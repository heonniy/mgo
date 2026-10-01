"""Process topology and NCCL group construction for EP + DP.

CRITICAL: ``pin_visible_device()`` must be called *before* ``import moe_infinity``.
Upstream's C++ ``ExpertDispatcher`` calls ``kNumDevices()`` at construction
(``core/parallel/expert_dispatcher.cpp:35``); with full visibility it would
spawn N fetch/exec threads per rank. By gating ``CUDA_VISIBLE_DEVICES`` to the
local rank's GPU before the import, the dispatcher sees a single GPU per rank.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from . import init_logging as ilog


def _gpu_numa_node(local_rank: int) -> Optional[int]:
    """Look up the NUMA node of GPU index ``local_rank`` via sysfs.

    Returns the integer NUMA node, or None if it can't be resolved (e.g.,
    sysfs path missing, container without /sys mounted). On the dev box,
    nvidia-smi reports the PCI BDF and /sys/bus/pci/devices/{BDF}/numa_node
    exposes the binding.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id",
             f"--id={local_rank}", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        # nvidia-smi prints e.g. "00000000:1A:00.0" — strip leading zeros
        # in domain so the /sys path matches.
        parts = out.split(":")
        if len(parts) == 3:
            bdf = f"{int(parts[0], 16):04x}:{parts[1].lower()}:{parts[2].lower()}"
        else:
            bdf = out.lower()
        path = f"/sys/bus/pci/devices/{bdf}/numa_node"
        with open(path, "r") as f:
            node = int(f.read().strip())
        return node if node >= 0 else None
    except Exception:
        return None


def pin_visible_device() -> int:
    """Restrict this process to a single GPU (the one matching LOCAL_RANK)
    and pin host memory allocations to that GPU's NUMA node.

    Returns the local rank that was pinned. Must run before any CUDA init
    and before importing ``moe_infinity``.

    NUMA binding (per-NUMA full-model replication):
        Archer's host RAM allocations and the file-backed mmap of the offload
        directory both page-fault into the calling process's local NUMA node.
        Without explicit pinning, the kernel scheduler may migrate the process
        between NUMA nodes, causing cross-NUMA QPI traffic on every PCIe fetch.
        We use libnuma (mbind/sched_setaffinity via ctypes) to lock the
        process to the GPU's NUMA node so all expert pages fault locally.
        Combined with per-rank offload directories + host_memory_ratio sized
        to fit the full model, this guarantees:
          * Every NUMA holds a full copy of the model in host RAM
          * Every PCIe fetch is local-NUMA → local-GPU (no cross-socket hop)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    # archer's autonomous LFU evict + trace-based prefetch are a SECOND
    # control loop that fights the Python GlobalCacheController.  Disable
    # both by default; ablation can opt in via MOE_EP_ARCHER_AUTONOMOUS=1
    # which leaves these knobs at their fork defaults.  These env reads
    # happen inside C++ static initializers — they must be set BEFORE
    # ``import moe_infinity`` (which loads the .so).
    if os.environ.get("MOE_EP_ARCHER_AUTONOMOUS", "0") != "1":
        os.environ.setdefault("MOE_EP_DISABLE_ARCHER_EVICT", "1")

    # NUMA-shared pinned host memory (replaces EP-shard): instead of each rank
    # loading 1/N expert share (which requires cross-rank NCCL fetches for the
    # other (N-1)/N), processes on the same NUMA share ONE pinned region via
    # POSIX shm + cudaHostRegister. All N/numa ranks see the full model in
    # local-NUMA RAM, no cross-rank fetch needed for host→GPU.
    # The C++ side reads MOE_INFINITY_NUMA_SHARED_HOST=1 to enable this path.
    # EP-shard env (MOE_INFINITY_EP_RANK/SIZE) intentionally NOT set here.

    # NUMA pinning. STRICT mode (2026-05-23): all host RAM allocations from
    # this process MUST come from the GPU's local NUMA node. If the local
    # NUMA OOMs, allocation FAILS rather than silently falling back to the
    # remote NUMA. This makes the "per-NUMA full-model replication, no
    # cross-socket access" invariant a hard guarantee, not a hint.
    # Combined with the per-rank offload directory (entry.py), each NUMA
    # ends up with a full local copy of the offloaded model and no rank
    # ever pulls expert bytes through QPI/UPI.
    node = _gpu_numa_node(local_rank)
    if node is None:
        ilog.error("pin_device",
                   f"could not resolve NUMA node for GPU {local_rank} "
                   f"(nvidia-smi/sysfs probe failed). Falling back to the "
                   f"kernel auto-balancer — cross-socket QPI traffic on PCIe "
                   f"fetches becomes possible and per-NUMA replication is no "
                   f"longer guaranteed.")
    if node is not None:
        os.environ["MOE_EP_NUMA_NODE"] = str(node)
        _applied = "none"
        try:
            import ctypes
            libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
            if libnuma.numa_available() >= 0:
                # Pin CPU affinity to this NUMA's cores. Kernel scheduler
                # can't migrate the thread off-node. ALWAYS applied — keeps
                # locality regardless of the memory-policy mode below.
                libnuma.numa_run_on_node(ctypes.c_int(node))

                # Memory-policy mode (2026-05-28 toggle, default STRICT):
                #   MOE_EP_NUMA_STRICT=1 (default): MPOL_BIND via
                #     numa_set_membind — host pages MUST come from this NUMA;
                #     if the node OOMs the allocation FAILS rather than
                #     spilling to remote NUMA. Enforces per-NUMA full-model
                #     replication as a hard guarantee.
                #   MOE_EP_NUMA_STRICT=0: MPOL_PREFERRED via numa_set_preferred
                #     — prefer this NUMA but spill to remote under pressure.
                #     Diagnostic aid: if a run dies exit=137 under strict, a
                #     soft policy survives, isolating "single-NUMA OOM" from
                #     "whole-cgroup OOM".
                # BIND and PREFERRED are mutually exclusive at the kernel
                # mempolicy level (setting PREFERRED after BIND OVERWRITES it —
                # verified 2026-05-23 via numa_get_membind read-back), so we
                # set EXACTLY ONE based on the toggle.
                _numa_strict = os.environ.get("MOE_EP_NUMA_STRICT", "1") != "0"
                if _numa_strict:
                    # Enable strict mode — membind OOMs instead of silently
                    # spilling to remote NUMA.
                    if hasattr(libnuma, "numa_set_strict"):
                        libnuma.numa_set_strict.argtypes = [ctypes.c_int]
                        libnuma.numa_set_strict(1)
                    # Build a nodemask containing only this node and apply as
                    # membind. ctypes argtypes/restype MUST be set explicitly —
                    # without them ctypes treats pointers as 32-bit ints and
                    # silently truncates on 64-bit, making numa_set_membind a
                    # no-op (verified via numa_get_membind 2026-05-23).
                    if (hasattr(libnuma, "numa_allocate_nodemask")
                            and hasattr(libnuma, "numa_bitmask_setbit")
                            and hasattr(libnuma, "numa_set_membind")):
                        libnuma.numa_allocate_nodemask.argtypes = []
                        libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
                        libnuma.numa_bitmask_setbit.argtypes = [
                            ctypes.c_void_p, ctypes.c_uint,
                        ]
                        libnuma.numa_bitmask_setbit.restype = ctypes.c_void_p
                        libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
                        libnuma.numa_set_membind.restype = None

                        mask = libnuma.numa_allocate_nodemask()
                        if mask:
                            libnuma.numa_bitmask_setbit(mask, node)
                            libnuma.numa_set_membind(mask)
                            if hasattr(libnuma, "numa_bitmask_free"):
                                libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]
                                libnuma.numa_bitmask_free(mask)
                            _applied = "membind-strict"
                elif hasattr(libnuma, "numa_set_preferred"):
                    # Soft policy: prefer local NUMA, allow remote spill.
                    libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
                    libnuma.numa_set_preferred.restype = None
                    libnuma.numa_set_preferred(node)
                    _applied = "preferred-soft"
        except (OSError, AttributeError) as _e:
            ilog.error("pin_device",
                       f"libnuma pinning failed ({_e!r}) — kernel "
                       f"auto-balancer in effect, NUMA locality not enforced.")
        ilog.log("pin_device",
                 f"pinned to NUMA {node} (CUDA_VISIBLE_DEVICES="
                 f"{os.environ['CUDA_VISIBLE_DEVICES']}, mempolicy={_applied}, "
                 f"strict={os.environ.get('MOE_EP_NUMA_STRICT', '1')})",
                 numa=node)
    return local_rank


@dataclass(frozen=True)
class ProcessTopology:
    """Unified EP topology — world is the EP group; each rank holds its own
    batch shard (implicit DP equal to world_size). No separate DP slabs."""
    world_size: int
    ep_size: int                # == world_size
    global_rank: int
    ep_rank: int                # == global_rank
    local_device: int           # always 0 after pin_visible_device

    # Process groups (Any to avoid importing torch at module load time).
    # ep_group == world_group in unified mode.
    ep_group: object = None
    world_group: object = None

    @property
    def is_rank0(self) -> bool:
        return self.global_rank == 0


def init_distributed(ep_size: int) -> ProcessTopology:
    """Initialize torch.distributed (NCCL). EP group == world group.

    Assumes torchrun has set RANK, WORLD_SIZE, LOCAL_RANK. ``pin_visible_device``
    must have been called first (this function will assert it).
    """
    import torch
    import torch.distributed as dist

    assert "CUDA_VISIBLE_DEVICES" in os.environ, (
        "pin_visible_device() must be called before init_distributed()"
    )
    assert os.environ.get("RANK") is not None, "torchrun env not set (RANK)"

    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if ep_size != world_size:
        raise ValueError(
            f"unified EP requires ep_size == world_size, got "
            f"ep_size={ep_size}, world_size={world_size}"
        )

    # After CUDA_VISIBLE_DEVICES gate, the only visible device is index 0.
    # Bind BEFORE NCCL init so the communicator latches onto the right GPU.
    # We do NOT pass device_id to init_process_group: doing so caused archer's
    # io_uring AIO read to fail with EINVAL during the dense-parameter copy
    # phase (apparently NCCL eagerly initialises the device context which
    # races with archer's per-rank AIO setup). The "Guessing device ID based
    # on global rank" warning is benign for us because each rank only has one
    # visible device; pass collective device_ids at call sites instead.
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        # Extend collective timeout for long-running operations:
        #   * First-time per-NUMA model materialize (~30 min, sometimes more).
        #   * Slow ranks at end-of-warmup barrier with many sample-by-sample
        #     decode steps.
        # Default 30 min is tight; bump to 2 h to leave headroom.
        from datetime import timedelta as _td
        _timeout_env = os.environ.get("MOE_EP_NCCL_TIMEOUT_MIN", "120")
        try:
            _timeout = _td(minutes=int(_timeout_env))
        except ValueError:
            _timeout = _td(minutes=120)
        # NCCL rendezvous over env:// blocks until ALL ranks call in — a rank
        # stuck at nccl_init.BEGIN means a peer never arrived (crashed earlier
        # in pin_device / import). The timeout above bounds the hang.
        with ilog.stage("nccl_init", world=world_size,
                        timeout_min=_timeout_env):
            dist.init_process_group(
                backend="nccl", init_method="env://", timeout=_timeout,
            )
    else:
        ilog.log("nccl_init", "already initialized, skipping")

    world_group = dist.group.WORLD
    ilog.log("init_distributed",
             f"END ep=world={world_size} global_rank={global_rank}")

    return ProcessTopology(
        world_size=world_size,
        ep_size=ep_size,
        global_rank=global_rank,
        ep_rank=global_rank,
        local_device=0,
        ep_group=world_group,
        world_group=world_group,
    )


def barrier(topology: Optional[ProcessTopology] = None) -> None:
    """Convenience barrier on the world group.

    Pins the collective to the local GPU via ``device_ids=[0]``; without this
    NCCL emits a "Guessing device ID based on global rank" warning and would
    try to use a device index that isn't visible to this rank (per-rank
    CUDA_VISIBLE_DEVICES gate).
    """
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier(device_ids=[0])
