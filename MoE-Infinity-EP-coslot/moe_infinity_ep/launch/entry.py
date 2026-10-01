"""MoE_EP top-level entry — mirrors upstream MoE but uses DistributedOffloadEngine.

CRITICAL import order: callers must invoke
``moe_infinity_ep.launch.distributed_setup.pin_visible_device()`` BEFORE
constructing this class, because importing ``moe_infinity`` triggers a C++
extension that probes ``torch.cuda.device_count()``.

Milestone 1 scope: ``MoE_EP.load_only(...)`` constructs the engine and the
model (registering weights with the per-rank archer engine), then barriers.
No generate/forward path is exposed yet — that arrives in Milestone 2.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

# Upstream is imported lazily inside methods to keep this module importable
# in environments where moe_infinity isn't installed yet (e.g., for static
# inspection / tests of config + topology only).

from .distributed_setup import ProcessTopology, barrier
from . import init_logging as ilog
from ..utils.config import Config


class MoE_EP:
    """Top-level multi-process MoE wrapper.

    Construction pattern (per rank):

        topology = init_distributed(cfg.parallel.ep_size, cfg.parallel.dp_size)
        engine = MoE_EP(cfg, topology)
        engine.load()       # weight registration + first-time offload (serialized)
        # M2+: engine.generate(input_ids, ...)
    """

    def __init__(self, cfg: Config, topology: ProcessTopology):
        self.cfg = cfg
        self.topology = topology
        self.engine = None
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load the model on all ranks. Serializes first-time offload to rank 0.

        After this returns on all ranks, every rank holds its own copy of the
        empty-shell model with weights registered to its per-rank archer
        engine (the actual weight bytes live in the on-disk offload directory).
        """
        ilog.log("load", "BEGIN", model=self.cfg.model.path,
                 ep_rank=self.topology.ep_rank,
                 world=self.topology.world_size)
        # Lazy imports — only valid once pin_visible_device() has run.
        from moe_infinity.entrypoints.big_modeling import MoE as _UpstreamMoE  # noqa: F401
        from moe_infinity.common.constants import MODEL_MAPPING_NAMES
        from moe_infinity.utils import ArcherConfig, get_checkpoint_paths
        from transformers import AutoConfig
        from accelerate.utils.versions import is_torch_version

        from ..runtime.distributed_engine import DistributedOffloadEngine

        if not is_torch_version(">=", "2.0"):
            raise RuntimeError("PyTorch >= 2.0 required.")

        model_path = self.cfg.model.path
        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Model path does not exist: {model_path}. "
                f"This entry point expects a local snapshot."
            )

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        architecture = model_config.architectures[0].lower()
        arch = next((a for a in MODEL_MAPPING_NAMES if a in architecture), None)
        if arch is None:
            raise RuntimeError(
                f"Architecture {architecture} not supported by upstream MoE-Infinity. "
                f"Supported: {list(MODEL_MAPPING_NAMES.keys())}."
            )
        model_cls = MODEL_MAPPING_NAMES[arch]
        checkpoint_paths = get_checkpoint_paths(model_path)

        # Per-rank offload directory — each rank gets its own copy of the
        # offloaded weights. This is what guarantees NUMA-local replication:
        # rank 0 faults its pages into NUMA 0, rank 1 into NUMA 1, etc., with
        # no kernel page-cache sharing between ranks. Also avoids concurrent
        # write contention during first-time offload.
        #
        # The actual NUMA pinning is enforced earlier in
        # ``distributed_setup.pin_visible_device()`` via libnuma — that locks
        # this process's CPU affinity AND preferred-allocation node to the
        # GPU's NUMA, so archer's host_memory_ratio budget (default 0.30 of
        # the local NUMA = ~296GB on dev box) faults the full ~57GB model
        # into local-NUMA RAM. End result, per NUMA:
        #   * Full model resident in host RAM (no disk paging during inference)
        #   * Every PCIe fetch issued by ranks on that NUMA stays local-socket
        # Per-NUMA shared offload directory (instead of per-rank). Ranks on
        # the same NUMA node share one materialized model copy on disk —
        # NUMA-local host RAM still gets independent pages via libnuma
        # strict membind, so we keep the locality benefit but cut disk
        # duplication by a factor of (GPUs_per_NUMA).
        # Concurrent first-time materialize is serialized via flock so only
        # one rank per NUMA writes archer_index; others see it exist and skip.
        from .distributed_setup import _gpu_numa_node
        # ProcessTopology doesn't expose local_rank; LOCAL_RANK env is set by
        # torchrun and is the GPU index on this node.
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        _numa_id = _gpu_numa_node(_local_rank)
        if _numa_id is None or _numa_id < 0:
            _numa_id = 0
        per_rank_offload = f"{self.cfg.offload.path}_numa{_numa_id}"
        os.makedirs(per_rank_offload, exist_ok=True)

        # C2 fix (2026-05-27): flock is now used in BOTH modes.
        #
        # Prior behaviour: shm-shared mode disabled flock entirely, assuming
        # the follower disk-read skip (set later via MOE_INFINITY_NUMA_FOLLOWER)
        # was enough.  But the follower-skip only applies to the sparse-load
        # ReadTensor loop INSIDE archer.  The dense-load + archer_index
        # serialise path runs unconditionally on every rank that enters
        # engine.init, so both leader and follower race to fwrite the same
        # archer_index file → silent corruption on cold start → next run's
        # Deserialize fails or returns inconsistent tensor IDs.
        #
        # Fix: every rank acquires flock.  Materializer (first comer) holds
        # LOCK_EX during the archer_index write; everyone else takes LOCK_SH
        # and waits for the EX holder to release.  After LOCK_EX → LOCK_SH
        # downgrade, follower processes that already passed the file-exists
        # check still get LOCK_SH so concurrent reads of archer_index are
        # safe.
        _lock_path = per_rank_offload + ".init.lock"
        _archer_index = os.path.join(per_rank_offload, "archer_index")
        import fcntl as _fcntl
        import time as _time
        _lock_fd = open(_lock_path, "w")
        # flock(LOCK_EX) blocks until the materializer (first rank per NUMA)
        # finishes writing archer_index — on a cold start that's the full
        # ~30min disk materialize.  Log BEFORE the blocking call so a long
        # wait reads as "blocked on flock" rather than a silent hang, and the
        # elapsed makes a slow/dead materializer obvious.
        _index_exists = os.path.exists(_archer_index)
        _flock_t = _time.monotonic()
        ilog.log("flock", f"acquiring archer_index lock "
                 f"(index_exists={_index_exists}, "
                 f"want={'SH' if _index_exists else 'EX'})",
                 path=_lock_path)
        if _index_exists:
            _fcntl.flock(_lock_fd.fileno(), _fcntl.LOCK_SH)
            self._numa_lock_mode = "SH"
        else:
            _fcntl.flock(_lock_fd.fileno(), _fcntl.LOCK_EX)
            if os.path.exists(_archer_index):
                # Another rank materialised while we waited for EX → downgrade.
                _fcntl.flock(_lock_fd.fileno(), _fcntl.LOCK_SH)
                self._numa_lock_mode = "SH(post-wait)"
            else:
                self._numa_lock_mode = "EX(materializer)"
        self._numa_init_lock_fd = _lock_fd
        ilog.log("flock", f"acquired mode={self._numa_lock_mode} "
                 f"waited={_time.monotonic() - _flock_t:.2f}s",
                 path=_lock_path)
        print(f"[entry rank={self.topology.global_rank}] archer_index flock "
              f"acquired: mode={self._numa_lock_mode} "
              f"path={_lock_path}", flush=True)

        if self.topology.is_rank0:
            numa_node = os.environ.get("MOE_EP_NUMA_NODE", str(_numa_id))
            print(
                f"[entry] rank0 offload={per_rank_offload} (NUMA-shared) "
                f"numa_node={numa_node} "
                f"host_memory_ratio={self.cfg.offload.host_memory_ratio}",
                flush=True,
            )

        # archer's C++ trace-based prefetch is a SECOND control loop that
        # raced with our Python controller (silent drift source).  Default
        # OFF; opt in only via MOE_EP_ARCHER_TRACE_PREFETCH=1 for ablation.
        # YAML's ``execution.enable_prefetch`` now controls ONLY the Python
        # path (CommandDispatcher.enqueue_prefetch on misses), not archer.
        _archer_trace_prefetch = os.environ.get(
            "MOE_EP_ARCHER_TRACE_PREFETCH", "0",
        ) == "1"
        archer_cfg_dict = {
            "offload_path": per_rank_offload,
            "device_memory_ratio": self.cfg.offload.device_memory_ratio,
            "host_memory_ratio": self.cfg.offload.host_memory_ratio,
            "num_threads": 1,
            "prefetch": _archer_trace_prefetch,
        }
        # Optional: pre-warmed EAMC (.npy of ExpertTracer.trace_collection).
        # If set, archer auto-loads via ArcherConfig.trace_path during
        # OffloadEngine init. Used by the main_exp sweep to share one
        # warm-up's trace across all measurement runs of the same dataset.
        if self.cfg.offload.trace_path:
            archer_cfg_dict["trace_path"] = self.cfg.offload.trace_path
        archer_config_obj = ArcherConfig.load_from_json(archer_cfg_dict)

        self.engine = DistributedOffloadEngine(
            capacity=getattr(archer_config_obj, "trace_capacity", 16),
            config=model_config,
            topology=self.topology,
            ep_cfg={
                "policies": self.cfg.policies.__dict__,
                "cache_capacity_per_rank": self.cfg.offload.cache_capacity_per_rank,
                "sparse_hbm_ratio": getattr(
                    self.cfg.offload, "sparse_hbm_ratio", 0.0
                ),
                "execution": self.cfg.execution.__dict__,
            },
        )
        self.engine.ckpt_files = checkpoint_paths

        # NUMA-shared pinned host memory: ranks on the same NUMA share one
        # memfd-backed pinned region for archer's sparse host cache. Leader
        # (lowest local_rank on this NUMA) creates + populates; followers
        # mmap the same memfd via Unix-socket SCM_RIGHTS, register their
        # cuda mapping, then run engine.init which bump-allocates into the
        # shared region and skips the disk read.
        _shm_region = None
        if int(os.environ.get("MOE_INFINITY_USE_NUMA_SHARED_HOST", "1")):
            from .numa_shared_host import (
                setup_numa_shared,
                compute_sparse_bytes_for_model,
            )
            # Determine NUMA leader: lowest local_rank in this NUMA group.
            _ws = self.topology.world_size
            _same_numa = [r for r in range(_ws)
                          if _gpu_numa_node(r) == _numa_id]
            _shm_leader_rank = min(_same_numa) if _same_numa else _local_rank
            _is_shm_leader = (_local_rank == _shm_leader_rank)
            _follower_count = max(0, len(_same_numa) - 1)
            _shm_sock = f"/tmp/moe_numa{_numa_id}.sock"
            ilog.log("numa_shm.elect",
                     f"role={'LEADER' if _is_shm_leader else 'FOLLOWER'} "
                     f"leader_rank={_shm_leader_rank} "
                     f"numa_group={_same_numa} followers={_follower_count}",
                     numa=_numa_id)

            # Total bytes archer's sparse cache will hold (full model).
            _num_layers = int(getattr(model_config, "num_hidden_layers", 0))
            _num_experts = int(getattr(model_config, "num_experts", 0))
            _hidden = int(getattr(model_config, "hidden_size", 0))
            _intermediate = int(getattr(
                model_config, "moe_intermediate_size",
                getattr(model_config, "intermediate_size", 0),
            ))
            _expected = compute_sparse_bytes_for_model(
                _num_layers, _num_experts, _hidden, _intermediate, 2,
            )
            # +5% headroom for alignment / kAioAlignment padding inside archer.
            _shm_size = int(_expected * 1.05)

            # cgroup-aware sanity: archer is otherwise blind to the container
            # memory limit and will silently page-fault its way to OOM. Each
            # NUMA group creates its own memfd (no cross-NUMA sharing), so
            # the cgroup sees ``num_numas × _shm_size`` of pinned shmem on top
            # of per-rank overhead. Refuse to start if that already exceeds
            # the limit instead of letting the OOM killer fire mid-load.
            try:
                with open("/sys/fs/cgroup/memory.max") as _f:
                    _cg_str = _f.read().strip()
                _cg_max = (
                    int(_cg_str) if _cg_str and _cg_str != "max" else 0
                )
            except FileNotFoundError:
                _cg_max = 0
            # PREFLIGHT (2026-05-28): residue/headroom-aware guard. cgroup.current
            # is whatever the cgroup ALREADY holds before this run allocates —
            # idle baseline normally, but PREVIOUS-RUN RESIDUE (a stuck rank
            # holding a shared memfd) or a co-tenant if elevated. Measured: a
            # crashed run leaves a survivor pinning the full shm region until the
            # last holder dies (window up to the NCCL watchdog, ~120min). Folding
            # current into the budget catches that BEFORE the kernel OOM-kills us.
            try:
                with open("/sys/fs/cgroup/memory.current") as _f:
                    _cg_cur = int(_f.read().strip())
            except (FileNotFoundError, ValueError):
                _cg_cur = 0
            if _cg_max > 0:
                # When nvidia-smi probing fails (`_gpu_numa_node` returns
                # None) we MUST NOT underestimate — None collapses into a
                # single set entry → cgroup check sees half the memory we
                # actually need.  Treat unknowns as distinct (per local_rank
                # bucket) so the worst case dominates.
                _numa_ids = [_gpu_numa_node(r) for r in range(_ws)]
                _resolved = {n for n in _numa_ids if n is not None}
                _unresolved = sum(1 for n in _numa_ids if n is None)
                _num_numas = max(1, len(_resolved) + _unresolved)
                _GB = 1 << 30
                # Per-rank host overhead BEYOND the shared sparse region:
                # Python interpreter, HF from_pretrained transient (state_dict
                # held in RAM until archer takeover), per-rank dense weights
                # pinned by torch, and the AIO pinned pool. This scales with
                # WORLD_SIZE — the previous flat 100GB ignored that and
                # under-counted multi-rank runs (4 ranks → real overhead is
                # ~4× a single rank's, not a constant). Override via
                # MOE_EP_PER_RANK_OVERHEAD_GB if a model's transient differs.
                _per_rank_overhead_gb = float(
                    os.environ.get("MOE_EP_PER_RANK_OVERHEAD_GB", "40")
                )
                _overhead = int(_ws * _per_rank_overhead_gb * _GB)
                _expected_cg = _num_numas * _shm_size + _overhead
                # Safety margin on top of current + footprint. Also absorbs the
                # small per-rank jitter in reading memory.current (shared cgroup,
                # reads race by ~1GB across ranks) so the fail/pass decision is
                # effectively identical on every rank → symmetric abort, no hang.
                _margin = int(
                    float(os.environ.get("MOE_EP_PREFLIGHT_MARGIN_GB", "20"))
                    * _GB
                )
                _need = _cg_cur + _expected_cg + _margin
                if self.topology.is_rank0:
                    print(
                        f"[entry rank0] cgroup sanity: numas={_num_numas} "
                        f"shm/numa={_shm_size/_GB:.1f}GB "
                        f"per_rank_overhead={_per_rank_overhead_gb:.0f}GB×{_ws}"
                        f"={_overhead/_GB:.1f}GB "
                        f"footprint={_expected_cg/_GB:.1f}GB "
                        f"current(before)={_cg_cur/_GB:.1f}GB "
                        f"margin={_margin/_GB:.0f}GB "
                        f"need={_need/_GB:.1f}GB cgroup_max={_cg_max/_GB:.1f}GB",
                        flush=True,
                    )
                if _need > _cg_max:
                    _residue_hint = (
                        f" cgroup.current is ALREADY {_cg_cur/_GB:.1f}GB before "
                        f"this run allocates anything — PREVIOUS-RUN RESIDUE is "
                        f"suspected (a stuck rank holding a shared memfd). Run "
                        f"`scripts/cleanup_prev_run.sh` (or diag_residue.sh "
                        f"leftover), wait for memory.current to drop, then retry."
                        if _cg_cur > 2 * _per_rank_overhead_gb * _GB else ""
                    )
                    raise RuntimeError(
                        f"preflight: would exceed cgroup memory.max. "
                        f"current={_cg_cur/_GB:.1f}GB + footprint "
                        f"({_num_numas} NUMA × {_shm_size/_GB:.1f}GB + "
                        f"{_ws}×{_per_rank_overhead_gb:.0f}GB overhead = "
                        f"{_expected_cg/_GB:.1f}GB) + margin {_margin/_GB:.0f}GB "
                        f"= {_need/_GB:.1f}GB > memory.max {_cg_max/_GB:.1f}GB."
                        f"{_residue_hint} Aborting deterministically before the "
                        f"kernel OOM-kills a rank (exit=137). Options: raise the "
                        f"cgroup limit, fewer GPUs/NUMAs, or lower "
                        f"MOE_EP_PER_RANK_OVERHEAD_GB if over-conservative."
                    )

            _shm_region = setup_numa_shared(
                numa_id=_numa_id, size=_shm_size,
                is_leader=_is_shm_leader, sock_path=_shm_sock,
                follower_count=_follower_count,
            )
            # Keep alive for process lifetime. archer holds raw pointers
            # into this region — releasing here would dangle them.
            # Reused-Process Risk: this MoE_EP instance is created per
            # torchrun child process today, but if the process gets reused
            # (e.g. in a long-running daemon) we'd leak shm.  Register an
            # atexit close so cgroup pages get accounted on shutdown.
            self._numa_shm_region = _shm_region
            import atexit as _atexit
            def _close_shm(region=_shm_region):
                try:
                    region.close()
                except Exception:
                    pass
            _atexit.register(_close_shm)

            # Follower disk-read skip (2026-05-27 refinement):
            #
            # Earlier design had follower waiting on a leader_done flag BEFORE
            # entering engine.init — that deadlocked because HF from_pretrained
            # (called inside engine.init) issues NCCL collectives that need
            # BOTH ranks already past the rendezvous.
            #
            # Resolution: still let BOTH ranks enter engine.init concurrently
            # (so HF NCCL proceeds normally), but inside the C++ sparse-load
            # loop the FOLLOWER skips kArcherTensorHandle->ReadTensor.  Bump
            # offsets still advance in lockstep (sparse_nodes iteration is
            # deterministic), so the follower's torch views in kTensorIndex
            # point to the SAME shm offsets the leader writes.
            #
            # The cross-rank sync point is the existing barrier() call further
            # down (entry.py:~333) — by the time it returns, the leader's full
            # sparse-load is complete and its writes are visible to the
            # follower via the shared memfd backing.  No tensor data is read
            # between sparse-load end and that barrier (verified in flow).
            #
            # Benefit: halves cold-start disk I/O per NUMA (~444 GB → ~222 GB
            # for Qwen3-235B).  Page cache helps when both read concurrently,
            # but the second read still pays kernel copy overhead.
            os.environ["MOE_INFINITY_NUMA_FOLLOWER"] = (
                "0" if _is_shm_leader else "1"
            )

        # Ranks sharing a NUMA dir are serialised by the flock acquired above.
        # First rank per NUMA materializes archer_index; later ranks reuse it.
        try:
            # engine.init drives archer's per-rank engine construction; the
            # leader's sparse-load (disk → memfd) runs inside this on a cold
            # start and is the single longest init stage (~30min for the full
            # model). from_pretrained additionally fires HF NCCL collectives
            # that need BOTH ranks present. Bracket each so the logs show
            # exactly which one a stuck run is sitting in.
            with ilog.stage("engine.init", role=self._numa_lock_mode):
                with self.engine.init(cls=model_cls, ar_config=archer_cfg_dict):
                    with ilog.stage("from_pretrained"):
                        self.model = model_cls.from_pretrained(
                            model_path,
                            attn_implementation="eager",
                            trust_remote_code=True,
                        )
        finally:
            # Release the NUMA init lock (now acquired in both modes — see
            # C2 fix above).
            if self._numa_init_lock_fd is not None:
                try:
                    _fcntl.flock(
                        self._numa_init_lock_fd.fileno(), _fcntl.LOCK_UN
                    )
                    self._numa_init_lock_fd.close()
                    ilog.log("flock", f"released (mode was "
                             f"{self._numa_lock_mode})")
                except Exception as _e:
                    ilog.error("flock", f"release failed: {_e!r}")
        # The post-init barrier is the cross-rank rendezvous: followers wait
        # here until the leader's full sparse-load is visible via the shared
        # memfd. A rank stuck at barrier.BEGIN means a *peer* hasn't arrived
        # (check the other ranks' last stage), not a local fault.
        with ilog.stage("post_init_barrier"):
            barrier()
        self._loaded = True
        ilog.log("load", "END model loaded on this rank")

    def report(self) -> dict:
        """Lightweight rank-local status snapshot, useful for the smoke test."""
        rep = {
            "global_rank": self.topology.global_rank,
            "ep_rank": self.topology.ep_rank,
            "world_size": self.topology.world_size,
            "loaded": self._loaded,
        }
        if self._loaded and self.model is not None:
            try:
                n_params = sum(p.numel() for p in self.model.parameters())
                rep["model_param_count_after_offload"] = int(n_params)
            except Exception as e:
                rep["model_param_count_after_offload"] = f"<error: {e}>"
        return rep
