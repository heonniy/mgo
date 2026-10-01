from __future__ import annotations

"""Pre-CUDA process setup.

Call pin_rank_before_cuda_import() before importing torch or moe_infinity.
The legacy C++ extension sizes its per-device state from CUDA visibility at
import/construction time, so each torchrun process must see exactly one GPU.
"""

import ctypes
import os
import re
import subprocess
from pathlib import Path


def _original_visible() -> list[str] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None or value.strip() == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _physical_selector(local_rank: int) -> str:
    visible = _original_visible()
    if visible is None:
        return str(local_rank)
    if local_rank >= len(visible):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} outside CUDA_VISIBLE_DEVICES={visible}"
        )
    return visible[local_rank]


def _pci_bus(selector: str) -> str | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={selector}",
                "--query-gpu=pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        return out.lower()
    except Exception:
        return None


def _numa_node(selector: str) -> int | None:
    bus = _pci_bus(selector)
    if not bus:
        return None
    candidates = [
        bus,
        re.sub(r"^00000000:", "0000:", bus),
    ]
    for bdf in candidates:
        p = Path("/sys/bus/pci/devices") / bdf / "numa_node"
        if p.exists():
            value = int(p.read_text().strip())
            if value >= 0:
                return value
    # Virtualized hosts may expose GPU PCI NUMA=-1 while the OS has exactly
    # one memory node. There is no locality ambiguity in that case.
    nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
    if len(nodes) == 1:
        return int(nodes[0].name[4:])
    return None


def _apply_numa(node: int, strict: bool) -> str:
    try:
        lib = ctypes.CDLL("libnuma.so.1", use_errno=True)
        if lib.numa_available() < 0:
            if strict:
                raise RuntimeError("libnuma is unavailable")
            return "libnuma-unavailable"
        if lib.numa_run_on_node(ctypes.c_int(node)) != 0:
            raise OSError(ctypes.get_errno(), "NUMA CPU binding failed")

        if strict and hasattr(lib, "numa_allocate_nodemask"):
            lib.numa_allocate_nodemask.restype = ctypes.c_void_p
            lib.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            lib.numa_set_membind.argtypes = [ctypes.c_void_p]
            mask = lib.numa_allocate_nodemask()
            if mask:
                lib.numa_bitmask_setbit(mask, node)
                if hasattr(lib, "numa_set_strict"):
                    lib.numa_set_strict(ctypes.c_int(1))
                # numa_set_membind returns void and only prints failures.
                # Verify the kernel policy instead of claiming strict bind.
                lib.numa_set_membind(mask)
                if hasattr(lib, "numa_bitmask_free"):
                    lib.numa_bitmask_free.argtypes = [ctypes.c_void_p]
                    lib.numa_bitmask_free(mask)
                mode = ctypes.c_int()
                bits = (ctypes.c_ulong * ((node // 64) + 1))()
                lib.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                                             ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
                ret = lib.get_mempolicy(ctypes.byref(mode), bits, len(bits) * 64, None, 0)
                if ret != 0 or mode.value != 2 or not (bits[node // 64] & (1 << (node % 64))):
                    raise RuntimeError("kernel did not apply strict NUMA memory binding")
                return "membind-strict"

        if strict:
            raise RuntimeError("cannot allocate a strict NUMA nodemask")

        if hasattr(lib, "numa_set_preferred"):
            lib.numa_set_preferred.argtypes = [ctypes.c_int]
            lib.numa_set_preferred(node)
            return "preferred"
        return "cpu-affinity-only"
    except Exception as exc:
        if strict:
            raise RuntimeError(f"strict NUMA binding failed for node {node}") from exc
        return f"numa-failed:{exc!r}"


def pin_rank_before_cuda_import(strict_numa: bool | None = None) -> dict:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    selector = _physical_selector(local_rank)

    # Must happen before importing torch/moe_infinity in the caller.
    os.environ["CUDA_VISIBLE_DEVICES"] = selector
    os.environ.setdefault("MOE_EP_DISABLE_ARCHER_EVICT", "1")
    os.environ.setdefault("MOE_EP_NATIVE_NUMERICS", "1")
    if os.environ["MOE_EP_NATIVE_NUMERICS"] == "1":
        os.environ.setdefault("MOE_EP_SLOT_VIEWS", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    node = _numa_node(selector)
    if strict_numa is None:
        strict_numa = os.environ.get("MGO_V2_NUMA_STRICT", "1") != "0"
    policy = "unknown"
    if node is not None:
        os.environ["MGO_V2_NUMA_NODE"] = str(node)
        policy = _apply_numa(node, strict_numa)
    elif strict_numa:
        raise RuntimeError("GPU NUMA locality is unknown; cannot enforce strict binding")

    return {
        "local_rank": local_rank,
        "visible_gpu": selector,
        "numa_node": node,
        "numa_policy": policy,
        "strict_numa": strict_numa,
    }
