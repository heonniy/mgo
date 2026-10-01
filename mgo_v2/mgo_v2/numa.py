from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def _parse_cpu_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def visible_physical_gpu(local_rank: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return local_rank
    ids = [x.strip() for x in visible.split(",") if x.strip()]
    if local_rank >= len(ids):
        raise RuntimeError("LOCAL_RANK outside CUDA_VISIBLE_DEVICES")
    try:
        return int(ids[local_rank])
    except ValueError:
        # UUID-based CUDA_VISIBLE_DEVICES: nvidia-smi ordering cannot be
        # inferred safely here, so use local rank and report it.
        return local_rank


def gpu_pci_bus_ids() -> dict[int, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    text = subprocess.check_output(cmd, text=True)
    out = {}
    for line in text.strip().splitlines():
        idx_s, bus = [x.strip() for x in line.split(",", 1)]
        out[int(idx_s)] = bus.lower()
    return out


def gpu_numa_node(physical_gpu_index: int) -> int:
    bus = gpu_pci_bus_ids()[physical_gpu_index]
    candidates = [
        Path("/sys/bus/pci/devices") / bus,
        Path("/sys/bus/pci/devices") / re.sub(r"^00000000:", "0000:", bus),
    ]
    for base in candidates:
        p = base / "numa_node"
        if p.exists():
            node = int(p.read_text().strip())
            if node >= 0:
                return node
    nodes = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
    if len(nodes) == 1:
        return int(nodes[0].name[4:])
    return -1


def numa_cpus(node: int) -> list[int]:
    if node < 0:
        return []
    p = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not p.exists():
        return []
    return _parse_cpu_list(p.read_text())


def pin_process_to_local_rank(local_rank: int) -> dict:
    physical = visible_physical_gpu(local_rank)
    node = gpu_numa_node(physical)
    cpus = numa_cpus(node)
    if cpus:
        os.sched_setaffinity(0, cpus)
    return {
        "local_rank": local_rank,
        "physical_gpu": physical,
        "numa_node": node,
        "cpus": cpus,
    }


def recommended_env() -> dict[str, str]:
    return {
        "NCCL_P2P_DISABLE": "0",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        "MOE_EP_DISABLE_ARCHER_EVICT": "1",
    }
