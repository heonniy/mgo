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
        # Linux sysfs normally uses lowercase domain:bus:device.function.
        out[int(idx_s)] = bus.lower()
    return out


def gpu_numa_node(gpu_index: int) -> int:
    bus = gpu_pci_bus_ids()[gpu_index]
    p = Path("/sys/bus/pci/devices") / bus / "numa_node"
    if not p.exists():
        # nvidia-smi may report 00000000:xx while sysfs uses 0000:xx.
        short = re.sub(r"^00000000:", "0000:", bus)
        p = Path("/sys/bus/pci/devices") / short / "numa_node"
    if not p.exists():
        return -1
    return int(p.read_text().strip())


def numa_cpus(node: int) -> list[int]:
    if node < 0:
        return []
    p = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not p.exists():
        return []
    return _parse_cpu_list(p.read_text())


def pin_process_to_gpu_numa(gpu_index: int) -> dict:
    node = gpu_numa_node(gpu_index)
    cpus = numa_cpus(node)
    if cpus:
        os.sched_setaffinity(0, cpus)
    return {"gpu": gpu_index, "numa_node": node, "cpus": cpus}


def recommended_env() -> dict[str, str]:
    return {
        "NCCL_P2P_DISABLE": "0",
        "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE", "1"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
        # Legacy Archer must not independently evict while mgo_v2 owns cache.
        "MOE_EP_DISABLE_ARCHER_EVICT": "1",
    }
