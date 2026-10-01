#!/usr/bin/env python3
import json
import os
import subprocess

from mgo_v2.numa import gpu_pci_bus_ids, gpu_numa_node, numa_cpus


def main():
    topo = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        text=True,
        capture_output=True,
        check=False,
    )
    result = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_pci": gpu_pci_bus_ids(),
        "gpu_numa": {},
        "nvidia_smi_topo": topo.stdout,
    }
    for gpu in sorted(result["gpu_pci"]):
        node = gpu_numa_node(gpu)
        result["gpu_numa"][gpu] = {
            "node": node,
            "cpus": numa_cpus(node),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
