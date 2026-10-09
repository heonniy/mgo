"""Host-only NUMA utilities; importing this module never initializes CUDA."""
import ctypes as C
import json
import os
import re
import subprocess
from pathlib import Path

GPUS = (0, 1, 4, 5)
NODES = (0, 0, 1, 1)
ROOT = Path('/data2/esjung/mgo-results/pcie_topology_ablation_20261009')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def cpu_set(node):
    text = Path(f'/sys/devices/system/node/node{node}/cpulist').read_text().strip()
    result = []
    for interval in text.split(','):
        ends = interval.split('-')
        result.extend(range(int(ends[0]), int(ends[-1]) + 1))
    return result


def affinity(rank):
    # Two disjoint 16-CPU sets per node, derived from the real host.
    all_cpus = cpu_set(NODES[rank])
    half = len(all_cpus) // 2
    pair_rank = rank % 2
    return all_cpus[pair_rank * half:(pair_rank + 1) * half]


def available_bytes():
    return next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def bind_pages(address, length, node):
    lib = C.CDLL('libnuma.so.1', use_errno=True)
    lib.mbind.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_void_p, C.c_ulong, C.c_uint]
    lib.mbind.restype = C.c_long
    mask = C.c_ulong(1 << node)
    if lib.mbind(address, length, 2, C.byref(mask), 64, 0):
        raise OSError(C.get_errno(), f'mbind node {node}')


def page_nodes(address, length, node, stride=4 * 2**20):
    lib = C.CDLL('libnuma.so.1', use_errno=True)
    lib.move_pages.argtypes = [C.c_int, C.c_ulong, C.POINTER(C.c_void_p), C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int]
    lib.move_pages.restype = C.c_long
    offsets = list(range(0, length, stride))
    addresses = (C.c_void_p * len(offsets))(*(address + offset for offset in offsets))
    statuses = (C.c_int * len(offsets))()
    if lib.move_pages(0, len(offsets), addresses, None, statuses, 0) < 0:
        raise OSError(C.get_errno(), 'move_pages locality query')
    counts = {str(value): list(statuses).count(value) for value in set(statuses)}
    if set(statuses) != {node}:
        raise RuntimeError(f'Unexpected sampled NUMA placement: expected {node}, got {counts}')
    return dict(samples=len(offsets), stride_bytes=stride, node_counts=counts)


def mapping_receipt(address):
    address = int(address)
    for line in Path('/proc/self/maps').read_text().splitlines():
        interval = line.split()[0]
        start, end = (int(value, 16) for value in interval.split('-'))
        if start <= address < end:
            prefix = f'{start:x} '
            numa = next((row for row in Path('/proc/self/numa_maps').read_text().splitlines()
                         if row.startswith(prefix)), None)
            return dict(maps=line, numa_maps=numa)
    raise RuntimeError('Shared source VMA not found')


def topology():
    fields = 'index,uuid,pci.bus_id,name,memory.total,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max'
    text = subprocess.check_output(['nvidia-smi', '-i', '0,1,4,5', f'--query-gpu={fields}', '--format=csv,noheader,nounits'], text=True)
    devices = []
    for line in text.splitlines():
        index, uuid, bus, name, total, gen, width, max_gen, max_width = (x.strip() for x in line.split(','))
        index = int(index)
        bus = re.sub(r'^00000000:', '0000:', bus.lower())
        node = int((Path('/sys/bus/pci/devices') / bus / 'numa_node').read_text())
        assert index in GPUS and node == NODES[GPUS.index(index)]
        devices.append(dict(physical_gpu=index, uuid=uuid, pci_bus=bus, name=name,
                            numa_node=node, total_mib=int(total), link_gen=int(gen),
                            link_width=int(width), max_link_gen=int(max_gen), max_link_width=int(max_width)))
    assert {row['physical_gpu'] for row in devices} == set(GPUS)
    nodes = {}
    for node in (0, 1):
        rows = Path(f'/sys/devices/system/node/node{node}/meminfo').read_text().splitlines()
        mem = {line.split()[2].rstrip(':'): int(line.split()[3]) * 1024 for line in rows if line.endswith('kB')}
        nodes[str(node)] = dict(cpus=cpu_set(node), memory=mem)
    return dict(devices=devices, nodes=nodes, fixed_affinity={str(gpu): affinity(rank) for rank, gpu in enumerate(GPUS)},
                host_available_bytes=available_bytes(), topology_matrix=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True))


if __name__ == '__main__':
    write(ROOT / 'topology.json', topology())
    print(ROOT / 'topology.json')
