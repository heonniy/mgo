"""One file-backed pinned expert pool per NUMA group, shared by its two ranks."""
import ctypes as C
import json
import mmap
import os
import time
from pathlib import Path

import torch

from pcie_host import bind_pages, mapping_receipt, page_nodes, write


class SharedPinnedMapping:
    def __init__(self, path, size, node, *, create=False):
        self.path = Path(path)
        self.size = int(size)
        self.node = int(node)
        self.registered = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        self.fd = os.open(self.path, flags, 0o600)
        try:
            if create:
                os.ftruncate(self.fd, self.size)
            if os.fstat(self.fd).st_size != self.size:
                raise RuntimeError('Shared expert source size mismatch')
            self.mapping = mmap.mmap(self.fd, self.size, flags=mmap.MAP_SHARED,
                                     prot=mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(self.fd)
        self.address = C.addressof(C.c_char.from_buffer(self.mapping))
        if create:
            bind_pages(self.address, self.size, self.node)
        self.tensor = torch.frombuffer(self.mapping, dtype=torch.bfloat16)
        self.driver = C.CDLL('libcuda.so.1')
        self.driver.cuMemHostRegister_v2.argtypes = [C.c_void_p, C.c_size_t, C.c_uint]
        self.driver.cuMemHostRegister_v2.restype = C.c_int
        self.driver.cuMemHostUnregister.argtypes = [C.c_void_p]
        self.driver.cuMemHostUnregister.restype = C.c_int

    def register(self):
        if self.registered:
            raise RuntimeError('Shared source already registered')
        # PORTABLE enables use by CUDA contexts in this process. Each peer
        # process registers its mapping of the same physical tmpfs pages.
        status = self.driver.cuMemHostRegister_v2(self.address, self.size, 1)
        if status:
            raise RuntimeError(f'cuMemHostRegister_v2 failed: CUDA status {status}')
        self.registered = True
        if not self.tensor.is_pinned():
            raise RuntimeError('PyTorch does not recognize the registered source as pinned')
        stat = self.path.stat()
        return dict(path=str(self.path), device=stat.st_dev, inode=stat.st_ino,
                    mapped_bytes=self.size, unique_bytes_per_node=self.size,
                    pid=os.getpid(), numa_node=self.node, pinned=True,
                    locality=page_nodes(self.address, self.size, self.node),
                    mapping=mapping_receipt(self.address))

    def unregister(self):
        if self.registered:
            status = self.driver.cuMemHostUnregister(self.address)
            if status:
                raise RuntimeError(f'cuMemHostUnregister failed: CUDA status {status}')
            self.registered = False


def build_numa_shared_expert_store(experts, bytes_per_expert, root, rank):
    node = 0 if rank < 2 else 1
    path = Path(root) / f'node{node}.bin'
    ready = path.with_suffix('.ready.json')
    size = len(experts) * bytes_per_expert
    if rank % 2 == 0:
        pool = SharedPinnedMapping(path, size, node, create=True)
        from .pinned_h2d import copy_expert_to_stage
        rows = pool.tensor.view(len(experts), bytes_per_expert // 2)
        for key, tensors in enumerate(experts):
            copy_expert_to_stage(rows[key], tensors, 'torch')
        receipt = pool.register()
        write(ready, receipt)
    else:
        deadline = time.monotonic() + 900
        while not ready.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f'NUMA {node} shared source initialization')
            time.sleep(0.1)
        leader = json.loads(ready.read_text())
        pool = SharedPinnedMapping(path, size, node)
        receipt = pool.register()
        assert receipt['inode'] == leader['inode'] and receipt['device'] == leader['device']
        rows = pool.tensor.view(len(experts), bytes_per_expert // 2)
    sources = [(rows[key],) for key in range(len(experts))]
    receipt.update(source='numa_shared_full_pinned', bytes=size,
                   pair_ranks=[node * 2, node * 2 + 1],
                   physical_unique_global_bytes=size * 2)
    return pool, sources, receipt
