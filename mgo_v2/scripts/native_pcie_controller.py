"""Thin array/ABI adapter to the C++ PCIe controller; build before timing."""
import ctypes as C
import fcntl
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

SOURCE = Path(__file__).with_suffix('.cpp')
FLAGS = ('-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off')


class State(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in
                ('capacities', 'slots', 'owner', 'primary', 'last', 'seen', 'lost', 'birth', 'reuses', 'gates')] + [('stride', C.c_int)]


class Output(C.Structure):
    _fields_ = [(name, C.c_void_p) for name in
                ('targets', 'effective', 'masses', 'lengths', 'destinations', 'fetches', 'quotas', 'row')]


@lru_cache(None)
def load_engine():
    digest = hashlib.sha256(SOURCE.read_bytes() + ' '.join(FLAGS).encode()).hexdigest()[:16]
    root = Path('/data2/esjung/cache/native_pcie_controller')
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'controller_{digest}.so'
    with (root / f'{digest}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            temporary = path.with_suffix(f'.{os.getpid()}.tmp')
            try:
                subprocess.run(['g++', *FLAGS, str(SOURCE), '-o', str(temporary)], check=True, capture_output=True, text=True)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    library = C.CDLL(str(path))
    library.mgo_pcie_create.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p]
    library.mgo_pcie_create.restype = C.c_void_p
    library.mgo_pcie_destroy.argtypes = [C.c_void_p]
    library.mgo_pcie_trace.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
    library.mgo_pcie_trace.restype = C.c_int
    library.mgo_pcie_step.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int,
                                   C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                                   C.POINTER(State), C.POINTER(Output)]
    library.mgo_pcie_step.restype = C.c_int
    library.mgo_pcie_quotas.argtypes = [C.c_int, C.c_int, C.c_int, C.c_void_p]
    library.mgo_pcie_quotas.restype = C.c_int
    return library


def native_quotas(m, group_balanced=False, event=0):
    result = np.empty(4, np.int64)
    status = load_engine().mgo_pcie_quotas(m, int(group_balanced), event, result.ctypes.data)
    if status:
        raise ValueError('Invalid native quota input')
    return result


class NativePcieController:
    def __init__(self, policy, seed, quota_mode, peer_costs):
        if len(policy.capacities) != 4 or policy.substitution or policy.policy not in (0, 1, 7):
            raise ValueError('Native PCIe controller requires exact-only R4 BR/CA/Near')
        if quota_mode not in ('rank_order', 'group_balanced'):
            raise ValueError('Unknown PCIe quota mode')
        self.library = load_engine()
        self.costs = None
        if peer_costs is not None:
            self.costs = np.ascontiguousarray(peer_costs, dtype=np.int64)
            if (policy.policy != 1 or self.costs.shape != (4, 4) or
                np.any(self.costs < 0) or np.any(np.diag(self.costs)) or np.max(self.costs) > 1000000):
                raise ValueError('Invalid measured peer-cost matrix')
        self.handle = self.library.mgo_pcie_create(seed, policy.policy, int(quota_mode == 'group_balanced'),
                       None if self.costs is None else self.costs.ctypes.data)
        if not self.handle:
            raise RuntimeError('Native PCIe controller creation failed')
        self.state = State(*(getattr(policy, name).ctypes.data for name in
                           ('capacities', 'slots', 'owner', 'primary', 'last', 'seen', 'lost', 'birth', 'reuses', 'gates')),
                           policy.slots.shape[1])
        self.capacity = 0
        self.topk = 0
        self.targets = np.arange(128, dtype=np.int64)
        self.fetches = np.empty((128, 5), np.int64)
        self.quotas = np.empty(4, np.int64)
        self.row = np.empty(48, np.float64)

    def apply(self, event, selected, weights, origins, gates):
        selected = np.ascontiguousarray(selected, dtype=np.int64)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        origins = np.ascontiguousarray(origins, dtype=np.int64)
        gates = np.ascontiguousarray(gates, dtype=np.float32)
        n, topk = selected.shape
        if weights.shape != selected.shape or origins.shape != (n,) or gates.shape != (128,):
            raise ValueError('Native controller metadata shape mismatch')
        if n > self.capacity or topk != self.topk:
            self.capacity = max(n, self.capacity, 1)
            self.topk = topk
            self.effective = np.empty((self.capacity, topk), np.int16)
            self.masses = np.empty((self.capacity, topk), np.float64)
            self.lengths = np.empty(self.capacity, np.int8)
            self.destinations = np.empty((self.capacity, topk), np.int8)
            self.output = Output(*(getattr(self, name).ctypes.data for name in
                                 ('targets', 'effective', 'masses', 'lengths', 'destinations', 'fetches', 'quotas', 'row')))
        count = self.library.mgo_pcie_step(self.handle, event, n, topk,
                    selected.ctypes.data, weights.ctypes.data, origins.ctypes.data,
                    gates.ctypes.data, C.byref(self.state), C.byref(self.output))
        if count < 0:
            raise RuntimeError(f'Native PCIe controller failed with status {count}')
        return (self.targets, self.effective[:n], self.masses[:n], self.lengths[:n],
                self.destinations[:n], self.fetches[:count], self.row)

    def close(self):
        if self.handle:
            self.library.mgo_pcie_destroy(self.handle)
            self.handle = None

    def trace(self):
        count = self.library.mgo_pcie_trace(self.handle, None, 0)
        if count < 0:raise RuntimeError('Native trace length failed')
        output = np.empty((count, 61), np.float64)
        assert self.library.mgo_pcie_trace(self.handle, output.ctypes.data, count) == count
        return output

    def __del__(self):
        if getattr(self, 'handle', None):
            self.close()
