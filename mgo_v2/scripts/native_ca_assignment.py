"""Build and load the exact, CPU-only CA Hungarian kernel before measurement."""
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess


SOURCE = Path(__file__).with_suffix('.cpp')
FLAGS = ('-O3', '-std=c++17', '-shared', '-fPIC')


def load_kernel():
    digest = hashlib.sha256(SOURCE.read_bytes() + ' '.join(FLAGS).encode()).hexdigest()[:16]
    cache = Path.home() / '.cache' / 'mgo_native_ca'
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f'ca_flow_{digest}.so'
    with (cache / f'{digest}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = cache / f'{library.name}.{os.getpid()}.tmp'
            try:
                subprocess.run(['g++', *FLAGS, str(SOURCE), '-o', str(temporary)],
                               check=True, capture_output=True, text=True)
                temporary.replace(library)
            finally:
                temporary.unlink(missing_ok=True)
    module = ctypes.CDLL(str(library))
    kernel = module.mgo_ca_min_cost_flow
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    kernel.restype = ctypes.c_int
    return kernel


ca_min_cost_flow = load_kernel()


from functools import lru_cache
from types import FunctionType
from numba import njit
import numpy as np


@njit
def native_balanced_assignment(demand, experts, world, randomized, quota=None):
    assert quota is None  # Legacy CA_NATIVE keeps its frozen rank-order quota.
    assert not randomized and demand.shape == (128, world)
    assert demand.flags.c_contiguous and experts.flags.c_contiguous
    assignment = np.empty(len(experts), np.int64)
    status = ca_min_cost_flow(demand.ctypes.data, world, experts.ctypes.data,
                              len(experts), world, assignment.ctypes.data)
    assert status == 0
    return assignment


@lru_cache(maxsize=1)
def fast_ca_step(legacy_step):
    globals_copy = legacy_step.py_func.__globals__.copy()
    globals_copy['balanced_assignment'] = native_balanced_assignment
    original = legacy_step.py_func
    clone = FunctionType(original.__code__, globals_copy,
                         'step_with_native_ca', original.__defaults__, original.__closure__)
    clone.__module__ = original.__module__
    return njit(cache=False)(clone)
