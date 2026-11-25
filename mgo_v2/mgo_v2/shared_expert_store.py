"""One host-shared, page-locked routed-expert source for all local ranks.

Approved deviation from the rank-private full pinned copy (which would need
world x the expert bytes of host memory): an offline prep step writes every
routed expert once into a tmpfs file (row-major, key = layer*experts+expert,
row = gate|up|down, each in its checkpoint (out,in) layout, flattened).  Every
rank maps that file MAP_SHARED and registers it with cudaHostRegister, so the
PriorityH2DScheduler keeps its direct pinned-source DMA path (no staging copy).

The manifest is written last by the builder; a store without a manifest, or
whose manifest disagrees with the checkpoint identity/geometry, is refused.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

FORMAT = 'mgo-shared-expert-rows-bf16-v1'
DATA = 'experts.bf16'
MANIFEST = 'manifest.json'
DEFAULT_ROOT = Path('/dev/shm/mgo_qwen235_expert_store_v1')
SAMPLE_ROWS = 16


def sample_rows(rows, count=SAMPLE_ROWS):
    """Deterministic spread of rows whose sha256 is checked at every open."""
    return sorted({int(x) for x in np.linspace(0, rows - 1, count)})


def row_digest(buffer):
    return hashlib.sha256(memoryview(buffer)).hexdigest()


def _identity_without_mtime(identity):
    # Shard bytes and index/config digests identify the checkpoint content.
    return dict(model=identity['model'], index_sha256=identity['index_sha256'],
                config_sha256=identity['config_sha256'],
                shards={k: dict(bytes=v['bytes']) for k, v in identity['shards'].items()})


class SharedPinnedStore:
    """Mapped and registered store. ``pool`` is a (rows, elements) BF16 tensor."""

    def __init__(self, pool, sources, receipt, host_ptr):
        self.pool = pool
        self.sources = sources
        self.receipt = receipt
        self._host_ptr = host_ptr

    def close(self):
        if self._host_ptr is not None:
            import torch
            torch.cuda.synchronize()
            rc = torch.cuda.cudart().cudaHostUnregister(self._host_ptr)
            self._host_ptr = None
            if int(rc) != 0:
                raise RuntimeError(f'cudaHostUnregister failed: {rc}')


def open_shared_store(spec, root=None, verify_sample=True):
    """Map the shared store and page-lock it for direct H2D DMA on this rank."""
    import torch
    from .model_loader import checkpoint_identity
    root = Path(root or os.environ.get('MGO_SHARED_EXPERT_STORE', DEFAULT_ROOT))
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f'shared expert store has no committed manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != FORMAT or manifest.get('status') != 'COMPLETE':
        raise ValueError('incomplete or foreign shared expert store')
    rows, row_bytes = spec.keys, spec.expert_bytes
    if (manifest['rows'], manifest['row_bytes'], tuple(manifest['geometry'])) != (rows, row_bytes, spec.geometry):
        raise ValueError('shared expert store geometry disagrees with the active model')
    identity, _ = checkpoint_identity(spec.path)
    if _identity_without_mtime(identity) != _identity_without_mtime(manifest['identity']):
        raise ValueError('shared expert store was built from a different checkpoint')
    data = root / DATA
    total = rows * row_bytes
    if data.stat().st_size != total:
        raise ValueError('shared expert store size mismatch')
    started = time.perf_counter()
    flat = torch.from_file(str(data), shared=True, size=total // 2, dtype=torch.bfloat16)
    pool = flat.view(rows, row_bytes // 2)
    mapped = time.perf_counter()
    sample = {}
    if verify_sample:
        raw = pool.view(torch.uint8)
        for row in sample_rows(rows):
            digest = row_digest(raw[row].numpy())
            if digest != manifest['sample_sha256'][str(row)]:
                raise ValueError(f'shared expert store row {row} digest mismatch')
            sample[str(row)] = digest
    verified = time.perf_counter()
    ptr = pool.data_ptr()
    # cudaHostRegisterDefault: page-lock the existing shared mapping in place.
    rc = torch.cuda.cudart().cudaHostRegister(ptr, total, 0)
    if int(rc) != 0:
        raise RuntimeError(f'cudaHostRegister of the shared expert store failed: {rc}')
    registered = time.perf_counter()
    if not pool.is_pinned():
        torch.cuda.cudart().cudaHostUnregister(ptr)
        raise RuntimeError('registered shared expert store is not reported as pinned')
    sources = [(pool[key],) for key in range(rows)]
    receipt = dict(source='shared_pinned', shared=True, path=str(data),
                   manifest=str(manifest_path), bytes=total, rows=rows, row_bytes=row_bytes,
                   map_seconds=mapped - started, sample_verify_seconds=verified - mapped,
                   register_seconds=registered - verified,
                   init_seconds=registered - started, verified_sample_rows=len(sample),
                   cuda_host_register_flags=0, store_build=manifest.get('build'),
                   checkpoint_index_sha256=manifest['identity']['index_sha256'])
    return SharedPinnedStore(pool, sources, receipt, ptr)
