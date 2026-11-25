"""Offline build of the host-shared routed-expert store (see mgo_v2/shared_expert_store.py).

Reads routed expert tensors directly from the safetensors shards and writes
row key=layer*experts+expert, row = gate|up|down (BF16, checkpoint layout,
flattened) into one tmpfs file. Space is reserved with posix_fallocate before
any copy, so an over-full tmpfs fails immediately instead of SIGBUS mid-copy.
The manifest (with checkpoint identity) is committed LAST.

Usage:
  build_shared_expert_store.py --model /home/hwlee/model/Qwen3-235B-A22B-Instruct-2507 \
      --out /dev/shm/mgo_qwen235_expert_store_v1 [--workers 16]
Cleanup (owner decision, never automatic):
  rm -rf /dev/shm/mgo_qwen235_expert_store_v1
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mgo_v2.model_spec import spec_from_config  # noqa: E402
from mgo_v2.shared_expert_store import (DATA, FORMAT, MANIFEST, row_digest,  # noqa: E402
                                        sample_rows)

EXPERT = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
PART = {'gate_proj': 0, 'up_proj': 1, 'down_proj': 2}


def copy_shard(job):
    model, shard, names, data_path, spec_tuple = job
    import torch
    from safetensors import safe_open
    layers, hidden, inter, experts = spec_tuple
    p = hidden * inter
    row_elems = 3 * p
    out = np.memmap(data_path, dtype=np.uint16, mode='r+', shape=(layers * experts, row_elems))
    seen = []
    with safe_open(str(Path(model) / shard), framework='pt', device='cpu') as f:
        for name in names:
            m = EXPERT.fullmatch(name)
            layer, expert, part = int(m[1]), int(m[2]), PART[m[3]]
            t = f.get_tensor(name)
            expected = (hidden, inter) if part == 2 else (inter, hidden)
            assert t.dtype == torch.bfloat16 and tuple(t.shape) == expected, (name, t.dtype, t.shape)
            key = layer * experts + expert
            out[key, part * p:(part + 1) * p] = t.contiguous().view(torch.int16).numpy().reshape(-1).view(np.uint16)
            seen.append((key, part))
    out.flush()
    del out
    return seen


def main(a):
    from mgo_v2.model_loader import checkpoint_identity
    spec = spec_from_config(a.model)
    identity, weight_map = checkpoint_identity(a.model)
    rows, row_bytes = spec.keys, spec.expert_bytes
    total = rows * row_bytes
    expert_names = {n: s for n, s in weight_map.items() if EXPERT.fullmatch(n)}
    if len(expert_names) != rows * 3:
        raise SystemExit(f'expected {rows*3} routed expert tensors, found {len(expert_names)}')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'refusing to overwrite nonempty store {out}')
    out.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(out)
    free = st.f_bavail * st.f_frsize
    margin = a.margin_gib * 2**30
    print(json.dumps(dict(event='SPACE', store_bytes=total, free_bytes=free, margin_bytes=margin)), flush=True)
    if free < total + margin:
        raise SystemExit(f'INSUFFICIENT SPACE on {out}: need {total + margin} B '
                         f'({(total + margin) / 2**30:.1f} GiB incl. margin), free {free} B ({free / 2**30:.1f} GiB)')
    data = out / DATA
    started = time.time()
    fd = os.open(data, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.posix_fallocate(fd, 0, total)
    finally:
        os.close(fd)
    reserved = time.time()
    print(json.dumps(dict(event='RESERVED', seconds=reserved - started)), flush=True)
    by_shard = {}
    for name, shard in expert_names.items():
        by_shard.setdefault(shard, []).append(name)
    jobs = [(a.model, shard, sorted(names), str(data),
             (spec.layers, spec.hidden, spec.moe_intermediate, spec.experts))
            for shard, names in sorted(by_shard.items())]
    loaded = np.zeros((rows, 3), np.bool_)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for i, seen in enumerate(pool.imap_unordered(copy_shard, jobs), 1):
            for key, part in seen:
                assert not loaded[key, part]
                loaded[key, part] = True
            if i % 8 == 0 or i == len(jobs):
                print(json.dumps(dict(event='SHARDS', done=i, total=len(jobs),
                                      seconds=time.time() - started)), flush=True)
    if not loaded.all():
        raise SystemExit('incomplete routed expert store')
    copied = time.time()
    raw = np.memmap(data, dtype=np.uint8, mode='r', shape=(rows, row_bytes))
    sample = {str(r): row_digest(raw[r]) for r in sample_rows(rows)}
    del raw
    manifest = dict(format=FORMAT, status='COMPLETE', identity=identity, geometry=list(spec.geometry),
                    spec=spec.receipt(), rows=rows, row_bytes=row_bytes, bytes=total,
                    layout='key=layer*experts+expert; row=gate(I,H)|up(I,H)|down(H,I) BF16 flattened',
                    sample_sha256=sample,
                    build=dict(started_unix=started, reserve_seconds=reserved - started,
                               copy_seconds=copied - reserved, total_seconds=time.time() - started,
                               workers=a.workers, host=os.uname().nodename),
                    cleanup=f'rm -rf {out}')
    tmp = out / (MANIFEST + '.tmp')
    tmp.write_text(json.dumps(manifest, indent=2) + '\n')
    tmp.replace(out / MANIFEST)
    print(json.dumps(dict(event='COMPLETE', bytes=total, seconds=time.time() - started, path=str(data))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--margin-gib', type=int, default=32)
    main(p.parse_args())
