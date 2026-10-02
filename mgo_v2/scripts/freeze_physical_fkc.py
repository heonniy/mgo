#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import gzip
import hashlib
import json
from pathlib import Path
import resource
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
import sys
import time
import numpy as np
from replica_pareto_cpu import ReplicaReplay
from frozen_replica_schedule import FrozenReplicaState, compile_event

ROOT = Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/physical_fkc_20261003')
PACKET = Path(__file__).resolve().parents[1] / 'experiments/fetch_comm_pareto_p2p_20261002'
CAPACITIES = [461, 461, 461, 460]


def sha(path):
    with path.open('rb') as f:
        h = hashlib.file_digest(f, 'sha256')
    return h.hexdigest()


def main():
    start = time.time(); ROOT.mkdir(exist_ok=False)
    receipts = json.loads((PACKET / 'exact_payload_capture_summary.json').read_text())
    cpu = json.loads((PACKET / 'replica_pareto_screen.json').read_text())
    for name in ('replica_pareto_cpu.py',):
        path = Path(__file__).with_name(name)
        assert sha(path) == cpu['provenance']['source_sha256']['mgo_v2/scripts/' + name]
    for item in receipts['raw_receipts']: assert sha(Path(item['path'])) == item['sha256']
    source = json.loads(Path(receipts['raw_receipts'][0]['path']).read_text())
    metadata = dict(status='PASS', plan_commit='145b3e8d53e08b23a4bda9c098c16c5ca406f92a',
                    cpu_result_commit='dc7b099d006344d862cbcd9df4e0f6f352a1ba1f', capacities=CAPACITIES,
                    raw_receipts=receipts['raw_receipts'], schedules={},
                    source_sha256={n: sha(Path(__file__).with_name(n)) for n in
                                   ('replica_pareto_cpu.py', 'frozen_replica_schedule.py', 'freeze_physical_fkc.py')})
    for label, rho in (('F', 0), ('K', .25), ('C', .75)):
        replay = ReplicaReplay(CAPACITIES, rho); applied = FrozenReplicaState(CAPACITIES, replay.cap)
        path = ROOT / f'schedule_{label}.jsonl.gz'
        totals = {}; phases = dict(prefill={}, decode={})
        with path.open('xb') as output, gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as compressed:
            for i, raw in enumerate(source['events']):
                assert raw['event'] == i and raw['layer'] == i % 48
                origins = np.asarray(raw['origin_ranks'], dtype=np.int64)
                selected = np.asarray(raw['raw_selected_experts'], dtype=np.int64)
                seen = set(replay.seen)
                row, ds, traffic, ops = replay.event(raw['layer'], origins, selected)
                event = compile_event(i, raw['layer'], origins, selected, replay, row, ds, traffic, ops, seen)
                applied.apply(event)
                assert applied.state() == replay.state()
                compressed.write((json.dumps(event, separators=(',', ':')) + '\n').encode())
                for key in ('first_copy_fetches', 'reload_fetches', 'replica_fetches', 'total_fetches',
                            'expert_h2d_bytes', 'peer_activation_bytes', 'remote_token_rank_pairs'):
                    totals[key] = totals.get(key, 0) + row[key]
                    p = phases['prefill' if i < 48 else 'decode']; p[key] = p.get(key, 0) + row[key]
                if (i + 1) % 48 == 0: print(label, i + 1, flush=True)
        expected = next(p for p in cpu['points'] if p['rho'] == rho)
        assert all(expected['full'][k] == v for k, v in totals.items())
        assert all(expected[phase][k] == v for phase, data in phases.items() for k, v in data.items())
        metadata['schedules'][label] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                                          rho=rho, duplicate_cap=replay.cap, events=432,
                                          action_state_parity_events=432, cpu_screen_total_parity=True,
                                          full=totals, **phases)
        path.chmod(0o444)
    metadata.update(elapsed_seconds=time.time() - start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    assert 'torch' not in sys.modules
    (PACKET / 'physical_fkc_schedule_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__': main()
