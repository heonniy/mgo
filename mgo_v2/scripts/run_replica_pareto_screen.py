#!/usr/bin/env python3
"""One bounded CPU-only five-budget replay of the frozen exact trace."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[var] = '1'
import resource
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from replica_pareto_cpu import ReplicaReplay, IndependentZeroReplay, ROW_BYTES, EXPERT_BYTES

CAPACITIES = [461, 461, 461, 460]
RHOS = [0, .125, .25, .5, .75]
SUM_FIELDS = ('first_copy_fetches reload_fetches replica_fetches total_fetches expert_h2d_bytes '
              'peer_activation_bytes dispatch_bytes combine_bytes remote_token_rank_pairs '
              'remote_expert_routes raw_expert_routes pre_event_local_exact_hits pre_event_global_hits '
              'global_resident_remote_services final_local_services greedy_peer_bytes_saved '
              'global_miss_expert_events').split()


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows):
    result = {key: sum(r[key] for r in rows) for key in SUM_FIELDS}
    result.update(events=len(rows),
                  mean_duplicate_slot_fraction=float(np.mean([r['duplicate_slots'] / sum(CAPACITIES) for r in rows])),
                  peak_duplicate_slot_fraction=max(r['duplicate_peak_in_event'] for r in rows) / sum(CAPACITIES),
                  mean_unique_resident_experts=float(np.mean([r['unique_resident_experts'] for r in rows])),
                  mean_resident_copies=float(np.mean([r['resident_copies'] for r in rows])))
    routes = result['raw_expert_routes']
    result.update(local_exact_hit_fraction=result['pre_event_local_exact_hits'] / routes,
                  global_resident_remote_service_fraction=result['global_resident_remote_services'] / routes,
                  final_local_service_fraction=result['final_local_services'] / routes,
                  pre_event_global_hit_fraction=result['pre_event_global_hits'] / routes)
    assert result['total_fetches'] == sum(result[k] for k in ('first_copy_fetches', 'reload_fetches', 'replica_fetches'))
    assert result['expert_h2d_bytes'] == result['total_fetches'] * EXPERT_BYTES
    assert result['peer_activation_bytes'] == result['dispatch_bytes'] + result['combine_bytes']
    return result


def pareto(points):
    def xy(p):
        return p['decode']['peer_activation_bytes'], p['decode']['expert_h2d_bytes']
    frontier = [p for p in points if not any(all(a <= b for a, b in zip(xy(q), xy(p))) and xy(q) != xy(p) for q in points)]
    f = min(points, key=lambda p: (xy(p)[1], xy(p)[0], p['rho']))
    c = min(points, key=lambda p: (xy(p)[0], xy(p)[1], p['rho']))
    unique = {xy(p) for p in frontier}
    peer_drop = (xy(f)[0] - xy(c)[0]) / xy(f)[0] if xy(f)[0] else 0
    h2d_rise = (xy(c)[1] - xy(f)[1]) / xy(f)[1] if xy(f)[1] else 0
    knee = None
    if len(unique) >= 3 and xy(f)[0] != xy(c)[0] and xy(f)[1] != xy(c)[1]:
        # Largest distance toward the lower-left from normalized F--C chord.
        interior = [p for p in frontier if xy(p) not in (xy(f), xy(c))]
        def benefit(p):
            x = (xy(p)[0] - xy(c)[0]) / (xy(f)[0] - xy(c)[0])
            y = (xy(p)[1] - xy(f)[1]) / (xy(c)[1] - xy(f)[1])
            return (1 - x - y) / 2**.5
        candidate = max(interior, key=lambda p: (benefit(p), -p['rho']))
        if benefit(candidate) > 0:
            knee = candidate['rho']
    go = len(frontier) >= 3 and len(unique) >= 3 and peer_drop >= .1 and h2d_rise >= .1
    return dict(nondominated_rhos=[p['rho'] for p in frontier], distinct_nondominated_coordinates=len(unique),
                F_rho=f['rho'], C_rho=c['rho'], K_rho=knee,
                F_to_C_decode_peer_reduction_fraction=peer_drop,
                F_to_C_decode_h2d_increase_fraction=h2d_rise,
                decision='GO_FOR_OWNER_REVIEW' if go else 'STOP_FOR_OWNER_REVIEW',
                physical_followup_started=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, default=Path(__file__).resolve().parents[1] / 'experiments/fetch_comm_pareto_p2p_20261002')
    args = parser.parse_args(); packet = args.packet
    started = time.time()
    receipt_path = packet / 'exact_payload_capture_summary.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'PASS' and receipt['decode_forwards'] == 8
    assert receipt['events'] == 432 and receipt['substitution_count'] == 0
    assert receipt['hidden_size'] * receipt['element_bytes'] == ROW_BYTES
    for raw in receipt['raw_receipts']:
        assert Path(raw['path']).stat().st_size == raw['bytes']
        assert sha(raw['path']) == raw['sha256']
    source_paths = [Path(__file__), Path(__file__).with_name('replica_pareto_cpu.py'),
                    Path(__file__).with_name('test_replica_pareto_cpu.py'),
                    packet / 'REPLICA_PARETO_SCREEN.md', packet / 'REPLICA_REPLAY_PROTOCOL.md', receipt_path]
    provenance = dict(plan_commit='93556b94003137f43196876f630206457fc8d844',
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=packet, text=True).strip(),
                      capture_result_commit='ed7f82b6a323460c18374c45f24018bc48a899f4',
                      input_receipts=receipt['raw_receipts'],
                      source_sha256={str(p.relative_to(packet.parents[2])): sha(p) for p in source_paths},
                      python=sys.version, numpy=np.__version__, address_space_limit_bytes=2 * 1024**3,
                      threads=1, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                      new_model_runs=0, new_gpu_runs=0)
    raw = json.loads(Path(receipt['raw_receipts'][0]['path']).read_text())
    assert raw['status'] == 'PASS' and len(raw['events']) == 432
    events = []
    for i, event in enumerate(raw['events']):
        assert event['event'] == i and event['layer'] == i % 48 and event['step'] == i // 48
        selected = np.asarray(event['raw_selected_experts'], dtype=np.int64)
        origins = np.asarray(event['origin_ranks'], dtype=np.int64)
        assert selected.shape == (len(origins), 8) and np.all((selected >= 0) & (selected < 128))
        if i >= 48:
            assert len(origins) == 32 and np.array_equal(np.bincount(origins, minlength=4), [8] * 4)
        events.append((event['layer'], origins, selected))
    del raw
    rows = []; points = []; checked_zero = 0
    status = dict(status='RUNNING', capacities=CAPACITIES, global_slots=sum(CAPACITIES), rhos=RHOS,
                  row_bytes=ROW_BYTES, expert_bytes=EXPERT_BYTES, provenance=provenance, points=points)
    write_json(packet / 'replica_pareto_screen.json', status)
    try:
        for rho in RHOS:
            t0 = time.time(); replay = ReplicaReplay(CAPACITIES, rho)
            reference = IndependentZeroReplay(CAPACITIES) if rho == 0 else None
            one = []
            for i, (layer, origins, selected) in enumerate(events):
                row, dest, tr, operations = replay.event(layer, origins, selected)
                if reference is not None:
                    state, rd, sd, sc, first, reloads = reference.event(layer, origins, selected)
                    assert replay.state() == state
                    assert np.array_equal(dest, rd) and np.array_equal(tr['dispatch'], sd) and np.array_equal(tr['combine'], sc)
                    assert (row['first_copy_fetches'], row['reload_fetches']) == (first, reloads)
                    assert row['duplicate_slots'] == 0 and row['replica_fetches'] == 0
                    checked_zero += 1
                row = dict(rho=rho, event=i, step=i // 48, layer=layer,
                           phase='prefill' if i < 48 else 'decode', **row)
                one.append(row)
                if (i + 1) % 48 == 0:
                    print(json.dumps(dict(rho=rho, events=i + 1, elapsed_seconds=round(time.time() - t0, 2))), flush=True)
            point = dict(rho=rho, duplicate_cap=replay.cap, full=aggregate(one),
                         prefill=aggregate(one[:48]), decode=aggregate(one[48:]),
                         final_state_sha256=hashlib.sha256(repr(replay.state()).encode()).hexdigest(),
                         elapsed_seconds=time.time() - t0)
            points.append(point); rows.extend(one)
            status['completed_rhos'] = [p['rho'] for p in points]
            write_csv(packet / 'replica_pareto_events.csv', rows)
            write_json(packet / 'replica_pareto_screen.json', status)
            print('COMPLETED ' + json.dumps(point), flush=True)
        assert checked_zero == 432 and len(rows) == 2160
        assert 'torch' not in sys.modules
        status.update(status='PASS', pareto=pareto(points), elapsed_seconds=time.time() - started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      validation=dict(independent_rho_zero_events=checked_zero, total_events=len(rows),
                                      exact_destinations=True, per_rank_capacity=True, duplicate_cap=True,
                                      deterministic_lru_and_ties=True, send_receive_transpose=True,
                                      source_hashes_verified=True, torch_imported=False))
        flat = [dict(rho=p['rho'], duplicate_cap=p['duplicate_cap'], phase=phase, **p[phase])
                for p in points for phase in ('full', 'prefill', 'decode')]
        write_csv(packet / 'replica_pareto_screen.csv', flat)
        write_json(packet / 'replica_pareto_screen.json', status)
        print('FINAL ' + json.dumps(status['pareto']), flush=True)
    except BaseException as exc:
        status.update(status='FAIL', error=repr(exc), elapsed_seconds=time.time() - started)
        if rows: write_csv(packet / 'replica_pareto_events.csv', rows)
        write_json(packet / 'replica_pareto_screen.json', status)
        raise


if __name__ == '__main__':
    main()
