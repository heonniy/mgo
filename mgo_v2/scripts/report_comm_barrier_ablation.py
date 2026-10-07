"""Audit and summarize matched BR/CA token-collective barrier experiments."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic_events(path, ablated):
    d = read(path)
    assert d['status'] == 'PASS' and d['token_prefix_parity'] and d['decode_steps'] == 16
    grouped = defaultdict(list)
    for row in d['segments']:
        if 48 <= row['event_index'] < 17 * 48:
            grouped[row['event_index']].append(row)
    assert len(grouped) == 16 * 48
    events = {}
    for event, rows in grouped.items():
        by_phase = defaultdict(float)
        for row in rows:
            by_phase[row['phase']] += row['stream_seconds']
        forward = by_phase['forward_token_a2a_submit'] + by_phase['moe.forward_complete']
        returned = by_phase['return_token_a2a']
        barrier_dispatch = barrier_return = 0.0
        h2d_wait = by_phase['required_h2d_exposed_wait']
        if ablated:
            barriers = [i for i, row in enumerate(rows) if row['phase'] == 'global_barrier_wait']
            assert len(barriers) == 2, (event, barriers)
            dispatch, back = barriers
            assert any(row['phase'] == 'moe.ablation_required_h2d_complete' for row in rows)
            assert any(row['phase'] == 'moe.ablation_dispatch_local_ready' for row in rows)
            assert any(row['phase'] == 'moe.ablation_return_local_ready' for row in rows)
            barrier_dispatch = rows[dispatch]['stream_seconds']
            barrier_return = rows[back]['stream_seconds']
            forward = sum(row['stream_seconds'] for i, row in enumerate(rows)
                          if i > dispatch and row['phase'] in ('forward_token_a2a_submit', 'moe.forward_complete'))
            returned = sum(row['stream_seconds'] for i, row in enumerate(rows)
                           if i > back and row['phase'] == 'return_token_a2a')
            assert forward > 0 and returned > 0
        events[event] = dict(forward_collective_path=forward, return_collective_path=returned,
                             dispatch_barrier_wait=barrier_dispatch, return_barrier_wait=barrier_return,
                             exposed_h2d_wait=h2d_wait, expert=by_phase['moe.expert_compute'],
                             controller=by_phase['placement_controller_cpu'],
                             pre_return_local_pack=by_phase['moe.return_a2a'])
    return events


def summarize(normal, ablated):
    roots = {'normal': normal, 'ablation_OURS': ablated}
    traces = {}
    for rank in range(4):
        left = read(normal / f'trace_rank{rank}.json')
        right = read(ablated / f'trace_rank{rank}.json')
        assert left['route_sha256'] == right['route_sha256']
        assert left['teacher_tokens'] == right['teacher_tokens']
        traces[str(rank)] = left['route_sha256']
    modes = {}
    for label, root in roots.items():
        status = read(root / 'status.json')
        result = read(root / 'result.json')
        assert status['status'] == result['status'] == 'PASS'
        assert result['capture_policy'] == 'LA_CA_NEAR' and result['policy_pair'] == ['BR', 'CA']
        assert result['prefetch'] == 'off' and result['decode_steps'] == 63
        assert result['collective_barrier_ablation'] == (label == 'ablation_OURS')
        policies = {}
        for policy in ('BR', 'CA'):
            primary = [read(root / f'run{run}.json') for run in range(4)
                       if read(root / f'run{run}.json')['backend'] == policy]
            assert len(primary) == 2 and all(row['status'] == 'PASS' for row in primary)
            rank_events = {}
            native_waves = {}
            for rank in range(4):
                native_waves[str(rank)] = []
                for run in range(4):
                    row = read(root / f'run{run}_rank{rank}.json')
                    expected = 63 * 48 if label == 'ablation_OURS' else 0
                    assert row['dispatch_barriers'] == row['return_barriers'] == expected
                    assert row['finite_logits'] and row['no_compile'] and row['frozen_parity']
                    if row['backend'] == policy:
                        assert row['native_waves'] > 0 and row['native_groups'] > 0
                        native_waves[str(rank)].append(row['native_waves'])
                rank_events[rank] = diagnostic_events(root / f'diagnostic_{policy}_rank{rank}.json',
                                                       label == 'ablation_OURS')
            assert all(set(rows) == set(rank_events[0]) for rows in rank_events.values())
            per_event_critical = {}
            for metric in next(iter(rank_events[0].values())):
                per_event_critical[metric] = [max(rank_events[r][e][metric] for r in range(4))
                                              for e in sorted(rank_events[0])]
            policies[policy] = dict(
                primary=primary,
                mean_tpot_s=mean(row['TPOT'] for row in primary),
                mean_ttft_s=mean(row['TTFT'] for row in primary),
                decode_peer_bytes=primary[0]['decode_peer_bytes'],
                decode_h2d_bytes=primary[0]['decode_h2d_bytes'],
                native_waves_per_rank=native_waves,
                diagnostic_critical_ms_per_decode={metric: sum(values) * 1000 / 16
                                                   for metric, values in per_event_critical.items()},
                diagnostic_rank_ms_per_decode={str(r): {metric: sum(event[metric] for event in rank_events[r].values()) * 1000 / 16
                                                         for metric in next(iter(rank_events[r].values()))}
                                               for r in range(4)},
                diagnostic_sha256={str(r): sha(root / f'diagnostic_{policy}_rank{r}.json') for r in range(4)},
            )
        modes[label] = dict(source=str(root), source_commit=status['source_commit'], policies=policies)
    for rank in range(4):
        for run in range(4):
            left = read(normal / f'run{run}_rank{rank}.json')
            right = read(ablated / f'run{run}_rank{rank}.json')
            assert left['backend'] == right['backend']
            for key in ('validation', 'decode_bytes', 'hit', 'tokens', 'argmax_hash'):
                assert left[key] == right[key], (rank, run, key)
    return dict(status='PASS', cell='R4_C30_B16_L256_O64', trace_sha256=traces,
                definition='Uninstrumented 63-decode primary TPOT; separate 16-decode diagnostic. Critical ms/decode sums the maximum rank span per layer. Collective-path spans begin after the ablation barrier and include NCCL launch/completion and residual host/peer waiting, not wire-only kernel latency.',
                modes=modes)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('normal', type=Path)
    p.add_argument('ablated', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(summarize(a.normal, a.ablated), indent=2) + '\n')
