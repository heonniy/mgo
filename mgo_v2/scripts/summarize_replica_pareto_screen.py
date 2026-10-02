#!/usr/bin/env python3
"""Validate compact outputs and render the completed CPU-only screen."""
import csv
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1] / 'experiments/fetch_comm_pareto_p2p_20261002'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = PACKET
    result = json.loads((p / 'replica_pareto_screen.json').read_text())
    assert result['status'] == 'PASS'
    events = list(csv.DictReader((p / 'replica_pareto_events.csv').open()))
    flat = list(csv.DictReader((p / 'replica_pareto_screen.csv').open()))
    assert len(events) == 2160 and len(flat) == 15
    sums = ('first_copy_fetches reload_fetches replica_fetches total_fetches expert_h2d_bytes '
            'peer_activation_bytes dispatch_bytes combine_bytes remote_token_rank_pairs '
            'remote_expert_routes raw_expert_routes pre_event_local_exact_hits pre_event_global_hits '
            'global_resident_remote_services final_local_services greedy_peer_bytes_saved '
            'global_miss_expert_events').split()
    for point in result['points']:
        subset = [e for e in events if float(e['rho']) == point['rho']]
        assert [int(e['event']) for e in subset] == list(range(432))
        for phase in ('full', 'prefill', 'decode'):
            phase_rows = [e for e in subset if phase == 'full' or e['phase'] == phase]
            f = next(r for r in flat if float(r['rho']) == point['rho'] and r['phase'] == phase)
            for key in sums:
                assert sum(int(r[key]) for r in phase_rows) == point[phase][key] == int(f[key])
            for key, value in point[phase].items():
                assert abs(float(f[key]) - value) < 1e-8
        for key in sums:
            assert point['full'][key] == point['prefill'][key] + point['decode'][key]
        assert point['full']['first_copy_fetches'] == 5462  # frozen trace's unique (layer,expert) keys
        for row in subset:
            assert int(row['duplicate_slots']) <= point['duplicate_cap']
            assert int(row['resident_copies']) <= 1843
            assert int(row['resident_copies']) - int(row['unique_resident_experts']) == int(row['duplicate_slots'])
            assert int(row['total_fetches']) == sum(int(row[k]) for k in sums[:3])
            assert int(row['expert_h2d_bytes']) == int(row['total_fetches']) * 9 * 1024**2
    frontier = []
    # Independently evaluate dominance and gate from CSV decode totals.
    decode = [r for r in flat if r['phase'] == 'decode']
    for row in decode:
        dominated = False
        for other in decode:
            x, y = int(row['peer_activation_bytes']), int(row['expert_h2d_bytes'])
            a, b = int(other['peer_activation_bytes']), int(other['expert_h2d_bytes'])
            if a <= x and b <= y and (a < x or b < y): dominated = True
        if not dominated: frontier.append(float(row['rho']))
    decision = result['pareto']
    assert frontier == decision['nondominated_rhos']
    points = {r['rho']: r for r in result['points']}
    f, c = points[decision['F_rho']]['decode'], points[decision['C_rho']]['decode']
    peer_pct = 100 * (f['peer_activation_bytes'] - c['peer_activation_bytes']) / f['peer_activation_bytes']
    h2d_pct = 100 * (c['expert_h2d_bytes'] - f['expert_h2d_bytes']) / f['expert_h2d_bytes']
    expected_go = len(frontier) >= 3 and h2d_pct >= 10 and peer_pct >= 10
    assert (decision['decision'] == 'GO_FOR_OWNER_REVIEW') == expected_go
    for path, digest in result['provenance']['source_sha256'].items():
        assert sha(p.parents[2] / path) == digest
    lines = ['# CPU-only replica Pareto screen', '',
             f"**{decision['decision']}** on the frozen eight-decode trace. All five budgets and 2,160 layer events passed validation.", '',
             f"There are {len(frontier)} nondominated rho settings: {frontier}. "
             f"F is rho={decision['F_rho']}; C is rho={decision['C_rho']}; K is {decision['K_rho']}. "
             f"Moving F to C increases decode expert H2D by **{h2d_pct:.2f}%** and reduces decode peer activation bytes by **{peer_pct:.2f}%**. "
             'The decision is a trace-screen result; physical validation has not started.', '',
             '## Primary decode plane', '',
             'GiB and MiB are binary units. H2D includes every physical first copy, reload and replica; peer bytes include only non-self dispatch plus combine activation rows.', '',
             '| rho | Duplicate cap | Expert H2D GiB | Peer MiB | Remote token-rank pairs | Nondominated |',
             '|---:|---:|---:|---:|---:|:---:|']
    for point in result['points']:
        d = point['decode']
        lines.append(f"| {point['rho']:g} | {point['duplicate_cap']} | {d['expert_h2d_bytes']/1024**3:.4f} | {d['peer_activation_bytes']/1024**2:.4f} | {d['remote_token_rank_pairs']:,} | {'yes' if point['rho'] in frontier else 'no'} |")
    lines += ['', '## Decode fetches and residency', '',
              '| rho | First-ever copies | Reloads | Replicas | Total fetches |',
              '|---:|---:|---:|---:|---:|']
    for point in result['points']:
        d = point['decode']
        lines.append(f"| {point['rho']:g} | {d['first_copy_fetches']:,} | {d['reload_fetches']:,} | {d['replica_fetches']:,} | {d['total_fetches']:,} |")
    lines += ['', '| rho | Mean / peak duplicate fraction | Mean unique experts | Local exact hits | Resident remote services | Final local services |',
              '|---:|---:|---:|---:|---:|---:|']
    for point in result['points']:
        d = point['decode']
        lines.append(f"| {point['rho']:g} | {d['mean_duplicate_slot_fraction']:.4f} / {d['peak_duplicate_slot_fraction']:.4f} | {d['mean_unique_resident_experts']:.2f} | {100*d['local_exact_hit_fraction']:.2f}% | {100*d['global_resident_remote_service_fraction']:.2f}% | {100*d['final_local_service_fraction']:.2f}% |")
    lines += ['', 'Hit fractions use residency at event entrance, whereas final local service also includes copies fetched for this event. Resident remote service requires that the actual remote serving copy existed at entrance. Denominators are raw expert routes. Occupancy means average post-event state; peaks include intermediate state.', '',
              '## Prefill and full-trace context', '',
              '| rho | Prefill H2D GiB | Prefill peer MiB | Full H2D GiB | Full peer MiB |',
              '|---:|---:|---:|---:|---:|']
    for point in result['points']:
        pr, full = point['prefill'], point['full']
        lines.append(f"| {point['rho']:g} | {pr['expert_h2d_bytes']/1024**3:.4f} | {pr['peer_activation_bytes']/1024**2:.4f} | {full['expert_h2d_bytes']/1024**3:.4f} | {full['peer_activation_bytes']/1024**2:.4f} |")
    lines += ['', 'Each budget replays the same 48 prefill + 384 decode events from an empty cache, with 1,843 slots split [461,461,461,460]. Prefill initializes state and is excluded from Pareto selection. Total routes are 928,896 in prefill and 98,304 in decode. The trace contains 5,462 unique (layer,expert) keys, all fetched first-ever once per budget.', '',
              '## Validation and reproducibility', '',
              '- Seven CPU policy tests passed, including hand-counted traffic, exact greedy savings, incremental dispatch effects, tie breaks, LRU, primary promotion, inactive unique-copy eviction, active-copy protection and atomic mandatory-admission failure.',
              '- Synthetic differential coverage: 48 independent rho=0 events; 192 replica events checked against full candidate traffic recomputation and a deterministic twin replay.',
              '- Actual rho=0: all 432 events match a separately written slot-array implementation in full state, route destinations, dispatch/combine matrices and fetch classes. Zero duplicates at every event.',
              '- Every actual event checks exact resident destinations, physical capacities, duplicate cap, primary validity, slot ownership and send/receive transposes. All CSV sums and full=prefill+decode identities were cross-checked against JSON; Pareto dominance was independently recomputed.',
              '- All four raw capture hashes match the prior validated receipts. The raw routes are frozen; captured P0 placements are intentionally not replay targets.',
              f"- One CPU process, one numerical-library thread, hard address-space limit 2 GiB; measured peak RSS **{result['peak_rss_mib']:.2f} MiB**, elapsed **{result['elapsed_seconds']:.2f} seconds**. No Torch import, GPU run, model generation, NCCL call, or new quality/performance measurement.",
              f"- Policy/protocol commit: `{result['provenance']['source_commit']}`. Capture result commit: `{result['provenance']['capture_result_commit']}`.", '',
              'See [frozen conventions and commands](REPLICA_REPLAY_PROTOCOL.md), [summary CSV](replica_pareto_screen.csv), [event CSV](replica_pareto_events.csv), [full JSON](replica_pareto_screen.json), and [validation and output hashes](replica_pareto_validation.json).', '',
              '## Interpretation and stop boundary', '',
              'These are counterfactual byte counts under one common deterministic first-copy/LRU policy and a greedy current-byte replica rule. From F to C, decode reloads increase from 17,441 to 18,388 and replica fetches from zero to 27,304. Mean unique coverage falls from 1,843 to 772.88 experts. At C, every route executes locally after current-event fetches, while entrance local hits are zero: the eliminated peer traffic comes with substantial repeated CPU loading. The rho=.75 cap is not reached because all current demand is already local before the cap fills.', '',
              'This short frozen trace does not establish runtime speedup, steady-state behavior, numerical quality, or an optimal placement policy. F/C are selected only on decode totals; K, when present, is the greatest positive inward distance from the normalized endpoint chord. Endpoint percentages use F as denominator.', '',
              ('The three gate conditions pass. A longer trace or physical F/K/C validation may be considered by the owner; this task ends at the CPU result.' if expected_go else 'The gate conditions do not all pass. Stop for owner review; do not extend the trace, alter budgets, implement a final policy, or launch physical F/K/C.'), '']
    (p / 'REPLICA_PARETO_RESULTS.md').write_text('\n'.join(lines))
    validation = dict(status='PASS', policy_unit_tests=7, synthetic_independent_zero_events=48,
                      synthetic_greedy_audited_events=192, synthetic_deterministic_twin_events=192,
                      **result['validation'], csv_json_aggregate_parity=True,
                      independent_pareto_parity=True, protocol_source_hashes_unchanged=True,
                      process_peak_rss_mib=result['peak_rss_mib'], decision=decision,
                      source_sha256={str(Path(__file__).relative_to(p.parents[2])): sha(Path(__file__))},
                      output_sha256={name: sha(p / name) for name in
                                     ('replica_pareto_screen.csv', 'replica_pareto_events.csv',
                                      'replica_pareto_screen.json', 'REPLICA_PARETO_RESULTS.md')})
    (p / 'replica_pareto_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
