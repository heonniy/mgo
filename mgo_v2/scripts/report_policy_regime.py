"""Build the final report only from complete, audited Env1 and Env2 evidence."""
import hashlib
import json
from pathlib import Path

from policy_regime_paths import BASE, BASE_PACKET

POLICIES = ('BR', 'OLD_CA', 'FCA', 'LA_CA')
CELLS = {(cache, policy) for cache in ('C30', 'C60') for policy in POLICIES}


def build():
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    evidence = {}
    for environment in ('env1', 'env2'):
        packet = BASE_PACKET if environment == 'env1' else BASE_PACKET / 'ENV2'
        root = BASE if environment == 'env1' else BASE / 'ENV2'
        timing = read(packet / 'POLICY_REGIME_TIMING_RESULTS.json')
        mechanism = read(packet / 'POLICY_REGIME_MECHANISMS.json')
        workload = read(packet / 'POLICY_REGIME_WORKLOAD.json')
        assert timing['status'] == 'TIMING_AUDITED'
        assert mechanism['status'] == workload['status'] == 'PASS'
        assert {(x['setting'], p) for x in timing['rows'] for p in x['policies']} == CELLS
        for dataset in (mechanism, workload):
            assert len(dataset['rows']) == 8
            assert {(x['setting'], x['policy']) for x in dataset['rows']} == CELLS
        transports = {}
        for cache in ('C30', 'C60'):
            state = read(root / cache / f'POLICY_REGIME_{cache}_B128_H64/status.json')
            assert state['status'] == 'PASS'
            if environment == 'env2':
                check = state['transport_verification']
                assert check['status'] == 'PASS' and check['observed_paths'] == ['SHM']
                transports[cache] = check
        evidence[environment] = dict(timing=timing, mechanism=mechanism,
                                     workload=workload, transport=transports)

    lines = [
        '# Policy regime results: Env1 and historical Env2', '',
        'R4 GPUs 0/1/4/5; local B128; frozen decode64; BF16; V3 P2/T2; '
        'substitution OFF. C30 and C60 use independent BR denominators. '
        'The same frozen requests, routes, weights, and teacher tokens are reused.', '',
        'Env2 is the historical same-host SHM configuration: NCCL_CUMEM_ENABLE=0, '
        'NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, with other inherited NCCL overrides '
        'cleared. Both Env2 cache groups must show SHM channels before timing. '
        'This is not a measurement on physically NVLink-free hardware.', '',
        'Primary timing is unprofiled. Two-repeat estimates use the mean; '
        'three-repeat estimates use the median. All raw samples and full ranges '
        'remain in the timing JSON. A slow observation alone is not a reason '
        'to remove it. A positive point estimate is not proof of a gain; '
        'paired uncertainty is reported separately.', '',
        '| Environment | Cache | Policy | TPOT s [range] | E2E s [range] | TPOT gain | Paired gain 95% interval |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    stability_notes = []
    for env, data in evidence.items():
        for cell in data['timing']['rows']:
            for policy in POLICIES:
                row = cell['policies'][policy]
                t, e = (row['metrics'][k] for k in ('TPOT', 'E2E_wall'))
                ci = row.get('paired_gate', {}).get('gain', {}).get('TPOT', {}).get('CI95')
                ci_text = 'baseline' if ci is None else f'[{100*ci[0]:+.2f}%, {100*ci[1]:+.2f}%]'
                lines.append(f"| {env} | {cell['setting']} | {policy} | "
                             f"{t['value']:.6f} [{t['range'][0]:.6f}, {t['range'][1]:.6f}] | "
                             f"{e['value']:.3f} [{e['range'][0]:.3f}, {e['range'][1]:.3f}] | "
                             f"{100*row['TPOT_gain']:+.3f}% | {ci_text} |")
            unstable = cell['unstable']
            stability_notes.append(f"{env}/{cell['setting']}: unstable={unstable}; "
                                   f"repeat counts: {', '.join(p+'='+str(len(cell['policies'][p]['metrics']['TPOT']['samples'])) for p in POLICIES)}.")
    lines += ['', *stability_notes]
    lines += ['', '## Mechanism attribution', '',
              'Values below are means across the four rank-local durations per decode step (ms). '
              'They come from separate instrumented captures, not primary TPOT. '
              'Do not add them: phases can overlap, NCCL includes waiting, and host staging '
              'runs on its own CPU thread. Outside-COMM/COMPUTE H2D is an overlap definition, '
              'not proof of critical-path stall. One capture per cell is descriptive.', '',
              '| Env | Cache | Policy | H2D DMA | H2D outside | Forward NCCL | Return NCCL | Expert GPU | Current controller CPU | Prefetch controller CPU | Host staging CPU |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ('total_H2D_ms', 'outside_COMM_COMPUTE_H2D_ms', 'moe.forward_a2a.nccl',
            'moe.return_a2a.nccl', 'moe.expert_compute', 'CPU moe.current_controller',
            'CPU moe.prefetch_controller', 'CPU moe.host_staging')
    for env, data in evidence.items():
        for row in sorted(data['mechanism']['rows'], key=lambda x: (x['setting'], POLICIES.index(x['policy']))):
            values = ' | '.join(f"{row['rank_local_ms_per_step'][k]['mean']:.3f}" for k in keys)
            lines.append(f"| {env} | {row['setting']} | {row['policy']} | {values} |")
    lines += ['', 'Full rank min/max distributions, per-event expert/communication imbalance, '
              'remote packet counts, and critical-rank packet load are retained in each '
              '`POLICY_REGIME_MECHANISMS.json`. Actual decode copy counts, prefetch usage, '
              'and issued-copy precision are in `POLICY_REGIME_WORKLOAD.json`. '
              'Prefetch use does not by itself establish readiness or time saved.', '',
              'Nsight 2024.6 FCA stalled twice. Both failures remain archived. Attribution '
              'was recollected with isolated Nsight 2025.3 and device event completion '
              'tracing disabled; completed old captures are not mixed into the final '
              'duration comparison. Primary timing was not repeated for this recovery.', '',
              'Machine-readable source receipt hashes: `POLICY_REGIME_FINAL_SOURCES.json`.']
    return '\n'.join(lines) + '\n', dict(status='COMPLETE_EVIDENCE_LOADED', source_sha256=sources)


if __name__ == '__main__':
    report, sources = build()
    (BASE_PACKET / 'POLICY_REGIME_FINAL_RESULTS.md').write_text(report)
    (BASE_PACKET / 'POLICY_REGIME_FINAL_SOURCES.json').write_text(json.dumps(sources, indent=2) + '\n')
