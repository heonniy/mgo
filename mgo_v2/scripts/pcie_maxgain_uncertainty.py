"""Supplement frozen final results with uncertainty and observed administration."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    source = a.root / 'maxgain_validation.json'
    final = json.loads(source.read_text()); assert final['status'] == 'PASS' and final['stage'] == 'final'
    rng = np.random.default_rng(20261010); rows = []
    for c in final['comparisons']:
        r = np.array(c['R']['values']); g = np.array(c['G']['values'])
        assert len(r) == len(g) == 5
        indices = rng.integers(0, 5, (10000, 5))
        boot = 100 * (1 - np.median(g[indices], axis=1) / np.median(r[indices], axis=1))
        rows.append(dict(candidate=c['candidate'], median_gain_pct=c['gain_percent'],
            mean_gain_pct=float(100 * (1-g.mean()/r.mean())),
            exploratory_paired_bootstrap_ci95_pct=np.percentile(boot, [2.5, 97.5]).tolist(),
            paired_repeat_gain_pct=(100*(1-g/r)).tolist(),
            original_conversation_families=c['input_provenance']['known_conversation_families']))
    # First preflight stdout printed this completion timestamp. The original
    # receipt was subsequently replaced by the reproducible scripted preflight.
    first_end = 1791561860.798152
    observed_start = 1791561831.4522178  # Observed live phase.json before first CPU preflight.
    metric = a.root / 'family_3/G-NEAR/repeat5.json'
    assert observed_start < first_end < metric.stat().st_mtime
    second = json.loads(a.cpu_receipt.read_text())
    activity = dict(first_cpu_preflight_completion_observed_unix=first_end,
        first_timestamp_provenance='First CPU-only meta-preflight stdout in the tool execution record; original receipt was later overwritten by scripted preflight',
        overlapping_generation=dict(candidate='family_3', arm='G-NEAR', repeat=5,
            generation_start_observed_unix=observed_start, metric_file_mtime_unix=metric.stat().st_mtime),
        second_preflight_receipt=str(a.cpu_receipt), second_preflight_completion_unix=second['unix'],
        timing_limit='Observed first completion falls inside the observed generation window; start/duration and causal effect on GPU timing are not established. File mtime follows result writing and is not the exact GPU end timestamp.')
    report = dict(status='PASS', source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        comparisons=rows, seed=20261010, bootstrap_draws=10000, cpu_administration=activity,
        method='Exploratory paired repeat-block bootstrap of ratio of medians; five repeats per arm; not preregistered or selection-adjusted',
        timing_policy='All original timings remain retained; no exclusion, new winner selection or alteration of the frozen final cohort')
    (a.out/'UNCERTAINTY.json').write_text(json.dumps(report, indent=2)+'\n')
    lines=['# Best64 final result: uncertainty and administrative context', '',
        'The frozen final cohort remains unchanged. All five repeats per arm, including the slow fifth G-NEAR family_3 repeat, are retained. The preregistered ratio-of-medians winner is still family_3.', '',
        '| Candidate | Median TPOT reduction | Mean TPOT reduction | Exploratory paired 95% interval | Original conversation families |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lo,hi=row['exploratory_paired_bootstrap_ci95_pct']
        lines.append(f"| {row['candidate']} | {row['median_gain_pct']:.3f}% | {row['mean_gain_pct']:.3f}% | [{lo:.3f}%, {hi:.3f}%] | {row['original_conversation_families']} |")
    lines += ['', 'The 4.736% selected median reduction does not establish a stable universal benefit. Its exploratory interval includes zero; the mean reduction is 2.615%. Five-repeat bootstrap intervals are coarse, assume exchangeable repeat blocks and do not adjust for screening/selection or temporal drift. The mixed_0 batch spans 52 original conversations, while family_3 contains 64 distinct split records/input prefixes from one original conversation.', '',
        'Concurrent administration: the previously preserved Git-push receipts are in the frozen cohort. In addition, the first full-size CPU-only DeepSpeed meta/leaf preflight printed completion Unix timestamp 1791561860.798152, inside the observed family_3/G-NEAR/repeat5 generation window beginning 1791561831.4522178 and preceding the result file mtime. The reproducible second CPU preflight has its own preserved receipt and completed later. These observations do not establish the duration or causal effect of CPU work; they make quiet-host timing unproven for that repetition. No slow repeat is removed or replaced.', '',
        'Future primary runs avoid concurrent CPU builds, imports and analytical generation; read-only process/status observations remain small. The seven-policy cohort uses the original headline64, with independent cold-cache generations and separate diagnostics. Its results must not be mixed with this selected-input search.', '',
        'The raw final results, source IDs, full token/cache/trace checks, physical phase validation and separate breakdown remain in ../final/cohort/. This supplement explains the limits of their interpretation and changes none of their measurements.', '']
    (a.out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS', comparisons=rows)))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cpu-receipt',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
