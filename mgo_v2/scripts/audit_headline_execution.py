"""Audit completion separately from headline timing eligibility.

An unstable result is never promoted to a stable headline result. A completed,
bounded confirmation can close an execution obligation while retaining that
limitation. The explicit outcome map prevents choosing the fastest attempt.
"""
import argparse
import json
from pathlib import Path

from summarize_headline_r4 import SYSTEMS, audit, check_manifests, read


def main(args):
    specs = {s['cell']: s for s in read(args.root / 'WORKLOADS.json')['cells']}
    outcomes = read(args.outcomes)
    expected = {(cell, system) for cell in specs for system in SYSTEMS}
    assert {(r['cell'], r['system']) for r in outcomes} == expected
    assert len(outcomes) == len(expected), 'duplicate outcome row'
    for spec in specs.values():
        check_manifests(spec)
    rows = []
    for outcome in outcomes:
        row = dict(outcome)
        job = args.root / outcome['attempt']
        try:
            assert read(job / 'status.json')['status'] == 'PASS', 'job not finished successfully'
            assert read(job / 'result.json')['cell'] == outcome['cell'], 'wrong cell'
            result = audit(job, specs[outcome['cell']])
            assert result['system'] == outcome['system'], 'wrong system'
            if outcome.get('confirmation_of'):
                previous_job = args.root / outcome['confirmation_of']
                assert read(previous_job / 'status.json')['status'] == 'PASS'
                assert read(previous_job / 'result.json')['cell'] == outcome['cell']
                previous = audit(previous_job, specs[outcome['cell']])
                assert previous['system'] == outcome['system']
                assert previous['status'] == 'UNSTABLE', 'unnecessary confirmation'
                row['previous_attempt'] = previous
            if result['status'] == 'UNSTABLE':
                assert outcome.get('confirmation_of'), 'bounded confirmation remains outstanding'
            row.update(execution_complete=True, headline_eligible=result['status'] == 'PASS', result=result)
        except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
            row.update(execution_complete=False, headline_eligible=False, error=repr(exc))
        rows.append(row)
    complete = all(r['execution_complete'] for r in rows)
    stable = all(r['headline_eligible'] for r in rows)
    output = dict(
        status=('EXECUTION_COMPLETE' if stable else 'EXECUTION_COMPLETE_WITH_UNSTABLE_ROWS') if complete else 'INCOMPLETE',
        execution_complete=complete,
        stable_headline_panel_complete=stable,
        completed_rows=sum(r['execution_complete'] for r in rows),
        stable_rows=sum(r['headline_eligible'] for r in rows),
        rows=rows,
        notes=['Every reported attempt retains all three primary samples.',
               'Completion of a bounded confirmation does not certify timing stability.',
               'Source/binary provenance, resource caveats and final GPU cleanup require separate completion checks.'])
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in ('rows', 'notes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/hwlee/mgo-results/headline_r4_20261007'))
    parser.add_argument('--outcomes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
