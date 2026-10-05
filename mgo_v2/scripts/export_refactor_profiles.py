"""Sequential post-capture export and rank-local interval summary.

Run only outside primary GPU timing. Rank-local durations are never summed
into a wall time or interpreted as a critical path.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from analyze_refactor_nsys import analyze


def distribution(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def summarize(rows):
    assert len(rows) == 8 and all(row['status'] == 'PASS' for row in rows)
    assert len({row['decode_events'] for row in rows}) == 1
    assert rows[0]['decode_events'] in (48*8, 48*256)
    sections = {}
    for section in ('H2D', 'kernel_union_ms', 'cpu_nvtx'):
        keys = sorted(set().union(*(row[section] for row in rows)))
        values = {}
        for key in keys:
            present = [row[section].get(key) for row in rows]
            if section == 'cpu_nvtx':
                fields = sorted(set().union(*(v for v in present if v is not None)))
                values[key] = {
                    field: distribution([v[field] for v in present if v is not None])
                    for field in fields
                }
                values[key]['ranks_present'] = sum(v is not None for v in present)
            else:
                values[key] = distribution([v for v in present if v is not None])
                values[key]['ranks_present'] = sum(v is not None for v in present)
        sections[section] = values
    exclusive = [row['interval_decomposition']['aggregate_exclusive_ms'] for row in rows]
    sections['exclusive_ms'] = {
        key: distribution([row[key] for row in exclusive]) for key in exclusive[0]
    }
    return dict(
        status='PASS', ranks=8, primary_timing=False,
        across_rank_distributions=sections,
        interpretation='Each duration is a rank-local interval union. Min/median/max describe ranks, not repeat uncertainty. Do not sum ranks or phase maxima into wall time. Instrumented profiles are attribution only; unprofiled E2E/TPOT determine performance. Missing CPU ranges remain absent rather than being assigned zero.',
    )


def main(args):
    root = args.capture.resolve()
    state = json.loads((root / 'status.json').read_text())
    assert state['status'] == 'PASS' and state['primary_timing'] is False
    output = root / args.output_name
    output.mkdir(exist_ok=False)
    rows = []
    sources = []
    for rank in range(8):
        report = root / f'profile_rank{rank}.nsys-rep'
        receipt_path = root / f'rank{rank}.json'
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        assert receipt['status'] == 'PASS' and receipt['rank'] == rank
        assert report.is_file() and report.stat().st_size > 0
        database = output / f'rank{rank}.sqlite'
        with (output / f'rank{rank}_export.log').open('w') as log:
            subprocess.run(
                [args.nsys, 'export', '--type=sqlite', '--output', str(database), str(report)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600,
            )
        row = analyze(database, receipt)
        result = output / f'rank{rank}_intervals.json'
        result.write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
        sources.append(dict(rank=rank, report=str(report), report_bytes=report.stat().st_size,
                            receipt=str(receipt_path), receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                            analysis=str(result), analysis_sha256=hashlib.sha256(result.read_bytes()).hexdigest()))
        print(f'rank {rank}: exported and reconciled', flush=True)
    summary = summarize(rows)
    summary.update(capture=str(root), case=state['case'], sources=sources)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    parser.add_argument('--nsys', default='/usr/local/bin/nsys')
    parser.add_argument('--output-name', default='interval_analysis')
    main(parser.parse_args())
