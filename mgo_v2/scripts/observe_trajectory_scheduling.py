#!/usr/bin/env python3
"""Posthoc read-only scheduling observation of an already running CPU replay.

No new replay or policy changes. Samples are partial, not component-level CPU
profiles; retain their explicit coverage and never extrapolate to earlier runs.
"""
import json
from pathlib import Path
import time
from summarize_trajectory_study import ROOT

def main():
    state=json.loads((ROOT/'replay_status.json').read_text())
    pid=int(state['started']['16'])
    proc=Path('/proc')/str(pid)
    output=ROOT/'replay_scheduling_observations.jsonl'
    assert not output.exists()
    with output.open('w') as file:
        while proc.exists():
            try:
                run_ns,wait_ns,slices=map(int,(proc/'schedstat').read_text().split())
                candidates=list((ROOT/'replay/b16').glob('*timings.jsonl'))
                current=max(candidates,key=lambda p:p.stat().st_mtime_ns)
                row=dict(monotonic_ns=time.monotonic_ns(),unix=time.time(),pid=pid,
                    cpu_runtime_ns=run_ns,runqueue_wait_ns=wait_ns,timeslices=slices,
                    latest_timing_file=current.name,latest_timing_bytes=current.stat().st_size,
                    scope='posthoc partial main-thread scheduling samples; not component CPU time')
                file.write(json.dumps(row,separators=(',',':'))+'\n');file.flush()
            except (FileNotFoundError,ProcessLookupError):break
            time.sleep(1)
    print('Read-only scheduling observation ended.',flush=True)

if __name__=='__main__':main()
