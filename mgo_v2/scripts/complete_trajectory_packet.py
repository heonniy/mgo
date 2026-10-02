#!/usr/bin/env python3
"""Wait for the bounded replay matrix, then build the inspectable result packet."""
import json
from pathlib import Path
import subprocess
import sys
import time
from summarize_trajectory_study import ROOT,PACKAGE

def main():
    status_file=ROOT/'analysis_status.json'
    status=dict(status='WAITING_FOR_REPLAYS',completed=[])
    status_file.write_text(json.dumps(status,indent=2)+'\n')
    while True:
        try: replay=json.loads((ROOT/'replay_status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError): replay={}
        if replay.get('status')=='FAIL':raise SystemExit('Replay matrix failed')
        if replay.get('status')=='COMPLETE':break
        time.sleep(10)
    for script in ('summarize_trajectory_study.py','trajectory_state_examples.py','plot_trajectory_study.py','report_trajectory_study.py'):
        status.update(status='RUNNING',current=script)
        status_file.write_text(json.dumps(status,indent=2)+'\n')
        print('Starting',script,flush=True)
        code=subprocess.call([sys.executable,str(PACKAGE/'scripts'/script)],cwd=PACKAGE)
        if code:
            status.update(status='FAIL',exit_code=code)
            status_file.write_text(json.dumps(status,indent=2)+'\n')
            raise SystemExit(code)
        status['completed'].append(script)
    status.update(status='READY_FOR_VISUAL_REVIEW',finished_unix=time.time())
    status_file.write_text(json.dumps(status,indent=2)+'\n')
    print('Ready for rendered-figure review and final hash sealing.',flush=True)

if __name__=='__main__':main()
