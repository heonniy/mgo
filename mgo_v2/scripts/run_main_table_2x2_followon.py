"""Start the DeepSeek table only after the Qwen table has fully passed."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_main_table_2x2_qwen import SYSTEMS, WORKLOAD_ROOT, existing_pass


def qwen_missing():
    missing = []
    for dataset in ('ShareGPT', 'LMSYS-Chat-1M'):
        manifest = json.loads((WORKLOAD_ROOT / dataset / 'WORKLOADS.json').read_text())
        assert manifest['status'] == 'FROZEN'
        for cell in manifest['cells']:
            if cell['model'] != 'Qwen3':
                continue
            for system in SYSTEMS:
                prefix = (f'mt2_qwen_{dataset.lower().replace("-", "_")}'
                          f'_b{cell["local_batch"]}_l{cell["input_tokens"]}_{system}_r3')
                if existing_pass(prefix) is None:
                    missing.append(prefix)
    assert len(missing) <= 32
    return missing


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().split(') ', 1)[1][0] != 'Z'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    missing = qwen_missing()
    print(json.dumps({'qwen_pass': 32 - len(missing), 'qwen_total': 32,
                      'missing': missing}), flush=True)
    if args.check_only:
        return
    while process_exists(args.after_pid):
        time.sleep(30)
    missing = qwen_missing()
    if missing:
        print(json.dumps({'status': 'WAITING_FOR_QWEN_REPAIR', 'missing': missing}), flush=True)
        raise SystemExit(2)
    print(json.dumps({'status': 'QWEN_COMPLETE_START_DEEPSEEK'}), flush=True)
    script = Path(__file__).with_name('run_main_table_2x2_deepseek.py')
    subprocess.run([sys.executable, '-u', str(script)], check=True)


if __name__ == '__main__':
    main()
