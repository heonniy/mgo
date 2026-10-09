"""Use the audited Qwen headline guard for a two-GPU frozen batch."""

from pathlib import Path

import run_qwen_r8_job as base


base.PHYSICAL = (0, 1)
base.CELL = 'Qwen3_ShareGPT_R2_C30_B16_L512_O64'
base.ROOT = Path('/home/hwlee/mgo-results/qwen_r2_sharegpt_b16_l512_20261010')
base.WORKLOADS = base.ROOT / 'WORKLOADS.json'


if __name__ == '__main__':
    base.main()
