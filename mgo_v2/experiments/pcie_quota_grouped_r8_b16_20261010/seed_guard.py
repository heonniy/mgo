"""Use the existing guarded R8 launcher with one frozen seed manifest."""

import argparse
import sys
from pathlib import Path


PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / 'scripts'))
import run_qwen_r8_job as guard  # noqa: E402


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--workloads', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args, rest = parser.parse_known_args()
    guard.WORKLOADS = args.workloads.resolve()
    guard.ROOT = args.output_root.resolve()
    assert guard.WORKLOADS == guard.ROOT / 'WORKLOADS.json'
    assert guard.WORKLOADS.is_file()
    sys.argv = [sys.argv[0], *rest]
    guard.main()


if __name__ == '__main__':
    main()
