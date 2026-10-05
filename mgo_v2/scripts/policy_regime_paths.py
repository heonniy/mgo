"""Separate physical Env1 and historical SHM Env2 artifacts."""
import os
from pathlib import Path
ENVIRONMENT=os.environ.get('MGO_POLICY_REGIME_ENV','env1')
assert ENVIRONMENT in ('env1','env2')
BASE=Path('/home/hwlee/mgo-results/policy_regime_20261005')
BASE_PACKET=Path(__file__).resolve().parents[1]/'experiments/decode_prefetch_runtime_refactoring_20261004'
ROOT=BASE if ENVIRONMENT=='env1' else BASE/'ENV2'
PACKET=BASE_PACKET if ENVIRONMENT=='env1' else BASE_PACKET/'ENV2'
