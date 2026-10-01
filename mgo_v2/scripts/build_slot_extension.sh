#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${CUTLASS_DIR:?set CUTLASS_DIR to a CUTLASS checkout (validated with v3.5.1)}"
export MOE_BUILD_STORE_ONLY=1
export MAX_JOBS="${MAX_JOBS:-6}"
cd "$ROOT/MoE-Infinity-EP-archer-coslot"
"${PYTHON:-python}" setup.py build_ext --inplace
