#!/usr/bin/env bash
# Runs steps 0-4 in order; stops at the first step that fails. CPU only (the small pod is enough).
#   bash run_all.sh [RUN_DIR] [OUT_DIR]
set -euo pipefail
RUN_DIR=${1:-$HOME/runs/pi05_a2_pro_swap_100}
OUT=${2:-$RUN_DIR/se_variants}
PY=${PY:-$HOME/env/vla/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cd "$HERE"
run() { local name=$1; shift; echo "=== $name ==="; env -u PYTHONPATH "$PY" "$@" 2>&1 | tee "$OUT/$name.log"; }
run s0 s0_reproduce.py "$RUN_DIR" --out "$OUT"
run s1 s1_scale.py "$RUN_DIR" --out "$OUT"
run s2 s2_anchor.py --out "$OUT"
run s3 s3_features.py --out "$OUT"
run s4 s4_test.py --out "$OUT"
echo "all steps done; results in $OUT"
