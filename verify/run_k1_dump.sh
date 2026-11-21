#!/bin/bash
# Run every K1 case sequentially (skip finished ones). CPU only: 5 threads, nice 15.
# Usage: bash run_k1_dump.sh OUTDIR
cd "$(dirname "$0")"
OUT=${1:-/work/dwong/r2_verify/dump}; mkdir -p "$OUT"
export TMPDIR=/work/dwong OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
for c in $(python3 cases.py | head -n -1 | awk '{print $1}'); do
  [ -f "$OUT/${c}__meta.json" ] && continue
  while [ "$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)" -gt 30 ]; do echo "load high, waiting"; sleep 600; done
  nice -n 15 python3 k1_dump.py "$c" "$OUT" 5 || echo "FAILED $c"
done
echo ALL_DONE
