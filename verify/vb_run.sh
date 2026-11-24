#!/bin/bash
# Usage: bash vb_run.sh CASES.json JOBS.txt OUTDIR DEVICE THREADS
# Processes "case group domain" lines in order; claims each output with mkdir <out>.lock so that
# a GPU and a CPU worker can share a queue without duplicates.  Skips finished outputs.
set -u
CASES=$1; JOBS=$2; OUT=$3; DEV=$4; TH=$5
PY=/home/dwong/anaconda3/bin/python3
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH TMPDIR=/work/dwong
mkdir -p "$OUT"
while read -r c g dom; do
  [ -z "$c" ] && continue
  det=${c%%__*}
  if [ "$g" = sealed_test ]; then pack=/ceph/dwong/jepa/data/lucid_v1/$det/${det}_lucid_v1_sealed.h5
  else pack=/ceph/dwong/jepa/data/lucid_v1/$det/${det}_lucid_v1.h5; fi
  o="$OUT/${c}__${g}__${dom}.npz"
  [ -s "$o" ] && continue
  mkdir "$o.lock" 2>/dev/null || continue
  if nice -n 15 $PY va_dump.py "$CASES" "$c" "$pack" "$g" "$o.tmp.npz" --device "$DEV" --domain "$dom" --threads "$TH"; then
    mv "$o.tmp.npz" "$o"; rmdir "$o.lock"
  else echo "FAILED $c $g $dom"; rmdir "$o.lock"; fi
done < "$JOBS"
echo ALLDONE
