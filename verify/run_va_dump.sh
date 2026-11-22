#!/bin/bash
# Usage: bash run_va_dump.sh CASES.json OUTDIR GROUP[,GROUP...] [PACK_OVERRIDE_JSON] [DOMAIN]
# One GPU job at a time (CUDA_VISIBLE_DEVICES set by caller); skips finished outputs.
set -u
CASES=$1; OUT=$2; GROUPS_=$3; PACKS=${4:-}; DOMAIN=${5:-photon}
PY=/home/dwong/anaconda3/bin/python3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TMPDIR=/work/dwong
mkdir -p "$OUT"
for c in $($PY -c "import json;print(' '.join(x['case'] for x in json.load(open('$CASES'))['cases']))"); do
  det=${c%%__*}
  for g in ${GROUPS_//,/ }; do
    if [ -n "$PACKS" ]; then pack=$($PY -c "import json;print(json.load(open('$PACKS'))['$det'])"); else pack=/ceph/dwong/jepa/data/lucid_v1/$det/${det}_lucid_v1.h5; fi
    o="$OUT/${c}__${g}__${DOMAIN}.npz"
    [ -s "$o" ] && continue
    nice -n 15 $PY va_dump.py "$CASES" "$c" "$pack" "$g" "$o.tmp.npz" --device cuda:0 --domain "$DOMAIN" --threads 4 \
      && mv "$o.tmp.npz" "$o" || echo "FAILED $c $g"
  done
done
echo ALLDONE
