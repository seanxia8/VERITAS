#!/bin/bash
# V6 confirmation runs (CPU, nice 15). Usage: bash run_v6_confirm.sh DUMPDIR OUTDIR
cd "$(dirname "$0")"; D=$1; O=$2; mkdir -p "$O"
export TMPDIR=/work/dwong OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
nice -n 15 python3 v6_confirm.py raw "$O/confirm_raw.json"
nice -n 15 python3 v6_confirm.py centred "$D" "$O/confirm_centred.json"
for c in $(python3 cases.py | grep '^wcte' | awk '{print $1}') sk__init__s9000 sk__mae_mse__s9000; do
  [ -f "$O/${c}__confirm.json" ] || nice -n 15 python3 v6_confirm.py enc "$c" "$O" 5 || echo "FAILED $c"
done
echo ALL_DONE
