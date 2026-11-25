#!/bin/bash
# r2-sparse2 T1.6: rebuild every existing WCTE recipe (C2, C3, S1, S1n, S1p, S2 of r2-sparse; C3e, C2dc of
# r2-follow) with this code tree and diff it against the reference run's recipe.json (hash and fields), and
# compare the full initial model state with the reference snapshot_step_00000.pt (prefix "" = every key).
# Also builds the three new recipes (S1L, S2L, S2Lt) for the record. Read-only on the reference runs.
#   sparse2_recipe_check.sh <code tree> <out dir> <block width ns>
set -u
CODE=$1; O=$2; BW=$3; mkdir -p $O
PY=/home/dwong/anaconda3/bin/python3; S=$CODE/scripts/follow_recipes.py
SP=/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/train/runs; FO=/ceph/dwong/jepa/returns/r2_follow_2026-11-24/train/runs
B="--micro-batch 8 --data-kind lucid_v1_w2 --memory-fraction 0.40"
C2F="--loss fixed --placeholder sentinel"
run() { PYTHONPATH=$CODE/src TMPDIR=/work/dwong nice -n 15 taskset -c 0-5 $PY $S --cli "wcte $2 $B --seed $3" --ref $4 --out $O/$1.json ${5+--weights-prefix "$5"} 2>&1 | tail -1 | sed "s/^/$1 /"; }
for s in 9000 9001 9002; do
  run C2_s$s   "mae_repaired $C2F" $s $SP/wcte__mae_repaired__s${s}__dw2__lfixed__psent ""
  run S1_s$s   "sparse_mae --queries all" $s $SP/wcte__sparse_mae__s${s}__dw2__qall ""
  run S2_s$s   "sparse_time" $s $SP/wcte__sparse_time__s${s}__dw2 ""
done
run C3_s9000   "mae_repaired $C2F --embed fourier --decoder isolated --count-head" 9000 $SP/wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier__disol__cnt ""
run S1n_s9000  "sparse_mae --queries neg --neg-ratio 1.0" 9000 $SP/wcte__sparse_mae__s9000__dw2__qneg ""
run S1p_s9000  "sparse_mae --queries pos" 9000 $SP/wcte__sparse_mae__s9000__dw2__qpos ""
run C3e_s9000  "mae_repaired $C2F --embed fourier" 9000 $FO/wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier ""
run C2dc_s9000 "mae_repaired $C2F --decoder isolated --count-head" 9000 $FO/wcte__mae_repaired__s9000__dw2__lfixed__psent__disol__cnt ""
# new cells (diff against C2 / S2 recipes, for the record; differences expected in identity and flags)
for s in 9000 9001 9002; do
  run S1L_s$s  "sparse_mae --queries all --embed linear --decoder legacy" $s $SP/wcte__mae_repaired__s${s}__dw2__lfixed__psent
  run S2L_s$s  "sparse_time --embed linear --decoder full" $s $SP/wcte__sparse_time__s${s}__dw2
  run S2Lt_s$s "sparse_time --embed linear --decoder full --blocks time --block-width-ns $BW" $s $SP/wcte__sparse_time__s${s}__dw2
done
echo RECIPE_CHECK_DONE
