#!/bin/bash
# r2-sparse-sk T1.4 / leakage test (e): rebuild every earlier recipe with this code tree and diff it against
# the reference run's recipe.json (hash and fields) and its snapshot_step_00000.pt (every key, bitwise):
#   the 14 earlier WCTE recipes of r2-sparse2's check (C2, S1, S2 x 3; C3, S1n, S1p, C3e, C2dc), the three
#   WCTE S1L runs of r2-sparse2 and the three reused SK default (SK-D0) runs.
# Then builds the new recipes for the record: SK-C2 vs SK-D0 (test (e): only --loss/--placeholder may differ),
# SK-S1L / SK-S2L vs SK-C2's recipe, SK-S1A vs SK-S1L's recipe, W-S1A vs WCTE S1L (weights: shared keys equal).
# Read-only on every reference run.   sparse_sk_recipe_check.sh <code tree> <out dir>
set -u
CODE=$1; O=$2; mkdir -p $O
PY=/home/dwong/anaconda3/bin/python3; S=$CODE/scripts/follow_recipes.py
R=/ceph/dwong/jepa/returns
SP=$R/r2_sparse_2026-11-24/train/runs; FO=$R/r2_follow_2026-11-24/train/runs; S2=$R/r2_sparse2_2026-11-25/train/runs
declare -A D0=([9000]=$R/r2_lucid_2026-11-21/train/runs/sk__mae_repaired__s9000
               [9001]=$R/r2_tier1_2026-11-22/train/runs/sk__mae_repaired__s9001
               [9002]=$R/r2_follow_2026-11-24/train/runs/sk__mae_repaired__s9002)
W="--micro-batch 8 --data-kind lucid_v1_w2 --memory-fraction 0.40"
K="--micro-batch 8 --data-kind lucid_v1 --memory-fraction 0.9"
C2F="--loss fixed --placeholder sentinel"
run() { local det=$1 name=$2 cli=$3 seed=$4 ref=$5 B; [ $det = sk ] && B=$K || B=$W
  PYTHONPATH=$CODE/src TMPDIR=/work/dwong OMP_NUM_THREADS=6 nice -n 15 taskset -c 0-5 $PY $S --cli "$det $cli $B --seed $seed" \
    --ref $ref --out $O/$name.json --weights-prefix "" 2>&1 | tail -1 | sed "s/^/$name /"; }
for s in 9000 9001 9002; do
  run wcte C2_s$s   "mae_repaired $C2F" $s $SP/wcte__mae_repaired__s${s}__dw2__lfixed__psent
  run wcte S1_s$s   "sparse_mae --queries all" $s $SP/wcte__sparse_mae__s${s}__dw2__qall
  run wcte S2_s$s   "sparse_time" $s $SP/wcte__sparse_time__s${s}__dw2
  run wcte S1L_s$s  "sparse_mae --queries all --embed linear --decoder legacy" $s $S2/wcte__sparse_mae__s${s}__dw2__elin__dleg__qall
  run sk   SKD0_s$s "mae_repaired" $s ${D0[$s]}
done
run wcte C3_s9000   "mae_repaired $C2F --embed fourier --decoder isolated --count-head" 9000 $SP/wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier__disol__cnt
run wcte S1n_s9000  "sparse_mae --queries neg --neg-ratio 1.0" 9000 $SP/wcte__sparse_mae__s9000__dw2__qneg
run wcte S1p_s9000  "sparse_mae --queries pos" 9000 $SP/wcte__sparse_mae__s9000__dw2__qpos
run wcte C3e_s9000  "mae_repaired $C2F --embed fourier" 9000 $FO/wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier
run wcte C2dc_s9000 "mae_repaired $C2F --decoder isolated --count-head" 9000 $FO/wcte__mae_repaired__s9000__dw2__lfixed__psent__disol__cnt
[ "${ONLY_EXISTING:-0}" = 1 ] && { echo RECIPE_CHECK_DONE; exit 0; }
for s in 9000 9001 9002; do
  run sk   SKC2_s$s  "mae_repaired $C2F" $s ${D0[$s]}
  run sk   SKS1L_s$s "sparse_mae --queries all --embed linear --decoder legacy" $s ${D0[$s]}
  run sk   SKS1A_s$s "sparse_mae --queries all --embed linear --decoder legacy --absence-tokens" $s ${D0[$s]}
  run sk   SKS2L_s$s "sparse_time --embed linear --decoder full" $s ${D0[$s]}
  run wcte WS1A_s$s  "sparse_mae --queries all --embed linear --decoder legacy --absence-tokens" $s $S2/wcte__sparse_mae__s${s}__dw2__elin__dleg__qall
done
echo RECIPE_CHECK_DONE
