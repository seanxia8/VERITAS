#!/usr/bin/env bash
# Execution record. For reproduction, select fresh output paths; never overwrite evidence.
set -euo pipefail
cd /work/dwong/pmt_multienergy_20261016
env JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  PYTHONPATH=code/src:/work/dwong/pmt_lucid_20261016/upstream \
  timeout 1800 /home/dwong/anaconda3/bin/python3 -m noise_module_lucid.multienergy_data \
  --config multienergy_pilot_repaired.json --source source_balanced \
  --upstream /work/dwong/pmt_lucid_20261016/upstream \
  --manifest /work/dwong/pmt_lucid_20261016/code/configs/lucid_pmt/upstream_manifest.json \
  --out evidence/track_closure_02
# The trainer independently requires validation.json passed=true.
env CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=code/src \
  timeout 1800 /home/dwong/anaconda3/bin/python3 -m noise_module_lucid.multienergy_baseline \
  --config multienergy_pilot_repaired.json --data evidence/track_closure_02/digits.npz \
  --out evidence/training_01
