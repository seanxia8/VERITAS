# verify/ — independent recomputation of the Round 2 load-bearing numbers (r2-verify)

Runbook: `docs/round2/RUNBOOK_2026-11-21_R2_VERIFY.md`. All code here was written from scratch
for this phase. The repository is used only as a black-box encoder forward
(`point_encoder.LucidPointJepa` plus `r2_geometry.load_r2_geometry` for the token order). K0 also
calls the repository feature builder (`r2_store`) as the object under test, and K3 uses the
repository's `lucid_views.ViewMasker` to define the context view.

## Environment

Deepthought (`dt1`), CPU only: `nice -n 15`, ≤ 6 threads, `TMPDIR=/work/dwong`. The Python is
`/home/dwong/anaconda3/bin/python3` (torch 2.5.1, h5py 3.11, scikit-learn 1.5.1, SciPy 1.13.1). Run
from `verify/` inside a clone checked out at `origin/r2-verify` (default `R2V_REPO=/work/dwong/r2_verify_repo`).
The case table is read from `results/round2/20261121_verify/INPUTS.txt`.

```bash
cd /work/dwong/r2_verify_repo/verify
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/work/dwong
D=/work/dwong/r2_verify; R=../results/round2/20261121_verify
```

## One command per check

| Check | Command | Output | Time |
|---|---|---|---|
| V0 checkpoint hashes | `sha256sum -c $D/ckpt.sha256` (list built from INPUTS.txt) | `ckpt_sha256_check.log` | 5 min |
| V0 encoder configs | `nice -n 15 python3 v0_ckpt_configs.py $D/ckpt.sha256 $R/ckpt_configs.json` | `ckpt_configs.json` | 3 min |
| K0 features | `nice -n 15 python3 k0_features.py $R/k0_features.json` | `k0_features.json` | 5 min |
| K1/K2 token dump | `bash run_k1_dump.sh $D/dump` (5 threads; skips finished cases) | `$D/dump/<case>__{train,test,k2,meta}.*` | ≈ 7 h |
| K1 readouts + I1–I6 | `nice -n 15 python3 k1_readout.py $D/dump $R/k1_readout.json` | `k1_readout.json` | 10 min |
| K2 collapse + I7–I8 | `nice -n 15 python3 k2_collapse.py $D/dump $R/k2_collapse.json` | `k2_collapse.json` | 1 min |
| K3 SIGReg + I9–I11 | `git clone https://github.com/rbalestr-lab/lejepa.git $D/ext/lejepa_src` (commit c293d29), then `nice -n 15 python3 k3_sigreg.py $R/k3_sigreg.json $D/ext/lejepa_src` | `k3_sigreg.json` | 15 min |
| K4 ToF + CRLB | `nice -n 15 python3 k4_tof.py $R/k4_tof.json 100` | `k4_tof.json` | 10 min |
| K5 regime | `nice -n 15 python3 k5_regime.py $R/k5_regime.json` | `k5_regime.json` | 3 min |
| V6 investigation | `bash run_v6_confirm.sh $D/dump $D/confirm`, then copy `$D/confirm/*.json` to `$R/v6_confirm/` | `v6_confirm/*.json` | ≈ 1.5 h |
| V6 blind comparison | `python3 v6_compare.py $R` (reads committed values only at this step) | `v6_comparison.json` | < 1 min |

## Files

- `common.py`: detector table, geometry from the LUCiD npz files and the HDF5 attributes, the photon probe cohorts, the independent 9-feature builder, and the black-box encoder loader.
- `cases.py`: the 30 K1 cases parsed from INPUTS.txt.
- `k1_dump.py`: per-event pooled statistics (all-PMT mean, hit mean, three hit spatial moments) and the per-PMT moments for K2.
- `k1_readout.py`: closed-form SVD Ridge with 5-fold run-grouped CV, readouts (a)–(d), the controls and I1–I6.
- `k2_collapse.py`: the law-of-total-variance decomposition, event-mean spread, I7 and I8.
- `k3_sigreg.py`: Epps–Pulley SIGReg, the analytic expectation, N-scaling, 8×4 vs 1×32 accumulation and the comparison with the official implementation.
- `k4_tof.py`: Huber ToF vertex fit, timing CRLB and the simplified charge-only CRLB.
- `k5_regime.py`: occupancy quantiles.
- `v6_compare.py`: the comparison against committed values with the pre-registered tolerances. `v6_confirm.py` + `ridge_core.py`: my own re-implementation of the committed spatial-moment construction, written after the seal for the §4.1 diagnosis.
