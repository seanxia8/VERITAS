# WP2 bounded linear implementation — 16 October 2026

Status: engineering implementation, independent review pending. This is the linear subset; it does not establish full WP2 acceptance or authorize E1 grids. Waveform implementation and scientific/data/resource gates remain separate.

## Implemented scope

`linear.fit` returns row functionals for unregularized CCA, hard-isotropy prediction, pure noise GEVP, finite-beta prediction, PCA context autoencoding and noise-whitened PCA context autoencoding. The finite-beta objective has both encoders hard-isotropic and an optimized predictor; its exact symmetric eigenproblem follows THEORY.md. Rank-deficient cross-prediction receives a deterministic orthonormal target completion. No iterative convergence setting or outcome-informed tuning is hidden. Fixed tolerances, beta grid, ordering, completion and sample ridge are frozen in `configs/linear_solver_v1.json`, pending independent review before grids.

`oracle` checks retained joint covariance SPD before Cholesky, records context/residual rank ratios and canonical margins, and returns no recovered context on ineligibility. Eligible context functionals are pulled back through the context triangular block. Caller must supply the retained-coordinate marginal covariance; raw masks provide explicit indices. No undeclared jitter is used.

`ExactModel` in views.py follows the registered toy construction, exposes exact covariance/reference/gamma_n, and separates target-only versus both-view loading scaling. Sequential both-view calls preserve the toy RNG draw order, including independent context calibration noise. Sample noise arrays also expose paired event-noise components. A separate implementer owns the approved waveform append; the immutable tests below bind only ExactModel behavior.

Masks cover the four registered raw splits, with reproducible hashes. Role/cell/model-seed namespaces are hash-derived and independent; method names are excluded so paired methods reuse the same event draws. The namespace utility does not itself enforce cohort allocation, which belongs to later cohort generation.

CRLB uses a full-rank Fisher inverse, and amplitude OF uses known-other-parameters information. Central differences expose the registered step and half-step check. Metrics preserve undefined constant nuisance targets. General probe cross-validation, R1–R10 tables, training loops and report regeneration are not implemented by this subset.

## Bounded verification

Command:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RECONSTRUCTION_DISABLE_TORCH_COMPILE=1 /private/tmp/jepa-venv-cpu311/bin/python -m pytest src/jepa_noise/tests/test_linear.py -q
```

Result: **13 passed in 0.38 s**; tool wall time 0.591466 s. The fixtures check population no-sharing recovery, invertible-map invariance/pullback, the existing principal-angle utility, conditional oracle recovery, annihilation and singular-joint rejection, finite-beta nuisance preference, target null-space completion, finite-difference reduced gradient, objective equality, noise/event/calibration covariance within 2% on a bounded d=4 fixture, small-fixture original toy draw order, independent nuisance R² sanity, constant-target handling, raw masks and namespace pairing, Fisher/OF agreement, and finite-difference half-step stability. These are designed implementation fixtures, not the registered E1a sample grid or evidence that R1–R5 pass.

## Review source binding

The following hashes bind this linear review slice. `views.py` is intentionally excluded from whole-file hashing during the independent waveform append; reviewer should inspect ExactModel and its bounded draw-order/covariance tests, then bind the complete file before integration acceptance.

```json
{
  "src/jepa_noise/linear.py": "0b722975ca38cb7ece228eaf70051ede572e3e900cd2223adf4373c91ab2a767",
  "src/jepa_noise/masks.py": "56e7e8d208b97556d1bde7fd0baec5c086229e399e7651e2358f731be1418053",
  "src/jepa_noise/metrics.py": "f029701abf8a8cb16b811c8f28f51cdfd2d48a8fb764d6fd8cbcaa46155c7824",
  "src/jepa_noise/crlb.py": "0e00a3ccc9fc22ff9dbba8bf081d4ef937d894b7545e9719e3d74f3a27a45647",
  "src/jepa_noise/configs/linear_solver_v1.json": "52fa6a91556259144aff7925555dc7133c6135806db8c94bdad834d1e7899fa7",
  "src/jepa_noise/tests/test_linear.py": "df55cb5c241cea27d1ac77c383842a63754cdab4150d284fb1bfe7b5a831ed9a"
}
```
