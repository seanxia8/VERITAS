# `latent_monitor.estimators` — the four linear classes

The four main-text representation classes of Paper 1, copied from the Paper 1
experiment repository so they can be used as **linear subjects** in the
controlled-variable protocol. Plan: `docs/EXPERIMENT_DESIGN.md` §IV (tentative).

| module | class | what it is | source (`noise-weighted-subspace-reconstruction` @ `ea076ff`, 2026-09-10) |
|---|---|---|---|
| `of.py` | 1 · OF | fixed rank-1 template, GLS / matched-filter amplitude, frequency-domain (inverse-PSD weights) and metric forms; predicted variance under Σ̂ ≠ Σ | `src/noise_geometry/filters/optimal.py` (unchanged) + `gls_amplitude_metric`, `OptimalFilter` |
| `cw_pca.py` | 2 · CW-PCA / EMPCA, IsoPCA | whiten, top-k SVD, map back; Σ̂⁻¹-orthogonal projection; principal angles in a metric | `src/noise_geometry/subspace/pca.py`, `subspace/angles.py` (unchanged) + `SubspaceFit.encode` |
| `tied_linear_ae.py` | 3 · tied linear AE | closed form and independently trained (L-BFGS-B / Adam) tied AE under the Σ̂⁻¹ loss, with optimality gap, principal angle and M-orthonormality diagnostics against the EMPCA optimum | `src/noise_geometry/autoencoders/trained.py`, `autoencoders/linear.py` (`.npz` I/O and the S3 driver dropped) |
| `nfpa.py` | 4 · NFPA, Iso-MPCA | separable whitening, alternating Rayleigh–Ritz (covariance-weighted Tucker-2) with restarts, Kronecker basis for angles | `experiments/synthetic/nfpa_confirmatory/{methods,covariance,generator}.py` — numerical core unchanged, `Dataset` coupling replaced by explicit `(Sigma_c, Sigma_t)` |
| `identification.py` | — | **new**: tangent basis `{s, ∂_t s, ∂_θ s}`, the k×k mixing matrix between a recovered basis and the physical directions, and the canonical gauge fix | — |
| `_metric.py` | — | shared metric helpers (diagonal vector or full SPD inverse covariance; symmetric roots) | consolidated from the sources above |

Conventions: row-wise samples `(n, d)` for classes 1–3 (a `C×T` trace is
flattened with `order="F"` when needed), `(n, C, T)` for class 4. The metric
is `Σ̂⁻¹` as a 1-D weight vector or a 2-D SPD matrix. Every fit exposes
`encode` / `reconstruct` so it can back a `Subject` stage.

Not copied: the untied AE ablation (`autoencoders/untied.py`, P1-E12), the
CRESST multi-template OF bank and the TIDMAD quadrature OF (`noise_geometry/
{cresst,tidmad}/`), and the confirmatory harness itself. They stay in the
source repository; the archival `basis_whitened` reshape defect noted there
is not reproduced here.

```bash
PYTHONPATH=src python -m pytest src/latent_monitor/tests/test_estimators.py -q   # 7 tests, ~10 s
```
