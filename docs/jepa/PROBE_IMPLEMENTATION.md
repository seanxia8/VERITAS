# Generic grouped ridge probes — implementation pending independent review

This implements runbook §1.3's alpha grid and training-only scaling in `src/jepa_noise/probes.py`. It does not approve A3, assign E1 cohort sizes, create event namespaces or choose an intercept convention. No empirical probes or outcome grids have run.

`grouped_ridge(features, targets, groups, *, fit_intercept)` requires explicit caller-provided labels for exactly five whole-group folds. It cannot infer groups from row order. The caller must supply the five fold-group assignments and verify their scientific independence and absence of overlapping subjects/windows. For datasets with more than five natural groups, the caller must first provide an explicit validated assignment of those natural groups to five fold groups; this utility does not silently create that assignment. `fit_intercept` is a mandatory boolean with no default; A3's proposed convention remains unapproved.

For every output and each alpha in `{1e-6,1e-4,1e-2,1,100}`, the scaler and ridge model fit only the four training groups. The utility selects the lowest pooled held-out event MSE, resolving exact ties toward larger alpha. No relative tolerance is introduced. Final scaling and fitting use all supplied probe-training events only. Test features/targets are accepted only by the separate evaluation function and cannot alter alpha, scaler or coefficients. Ridge solves the conventional unnormalized sum-of-squared-error plus alpha times coefficient squared norm; the fitted intercept, when explicitly requested, is unpenalized.

Constant feature columns are detected by exact equality, centered exactly and assigned scale one. Nonconstant columns use population standard deviation (`ddof=0`). Nonfinite inputs and invalid numerical scaling raise explicitly. Exact constant training targets return per-output `not_evaluable`, no fit; constant test targets likewise return `not_evaluable`. Other outputs retain their own independent results. Evaluation returns per-target MSE/R² without inventing CRLB support or a cross-target common-valid set; those contracts remain the campaign caller's responsibility.

## Scope and remaining interpretation

The runbook specifies five grouped folds and lowest validation MSE but does not explicitly define weighting for unequal fold sizes. This implementation uses pooled event MSE, equivalent to weighting each fold by its held-out event count. Equal-size folds give the usual mean-fold MSE. The chosen interpretation is exposed here for independent review; it is not a new cohort or author-approved acquisition design. The utility stores fold training/held-out indices and fitted scalers/coefficient evidence for leakage checks, and fits no public or scientific data in its tests.

It is not yet wired to any campaign. Sample counts, IDs, calibration matrices, method information labels, metric target support, and the required intercept choice remain external, explicit inputs. No A3 proposal is converted into a default.

## Bounded verification

Both test invocations used Python `/private/tmp/jepa-venv-cpu311/bin/python`, `PYTHONPATH=src`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `RECONSTRUCTION_DISABLE_TORCH_COMPILE=1`, and a ten-second SIGALRM ceiling. The first invocation passed seven tests in 0.15 s (measured pytest-wrapper time 0.348486 s; tool wall 0.447671 s). A targeted constant-feature rounding improvement then passed the same seven tests in 0.13 s (wrapper 0.278924 s; tool wall 0.382418 s). Total measured tool wall for both is under one second; no timeout occurred.

Fixtures check a direct ridge reference, explicit intercept argument, exact-alpha tie behavior, held-out perturbation invariance of each affected fold's training fit, test perturbation invariance of final fit/alpha, whole-group separation, constant target/feature behavior, nonfinite rejection and event-weighted validation with unequal groups. Held-out perturbations may legitimately change validation loss and chosen alpha; the test verifies that they cannot change that fold's fitted training scaler or coefficients. External test perturbations cannot change either selection or final fit.

SHA256 source binding:

```text
24071be3084dd731cae60e91f7c8c69081c463218c15610c820903a6a5c4dcfa  src/jepa_noise/probes.py
3cc19aef27c3418fcd56359b406aefe119fd56e52cc017e6e69617d43b7073d7  src/jepa_noise/tests/test_probes.py
```
