# E1c A2 engineering implementation

Scope: approved `E1C_CONFIG_PROPOSAL.md` A2 (registration `f25c7ab`), runbook §4.2/WP3. This is an implementation note, not an E1c run, literature reproduction result or R5 outcome. New files are `src/jepa_noise/dynamics.py` and `tests/test_dynamics.py`; existing packages remain unchanged. Pre-outcome independent review, complete campaign projection and remaining-resource allocation are still required.

## Fixed moments and stop-gradient mathematics

The data builder constructs `ExactModel(seed=seed,theta_scale=.4,eta_scale=3,share='target')` for each approved seed/gamma. It retains the source loading seed and replaces the sample RNG by the independent `E1c/data` namespace keyed by seed and gamma. The resulting fixed batch has 100000 paired samples, centered once with empirical moments divided by n. The builder returns moments and true separate loadings for reuse across depths. Initial/final data RNG states and namespace keys are returned. It is deliberately **not called** by engineering tests. The existing ExactModel helper also draws auxiliary calibration noise; E1c does not use it, but projection must charge that generation cost.

Encoder order is `W=W_(L-1)...W_0`, predictor P. Let `C=E[x_c x_c.T]`, `T=E[x_t x_t.T]`, `K=E[x_c x_t.T]`, and V be a value-copy of W held constant during differentiation. With A=PW, the loss is `0.5 tr(A C A.T)-tr(A K V.T)+0.5 tr(V T V.T)`. Therefore `G_A=A C-V K.T`, `G_P=G_A W.T`, and `G_W=P.T G_A`. Each layer gradient is `suffix.T G_W prefix.T`. All layers and predictor update simultaneously with learning rate .001, using gradients at the same old iterate and fixed target V. There is no d divisor, decay, clipping, momentum, isotropy constraint or EMA.

Finite differences explicitly hold V fixed. Recomputing V at a perturbed encoder differentiates another loss; a test verifies this produces a different derivative. Direct sample residual loss and predictor gradients independently check the moment formula and reduction. The default target is refreshed only when evaluating the next iterate, as required by the tied stop-gradient model.

## Initialization, measurements and stops

Every layer has a separate `E1c/init/layer/<index>` Gaussian namespace and the square predictor uses `E1c/init/predictor`, each keyed by seed with SD .001. Depth is deliberately absent from these keys, so shared layer indices and predictor are exactly paired across depths. Production runner validates d40, registered depths, n100000 empirical-moment source and fixed100000-step horizon. A shorter explicitly marked engineering call returns `engineering_incomplete` and an unevaluable R5, never a shortened scientific result.

Absolute measurements are `||W A_i|| sd(theta_i)` and `||W B_c|| sd(eta)`, where B_c is the actual context nuisance loading. Relative measurements divide by Frobenius norm(W) and remain diagnostics. Initialization is recorded as step0 and every completed update adds a measurement; complete curves have100001 rows. Existing `metrics.learning_order` supplies the unchanged runbook growth/floor/terminal-plateau/sustained-crossing censoring checks and requires every mode to be eligible. No apparent collapse is made eligible through normalization.

The caller must pass a positive remaining CPU allocation. Resource exhaustion and numerical instability preserve partial curves, final parameters, status and reason. They do not cause automatic restart or a replaced seed. There are no100000-step outcomes or training benchmarks in this implementation work.

## Diagonal engineering fixture and limits

A separate population fixture has positive diagonal marginal moments and diagonal cross moments. Tests use diagonal square layers/predictor and verify that one simultaneous update preserves coordinate decoupling. This supplies a source-compatible diagonal algebra construction for the equations cited by A2. It is not a reproduction of a published full trajectory, and no published convergence theorem is imported or asserted for the campaign's Gaussian random-loading initialization. Population fixture moments never replace the finite empirical batch in the production runner.

No additional scientific choice was required for the algebraic engine. The exact namespace strings and matrix-product order above are explicit implementation conventions for independent review. Full trajectory execution, choice-free100000-step curve retention/checkpoint serialization, immutable manifests, budget enforcement around caller preprocessing, R5 depth comparisons and review acceptance remain orchestration work. Long-running statistical outputs are not produced by unit tests. The engine does not itself serialize checkpoints to disk; callers must persist returned parameters/curves and manage external job termination safely.

## Verification and provenance

Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src /private/tmp/jepa-venv-cpu311/bin/python -m pytest src/jepa_noise/tests/test_dynamics.py -q`, executed under a30-second subprocess timeout. **8 tests passed in .14 seconds**; measured subprocess active wall was **.5587219582 seconds**, one numerical thread. No test failure occurred. Tests include intentional numerical-failure/resource-stop fixtures, whose expected STOP outputs passed. `git diff --check` passed. No campaign batch,100000-step trajectory, substantial benchmark, GPU job, commit or push was run.

SHA256:

```text
60cdd08b6f49cecc4900ae29dfdc79e26bf994f33ddfd9b965ed6578dcc91f6a  src/jepa_noise/dynamics.py
4c7886c812e91150e3f56f4bf4afd631537111745c1fe94d2afd747d82cbc562  src/jepa_noise/tests/test_dynamics.py
```
