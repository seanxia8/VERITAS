# JEPA pre-launch review — 16 October 2026

**Verdict: revise before registration and training.** The original toy calculation reproduces, but it does not validate the full theory, data contract or executable experiment specification. No runbook/prompt changes, registration commit, remote job or training were performed during this review.

Two fresh-context agents independently reviewed (1) theory/scientific design and (2) execution/repository compatibility. The coordinating agent separately reproduced the original toy, checked a population counterexample and verified TIDMAD's published channel contract. Their detailed reports are `INDEPENDENT_PRELAUNCH_REVIEW_2026-10-16.md` and `EXECUTION_PRELAUNCH_REVIEW_2026-10-16.md` in this folder.

## Priority findings

### 1. P5 requires a post-innovation signal-rank condition (P1)

Runbook lines 200–207 claim the context-first joint whitener recovers the physics reference subspace. Lower-triangular whitening prevents target-to-context leakage, but its target innovation can cancel the signal as well as the correlated noise. With independent unit-variance theta, eta and private noises, take context `[theta + eta + e1, e2]` and target `[theta/4 + eta/2 + e3, e4]`. The noise covariance is positive definite and the original target physics loading has rank one. After context-first whitening the target physics loading is zero; the cross-view covariance is approximately `1.2e-16`. CCA therefore cannot identify the desired physics direction.

Reproduction: `check_p5_counterexample.py`, with output in `P5_COUNTEREXAMPLE_OUTPUT.txt`. Restrict the proposition to cases in which the residual target physics loading retains the required rank and supply a proof. Retain an annihilation case as a negative control. A successful random toy does not establish the unconditional statement.

### 2. TIDMAD's channel contract is wrong (P1)

Runbook lines 579–580 describe ch0 as injection reference and ch1/ch2 as two SQUIDs; lines 601–604 propose a ch1-to-ch2 split. The release's training files have **CH1 SQUID and CH2 injected reference**, as confirmed both by `src/tidmad_transformer/data.py:1–6` and the [TIDMAD paper, section 2](https://arxiv.org/html/2406.04378v2#S2). A channel split following the current prescription would use the clean reference as a view, changing the scientific task and violating the intended no-reference-input rule.

Bind roles to exact HDF5 paths, verify actual file schemas, and remove unavailable channel splits or choose a justified alternative using measured data only. Do not infer a second noisy detector channel from channel numbering.

### 3. E2 does not implement the declared temporal gap (P1)

Runbook lines 413–414 define E1b context `[0,112)` and target `[144,256)`. Lines 460–463 instead use 32-sample patches 0–3 and 4–7 and describe masking one loss patch. The intended gap `[112,144)` intersects both patches 3 and 4. Excluding a loss term does not remove those samples from encoder attention.

Choose an explicit sample/token-compatible partition, state any difference from E1b, and test invariance of context outputs to all target/gap samples before training. Freeze the corrected split before computing sharing levels, mask scores or experiment cells.

### 4. R6 conflicts with the development-cell exclusion (P1)

Runbook line 242 requires a three-level nominal-SNR correlation; lines 533–534 reserve the mid/nominal cell for tuning/smoke and say it is never pooled. As written, the primary statistic cannot use all three specified levels while honoring that exclusion.

Declare a separate development dataset/seed population and an independent mid/nominal evaluation cell, or explicitly revise the primary statistic. Do not silently put tuning observations into the reported primary correlation.

### 5. Criterion M cannot be calculated from noise-only records alone (P1)

Lines 214–220 define a difference between an event-signal canonical correlation and a noise correlation. Lines 499–501 instruct the implementer to compute it from noise-only records. The event-signal term is not identifiable from those records alone.

Specify the disjoint development event/noise data, the covariance subtraction/regularization procedure, and sample dimensionality for M. Keep its selection independent of held-out evaluation. Alternatively define a genuinely noise-only criterion and acknowledge that it is a different design.

### 6. E1b lacks the fixed linear signal subspace assumed by its theorem-based checks (P1)

The theory defines `x = A theta + noise` with a fixed rank-k signal map. E1b uses the nonlinear waveform family varying amplitude, arrival time and decay time (`src/latent_monitor/tier1.py`, `EventFamily`). A three-parameter nonlinear family need not span a rank-three linear space. An exact covariance of noise alone does not define the missing fixed signal matrix.

Separate an exact linearized experiment with a declared expansion point and valid range from the nonlinear empirical waveform experiment; or explicitly define and justify the reference subspace and revise the interpretation of angle thresholds. Do not label a nonlinear result a mandatory reproduction of P3.

### 7. The exact full-record covariance has a DC nullspace (P1)

The execution reviewer verified zero DC power in the current Tier-1 reference generator. The full 8-channel, 256-sample noise covariance therefore has rank at most 8 x 255. Runbook lines 508–512 require an inverse lower Cholesky of that full covariance, which does not exist as an exact positive-definite operation. The full-record ordinary Fisher inverse also needs correction. The existing whitener's unit treatment of the zero bin is not proof of positive noise variance. Define a consistent support-restricted model or a revised generating law before claiming an exact oracle; see execution finding E01.

### 8. Noise anchoring's exact theorem and practical objective must be separated (P1)

P4 establishes the constrained noise-only GEVP solution. It does not prove that prediction plus a finite-beta penalty, or the soft-SIGReg deep objective, has that same solution. The scientific reviewer supplies a realizable two-axis population counterexample where beta=10 still favors nuisance. Keep R4 as an empirical hypothesis, narrow the summary's exact-recovery wording, and do not treat legitimate failure of that hypothesis as a software defect.

### 9. E3 protocol and campaign resource accounting remain incomplete (P1)

Before E3, specify file/window roles, probe splits, training steps and batch size, hyperparameter transfer, input axes, crop compatibility and target extraction. Validate amplitude variance before computing amplitude R-squared. Before substantial smoke/tuning, account for the whole campaign; repeat the E3 cost projection after its fallback crop and stop if it still exceeds the cap. Detailed evidence and remedies are in execution findings E06–E07; E08–E10 cover dependency imports, optimizer parameter coverage and metric definitions.

## Checks completed and limits

The existing `toy_check_2026-10-16.py` was run with `.venv/bin/python`:

| Quantity | No shared nuisance | Shared nuisance |
|---|---:|---:|
| CCA maximum angle to reference | 0.9 degrees | 89.3 degrees |
| Noise-anchored maximum angle | 1.0 degrees | 0.9 degrees |
| Per-view whitening canonical correlations | Unchanged | Unchanged |

All match Appendix A.1. This supports the displayed toy, not P5 (which the script does not test) or the deep/real-data predictions. The finite-beta combined objective must be distinguished from P4's constrained noise-only GEVP theorem: an empirical prediction may fail without being a code bug.

The independent reports contain additional execution and metric-definition findings, including E3 partitions/schedule and covariance support. Resolve their substantiated blockers before launching the affected work package. This is a pre-launch review, not exhaustive proof verification or a completed novelty assessment.

## Recommended repair order

1. Correct P5's assumptions and distinguish theorem statements from finite-beta/deep empirical hypotheses.
2. Define E1b's signal reference and covariance support.
3. Repair TIDMAD roles, view candidates and evaluation partitions/targets.
4. Make sample masks, criterion-M fitting data and development/evaluation populations explicit.
5. Resolve the execution-review findings, run bounded analytic/interface checks, and obtain a fresh review of the changed specification.
6. Only then create the registration commit and execute WP0 onward under the existing push/GPU permissions.

The launch prompt is a runnable instruction to commit/freeze the current plan. Do not hand it to an executing agent unchanged while these findings remain open. The scope of this turn was review; no repair has been silently applied.

## Reviewed source hashes

- Runbook: `61ab3399f3ff32884e57054960f3e6040674f0a4a4d897243b69461f16e33a59`
- Launch prompt: `02a6dc9f9e47040667b9f6237086237897e37a3fa1ad2b467fa053326c3babd6`
- Toy script: `fe29f96554bb0fd8ccba324b87b77be1eac716105c491ec2572c557a235d1283`
