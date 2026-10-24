# WP5 — LUCiD arms, anchor penalty and probe forms (development status)

Owner: WP5 implementer (objective/penalty layer only). Scope (new files only):
`src/jepa_noise/lucid_arms.py`, `src/jepa_noise/tests/test_lucid_arms.py`,
`src/jepa_noise/configs/lucid_arms_v0.json`, this file.
Binding: runbook v4 §3.5.6 (seven arms, the `mae_chi2` covariance, the anchor
penalty, the four probe forms), §3.5.5 (encoder/read-out/masking), §3.5.3
(nominal twin, dark-noise-only records, gains before threshold), §3.5.4
(views/masks), §3.4 (A6 arms), §3.5.8 (seed floor) and §3.5.9 (R6–R8, R7b, R11,
R11b, R12). No existing file was modified; no cohort, checkpoint or trainer is
part of this WP, and the no-touch packages were not imported.

**Status: development only.** The readout is `readout_synthetic_unrepaired`; no
number here may enter `PREDICTIONS.md`, a registered cell or a gate. Local tests
use the synthetic geometry `lucid_bombs.from_synthetic` (76 PMTs / 4 modules × 19
in the small fixture, NOT WCTE) and tiny synthetic runs.

## What is implemented

1. **The seven §3.5.6 arms plus the secondary dark variant**
   (`LucidArm`, `build_arm(arm, geometry, ...)`), each a loss on the
   `point_encoder` outputs and masks:
   - `mae_mse`: a linear head on the predictor's target-PMT outputs regresses
     the hidden PMTs' token features — log charge, time (masked to hit PMTs) and
     hit flag — each by squared error.
   - `mae_chi2`: a 2-output head predicts `mu_p >= 0` (softplus), the expected
     **recorded** charge (zero below threshold), and the first-hit time. The
     charge term is the registered module covariance below plus a Gaussian time
     NLL (reference transit-time sigma 1.5 ns = 0.015 in the 100 ns token unit)
     on the hidden hit PMTs. Returned with
     `metrics["uses_oracle_noise_information"] = True`.
   - `ijepa_ema`: an architecture-identical EMA copy of the encoder (cosine tau
     0.996 → 1 over the registered 10 000 steps, `update_target(step)`), target
     tokens layer-normalised (`LayerNorm(elementwise_affine=False)`), prediction
     L2. The target encoder is never in an optimiser group.
   - `sigreg_jepa`: shared weights, no stop-gradient and no EMA; prediction L2
     (MSE between the predictor output and the **target-only** read-out at the hidden
     PMTs, `encode(features, hidden_mask=~target)`; the full unmasked pass is not
     the label any more, see DECISIONS 2026-10-23) plus
     `lambda * SIGReg(context_pooled, target_tokens)` with the pinned
     `jepa_noise.sigreg` implementation and its own reduction.
   - `sigreg_jepa_anchored`: `sigreg_jepa` + β × anchor penalty.
   - `anchor_only`: SIGReg + the same penalty, no prediction loss (the predictor
     is constructed for architecture identity but receives no gradient).
   - `oracle_nominal_twin`: the `sigreg_jepa` objective on the nominal twin
     read-out. The arm refuses to run without `oracle=True` **and** a twin batch
     (mirroring `lucid_bombs.nominal_twin`); every other arm refuses oracle
     access.
   - `sigreg_jepa_anchored_dark` (secondary, R7b): draw 1 only (dark overlay, no
     per-module rescale), its own β.
   - β is required for `sigreg_jepa_anchored`, `anchor_only` and the dark variant,
     and must be in the registered grid; no value is chosen in this WP.
2. **`mae_chi2` covariance** (`chi2_module_terms`, `chi2_dense_module`,
   `LucidArm._forward_mae_chi2`). Registered formula, with `r = q - mu` for the
   hidden PMTs of module `m`:

   ```
   C_m = diag(mu_p (1 + sigma_spe^2) + v_dark + s_p^2 mu_p^2) + s^2 mu_m mu_m^T
   loss = 0.5 * sum_m (r^T C_m^-1 r + ln det C_m)
   ```

   `s` and `s_p` are the **cell's true spreads** (passed at build time), and
   `sigma_spe = 0.3`, `v_dark = 0.0` come from the reference generator config.
   Computed with Woodbury and the determinant lemma, `D` the diagonal part and
   `u = mu_m`:

   ```
   C^-1 = D^-1 - s^2 D^-1 u (1 + s^2 u^T D^-1 u)^-1 u^T D^-1
   ln det C = sum ln D + log1p(s^2 u^T D^-1 u)
   ```

   Vectorised per module: the hidden PMTs are scattered into `(B, M, K)` slots
   (`K = pmts_per_module`); non-hidden slots are neutral (`D = 1`, `u = r = 0`)
   and contribute exactly zero, so the loop over modules is a tensor contraction.
   The loss is the registered event sum over modules, mean over the batch.
3. **Anchor perturbation and penalty** (`LucidArm.perturb`, `anchor_penalty`,
   `DarkBank`, `dark_bank_from_store`). Per event, one perturbed copy `x~` with
   - an independent dark-noise-only window overlaid: union of hits; charges add
     where a PMT is hit in both; the earlier time kept; the dark charge alone
     where only the dark window hits;
   - every module's charges multiplied by an independent `N(1, 0.01^2)` factor
     (the assumed per-module calibration uncertainty).

   `penalty = beta * mean_{batch, modules} ||z_m(x~) - z_m(x)||^2 /
   trace(Cov(z))`, where `z_m` is the module-pooled **unmasked** read-out token
   (Tier-1 pattern in `jepa.py`) and the trace covariance uses the flattened
   `(B * M)` module tokens. The trace denominator keeps its gradient. The
   rescale is applied in the log-charge domain, so a unit factor and an empty
   dark record leave the features bit-identical and the penalty is exactly zero.
   The dark variant uses `with_rescale=False`; both variants share the code path.
4. **The four probe forms** for the six registered targets:
   - `build_probe_dataset(runs, kind, form)` with
     `form ∈ {per_event, run_mean, propagated, raw_input}` and
     `kind ∈ {vertex, log_yield, module_gain, pmt_gain, f2_variable, age}`.
     `run_mean` is the primary form (`PRIMARY_PROBE_FORM`, and the config's
     `probes.primary_form`).
   - `per_event`: full-input read-out features; vertex/log yield/F2/age use the
     mean over all tokens, module gain the module-pooled token, per-PMT gain
     each hit PMT's token (one ridge shared by all PMTs).
   - `run_mean`: the per-event features averaged over a run's probe events; R²
     over (run, module), (run, PMT) or runs.
   - `propagated`: the same probe on the context-encoder outputs at masked
     positions with the standard training mask (`within`: odd PMTs; `across`:
     the target half), run-mean over events in which the position is masked.
   - `raw_input`: the ceiling on raw token features (module-mean raw features
     for module targets); the encoder is never touched, and the test proves it
     by building this form from NaN embeddings.
   - Ridge: v3 §1.3 via `jepa_noise.probes` (`ALPHAS = (1e-6, 1e-4, 1e-2, 1,
     100)`, five run-grouped folds, fold-internal standardisation, ties to the
     larger alpha); `fit_probe` / `probe_predictions` / `probe_r2`; constant
     targets are `not_evaluable` with the probe module's reason.
   - `random_init_floor(arm, geometry, seed=the arm's first seed)` returns an
     untrained encoder of the same §3.5.5 architecture; `probe_data` freezes a
     run's full-input and context read-outs and aligns its truth to the events
     `tokenise` kept. For `oracle_nominal_twin` the caller freezes the probe on
     the **twin** batch (the arm's own input) while the truth targets stay the
     same, as §3.5.6 requires.
   - `squared_correlation` + `probe_predictions` expose R11's cross-leakage
     statistic; the full R11/R11b/R12 evaluation is the evaluation layer's job.
5. **Optimiser groups** (`LucidArm.parameter_groups`,
   `check_parameter_coverage`): Muon for 2-D matrices inside encoder/predictor
   blocks, AdamW for embeddings, heads, norms and queries; the EMA target
   encoder is excluded; every trainable non-EMA parameter appears exactly once.
   **No learning rate, schedule or optimiser is constructed here** — all are
   `pending_review` in the config.

## Exact code paths

| Item | Path |
|---|---|
| arms, anchor, covariance, probes | `src/jepa_noise/lucid_arms.py` (886 lines) |
| tests | `src/jepa_noise/tests/test_lucid_arms.py` (313 lines) |
| config | `src/jepa_noise/configs/lucid_arms_v0.json` |

Entry points: `la.load_config()`, `la.ArmSettings.from_config(cfg, cell=(s, s_p))`,
`la.build_arm(arm, geometry, settings=..., beta=..., seed=..., dark_bank=...)`,
`arm(batch, step=...)`, `arm.anchor_penalty(z, z_tilde)`, `arm.perturb(...)`,
`arm.update_target(step)`, `la.probe_data(model, batch, truth, geometry)`,
`la.build_probe_dataset(runs, kind, form)`, `la.probe_r2(train, test)`,
`la.random_init_floor(arm, geometry, seed=...)`.

## Tests and numbers

Command (single-threaded BLAS, CPU):

```
PYTHONPATH=src /private/tmp/jepa-venv-cpu311/bin/python -m pytest \
  src/jepa_noise/tests/test_lucid_arms.py -q -s
```

14 tests, about 2 s standalone wall on the Mac CPU (12.9 s while the full suite ran concurrently). Coverage: every arm's forward+backward
on batch 2 with finite gradients; distinct loss values; the two required
`mae_chi2` covariance tests; anchor zero/positive/gradient properties; the dark
variant; oracle access refusal; all four probe forms; constant targets; the raw
ceiling; the encoder probe path; the random-init floor; parameter groups and the
EMA update.

Arm losses in the tiny smoke (SMALL geometry, batch 2, `within`, seed 0,
β = 10 where applicable, a 3-record dark bank):

| Arm | loss |
|---|---|
| `mae_mse` | 4.6665 |
| `mae_chi2` | 1781.9683 (charge 569.8783 + time NLL 1212.0900) |
| `ijepa_ema` | 2.6977 |
| `sigreg_jepa` | 5.3794 |
| `sigreg_jepa_anchored` | 5.6961 |
| `anchor_only` | 0.5906 |
| `oracle_nominal_twin` | 5.3794 (= `sigreg_jepa`: same seed, twin batch = the batch) |
| `sigreg_jepa_anchored_dark` | 5.6941 |

The `mae_chi2` value is large because the untrained softplus mean is ~0.7 PE
against charges of several PE (the charge term is the registered sum over hidden
PMTs, not a mean per PMT) and the untrained time head predicts ~0 against first
hits at ~50 ns with `sigma_t = 1.5 ns` (the time NLL is a mean over 64 hidden
hits here). The gradient is finite; `chi2_floor_fraction = 0.0` at init (no mean
variance was clamped). These are smoke numbers, not trained values.

**`mae_chi2` covariance tests.**

(a) *Woodbury vs dense, one module.* `chi2_dense_module` builds `C_m` densely and
computes `½(rᵀC⁻¹r + ln det C)`; the Woodbury terms agree to
`rtol = 1e-9` (in fact exactly in float64 on the fixture), and a module with no
hidden PMTs contributes exactly `0.0`. The fixture also shows that with
`v_dark = 0` and `mu = 0` the dense covariance is singular (the call raises), and
that the configured floor makes it finite — the reason the floor is a reported
gate.

(b) *Monte-Carlo covariance.* `simulate_module_readout` draws the `C_m` model at
fixed `mu = (4, 6, 9, 12, 8)` PE, `s = 0.10`, `s_p = 0.10`, `sigma_spe = 0.3`,
`v_dark = 0`, 400 000 samples, seed 7, and compares the empirical covariance with
`C_m` entry by entry. Relative error (%):

```
[[0.13 2.76 0.54 2.91 4.25]
 [2.76 0.16 0.05 0.33 4.19]
 [0.54 0.05 0.09 0.00 1.32]
 [2.91 0.33 0.00 0.10 1.15]
 [4.25 4.19 1.32 1.15 0.27]]
max = 4.25 %   (requirement: < 10 %)
```

The largest errors are the smallest off-diagonal entries (`s² mu_i mu_j` with
`mu_i = 4`), which carry the largest Monte-Carlo relative noise; the diagonal
errors are 0.1–0.3 %.

## Config freeze status

`configs/lucid_arms_v0.json` (`jepa-lucid-arms-v0`) is **development**, not
frozen for outcomes.

| Block | Status |
|---|---|
| β grid {0.1, 1, 10, 100, 1000}; per-arm β | grid registered; each β `null` → `pending_development_cohort` (§3.5.6 largest-β rule) |
| EMA τ 0.996 → 1 cosine, 10 000 steps | registered by §3.5.6/§3.5.5 |
| SIGReg slices/quadrature/λ = 0.02, reduction | pinned to `sigreg.py` and its source SHA; no new numerics |
| `mae_chi2` σ_spe = 0.3, v_dark = 0.0, time σ = 1.5 ns | from `lucid_bombs` / the generator config |
| variance floor 1e-6 PE² | **reported gate**, not a decision (see ambiguities) |
| anchor rescale SD 0.01, dark-overlay convention | registered by §3.5.6 |
| cell `s = 0.05`, `s_p = 0.10` | the development cell; the main grid passes `s` per cell |
| optimiser, lr, schedule, warmup, clip | **`pending_review`** (every field) |
| probes: run-mean primary, alpha grid, 5 run-grouped folds, fold-internal standardisation | registered by §3.5.6 / v3 §1.3 |

## Engineering choices (one line each)

- Mean recorded charge `mu = softplus(head)`; the charge residual uses the
  hit-flagged `exp(log q)` from the token features (0 for unhit PMTs).
- The variance floor is applied per PMT to `D` only (`variance_floor_pe2`,
  1e-6 PE²), so non-hidden slots stay exactly neutral.
- `mae_chi2`'s time NLL is `0.5 * mean_hit((t - t_true)/sigma_t)^2`, a mean over
  hidden hit PMTs (the charge term stays the registered sum over modules).
- `mae_mse` trains the hit flag by squared error without a sigmoid (registered
  wording: "squared error on ... hit flag"), and the time term is masked to hits.
- `mae_chi2` carries a linear head with 2 outputs; `mae_mse` 3 outputs.
- The hidden PMT set for the arm losses is `batch.hidden` (natural occupancy);
  rectangular hidden views are required, and the balanced-occupancy ablation is
  a probe-time option, not an arm loss.
- The time token is compared in the 100 ns unit; `sigma_t = 1.5/100` from the
  config, not refitted.
- `ijepa_ema`'s target LayerNorm has no affine parameters; the target encoder is
  a deep copy excluded from the parameter groups.
- `anchor_only` builds the predictor for architecture identity but gives it no
  gradient (the registered "no prediction loss").
- The anchor's `z_m(x)` reuses the unmasked read-out already computed by
  `sigreg_jepa`; only the perturbed copy costs an extra encode.
- The trace denominator flattens the `(B * M)` module tokens and uses the
  unbiased `B*M - 1` divisor, matching the Tier-1 `_trace_cov`.
- The dark overlay applies the rescale in the log-charge domain and does not
  re-threshold the hit set (the rescale is a calibration perturbation, not a
  readout re-simulation).
- `LucidArm.__init__` seeds `torch.manual_seed(seed)` so the init, the floor hook
  and the anchor draw sequence are reproducible.
- The probe fold map is "sorted runs, round-robin over five folds"; the probe
  ridge reuses `probes.grouped_ridge` unchanged.
- `run_mean` refuses `vertex` and `log_yield` because their truth is per event
  (no run-level target exists).
- Probe truth per event is aligned by `event_id` when `truth["event_ids"]` is
  present, because `tokenise` drops N_min-excluded events.
- `raw_input` mean features include the fixed geometry columns; they become
  constant columns and the ridge's fold-internal standardisation sets their
  scale to 1.

## Pending

- **WP5 next task:** trainer and cohort wiring — loader over the generated runs,
  the frozen optimiser/lr/schedule, checkpointing, the β development sweep and
  the 162/39 registered trainings.
- **GPU smoke** on the §6.1 hosts (forward/backward memory and throughput of the
  anchored arms, which need three encoder passes).
- **Scaling benchmark** (§3.5.5, no training): 2k/10k/40k tokens, peak memory and
  events/s.
- **Evaluation layer:** run-directory truth assembly (beyond
  `truth_from_state`), probe-train/probe-test set construction, random-init
  floors per arm, R6–R8, R7b, R11, R11b, R12 and the cross-leakage criteria.
- **WP5 config review:** freeze the optimiser, learning rate, schedule, warmup,
  clipping and the per-arm β values.

## Ambiguities and open questions (reported, not decided)

1. **`v_dark = 0` makes the registered covariance singular at zero mean.** A
   hidden unhit PMT has `mu = 0`, so `C_m = diag(0)` and `ln det C_m = -inf`. The
   config's `variance_floor_pe2 = 1e-6` keeps the loss finite but changes the
   log-determinant term for exactly those PMTs, i.e. it changes a registered
   loss. Flagged as `mae_chi2.variance_floor_status =
   gate_reported_not_decided`; options are (a) keep the floor, (b) use the
   generator's dark-count Poisson variance instead of `v_dark = 0`, (c) replace
   the log-determinant on zero-mean PMTs. This is a gate, not an engineering
   choice.
2. **"Hidden PMTs" in the module covariance.** `C_m` is written for "the hidden
   PMTs of module m"; the implementation uses the hidden-submatrix of the full
   module covariance (`s² mu_h mu_hᵀ` restricted), which is exactly the
   submatrix — recorded here because the alternative (recompute `mu_m` over all
   19 PMTs and then restrict) is not the same matrix.
3. **Run-level truth for the physics probes.** R6/R7's physics quantity is the
   vertex resolution (not a probe R²), so no run-mean vertex target is defined
   here; the evaluation layer must keep vertex/yield probes per event.
4. **The anchor read-out.** The penalty is defined on `z_m`, the module-pooled
   read-out token; the runbook does not say whether `z_m` is taken from the
   masked or unmasked encoding. The Tier-1 pattern (unmasked) is used, matching
   `jepa.py`.
5. **`anchor_only`'s SIGReg target set.** With no predictor there are no
   predicted tokens, so SIGReg uses the context-pooled value and the full-input
   target-token value (the `sigreg_jepa` reduction) rather than a prediction.
   If the author intended SIGReg on the context side only, the loss changes.
6. **Beta for `anchor_only`.** §3.5.6 says the development cohort chooses β for
   `sigreg_jepa_anchored`, `anchor_only` and the dark variant; the code requires
   a β for all three and selects none.


## Resolution of the variance-floor ambiguity (2026-10-17, Lead ruling)

`v_dark = 0` was a placeholder from the pre-generator scaffold. The generator does draw dark
counts (4200 Hz over the 1000 ns window = 4.2e-3 counts per PMT per window), so the reference
dark-charge variance is `lambda_dark (1 + sigma_spe^2) = 0.004578 PE^2`; the runbook defines
`v_dark` as exactly this quantity. With `v_dark > 0` the module covariance is positive definite even
for a hidden unhit PMT with `mu = 0`, so **no variance floor is applied** (`variance_floor_pe2 = 0`,
retained only as a guard that is required when a non-registered configuration sets `v_dark = 0`).
This is an implementation correction against the registered definition, not a change to a
registered loss: the log-determinant is now finite because the covariance is the correct one. The
detectability config's `v_dark` is updated to the same value; the frozen decision JSON is re-run so
that the evidence matches the committed config (the ceilings shift negligibly — dark charge is
~1e-2 PE against signal charges of 10^1–10^2 PE — and no pass/fail outcome moves).

Other readings confirmed and recorded: `C_m` is the hidden-PMT submatrix of the module's
covariance; `run_mean` is primary but the vertex/log-yield targets stay per-event (they are
per-event truths); the anchor's `z_m` is the unmasked module-pooled read-out (Tier-1 pattern);
`anchor_only` regularises the context-pooled and full-input target tokens (no predictor exists in
that arm); `anchor_only` requires its own β selected on the development cohort like the other
anchored arms.
