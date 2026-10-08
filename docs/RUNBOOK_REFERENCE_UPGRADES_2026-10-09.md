# Runbook — Paper 3 reference upgrades, work packages WP1–WP8

**Date:** 9 October 2026. **Kickoff prompt:** `docs/AGENT_PROMPT_REFERENCE_UPGRADES_2026-10-09.md`.
**Design sources (read first):** `docs/REFERENCE_UPGRADES_2026-10-08.md` (R1–R4, decisions §5),
`docs/PREREGISTRATION.md` §§1–5 and §8 (items 1–3, 5, 10–12 decided or revised 8–9 Oct),
`docs/TODO.md` ("Reference upgrades" block).
**Repository:** `/Users/wongdowling/Documents/VERITAS`, branch `dev`, remote `origin` =
`github.com/seanxia8/VERITAS`. Shared with Junjie Xia.

This runbook is for a lead agent that can spawn subagents. It covers eight work packages. It does
**not** cover Phase B, Phase C, building `nuradio_simulation`, editing `latex/paper3_proposal.tex`,
or freezing anything.

| WP | Task | Source | Kind | Lane |
|---|---|---|---|---|
| WP1 | R3 privileged-basis rule: hook declarations, admissibility check, eigengap guard, rotation test | REFERENCE_UPGRADES §2 R3 | code + tests | A |
| WP2 | R4 abstention reason codes | §2 R4 | code + tests | A |
| WP3 | Tier-1 κ_m = 1.10 declared; {1.05, 1.10, 1.25} sensitivity output | PREREGISTRATION §8 item 1 | code + tests | A |
| WP4 | R1 generator: `double_pulse` separation axis, copula option, overlap and tangent statistics | §2 R1 | code + tests | B |
| WP5 | R2 first-order N-harm prediction on the linear subject (exact), plus `jac_input` | §2 R2 | code + tests + dev run | C |
| WP6 | R1(b) linear-subject ρ sweep (development, non-citable) | §2 R1(b), D-R1 | dev run + report | B (after WP4) |
| WP7 | Draft transformer trainability gate (the D10 replacement trigger) | PREREGISTRATION §8 item 10 | doc draft | D |
| WP8 | One-page note for Junjie | REFERENCE_UPGRADES §5 | doc draft | D |

---

## 0. Hard rules

1. **No push, no freeze.** Never run `git push`, `freeze_part`, or anything that creates
   `protocol/frozen/`. The author pushes and opens the PR.
2. **Local commits only by the lead**, one per work package, after its review passes (§4). Subagents
   never commit. No `--no-verify`, no history rewriting, no amending commits that already exist.
3. **The two claims do not move.** Do not change: the Claim-1 or Claim-2 primary estimands,
   `GENERIC_COMMITTED`, `INTERMEDIATE_COMMITTED`, the task-sensitive score, the split fractions,
   the held-out families, or any existing threshold other than the Tier-1 κ_m status (WP3).
4. **Existing numbers must not change.** With all new options at their defaults, the development
   smoke must reproduce the baseline outputs exactly (§2.3). New outputs are added as new keys.
5. **Every number comes from a command you ran.** Quote it. A failed step is reported with the
   exact error, not estimated around.
6. **Development outputs are non-citable.** Write them under `results/<name>_dev_<date>/` with a
   `README.md` that says so. Nothing in `results/confirmatory/`.
7. **Do not touch** `archive/`, `docs/archive/`, existing `results/*` directories, `latex/`, or
   `docs/REVISION_REPORT_*` files written before today.
8. **Backward compatibility.** Keep public names and signatures. Add parameters with defaults that
   reproduce today's behaviour. Do not add required methods to the `Subject` protocol
   (`hypergraph_subject.py` and other subjects must keep satisfying it).
9. **Stop and ask the author** (§6) instead of deciding anything that changes a claim, a committed
   score, a registered family, or an existing number.

## 1. Environment and preflight (lead, before spawning anyone)

```bash
cd /Users/wongdowling/Documents/VERITAS
git status -sb                      # expect: ## dev...origin/dev, clean
git log --oneline -3                # record HEAD
ls .venv/bin/python && .venv/bin/python -c "import numpy, scipy, pytest; print(numpy.__version__, scipy.__version__)"
PYTHONPATH=src .venv/bin/python -m pytest src/latent_monitor/tests -q     # README: 94 passed, 1 skipped (no torch)
.venv/bin/python -c "import torch; print(torch.__version__)" || echo "no torch (TransformerSubject tests skip)"
```

If `.venv` lacks numpy/scipy/pytest, use `uv run --package latent-monitor --with pytest python -m pytest …`
or create `.venv-cpu` as in `README.md` "Running the tests". Record which interpreter you used and
use it everywhere. If the baseline suite is not green, **stop** and report; do not fix unrelated
failures.

### 1.1 Baseline smoke (the regression anchor)

```bash
PYTHONPATH=src .venv/bin/python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_2026-10-09_baseline
```

Keep this directory. §2.3 compares against it.

## 2. Orchestration

### 2.1 Lanes and file ownership

Run the lanes in parallel. A file belongs to exactly one lane; a subagent edits only files its lane
owns. If a task needs a file owned by another lane, the subagent stops and tells the lead.

| Lane | WPs (in order) | Owns |
|---|---|---|
| A — protocol | WP1 → WP2 → WP3 (one subagent, sequential: all three touch `protocol/smoke.py`) | `cumulant.py`, `reference.py`, `support.py`, `protocol/abstention.py`, `protocol/consequence.py`, `protocol/features.py`, `protocol/smoke.py`, new `hooks.py`, tests `test_privileged_basis.py`, `test_abstention_reasons.py`, `test_kappa_sensitivity.py`, plus edits to `test_protocol*.py` that WP3 needs |
| B — generator | WP4 → WP6 (one subagent) | `tier1.py`, new `run_r1_sweep.py`, test `test_tier1_r1.py` |
| C — Jacobian | WP5 | `linear_subject.py`, `torch_subject.py`, new `nharm.py`, new `run_nharm.py`, test `test_nharm.py` |
| D — docs | WP7, WP8 (one subagent, after reading the others' specs, not their code) | new `docs/DRAFT_TRAINABILITY_GATE_2026-10-09.md`, new `docs/NOTE_FOR_JUNJIE_2026-10-09.md` |
| Lead | integration, reviews, commits, final docs | `docs/TODO.md`, `docs/PREREGISTRATION.md`, `docs/REFERENCE_UPGRADES_2026-10-08.md`, `docs/INDEX.md`, `docs/EXPERIMENT_DESIGN.md`, `README.md`, the final report |

Lane C needs physical tangents from the Tier-1 generator. It must not edit `tier1.py`; it renders
perturbed waveforms through a local helper or asks Lane B to expose
`EventFamily.waveform_from_params` (WP4 item 1), whichever lands first. The lead decides.

If your tool supports isolated git worktrees per subagent, use one per lane and merge in the order
A, B, C, D. Otherwise all lanes share the working tree and the ownership table is binding.

### 2.2 Per-WP loop

1. **Implement** (lane subagent). Spec in §3; read the cited source sections first.
2. **Self-check** (same subagent). Its new tests pass; the full `latent_monitor` suite passes.
3. **Review** (a *fresh* reviewer subagent that did not write the code). It gets the WP spec, the
   diff (`git diff` limited to the lane's files) and the test output. It checks the acceptance
   list, the hard rules in §0, and the maths; it answers ACCEPT or a numbered list of defects.
4. **Fix** (lane subagent) and re-review until ACCEPT, at most three rounds; then escalate to the
   author (§6).
5. **Integrate** (lead). Full suite plus the regression check of §2.3, then one local commit:
   `WP<n>: <summary>` with a body listing files and the test count.

### 2.3 Regression check (lead, after every integration)

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/latent_monitor/tests -q
PYTHONPATH=src .venv/bin/python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_2026-10-09_after_WP<n>
```

Compare the new smoke JSON with the baseline: every key present in the baseline must have an
identical value, except the Tier-1 κ_m status/source strings after WP3. New keys are allowed. Write
a small script for this (`scripts/compare_smoke.py`, owned by the lead) and quote its output. Any
other difference is a stop condition.

---

## 3. Work packages

### WP1 — R3 privileged-basis rule

**Read:** REFERENCE_UPGRADES §2 R3; PREREGISTRATION §2 (the 8 Oct R3 sentence); `cumulant.py`
(module docstring, `project_diagonal_third`, `pca_basis`); `protocol/features.py` (where hook
statistics are built); `tests/test_spine.py::test_invariance_group_of_energy_splits_and_task_length`.

**Build:**
1. `hooks.py`: a declaration of every hook of `LinearSubject` and `TransformerSubject` as
   privileged or not, with a one-line reason each. Privileged means an elementwise nonlinearity sits
   between this hook and the next linear map (post-activation MLP units). The linear subject has no
   privileged hook. For `TransformerSubject`, read its `_stages` and decide per hook; if a hook is
   ambiguous, mark it non-privileged and say why.
2. An admissibility registry: each hook statistic in `features.py` is labelled `invariant`
   (O(d)-invariant: shrunk Mahalanobis with a scaled-identity target, energy splits, Frobenius
   norms), `chart` (diagonal statistics in a reference-PCA chart) or `per_unit` (coordinate-basis
   statistics). A function `check_admissible(stat, hook)` raises on `per_unit` at a non-privileged
   hook. Audit `features.py` and list every statistic with its label in the module docstring. If an
   existing statistic is `per_unit` on a non-privileged hook, **do not change it**; record it and
   stop for the author (§6).
3. Eigengap guard for `chart` statistics. On the reference fit, compute the relative gaps
   (λ_i − λ_{i+1}) / λ_i of the retained PCs. Add a declared, unfrozen constant
   `CHART_EIGENGAP_MIN` (proposed 0.05; say so in the docstring). If any retained gap is below it,
   the reference records `chart_degenerate = True`. Report the gaps in the reference's `to_dict`.
   **Do not drop features in this WP**: changing the arm feature set would change existing numbers.
   The flag is reported; what happens when it is set is an author decision (§6).
4. Deterministic sign convention for PCA charts if none exists: flip each PC so that its reference
   third moment is ≥ 0. Check first whether a convention exists; if one does, keep it.

**Tests (`test_privileged_basis.py`):**
- *Function-preserving rotation.* Build a `LinearSubject` as the existing tests do. For a random
  Q ∈ O(k), form the rotated subject with z' = Qz: A' = QA, O' = OQᵀ, D' = DQᵀ (check these against
  the code's shapes). Assert that outputs and reconstructions are unchanged (allclose), that every
  `invariant` statistic on `z` is unchanged, that `chart` statistics are unchanged when the
  eigengap condition holds, and that a deliberately `per_unit` statistic (e.g. per-coordinate
  variance of z) **does** change. The last assertion shows the test has power.
- `check_admissible` raises for `per_unit` at a non-privileged hook and passes otherwise.
- The eigengap flag is set on a constructed degenerate spectrum and clear on a separated one.

**Accept when:** tests pass; the full suite passes; the smoke regression is identical; the
`features.py` audit list is in the docstring; no existing feature changed.

### WP2 — R4 abstention reason codes

**Read:** REFERENCE_UPGRADES §2 R4; PREREGISTRATION §2 (R4 sentence); `protocol/abstention.py`;
`protocol/smoke.py` §3 and §5 (the abstention novelty is the max of the calibrated
`gr_support_novelty` and `gr_out_of_span`); `reference.py` (`P_resolved`, `P_weak`).

**Important design fact.** Abstention is triggered only by out-of-support novelty (support score or
residual out-of-span energy). So every abstained window is out of estimated support, and the
P_weak share decides how to read it. Implement:
- `weak_share = ‖P_weak Δz̄‖² / ‖Δz̄‖²` per window, with Δz̄ the window's mean deviation from the
  reference mean (0 when ‖Δz̄‖ = 0).
- A declared, unfrozen constant `WEAK_SHARE_MIN = 0.5`.
- Among abstained windows: **B** if `weak_share ≥ WEAK_SHARE_MIN` (out of support *and* mainly in
  directions the measurement cannot resolve: retraining alone will not help); **O** otherwise (out of
  support in resolved directions: wider training could cover it).
- Diagnostic only: the count of **non-abstained** windows with `weak_share ≥ WEAK_SHARE_MIN`,
  reported as "U, not abstained".
- Record which component triggered each abstention (`support`, `out_of_span`, or both).

Add the codes to the abstention report (counts overall, per contract and per family) and to the
joint decision table as a new column. No new endpoint, no threshold change.

**Docs (lead, at integration):** update the R4 sentence in PREREGISTRATION §2 and REFERENCE_UPGRADES
§2 R4 to this O/B reading, with one line explaining why U-only cannot occur among abstentions under
the current trigger.

**Tests (`test_abstention_reasons.py`):** synthetic reference with a known P_weak; a window displaced
along P_weak and flagged novel → B; along P_resolved → O; zero displacement → O with weak_share 0;
a non-abstained window along P_weak → counted as "U, not abstained".

**Accept when:** tests pass; smoke regression identical except new keys; the linear subject's
resolved rank is reported next to the codes (if it is full rank, P_weak is empty and every code is O;
say so in the smoke summary rather than hiding it).

### WP3 — Tier-1 κ_m declared, sensitivity curve

**Read:** PREREGISTRATION §5 and §8 item 1; `protocol/consequence.py::HarmThreshold`;
`protocol/mode.py` (what `declared` changes); `smoke.py` line with
`HarmThreshold(1.10, "provisional_dev", …)`; `tests/test_protocol.py` around `HarmThreshold`.

**Build:**
1. Tier-1 smoke threshold: `HarmThreshold(1.10, "declared", source="controlled-tier convention, not a physics requirement; PREREGISTRATION §8 item 1, decided 9 Oct 2026", …)`.
   Update the `HarmThreshold` docstring: `declared` means a frozen value with a cited source, which
   may be a declared convention.
2. **Check `mode.py` first.** Development runs must stay non-citable after this change, and
   confirmatory mode must still refuse (no freeze, dirty tree, no hashes). If `declared` makes any
   dev output `citable: true`, stop and report instead of working around it.
3. Supporting output `claim2.kappa_sensitivity`: for κ ∈ {1.05, 1.10, 1.25}, the primary paired
   ΔAUROC_harm point estimate, both absolute AUROCs, the harm prevalence, and the number of harmful
   cells. No intervals, no inference, labelled supporting. The primary stays at 1.10.

**Tests (`test_kappa_sensitivity.py`, plus edits in `test_protocol.py`):** the three κ values appear;
the 1.10 entry equals the primary; a dev run is still non-citable; confirmatory still refuses.

**Accept when:** tests pass; smoke regression identical except the κ status/source strings and the
new key.

### WP4 — R1 generator

**Read:** REFERENCE_UPGRADES §2 R1; PREREGISTRATION §4 (R1 sentence) and §8 items 3, 11;
`tier1.py` (`EventFamily`, `event_cells`, `Cell.event`, `implied_whitener`).

**Build:**
1. `EventFamily.waveform_from_params(params, n_samples)`: render the waveform from an explicit
   parameter dict. `waveform(event_id, …)` calls it, so both paths agree exactly.
2. `EventFamily.second_offset: float = 0.2`, used by `double_pulse` in place of the literal 0.2
   (keep the `min(…, 0.9)` cap). Default output must be bit-identical to today.
3. `double_pulse_severity_cells(ref, offsets=(0.05, 0.10, 0.20, 0.40))`: evaluation-only cells with
   family name `double_pulse` (so the held-out logic applies unchanged) and labels
   `event_in_span:double_pulse@dt=<offset>`. Do not add them to `all_cells` (that would change the
   smoke); expose them for WP6 and future runs.
4. Copula option: `EventFamily.copula_rho: float = 0.0` on the pair (amplitude, τ_t), Gaussian
   copula mapped to the existing uniform ranges. When `copula_rho == 0` use the existing code path
   unchanged (bit-identity). Document that ρ is the copula (latent-normal) correlation, not the
   Pearson correlation of the uniforms, and report both in tests.
5. `correlation_cells(ref, rho_ref, rho_tests)`: cells labelled `event_in_span:corr@rho=<ρ>`
   (in-support-but-rare) and `event:corr@rho=<ρ>` for ρ_test = −ρ_ref (outside estimated support),
   matching PREREGISTRATION §4.
6. Statistics (functions, not features): `pulse_overlap_cosine(cell, event_ids)`, the Σ̂⁻¹ cosine
   between the whitened first and second pulse templates; `tangent_cosine(cell, event_ids, a, b)`,
   the Σ̂⁻¹ cosine between finite-difference tangents ∂g/∂a and ∂g/∂b (central differences, step
   declared).

**Tests (`test_tier1_r1.py`):** bit-identity of default draws and waveforms for 100 event ids
against values computed before your change (store them in the test from the baseline commit);
empirical rank correlation within tolerance of the target for ρ ∈ {0.5, 0.8, −0.8}; the offset
axis changes only the second pulse; `pulse_overlap_cosine` across the declared offsets (report the
values; assert monotone decrease only if it holds, otherwise report it as a finding for the author);
tangent cosine finite and in [−1, 1].

**Accept when:** tests pass; smoke regression identical.

### WP5 — R2 first-order N-harm prediction

**Read:** REFERENCE_UPGRADES §2 R2 (formula, interference split, scope); PREREGISTRATION §1 (R2
sentence); `subject.py`; `linear_subject.py` (`represent`, `jac_output`, pooling); `tier1.py`
(`sigma_covariance_cells`, `Cell.noise_record`, `noise_batch`); `smoke.py` for how K is computed
(`K_amp / ref_amp`).

**Overlap note.** VERITAS plan v3 WP-C (the worst-case N direction, the top generalised eigenvector
of the pullback) is not in this repository and is not implemented. Implement R2 so that the same
quantities also give the worst-case direction as a by-product, labelled "WP-C-compatible".

**Build:**
1. `jac_input(subject, X, geometry)` in `nharm.py`: ∂y/∂x̃ with respect to the Σ̂-whitened input,
   shape `(n, n_targets, C·N)`. Use a `jac_input` method if the subject has one; give
   `LinearSubject` a closed-form method (y is affine in x̃, so J is constant: compose the pooling
   weights, `A` and `O`); give `TransformerSubject` an autograd method (tests skip without torch);
   fall back to central finite differences otherwise. Do **not** add it to the `Subject` protocol.
2. Estimator that avoids forming Σ̃ (C·N = 2048 on Tier 1): pass whitened noise-only records
   through J and take sample covariances of the outputs,
   `ΔC = Cov_r(J w̃) − Cov_ref(J w̃)`, averaged over reference events when J varies;
   `v_K = ΔC[K, K]`. Report a bootstrap SE over records.
3. Interference split for the K row r = e_Kᵀ J, with whitened physical tangents T (from WP4's
   `waveform_from_params`, or a local renderer if WP4 has not landed): cross-talk c = rT;
   variance-inflation factor [(TᵀT)⁻¹]_KK · [TᵀT]_KK; in-span and out-of-span shares of ‖r‖²;
   per cell, the three terms r_in Δ r_inᵀ, 2 r_in Δ r_outᵀ, r_out Δ r_outᵀ estimated the same
   record-based way (project records onto span(T) and its complement).
4. Worst-case direction (WP-C-compatible): for one output row r, the unit whitened noise
   perturbation that moves y_K most under the reference metric is Σ̃_ref rᵀ / ‖Σ̃_ref^{1/2} rᵀ‖
   (just rᵀ/‖r‖ when Σ̃_ref = I). Report it as a vector summary (its in-span / out-of-span split
   and its overlap with each tangent), not the full vector. One function, one output key.
5. `run_nharm.py --out results/nharm_dev_<date>`: linear subject, the four `sigma_cov:*` cells.
   Output per cell: predicted v_K with SE, realised output-variance excess on noise-only records,
   realised K (as in smoke), the three-term split, cross-talk, VIF; a rank correlation across the
   four cells labelled descriptive (n = 4 < 10). README says non-citable.

**Tests (`test_nharm.py`):** for the linear subject the predicted excess equals the realised output
variance excess within Monte-Carlo error (this is the exactness check); with c = e_K the in-span
share equals the Cramér–Rao row norm; the three terms sum to v_K; finite-difference and closed-form
Jacobians agree for the linear subject; the WP-C direction is a unit vector maximising the
quadratic form on a constructed case.

**Accept when:** tests pass; smoke regression identical (WP5 adds no smoke key); the dev run
completes and its README states the exactness check result with numbers.

### WP6 — R1(b) linear sweep (development)

**Depends on:** WP4 integrated.
**Read:** REFERENCE_UPGRADES §2 R1(b); PREREGISTRATION §8 item 11 (D-R1: linear subject only);
how the smoke fits the linear subject on `reference_fit` windows.

**Run** `run_r1_sweep.py --out results/r1b_linear_sweep_dev_<date>`:
- For ρ_train ∈ {0, 0.5, 0.8, 0.95}: fit the linear subject on a reference family with
  `copula_rho = ρ_train` (same fitting recipe and sizes as the smoke).
- Evaluate on ρ_test ∈ {0, −ρ_train} (skip duplicates at ρ_train = 0) and on the same-ρ clean cell.
- Report: amplitude harm K relative to the same-ρ clean cell; the tangent cosine
  (amplitude, τ_t); support novelty AUROC of the ρ_test = 0 cell (in-support-but-rare) and of
  ρ_test = −ρ_train (outside support) against same-ρ clean windows; seeds 0–4.
- Optional, cheap: the WP4 separation axis on the linear subject (K versus Δt and overlap cosine).
- `REPORT.md`: tables, the declared direction from REFERENCE_UPGRADES §2 R1, whether it held, and
  "development, non-citable". A linear subject may show little harm for |ρ| < 1 (the linear-limit
  statement predicts change mainly as K becomes singular); report that plainly if so.

**Accept when:** the run completes for all cells and seeds; numbers in the report are copied from
the output files; a reviewer re-derives two table entries from the raw output.

### WP7 — Draft trainability gate (author decides)

Write `docs/DRAFT_TRAINABILITY_GATE_2026-10-09.md`, at most two pages. Read PREREGISTRATION §8
item 10, `REVISION_REPORT_2026-09-16_pass2.md` §6 (Phase C), `torch_subject.py::train_reference`,
`TESTBEDS.md` §2.
- Purpose: the predeclared trigger under which the hypergraph subject replaces the transformer.
- Candidate criteria, each with a proposed number and a one-line reason, as tick-boxes for the
  author. At least: clean-reference amplitude error relative to the linear subject on the same
  events; seed stability across ≥ 3 seeds; convergence (no NaN, validation plateau); the
  linearisation check of the Jacobian at reference points; resolved rank of M_recon at least the
  number of physical targets.
- What happens on failure (replace, never add; report the failed gate).
- What it must not do: use evaluation cells, or be tuned after seeing Phase C results.
- Mark every number as a proposal. Do not edit PREREGISTRATION.

### WP8 — Note for Junjie

Write `docs/NOTE_FOR_JUNJIE_2026-10-09.md`, at most one page, in English, plus a 5-line message
version at the end for Slack or email.
- What changed: the PR from `dev` (docs and, by then, WP1–WP6 code), the R1–R4 upgrades in one
  line each, the 9 Oct decisions (D-R1 no; D10 replacement-only; Tier-1 κ_m convention; Claim-2
  rule form; severity principles).
- What we need from him, as numbered asks: (1) confirm the transfer arm, provisionally arm B
  (NuRadioMC), with the reasons from PREREGISTRATION §8 item 5, or argue for LUCiD; (2) the
  published resolution requirement that sets that arm's κ_m; (3) agreement on the proposal wording
  in REFERENCE_UPGRADES Appendix B; (4) comments on the WP7 gate draft; (5) the still-open title
  and two-claim framing item.
- Neutral tone, no claims of results.

---

## 4. Integration order and commits

1. Preflight and baseline (§1).
2. Spawn lanes A, B, C, D together.
3. Integrate as each WP is accepted: WP1, WP2, WP3 (lane A order), WP4, WP5, WP6, then WP7–WP8.
   One local commit per WP after §2.3 passes.
4. Final lead pass:
   - `docs/TODO.md`: tick what is done, with the commit hash; leave the rest open.
   - `docs/REFERENCE_UPGRADES_2026-10-08.md` §6 "Order": mark items done with dates.
   - `docs/PREREGISTRATION.md`: only the R4 wording fix (WP2) and pointers to the new tests; no
     threshold or claim change.
   - `docs/INDEX.md`: add the runbook, prompt, WP7 and WP8 files, and the final report.
   - `README.md`: update the test count line.
   - Final report `docs/REVISION_REPORT_2026-10-09_reference_upgrades.md` (template §7).
5. A final **independent reviewer subagent** reads the whole diff since the baseline commit, the
   final report and the dev outputs, and answers ACCEPT or defects. Fix, then commit the report.
6. Stop. Do not push.

## 5. Time budget

Lane A about 1.5 days, lane B about 1.5 days (WP6 runtime minutes to an hour on CPU), lane C about
1 day, lane D about half a day. Wall clock with parallel lanes: about 2 days. If a WP exceeds twice
its budget, stop that lane and report.

## 6. Stop conditions — ask the author

- The baseline suite or smoke fails before any change.
- Any existing smoke value changes (other than the WP3 κ strings).
- An existing hook statistic is `per_unit` on a non-privileged hook (WP1 item 2).
- `chart_degenerate` is set on the smoke reference (WP1 item 3): what to do with the features is a
  design decision.
- `declared` κ_m makes a development output citable (WP3 item 2).
- The WP5 exactness check fails for the linear subject.
- A WP needs a file owned by another lane, or a change to the `Subject` protocol.
- Three review rounds without ACCEPT.

When stopping, leave the tree with all accepted WPs committed and the blocked WP's changes
uncommitted, and write the blocker into the final report.

## 7. Final report template

```
# Reference upgrades WP1–WP8 — implementation report (date)

Baseline: commit <hash>, interpreter <path>, suite <N passed, M skipped>.
Final:    commit <hash>, suite <N passed, M skipped>, smoke regression <identical | list>.

| WP | Status | Commit | Tests added | Reviewer verdict | Notes |

## WP1 … WP8
What was built (files), what the tests show (numbers quoted), open points.

## Development results (non-citable)
WP5 exactness check; WP5 per-cell table; WP6 tables and whether the declared direction held.

## Decisions needed from the author
Numbered, each with the evidence and the options.

## Not done / deviations from the runbook
```
