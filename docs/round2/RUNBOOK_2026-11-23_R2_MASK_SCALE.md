# Runbook — Shortcut budget and mask scale: interpolation diagnostic, patch-size sweep, readout-seeded LUCiD fit (r2-mask), 23 Nov 2026

**Phase** `r2-mask`. **Branch** `mask-scale`, created from `origin/tier1`. **Worktree** `~/Documents/VERITAS-worktrees/mask-scale`. **Results** `results/round2/20261123_mask/`.

Runs in parallel with `r2-tier1` (still finishing its gate report) and must not disturb it.

---

## 0. Why this runbook exists

The first un-bundled Tier-1 numbers (SK, sealed set, 2 seeds) say that the time-loss weight changes the final vertex error by 2–3 % (not clear), while the mask changes it by 13–14 % (clear). The candidate lead claim is therefore:

> On a fixed sensor array, masked pretraining learns only what cannot be predicted more cheaply. Cheap routes are (i) the PMT's identity alone and (ii) interpolation from visible neighbours. How much target variance is left after both can be computed before training, and it predicts the pretraining gain.

This runbook tests that claim cheaply, before Friday's decision, with three pieces: **M1** a network-free diagnostic, **M2** a patch-size sweep, **M3** a trial of the frozen readout as the starting point of LUCiD's likelihood fit.

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `mask-scale`:

1. **Author, 23 Nov 2026, 19:54 CET:** "我同意，我把这三件事写成一份短 runbook。" (Translation: "I agree; write these three things up as a short runbook." The three things: the CPU interpolation diagnostic, the patch-size sweep, and the 200-event seed trial; the SK part of the sweep was offered at about 30 GPU-h.)
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决". Never stop to ask; record decisions in `DECISIONS.md` and `$CC_STATE/notes/r2-mask.md`.
3. **Author, 22 Nov 2026:** do not depend on the collaborator.

### 0.2 Binding rules

- **Pre-registration.** Copy §1 verbatim to `docs/round2/PREREG_2026-11-23_R2_MASK.md`, append the M1 numbers it asks for, commit and push **before any M2 run is evaluated**.
- **One factor.** Every M2 cell is arm `mae_repaired` with the default recipe of that detector (encoder window, shuffled loader, 50 ns time unit, 10,000 steps, micro-batch of the existing run). Only the mask changes.
- **Sealed set and frozen readouts.** The rules of `RUNBOOK_2026-11-22_R2_TIER1_GATE.md` §T2.3 and its Appendix A apply unchanged. Read sealed packs only through `scripts/tier1_eval.py`; log each read in `results/round2/20261123_mask/SEALED_ACCESS_LOG.md`.
- **LUCiD does the physics.** No analytic-generator physics anywhere.
- **Do not disturb `r2-tier1`.**
  - Never touch its worktree, branch, remote clone, returns or queues. Read its results only from `origin/tier1`.
  - deepthought2 allows two GPU jobs in total. Use a lane there only when `r2-tier1` has none running; otherwise send WCTE runs to HTCondor.
  - HTCondor: at most 4 concurrent jobs from this phase.
  - CPU: ≤ 6 cores on `dt1` at `nice -n 15`; portal1 up to one sixth of its cores, leaving ≥ 8 free.
- **Safety.** Main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Push only to `origin/mask-scale`, fast-forward only. Never modify existing data, checkpoints or returns. JAX on shared GPUs: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`.
- **Idle polling** at most every 25 minutes.

### 0.3 Paths and budget

| What | Path |
|---|---|
| Remote clone | `/ceph/dwong/jepa/r2_mask_repo` |
| Returns | `/ceph/dwong/jepa/returns/r2_mask_2026-11-23/` (`R2_REPAIR_OUTPUT_ROOT=<this>/train`) |
| Scratch | `/work/dwong/r2_mask/` |

GPU ledger line `R2_mask_gpu`, cap **40 GPU-h**: WCTE trainings ≈ 5, SK trainings ≈ 20, scoring ≈ 6, M3 ≤ 3. Drop from the bottom of the M2 list if the projection exceeds the cap.

Tooling, hosts and batch layout are those of `r2-tier1`: read `$CC_STATE/notes/r2-tier1.md` (read-only) first.

---

## 1. Pre-registration (copy verbatim to `PREREG_2026-11-23_R2_MASK.md`)

### 1.1 Mask configurations

All hide about half of the PMTs.

| Name | Rule |
|---|---|
| `parity` | `within`: the same PMTs in every event |
| `random` | a fresh uniform half of the PMTs per event |
| `patch(s)` | PMTs ordered along the Hilbert curve and cut into contiguous patches of s PMTs; a fresh half of the patches per event. With an odd number of patches, hide ⌈count/2⌉. |

Sizes: WCTE s ∈ {8, 32, 128, 256}; SK s ∈ {43, 173, 692}. The defaults (32 and 173) are unchanged and their existing runs are reused. For every configuration record the realised hidden fraction and the fraction of PMTs that is never hidden.

### 1.2 Diagnostic quantities (M1; no neural network)

Targets and scalings are exactly those of the default training loss: hit flag; for hit PMTs, log-charge / 3 and time in units of 50 ns. All quantities are computed on `dev_probe_train`, with fitting and scoring on disjoint events.

- **Static predictor S:** each PMT's mean target over the fitting events.
- **Local predictor L:** one Ridge regression per detector and channel, shared across PMTs, from the same event's k = 8 nearest *visible* PMTs (their hit flags, log-charges, times and distances) plus the static mean.
- **Explained fractions** on scoring events: E_S and E_L (1 − MSE / variance), per channel and combined with the default loss weights.
- **Shortcut budget:** static = E_S; local = E_L − E_S; remainder **U = 1 − E_L**. U contains irreducible noise, which is the same for every mask on a detector, so only differences in U between masks are interpreted.
- **AvgDist:** mean distance from a hidden PMT to its nearest visible PMT, in metres and in units of the median nearest-neighbour spacing.
- **Correlation length ℓ_c:** the distance at which a single visible neighbour's explained fraction of log-charge falls to half of its nearest-neighbour value.

Report photon bombs (primary), muons and electrons separately.

### 1.3 Outcomes

Mean 3-D vertex error on sealed photon bombs with the frozen spatial-moment Ridge readout, as gain over the run's matched init (step-0 snapshot). Secondary: μ and e direction, energy resolution, and the MLP and attentive probes. Statistics (SE_boot, SE_seed, N_σ, "clear") as in `PREREG_2026-11-22_R2_TIER1.md` §2.2.

### 1.4 Predictions

- **Q1 (ordering of U).** U(parity) ≤ U(random) < U(patch), and U(patch(s)) increases with s.
- **Q2 (U predicts the gain).** Over the configurations {random, patch(s)} of a detector, the Spearman correlation between U and the vertex gain is ≥ 0.8. The direction gain has the same sign of correlation.
- **Q3 (scale).** On each detector, the configuration at which half of the gain from `random` to the best configuration is reached has AvgDist within a factor 3 of ℓ_c.
- **Q4 (destination, not rate).** The contrast patch(default) vs `random` on SK is at least as large at 10,000 steps as at 5,000 steps (unlike the time-unit contrast, which shrank).

### 1.5 Kill criterion

**KM.** On SK the best and worst configurations differ by less than 5 % or the difference is not clear, **and** the Spearman correlation of Q2 is below 0.5 on both detectors. Then the shortcut-budget claim is not the lead.

Everything else is exploratory and is labelled so.

---

## 2. Stages

Order: T0 → M1 (CPU) and code for M2 in parallel → push PREREG with the M1 numbers → evaluate M2 → M3 any time. Submit M2 trainings as soon as the code is tested; they may train before the PREREG push, but none may be evaluated before it.

### T0 — Setup (≤ 20 min)

Status file, notes, commit this runbook, remote clone at the branch head. Confirm that `--mask`, `scripts/tier1_eval.py`, the sealed packs and `src/jepa_noise/tier1_yardstick.py` exist on this branch. Inventory `dt2`, the batch queue and whether `r2-tier1` lanes are running.

### M1 — Shortcut-budget diagnostic (CPU only)

1. Implement `scripts/mask_shortcut_budget.py` per §1.2, using the real masker (`ViewMasker`) so that the hidden sets are those of training. 2,000 fitting and 2,000 scoring photon bombs per detector; 1,000 + 1,000 each for muons and electrons.
2. Run WCTE and SK for every configuration of §1.1; then JUNO and HK for `parity`, `random` and the default patch (exploratory).
3. Unit tests: with a synthetic field that is exactly linear in position, L explains ≈ 100 % under `random`; with a per-PMT constant field, S explains 100 %.
4. Output `SHORTCUT_BUDGET.md` and `.json`: the table of E_S, E_L − E_S, U, AvgDist per configuration, and ℓ_c per detector. Append the WCTE and SK numbers to the PREREG file and push.

### M2 — Patch-size sweep

1. Code: add `--patch-size` for `--mask spatial_patch` (default unchanged, so the default recipe keeps its hash; test it), the ⌈count/2⌉ rule for non-default sizes, and the run-name suffix `__ps<size>` (after `__m…`, before `__ts…`). Golden test: default-size masks identical to the base commit for 200 event ids. One fresh-context review of the diff.
2. Runs (seeds A = 9000, B = 9001, C = 9002):

| Priority | Detector | Cells | New runs | ≈ GPU-h |
|---|---|---|---|---|
| 1 | WCTE | patch 8, 128, 256 × {A, B, C} | 9 | 5 |
| 1 | SK | patch 43, 692 × A | 2 | 10 |
| 2 | SK | patch 43, 692 × B | 2 | 10 |

   Reused from `r2-tier1` (read-only): `random` and default-patch cells at 50 ns on both detectors, with their seeds.
3. Scoring: `scripts/tier1_eval.py` on validation and sealed, final step and step 5,000 (for Q4), with the Ridge, MLP (3 seeds) and attentive (3 seeds) probes; matched init from each run's step-0 snapshot.
4. Output `MASK_SCALE.md` and `.json`: error and gain per configuration and seed; Q1–Q4 and KM verdicts; one figure of gain against U and one of error against patch size, per detector.

### M3 — Readout-seeded LUCiD fit (time box 4 agent-hours, ≤ 3 GPU-h)

Question: can a frozen readout replace an expensive seed search for LUCiD's likelihood fit?

1. Events: the 200 `dev_probe_test` photon bombs per detector (SK, WCTE) already used by the Tier-1 yardstick Y2.
2. Starting points for (x, y, z): (a) truth (exists); (b) charge-weighted centroid (extend to all 200); (c) the frozen Ridge readout of `<det>__mae_repaired__s9000`, fitted on `dev_probe_train`; (d) the same readout on that run's matched init; (e) the raw-feature baseline. Start ln N from a linear fit to total charge on `dev_probe_train`; start t₀ from the median time residual at the starting vertex.
3. Fits: LUCiD's loss as in the yardstick, counts-only (primary, because the time term has an unresolved t₀ offset) and joint, same iteration budget for every start.
4. Report per start: mean and median final 3-D error, the fraction of events ending within twice the truth-start median error ("good basin"), iterations and wall time.
5. Registered expectation, appended to the PREREG file before running: on SK the good-basin fraction is ≥ 80 % for start (c) and ≤ 30 % for start (b). **Drop this piece** if (c) is below 50 %.
6. Output `SEED_TEST.md` and `.json`.

### T9 — Report

`results/round2/20261123_mask/MASK_REPORT.md` (≤ 2 pages): the shortcut-budget table, the sweep with Q1–Q4 and KM, the seed test, deviations, and a one-paragraph statement of what the lead claim may and may not say. `docs/round2/ACCEPTANCE_2026-11-23_R2_MASK.json`. Status `done`; print the report in full.

**Target: Thu 26 Nov, 20:00 CET.** Commit what exists then, and finish the rest afterwards.

---

## Launch (Mac Terminal, its own window, in the foreground)

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_mask.sh
```
