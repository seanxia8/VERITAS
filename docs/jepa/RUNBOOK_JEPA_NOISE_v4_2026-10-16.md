# RUNBOOK v4 — Predictable Is Not Physical: JEPA on detector signals

_16 October 2026, evening. Written by Claude at the author's request, after an audit of the v3
execution on branch `JEPA` (HEAD `f021d3c`). Owner: Dowling Wong; collaborator: Junjie Xia.
Fixed base `0cae6b0`. Amended 20 October 2026 by A9 (§3.7). **A10 (§3.8, 22 Oct 2026) is a draft and is not in force until registered.**_

**Authority.** v4 becomes binding when the orchestrator commits it, unchanged, as
`jepa: register runbook v4 (author revision)`. Its amendments A3 (with modifications), A4, A5, A6, A7
and A8 are author decisions made through this revision; no separate gate is needed for them.
**A9 (§3.7, 20 Oct 2026)** registers the author decisions made after registration (already quoted
verbatim in `DECISIONS.md`). The orchestrator commits this amended file unchanged, as
`jepa: register runbook amendment A9 (author revision)`.

**A7 (17 Oct 2026) changes the main deep experiment.** E2 now runs on LUCiD photon bombs, and the
Tier-1 E2 shrinks to a bridge. **A8 (17 Oct 2026)** revises A7 after a second review of Junjie's
comments; it is folded into §3.5 and listed in §3.6. Where §3.5 conflicts with an earlier section,
§3.5 wins; where §3.7 (A9) conflicts with §3.5, §6.1 or §8, §3.7 wins. The companion design doc is
"JEPA Project Design: Predictable Is Not Physical (v0.3)".

**v3 is kept as history.** `docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md` stays byte-identical, and
every v3 section that v4 does not override stays in force (list in §2). Amendments A1 and A2 stay
in force; A4 changes some of the values A1 fixed.

---

## 0. Audit of the v3 execution: what happened and what v4 does about it

| # | Finding | Evidence | v4 response |
|---|---|---|---|
| F1 | The agent rewrote v2 into v3 before registering it, with 14 repairs plus new design choices. Most repairs are correct; some new choices were the agent's own (64-sample gap, 1% white noise, linearised E1b, frequency endpoint in R9, CPAL dates demoted to "planning notes"). | `reviews/REVISION_REPORT_JEPA_V3_2026-10-16.md` | The correct repairs are kept (§2). The CPAL timeline is restored (§9). No further runbook rewrites by agents (§5). |
| F2 | **E2 design dead end.** All three γ_n targets selected line level 0, so R6–R8 became "not evaluable". | `results/jepa/20261016_wp2_E2_LEVEL_SELECTION.json`, γ_n 0.966…0.995 | Root cause below. Fixed by A4 (§3.2), and the fix is checked in §A. |
| F3 | E1, which is numpy-only and cheap, never ran. The agent blocked all remote work on a torch environment install that hit the file-count quota and then the 600 s timeout. | `GATE_G6.md`, `STORAGE_RECOVERY.md` | E1 runs locally now. A torch environment is not on E1's critical path (§6). |
| F4 | Files were deleted outside the JEPA area on Deepthought: about 22k `.pyc` files in other campaigns' environments (`paper1_env`, `meg_env`, `x3_env`, `dpo_batch3`). It is functionally harmless, since the files regenerate, but it was out of scope. The record says the author authorised it. | `results/jepa/20261016_storage_recovery/` | New hard rule in §6. |
| F5 | Repo bloat: about 94k inserted lines, about 25k of them infrastructure audit logs (quota inventories, cleanup JSONL). | `git diff --stat 0cae6b0...JEPA` | Hygiene rules and a one-time cleanup (§7). |
| F6 | Gate inflation. G6 was used for unspecified engineering details. Amendment A3 sat unapproved and blocked the whole of E1. | `GATE_G6.md`, `E1_PROBE_CONFIG_PROPOSAL.md` | A3 is decided here (§3.1). A decision-default rule and a narrower G6 (§5). |
| F7 | Correct catches by the agent, all adopted: TIDMAD channel roles (CH1 = SQUID, CH2 = injection reference); v2's "ch0 reference" was wrong. Conditional P5 eligibility. The finite-β ≠ GEVP distinction. Context-first oracle with gaps discarded. Independent development cohort. Constant-target handling in R². | v3 §2.2, §7.1 | Kept unchanged. |

**Root cause of F2** (population numbers from the v3 artifacts, re-derived in §A):
- The Tier-1 reference cell's base noise is **shared across channels** (`corr_strength = 0.4`)
  and strongly coloured in time. Its noise canonical correlation is already 0.74 (channel split)
  and 0.61 (time split) with **no** added nuisance. Any added common mode pushes γ_n above 0.95,
  so "top noise canonical correlation" saturates and cannot define levels.
- The time and block splits put the pulse onset in the context and only the tail in the target.
  The target signal therefore has rank 2, and the third physics canonical correlation is about
  2e-4: there is nothing for JEPA to predict.
- E1b's θ perturbation SDs (0.15, 0.005, 0.002) are about 6× smaller than the parameter spread
  E2 actually trains on, so E1b physics was much weaker than E2 physics.

The γ_n level rule was my v2 design. The agent was right to stop rather than improvise.

---

## 1. Status at v4 registration

- Done:
  - v3 registered; local environment accepted; Appendix A toy and P5 counterexample reproduced.
  - `THEORY.md` (P1–P5).
  - WP2 linear package accepted by an independent review.
  - A1 waveform law implemented and population-checked (36 cell/split rows).
  - E1c engine and grouped probes engineered.
- Not run: any E1–E4 outcome grid. GPU use is zero. Remote torch environment incomplete.
- Flags for the v4 launch: `PUSH_JEPA=yes`, `GPU_ALLOWED=yes` (on the hosts and within the
  limits of §6.1–§6.2), `DEEPTHOUGHT_GPUS=4`, `DEEPTHOUGHT2_GPUS=2`, `CONDOR_NEMO2=no`,
  `OPTIONAL_WPS=WP7`.

---

## 2. What stays in force from v3

v3 sections that stay in force (cited as "v3 §x"):
- §1.1 isolation;
- §1.2 provenance, as simplified by §7;
- §1.3 statistical and tuning contract;
- §2.1–2.3 mathematical contract (P1–P5, criterion M);
- §4 package and numerics, including §4.1 (1% independent white addition, with ε fixed from the
  **A4** base cell) and §4.2 (CRLB, probes, E1c protocol);
- WP4's arm definitions and tests, except the default split (§3.2) and the arm set (A6, §3.4);
- §7 E3 in full;
- §8 WP7;
- Appendix A.

v3 sections overridden by v4:
- §2.4 R1, R2, R6, R7, R8 → §4;
- §3.1 hosts and environments → §6;
- §3.2 accounting granularity → §6;
- §1.4 gate usage → §5;
- §5 E1b design → §3.2;
- §6 WP5 split and level rule → §3.2;
- completion dates → §9.

---

## 3. Amendments registered by v4

### 3.1 A3 (E1 cohorts, calibration and estimator information): approved with modifications
`docs/jepa/E1_PROBE_CONFIG_PROPOSAL.md` is approved as written, except:
1. **E1b cell list** follows A4: 13 unique cells, not 9.
2. **Streaming, not storing.** E1b subject and calibration data are generated in chunks of at
   most 10 000 events. Sufficient statistics (sums and cross-products) accumulate in float64.
   Raw n = 1e5 arrays are never held in memory or written. Probe cohorts of 10 000 + 10 000 may
   be materialised.
3. **M diagnostic development setting:** g = 0.02, line = 0, development seed 100, A4 base. Use
   the A4 candidate order (channel, interleaved, time, block).
4. **Ridge-fit counts** in A3 are planning numbers, not a gate. Probes act on k ≤ 3-dimensional
   embeddings and cost seconds.

### 3.2 A4 (E1b/E2 design repair)
1. **Base noise for E1b and E2:** the Tier-1 reference cell with base `corr_strength = 0`
   (channel-private), keeping its temporal PSD unchanged. The v3 §4.1 white addition stays; ε is
   recomputed from this base. A1's added line and common-mode laws, bases and normalisations are
   unchanged.
2. **E1b θ perturbation SDs** = SDs of E2's uniform parameter ranges:
   (3, 0.1, 0.04)/√12 = **(0.866, 0.0289, 0.0115)**, at the same θ0 = (4.5, 0.35, 0.08) and the
   same fixed J0. This is a linear fixture by construction (s = J0 δθ), so larger SDs do not break
   linearity.
3. **Splits:**
   - **channel** (channels 0–3 → 4–7, all samples) is the **primary** split for E1b and E2;
   - **interleaved** ({0,2,4,6} → {1,3,5,7}) is M's alternative;
   - **time** and **block** (v3 sample definitions) are **diagnostic only**: rank-deficient
     target, reported, never used for a registered prediction.
4. **E1b grids** (13 unique cells, 5 seeds, all four splits):
   - common-mode strength g ∈ G = {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9} at line 0;
   - line multiplier ∈ {0.75, 1.5, 2.25, 3} at g = 0;
   - the cell (g = 0, line = 0) is the clean reference.
5. **Displacement D** (the design quantity replacing γ_n for levels): the population maximum
   principal angle between the top-3 context canonical directions (linearised fixture, A4
   covariances, given split) and the Paper 1 reference R_c (top-3 GEVP of
   (A_c Σ_θ A_cᵀ, N_cc)). γ_n stays a reported quantity.
6. **E2 levels:** on the channel split, choose the three g ∈ G \ {0} whose D is nearest to
   **10°, 45° and 80°** (ties → smaller g), line = 0. Compute and freeze this before any E2
   training. The expected selection from §A is g = 0.005 (11.3°), 0.02 (48.0°) and 0.05 (77.0°).
   - If the recomputation differs from §A by more than 1°, stop and report it as a bug check.
     That is not a design gate.
   - If the selected g values are not distinct, take the next-nearest distinct value; record
     this, no gate.
7. **E2 masks:**
   - Every E2 arm uses channel tokens 0–3 (all 8 time patches) → channel tokens 4–7. No gap is
     needed, because the base noise is channel-private.
   - There is no M-selected arm in E2 (A6). Under A4, the channel and interleaved candidates
     agree to within 0.5°.
   - The joint-whitening arm's context-first Cholesky orders the context channels first.
   - Leakage tests cover channel visibility.
8. **E2 nuisance targets:** `common_0` and `common_1`. Line targets are undefined at line = 0.
9. **R8 shift:** multiply the common-mode component's amplitude by 2 (variance ×4) at test time.
   Normalisers, encoders and probes stay frozen.
10. E2 SNR cells are unchanged: nominal, and sub-noise (amplitude × 0.25). The development cohort
    is (mid level, nominal), independent IDs as in v3.

### 3.3 A5 (process): §5–§7 below.

### 3.4 A6 (E2 arm set: seven arms; author decision, 16 Oct 2026)
E2 trains exactly these seven arms, using v3's names and definitions:

| v3 # | Arm | Role |
|---|---|---|
| 1 | `mae_mse` | Baseline: raw-sample reconstruction |
| 2 | `mae_chi2` | Strong baseline: noise-model (full Σ) weighted reconstruction |
| 3 | `ijepa_ema` | JEPA with EMA target (DINO/I-JEPA-like); depth sweep |
| 4 | `sigreg_jepa` | Main object (LeJEPA-style); run first; depth sweep |
| 6 | `sigreg_jepa_anchored` | Main method (noise-record anchor) |
| 8 | `anchor_only` | Control: is the prediction loss needed? |
| 9 | `sigreg_jepa_jointwhite` | Oracle upper bound (needs full Σ_n) |

What changes as a result:
- **Removed from E2:**
  - arm 5 `sigreg_jepa_masked`: its two candidate splits are statistically identical under A4. It
    stays in E1 (masks/criterion M) and in E3 (v3 §7, where real lines make the choice matter).
  - arm 7 `hybrid_chi2_jepa`.
- **SimCLR** is not added; it is future work.
- **Development cohort** tunes only β (arms 6 and 8). The λ_J search is dropped.
- **Grid:** 7 arms × 6 cells × 3 seeds = 126 primary runs. The depth sweep covers arms 3, 4 and 6
  on the (high, nominal) cell. The smoke run covers all 7 arms.
- **Predictions:**
  - R7 compares `sigreg_jepa_anchored` (and reports `mae_chi2` beside it) against `sigreg_jepa`.
    The masked-arm clause no longer applies in E2.
  - Also report, without a pass criterion, the gap between anchored JEPA and `mae_chi2`: the
    noise-record method versus the noise-model method.
- **E3 arm list:** unchanged from v3 §7.3.
- **WP7 (E4):** unchanged; it uses arms 3, 4 and 6.

A7 (§3.5) moves this seven-arm set to the LUCiD experiment, and reduces the Tier-1 E2 to a
bridge.

### 3.5 A7 (LUCiD as the E2 testbed; Tier-1 bridge; LUCiD encoder; seed floor; R11; author decision, 17 Oct 2026), as revised by A8

This section is the A7 text with the A8 revisions (§3.6) folded in. Text marked **[A8]** is new or
changed in A8.

**3.5.1 Scope.**

| Experiment | What it is now |
|---|---|
| E1 (E1a, E1b, E1c) | Unchanged (v4 §3.1–3.2). |
| E2 bridge | The Tier-1 E2 of §3.2/§3.4, replaced by §3.5.7. |
| E2 main | LUCiD photon bombs (§3.5.2–3.5.6). |
| E2 secondary **[A8]** | R11, R11b, dark-window-only anchor, aging trend, occupancy ablation (§3.5.6), run in that order within the cap. |
| Scaling benchmark **[A8]** | No training; encoder memory and throughput at 2k, 10k and 40k tokens (§3.5.5). |
| E3 | Optional. Run only the v3 §7.2 coherence step (no training) unless a band reaches MSC ≥ 0.5; training then needs an author go. |
| E4 | Optional, on the frozen LUCiD encoders. Same fault logic, with the faults mapped to LUCiD: dead mPMT module; timing jitter on one module; global gain ×1.2. |

There is no depth sweep on LUCiD.

**3.5.2 Dependencies (these override v3 §1.1's "do not incorporate PMT work").**
- **PMT readout.** The JEPA branch may merge `codex/junjie-pmt-response` at one reviewed SHA, named
  by the author in a decision-log line. Do this only after the PMT runbook v2 repairs R-1 to R-3
  are accepted.
- **Before the merge [A8]:** WP4b development (generator, encoder, tests, smoke, detectability
  check) may use the current readout, labelled `readout_synthetic_unrepaired`. Nothing produced
  with it is a reported result; the detectability check is re-run on the merged readout.
- **Read-only.** JEPA code only **imports** `src/noise_module_lucid/` and `configs/lucid_pmt/`.
  It never edits them; fixes go back through the PMT branch.
- **LUCiD.** Pinned revision `eaf2e89f579d6679e90020c732ac2c846895d691`. Research use is confirmed
  by Junjie. The missing upstream LICENSE file is recorded; nothing is redistributed.
- **If the PMT repairs are not accepted by 31 Oct:** write gate G-PMT with these options.
  - (A) Wait.
  - (B) Run E2 main on the current synthetic readout, labelled `readout_synthetic_unrepaired`, and
    re-run once the repairs land.

  Recommended: (B) for development, (A) for reported results.

**3.5.3 Data generator** (`src/jepa_noise/lucid_bombs.py`).
- **Geometry.** WCTE in LUCiD: 105 mPMT modules × 19 PMTs. Use the physical `mpmt_id` map from
  the PMT branch.
- **Events.** Isotropic point sources ("photon bombs"):
  - vertex uniform in a fiducial cylinder (radius and half-height = 80% of the inner detector);
  - emission at the trigger time (t = 0);
  - **photon yield [A8]** log-uniform in [Y_lo, Y_hi], spanning low to high occupancy. Y_lo: events
    at Y_lo have a median of about 200 hit PMTs. Y_hi: the lower of (a) the yield at which the
    median module charge reaches 300 PE and (b) the yield at which the median hit fraction reaches
    0.95. Set both on development data before training, adjust only through §3.5.3a, and record
    them. Report physics and detector-state metrics also by yield tercile;
  - spectrum and transport (prompt, no scattering, reflection or absorption beyond the attenuation
    factor below) follow the PMT `reference.json`.
- **Readout.** The PMT chain from the merged SHA: QE·CE detection, single-photon charge, transit-time
  spread, dark counts, fixed threshold, physical walk curve (PMT v2 D-TQ). Gains act on the charge
  before the threshold, so a gain change can switch a PMT between hit and unhit.
- **Nominal twin [A8].** For every event the generator also writes a twin readout with all gains
  nominal (1) and λ = λ0, using the same photon, single-photon-charge, transit-time and dark-count
  draws (keyed RNG). Only the oracle arm reads it.
- **Runs [A8].** All events in a run share one detector state.
  - Training and development-training sets: runs of 1000 events (100 runs per 100k).
  - Probe-train, probe-test and robustness sets: runs of **50** events, so each 10k set holds 200
    distinct states. Every set has its own runs.
  - For each drift setting, one probe-train and one probe-test set serve both splits; the N_min
    exclusion (§3.5.4) applies to training events only.
  - The training loader samples events uniformly across all runs; no batch is drawn from a single
    run.
  - Event IDs and namespaces are as in v3 §6 WP5, with the prefix `lucid/`.
- **Detector-state laws** (truth targets per event = its run's values). The gain of PMT p in module
  m is gain_m · gain_p. **[A8]** Each factor is clipped to [0.5, 1.5] separately; count and report
  clipped values.
  - **F1, module gain drift** (main cells): gain_m = 1 + s·u_m, u_m ~ N(0, 1) independent per module
    and run. Main cells: s ∈ {0.02, 0.05, 0.10}.
  - **F3, independent per-PMT gain [A8]** (every LUCiD cell, including R11, R11b and T): gain_p =
    1 + s_p·w_p, w_p ~ N(0, 1) independent per PMT and run, s_p = 0.05 (adjusted only through
    §3.5.3a). It is private in both splits.
  - **F2, attenuation change** (water-quality-like; R11 and R11b only): attenuation length
    λ = λ0·exp(a·v) (always positive), v ~ N(0, 1) per run, a set by §3.5.3a from {0.10, 0.20, 0.50}. Its uniform part
    is degenerate with the yield; only its distance-dependent part is informative.
  - **F2′, fallback [A8]** (used instead of F2 only if §3.5.3a finds F2 undetectable at a = 0.50):
    a smooth vertical gain gradient, gain_p ← gain_p · (1 + b·v·z_p/H), with z_p the PMT height from
    the detector centre, H the inner half-height, v ~ N(0, 1) per run, b ∈ {0.05, 0.10} by
    §3.5.3a. It models temperature stratification and keeps the "smooth vs module-wise" contrast.
    The F1 truth stays ln(1 + s·u_m) only; F2′'s projection onto modules is not part of it, and the
    §3.5.3a ceilings for F1 and F2′ are computed with both profiled jointly.
  - **Law R11b, co-varying [A8]:** one scalar c ~ N(0, 1) per run drives both: gain_m = 1 + s·c·k_m
    and the F2 (or F2′) variable v = c (same exponential form for λ), with k_m ~ N(0, 1) drawn once (seed namespace
    `lucid/k_R11b`) and fixed, s = 0.05. Used for **training** the R11b encoders only. R11b's probe
    sets use the independent R11 law.
  - **Law T, aging trend [A8]:** gain_m(r) = 1 + s_T·k′_m·(r/R), with k′_m ~ N(0, 1) drawn once
    (seed namespace `lucid/k_T`, independent of `lucid/k_R11b`), R = 100 training runs,
    r = 0…R−1, s_T = 0.10. In-range probe sets draw r/R uniformly in [0, 1); the extrapolation
    set uses r/R ∈ [1.9, 2.1].
  - Ask Junjie to confirm s, s_p and a (or b) in `DECISIONS.md`; any change comes before training.
- **Noise-only records:** dark-noise-only readout windows (no source), generated per run.
- **Storage:** hits as sparse arrays. Generation runs on the CPU host. Benchmark 1000 events first
  and project the total.

**3.5.3a Detectability check [A8]** (before any training, on development data; re-run on the
merged readout).
- For each factor and setting (F1 at each s, F3, F2 or F2′, law T), compute the Fisher information
  for the factor from the generator's true expected charges: Poisson statistics inflated by
  (1 + σ_spe²), with the vertex, the yield, all other module gains and the other factors treated
  as unknown (profiled).
- Report two ceilings, with σ_f the factor's spread: per event, R²_max = σ_f² / (σ_f² + 1/I_event);
  per run, the same with I_run = the sum of I_event over a run's 50 events. Both are lower bounds:
  profiling ignores the known yield prior, which run-mean probes can exploit.
- Every factor used by a registered criterion needs a **run-level ceiling ≥ 0.6** at the median
  yield. If it falls short, adjust in this order, recording each step in `DECISIONS.md`:
  1. raise Y_hi to its bound (b) above;
  2. F2: the next a in {0.10, 0.20, 0.50}, then switch to F2′ (b = 0.05, then 0.10);
  3. F3: s_p up to 0.10.
- F1 cells whose ceiling is still below 0.6 stay in the grid, because they show the trap's lower
  limit, but they are excluded from the R6 and R7 criteria. If no F1 cell reaches 0.6, or F2/F2′
  cannot, write G6 for the dependent predictions. If F3 cannot, R12 is `not_evaluable`; no gate.
- Freeze the ceilings in `DECISIONS.md` and report them beside every detector-state R².

**3.5.4 Views and masks.**
- **Across modules** (`across`): order modules along a 3D Hilbert curve of their centres. Per event,
  a random contiguous half of that order is the target; the rest is the context.
- **Within modules** (`within`): context = PMTs with even `pmt_id_in_mpmt` in every module;
  target = odd. The partition is fixed, the same for every event.
- **Minimum hits:** every target view must contain ≥ N_min hit PMTs (N_min = 20, checked on
  development data). For `across`, redraw the random half up to 3 times; an event that still fails
  is excluded and counted. For `within`, a failing event is excluded and counted **[A8]**. Report
  the exclusion and redraw rates per cell.
- **Hit fractions:** report the hit fraction per view.
- **Occupancy ablation** (secondary; LeJEPA, `within`, s = 0.05, 3 seeds): natural occupancy vs
  balanced views. **[A8]** Balancing subsamples unhit PMTs in the **context** view only, to match its
  hit count; every target PMT is still queried, so the query set reveals nothing about target hits.
  The leakage test also runs in this mode.

**3.5.5 LUCiD encoder WP** (`src/jepa_noise/point_encoder.py`; a new WP4b before WP5).
- **Tokens [A8]:** one per PMT, all PMTs (hit and unhit). Features, all with fixed, data-independent
  scaling (no per-event normalisation):
  - position and facing direction (divided by fixed detector dimensions);
  - **log charge**, ln(q / 1 PE), for hit PMTs, and 0 for unhit PMTs. A module gain is then an
    additive offset on that module's hit tokens, matching the linear theory;
  - first-hit time relative to the trigger, divided by a fixed 100 ns, 0 for unhit PMTs. **[A9]**
    The trigger is the readout's NHits rule, after the prompt-window cut (§3.7.3);
  - hit flag.
- **Architecture:** serialised point transformer.
  - Sort tokens along a 3D Hilbert curve; attention within windows of 64 tokens, with the window
    offset shifted by 32 on alternate layers.
  - 4 blocks, d = 128, 4 heads, pre-norm.
  - **Read-out point [A8]:** the representation used by every loss and probe is the output of the
    last block **before** any final normalisation layer. The `ijepa_ema` target adds its own token
    normalisation, as v3 arm 3 defines.
  - The serialisation code may be adapted from pimm (MIT), with attribution. No pretrained Panda
    weights.
- **Training [A8]:** 10 000 steps, batch 32, the repository's AdamW/Muon conventions as in v3 WP4,
  numeric settings frozen in the reviewed config before outcomes.
- **Masking:** the context encoder replaces target-view tokens with a learned mask embedding before
  the first block. The predictor (2 blocks, d = 64) reads the context outputs plus position queries
  at target PMTs.
- **Tests:**
  - shapes;
  - **leakage:** zero gradient of context outputs with respect to every target-PMT input;
  - **amplitude preservation [A8]:** a linear probe recovers per-PMT log charge (hit PMTs) from the
    read-out tokens. Criterion R² ≥ 0.9 for `sigreg_jepa` and `mae_mse`, after the smoke and again
    on the final encoders. A failure on the final encoders is reported as a finding, with that
    arm's detector-state R² flagged; no gate. Other arms are reported without a criterion: the
    anchor is meant to remove module-wide charge scale, so a lower value there is a finding, not a
    failure;
  - 10-step determinism;
  - **memory:** batch 32 within the §6.2 per-process cap (about 21 GB on L40S, 9.6 GB on a Titan,
    with gradient accumulation allowed there); record throughput.
- **Scaling benchmark [A8]** (no training; 2k and 10k on a Titan, 40k on L40S, ≤ 1 GPU-hour each): synthetic layouts of N ∈ {2 000, 10 000, 40 000} PMTs on
  a cylinder surface, hit fraction 0.2, batch 8. Time the forward and backward pass of context
  encoder + predictor, and record peak memory and events per second. If a size does not fit in
  memory, record that as the result.

**3.5.6 E2 main and secondary cells.**
- **Arms:** the seven A6 arms, mapped to LUCiD as follows.

  | Arm | LUCiD version |
  |---|---|
  | `mae_mse` | squared error on hidden PMTs' token features: log charge, hit flag, and time where hit |
  | `mae_chi2` **[A8]** (LUCiD version; replaces A7's `mae_poisson`) | second-moment quasi-likelihood of hidden PMTs' charges with the covariance below, plus a Gaussian time NLL (reference transit-time σ) on hidden hit PMTs |
  | `ijepa_ema` | as defined in v3 |
  | `sigreg_jepa` | as defined in v3 |
  | `sigreg_jepa_anchored` | `sigreg_jepa` plus the LUCiD anchor penalty below |
  | `anchor_only` | SIGReg plus the same penalty, no prediction loss |
  | `oracle_nominal_twin` **[A8]** | `sigreg_jepa` objective on the nominal twin readout (§3.5.3): same events, all gains nominal. Probes and floors use the arm's own input; truth targets are the same. Run in the six main cells only |

  **`mae_chi2` covariance [A8].** The head predicts μ_p ≥ 0 (softplus), the expected *recorded*
  charge of each hidden PMT (zero below threshold), and unhit PMTs have recorded charge 0. For the
  hidden PMTs of module m, with residual r = q − μ:
  C_m = diag(μ_p(1 + σ_spe²) + v_dark + s_p²μ_p²) + s²·μ_m μ_mᵀ,
  where σ_spe is the reference single-photon charge spread, v_dark the reference dark-charge
  variance per window, and s, s_p the cell's true spreads. Loss = ½ Σ_m (rᵀ C_m⁻¹ r + ln det C_m),
  computed with the Woodbury identity and the determinant lemma. It is the target-marginal analogue
  of the toy's full-Σ_n χ², a second-moment approximation that ignores threshold censoring, and is
  labelled as using noise-model information. Unit tests: agreement with a dense computation on one
  module; and the Monte-Carlo covariance of the readout at fixed μ, s and s_p within 10% of C_m.

  **Anchor penalty [A8].** Per training event, one perturbed copy x̃ with both draws:
  1. overlay an independent dark-noise-only window: union of hits; where a PMT is hit in both,
     charges add and the earlier time is kept;
  2. multiply **every** module's charges by an independent factor ~ N(1, 0.01²), the assumed
     per-module calibration uncertainty.

  Penalty = mean over modules of ‖z_m(x̃) − z_m(x)‖², divided by the trace of the module-token
  covariance, where z_m is the module-pooled read-out token. β grid {0.1, 1, 10, 100, 1000}. On
  the development cohort, choose the **largest** β whose vertex resolution is ≤ 1.1 × that of the
  development `sigreg_jepa` (the same physics tolerance as R7); this replaces the v3 §1.3 β rule for
  LUCiD, whose ties-to-smaller rule would favour an inactive anchor. The **dark-window-only variant**
  `sigreg_jepa_anchored_dark` (draw 1 only, its own β) is a secondary run that tests specificity
  (R7b).
- **Main cells:** s ∈ {0.02, 0.05, 0.10} × split ∈ {across, within} = 6 cells, all with F3.
- **Development cohort:** s = 0.05, `within`, with independent IDs and seeds. In order:
  1. Y_lo, Y_hi, N_min and the detectability check (§3.5.3a);
  2. the physics-probe type decision (below), from one development seed each of `sigreg_jepa` and
     `mae_mse`;
  3. β for `sigreg_jepa_anchored`, `anchor_only` and `sigreg_jepa_anchored_dark` (one seed per β).
- **Seeds:** 5 for `sigreg_jepa`, `sigreg_jepa_anchored` and `mae_mse`; 3 for the others. That is
  3 × 6 × 5 + 4 × 6 × 3 = 162 trainings.
- **Secondary runs [A8]**, all `within`, run in this order within the cap:
  1. R11: F1 (s = 0.05), F2 (or F2′) and F3 varying independently; `sigreg_jepa`,
     `sigreg_jepa_anchored`, `mae_mse`; 5 seeds (15).
  2. R11b: trained on law R11b plus F3; same three arms; 3 seeds (9).
  3. Dark-window-only anchor: s = 0.10; 3 seeds (3).
  4. Aging trend: law T plus F3; same three arms; 3 seeds (9).
  5. Occupancy ablation (§3.5.4): 3 seeds (3).

  Total 39. With the main grid, 201 primary LUCiD trainings, plus 17 development trainings
  (2 for the probe decision, 15 for β).
- **Probes [A8]** (ridge as in v3 §1.3, cross-validation folds grouped by run, fitted on probe-train
  and scored on probe-test; targets ln gain_m and ln gain_p).

  | Target | Full-input probe (frozen encoder, unmasked input) |
  |---|---|
  | vertex (x, y, z), log yield | mean over all read-out tokens; ridge, or MLP after the decision below |
  | module gain ln gain_m (F1, law T) | module-pooled tokens (mean over its 19 PMTs); one ridge shared by all modules |
  | per-PMT gain ln gain_p (F3) | each hit PMT's token; one ridge shared by all PMTs |
  | F2 / F2′ variable v | mean over all read-out tokens; ridge |
  | age r/R (law T) | mean over all read-out tokens; ridge |

  Every detector-state probe is reported in four forms, each with its random-init floor (the same
  probe on an untrained encoder of the same architecture, init seed = the arm's first seed):
  1. **per event** on the full-input features above;
  2. **run-mean:** the features averaged over a run's 50 probe events, R² over (run, module), (run,
     PMT) or runs. **This is the primary form for every registered criterion**, because a drift
     state is a run property and a single event carries little information about it (§3.5.3a);
  3. **propagated:** the same probe on the context-encoder outputs at masked positions, with the
     standard training mask applied: in `within`, a module's masked (odd) PMTs; in `across`, the
     PMTs of modules in the target half. Run-mean over the events in which the position is masked.
     It measures what the context carries about the hidden part, without input passthrough;
  4. **raw-input ceiling:** the same probe on the raw token features (module-mean raw features for
     module targets).
  - **Physics-probe type** (decided once, step 2 of the development cohort): if the linear vertex
    resolution of both development encoders is worse than 3 × the reference fit, the primary
    physics probe becomes an MLP (2 hidden layers of 256, early stopping on 10% of probe-train,
    fixed seed) for every cell and arm. Record it in `DECISIONS.md`; no gate.
- **Physics reference [A8]:** the VERITAS time-of-flight vertex fit, `consequence.fit_vertex`
  (default `FitConfig`, v_group from `reference.json`), on the same probe-test events, using
  recorded charges and times without gain correction. Resolution = mean 3D vertex error over
  events whose fit is `ok`; report the failure rate beside it. Report the ratio model / fit.
- **Robustness:**
  - main cells: robustness sets at s = 0.20 (F3 unchanged), outside the training range. The
    robustness ratio is vertex resolution there divided by resolution at s = 0. R8 uses encoders
    trained at `within`, s = 0.10, so s = 0.20 is "drift ×2";
  - aging trend **[A8]:** vertex resolution on the extrapolation set (r/R ∈ [1.9, 2.1]) divided by
    that on the in-range probe-test set.
- **Compute cap [A8]:** ≤ 80 L40S GPU-hours (deepthought2; **[A9]: 170**, grid at ladder rung (1), §3.7.2) for the development cohort, main grid,
  secondary runs and their evaluation; ≤ 300 Titan GPU-hours (Deepthought) for smokes, the scaling
  benchmark, the bridge, generation and other evaluation (§6.1). Generation has a separate CPU cap
  of 48 wall-hours (over → G2). Benchmark first (v3 §3.2). For
  LUCiD this fallback replaces v3 §3.2's: if the projection exceeds the cap, (1) use 3 seeds for
  every main arm; (2) drop secondary runs from the end of the list; (3) reduce every LUCiD arm to
  5 000 steps; (4) G2. Record each step in `DECISIONS.md`.

**3.5.7 Tier-1 bridge** (replaces the Tier-1 E2 of §3.2.6–§3.4).
- **Data:** the A4 law at g = 0.02, nominal SNR, channel split.
- **Arms:** `sigreg_jepa`, `sigreg_jepa_anchored` and `mae_chi2`, 3 seeds each.
- **β [A8]:** the v3 §1.3 grid and rule on an independent Tier-1 development cohort (g = 0.02,
  nominal, development seed 100).
- **Ablation:** `sigreg_jepa` with *parallel* factorised attention (time branch ‖ channel branch,
  concatenated, then projected), 3 seeds.
- **Input and read-out [A8]:** `normalise_input_sequence` (the per-event z-score in
  `current_compact.forward`) is replaced by the fixed per-channel scaling of v3 WP4. The v3 WP4
  token grid `(B, N_patch, C, d)` is taken **before** `final_proj_norm` (the RMSNorm after the
  channel layers), for every loss and probe. The `ijepa_ema` target adds its own normalisation as
  in v3.
- **Amplitude-preservation probe:** recover per-channel pulse amplitude with R² ≥ 0.9, both after
  the time block and from the read-out tokens, for `sigreg_jepa` and `mae_chi2`.

**3.5.8 Seed floor (applies to every comparison in E2).**
- For each metric and cell, an arm's spread is the SD over its seeds.
- **[A8]** The seed floor for a comparison of arms a and b is the largest of SD_a, SD_b and the
  median SD over all arms in that cell for that metric (this stabilises 3-seed SDs).
- For a comparison between two cells of one arm (R6), the floor is the largest of that arm's SDs
  in the two cells and the larger of the two cells' median SDs.
- A difference is claimed only if |Δ mean| > 2 × the seed floor.
- Also report linear CKA between seeds of the same arm, on probe-test embeddings.

**3.5.9 Predictions** (replace v4 §4's R6–R8 and add B1, R7b, R11, R11b and R12; R1–R5 and R3b are
unchanged). All LUCiD detector-state R² values are run-mean (§3.5.6), and only F1 cells passing
§3.5.3a enter R6 and R7.

| ID | Prediction | Criterion |
|---|---|---|
| B1 | The deep transformer follows the theory (bridge) | `sigreg_jepa` nuisance R² > `sigreg_jepa_anchored` in every seed, and the difference exceeds the seed-floor rule |
| R6 | The split decides the trap (LUCiD) | `sigreg_jepa` **propagated** module-gain R² minus its random-init floor (same split and mask): `within` > `across` at every eligible s, beyond the seed-floor rule, **and** in `within` the excess over the floor > 2 × the seed floor, so the gain is learned rather than passed through. Full-input R² − floor reported beside it |
| R7 | The anchor removes the drift and keeps physics (LUCiD) | `within`, s = 0.10: anchored full-input module-gain R² < plain in every seed, beyond the seed-floor rule; anchored seed-mean vertex resolution ≤ 1.1 × plain |
| R7b **[A8]** | The anchor is specific to the nuisance it is shown (LUCiD) | `within`, s = 0.10: `sigreg_jepa_anchored_dark` full-input module-gain R² ≥ `sigreg_jepa` − 2 × the seed floor (seeds 0–2 of both). Attenuation (or F2′) R² of anchored vs plain in the R11 cell is reported without criterion |
| R8 | Absorbed drift costs robustness (LUCiD) | Encoders trained at `within`, s = 0.10: robustness ratio of plain > anchored in ≥ 4 of 5 paired seeds (3 of 3 if the seed fallback applies). **[A8]** Secondary, on the aging-trend cell: plain > anchored in 3 of 3 paired seeds |
| R11 **[A8 criterion]** | Effects with different geometric patterns, varying independently, are separable (LUCiD) | In `sigreg_jepa` on the R11 cell: F1 R² ≥ 0.5 and F2 (or F2′) R² ≥ 0.5, each above its random-init floor by more than 2 × the seed floor; and cross-leakage ≤ 0.1, where cross-leakage is (i) the squared correlation, over (run, module) pairs, between the F1 probe's output and the F2 variable, and (ii) the squared correlation, over runs, between the F2 probe's output and the run's module-mean ln gain_m |
| R11b **[A8]** | Effects that drift together in training stay entangled (LUCiD) | `sigreg_jepa` trained on law R11b, probes fitted and scored on R11-law probe sets: at least one R11 condition fails. Evaluable only if R11 passes; otherwise `not_evaluable` |
| R12 **[A8]** (control) | Only shared aging is carried across views (LUCiD) | `sigreg_jepa` **propagated** per-PMT gain R² ≤ its random-init floor + 0.05 in both splits, while R6 passes. Evaluable only if F3 passes §3.5.3a and R6 passes. A control expected by construction; the full-input per-PMT R² is reported per split |

R9 (TIDMAD) applies only if E3 training is approved after the coherence step.

### 3.6 A8 (design doc v0.2.1: second review of Junjie's comments, and an independent runbook review; author decision, 17 Oct 2026)

A8 is folded into §3.5 above. What it changed, and why:

| Change | Where | Reason |
|---|---|---|
| Independent per-PMT gain F3 in every LUCiD cell; control R12 | §3.5.3, §3.5.9 | Per-PMT aging at different rates is private across views; only its shared part is detector state |
| Aging-trend law T and its robustness ratio | §3.5.3, §3.5.6, R8 | Gain drift is a slow trend, not a step; for a shuffled event-level encoder only the range of states matters |
| Co-varying law R11b and prediction R11b; R11 criterion = own R² plus cross-leakage | §3.5.3, §3.5.9 | Effects are degenerate in one snapshot; separability must be tested both ways |
| LUCiD `mae_chi2` with shared module-gain covariance replaces `mae_poisson` | §3.5.6 | A per-PMT Poisson likelihood only matches per-view whitening, which P2 says does not help |
| Read-out before any final norm; log-charge tokens; amplitude probe on read-out tokens; bridge input without per-event z-score | §3.5.5, §3.5.7 | The final RMSNorm rescales each token, and `normalise_input_sequence` z-scores each event; log charge makes gain additive |
| Detectability check; wide yield range; run-mean probes primary; F2 strength set by the check, with the F2′ fallback | §3.5.3, §3.5.3a, §3.5.6 | Runbook review: at about 1 PE per hit PMT a few-% drift is nearly invisible in one event, which would make R11 fail and the null predictions pass by default |
| Propagated and raw-input probes; R6 and R12 use the propagated probe | §3.5.6, §3.5.9 | Runbook review: full-input probes can read gains straight from the input |
| Anchor penalty on module-pooled tokens, every module rescaled, wider β grid | §3.5.6 | Runbook review: a 1% rescale of one module barely moved the global pool, so the anchor could be inactive |
| R7b tests the dark-window-only anchor on module drift | §3.5.9 | Runbook review: attenuation lies inside the span of per-module rescalings, so v0.2.1's attenuation test did not probe an unseen nuisance. The design doc was updated to match |
| Oracle = `sigreg_jepa` on a nominal-gain twin readout | §3.5.3, §3.5.6 | Dividing recorded charges leaves the gain's effect on which PMTs are hit; the twin removes gain effects before threshold, and attenuation is not varied in the main cells |
| Physics reference named correctly (VERITAS time-of-flight fit, not a likelihood fit) | §3.5.6 | `consequence.py` is VERITAS's own Cauchy time-of-flight fit |
| Probe CV grouped by run; shared probe sets across splits; runs of 50 in probe sets; loader mixes runs | §3.5.3, §3.5.6 | With runs of 1000, a 10k probe set held only 10 drift states |
| Occupancy balancing in the context view only | §3.5.4 | Subsampling target PMTs revealed the target hit pattern |
| Seed-floor median stabiliser and a rule for cross-cell comparisons | §3.5.8 | SDs from 3 seeds are noisy |
| Training length, full LUCiD GPU accounting, LUCiD fallback order; bridge β | §3.5.5–§3.5.7 | Steps, development runs and the bridge β were unspecified |
| Scaling benchmark | §3.5.5 | Scalability to more than 10k sensors is measured, not argued |
| Second review pass: R6 measured as excess over the random-init floor; λ = λ0·exp(a·v); R11 margin in seed-floor units; LUCiD β = the largest within the R7 physics tolerance; F3 shortfall makes R12 not evaluable; ceilings labelled lower bounds; generation re-benchmarked at the frozen yield range | §3.5.3, §3.5.3a, §3.5.6, §3.5.9 | The first fixes made R6 pass almost by construction, allowed λ ≤ 0 at a = 0.50, and could make R11's margin unreachable |
| Execution on two shared servers: Deepthought's 4 Titans (development, bridge, generation, evaluation) and 2 of deepthought2's L40S (all LUCiD criterion runs); one GPU type per comparison set; environment order with a gate; Condor NEMO2 off by default | §6, §6.1 | Author decision, 17 Oct 2026; v3 stalled building environments under the file-count quota |
| Resource guard with OK / HOLD / SHED levels, acting only on our own processes | §6.2 | Author: the experiments must never crash or fill the shared servers |

### 3.7 A9 (follow-ups, budget, readout and token time, frozen values; author decisions 17–20 Oct 2026)

A9 registers, as runbook text, the author decisions made after v4 was registered. Each one is
already quoted verbatim, with its date, in `docs/jepa/DECISIONS.md`; that record stays the
authority for the exact wording. **Where A9 conflicts with §3.5, §6.1 or §8, A9 wins.**
_Corrected 21 Oct 2026: grid volume (§3.7.2), physics-reference input (§3.7.3) and the
detectability record (§3.7.4), following the differences listed in `DECISIONS.md` at `368e1c2`._

A9 changes no registered prediction criterion and no registered verdict. It changes four things:
- two labelled follow-ups;
- the LUCiD budget and the protocol the grid runs;
- the readout and the token time;
- the record of values that the registered procedures have now frozen.

**3.7.1 Follow-ups B1-ext and R5-ext** (approved 17 Oct 2026; scoped 18 Oct 2026).
- **B1-ext.** A development β trade-off on the bridge cell, β ∈ {10, 100, 1000}, with the LUCiD
  selection rule: the largest β whose physics stays within 1.1 × plain, per parameter.
  - Budget ≤ 4 Titan GPU-hours.
  - The approved 3-seed anchored run at the selected β is not run as a second B1. β = 10 was
    selected, so that grid would duplicate the registered B1 anchored arm (same cell, arm, seeds
    and β); its partial runs are labelled a replication.
- **R5-ext.** E1c with balanced initialisation, per-layer SD = (1e-3)^(1/L), so the end-to-end
  scale matches depth 1. Same learning rate, steps and eligibility rules; config frozen before the
  rows run; CPU only.
- **Both** are labelled `registered: false` with their `extension` name, written outside the
  registered result directories, and reported beside B1 and R5. Neither replaces a verdict.
- **Recorded outcomes (18 Oct 2026):**
  - B1-ext: only β = 10 qualifies. β = 100 fails on t0 (ratio 2.44 to plain); β = 1000 fails on all
    three parameters and lowers nuisance R² below plain (0.445 vs 0.542). Stronger anchoring buys
    no nuisance protection and costs physics, so B1's failure is not "the anchor was too weak".
    2.45 of the 4 GPU-h were used.
  - R5-ext: 7 of 10 rows are evaluable (the uniform-init E1c had none). The R5 ordering rule fails
    at γ = 1.0 (2 of 5 seeds) and is not evaluable at γ = 0.5 (2 of 5 seeds evaluable).

**3.7.2 Budget and grid protocol** (G2 ruling 18 Oct 2026; G2 amendment 20 Oct 2026).
- **L40S cap: 170 GPU-hours**, replacing the 80 h in §3.5.6 and §6.1. It covers the development
  cohort, the main grid, the secondary runs and their evaluation. The Titan 300 h, CPU 48 h, quota
  and §6.2 limits are unchanged, and so is the limit of at most 2 jobs on deepthought2.
- **TF32 stays off.** Measured on the L40S: 1.22× average (1.15× steady) steps/s, below the 1.5×
  threshold. The §6.1 fp32 rule stands.
- **Grid protocol: §3.5.6 ladder rung (1) only.**
  - 10 000 steps, 3 seeds for every main arm, all secondary runs retained.
  - Recorded as `fallback_rung: 1`. That is weaker than the registered protocol only in the seed
    count, and R8's "3 of 3" clause applies.
  - Expected volume: 165 runs (6 main cells × 21; R11 15, at its registered 5 seeds because it is a
    secondary group; R11b 9, dark-window anchor 3, law T 9, occupancy 3). Option A would be 159,
    with R11 also cut to 3 seeds.
- **Start rule.** Before the first grid job, re-project with the measured per-arm walls, including
  evaluation.
  - Start only if ledger + projection ≤ 160 GPU-h.
  - Otherwise run option A: rungs (1) + (3), 3 seeds and 5 000 steps. It is recorded as a weaker
    protocol, with no new gate.
- **Inside the grid, training length never changes.** If the budget tightens, apply rung (2):
  - before each secondary group, if ledger + remaining projection exceeds 170 h, drop secondary
    groups from the end of the registered list (occupancy, aging trend, dark-window, R11b, R11);
  - record each drop, and mark the affected predictions `not_evaluable`.
- **Secondary groups run after the main arms** wherever the per-cell data lifecycle allows
  (engineering choice).
- **Configs.** The registered config stays the default for development, smokes and tests. Each
  fallback is a named derived config, `derived_from` the registered one, recording its rung, and
  every run manifest records `fallback_active` and `fallback_rung`. The option-A config stays
  committed and unused unless the start rule selects it.
- **Titan memory shape** (engineering; no registered numeric moves): micro-batch 8 with gradient
  accumulation 4, so the effective batch is still 32.

**3.7.3 Readout for reported runs, and token time** (author, 18 Oct 2026).
- **PMT readout.**
  - Reported LUCiD runs use PMT branch `codex/junjie-pmt-response` at `5f4ecc0`, merged into
    `JEPA` as `4e07c0c`, readout label `readout_pmt_r6`.
  - It supersedes `0fba034` (development only, dark counts off) and `b341037` (R-6).
  - Dark counts: 1 kHz per PMT, in the 1 µs readout window [−200, 800] ns, about 2 dark hits per
    event. `v_dark` = λ_dark(1 + σ_spe²) = 0.00109 PE², derived from the reference config.
  - The labels `readout_dark_off` and `readout_synthetic_unrepaired` are legacy; no reported result
    uses them.
- **Fixed detector response.** The reference response has a per-module `group_parity` gain (odd
  modules +3%) and a per-module timing width. Both are fixed detector response: they are present in
  the measured readout and in the nominal twin. The twin (§3.5.3, §3.5.6) removes only the injected
  F1 and F3 gains.
- **Token time** replaces "first-hit time relative to the trigger" in §3.5.5:
  - **t_token = t_digit − t_trigger, after the prompt-window cut.**
  - t_trigger is the readout's NHits trigger: N = 4 digits within 20 ns
    (`readout.nhits_trigger_time(n=4, window_ns=20)`).
  - The prompt window is [−20, +60] ns around the trigger (`readout.prompt_window_mask`). Digits
    outside it are dropped and counted.
  - With no trigger, the origin falls back to the readout-window start and is counted. No window cut
    is applied, following the imported mask.
  - The rule is imported from `noise_module_lucid.readout`, never reimplemented. Dataset generation
    refuses to run without it.
- **The true emission time is truth only.** It, and the per-event emission offset, never enter an
  input, token, feature, mask or probe. Any code path that would use them as input is a defect.
- **Leak guard.** Each event gets an emission-time offset, uniform in [−50, +50] ns from the keyed
  RNG, added to source photon arrivals and recorded as truth. The readout window is unchanged.
  Invariance tests:
  - dark counts off: a constant shift of all photon times gives bit-identical tokens, and a trigger
    time shifted by exactly that constant;
  - dark counts on: the same holds for the source-derived digits only. Report how often a dark
    coincidence moves the trigger (`dark_moved_trigger_rate`).
- **Counters per run:** `n_no_trigger`, `n_prompt_dropped`, `n_dark_moved_trigger` and its rate.
- **Physics reference unchanged.** The time-of-flight fit reads the trigger-relative digit times and
  fits t0 itself, so a per-event time origin is absorbed.
- **Development cohort redone on this input.** All earlier development runs are kept, labelled
  `superseded_by_token_time_origin_change`. Reason: with trigger t = 0 at the true emission time,
  `mae_mse`'s linear vertex (0.093 m) beat a reference fit that must estimate t0.
- **R7b reporting [A9].** The dark window has about 2 dark hits against about 900 source hits. The
  dark-only draw's measured embedding change is reported next to R7b, so that a pass is not
  over-read.

**3.7.4 Detectability check: F3 marginalised, values frozen** (author, 18 Oct 2026; frozen on the
`readout_pmt_r6` readout).
- **Rule change to §3.5.3a.** F3 is treated as random, not known: each PMT's variance in the Fisher
  ceilings gains s_p²·μ_p², the same term as the §3.5.6 `mae_chi2` covariance. The adjustment ladder
  restarts from the registered s_p = 0.05. Old and new ceilings are recorded side by side; no
  pass/fail changed, and the largest shift was −3.5e−3.
- **Frozen outputs of the registered procedure** (not new choices). Decision JSON: the consistency
  re-freeze `results/jepa/20261018_wp4b_detectability_r6/wp4b_detectability_wcte_r6_consistency.json`
  (file sha256 `411190bc…`); it supersedes `wp4b_detectability_wcte_r6.json` (`1e4407ac…`), with the
  same decisions.
  - Yields: Y_lo = 6 298.44 photons (median 199 hit PMTs); Y_hi = 539 434 (0.95 hit-fraction
    bound); median 58 289.
  - Ladder steps taken: Y_hi 320 000 → 539 434; F2 a 0.1 → 0.2 → 0.5; s_p 0.05 → 0.10.
  - Run-level ceilings: F1 0.684 and 0.890 at s = 0.05 and 0.10, which reach 0.6; s = 0.02 does
    not (0.257). F2 0.919, law T 0.986.
  - F2 is used at a = 0.50; F2′ is not used.
  - F3 fails at s_p = 0.10 (0.323), so **R12 is `not_evaluable`, with no gate**, as §3.5.9
    provides.
  - Law T passes. R6 and R7 are evaluable. G6 is not triggered.
- **Generation** is re-benchmarked at the frozen yields (§3.5.6).

**3.7.5 Development cohort decisions** (frozen 20 Oct 2026; β table accepted by the author 20 Oct
2026).
- **Physics probe form (step 2): ridge.** `sigreg_jepa` 0.32363 m (2.272× the reference
  0.142446 m) and `mae_mse` 0.11349 m (0.797×) are not both above 3×.
- **β (step 3)**, by the rule "largest β with vertex resolution ≤ 1.1 × plain", that is ≤ 0.355989 m:
  - `sigreg_jepa_anchored` β = 1000 (0.20788 m) and `sigreg_jepa_anchored_dark` β = 1000
    (0.21010 m), both recorded as `edge_selected`. The grid is not extended, and there is no
    re-selection after results.
  - `anchor_only`: no β qualifies (best 0.44891 m). It runs in the grid at β = 1000, labelled
    `beta_not_qualified`, as a descriptive control. Any clause that needs a qualified
    `anchor_only` β is `not_evaluable` from it. The finding that the anchor alone reaches the
    physics threshold at no β is reported as a development result.

**3.7.6 What A9 does not change.**
- The registered predictions (§3.5.9, §4, v3 §2.4) and every verdict recorded so far. E1 stands
  as: R1 fail, R2 fail (independent recompute recorded beside it), R3 `not_evaluable`, R3b pass,
  R4a fail, R4b fail, R5 `not_evaluable`. B1 fails.
- The §6.1 comparability rule, the §6.2 guard limits, the no-touch packages, and the §9 timeline.


### 3.8 A10 (mixed event domains: photon, muon and electron bombs; author instruction 22 Oct 2026 — DRAFT, NOT IN FORCE)

**Status.** Draft amendment for registration. It is **not in force**: no grid, dev cohort or
outcome-generating run may use it until the orchestrator commits this section unchanged as
`jepa: register runbook amendment A10 (author revision)`. Until then the registered §3.5/§3.7 text
is authoritative. The proposal and its options are in
[GATE_A10_MIXED_DOMAINS_2026-10-22.md](GATE_A10_MIXED_DOMAINS_2026-10-22.md).

**Authority.** Junjie (21 Oct 2026) proposed adding μ/e particle bombs because the isotropic photon
bomb only tests ToF, while Cherenkov adds directionality and particle physics; he leans toward
mixing the datasets. The author (22 Oct 2026) accepted option A and required **both μ⁻ and e⁻**.
Junjie's message is design input, not approval of the mixing ratio, generator mode or criteria.

**A10.1 Event domains (data factor, not a method).** E2 keeps one method — SIGReg-JEPA plus the
noise anchor, one encoder and one probe protocol (§3.5.5–§3.5.6) — and adds two event domains
beside the isotropic photon bomb:
- `photon`: unchanged isotropic point source (§3.5.3); truth vertex and log yield.
- `muon`: μ⁻ track; Cherenkov emission along the track, `cos θ_C = 1/(β n)`; truth vertex,
  direction, particle type, initial energy, deposited energy and containment.
- `electron`: e⁻ shower; Cherenkov emission from the shower axis/profile; the same truth fields.
A domain is a data-domain factor. The photon bomb is a simplified isotropic domain, not a literal
subset of a Cherenkov event; per-domain results are reported, pooled results secondary.

**A10.2 Run structure and mixing.** Each run carries one detector state, shared by every domain in
that run. Default training-run composition is **334 photon + 333 muon + 333 electron per 1000
events (≈1:1:1)**; probe, robustness and dark-only runs are 17 + 17 + 16 per 50 events. The class
label is metadata only and never a model input; batches are class-balanced by construction, with
optional exact per-batch stratification if the development domain classifier shows imbalance.
A **2:1:1** photon:μ:e fallback is allowed only through the budget fallback in A10.6 and is
recorded. The ratio is a pre-registered parameter; changing it needs a further amendment.

**A10.3 Generator.** Two modes, one label per manifest:
- `photonsim`: PhotonSim/Geant4 μ⁻ (and e⁻, if supplied) events transported through the pinned WCTE
  geometry. Existing μ⁻ samples: 500/1000/1500 MeV, 20 events each. No e⁻ generator exists in this
  repository at the time of drafting.
- `analytic_cherenkov`: a fast parameterised track/shower Cherenkov source in the JEPA generator,
  using the pinned geometry, attenuation and per-photon readout; labelled `simplified_cherenkov`.
**Default: hybrid.** Production uses `analytic_cherenkov`; a mandatory validation subset of real
PhotonSim μ⁻ events must reproduce the hit-count, charge and time distributions within pre-declared
tolerances before the grid. Electron production is analytic until Junjie supplies PhotonSim e⁻
samples. Modes are never silently mixed; a combined dataset records both.

**A10.4 Particle truth and probes.** Per-domain probes:
- photon: vertex, log yield;
- muon: vertex, direction, deposited energy;
- electron: vertex, direction, deposited energy, particle type;
- nuisance in every domain: module gain (F1), per-PMT gain (F3), attenuation (F2), age (law T).
Run-mean is primary; the propagated probe is used for R6; the raw-input ceiling and per-domain
random-init floor are reported. Energy probes use the contained subset (deposited energy inside the
instrumented volume; escaping events are diagnostic); direction and vertex probes use all events.
Particle-type probes use the μ and e domains jointly. A domain classifier on frozen embeddings
reports domain AUC as a diagnostic, never as a criterion.

**A10.5 Predictions.** R6 and R7 are evaluated per domain. The **particle domain (μ + e) is the
primary physics-retention domain**; the photon domain is the isotropic control; the pooled result is
secondary. R6 and R7 criteria are otherwise unchanged (§3.5.9). Optional **R13**: in the particle
domain, direction and deposited-energy probes exceed their random-init floors by more than
2 × the seed floor. R11/R12 may be extended to the particle domains by a later amendment; R1–R5 and
B1 are unchanged.

**A10.6 Detectability, budget and fallback.** The §3.5.3a detectability check is re-run on the mixed
sample, per domain, and frozen before training. A domain whose F1 run-level ceiling is < 0.6 is
excluded from R6/R7; if no domain passes, G6 for the dependent predictions. F3/F2/T ceilings are
re-recorded. Training volume is unchanged if mixed replaces photon-only (same rung-1 run count);
generation, storage and per-domain analysis increase and are re-projected against the 160 h start
rule and 170 h cap. Fallbacks, in order: (1) 2:1:1 mixing; (2) electron domain deferred; (3) E3/E4
dropped; (4) G2. Each step is recorded.

**A10.7 What A10 does not change.** E1, R1–R5, B1, the resource guard and shared-server rules, the
no-touch packages, the class-label-free input contract, the one-method framing, and the registered
arms/objectives of §3.5.6. The photon bomb is not reclassified as N or S; domains are a data factor.

**A10.8 Author decisions (22 Oct 2026; registration slots filled).**
- **Generator: hybrid.** Analytic Cherenkov production for the grid; a mandatory PhotonSim μ⁻
  validation subset; electron analytic until Junjie supplies PhotonSim e⁻ samples.
- **Electron source: analytic first.** Replacing it with full PhotonSim e⁻ needs a further amendment.
- **Mixing: 1:1:1** photon:μ:e (334/333/333 per 1000-event training run; 17/17/16 per 50-event
  probe run). The 2:1:1 ratio is only the recorded budget fallback of A10.6.
- **Primary domain: particle (μ + e) primary for R6/R7**; the photon domain is the isotropic
  control; the pooled result is secondary.
- **Remaining values (not design slots):** particle energy ranges/topology (defaults: μ⁻
  300–1500 MeV, e⁻ 100–1000 MeV log-uniform; energy probes on the contained subset) and the drift
  values s, s_p and a (pending Junjie). They do not change the domain or mixing design.

---

## 4. Registered predictions: changed rows only

All other rows (R3, R3b, R4a, R4b, R5, R9, R10) are as in v3 §2.4. **The Tier-1 R6, R7 and R8 rows
below are superseded by §3.5.9** (A7/A8); they are kept only as the record of the A4 design.

| ID | Prediction | Criterion |
|---|---|---|
| R1 | Linear JEPA (CCA) recovers R_c when no noise is shared | E1a as v3. **Plus E1b**: channel split at (g = 0, line = 0), n = 1e5, median max angle ≤ 10° (population design value 0.0°). |
| R2 | Shared noise displaces it | E1a as v3. **Plus E1b**: channel split, Spearman(g, seed-mean angle) ≥ 0.8 over G; and the seed-mean angle at g = 0.05 is ≥ 45°. |
| R6 | Deep JEPA puts shared noise in its embedding as displacement grows | E2, nominal SNR, separately for `ijepa_ema` and `sigreg_jepa`: Spearman over the three levels of seed-mean nuisance R² (mean of `common_0`, `common_1`) ≥ 0.8. |
| R7 | The noise anchor cuts nuisance and keeps physics | At the **high** level (selected g), nominal and sub-noise: `sigreg_jepa_anchored` has lower nuisance R² than `sigreg_jepa` in every paired seed, and its seed-mean physics MSE/CRLB per parameter is at most 1.1 × that of `sigreg_jepa`. Also report `mae_chi2` beside it (A6). |
| R8 | Learning the shared noise costs robustness | As v3, but the shift is common-mode amplitude × 2 (§3.2.9). |

---

## 5. Decision and gate rules (A5)

- **Decision default.** Any choice that does not change a registered prediction, metric, cell,
  dataset role or endpoint is an engineering decision. Examples: chunk sizes, solver tolerances
  within reviewed configs, file layout, plotting, retry timing, environment location.
  - Make the simplest defensible choice.
  - Log it as **one line** in `docs/jepa/DECISIONS.md` (date, choice, reason).
  - Continue. No gate.
- **G5** applies only to changes of the registered scientific specification.
- **G6** applies only when:
  1. required data are unavailable or invalid; or
  2. a population or feasibility check shows a registered prediction cannot be evaluated *after*
     applying v4's own fallbacks.
- **G2** budget caps as v3.
- **Gate files**: one page at most, ending with **Options A/B/(C), one marked recommended**, so the
  author can answer with a letter. Work that does not depend on the gate continues.
- **No agent rewrites the runbook.** Proposed specification changes go into a gate file. Only the
  author, or Claude at the author's request, edits it.
- **Reviews:**
  - WP2 is done.
  - After E1: a light independent recompute of R1–R5 from the raw tables.
  - WP5 independent review.
  - Final review.
  - No extra prelaunch review rounds.
- **Report claims of author approval verbatim.** Quote the author's words, with date and
  context, in the decision log. Never infer approval.

---

## 6. Hosts, environments and the shared filesystem (overrides v3 §3.1, §3.2 accounting)

- **E1 (E1a, E1b, E1c) is numpy/scipy only.** Run it now on the Mac, in the already-accepted local
  environment. Deepthought is optional: if used, it needs only a numpy/scipy environment, a few
  hundred files.
  - Measure the Mac's memory first (`sysctl hw.memsize`). With the streaming rule (§3.1.2), peak
    memory per worker should stay under 4 GB.
  - Workers = min(physical cores − 2, floor(free GB / 4)).
- **GPU work runs on two shared servers** (author decision, 17 Oct 2026; §6.1–§6.2): Deepthought's
  four GTX Titan 12 GB cards and two of deepthought2's L40S cards. Condor (NEMO2) is off unless the
  author sets `CONDOR_NEMO2=yes`. Use the repository's `torch==2.5.1` pin (cu124 build, or cu118 if
  a host's driver is older than 550; record the build).
  - Smoke first, within v3's benchmark caps.
  - Record the host and GPU model for every run.
- **Deepthought CPU torch environment** is optional, for evaluation only, and never blocks E1–E3.
  If wanted:
  1. confirm at least 60 000 free quota entries under `/ceph/dwong`;
  2. install with `PYTHONDONTWRITEBYTECODE=1` and `pip --no-compile`;
  3. one attempt, 45-minute limit.

  Alternatively, use an existing CPU torch environment **read-only**: record its path and
  versions; torch ≥ 2.5 is acceptable for CPU evaluation; add a 10-batch equivalence test against
  the local environment.
- **Hard rule on shared storage.** On Deepthought and deepthought2, never delete, move or modify
  anything outside `/ceph/dwong/jepa/`, caches included, unless the author's chat message names the
  exact paths for that action. A quota problem is a gate with options, not a cleanup. Never touch,
  signal or renice another user's process.
- **Budget accounting:**
  - Campaign wall-hours and GPU-hours, rounded to 0.1 h, in `RESOURCE_LEDGER.json`.
  - One row per job or batch. No per-second ledgers.
  - Failed attempts count.
  - v3 caps unchanged, except that all LUCiD work has its own caps and fallback order (§3.5.6).

### 6.1 Hosts and execution path (A8, 17 Oct 2026; author decision on GPU use)

**Author's decision (17 Oct 2026), recorded as given:** use both servers.
- Deepthought: all four GPUs may be used, but not all of them fully loaded.
- deepthought2: its L40S cards are shared with other users; we may use two, without filling them.
- The agent monitors the servers so that our experiments never crash or saturate them.

Launch-prompt flags: `DEEPTHOUGHT_GPUS=4`, `DEEPTHOUGHT2_GPUS=2`, `CONDOR_NEMO2=no`.

| Work | Where | Why |
|---|---|---|
| WP0-v4, WP2-v4, E1 (WP3), all code, unit tests, CPU smokes on tiny inputs, analysis of returned metrics | the author's Mac | numpy-only or small |
| Development and debugging runs, WP4b smokes, the whole bridge (WP4a, all arms and the ablation), the detectability check, LUCiD generation, probe fitting and evaluation of returned embeddings, the 2k and 10k scaling sizes | **Deepthought**: 4 × GTX Titan 12 GB, 40 CPU cores (`ssh deepthought`) | fast iteration, no queue |
| Every training whose output enters a registered LUCiD criterion: development cohort (probe decision, β), main grid, secondary runs, robustness evaluation; the 40k scaling size | **deepthought2**: 2 L40S cards, shared (`ssh deepthought2`) | faster cards; one GPU type for every LUCiD comparison |
| Anything | Condor NEMO2 | off unless the author sets `CONDOR_NEMO2=yes` (last bullet below) |

- **Comparability.**
  - Every training inside one comparison set runs on one GPU type: all LUCiD criterion runs on
    L40S, the whole bridge on Titans.
  - Train in fp32 with TF32 disabled everywhere (`torch.backends.cuda.matmul.allow_tf32 = False`,
    `torch.backends.cudnn.allow_tf32 = False`). **[A9]** TF32 was benchmarked on the L40S (1.22×,
    below the 1.5× threshold) and stays off (§3.7.2).
  - If deepthought2 is unusable for more than 3 consecutive days during WP5, write a gate:
    (A) wait (recommended); (B) rerun the entire remaining comparison sets on Titans, within the
    Titan cap. Never split a comparison set across GPU types.
- **Code** reaches both hosts only through GitHub: push `JEPA`, then clone one directory per SHA
  under `/ceph/dwong/jepa/code/<sha>/`.
- **Working root** on both hosts is `/ceph/dwong/jepa/`. If deepthought2 does not mount it, write
  a gate asking the author to name a path there; no work is written on deepthought2 before that.
- **Python environment, in this order:**
  1. an existing GPU torch environment on the host (for example the one the PMT branch trained
     with), used **read-only**: record its path and versions; run a 10-batch equivalence test
     against the local environment; never modify it. If it lacks JAX/LUCiD, use 2 for a separate
     generation environment;
  2. one new environment under `/ceph/dwong/jepa/env/<name>/`, only if preflight shows at least
     100 000 free entries under `/ceph/dwong` after the planned data files. Install with
     `PYTHONDONTWRITEBYTECODE=1` and `--no-compile`; one attempt, 45-minute limit; record the
     entries used. One environment serves both hosts if they share `/ceph`, OS and a compatible
     driver;
  3. otherwise gate G-ENV: (A) the author names a local-disk path outside `/ceph` (recommended);
     (B) one single-file container image under `/ceph/dwong/jepa/`; (C) set `CONDOR_NEMO2=yes`.
  - Check `nvidia-smi`: a driver ≥ 550 allows the cu124 build; otherwise use cu118.
  - Check whether the installed JAX supports the Titans' compute capability. If not, LUCiD
    generation runs on Deepthought's CPUs.
- **Running a job** (no scheduler):
  - Each job runs in its own tmux session `jepa_<run_id>`, started by `scripts/jepa_run.sh`. The
    script:
    - sets `CUDA_VISIBLE_DEVICES` to one GPU chosen by the guard (§6.2);
    - runs `nice -n 10`, with at most 4 BLAS threads and 4 data-loader workers;
    - wraps the job in `timeout` at 1.5 × its benchmarked time;
    - calls `torch.cuda.set_per_process_memory_fraction` at the §6.2 cap;
    - logs to `/ceph/dwong/jepa/jobs/<run_id>/` and writes a PID file there.
  - One JEPA job per GPU.
  - One manifest row per run (run_id, SHA, config hash, arm, cell, seed, GPU type). The orchestrator
    starts the next pending run only when the guard status says OK.
  - **Memory:** batch 32 must fit within the §6.2 per-process cap: about 21 GB on L40S, 9.6 GB on a
    Titan. If it does not fit on a Titan, use gradient accumulation with the same effective batch
    of 32, and record it.
- **Checkpoints.**
  - Save at least every 30 minutes, and resume after a stop.
  - A job stopped by the guard (§6.2) or by an infrastructure fault is resumed from its checkpoint
    under the same run ID. Record the interruption.
  - A scientific failure (NaN, divergence) is recorded, never re-tuned.
- **Data.**
  - Shards of at most 100 MB under `/ceph/dwong/jepa/data/<dataset_id>/`, with a manifest and
    `SHA256SUMS`, read in place by the jobs.
  - float16 storage, or on-the-fly generation from keyed seeds, is an engineering decision, made on
    the benchmark, provided every arm sees identical events.
- **Returns.**
  - Metrics JSON, float16 probe-set embeddings for the §3.5.6 read-out forms, checkpoints and logs
    go to `/ceph/dwong/jepa/returns/<run_id>/`.
  - The compact parts are rsynced to `results/jepa/<run_id>/` with hash verification (v3 §1.2).
  - Checkpoints and raw embeddings stay on `/ceph`, with path and hash recorded.
- **Caps and ledger** (`RESOURCE_LEDGER.json`, GPU-hours = job wall-clock × 1 GPU, per GPU type,
  failed and interrupted attempts included):
  - L40S (deepthought2): 80 GPU-hours for all LUCiD criterion work (§3.5.6). **[A9]: 170 GPU-hours
    (§3.7.2).**
  - Titan (Deepthought): 300 GPU-hours for development, smokes, the bridge, generation and
    evaluation. The bridge counts here instead of under v3's E2 cap.
  - CPU: LUCiD generation 48 wall-hours (§3.5.6), on at most 16 cores.
- **Condor NEMO2** (only if `CONDOR_NEMO2=yes`):
  - Submit from Deepthought.
  - Build environments inside the job sandbox with `uv` (Python 3.11), modelled on
    `scripts/train_compact_l40s/submit.generated.jdl` (L40S, `accounting_group = delight`,
    `use_x509userproxy = False`); at most 8 jobs at once.
  - Ledger from `condor_history`.
  - NEMO2 L40S runs join an L40S comparison set only after a 10-batch equivalence test against
    deepthought2.

### 6.2 Resource guard: never crash or fill the shared servers

These limits apply to **our** processes only, on each host:

| Limit | Deepthought (4 × Titan 12 GB) | deepthought2 (L40S, shared) |
|---|---|---|
| GPUs in use by JEPA | ≤ 4, but at most **3 training jobs** at once. The 4th GPU only for jobs ≤ 30 min (smokes, evaluation, probes) | ≤ 2 |
| GPU memory per JEPA process | ≤ 80% of the card (about 9.6 GB) | ≤ 45% of the card (about 21 GB) |
| Start on a GPU only if | no other user's process holds > 1 GB on it, and its 5-minute mean utilisation is < 20% | other users' memory + our cap ≤ 85% of the card, and its 5-minute mean utilisation is < 60% |
| CPU cores, all JEPA processes | ≤ 16 of 40 | ≤ 16, or 40% of the cores if fewer |
| RAM, all JEPA processes | ≤ 128 GB, and host free RAM never below 20% | ≤ 64 GB, and host free RAM never below 20% |
| Host load | start nothing if the 15-minute load > 0.7 × cores | same |
| Storage | write no new data if free entries under `/ceph/dwong` < 30 000, or if JEPA data exceed 2 TB | same |

**The guard** (`scripts/jepa_guard.py`, part of the JEPA code):
- It runs in tmux `jepa_guard` on each host, every 2 minutes, whenever a JEPA job runs there.
- It reads:
  - `nvidia-smi`: per GPU memory, utilisation, and processes with their owners;
  - `/proc/meminfo` and the load average;
  - the quota or entry count;
  - the CPU and RSS of our processes.
- It writes `/ceph/dwong/jepa/jobs/guard_<host>.json` (current state) and appends events to
  `guard_<host>.log`.
- It sets one of three levels:
  - **OK:** new jobs may start within the limits.
  - **HOLD:** a start condition fails, or another user starts heavy work on a GPU we use (their
    memory > 2 GB or utilisation > 30%). No new starts on that GPU or host; running jobs continue.
  - **SHED:** host free RAM < 10%, total memory on any GPU > 95%, 15-minute load > 0.9 × cores,
    free entries < 10 000, or one of our processes exceeds its own cap. The guard sends `SIGTERM`
    to the most recently started JEPA job on that host, which checkpoints and exits. It re-checks
    after 5 minutes and repeats while SHED persists. A shed job returns to the queue and resumes
    from its checkpoint later.
- It acts only on processes listed in our PID files under `/ceph/dwong/jepa/jobs/`, owned by
  `dwong` and tagged `jepa_` in the command line. It never touches anything else.
- **Before first use:** unit tests with synthetic readings for each level, then a 10-minute dry run
  on each host (log only, no signals).
- **The orchestrator:**
  - reads the guard status at every check, and launches only on a status younger than 2 minutes;
  - logs each HOLD or SHED event as one line in `docs/jepa/INFRA_LOG.md`;
  - reports to the author if SHED fires more than twice in 24 hours, or if HOLD blocks a host for
    more than 24 hours.

---

## 7. Repository hygiene (A5)

- **Commit:** code, configs, manifests, summary tables, figures, `SHA256SUMS` and short READMEs.
- **Size limits:**
  - each experiment's results directory ≤ 20 MB;
  - each JSON ≤ 1 MB;
  - per-cell raw population matrices stay on the run host (path and hash recorded).
- **Infrastructure logs** (quota, cleanup, environment setup, retries) stay on the host. The repo
  gets at most 20 lines per event in `docs/jepa/INFRA_LOG.md`.
- **One-time cleanup in WP0-v4:**
  - Remove from the tree the bulky files in `results/jepa/20261016_storage_recovery/` and
    `results/jepa/20261016_wp0_remote_env*/`. Keep a README, `SHA256SUMS` and one summary JSON per
    directory. Git history retains everything.
  - Commit the cleanup separately as `jepa: v4 hygiene (history retained)`.

---

## 8. Work plan, in order

**WP0-v4: register and clean up.**
1. Commit v4 unchanged, together with this runbook's companions: the v4 prompt and
   `check_v4_levels.py`.
2. Verify the CPAL 2027 Recent Spotlight call and deadline on the official site, and record the
   URL and date in `DECISIONS.md`.
3. Do the §7 cleanup.
4. Update `PREDICTION_STATUS.json`: R6–R8 go from `not_evaluable` to `not_run`, with reason
   "v4 A4/A7 redesign"; add B1, R7b, R11, R11b and R12 as `not_run`, with reason "v4 A7/A8".

**WP2-v4: implement A4 in `views.py`.**
- Changes:
  - a base-ρ parameter, default 0 for the A4 law;
  - the θ SD change for E1b;
  - the extended G grid and the line grid at g = 0;
  - the `channel` default for E2.
- Tag the law `v4-A4-waveform-v1`, and keep the v3 law selectable for reproduction.
- Recompute population tables for 13 cells × 4 splits, including D. They must match §A to 0.5°.
- A fresh reviewer re-runs the §A script and the WP2 population tests.

**WP3: E1** (local, now). Run E1a, then E1b, then E1c, as in v3 §5 with A3 and A4.
- Accept: `PREDICTIONS.md` for R1–R5 regenerated by `python -m jepa_noise.report E1`.
- Light independent recompute.

**WP4a: toy torch implementation and bridge** (§3.5.7).
- `current_compact` subclass with the A4 channel masks, reading tokens before `final_proj_norm`.
- Arms: `sigreg_jepa`, `sigreg_jepa_anchored` and `mae_chi2`, plus the parallel-attention ablation.
- Run the amplitude-preservation probe (after the time block and on the read-out tokens).
- Run the bridge (B1).

**WP4b: LUCiD stack** (§3.5.2–3.5.6). Can start in parallel with WP3.
1. Merge the PMT branch at the author-named SHA. Until then, develop on the unrepaired readout as
   §3.5.2 allows (no reported results).
2. Build `lucid_bombs.py` with the F1, F2, F2′, F3, R11b and T laws, the nominal twin and the run
   lengths of §3.5.3; benchmark 1000 events.
3. Development cohort step 1 (§3.5.6): Y_lo, Y_hi, N_min and the detectability check (§3.5.3a).
   Re-benchmark generation at the frozen [Y_lo, Y_hi] before projecting the 48 h CPU cap. Re-run
   the check after the PMT merge.
4. Build the point encoder (log-charge tokens, read-out before any final norm) and its tests.
5. Run the scaling benchmark.
6. Implement the seven LUCiD arms, the dark-window-only anchor variant and the four probe forms of
   §3.5.6, including the random-init floor. Unit-test the `mae_chi2` covariance (dense computation
   on one module; Monte-Carlo readout covariance).
7. 500-step smoke of all LUCiD arms on a Titan, then a short check on deepthought2, including the
   leakage and amplitude-preservation tests. The guard (§6.2) is tested and running before the
   first GPU job on each host.

**WP5: E2 main and secondary runs on LUCiD** (§3.5.6).
1. Development cohort steps 2 and 3 (§3.5.6): the physics-probe type decision, then β.
2. Main grid.
3. Secondary runs, in the §3.5.6 order, within the cap.
4. Robustness tests (main cells and the aging-trend extrapolation); amplitude-preservation test on
   the final encoders.
5. Seed-floor analysis.
6. Independent review (recompute B1, R6–R8, R7b, R11, R11b and R12 from raw outputs).

**WP6: E3.** The coherence step only (v3 §7.2). Training needs an author go.

**WP7** (optional E4) on the frozen LUCiD encoders (A7 §3.5.1).

---

## 9. Timeline (CPAL 2027 Recent Spotlight; deadline to verify in WP0-v4, planned 18 Jan 2027)

| By | Milestone |
|---|---|
| 24 Oct | WP0-v4 (registering v4 with A6, A7 and A8) and WP2-v4 accepted; E1a complete |
| 31 Oct | E1b and E1c complete; PMT repairs merged at a named SHA (or G-PMT); LUCiD generator benchmarked |
| 14 Nov | WP4a bridge done; WP4b encoder tests, scaling benchmark, guard and GPU smokes on both hosts |
| 12 Dec | WP5 (E2 main and secondary runs) complete and reviewed |
| 22 Dec | WP6 coherence step; WP7 if time allows |
| 8 Jan | `REPORT.md` and final review; the author drafts the manuscript |

If a milestone slips by more than a week, write a one-page gate with descoping options. Do not
silently compress the work.

---

## Appendix A. v4 design check (population, 16 Oct 2026)

Run with `PYTHONPATH=src python docs/jepa/check_v4_levels.py`, at `f021d3c`, using `Tier1Views`
for J0, the temporal covariance and the A1 bases. "Angle" is the max principal angle between the
top-k CCA context directions and the top-k R_c. The cc values are canonical correlations.

**v3 design** (base ρ = 0.4, v3 θ SD, channel split):

| g | γ_n | 3rd physics cc | angle k = 3 |
|---|---|---|---|
| 0 | 0.736 | 0.028 | 18.4° |
| 0.005 | 0.787 | 0.027 | 74.0° |
| 0.05 | 0.923 | 0.026 | 84.6° |
| 0.45 | 0.988 | 0.025 | 89.2° |

The third physics mode is essentially absent, so the design is uninformative.

**A4 design** (base ρ = 0, E2-matched θ SD, channel split):

| g | 0 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.225 | 0.45 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| γ_n | 0.000 | 0.357 | 0.526 | 0.690 | 0.847 | 0.917 | 0.962 | 0.980 | 0.990 |
| physics cc (1, 2, 3) | 0.999, 0.999, 0.682 | | | | | | | | 0.999, 0.999, 0.658 |
| angle k = 3 | **0.0°** | **11.3°** | 22.3° | **48.0°** | **77.0°** | 84.1° | 87.5° | 88.8° | 89.4° |
| angle k = 1 / 2 (≤) | 2.1° / 0.0° | | | | | | | | 2.6° / 1.8° |

Reading: shared common mode displaces the **weakest** physics direction first, which is exactly
P1's prediction. The interleaved split gives the same values within 0.5°. Selected E2 levels:
g = 0.005, 0.02 and 0.05.
