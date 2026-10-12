# Runbook — re-scoping N towards the two claims (LUCiD response N, TIDMAD controlled N, arm portfolio)

**Date:** 12 October 2026. **Trigger:** collaborator comment on the LUCiD noise module
(`docs/ISSUE_LUCID_NOISE_MODULE.md`), reproduced verbatim in Appendix A; the assessment of it is
Appendix B.
**Design sources (read first):** `docs/EXPERIMENT_DESIGN.md` §§I.3, II.1–II.5, II.7, III.6;
`docs/PREREGISTRATION.md` §§1, 8; `src/latent_monitor/protocol/labels.py` (the origin taxonomy);
`docs/RUNBOOK_REFERENCE_UPGRADES_2026-10-09.md` (the WP1–WP8 lanes this runbook must not collide with).
**Repository:** `seanxia8/VERITAS` (formerly ORACLE), branch `rescope-n` off `dev` @ `37137c3`.

This runbook covers five work packages. It does **not** cover Phase B, Phase C, any training run,
building `nuradio_simulation`, a JAXTPC or LIGO arm, reviving the HeST/DELight trace stage, editing
`latex/paper3_proposal.tex`, or freezing anything.

| WP | Task | Kind | Status at writing |
|---|---|---|---|
| WP-A | Scope revision in the design documents (taxonomy, LUCiD role, training protocol, arm portfolio) | docs | to do |
| WP-B | LUCiD detector-response N families (`noise_module_lucid/response.py`) | code + tests | to do |
| WP-C | TIDMAD controlled N injection on the noise-only records (`tidmad_transformer/n_injection.py`) | code + tests | to do |
| WP-D | Arm-portfolio note for Junjie and the reply to the comment | docs | to do |
| WP-E | Verification: suites, smoke regression, independent review | check | to do |

---

## 0. Hard rules

1. **No push, no freeze.** No `git push`, no `freeze_part`, no `protocol/frozen/`. The author pushes.
2. **The two claims do not move.** Claim-1 and Claim-2 estimands, `GENERIC_COMMITTED`, the
   task-sensitive score, splits, held-out families and thresholds are untouched.
3. **No protocol file is edited.** `src/latent_monitor/**` belongs to the WP1–WP8 lanes
   (`RUNBOOK_REFERENCE_UPGRADES_2026-10-09.md` §2.1). In particular `protocol/labels.py` keeps its
   `ORIGIN` tuple; a new origin label is an author decision (D-M below), not an edit made here.
4. **The development smoke must reproduce the 9 Oct anchor exactly**
   (`scripts/compare_smoke.py` against `results/latent_monitor_smoke_dev_2026-10-09_baseline/`).
5. **No LUCiD import, no LUCiD-dependent test** (gate A0 is still open). Everything in WP-B runs on
   numpy arrays in LUCiD's output convention.
6. **No torch, no data download in WP-C tests.** TIDMAD files are not in the repository; the
   tests use synthetic arrays and a synthetic HDF5 file.
7. **Backward compatibility.** Existing public names, signatures and the existing
   `interventions.INTERVENTIONS` registry are unchanged; new families are added beside them.
8. **Decisions that belong to the author or to Junjie are drafted, never taken** (§2.2).
9. **Every number in a report comes from a command that was run.**

---

## 1. What the comment changes, in one table

| comment point | verdict | action here |
|---|---|---|
| Electronics noise is a weak N for an optical arm: thresholding removes it, harm ≈ 0 (Claim 2 empty) and a generic monitor flags what is visible (Claim 1 null) | **accepted** | WP-A: freeze the electronics part of `noise_module_lucid`; it is not a claim-bearing family set |
| Module loss, gain drift, timing drift still harm optical reconstruction | **accepted** | WP-B: these become the LUCiD N families |
| The N–S question has content where the signal sits at/below the noise floor (TIDMAD, DELight) | **accepted** | WP-C (TIDMAD); see corrections below |
| "TIDMAD uses only real recorded noise" | **correct** | WP-C adds controlled N on the 208 noise-only files |
| "DELight adds none" | **half right** (corrected in review round 1) | no DELight simulation chain exists, but a sub-noise TES chain with noise does: HeST → `qp_simulator` → `noise_module` with the HeRALD TES placeholder budget (`src/herald_simulation/`), demoted 18 Sep because it stops at quasiparticle arrival times; `noise_module.budgets.al2o3_athermal` (DELight-like) is not wired into any chain. Revival is conditional (§2.1 R-6) |
| (not mentioned) the sub-noise regime is already in the design | **addition** | Tier-1 ORACLE-Cov (primary, built) and the planned transfer arm B NuRadioMC (Askaryan pulses near the thermal floor, coherent Galactic noise, reconstruction on noisy traces; surveyed, pilot not run) |
| Optical N → detector response: QE, angular acceptance, gain and timing calibration | **accepted with a taxonomy split** | WP-B; angular acceptance is mis-modelling, not N (R-2) |
| Medium mis-modelling (scattering, absorption): N or its own category? | **own category, excluded from both claims** | WP-A records it; D-M asks whether it gets an origin label |
| Inject per-PMT QE variance / an efficiency factor | **split** | a *static* spread is part of the detector definition (reference); a *change* after training is N — and efficiency (photon thinning) is kept separate from gain (charge scale) |
| Different PMT angular-acceptance model | **not N** | listed in `response.EXCLUDED` with the reason; not implemented |
| Time-walk in the front-end trigger | **accepted, best of the three** | WP-B derives it from the SPE template by threshold crossing; a threshold change gives a charge-correlated timing bias that feeds vertex reconstruction |
| Train on baseline LUCiD and on each implementation; a few RNG seeds per category | **seeds accepted; per-variant retraining rejected as the claim design** | WP-A (R-4): train on baseline only, seeds ≥ 3 (target 5); variants are test-time interventions on the frozen model. Retraining per variant measures training-distribution dependence (the C0 / Paper 1 question) and is at most a supporting row |
| Prioritise TIDMAD and DELight; consider LIGO | **TIDMAD yes; DELight conditional; LIGO future work** | R-5, R-6 |
| Consider a LArTPC arm (JAXTPC) | **replacement candidate, not an addition** | WP-D: a transfer-arm decision for Junjie (JAXTPC is from DeepLearnPhysics); if chosen it replaces NuRadioMC |

---

## 2. Decisions

### 2.1 Adopted for implementation (12 Oct; author confirms on review)

- **R-1 — LUCiD electronics noise frozen.** `noise_module_lucid` 0.3 electronics families stay in
  the package (they are correct and tested) but enter no claim cell. No further investment.
- **R-2 — N for a detector response is a change relative to the training-time calibration.**
  N = a change of the acquisition/response chain after the model was trained (efficiency loss,
  gain drift, timing-calibration drift, discriminator threshold change, module loss). A static
  per-PMT spread present at training time is part of the detector definition (the reference cell).
  **Model versus condition:** a different *simulation model* of the angular acceptance or of the
  medium (scattering, absorption) is a separate category **M**, excluded from both claims and from
  fitting and calibration; a *physical condition* that changes after training (water-transparency
  drift, acceptance change from concentrator ageing or a field change) is N (not implemented here).
  The front-end electronics families are N by this rule; R-1 sets them aside as weak, not as M.
- **R-3 — LUCiD N families** (WP-B): `efficiency_loss` (binomial photon thinning), `gain_drift`
  (charge scale, kept separate from efficiency), `timing_drift` (per-channel offsets),
  `threshold_walk` (time-walk from a discriminator threshold change), `module_loss` (a crate/board
  dead). Origin `N_struct` in the existing taxonomy.
- **R-4 — Training protocol for the realism arms.** One training distribution (the baseline
  reference), model seeds ≥ 3 (target 5) — the seeds are the outermost resampling level already
  declared in `PREREGISTRATION.md` §1. Interventions are applied at test time to the frozen
  models. Per-variant retraining is an optional supporting row, never claim-bearing.
- **R-5 — TIDMAD gets controlled N** on its noise-only records (WP-C), so the arm has declared N
  families instead of only the unchosen real noise. The two-Σ̂ contrast (WP10) is unchanged.
- **R-6 — No new arm is added.** LArTPC/JAXTPC is a replacement candidate for the transfer arm
  (Junjie); LIGO stays future work; the HeST/DELight trace stage is revived only if Phase B shows
  that a second sub-noise arm is needed.

### 2.2 Drafted, not taken (recorded in `PREREGISTRATION.md` §8 as open items)

- **D-M** — does category M get its own origin label in `protocol/labels.py`, or is it simply kept
  out of every cell? (Author; touches lane-A territory, so after WP1–WP3.)
- **D-T** — the transfer arm: NuRadioMC (provisional) or JAXTPC (replacement) — and its κ_m source.
  (Junjie.)
- **D-L** — does LUCiD stay as a null-expected control arm at all, given A0? (Author.)
- **D-S** — TIDMAD S families (clean rare signals: injected sinusoids at frequencies sparse in
  training) — needed before TIDMAD can carry Claim 1; design only here. (Author.)

---

## 3. Work packages

### WP-A — Scope revision in the design documents

Files: `docs/EXPERIMENT_DESIGN.md` (a dated "12 Oct re-scope" block in §II.1 status, §II.3.4,
§II.5, §II.7 header, §III.2.4), `docs/TESTBEDS.md` (status note), `docs/TODO.md` (new block),
`TODO.md` (LUCiD section re-prioritised), `docs/PREREGISTRATION.md` §8 (items 13–16 as open,
plus R-4 recorded as a declared training rule *draft*), `docs/INDEX.md`.

Rules: add dated notes; do not rewrite history; do not change any existing decision text except
to append "superseded 12 Oct" pointers where R-1…R-6 supersede it.

**Accept when:** every R-item and D-item appears exactly once in its home document; no claim text
changed (diff check on §1 of `PREREGISTRATION.md`).

### WP-B — LUCiD detector-response N families

New `src/noise_module_lucid/response.py` (MIT, no LUCiD import):

1. `efficiency_loss_hits(pmt_ids, survival, rng)` → keep-mask (exact binomial thinning per photon);
   `efficiency_loss_counts(pe_counts, survival, rng)` (integer pe waveform, exact);
   `scaled_qe(per_pmt_qe, survival)` (the LUCiD-config route, for after A0). A float
   (charge-smeared) waveform is refused: thinning must happen before charge smearing.
2. `gain_drift_hits(charges, pmt_ids, gain)` and `gain_drift_pe(pe_waveform, gain)` — charge scale
   only; hit count unchanged (the signature that separates it from efficiency loss).
3. `timing_drift_hits(times, pmt_ids, offsets_ns)`; `timing_drift_to_mv(pe_waveform, offsets_ns)`
   (renders the mV trace with a fractionally delayed SPE template — the sub-sample form);
   `timing_drift_trace(trace, offsets_s, fs)` for **integer-sample** offsets only (an FFT phase ramp
   on a sharp SPE pulse rings; corrected in review round 1).
4. `leading_edge_time(amplitude_pe, threshold_mv, template)` (the walk curve from the SPE template,
   linear interpolation, NaN below threshold), `leading_edge_times(trace, threshold_mv, fs)`
   (first crossing per channel on a mV trace), and `threshold_walk_hits(times, charges,
   threshold_ref_mv, threshold_new_mv)` → shifted times plus a lost-hit mask.
5. `module_loss_trace(trace, dead_channels)` / `module_loss_hits(pmt_ids, dead_channels)` and
   `dead_channels_from_groups(groups, dead_group_ids)`.
6. `RESPONSE_FAMILIES` registry (name → origin, kind, description, default severity grid) and
   `EXCLUDED` (static QE spread → reference; angular acceptance, scattering, absorption → M), with
   `describe_response()`.
7. Tests `tests/test_response.py`: binomial mean/variance of thinning; gain changes charge not
   count; timing offsets exact for integer shifts; walk monotone decreasing in charge and
   increasing in threshold, zero shift when thresholds equal, sub-threshold hits flagged lost;
   module loss zeroes exactly the dead channels; registry entries all `N_struct`; refusal paths.
8. Package `__init__` re-exports; version 0.3.0 → 0.4.0; `README.md` and
   `docs/noise_module_lucid.md` gain a short "detector-response N (12 Oct)" section.

**Accept when:** `pytest src/noise_module_lucid` passes; the existing 40 tests unchanged.

### WP-C — TIDMAD controlled N injection

New `src/tidmad_transformer/n_injection.py` (numpy + `noise_module`; h5py only inside the file
reader; no torch):

1. Work on the **float** readout (after the release's `int8 + 128` shift) of a noise-only record.
2. Families (origin in brackets): `added_coloured_noise` (N_cov — additive Gaussian noise with a
   declared PSD from `noise_module`, so Σ changes by a known amount), `band_line` (N_cov — a narrow
   line from the `noise_module` grammar), `gain_change` (N_struct), `baseline_drift` (N_struct,
   slow random-walk/linear drift), `glitches` (origin `unknown` — the undeclared held-out family on every arm; `noise_module.ArtifactInjector`),
   `sampling_jitter` (N_struct, first-order slope form, valid in the analysed band), `dropout`
   (N_struct, dead time at the analogue zero).
3. `requantise_int8(x)` back to the release's integer convention with a saturation count, so an
   injected record goes through the same reader path as a real one; the saturation fraction is
   recorded (the C1 int8-saturation lesson).
4. `NCell` spec (family, severity, seed) and `inject(record, cell, fs)` returning the corrupted
   record and a provenance dict (family, origin, severity, seed, realised added variance, saturation
   fraction, `noise_module` version).
5. `inject_file(path, contract, start, length, cell)` reader helper (uses `data.read_channel_pair`).
6. `DEFAULT_CELLS` (a small severity grid per family; each family's grid straddles a declared
   range and includes one point beyond it, per `PREREGISTRATION.md` §8 item 3 principles — the
   grid is development-only and not registered).
7. Tests `tests/test_n_injection.py`: added variance matches the declared PSD integral within a
   confidence bound; gain exact; dropout exact; drift bounded and slow (low-frequency power);
   glitch count matches the contract; requantise round-trip on integer input and saturation
   counting; determinism by seed; a synthetic-HDF5 `inject_file` round trip. Must run without torch
   (the existing torch-dependent TIDMAD tests are not touched).

**Accept when:** the new test file passes in a torch-free environment.

### WP-D — Arm-portfolio note and reply

1. `docs/NOTE_ARM_PORTFOLIO_2026-10-12.md` — one page for Junjie: what changed (R-1…R-6), the two
   corrections, the D-T question (NuRadioMC versus JAXTPC as *the* transfer arm), the training
   protocol R-4, and what is not being done.
2. Appendix C of this runbook — the reply to the comment, ready to paste.

### WP-E — Verification

1. `pytest` on `src/noise_module`, `src/noise_module_lucid`, `src/latent_monitor`, and the new
   TIDMAD test file. Quote the counts.
2. Smoke regression: run `latent_monitor.protocol.smoke` to a scratch directory and
   `scripts/compare_smoke.py` against the 9 Oct anchor — must be identical (no protocol file
   changed, so any difference is a stop condition).
3. Independent review by a fresh agent that has not seen the implementation conversation: physics
   of each family, taxonomy consistency with `labels.py`, and whether any document now
   contradicts another.

---

## 4. File ownership (no collision with WP1–WP8)

| this runbook writes | WP1–WP8 lanes write |
|---|---|
| `src/noise_module_lucid/**`, `src/tidmad_transformer/n_injection.py`, `src/tidmad_transformer/tests/test_n_injection.py`, the docs listed in WP-A/WP-D | `src/latent_monitor/**`, `docs/DRAFT_TRAINABILITY_GATE_*`, `docs/NOTE_FOR_JUNJIE_2026-10-09.md` |

Shared files touched by both (`docs/TODO.md`, `docs/INDEX.md`, `docs/PREREGISTRATION.md` §8):
this runbook only **appends** dated blocks/items, so a merge is a trivial concatenation.

## 5. Stop conditions — ask the author

- Any smoke value changes.
- A family cannot be implemented without a LUCiD import or without torch.
- A document already states a decision that contradicts R-1…R-6 in a way an appended note cannot
  reconcile.
- The independent review finds a family whose origin label is wrong under R-2.

## 6. Final report

### Review round 1 (12 Oct, independent agent) — findings and resolution

| finding | severity | resolution |
|---|---|---|
| `timing_drift_trace` FFT phase ramp on a sharp SPE pulse: 0.5 ns shift of a 30 pe pulse gave a 4.4 mV precursor and moved the 1 mV edge ~15 ns early | P0 | trace form restricted to integer-sample shifts (non-circular); new `timing_drift_to_mv` renders sub-sample drift from the pe waveform with a delayed template (exact, causal; tested against the analytic pulse) |
| electronics noise listed in `EXCLUDED` as "not N" | P1 | moved to `FROZEN`: N by R-2, set aside as weak (R-1) |
| glitches `N_struct` on TIDMAD but the undeclared held-out family on Tier 1 | P1 | TIDMAD glitches → origin `unknown` |
| first-order jitter claimed valid for any bandwidth | P1 | docstring states the 2πfσ ≪ 1 limit and why it holds in the analysed band |
| M overreach: physical drifts after training are N | P1 | R-2 split into model (M) versus condition (N) everywhere |
| stale arm-A electronics text in §§II.1, II.8, III.5, III.6 | P1 | dated superseded pointers added |
| PREREGISTRATION item 5 LUCiD option vs items 14/15 | P1 | item 14 reconciles it |
| "null-expected control" vs harm-bearing families | P1 | reworded: Claim 2 non-empty; Claim 1 possibly null (generic-visible) |
| TIDMAD: partition with the generic arm's noise-only statistics; `K_rel` undefined on noise-only windows | P1 | §II.5.4 and item 16: declared disjoint partition and signal-on-top required; detection rows only until then |
| DELight / NuRadioMC statements overstated | P1 | corrected in this runbook, TESTBEDS, NOTE, reply |
| gain "hit count unchanged" only at hit level | P1 | stated as an idealisation; fixed-mV discriminator case documented (compose with `threshold_walk_hits`) |
| walk normalised to the analytic peak, trace path to the sampled max | P2 | default `template_fs=FS_L` matches `to_mv`; tested against a `to_mv` trace (≤ 0.15 ns) |
| scalar gain sign unchecked; banker's rounding in `requantise_int8`; weak tests (`hasattr`, `<=`, glitch counts only) | P2 | fixed |
| dropout fills at 128 (a step); `inject_file` bypasses `read_channel_pair` | P2 | kept deliberately, documented (dead time reads analogue zero; noise-only files have no reference channel) |


What was built (files, test counts), the smoke comparison output, review findings and their
resolution, the D-items still open, and the commit list.

---

## Appendix A — the comment (verbatim)

> Effort is currently spread across several arms, and much of it goes into noise engineering
> rather than the two claims. A reminder of the goal: Claim 1 asks whether intermediate
> representations separate N (acquisition corruption: module loss, hit thinning, jitter, drift,
> glitches, covariance changes) from S (clean events from sparsely represented regions of the
> training distribution), beyond what a strong generic monitor already does. Claim 2 asks whether
> alarm magnitude tracks the scientific harm. E (deterministic contract faults) is handled by a
> validation gate.
>
> **Why the LUCiD electronics-noise arm is weak for both claims.**
> In optical detectors, electronics noise is small next to the single-photoelectron pulse, and
> hit-finding thresholds remove it before reconstruction. Two consequences follow. (a)
> Reconstruction is barely affected, so harm is close to zero and Claim 2 has nothing to rank.
> (b) Where the noise is visible, simple input- or output-level features already flag it, so
> Claim 1 should come out null against a generic monitor. This applies to electronics noise
> specifically. Module loss, gain drift and timing drift still harm optical reconstruction and
> stay relevant.
>
> **Where the N–S question has real content.** In TIDMAD and DELight the signal sits at or below
> the noise floor, with overlapping spectral and temporal shape, and no threshold step removes
> the noise. Physics-level features can't easily separate N from S there. Yet these are the arms
> with the least N machinery today: TIDMAD uses only real recorded noise, and DELight adds none.
>
> **Proposed re-scoping.**
> - For optical detectors, move N towards detector response (QE, angular acceptance, gain and
>   timing calibration). Decide separately whether medium mis-modelling (scattering, absorption)
>   counts as N or as its own category, since it is not a corrupted measurement.
> - Prioritize controlled N injection on TIDMAD and DELight if the "conventional" noise is higher
>   priority. Also consider a LIGO arm.
> - Consider a LArTPC arm, where coherent electronics noise across wire planes and the induction
>   effect is comparable to the signal and noise filtering is a real reconstruction step. JAXTPC
>   (https://github.com/DeepLearnPhysics/JAXTPC) is a candidate baseline.
>
> - Inject variance on PMT QE (quantum efficiency), currently a constant for all PMTs. Or just an
>   efficiency factor in general to include gain and collection efficiency.
> - Inject a different model of PMT angular acceptance.
> - Inject a time-charge correlation ("time-walk effect") in the front-end trigger. Higher charge
>   hit has signal cross threshold earlier.
>
> Train models with the baseline LUCiD and then with different implementations to observe the
> representation change. Each training category should include a few model with RNGs to gauge
> the stochasticity.

## Appendix B — assessment (12 Oct)

The diagnosis is right: an electronics-noise family in an optical arm gives Claim 2 nothing to
rank and Claim 1 an expected null. Two factual points need qualifying — a sub-noise TES chain with
noise exists (HeRALD-shaped, demoted; not a DELight simulation), and the sub-noise regime the
comment asks for is already the primary substrate (Tier 1, built) and the regime of the planned
transfer arm B (NuRadioMC, pilot not run). Two parts of the prescription work against the
comment's own premise that effort is spread too thin: adding LIGO and LArTPC arms (the
pre-registration shape is one main nonlinear subject, one linear control, one transfer arm), and
retraining one model per LUCiD variant (that is a different study from the frozen-model claims).
The proposed response families need the R-2 split before they are built: static spread is the
reference, a change after training is N, and angular-acceptance or medium differences are M.

## Appendix C — reply to the comment (ready to paste)

> Thanks — I agree with the core diagnosis and we're acting on it.
>
> **Electronics noise on LUCiD.** Agreed: thresholds remove it before reconstruction, so harm is
> ≈ 0 (nothing for Claim 2 to rank) and a generic monitor catches whatever is visible (Claim 1
> null). The electronics part of `noise_module_lucid` is frozen as is and enters no claim cell.
>
> **Optical N moves to detector response**, with one taxonomy rule: N is a change of the
> response *after* the model was trained. These families do harm reconstruction, so Claim 2 has
> something to rank; Claim 1 may still be null where they are visible to a generic monitor (gain via
> the charge histogram, module loss via hit counts) — LUCiD would then be a useful null control. Concretely, five families: efficiency loss (binomial
> photon thinning), gain drift (charge scale — kept separate from efficiency because the
> signatures differ: thinning changes hit counts and Poisson statistics, gain only the charge
> scale), per-channel timing drift, time-walk from a discriminator-threshold change (derived from
> the SPE template by threshold crossing — this one has real harm, since timing feeds the vertex),
> and module loss. A *static* per-PMT QE spread present at training time is part of the detector
> definition, not N. A different *simulation model* of the angular acceptance or of the medium
> (scattering/absorption) is mis-modelling — a separate category, kept out of both claims — while a
> physical drift after training (water transparency, concentrator ageing) would be N.
>
> **Training.** We keep the frozen-model design: train on the baseline only, with ≥ 3 (target 5)
> seeds, and apply the variants at test time. Retraining one model per variant asks how the
> training distribution shapes the representation — interesting, but a different question from
> the two claims; at most a supporting row.
>
> **Sub-noise arms.** Two qualifications: there is no DELight simulation chain, but a sub-noise TES
> chain with noise does exist (HeST → QP simulator → `noise_module` with the HeRALD TES budget,
> demoted on 18 Sep because it stops at quasiparticle arrival times), and the regime you describe
> is already the design's primary substrate (Tier 1, controlled covariance injection) and the
> regime of the planned transfer arm, NuRadioMC (Askaryan pulses near the thermal floor, coherent
> Galactic noise, reconstruction on noisy traces; pilot not yet run). TIDMAD is a fair point — we've
> added controlled N injection on its 208 noise-only files; S families and a signal-on-top for the
> consequence variable come next.
>
> **New arms.** Adding LIGO and LArTPC would spread effort further, which is the opposite of your
> first point. LArTPC is physically the most convincing (coherent-noise filtering is a real
> reconstruction step), so I'd treat JAXTPC as a *replacement* candidate for the transfer arm, not
> an addition — that's a decision for us together. LIGO stays future work; DELight is revived
> only if Phase B shows we need a second sub-noise arm.
