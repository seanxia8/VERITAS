# Claims Ledger v2 — Round 2 (VERITAS / LUCiD multi-detector), 22 Nov 2026

Supersedes `CLAIMS_LEDGER_2026-11-21.md` (v1, kept as history). v2 folds in three things:
1. the blind recomputation `r2-verify` (analytic-world numbers; `results/round2/20261121_verify/VERIFY_REPORT.md`, seal `b755b23`);
2. the LUCiD regeneration `r2-lucid` (`results/round2/20261121_lucid/LUCID_REGEN_REPORT.md`, branch `lucid-mc`);
3. the wording corrections from the steering review of 22 Nov (`review-response-paper-steering-2026-11-22.md`, §6.1).

## 0. Rules, worlds and status levels

**Rules (unchanged).** Only **V** claims may appear in an abstract or title. **R** claims may appear in the body with caveats. **S** and **P** claims appear only as labelled observations or hypotheses. **X** claims must not be made.

**Two worlds.** Every claim now carries one status per world.
- **A — analytic world.** Our own generator (`lucid_bombs` / `r2_generator`): straight-line prompt light, exp(−d/λ), no scattering, fixed-σ Gaussian first-hit time. It has an exact likelihood and CRLB. Never call it "LUCiD simulation".
- **L — LUCiD world.** LUCiD `e651e836`, sampled Monte Carlo (tier A1) with LUCiD's digitiser, 120,000 events per detector.

| Level | Meaning |
|---|---|
| **V** | Recomputed blind by an independent implementation within a pre-registered tolerance, with every attached invariant passing. |
| **R** | Holds across ≥ 2 seeds or ≥ 2 detectors, or reproduced by an internal re-run. One code path. |
| **S** | One run or one seed, or one code path with no replication. |
| **P** | Hypothesis with indirect evidence, or a result with an unresolved confound. |
| **X** | Contradicted or withdrawn. |

**Current count.** A-world: **V = 1** (C21, descriptive). L-world: **V = 0**; no LUCiD number has been blind-recomputed.

**Defaults for all numbers below.** Mean 3-D vertex error on photon-bomb probe-test events, seed 9000 unless stated. In the L-world, SK / JUNO / HK trained arms have **one seed**; WCTE has two; random init has two everywhere.

## 0.1 Definitions that must appear in any paper

- **Spatial-moment readout** (the primary readout). Ridge on 6 × 128 features built from hit tokens: hit mean; charge-weighted mean (w ∝ q³); early-time-weighted mean (w ∝ q³·e^(−4Δt/T) over prompt hits); cross-moments with positions centred on the per-event hit centroid. An independent, simpler spatial readout gives errors 1.3–2.6× larger (`r2-verify` §4.1), so the definition must be stated exactly.
- **"Plain MAE" is not a standard MAE.** Arm `mae_mse` uses the `within` split: a **fixed parity mask** (even PMT slots visible, odd slots hidden, the same in every event). Per-event random masking has never been run. `mae_repaired` uses per-event Hilbert spatial patches (`spatial_patch`). The "repair" changes **four** things at once: fixed-parity → per-event patches; the time-loss unit (window scale → 50 ns); the loader (run-contiguous → shuffled within batch); and the encoder attention window (64 → 64 / 384 / 320 / 640 on WCTE / SK / JUNO / HK).
- **The committed "init" is not config-matched on the large detectors.** It is the default encoder (window 64). Repaired arms on SK / JUNO / HK use larger windows, so their "gain over init" compares two encoder configurations. The matched control is each run's step-0 snapshot; it has not been scored yet (Tier-1 campaign, stage T3). Until then, every "relative to init" number for a repaired arm on SK / JUNO / HK carries this caveat (C12–C15, C24, C26, C27).
- **Time-loss unit.** `--time-loss-scale-ns` changes only the weight on the time term of the loss (`time_loss_weight`). The input time feature keeps the detector's window scale. It exists for `mae_repaired` only.
- **Token time** is the digitised first-hit time minus the NHits trigger time (trigger-relative), inside the prompt window. Absolute and trigger times are stored, so a median-referenced variant needs re-tokenising only.
- **Collapse metric.** f_event (event-mean variance fraction). Do not use ratios of the across-event *fraction* (`r2-verify` §4.3).

## 0.2 Known errors in committed artefacts

| Artefact | Error | Fix |
|---|---|---|
| `TOY_VS_LUCID.md` / `.json` (`lucid-mc`) | The WCTE `mae_repaired` seed-9000 row (vertex and particle tables) is read from the **5 ns sweep run**. `lucid_rows()` keys runs by (detector, arm, seed), so sweep runs overwrite the default run. Correct default values: spatial Ridge 0.0521 m, MLP 0.0465 m, all-mean 0.141 m, f_event 0.185. | Skip `__ts*` runs in the builder (or key by time unit) and rebuild. Not yet done. |
| Same file, ToF line | The Huber ToF fit starts at the detector centre, uses every hit in the window and has no late-light or dark-hit model. It is **not** an information measure (see C10, X8). | Replace with a likelihood yardstick (planned E5a). |

---

## A. Evaluation and readout

### C1 — All-token mean pooling understates what the representations contain
- **Claim.** Ridge on the mean over all PMT tokens gives a much larger vertex error than the spatial-moment readout on the same frozen encoder, for every arm including random init.
- **A-world.** 5–10× on SK / JUNO / HK (e.g. SK MAE 0.93 → 0.13 m). The all-mean values reproduce blind (30/30). The committed spatial-moment values reproduce only with the committed feature definition (11/11, after unblinding); invariant I1 (≥ 3× with a simpler independent readout) fails in 3 of 21 cases. **Status R**, with the readout definition stated.
- **L-world.** Ratio all-mean ÷ spatial: 1.7–8.1×. Init 3.0 / 7.0 / 6.3 / 8.1× (WCTE / SK / JUNO / HK); repaired MAE 2.7 / 2.4 / 2.6 / 2.5×; repaired JEPA 3.1 / 5.5 / 5.2 / 6.8×; fixed-label 1.7 / 4.2×. The report's "3–8×" is wrong for the repaired MAE and for fixed-label on WCTE. **Status R** (4 detectors), not blind-recomputed.
- **Upgrade.** Blind recomputation on LUCiD; attentive and MLP probes (probe robustness, see C28).

### C2 — Error grows with detector size for every arm; under all-mean pooling this is a readout effect
- **A-world.** HK ÷ SK under all-mean pooling is **1.39–1.89 over both seeds** (v1 quoted seed 9000 only, 1.63–1.89). Radius ratio 1.91. Invariant I4 (within ±20 % of 1.91) fails for 2 of 7 arm-seeds. **Status R** with the corrected range.
- **L-world.** HK ÷ SK under all-mean: init 2.00 / 2.04, repaired MAE 1.88, repaired JEPA 2.25. Under the spatial readout the ratio is also 1.72–1.81, and the normalised repaired-MAE error is 2.6 / 2.1 / 2.4 % of R_fid (SK / JUNO / HK), against 0.9 / 0.7 / 0.7 % in the A-world. **Status S.**
- **Consequence (unchanged).** SK→HK degradation cannot test hypotheses about the encoders.

### C3 — Arm ordering under the spatial-moment readout
- **A-world.** MAE < init on 4 detectors × 2 seeds; fixed-label JEPA worst on 4 detectors. Confirmed by the independent readout (I2, I3 pass). init vs position-free is not robust. **Status R.**
- **L-world.** Repaired MAE < plain MAE < repaired JEPA < init ≪ fixed-label wherever trained. WCTE: both seeds (**R**). SK: one seed (**S**). JUNO / HK: repaired arms and init only (**S**).
- **Caveat.** "MAE" here is the fixed-parity-mask MAE (§0.1). The ordering between the two repaired arms is probe-dependent (C28).

### C19 — Random initialisation is a strong baseline
- **A-world.** Init beats the raw-feature spatial baseline by 43–69 % on SK / JUNO / HK. I6 passes with an independent raw baseline. **Status R.**
- **L-world.** Init beats raw features on all four detectors: 0.066 vs 0.075, 0.527 vs 0.667, 0.423 vs 0.538, 0.907 vs 1.188 m (13 / 21 / 21 / 24 %). **Status R** (two init seeds). The margin is smaller than in the A-world.
- **Consequence (unchanged).** Every pretraining claim is stated as a gain over init.

### C22 — Retraining reproduces frozen checkpoints
- A-world, unchanged: **R.**

### C28 — Conclusions depend on the probe (new)
- **Observation (L-world).** On JUNO the MLP readout ranks repaired JEPA (0.336 m) ahead of repaired MAE (0.367 m); Ridge ranks them the other way (0.379 vs 0.288 m). The WCTE time-unit sweep is monotone under Ridge and all-mean, not under MLP (C25). The MLP probe is worse than Ridge for init on SK (0.62 vs 0.53 m), so it is not simply a stronger probe.
- **Status S.** No claim about the ordering of the two repaired arms, and no dose-response claim, without MLP (≥ 3 probe seeds) and attentive probes.

---

## B. Simulation, regime and yardsticks

### C8 — The analytic world is internally self-consistent
- **A-world.** Joint ML fits reach 2.4 / 1.5 / 1.6 / 1.2 cm against CRLBs of 1.9 / 1.4 / 1.5 / 1.1 cm. Independent ToF fit agrees on 4/4 detectors (within 8 %); independent timing-only CRLB agrees on SK / JUNO / HK (within 3 %) and differs on WCTE by design (timing-only vs joint). **Status R.**
- **L-world.** Not applicable. Validation of the LUCiD data is the six gates in `VALIDATION.md` (all pass); that is a statement about our pipeline, not a physics-fidelity claim.

### C9 — Gap between the best frozen readout and the information bound
- **A-world.** 8–16× on SK / JUNO / HK (committed readout); 12.6–28.8× with the independent simple readout against an independent ToF fit. **Status R.**
- **L-world.** **Not testable.** No likelihood yardstick exists yet. Must not be stated for LUCiD data.

### C10 — Frozen readouts sit between charge-only and timing-limited information
- **A-world.** **S** (unchanged).
- **L-world.** **Not testable.** "Timing is largely unused" must not be claimed on LUCiD. The crude ToF fit (medians 0.139 / 0.217 / 0.271 / 0.277 m) is a poor estimator, not a bound. Inference, not yet measured: with 1-PE jitter of 0.95–3.5 ns over ~10³ hits, a proper timing fit should reach centimetres. The validation gate G2 finds timing-residual peaks within ±0.9 ns under the n = 1.33 straight-line reference, so the index is probably not the problem.

### C17 — HK Gate V
- **Closed / obsolete.** Gate V compared LUCiD transport with the analytic generator, which is no longer the data source.

### C21 — Regime
- **A-world.** Bright, high occupancy: HK cohort median 2.6 × 10⁴ PE and 9.6k hit PMTs (49 %); the v1 numbers (11k–17k PMTs) described displays between the median and the 95 % quantile. **Status V (descriptive).**
- **L-world.** Deliberately dim: photon-bomb median 2.3–2.5k detected PE on every detector (q5 ≈ 200, q95 ≈ 3 × 10⁴). Late light 1.5 % (WCTE), 12–21 % (SK / JUNO / HK). **Status S** (from manifests).

### C18 — Worst events
- **A-world.** All-mean pooling shrinks outer-quartile vertices inward by 0.5–1.0 m; the spatial readout is unbiased within ±0.08 m per quartile (`r2-verify`, full cohort). **Status R.**
- **L-world.** The worst events are the dimmest (110–200 hits) on all four detectors, for both repaired arms. **Status S.**

---

## C. JEPA failure and repair

### C4 — Fixed-label JEPA collapses to static layout
- **A-world.** f_event 7.8 × 10⁻⁵ / 2.4 × 10⁻⁴ / 2.1 × 10⁻³ / 3.2 × 10⁻⁴ (WCTE / SK / JUNO / HK); static-layout share 0.997 / 0.986 / 0.895 / 0.970. Both reproduce blind (16/16; f_event within 4–10 %). f_event(MAE) ÷ f_event(fixed-label) ≈ 140 on SK. Invariant I7 (≥ 50× in the across-event *fraction*) fails because MAE tokens are themselves ≈ 90 % static; that wording is withdrawn (X11). **Status R.**
- **L-world.** f_event 0.0002 (WCTE), 0.0001 (SK); f_layout 0.993 / 0.989; spatial error 0.153 / 1.151 m, worse than raw features. **Status R** (2 detectors, one seed each).

### C5 — Mechanism: static-geometry shortcut
- **S**, unchanged. The ablations that separate it from C6 (EMA only vs batch fix only) have not been run.

### C6 — SIGReg was computed per micro-batch on run-contiguous batches
- **Claim (corrected).** The Epps–Pulley statistic scales with N (1.61 at N = 4, 12.87 at N = 32). With micro-batch 4 the effective regularisation **magnitude** was ≈ 1/8 of the intended value.
- **Evidence.** N-scaling reproduced blind to 5 × 10⁻⁶; the independent statistic matches official LeJEPA to 2.5 × 10⁻⁸ (I9, I11 pass). I10 fails: on the collapsed embeddings the gradient *direction* is nearly unchanged (cosine 0.988), so any wording about direction is withdrawn (X10).
- **Status R** (code-level; world-independent).

### C7 — Share of the JEPA failure explained by the batch bug
- **P**, unchanged.

### C13 — The repaired JEPA does not collapse
- **A-world.** f_event 0.096 / 0.123 / 0.132 / 0.085. **R.**
- **L-world.** f_event 0.171 / 0.125 / 0.129 / 0.066 (WCTE seed 9001: 0.153). **R.**
- **Caveat.** The arm carries MAE-like auxiliary heads (masked charge, time, occupancy).

### C14 — The repaired JEPA beats init on WCTE
- **A-world.** 0.0465 vs 0.0525 m, one seed. **P.**
- **L-world.** 0.0629 vs 0.0656 m (−4 %) and 0.0612 vs 0.0662 m (−8 %), two seeds. **S** (small effect; auxiliary heads; probe-dependent size).

### C15 — Repaired JEPA against init on the large detectors (restated; world-dependent)
- **A-world.** At init level: +3 % / +21 % / −2 % (SK / JUNO / HK). **S.**
- **L-world.** Better than init: 0.449 vs 0.527, 0.379 vs 0.423, 0.812 vs 0.907 m (−15 % / −10 % / −10 %). **S** (one seed; HK JEPA at window 128).
- The v1 wording ("at init level on large detectors") is false as a general statement.

### C16 — JEPA keeps global quantities, MAE spatial ones
- **P.** See C27 for the L-world energy numbers. Auxiliary heads confound any JEPA-specific reading.

### C20 — Iteration-0 identity
- **P / S**, unchanged.

---

## D. MAE behaviour, loss weighting and tasks

### C11 — The original MAE's time term was a negligible share of the loss
- **A-world.** 0.04 % (SK, JUNO), 0.08 % (HK). **R.**
- **L-world.** 0.45 % (WCTE), 2.0 % (SK) for the plain MAE; 61–90 % for the repaired MAE on the large detectors. **S** (from logs).
- **Caveat.** Loss share is not gradient share.

### C12 — The repaired MAE improves on the plain MAE
- **A-world.** Spatial 0.128 → 0.119, 0.096 → 0.091, 0.186 → 0.167 m; all-mean 0.93 → 0.39 m (SK). **S.**
- **L-world.** Spatial: WCTE 0.0575 → 0.0521 m (−9 %; seed 9001 −8 %), SK 0.427 → 0.356 m (−17 %). All-mean on SK: 2.06 → 0.84 m. **R** on WCTE, **S** on SK.
- **Attribution: P.** The bundle changes the mask (fixed parity → per-event patches), the time unit, the loader and, on SK / JUNO / HK, the encoder attention window (§0.1). On WCTE, patch masking at a 450 ns unit already gives 0.0541 m, i.e. about two thirds of the gain comes without tightening the time unit.

### C24 — Relative gains over init across worlds (new)
- **Observation.** Repaired MAE gains −22 / −35 / −26 / −36 % in the A-world and −21 / −32 / −32 / −31 % in the L-world (WCTE / SK / JUNO / HK). Plain MAE gains shrink: −24 / −30 % (A) to −12 / −19 % (L) on WCTE / SK. Absolute gains in metres are larger on LUCiD because all errors are 1.25–3.7× larger (fixed-label on WCTE: 1.09×).
- **Status S.** "Gains grow under realism" is false in relative terms for both MAE arms (X7).

### C25 — Time-unit dose-response on WCTE (new)
- **Observation (L-world, repaired MAE, seed 9000, patch masking fixed).** Time unit 450 / 50 / 15 / 5 / 1.5 ns → spatial Ridge 0.0541 / 0.0521 / 0.0512 / 0.0501 / 0.0485 m; all-mean 0.143 / 0.141 / 0.134 / 0.122 / 0.118 m; MLP 0.047 / 0.047 / 0.053 / 0.051 / 0.045 m (not monotone); f_event 0.198 → 0.134; token time R² 0.957 → 0.986.
- **Status S.** One seed, one detector, 10 % over a 300-fold change of unit; step sizes are comparable to the WCTE seed spread.

### C26 — Direction reconstruction (new)
- **Observation (L-world, median angle, μ / e).** SK: init 18.6° / 11.2°, plain MAE 9.1° / 7.4°, repaired MAE 5.6° / 4.7°, repaired JEPA 9.5° / 7.8°, fixed-label 18.3° / 16.5°. HK: init 16.2° / 13.7°, repaired MAE 5.6° / 4.6°, repaired JEPA 9.3° / 7.3°. JUNO: init 9.9° / 8.5°, repaired MAE 6.0° / 5.7°, repaired JEPA 6.1° / 5.7°.
- **Status S** (one seed; SIREN surrogate particles).

### C27 — Energy moves the other way (new)
- **Observation (L-world, residual variance 1 − R²).** Electron energy on SK: init 0.006, plain MAE 0.004, repaired MAE 0.014, repaired JEPA 0.007. On HK: init 0.006, repaired MAE 0.018, repaired JEPA 0.007. Muon energy on SK: 0.016 / 0.010 / 0.019 / 0.022; on HK: init 0.014, repaired MAE 0.023, repaired JEPA 0.019. JUNO differences are within rounding.
- **Reading.** The time-dominated repaired MAE is worse than random init on energy while it is best on vertex and direction; the charge-dominated plain MAE is best on energy.
- **Status P.** One seed, R² rounded to three decimals, bundle-confounded. Needs re-scoring as σ_E/E on a sealed split. μ/e AUC (≥ 0.98 for all non-collapsed arms) also needs re-scoring as mis-identification at fixed efficiency. "Energy and μ/e are saturated" is withdrawn (X9).

---

## E. Round-1 and Paper-3 claims affected

- **C-P3** (Paper 3 vertex costs read with the all-mean probe): needs re-read. Unchanged.
- **C-R1** (trigger-time jitter is the operative shared nuisance on WCTE): **S**, A-world, not re-audited. Relevant again because L-world token time is trigger-relative (§0.1).
- **C-P3b**: open. Unchanged.

---

## F. Retracted (X)

| ID | Statement | Why |
|---|---|---|
| X1 | "No view, label, regulariser or de-jittering change repairs the collapse" | EMA target plus the batch fix removes it (C13). |
| X2 | "Position-free token log-charge R² of 0.94–0.99 shows the JEPA branch has no bug" | Random init reaches 0.999; a real batch bug exists (C6). |
| X3 | "HK Gate V 2.07 indicates an error in the HK analytic model" | Validation-condition mismatch (C17, now obsolete). |
| X4 | "JEPA also degrades from SK to HK, so 'problem too easy' is falsified" | Readout artefact common to all arms (C2). |
| X5 | "The `within` split changes the target set per event" | It is a fixed parity split. |
| X6 | "MAE < init < position-free < fixed-label on all four detectors" | init vs position-free is not robust (C3). |
| X7 | "Pretraining gains over init are larger on LUCiD than in the toy" (as a relative statement) | True in metres only; relative gains are equal or smaller for both MAE arms (C24). |
| X8 | "Timing carries far less information on LUCiD", "the timing advantage collapses under realism" | Inferred from a crude ToF fit that is not an information measure (C10). |
| X9 | "μ/e and energy are saturated" | The metrics are; residual variances differ 2–5× between arms (C27). |
| X10 | "The SIGReg bug changed the gradient direction" | Magnitude ≈ 1/8; cosine 0.988 on collapsed embeddings (C6). |
| X11 | "Fixed-label across-event fraction is ≥ 50× below MAE" | Wrong metric; use f_event (C4). |
| X12 | "The simulation is LUCiD" for any round-1 or round-2 number before 21 Nov | Analytic generator on LUCiD geometries (§0). |

## G. Wording that is forbidden until new evidence exists

- "Information ceiling", "CRLB" or "timing largely unused" for LUCiD data (until a likelihood yardstick exists).
- Any attribution inside either repair bundle ("the loss scale explains the repaired MAE", "EMA fixes the collapse").
- "The likelihood-matched loss is optimal."
- "JEPA fails on detector data."
- Performance statements about SK, HK or JUNO as experiments: these are LUCiD geometries; "JUNO" is a water sphere; WCTE uses the `hk` digitiser preset; particles are SIREN surrogates.
- "First self-supervised study on water-Cherenkov data"; "foundation model".
- "Loss share" used as if it were gradient share.

## H. What would move claims to V

1. Blind recomputation of the LUCiD tables (C1, C3, C4, C12, C13, C19 in the L-world).
2. A sealed probe-test split (≥ 10k events per detector) untouched by any readout design decision, with paired bootstrap.
3. Seeds: ≥ 2 on SK for every arm used in a headline.
4. Un-bundling runs (mask × time unit at fixed loader) for C12, C25, C27; JEPA ablations for C5, C7, C14–C16.
5. A likelihood yardstick on LUCiD (counts-only, time-only, joint) for C9, C10.

Pre-registered falsification and kill criteria for the next campaign are recorded in that campaign's runbook, not here.
