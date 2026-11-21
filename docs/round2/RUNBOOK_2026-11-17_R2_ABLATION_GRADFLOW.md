# Addendum v2 — Round 2 Bug-vs-Design Attribution, Gradient Flow and the Large-Detector JEPA Gap

Version 2, 20 Nov 2026 (v1 of 17 Nov was never launched; v2 replaces it). Status: authorised for unattended execution; `r2-repair` is done (17 Nov 20:36 UTC).
Extension: `r2_ablate_2026-11-17`, `registered: false`.
Parent: `docs/round2/RUNBOOK_2026-11-17_R2_REPAIR.md`. All of its §0.1 safety constraints hold unchanged. Inputs: `results/round2/20261117_repair/` (report, probes, retrain readings, snapshots).
Branch: `dev` in worktree `~/Documents/VERITAS-worktrees/r2` (remote clone `/work/dwong/r2_repo` on `dt1`/`dt2`).

---

## 0. Questions this addendum must answer

The collaborator (Junjie Xia) asked six technical questions. Each is tracked to a stage, and the final report answers each one explicitly with numbers.

| ID | Question | Stage |
|---|---|---|
| Q1 | Do MAE and JEPA start from the same weights at iteration 0? | A1 |
| Q2 | Given the same initial state, where do their gradient flows differ? | A2, A5 |
| Q3 | Is the JEPA failure a bug or a fundamental (design) problem? | A3, A4 |
| Q4 | After repair, why does JEPA beat initialisation on WCTE but stay at initialisation level on SK/JUNO/HK? | A3b, A2-G8 |
| Q5 | Does "SK→HK gets worse" test the "problem too easy for JEPA" hypothesis, and in what physical regime are we? | A4.6 |
| Q6 | What does latent evolution look like when viewed through a lens that can see vertex information? | A2.5 |

### 0.1 Evidence from `r2-repair` that shaped v2

1. **Repaired JEPA reaches initialisation level only.** It no longer collapses (event fraction 0.08–0.13). Spatial-moment vertex error:

   | | WCTE | SK | JUNO | HK |
   |---|---:|---:|---:|---:|
   | repaired JEPA | 0.047 | 0.189 | 0.149 | 0.256 m |
   | init | 0.053 | 0.183 | 0.123 | 0.260 m |
   | repaired MAE | 0.041 | 0.119 | 0.091 | 0.167 m |

   HK JEPA ran at the W128 fallback, while HK MAE ran at W640.
2. **Step-0 function differs between arms.** In the SK UMAP grid, the step-0 panels of `sigreg_jepa` and `target_posfree` are identical to each other. Those of `jepa_repaired` and `mae_repaired` are identical to each other but differ from the first pair. Parameters may be identical while the function differs (window W64 vs W384, or input or feature handling). Q1 therefore needs both a parameter-identity check and a function-identity check.
3. **Random initialisation is not a blob.** At step 0 the event-mean embedding is a one-dimensional arc ordered by log light yield. Input features pass through the residual stream, so `init` already carries substantial information (spatial-moment vertex 0.18 m on SK).
4. **The UMAP lens is blind to vertex.** No arm shows vertex-radius structure at any step, because the UMAP input (event-mean tokens) discards spatial information by construction.
5. **Repaired JEPA barely moves from init.** Its event-space geometry stays close to the step-0 arc for all 10k steps. Its training prediction loss is already low at step 50 (≈0.25, falling to ≈0.05), and its SK spatial vertex improves only from 0.26 to 0.23 m on the 1,000-event evolution cohort. The repaired JEPA appears to still use the parity (`within`) view; only repaired MAE switched to keyed Hilbert patches. Hypothesis: the latent prediction task remains locally solvable, so it carries little global (vertex) signal.
6. **Worst events are probe shrinkage.** The worst HK MAE events are low-yield (≈3–4×10³ PE) with vertices at the fiducial bottom corner. All mean-pool predictions are shrunk 8–12 m towards the centre, while joint ML and the spatial-moment readout land on truth.
7. **Photon bombs are a bright, high-occupancy regime.** Median HK displays have 3×10⁴–1.6×10⁵ PE with 11k–17k of 19,746 PMTs hit. Conclusions apply to this regime unless stratified.

## 0.2 Authority

Quote verbatim in `docs/round2/DECISIONS.md` at Stage A0:
1. **Author, 17 Nov 2026, 10:08 CET:** "现在按照你的建议写addendum".
2. **Author, 20 Nov 2026, 22:45 CET:** "修订我们的计划，junjie的问题很尖锐很有技术性". (Translation: "Revise our plan; Junjie's questions are sharp and technical.")
3. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".
   - Do not stop to ask or pause between stages.
   - At every decision point take your own recommended option and record one line in `DECISIONS.md` and `$CC_STATE/notes/r2-ablate.md`.
   - Continue until every stage is committed and pushed.

## 0.3 Where to write, compute, budget

- **Repository outputs:** `results/round2/20261117_ablate/`.
- **Ceph:** `/ceph/dwong/jepa/returns/r2_ablate_2026-11-17/`.
- **Deepthought scratch:** `/work/dwong/r2_ablate/`.
- **Read only:**
  - `r2-repair` returns and snapshots under `/ceph/dwong/jepa/returns/r2_repair_2026-11-17/`;
  - Round-1 and Round-2 data and checkpoints.

  Never modify, cancel or re-run anything that belongs to `r2-repair`.
- **Compute:**
  - Deepthought CPU: ≤ 20 cores, `nice -n 10`, `TMPDIR=/work/dwong`.
  - Titans and deepthought2 L40S: parent §0.1 guard rules.
  - HTCondor batch on TOpAS A100 or NEMO2 L40S: ≤ 8 concurrent jobs.
- **GPU ledger:** new line `R2_ablate_gpu`, cap **25 GPU-h**. Scope decision of 21 Nov 2026: JEPA attribution is now a supporting/fallback line of the paper, so GPU is reserved for the main-line experiments.
  - Projection: SK runs ≈ 6 GPU-h each, WCTE runs < 1 GPU-h each. HK W640 (≈ 12–16 GPU-h) runs only if budget remains.
  - If the projection exceeds the cap, drop runs from the bottom of §A3.3 and record each drop.

---

## Stage A0 — Preflight

1. Fast-forward the worktree to `origin/dev`.
2. Copy this file from `~/Documents/VERITAS/docs/round2/`, record its SHA256 with the §0.2 quotes, then commit and push.
3. Sync `/work/dwong/r2_repo` on `dt1` and `dt2`.
4. Write `$CC_STATE/status/r2-ablate.json` (`state: running`) and start `$CC_STATE/notes/r2-ablate.md`.
5. Inventory every `r2-repair` snapshot run (WCTE ×6, SK ×4, nine snapshots each) and every final checkpoint (JUNO and HK repaired arms) into `results/round2/20261117_ablate/INPUTS.json`, with SHA256s.
6. Record, per arm, the exact view and masking scheme, window, input feature transform and time scaling, loss terms, SIGReg mode, micro-batch and optimiser groups. Confirm whether `jepa_repaired` uses the `within` parity view.
7. Expose these as independent opt-in flags, with defaults reproducing existing arms exactly:
   - **F1:** shuffled loader + exact full-batch SIGReg;
   - **F2:** EMA stop-gradient teacher;
   - **F3a:** hit-only latent loss + event/hit-token SIGReg;
   - **F3b:** auxiliary observable head and occupancy head. These predict masked raw observables, an MAE-like reconstruction term;
   - **F4:** masking scheme (`within` parity vs keyed Hilbert patches as in `mae_repaired`).

## Stage A1 — Step-0 identity (Q1; CPU)

1. **Parameter identity.** For each detector, compare every shared `jepa.encoder.*` and `jepa.predictor.*` tensor of every seed-9000 `snapshot_step_00000.pt`, pairwise and against a freshly built `build_arm(..., seed=9000)`. Report the maximum absolute difference per tensor group.
2. **Function identity.** On the same 32-event batch, compare encoder outputs (full input and context-masked input) across arms at step 0 (maximum absolute difference, CKA). Explain every non-identical pair by its configuration difference (window, feature transform, view). This must account for two observations:
   - the Stage-5 UMAP observation in §0.1 item 2;
   - on WCTE, where every arm uses W64, the `jepa_repaired` step-0 snapshot already differs from the common initialisation shared by all other arms (f_event 0.0746 vs 0.0787; spatial-moment vertex 0.0661 vs 0.0638 m on the evolution cohort; effective rank 1.58 vs 1.53).
3. **Batch and mask identity.** Compare the first training batch's event IDs and target or masked indices per arm.
4. **Optimiser.** Record parameter groups, learning rates, weight decay and schedule per arm.
5. **Matched-init pairs.** Define the pairs that share both parameters and function at step 0, for example legacy MAE / fixed-label / position-free at W64. These pairs are the only ones used for the step-0 comparisons G4 and G7.

Output: `step0_identity.json`.

## Stage A2 — Gradient flow on existing snapshots (Q2; CPU)

Inputs: all `r2-repair` snapshot runs (WCTE ×6, SK ×4). Use K = 8 fixed batches of 32 dev-train events per detector (seed `20261117`), with the same events for every arm and snapshot.
- **Primary analysis:** each arm uses its own training view and generator.
- **Secondary analysis:** on matched-init pairs at step 0, use an identical common view, so that only the loss differs.

Compute in FP32 on CPU, deterministic, with dropout off.

### A2.1 Quantities

- **G1 — per-layer gradient norms.** For the total loss and for each loss component.
- **G2 — input-column attribution.** ‖∂L/∂W_in[:, c]‖ for the 9 input columns, grouped as geometry (0–5), `ln q` (6), time (7) and hit (8). Also input saliency E|∂L/∂x| over hit and unhit tokens.
- **G3 — label-path vs context-path share** of the encoder gradient, for JEPA arms. Must be exactly 0 for EMA arms (sanity check).
- **G4 — step-0 cross-arm cosine per layer**, matched-init pairs only.
- **G5 — per-layer gradient SNR** across the K batches.
- **G6 — SIGReg against prediction.**
  - cosine and norm ratio of ∇SIGReg vs ∇prediction;
  - SIGReg gradient from 8×4 accumulation vs 1×32.
- **G7 — first optimiser update Δθ** from step 0 on batch 0. Per-layer cosine between matched-init arms, and the column split for `input_proj`.
- **G8 — alignment with vertex information (new).**
  - At each snapshot, fit the spatial-moment Ridge vertex probe on that snapshot's frozen features, using a fixed 1,000-event probe-train set disjoint from the K batches.
  - Then compute ∇θ of the probe's MSE on the K batches, with the probe weights held fixed.
  - Report the per-layer cosine between ∇θ(SSL loss) and ∇θ(probe loss). A positive value means the self-supervised objective locally pushes the encoder towards features from which the vertex is linearly decodable through spatial moments.
  - Report it for every arm and snapshot, and for `init`.

### A2.2 Pre-registered expectations

Report each as confirmed or refuted.
- **H1:** parameters are identical across all seed-9000 arms of a detector. Function is identical only within matched configurations.
- **H2:** in `sigreg_jepa`, the label-path share and the geometry-column share of the `input_proj` gradient grow over the first 500 steps and exceed those of `mae_mse` by step 500.
- **H3:** the time-column share is < 1% of the event columns for `mae_mse` and ≥ 10× larger for `mae_repaired`.
- **H4:** SIGReg gradient from 8×4 accumulation vs 1×32: cosine < 0.9 and norm ratio far from 1.
- **H7:** on SK, the G8 cosine for `mae_repaired` exceeds that of `jepa_repaired` at every snapshot from step 200 onwards.

### A2.5 Latent evolution with a vertex-aware lens (Q6)

For the same snapshots and 1,000 held-out events as `r2-repair`, fit UMAP on three inputs:
- (i) event-mean tokens, as before;
- (ii) hit-masked spatial-moment features [z̄_hit, z̄·x̃, z̄·ỹ, z̄·z̃] after PCA to 50 dimensions;
- (iii) a token-level sample (hit PMTs of 50 events).

Colour by true vertex x and z (not only radius), log yield and wall distance. Alongside the pictures, report a quantitative counterpart per input and step: 10-nearest-neighbour vertex regression error in the UMAP-input space. Pictures are illustrative; the kNN numbers are the evidence.

Outputs: `gradflow/gradflow.json`, plus figures under `gradflow/figures/` and `evolution_spatial/`. Plot one measure per axis, never dual axes.

---

## Stage A3 — Factorial ablation and the large-detector gap (Q3, Q4; GPU)

### A3.1 Arms

Training: 10,000 steps with the cohorts, dataset seeds and micro-batch of the matching `r2-repair` runs (SK micro-batch 4, so that F1-off reproduces the bug). Each run saves nine snapshots.

| ID | F1 batch fix | F2 EMA teacher | F3 target redesign | F4 masking | Window | Status |
|---|:---:|:---:|:---:|---|---|---|
| L0 `sigreg_jepa` | – | – | – | within | legacy | exists |
| A1 `abl_batchfix` | ✓ | – | – | within | legacy | new |
| A2 `abl_ema` | – | ✓ | – | within | legacy | new |
| A3 `abl_batchfix_ema` | ✓ | ✓ | – | within | legacy | new |
| A4 `abl_full_legacywin` | ✓ | ✓ | ✓ | within (as R) | legacy | new |
| RL `jepa_repaired_latent_only` | ✓ | ✓ | F3a only (no F3b heads) | within | repaired | new (pure latent prediction: is the repaired-JEPA gain from latent prediction or from the MAE-like auxiliary heads?) |
| R `jepa_repaired` | ✓ | ✓ | ✓ | within (verify in A0) | repaired | exists |
| P `jepa_repaired_patch` | ✓ | ✓ | ✓ | **keyed Hilbert patches** (as `mae_repaired`) | repaired | new |
| H `jepa_repaired_w640` (HK only) | ✓ | ✓ | ✓ | as R | **W640** | new |

### A3.2 Tests

In `src/jepa_noise/tests/test_r2_ablate.py`:
- (a) the L0 preset reproduces the legacy loss and gradients;
- (b) the R preset reproduces `jepa_repaired`, and the RL preset differs from R only by removing the F3b loss terms;
- (c) F1 gives identical SIGReg for 1×32 and 8×4;
- (d) F2 gives a zero label-path gradient;
- (e) F4 gives exactly the `mae_repaired` patch layout for identical seeds.

Run the full native regression suite before production (baseline 992 pass / 0 fail).

### A3.3 Priority

Drop from the bottom if the cap is threatened. With the 25 GPU-h cap, expect items 1–3 to run; later items run only if budget remains.
1. SK A1.
2. WCTE A1, A2, A3, A4, RL and P (seed 9000), plus RL and R at seed 9001.
2b. SK RL.
3. SK P.
4. SK A3.
5. WCTE seed 9001 for L0, A1, A2, A3, R, P and `mae_repaired` (seed-noise band). The WCTE claim "repaired JEPA beats init" rests on a 6 mm single-seed difference, while frozen position-free JEPA differs by 6 mm between seeds 9000 and 9001. R seed 9001 is therefore required before that claim is used.
6. HK H.
7. SK A2.
8. SK A4.

### A3b — Why repaired JEPA stays at initialisation level (Q4; CPU, on snapshots of R, P, `mae_repaired`, `init`)

- **D1 — prediction difficulty.** Compare the trained predictor's latent loss on masked hit tokens with three baselines:
  - (i) the mean teacher latent of the k = 2, 4, 8 nearest visible context PMTs (pure spatial interpolation);
  - (ii) the step-0 predictor;
  - (iii) the per-PMT dataset mean.
- **D2 — teacher drift.** Linear CKA between teacher tokens at step t and at step 0. Spatial-moment vertex error of the teacher's own tokens vs step.
- **D3 — target information.** Spatial-moment vertex probe using only the teacher-target tokens of masked PMTs.
- **Pre-registered expectations:**
  - **H5:** on SK, k-NN interpolation (best k) attains a latent loss within 20% of the trained predictor's for R.
  - **H6:** P improves SK spatial-moment vertex error over R by more than 2 cm.
- **Verdict rules for Q4:**
  - **"masking too local":** P recovers ≥ 50% of the R → `mae_repaired` gap on SK and H5 holds.
  - **"window":** H recovers ≥ 50% of the HK R → `mae_repaired` gap.
  - **"teacher anchored at init":** D2 CKA stays > 0.9 throughout and neither of the above holds.
  - Otherwise **"objective-level limitation, unresolved"**, with the measured numbers.

## Stage A4 — Evaluation, attribution and regime

1. For every new run, run the standard and repaired probe suites on the full matched cohorts, plus snapshot evolution on the `r2-repair` 1,000-event cohort.
2. **Attribution metric:** spatial-moment Ridge mean 3-D vertex error (m and fraction of R_fid). Always report the full-cohort `init` value next to it, and state which cohort a number comes from. The 1,000-event evolution cohort gives larger errors than the full cohort. Also report:
   - f_event, f_layout and effective rank;
   - all-mean vertex;
   - μ direction;
   - energy R²;
   - μ/e AUC.
3. **Decomposition**, with gap G = e(L0) − e(R):
   - bug share b = (e(L0) − e(A1)) / G;
   - teacher share t = (e(L0) − e(A2)) / G;
   - interaction i = [e(L0) − e(A1) − e(A2) + e(A3)] / G;
   - redesign share r = (e(A3) − e(A4)) / G;
   - window residual w = (e(A4) − e(R)) / G;
   - masking effect m = (e(R) − e(P)), relative to the R → `mae_repaired` gap;
   - auxiliary-head effect a = (e(RL) − e(R)), relative to the R vs `init` difference. If a accounts for most of the improvement over `init`, the report must say that the WCTE gain comes from the MAE-like heads, not from latent prediction.
4. **Verdicts for Q3** (per detector):
   - **"bug primary":** b ≥ 0.5 and e(A1) ≤ e(init).
   - **"teacher primary":** t ≥ 0.5 and b < 0.25.
   - **"both required":** A3 recovers ≥ 0.75 G with b, t < 0.5.
   - **"redesign required":** A3 recovers < 0.5 G and r ≥ 0.25.
   - Otherwise **"mixed"**.
   - On WCTE, a share counts only if its numerator exceeds twice the seed half-range. SK verdicts are single-seed; say so wherever an SK verdict is quoted.
5. **Representation-specific gains.** Repaired JEPA beats repaired MAE on energy R² (SK 0.965 vs 0.951, HK 0.984 vs 0.957), while losing on vertex. For every final model and for `init`, tabulate the gain over `init` on each task (vertex, direction, energy, μ/e). This tests whether the two objectives absorb global and spatial information differently.
6. **Regime (Q5).**
   - Tabulate per detector the PE and hit-occupancy distributions of the evaluation cohort.
   - Re-report the main vertex comparison in yield quartiles and wall-distance quartiles, in metres and as a fraction of R_fid, with the ML/CRLB value of each stratum.
   - State explicitly that SK→HK degradation under the all-mean probe is a readout artefact shared by all arms, so it cannot test the "problem too easy" hypothesis. Then state what the stratified, normalised numbers do say: whether the JEPA–MAE gap widens or narrows towards the low-yield, low-occupancy strata.

Outputs: `ATTRIBUTION_TABLE.md`, `attribution.json`, `REGIME_TABLE.md`.

## Stage A5 — Gradient flow on the new runs (CPU)

Re-run A2 (G1–G8) and A3b (D1–D3) on every A3 run. The key contrasts:
- A1 vs L0 (G6, G2);
- A2 vs L0 (G3);
- P vs R (G8, D1);
- H vs R (G8).

## Stage A6 — Report and acceptance

1. Write `results/round2/20261117_ablate/ABLATION_REPORT.md`:
   1. **Answers to Q1–Q6.** Each answer is one short paragraph with its numbers and verdict, written so it can be forwarded to the collaborator.
   2. Step-0 identity.
   3. Gradient flow (H1–H4, H7) with figures.
   4. Attribution (Q3) per detector.
   5. Large-detector gap (Q4: D1–D3, H5, H6, verdict).
   6. Vertex-aware latent evolution (Q6).
   7. Regime and stratified results (Q5).
   8. Representation-specific gains.
   9. Limitations: single SK seed, photon-bomb regime, the 1,000-event vs full-cohort difference.
2. Update `docs/round2/STATE.md`, `DECISIONS.md` and `RESOURCE_LEDGER.json` (`R2_ablate_gpu`).
3. Write `docs/round2/ACCEPTANCE_2026-11-17_R2_ABLATE.json`, with PASS/FAIL per deliverable and SHA256s.
4. Commit and push to `origin/dev`.
5. Write `$CC_STATE/reports/r2-ablate.md` and set `status/r2-ablate.json` to `done`.
6. Print the report in full.

## Launch (Mac Terminal)

```bash
nohup bash ~/Documents/VERITAS/docs/round2/claude-code/run_ablate_after_repair.sh >/dev/null 2>&1 &
```

`r2-repair` is already done, so the phase starts immediately. Monitor with:
- `tail -f ~/Documents/VERITAS_offrepo/claude-code/logs/r2-ablate.log`
- `cat ~/Documents/VERITAS_offrepo/claude-code/notes/r2-ablate.md`
