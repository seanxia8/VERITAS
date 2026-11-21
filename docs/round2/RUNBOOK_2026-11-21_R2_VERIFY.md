# Runbook — Independent Recomputation of Load-Bearing Numbers (r2-verify), 21 Nov 2026

Status: authorised for unattended execution, CPU only. It runs in parallel with `r2-ablate` and shares neither its worktree nor its remote clone.
Extension: `r2_verify_2026-11-21`, `registered: false`.
Companion: `docs/round2/CLAIMS_LEDGER_2026-11-21.md`. Each K-check here upgrades named claims from R to V.
Branch: `r2-verify` (created from `origin/dev`) in worktree `~/Documents/VERITAS-worktrees/r2-verify`. Remote clone: `/work/dwong/r2_verify_repo` on Deepthought (`dt1`).

---

## 0. Why, and the independence protocol

The paper's conclusions rest on a small number of numbers. Every one of them was produced by a single code path written by the same agent family that wrote the training code. This runbook recomputes those numbers with **new, minimal, independent code** and compares them **blind** against the committed values.

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `r2-verify`:
1. **Author, 21 Nov 2026, 00:04 CET:** "写一份关键数字独立复算的 runbook。它可以和 r2-ablate 并行，只用 CPU。" (Translation: "Write a runbook for independent recomputation of the key numbers. It can run in parallel with r2-ablate, CPU only.")
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".
   - Do not stop to ask.
   - At each decision point take your own recommended option, record it in one line, and continue to the end.

### 0.2 Independence rules

These are binding until Stage V6.

1. **Do not open** any committed result, report or ledger:
   - `results/round2/**`, except the two metric-free extraction commands in §V0.4;
   - `docs/round2/*REPORT*`;
   - `docs/round2/CLAIMS_LEDGER_*`, `docs/round2/ROOT_CAUSE_ANALYSIS_*`;
   - `~/Documents/VERITAS_offrepo/claude-code/reports/*`, `notes/r2-repair.md`, `notes/r2-ablate.md`.
2. **Do not open or import** the repository's evaluation, probe, physics-ceiling or regulariser code:
   - `src/jepa_noise/ext_r2*.py`, `r2_probe_repair.py`, `ext_readout.py`, `ext_shortcut.py`;
   - `abs_render.py`, `cherenkov_validation.py`, `sigreg.py`;
   - the probe functions in `lucid_arms.py` (`build_probe_dataset`, `probe_r2` and anything named `*probe*`, `*vertex*`, `*_mean_tokens*`).
3. **Allowed repository code** is only what is needed to run a frozen encoder forward:
   - the model classes, `build_arm` and `LucidArm.jepa.encode`;
   - the feature builder used to construct encoder inputs.

   The feature builder is itself checked in K0.
4. **Write all verification code from scratch** in `verify/` on branch `r2-verify`: plain NumPy, SciPy, scikit-learn or PyTorch, about 50–300 lines per check. No copying from the repository.
5. **Never modify** repository source outside `verify/`, any data, any checkpoint, or any return of another phase.
6. **Pre-registered tolerances (§V6) are fixed now.** They may not be changed after seeing committed values.

### 0.3 Compute and safety

- **CPU only:** Deepthought `dt1`, at most **6 cores**, `nice -n 15`, `TMPDIR=/work/dwong`. Before each heavy job, check the 1-minute load; if it exceeds 30, wait 10 minutes and re-check. No GPUs. No batch jobs.
- **Write locations:**
  - repository: `verify/` and `results/round2/20261121_verify/` on branch `r2-verify`;
  - Ceph scratch: `/ceph/dwong/jepa/returns/r2_verify_2026-11-21/`;
  - Deepthought scratch: `/work/dwong/r2_verify/`.
- **Parent constraints.** All parent safety constraints of `RUNBOOK_2026-11-17_R2_REPAIR.md` §0.1 hold:
  - never touch the main checkout;
  - never force-push;
  - never push to `main`;
  - never modify round-1 or round-2 data or checkpoints.
- **Branch.** Push only to `origin/r2-verify`. Do not merge into `dev`; the author merges after review.

---

## Stage V0 — Setup

1. Confirm the worktree `~/Documents/VERITAS-worktrees/r2-verify` is on branch `r2-verify` (the launcher creates it). Copy this runbook into `docs/round2/`, record its SHA256 and the §0.1 quotes in `docs/round2/DECISIONS.md`, then commit and push to `origin/r2-verify`.
2. On `dt1`, create `/work/dwong/r2_verify_repo` as a clone checked out at `origin/r2-verify`. Never use `/work/dwong/r2_repo`, which belongs to the other phases.
3. Write `$CC_STATE/status/r2-verify.json` with `{"phase": "r2-verify", "state": "running", "started": <UTC>}` and start `$CC_STATE/notes/r2-verify.md`.
4. Obtain checkpoint and data paths with these metric-free commands only. Run them from the worktree root and save their output to `results/round2/20261121_verify/INPUTS.txt`.

```bash
python3 -c 'import json;d=json.load(open("results/round2/20261117_repair/probes/probe_repair_results.json"));[print(r["detector"],r["arm"],r["seed"],r["step"],r["checkpoint"],r.get("checkpoint_sha256"),r["hdf5"],r["n_probe_train"],r["n_probe_test"]) for r in d["runs"]]'
python3 -c 'import json;d=json.load(open("results/round2/20261117_repair/retrain/retrain_readings.json"))["runs"];[print(k,v["checkpoint"]["file"],v["checkpoint"]["sha256"]) for k,v in d.items()]'
```

5. Verify each checkpoint's SHA256 before use.

## Stage V1 — K0: encoder inputs rebuilt from raw data

1. Inspect the HDF5 schema of each detector's probe pack (`h5py`) and record the datasets used.
2. Independently rebuild the 9 per-PMT input features, documenting each formula from the data attributes:
   - PMT position (3) and direction (3) in the encoder's coordinate convention;
   - `ln q`;
   - time, with the detector's time scaling;
   - hit flag.
3. On 64 events per detector, compare with the repository feature builder (allowed). Report the maximum absolute difference per column.
4. **Pass:** ≤ 1e-5 relative per column. If it fails, record the discrepancy and use the repository features for K1, flagged.

## Stage V2 — K1: readout table, recomputed

Cases:
- WCTE, SK, JUNO and HK × {init s9000, init s9001, `mae_mse` s9000, `mae_mse` s9001, `sigreg_jepa` s9000, `target_posfree` s9000, `target_posfree` s9001};
- WCTE `jepa_repaired` s9000 and `mae_repaired` s9000 (repair retrain runs).

1. **Token dump.** Use the repository encoder forward as a black box on full, unmasked inputs. Store float32 tokens for the probe-train and probe-test events, taken from the HDF5 groups named `*probe_train*` / `*probe_test*` (or the split the data defines). Record event counts.
2. **Independent readouts.** Implement each of these from scratch:
   - (a) **all-PMT mean + Ridge**;
   - (b) **hit-masked mean + Ridge**;
   - (c) **hit-masked spatial moments + Ridge**: [z̄_hit, mean_hit(z·x̃), mean_hit(z·ỹ), mean_hit(z·z̃)] with PMT coordinates scaled to unit range per detector;
   - (d) **raw-feature spatial moments + Ridge**: the same construction applied to the 9 standardised raw features, with no encoder;
   - (e) **controls:**
     - predict the training-mean vertex;
     - shuffled-label Ridge on (c);
     - two-seed init spread.
3. **Probe protocol.** Ridge α chosen by 5-fold run-grouped CV on probe-train only, over a log grid 1e-6 … 1e4. Features standardised on probe-train.
4. **Metrics.**
   - Mean 3-D error (m) and error/R_fid, with R_fid taken from the geometry in the data.
   - For (a) and (c), also the mean radial bias (predicted r − true r) in four true-radius quartiles.
5. **Invariants to report.** These are pre-registered; record each as PASS or FAIL.
   - **I1:** on SK, JUNO and HK, (c) is at least 3× better than (a) for every case.
   - **I2:** (c) MAE < (c) init on all four detectors, in both seeds.
   - **I3:** (c) `sigreg_jepa` is worse than (c) init on all four detectors.
   - **I4:** for every arm, (a) HK ÷ SK lies within ±20% of R_fid(HK) ÷ R_fid(SK).
   - **I5:** the shuffled-label error is within 10% of the predict-the-mean error.
   - **I6:** (c) init < (d) raw on SK, JUNO and HK.

## Stage V3 — K2: collapse metrics from first principles

For the K1 cases, on 200 probe-test events:
1. Decompose token variance:
   - total = variance over (event, PMT) of z;
   - **static layout** = variance over PMTs of the per-PMT mean over events;
   - **across-event, same PMT** = mean over PMTs of the variance over events.

   Report each as a fraction of the total, separately over all PMTs and over hit PMTs only.
2. Event-mean spread: trace of the covariance over events of the all-PMT mean token, divided by total token variance.
3. **Invariants.**
   - **I7:** for `sigreg_jepa` on WCTE, SK and HK, the across-event fraction is ≥ 50× smaller than for `mae_mse` of the same seed, and the static-layout fraction is ≥ 0.95.
   - **I8:** JUNO `sigreg_jepa` is less extreme than SK/HK (to be checked against the committed value only in V6).

## Stage V4 — K3: SIGReg statistic, independent implementation

1. Implement the Epps–Pulley characteristic-function statistic from the LeJEPA paper (Balestriero & LeCun, arXiv 2511.08544): random 1-D projections, an empirical characteristic function against N(0, 1), the integration weights and the N scaling. If the official implementation can be installed (pip or GitHub through the allowlist), also run it on the same inputs.
2. **Inputs.**
   - Synthetic: Gaussian, and a deliberately non-Gaussian sample with a known answer.
   - Real: context embeddings from the SK `sigreg_jepa` s9000 checkpoint on 256 dev-train events.
3. **Report** the statistic at N = 4, 8, 32 and 256 on nested subsets. Report the mean over 8 disjoint micro-batches of 4 against the value on the union of 32. Report the gradient cosine between the 8×4-accumulated and 1×32 statistics with respect to the embeddings.
4. **Invariants.**
   - **I9:** for a fixed non-Gaussian distribution, the statistic grows ≈ linearly in N.
   - **I10:** the 8×4 accumulation does not equal the 1×32 statistic (ratio far from 1), and the gradient cosine is < 0.9.
   - **I11:** the independent and official implementations agree to 1e-4 relative, where the official one is available.

## Stage V5 — K4/K5: physics ceiling and regime, independently

**K4 — ToF-only vertex fit and timing CRLB, with a deliberately simpler model.**
1. For 100 probe-test photon-bomb events per detector, take first-hit times tᵢ and PMT positions rᵢ for hit PMTs inside the detector's prompt window, read from the data attributes.
2. Fit (v, t₀) by robust least squares on tᵢ = t₀ + n_g ‖rᵢ − v‖ / c:
   - Huber loss with scale 3 ns;
   - 8 random starts;
   - group index n_g from the data or simulator attributes; if absent, 1.38, recorded.
3. Report mean and median 3-D error and the convergence rate.
4. Compute the timing-only CRLB: J = Σᵢ ∇tᵢ ∇tᵢᵀ / σ² over (v, t₀), with σ equal to the single-photoelectron timing spread from the data attributes (or 1.5 ns, recorded). Report the median √tr(J⁻¹)_pos.
5. Optional **K4b:** a charge-only CRLB from a Poisson model with expected charge μᵢ ∝ ε cosθᵢ / dᵢ² × exp(−dᵢ/λ). State the simplification.
6. **Invariants.**
   - **I12:** the ToF fit converges for ≥ 95% of events, with mean error ≤ 5 cm on SK, JUNO and HK.
   - **I13:** the ToF fit error is ≥ 5× smaller than the best (c) readout from K1 on each of SK, JUNO and HK.

**K5 — Regime.** For the full probe-test cohort of each detector, report quantiles (5, 25, 50, 75, 95%) of:
- total PE;
- number of hit PMTs;
- hit fraction of PMTs.

There is no invariant; these are descriptive.

## Stage V6 — Blind comparison, then investigation

Only now may committed results be opened.

1. Read the committed values for each recomputed number from:
   - `results/round2/20261117_repair/probes/probe_repair_results.json`;
   - `retrain/retrain_readings.json`;
   - `diagnostics/latent_variance_cka.json`;
   - `diagnostics/a6_training_checks.json`;
   - `diagnostics/physics_ceiling.json`.
2. **Pre-registered tolerances:**

   | Check | Tolerance |
   |---|---|
   | K1 probe errors | \|Δ\| ≤ max(10% relative, 3 mm) |
   | K2 fractions | same order of magnitude; layout fractions within 0.03 absolute |
   | K3 | invariants I9–I11, plus N-scaling ratios within 10% |
   | K4 ToF mean error | within ±35% of the committed ToF-only value (models differ by design) |
   | K4 CRLB | within ±35% of the committed value |

3. **Verdict per claim:**
   - **V** if every check tied to the claim passes its tolerance and its invariants.
   - **R (unchanged)** if invariants pass but a tolerance fails.
   - **Downgrade flag** if an invariant fails.

   Map C1 → K1 (I1, I5); C2 → K1 (I4); C3 → K1 (I2, I3); C19 → K1 (I6); C4 → K2 (I7); C6 → K3; C8, C9 → K4 + K1; C21 → K5.
4. **Investigate discrepancies.** For every tolerance or invariant failure you may now read the relevant repository code to find the cause. Write the diagnosis and a proposed fix in the report. **Do not change repository code outside `verify/`.**

## Stage V7 — Report and acceptance

1. Write `results/round2/20261121_verify/VERIFY_REPORT.md` with:
   1. a summary table (claim → check → recomputed → committed → tolerance → verdict);
   2. the invariants I1–I13 with PASS or FAIL;
   3. the K5 regime table;
   4. discrepancies and diagnoses;
   5. the independence statement: which files were read before V6 and the exact code used.
2. Write `verify/README.md` describing how to rerun every check with one command per check.
3. Write `results/round2/20261121_verify/ACCEPTANCE_R2_VERIFY.json` with per-deliverable SHA256 and PASS/FAIL.
4. Commit and push to `origin/r2-verify`.
5. Write `$CC_STATE/reports/r2-verify.md` and set `status/r2-verify.json` to `done`.
6. Print the report in full.

## Launch (Mac Terminal, any time; independent of r2-ablate)

```bash
nohup bash ~/Documents/VERITAS/docs/round2/claude-code/run_verify.sh >/dev/null 2>&1 &
```

Monitor with:
- `tail -f ~/Documents/VERITAS_offrepo/claude-code/logs/r2-verify.log`
- `cat ~/Documents/VERITAS_offrepo/claude-code/notes/r2-verify.md`

**Limitation.** Blindness is enforced by instruction, not by file permissions: the worktree contains `results/round2/`. The report's independence statement (V7.1.5) is the audit trail.
