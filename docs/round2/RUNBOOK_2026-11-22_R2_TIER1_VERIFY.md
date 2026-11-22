# Runbook — Blind recomputation of the LUCiD headline numbers and of the Tier-1 gate contrasts (r2-tier1v), 22 Nov 2026

**Phase** `r2-tier1v`. **Branch** `tier1-verify`, created from `origin/lucid-mc`. **Worktree** `~/Documents/VERITAS-worktrees/tier1-verify`. **Results** `results/round2/20261122_tier1_verify/`.

This phase runs in parallel with `r2-tier1` (branch `tier1`), which trains and evaluates. You recompute its headline numbers independently and blind. The two phases exchange information only through the files named in §0.2.

---

## 0. Why this runbook exists

The project rule is that only numbers recomputed blind by an independent implementation may enter an abstract. `r2-verify` did this for the analytic-world numbers. No LUCiD number has been recomputed yet, and the Tier-1 gate (Fri 27 Nov) will choose the paper's lead claim from a handful of contrasts. Both need an independent check before the author relies on them.

### 0.1 Authority

Quote verbatim in `docs/round2/DECISIONS.md` on branch `tier1-verify`:

1. **Author, 22 Nov 2026, 11:32 CET:**
   > "写runbook，把没修的，以及这个方案写进去，不要问junjie，他是共同作者但是没时间。我们继续推进，按照spotlight去进行"

   (Translation: "Write the runbook. Put the unfixed items and this plan into it. Do not ask Junjie; he is a co-author but has no time. We keep moving, aiming at the Spotlight track.") One of the unfixed items is that no LUCiD number has been recomputed blind.
2. **Standing rule (Author, 30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决". Never stop to ask. Record each decision as one line in `DECISIONS.md` and `$CC_STATE/notes/r2-tier1v.md`.

### 0.2 Independence rules (binding)

There are two seals: **seal A** (existing LUCiD checkpoints) and **seal B** (the new Tier-1 cells). A seal is a pushed commit containing every independent number of that part. Until a seal is pushed, the corresponding results of the other phase stay closed.

**You may read at any time:**
- this runbook;
- your own code in `verify/` (from `origin/r2-verify`, see VA0) and its `README.md`;
- repository code needed to run an encoder as a black box and to read data: `point_encoder.py`; in `lucid_arms.py` only the model class, `encode_tokens`, `build_arm` and `random_init_floor`; in `ext_r2.py` only `encoder_config_for` and `build_r2_arm`; `r2_store.py`; `r2_geometry.py`; `data_roots.py`; `lucid_views.ViewMasker` as a black box; `src/jepa_noise/configs/lucid_v1.json`; the frozen geometry files;
- data packs, their manifests and `results/round2/20261121_lucid/DATA_MANIFEST_*.json`;
- in the returns trees: `checkpoint.pt`, `snapshot_step_*.pt`, and the identity, encoder and scaling keys of the `recipe.json` beside them;
- from `origin/tier1`, by `git show` only: `results/round2/20261122_tier1/verify_inputs/VERIFY_INPUTS_A.json` and `_B.json`, `docs/round2/READOUT_SPEC_v1.md`, `docs/round2/PREREG_2026-11-22_R2_TIER1.md`.

**Closed until the matching seal is pushed:**
- everything else on `origin/tier1`;
- `results/round2/20261121_lucid/` except the data manifests;
- any `final.json`, evaluation output, table or report in a returns tree;
- `$CC_STATE/reports/`, and the notes and logs of other phases;
- in `~/Documents/VERITAS/docs/round2/`: `CLAIMS_LEDGER_*`, `REVIEW_PROMPT_*`, `RUNBOOK_2026-11-22_R2_TIER1_GATE.md`, `RUNBOOK_2026-11-21_R2_LUCID_REGEN.md`;
- repository evaluation and probe code: `ext_r2_repair_probes.py`, `r2_probe_repair.py`, `probes.py`, `ext_r2_results.py`, `scripts/lucid_toy_vs_lucid.py`, `scripts/tier1_eval.py`.

Blindness is enforced by instruction, not by permissions. If you open a closed file by mistake, say so in the report with the time and what you saw. The git history is the audit trail.

### 0.3 Safety and compute

- Main checkout `~/Documents/VERITAS`: never switch, stash, reset, clean, add or commit. Before every commit, check that `git rev-parse --show-toplevel` is the `tier1-verify` worktree. Push only to `origin/tier1-verify`, fast-forward only.
- Never modify data, checkpoints, returns or other phases' files. Sealed packs are read-only inputs.
- **CPU.** Deepthought: ≤ 6 cores at `nice -n 15`. portal1 (`p1`): up to one sixth of its cores at `nice -n 15`, leaving ≥ 8 free; halve the workers when other users' load exceeds a quarter of the cores.
- **GPU, encoder forward passes only.** At most one job at a time on a free Titan on Deepthought. Never use deepthought2 or HTCondor; the other phase owns them. Book it under `R2_tier1v_gpu`, cap 6 GPU-h, in `docs/round2/RESOURCE_LEDGER.json` on this branch. If no Titan is free, use CPU.
- Remote clone `/work/dwong/r2_tier1v_repo`; scratch `/work/dwong/r2_tier1v/`.
- While only waiting, poll at most every 25 minutes in one foreground Bash call.

---

## 1. Definitions (frozen; identical to Appendix A of the other phase's runbook)

Token features x_p ∈ ℝ⁹ per PMT: scaled position r_p (indices 0–2), direction (3–5), ℓ_p = ln(max(q_p, 10⁻⁶ PE) / 1 PE) for hit PMTs (index 6), τ_p = t_p / T_scale (index 7), hit flag h_p (index 8). z_p ∈ ℝ¹²⁸ is the token embedding from `encode_tokens` of the full, unmasked event. N is the number of PMTs.

- **(a) All-mean.** z̄ = (1/N) Σ_p z_p.
- **(c) Spatial-moment latent (768 features).** Let H = {p : h_p = 1} (all tokens if H is empty) and P = {p ∈ H : τ_p ≥ 0}.
  - m₁ = mean of z_p over H.
  - m₂ = Σ_H w^q_p z_p with w^q_p ∝ exp(3 ℓ_p), normalised over H.
  - m₃ = Σ_P w^e_p z_p with w^e_p ∝ exp(3 ℓ_p − 4 (τ_p − min_P τ)), normalised over P; if P is empty, m₃ = m₂.
  - C = (1/|H|) Σ_H (r_p − r̄) z_pᵀ ∈ ℝ³ˣ¹²⁸, with r̄ the mean of r_p over H.
  - Feature vector: [m₁, m₂, m₃, vec(C)].
- **(d) Raw-feature baseline (33 features).** The construction of (c) with z_p replaced by (ℓ_p, τ_p, h_p), plus 15 summaries: |H|/N; ln max(|H|, 1); mean and SD of ℓ over H; mean and SD of τ over P; r̄; Σ w^q_p r_p; Σ w^e_p r_p.
- **Ridge protocol.** Features standardised on the training split; one Ridge per target coordinate with intercept; α ∈ {10⁻⁶, 10⁻⁴, 10⁻², 1, 100} chosen by five-fold cross-validation with folds grouped by run; vertex error = mean Euclidean distance on the test split.
- **f_event.** The variance of the per-event mean token embedding divided by the total token variance, as in your `verify/k2_collapse.py`.
- **Splits.** Probes are fitted on `dev_probe_train` photon bombs. They are tested on `dev_probe_test` and on `sealed_test` photon bombs.
- **Matched init** of a run is its `snapshot_step_00000.pt`. **Legacy init** is the default-configuration random encoder at seeds 9000 and 9001, as in `r2-verify`.

Your own `verify/v6_confirm.py` already re-implements forms (c) and (d). Check it against this section line by line before use.

---

## 2. Stages

### VA0 — Setup

1. Write `$CC_STATE/status/r2-tier1v.json` (running) and the notes file. Commit this runbook.
2. Bring your earlier code into the branch: `git fetch origin r2-verify`, then `git checkout origin/r2-verify -- verify`. Read `verify/README.md`. Commit.
3. Create the remote clone and scratch directory.
4. **K0 on LUCiD data.** For 200 events per detector, show that your independent feature rebuild equals the repository tokeniser bit for bit on the LUCiD packs. A mismatch is investigated before anything else.

### VA1 — Part A inputs

1. Discover the existing LUCiD checkpoints yourself: under `/ceph/dwong/jepa/returns/r2_lucid_2026-11-21`, find `checkpoint.pt` and `snapshot_step_00000.pt`, and read only the identity, encoder and scaling keys of each `recipe.json`. Open nothing else in that tree.
2. Start the `dev_probe_train` and `dev_probe_test` feature dumps at once; they do not need the manifest.
3. Poll for `VERIFY_INPUTS_A.json` on `origin/tier1`. When it appears, reconcile it with your list (paths, SHA256, configurations), record any difference, and add the sealed packs it names. If a case is in the manifest and not in your list, or the reverse, compute it anyway and say so.

### VA2 — Part A quantities

For every case (existing LUCiD runs of all four detectors, legacy init at both seeds, and each run's matched init):
- readouts (a) and (c), and baseline (d) once per detector, on `dev_probe_test` and on `sealed_test`;
- f_event;
- controls: shuffled-label Ridge and predict-the-mean.

**Invariants, registered here.**

| ID | Statement |
|---|---|
| J1 | (c): repaired MAE is better than its matched init and than the legacy init, on all four detectors |
| J2 | (c): fixed-label JEPA is the worst trained arm and is worse than the legacy init, on WCTE and SK |
| J3 | (c): plain MAE is better than the legacy init on WCTE (both seeds) and on SK |
| J4 | f_event: fixed-label JEPA < 0.005 on WCTE and SK; repaired JEPA > 0.03 on all four detectors |
| J5 | (c) legacy init is better than (d) on all four detectors, both init seeds |
| J6 | (a) ÷ (c) ≥ 1.5 for every case |
| J7 | The shuffled-label error is within 10 % of the predict-the-mean error, for every case |
| J8 | (c): repaired JEPA is better than the legacy init on all four detectors |
| J9 | For every case, (c) on `sealed_test` is within 10 % of (c) on `dev_probe_test` |

**Tolerances against the other phase's values:** (a), (c), (d) within 5 % relative; f_event within 15 % relative; the same ordering of arms per detector and seed.

### VA3 — Seal A, comparison, report

1. Commit and push `table_a_blind.json` (every number above) with the message "BLIND SEAL A".
2. Then open, from `origin/tier1`, `results/round2/20261122_tier1/RESCORE_LUCID.json` and the rebuilt `TOY_VS_LUCID.json`. If they are not there yet, continue with part B preparation and come back.
3. Compare under the tolerances. For each discrepancy, diagnose the cause; after the seal you may read the other phase's evaluation code for that purpose.
4. `VERIFY_A_REPORT.md`: one row per claim and case with recomputed value, the other phase's value, tolerance result, invariants and verdict. Verdict rule as in `r2-verify`: **V** = every tolerance and invariant passes; **R** = invariants pass and a tolerance fails for a diagnosed definitional reason; **flag** = an invariant fails.

Target: seal A by Wed 25 Nov.

### VB1 — Part B quantities (the gate)

1. Poll for `VERIFY_INPUTS_B.json`. It lists the Tier-1 cells (detector, mask, time unit, seed), their final, 5,000-step and step-0 checkpoints, and the registered contrasts by run id. `PREREG_2026-11-22_R2_TIER1.md` defines the contrasts, the statistics and the criteria.
2. For every listed checkpoint: readout (c), and (a), on the sealed photon bombs.
3. Compute each registered contrast exactly as PREREG §2.2–§2.3 prescribes, with your own bootstrap and seed standard error: T̄, per-seed values, SE_boot, SE_seed, N_σ, clear or not.
4. Evaluate K1, K2, F1 and F5 from your numbers and state the outcome they imply. F4, K3 and K4(b) depend on probes and fits this phase does not recompute; say so.
5. **Sign check for P3.** Electron energy on the sealed electrons, SK at patch: Ridge on ln E from the features of (c); 68 % half-width of (Ê − E)/E; the contrast 450 → 5 with its sign and whether it is clear.

### VB2 — Seal B, comparison

1. Commit and push `table_b_blind.json` with the message "BLIND SEAL B".
2. Then open `GATE_TABLE.json` on `origin/tier1`.
3. Tolerances: each cell's error within 5 % relative; each contrast T̄ within 2 percentage points; the same clear / not-clear classification; the same outcome from K1, K2, F1 and F5.
4. If the manifest grows after the seal, treat the new cells as part B2 with their own seal.

Targets: seal B by Thu 26 Nov, 20:00 CET. If manifest B is not on `origin/tier1` by Thu 26 Nov, 12:00 CET, verify what exists and state the coverage. Final extension until Fri 27 Nov, 06:00 CET, then finish.

### VB3 — Report and acceptance

1. `VERIFY_B_REPORT.md`: the contrast table, tolerance results, your outcome against the other phase's outcome, discrepancies and diagnoses.
2. An independence statement: what you read before each seal, what after, and any slip.
3. `ACCEPTANCE_R2_TIER1V.json`: PASS / FAIL / SKIPPED per deliverable.
4. Set the status file to `done`, write the phase report, and print it in full.

---

## 3. Deliverables (all on `origin/tier1-verify`, under `results/round2/20261122_tier1_verify/`)

`table_a_blind.json` · `VERIFY_A_REPORT.md` · `table_b_blind.json` · `VERIFY_B_REPORT.md` · `ACCEPTANCE_R2_TIER1V.json` · new or changed code under `verify/` with an updated `verify/README.md` · `DECISIONS.md` and `RESOURCE_LEDGER.json` entries.

## 4. Stops

None beyond the safety rules. A failed invariant or tolerance is a result to report, not a reason to stop.

## Launch (Mac Terminal, its own window, in the foreground)

```bash
bash ~/Documents/VERITAS/docs/round2/claude-code/run_tier1v.sh
```
