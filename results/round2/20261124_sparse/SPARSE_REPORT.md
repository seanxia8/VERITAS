# r2-sparse report — sparse hit tokens on WCTE (24 Nov 2026)

Runbook `docs/round2/RUNBOOK_2026-11-24_R2_SPARSE_WCTE.md`; pre-registration `docs/round2/PREREG_2026-11-24_R2_SPARSE.md` (pushed 0083758 with the window and r, before any training). Real LUCiD data only (`lucid_v1`, LUCiD e651e836, tier A1; re-packed as `lucid_v1_w2`). 21 new WCTE trainings (7 cells × seeds 9000/9001/9002, 10,000 steps), all complete; 48 scorings (21 runs + 3 C0 runs, each at step 10,000 and step 0, validation and sealed). GPU 16.3 of 28 GPU-h (`R2_sparse_gpu`). Tables: `SPARSE_TABLE.md/.json`; decisions: `docs/round2/DECISIONS.md` (r2-sparse section).

## 1. Window study and chosen window (W0, W1)

From the cached `dev_probe_train` digits (2,000 bombs, 500 μ, 500 e per detector; `WINDOW_STUDY.md`):
- **WCTE has no late-light problem.** No PMT has a second digit (LUCiD's 200 ns integration window spans the whole event), and the share of signal digits later than +60 ns is 3 × 10⁻⁶. SK, JUNO and HK do lose late light (0.8 / 3.6 / 1.3 % of first-per-PMT signal digits lie beyond the window end). SK and HK caches are truncated by LUCiD's trigger gate; JUNO's, like WCTE's, is not.
- **The early "signal" digits are dark-opened windows, not early light.** 25 % of WCTE events have a physics-flagged digit before trigger − 20 ns. These digits are flat in time over the 100 ns dark pad and carry normal hit charges. Their rate matches the dark-noise expectation (0.309 per event observed vs 0.317 expected; 25.2 % vs 25.4 % of events). They are dark photoelectrons that opened the integration window on a PMT that receives light later; LUCiD flags such a digit as physics. So no early direct light is lost on WCTE. The old readout does drop these PMTs entirely, charge included (F3).
- **W1 rule, applied as registered:** P = max(50, q99.5 + 10) = 110 ns. A = 50 ns, the smallest multiple of 50 ns that contains ≥ 99.5 % of first-per-PMT signal digits (it contains 99.996 %). The WCTE window is **[−110, +50] ns** (old: [−20, +60] ns). The post-trigger side is 10 ns shorter than before, as the rule dictates.

## 2. Data validation (W2)

`lucid_v1_w2` re-reads the cached digits for `dev_train`, `dev_probe_train`, `dev_probe_test` and `sealed_test`, with LUCiD dark noise filling the part of the window the cache does not cover. No regeneration was needed. Three passes failed validation; each was diagnosed, kept on disk as superseded, and never used:
1. The dark-digit density was low at the window start. Cause: LUCiD's jitter shifts digits late. Fix: start generating the dark extension 20 ns before the window.
2. The runbook's deposit proxy (earliest physics digit) was contaminated by the dark-opened digits above. Fix: a first-light proxy, the earliest physics digit that has a physics neighbour within 20 ns.
3. A 1.5 ns gap remained between the cached and the extension dark noise. It was calibrated once on `dev_probe_train` and closed.

On the data of record (`shards_w2b`), all four cohorts pass every check:
- event IDs and truths are identical to v1;
- every v1 hit is present and bit-identical, except on PMTs whose first digit became an extension dark digit;
- triggers are equal in 99.99 % of events;
- signal hits are ≥ v1 on the common window;
- dark density is flat within ±10 % per 10 ns (max 3.1 % on `dev_train`). This excludes [−20, +10) ns, where the NHits trigger itself pulls dark digits to τ = 0, as it does in v1.
- minimum charge is 0.149 PE.

Dark hits rise from 0.38 to 0.75 per event, consistent with the expected 0.72 for the 160 ns window. The S1n ratio was fixed before training at **r = 1**: WCTE's median kept fraction is 0.463, against SK's 0.386 at r = 2.

## 3. Results (sealed photon bombs, frozen spatial-moment Ridge; T = test / base − 1)

| ID | Contrast | T̄ | per seed | N_σ | clear | prediction → verdict |
|---|---|---|---|---|---|---|
| R1 | C1 vs C0 (window) | +1.4 % | +1.8 / +0.2 / +2.1 % | 1.2 | no | none (reported) |
| R2 | C2 vs C1 (fixed loss) | +2.3 % | +1.5 / +3.8 / +1.8 % | 2.1 | yes | ≤ +2 % → **fails** (narrowly) |
| R2b | C3 vs C2 (embedding + isolated decoder + count head) | +32.3 % | +35.0 / +34.4 / +27.5 % | 28.7 | yes | ≤ +2 % → **fails** |
| R3 | S1 vs C3 (sparse encoder) | +16.5 % | +15.3 / +15.5 / +18.6 % | 14.6 | yes | ≤ +2 % → **fails** |
| R4 | S1n vs S1 (negative sampling) | −0.2 % | −0.2 / −0.3 / −0.1 % | 0.2 | no | within ±3 % → **holds** |
| R5 | S1p vs S1 (positives only) | +2.5 % | +2.1 / +2.1 / +3.1 % | 2.2 | yes | S1p worse, clear → **holds** |
| R6 | S2 vs C3 (time-ordered position prediction) | −13.4 % | −15.4 / −12.4 / −12.3 % | 11.8 | yes | no sign (reported) |

Seed-mean sealed vertex errors: C0 0.0529, C1 0.0536, C2 0.0549, C3 0.0726, S1 0.0846, S1n 0.0844, S1p 0.0866, S2 0.0629 m. Raw-feature baselines are 0.0753 m (v1) and 0.0852 m (w2). The validation cohort and the MLP and attentive probes have the same signs for R2b, R3 and R6.

**Decision (§2.6): report and stop. The sparse family is a negative result on WCTE.**
- S2 meets every part of the rule except the last. It beats C3 by ≥ 5 % and clearly on vertex (−13 %), μ direction (−24 %) and e direction (−20 %), with MLP and attentive probes of the same sign. But it is worse than the best dense control (C1, 0.0536 m) on the primary outcome, so it does not qualify.
- S1 and S1n are clearly worse than C3.
- R2 fails narrowly (+2.3 %, N_σ 2.1) and R2b fails strongly, so neither the fixed loss nor the new embedding becomes a default.

## 4. What this means for the co-author's proposals

- **Fixed positional code + content MLP, no direction (with the isolated decoder and count head, as registered in C3): rejected on WCTE.** C3 reaches the same pretext losses as C2 (charge 0.353 vs 0.349; hit and time terms equal), so the cause is not a training failure. Its tokens are less useful to the frozen moment readout: +32 % vertex error, +15–18 % on the MLP and attentive probes. R2b measures the three components together, so this study cannot attribute the loss to one of them.
- **Hit-only tokens: rejected.** The sparse encoder is a further +16.5 % worse than its dense control C3.
- **Negative subsampling (r = 1):** no effect (R4 holds, −0.2 %); on WCTE it changes little, as predicted.
- **Absence information matters, as predicted** (R5 holds). Querying unhit PMTs helps by 2.5 % (clear), and by 4 % on directions.
- **Time ordering with many-to-one position prediction (S2): the most promising idea, but not good enough here.**
  - S2 is the best sparse model by far: −25 % vertex versus S1.
  - It has the best attentive-probe vertex (0.0599 vs 0.0641 m for C1) and the best direction resolution of all cells (μ 5.92° vs 6.50°, e 5.33° vs 5.78° for C1). These comparisons with C1 are exploratory.
  - Its Ridge vertex and μ/e identification (mis-ID 0.167 vs 0.054) are worse than C1's.
  - Diagnostics: its predicted mass sits mostly on hidden hits of *other* time blocks (66 %), only 8 % on hits of its own block, and 26 % on unhit PMTs. It learns where the light is, not when.
- **Count head:** well calibrated in every arm (prediction / truth ≈ 1.00, r ≥ 0.995).
- **The wider window (C1 vs C0): +1.4 %, not clear.** It keeps the dark-opened PMTs but with dark-noise times. The raw readout gets worse (0.0753 → 0.0852 m); the learned readouts barely change.

## 5. Deviations and caveats

- W2 deviations:
  - The deposit proxy was refined (first-light proxy plus a 1.5 ns offset calibrated on dev data), and a 20 ns generation margin was added. Without these the extension fails the runbook's own uniformity check.
  - Validation criteria were stated before the data of record was validated: A = 50 ns makes v1 hits in (50, 60] ns absent by construction, and the NHits trigger zone is excluded from the uniformity check.
  - The superseded passes stay on disk (nothing under `/ceph/dwong/jepa/data` is deleted).
- Batch scoring returned the shipped run directories, and the collector renamed the originals to `*.prev_<cluster>`. All six copies were verified byte-identical.
- Native suite: 1,603 passed. All 7 failures are pre-existing or environmental (al2o3 data files and a git `code_sha` missing from an archive snapshot, and D14).
- The fresh-context review returned PASS.
- Sealed packs were read for metrics only through `scripts/tier1_eval.py`. The re-pack, packing, structural validation and staging copy of the sealed cohort are logged in `SEALED_ACCESS_LOG.md`.
- Tier-1's smaller time units, which work better on WCTE, are not used in any arm; the contrasts are internal. WCTE is the least diluted detector, so the loss fix (F2) remains untested where it matters most (SK).
