# Runbook — Round 2: the round-1 recipe on SK, HK and JUNO

Date: 11 Nov 2026. Status: v3 (v2 added the receptive-field-matched MAE control; v3 keeps the
time-scale rule with diagnostics, adds a conditional third seed and five random-init seeds), design
not yet independently reviewed, implementation not started.
Extension `r2_multidetector_2026-11-11`, `registered: false`.

Branch: `dev`, after the repository cleanup (`docs/round2/RUNBOOK_2026-11-11_REPO_CLEANUP.md`) has
tagged `round1-final` and fast-forwarded `dev` to it. Commit this file unchanged as
`docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` before any implementation commit and record its
sha256. Results go to `results/round2/`. The round-1 recipe this runbook copies is documented in the
Project doc "LUCiD round-1 recipe: samples, encoder and training" and in §2 below; every number this
runbook needs is quoted here.

---

## 0. Authority

Quote all six lines verbatim in `docs/round2/DECISIONS.md`.

1. The author, 11 Nov 2026, 21:30 CET:
   > "根据第一轮的configuration写一个可以follow的runbook，同样的configuration，对于HK，SK，JUNO三种simulation，每种simulation用同样的configuration。有了dataset之后更正后的JEPA，MAR，random init三种结果train后prob，然后做对比"

   ("MAR" means MAE.) This authorises data generation for the three detectors, training of the arms
   in §5, probing and the comparison.
2. The author, 11 Nov 2026, answering the design questions: JEPA arm = **"Fixed-label + position-free
   (Recommended)"**; GPU budget = **"New round-2 ledger, cap 60 h (Recommended)"**.
3. The author, 11 Nov 2026, on the cleanup: **"Tag + fast-forward dev only (Recommended)"** (round 2
   continues on `dev`).
4. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

   Fix routine problems yourself (bugs, paths, environment, guard HOLD/SHED, launcher timeouts, ssh
   drops), write one `DECISIONS.md` line each, and continue. Stop only for §9.
5. The author, 11 Nov 2026, 23:05 CET, adding the receptive-field control:
   > "加入这三条建议：只加 MAE 一个 arm、一个 seed；窗口按 64 × N_PMT / 1,995 缩放：SK 约 384，JUNO 约 320，HK 约 640；三个探测器加起来大约多 8–12 L40S h，60 h 的账本还装得下。"

   This authorises the arm `mae_mse_rf` of §5.1 (one seed per detector) inside the same 60 h cap.
6. The author, 11 Nov 2026, 23:18 CET, accepting three design recommendations (keep the time-scale
   rule of §2.2 and log its effect; a pre-declared conditional seed 9002; random init with five seeds):
   > "按照你的建议更新 runbook"

**Not authorised:** any change to a round-1 file listed in an accepted `<P>`; any change to a round-1
dataset, checkpoint or committed result; any arm, seed or step count beyond §5 and the conditional
rule of §8; any change to a reading rule or threshold of §7 after outcomes are seen; any GPU use beyond
the cap of §8.

---

## 1. Goal and questions

Round 1 trained every encoder on one simulated WCTE dataset. Round 2 repeats the recipe, unchanged, on
three larger water-Cherenkov geometries from LUCiD and compares the same three kinds of encoder.

| ID | Question |
|---|---|
| Q1 | Does the fixed-label SIGReg-JEPA collapse to an event-invariant code on SK, HK and JUNO as on WCTE? |
| Q2 | Does the position-free-target JEPA escape the collapse on every detector? |
| Q3 | Does MAE beat the random-init floor on every detector? |
| Q4 | Is the WCTE ranking (MAE < position-free JEPA < random init < fixed-label JEPA, vertex error) the same on every detector? |
| Q5 | How much of any drop on the larger detectors comes from the encoder's smaller relative receptive field? (The 64-token windows cover about 10 % of WCTE but about 1 % of HK.) Answered by the receptive-field-matched MAE control. |

---

## 2. The configuration

### 2.1 Same as round 1, unchanged (do not edit)

| Item | Value (round 1, WCTE) |
|---|---|
| Generator | `analytic_cherenkov` with `lucid_bombs_v2_calibrated.json` (sha256 `90ac1601…`) values |
| Domains | photon bomb : muon : electron = 0.334 : 0.333 : 0.333, deterministic `domain_plan` (334/333/333 per 1,000; 17/17/16 per 50) |
| Vertex | uniform in volume, fiducial fraction 0.8 |
| Muon | E log-uniform 300–1,500 MeV; range 4.3 mm/MeV; cone width 0.08 in cos θ; 24 emission points |
| Electron | E log-uniform 100–1,000 MeV; shower length 0.8 m; cone width 0.16 |
| Track light | 2,248.54 photons/MeV × deposited energy |
| Spectrum, QE, CE | uniform 300–600 nm; QE (0.10, 0.18, 0.24, 0.25, 0.22, 0.16, 0.08) at 300–600 nm; CE 0.95 |
| Transport | prompt direct only; no scattering, no reflection; attenuation λ₀ = 50 m; n_group 1.33 |
| Readout | SPE σ 0.3; threshold 0.25 PE; walk calibrated at 0.25 PE (4 mV/PE); jitter 1.5 ns; emission offset ±50 ns |
| Fixed response | odd (pseudo-)modules +3 % charge gain, +5 % timing width |
| Dark noise | 1 kHz per PMT in the readout window [−200, 800] ns |
| Trigger window | NHits window W = 20 ns |
| Drift laws | F1 module gain s = 0.05 (training and N probes), s = 0.30 (D probes); F3 per-PMT gain spread 0.10; clip [0.5, 1.5] |
| Seeds | dataset seed 20261017; keyed namespaces unchanged (so the same event ID gives the same domain, normalised vertex, direction and energy draw on every detector) |
| Tokens | one per PMT; 9 features (position, direction, ln q, time, hit flag); fixed scaling |
| Views | within split (fixed parity of the slot inside each module); `n_min = 20` for training, 0 for probes |
| Encoder | `EncoderConfig()` defaults: d 128, 4 blocks, 4 heads, window 64, shift 32, d_pred 64, 2 predictor blocks, full predictor attention; spectral muP init |
| Training | 10,000 steps; effective batch 32 (4 events from each of 8 runs); Muon (lr 1e-3, momentum 0.95, Nesterov, 5 NS steps) + AdamW (lr 1e-3, betas 0.9/0.999, eps 1e-8, wd 0); constant lr; no warmup, no decay, no clipping; fp32, TF32 off; checkpoint every 500 steps; final step frozen |
| Objectives | as `lucid_arms_v0.json`: SIGReg 1,024 slices, t ∈ [0, 3], 17 nodes, λ 0.02; MAE head 128 → 3, losses on ln q, hit flag, time (hit targets only) |
| Probes | as round 1 (§6) |

### 2.2 Forced by the geometry, fixed by rules (each rule reproduces round 1 on WCTE exactly)

| Quantity | Rule | WCTE | SK | HK | JUNO |
|---|---|---|---|---|---|
| Geometry source | LUCiD pin `eaf2e89`; real layout where LUCiD has one | `wcte_geometry.npz` | `sk_geometry.npz` (11,096 PMTs) | `hk_geometry.npz`, **active** PMTs only (19,746; the 19,036 `inactive_*` entries are excluded) | `JUNO_geom_config.json` (sphere R 17.5 m, 10,000 sensors, r 0.25 m), positions generated once by LUCiD's own `generate_detector` and frozen to an `.npz` with its sha256 |
| Shape | from the file | cylinder R 1.577, H 2.777 m | cylinder R 16.962, H 36.378 m | cylinder R 32.430, H 65.929 m | sphere R 17.5 m |
| Modules | real `mpmt_id` where present; otherwise **pseudo-modules**: k = the divisor of N_PMT closest to 19 (ties → larger); cut the PMTs, sorted in token Hilbert order (12 bits), into consecutive groups of k; `pmt_id_in_mpmt` = rank in group (1-based) | real, k 19, 105 modules | k 19, 584 | k 18, 1,097 | k 20, 500 |
| Fiducial volume | 0.8 × each dimension | 0.8 R × 0.8 H | same | same | 0.8 R sphere |
| Position scaling | divide by the half-extent of the tank's bounding box | (R, R, H/2) | (R, R, H/2) | (R, R, H/2) | (R, R, R) |
| Hilbert bounding box | the tank's bounding box | [−R, R]² × [−H/2, H/2] | same | same | [−R, R]³ |
| Longest direct path L | cylinder √((2R)² + H²); sphere 2R | 4.20 m | 49.74 m | 92.48 m | 35.00 m |
| Crossing time t_cross | L · 1.33 / c | 18.6 ns | 220.7 ns | 410.3 ns | 155.3 ns |
| Prompt window | (−20 ns, ceil₁₀(t_cross + 40 ns)) | (−20, 60) | (−20, 270) | (−20, 460) | (−20, 200) |
| Time feature scale | prompt upper edge / 0.6 | 100 ns | 450 ns | 766.7 ns | 333.3 ns |
| NHits N | smallest N ≥ 4 with 50 · P(Poisson(λ₂₀) ≥ N) ≤ 1e-5, λ₂₀ = 1 kHz × N_PMT × 20 ns (expected dark-only triggers per 1 µs window) | 4 | 6 | 8 | 6 |
| Photon-bomb yield range | the round-1 §3.5.3a definitions in relative terms: Y_lo at median hit fraction 0.100 (= 200 / 1,995); Y_hi at the lower of 0.95 median hit fraction or 300 PE median (pseudo-)module charge; found by the round-1 detectability procedure on 1,000 development events | 6,298 – 539,434 (round 1) | derived | derived | derived |
| Attention window, **`mae_mse_rf` control only** | W_rf = the multiple of 64 nearest to 64 × N_PMT / 1,995; shift = W_rf / 2. All other arms keep 64 / 32 | 64 / 32 | 384 / 192 | 640 / 320 | 320 / 160 |

The values in the table were computed at design time; the X0 step (§4.2) recomputes them from the
files and stops (S4) if any differs. The readout window [−200, 800] ns stays; X0 checks that it
contains every prompt window plus the ±50 ns emission offset.

**Why the time scale follows the prompt window, not a fixed 100 ns (decided, author line 6).** A fixed
100 ns would keep 1 ns = 0.01 feature units everywhere, but the time feature's spread inside an event
grows with the crossing time, and the MAE loss adds the time MSE to the ln q and hit MSEs unweighted.
With a fixed 100 ns the time term on SK and HK would grow by roughly two orders of magnitude relative
to WCTE (a rough estimate) and turn MAE into a mainly time-reconstruction objective. Scaling by the
prompt window keeps that shift to a few times. The smaller numeric range loses no information: the
input projection is learned, and float32 resolves well below 1 ns. The effect is logged (§6) so any
imbalance is visible in the report.

### 2.3 Known limitations kept on purpose (record, do not fix)

- λ₀ = 50 m was harmless at WCTE scale (paths under 4.2 m) but attenuates strongly over 35–92 m. It
  stays 50 m because the configuration is held fixed; the report states it.
- No scattering or reflection; LUCiD's own SK physics configs include them.
- The LUCiD JUNO config is water, not scintillator.
- The track light yield was calibrated on WCTE (Gate V); §4.2 re-checks it per detector.

---

## 3. Cohorts and storage

### 3.1 Cohorts, per detector (all mixed 1:1:1, event-ID ranges as round 1)

| Cohort | Events | Run length | F1 s | Event IDs from | Stored extras |
|---|---:|---:|---:|---:|---|
| `dev_train` | 100,000 | 1,000 | 0.05 | 1,000,000 | none |
| `dev_probe_train` | 10,000 | 50 | 0.05 | 1,200,000 | twin hits, `expected_pe` |
| `dev_probe_test` | 10,000 | 50 | 0.05 | 1,210,000 | twin hits, `expected_pe` |
| `dev_probe_train_s030` | 10,000 | 50 | 0.30 | 1,200,000 | twin hits, `expected_pe` |
| `dev_probe_test_s030` | 10,000 | 50 | 0.30 | 1,210,000 | twin hits, `expected_pe` |

The s = 0.30 sets are generated now (cheap) so a later repair study needs no regeneration; this runbook
does not analyse them. Twin hits and the per-PMT `expected_pe` are written only for probe cohorts:
training never reads them.

### 3.2 One file per detector

- Generate per-run shards under `/work/dwong/r2_gen/<detector>/` with the round-1 per-run layout
  (`hits.npz`, `twin_hits.npz`, `events.npz`, `manifest.json`, `SHA256SUMS`).
- Pack each detector into **one HDF5 file**: `/ceph/dwong/jepa/data/round2/<detector>/<detector>_r2.h5`.
  - One group per cohort; datasets `run_id`, `event_ids`, `offsets`, `pmt_id` (int32), `charge_pe`
    (float32), `time_ns` (float32), `twin_*` (probe cohorts), the truth table, the per-run state sha256;
    attributes: geometry and config hashes, the §2.2 derived values, every run manifest.
  - Chunked by run, lossless compression (gzip level 4 or lzf).
- Verify: re-read every run from the HDF5 and compare with the shard byte for byte; record the file's
  sha256 in `results/round2/<date>_data/DATA_MANIFEST.json`; then remove the shards.
- Expected size (estimates): SK ≈ 13 GB, HK ≈ 24 GB, JUNO ≈ 12 GB. Free bytes ≥ 2 × the projection on
  `/work` and `/ceph` (S5). The single file costs one Ceph entry per detector.

### 3.3 Loader

- A new reader serves training batches and probe sets from the HDF5, keeping the round-1 sampling:
  4 events from each of 8 runs, without replacement.
- Cache **sparse** runs in RAM and build dense tokens per batch; the round-1 dense cache (about 13 GB
  at WCTE) would need 65–125 GB here.
- Test: tokens from the HDF5 reader equal tokens from the round-1 per-run reader on the same runs,
  bit for bit.

---

## 4. Implementation, review and preflight

### 4.1 Code (new modules only)

Round-1 files that appear in any accepted `<P>` stay byte-identical (their gates check them). Add:

- `src/jepa_noise/r2_geometry.py`: a geometry type for LUCiD cylinders without modules and for the
  sphere; pseudo-modules (§2.2); fiducial sampling, distance to boundary, Hilbert bounding box and
  position scaling for both shapes; `surfaces` label `sphere` for JUNO.
- `src/jepa_noise/r2_generator.py`: the round-1 generator applied to an `r2_geometry` detector, reusing
  the `lucid_bombs` primitives; trigger and prompt window passed explicitly to the imported
  `nhits_trigger_time(n=…, window_ns=…)` and `prompt_window_mask(window_ns=…)`; time scale from §2.2.
- `src/jepa_noise/r2_store.py`: HDF5 pack, verify and the sparse-cache reader (§3).
- `src/jepa_noise/ext_r2.py`, `scripts/jepa_r2.py`, `src/jepa_noise/tests/test_r2_*.py`.
- CLI modes: `dry-run`, `preflight --detector`, `generate --detector`, `pack --detector`,
  `smoke --detector --arm`, `train --detector --arm --seed`, `evaluate --detector`, `report`.

**Required tests (synthetic where possible):**

1. **WCTE identity:** on WCTE, `r2_geometry` returns the round-1 geometry (positions, directions, module
   index, slot, Hilbert order, scaling) exactly, and every §2.2 rule returns the round-1 value.
2. **WCTE regeneration:** `r2_generator` on WCTE regenerates one stored round-1 `dev_train` run
   (1,000 events) with identical hits (same PMTs, charges, times bit for bit).
3. Pseudo-modules: equal size, contiguous in Hilbert order, deterministic; parity split halves each.
4. Sphere: fiducial points inside 0.8 R, uniform in volume (radial CDF test); distance to boundary
   against a brute-force ray march.
5. Derived values: each §2.2 rule against hand values.
6. Store: pack → read round trip; HDF5 tokens equal per-run tokens; sampling protocol unchanged.
7. Arms: the four trained arms build on every geometry; the position-free target zeroes exactly
   columns 0–5 in the label pass (round-1 R-2 check); `mae_mse_rf` differs from `mae_mse` only in
   `window` and `shift` (a config diff of exactly those two fields), and with W_rf = 64 it is identical
   to `mae_mse`.
8. Forbidden paths: no write under round-1 data or checkpoint paths; no training mode outside §5 and
   the §8 conditional rule.
9. The §8 seed-9002 trigger logic on synthetic readings (each trigger, no trigger, overlap, a skip when
   the remaining budget is too small); `init` seeds 9000–9004 build and are reproducible.
10. The §6 loss-share logger returns shares that sum to 1 and match the logged loss terms.

### 4.2 Preflight X0, per detector

1. Geometry: load, record hashes; build pseudo-modules; recompute the §2.2 table (S4 on any mismatch).
2. Trigger check: dark-only windows (≥ 10,000 per detector) give no trigger at the derived N; photon
   bombs at Y_lo and each particle domain trigger with efficiency ≥ 99 % (S3 otherwise).
3. Yield range: the round-1 detectability procedure on 1,000 development events (S8 if it does not
   converge).
4. **Gate V per detector:** transport the 20 public PhotonSim 500 MeV μ⁻ events through LUCiD into the
   detector (`cherenkov_validation`, holdout `e500_010..019`) and score the analytic model with the
   round-1 metrics: yield ratio in [0.5, 2.0], module-pattern Pearson ≥ 0.85, PMT log-charge Pearson ≥
   0.70, median time residual ≤ 5 ns. The yield per MeV is **not** refitted (S2 on failure).
5. Benchmark 1,000 mixed events: events/s per process, hits per event, bytes per event; project CPU
   wall time and disk for §3 (S5, S7).
6. Encoder smoke projection (after §5.2): step time and peak memory per arm.

**WCTE regression (once, before any detector):** with the new code, reproduce the round-1 `init9000`
and `mae_within_trigger9000` full-input vertex on the WCTE probe sets within 1e-3 m, read through the new
HDF5 reader from a WCTE pack of the two probe cohorts (S6 on failure).

### 4.3 Review and acceptance

A fresh-context reviewer reads this runbook and the full diff; blocking checklist with file:line
evidence: (i) round-1 files untouched; (ii) every §2.1 value reaches the generator unchanged;
(iii) the §2.2 rules and their WCTE identity; (iv) pseudo-modules and sphere geometry; (v) store and
loader equivalence; (vi) arms, readings, budget and stops. Commit the review and
`docs/round2/ACCEPTANCE_2026-11-11_R2.json` (runbook sha256, impl commit, verdict PASS) before any
generation.

---

## 5. Arms and training

### 5.1 Arms, per detector

| Arm | Definition | Seeds |
|---|---|---|
| `sigreg_jepa` (fixed-label JEPA) | round-1 `sigreg_jepa`, target-only label pass (label-leak fix) | 9000, 9001 |
| `target_posfree` (position-free JEPA) | round-1 R-2: the label pass gets columns 0–5 zeroed, target stays trainable | 9000, 9001 |
| `mae_mse` (MAE) | round-1 `mae_mse` | 9000, 9001 |
| `init` (random init) | the encoder at initialisation, no training | 9000, 9001 (used by the readings); 9002, 9003, 9004 (floor band only, §7) |
| `mae_mse_rf` (receptive-field-matched MAE, control) | `mae_mse` with `EncoderConfig(window=W_rf, shift=W_rf/2)` from §2.2; data, objective, optimiser, schedule and every other setting identical | 9000 only |

Seven training runs per detector, 21 in total, plus at most the conditional seed-9002 runs of §8.
`mae_mse_rf` runs after the six main runs of its detector, and on HK only if it still fits the cap
after them (§8). The five `init` encoders cost no training, only probe forward passes.

### 5.2 Running

- Order: SK, then JUNO, then HK.
- Per detector: one 20-step smoke per trained arm, `mae_mse_rf` included (own root, archived, never
  resumed), to measure step time and peak memory; then the six main runs, then `mae_mse_rf`.
- **Micro-batch:** the largest of 8, 4, 2, 1 whose smoke peak stays ≤ 45 % of the L40S memory; the
  effective batch stays 32 (engineering, recorded).
- Host: deepthought2 L40S through `scripts/jepa_run.sh` (no guard bypass); at most two jobs at once;
  `setsid` for long jobs; per-run wall cap = 1.5 × the smoke projection.

---

## 6. Evaluation (as round 1)

On `dev_probe_train` / `dev_probe_test`, for every model in §5.1:

| Reading | Input | Probe | Metric |
|---|---|---|---|
| Vertex | photon-bomb events; event mean of all read-out tokens (full input) and the half pass | ridge per coordinate, α by 5-fold run-grouped CV | mean 3-D error (m) and error / fiducial radius |
| Light yield | photon-bomb events, event mean | ridge on log yield | RMS resolution |
| Token log-charge R² | ≤ 32 hit tokens per event (seed 0) | ridge | R² |
| f_layout, f_event | read-out tokens | — | variance shares (round-1 code) |
| Effective rank | event means | — | exp(entropy) |
| Module gain R² | run-mean embedding, (pseudo-)module gains | ridge | R² minus the `init` floor |
| μ/e AUC | muon and electron events | linear | AUC |
| Direction | muon / electron events | MLP (round-1 settings) | median angle |
| Deposited energy | electrons; contained muons if ≥ 500 | ridge | R² |

Uncertainty: paired run-cluster bootstrap, 1,000 resamples, seed 20261103.

**Time-scale diagnostics (logged, no threshold), per detector:**
- the standard deviation of the time feature over hit tokens of `dev_train` (and of WCTE round 1 for
  reference);
- for `mae_mse` and `mae_mse_rf`: the share of each loss term (ln q, hit, time) in the total at the
  start (mean over the first 100 steps) and at the end (mean over the last 100 steps);
- these numbers appear in `READINGS_SUMMARY.md` next to the WCTE values, with one sentence on whether
  the time term's share moved by more than 2× relative to WCTE.

---

## 7. Readings (fixed now)

Per detector:

- **R2-a (Q1), fixed-label collapse:** `sigreg_jepa` class on full-input f_event: *event-invariant* if
  ≤ 1e-3 in both seeds; *event-dependent* if ≥ 0.01 in both; else *intermediate*.
- **R2-b (Q2), position-free recovery:** *recovered* if, in both seeds, full-input vertex < the `init`
  vertex of the same seed, token ln q R² ≥ 0.5, and f_event ≥ 0.01 (the round-1 rule); *not recovered* if
  vertex ≥ `init` or R² < 0.5 in both seeds; else *unclear*.
- **R2-c (Q3), MAE above floor:** *yes* if, in both seeds, MAE vertex − `init` vertex < 0 with a
  bootstrap CI excluding 0; *no* if ≥ 0 with a CI excluding 0 in both; else *unclear*.
- **R2-d (Q4), ranking (descriptive):** the order of seed-mean vertex error across the four arms, and
  Kendall τ against the WCTE order (MAE < position-free < init < fixed-label).

- **R2-e (Q5), receptive-field effect (descriptive):** per detector, `mae_mse_rf` minus `mae_mse`
  (both seed 9000) on vertex (with a paired bootstrap CI), token ln q R², f_event and the particle
  tasks. No pass/fail: one seed, one control. If the CI on vertex excludes 0 and the difference is
  larger than the round-1 MAE seed spread (about 1.5 cm), the report says the main-arm comparison
  across detectors is confounded by receptive field and names by how much.

- **Floor band (descriptive):** per detector, the mean and range of the five `init` encoders' vertex
  error, token ln q R² and particle-task numbers. It is reported beside R2-b and R2-c; it does not
  enter their rules, which stay seed-matched (9000 against 9000, 9001 against 9001).
- **Seed 9002 (if triggered, §8):** reported as an added seed beside the two pre-planned ones, with the
  trigger that caused it. The readings are recomputed with three seeds using the same rules, "both
  seeds" becoming "all three seeds", and both versions are reported.

Across detectors (descriptive): the readings side by side with WCTE (round-1 committed values),
vertex / fiducial radius, and the particle-task table. No pass/fail across detectors.

---

## 8. Budget

- **GPU:** new ledger line `R2_l40s_deepthought2`, cap **60 L40S h** (author line 2), separate from the
  round-1 ledger (134.15 / 170, unchanged). Projection at design time (to be replaced by X0
  measurements): main-arm training SK ≈ 13–17 h, JUNO ≈ 12–15 h, HK ≈ 24–30 h; `mae_mse_rf` ≈ 2 h
  (SK) + 1.7 h (JUNO) + 3.5–4 h (HK), about 1.25–1.45× a standard MAE run because only the attention
  term grows with the window; evaluation ≈ 5 h; smokes ≈ 2 h. The total sits close to the cap, so the
  fallback below is expected to matter.
- **Fallback, fixed now, in this order:** before HK training, re-project HK from the SK and JUNO
  measured step times.
  1. If the remaining cap cannot hold six HK main runs, `mae_mse_rf` on HK and evaluation, drop
     `mae_mse_rf` on HK first.
  2. If six HK main runs plus evaluation still do not fit, train the HK main arms with seed 9000 only
     (three runs) and report it.
  3. If even that does not fit, S1.
- **Conditional seed 9002, fixed now:** after the evaluation of a detector, train seed 9002 for an arm
  on that detector only if one of these holds:
  - R2-b or R2-c reads *unclear* → seed 9002 for the arm that reading concerns (`target_posfree` or
    `mae_mse`, with `init` seed 9002 already available as its seed-matched floor);
  - the seed ranges of `mae_mse` and `target_posfree` vertex error overlap (the ranking between them is
    undecided) → seed 9002 for both;
  - R2-a reads *intermediate* → seed 9002 for `sigreg_jepa`.

  Priority: below every main run and below `mae_mse_rf` on every detector. Seed-9002 runs use only
  budget left after those, and are skipped (not stopped) when the remaining cap cannot hold them plus
  their evaluation. A skipped trigger is reported with its reason. No other seeds are added for any
  other reason.
- **CPU:** generation within the remaining round-1 generation cap (18.43 / 48 wall-h used), ≤ 16
  cores, `nice -n 10`; evaluation and reports ≤ 12 wall-h.
- **Disk / entries:** §3.2; Ceph free entries ≥ 30,000 after projected writes.

---

## 9. Stops (the only ones)

- **S1** GPU projection over the cap after the fallback.
- **S2** Gate V fails for a detector: stop that detector, continue the others.
- **S3** Trigger efficiency < 99 % at Y_lo or for a particle domain with the derived N.
- **S4** A recomputed §2.2 value differs from the table.
- **S5** Disk or entries below the floor.
- **S6** WCTE regression or a WCTE identity test fails after one fix attempt.
- **S7** CPU projection beyond the remaining generation cap.
- **S8** The yield-range procedure does not converge.
- **S9** Two consecutive review BLOCKs on the same item; any reading change after outcomes.

On a stop: finish what is safe, record it in `EXECUTION.json`, report, and continue independent items
(another detector).

---

## 10. Deliverables

- `results/round2/<date>_data/DATA_MANIFEST.json` (files, sha256, sizes, derived values, benchmarks).
- `results/round2/<date>_x0/<detector>.json` (preflight, trigger, yield range, Gate V).
- `results/round2/<date>_train/runs.json` (per run: arm, seed, steps, micro-batch, wall, GPU h, peak memory).
- `results/round2/<date>_eval/readings.json`, `READINGS_SUMMARY.md`, figures: vertex per arm per
  detector (with WCTE) and the five-seed `init` floor band, f_event per arm, particle-task table, the
  `mae_mse_rf` − `mae_mse` difference per detector, and the time-scale diagnostics of §6.
- Any seed-9002 runs, each with its trigger, or the skipped triggers with their reasons.
- `docs/round2/STATE.md`, `DECISIONS.md` (all §0 quotes), `RESOURCE_LEDGER.json` (the new line).

**Report to the author** (English, ≤ 1 page): stops; per detector the readings R2-a to R2-e with numbers;
the cross-detector table; GPU and CPU used against caps; deviations.
