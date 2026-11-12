# Runbook addendum — Round 2: Gate V becomes informational; photon-statistics ceiling as a diagnostic

Date: 12 Nov 2026 (v2, 10:10). Parent: `docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` (sha256
`366f0680a5b262a90774bf725fd6ee75c516766789a605807b22978630572afe`). v1 (10:01) made Gate V
ceiling-relative. Before any agent ran v1, the author replaced that with this version.

---

## 0. Authority

- **Author, 12 Nov 2026, 10:08, after learning that only preflight had run:** "啊？只是preflight？
  能不能一步步往下走，我会handsoff now，直接从生成数据开始".
- **Author, 12 Nov 2026, 10:01** (option B, the ceiling diagnostic): "做B，那现在我把run——chain停掉了".
- **Standing rule (30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".

**What this changes:**
- **Gate V is informational for SK, JUNO and HK.** The parent's stop S2 no longer applies. Every
  detector proceeds to generation, training and evaluation as soon as its other X0 items allow.
- **Recorded outcome:** Gate V (absolute, round-1 thresholds) failed for SK and JUNO; HK's result is
  pending. The analytic model is therefore not validated against PhotonSim + LUCiD at these scales. Every
  round-2 result and the report carry this caveat.
- **The photon-statistics ceiling (§2) is a diagnostic**, computed on spare CPU. It is reported next to
  the Gate V values and never gates or delays anything.

**What does not change:** all other parent rules, values and stops: S1, S3–S9, budget, ledger, order
SK → JUNO → HK, the HK fallback and the seed-9002 rule. "No refit of the yield per MeV" also stands.

---

## 1. Gate V record

**Gate V so far** (round-1 harness `cherenkov_validation`, holdout `e500_010..019`, thresholds
module-pooled Pearson ≥ 0.85 and PMT log-charge Pearson ≥ 0.70 for every event):

| Detector | Module, median | Module, worst event | Log-charge, median | Log-charge, worst event | Yield ratio |
|---|---:|---:|---:|---:|---:|
| SK | 0.829 | 0.778 | 0.692 | 0.665 | 1.19 |
| JUNO | 0.888 | 0.845 | 0.761 | 0.729 | 1.08 |

WCTE in round 1 had a module median of 0.902 (worst event 0.877) and a log-charge median of 0.815.

**Open question for the report:**
- The reference pattern is a QE-weighted sum over a finite list of incident photons. In large detectors
  those photons spread over far more channels, so Pearson against the reference is capped below 1 even
  for a perfect model.
- The diagnostic below measures that cap. Whether a low value is noise or model mismatch is reported,
  not decided.

---

## 2. Ceiling diagnostic (what a perfect analytic model would score)

For each detector d ∈ {WCTE, SK, JUNO, HK} and each holdout event e:

1. **Expected pattern.** Recompute μ, the analytic expected detected charge per PMT, with the exact
   `compare_event` code path, geometry and `cfg` of that detector's Gate V. It must reproduce the
   committed `expected_detected` to a relative 1e-6; if it does not, report that and skip that
   detector's diagnostic.
2. **Photon totals.** Λ = `n_photons` and w̄ = `observed_detected` / `n_photons`, both from the committed
   Gate V row.
3. **Perfect-model references.** Draw K = 500 times, with RNG seed 20261112 keyed by (detector, event,
   draw). Each draw distributes the Λ incident photons over PMTs as multinomial(Λ, μ / Σμ), with
   detected charge n* · w̄.
4. **Score each draw** with the harness's metric arithmetic (same `keep` rule, `log1p`, module pooling).
   The event ceiling c_{d,e,m} is the median of the draws; also record their p05 and p95.
5. **Detector ceiling.** C_{d,m} is the median over events. The normalised score is
   ρ̃_{d,e,m} = observed / c_{d,e,m}.
6. **Optional.** The split-half reliability of the reference (Spearman–Brown, then √), if the per-photon
   transported events are available or cheap to regenerate. Also report the mean incident photons per
   hit PMT and per pseudo-module.

**Implementation:** `src/jepa_noise/ext_r2_gatev.py` with tests:
- the draw keeps Λ;
- a synthetic perfect-model event scores at its own ceiling;
- μ reproduces the committed `expected_detected`.

The ceiling code gets a short fresh-context review before its numbers go into the report.

---

## 3. Order and machines

1. **Mac:** commit this addendum unchanged on `dev`, record its sha256, push.
2. **Immediately resume the parent's r2-run in order:**
   - **SK** (X0 items 1–3 and 5 are recorded): generation and pack (§3), smokes (§5.2), training (§5.1),
     evaluation (§6), readings (§7), seed-9002 rule (§8);
   - **then JUNO**, the same way;
   - **then HK**, once its X0 yield range and benchmark exist. Its Gate V result, whenever it arrives, is
     informational. If the HK X0 job died, rerun only what generation needs (yield range, benchmark).
3. **Ceiling diagnostic:** on spare Deepthought CPU (≤ 4 cores, `nice -n 10`) while generation or
   training runs. It never pre-empts generation. CPU ≤ 3 wall-h, outside the generation cap.
4. **Records:**
   - `EXECUTION_sk.json` and `EXECUTION_juno.json` get a resume record that keeps the original S2 record
     and cites this addendum;
   - `DECISIONS.md` gets this §0;
   - `STATE.md` is updated.

---

## 4. Deliverables

- `results/round2/20261112_gatev_ceiling/`: `gatev_ceiling.json` (per detector and event: Λ, w̄, c,
  p05, p95, observed, ρ̃; per detector C) and `GATEV_CEILING.md`.
- The r2-report states, per detector:
  - the absolute Gate V values with "failed / passed";
  - the ceilings and ρ̃;
  - the caveat sentence.
