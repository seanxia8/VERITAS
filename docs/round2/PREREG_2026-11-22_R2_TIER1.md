# Pre-registration — r2-tier1 (Tier-1 decision campaign), 22 Nov 2026

Copied verbatim from `docs/round2/RUNBOOK_2026-11-22_R2_TIER1_GATE.md` §2 (runbook SHA256 `43c8635c18a1067182e5bbf642dc2d3ab27a964fa815e0cc8da4c2a8d77236f3`) before the first production training job. Later changes are dated amendments appended below; the text above the amendments line is never edited.

## 2. Pre-registration (copy verbatim to `PREREG_2026-11-22_R2_TIER1.md`)

### 2.1 Cells and notation

- A cell is (detector, mask, time unit, seed), with everything else as in the existing `mae_repaired` run of that detector (encoder window, shuffled loader, 10,000 steps).
- Masks: `patch` = `spatial_patch`; `random` = per-event uniform half of the PMTs; `parity` = `within`.
- Time units in ns. Seeds A = 9000, B = 9001, C = 9002.
- e(m, u, s): mean 3-D vertex error in metres on the **sealed** photon bombs, frozen spatial-moment Ridge readout (Appendix A).
- Per-seed contrast T(u₁→u₂; m, s) = e(m, u₂, s) / e(m, u₁, s) − 1. T̄ is its mean over seeds.

### 2.2 Statistics

- SE_boot: paired bootstrap over sealed test events (2,000 resamples) of the per-seed contrast, combined over seeds.
- SE_seed: SD_pooled / √n_seeds, where SD_pooled is the RMS, over the registered contrasts of that detector, of the across-seed SD of the per-seed contrast.
- N_σ = |T̄| / max(SE_boot, SE_seed).
- A contrast is **clear** when N_σ ≥ 2 and it has the same sign in every seed.
- **Borderline rule.** If a registered SK contrast has 1.5 ≤ N_σ < 2.5, train seed C for its two cells (inside the cap, dropping from the bottom of the priority list) and recompute.

### 2.3 Registered contrasts

| ID | Detector | Contrast |
|---|---|---|
| CT-p50 | SK | T(450→50; patch) |
| CT-p5 | SK | T(450→5; patch) |
| CT-r50 | SK | T(450→50; random) |
| CM-450 | SK | e(patch, 450) / e(random, 450) − 1 |
| CM-50 | SK | e(patch, 50) / e(random, 50) − 1 |
| WT-p | WCTE | T(450→50), T(450→5), T(450→1.5) at patch |
| WT-r | WCTE | T(450→50), T(450→5), T(450→1.5) at random |
| WM | WCTE | e(patch, u) / e(random, u) − 1 at u = 450 and u = 5 |

Secondary outcomes, same contrasts: μ and e direction (median angle), electron and muon energy resolution (68 % half-width of (Ê − E)/E), μ mis-identification at 99 % e efficiency.

### 2.4 Predictions

- **P1 (sign).** CT-p50, CT-p5, CT-r50 and all WT contrasts are negative.
- **P2 (monotone).** On SK at patch, e falls along 450 → 150 → 50 → 15 → 5. Only a clear reversal of ≥ 3 % between adjacent units counts against it.
- **P3 (trade-off).** On SK at patch, electron energy resolution worsens from 450 to 5 (contrast positive and clear). Muon energy has the same sign.
- **P4 (knee).** Let u₅₀ be the unit at which the time term is half of the end-of-training loss of the default (50 ns) run of that detector, computed from its training log (§T5.2) and appended to this file as a dated amendment **before any E-cell is evaluated**. Let u_half be the largest tested unit at which the reduction from 450 reaches half of the total reduction to the finest tested unit. Prediction: u₅₀ / 3 ≤ u_half ≤ 3 u₅₀ on each detector, and u_half(SK) > u_half(WCTE).
- **P5 (specificity).** With the charge-loss weight × 100 at (patch, 50), the vertex error is not more than 3 % lower than at (patch, 50), and the electron energy resolution is not worse.
- **P6 (direction).** The direction contrasts have the sign of the vertex contrasts.

### 2.5 Falsification, kill criteria and outcomes

- **K1.** On SK, both CT-p50 and CT-p5 are smaller than 5 % in magnitude or not clear. Then A′ is dropped. If CM-450 or CM-50 is ≤ −10 % and clear, the outcome is **A-alt**; otherwise **P2**.
- **K2.** P2 fails by a clear reversal, or the total reduction 450 → 5 on SK at patch is below 5 %. Same consequence as K1.
- **F1.** CT-p5 ≤ −10 % with N_σ ≥ 3 and P2 holds: outcome **A′ strong**.
- **A′ weak.** −10 % < CT-p5 ≤ −5 %, clear, and P3 confirmed. The claim is then worded as allocation between channels, not as a gain.
- **F4 / K3 (probe robustness).** The sign of CT-p5 under the MLP probe (mean of 3 probe seeds) and under the attentive probe (mean of 3) must equal the Ridge sign. A clear opposite sign under the attentive probe gives outcome **P2** with "probe-dependent conclusions" as a fourth pitfall. Non-clear differences are reported only.
- **F5 (wording, registered both ways).** Compare CT-p5 on SK at 5,000 and at 10,000 steps (snapshots). If its magnitude at 10,000 is at most half of that at 5,000: "the loss scale sets the order and rate of learning". If it is at least as large: "at this budget the loss scale selects the solution". Otherwise: "undetermined at 10,000 steps".
- **K4 (data consistency).** (a) A headline LUCiD number fails blind recomputation beyond tolerance with no diagnosed definitional cause; or (b) the joint yardstick fit started at truth (§T6) has a mean bias above 5 standard errors in any coordinate on SK or WCTE. Then finish running jobs, start nothing new, and report.

| Outcome | Condition |
|---|---|
| A′ strong | F1 met; K2, K3, K4 not fired |
| A′ weak | as defined above; K2, K3, K4 not fired |
| A-alt | K1 or K2 fired and a mask contrast ≤ −10 % and clear |
| P2 | otherwise |

### 2.6 Not registered

Everything else (JUNO / HK, the budget axis beyond F5, yardsticks, the parity bridge) is exploratory and is labelled so.

---

## Amendments (dated, append-only)

