# Claims ledger v2 — proposed changes after the Tier-1 gate (r2-tier1)

Proposal only; the ledger itself (`docs/round2/CLAIMS_LEDGER_2026-11-22.md`) is not edited. Evidence: `GATE_TABLE.json/.md`, `GATE_G1.md`, `RESCORE_LUCID.md`, `BUDGET_AXIS.md`, `YARDSTICKS.md` in `results/round2/20261122_tier1/`. All L-world numbers below are sealed-test (`sealed_test`, 30,000 events per detector, ≈10,000 photon bombs), frozen spatial-moment Ridge readout (READOUT_SPEC_v1), step 10,000, unless stated. "V" requires the blind recomputation of r2-tier1v; where its part B is still pending, the proposal is R at most.

## Changed claims

| Claim | Current (v2) | Proposed | Evidence and reason |
|---|---|---|---|
| **C12** attribution (repaired MAE gain) | Attribution **P** (bundle) | **R (L-world): the mask is the main factor on SK; the time unit is a small, unclear factor on SK and a clear but small one on WCTE.** | SK, 2 seeds: patch vs random mask −12.7 % (450 ns) and −13.9 % (50 ns), both clear (N_σ 5.7, 6.2); 450 → 50 ns at patch −3.3 % (N_σ 1.5, not clear), 450 → 5 ns −2.4 % (N_σ 1.1). Outcome **A-alt** (K1, K2 fired). The wording "the loss scale explains the repaired MAE" stays forbidden (§G). |
| **C25** time-unit dose-response on WCTE | **S** (one seed) | **R (L-world, WCTE):** finer time unit lowers the vertex error, 450 → 1.5 ns −8.7 % at patch (3 seeds, clear, N_σ 9.6) and −5.1 % at random (clear, N_σ 5.7); 450 → 50 ns −2.9 % at patch (clear, N_σ 3.3) but −1.3 % at random (not clear) | WT-p / WT-r contrasts in `GATE_TABLE`; MLP and attentive probes in the secondary table. V after r2-tier1v part B. |
| **New C29** dose-response is detector-dependent | — | **R:** on SK the patch curve is flat within ±3 % from 450 to 5 ns (seed means 0.373 / 0.370 / 0.361 / 0.361 / 0.364 m), on WCTE it falls monotonically | P2 holds formally (no clear reversal), but the total 450 → 5 ns reduction on SK (2.4 %) is below the 5 % of K2. |
| **New C30** registered knee | — | **S (exploratory reading of P4):** u_half lies within a factor 3 of u₅₀ on both detectors (SK 50 vs 62.6 ns; WCTE 15 vs 6.5 ns) and is larger on SK | P4 verdict in `GATE_TABLE`; on SK the curve is nearly flat, so u_half is poorly defined there. |
| **C27** energy moves the other way | **P** | **R (SK, sealed, σ_E/E):** from 450 to 5 ns at patch the electron energy resolution does not clearly change (+0.8 %, N_σ 0.2) while the muon resolution worsens (+12 %, clear, N_σ 2.3; seed spread large); P3 is not confirmed. A ×100 charge-loss weight makes both vertex (+2.5 %) and electron energy resolution (+4.3 %, clear) worse (P5 fails) | Re-scored as 68 % half-width of (Ê − E)/E on the sealed split, as §H asked. Statement "time-dominated MAE is worse than init on energy" to be re-worded from `RESCORE_LUCID.md` (sealed). |
| **C3** arm ordering, L-world | SK **S** | SK stays **S** for arms other than mae_repaired (one seed each); mae_repaired on SK now has 2 seeds at the default recipe (s9000, s9001: 0.368 / 0.353 m sealed) → **R** for "repaired MAE < init" on SK | RESCORE (sealed, matched and legacy init). |
| **C19** init strong baseline, L-world | R | **R**, unchanged, now on sealed: init beats raw features on all four detectors (SK 0.543 vs 0.670 m; JUNO 0.436 vs 0.552; HK 0.932 vs 1.203; WCTE 0.066 vs 0.075) | RESCORE sealed rows. |
| **C24** relative gains over init | S | Restate with **matched init** (step-0 snapshot of the same run): SK mae_repaired gain 0.180 m over matched init vs 0.175 m over legacy init; JUNO 0.156 vs 0.144; HK 0.307 vs 0.285; WCTE identical (same window) | D8/D9 closed: gains are now quoted against the matched init. |
| **C28** probe dependence | S | **R:** the mask contrasts keep their sign and clarity under MLP (2 forms) and attentive probes on SK; CT-p50 is clear under MLP and attentive (−7 % to −10 %) although not under Ridge | Secondary table of `GATE_TABLE`. F4/K3: no clear opposite sign under the attentive probe (K3 not fired). |
| **C9 / C10** (L-world) | Not testable | **Exploratory only (surrogate):** LUCiD's expectation model gives a time share of the vertex information of 0.24 (SK) and 0.007 (WCTE) at σ_TTS = 2.1 ns; achieved fits from truth reach 0.18 m (SK) and 0.03 m counts-only (WCTE), far from the surrogate (≈1 cm). | `YARDSTICKS.md`. Label "surrogate"; never "CRLB" or "information ceiling"; §G wording stays forbidden. |
| **C11** loss share, L-world | S | Add u₅₀ (P4 amendment): 62.6 ns (SK), 6.5 ns (WCTE); loss share is still not gradient share | `PREREG` amendment 1, `U50.json`. |

## Unchanged, restated

- **§G forbidden wording** stays in force, in particular "the loss scale explains the repaired MAE" (the gate gives A-alt: the mask carries the SK gain) and "information ceiling / CRLB / timing largely unused" for LUCiD data.
- **K4** did not fire on the recorded readings (blind recomputation part A agrees within 0.2 % on the vertex readouts; the yardstick joint fit is unbiased in x, y, z). Under a strict all-parameter reading of K4(b) it would fire through the nuisance t₀; `GATE_G1.md` states this.

## Moves towards V (§H)

- H1 (blind recomputation): part A done by r2-tier1v (27/27 vertex readouts within 0.2 %); part B (E-cells) pending → C12-attribution, C25, C29 can move to **V** when it reports within tolerance.
- H2 (sealed split): done (`sealed_test`, 30,000 events per detector, all four detectors). Evaluation reads are logged; generation/QA/manifest accesses by other code were not logged at the time and are disclosed in the addendum of `SEALED_ACCESS_LOG.md` (none touched a readout).
- H3 (≥ 2 SK seeds): done for the registered SK cells (seeds 9000, 9001); seed C was not trained (GPU cap; no registered SK contrast fell in the borderline band 1.5 ≤ N_σ < 2.5 — CT-p50 is at 1.49).
- H4 (un-bundling): done for mask × time unit at fixed loader and window.
- H5 (likelihood yardstick): surrogate only (LUCiD's expectation model, not a likelihood of the sampled data); stays open for Tier 2.
