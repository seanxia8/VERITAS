# VERIFY_B_REPORT — blind recomputation of the Tier-1 gate contrasts (r2-tier1v, part B)

Seal B: `a8b6f3c` ("BLIND SEAL B"), pushed 2026-11-24 10:34Z, before GATE_TABLE.json was opened. Inputs: VERIFY_INPUTS_B.json (sha256 `e0aa3369e0b87a95…`, 180 entries = 135 unique (run, step); the `.prev_*` copies are byte-identical). Readout: frozen spatial-moment Ridge (runbook §1 (c)) fitted on `dev_probe_train` photon bombs, e = mean 3-D vertex error on the sealed photon bombs. Statistics follow PREREG §2.2: own paired bootstrap (2,000 resamples, one index draw shared by all cells), SE_seed = SD_pooled/√n over the detector's registered contrasts, N_σ = |T̄|/max(SE_boot, SE_seed), clear = N_σ ≥ 2 with one sign in every seed.

Coverage: 132 of 135 checkpoints. The three `wcte__…__trmedian` checkpoints (exploratory, not registered) were not recomputed, because their median time reference is implemented only in origin/tier1 code, which stayed closed before the seal. All 45 step-0 checkpoints have bitwise-identical encoder weights within each (detector, seed), so one dump per seed stands for the others.

## Registered contrasts (sealed_test, step 10,000)

| Contrast | T̄ recomputed | per seed | SE_boot | SE_seed | N_σ | clear | T̄ r2-tier1 | ΔT̄ (pp) | N_σ r2-tier1 | clear r2-tier1 | Tolerance |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|---|---|
| CT-p50 | -3.33% | A -0.8%, B -5.9% | 0.0033 | 0.0223 | 1.49 | no | -3.33% | -3.0e-05 | 1.49 | no | pass |
| CT-p5 | -2.39% | A -0.8%, B -3.9% | 0.0036 | 0.0223 | 1.07 | no | -2.39% | -3.4e-05 | 1.07 | no | pass |
| CT-r50 | -1.96% | A +1.9%, B -5.8% | 0.0027 | 0.0223 | 0.88 | no | -1.96% | +6.6e-06 | 0.88 | no | pass |
| CM-450 | -12.71% | A -11.7%, B -13.8% | 0.0038 | 0.0223 | 5.71 | yes | -12.71% | +2.2e-05 | 5.71 | yes | pass |
| CM-50 | -13.90% | A -14.0%, B -13.8% | 0.0037 | 0.0223 | 6.24 | yes | -13.90% | -1.1e-05 | 6.24 | yes | pass |
| WT-p:450->50 | -2.95% | A -2.7%, B -3.3%, C -2.9% | 0.0025 | 0.0091 | 3.25 | yes | -2.95% | -6.3e-07 | 3.25 | yes | pass |
| WT-p:450->5 | -6.98% | A -6.8%, B -8.3%, C -5.8% | 0.0026 | 0.0091 | 7.68 | yes | -6.98% | -1.3e-06 | 7.68 | yes | pass |
| WT-p:450->1.5 | -8.73% | A -9.2%, B -9.3%, C -7.7% | 0.0027 | 0.0091 | 9.62 | yes | -8.73% | +7.8e-07 | 9.62 | yes | pass |
| WT-r:450->50 | -1.30% | A -1.6%, B -2.0%, C -0.4% | 0.0021 | 0.0091 | 1.43 | no | -1.30% | -1.1e-05 | 1.43 | no | pass |
| WT-r:450->5 | -5.68% | A -8.1%, B -6.0%, C -3.0% | 0.0026 | 0.0091 | 6.26 | yes | -5.68% | -6.5e-06 | 6.26 | yes | pass |
| WT-r:450->1.5 | -5.14% | A -7.3%, B -5.1%, C -3.0% | 0.0028 | 0.0091 | 5.65 | yes | -5.14% | -6.8e-06 | 5.65 | yes | pass |
| WM:450 | -3.41% | A -3.9%, B -4.2%, C -2.1% | 0.0032 | 0.0091 | 3.75 | yes | -3.41% | -3.7e-06 | 3.75 | yes | pass |
| WM:5 | -4.71% | A -2.6%, B -6.6%, C -4.9% | 0.0030 | 0.0091 | 5.18 | yes | -4.71% | +1.5e-06 | 5.18 | yes | pass |

SD_pooled: SK 0.0315 (r2-tier1 0.0315), WCTE 0.0157 (0.0157). Contrasts within 2 pp: 13/13 (max |ΔT̄| 3.4e-05 pp); same clear/not-clear classification: 13/13.

Cell errors: 30/30 within 5 % (max |rel| 4.5e-07). These are the 20 sealed cells of BUDGET_AXIS.json at steps 5,000 and 10,000, plus the seed-mean P2/P4 curves of GATE_TABLE.json; GATE_TABLE has no other per-cell errors (`vb_compare_gate.json`).

## Criteria and outcome

| Item | Recomputed | r2-tier1 | Same |
|---|---|---|---|
| K1 | True | True | yes |
| K2 | True | True | yes |
| F1 | False | False | yes |
| F5 | the loss scale sets the order and rate of learning | the loss scale sets the order and rate of learning | yes |
| A_weak | False | False | yes |
| P2_holds | True | True | yes |
| P1 | True | True | yes |
| P3_electron_clear_positive | False | False | yes |
| outcome | A-alt | A-alt | yes |

- **K1** fires: |CT-p50| = 3.3 % and |CT-p5| = 2.4 % are both below 5 % and neither is clear (N_σ 1.49, 1.07). CM-450 = −12.7 % (N_σ 5.71) and CM-50 = −13.9 % (6.24) are ≤ −10 % and clear, so the outcome is **A-alt**.
- **K2** fires as well: the total reduction 450 → 5 ns on SK at patch is 2.4 % < 5 %. P2 itself holds (no clear reversal ≥ 3 %; adjacent-unit chain on seed A only, because 150 and 15 ns exist only at s9000).
- **F1** not met (CT-p5 > −10 %). **A′ weak** not met (CT-p5 not ≤ −5 %, not clear, P3 not confirmed).
- **F5**: |CT-p5| = 5.9% at 5,000 steps vs 2.4% at 10,000 steps (at most half), so "the loss scale sets the order and rate of learning". Same as r2-tier1.
- **Borderline rule** (PREREG §2.2): no registered SK contrast has 1.5 ≤ N_σ < 2.5 (CT-p50 is at 1.493), so no seed C is required. This agrees with the manifest, which lists no SK seed C.
- **P3 sign check** (SK patch, sealed contained electrons, Ridge on ln E from (c), 68 % half-width of (Ê − E)/E): contrast 450 → 5 = +0.59% (A +0.21%, B +0.97%), N_σ 0.61, positive but not clear, so P3 is not confirmed. r2-tier1: +0.76%, N_σ 0.21. Same sign and classification; ΔT̄ = -0.17 pp, within the 2 pp contrast tolerance.
- **P1**: all CT and WT contrasts negative (CT-r50 has mixed seed signs). Same as r2-tier1.
- **Outcome implied by K1, K2, F1, F5: A-alt.** This is identical to GATE_TABLE.json. It assumes F4/K3 and K4(b) did not fire; those depend on MLP/attentive probes and yardstick fits that this phase does not recompute (r2-tier1 reports them not fired). Part A found no headline failure without a definitional cause, so K4(a) does not fire on the recomputation side either.

## Discrepancies and diagnosis

- None beyond tolerance. Contrast T̄, per-seed values, cell errors and SD_pooled agree to ≤ 5e-7 relative, so the encoder outputs and the Ridge are effectively the same. SE_boot agrees to about 3 %; the bootstrap index draws differ (own seed vs cohort-hashed seed), as expected for independent resamples.
- P3 N_σ (0.61 vs 0.21): a definitional difference in SE_seed. I used P3's own across-seed SD (one contrast, recorded in DECISIONS.md); r2-tier1 pools over the secondary-outcome contrasts. The residual ΔT̄ of 0.17 pp comes from the energy-probe implementation details; it is within tolerance and changes neither the sign nor the classification.

## Independence statement

- **Before seal A (`8b5d2aa`, 22 Nov 20:56Z):** this runbook; my `verify/` code; black-box repository code (`point_encoder.py`, the allowed parts of `lucid_arms.py`, `data_roots.py`, `r2_store.py` as on this branch); data packs, manifests and `DATA_MANIFEST_*.json`; checkpoints with the identity, encoder and scaling keys of `recipe.json`; and, by `git show` from origin/tier1, VERIFY_INPUTS_A.json, READOUT_SPEC_v1.md and PREREG_2026-11-22_R2_TIER1.md. Disclosure: READOUT_SPEC_v1.md contains the other phase's validation values for one case (wcte__mae_repaired__s9000, dev); I read it at 19:49Z, after my dev numbers were computed (16:17Z).
- **Between seal A and seal A2 (`3ecf960`):** the rebuilt TOY_VS_LUCID.json, RESCORE_LUCID.json without its HK/JUNO sealed rows (filtered unread), and `ext_readout.layout_fractions` for the f_event diagnosis.
- **Before seal B (`a8b6f3c`, 24 Nov 10:34Z):** in addition, VERIFY_INPUTS_B.json. I opened no other origin/tier1 file, no GATE_TABLE, no tier1 evaluation or probe code, no other phase's notes, logs or reports, and no returns-tree outputs. While checking load on Deepthought I saw the command line of an r2-tier1 `tier1_eval.py` process in `ps` (a process name only, no values).
- **After seal B:** GATE_TABLE.json, GATE_TABLE.md and BUDGET_AXIS.json, for the comparison.
- No slip: no closed file was opened before its seal.
