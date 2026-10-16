# PMT issue 5 status

16 October 2026. **PARTIALLY COMPLETE — engineering development only.** WP0 complete; WP1–WP5 mechanisms implemented and synthetically tested, with target PMT, physical mapping and actual LUCiD/track validation outstanding. WP6/WP8 drafts prepared; WP7 not run. WP9 engineering closeout complete; physics/scientific closeout pending.

Base `0cae6b061e488e319643cfb0ba820318b8923578`; starting HEAD `84f65348a438e1b8a4594246afd3d60785518d76`; branch `codex/junjie-pmt-response`, clean at start. Fixed base verified ancestor; local dev remains the base. No applicable AGENTS.md found in this worktree or parent directories. Protected main checkout, JEPA, manuscript and historical results are not task outputs.

Sources: live [issue #5](https://github.com/seanxia8/VERITAS/issues/5) still has the four comments recorded in the runbook; [issue #4](https://github.com/seanxia8/VERITAS/issues/4) and its single acknowledgement were checked. No newer scientific design change found. See [source/stage contract](PMT_SOURCE_AND_STAGE_CONTRACT.md) for thesis verification and upstream code locators.

| Requirement | Implementation | Tested | Physics reviewed | Evaluated | Unresolved boundary |
|---|---|---|---|---|---|
| J1 | Complete exclusion | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | No new electronics claims |
| J2 | Synthetic mechanism passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Target QE/CE and upstream photon exporter pending; no causal identifiability claim |
| J3 | Synthetic mechanism passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Gaussian digit approximation; no measured target curve or asymmetric thesis reproduction |
| J4 | Synthetic mechanism passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Physical crate map and field calibration missing; proxy only |
| J5 | Baseline fixture passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | WP7 >=3/target5 seed training not run |
| J6 | Synthetic fixture passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Actual prompt Cherenkov sample unavailable |
| J7 | Point-source engineering passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Track/cone validation and D-K primary policy pending |
| J8 | Research-use record updated | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | Licence file absent at retrieval; redistribution/release unapproved |
| J9 | Decision preparation complete | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | D-S/D-T and real-data transfer execution deferred |
| J10 | Draft only | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | Physics validation, precision design, models and held-out comparison remain |
| J11 | Engineering docs/tests complete | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Scientific issue remains partially complete |

The [final report](PMT_ISSUE5_FINAL_REPORT.md) maps every requirement to source, code and evidence. All output is development/non-citable, independently of simulator name. Tests: baseline PMT 68; final PMT 93, noise 145, latent/TIDMAD 104 passed; two optional-dependency skips. These are separate suites, not 342 independent physics validations.
