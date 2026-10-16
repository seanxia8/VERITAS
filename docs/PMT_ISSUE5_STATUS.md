# PMT issue 5 status

16 October 2026. **PARTIALLY COMPLETE — engineering development only.** Real pinned-LUCiD prompt transport passes across 500/1,000/1,500 MeV after a retained candidate-lookup failure and repair. Three GPU seeds pass pooled direction/vertex development gates but all fail initial-energy trainability (39.0–41.8% MAE; required <30%). Target-PMT calibration, energy/containment and primary protocol decisions remain open. See [multi-energy report](PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md) and [author decisions](PMT_CALIBRATION_AND_PROTOCOL_DECISIONS_2026-10-16.md).

Base `0cae6b061e488e319643cfb0ba820318b8923578`; starting HEAD `84f65348a438e1b8a4594246afd3d60785518d76`; branch `codex/junjie-pmt-response`, clean at start. Fixed base verified ancestor; local dev remains the base. No applicable AGENTS.md found in this worktree or parent directories. Protected main checkout, JEPA, manuscript and historical results are not task outputs.

Sources: live [issue #5](https://github.com/seanxia8/VERITAS/issues/5) still has the four comments recorded in the runbook; [issue #4](https://github.com/seanxia8/VERITAS/issues/4) and its single acknowledgement were checked. No newer scientific design change found. See [source/stage contract](PMT_SOURCE_AND_STAGE_CONTRACT.md) for thesis verification and upstream code locators.

| Requirement | Implementation | Tested | Physics reviewed | Evaluated | Unresolved boundary |
|---|---|---|---|---|---|
| J1 | Complete exclusion | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | No new electronics claims |
| J2 | Synthetic mechanism passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Real photon exporter and exact paired controls pass; target QE/CE pending; no causal identifiability claim |
| J3 | Synthetic mechanism passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Gaussian digit approximation; no measured target curve or asymmetric thesis reproduction |
| J4 | Synthetic mechanism passes | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | 105 physical mPMT groups verified; crate map, measured group response and field calibration pending |
| J5 | Three-seed track baseline completed; energy fails | Multi-energy real-photon GPU pilot | Pending (J8: not applicable) | Scientific comparison not run | Pooled direction/vertex gates pass; all energy gates fail; primary WP7 claim comparisons not run |
| J6 | Real prompt integration passes | Independent geometry/time closure | Pending (J8: not applicable) | Scientific comparison not run | 60 physical events at three energies; 15,360-ray closure passes after retained repair; full detector physics review pending |
| J7 | Point-source, local cone and track development checks | Synthetic point + real track baseline | Pending (J8: not applicable) | Scientific comparison not run | Direction/vertex development pass; initial-energy trainability fails; primary endpoints and D-K pending |
| J8 | Research-use record updated | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | Licence file absent at retrieval; redistribution/release unapproved |
| J9 | Decision preparation complete | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | D-S/D-T and real-data transfer execution deferred |
| J10 | Draft only | Documentation/source check | Pending (J8: not applicable) | Scientific comparison not run | Physics validation, precision design, models and held-out comparison remain |
| J11 | Engineering docs/tests complete | CPU synthetic/regression | Pending (J8: not applicable) | Scientific comparison not run | Scientific issue remains partially complete |

The [final report](PMT_ISSUE5_FINAL_REPORT.md) maps every requirement to source, code and evidence. All output is development/non-citable, independently of simulator name. Tests: baseline PMT 68; final PMT 93, noise 145, latent/TIDMAD 104 passed; two optional-dependency skips. These are separate suites, not 342 independent physics validations.

Follow-up evidence: 94 local PMT tests pass (one optional skip); remote padding/permutation tests pass; source audit, 6,144-ray closure, real-photon paired controls and all three GPU trainability seeds pass. Full remote PMT suite: 95 passed, no skips. These do not close the scientific issue.

Latest multi-energy evidence: 96 local tests pass with two optional PyTorch skips; 98 remote tests pass without skips. Source/code/config/dataset/checkpoint hashes and independent prediction metrics are verified. The held-out pilot has 12 physical events, 96 placements and only one contained placement; the 500 MeV source has prior development use. Every result remains development/non-citable.
