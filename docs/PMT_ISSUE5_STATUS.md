# PMT issue 5 status

_Rewritten 18 October 2026 under runbook v2 R-5 (`docs/RUNBOOK_JUNJIE_PMT_v2_2026-10-16.md`). Supersedes the 16 October table, whose rows repeated the same boilerplate; that version is in git history at `56179eb`._

_**21 October 2026 sync:** the R-4/4b/4c rows, the WP6 freeze and the WP8 row are current; the reply replacement is prepared in [`ISSUE_UPDATES_2026-10-21.md`](ISSUE_UPDATES_2026-10-21.md) (not posted). Open decisions and their defaults are in [`DECISION_REGISTER_2026-10-21.md`](DECISION_REGISTER_2026-10-21.md)._

**Overall: engineering development only. No primary claim comparison has been run, and nothing here is a reported result.** The `181b747`/`56179eb` GPU, remote and CERNBox runs are development evidence (C1 = A, `PMT_DECISIONS_LOG.md`). Author decisions of 18 Oct 2026 (`PMT_DECISIONS_LOG.md`):
- D-Q, D-J, D-G and D-K are adopted as binding.
- D-TQ's form a + b/√q is adopted; its values stay `synthetic_dev`.
- D-E is pending Junjie; (A), deposited energy, is the labelled development default.
- κ_m = 1.10 stays `provisional_dev` until the WP6 freeze.

Physics review with Junjie has not happened for any row. "Evaluated" means a primary WP7 comparison; none has been run.

| Req. | What exists | Tested by | Status | Open, and who decides |
|---|---|---|---|---|
| J1 Stop weak electronics-noise work | Claim cells exclude frozen electronics families | Documentation check | Done (engineering) | Nothing |
| J2 Wavelength QE and degeneracy | QE(λ)·CE detection with keyed draws; exact-product control and charge-matched weak control | `test_pmt_contract.py`; fixture `exact_degeneracy_equal` passes (R-1 re-run, 18 Oct); real-photon exact control passed in `181b747` | Synthetic mechanism works | Target-PMT QE/CE values (Junjie). No identifiability claim |
| J3 Fixed-threshold T-Q, wider low-Q timing | D-TQ walk (form adopted) `a + b/√q` (b = 2 ns, q_sat = 50 PE) on [0.25, 1000] PE; the polynomial form is kept as legacy. Charge-dependent Gaussian width. Slope cells b × 1.2 and b × 0.8, plus an offset-only control | `test_physical_tq.py`; `results/pmt_r1_r3_dev_2026-10-18/tq_validation.json` (conditional mean within 3σ_MC at 0.25/1/2/5/20 PE) | R-2 done, `synthetic_dev` | Measured or thesis-derived curve (Junjie). The promised real-photon slope control was not re-run (R-4 series closed without it). Asymmetric kernel not built (D-J) |
| J4 Module amplitude/width response | Coupled gain + width cell with isolated controls; factors applied per mPMT module (D-G) through the manifest rule | `test_pmt_contract.py`, `test_response_manifest.py`; fixture cells | Synthetic mechanism works | Crate/cable map and measured factors (Junjie) |
| J5 Baseline includes response; ≥ 3 seeds | 3-seed direction pilot (`181b747`) and 3-seed multi-energy pilot (`56179eb`), both development evidence | GPU pilots; R-3 constant-predictor table (`PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md`) | R-4/4b/4c development fails (vertex; R-4 energy too); recorded verdicts stand | D-E endpoint (Junjie) and "launch WP7" (author) |
| J6 Prompt-only first trial | Pinned LUCiD prompt transport on WCTE; independent geometry/time closure | Closure runs in `181b747`/`56179eb` | Development evidence | Full detector physics review |
| J7 Vertex/energy consequence, cone check | Point-source closure; local emission-angle closure; track pilot | Fixture closure (vertex < 0.3 m, energy < 10%) passes; local cone closure passes; reconstructed cone/Eloss not built | Vertex fails on tracks (0.81–0.88 × constant vs 0.8); energy endpoint ill-posed on escaping tracks; cone check is transport-level only | D-E endpoint (Junjie); reconstructed cone deferred (Junjie) |
| J8 LUCiD use clarification | Research-use statement recorded | Documentation check | Done | Licence file and redistribution remain unapproved |
| J9 TIDMAD N/S/harm, arm scope | Decision preparation only | — | Not started (WP8) | D-S, D-T (author/Junjie) |
| J10 Show scientific value | Experiment contract frozen v2; no claim comparison | — | Not started (WP7) | "launch WP7" (author); D-E pending (Junjie) |
| J11 Consistent docs/status | This file, decisions log, R-3 table | R-5 hygiene | In progress | Nothing |

## Runbook v2 repairs

| Item | State |
|---|---|
| R-1 single response manifest | Done. Pilots build from `configs/lucid_pmt/reference.json` (schema v2); a static scan test fails on any literal-built response; the fixture was re-run with 4000 photons (`results/pmt_r1_r3_dev_2026-10-18/fixture_run_02`) and all four checks pass |
| R-2 physical T-Q | Done locally. The real-photon slope control was promised for R-4 but never re-run after R-2; the R-4 series closed without it (gap, not a passing result) |
| R-3 relative gates | Done. Gates in `configs/lucid_pmt/trainability_gates.json` apply to future runs; energy is gated pooled and per subgroup (author, 18 Oct); the past-run table is in the multi-energy report |
| R-6 dark counts | Done. Keyed dark-count stage in the readout, rate in the manifest (1 kHz per PMT, `synthetic_dev`), `readout.dark_noise_window` for dark-only windows; fixture re-run in `results/pmt_r6_dev_2026-10-18` |
| E26 sub-domain threshold | Fixed: counted as `below_charge_domain`, not raised |
| `endpoint_identifiability.py` | Committed as `dev_unreviewed` with analytic tests; supports the D-E containment preflight |
| R-4 re-run | Done (development): containment range-based (100–350 MeV ≥ 50%); D-E deposited-energy target; 3 seeds. **Gates FAIL**: vertex in all seeds (0.96–1.00 × constant); E_dep in seed 32. Retained, not retuned. `results/pmt_r4_dev_2026-10-18` |
| R-4b | Done (development). The 2×2 validation diagnostic rules out the time unit and finds only a small dark-count effect; cause not identified. R-4b test run **FAILS** vertex in all seeds (0.99–1.00 × constant). `results/pmt_r4b_dev_2026-10-18` |
| Vertex diagnostic E40 | Done (development, validation only). Arm 0 reproduces `56179eb` exactly (inputs bit-identical, checkpoints byte-identical), so there is no code regression. **Cause: the window-start time origin** (+10-unit offset on hits; conditioning). Vertex ÷ constant is 0.65–0.68 with the trigger origin and 0.94–0.96 with window start. The energy target does not starve vertex. The fix rule (NHits trigger) was approved and tested; see the NHits row. `results/pmt_r4c_dev_2026-10-18` |
| NHits time-origin rule | N = 3 at `1d9845f` failed validation with dark on (E44). **N = 4 + prompt window [−20, +60] ns at `5f4ecc0`** (`readout.nhits_trigger_time`, `readout.prompt_window_mask`). E45 validation passes: dark on 0.79–0.84, dark off 0.75–0.81, truth-dark bound 0.76–0.82; fallback 0%; no pure-source digit dropped. **R-4c (test, once) FAILS vertex** in all seeds (0.81–0.88 × constant, limit 0.8), against 0.96–1.00 for R-4 and R-4b; direction and E_dep pass. Retained, not retuned. `results/pmt_r4c_dev_2026-10-18/e45`, `r4c_test` **R-4 series closed** (author, 18 Oct 2026, seventh message): R-4, R-4b and R-4c are development fails on the 60-event pilot; those test events are retired; the vertex gate is re-evaluated in WP7 on fresh events with the frozen rule. `5f4ecc0` named for the JEPA study (supersedes `b341037`). |
| κ_m precision study | Done: SD_K = 0.040, so κ_m = 1.10, **accepted** by the author (re-check if any WP7 cell holds fewer than 12 physical events) |
| WP6 contract | Frozen v2, `configs/lucid_pmt/wp6_contract_v2.json` (κ_m = 1.10; held-out choices confirmed); D-E-dependent entries `pending_D-E` |
| WP8 reply | Replacement text prepared 21 Oct in [`ISSUE_UPDATES_2026-10-21.md`](ISSUE_UPDATES_2026-10-21.md), not posted; the 18 Oct `PMT_ISSUE5_REPLY_DRAFT.md` is superseded |
| WP7 | Not started. Waits for Junjie's D-E answer and the author's "launch WP7"; the R-4 series is closed (development fails, retained) |
| R-5 hygiene | Runs 03/04 removed (history `56179eb`); every results directory ≤ 20 MB; checkpoint paths and hashes in `results/PMT_CHECKPOINTS.json` (host re-verification in R-4); this file rewritten |
