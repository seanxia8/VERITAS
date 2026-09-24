# ORACLE docs — index

_18 September 2026. The map of `docs/`. If you only read three files, read
`../README.md`, `EXPERIMENT_DESIGN.md`, and `TODO.md`._

## Start here

| file | role |
|---|---|
| [`../README.md`](../README.md) | project overview, what has been run, how to build and test |
| [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) | **the canonical design** — tiers, arms, controlled-variable protocol, the two-claim protocol (Part V) |
| [`TESTBEDS.md`](TESTBEDS.md) | **the canonical testbed inventory** — simulations and datasets that serve and do not, what is implemented |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | the `core` protocol part — **unfrozen draft**; confirmatory mode refuses without a freeze |
| [`TODO.md`](TODO.md) | paper/protocol TODO board (the testbed board is `../TODO.md`) |

## Design and plan

| file | role |
|---|---|
| [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) | canonical design (Parts I–V) |
| [`TESTBEDS.md`](TESTBEDS.md) | canonical testbed inventory (arms, autoencoder target, reserve candidates) |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | work packages, interfaces, acceptance criteria, gates |
| [`TWO_CLAIM_REVISION_PLAN.md`](TWO_CLAIM_REVISION_PLAN.md) | the two-claim revision plan (M1–M5, I1–I13, D1–D4) |
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | unfrozen pre-registration draft |
| [`DATASET_STRATEGY_2026-09-16.md`](DATASET_STRATEGY_2026-09-16.md) | dataset/simulator strategy memo (admission criteria; NuRadioMC, GWOSC, cryogenic options) |
| [`COLLIDER_ARM_2026-09-18.md`](COLLIDER_ARM_2026-09-18.md) | Tier-3b future-work collider domain-transfer arm (CMS/ATLAS) |
| [`noise_module_lucid.md`](noise_module_lucid.md) | the LUCiD front-end noise module explained (`src/noise_module_lucid/`) |

## Status and results

| file | role |
|---|---|
| [`RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md`](RESULTS_LATENT_MONITOR_TIER1_2026-09-06.md) | the Tier-1 linear-subject signature table (13/14) |
| [`REVISION_REPORT_2026-09-16.md`](REVISION_REPORT_2026-09-16.md) | pass-1 revision report |
| [`REVISION_REPORT_2026-09-16_pass2.md`](REVISION_REPORT_2026-09-16_pass2.md) | pass-2 revision report (two-claim consolidation) |
| [`REVISION_REPORT_2026-09-17_cumulant.md`](REVISION_REPORT_2026-09-17_cumulant.md) | Fisher-cumulant integration report (D7/D8/D10) |
| [`ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`](ARITRA_CUMULANT_INTEGRATION_2026-09-17.md) | design note for the third-cumulant rung |
| [`reviews/`](reviews/) | adversarial/self reviews, the LUCiD noise review, the physical-latent testbed memo |

## Prompts (agent instructions)

| file | role |
|---|---|
| [`IMPLEMENTATION_PROMPT_TWO_CLAIM.md`](IMPLEMENTATION_PROMPT_TWO_CLAIM.md) | the two-claim implementation pass |
| [`IMPLEMENTATION_PROMPT_CUMULANT.md`](IMPLEMENTATION_PROMPT_CUMULANT.md) | the cumulant integration pass |
| [`IMPLEMENTATION_PROMPT_MLST.md`](IMPLEMENTATION_PROMPT_MLST.md) | the MLST revision pass |
| [`REVIEW_PROMPTS.md`](REVIEW_PROMPTS.md) | reviewer prompts (§A before implementation, §B per milestone) |

## Archive

[`archive/`](archive/) holds superseded documents, indexed in
[`archive/README.md`](archive/README.md): the 5 Sep arms and latent-monitoring
plans, the testbed survey, the 3 Sep theme/novelty/dev notes, the audit, open
decisions, the earlier revision plan, and package docs.

## Related trees

| path | role |
|---|---|
| `../src/` | packages: `noise_module`, `noise_module_lucid`, `latent_monitor`, `herald_simulation`, `nuradio_simulation` (to build), `prometheus_simulation`, `qp_simulator`, `reconstruction_model`, `tidmad_transformer` |
| `../notebooks/` | executed walk-throughs and the `_build_nb*.py` sources of the two HeRALD/LUCiD notebooks |
| `../latex/` | the collaboration concept note (`paper3_proposal.tex`/`.pdf`) |
| `../results/` | checked-in development/feasibility outputs (non-citable unless stated) |
| `../scripts/` | local/Condor helpers and the NuBench feasibility scripts |
| `../sources/` | research notes and raw API dumps (`sources/README.md`) |
| `../reference/` | prior-art and testbed PDFs |
| `../archive/` | off-active-path code and notes |
