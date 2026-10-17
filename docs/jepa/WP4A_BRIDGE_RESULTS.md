# WP4a — Tier-1 bridge results (runbook v4 §3.5.7, §3.5.8, §4 B1)

Status: **complete, unreviewed development output.** Runs executed 2026-10-17 on `deepthought`
(4 × GTX TITAN X), code `f10c034` (+ the read-only aggregate mode `cd33ff1`), config
`src/jepa_noise/configs/bridge_v1.json` sha256 `2f7b3c5e…`. All 12 runs: A4 bridge cell (g = 0.02,
nominal, channel split, nonlinear), 10 000 steps, **batch 128** (v3 §6 WP4's registered Tier-1 E2
batch), full registered cohort ID ranges, `registered_run: true`, fp32 TF32-off.

**Data path:** `/ceph/dwong/jepa/returns/wp4a_grid_{sigreg,anchored,mae_chi2,parallel}/`,
combined aggregate `/ceph/dwong/jepa/returns/wp4a_grid_combined/` (compact copies in this repo
under `results/jepa/wp4a_grid_*/` and `results/jepa/wp4a_grid_combined/`). Embeddings (666 MB per
run) and checkpoints stay on `/ceph` with per-run `SHA256SUMS`; the benchmark and β-cohort runs are
`returns/wp4a_bench_sigreg_s0/` and `returns/wp4a_beta_dev/`.

## β selection (development cohort, independent IDs, seed 100)

One anchored run per β ∈ {0.1, 1, 10}, v3 §1.3 rule (mean over physics parameters of
development-validation MSE divided by the development probe-training variance):

| β | score |
|---|---|
| 0.1 | 0.104152 |
| 1 | 0.104057 |
| **10** | **0.097206** |

Selected **β = 10** (no tie); recorded in `DECISIONS.md`; `beta_decision.json` passed to the grid.

## B1 (registered) — **FAIL**

Criterion (v4 §3.5.9): `sigreg_jepa` nuisance R² > `sigreg_jepa_anchored` in **every** seed and the
difference exceeds the seed-floor rule (2 × floor, v4 §3.5.8).

| quantity | value |
|---|---|
| plain mean nuisance R² | 0.525128 |
| anchored mean nuisance R² | 0.515705 |
| paired difference (plain − anchored) | +0.009422 |
| per-seed differences (0, 1, 2) | **+0.08280**, −0.01625, −0.03828 |
| seed floor (largest of SD_plain 0.0157, SD_anchored 0.0513, median SD 0.0436) | 0.0512676 |
| `all_seeds_plain_greater` | **false** (seed 0 only) |
| `|Δmean| > 2 × floor` (0.10254) | **false** |

So the anchor did **not** reduce the encoded common-mode nuisance at the β the registered physics
rule selects; the direction holds in one of three seeds only. Reason code `not_every_seed`.

## R7 bridge half (supporting; the registered R7 is now the LUCiD row) — nuisance clause fails, physics kept

Same pairing: anchored nuisance R² is lower only in seed 0. The physics side is **kept** — anchored
/ plain physics-MSE/CRLB ratios: amplitude 0.984×, t0 **0.337×**, tau_t 0.981× (all ≤ 1.1×). The
anchored arm is markedly *better* on t0 in every seed (9.41/8.54/10.17 → 3.26/3.16/3.06), which is
reported without a criterion.

## Amplitude-preservation probe (v4 §3.5.7 test) — **criterion not met**

Criterion: per-channel pulse amplitude recovered with R² ≥ 0.9 from both the post-time-block
representation and the read-out tokens, for `sigreg_jepa` and `mae_chi2`. Mean over the 8 channels:

| arm | read-out tokens (seeds 0,1,2) | post-time block (seeds 0,1,2) |
|---|---|---|
| `sigreg_jepa` | **0.9122, 0.9004, 0.9097** (pass) | 0.8076, 0.8100, 0.8111 (fail) |
| `sigreg_jepa_anchored` | 0.8953, 0.9048, 0.9052 (2/3 pass) | 0.8164, 0.8087, 0.8169 (fail) |
| `mae_chi2` | 0.7718, 0.7976, 0.7955 (fail) | 0.7741, 0.7768, 0.7741 (fail) |
| `sigreg_jepa_parallel` | 0.7775, 0.7515, 0.7973 (fail) | 0.7799, 0.7790, 0.7777 (fail) |

Random-init floor (same architecture, arm's first seed): post-time 0.7376, tokens reported per arm
in `aggregate.json`. Pattern: **post-time fails systematically** (all 12 runs, only just above the
floor); **tokens are arm-specific** — ~0.90–0.91 for `sigreg_jepa`/anchored but 0.75–0.80 for
`mae_chi2` and the parallel ablation. Reported as a finding; no re-tuning.

## Other reported quantities

- Seed CKA on probe-test embeddings: `sigreg_jepa` 0.94064, `sigreg_jepa_anchored` 0.99843,
  `mae_chi2` 0.90539 (anchored is by far the most seed-reproducible).
- `mae_chi2` beside R7: nuisance R² mean 0.4506, physics MSE/CRLB means amplitude 1.90, t0 76.7,
  tau_t 1.60.
- Physics MSE/CRLB (seed means) — plain: amplitude 1.165, t0 9.37, tau_t 1.100; anchored: 1.146,
  3.16, 1.079; mae_chi2: 1.90, 76.7, 1.60; parallel: 1.62, 15.6, 1.60.
- Data footprint (read-only walk of the four source roots): 8 104 213 028 B / 88 files.
- Seed floor rule uses ddof = 1; 3 seeds per arm.

## Incidents and provenance

- The grid ran as four guard-gated jobs (three in parallel on GPUs 0/2/3; GPU 1 only held another
  user's idle kernel and was never used). Guard level OK throughout, 0 SHED.
- A combined-aggregation attempt by the `grid` mode re-executed a completed run on CPU for 13 min
  before being killed; it damaged `wp4a_grid_sigreg` s0/s1 `run_status.json` and s0's final
  `checkpoint.pt` artifact (sha preserved in the manifest). Both runs' `probes.json` and
  `embeddings.npz` are original and their manifest rows say complete; the combined aggregate
  records both as `status_anomalies` with the `--complete-run` declaration. Nothing was restored or
  fabricated. Details: `INFRA_LOG.md`, `PREFLIGHT_A8_deepthought.md` §14–§15.
- GPU-hours: 6.52 Titan GPU-h for benchmark + β cohort + 12-run grid (4.22 h of it inside the
  per-run walls), charged in `RESOURCE_LEDGER.json`.

## What this means for the study

The Tier-1 bridge's registered prediction B1 **fails**: at the registered β selection, the
noise-record anchor does not reduce the nuisance R² of the deep encoder in every seed, although it
preserves (and for t0 improves) the physics probes and is more seed-reproducible. This is a
scientific outcome to report, not a fault to repair; the LUCiD main experiment (R6–R8, R11, R11b,
R12) is where the registered deep-E2 claims now live (§3.5.9).

## Diagnostics (added 2026-10-17, post hoc; no registered number changes)

Two post hoc checks were run after the B1 verdict was recorded. Full detail, commands,
JSON and the figure are in `docs/jepa/E1_BRIDGE_FINDINGS.md` and
`results/jepa/20261017_e1_findings/`. Nothing below re-tunes, re-selects or re-runs a
registered cell.

**The positive half of B1 is strong.** Plain `sigreg_jepa` nuisance R² is **0.5251**
against a random-init floor of **−0.00358** (same architecture and probe protocol at
initialisation; the value is identical in all 12 runs), and the two defined nuisance
targets in seed 0 are common_0 0.5812 and common_1 0.5034. That is direct evidence that
a trained deep transformer encodes the shared common mode (P1): the probe is not merely
fitting noise. The anchored arm is 0.5157, so the registered paired differences
(+0.0828, −0.0163, −0.0383) show the anchor neither added nor removed common-mode
information consistently — which is what B1's failure records.

**Why the anchor may have been too weak (diagnostic hypothesis, not a licence to
retune).** β = 10 was the *top* of the registered β grid {0.1, 1, 10}, and the
registered β-selection score was still improving at that boundary (0.104152 → 0.104057
→ 0.097206). Independently, R4b fails at β = 10 in the *exact linear model*: 9 of the
42 E1a cells exceed 5° even though that path has no optimiser. Both facts are
consistent with the anchor strength the B1 criterion would need lying beyond the
registered grid. This is a hypothesis only: the author has separately approved a
labelled follow-up (β ∈ {10, 100, 1000} with the LUCiD selection rule), and neither
this note nor the new findings file changes a registered verdict.
