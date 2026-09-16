# ORACLE pre-registration — `core` part

**status: DRAFT — UNFROZEN. No hash exists. Nothing below is approved.**

_16 September 2026 (two-claim revision, `TWO_CLAIM_REVISION_PLAN.md`). This is
the local, unfrozen draft of the `core` part of the protocol. It is the file
`latent_monitor.protocol.mode.freeze_part("core", …)` would hash; that function
is never called by any runner in this repository, `protocol/frozen/` does not
exist, and every confirmatory run refuses to start
(`require_gates`, `require_run_dependencies`). Freezing requires the decisions
in §8 and collaborator agreement; if pre-registration is lodged externally, the
venue and identifier are recorded in §8 and this file mirrors it. Values marked
**PENDING** are decisions, not placeholders to be filled by a script._

## 1. The two claims and their primary estimands

| claim | primary estimand | scores / arms (fixed without evaluation labels) | inference rule (unfrozen) |
|---|---|---|---|
| **Claim 1 — incremental attribution value** | ΔF1_attr = macro-F1{N,S}(`full_intermediate`) − macro-F1{N,S}(`generic_rich`) on the **hard matched** evaluation set | `generic_rich` (committed generic arm: input quality, outputs, uncertainty proxy, final z, operational noise-only statistics, reference distances of input/output/pre-output, z-energy splits, out-of-span, support novelty); `full_intermediate` = `generic_rich` + strictly internal hooks (channel, token: reference distances of pooled mean and second moment, reference principal coordinates) | benefit: 95 % interval above 0 **and** ΔF1 ≥ 0.10; equivalence: interval ⊂ [−0.05, +0.05]; detriment: interval < 0; no reading below 30 windows / 10 event groups or with a degenerate interval; interval by bootstrap over event groups (supporting: family-outer hierarchical) |
| **Claim 2 — incremental harm ranking** | ΔAUROC_harm = A_{K≥κ_m}(task-sensitive) − A_{K≥κ_m}(generic, committed) over **all held-out intervention cells**, cell weighting `uniform_cell` | generic committed = max of clean-calibrated {`gr_input_maha`, `gr_output_maha`, `z_mahalanobis`}; task-sensitive = clean-calibrated ‖z − z̄‖_{M_task}, W_y one-hot on the declared K target | interval by hierarchical bootstrap, outer unit = cell (family / perturbation seed where declared), event groups nested, model seeds outermost when > 1; **descriptive only below 10 outer units**; the acceptance point estimate and the equivalence rule are **PENDING** |

Required supporting results (Claim 1): inclusive and hard results with retained fractions and overlap failures; confusion matrices; family and held-out-severity results; the capacity-matched (`generic_rich_matched`) and hook-drop controls; `intermediate_only` alone; the pass-1 diagnostics (`all_generic`, `all_generic_no_noise`, `noise_only`, `full_layerwise_legacy`); operational vs privileged variants; abstention (unknown AUROC, risk–coverage curve/AUC, retained coverage, retained-known F1 with counts); the joint detect → abstain → attribute table over clean, N, S, mixture, unknown. (Claim 2): both absolute AUROCs; AUPRC with prevalence; the four alarm–harm quadrants with counts at a cell-level threshold; missed-harm rate at the alert budget; benign valid-rare cell rejection and the window-level estimator; breakdowns by contract/family/severity; conditional strong-alarm triage with the number of cells dropped; the supporting intermediate score.

Supporting analyses (never numbered as claims, no success thresholds): detection power and delay at 1 % FAR with a binomial interval on the realised rate; cost; probes; designed and task-specific controls (positive controls; linear subject only); resolvability rank and patching.

## 2. Mathematical spine (M1–M5)

M_recon = J̃_gᵀ J̃_g with J̃_g = Σ̂^{-1/2} ∂g/∂z (`jac_recon`): the Gaussian Fisher information under the *assumed* Σ̂, unit-free, local; its eigen-directions above the rank threshold are `P_resolved`, below `P_weak` (legacy names `P_exc/P_unexc` kept for the 6 Sep artifacts). M_task = J_yᵀ W_y J_y, W_y in physics-output units (`TaskMetric.one_hot` on the declared K target, or `from_resolutions`); J_yᵀ Σ⁻¹ J_y is not formed. Training support: `support.SupportEstimator` (Ledoit–Wolf Mahalanobis ∨ kNN on reference z, clean-quantile standardised), validated on constructed in/out-of-support controls with floor AUROC 0.9 before use. Signatures are conditional (`EXPERIMENT_DESIGN.md` §III.1). Invariance per statistic: shrunk reference distances — orthogonal and isotropic scale, not shear; energy splits — orthogonal and isotropic scale; task length — any invertible reparameterisation with covariant J_y; Jacobian singular subspaces — orthogonal only. Companion-paper results carry no central argument.

## 3. Information contract

Phases: `reference_fit` → `alarm_time` → `delayed_label` → `evaluation_only` (`protocol.availability`). Alarm-time arms consume `alarm_time` features only (manifest) built through `AlarmTimeInputs` (window, random-trigger records, geometry — no truth/twin/label/realised-Σ field) into a locked, source-tagged `FeatureBatch` (data flow). Residual limitation: a builder that lies about a tag is not caught; code review is part of the contract. Noise-only records operational: ORACLE-Cov, arm B, TIDMAD (208 no-injection files), arm A once built; Prometheus **no** (privileged). Covariance provenance: operational statistics use Σ̂; the realised Σ and κ_cond are evaluation quantities. Likelihood object per arm: ORACLE-Cov and arm B — exact Gaussian Kronecker full covariance; TIDMAD — diagonal inverse-PSD approximation; non-Gaussian-tail cells — the whitened quadratic form is a diagnostic, not a likelihood. Reference distances: Ledoit–Wolf shrinkage; PCA to n_ref/5 components when the hook dimension exceeds it; each score mapped to the clean-null scale (`NullCalibrator`, fitted on `calibration` windows) before any combination. Measured: distance orderings are not stable at n_ref ≲ 100 — the trained run's reference-fit count is an output of the precision study (§6).

## 4. Splits and hold-outs

Unit: the event group. Partitions: `reference_fit` 30 % (clean subject/reference/null fitting, never supervised examples), `attribution_train` 20 %, `development` 15 % (tuning), `calibration` 15 % (clean windows: FAR, conformal, null calibration), `evaluation` 20 % (`protocol.splits`; smoke values, unfrozen). Held-out for evaluation only — Tier 1: families `line_pickup` (N) and `double_pulse` (S); severities **PENDING**; perturbation seeds **PENDING**; undeclared families (Tier 1 smoke: `glitch`; arm B: WIMP; Prometheus: U1–U4) enter no fit, tuning or calibration and are scored only on evaluation groups. Hard matching: 1:1 nearest neighbour without replacement inside caliper 1.0 on {`in_rms`, `in_kurtosis`, `z_mahalanobis`, `gr_output_maha`} standardised by the clean pool; retained fractions and overlap failures reported; a declared conditional design, not causal identification. Content matching on Prometheus as before (caliper 1.0, 1:1, unmatched N retained in `N*_unmatched`).

## 5. Origin, harm, K and κ_m

Origin ∈ {clean, N_cov, N_struct, G, S_in_span, S_support, mixture, constructed, unknown}; harm ∈ {benign, harmful, undefined}; **EF** (evaluation-contract fault) / **EV** (event variation); κ_cond / κ_m. K per arm, cell level (mean over the cell's evaluation windows, baseline = the same events through the clean reference cell):

| arm | K_phys (Claim 2) | K_diag (diagnostic only) | κ_m | status |
|---|---|---|---|---|
| ORACLE-Cov (Tier 1) | \|ŷ_amplitude − amplitude\| / clean-reference mean on the same events | raw/whitened reconstruction residual | **PENDING** — dev uses the provisional 10 % (`HarmThreshold(1.10, "provisional_dev")`) | provisional_dev |
| HeST (arm B) | declared: whitened trace error and the amplitude readout | same | **PENDING** | pending |
| LUCiD (arm A) | **open** (§II.9) | — | **PENDING** | pending |
| TIDMAD (arm C) | K_rel on the frozen 20-file subset | K_dev (non-scientific) | **PENDING** | pending |
| Prometheus | angular error (frozen DynEdge) | — | **PENDING** — a physical resolution change, not a percentage | pending |

Zero baseline: K undefined, reported as such. A ranking score is never described as a probability.

## 6. Statistics, gates, precision

FAR: 1 % on non-overlapping clean calibration windows, realised rate reported with a Clopper–Pearson interval (`far_precision`); ≥ 100 windows give one-step resolution only; the calibration count, the reference-fit count and the outer-unit counts for Claim 2 are outputs of a **development precision study (Phase B, not run)**. Cell-level alert threshold: pseudo-cell bootstrap of clean windows with the declared aggregate (`cell_alarm_threshold`). Multiplicity: the combined scores are calibrated as single statistics; a run-level false-alert budget over several monitors is **PENDING**. Modes: `dev` (never citable); `confirmatory` refuses without `protocol/frozen/{core,counts,bridge}.json` matching their files, a freeze commit that is an ancestor of HEAD, a clean tree (or a frozen source-tree hash), a frozen environment-lock hash, a data-manifest hash, a model/checkpoint hash, a declared κ_m, and a destination under `results/confirmatory/` (`require_run_dependencies`).

## 7. Designed controls

Families `output_null`, `output_aligned`, `random`, `task_aligned`, `task_null` from the frozen head at the reference mean; norm matched in a declared metric (`euclidean` or `null_mahalanobis`); realised per-output loss is the endpoint; `linearization_check` reported; a zero-rank subspace is a reported skip; **linear subject only** (`NonlinearSubjectUnsupported` otherwise) until a constrained local inverse with an input-validity check exists. Positive controls, not transfer evidence.

## 8. Decisions required before this file can be frozen

1. κ_m per arm (§5) from a scientific requirement, with its source.
2. The Claim-2 acceptance point estimate and equivalence rule; whether the Claim-1 margins are achievable (Phase B).
3. Held-out severities and seed ranges; cell weighting scheme if not uniform; run-level FAR budget.
4. Calibration, reference-fit and outer-unit counts (Phase B).
5. Junjie's agreement on the title and the two-claim framing; the transfer arm (B, A or Prometheus) and its K; gates A0, B0, D5.
6. External pre-registration venue and identifier, if any: **none yet**.

When these are settled, `freeze_part("core", docs/PREREGISTRATION.md, protocol/frozen)` records the SHA-256 and commit; the environment lock, data manifest and model hashes are frozen alongside; from then on this file is edited only with a new part version.
