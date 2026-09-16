# `latent_monitor` — controlled-variable latent monitoring

Implements `docs/EXPERIMENT_DESIGN.md` §§III.2–III.4: the `Subject`
protocol, the reference-cell projectors, per-event Δz statistics against
paired twins, the pre-registered attribution lookup, abstention, and the
adjustments that follow from a diagnosis. Follows the `oracle_cov`
`Subject` interface (`represent` / `outputs` / `jac_recon` / `jac_output`)
so it can be lifted into `noise-weighted-subspace-reconstruction` later.

```bash
PYTHONPATH=src python -m latent_monitor.run_table --out results/latent_monitor_tier1
PYTHONPATH=src python -m pytest src/latent_monitor/tests -q      # 85 tests (1 skipped without torch), ~25 s CPU
PYTHONPATH=src python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_<date>   # the two-claim CPU smoke (dev only, NOT CITABLE)
```

## What is in here

| module | what |
|---|---|
| `whitening.py` | `KroneckerWhitener` — Σ̂ = Σ_c ⊗ Circ(S(f)) as a named, replaceable parameter; `for_channels(C)` carries the correlation *assumption* to another sensor count; `estimate_kronecker` from noise-only records; κ(Σ̂⁻¹Σ) by Kronecker factors |
| `subject.py` | the `Subject` protocol and the six hooks `whitened → channel → token → z → pre_output → output` |
| `linear_subject.py` | the analytic subject: PCA encoder, geometry-weighted pooling, decoder, least-squares head; `with_sigma_hat` is the whitening lemma as an adjustment (GLS re-derivation, decoder untouched); `refit_stage` = stage-restricted LoRA |
| `torch_subject.py` | S1: the compact transformer wrapped as a `Subject` — whitening in front, an explicit position embedding (zero at init), a ridge *probe* decoder, autograd `jac_output`, exact self-patching of pooled hooks |
| `tier1.py` | paired Tier-1 cells from `noise_module`: reference; Σ-covariance (corr ↑/↓, bandwidth, line); Σ-structural (gain drift, channel loss, jitter); geometry (½C, 2C on the same box); event (out-of-span glitch; in-span oscillation, double pulse). Every cell also carries noise-only records |
| `reference.py` | `fit_reference`: P_out/P_null from J_y, `P_resolved/P_weak` from the measurement metric M_recon = J̃_gᵀ J̃_g (legacy names `P_exc/P_unexc` kept), null distributions from ref-vs-ref twins and noise-record halves |
| `task_metric.py` | M_recon (measurement, unit-free, assumed Σ̂) versus M_task = J_yᵀ W_y J_y (physics-output units, no Σ); the task length of Claim 2; `InvarianceReport` |
| `support.py` | training-support novelty (Ledoit–Wolf Mahalanobis ∨ kNN on reference z), validated on constructed controls before abstention may use it |
| `statistics.py` | per-cell statistics: mean-shift norm (null metric), per-event alarm, energy splits, noise-only z-variance ratio, residual PSD (smoothed and single-bin) and channel-correlation shifts, out-of-span fraction, layer profile, consequence, conditional-on-alarm AUROC, abstention rate |
| `lookup.py` | `calibrate` (thresholds fixed once from the reference null) and `attribute` — the rule order *is* the decision procedure |
| `designed.py` | positive controls: output-null / output-aligned / random and the task-specific `task_aligned` / `task_null` (from M_task); exact for the tied linear subject and **refused** for nonlinear subjects (`NonlinearSubjectUnsupported`); norm matched in a declared metric; `NullSpaceUnavailable` is an explicit skip; `linearization_check` |
| `protocol/` | the two-claim protocol (`docs/EXPERIMENT_DESIGN.md` Part V, `docs/PREREGISTRATION.md`): `labels` (origin vs harm, EF/EV), `availability` (manifest + typed `AlarmTimeInputs` + source-tagged `FeatureBatch` + contract), `splits` (five event-group partitions), `features` (typed alarm-time builder; shrunk/PCA reference distances; `NullCalibrator`), `arms` (primary/control/diagnostic arms, one classifier, one grid; macro-F1 policy), `consequence` (paired ΔAUROC, quadrants, cell threshold, valid-rare cell/window rejection, hierarchical bootstrap, FAR precision), `abstention` (conformal on clean calibration, unknown AUROC, risk–coverage), `matching` (hard contrasts, joint decision table), `mode` (dev/confirmatory, freeze + run-dependency gates, provenance), `smoke` |
| `adjust.py` | `rewhiten`, `activation_patch`, `damage_patch`, `refit_stage` |
| `run_table.py` | the whole §1 table + adjustments → `table.json`, `table.md`, `adjustments.json` |
| `estimators/` | the four linear representation classes of Paper 1 — `of.py`, `cw_pca.py`, `tied_linear_ae.py`, `nfpa.py` — plus `identification.py` (tangent-basis gauge fix); copied from the Paper 1 experiment repository, see `estimators/README.md` and `docs/EXPERIMENT_DESIGN.md` §IV |

## What the 6 Sep table is, and is not (16 Sep)

Every Δz statistic in `statistics.py` is computed against the paired clean
twin and the noise-only statistics are cell-level aggregates — replay-side
information, `evaluation_only` in `protocol.availability`. The table is
development evidence of the *signatures*, not alarm-time attribution and not
a C2/C4 result. Alarm-time features live in `protocol.features`; the noise-only
statistics are acquisition-quality features of the **generic** arm wherever the
acquisition supplies random triggers, and the `noise_only` ablation assigns any
attribution gain to its source.

## The discriminator the lookup rests on

**An acquisition change shows in noise-only (random-trigger) records; a
physics change cannot.** Σ-type cells move the noise-only z-variance ratio,
residual PSD or residual channel correlation off their calibrated null;
event-type cells leave all three exactly unchanged. Within the Σ side, a
consistent mean shift separates structural N from covariance-type Σ; within
the physics side, the out-of-span fraction of the paired change separates
support shifts (abstain) from supported-but-rare physics (recalibrate the
head). The designed families are recognised by isotropy: random per-event
directions with a large per-event alarm and a small mean shift.

## Verified on the linear subject (see `results/latent_monitor_tier1/table.md`)

13 of 14 cells attribute as predicted at C=8/N=256 and at three other
seeds/sizes; the fourteenth is documented, not wrong: **timing jitter on a
trace whose noise has a shared cross-channel component decorrelates that
component**, so its noise-only signature is a covariance change — N by
contract, Σ-covariance in the latent. Three findings the plan must carry:

1. Re-whitening restores the noise-only variance ratio to 1.00 on every
   Σ-covariance cell and leaves z on a pure signal unchanged to 0.1% — with
   the decoder and head untouched — but the consequence cost of a κ≈5–12
   mismatch was already small for a linear subject. The *alarm* is where κ
   shows; the consequence is second-order.
2. For the linear subject a granularity change is a **gain** on z: the
   consequence is repaired by refitting the output head (→0.99) or the
   channel stage, and *not* by the three geometry-pooling weights. The plan's
   G-row repair ("embedding + pooling only") is therefore a prediction about
   nonlinear subjects, not a general fact.
3. Activation patching is flat (1.0 at every stage) for every input-side
   corruption: the linear subject has no stage that creates damage. C5 is a
   question for the transformer.

## Two constraints inherited from the arms plan

The realized-covariance estimator has a floor set by N/C (≳500 for κ_floor
≲ 1.1), and one covariance must span every record of a cell —
`MultiChannelNoiseGenerator(..., freeze_channel_structure=True)` (WP-N1).
`tier1.Cell` does both.
