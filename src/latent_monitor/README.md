# `latent_monitor` — controlled-variable latent monitoring

Implements `docs/LATENT_MONITORING_PLAN_2026-09-05.md` §§2–4: the `Subject`
protocol, the reference-cell projectors, per-event Δz statistics against
paired twins, the pre-registered attribution lookup, abstention, and the
adjustments that follow from a diagnosis. Follows the `oracle_cov`
`Subject` interface (`represent` / `outputs` / `jac_recon` / `jac_output`)
so it can be lifted into `noise-weighted-subspace-reconstruction` later.

```bash
PYTHONPATH=src python -m latent_monitor.run_table --out results/latent_monitor_tier1
PYTHONPATH=src python -m pytest src/latent_monitor/tests -q      # 35 tests, ~20 s CPU
```

## What is in here

| module | what |
|---|---|
| `whitening.py` | `KroneckerWhitener` — Σ̂ = Σ_c ⊗ Circ(S(f)) as a named, replaceable parameter; `for_channels(C)` carries the correlation *assumption* to another sensor count; `estimate_kronecker` from noise-only records; κ(Σ̂⁻¹Σ) by Kronecker factors |
| `subject.py` | the `Subject` protocol and the six hooks `whitened → channel → token → z → pre_output → output` |
| `linear_subject.py` | the analytic subject: PCA encoder, geometry-weighted pooling, decoder, least-squares head; `with_sigma_hat` is the whitening lemma as an adjustment (GLS re-derivation, decoder untouched); `refit_stage` = stage-restricted LoRA |
| `torch_subject.py` | S1: the compact transformer wrapped as a `Subject` — whitening in front, an explicit position embedding (zero at init), a ridge *probe* decoder, autograd `jac_output`, exact self-patching of pooled hooks |
| `tier1.py` | paired Tier-1 cells from `noise_module`: reference; Σ-covariance (corr ↑/↓, bandwidth, line); Σ-structural (gain drift, channel loss, jitter); geometry (½C, 2C on the same box); event (out-of-span glitch; in-span oscillation, double pulse). Every cell also carries noise-only records |
| `reference.py` | `fit_reference`: P_out/P_null from J_o, P_exc/P_unexc from the pullback Fisher, null distributions from ref-vs-ref twins and noise-record halves |
| `statistics.py` | per-cell statistics: mean-shift norm (null metric), per-event alarm, energy splits, noise-only z-variance ratio, residual PSD (smoothed and single-bin) and channel-correlation shifts, out-of-span fraction, layer profile, consequence, conditional-on-alarm AUROC, abstention rate |
| `lookup.py` | `calibrate` (thresholds fixed once from the reference null) and `attribute` — the rule order *is* the decision procedure |
| `designed.py` | the output-null / output-aligned dissociation, exact for a linear decoder |
| `adjust.py` | `rewhiten`, `activation_patch`, `damage_patch`, `refit_stage` |
| `run_table.py` | the whole §1 table + adjustments → `table.json`, `table.md`, `adjustments.json` |

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
