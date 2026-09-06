# Results — the §1 table on the linear subject, Tier-1 cells (6 September 2026)

_Implements `LATENT_MONITORING_PLAN_2026-09-05.md` WPs N1, L0, S1, H0, H1. Code:
`src/latent_monitor/`, `src/herald_simulation/`, `noise_module.tes_budget`,
`MultiChannelNoiseGenerator.freeze_channel_structure`. Numbers below are from
`results/latent_monitor_tier1/table.md` (C=8, N=256, k=6, seed 0) and were
reproduced at seeds 1–3 and at C=6/N=128 and C=16/N=512._

## The table holds: 13 match, 1 documented, 0 mismatch

| cell | attributed | mean-shift | noise-only z-var | out-of-span | layer peak | consequence |
|---|---|---:|---:|---:|---|---|
| Σ-cov: corr ↑ / ↓ | sigma_cov | 0.7 / 1.1 | 1.67 / 0.32 | 1.00 | whitened | ≈1 |
| Σ-cov: bandwidth ↓ | sigma_cov | 0.3 | 1.09 (PSD dev 0.43) | 1.00 | channel | ≈1 |
| Σ-cov: line pickup | sigma_cov | 0.4 | 0.91 (single-bin dev 7.0) | 1.00 | whitened | ≈1 |
| Σ-struct: gain drift | sigma_struct | 649 | 0.88 (chan-corr 0.38) | 0.98 | channel | 1.05 |
| Σ-struct: channel loss | sigma_struct | 2604 | 0.56 | 0.95 | channel | 1.65 |
| Σ-struct: timing jitter | **sigma_cov** (documented) | 1.4 | 0.55 | 1.00 | whitened | ≈1 |
| G: 4 ch / 16 ch | geometry | 2950 / 2545 | 0.90 / 0.92 | 0.12 / 0.06 | token | 1.8 / 1.6 |
| E: glitch (out-of-span) | event → abstain | 602 | 1.00 | 0.84 | channel | 1.10 |
| E: oscillation, double pulse (in-span) | event_in_span | 950 / 9413 | 1.00 | 0.17 / 0.02 | channel / whitened | 1.76 / 4.58 |
| designed output-null | output_null | 9.5 | 1.00 | 0.00 | — | 1.00 |
| designed output-aligned | output_aligned | 10.2 | 1.00 | 0.00 | — | 3.66 |

Thresholds were calibrated once on the reference null (q99) and never
re-tuned: mean-shift 5.44, per-event alarm 4.53, z-variance band 0.85–1.15,
smoothed PSD deviation 0.17, single-bin 0.29, channel-correlation 0.13.

## What the table taught, beyond confirming the plan

**The N-vs-S discriminator is noise-only records.** Every Σ-type cell moves
at least one noise-only statistic off its null; every event-type cell leaves
all three at exactly their reference value (variance ratio 1.000, PSD and
channel-correlation deviation 0.000). A random trigger cannot see physics.
This is simpler and more robust than any latent-side statistic, and the
lookup is built on it.

**Jitter that decorrelates a shared component is a covariance change.** The
plan's structural-N row assumed a mean-shift signature. With a shared
cross-channel noise component, per-channel timing jitter destroys the
correlation and shows as noise-only variance 0.55 with no consistent mean
shift — Σ-covariance in the latent, N by contract. The 2026-09-02 Tier-1
review already said N is not one signature; this is a second instance.

**Supported-but-rare physics is its own row.** The oscillation and
double-pulse families are *inside* the span the encoder excites: z moves a
lot, the residual and every noise-only statistic do not, and the consequence
rises (up to 4.6×). The representation is intact; the head is not. The
lookup names this `event_in_span` with the adjustment "recalibrate or extend
the output head" — cheaper than abstaining and different in kind from a
support shift, which the out-of-span fraction (0.84 vs 0.02–0.17) separates.

**Re-whitening is the whitening lemma, not a layer swap.** Replacing Σ̂ while
keeping the encoder fitted under the old Σ̂ multiplied the consequence by
~4–5. The correct adjustment keeps the raw signal basis S = W⁻¹D and
re-derives the encoder as the GLS projection onto S under Σ̂′ (temporal GLS
per channel, GLS channel weights Σ_c′^{-1/2}·1, gain matched). Done that way it
restores the noise-only variance ratio to 1.00 on all four Σ-covariance cells
and leaves z on a pure signal unchanged to 0.1%, with decoder and head
untouched. The κ ≈ 5–12 mismatches themselves cost the linear subject almost
nothing in consequence — the alarm is where κ shows.

**Geometry repairs through the head, not the pooling.** For the linear
subject a granularity change is a gain on z: refitting the output head brings
the consequence from 1.8 to 0.99, refitting the channel stage does the same,
and the three geometry-pooling weights cannot (1.8 → 1.8–1.9). The plan's
G-row repair is a prediction about nonlinear subjects.

**Patching is flat.** Substituting the clean stage into the perturbed pass
recovers 100% at every stage for every input-side corruption, and the reverse
patch transmits 100%. The linear subject has no stage that creates damage;
C5 is a transformer question.

## Two implementation facts that would otherwise poison a real run

- A frozen channel structure is required for one Σ̂ to span a cell
  (`freeze_channel_structure=True`; pooled κ converges only then).
- `reconstruction_model.models.current_compact.AbsolutePositionalEmbedding`
  created its parameter with `torch.empty` and never initialised it, so a
  freshly `init_weights()`-ed model started from garbage (|pos_embed| ~ 1e31,
  training diverged nondeterministically). Fixed to N(0, 0.02) like every
  other model in the package; checkpoints are unaffected.

## HeST arm: the chain runs end to end

`herald_simulation` builds the reference cell (HeRALD_v1, 24 CPDs, ER 1 keV,
TES_HERALD_V1) and thirteen one-factor cells — 24→1 and 2-sensor geometries,
four Σ-covariance, three structural, three event-type, one WIMP U — all paired
by `event_id` (initial quasiparticle population bit-identical across
geometries, tested). Four events per cell at 1% QP thinning: 34 s total on two
CPU cores. The matched-cell κ floor at N/C = 683 is 1.18, as the arms plan
predicted. Constants are placeholders until read from arXiv:2307.11877.

## Not done, and why

- **LUCiD** — gated on the licence (A0); `noise_module_lucid` is a preset plus
  a units bridge once it lands.
- **TIDMAD WP10** — needs the data and a GPU.
- **The transformer table** — `TransformerSubject` wraps the compact model,
  passes the protocol end to end on CPU, and refits its explicit geometry
  stage; running the §1 table on a *trained* transformer is the next
  experiment, not a smoke test.
