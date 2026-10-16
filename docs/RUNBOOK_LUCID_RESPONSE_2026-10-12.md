# Runbook — LUCiD arm with detector-response N (supersedes `RUNBOOK_LUCID_2026-09-24.md`)

> **16 October 2026 update:** the historical A0 research-use block and email request below are superseded by [Junjie's clarification](https://github.com/seanxia8/VERITAS/issues/5#issuecomment-6133100584). Research use is clarified; licence-file/redistribution status remains separate. Current response/stage and engineering status: [PMT source contract](PMT_SOURCE_AND_STAGE_CONTRACT.md) and [PMT issue status](PMT_ISSUE5_STATUS.md). Old toy results and threshold-walk semantics remain historical evidence.


_12 October 2026. The 24 Sep runbook built the arm around front-end **electronics** noise; that
part is frozen (`RUNBOOK_RESCOPE_N_2026-10-12.md`, R-1). This runbook carries the arm with the
**detector-response** N families, a declared readout chain, and an independent consequence
variable K. Every command runs from the repository root with `PYTHONPATH=src`._

| step | what | status (12 Oct) |
|---|---|---|
| 0 | Gate A0 (LUCiD licence) | **open** — upstream `CIDeR-ML/LUCiD` @ `eaf2e89` (6 Oct) still ships no licence; email draft in Appendix A |
| 1 | Readout chain + response cells | **built** — `readout.py`, `dataset.response_cells` / `run_response_matrix` |
| 2 | Consequence variable K | **built (draft)** — `consequence.py`, two endpoints; κ_m `provisional_dev` (D-K) |
| 3 | Development run on the toy source | **done** — `results/lucid_response_toy_dev_2026-10-12/` (non-citable) |
| 4 | LUCiD photon adapter | to build **after A0** |
| 5 | Monitored subject on digits; baseline-only training, ≥ 3 (target 5) seeds | to build (GPU) |
| 6 | Claims: test-time response cells on the frozen subjects | after 5 |

---

## 0. Gate A0 and what it blocks

LUCiD has no licence (`README.md`: "The license is being finalized and will be added shortly";
`CONTRIBUTING.md` defers to it). Re-checked 12 Oct on the upstream `main` (`eaf2e89`, 6 Oct).
Until a permissive licence lands:

- **allowed:** everything in `noise_module_lucid` (ours, MIT, no LUCiD import), the toy source, the
  readout chain, K, and development runs on toy photons;
- **not allowed:** a `src/lucid_simulation` package, any test or committed result that imports LUCiD,
  any released dataset generated with LUCiD.

**Inconsistency to resolve (author):** `EXPERIMENT_DESIGN.md` §II.3.2 says "no LUCiD work starts
before a permissive licence"; the 24 Sep runbook §0 allowed internal production and analysis
without release. This runbook follows the stricter reading.

The maintainers per `CITATION.cff` are Omar Alterkait (contact listed: omar.alterkait@tufts.edu),
César Jesús-Valls, Ryo Matsumoto, Patrick de Perio and **Kazuhiro** Terao (earlier docs said
"Kensuke" — corrected). Terao's group is at SLAC, so Junjie is the natural second channel.

## 1. The readout chain (what the N families act on)

`noise_module_lucid.readout.readout_chain(photon_sensor, photon_time, n_channels, params, seed)` →
per-PMT digits `(sensor, charge, time)`, the observable LUCiD's own reconstruction uses.

| stage | parameter (`ReadoutParams`) | family that moves it |
|---|---|---|
| 1 efficiency (per-photon binomial) | `survival` | `efficiency_loss` |
| 2 digitisation (count, first arrival) | — | — |
| 3 charge `n + √n σ z` (one fixed `z` per channel) | `spe_sigma` | — (reference) |
| 4 gain | `gain` | `gain_drift` |
| 5 discriminator: leading edge at `threshold_pe`, walk corrected with the curve calibrated at `walk_calibration_pe` | `threshold_pe` | `threshold_walk` |
| 6 timing calibration | `timing_offsets_ns` | `timing_drift` |
| 7 dead channels | `dead_channels` | `module_loss` |

Rules: a cell moves **one** parameter (`readout.with_change` refuses two); random draws come from
per-stage generators seeded by the event id, so every cell is paired with the reference event by
event; the walk-correction curve is a training-time constant, so the reference has zero residual
walk and a threshold or gain change leaves a charge-dependent residual (and, at a fixed-mV
threshold, a gain drop also loses small hits).

Cells: `dataset.response_cells(n_channels, group_size=64)` — reference + every family at every
development severity of `response.RESPONSE_FAMILIES`. Drifts act coherently per crate on every
other crate (so they cannot be absorbed by a global scale); threshold walk is detector-wide;
module loss kills `k` crates spread over the detector. Origin `N_struct`; reference `clean`.

## 2. The consequence variable K (draft, D-K)

`noise_module_lucid.consequence`, independent of the monitored model and of LUCiD's `recon.py`:

- **vertex**: `fit_vertex` — point-source time-of-flight fit (profiled early-edge `t0`, Cauchy loss
  for late light, √charge weights, coarse grid + Nelder–Mead); `err_v = |v̂ − v|` in metres;
- **energy**: `E_hat = c · Q_total` with `c` fitted on the **reference cell only** (training-time
  energy scale); `err_E = |E_hat − E| / E`;
- `K_x(cell) = mean_e err_x(cell) / mean_e err_x(reference)` over the same events, paired bootstrap
  95 % interval; draft primary **`K = max(K_vertex, K_energy)`**;
- κ_m = **1.10** `provisional_dev` (the Tier-1 controlled-tier convention mirrored). To declare it,
  the author picks either the convention (as Tier 1) or a physics requirement (a published vertex /
  energy-resolution requirement of the target experiment) — `PREREGISTRATION.md` §8 item 17.

Why two endpoints: on the toy, timing drift moves only the vertex, gain drift and efficiency loss
only the energy scale, and threshold walk and module loss both — a single endpoint would hide
families. κ_m semantics: harm when either endpoint's mean error exceeds the reference by 10 %;
since K_energy ≈ |scale bias| / resolution, the same bias is "more harmful" at better resolution —
part of the D-K decision. `run_response_matrix` also reports a dead-map-corrected energy endpoint
(`K_energy_deadcorr`, `K_deadcorr`) for D-K 17(c).

## 3. Development run on the toy source (done)

```bash
PYTHONPATH=src python -m noise_module_lucid --response-toy --out runs/lucid_response_toy \
    --n-events 200 --toy-photons 800
```

`toy.py` is a 482-PMT WCTE-scale cylinder with an isotropic point source — **not** LUCiD, **not** a
detector simulation, non-citable. Result: `results/lucid_response_toy_dev_2026-10-12/README.md`.
Every family moves a physical endpoint. Only the timing and threshold grids straddle κ_m; the
gain/efficiency grids are harmful at every point (they need sub-percent severities once the real
resolution is known); module loss is non-monotone after the dead-map correction. No monitor was
trained.

## 4. LUCiD photon adapter (after A0)

`run_response_matrix` needs events with `event_id`, `vertex`, `photon_sensor`, `photon_time`,
`true_energy`. Build `lucid_events(...)` that runs LUCiD in per-photon mode (detected photons with
sensor index and arrival time, **before** LUCiD's own digitiser — our readout chain replaces it,
otherwise charge smearing and the discriminator would be applied twice) and wraps them in that
interface. Record the LUCiD commit, geometry hash and placed-sensor count in provenance. Then
re-run §3 on LUCiD photons and compare with the toy table (parity on directions, not values).

Physics sources: the isotropic flasher first (LUCiD's default dataset source; vertex is defined);
muon/electron tracks via the PhotonSim path next — for tracks the point-source K measures a
light-weighted emission point, which is still a fixed endpoint, but declare it as such.

## 5. Monitored subject (GPU)

- **Input:** per-PMT digits as a `(n_channels, 2)` tensor (charge, time; zeros for unhit PMTs) —
  the electronics waveform is no longer the substrate. Write the digits → tensor adapter.
- **Targets:** vertex (3) and energy (1) → declared `y`, hence `J_y` and `M_task`.
- **Training (R-4):** on the **reference cell only**, ≥ 3 (target 5) model seeds; the seeds are the
  outermost resampling level. No per-variant retraining.
- **Freeze and wrap** with `latent_monitor.torch_subject` as in the 24 Sep runbook §6; gate: beats a
  trivial baseline on vertex+energy, hooks finite and stable on the reference cell.

## 6. Claims on the frozen subjects

Apply every response cell at **test time** to the frozen subjects. Claim 2: rank cells by K ≥ κ_m
with the task-sensitive score against the committed generic score. Claim 1: N (these cells) versus
S (material WbLS vs water, PhotonSim particle types) on the hard-matched set. Expectation recorded
before running: gain drift and module loss may be visible to the generic arm (charge histogram, hit
count), so Claim 1 may be null on this arm — that is a reportable control result (D-L).

## Appendix A — gate A0 email (draft, to send from your own address)

> **To:** Omar Alterkait <omar.alterkait@tufts.edu>; cc Kazuhiro Terao (via Junjie Xia if no direct address)
> **Subject:** LUCiD licence — permission to use it in a detector-diagnostics study
>
> Dear Omar, dear Kazu,
>
> I'm a PhD student at KIT (ETP), working with Junjie Xia on a study of failure diagnostics for
> learned reconstruction in particle detectors. LUCiD is the only water-Cherenkov simulator we
> found with per-PMT outputs and exact event pairing across geometries, and we would like to use
> it as one of our simulation arms, citing arXiv:2602.24129.
>
> The repository currently has no LICENSE file (the README says the licence is being finalised),
> so we can't build on it or release derived datasets yet. Would you be willing to add a permissive
> licence such as MIT or BSD-3-Clause? If you have a different licence in mind, a short note on
> what it will be would already help us plan.
>
> We only use LUCiD's photon output; our readout and noise models are a separate MIT package that
> depends on nothing in LUCiD. We are happy to share results or contribute back.
>
> Many thanks and best regards,
> Dowling Wong
