# Gate A10 — mixed event domains for JEPA E2 (photon, muon, electron) — G5 proposal

**Status: DRAFT — author decisions recorded; pending registration.** This is a G5 amendment proposal for the registered JEPA runbook
([RUNBOOK_JEPA_NOISE_v4_2026-10-16.md](RUNBOOK_JEPA_NOISE_v4_2026-10-16.md)). The author's
instruction of 22 October 2026 is: **both μ⁻ and e⁻ particle bombs are required**, option A is
accepted, and the §8 slots are decided below. A10 must be registered as a dated amendment before
any outcome-generating run uses it; the registered §3.5/§3.7 text remains authoritative until then.

**Authority trail.** Junjie's message of 21 Oct 2026 proposed muon/electron particle bombs on top of
the isotropic photon bomb (Cherenkov adds directionality and particle physics; he leans toward
mixing the datasets). The author accepted that on 22 Oct 2026 and required both species. This gate
records the design consequence; it does not claim Junjie approved the mixing ratio or the generator
mode.

---

## 1. What changes, and why this is G5

The E2 dataset is part of the registered scientific specification: event classes, probe targets,
predictions R6/R7 and the budget all change. Under runbook §5 a change to a registered dataset role,
prediction, metric or cell needs a **G5 amendment**, not an engineering decision. A grid launched on
the old photon-only dataset would be superseded by the mixed dataset, so **the full grid stays paused
until A10 is registered**.

**Not a method change.** The encoder, objective, anchor, controls and probes stay exactly those of
§3.5.6. Domains are a data factor:

```
one method      SIGReg-JEPA + noise anchor (§3.5.5–§3.5.6)
three domains   photon bomb (isotropic) | muon bomb (track) | electron bomb (shower)
one run         one detector state, shared by all domains in that run
one readout     per-domain probes, per-domain random-init floors, pooled secondary
```

## 2. Domain definitions

| Domain | Event | Physics truth targets | Nuisance targets |
|---|---|---|---|
| `photon` | isotropic point source, fiducial vertex, log-uniform yield (unchanged §3.5.3) | vertex, log yield | module gain (F1), per-PMT gain (F3), attenuation (F2), age (T) |
| `muon` | μ⁻ track; Cherenkov emission along the track, cone `cos θ_C = 1/(β n)`; energy and direction sampled per event | vertex, direction, deposited energy (contained primary case; escaping diagnostic), particle type μ | same four |
| `electron` | e⁻ shower; Cherenkov emission from the shower axis/profile; direction and deposited energy sampled per event | vertex, direction, deposited energy, particle type e | same four |

The photon bomb is a **simplified isotropic domain**, not a literal subset of a Cherenkov event: the
detector response is shared, the event physics is not. Results are reported **per domain** and
pooled only as secondary.

## 3. Run composition and class-balanced sampling

Each run has **one detector state**; every domain in the run uses that same state (so module gain,
per-PMT gain, attenuation and aging are seen identically by all domains).

| Set | Default per run | Note |
|---|---|---|
| training run (1000 events) | 334 photon + 333 muon + 333 electron (≈1:1:1) | the class-balanced default; exact per-run counts in config |
| probe-train run (50) | 17 + 17 + 16 | run-mean probes per domain |
| probe-test run (50) | 17 + 17 + 16 | disjoint runs from probe-train |
| robustness run (50) | 17 + 17 + 16 | s = 0.20 and aging extrapolation |
| dark-only / nominal twin | per run | shared by domains |

- **Class label is metadata only**; it never enters the model input. The model sees only measured
  PMT charge/time/position/occupancy.
- Batches are class-balanced by construction (fixed run composition); an exact per-batch
  stratification is optional if the development domain classifier shows imbalance.
- Changing the ratio is a further amendment. A 2:1:1 fallback (photon:μ:e) is allowed only if the
  generation-cost projection exceeds the cap, and is recorded.

## 4. Generator: two modes, one label

| Mode | What it is | Use |
|---|---|---|
| `analytic_cherenkov` | fast parameterised track/shower Cherenkov source in the JEPA generator; photon emission along the track with the Cherenkov angle, attenuation and geometry from the pinned LUCiD model | scale production (100k events), labelled `simplified_cherenkov` |
| `photonsim` | PhotonSim/Geant4 muon (and electron, if supplied) samples transported through the pinned WCTE geometry | validation subset and, if the budget allows, reported production |

**Recommendation: hybrid.** Production uses `analytic_cherenkov` with a mandatory validation subset
of real PhotonSim events: the generator must reproduce, within pre-declared tolerances, the PhotonSim
muon hit-count/charge/time distributions at 500/1000/1500 MeV. Electron production is analytic until
Junjie supplies PhotonSim e⁻ samples. No mode is silently mixed: every manifest records the mode,
and a combined dataset records both.

Existing data: PhotonSim water μ⁻ at 500/1000/1500 MeV, 20 events each (PMT work), usable for the
validation subset. **No electron generator exists in the repo.**

## 5. Probes and predictions (proposed amendment text)

- **Per-domain probes:** photon → vertex, log yield; muon → vertex, direction, deposited energy;
  electron → vertex, direction, deposited energy, particle type; nuisance → module gain, per-PMT
  gain, F2 variable, age. Run-mean primary, propagated for R6, raw-input ceiling, per-domain floor.
- **R6/R7 per domain.** Particle domain (muon + electron) is the **primary** physics-retention
  domain; photon domain is the isotropic control; pooled result secondary. Criteria otherwise
  unchanged (§3.5.9).
- **Optional new R13:** in the particle domain, direction and deposited-energy probes exceed their
  random-init floors by more than 2 × the seed floor. No direction/energy claim is made for the
  photon domain.
- R11/R12 may be extended to the particle domains; R1–R5 and B1 are unchanged.
- **Domain classifier diagnostic:** a classifier on frozen embeddings reports domain AUC; it is a
  diagnostic, never a criterion. A high AUC triggers a per-domain review, not a retune.

## 6. Detectability, budget, timeline

- **Detectability (§3.5.3a) re-run on the mixed sample**, per domain. If an F1 cell's run-level
  ceiling is < 0.6 in a domain, that domain is excluded from R6/R7; if no domain passes, G6 for the
  dependent predictions. F3/F2/T ceilings are re-recorded.
- **Training volume** is unchanged if mixed replaces photon-only (same 165 rung-1 runs); generation,
  storage and per-domain analysis increase. Re-project ledger + grid against the 160 h start rule
  and 170 h cap before relaunch.
- **Fallbacks, in order:** (1) 2:1:1 mixing; (2) electron domain deferred to a later phase; (3)
  E3/E4 dropped; (4) G2. Each step is recorded.
- **Timeline:** +1–2 weeks for generator, validation, mixed smoke, dev cohort and registration; the
  CPAL date is not moved silently. If the projection misses it, write a descoping gate.

## 7. Options considered

| Option | Content | Assessment |
|---|---|---|
| **A (selected)** | three domains, μ and e both, hybrid generator, particle primary | selected; strongest physics content; uses existing muon data; needs electron generator; costs generation |
| B | photon + muon only, electron deferred | cheaper, still directional; fails the author's "both" instruction |
| C | keep photon-only grid and add particle as a separate experiment | cleanest control, but two datasets/two papers; does not answer Junjie's point inside JEPA |

## 8. Author decisions (22 October 2026) and remaining values

**Decided by the author (22 Oct 2026):**

1. **Generator: hybrid** — analytic Cherenkov production for the grid plus a mandatory PhotonSim μ⁻
   validation subset.
2. **Electron source: analytic first**; replacing it with full PhotonSim e⁻ needs a later amendment.
3. **Mixing: 1:1:1** photon:μ:e (334/333/333 per 1000-event training run; 17/17/16 per 50-event
   probe run).
4. **Primary domain: particle (μ + e) primary for R6/R7**, photon domain the isotropic control;
   pooled result secondary.

**Still needed as values, not design choices:**

- Particle energy ranges and contained/through-going selection (defaults: μ⁻ 300–1500 MeV, e⁻
  100–1000 MeV log-uniform; energy probes on the contained subset).
- Drift values s, s_p, a from Junjie (separate from A10).

## 9. What A10 does not change

E1 and R1–R5, B1, the guard and shared-server rules, the no-touch packages, the class-label-free
input contract, the one-method framing, and the registered §3.5.6 arms/objectives.

## 10. Registration steps

1. The author confirmed the decisions in §8 on 22 Oct 2026; they are recorded in `DECISIONS.md` and
   in A10 §A10.8.
2. The runbook text in [AMENDMENT_A10_MIXED_DOMAINS_2026-10-22.md](AMENDMENT_A10_MIXED_DOMAINS_2026-10-22.md)
   is inserted as §3.8 and committed unchanged as
   `jepa: register runbook amendment A10 (author revision)`.
3. Configs are versioned (`lucid_bombs_v1_mixed.json`, `lucid_run_v0_mixed.json`), not edited in
   place; hashes recorded.
4. Detectability re-run on the mixed sample and frozen before any training.
5. Mixed 500-step smoke + development cohort; then launch the grid.

## References

- Runbook: [RUNBOOK_JEPA_NOISE_v4_2026-10-16.md](RUNBOOK_JEPA_NOISE_v4_2026-10-16.md) §3.5.3,
  §3.5.3a, §3.5.6, §3.5.9, §5.
- Predictions: [PREDICTION_STATUS.json](PREDICTION_STATUS.json), [PREDICTIONS.md](PREDICTIONS.md).
- Generator: `src/jepa_noise/lucid_bombs.py`; runner: `src/jepa_noise/lucid_run.py`.
- Diff plan: [DIFF_PLAN_A10_LUCID_2026-10-22.md](DIFF_PLAN_A10_LUCID_2026-10-22.md).
- Author note: [NOTE_FOR_JUNJIE_2026-10-21.md](NOTE_FOR_JUNJIE_2026-10-21.md).
