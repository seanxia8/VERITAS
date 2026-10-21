# JEPA study — one-page report for Junjie (21 October 2026)

_Status: development only. No claim comparison has run. This note deliberately reports **one method line**; the full pre-registration is [RUNBOOK_JEPA_NOISE_v4_2026-10-16.md](RUNBOOK_JEPA_NOISE_v4_2026-10-16.md) and the verdicts are in [PREDICTION_STATUS.json](PREDICTION_STATUS.json)._

## 1. The question

Can a predictive self-supervised representation (JEPA-style) **prioritise shared nuisance over physics** — here, module-gain drift shared between two views — and can a **noise-only anchor** remove that nuisance without losing physics? This is a representation-methodology study, not a detector-physics measurement.

## 2. The method (one method, not a menu)

**Primary subject: SIGReg-JEPA on LUCiD WCTE.** One data generator, one encoder, one probe protocol.

- **Event domains:** isotropic **photon bombs** and, per your 21 October message, **muon/e- particle bombs** with real Cherenkov directionality. Mixed training, per-domain probes; the photon bomb is a simplified isotropic domain, not literally a subset of a Cherenkov event.
- **Input:** the 1,995 WCTE PMTs, one token per PMT: position/direction (fixed scaling), log charge, first-hit time relative to the NHits trigger, hit flag. Prompt-only transport, uniform angular acceptance, no scattering/reflection/absorption beyond attenuation.
- **Model:** serialised point transformer, 4 blocks, d = 128, 4 heads, window attention over 64 tokens; predictor 2 blocks, d = 64; objective = prediction loss + SIGReg.
- **Treatment (the only two contrasts in the main line):**
  1. **view split: `within` vs `across` modules** (R6) — does the split decide whether shared module gain is learned?
  2. **noise anchor vs plain** (R7) — the anchor overlays an independent dark-noise-only window and a 1% per-module charge rescale, with β chosen on the development cohort (β = 1000). It is the **only remedy** in the main line.
- **Readout:** frozen encoder, run-mean ridge probes; the propagated probe for R6; vertex resolution from the frozen VERITAS time-of-flight fit for R7; every score is compared with a **random-init floor**.

**Everything else is a control, not a method:**

| Role | Item |
|---|---|
| Baselines | `mae_mse`, `mae_chi2` (noise-model weighted reconstruction) |
| Ablation | `ijepa_ema` (EMA target instead of shared weights) |
| Control | `anchor_only` (SIGReg + anchor, no prediction loss) |
| Oracle upper bound | `oracle_nominal_twin` (nominal-gain twin readout) |
| Theory | E1 linear population checks (CCA / GEVP / conditional oracle) — motivation, not a separate method |
| Optional extensions | E3 (TIDMAD real noise), E4 (fault monitor) — not part of the main claim |

## 3. What we have run and found so far

- **E1 (analytic/linear):** R1/R2/R4 fail, R3b passes, R3/R5 not evaluable. Shared noise can displace CCA; finite-β and deep-network recovery are not guaranteed.
- **E2 bridge (B1):** **fail** — anchored vs plain nuisance R² differs by 0.0094, below the 2 × seed-floor threshold 0.1025. The theory-to-transformer bridge is not established.
- **LUCiD development cohort:** β table frozen (anchored β = 1000; vertex 0.208 m vs plain 0.324 m, reference 0.142 m). **Dev floors show every arm below the random-init module-gain floor** (0.39438): plain −0.004, `mae_mse` 0.315 — at development scale JEPA does **not** learn F1 module gain, so R6/R7 may be null.
- **Full grid:** not yet run. The first launch hit a harness bug (a device mismatch when evaluating the random-init floor); training and probes themselves completed. Relaunch after the one-line fix.
- No claim comparison has run; all numbers above are development evidence.

## 4. What we need from you (physics, not method)

1. **Drift values.** F1 module gain s = 2/5/10%; F3 per-PMT gain s_p = 5–10%; F2 attenuation a = 0.50; aging 10% over 100 runs; gains clipped to [0.5, 1.5]. Are these realistic for WCTE, or should a different range be used?
2. **Particle bombs (your 21 October message).** We will add **both μ⁻ and e⁻** domains, not just muons. They add directionality and the Cherenkov cone, while the photon bomb only tests isotropic light and ToF. Which energies and topology (contained vs through-going) should we use for each species, and do you have PhotonSim/Geant4 samples you want us to use? At JEPA scale we plan a hybrid: fast analytic Cherenkov production for the grid, validated against real PhotonSim events; at present we have 500/1000/1500 MeV μ⁻ PhotonSim samples (20 events each) and no e⁻ generator.
3. **Mixing.** Mixed training with per-domain probes (our preference), or pretrain on photon bombs then fine-tune on particle bombs? We will register the dataset/target change as an A10/G5 amendment before running.
4. **Scope.** Is it right to frame the paper as representation methodology with E3 (TIDMAD) and E4 (fault monitor) as optional extensions? If the full grid is null, we will report it as a pre-registered null result, not retune.

## 5. Limits

Development only; synthetic response values; no detector calibration; a null result is possible and will be reported. E1 is analytic; E2/E3/E4 are empirical. No claim comparison exists yet.

---

### Short reply to your 21 October message

> Hi Junjie,
>
> Thanks — this is very helpful, and I agree. The photon bomb only tests isotropic light and ToF; a μ⁻ or e⁻ event adds directionality, the Cherenkov cone and particle physics, which is exactly what we want the representation to preserve. We will add **both μ⁻ and e⁻**, keeping it as **one method with three event domains**, not three methods: same encoder/objective/probe, mixed training with class-balanced sampling, and per-domain probes (vertex/yield for photon bombs; vertex/direction/deposited energy for μ and e; particle type across μ/e). The photon bomb is a simplified isotropic domain rather than a literal subset of a Cherenkov event, so we report per-domain results and the mixed result.
>
> Practical questions: (1) energies and topology for μ⁻ and e⁻ (contained vs through-going); do you have PhotonSim/Geant4 samples, especially e⁻? (2) mixing — mixed training with per-domain probes, or pretrain then fine-tune? I lean to the former. (3) the drift values s/s_p/a are still pending. We will register the dataset/target change as an A10 amendment before running.
>
> We have 500/1000/1500 MeV μ⁻ PhotonSim samples from the PMT work (20 events each) but no e⁻ generator; at JEPA scale we plan hybrid analytic production validated on PhotonSim.
>
> Thanks,
> Dowling
