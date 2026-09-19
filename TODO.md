# ORACLE — root TODO: testbeds & experiment workstreams

_Created 18 September 2026 from the working note. This is the **testbed /
experiment** board. The **paper and protocol** board is `docs/TODO.md`; the
canonical design is `docs/EXPERIMENT_DESIGN.md`, the canonical inventory
`docs/TESTBEDS.md`. Do not duplicate the protocol items here._

Priority order this pass: **1) LUCiD reading + noise module**, **2) FASER
feasibility**, **3) XLZD feasibility**, **4) CMS+ATLAS feasibility**.

Status vocabulary: *not started* · *reading* · *feasibility* · *design* ·
*building* · *built* · *validated* · *blocked*.

---

## 1. LUCiD + customised noise module  ← top priority

Reference: `docs/EXPERIMENT_DESIGN.md` §II.3, §II.7, §III.6;
`docs/TESTBEDS.md` §1.1; `docs/reviews/LUCID_NOISE_REVIEW_2026-09-11.md`;
new package `src/noise_module_lucid/`; explanation `docs/noise_module_lucid.md`.

### 1.1 Reading
- [ ] Read the LUCiD forward model end to end: geometry (`lucid/geometry`),
      sources, `sensor_response.build_make_hits_waveform`, `digitizer` (QE, SPE,
      TTS, dark noise, TDC), `fitting/` (recon, Fisher).
- [ ] Read the noise-relevant LUCiD internals and record, per term, whether it
      is a **signal-carrier statistic** (inside LUCiD) or **acquisition noise**
      (ours to add). Summary goes in `docs/noise_module_lucid.md` §2.

### 1.2 Licence / gate A0
- [ ] Email Kensuke Terao and Omar Alterkait: add MIT or BSD-3 (`CITATION.cff`
      and `CONTRIBUTING.md` exist; likely an oversight). Re-check upstream.
- [ ] No `src/lucid_simulation` package and no LUCiD-dependent test before A0.
      The noise package itself has **no LUCiD dependency** and may be built now.

### 1.3 Geometry by JSON
- [ ] Adopt LUCiD's four-key geometry JSON as the arm's geometry contract.
- [ ] Record placed-sensor count (not requested), `apply_translation=False`,
      `max_candidates_per_ray` assertion (`TESTBEDS.md` §1.1 traps).
- [ ] Define the covariance unit: crate / string / angular-sector × height band
      (16–64 PMTs). `src/noise_module_lucid/grouping.py`.

### 1.4 Noise module (build)
- [ ] `src/noise_module_lucid/` package: `presets.py`, `units.py`, `grouping.py`,
      `adapter.py`, `interventions.py`, `tests/`, provenance record.
- [ ] Preset V2 for the 512 ns grid (done in the notebook helper, promote it).
- [ ] **Re-parameterise the preset for the 16–32 µs `window_ns`** the κ cells
      need (df 30–60 kHz): DC-DC switching lines 100 kHz–2 MHz, three decades of
      1/f, long-cable reflection comb. This is the main open physics decision.
- [ ] Matched-cell κ floor calibration per (C, N); report alongside every swept κ
      (N/C ≳ 500 for κ_floor ≲ 1.1).
- [ ] Acceptance per `EXPERIMENT_DESIGN.md` §II.7.6 (AC-1…AC-7).

### 1.5 Noise / background definitions to change or add (from the note)
Noise (acquisition, goes into Σ or a structural family):
- [ ] **Thermal noise** — PMT photocathode workfunction, auto/thermionic electron
      release (dark-count rate). Note: LUCiD models dark counts as *events*; the
      **rate fluctuation / electronics** part is ours. Decide the split.
- [ ] **Radioactive (radon in water)** — background *events* (never Σ). Rate,
      spectrum, spatial distribution; add as an S/U family, not noise.
- [ ] **Cable length → lagged signal** — a per-channel delay/reflection in the
      transfer function (`reflection`/`filtered`), structural N.
- [ ] **Electronics random sampling in ADC/TDC** — aperture jitter and
      quantisation (`noise_module.psd_resampling`, `artifact_injector`);
      a digitiser-contract N family.
- [ ] **PMT pre-/after-pulse** — out-of-trigger hit; in-gate contamination from a
      different event. Mostly IWCD / near detector with pileup. Model as a
      **background/event** family (pileup), not covariance.
- [ ] Clock/switching pickup, broadband common mode, alias fold, gain drift,
      channel loss (already in V2 / `interventions.py`).

Physics / background (events, `z_shape`/S families):
- [ ] **Diffusive supernova neutrino background** as signal (DSNB).
- [ ] **Atmospheric neutrinos** — higher-energy tail leaking into the signal
      band (Cherenkov radiation). Background event family.
- [ ] **Neutrino–nucleus interaction channel** — QE vs DIS. Physics S family.

### 1.6 Notebook
- [ ] Update `notebooks/noise_models_herald_lucid.ipynb` (builder
      `notebooks/_build_nb2.py`) to import `noise_module_lucid` instead of the
      local `pmt_frontend_v2.py` helper; re-run against a LUCiD clone so the
      stored outputs show V2.
- [ ] Update `notebooks/one_event_herald_lucid.ipynb` (builder `_build_nb1.py`)
      the same way.
- [ ] Fix the hard-coded builder output path (`/home/claude/...`).

---

## 2. FASER (feasibility check)

- [ ] Literature: FASER and FASERν; the emulsion detector and the
      scintillator/tungsten spectrometer readout; what public data/simulation
      exists.
- [ ] Does it fit the ORACLE admission criteria (`DATASET_STRATEGY` §3)?
      Emulsion plates are **offline, integrating, no waveform** — likely a
      **hit/image-level** arm, not a trace arm.
- [ ] **Multichannel noise?** Check whether there is a per-channel electronic
      readout with a covariance structure (scintillator/calorimeter channels)
      or only emulsion images.
- [ ] Verdict: claim-bearing / supporting / future work; one paragraph + a
      `docs/TESTBEDS.md` row if it survives.

## 3. XLZD (feasibility check)

- [ ] Target physics: **0νββ** (and dark matter / solar ν).
- [ ] Readout: PMT cable + electronics noise; **cryogenic**; **superinsulation**;
      **skin PMT** (E-field boundary). Decide which are N (acquisition), which
      are G (geometry/field), which are events.
- [ ] Feasibility against §3 admission criteria; is there an open simulation or
      an open trace dataset (cf. the LZ/XENON "would not serve" rows in
      `docs/TESTBEDS.md` §4.2 — re-check, XLZD is newer)?
- [ ] Verdict + `TESTBEDS.md` row.

## 4. CMS + ATLAS (feasibility check)

Documented in `docs/COLLIDER_ARM_2026-09-18.md` (Tier 3b). Feasibility tasks:
- [ ] **Detector card** path: Delphes cards, one shared HepMC sample through
      `ATLAS` / `CMS` / `CMS_PhaseII` / `CLICdet` / `ILD` / `IDEA`; confirm the
      exact card set and pairing.
- [ ] **Full sim** path: Key4hep/DD4hep detector concepts (CLICdet → CLD) with a
      shared generator file; budget the GEANT4 run.
- [ ] **Open dataset** path: ATLAS Open Data (2025 beta, record 93910; event-gen
      batch 160000) and CERN Open Data (CMS 2010–2015); confirm schema, licence,
      level (reco/object, not raw traces).
- [ ] Prior work to cite: `2606.14373` (MLPF latents feed downstream tasks),
      `2604.12364` (jet→neutrino FM transfer), `2512.00187` (cross-geometry
      shower transfer), `2305.11531` (GAAMs), `2503.00131` (MLPF CLICdet→CLD),
      `2608.15166` (DANTE detector domain shift), `2608.18190` (nuisance vs
      physics shift), `2609.00611` (Panda V2), `2510.24066` (OmniLearned),
      `2605.29283` (fmbench). Position against: `2501.13789` (CMS ML DQM),
      `2309.10157` / `2407.20278` (CMS ECAL autoencoder AD).
- [ ] Verdict: G-transfer only, not claim-bearing (no controllable acquisition,
      no paired N counterfactual).

---

## 5. Other testbeds (status, lower priority)

### 5.1 Synthetic — ORACLE-Cov (Tier 1)
- [x] Linear-subject signature table (13/14). See `docs/TODO.md` for the
      trained-subject and Phase-B items.

### 5.2 HeST + homebrew QP simulator + noise (demoted to optional case study)
- [ ] **Validity caveat:** TESSERACT / SPICE-HeRALD upstream; authors are grad
      students; the trace, electronics, noise and harm proxy are project-authored.
      Keep as a case study, not claim-bearing (`DATASET_STRATEGY` §8).
- [ ] Postdoc email sent — track reply; gate B0 (copyright line) retired with the
      arm.
- [ ] Replace `HERALD_V1_PLACEHOLDER` with `from_paper` values only if revived.

### 5.3 NuRadioMC / NuRadioReco + customised noise module (Tier-2 arm B)
- [ ] 3-day pilot (4 cells, 1–5k events): identical truth across detector JSONs;
      clean/noisy traces + independent reconstruction K export; hard-contrast
      overlap; bounded compute (`DATASET_STRATEGY` §5, §10).
- [ ] Build `src/nuradio_simulation/` adapter with native and controlled noise
      modes; per-group κ floor.

### 5.4 Prometheus (backup)
- [ ] Keep as the LUCiD fallback with a waveform bridge; frozen DynEdge public
      model, angular-error K. D5 re-inference offset must resolve first.

### 5.5 TIDMAD (自带模型 / own released model) and CRESST
- [ ] TIDMAD: two-Σ̂ training contrast on real noise; `K_rel` must survive
      scrutiny. Code exists; needs data + GPU.
- [ ] CRESST: real TES pulse data, per-detector random-trigger noise; confirm
      licence/DOI with the ORIGINS DMDC before any dependency.

---

## 6. Cross-cutting

- [ ] Keep `docs/TESTBEDS.md` canonical: when an arm changes status, edit the
      tables there and cite the evidence file — do not add another survey.
- [ ] Keep the new `sources/` research notes indexed (`sources/README.md`).
- [ ] Do not add a fourth *claim-bearing* arm: "three is the number"
      (`docs/EXPERIMENT_DESIGN.md` §I.4).
