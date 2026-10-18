# PMT Decisions Log

## Author confirmations (runbook v2 §2)

### C1 -- Retroactive authorisation of GPU/remote/CERNBox runs

**Answer (17 Oct 2026):** "追认" (ratified) -- option **(A)**. The `181b747` / `56179eb` GPU, remote and CERNBox runs are accepted retroactively as **development evidence**. Their recorded verdicts stand, and nothing in them is a reported result.

### C2 -- Flags for the next execution

**Answer (17 Oct 2026):** "你的建议合理，采纳" (the recommendation is adopted) -- `PMT_REMOTE=yes`, `PMT_GPU=yes`, `PMT_DATA_CAP_GB=5`.

Authorisation relied on for remote/GPU work, quoted from the launch prompt `docs/AGENT_PROMPT_JUNJIE_PMT_v2_2026-10-16.md` (committed `c2b0a3a`, 18 Oct 2026): "PMT_REMOTE=yes # C2, author, 17 Oct 2026; within runbook v2 §2 shared-server rules", "PMT_GPU=yes # one Titan job at a time on Deepthought only (§2)", "PMT_DATA_CAP_GB=5".

---

## Author message, 18 Oct 2026 (verbatim)

> Good work. R-1 to R-3 and R-5 are accepted. Decisions below, then continue.
> Record this message verbatim in `docs/PMT_DECISIONS_LOG.md` (author, 18 Oct 2026).
>
> ## Author decisions
>
> 1. **SHA `0fba034` is accepted for the JEPA merge** (development). The JEPA study will merge it.
>    Reported JEPA runs will wait for the R-6 SHA below.
> 2. **Energy gate per subgroup:** apply it per subgroup as well. Energy MAE must be ≤ 0.8 × constant
>    pooled **and** ≤ 1.0 × constant in every energy subgroup, as for the vertex. A pooled-only gate
>    would hide the regression to the mean seen at 1000 MeV.
> 3. **§3 defaults:**
>    - D-Q, D-J, D-G and D-K are **adopted** as binding.
>    - For D-TQ, the functional form a + b/√q is adopted, but its values stay `synthetic_dev` until
>      Junjie gives a measured curve.
>    - D-E is pending Junjie. Use (A), deposited energy, as the development default, labelled as
>      such.
>    - κ_m = 1.10 stays `provisional_dev` until the WP6 freeze.
> 4. **`endpoint_identifiability.py`:** read it and report in five lines or fewer what it computes
>    and which run produced it.
>    - If it supports D-E (containment or identifiability of initial energy), commit it with a test
>      and label it `dev_unreviewed`.
>    - Otherwise leave it untracked and list it in the status file.
> 5. **E26:** fix it now. A threshold below 0.25 PE is counted, not raised, with a test.
>
> ## New repair R-6 (before R-4; needed by the JEPA study)
>
> The reference response has `dark_counts: off`. The JEPA study needs dark counts, both as private
> noise and for its anchor's dark-noise-only windows.
> - **If the readout already implements a dark-count stage:** expose the rate in `reference.json`
>   (`dark_rate_hz`, synthetic default 1 kHz per PMT, `synthetic_dev`) and add a public function that
>   generates a **dark-noise-only readout window** for a given seed and run.
> - **If it does not:** implement the stage in the readout. Dark hits are uniform in the readout
>   window, with single-photon charge and the same T-Q walk and timing width, under keyed RNG; stage
>   owner: readout.
> - **Tests:**
>   - with the rate at 0, output identical to the current reference;
>   - the dark-hit count matches rate × window × number of PMTs within 3σ;
>   - the dark-only window contains no source hits;
>   - determinism under the keyed RNG.
> - Commit, push, and **report the new SHA**. That is the SHA JEPA uses for reported runs.
>
> ## Then, in order
>
> 1. **R-4** under the C2 flags and the runbook v2 §2 shared-server rules.
>    - First the generation-only containment check: does any μ energy range in 100–400 MeV reach
>      ≥ 50% containment in WCTE?
>    - Then the re-run, with vertex and direction under the R-3 gates, and energy on the deposited
>      endpoint, labelled development.
>    - 3 seeds; data reads ≤ 5 GB.
> 2. **WP6:** freeze the experiment contract with everything adopted above. Cells that depend on D-E
>    stay `pending_D-E`.
> 3. **WP8:** draft the reply to Junjie in `docs/PMT_ISSUE5_REPLY_DRAFT.md`, not posted.
>    - It summarises what was fixed: the response manifest, the 1/√q walk, the honest gates with the
>      constant-baseline table, dark counts.
>    - It asks his open decisions as a numbered list: D-E; a measured T-Q curve; the target QE/CE
>      for the WCTE 3-inch PMTs; a crate/cable map; the dark rate; D-T/L/M/S.
> 4. **WP7** only after I write "launch WP7".
>
> ## Report
>
> After R-6, and again after R-4 and WP8:
> - tests and their results;
> - the R-6 SHA;
> - the containment result;
> - R-4 gate table against the constant baseline;
> - GPU-hours and data read;
> - open decisions.

---

## Author message, 18 Oct 2026, second (verbatim)

> Option 1. Run the range-based check now, then continue with the R-4 re-run, WP6 and the WP8 draft.
>
> - Energies are kinetic. Use 100–400 MeV kinetic, in steps of 50 MeV.
> - Cross-check your range function against the PDG table of muon CSDA range in water at 2–3
>   energies, cite the source, and state the multiple-scattering caveat.
> - Report the contained fraction per energy, with the same vertex and direction sampling as the
>   pilot, on the pinned WCTE cylinder (R = 1.577 m, H = 2.777 m).
> - Do not build PhotonSim now. If some range reaches ≥ 50% containment, record it, and leave
>   generating new contained tracks as a WP7 planning item.
> - D-E development definition: E_dep = kinetic energy on entering the volume (or at the vertex)
>   minus kinetic energy on exiting, with exit energy 0 for a stopped track. Primary muon only,
>   from PhotonSim's per-segment β. Record it in the decisions log.
>
> Record this message verbatim (author, 18 Oct 2026). Report after R-4 and WP8 in the agreed format.

**Containment result (R-4 step 1, 18 Oct 2026; `results/pmt_r4_dev_2026-10-18/containment/containment.json`).**
- Method: range-based and straight-track, with pilot sampling, on the pinned WCTE cylinder (R = 1.5771 m, H = 2.7770 m).
- Contained fraction at 100, 150, 200 and 250 MeV kinetic: 1.000 at each.
- At 300 MeV: 0.938. At 350 MeV: 0.687. At 400 MeV: 0.346. The binomial SE is ≤ 0.0011 at every energy.
- **μ− kinetic energies 100–350 MeV reach ≥ 50% containment; 400 MeV does not.**
- With ±5% range, 350 MeV gives 0.58–0.78 and 400 MeV gives 0.24–0.47.
- The CSDA range agrees with the PDG muon table in liquid water to ≤ 0.07% at 100–400 MeV.
- Caveat: multiple scattering and straggling are ignored, so the straight-track fraction is approximately a lower bound. At 500 MeV it gives 1.2%, against the pilot's Geant4 count of 3/160 (1.9%).
- **WP7 planning item:** generate new contained μ− tracks at 100–350 MeV kinetic. No PhotonSim source exists below 500 MeV, and the author said not to build PhotonSim now.

**D-E development definition (author, 18 Oct 2026):** E_dep = T(entry, or vertex) − T(exit), with T(exit) = 0 for a stopped track. Kinetic energies, primary muon only, from PhotonSim's per-segment β. Development default, labelled as such; the endpoint itself is pending Junjie.

---

## Scientific decisions (runbook v2 §3), status after the author's 18 Oct decisions

| # | Decision | Setting | Status |
|---|---|---|---|
| D-TQ | T-Q walk shape | `walk(q) = a + b/sqrt(q)` on [0.25, q_sat]; constant above q_sat. a=0, b=2 ns, q_sat=50 PE. Intervention b × (1 ± 0.2), plus a separate offset-only control | **Form adopted** (18 Oct); values `synthetic_dev` until Junjie gives a measured curve |
| D-Q | Charge domain | [0.25, 1000] PE everywhere; overflow clamped and counted | **Adopted, binding** (18 Oct) |
| D-J | Timing kernel | Gaussian approximation; fraction of digits before photon arrival reported per cell | **Adopted, binding** (18 Oct) |
| D-G | Intervention groups | mPMT modules (physical map from pinned file) | **Adopted, binding** (18 Oct) |
| D-K | Fit failures | Unevaluable if failure rate > 10%; otherwise complete-case K with failure-rate difference reported | **Adopted, binding** (18 Oct); κ_m = 1.10 stays `provisional_dev` until the WP6 freeze |
| D-E | Energy endpoint | (A) Deposited energy inside instrumented volume; initial energy supporting on contained events only; containment preflight before training | **Pending Junjie**; (A) is the labelled development default |
| R-3 energy gate | Energy MAE gate | ≤ 0.8 × constant pooled **and** ≤ 1.0 × constant in every energy subgroup | **Author decision** (18 Oct), replaces E17 |
| D-T/L/M/S | Portfolio | Unchanged from v1 WP8 | open with Junjie |

---

## Engineering decisions

_One line each, logged as made (18 Oct 2026)._

- E1. The pre-existing draft of this log (untracked, 18 Oct 00:29) called the §3 defaults "adopted". That word was changed to "development setting" to match v2 §3.
- E2. Interpreter: `/Users/wongdowling/Documents/VERITAS/.venv/bin/python` (Python 3.12.7, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1), the same one the 16 Oct runs recorded. It is only executed, with `-B` and `-p no:cacheprovider`, and never written to. It has no torch, so the two torch tests skip locally.
- E3. `src/noise_module_lucid/endpoint_identifiability.py` was found untracked (mtime 16 Oct 21:48, after `56179eb`), and no prompt mentions it. It was left uncommitted and unused pending author review.
- E4 (R-1). Manifest schema `pmt-response-v2`. Per-channel factors are a rule on a channel-to-group map (`base + delta × (group mod 2)`), so one manifest serves the 76-sensor fixture and WCTE. The values are unchanged from v1 (gain +3%, width +5% on odd groups).
- E5 (R-1). The integration and control pilots now group by physical mPMT module (D-G). Before, `lucid_integration.response` used `channel // 19`, which is not the module map. The multi-energy pilot already used mPMT.
- E6 (R-1). One QE table everywhere: the pilots' 5-anchor table (200–800 nm, zero at the ends). It equals the v1 fixture table on 300–600 nm, the toy spectrum's range.
- E7 (R-1). Fixture photon yield restored to the v1-declared 4000 (it was cut to 1000 only to fit the old 100 PE domain). Seeds, closure bounds and tolerances are unchanged.
- E8 (R-1). The literal-response scan flags a response constructor or a `replace` of a response field whose argument contains a numeric constant and no attribute of an existing response (`np.*` does not count). Bare `ReadoutParams()` is allowed only in `dataset.py`/`readout.py`, as the legacy toy chain with no QE or T-Q.
- E9 (R-2). The walk uses a = 0 on every channel, a single curve rather than the old per-channel 2 + 0.05·(i mod 3) constants, which cancelled in calibrated time anyway.
- E10 (R-2). Fixture T-Q cells: `fixed_tq_slope_up` (b × 1.2), `fixed_tq_slope_down` (b × 0.8) and `tq_offset_control` (a + 1 ns), on all channels. They replace the old odd-channel polynomial cell.
- E11 (D-Q). Clamping acts on the measured, gain-scaled charge before the discriminator, walk and width. The reported digit charge is the clamped value. The count is `overflow_clamped`. Without a declared `charge_domain_pe` (legacy), out-of-domain charge still raises.
- E12 (D-J). `digits_before_arrival` counts digits whose raw time minus first photon arrival is negative: cell walk plus jitter in T-Q mode, the template walk in legacy mode, before any calibration subtraction or offsets.
- E13 (R-2). The T-Q validation uses one photon per channel, flat QE, spe_sigma = 0 and gain = q, with a zero reference walk to expose the raw curve; N = 20000 and seed 71. Each charge point has its own event ID (1000 + 100·q). Jitter is keyed per (seed, event, channel), so the five points are independent tests. The first version reused one event and gave one draw five times; the independent review caught it.
- E14 (R-2). The charge-dependence floor in the paired control is a peak-to-peak spread above 1e-9 ns, because floating-point rounding gives a constant shift a spread of about 1e-15.
- E15–E23 (R-3, implemented by a subagent and reviewed):
  - E15. The constant predictor is the pooled training-split mean, evaluated unchanged in every subgroup.
  - E16. The vertex and energy gates use ≤; the direction gate uses a strict < 20°, both pooled and in every energy subgroup.
  - E17. Energy ratios per subgroup are reported, not gated.
  - E18. An undefined denominator, a non-finite metric or an empty or missing subgroup fails the gate, with a reason.
  - E19. Expected subgroups are `cfg['energies_MeV']`.
  - E20. The energy endpoint is chosen by column and name, and must be encoded as log(E/1000 MeV).
  - E21. The old absolute gates are removed for future runs, and the MSE-below-constant gates are kept.
  - E22. The past run's constant was reconstructed from the seeded placements and the +z gun axis, and it reproduces every recorded constant metric exactly.
  - E23. Thresholds are in `configs/lucid_pmt/trainability_gates.json`.
- E25. Independent review fixes:
  - D-Q/D-J counts are recorded per event and per run in both LUCiD pilots.
  - `overflow_clamped` counts final digits only.
  - In T-Q mode the raw delay includes the template residual when the threshold differs from calibration.
  - The literal scan is now builder-only for response constructors. Literal-valued field replacements are allowed only inside `ResponseCellSpec(...)` or a declared construction. It flags all four HEAD pilots. Known blind spot: a literal bound to a name first.
  - The exact QE×2/CE÷2 control moves to `response_manifest.exact_product_control`.
  - The offset control gets its own family, `tq_offset_control`.
  - The R-3 energy gate fails closed unless the run declares the endpoint `deposited_energy_in_volume`.
  - The past-run reproduction test skips when `results/` is not shipped.
  - Note added to the integration results README on the mPMT regrouping.
- E26. Fixed 18 Oct per author decision 5. A fired digit whose charge is below the declared domain is rejected and counted as `below_charge_domain`, never extrapolated (test `test_threshold_below_charge_domain_is_counted_not_raised`). Before the fix, a threshold below 0.25 PE raised inside the walk curve.
- E27 (R-3, author decision 2). The energy gate is now also per subgroup: ≤ 1.0 × constant in every energy subgroup (`energy_relative_mae_ratio_max_subgroup`). The fail-closed D-E endpoint rule applies to every energy gate. This replaces E17.
- E28 (author decision 4). `endpoint_identifiability.py` supports D-E (in-volume path length, containment, initial-energy identifiability), so it is committed as `dev_unreviewed` with analytic tests. The code is unchanged apart from the label. Caveat for review: `endpoint_identifiability()` builds its constant from the mean of the evaluated set's true energy, not the training-split mean of R-3, so it is a diagnostic, not a gate.
- E29 (R-6). The readout had no dark-count stage: the `pulses.py` dark pulses act on legacy waveform traces of the frozen electronics path. So the stage was implemented in `readout_chain`:
  - each PMT gets Poisson(rate × window) dark photoelectrons from a keyed uniform (identity = channel), with times uniform in `readout_window_ns` (identity = channel × 4096 + rank);
  - they bypass QE/CE/survival and join the detected photoelectrons before digitisation, so they share the SPE charge, gain, discriminator, T-Q walk and width;
  - there is one digit per PMT, so a dark photoelectron earlier than the source light sets that digit's time;
  - `n_dark_true` is evaluation-only truth and is not in `OperationalDigits`.
- E30 (R-6). `readout_window_ns` = [−200, 800] ns (1 µs, forward event time), `synthetic_dev`. It bounds dark-hit generation only; source photons are not cut by it. At 1 kHz that gives an expected 0.076 dark photoelectrons per fixture event and about 2.0 per WCTE event (1995 PMTs).
- E31 (R-6). `dark_noise_window(params, n_channels, seed=, run=, window=)` keys all its draws on stream `dark_window:<window>` with event ID = run, so a dark-only window never shares draws with a physics event that has the same ID. Rate 0 adds no draws and no count keys, and the digits are byte-identical to `0fba034` (golden hash in `test_dark_counts.py`).
- E33 (R-6 review). The LUCiD pilots' time feature is now `(time − window start) / window length` (`lucid_integration.time_feature`), replacing `(time − first digit time) / 20 ns`. With dark counts on, the earliest digit was a dark hit in about a third of WCTE events, which moved every source time. The window start is the operational, trigger-defined origin. This changes the R-4 input distribution relative to `181b747`/`56179eb`, which are not re-scored.
- E34 (R-6 review). Smaller fixes:
  - `dark_noise_window` uses an empty `detected_pe` batch for responses without QE, and requires integer seed, run and window.
  - The dark-count uniform is clamped below 1.0, where the Poisson ppf is infinite (p = 2⁻⁵³).
  - The per-PMT bound now admits exactly 4096.
  - The `n_pe_true` comment now says it includes dark photoelectrons.
  - The `endpoint_identifiability()` docstring now states its constant: the evaluated set's mean energy in MeV. The code is unchanged and the test distinguishes the two definitions.
  - Known and left: `below_charge_domain` is counted before the dead-channel mask.
- E35 (R-4). The run config is `configs/lucid_pmt/r4_pilot.json`. It keeps the `56179eb` splits, placements (augmentation seed) and architecture, and changes the energy target to D-E, with new model seeds 30–32. Code is staged on the host from `git archive 72985f2`. The trainer's gate failure stops the run as designed; it is retained and nothing is retuned.
- E36 (R-4). D-E implementation:
  - T is taken from β at each segment start.
  - Along a segment, T is linear in path length.
  - At the vertex, T is the generator kinetic energy.
  - The exit point is found by bisection on the first boundary crossing.
  - Re-entries are counted, not added (none occurred).
  - `Segment_Edep` inside the volume is recorded as a cross-check only.
- E32 (R-6). The R-2 T-Q tests and `tq_validation.py` set `dark_rate_hz = 0` to validate the walk in isolation. The pilots and fixture use the manifest's 1 kHz.
- E24 (R-5). Runs 03/04 were removed from the tree (history `56179eb`). Checkpoint paths and hashes are in `results/PMT_CHECKPOINTS.json`. The integration-pilot root is inferred from the README and is re-verified on the host in R-4.

---

## Remote/GPU job log

| Date | Host | Job | Wall time | GPU-hours | Data read |
|---|---|---|---|---|---|
| 18 Oct 2026 | local | R-1 fixture re-run (`results/pmt_r1_r3_dev_2026-10-18/fixture_run_01`, pre-review) | 6.4 s | 0 | 0 |
| 18 Oct 2026 | local | R-1 fixture re-run after review fixes (`fixture_run_02`, final) | 5.3 s | 0 | 0 |

| 18 Oct 2026 | local | R-6 fixture runs (`results/pmt_r6_dev_2026-10-18/fixture_run_01`, `_02`) | 5.7 s, 6.9 s | 0 | 0 |
| 18 Oct 2026 | local | R-4 containment check (range-based, `results/pmt_r4_dev_2026-10-18/containment`) | about 1 min | 0 | 0 |
| 18 Oct 2026 | deepthought (CPU, read-only) | geometry query; directory listings; `nvidia-smi`; checkpoint re-hash | < 3 min total | 0 | 0 |
| 18 Oct 2026 | CERNBox (WebDAV) | PhotonSim directory listing (PROPFIND) | seconds | 0 | a few KB |
| 18 Oct 2026 | deepthought (CPU, 2 threads) | segment-ordering probe, one 500 MeV event | < 1 min | 0 | 1.2 MB |
| 18 Oct 2026 | deepthought (CPU, 4 threads, nice 10) | R-4 primary-segment extraction (`segments_01`) | 56 s | 0 | 91.2 MB |
| 18 Oct 2026 | deepthought (CPU, 4 threads, nice 10) | R-4 data generation (`track_closure_01`), existing photon sources | 14 min 26 s | 0 | 0 |
| 18 Oct 2026 | deepthought (GPU 3, TITAN X, 25% memory) | R-4 training, 3 seeds × 100 epochs (`training_01`) | 55 s | 0.015 | 0 |

R-4 totals: 0.015 GPU-hours, about 92.5 MB of source reads against the 5 GB cap. Before training, a 5-minute sample showed every GPU at 0% mean utilisation, no process above 1 GB, and no `jepa_*` training session. Only `jepa_guard` was running. No other process was touched, signalled or reniced. Nothing was written under `/ceph`.

_R-1 to R-3 used no remote or GPU job; they ran on local CPU._
