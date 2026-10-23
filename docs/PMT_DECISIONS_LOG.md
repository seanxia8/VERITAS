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

## Author message, 18 Oct 2026, third (verbatim)

> Recorded decisions (author, 18 Oct 2026). Record this message verbatim in the decisions log.
>
> 1. Vertex: the validation-split diagnostic is approved, with the fix fixed in advance by principle.
>    - The time feature keeps its origin at the readout-window start, so dark hits cannot move it,
>      but uses a fixed 20 ns unit, comparable to the prompt-light spread, instead of 1 µs. Record
>      this rule before running anything.
>    - On the validation split only, run 2×2: unit {20 ns, 1 µs} × dark counts {on, off}. It
>      diagnoses the cause; it does not choose the fix.
>    - Then re-run R-4 once on the test events with the rule above, as R-4b, under the same gates and
>      3 seeds. Report it beside the failed R-4, which stays as recorded.
>    - State that training uses only 480 placements, and whether the gates look data-limited.
> 2. κ_m: run the precision study first: repeat reference runs and measure the SD of K. Then
>    κ_m = max(1.10, 1 + 2·SD_K). Freeze it in WP6 v2.
> 3. WP6 held-out choices are confirmed: timing_width as the held-out family, and modules with
>    index mod 3 = 2.
> 4. WP8 draft:
>    - Do not post. Update it after R-4b with one line on the vertex cause and result.
>    - In D-L, add "(for Paper 3 / ORACLE; the JEPA study uses LUCiD regardless)".
>    - Add a question 7: are module gain drifts of 2/5/10% per run, an independent per-PMT gain
>      spread of 5%, and a 10% attenuation-length change realistic for WCTE?
>    - In D-E, state our preference: (A), with the energy endpoint on contained μ⁻ at 100–300 MeV
>      kinetic, where deposited equals initial energy. E_dep on escaping tracks is kept as a
>      diagnostic (primary only).
> 5. WP7 waits for R-4b, the precision study and Junjie's D-E answer.
>
> Report after R-4b and the precision study.

**Time-feature rule, recorded before any R-4b diagnostic or run (18 Oct 2026):** the digit time feature is `(t − readout-window start) / 20 ns`. The origin is the trigger-defined readout-window start, so dark hits cannot move it. The unit is a fixed 20 ns, comparable to the prompt-light spread. This replaces E33's 1 µs unit (window length) for R-4b and every later run. The 2×2 validation diagnostic, unit {20 ns, 1 µs} × dark counts {on, off}, diagnoses the cause only and does not choose the fix.

---

## Author message, 18 Oct 2026, fourth (verbatim)

> Recorded (author, 18 Oct 2026). Record this verbatim. κ_m = 1.10 is accepted; re-check it if any
> WP7 cell holds fewer than 12 physical events.
>
> One more validation-only diagnostic is approved, extended so that it can be decisive. 3 seeds each,
> test events untouched, checkpoint selection as before (state the caveat).
>
> Arm 0, exact reproduction of 56179eb under the current code: first-digit time origin, initial-energy
> target, dark counts off, seeds 20–22, same loss weights and checkpoint rule.
> - If it does not reach about 0.73–0.88, there is a code regression between 56179eb and now. Then
>   bisect the commits on validation only, and report the commit that breaks it.
>
> Arms 1–4: time origin {trigger time, window start} × energy target {initial, deposited}, dark
> counts off.
>
> Also log, for every arm, each head's loss at initialisation and at the selected checkpoint (vertex,
> direction, energy), and the relative scale of the energy target. A change of target can unbalance
> the multi-task loss and starve the vertex head. Report whether that happened.
>
> This diagnoses the cause; it does not pick a fix. Propose the fix afterwards as a principle-based
> rule, recorded before any test-event re-run, as with the 20 ns rule. R-4 and R-4b stay as recorded.
> WP7 waits for this, for Junjie's D-E answer and for my "launch WP7". Update the WP8 draft with one
> line on the outcome; do not post it.

**Diagnostic design, recorded before any arm runs (18 Oct 2026, agent; E40):**
- All five arms: dark counts off, 20 ns unit, seeds 20–22, the `56179eb` loss (equal weights; vertex in 0.35 m units) and checkpoint rule (minimum validation loss). Evaluation on the validation split only; the test split is never predicted. Caveat: the validation split also selects the checkpoint, which flatters every arm equally.
- Time origins:
  - first digit: `(t − earliest digit time) / 20 ns`, the `56179eb` formula;
  - trigger time: `t / 20 ns`, where t = 0 is the trigger to which `readout_window_ns` = [−200, 800] ns is referred (in this simulation it coincides with the primary interaction time), so prompt hits sit near 0;
  - window start: `(t + 200 ns) / 20 ns`, the R-4b rule, so prompt hits sit near +10.
- Every arm uses seeds 20–22, so the seed is not a confound between arms. The window-start × deposited arm repeats R-4b's `diag_u20_nodark` (seeds 30–32) with seeds 20–22, which isolates the seed effect.
- Arm 0 checks:
  - its inputs are compared bit for bit with the `56179eb` data set, and its checkpoint hashes with the recorded `56179eb` hashes (deterministic algorithms, same GPU model);
  - like-for-like reference: the `56179eb` checkpoints are also scored on the validation split (CPU inference only), because the 0.73–0.88 range was measured on test;
  - the author's criterion is applied to arm 0's pooled validation vertex ratio. If it is not about 0.73–0.88, the commits between `56179eb` and now that touch the data or training path are bisected on validation only.
- Loss logging, every arm and seed:
  - per-head loss (direction MSE, vertex MSE in 0.35 m units, log-energy MSE) on train and validation, at initialisation and at the selected checkpoint, plus per epoch;
  - each head's gradient norm on the shared per-sensor encoder at those two points;
  - the scale of each target: the training-split constant-predictor loss per head, and the SD of the log-energy target.
- Starvation reading, fixed in advance: the target switch starves the vertex head if, with the deposited target, energy's share of the initial loss or of the encoder gradient rises clearly above its share with the initial target, and vertex fails only with the deposited target.
- The logging adds no random draws and keeps the summed loss in the same order, so the training trajectory is unchanged.


**Proposed time-feature rule (agent, 18 Oct 2026).** Approved by the author on 18 Oct 2026 (fifth message), with N = 3; see E43 for the corrected dark bound. Recorded before any test-event re-run.

- **Principles.** The time feature's origin must be:
  1. operational: computed from the digits alone, never from truth;
  2. robust to dark hits;
  3. centred, so prompt hits sit near 0 on the 20 ns unit, like the other input features.

  Each origin tested so far fails one principle:
  - the window start fails (3), which E42 shows is the cause of the vertex regression;
  - the first digit fails (2), as E33 found;
  - the trigger time t = 0 meets (2) and (3) but fails (1) in this simulation, because it is the true interaction time.
- **Rule.** The origin is a simulated NHits trigger: the earliest digit time t such that at least N digits lie in [t, t + 20 ns]. N is the smallest integer for which the dark-only probability of such a coincidence anywhere in the 1 µs readout window is below 10⁻³, at the manifest dark rate and channel count. At 1 kHz × 1995 PMTs (0.040 expected dark digits per 20 ns), N = 3 (P ≈ 5×10⁻⁴; N = 2 gives 0.039). The unit stays 20 ns.
- **No trigger.** If no such cluster exists, the origin is the window start and the placement is counted as untriggered. The count is reported, never dropped silently.
- **Expectation.** Arm 1 (truth t = 0) is an optimistic bound: about 0.68 × constant on validation. The first-digit arm (0.78) is closer to what a hit-derived origin should give without dark counts.
- **Open choice for the author.** Run one validation-only check of this rule (dark counts on and off, seeds 20–22) before the single test-event re-run, or go straight to the test re-run.

---

## Author message, 18 Oct 2026, fifth (verbatim)

> Recorded (author, 18 Oct 2026). Record this verbatim.
>
> 1. The NHits time-origin rule is approved as proposed:
>    - origin = the earliest digit time t with ≥ N digits in [t, t + 20 ns];
>    - N = 3, from the stated dark-coincidence bound (< 1e-3 per 1 µs window at 1 kHz and 1995 PMTs);
>    - fallback to the window start, counted per cell.
>    Implement it in the readout package as a public function, for example
>    nhits_trigger_time(digits, n=3, window_ns=20), with tests:
>    - a dark-only window almost never triggers (measure the rate);
>    - a prompt event triggers within 1 ns of its first true prompt cluster;
>    - the result is unchanged when dark counts are added to an event that has a prompt cluster.
> 2. Validate first, on the validation split only: dark counts on and off, seeds 20–22. This checks
>    that the rule works and reports the fallback rate; it does not tune anything.
>    - If both arms are within the 0.73–0.88 band of arm 0 and the fallback rate is < 1%, re-run
>      once on the test events as R-4c, beside R-4 and R-4b.
>    - Otherwise, report and stop.
> 3. Commit, push, and report the SHA that contains the function. The JEPA study will adopt the same
>    origin for its tokens.
> 4. WP7 still waits for Junjie's D-E answer and my "launch WP7". Update the WP8 draft with one line on
>    R-4c; do not post it.

**Implementation findings and validation design, recorded before any validation run (18 Oct 2026, agent; E43):**
- `readout.nhits_trigger_time(digits, n=3, window_ns=20.0)` implements the rule as approved. The time-feature origin `"nhits"` falls back to the window start, and `multienergy_data` counts fallbacks per placement and per (role, energy) cell.
- **Three measured properties differ from the approval's premises** (`results/pmt_r4c_dev_2026-10-18/nhits_rule/nhits_measured.json`):
  1. **Dark-only rate at N = 3:** about 1.2×10⁻³ per 1 µs window, above the stated < 10⁻³. The proposal's 5×10⁻⁴ counted 50 fixed 20 ns bins, but the rule slides from every digit: about 2.0 dark digits × P(≥ 2 more within 20 ns). Applied correctly, the principle gives N = 4, which shows no triggers in 20 000 windows. **This was my error in the proposal.**
  2. **Prompt accuracy:** with the timing width off, the trigger is within 1 ns of the true first prompt cluster. With the reference width it is early by a median of 0.6–3.3 ns (worst 5.8 ns), more so with more hit PMTs, because the earliest of many jittered digits sits in the low tail. That is 0.03–0.17 time-feature units at the median.
  3. **Dark invariance:** adding dark counts changes the trigger in 26 of 600 prompt events (4.3%, 95% CI 2.8–6.3%). A dark digit up to 20 ns before the prompt cluster can complete an earlier cluster, so the move is at most 20 ns (1 unit). 22 changes move the trigger more than 1 ns earlier; 21 of those have a dark digit in the preceding 20 ns. The remaining one (event 85) is a dark-only three-fold coincidence at −164 ns, which set the trigger 262 ns (13 units) before the prompt light. That is the N = 3 false-trigger mode of finding 1, and N = 4 would have excluded it.

  The tests assert these measured properties, not the stated ones.
- **Validation, as approved:** N = 3, origin `"nhits"`, 20 ns unit, deposited-energy target (the R-4 endpoint), dark counts on and off, seeds 20–22, validation split only. Checkpoint selection is unchanged, with the same caveat.
- **Criterion, fixed now:**
  - each arm passes if every seed's pooled validation vertex ratio is ≤ 0.88, the upper edge of arm 0's band (a value below 0.73 is better than arm 0 and counts as passing);
  - and the fallback rate over all placements is < 1%.
- **Test-event re-run held for the author** even if the criterion is met, because finding 1 shows the approved N rests on a bound that does not hold. The test events are single-use, and the author may prefer N = 4. Nothing is tuned.

---

## Author message, 18 Oct 2026, sixth (verbatim)

> Recorded (author, 18 Oct 2026). Record this verbatim.
>
> 1. N = 4, as the stated dark-coincidence principle requires once the window slides from every digit.
>    N = 3 is superseded. Update the tests to assert the bound (0 dark-only triggers in ≥ 20 000
>    windows is acceptable evidence).
> 2. Before any run, commit this principle-based prompt-window rule:
>    - a digit with t − t_trig outside [−20 ns, +60 ns] is out-of-time; it is dropped from the model
>      input and counted per cell;
>    - basis: maximum direct time of flight in WCTE (about 4.2 m / 0.225 m/ns ≈ 19 ns), plus the walk
>      (≤ 4 ns), jitter and the early-trigger tail (≤ 6 ns), with margin;
>    - with no trigger (fallback), keep the window start and drop nothing; count it.
> 3. Validation split only, seeds 20–22, three arms:
>    (a) NHits N = 4 + prompt-window rule, dark on;
>    (b) the same, dark off;
>    (c) diagnostic upper bound: dark on, but only truth-labelled dark digits removed (uses truth;
>        never a candidate fix).
>    Pass mark, recorded before the run: arm (a) ≤ 0.88 in every seed, and within 0.05 of arm (b)
>    in the seed mean. Report the dropped-digit counts and how many source digits were dropped
>    (should be about 0).
> 4. If (a) passes, run R-4c once on the test events with the rule, beside R-4 and R-4b. If it fails,
>    report and stop.
> 5. Commit and push, and report one SHA containing nhits_trigger_time(n=4) and the prompt-window rule
>    as a public function. The JEPA study adopts both from that SHA.
> 6. WP7 still waits for Junjie's D-E answer and my "launch WP7". WP8: one line on R-4c; do not post.

**Implementation and validation design, recorded and committed before any run (18 Oct 2026, agent; E45):**
- **Rule code.**
  - `readout.NHITS_N = 4`; N = 3 is superseded.
  - New public functions: `readout.prompt_window_mask(digits, trigger_ns, window_ns=PROMPT_WINDOW_NS)`, with `PROMPT_WINDOW_NS = (-20.0, 60.0)` as a closed interval, and `readout.select_digits(digits, keep)`. With no trigger (`None`) every digit is kept.
  - `multienergy_data` (config `prompt_window: true`, needs origin `"nhits"`):
    - computes the trigger on the digits;
    - drops out-of-time digits from the model input (charge, time and mask columns);
    - counts per placement and per (role, energy) cell, plus fallbacks:
      - `prompt_dropped`;
      - `prompt_dropped_source`: dropped digits holding at least one source photoelectron, from evaluation-only truth;
      - `prompt_dropped_source_no_dark`: the same, but with no dark photoelectron.
  - A source digit can be out-of-time when an earlier dark photoelectron on the same PMT set its measured time. The test suite saw this in 2 of 40 synthetic events, at −250 and −56 ns. The rule then drops it correctly, because its time is dark, so the two source counts are reported separately.
  - Restricting to [t_trig − 20, t_trig + 60] never moves the trigger: no earlier ≥ 4 cluster exists, and the cluster at t_trig is kept. A test checks this.
- **Tests.**
  - 0 dark-only triggers at N = 4 in 20 000 keyed windows (95% upper limit 1.5×10⁻⁴).
  - The window's bounds and basis.
  - The fallback drops nothing.
  - No pure-source digit is dropped in 40 synthetic prompt events with dark counts. Any dropped source digit lies before the trigger, with its time set by dark. The window leaves the trigger unchanged.
- **Arms** (validation split only, seeds 20–22; deposited-energy target, 20 ns unit, origin `"nhits"` with N = 4; `configs/lucid_pmt/r4c/e45_*.json`):
  - (a) prompt window, dark on;
  - (b) prompt window, dark off;
  - (c) dark on; digits whose photoelectrons are all truth-labelled dark are removed *before* the trigger, then the trigger and prompt window are applied as in (a). So (c) differs from (a) only by perfect dark removal, and from (b) only by dark photoelectrons that merged into source digits. **Diagnostic upper bound; uses truth; never a candidate fix.** Counted as `truth_dark_dropped`.
  - Checkpoint selection is unchanged, with the same caveat: the validation split also selects the checkpoint.
- **Pass mark** (the author's, fixed now):
  - arm (a)'s pooled validation vertex ÷ constant is ≤ 0.88 in every seed;
  - and |mean(a) − mean(b)| ≤ 0.05 over seeds 20–22.

  Arm (c) is reported, not gated. Reported counts: dropped digits per cell, dropped source digits (expected about 0), and fallbacks.
- **R-4c, only if (a) passes:**
  - once on the test events (`r4c_test.json`), on the arm (a) data set;
  - seeds 30–32, the same seeds and gates as R-4 and R-4b, so the three sit side by side;
  - the checkpoint rule is unchanged. If (a) fails: report and stop.

## Author message, 18 Oct 2026, seventh (verbatim)

> Recorded (author, 18 Oct 2026). Record this verbatim.
>
> 1. 5f4ecc0 is named for the JEPA study; it supersedes b341037.
> 2. The R-4 series is closed. R-4, R-4b and R-4c are recorded as development fails on the 60-event
>    pilot. Run nothing more on those test events. The vertex gate is re-evaluated in WP7 on fresh
>    events with the frozen rule (NHits N = 4, prompt window [−20, +60] ns, 20 ns unit).
> 3. WP8: rewrite the vertex part of the R-4 section as one compact paragraph and a table (R-4 /
>    R-4b / R-4c, vertex pooled per seed). Include the cause (window-start origin; conditioning), the
>    final rule, and the outcome (0.81–0.88, gate missed).
>    - Wherever the trigger-time arm (0.65–0.68) is mentioned, state that t = 0 is the true emission
>      time in this simulation and is not available in data.
>    - Do not post it; tell me when it is ready and I will post it myself.
> 4. WP7 still waits for Junjie's D-E answer and my "launch WP7". Until then, nothing new on the hosts.

**Actions (agent, 18 Oct 2026):**
- The R-4 series is closed. The R-4, R-4b and R-4c verdicts stand as development fails on the 60-event pilot, and those test events are retired.
- WP7's frozen time rule is NHits N = 4, the prompt window [−20, +60] ns and a 20 ns unit, from `5f4ecc0`.
- WP8 draft: the vertex part of the R-4 section is rewritten as one paragraph and a table. The trigger-time arm carries the caveat that t = 0 is the true emission time and is not available in data.
- Nothing new on the hosts.

## Author message, 18 Oct 2026, eighth (verbatim)

> 7. **Realistic drift sizes (for the JEPA study).** We inject module gain drifts of 2/5/10% per run;
>    only 5% and 10% are visible enough in single WCTE runs to test. An independent per-PMT gain spread
>    stays undetectable even at 10%. A water-quality-like attenuation change only becomes visible at
>    a 50% change in attenuation length, because WCTE is small. Are 5–10% module drifts and 10%
>    per-PMT spreads realistic? Is a 50% attenuation change plausible at all? If not, would you
>    prefer a different smooth second factor, for example a vertical gain gradient from temperature
>    stratification?

**Action (agent, 18 Oct 2026):** this replaces question 7 in the WP8 draft word for word. The visibility statements come from the JEPA study and were not re-derived here. The draft is not posted.

## Author message, 18 Oct 2026, ninth (verbatim, SHA statement)

> PMT SHA named (author, 18 Oct 2026; record verbatim): "5f4ecc0 on codex/junjie-pmt-response is
> accepted for JEPA reported runs; it supersedes b341037."

**Action (agent, 18 Oct 2026):**
- Recorded the statement above.
- Checked `git diff 5f4ecc0 HEAD`: no changes under `src/` or `configs/`. The commits after `5f4ecc0` touch only `docs/` (3 files) and `results/` (45 files).
- The rest of that message (merge `5f4ecc0`, token time, emission-time offset and invariance tests, regenerating datasets, the β table, the 201-run grid) is addressed to the JEPA study. It is not acted on in this branch (runbook §6: never touch the JEPA branch, its files or its jobs).

## Scientific decisions (runbook v2 §3), status after the author's 18 Oct decisions

| # | Decision | Setting | Status |
|---|---|---|---|
| D-TQ | T-Q walk shape | `walk(q) = a + b/sqrt(q)` on [0.25, q_sat]; constant above q_sat. a=0, b=2 ns, q_sat=50 PE. Intervention b × (1 ± 0.2), plus a separate offset-only control | **Form adopted** (18 Oct); values `synthetic_dev` until Junjie gives a measured curve |
| D-Q | Charge domain | [0.25, 1000] PE everywhere; overflow clamped and counted | **Adopted, binding** (18 Oct) |
| D-J | Timing kernel | Gaussian approximation; fraction of digits before photon arrival reported per cell | **Adopted, binding** (18 Oct) |
| D-G | Intervention groups | mPMT modules (physical map from pinned file) | **Adopted, binding** (18 Oct) |
| D-K | Fit failures | Unevaluable if failure rate > 10%; otherwise complete-case K with failure-rate difference reported | **Adopted, binding** (18 Oct); κ_m = 1.10 **accepted** (author, 18 Oct, fourth message); re-check if any WP7 cell holds fewer than 12 physical events |
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
- E37 (R-4b). Results:
  - The 2×2 diagnostic (validation only) rules out the time unit as the cause, and dark counts as more than a small effect (about 0.05). All arms give vertex 0.95–1.00 × constant.
  - With dark counts off, the inputs equal those of `56179eb` except for a per-event constant time offset (+9.9 to +10.3 units on hits; unhit sensors stay at 0) and the energy column.
  - The cause is not identified. The open candidates are the time origin, the energy-target switch and the model seeds.
  - R-4b (test, 20 ns rule) fails vertex in all seeds (0.99–1.00) and E_dep in seed 30.
  - Vertex is not data-limited at 96–192 training placements; E_dep is.
  - Recorded beside R-4; nothing was retuned.
- E38 (κ_m). The precision study (32 readout re-draws × 96 validation placements, frozen R-4b checkpoints) gives SD_K = 0.040 for K = max(K_vertex, K_energy), with SD = 0.028 for vertex and 0.064 for energy. Then 1 + 2·SD_K = 1.080, so **κ_m = 1.10**, frozen in `wp6_contract_v2.json`.
- E39 (R-4b, data). The 1 µs + dark arm reuses the R-4 data set (`track_closure_01`): its settings are identical, (t − lo)/(hi − lo) with hi − lo = 1000 ns. The diagnostic uses the validation split for both checkpoint selection and evaluation. That flatters every arm equally and is used only to compare arms.
- E41 (E40 run). The first arm 2 generation (code `65fba28`) failed on its second placement: in the deposited-energy branch of `multienergy_data`, `origin=raw['vertex_m']` shadowed the new time-origin variable. Fixed at `91ef3f4` (renamed `time_origin`). Arms 0, 1 and 3 never enter that branch. Every arm was generated or trained with `91ef3f4`, apart from arms 0, 1 and 3's data, which came from `65fba28`. Their generation path is identical, and arm 0's bit-identical match with `56179eb` confirms it.
- E42 (E40 result, validation only). Vertex RMSE ÷ constant, seeds 20–22:
  - arm 0 (first digit, initial) 0.782;
  - arm 1 (trigger, initial) 0.681;
  - arm 2 (trigger, deposited) 0.652;
  - arm 3 (window start, initial) 0.961;
  - arm 4 (window start, deposited) 0.936.

  Findings:
  - Arm 0's inputs are bit-identical to `56179eb`'s, and its checkpoints byte-identical, so there is no code regression and no bisection.
  - **The cause is the window-start origin:** a constant +10-unit offset on hit sensors against an unhit 0. It is a conditioning effect; the timing information is unchanged.
  - The energy target does not starve the vertex head. Vertex is slightly better with the deposited target, and energy takes 4–14% of the loss at the selected checkpoint.
  - The seed effect is small (arm 4: 0.936 against R-4b's 0.948 with seeds 30–32).

  Details: `results/pmt_r4c_dev_2026-10-18`. Nothing was retuned, and the test split was not predicted.
- E44 (NHits validation, validation only; code `1d9845f`). Vertex ÷ constant, seeds 20–22:
  - dark off: 0.797, 0.748, 0.814;
  - dark on: 0.886, 0.921, 0.924;
  - fallback 0 of 480 placements in both arms.

  **The E43 criterion fails** (dark on is above 0.88 in every seed), so R-4c was not run, as the author's rule requires.
  - The rule fixes the conditioning problem without dark counts.
  - With dark counts, vertex loses about 0.12. An untested hypothesis is that dark-digit time features (about −15 to +35 units against prompt hits near 0–2) act as outliers under masked max pooling.

  Nothing was tuned. Details: `results/pmt_r4c_dev_2026-10-18/nhits_validation`.
- E46 (E45 validation result, validation only; code `5f4ecc0`). Vertex ÷ constant, pooled, seeds 20 / 21 / 22:
  - (a) N = 4 + prompt window, dark on: 0.840 / 0.789 / 0.807 (mean 0.812);
  - (b) the same, dark off: 0.797 / 0.748 / 0.814 (mean 0.786);
  - (c) truth-dark removed (upper bound): 0.775 / 0.755 / 0.824 (mean 0.785).

  **The pass mark is met:** every seed of (a) is ≤ 0.88 (maximum 0.840), and mean(a) − mean(b) = +0.026 (≤ 0.05). So R-4c runs once on the test events, as recorded in E45.
  - Fallback: 0 of 480 placements in every arm.
  - **Dropped digits, arm (a):**
    - 710 of 197 455 digits in all roles; 147 of 39 557 in validation; per validation cell 59, 45 and 43 (500, 1000 and 1500 MeV).
    - Source digits dropped: 32, every one holding a dark photoelectron that set its time before the trigger. Pure-source digits dropped: 0.
  - Arm (c) removed 736 truth-dark digits; its prompt window then dropped only those same 32 dark-timed source digits.
  - Arm (b)'s checkpoints are byte-identical to E44's dark-off checkpoints. With dark counts off, N = 4 and N = 3 give the same trigger on every placement, and the window drops nothing.
  - **Reading:** the window recovers most of the dark-count loss. (a) is within 0.03 of both (b) and the truth upper bound (c), against a loss of 0.12 in E44. The direction and E_dep gates are unchanged in character. Nothing was tuned.

  Details: `results/pmt_r4c_dev_2026-10-18/e45`.
- E47 (R-4c, test events, once; code `5f4ecc0`). NHits N = 4 + prompt window, dark on, deposited-energy target, seeds 30–32, on the E45 arm (a) data set; same gates and checkpoint rule as R-4 and R-4b.

  | Seed | Vertex, pooled | Vertex, worst energy | Direction | E_dep, pooled | E_dep, worst energy | Verdict |
  |---|---|---|---|---|---|---|
  | 30 | 0.812 | 0.88 (500 MeV) | 7.8° | 0.642 | 0.67 | FAIL (vertex pooled) |
  | 31 | 0.877 | 0.93 (1000 MeV) | 9.2° | 0.570 | 0.72 | FAIL (vertex pooled) |
  | 32 | 0.883 | 1.02 (500 MeV) | 7.8° | 0.640 | 0.69 | FAIL (vertex pooled and 500 MeV) |

  **R-4c fails the vertex gate (≤ 0.8 × constant pooled) in all three seeds.** Direction and E_dep pass in every seed and cell. Vertex is much better than R-4 (0.96–1.00) and R-4b (0.99–1.00), but not enough for the gate.
  - On test, seeds 30–32 sit about 0.04 above E45 (a) on validation (0.81–0.88 against 0.79–0.84, seeds 20–22). Different seeds and splits, and the validation split also selected the checkpoints.
  - The verdict is retained, with no retuning and no further test-event run. Details: `results/pmt_r4c_dev_2026-10-18/r4c_test`.
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

| 18 Oct 2026 | deepthought (CPU, 2 × 4 threads, nice 10) | R-4b generation: `data_u20_dark`, `data_u20_nodark`, `data_u1000_nodark`, `data_precision` | 16:52, 17:02, 15:10, 6:45 | 0 | 0 |
| 18 Oct 2026 | deepthought (GPU 3, TITAN X, 25% memory) | R-4b GPU sequence: 2×2 diagnostic, 2 train-subset arms, R-4b (7 runs × 3 seeds) | 338 s in total | 0.094 | 0 |
| 18 Oct 2026 | deepthought (CPU, 4 threads, nice 10) | κ_m precision inference (`precision_01`) | 30 s | 0 | 0 |

| 18 Oct 2026 | deepthought (CPU, 2 × 4 threads, nice 10) | E40 generation: `r4c_data_arm0`, `_arm1`, `_arm3`; `_arm2` failed (E41); `_arm2b` retry | 14:34, 14:45, 13:15; 0:56; 13:17 | 0 | 0 |
| 18 Oct 2026 | deepthought (GPU 0, TITAN X, 25% memory) | E40 training: 5 arms × 3 seeds, validation only (`r4c_arm0`–`4`) | 280 s in total | 0.078 | 0 |
| 18 Oct 2026 | deepthought (CPU, nice 10) | E40 input comparisons and `56179eb` validation reference | < 2 min | 0 | 0 |
| 18 Oct 2026 | local | NHits rule measurement (`nhits_rule/measure_nhits.py`, 20 000 dark windows, 600 prompt events) | about 4 min | 0 | 0 |
| 18 Oct 2026 | deepthought (CPU, 2 × 4 threads, nice 10) | NHits validation generation: `nhits_data_dark`, `nhits_data_nodark` | 14:46, 14:18 | 0 | 0 |
| 18 Oct 2026 | deepthought (GPU 0, TITAN X, 25% memory) | NHits validation training: 2 arms × 3 seeds, validation only | 111 s | 0.031 | 0 |
| 18 Oct 2026 | local | E45 test suite (20 000 dark windows at N = 4) | about 3 min | 0 | 0 |
| 18 Oct 2026 | deepthought (CPU, 2 × 4 threads, then 4 threads, nice 10) | E45 generation: `e45_data_a`, `_b`, `_c` | 14:32, 14:43, 13:22 | 0 | 0 |
| 18 Oct 2026 | deepthought (GPU 0, TITAN X, 25% memory) | E45 training: 3 arms × 3 seeds, validation only | 169 s | 0.047 | 0 |
| 18 Oct 2026 | deepthought (GPU 0, TITAN X, 25% memory) | R-4c training and test evaluation, 3 seeds (`r4c_test`) | 56 s | 0.016 | 0 |

E45 and R-4c totals: 0.063 GPU-hours, with no new source reads. Before each GPU job, a 5-minute sample (`gpu_sample_pre_e45.csv`, `gpu_sample_pre_r4c_test.csv`) showed all GPUs at 0%, no `jepa_*` training session, and only another user's notebook kernel at ≤ 802 MiB per card. Cumulative GPU use in this session: 0.281 GPU-hours. Cumulative source reads: about 92.5 MB.

NHits totals: 0.031 GPU-hours, with no new source reads. Before training, a 5-minute sample showed all GPUs at 0%, no `jepa_*` training session, and only another user's notebook kernel at ≤ 802 MiB per card. Cumulative GPU use in this session: 0.218 GPU-hours. Cumulative source reads: about 92.5 MB.

E40 totals: 0.078 GPU-hours, with no new source reads. Before training, a 5-minute sample showed all GPUs at 0% mean utilisation and no `jepa_*` training session. The only other GPU process was another user's notebook kernel, at ≤ 802 MiB per card. GPU 0 was used, the same index as `56179eb`. Cumulative GPU use in this session: 0.187 GPU-hours. Cumulative source reads: about 92.5 MB.

R-4b totals: 0.094 GPU-hours, with no new source reads. Before the run, a 5-minute sample showed all GPUs at 0%, no process above 1 GB, and no `jepa_*` training session. Cumulative GPU use in this session: 0.109 GPU-hours. Cumulative source reads: about 92.5 MB.

R-4 totals: 0.015 GPU-hours, about 92.5 MB of source reads against the 5 GB cap. Before training, a 5-minute sample showed every GPU at 0% mean utilisation, no process above 1 GB, and no `jepa_*` training session. Only `jepa_guard` was running. No other process was touched, signalled or reniced. Nothing was written under `/ceph`.

_R-1 to R-3 used no remote or GPU job; they ran on local CPU._

---

## Author message, 23 Oct 2026 (verbatim, R-7)

> New repair R-7: literature values from Junjie's thesis. Do it **before step 1** of my D-E message
> (the generation check), so the new events use the new response. Record this message verbatim in
> `docs/PMT_DECISIONS_LOG.md` (author, 23 Oct 2026). CPU only.
>
> **Source.** J. Xia, *Upgrade and Calibration of Super-Kamiokande's Inner Photodetectors*, MSc
> thesis, University of Tokyo, July 2019. Cite it; do not commit the PDF.
> - It measures **50 cm** Super-K (R3600) and Hyper-K Box&Line (R12860) PMTs, not WCTE's 3-inch PMTs.
> - So its values are **proxies**. Label each one `literature_proxy_hk_20in` (or `literature_sk`),
>   with the section, figure or equation it comes from.
> - A WCTE value from Junjie replaces a proxy as soon as it arrives.
> - The figure values below were digitised by me by eye: about ±0.1 ns, and ±1 percentage point of QE.
>
> ## 1. Adopt (new schema `pmt-response-v3`)
>
> 1. **`threshold_pe` = 0.25, unchanged.** Source: SK's −0.69 mV threshold, about 0.25 PE (§6.1).
>    Status `literature_sk`.
> 2. **`spe_sigma` = 0.30, unchanged.** Source: HK B&L SPE charge resolution 30.8% (§4.3).
>    Status `literature_proxy_hk_20in`.
> 3. **QE curve, replaced.** The current synthetic curve peaks at 0.50, which no bialkali PMT
>    reaches. Use the HK R12860 average from Fig. 4.3 (thesis ref. [21]):
>
>    | λ (nm) | 280 | 300 | 320 | 340 | 350 | 370 | 400 | 420 | 450 | 480 | 500 | 520 | 540 | 560 | 600 | 650 | 700 |
>    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
>    | QE | 0 | 0.01 | 0.10 | 0.23 | 0.27 | 0.29 | 0.305 | 0.29 | 0.25 | 0.19 | 0.165 | 0.13 | 0.07 | 0.045 | 0.015 | 0.002 | 0 |
>
>    Keep zero-QE anchors outside 280–700 nm, as now.
> 4. **`collection_efficiency` = 0.87**, up from 0.80. Source: HK B&L CE over the full 50 cm (§4.2).
>    Peak QE × CE ≈ 0.27, which matches the HK requirement of 26% (Table 4.1). Record that check.
> 5. **Timing width: the split-normal kernel, Eq. 6.3**, replacing the symmetric Gaussian (the D-J
>    approximation). This is the kernel Junjie asked for in issue #5.
>    - The probability density is √(2/π)/(σ_e + σ_l) · exp(−(t − t₀)²/2σ²), with σ = σ_e for t < t₀
>      (early side) and σ = σ_l for t ≥ t₀ (late side).
>    - t₀ (the mode) = photon arrival + walk(q). The walk, the T-Q intervention and the
>      `digits_before_arrival` counter keep their meaning.
>    - The mean shifts by √(2/π)(σ_l − σ_e); record it.
>    - σ_e and σ_l, from Fig. 6.4(b), HK PMTs. In Fig. 6.3, σ_r (blue) is the early side and σ_l
>      (magenta) is the late side:
>
>      | q (PE) | 0.25 | 0.5 | 1 | 2 | 3 | 5 | 10 | 30 | 100 | 1000 |
>      |---|---|---|---|---|---|---|---|---|---|---|
>      | σ_e (ns) | 2.0 | 1.53 | 1.26 | 1.04 | 0.98 | 0.86 | 0.72 | 0.65 | 0.61 | 0.75 |
>      | σ_l (ns) | 2.9 | 1.82 | 1.52 | 1.76 | 1.44 | 1.13 | 0.83 | 0.60 | 0.47 | 0.49 |
>
>      Interpolate linearly in log q. The bump at 2 PE is real in the data; keep it.
>    - The per-module `timing_width_factor` multiplies both σ_e and σ_l.
>    - Treat the values as standard deviations, since §6.2 says they feed Eq. 6.3. Eq. 6.2 omits the
>      factor 2, so this reading may be off by √2. List it as a question for Junjie.
> 6. **Tests:**
>    - QE table interpolation, and the zero anchors.
>    - Peak QE × CE within 0.25–0.29.
>    - At 0.25, 1, 2, 10 and 100 PE: the sampled early and late SDs match the table within 3σ, and
>      the mode equals arrival + walk.
>    - The T-Q intervention still changes only the walk.
>    - Dark hits use the same kernel.
>    - The v2 response is still reproducible by its manifest hash.
>
> ## 2. Record, do not adopt
>
> Add these to the decisions log as literature comparisons. No parameter changes.
> - **T-Q magnitude** (Fig. 5.33, one 20-inch channel, SK electronics). Relative to 0.25 PE, the hit
>   comes earlier by about 3 ns at 0.4 PE, 7 at 0.8, 12 at 4, 17 at 40 and 21.5 at 400 PE. It keeps
>   rising by about 4.5 ns per decade above 4 PE.
>   - Our `a + b/√q` with b = 2 ns spans about 3.7 ns and is flat above 50 PE.
>   - Keep our form and values: a 3-inch tube's pulses are faster, and D-TQ stays as adopted.
>   - Add a question for Junjie: does the WCTE walk keep rising at high charge?
> - **Dark rate:** 4.2 kHz (SK) and 8.3 kHz (HK B&L at 15 °C) are 20-inch values and scale with
>   cathode area, so they do not transfer. Keep 1 kHz.
> - **Charge linearity** within 5% up to 470 PE, and afterpulses below 5% (§4.3, §4.5). Pre- and
>   after-pulses stay off.
> - **SPE charge shape** (Eq. 6.1: SPE and 2PE peaks plus backscatter terms). The Gaussian stays for
>   now.
> - **Gain drift**, for the JEPA study's framing only, not a PMT parameter:
>   - SK-IV's average gain rose about 17% over roughly 10 years (re-tuned to 85.6% of the drifted
>     value), and the PMT-to-PMT spread grew to 12% (4% after re-tuning);
>   - HK B&L gain varied 1% RMS over 2 years.
>
> ## Constraints
>
> - **JEPA stays pinned at `5f4ecc0`.** Do not ask it to merge v3, and do not touch its files.
> - Past results (R-4, R-4b, R-4c) keep their v2 labels. Nothing is re-run on the 60 pilot events.
> - Commit, push, and report:
>   - the v3 SHA;
>   - the tests;
>   - a one-line comparison: hit PMTs per event and mean digit time spread, v2 against v3, on the
>     existing fixture.

**R-7 literature comparisons (author, 23 Oct 2026; record, do not adopt -- no parameter changes).**
- **T-Q magnitude** (Fig. 5.33, one 20-inch channel, SK electronics): relative to 0.25 PE the hit comes earlier by about 3 ns at 0.4 PE, 7 ns at 0.8, 12 ns at 4, 17 ns at 40 and 21.5 ns at 400 PE, still rising by about 4.5 ns per decade above 4 PE; our `a + b/√q` with b = 2 ns spans about 3.7 ns and is flat above 50 PE. Our form and values are kept (3-inch tube pulses are faster; D-TQ stays as adopted).
- **Dark rate:** 4.2 kHz (SK) and 8.3 kHz (HK B&L at 15 °C) are 20-inch values that scale with cathode area and do not transfer; kept at 1 kHz.
- **Charge linearity** within 5% up to 470 PE and afterpulses below 5% (§4.3, §4.5); pre- and after-pulses stay off.
- **SPE charge shape** (Eq. 6.1: SPE and 2PE peaks plus backscatter terms); the Gaussian-based digit charge smearing stays for now.
- **Gain drift** (for the JEPA study's framing only, not a PMT parameter): SK-IV average gain rose about 17% over roughly 10 years (re-tuned to 85.6% of the drifted value) and the PMT-to-PMT spread grew to 12% (4% after re-tuning); HK B&L gain varied 1% RMS over 2 years.

The R-7 adopted items are unaffected by these comparisons: `threshold_pe` = 0.25 (`literature_sk`, §6.1), `spe_sigma` = 0.30 (§4.3), the Fig. 4.3 HK R12860 QE curve, `collection_efficiency` = 0.87 (§4.2) and the Eq. 6.3 split-normal timing kernel keep their R-7 values (these five bullets are comparisons only).

**Open questions for Junjie from R-7.**
- Is the §6.2/Fig. 6.3 timing width read as a standard deviation (σ_e early, σ_l late, feeding Eq. 6.3) or does Eq. 6.2's missing factor of 2 mean our reading is off by √2? R-7 adopts the standard-deviation reading for now; the values are labelled `literature_proxy_hk_20in`.
- Does the WCTE T-Q walk keep rising at high charge (Fig. 5.33 keeps rising at about 4.5 ns/decade above 4 PE)? Our `a + b/√q` is flat above 50 PE, so a WCTE measurement decides.

The R-7 values are proxies, labelled `literature_proxy_hk_20in` (HK 20-inch R12860 Box&Line) or `literature_sk` (Super-K R3600 50 cm), from J. Xia, *Upgrade and Calibration of Super-Kamiokande's Inner Photodetectors*, MSc thesis, University of Tokyo, July 2019, with section/figure/equation per value; a WCTE value from Junjie replaces a proxy as soon as it arrives, and the PDF is not committed.

**Engineering decisions and recorded checks (agent, 23 Oct 2026; R-7).**

- E48 (R-7). Schema `pmt-response-v3` in `configs/lucid_pmt/reference_v3.json` is the new default reference; `configs/lucid_pmt/reference.json` (`pmt-response-v2`) is frozen byte-identical and its file hash (`17636f9bba69e10ea59bcbd83a884c003b4709ea5e1a6fabd94178a0be402e68`) is pinned in `response_manifest.V2_SHA256` and checked by `load_manifest`. Past results (R-4, R-4b, R-4c) and the JEPA SHA `5f4ecc0` keep their v2 labels. JEPA was not asked to merge v3 and none of its files (`src/jepa_noise/`, `configs/jepa*`, `docs/jepa/`, `results/jepa/`) were touched.
- E49 (R-7). QE curve replaced by the HK R12860 average from Fig. 4.3: the 17 author-transcribed anchors 280-700 nm plus the v2 zero-QE anchors at 200 and 800 nm; piecewise-linear in wavelength; status `literature_proxy_hk_20in`. The old synthetic curve's peak 0.50 is gone. At 300-600 nm the LUCiD pilots' photon band is fully inside the new table.
- E50 (R-7). `collection_efficiency` = 0.87 (section 4.2). Peak QE x CE = 0.305 x 0.87 = 0.26535, inside the required 0.25-0.29 and matching the HK requirement of 26% (Table 4.1); recorded in the manifest's `literature_provenance.checks` and asserted by a test.
- E51 (R-7). `timing_response.SplitNormalWidth` implements Eq. 6.3: density sqrt(2/pi)/(sigma_e + sigma_l) * exp(-(t - t0)^2 / (2 sigma^2)), sigma_e for t < t0 and sigma_l for t >= t0. The Fig. 6.4(b) sigma_e/sigma_l table is interpolated linearly in log10 q and the 2 PE bump is kept. The mode is t0 = photon arrival + walk; the sampled mean is t0 + sqrt(2/pi)(sigma_l - sigma_e), recorded on the class and in the manifest conventions. The per-module `timing_width_factor` scales both sigmas, so the inverse-CDF draw scales by the same factor (the ratio, and therefore the early/late probability, is unchanged).
- E52 (R-7). The readout draws one keyed uniform per channel for the digit width (identity = channel, as in v2) and maps it through the declared kernel; dark photoelectrons pass through the same width stage, so there is no separate dark kernel. Moving the T-Q walk leaves the width draw unchanged: the walk interventions remain exactly the declared curve shift. The v2 symmetric-Gaussian path is arithmetically unchanged.
- E53 (R-7). Reading recorded: the thesis widths are treated as standard deviations (section 6.2 says they feed Eq. 6.3), and the possible sqrt(2) reading (Eq. 6.2 omitting the factor 2) is listed as an open question for Junjie above.
- E54 (R-7). Test code changes outside the new file: the pre-R-6 dark-count golden hash is now pinned to the v2 manifest (it must keep reproducing v2, not v3); the physical T-Q mean check adds the recorded split-normal mean shift; the early-digit fraction check uses the analytic split-normal CDF; `pmt_development`'s QE-shape control resamples the v1/v2 anchor pattern onto the v3 table and its plot shows sigma_e and sigma_l.
- E55 (R-7). New test file `src/noise_module_lucid/tests/test_r7_response_v3.py` (15 tests): QE table interpolation and zero anchors; peak QE x CE; analytic normalisation, CDF, inverse CDF, mean shift and log-q interpolation of the split normal; sampled widths, half probability, mean and KDE mode plus a KS test against the analytic CDF at 0.25, 1, 2, 10 and 100 PE; width-factor scaling of both sigmas; T-Q intervention changes only the walk; dark digits carry the same kernel; v2 reproducible by its manifest hash; v3 default and provenance labels. Full `src/noise_module_lucid/tests` run: 211 passed, 2 skipped (the two torch tests skip locally).

- E56 (R-7). WP6 contract v2 is left bound to the v2 reference (it is frozen and WP7 is not launched); re-freezing the contract on v3 is a WP7 planning item, not part of R-7. Nothing was re-run on the 60 pilot events and no GPU or remote host was used.

**R-7 fixture comparison, v2 against v3 (24 evaluation events of the existing 76-sensor synthetic fixture; `results/pmt_r7_dev_2026-10-23/fixture_comparison.json`, `scripts/pmt_r7_v2_v3_fixture_comparison.py`).** Hit PMTs per event 75.96 -> 75.12 (delta -0.83); mean within-event digit time spread 3.081 -> 3.235 ns (delta +0.154 ns); mean digit time 21.121 -> 21.297 ns. Only the response manifest changes: same events (30000-30023), same per-event seeds, dark-count stage on at 1 kHz in both. Caveats, all properties of this fixture: it is occupancy-saturated (4000 photons over 76 sensors, about 11 PE/PMT in v2 and 6 in v3), so the hit-PMT count sits near the 76-sensor ceiling and is compressed; and the about 3 ns within-event spread is dominated by time of flight across the cylinder, so it is only weakly sensitive to the kernel change. Nothing was re-run on the 60 pilot events.
