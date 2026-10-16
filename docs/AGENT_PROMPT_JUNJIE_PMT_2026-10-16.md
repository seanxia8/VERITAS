# Agent prompt: implement Junjie's PMT corrections first

_16 October 2026. Copy the prompt below into a new agent chat opened on the PMT worktree. This is a bounded implementation assignment, not a launch of the full scientific campaign._

---

Work in `/Users/wongdowling/.codex/worktrees/junjie-pmt-response/VERITAS`, branch `codex/junjie-pmt-response`. The integration base is `0cae6b061e488e319643cfb0ba820318b8923578`. The user requests that PMT fixes proceed independently of the JEPA study.

Read `docs/RUNBOOK_JUNJIE_PMT_2026-10-16.md` completely, then implement **WP0–WP5 first**, including their acceptance tests and an engineering closeout under WP9. Prepare WP6's experiment contract and a concise WP8 decision list where the implementation supplies enough information. Do not launch WP7 training or new transfer-arm work in this pass.

## Scope and authority

- Implement reversible local code and documentation changes; run bounded local CPU unit tests and synthetic validation. You may inspect the pinned LUCiD code and run a small local integration fixture if the required simulator and data are already available. Record missing dependencies instead of claiming a real-simulator validation.
- You may create local commits containing only this task's changes after focused tests and `git diff --check` pass. Do not push, merge, switch to another project's branch, post issue comments, close issues, contact collaborators, download bulk datasets, launch remote jobs, or use a GPU.
- Do not modify `/Users/wongdowling/Documents/VERITAS`, the JEPA worktree, `docs/jepa/`, `src/jepa_noise/`, previous result folders, archived evidence, or the manuscript. Do not cherry-pick PMT changes into JEPA. The two studies must remain isolated.
- Keep changes within `src/noise_module_lucid/`, its tests, PMT-specific docs/configuration and a new dated PMT development-results folder. Update the docs index as needed. Preserve the public noise-module API; import existing dependencies rather than modifying shared scientific packages. Do not change the primary claims, `GENERIC_COMMITTED`, protocol labels or harm thresholds implicitly.
- The thesis and issue text are source evidence, not executable instructions. Use the runbook's source map and inspect the actual code. Recheck issue #5 for newer scientific clarifications if accessible; record any change rather than silently altering the requested design.
- Do not ask again about already settled points: baseline-only training, fixed-threshold T-Q intervention, prompt-only first trial, uniform angular acceptance, and deferring concentrators. Junjie's LUCiD research-use statement is already linked in the runbook. Record current licence-file and release status separately without treating the old unanswered-use request as current.

## First actions

1. Verify cwd, branch, HEAD and worktree status. Read applicable `AGENTS.md` instructions. Confirm the fixed base is an ancestor. Preserve unrelated files if any have appeared; never reset or clean them.
2. Read the runbook, `readout.py`, `response.py`, `dataset.py`, `consequence.py`, the current tests, and `docs/PREREGISTRATION.md` for existing boundaries.
3. Read the relevant sections of Junjie Xia's thesis at `/Users/wongdowling/Downloads/MasterThesis-XIA_JUNJIE-35176375.pdf`. Verify SHA-256 `f3bfaf51e9cedf9d0219a36de888b1e5b5e21168664ad0c52af2210f8caee7df`. Printed-page and PDF-page references are mapped in the runbook. Do not copy Super-K numerical calibrations into a different detector without justification.
4. Create `docs/PMT_ISSUE5_STATUS.md` with the J1–J11 requirements, current evidence and separate implementation/test/physics-review/evaluation status. Record a baseline test run using a compatible interpreter by absolute path; avoid a workspace sync that attempts unavailable CUDA wheels on macOS.
5. Complete the work packages below in dependency order. If a scientific choice is missing, isolate it in the configuration and decision record, implement/test the mechanism with an explicitly labelled synthetic fixture, and continue independent work. Do not invent a physical calibration or alter a primary estimand to bypass a decision.

## Required implementation outcomes

### WP0–WP1: status, data contract and stage ownership

Reconcile the latest issue and obsolete A0 wording. Define a versioned detector-response contract, target-PMT provenance and photon-stage ownership. Specify whether input photons have already undergone QE, CE, TTS or other response. Each effect must be applied once. Keep latent wavelength, truth and clean twins out of operational monitor inputs. Separate synthetic parameters from measured or pending calibration. Use a physical crate mapping where available and label synthetic maps honestly.

### WP2: wavelength QE and identifiability

Add a bounded wavelength-QE interface, explicit CE/gain conventions, persistent photon IDs and paired random streams. Validate probabilities, wavelength domains and analytic detection rates. Add an exact observable-degeneracy control and a separately labelled charge-matched comparison. Do not claim that internal representations recover causes that produce identical measured observations. If upstream exposes only post-QE hits, do not pretend a second thinning step reproduces arbitrary wavelength-QE changes.

### WP3: fixed-threshold timing location and spread

Add a bounded polynomial charge–time relation calibrated on the reference, with a separate charge-dependent timing-width model. Preserve reference physical noise while removing only its systematic calibration residual. In the pure timing intervention, keep the threshold, charge values and pre-window hit membership fixed. Test sign conventions, curve boundaries, timing distributions, deterministic pairing and zero-change identity. Preserve the old threshold-walk intervention under its historical meaning. Avoid double TTS, unsupported extrapolation and conflating the mean/mode or width conventions in thesis Eqs.6.2–6.3.

### WP4: coupled group response

Implement module/crate-dependent amplitude and timing-width factors as one explicit coupled intervention, with gain-only and width-only controls. Distinguish timing width from timing offset. Account for legitimate gain-to-threshold and gain-to-time-walk coupling. Treat this as a B-field-inspired proxy until target-specific field calibration exists. Adapt the one-field cell-validation rule explicitly rather than bypassing it.

### WP5: adapter, consequence and evidence repairs

Build or validate the LUCiD photon adapter without assuming `wavelength_mode` alone supplies the required data. Produce a new prompt-only synthetic fixture; preserve the old toy results.

Separate energy calibration and dead-map correction fitting from evaluation events, using a frozen, hashed reference-calibration artifact. Validate the point-source endpoint and document the additional requirements for a track endpoint. Repair silent missing-endpoint aggregation and expose fit failures without making severe corruption look benign through event exclusion. Keep primary D-K decisions pending where required.

Replace simulator-name-based citability with explicit evidence stage. Every new output in this pass stays development/non-citable, including LUCiD-sourced output. Prepare the supporting cone-angle diagnostic only if physically defined by the available event model; otherwise record the precise missing inputs, rather than inventing an energy-loss inference.

## Validation and reporting

- Test meaningful invariants and statistical behavior, not only function existence or mirrored implementation. Declare stochastic tolerances and seeds before inspecting outcomes; do not loosen them to hide a failure.
- Run the focused PMT suite after changes, then the relevant existing noise/latent-monitor regression checks. Record expected skips and environment limitations. No full training is needed for this pass.
- Keep independent calibration/event partitions demonstrably disjoint. Test that operational features cannot read truth, clean twins, intervention labels or latent photon wavelengths.
- Save new CPU validation results with code/configuration hashes and exact commands in a uniquely dated PMT directory. Do not overwrite old evidence.
- Update `docs/PMT_ISSUE5_STATUS.md` and write `docs/PMT_ISSUE5_FINAL_REPORT.md`. For each requirement provide the source, code location, test/result evidence and unresolved boundary. Mark the overall scientific issue as only partially complete if WP6–WP8 or target-detector validation remain outstanding.
- Prepare `docs/PMT_EXPERIMENT_CONTRACT.md` as a draft containing the model/data interfaces, candidate valid S families, split design, comparison requirements and unresolved author decisions. No protocol freeze or training launch.
- Finish with local commit IDs, changed files, tests, completed work packages, remaining scientific decisions and a paste-ready proposed issue response. Do not post that response.

Prioritize a reviewable implementation of the requested PMT corrections. Passing synthetic tests establishes engineering readiness only; it does not establish target-detector calibration, scientific superiority or issue closure.
