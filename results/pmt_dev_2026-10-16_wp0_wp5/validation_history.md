# Validation history

Before any new stochastic outcomes, test_pmt_contract.py declared seeds 4/71, six-SE rate and mean bounds, 3% width tolerance, and 2 mm noiseless closure.

First run: 85 passed, 1 failed (4.37 s). The noiseless point-source test returned error 0.056840054 m despite optimizer success. The inherited Nelder-Mead default simplex is nearly collapsed when the coarse-grid start is zero. Repair: use a detector-scale initial simplex (5% radius/half-height). Test bound unchanged. Historical toy result folders remain untouched.

Second suite: 86 passed in 4.73 s; after channel/frozen-fit checks: 87 passed in 4.36 s.

First end-to-end attempt (`prompt_run_01`) stopped before calibration output: charge exceeded the declared [.25,100] PE T-Q domain for a 4000-photon source in the small 76-channel fixture. No extrapolation occurred. A separate source-manifest serialization failed on NumPy int64 group IDs; normalized to Python ints. These are engineering failures, not scientific results.

For `prompt_run_02`, set source yield to 1000 incident photons, keeping geometry, response curves/domain, seeds, closure bounds and all test tolerances unchanged. This contains the synthetic workload within the declared calibration domain; it is not a target-detector calibration adjustment or a severity search. Retain the failed folder and this record.

Second fixture stopped before calibration output on **lower** domain boundary: charges 0.2445–0.2483 PE fired the historical sampled-template discriminator. Inspection of response._template_peak_gain explains its continuous peak exceeding the sampled PE peak. First run's upper-bound violations were independently confirmed (102.9–135.2 PE). For the new polynomial mode, define the discriminator in continuous PE peak units (`template_fs=None`), so .25 PE means exactly .25 PE. The legacy mode retains sampled-template semantics. Added boundary regression at .245/.25 PE. No domain/tolerance changed. `prompt_run_03` uses the same 1000-photon workload and seeds as run 02.

Noise regression direct run stopped during session setup: existing conftest regenerates shared noise-module .dat files outside task scope. Run that unchanged package from a temporary copy instead; do not alter shared package files. This was a sandbox write failure, not a test assertion or approval rejection.

Run 03 passed all four engineering checks. Run 04 added explicit QE spectrum integrals and refreshed code hashes. Final audit found duplicate dead channel indices could double-count correction shares; these now fail validation (one new test). Coupled causal-control fields are now persisted in per-cell provenance, and legacy electronic outputs no longer use the stale A0 flag. Run 05 binds those final code hashes; scientific inputs, seeds and tolerances are unchanged. Final PMT suite: 93 passed in 5.10 s. Noise: 145 passed; latent/TIDMAD: 104 passed, 2 skipped.

Staged diff check caught CSV writer CRLF and pytest traceback trailing spaces. Normalized only new evidence CSV/log line endings/trailing whitespace; parsed CSV rows verified identical. Before/after hashes are in formatting_normalization.json. No old evidence or scientific values changed.

18 October 2026 (runbook v2 R-5): `prompt_run_03`, `prompt_run_04` and `prompt_run_04.log` were removed from the tree as duplicates of the final `prompt_run_05`. They remain in git history at `56179eb`, and their recorded verdicts are unchanged.
