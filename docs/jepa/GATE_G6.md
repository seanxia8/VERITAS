# G6 — unresolved execution inputs

Status: OPEN, scoped. Registration remains valid; this gate is not a new specification P1. No affected outcome grid has run.

## Waveform generating law

Runbook §5 requires additive narrow-line noise, exact physical covariance, independent sine/cosine targets and two declared common-mode temporal basis coefficients. Final execution review, “Remaining intentional gates,” explicitly leaves bandwidth/normalization and common-mode construction to source/config review. `src/latent_monitor/tier1.py:201` defines the reference base, and its existing line variant at lines 228–233 is a normalized composite PSD variant, not the declared independently added component. It therefore cannot uniquely supply the JEPA additions. The time-basis covariance, channel loading, strength convention and whether the two sweeps are separate axes or a Cartesian grid must be resolved explicitly before derived configs/manifests can be frozen. Author approved amendment A1 in GATE_G5.md on 2026-10-16 before outcomes. The generating law is now specified; implementation and full covariance validation have passed independently; all nine fixed cells now have population eligibility evidence. Actual used-host acceptance remains open.

Affected: E1b, E2 development/calibration and training, E3's transfer of E2 beta, and E4. Independent exact-model analytic checks and source review may continue. See GATE_G5.md for the approved resolution. THEORY.md retains the pre-approval analysis for traceability.

## Prospective numerical configuration

WP2's fixed linear solver configuration has now been independently accepted at SHA256 52fa6a91556259144aff7925555dc7133c6135806db8c94bdad834d1e7899fa7 (see reviews/WP2_INDEPENDENT_REVIEW_2026-10-16.md). WP4's numeric optimizer/SIGReg configuration still requires pre-outcome review. No sample grid or substantial smoke is cleared while these are missing. Population identity/gradient fixtures are permitted implementation checks. This item may be resolved by a reviewed fixed numerical configuration without changing scientific endpoints.

## E3 manifest

No accepted E3 manifest, release schema, acquisition roles or target-spread verification exists yet. Bounded remote metadata discovery does not establish data absence or eligibility. Do not train, compute calibration feature/coherence outcomes or broaden file selection before the prescribed candidate manifest exists. Remote source evidence is in PREFLIGHT_REMOTE.md when finalized.

Resolution must identify exact inputs, source hashes, affected scope and acceptance evidence. No G4 feasibility result or empirical prediction failure is inferred from missing inputs.

## Remote environment STOP

Dedicated Python3.11/torch2.5.1 CPU environment installation on Deepthought failed with `Disk quota exceeded` (exit 2). Failed attempt is preserved; no retry, cleanup or alternate environment was silently adopted. WP0 remote import acceptance is incomplete. This is an infrastructure/data-access preflight blocker, not evidence that campaign G2 resource caps were reached. See PREFLIGHT_REMOTE.md and its immutable remote return manifest.

## E2 fixed-grid level selection — STOP

Runbook §6 WP5 requires three distinct line levels nearest population gamma_n targets .1/.5/.9, ties lower variance. At the approved A1 common strength .45 and default time split, the fixed line grid 0/.75/1.5/2.25/3 gives gamma_n .9662047091901853/.9866537783591497/.9916848481112152/.9939612308755621/.9952591073661169. All three targets therefore select line multiplier0. Distinct levels cannot be selected. Evidence and source hashes: `results/jepa/20261016_wp2_E2_LEVEL_SELECTION.json`; independent recomputation confirmed the same selection; see the WP2 review addendum.

Stop the dependent E2 comparison, E3 transfer of selected E2 beta, and WP7 dependent execution. Preserve these values; do not expand the grid, alter A1 strength, switch default split or tune after this feasibility finding. Independent WP2/E1 work can continue subject to its own gates. This is a design-feasibility G6, not empirical failure of R6–R8, and no GPU job has run.

Population rank qualification: original time/block target signal rank is2, while the conditional innovation target rank is3. Their P5 recovery checks are eligible; these are not evidence for unconditioned P3 recovery. All36 cell/split P5 checks passed. Remote quota recovery and its sole fresh-environment retry are documented separately; the initial failure is preserved.

## E1 missing probe/cohort contract

Runbook §1.2 forbids inventing unlisted scientific choices. A3 in E1_PROBE_CONFIG_PROPOSAL.md provides a concrete, currently unapproved proposal for probe sample counts, groups, noise-calibration estimator information and M diagnostic scope. Author decision requested before dependent E1 sample grids. A2 dynamics engineering may continue under its already approved specification.

## Sole remote infrastructure retry exhausted

The authorized fresh-directory retry reached its600second allocation and exited124 on2026-10-16 at15:32:24UTC during CPU torch installation. Imports were not reached. Quota headroom was restored, but installation throughput did not permit completion within the allocated retry. Preserve both environments/attempts. No second infrastructure retry, environment substitution or dependent remote experiment is authorized by this recovery record. Compact executor evidence and exact timing were returned and verified; see PREFLIGHT_REMOTE.md and STORAGE_RECOVERY.md. This is infrastructure STOP, not a measured scientific failure or campaign G2.
