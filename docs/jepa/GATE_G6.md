# G6 — unresolved execution inputs

Status: OPEN, scoped. Registration remains valid; this gate is not a new specification P1. No affected outcome grid has run.

## Waveform generating law

Runbook §5 requires additive narrow-line noise, exact physical covariance, independent sine/cosine targets and two declared common-mode temporal basis coefficients. Final execution review, “Remaining intentional gates,” explicitly leaves bandwidth/normalization and common-mode construction to source/config review. `src/latent_monitor/tier1.py:201` defines the reference base, and its existing line variant at lines 228–233 is a normalized composite PSD variant, not the declared independently added component. It therefore cannot uniquely supply the JEPA additions. The time-basis covariance, channel loading, strength convention and whether the two sweeps are separate axes or a Cartesian grid must be resolved explicitly before derived configs/manifests can be frozen. Author approved amendment A1 in GATE_G5.md on 2026-10-16 before outcomes. The generating law is now specified; implementation, exact covariance and eligibility validation remain open.

Affected: E1b, E2 development/calibration and training, E3's transfer of E2 beta, and E4. Independent exact-model analytic checks and source review may continue. See GATE_G5.md for the approved resolution. THEORY.md retains the pre-approval analysis for traceability.

## Prospective numerical configuration

WP2's fixed solver and WP4's numeric optimizer/SIGReg configuration require pre-outcome review. No sample grid or substantial smoke is cleared while these are missing. Population identity/gradient fixtures are permitted implementation checks. This item may be resolved by a reviewed fixed numerical configuration without changing scientific endpoints.

## E3 manifest

No accepted E3 manifest, release schema, acquisition roles or target-spread verification exists yet. Bounded remote metadata discovery does not establish data absence or eligibility. Do not train, compute calibration feature/coherence outcomes or broaden file selection before the prescribed candidate manifest exists. Remote source evidence is in PREFLIGHT_REMOTE.md when finalized.

Resolution must identify exact inputs, source hashes, affected scope and acceptance evidence. No G4 feasibility result or empirical prediction failure is inferred from missing inputs.

## Remote environment STOP

Dedicated Python3.11/torch2.5.1 CPU environment installation on Deepthought failed with `Disk quota exceeded` (exit 2). Failed attempt is preserved; no retry, cleanup or alternate environment was silently adopted. WP0 remote import acceptance is incomplete. This is an infrastructure/data-access preflight blocker, not evidence that campaign G2 resource caps were reached. See PREFLIGHT_REMOTE.md and its immutable remote return manifest.
