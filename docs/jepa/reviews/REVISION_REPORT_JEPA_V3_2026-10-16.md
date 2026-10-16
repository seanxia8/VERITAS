# JEPA v3 revision record — 16 October 2026

Status: revised specification, unregistered. This revision task authorizes documentation repair and bounded local verification, not registration, push, remote jobs or training. The historical v2 reviews retain their original verdicts and line references; their source snapshots are in `snapshots/v2/`.

## Disposition of the prelaunch findings

| Finding | v3 repair | Verification still required before affected execution |
|---|---|---|
| P5 unconditional oracle claim | §2.2 requires innovations-adjusted signal rank, canonical margin and negative control | Population proof and eligible/ineligible fixtures in WP1–WP2 |
| P4 theorem conflated with finite-beta/deep training | §0/§2 separate pure constrained GEVP from empirical training hypotheses | Combined-objective gradient/convergence checks; negative outcomes retained |
| E1b nonlinear family lacks fixed A | WP3 uses a fixed Jacobian and declared perturbation distribution | Actual view/residual ranks and canonical margins; tail-only views may be ineligible |
| Singular Tier-1 covariance | §4.1 declares actual independent white noise with fixed 1% reference variance, used in generation and covariance | Actual covariance agreement and Fisher checks; protected generator unchanged |
| Incompatible E2 gap | E1b/E2 use sample-aligned [0,96), [96,160), [160,256) and matching tokens | Raw-input visibility and patch boundary checks |
| M cannot use noise alone | §2.3 specifies independent event/noise calibration, projection, regularization and tie rule | Estimator implementation and manifests; M is a heuristic |
| R6 lacks independent middle cell | WP5 has separate development IDs and all six evaluation cells | Namespace/hash checks and raw contrast recomputation |
| Incorrect TIDMAD channels | §7 binds CH1 SQUID and CH2 reference; temporal SQUID-only views | Actual HDF5 schema and no-reference-input tests |
| Undefined E3 data/target protocol | §7 defines ordered eligibility, role manifests, guards, schedule and target spread | Data availability and label eligibility remain unmeasured |
| Incomplete compute accounting | §3 gates before substantial smoke; full campaign ledger and E3 fallback reprojection | Host timing and resource availability remain unmeasured |
| Dependencies and parameter coverage | CPU/GPU torch compatibility, jaxtyping import, exact optimizer membership | Import and parameter-coverage tests on actual execution hosts |
| Unspecified scoring/tuning | §1 fixes grids, grouped probes, aggregation, ties and descriptive uncertainty | Reviewed fixed numerical solver and optimizer/SIGReg config before grids |
| Stop-gradient theorem overreach | §4.2/E1c define finite-horizon empirical dynamics and censoring | No claim that a random-loading model satisfies published diagonal assumptions |
| Toy parameter mislabel | Printed `gamma_n` renamed `gamma_load` only | Compare unchanged numeric output to historical toy |

## Explicit design changes

These are prospective design choices, not evidence of improved results: the actual 1% independent white-noise component; a 64-sample E2 gap; linearized E1b; independent development/evaluation populations; projected heuristic M; and E3 log-frequency R² as the primary physics-retention target. Amplitude R² is supporting only when its declared variation threshold holds. Reports must identify v3 and cannot compare its endpoints to v2 as if the protocols were identical.

Exact solver/optimizer settings and actual data feasibility are implementation gates, not completed checks. Existing toy and P5 counterexample results are disclosed prior evidence. No new model outcomes were used in this revision. Lack of rank or target variation must yield a documented ineligible result or the specified gate, never silent redesign.

## Review and verification

The subsequent `SCIENTIFIC_V3_REVIEW_2026-10-16.md` and `EXECUTION_V3_REVIEW_2026-10-16.md` provide independent review of this revised specification. Any remaining material finding must be resolved or explicitly recorded as blocking its dependent work before registration/execution. Local bounded checks are recorded in `V3_LOCAL_CHECKS_2026-10-16.txt`.

The two-file handoff remains intentional: send the launch prompt; it directs the executing agent to the full runbook and review record. Flags are now inside the copied prompt so permissions are not lost when forwarding.

## Follow-up repairs from independent v3 review

The second read caught additional specification defects, repaired before handoff: E1c learning time now requires absolute growth over initialization and terminal stabilization, with collapse/flat-curve negative controls; P5 explicitly requires the retained joint covariance to be positive definite; E2/E3 ordinary and M-selected masks are assigned to named arms; MAE/hybrid reconstruction heads read context-encoder outputs with prediction-level leakage tests; and remote returns include checksums and raw-evidence access. The E3 fallback retains named baseline/selected mask roles. Independent review reports retain their initial findings and append final dispositions so the repair history is auditable.

Local checks passed: historical v2 hashes, exact toy label-only diff, reproduced toy and P5 counterexample, E2 patch/sample agreement, E3 STFT support and fallback lengths, and R5 synthetic growth/collapse eligibility. These checks do not validate a future implementation or actual TIDMAD data. The branch HEAD remains the original base; no registration commit was created.
