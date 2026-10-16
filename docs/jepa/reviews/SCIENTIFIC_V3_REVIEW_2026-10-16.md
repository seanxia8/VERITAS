# Independent scientific review of JEPA v3

Date: 16 October 2026. Scope: actual revised `RUNBOOK_JEPA_NOISE_2026-10-16.md` and `AGENT_PROMPT_JEPA_2026-10-16.md` on disk, including mathematical assumptions, masks, calibration/evaluation partitions and predictions. No experiments, training, remote access or plan edits were performed for this review.

**Final verdict: ACCEPT for the scientific specification at the hashes recorded below. The scoped R5 blocker and P5 joint-SPD clarification have both been resolved and independently reread.** This is not execution authorization, data acceptance or confirmation that future G6 checks will pass. No scoped P1 remains after the follow-up correction recorded below; the principal v2 theorem and masking defects have been addressed.

## Finding retained for traceability — resolved in the same review cycle

### [P1, E1c only] Encoder-norm normalization can classify a collapsing model as having learned every mode

**Quoted premise:** runbook line 180: “Normalize each curve by `norm(W)` to remove common scaling,” followed by “require final normalized contribution >1e-6” and a persistent crossing of 50% of final value. R5 at line 119 interprets those times as learning order.

**Counterexample by algebra:** take any nonzero initialization `W0` for which `W0 A_i` and `W0 B` are nonzero, and let `W(t)=a(t) W0` with positive `a(t)` tending to zero. Every specified normalized contribution is

```
||W(t) A_i|| sd(theta_i) / ||W(t)||
    = ||W0 A_i|| sd(theta_i) / ||W0||.
```

It is constant, positive, and can easily exceed 1e-6 while the unnormalized representation vanishes. All 50%-final crossings occur at the first step and persist for 100 steps. The rule therefore calls an unlearned or globally collapsing encoder eligible, despite the intended nonlearning/censoring convention. Even without collapse, an initially nonselective random map can start above half of its eventual normalized contribution; the time then measures relative allocation, not acquisition of a feature.

**Consequence:** R5 can yield apparently valid learning times for a trajectory that contains no learning. This is a construct-validity defect in the metric, not a prediction that R5 must succeed or a claim that the scheduled GD will necessarily collapse.

**Correction:** retain normalized curves as relative-allocation diagnostics only. Define primary learning times from absolute mode contributions with a frozen initialization baseline, a prespecified minimum absolute increase/noncollapse condition, and a terminal stabilization condition. A baseline-relative threshold such as half of the eligible final increase is suitable once its growth and noise-floor criteria are explicitly frozen. Require all modes to pass eligibility before forming R5; no positive growth or no stabilized final level means censored/not-evaluable. Alternatively rename the current measure to relative-allocation timing and narrow the scientific claim, but it cannot by itself identify learning onset.

## Minor clarification

**P5 joint positive definiteness:** line 75 explicitly assumes positive-definite marginal noise covariances, while line 87 asks for an inverse lower Cholesky of the joint noise matrix. Positive-definite marginals alone do not imply a positive-definite joint matrix. State `N_joint ≻ 0` explicitly in P5. This is a theorem-domain clarification rather than a new failure of the configured waveform law: lines 168–172 already add a positive white component there, and invalid covariance handling is gated. A singular joint-noise fixture should be rejected before Cholesky rather than repaired with undeclared jitter.

## Addressed findings and scientific boundaries

- **P3/P5:** lines 75–89 distinguish population/global-optimum results and require residual target rank and conditioning; line 87 discards gaps before the two-view oracle transform. The annihilation example is correctly a negative control. Sample angular accuracy remains a separate empirical claim. Subject to explicit joint SPD, the revised P5 statement is sound.
- **P4:** lines 86, 117–118 and 194 correctly separate the pure GEVP from finite-beta hard-constrained optimization and the deep penalty. A finite-beta failure is now a legitimate finding, not a theorem-test failure.
- **Masks:** lines 204–217 and 229–231 align the sample gap with two full patches and test both context and target visibility. Oracle target dependence on raw context is appropriately acknowledged.
- **E1b:** a fixed Jacobian fixture is specified at line 204, with numerical rank and residual-rank eligibility checks at 217. A tail-only target may fail these checks; retaining that as an ineligible diagnostic is appropriate. No replacement split is silently authorized.
- **M:** lines 91–105 define event-normalized cross-signal strength, projected dimensions, calibration sources and deterministic selection; it is explicitly a heuristic. This resolves the impossible noise-only signal estimate. Real-data stationarity and finite-sample quality remain empirical limitations, not an exact latent recovery claim.
- **E2 cohorts:** lines 253–269 provide a separate development cohort and independently seeded evaluation middle-level cohort. The three-level R6 statistic and coefficient selection are now defined without reusing a tuned checkpoint.
- **E3:** lines 279–310 state the corrected single-SQUID/reference contract, forbid CH2 model inputs, supply file/interval roles and guards, and gate insufficient data/target spread. File eligibility, frequency spread, actual independence units, stationarity and resource feasibility require future inspection; a possible G6 stop is not a contradiction in the scientific specification. The frequency endpoint is explicitly disclosed as a v3 change. This review did not inspect raw data or verify that the chosen release files can satisfy the eligibility rules.
- **Statistics:** lines 46–53 distinguish descriptive seed effects from powered confirmatory claims, support not-evaluable statuses, and define probe tuning. The small seed counts still constrain precision but are no longer presented as inferential guarantees.
- **Launch prompt:** requires review dispositions and no unresolved P1 before registration, respects G0–G6, and does not convert a review request into execution authorization. Numeric solver/optimizer/SIGReg configurations are deliberately deferred to pre-outcome review gates; this is explicit future work rather than silently missing authorization.

The E1c metric is now repaired; once the joint-SPD assumption is explicitly stated, no additional scientific blocker was identified in this read. Implementation correctness and dataset feasibility remain subject to the runbook's prospective tests and gates.

## Follow-up inspection: R5 correction verified

The orchestrator revised §4.2 after this finding. I reread the actual on-disk text: primary curves now use absolute contributions q(t), frozen q0, final last-1000 mean qf, finite-value checks, qf>1e-6, qf>=2q0, adjacent terminal-block means within .05qf, and a persistent baseline-relative half-increase crossing. Zero, constant and globally shrinking negative controls plus a rising plateau positive control are mandatory. Normalized curves are diagnostic only. This resolves the concrete collapse/initialization counterexample. It remains a prespecified empirical timing definition, not proof that all modes converge or that R5 will pass. No experiments were run in this follow-up. Original line references above describe the draft inspected before the same-day follow-up edits.

## Final disposition and source binding

Reread the final P5 text at runbook line 89: it explicitly requires the marginal **joint** retained-coordinate noise matrix to be positive definite, states that marginal positive definiteness alone is insufficient, and rejects singularity without undeclared jitter. The minor clarification is resolved. Together with the verified R5 correction, no scientific P1 remains from this review. Final verdict is **ACCEPT of the specification**, subject to its explicit prospective implementation, data, calibration and resource gates; this is not launch authorization or evidence that empirical predictions will pass.

SHA-256 of final reviewed source:

```
5f8de710a39e74a6f76235eeaf46c4df0907bfbe5145b1d3aa88e3a9fff1e6cd  docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md
1a30efead8f277fed982c87c82f264078eda368e035d6eada7b85032970097f4  docs/jepa/AGENT_PROMPT_JEPA_2026-10-16.md
```

Historical findings and conditional wording above are retained to document the review-and-correction sequence; this final disposition supersedes their open status.
