# Launch prompt — JEPA v3 noise-weighting study

Send everything below the divider to the executing agent. The runbook is referenced by path; you do not need to paste it separately. This prompt is for a subsequent execution task. Editing or reviewing it does not launch the study.

---

PUSH_JEPA=yes
GPU_ALLOWED=yes
OPTIONAL_WPS=WP7

You are the orchestrator for VERITAS on branch `JEPA`, local checkout `/Users/wongdowling/Documents/VERITAS`. Read `docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md` in full, version **v3**, then `docs/jepa/reviews/REVISION_REPORT_JEPA_V3_2026-10-16.md` and the latest v3 independent review. The runbook is the scientific and execution contract. The author explicitly authorizes the orchestrator to push branch `JEPA` to origin and use the authorized Deepthought GPU allocation through the runbook's Condor workflow, within its budgets and gates. No further permission is needed for those scoped actions. Optional work can be `none`, `WP7`, or `WP7,WP8`.

1. First verify branch, base SHA, source hashes, local changes and review dispositions. Preserve unrelated changes and the separate PMT worktree/branch. Do not register a specification with unresolved P1 findings. Known toy/counterexample outputs are prior development evidence.
2. After those checks, execute WP0 through WP6 in order, then the selected optional work subject to its own budget/gates. Create the v3 registration commit before new outcome-generating grids. Reproduce Appendix A and the P5 annihilation negative control. Only the orchestrator commits.
3. Delegate bounded implementation work and remote execution to separate subagents. Use fresh-context independent reviewers at WP2, WP5 and completion, supplying source artifacts rather than implementer conversations. Integrate their findings and verify acceptance yourself.
4. Follow §3 for Deepthought transfer, environments and resources. Remote code arrives through GitHub at a pinned SHA. With PUSH_JEPA=no, if the needed SHA is absent from origin/JEPA, record G0 and wait for the author to push; local independent checks may continue. No GPU work with GPU_ALLOWED=no. No unbudgeted CPU substitute for GPU training. Remote executors never commit/push and return evidence as prescribed in §1.2.
5. Resolve fixed solver/optimizer/SIGReg settings and data manifests through the specified pre-outcome review gates. Project the entire campaign before substantial smoke/tuning, including probes, depth, oracle and fallback costs. Keep a cumulative resource ledger; recheck the E3 fallback and stop if still over budget.
6. Enforce G0–G6. Record evidence in `docs/jepa/GATE_<id>.md` and stop the affected dependent work pending resolution. Do not alter registered thresholds, expand data, retune after scientific failure or silently introduce another fallback. Preserve negative/ineligible/censored results. P5 requires residual signal rank; finite-beta/deep recovery is an empirical prediction, not a theorem unit test.
7. Verify sample/token visibility, exact optimizer coverage, independent development/evaluation cohorts, actual simulated covariance, CRLB support, and all TIDMAD roles. CH2 is a delayed label only. E1b is linearized; E2 is nonlinear. E3's primary retained-physics endpoint is frequency, with amplitude only if eligible.
8. Respect the no-touch list and PMT isolation. Complete `docs/jepa/COMPLETION.md` and `docs/jepa/REPORT.md` with each WP's evidence, R1–R10 outcomes (including not-evaluable), budget, fired gates and remaining decisions. Engineering checks, empirical outcomes and manuscript readiness are distinct.

Start by reporting the verified version/branch and unresolved gates, then carry out the authorized steps. Push/GPU permission is explicitly granted above; all scientific, resource and acceptance gates still apply. This author-approved permission update supersedes the earlier prompt's `no` flags; historical review hashes refer to the prompt before this update.
