# Independent execution review of JEPA v3 — 16 October 2026

**Final verdict: ACCEPT for the reviewed execution specification after the repairs recorded below. No open P1/P2 execution findings remain in this review. This is not execution approval.** The major v2 execution defects are repaired. One remaining scientific assignment is ambiguous, and two small implementation/handoff repairs should be made before affected work. No remote commands, experiments, training, downloads or commits were performed.

Reviewed files and SHA-256:

```text
RUNBOOK_JEPA_NOISE_2026-10-16.md
15e64a36ea57dbf170209fa388789e09f5ebcfaa7edc900ed433ea8983c921e3
AGENT_PROMPT_JEPA_2026-10-16.md
1a30efead8f277fed982c87c82f264078eda368e035d6eada7b85032970097f4
```

Line numbers refer to those files at those hashes. Source cross-checks use repository HEAD `0cae6b061e488e319643cfb0ba820318b8923578`; both runbook and prompt remained untracked at inspection. `RUNBOOK` and `PROMPT` below denote these reviewed files.

## Initial findings, all subsequently resolved

### V3-E01 — P1: assign the fixed baseline split explicitly, especially in E3

**Quoted premise:** RUNBOOK 241: masked arm uses “one split selected by M”; 300: “Freeze chosen gap before SSL training”; 308: E3 must “Retain E2 objective definitions”. Its two E3 candidate masks are listed at 295–298, but no sentence assigns `time_short` or `time_long` to the four non-masked E3 arms. RUNBOOK 271 also says oracle transformations use “the fixed selected split”, although arm 9 inherits arm 4, not the M-selected arm 5.

**Consequence:** There are two literal readings: use the M-selected gap for every E3 arm, making `sigreg_jepa_masked` identical to plain SIGReg; or use an unspecified fixed default for plain/anchored/EMA/MAE. These are different comparisons. In E2, the phrase about the oracle can similarly select either the ordinary time mask or the M-selected mask. The protocol must not leave this scientific assignment to executor preference or post-calibration inference.

**Minimal exact repair:** Add: “E2 arms 1–4 and 6–9 use the fixed `time` split. Only arm 5 uses M's selected split. The oracle uses that fixed `time` split for both training and frozen-probe preprocessing. E3 non-masked arms use `time_short`; only `sigreg_jepa_masked` uses M's selected candidate. Under the 24-frame fallback, `time_short` means `[0,8) -> [16,24)`, and `time_long` means `[0,4) -> [20,24)`.” If another baseline is intended, explicitly name it instead. Record when M selects the baseline itself: then masked/plain are a deliberate null contrast, not two different interventions.

### V3-E02 — P2: bind reconstruction-head inputs and test the reconstruction prediction path

**Quoted premise:** RUNBOOK 231 introduces a target-only encoder and a predictor; 237 calls arm 1 “Context-masked reconstruction via linear target-token head”; 238 says arm 2 uses the same reconstruction. The head's source tensor is not explicitly named.

**Source evidence:** The imported backbone can expose all token positions before pooling (`src/reconstruction_model/model.py:265–269`). With the v3 architecture, there are multiple target-position tensors: context-encoder masked-position outputs, target-only encoder outputs, and predictor outputs. A head attached to the target-only tensor receives the true target samples. The current zero-dependence test concerns encoder visibility and can pass even if the reconstruction branch is connected incorrectly.

**Consequence:** A mistaken target-head routing would turn the MAE control into target autoencoding and corrupt its comparison, without necessarily violating the stated encoder tests. This is a preventable ambiguity, not evidence of a bug in code that has not yet been implemented.

**Minimal exact repair:** Specify the intended tensor, for example: “Arms 1/2 reconstruction heads read target-position outputs of the context encoder, whose raw target/gap patches were replaced before attention; they never read target-encoder outputs. The hybrid reconstructs through this same branch.” Alternatively use predictor outputs, but declare the additional projection/head dimensions. Add zero-gradient and finite-perturbation tests from the final reconstruction **prediction** to raw target/gap samples; the loss itself may depend on target labels.

### V3-E03 — P2: restore an explicit remote evidence return and review-access contract

**Quoted premise:** PROMPT 16 says executors “return evidence as prescribed in §1.2”. RUNBOOK §1.2 (31–42) prescribes provenance and local-only committing, but no evidence-transfer procedure. §3.1 names `raw` and `returns` directories; Appendix A.3 (374) defines executor responsibilities without stating who prepares summaries, writes checksums or retrieves review inputs.

**Consequence:** An executor can finish correctly while the orchestrator lacks a specified route to per-seed raw evidence for independent recomputation. The dangling section reference is a handoff defect; it does not authorize remote commits or an uncontrolled raw-data copy.

**Minimal repair:** Define one transfer contract under §1.2: executor stores raw arrays/checkpoints under `/ceph/dwong/jepa/raw/<experiment>/`, writes per-seed results/provenance and an artifact manifest with hashes, generates small summaries under `returns/<experiment>/`, and reports exact paths plus exit status. The orchestrator retrieves those summaries to the registered local results path and verifies hashes before committing. State that reviewers either read immutable raw artifacts on the run host or retrieve the explicitly listed bounded inputs, verifying the manifest. Update PROMPT's reference to the actual subsection. Use a host-qualified path for L40S returns rather than assuming all raw files physically reside on Deepthought.

## Repairs verified in v3

- **Noise support:** §4.1 now adds real independent sample noise to the generating law, with matching covariance/CRLB treatment; it no longer claims that an inversion floor fixes the physical DC nullspace. Gap samples are discarded before oracle construction.
- **Patch alignment:** E2 time masks now use context patches 0–2, gap 3–4 and target 5–7, exactly matching `[0,96)`, `[96,160)`, `[160,256)`. E3 96-frame and 24-frame boundaries are four-frame aligned. At n_fft4096/hop2048, 51200 samples indeed yield 24 frames. Declared gaps separate raw STFT supports.
- **Independent development:** mid/nominal is retained in evaluation, with distinct development IDs/model seeds. R6's three levels and seed-mean aggregation are compatible.
- **Criterion M:** separate event/noise calibration sources and explicit projected matrix expressions resolve the noise-only calculation defect. Its heuristic status and possible out-of-range signal strength are stated.
- **E3 roles and targets:** CH1/CH2 roles match the repository readers, CH2 is excluded from inputs, and frequency becomes the primary retained-physics endpoint with explicit spread requirements. File selection, window guards, overlays, calibration and test roles are now specified. Dataset feasibility is properly deferred to G3/G6 rather than asserted as verified.
- **Dependencies and optimizers:** `jaxtyping`, torch versions, import checks and exact optimizer coverage are explicit. No protected-package edit is needed.
- **Resources:** separate E1/E2/E3/E4 caps include preparation, failures, smoke, tuning, probes and reserve; benchmarks are bounded; E3 fallback is reprojected; WP8 needs its own approved budget. No permission to bypass GPU/push flags is introduced.
- **Empirical failures:** outcome states distinguish failure, ineligibility and not-run. A failed finite-beta or deep prediction is no longer forced into a theorem acceptance test.

## Remaining intentional gates, not additional defects

Solver convergence configuration, numeric optimizers/SIGReg implementation, missing data/metadata, covariance stationarity and live resource projections are explicitly gated before their affected outcomes. Their values need review and freezing, but the document does not authorize the executor to invent scientific defaults. The exact added-line bandwidth/normalization and common-mode construction must also be resolved in those source/config reviews under §§1.2 and 5; until then they are implementation inputs pending validation, not measured facts.

This review does not establish that the selected TIDMAD manifest exists, that its frequency-spread rules can be met, or that CPU/GPU budgets suffice. Those are live preflight/data checks. No tests or benchmarks were launched during this v3 re-review; checks consisted of source inspection, mask-support arithmetic and file hashing. The report retains the Markdown format used in the preceding execution review.


## Final reinspection and dispositions

The orchestrator repaired the specification during review. Final reviewed RUNBOOK SHA-256: `5f8de710a39e74a6f76235eeaf46c4df0907bfbe5145b1d3aa88e3a9fff1e6cd`. PROMPT SHA-256 remains `1a30efead8f277fed982c87c82f264078eda368e035d6eada7b85032970097f4`. Initial findings and their source-hash-bound line references above are preserved.

- **V3-E01 resolved:** final RUNBOOK 235 fixes E2's default time split; 275 names that same fixed split for oracle probes; 304 fixes E3 non-masked arms to time_short and M only to the masked arm, including honest identical-mask outcomes; 320 explicitly names both fallback candidates.
- **V3-E02 resolved:** final RUNBOOK 249 explicitly routes reconstruction heads 1/2/7 through context-encoder target-position outputs and adds final-prediction gradient and perturbation checks. Target-only encoder outputs are prohibited on that branch.
- **V3-E03 resolved:** final RUNBOOK 42 specifies compact return artifacts, rsync, verified hashes, immutable attempts, host-qualified raw paths, reviewer access, and GPU-host transfer staging. The prompt's §1.2 reference now resolves to an actual handoff contract.

Final execution-contract verdict is ACCEPT, conditional only on the already explicit subsequent implementation/data/compute gates. This review does not clear live host or dataset feasibility, authorize a launch, validate unimplemented code, or imply scientific success. Only document/source inspection and hashing were performed. No remote jobs, training, commits or outcome-generating checks were launched.
