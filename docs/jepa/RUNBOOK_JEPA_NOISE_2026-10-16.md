# RUNBOOK — Predictable Is Not Physical: JEPA on detector signals

_16 October 2026, v3, revised after independent prelaunch review. Owner: Dowling Wong (KIT); collaborator: Junjie Xia. Branch `JEPA`, fixed base `0cae6b061e488e319643cfb0ba820318b8923578`. Status: unregistered specification; no experiment authorized by merely reading this document._

Companion: `AGENT_PROMPT_JEPA_2026-10-16.md`. Historical v2 is preserved in `reviews/snapshots/v2/`; reviews there and in the parent reviews directory retain their original line references. The repair record is `REVISION_REPORT_JEPA_V3_2026-10-16.md`. This document replaces v2 rather than layering exceptions over it.

## 0. Scientific question and evidence levels

Can predictive representation learning prioritize detector noise shared between views over useful physics, and can noise-only calibration records improve the tradeoff? We distinguish three objects:

1. An idealized **hard-constrained population linear objective**, related to CCA under the conditions in §2.
2. A pure noise-anchored generalized eigenvalue problem (GEVP), with an exact subspace result.
3. Practical finite-penalty and deep JEPA objectives, whose behavior is **empirical**, not guaranteed by either result above.

Shared nuisance can displace weak physics in a restricted embedding. It need not do so for every loading geometry or noise level. Invertible per-view CCA invariance does not imply invariance of every objective called chi-square, finite-sample regularized estimator, or neural network.

Three candidate remedies are tested: selecting views using calibration signal/noise statistics, an independent-noise overlay penalty, and context-first joint noise whitening as an oracle. The oracle additionally requires residual target signal rank; removing noise correlation can remove shared physics too. Appendix A's toy tests CCA and the pure GEVP only.

E1 tests analytic mechanisms, E2 tests nonlinear simulated waveforms, and E3 tests hardware-injected signals in measured TIDMAD noise. Optional E4 asks whether predictive surprise detects faults. This is separate from VERITAS/Paper 3's N-versus-S and harm-ranking claims; it does not change their protocol.

## 1. Scope, registration and execution rules

### 1.1 Files and isolation

- Work only in the JEPA checkout on branch `JEPA`. PMT work is on a separate branch/worktree; do not incorporate it during this campaign.
- New code: `src/jepa_noise/`; documentation: `docs/jepa/`; new results: `results/jepa/<date>_<experiment>/`.
- No edits to `src/latent_monitor/`, `src/noise_module/`, `src/noise_module_lucid/`, `src/reconstruction_model/`, `src/tidmad_transformer/`, `docs/PREREGISTRATION.md`, `docs/EXPERIMENT_DESIGN.md`, `latex/`, or historical results. Import/subclass/wrap locally.
- Outside these directories, allow only the JEPA index entry, its root workspace-member entry and the resulting dependency lock update. Leave `docs/LUCID_MODEL_LINEUP_2026-10-09.md` untracked and unchanged.
- Existing missing interfaces are implemented in JEPA wrappers, never by changing another study's scientific configuration.

### 1.2 Registration and provenance

A revision/review request does not execute the launch prompt. Before a separately launched study:

1. Verify branch/base and review the v3 repair record and any subsequent review. Resolve open P1 findings before registration.
2. Commit the received runbook, prompt, toy, prelaunch reviews, counterexample and revision record as `jepa: register runbook v3`. Known toy and review-counterexample results are disclosed as prior development evidence, not prospective discoveries.
3. Freeze each experiment's derived configuration and data manifest before its first outcome-generating run. Only the deterministic derivation rules in this document may fill derived fields. A scientific choice not specified here triggers G5/G6; it is not permission to tune silently.
4. Subsequent changes to predictions, masks, metrics, data-selection rules or cell definitions require an author-recorded G5 amendment. Reports must retain pre-amendment results.

Only the orchestrator commits. No push unless `PUSH_JEPA=yes`; no L40S use unless `GPU_ALLOWED=yes`. Do not merge into `dev`/`main`. Deepthought receives committed code via GitHub; absence of an accessible required SHA triggers G0. Use fixed-base diffs against `0cae6b0`, not a moving `dev`.

Executors place compact configs/manifests, hash inventories, logs and metric tables under `/ceph/dwong/jepa/returns/<run_id>/`; the orchestrator copies these via rsync to the matching `results/jepa/<run_id>/` and verifies hashes. Never overwrite a prior attempt. For L40S jobs, record the actual host-qualified artifact/return paths and use the registered Condor transfer mechanism; do not assume GPU artifacts reside on Deepthought. Large raw arrays/checkpoints remain at recorded absolute host paths with SHA256 hashes; reviewers receive read access to those exact inputs and must recompute required statistics before acceptance. A summary without accessible raw evidence is insufficient.

Every run stores code SHA, tracked/untracked status, config/data/split hashes, complete environment inventory, RNG seeds and states, host, start/end times, elapsed/cumulative resources and exit status. Do not start with unrecorded dirty scientific inputs. Store large arrays/checkpoints on the run host; no committed file above 50 MB. New outputs remain `development output, not reviewed` until author review; review alone does not confer confirmatory status.

### 1.3 Statistical and tuning contract

- E1 seeds 0–4; E2/E3 seeds 0–2, optionally E2 seeds 0–4 only if the budget decision is recorded before the grid. The same seed population is used for all paired arms. Development model seeds 100–102 never enter reported evaluation aggregates.
- Freeze encoders at the final scheduled step, not the best test/probe checkpoint. Log divergence/collapse as failures; do not replace an unfavorable seed.
- Fit ridge probes on frozen **unmasked** embeddings. Per-output alpha grid `{1e-6,1e-4,1e-2,1,100}`; five grouped folds on probe-train only; choose lowest validation MSE, ties favor larger alpha. Standardization fits within each fold; final scaler fits probe-train only. For E3, grouping is source acquisition/file interval, not shuffled overlapping windows.
- Report individual physics errors and nuisance R² values. Primary nuisance R² is the unweighted mean over declared nonconstant nuisance coefficients; constant targets are undefined, not zero. Compare only the declared common valid target set; a missing required target makes the associated prediction not evaluable.
- `physics_MSE_CRLB[j] = mean(error_j^2) / mean(CRLB_j)` over the same test events. This is an efficiency diagnostic; biased/nonlinear estimators need not attain a CRLB. Invalid Fisher/target cases are counted, not silently dropped.
- Development choice of beta and hybrid lambda: minimize the mean over physics parameters of validation MSE divided by their development probe-training variance. Use development probe-train/validation only; ties within 1e-8 relative favor the smaller coefficient. No optimization using nuisance or final test scores. Grid beta `{0.1,1,10}`, hybrid lambda `{0.1,1}`. Apply the chosen beta to both anchored and anchor-only arms.
- Report paired seed effects, mean, SD, all values and a 95% t interval across paired seed effects when at least three exist. For ratios use log ratios where positive, reporting original values too. These small-seed intervals and win-count criteria are **descriptive pilot evidence**, not powered confirmatory tests. No headline significance claim based only on 2/3 or 4/5 wins. R6 computes Spearman on the three seed-mean noise levels, separately per encoder arm; report seedwise correlations as supporting.
- All outcome tables support `pass`, `fail`, `not_evaluable`, `not_run`; attach reason codes. A failed empirical prediction is a scientific finding, not a reason to alter code/hyperparameters until it passes.

### 1.4 Delegation and gates

Use one implementer per WP, bounded executors for remote runs, and a fresh-context independent reviewer at WP2, WP5 and final closeout. No concurrent writers to the same files; only the orchestrator commits. Reviewers see source and raw results, not implementer conversation. Continue independent work when a gate blocks only a dependent step.

| Gate | Trigger and required response |
|---|---|
| G0 | Required SHA unavailable on remote and push not allowed; record and wait for publication/authorization. |
| G1 | Analytic identity, negative-control handling, input-leakage test or Appendix A toy fails; stop dependent experiments. A correctly implemented failed empirical prediction does not trigger G1. |
| G2 | Any campaign projection or actual cumulative resource use reaches its cap, including after a fallback; stop launch/remaining jobs within this campaign and preserve output. |
| G3 | Download exceeds 50 GB or needs credentials; no bulk acquisition until resolved. |
| G4 | No eligible E3 band has calibration noise MSC ≥0.5 under any declared split; report E3 feasibility negative, do not train it or search new splits. |
| G5 | Any change to a registered scientific specification or unlisted fallback; record amendment before dependent work. |
| G6 | Missing/invalid data contract, calibration stationarity, signal-rank eligibility, metric target or manifest; record precisely, do not fabricate defaults. |

Write `docs/jepa/GATE_<id>.md` with evidence, affected scope and proposed bounded resolution. Preserve STOP output. No automatic recollection or reselection after a scientific failure.

## 2. Mathematical contract

### 2.1 Linear model

Centered views satisfy `x_c=A_c theta+B_c eta+e_c`, `x_t=A_t theta+B_t eta+e_t`. Physics, nuisance and private residuals are mutually uncorrelated, and the two private residuals are uncorrelated. State Gaussianity where independence/Gaussian formulae are used. `Sigma_theta` and noise view covariances are positive definite and `A_c` has full column rank k. Record any violated assumption as an ineligible theorem test.

`R_c = span(N_cc^-1 A_c)`. `gamma_load` scales `B_t` only in the main exact model; nuisance remains in context. `gamma_n` is the population top noise canonical correlation computed from exact generator covariance. Do not label loading amplitude as a measured canonical correlation.

Idealized CCA analogue: minimize prediction MSE over separate linear context/target maps and predictor under hard target isotropy, with k-dimensional full-rank optimal context map and nonzero selected canonical correlations. Its global optimal context row space is the canonical subspace. This is not a claim about every GD fixed point or soft SIGReg. The stop-gradient tied-encoder model is a separate empirical model.

### 2.2 Propositions and mandatory counterexamples

- **P1:** cross-covariance is `A_t Sigma_theta A_c.T + B_t Sigma_eta B_c.T`. Canonical modes can mix physical and nuisance factors; a strong shared nuisance **can** displace physics. Factor-by-factor rankings require a separable-mode construction.
- **P2:** population/unregularized CCA canonical correlations are invariant to invertible per-view linear transforms; map directions back before comparing. This does not prove invariance of arbitrary weighted losses, ridge estimators or deep objectives. For scalar bands with equal per-view signal/noise powers, coherency is `(c_s S+c_n J)/(S+J)`; test that assumption before using this special formula.
- **P3:** with zero cross-view noise, full-rank target signal and the conditions above, the global optimum recovers `R_c`. Prove by the range of `Sigma_cc^-1 Sigma_ct` and Woodbury. Check smallest signal singular value and spectral gap, not only nominal rank.
- **P4:** minimizing `tr(W N_cc W.T)` under `W Sigma_x W.T=I` yields the smallest generalized eigenvectors of `(N_cc,Sigma_x)` and recovers `R_c`. Noise-only calibration covariance must match event noise covariance. Independent zero-mean overlay noise and a linear encoder yield this trace penalty. **Prediction plus finite beta is a different objective**, and normalized deep overlay plus soft SIGReg is another approximation. Test them empirically; do not promise P4 recovery.
- **P5, conditional oracle:** first discard all gap/unused samples. Form the marginal joint noise covariance of retained `(context,target)` coordinates, require `N_joint` positive definite (marginal positive definiteness alone is insufficient), and whiten with inverse lower Cholesky. Reject a singular joint covariance before factorization; do not add undeclared jitter. Context depends only on raw context. The target signal loading before its final invertible block scaling becomes `A_t - N_tc N_cc^-1 A_c`. Recovery of `R_c` additionally requires this residual loading to retain rank k. Require smallest/largest singular-value ratio >1e-8 for both context and residual target, plus kth population canonical correlation >1e-8 and a retained/discarded canonical-correlation gap >1e-8. Record both margins; these numerical eligibility thresholds do not guarantee finite-sample angular accuracy. Verify eligible cells using population matrices. If rank fails, report `oracle_rank_ineligible`; retain output as a negative control, never claim recovery. Post-whitening correlations need not equal no-sharing correlations.

Test `reviews/check_p5_counterexample.py`: a valid SPD example has zero residual target physics. This must be correctly **rejected as recovery-ineligible**, not made to produce the reference subspace. Also test a finite-beta example where nuisance remains preferred; a pure GEVP theorem does not override that result.

### 2.3 Criterion M, operational calibration heuristic

M is an empirical mask-selection heuristic, not a theorem. It requires both calibration event windows and independent calibration noise windows. No probe-test inputs/labels are used.

For each declared candidate split, flatten its real coordinates and apply a fixed seed-2718 Gaussian random projection to `d=min(8,d_c,d_t)` coordinates per view, with QR-orthonormal columns. Seed/arrays are stored. Fit centering on calibration subsets only. Use at least `max(200,20*d)` nonoverlapping records of each type. E1/E2 use 4096 per type. E3 records come from its manifest.

Let `S_cc,S_tt,S_ct` be event covariance blocks and `N_cc,N_tt,N_ct` noise blocks in projected coordinates. With `eps(S)=1e-3*trace(S)/d`, define:

```text
H_signal = (S_cc+eps(S_cc)I)^(-1/2) (S_ct-N_ct) (S_tt+eps(S_tt)I)^(-1/2)
H_noise  = (N_cc+eps(N_cc)I)^(-1/2) N_ct (N_tt+eps(N_tt)I)^(-1/2)
M = kth singular value(H_signal) - top singular value(H_noise)
```

Use k=3 for E1/E2, k=1 for E3. The signal term is an event-normalized cross-signal strength, **not signal-only CCA**; finite estimates may exceed one and must be reported rather than clipped. Degenerate covariance/insufficient data is G6. Choose maximum M; ties within 1e-8 use candidate order as listed. Freeze scores, projections and selected split before model training. Primary mask is selected once at the independent development/calibration setting and held fixed across an experiment's reported cells. Record all candidates, not only the winner.

### 2.4 Prospective pilot predictions

These replace v2's table before registration; no new outcome has been used to choose them. Exact theorem tests are separate from finite-sample pilot success criteria.

| ID | Prediction / population | Descriptive criterion |
|---|---|---|
| R1 | Exact-model CCA at zero cross-noise, E1a n=1e5 | Median max angle ≤5 degrees across seeds, per declared signal/noise-scale cell. E1b is diagnostic, not a P3 gate. |
| R2 | Shared-noise displacement, E1a | Per scale cell, Spearman over seed-mean angles at seven gamma levels ≥0.8; report all cells. At theta scale .4, eta scale 3, gamma_load=1, median angle ≥45 degrees. A nonmonotone curve is a possible finding. |
| R3 | Unregularized per-view invariance | Maximum sample canonical-correlation difference ≤1e-8 on E1a with n≫d. |
| R3b | Conditional joint whitening | Eligible E1a cells: population recovery within 1e-6 rad; sample median angle ≤5 degrees at n=1e5. Annihilation control must report rank ineligibility. No fixed bound equating transformed correlations to original physics correlations. |
| R4a | Pure noise GEVP recovery | Population ≤1e-6 rad; sample median ≤5 degrees at n=1e5 on E1a per cell. |
| R4b | Finite-beta empirical recovery | Combined linear objective at beta=10: median angle ≤5 degrees per E1a cell. Failure is a result, not G1. Also report beta=.1,1 and E1b diagnostics. |
| R5 | Depth and learning order | E1c: at gamma_load=.5 and 1, delta(L=3)<delta(L=1) in ≥4/5 seeds; delta=t_eta-min_i(t_theta_i). Nonlearned/censored modes make the corresponding seed not evaluable, not a win. |
| R6 | Shared noise in E2 embeddings | Separately for EMA and SIGReg JEPA: Spearman over the three nominal-SNR seed-mean nuisance R² values ≥.8. All three evaluation cells are independent of development. |
| R7 | Remedies preserve physics while reducing nuisance | E2 high/nominal and high/sub-noise: anchored and masked each lower nuisance R² in every paired seed; per-parameter seed-mean MSE/CRLB ≤1.1 times plain SIGReg. |
| R8 | Robustness to changed line amplitude | E2 high/nominal: multiply line waveform amplitude by 2 at test time (power x4); average per-parameter shifted/clean MSE ratio is lower for anchored than plain in ≥2/3 paired seeds. |
| R9 | Real-noise lines versus useful signal, E3 | Plain has a larger fraction of top-10 sensitivity bands labelled as noise lines than anchored in ≥2/3 seeds; anchored **frequency** probe seed-mean R² ≥.5 and ≥.9 times plain frequency R². Target is log10 injection frequency. Amplitude R² is supporting only when its target variance is adequate. |
| R10 optional | Surprise depends on type of fault | E4 macro-average full AUROC ≥.8 for breaking faults and ≤.65 for preserving faults; also report each fault and TPR at FPR .05. |

R9 uses frequency because selected hardware injections may have essentially fixed amplitude. No observed amplitude fluctuation is promoted to an independently varied physical signal. This changes v2's primary physics-retention endpoint explicitly; all reports must identify v3. For inference beyond pilot description, a separate power/precision and multiplicity plan is required.

## 3. Environments, data integrity and resource accounting

### 3.1 Hosts

Mac: CPU only, explicit interpreter from README's separate CPU environment; never resolve the CUDA workspace pin with bare `uv sync`. Deepthought: dedicated `/ceph/dwong/jepa/{code,venv,logs,raw,returns}`; one clone per SHA; isolated `tmux -L jepa` sessions; do not touch other studies' processes/sockets.

Use Python 3.11 and torch 2.5.1 CPU on CPU hosts, torch 2.5.1+cu124 on authorized L40S; record any unavoidable platform variation and run equivalence checks. Include numpy, scipy, pandas, matplotlib, h5py, pytest, scikit-learn and **jaxtyping**. Freeze resolved versions before experiments. Test imports of `reconstruction_model.model`, `tidmad_transformer.model` and `jepa_noise` via `PYTHONPATH=src`. Disable implicit compilation for CPU checks where the repository supports it; record the setting.

Deepthought preflight: hostname, nproc, uptime, free memory/disk, torch version/thread count, Git remote SHA and import checks. Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Worker count is the integer min of 32 and floor(nproc minus 15-minute load minus 4), capped further by memory and other active authorized campaigns; if below one, wait without launching. Capture logs and completion status from JEPA sessions only. Read the actual log files, since redirected output need not appear in tmux capture.

L40S uses the repository Condor submission pattern and the author's allocation only with `GPU_ALLOWED=yes`. Pin datasets/configurations/software before comparing CPU/GPU outcomes; "only device differs" must be verified, not assumed.

### 3.2 Campaign budgets apply before smoke and tuning

| Campaign | CPU cap | Authorized L40S cap |
|---|---|---|
| E1 grids including linear GD | 24 wall-hours at recorded workers | Not used |
| E2 including WP4, tuning, all cells/depths/evaluation | 72 wall-hours at recorded workers | 48 GPU-hours |
| E3 including coherence, tuning/probes and training | 48 wall-hours at recorded workers | 24 GPU-hours |
| Optional E4 | 4 CPU wall-hours | 2 GPU-hours |

Caps are independent and cannot borrow from each other. Use aggregate worker-hours divided by the fixed allocated worker count, plus serial preparation/evaluation time, as the CPU budget accounting measure; also cap actual campaign wall-clock active execution. Sum GPU-hours across all devices/jobs. Maintain one coordinator-owned ledger including failed attempts and preprocessing. Explicitly allocate RAM, workers and devices if PMT or another study runs concurrently.

Before substantial smoke/tuning, benchmark up to 300 steps for E2 arms 1/4/6, an oracle batch/preprocessing pass and deepest encoder; E3 up to 200 steps; E1 representative solvers. Bound each benchmark at 10 CPU minutes or .25 GPU-hours, and the entire E2/E3 benchmark phase at one CPU hour or one GPU-hour, charged to the campaign. Timeout means incomplete benchmark, not permission to run longer silently.

Project **all** jobs: calibration, smoke, coefficient searches, primary grid, depths, probes, figures, oracle cost and a 25% reserve. E2 fallback order before grid: five to three seeds, then 10000 to 5000 steps for every arm including development searches. Reproject. E1 has no automatic grid reduction. E3's one input-size fallback is defined in §7; reproject it too. If still over budget, G2. No arm/cell removal or further fallback without G5.

Every job receives a conservative remaining-budget allocation and checkpoints safely on termination; the coordinator stops launching when the ledger is exhausted. Benchmark/smoke success does not authorize remote/GPU execution absent the launch flags.

## 4. Package and shared numerical contracts

Implement only `src/jepa_noise/`: `views.py`, `linear.py`, `masks.py`, `metrics.py`, `crlb.py`, `backbone.py`, `sigreg.py`, `jepa.py`, `tidmad.py`, `train.py`, `run.py`, `report.py`, `configs/`, `tests/` and package metadata.

- `ExactModel(..., share="target"|"both")` returns samples, separate noise records, exact covariances, R_c and population gamma_n. `share="both"` reproduces Appendix A's random-draw order only.
- `Tier1Views(..., signal_mode="linearized"|"nonlinear", split=...)` returns raw data, explicit raw-sample index arrays, signal/noise latent targets and covariance of the actual generating law.
- `linear.fit` returns context functionals. Principal angles are measured between row spaces, mapped back to original context coordinates; compare against the existing principal-angle utility.
- CCA: unregularized for exact identities/R3; otherwise ridge `1e-6*trace(S)/d` on sample marginals, reported. Numerical conditioning failures are not hidden by tuning ridge.

### 4.1 Positive-definite JEPA waveform noise

The imported Tier-1 variance convention has zero DC power. Do not use its whitener's unit scaling for DC as if it were physical variance. In the **JEPA wrapper only**, add independent Gaussian sample noise with variance

`epsilon_dc = .01 * mean(diag(N_reference_raw))`

per channel/sample; fix this value once from the unperturbed base generator, across every E1b/E2 cell and SNR. Include the same addition in all event/noise records and all exact covariance/CRLB calculations: `N_actual=N_raw+epsilon_dc*I`. This is a declared 1% white-variance modification of the experiment, not an inversion jitter. Preserve the imported package unchanged. Test actual generated variance and positive definiteness; this component belongs to private noise, not probed nuisance eta.

Discard gap samples before constructing the oracle covariance. Embed transformed visible samples back into their original positions with zero unused samples, and apply the matching raw-input dependency tests. Never whiten the entire record symmetrically then mask it.

### 4.2 CRLB, probes and learning curves

Use the exact modified `N_actual`. For nonlinear waveforms compute eventwise Jacobian by central differences with h=1e-4 times each parameter's declared range; test h/2 stability on fixed reference fixtures. Evaluate `J.T solve(N_actual,J)` and its inverse only when full rank. OF amplitude variance cross-check is `1/I_aa` when other parameters are known, within 1%; not the joint-parameter inverse diagonal. No pseudoinverse disguised as an exact inverse.

For E1c: full-batch GD, square tied depth-L encoders and square predictor on d=40, L=1/2/3, Gaussian weight SD=1e-3, learning rate 1e-3, 100000-step horizon, no early stopping; if numerical instability occurs record failure. Test gradients/stop-gradient semantics independently. First reproduce a diagonal-covariance fixture compatible with the cited theory; random-loading detector generalization is explicitly empirical. Mode contribution is `norm(W A_i)*sd(theta_i)` or `norm(W B)*sd(eta)` using the true separate loading, not projection of mixed observations. Use absolute contribution q(t) for the primary learning-time metric; q(t)/norm(W) is diagnostic only. Freeze q0=q(0). Let qf be the last-1000-step mean. Eligibility requires finite q throughout, qf>1e-6, qf>=2*q0, and absolute difference between the means of steps 98001–99000 and 99001–100000 <=.05*qf. A shrinking, flat or nonplateaued mode is not eligible. Define the first crossing of q0+.5*(qf-q0), sustained for 100 steps, as learning time. Test zero, constant and globally shrinking curves as ineligible, plus a rising plateau as eligible. A never-reached crossing or unlearned mode is censored; report it, never treat it as immediate learning. R5's minimum over physics modes uses only valid crossings and requires all mode-learning eligibility checks first.

## 5. WP0–WP3: preflight, theory and linear evidence

### WP0 — Register and preflight, only when launched

Read review dispositions; verify no unresolved P1. Verify branch/base and permissions. Make the v3 registration commit, then reproduce Appendix A and the P5 negative control. Create package skeleton and its workspace/index entries. Complete environment/import/resource preflight on each authorized run host. If remote code is unavailable, G0; local checks may continue. Acceptance is a complete `PREFLIGHT.md`, registered source hashes and passing imports on hosts actually used, not a fabricated remote pass.

### WP1 — Theory note

Write `THEORY.md` with P1–P5 assumptions/proofs, M's definition and limitations, finite-beta distinction, covariance support and mapping conventions. Required analytic tests use population matrices, tolerance 1e-8 for matrix identities and 1e-6 rad for subspaces. Include eligible P5 and annihilation counterexample, finite-beta negative case, invertible-map invariance and mutually uncorrelated-factor assumptions. Document which literature results concern hard constraints, soft SIGReg or tied stop-gradient models.

### WP2 — Linear package

Implement CCA, a hard-isotropy linear prediction solver, pure noise GEVP, finite-beta prediction solver with **both** context and target hard isotropy, and oracle mapping. Penalize context `tr(W_c N_cc W_c.T)`; beta grid .1/1/10. The hard-isotropy solver is an idealized linear control, not the literal deep SIGReg implementation. Define its optimizer/convergence settings in a versioned solver config before sample-grid outcomes; use a population objective/gradient check and analytic solution comparison. If no fixed solver config has been reviewed, G6 before grids.

Name `R_c` the **noise-weighted reference**, not an implemented MAE training algorithm. Linear `mae_mse` control uses top-k context PCA. A whitened-reconstruction control uses top-k PCA of the context after exact noise whitening, mapping directions back; state this is context autoencoding, distinct from deep masked reconstruction. OF is a physics-estimation baseline. Tied stop-gradient experiments follow §4.2, not an assumed CCA convergence test.

Acceptance: theorem checks, negative controls, finite-difference gradients, independent-target nuisance R² sanity tests, exact masks and deterministic pairing, statistical covariance check within 2%, CRLB test and independent WP2 review. Recovery failure in a finite-beta or stop-gradient experimental arm does not fail a theorem unit test. Log all convergence outcomes.

### WP3 — E1 experiments

**E1a exact model:** d=40 per view, k=3, r=1; theta scale {.1,.4,1}, eta scale {1,3}, gamma_load {0,.1,.25,.5,.75,.9,1}, n {1e3,1e4,1e5}, seeds 0–4. Methods: CCA, hard-isotropy prediction, PCA reconstruction controls, pure GEVP, finite-beta solver, conditional oracle, OF. Tied dynamics are isolated in E1c. Angles use n=1e5 for R1/R2/R3b/R4; smaller n are convergence diagnostics. Add a deliberately annihilating P5 fixture outside the primary grid, labelled negative control. Do not discard rank-ineligible random cells; retain their records and separate theorem eligibility from pilot outcomes.

**E1b linearized waveform fixture:** take the eight-channel, 256-sample template at `theta0=(4.5,.35,.08)` in its native units. Set fixed `J0` to central differences there and generate `s=J0 delta_theta`, with independent centered Gaussian perturbations of SD `(.15,.005,.002)`. Thus `A=J0` is fixed and R_c is well-defined for each split. This is a deliberately linearized fixture; do not regenerate nonlinear waveforms from perturbed parameters and call them linear.

Use the JEPA noise law §4.1 and exact raw-sample splits:

| Candidate order | Context | Target | Unused gap/other |
|---|---|---|---|
| time | all channels, [0,96) | all channels, [160,256) | [96,160) |
| channel | channels 0–3, all samples | channels 4–7, all samples | none |
| interleaved | channels 0,2,4,6 | channels 1,3,5,7 | none |
| block | channels 0–3, [0,96) | channels 4–7, [160,256) | all other samples |

Separately sweep added narrow-line variance `{0,.75,1.5,2.25,3}` times the reference white-component variance and common-mode strength `{0,.225,.45,.675,.9}`. Line frequency is the non-DC rFFT bin maximizing the reference noiseless template's spectral power (ties lower frequency), fixed before any probe outcome. Generate declared shared components separately and add to the base; exact total covariance is their sum, including the independent white addition. Record the line model's bandwidth/PSD normalization and all actual covariances in the config. Existing colored base noise may remain shared at zero added line; **zero knob is not zero cross-noise**. E1b angle results are diagnostic extensions; no mandatory P3 test at a falsely independent baseline.

Before sample grids, evaluate each fixed split using population Jacobians: record both view ranks, residual target rank, whitened kth canonical correlation and eigengap. A time split containing only a late exponential tail may not identify amplitude and onset separately. Mark such splits theorem-ineligible and retain their diagnostic records; do not search replacement splits to restore rank. The full-rank channel split is also subject to this check. If no declared split supports a requested theorem check, G6 for that check only.

Line nuisance targets are its independent sine/cosine coefficients. Common-mode targets are coefficients of its two declared leading temporal basis vectors; deterministic eigenvector sign convention and eigengap report required. At zero variance that nuisance target is undefined. Use n=1e5 and five seeds. Report signal rank/conditioning for every view; absent target information is a finding, not a new split search.

**E1c learning order:** exact-model theta scale .4, eta scale 3, gamma_load {.5,1}, five seeds and §4.2 depth protocol. Publish each mode's curve, censoring and R5. A finite-budget inconclusive result remains inconclusive.

Outputs: method/cell/seed tables with status, canonical spectra, angles, nuisance probes, physical error, convergence and conditioning. Figures F1 angles, F2 spectra, F3 sample convergence, F4 learning curves. `report E1` must regenerate `PREDICTIONS.md` byte-identically. Accept complete honest reporting, not compulsory passing of R1–R5.

## 6. WP4–WP5: nonlinear Transformer experiment E2

### WP4 — Architecture, objectives and tests

Wrap `reconstruction_model.model.Transformer`, retaining its factorized backbone. Use C=8, N=256, patch length=stride=32, d_model=128, n_head=4, depths (1,1), batch=128. Return `(B,N_patch,C,d)` before pooling/heads. Fixed per-channel mean/std are fit on disjoint calibration-noise records, stored and held fixed under shifted robustness tests. No per-event normalization.

Use the exact sample masks from §5: time context patches 0–2, gap patches 3–4, targets 5–7. The context encoder replaces **all noncontext tokens** after their independent patch embedding; none may attend through an unmasked gap. Target encoder sees target-only tokens, with all other positions masked, for the clean two-view comparison. Predictor has d_pred=64, two full-attention blocks and target positional queries; it reads context outputs plus queries only. The gradient and finite-perturbation test must show zero dependence of context outputs on every raw target **and gap** sample, and target outputs on raw context/gap samples before oracle preprocessing. Oracle target innovations legitimately depend on context; context must still be invariant to raw target/gap.

The default split for E2 arms other than arm 5 is the fixed time split; only arm 5 uses M selection among §5 candidates. All arms use the same data, scheduled steps and backbone. Explicitly construct optimizer groups locally; every trainable non-EMA parameter, including mask tokens, positional queries, predictor and heads, occurs exactly once. EMA target has no optimizer/gradient updates. Retain repository Muon/AdamW conventions and freeze numeric optimizer parameters in the reviewed config before outcomes.

| Arm | Objective |
|---|---|
| 1 `mae_mse` | Context-masked reconstruction via linear target-token head; squared error in measurement coordinates. |
| 2 `mae_chi2` | Same reconstruction with exact target marginal `N_actual` weighting; oracle-noise information explicitly labelled. |
| 3 `ijepa_ema` | EMA target, tau .996→1 cosine; layer-normalized target-token embeddings; prediction L2. |
| 4 `sigreg_jepa` | Shared weights, no stop-gradient/EMA, prediction L2 plus SIGReg. This is an adaptation, not a literal reproduction claim. |
| 5 `sigreg_jepa_masked` | Arm 4 with one split selected by M on independent development calibration data, fixed across all reported cells. |
| 6 `sigreg_jepa_anchored` | Arm 4 plus beta times independent-noise overlay squared embedding change divided by trace of embedding covariance. |
| 7 `hybrid_chi2_jepa` | Arm 2 plus lambda_J times the complete arm-4 objective. |
| 8 `anchor_only` | SIGReg plus arm-6 penalty, no prediction; test whether prediction adds value. |
| 9 `sigreg_jepa_jointwhite` | Arm 4 after two-view context-first Cholesky using actual covariance; gap discarded. Label oracle; P5 is not a nonlinear recovery theorem. |

Reconstruction heads in arms 1/2 and the reconstruction branch of arm 7 read target-position outputs of the context encoder, with raw target/gap tokens replaced before attention. They never read target-only encoder outputs. Test zero gradient and finite-perturbation invariance of the final reconstruction prediction to raw target/gap samples; the loss may depend on its target labels.

SIGReg applies to context pooled embeddings and target token embeddings, averaging the two regularizer values. Source and pin its implementation, numerical default coefficient, projection count and loss reduction from the primary reference before WP4 smoke. The coefficient is not chosen by claim performance. The overlay uses unmasked pooled embeddings, independent calibration-noise draws, and denominator `max(trace(Cov(z)),1e-6)` with gradients through the trace. Apply identically in arms 6/8. These finite-penalty objectives have no exact P4 guarantee.

Tests: shapes, exact raw-input visibility, optimizer coverage, EMA update, SIGReg Gaussian/collapse calibration, ten-step CPU determinism, positive covariance, no-overlap IDs and source tags. After budget approval run 500-step development smoke for all nine arms; each JEPA arm should have pooled std>.1 and token effective rank>4. Collapse is reported as a trainability failure; do not change objectives secretly. WP4 acceptance is tested implementation plus documented smoke outcome; G6 if trainability is inadequate for meaningful comparison.

### WP5 — Cohorts, evaluation and resources

Use **nonlinear** EventFamily waveforms plus §4.1 noise. Three line levels target population gamma_n nearest {.1,.5,.9} using the declared E1b line grid, ties lower variance. Record actual levels; if distinct levels cannot be selected, G6 rather than changing the grid. Two SNRs: baseline amplitude range and range multiplied by .25. All six evaluation cells remain present.

The development distribution is mid/nominal but its data and model seeds are independent of the six evaluation cohorts. It is not the evaluation mid/nominal cohort. Use integer event IDs as follows, with independent seed namespaces per role/cell/model seed:

| Role | IDs (inclusive) |
|---|---|
| Evaluation-cohort subject training | 0–199999 |
| Evaluation-cohort probe training | 200000–209999 |
| Evaluation-cohort probe test | 210000–219999 |
| Development subject training | 1000000–1199999 |
| Development probe training / validation | 1200000–1209999 / 1210000–1219999 |
| M development event calibration | 1300000–1304095 |
| Calibration noise / overlay noise | namespaces `noise_cal` / `noise_overlay`, disjoint from all event/noise-test streams |

Use seed-keyed namespaces, not integer offset assumptions alone, and test no shared random state. Physics twins are paired across methods/shifted tests within a cohort. M gets 4096 development event and 4096 development noise records. Reference normalization for each evaluation cell uses 4096 independent noise-calibration records from that cell; robustness shifts never refit it.

Primary 10000 steps, with budgeted fallback to 5000 before outcomes (§3.2). Depth sweep (2,2)/(4,4) for arms 3/4/6 on high/nominal, same scheduled steps and seeds. Beta/lambda chosen once on the independent development cohort via §1.3. A second independent mid/nominal evaluation model is trained normally; no tuned checkpoint enters R6.

Evaluate frozen unmasked encoders; exact preprocessor is part of each arm (oracle transformations use the fixed time split). Physics probes target amplitude/t0/tau_t, nuisance probes line I/Q; common-mode coefficients are supporting where nonconstant. E2 local tangent angles, if reported, are eventwise diagnostics against J(theta), not one exact global R_c. R8 uses identical physics/noise base draws with line amplitude x2; freeze all normalizers, encoders and probes.

Outputs F5 nuisance versus gamma_n, F6 physics/CRLB, F7 robustness, F8 depth; tables per target/seed and R6–R8. Fresh reviewer recomputes raw contrasts and checks the separate middle-level cohorts. Scientific acceptance means traceable outcomes including failures.

## 7. WP6: real TIDMAD noise E3

### 7.1 Exact channel roles and deterministic manifest

The published release has `timeseries/channel0001/timeseries` = SQUID input and `timeseries/channel0002/timeseries` = injected reference, as also declared by `tidmad_transformer.data`. There is no assumed second SQUID. **CH2 is never an encoder, target-encoder, reconstruction-loss or noise-overlay input.** It is delayed-label information for probe targets only. No ch1→ch2 view split. Verify actual HDF5 paths/dtypes/rates against release metadata; mismatch is G6.

Use existing data first. Otherwise acquire at most four training, four validation and sixteen science files within the 50 GB gate; exact sizes are checked before download. Selection is by published file index, not observed model scores. Before calculating feature/coherence outcomes, list all candidate files, hashes, acquisition IDs, injection-frequency metadata and sizes in `E3_MANIFEST.json`.

A file is eligible only if its published injection metadata are wholly inside the retained frequency interval and contain at least five distinct frequencies. Select the first four eligible train and first four eligible val files, ascending official index; check the additional target-spread conditions separately before training. Do not substitute based on coherence/model results. If metadata cannot establish eligibility, inspect CH2 labels only, record that screening and selection before feature analysis, or G6 if ambiguity remains. Choose the first sixteen science files in official order. The paper/metadata define acquisition grouping; adjacent files from a continuous run are not claimed independent acquisitions.

Assign selected train files 1–2 to SSL training, 3 to calibration/mask selection, 4 to probe training; selected val files 1–3 to probe training, 4 to final test. The four probe-training files allow five deterministic **blocked** folds: split their ordered usable intervals into five approximately equal duration blocks, with no overlapping raw samples and a one-second boundary guard. State that these are blocked within-acquisition validation, not five independent experiments. Scientific uncertainty across acquisition periods is limited by the actual manifest.

Science files 1–4 are calibration, 5–12 overlays/training noise, 13–16 final noise diagnostics. No science test window is used to label line bands. Within every role use nonoverlapping windows with stride at least twice window length; trim one second at role boundaries. Exclude injection transitions using CH2 label stationarity: fit a sinusoid separately to first/second halves; frequency agreement within one FFT bin and relative amplitude difference ≤.1. Record rejected fractions and reasons. All masks/normalizers/line labels use calibration only; locked manifest precedes SSL training. If any role lacks 200 valid windows or adequate target frequency spread, G6, no post-hoc file expansion.

### 7.2 Coordinates, masks and calibration feasibility

Use the existing complex STFT: n_fft=4096, hop=2048, Hann, no centering. Cast raw int8 to float and use signed physical ADC units (subtract the reader's +128 offset), documenting the conversion. Retain bins 1–327, excluding DC; primary window 198656 samples gives 96 frames. Feed paired real/imaginary features per frequency token to a local subclass of the TIDMAD backbone. Do not create an imaginary sensor axis. Fixed means/std from calibration science windows; no event-wise scaling.

With four-frame patches define candidate order:

| Name | Context frames | Gap frames | Target frames |
|---|---|---|---|
| time_short | [0,44) | [44,52) | [52,96) |
| time_long | [0,40) | [40,56) | [56,96) |

Both views use all retained bands. Verify raw-STFT supports do not overlap, including Hann support; no partial patch crosses a role boundary. M uses calibration hardware-injected **SQUID** windows and calibration science noise windows only; CH2 is excluded from M. The four non-masked E3 arms use fixed time_short. Only sigreg_jepa_masked uses the M-selected candidate, frozen before SSL training. If M chooses time_short, record that the masked/plain contrast has identical masks; do not force a different winner. Minimum calibration samples is max(200,20*d) under §2.3's projected estimator.

For each retained band compute Welch-style averaged cross-periodograms over paired equal-length views, including each frame lag's pairing definition in config; no single-sample coherence estimator. Use matching ordinal frames in context/target, aggregate over windows, and record the time lag. Estimate noise MSC and J on science calibration. Define lines as J>5 times the running median of the neighboring ±50 retained bins, excluding the bin itself. Mark line labels once. Equal view PSD assumption is checked by target/context ratios: .8–1.25 on eligible bands; otherwise report general coherency `S_ct/sqrt(S_cc S_tt)` rather than the equal-power formula. Do not force agreement.

Report injection-frequency-conditioned event coherence and noise coherence separately. No arithmetic signal/noise subtraction across nonstationary acquisitions is called exact ground truth. Compare calibration versus final science PSDs after all choices are fixed; if median band ratio lies outside [.5,2], report unmatched-noise limitation and G6 before interpreting P4-like recovery. G4 if no calibration line/eligible band has noise MSC≥.5 for either candidate.

### 7.3 Training and targets

Five arms: mae_chi2, ijepa_ema, sigreg_jepa, sigreg_jepa_masked, sigreg_jepa_anchored. Retain E2 objective definitions, fixed per-band standardization, d_model128, four-frame patches, one temporal/one frequency layer, 4 heads, batch32, 5000 steps, seeds0–2. Optimizer numeric settings and SIGReg are inherited from the reviewed E2 config; anchored beta is transferred from E2 with no E3 retuning. The mae_chi2 arm uses a diagonal per-bin variance estimate from E3 calibration noise, labelled an approximation, not exact likelihood. Clip variance only at fixed 1e-6 times median retained-bin variance and record clipped bins.

Use only SQUID windows for SSL and science noise for overlays. Primary probe target is log10 injection frequency derived by least-squares sinusoid fitting to CH2 after constant-offset removal, within the retained band. Record estimator resolution, transition exclusions and residuals. No ground-truth frequency is fed into any encoder or mask selection. Require ≥5 distinct stable frequencies and log-frequency SD≥.1 in probe-train and probe-test before R9 is eligible. If not, G6 rather than scoring a constant target.

Supporting amplitude probe uses the same fitted sinusoid amplitude only if its CV≥.05 and all values are finite in both probe partitions. Otherwise record `amplitude_not_identifiable_from_selected_range`; do not interpret digitization fluctuations as amplitude variation. Standard/weak acquisition variants are not mixed without a registered acquisition-block design.

### 7.4 Budget fallback and metrics

Budget includes coherence, manifests, smoke, training, probes and sensitivity. One fallback: 24 frames (51200 samples), same retained frequency bins. This avoids changing frequency coverage. Use time_short `[0,8) -> [16,24)` and time_long `[0,4) -> [20,24)`, with their intermediate frames as gaps. Rebuild deterministic nonoverlapping window intervals on the **same files**, rerun calibration-only selection and reproject before training. Preserve the initial feasibility report. If cost remains above cap, G2; no additional narrowing or file changes. Decision is based only on cost, not R9 outcomes.

Primary sensitivity is mean squared Frobenius Jacobian energy per physical frequency band, summing paired real/imaginary components and frames, then normalized across retained bands. Differentiate with respect to calibration-standardized inputs for all arms, evaluated on held-out final-test windows. Fix top-10 tie break to lower frequency. If total sensitivity is zero/nonfinite, fail trainability, not zero line preference. Line labels remain calibration-defined. F9 shows calibration coherence/noise power, F10 sensitivity spectra. R9 compares fraction of top-10 bands labelled lines and held-out frequency probes. Report amplitude diagnostics separately, along with PSD stationarity and data exclusions.

Acceptance: reproducible manifest and splits, no CH2 input path, frozen parameters, R9/status and per-seed tables, resource ledger and honest stationarity limitations. These are real detector noise results with hardware injection, not a discovery claim or proof of every P4 assumption.

## 8. Optional work and completion

### WP7 — E4 surprise as a monitor

Only after WP5 acceptance and if named in launch flags. Use frozen evaluation-cohort encoders from arms3/4/6 on mid/nominal and high/nominal. Predictor L2 averaged over target tokens under the trained fixed split; **do not introduce untrained random mask patterns**. Baseline is frozen arm1 reconstruction error on identical events. New event namespace `E4_test`, 10000 clean/fault-paired events per seed/cell; separate `E4_cal` 2000 clean events sets FPR .05 threshold.

Breaking hypotheses: independent white-noise amplitude x2 on channels6–7, glitch, dead channel7, timing displacement ±3 samples on channels4–7 with zero padding. Preserving hypotheses: line amplitude x2, common-mode amplitude x2, global gain x1.2. Define these by actual generator components; do not claim they mathematically preserve every learned dependency. Report per-fault full AUROC and realized FPR/TPR; family summary is the unweighted mean of fault AUROCs. R10 is empirical.

### WP8 — Optional ToST

Not selected by default. Only after WP5 acceptance and explicit inclusion in launch flags, with a separately approved compute allocation and two-week calendar limit. Replace attention locally, keep objectives and masks of arms4/6. Stop if unstable by day7. Per-layer comparisons use E2 tangent diagnostics and nuisance dependence, not a nonexistent fixed nonlinear R_c. This does not add a subject to Paper 3.

### Completion

- [ ] Reviewed v3 registration precedes new outcome-generating grids; `git log --reverse --format=%s 0cae6b0..JEPA` identifies it.
- [ ] Fixed-base diff touches allowed paths only; protected packages and PMT worktree unchanged.
- [ ] Environment/import, covariance, masks, optimizer coverage, analytic and negative-control tests pass on used hosts.
- [ ] WP2/WP5/final independent reviews state exactly what was rerun.
- [ ] R1–R10 relevant rows include pass/fail/not-evaluable/not-run, numeric evidence and seed IDs; scientific failures preserved.
- [ ] Every experiment has immutable config, split/data hashes, provenance, tables, figures, resource ledger and SHA256SUMS.
- [ ] `python -m jepa_noise.report all` regenerates prediction tables byte-identically, excluding explicitly separate wall-clock metadata.
- [ ] Every gate has a disposition; incomplete work remains labelled incomplete.
- [ ] `COMPLETION.md` records commands/output; `REPORT.md` (2–4 pages) reports findings, limitations, costs and remaining decisions.

No submission, external posting or release is authorized. Original CPAL Spotlight dates are planning notes only; verify the official venue/call before relying on them. Science and protocol completeness take priority over a calendar target.

## Appendix A. Reproduction and source guidance

### A.1 Original toy

`python docs/jepa/toy_check_2026-10-16.py`: d40, k3, N200000, default_rng(0), theta SD .4, eta SD3; gamma scales both views. Preserve draw order. The printed knob is `gamma_load`, not population gamma_n.

| Quantity | gamma_load=0 | gamma_load=1 |
|---|---|---|
| Top canonical correlations | .922,.889,.880,.026 | .998,.920,.888,.880 |
| Max CCA/reference angle | .9 degrees | 89.3 degrees |
| Pure GEVP/reference angle | 1.0 degrees | .9 degrees |
| Per-view whitening spectra | unchanged | unchanged |

Tolerance .01 on correlations, .5 degrees on angles. This known development example does not test finite-beta GD, P5 or deep JEPA.

### A.2 Sources and claims to verify

- Littwin et al., [arXiv:2407.03475](https://arxiv.org/abs/2407.03475): tied deep-linear dynamics under the paper's assumptions; distinguish its reproduction from random-loading generalization.
- Balestriero & LeCun, [LeJEPA, arXiv:2511.08544](https://arxiv.org/abs/2511.08544): soft SIGReg and practical implementation. Pin version/code/default reductions rather than equating it with hard covariance constraints.
- Maes et al., [LeWorldModel, arXiv:2603.19312](https://arxiv.org/abs/2603.19312); Assran et al., [I-JEPA, arXiv:2301.08243](https://arxiv.org/abs/2301.08243): cite precise implementation differences.
- Potapov et al., [arXiv:2603.14131](https://arxiv.org/abs/2603.14131), and Huang, [VJEPA, arXiv:2601.14354](https://arxiv.org/abs/2601.14354): verify full-text overlap before making a novelty claim; absence of shared noise is not assumed from an abstract.
- [TIDMAD paper §2](https://arxiv.org/html/2406.04378v2#S2) and [official code](https://github.com/jessicafry/TIDMAD): CH1 readout, CH2 reference, physical data/metadata contract. Current package readers are implementation evidence, not a substitute for actual file validation.
- Paper1 noise-weighted geometry and `src/latent_monitor/estimators/`: bind the cited version before manuscript work.

### A.3 Handoff rules

Implementer: read this document and its WP; edit permitted files only; use synthetic/analytic fixtures before expensive runs; report acceptance evidence and scientific failures separately; do not commit independently. Executor: use fixed SHA/config/manifest and budget allocation; never edit or recollect after a scientific STOP. One infrastructure retry before any outcome is allowed only within the same budget, in a fresh directory preserving the first attempt. Reviewer: independently recompute theorem checks/negative controls and raw reported statistics, check data/mask visibility and all split hashes, and report ACCEPT / ACCEPT WITH FIXES / REJECT with evidence.

### A.4 Version history

v1/v2 are historical drafts. v3 responds to the 16 October prelaunch reviews: conditional P5, finite-beta distinction, exact linearized E1b, explicit physical covariance addition, patch-aligned masks, operational M estimator, independent development/evaluation cohorts, corrected TIDMAD channels and manifests, frequency-based R9, full resource accounting, optimizer/dependency checks and descriptive statistical interpretation. See the revision report for per-finding disposition and the fresh v3 review.
