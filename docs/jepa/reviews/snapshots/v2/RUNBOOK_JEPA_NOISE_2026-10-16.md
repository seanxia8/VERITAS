# RUNBOOK — "Predictable Is Not Physical": JEPA's implicit noise weighting on detector signals

_16 October 2026, v2 (independent review applied, same day). Owner: Dowling Wong (KIT).
Collaborator: Junjie Xia (SLAC/Stanford). Branch: `JEPA` of `seanxia8/VERITAS`, cut from `dev` at
`0cae6b0`. Target: CPAL 2027 Recent Spotlight Track (non-archival; deadline 18 Jan 2027 AoE), plus a
short arXiv preprint. Companion launch prompt: `docs/jepa/AGENT_PROMPT_JEPA_2026-10-16.md`. Toy
check: `docs/jepa/toy_check_2026-10-16.py`._

This runbook is written for an **orchestrating agent**. That agent implements and runs the
study, may spawn **subagents**, and executes compute on **Deepthought** (and, if the author
allows, on the L40S Condor nodes). Every work package (WP) ends in acceptance checks that a
different agent, or the author, can re-run. The study is done when every box in §9 is ticked
with the command output that shows it.

---

## 0. Summary of the science

JEPA (joint-embedding predictive architecture) is a self-supervised **training objective**: an
encoder is trained to predict the *embedding* of a masked target block from a visible context
block. In the linear regime, with an isotropic-embedding constraint (SIGReg, as in LeJEPA), its
fixed point is canonical correlation analysis (CCA) between the context and target views.

JEPA therefore learns **whatever the two views share, ranked by how predictable it is**. That
includes physics θ, and it also includes noise η that is coherent across the views: spectral
lines, common mode, 1/f.
- When no noise is shared, the JEPA subspace equals Paper 1's noise-weighted subspace
  span(Σ_n⁻¹A), and JEPA gets there without knowing Σ_n.
- When noise is shared, it outranks weak physics.

**Whitening each view on its own cannot change this.** That covers per-channel
standardisation, per-band PSD weighting and a χ² loss, because canonical correlations are
invariant under invertible per-view linear maps. Only information about the **cross-view**
noise structure can. That information is in noise-only records, which detectors record for free
with random triggers. Three ways to use it:
- **Noise-aware masking:** choose the split with the least cross-view noise coherence.
- **Noise-anchored penalty:** make the embedding insensitive to added noise-only records. Its
  linear fixed point is span(Σ_n⁻¹A) even when η is shared.
- **Joint whitening (oracle):** whiten the whole input with the full noise covariance,
  including the cross-view terms. It works, but needs that full (CT)×(CT) covariance.

The paper states P1–P4 (§2). It tests them on linear models (E1), a Tier-1 transformer (E2) and
real TIDMAD noise (E3). Optionally (E4), it tests JEPA surprise as a fault monitor.

**Toy check, 16 Oct (§A.1):**

| | η not shared | η shared |
|---|---|---|
| JEPA vs noise-weighted subspace | 0.9° | 89° |
| Noise-anchored fixed point vs noise-weighted subspace | 1.0° | 0.9° |

Per-view whitening leaves the canonical correlations unchanged.

---

## 1. Rules for every agent on this task

### 1.1 Scope and no-touch list
- New code goes only in **`src/jepa_noise/`** (a new uv workspace member). Docs go in
  **`docs/jepa/`**, results in **`results/jepa/<YYYY-MM-DD>_<exp>/`**.
- **Do not modify:**
  - `src/latent_monitor/` (any file)
  - `src/noise_module/`, `src/noise_module_lucid/`
  - `src/reconstruction_model/` (subclass it, never edit it)
  - `src/tidmad_transformer/` (import it, never edit it)
  - `docs/PREREGISTRATION.md`, `docs/EXPERIMENT_DESIGN.md`, `latex/`

  This study is **not** part of Paper 3's preregistered claims and must not change them.
- Allowed edits outside the new folders: one line in `docs/INDEX.md`, the workspace-member
  line in the root `pyproject.toml`, and the resulting `uv.lock` change.
- If something in a no-touch package is wrong or missing, record it in
  `docs/jepa/ISSUES.md`. Work around it inside `src/jepa_noise/` with a wrapper or a local
  re-implementation, and never patch the original.
- Panda V2 / point-cloud encoders (arXiv 2609.00611) are **out of scope** for this study. Our
  inputs are waveforms; Panda is a candidate for a later study.

### 1.2 Git
- Work only on branch `JEPA`. Never commit to, merge into, rebase onto or push `dev` or `main`.
- **The first commit is the registration commit.** It contains this runbook, the launch
  prompt and the toy script, exactly as received, with the message
  `jepa: register runbook v2`. After it, a change to §2.3 (predictions) needs Gate G5.
- Commit at least once per WP, with message `jepa: WP<n> <short title>`. End each message with
  the attribution lines your harness requires.
- Diffs and checks use the fixed base **`0cae6b0`** (for example
  `git diff 0cae6b0...JEPA`), never `dev`, because `dev` will move.
- **Pushing** `JEPA` to `origin` is allowed only if the launch prompt says `PUSH_JEPA=yes`.
  Deepthought receives code only through GitHub. If pushing is not allowed and the needed
  commit is not on `origin/JEPA`, stop at Gate G0.
- **Only the orchestrator commits.** Executors on Deepthought never commit or push. Summary
  outputs come back to the Mac by `rsync deepthought:/ceph/dwong/jepa/returns/<exp>/ results/jepa/<exp>/`,
  and the orchestrator commits them.
- Never commit a file over 50 MB. Raw arrays, checkpoints and embeddings stay on Deepthought
  under `/ceph/dwong/jepa/raw/`.
- `docs/LUCID_MODEL_LINEUP_2026-10-09.md` is the author's untracked file. Leave it untracked.

### 1.3 Research hygiene
- The predictions in §2.3 are **registered by the registration commit, before any run**.
  Do not change hyperparameters, masks, seeds, data or metrics in response to any prediction
  outcome.
- A failed prediction is a finding, not a bug. Rule out a bug with the analytic check (E1a)
  and the unit tests, then report the failure as it is.
- Every run writes `provenance.json` containing:
  - the git SHA, and the output of `git status --porcelain --untracked-files=no` (which must
    be empty, otherwise save the diff);
  - the config and its SHA-256, and the seed;
  - the host, `nproc`, and the Python, numpy and torch versions;
  - the start time, end time and wall time.
- Every results README says **"development output, not reviewed"** until the author
  reviews it.
- Seeds:
  - E1: at least 5 per cell.
  - E2: at least 3 per arm × cell (target 5).
  - Report mean ± s.d. and every seed's value.
- Probes (pre-specified):
  - ridge regression on frozen embeddings;
  - α chosen by 5-fold CV on the probe-train split only;
  - reported on a held-out probe-test split never used for training or selection;
  - inputs to the probe are always **unmasked** inputs through the shared/target encoder.

### 1.4 Subagents
You may spawn subagents. Templates are in §A.3.
- **Implementer**: one per WP or module.
- **Executor (Deepthought)**: runs frozen configs only. It never edits code or configs and
  never commits.
- **Independent reviewer**: a fresh agent that has **not** seen the implementer's
  conversation. Required at WP2, WP5 and at the end.

Do not run two subagents that write the same files at the same time. At most 2 concurrent git
network operations.

### 1.5 Gates: stop, write `docs/jepa/GATE_<id>.md`, wait for the author
- **G0**: the needed commit is not on `origin/JEPA`, and you may not push.
- **G1**: E1a (§5.3) or the §A.1 reproduction fails. Do not continue to E2.
- **G2**: the E2 compute projection (§6.3) exceeds the cap on the allowed host.
- **G3**: TIDMAD needs a download over 50 GB, or needs credentials.
- **G4**: the E3 coherence step (§7.2) finds no band with noise MSC ≥ 0.5 under any candidate
  split. E3 as designed then has nothing to show.
- **G5**: any wish to change a registered prediction, metric, threshold or cell definition.

---

## 2. Theory to implement and test

### 2.1 Model and definitions
The two views are

    x_c = A_c θ + B_c η + e_c
    x_t = A_t θ + B_t η + e_t

- θ ∈ R^k is physics, with covariance Σ_θ.
- η ∈ R^r is noise that can be shared across the views.
- e_c and e_t are noise **not** shared across the views (independent of each other).
- The context-view noise covariance is Σ_n,c = B_c Σ_η B_cᵀ + Σ_e,c.

The noise-weighted reference subspace R_c is the top-k generalised eigenvectors of
(A_c Σ_θ A_cᵀ, Σ_n,c), which equals span(Σ_n,c⁻¹ A_c). It is Paper 1's subspace, and the span of
the optimal filter (OF).

**γ_load** is the experimental knob: the loading of η on the target view, B_t = γ_load·B_t0.
η is always present in the context view's noise.

**γ_n** is the reported quantity: the **population** top canonical correlation between the
views' noise parts. Compute it from the generator's exact covariance, never from samples:
sample CCA on high-dimensional views has a large null floor.

**Linear JEPA (SIGReg form):** min over W_c, W_t, P of E‖P W_c x_c − W_t x_t‖², subject to
W_t Σ_tt W_tᵀ = I. The context rows span the top-k context canonical directions. **All angles
are measured with W_c.**

**Linear JEPA (stop-grad form):** the deep linear setting of Littwin et al. (NeurIPS 2024).

### 2.2 Propositions: write them up in `docs/jepa/THEORY.md` with proofs
- **P1 (shared-latent recovery).** The nonzero canonical correlations come only from (θ, η)
  through the cross-covariance A_t Σ_θ A_cᵀ + B_t Σ_η B_cᵀ. JEPA ranks factors by canonical
  correlation and cannot tell θ from η. A strong shared η outranks a weak θ.
- **P2 (per-view invariance).**
  - For invertible M_c and M_t, CCA(M_c x_c, M_t x_t) has the same canonical correlations,
    and its directions map back to the same functionals of x.
  - So per-view whitening, per-channel standardisation, per-band PSD weighting and χ² losses
    cannot change what linear JEPA prefers.
  - **This invariance does not hold for joint maps.** Whitening the stacked (x_c, x_t) with the
    full noise covariance, including B_t Σ_η B_cᵀ, makes the noise parts uncorrelated across
    views, which removes η from the canonical spectrum.
  - Per band: with complex coherencies c_s and c_n between views, the coherency of a band is
    (c_s S + c_n J)/(S + J) and the MSC is its squared modulus. Both are unchanged by any
    common per-band scaling of S and J. A spectral line narrower than the band resolution
    stays coherent after band-level whitening.
- **P3 (agreement with Paper 1).**
  - Claim: if B_t = 0 and A_t Σ_θ has full rank k, then span(W_cᵀ) = span(Σ_n,c⁻¹ A_c) = R_c.
  - Proof: the range of Σ_cc⁻¹ Σ_ct is span(Σ_cc⁻¹ A_c), and by Woodbury
    Σ_cc⁻¹ A_c = Σ_n,c⁻¹ A_c (I + Σ_θ A_cᵀ Σ_n,c⁻¹ A_c)⁻¹.
- **P4 (noise-anchored fixed point).**
  - Problem: min tr(W Σ_rec Wᵀ) subject to W Σ_x Wᵀ = I, where Σ_rec is the covariance of
    noise-only records.
  - Solution: the generalised eigenvectors of (Σ_rec, Σ_x) with the smallest eigenvalues ν.
  - If Σ_rec = Σ_n,c (stationary noise, no event-correlated noise), this is the problem
    A Σ_θ Aᵀ w = μ Σ_n w with μ = (1 − ν)/ν, so the span is R_c **whether or not η is shared**.
  - The deep version (§6) uses the overlay penalty E‖z(x + n_rec) − z(x)‖². For a linear z
    this equals tr(W Σ_rec Wᵀ).
- **P5 (joint whitening, oracle).** If x is whitened by the full joint noise covariance before
  the split, the JEPA span maps back to R_c. This needs the full covariance; P4 needs only noise
  records and an embedding-level penalty.
  - **The whitener must be block-lower-triangular, context first.** That is the Cholesky
    factor of the noise covariance in the order (context, gap, target); for a time split it
    is a causal innovations filter. Then the whitened context depends only on x_c.
  - A symmetric whitener (Σ^{-1/2}, or a symmetric Kronecker whitener) mixes target samples
    into the context. That leaks the target and must not be used.
  - Population check, 16 Oct (d = 40, k = 3), at γ_load ∈ {0.5, 1}:

    | | top canonical correlation | angle to R_c |
    |---|---|---|
    | plain JEPA | 0.993 / 0.997 | 85° |
    | after context-first joint whitening | 0.893 (= ρ_θ,1) | 0.0° |
- **Criterion M (mask selection).** Over a candidate family of splits:
  - for each split, estimate ρ_η^noise, the top canonical correlation of noise-only records
    between the views, and ρ_θ,k, the k-th event canonical correlation after subtracting the
    noise cross-covariance;
  - choose the split that maximises ρ_θ,k − ρ_η^noise;
  - use ridge-CCA with ε = 1e-3·tr(Σ)/d and n ≥ 20·d noise records;
  - report both quantities for every candidate.

**Objections to address in THEORY.md §Discussion**
1. "It's just CCA." Agreed. The contributions are the detector consequence (P1, P2), the three
   remedies and their costs, and the deep and real-data tests.
2. "In the linear case P4 does not need the prediction term." True. That is why E2 includes an
   `anchor_only` control.
3. Potapov et al. (arXiv 2603.14131) and VJEPA's noisy-TV toy (arXiv 2601.14354) do not treat
   noise shared across views. Cite both and state the difference.
4. "Just whiten properly." P2/P5: that needs the full cross-view covariance, at full spectral
   resolution. P4 and M get there with records and no Σ_n model.

### 2.3 Registered predictions (frozen by the registration commit)

| id | prediction | test | pass criterion |
|---|---|---|---|
| R1 | Linear JEPA ≈ R_c when η is not shared | E1a, E1b at n = 1e5 | median-over-seeds max angle ≤ 5° (E1a), ≤ 10° (E1b) at γ_load = 0 |
| R2 | The angle grows with sharing | E1a, E1b at n = 1e5 | Spearman(γ_n, angle) ≥ 0.8 over the grid; E1a angle at γ_load = 1, η scale 3 is ≥ 45° |
| R3 | Per-view invariance | E1a | max abs difference of **unregularised** sample canonical correlations before/after per-view whitening ≤ 1e-8 (n ≫ d) |
| R3b | Joint whitening removes η | E1a | after joint whitening, the top canonical correlation at γ_load = 1 is ≤ ρ_θ,1 + 0.02 and the angle to R_c is ≤ 5° |
| R4 | The noise-anchored remedy holds at every γ | E1a, E1b at n = 1e5 | GEVP solution, and GD at β = 10: angle ≤ 5° (E1a), ≤ 10° (E1b) at every γ_load. Masked arm in E1a is a sanity check only (true by construction) |
| R5 | Depth makes JEPA learn shared η earlier | E1c | stop-grad dynamics, γ_load ∈ {0.5, 1}. Δ = t_η − min_i t_θ,i (steps to 50% of final projection norm). Δ(L = 3) < Δ(L = 1) in ≥ 4/5 seeds at each γ_load |
| R6 | Plain JEPA puts η in its embedding as sharing grows | E2, nominal SNR, each of `ijepa_ema` and `sigreg_jepa` separately | Spearman(γ_n, nuisance R²) ≥ 0.8 over 3 levels × seeds |
| R7 | The remedies cut nuisance and keep physics | E2, (high, nominal) and (high, sub-noise) | nuisance R² of `sigreg_jepa_anchored` and `sigreg_jepa_masked` < `sigreg_jepa` in every paired seed; seed-mean physics MSE/CRLB, per parameter, not worse than `sigreg_jepa` by more than 10% |
| R8 | Learning η costs robustness | E2, (high, nominal) | scale the **line** component ×2 at test time. Degradation = probe-test MSE_shift/MSE_clean, averaged over the 3 parameters. `sigreg_jepa` degrades more than `sigreg_jepa_anchored` in ≥ 2/3 of paired seeds |
| R9 | Real noise lines take over plain JEPA | E3 | the fraction of the top-10 latent-sensitivity bands that are real-noise-line bands is larger for `sigreg_jepa` than for `sigreg_jepa_anchored` in ≥ 2/3 of paired seeds, **and** the anchored injected-amplitude probe R² is ≥ 0.9 × the plain arm's |
| R10 (opt.) | Surprise sees faults that break cross-view consistency, not those that keep it | E4 | mean full AUROC ≥ 0.8 on the breaking family and ≤ 0.65 on the preserving family. Also report TPR at FPR 5% |

---

## 3. Hosts and environments

### 3.1 Mac (development, unit tests, small E1)
- Repo `~/Documents/VERITAS`, branch `JEPA`, CPU only.
- Use the `.venv-cpu` recipe in the repo README ("Running the tests"). Never `uv sync` the CUDA
  torch pin on macOS.

### 3.2 Deepthought (E1 full grids, E3 coherence step, E2/E3 training if §6.3 allows)
- SSH alias `deepthought`. Work root **`/ceph/dwong/jepa/`**, with subfolders `code/`, `venv/`,
  `logs/`, `raw/`, `returns/`. Create `logs/` before any launch.
- **One clone per commit:** `/ceph/dwong/jepa/code/<sha>/`, a fresh clone of `origin` at that
  SHA. Concurrent runs never share a working tree.
- Environment: `/ceph/dwong/jepa/venv`, Python 3.11, with numpy, scipy, pandas, matplotlib,
  h5py, pytest, scikit-learn and torch (CPU wheel). Do not reuse other campaigns'
  environments. Import `reconstruction_model` and `tidmad_transformer` from `src/` via
  `PYTHONPATH=src`; do not install their CUDA pins.
- Launch:

      tmux -L jepa new-session -d -s <exp> "cd /ceph/dwong/jepa/code/<sha> && . /ceph/dwong/jepa/venv/bin/activate && export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src && python -m jepa_noise.run <config> > /ceph/dwong/jepa/logs/<exp>.out 2> /ceph/dwong/jepa/logs/<exp>.err"

- Monitor with `tmux -L jepa capture-pane -p -t <exp>` only. The default tmux socket and other
  sockets belong to Paper 1 campaigns: never attach to them, send keys to them or kill them.
- Git: `GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND='ssh -o BatchMode=yes'`.
- **Preflight** (record all output in `docs/jepa/PREFLIGHT.md`):
  - `hostname`, `nproc`, `uptime` (load average)
  - `free -g`, `df -h /ceph/dwong`
  - `nvidia-smi` (expected: absent)
  - `git ls-remote origin JEPA`
  - `python -c "import torch; print(torch.__version__, torch.get_num_threads())"`

  **Free cores** = nproc − 15-min load average. Use at most min(32, free cores − 4) workers.

### 3.3 L40S Condor (E2/E3 training; only if the launch prompt says `GPU_ALLOWED=yes`)
- Use the repo's `scripts/train_compact_l40s/` pattern, with the `torch==2.5.1+cu124` pin and
  the author's allocation.
- Use the same configs as on CPU. Only the device differs. Record the GPU model in provenance.

---

## 4. Package layout and interfaces (`src/jepa_noise/`)

```
src/jepa_noise/
  pyproject.toml   __init__.py
  views.py     # ExactModel (E1a), Tier1Views (E1b/E2): γ knobs, gaps, η ground truth, exact covariances
  linear.py    # cca, sigreg_linear, stopgrad_deep_linear, mae_mse, mae_chi2(=R_c), anchored (GEVP, GD), jointwhite, of
  masks.py     # split families, criterion M
  metrics.py   # principal angles (W_c), nuisance R², physics/CRLB, population γ_n
  crlb.py      # Fisher JᵀΣ⁻¹J for Tier-1 templates
  backbone.py  # TokenGridEncoder: current_compact subclass, fixed standardisation, returns (B, N_patch, C, d)
  sigreg.py    # SIGReg (random 1-D projections, Epps–Pulley vs N(0,1))
  jepa.py      # the E2 arms
  train.py  run.py  report.py  configs/  tests/
```

Binding interfaces:
- `views.ExactModel(d, k, r, theta_scale, eta_scale, gamma_load, share="target"|"both", seed)`.
  - Methods: `.sample(n) -> (Xc, Xt, theta, eta)`, `.noise_records(n) -> (Nc, Nt)`,
    `.population() -> dict` (all covariances, R_c, γ_n).
  - `share="both"` scales η in both views by γ and consumes the RNG in exactly the order of
    the toy script. It exists only to reproduce §A.1.
- `views.Tier1Views(cell, split, gap, line_scale, cm_strength, seed)`.
  - `cell` is a `latent_monitor.tier1.Cell` (from `reference_cell`).
  - `split ∈ {"time", "channel", "interleaved", "block"}`.
  - Returns event batches (full input, context/target index sets, θ, η), noise-only batches,
    and `.population()` (exact view covariances from the generator's implied covariance).
- `linear.fit(method, Xc, Xt, k, **kw) -> W_c (k×d_c)`.
- `metrics.max_principal_angle(W1, W2)` in degrees. It must agree with
  `latent_monitor.estimators.cw_pca.principal_angles`; test this.

---

## 5. WP0–WP3: preflight, theory, linear package, E1

### WP0: preflight and registration
1. Check the branch: `git branch --show-current` = `JEPA`, and
   `git merge-base --is-ancestor 0cae6b0 JEPA` succeeds.
2. Make the registration commit (§1.2): `docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md`,
   `docs/jepa/AGENT_PROMPT_JEPA_2026-10-16.md` and `docs/jepa/toy_check_2026-10-16.py`. Skip
   files that are already committed unchanged.
3. Run `python docs/jepa/toy_check_2026-10-16.py`. It must print the §A.1 table; if not, Gate G1.
4. Create the skeleton:
   - `src/jepa_noise/` with its `pyproject.toml`;
   - add it to the root workspace members;
   - `results/jepa/.gitkeep`;
   - the `docs/INDEX.md` line.
5. Gate G0 if needed, then run the Deepthought preflight (§3.2).

**Accept:**
- `docs/jepa/PREFLIGHT.md` is complete.
- `PYTHONPATH=src python -c "import jepa_noise"` works on both hosts.

### WP1: theory note
Write `docs/jepa/THEORY.md` covering §2.1–2.2 in full. Each of P1–P5 and M gets a numerical
check in `tests/test_theory.py`, using population matrices only, to 1e-8.

**Accept:** the tests pass. The WP2 reviewer re-derives P3, P4 and P5 on paper.

### WP2: linear package
Implement `views.ExactModel`, `linear.py`, `metrics.py`, `crlb.py` and `masks.py`.

- **CCA:** generalised symmetric eigenproblem. Unregularised for R3; ridge
  ε = 1e-6·tr(Σ)/d elsewhere, recorded.
- **`stopgrad_deep_linear`:** encoder W = W_L⋯W_1 for L ∈ {1, 2, 3}, linear predictor P,
  target = stop-grad(W). Init σ = 1e-3. Full-batch GD on population covariances. Log each θ
  mode's and η's projection norm per step.
- **`sigreg_linear`:** GD on the SIGReg-form objective (§2.1). Must converge to `cca` within
  1° on E1a with γ_load = 0.
- **`anchored`:** GEVP (Σ_rec, Σ_x) directly, plus GD (prediction + SIGReg + β·tr(WΣ_recWᵀ))
  with β ∈ {0.1, 1, 10}. At small β the GD solution is a CCA/anchored compromise; record it,
  but R4 uses the GEVP solution and β = 10.
- **`jointwhite`:** whiten the stacked views (context first) with the inverse lower Cholesky
  factor of the population joint noise covariance, run CCA, and map the context functionals
  back to x_c. Test that the whitened context has zero dependence on x_t.
- **`mae_chi2`:** R_c with the known Σ_n,c. It is the reference subspace for all angles.
- **CRLB:**
  - Fisher I(θ) = Jᵀ Σ_n⁻¹ J, with J = ∂(template)/∂(amplitude, t0, τ_t) by central
    differences (h = 1e-4 × range), from `latent_monitor.tier1.EventFamily.waveform` /
    `pulse_template_2`.
  - The CRLB is per event, [I⁻¹]_ii. Report its mean over probe-test events.
  - Cross-check: the OF amplitude variance must equal 1/I_aa (known t0 and τ), not
    [I⁻¹]_aa, to 1%.

Unit tests:
- P3 (≤ 1e-6 rad, population).
- R3-style invariance, unregularised (1e-8).
- P4 with η shared (≤ 1e-6 rad).
- P5 (≤ 1e-6 rad).
- `sigreg_linear` → `cca` (≤ 1°).
- `stopgrad_deep_linear` with γ_load = 0 → R_c (≤ 2°).
- Principal angles agree with `cw_pca`.
- Nuisance R² ≈ 0 for independent Z, and ≈ 1 for Z = η + small noise.
- The OF cross-check.
- Masks: on a constructed model where split A shares η and split B does not, M picks B.

**Accept:**
- All tests pass on both hosts.
- The reviewer reproduces §A.1 with `ExactModel(share="both")` (exact values), plus the
  population versions of R1–R4. Report in `docs/jepa/reviews/WP2_REVIEW.md`.

### WP3: E1 linear experiments (CPU)

**E1a: exact linear-Gaussian model** (`share="target"`).
- Grid:
  - d = 40, k = 3, r = 1
  - θ scale ∈ {0.1, 0.4, 1.0}
  - η scale ∈ {1, 3}
  - γ_load ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
  - n ∈ {1e3, 1e4, 1e5}
  - 5 seeds
- Methods: `cca`, `sigreg_linear`, `stopgrad_deep_linear` (L = 1, 2, 3), `mae_mse`,
  `mae_chi2`, `anchored` (GEVP; GD at β ∈ {0.1, 1, 10}), `jointwhite`, `masked` (sanity
  check), `of`.
- Outputs: `table.csv` with one row per method × cell × seed, holding the angle, nuisance
  R², canonical correlations, population γ_n and physics/CRLB.
- Figures:
  - F1: angle vs γ_n per method (n = 1e5).
  - F2: canonical spectra at γ_load ∈ {0, 1}.
  - F3: convergence in n. This is the only place n < 1e5 is used for claims.

**E1b: Tier-1 waveforms.**
- Cell: `latent_monitor.tier1.reference_cell(n_channels=8, n_samples=256)`.
- Splits are defined **in samples**:
  - time: [0, 112) → [144, 256), with a 32-sample gap (≥ the noise autocorrelation length;
    measure it and record);
  - channel: {0–3} → {4–7};
  - interleaved: {0, 2, 4, 6} → {1, 3, 5, 7};
  - block: channels 0–3 × [0, 112) → channels 4–7 × [144, 256).
- Two knobs, 5 levels each:
  - **line:** a narrow `line` component (type `line` in `noise_module.spectral.models`) at a
    frequency inside the pulse band, scale 0 … 3× the white floor. Shared across time splits.
  - **common mode:** `corr_strength` in mode `shared_private`, 0 … 0.9. Shared across channel
    splits.
- η ground truth: generate the shared part separately inside `views.py`, as two generator
  calls summed. Test that the summed covariance matches the single-call implied covariance to
  2%.
- η targets for probes: the line's in-phase and quadrature amplitudes at the line frequency;
  the common-mode coefficients on its top-2 temporal modes.
- Reference R_c: the exact context-view noise covariance from the generator's implied
  covariance.
- Same methods, metrics and figures as E1a, per knob and split. n = 1e5 for claims.

**E1c: learning order.**
- E1a at θ scale 0.4, η scale 3, γ_load ∈ {0.5, 1}.
- `stopgrad_deep_linear` with L ∈ {1, 2, 3}, 5 seeds.
- F4: learned-step per mode, coloured by depth.
- Compute Δ as in R5.

**Accept (WP3):**
- `results/jepa/<date>_E1_summary/PREDICTIONS.md` gives R1–R5 (and R3b) pass/fail, with
  numbers and seeds.
- `python -m jepa_noise.report E1` regenerates it byte-identically.
- Gate G1 applies.

---

## 6. WP4–WP5: deep JEPA and E2 (Tier-1 transformer)

### WP4: torch implementation

**`backbone.TokenGridEncoder`.**
- Subclass `reconstruction_model.model.Transformer` (the `current_compact` architecture).
- Override `forward` so that it:
  1. **replaces `normalise_input_sequence` with a fixed per-channel standardisation**, whose
     mean and std are estimated once per cell from noise-only records and stored as buffers;
  2. returns the token grid (B, N_patch, C, d) before the final mean-pool and the heads.

  Record this deviation in `docs/jepa/THEORY.md` §Implementation. The reason: per-row z-scoring
  uses statistics of masked target patches, which leaks targets into the context, and it
  removes the amplitude scale.
- Tier-1 config: C = 8, n_samples = 256, patch_len = 32, stride 32, giving 8 patches × 8
  channels = 64 tokens. Time splits use patches 0–3 → patches 4–7, with the gap implemented by
  masking out of the loss the patch straddling the gap. Record this mapping to the E1b sample
  split.
- **Primary config** (pre-registered): d_model = 128, n_head = 4, depth (1,1), batch 128,
  10 000 steps.
- **Depth sweep**: (2,2) and (4,4).

**Masking.** Attention is factorised (over time within a channel, then over channels within a
patch), so tokens cannot be dropped. Replace the context encoder's masked tokens with a learned
`[MASK]` embedding *after* patch embedding.
- **Leakage test:** the gradient of every context-encoder output with respect to every masked
  input sample is exactly 0.

**Predictor.**
- d_pred = 64, 2 blocks, full attention over the token grid.
- Inputs: context outputs at context positions, plus positional query tokens at target
  positions.
- Outputs: predictions at target positions.

**Arms.** All share the backbone, optimiser (`configure_optimisers`; the predictor and heads
go in the AdamW group), steps and data. Default split for all arms except 5: **time with gap**.

1. `mae_mse`: linear head d → patch_len on target tokens. Loss in measurement coordinates
   (unstandardise first).
2. `mae_chi2`: as 1, with the residual whitened by the marginal noise covariance of the
   target index set, built from the cell's implied covariance.
3. `ijepa_ema`:
   - EMA target encoder, with τ going 0.996 → 1.0 on a cosine schedule;
   - targets are layer-normed target-encoder outputs at target positions;
   - L2 loss.
4. `sigreg_jepa` ("SIGReg-JEPA, LeWorldModel-style"):
   - one shared encoder for context and target, no EMA, no stop-grad (as in
     LeJEPA/LeWorldModel);
   - loss = prediction L2 + λ·SIGReg;
   - SIGReg applies to **both** the pooled embeddings and the target-token embeddings
     (B·N_t rows);
   - λ is the LeJEPA default; read arXiv 2511.08544 and record the value and section;
   - log the effective rank of the token embeddings every 500 steps.
5. `sigreg_jepa_masked`: arm 4, with the split chosen by criterion M from {time, channel,
   interleaved, block}. Compute it once from noise-only records **before** training and
   commit the M report.
6. `sigreg_jepa_anchored`: arm 4 plus an overlay penalty β·E‖z(x + n_rec) − z(x)‖² / tr Cov(z),
   where z is the pooled embedding and n_rec is an independent noise-only record. β ∈ {0.1, 1,
   10} is chosen on the dev cell only, then frozen.
7. `hybrid_chi2_jepa`: arm 2 loss + λ_J·(arm 4 loss), with λ_J ∈ {0.1, 1} chosen on the dev
   cell.
8. `anchor_only`: SIGReg + the arm-6 penalty, **no** prediction loss (control for objection 2).
9. `sigreg_jepa_jointwhite` (oracle): arm 4 on input whitened before masking by the inverse
   **lower Cholesky** factor of the cell's full implied noise covariance. Build that covariance
   from the factors of `Cell.implied_whitener()`, with the index order (context samples, gap,
   target samples). Never use the symmetric whitener: it mixes target into context. The
   leakage test applies to the whitened input as well. Label this arm "oracle (needs Σ_n)"
   everywhere.

**Tests** (`tests/test_torch.py`):
- shapes;
- the leakage test;
- the EMA formula;
- SIGReg ≈ 0 for N(0, I) samples and large for collapsed ones;
- 10-step bit-determinism on CPU;
- anti-collapse smoke: pooled-embedding std > 0.1 and token effective rank > 4 after 500 steps
  on the dev cell, for every JEPA arm.

**Accept (WP4):** the tests pass on the run host, and a 500-step smoke run of all 9 arms
finishes (`results/jepa/<date>_E2_smoke/`).

### WP5: E2 runs
- **Cells:**
  - shared-noise level ∈ {low, mid, high}: the **line** knob levels whose population γ_n for
    the time split is closest to 0.1, 0.5 and 0.9 (record the actual values);
  - SNR ∈ {nominal (EventFamily default), sub-noise (amplitude range × 0.25)};
  - that gives 6 cells.
- **Dev cell** = (mid, nominal). It is used only for β, λ_J and the smoke run, reported
  separately and never pooled.
- **Ids:**
  - train events 0–199 999;
  - probe-train 200 000–209 999;
  - probe-test 210 000–219 999;
  - noise records from a disjoint id range;
  - the reviewer checks disjointness.
- **Evaluation** (frozen shared/target encoder, unmasked input, pooled embedding):
  - physics MSE/CRLB per parameter;
  - nuisance R² (line I/Q, common-mode coefficients);
  - robustness (R8);
  - a linearised-readout angle to R_c (diagnostic only).
- **Depth sweep:** arms 3, 4 and 6 at (2,2) and (4,4) on the (high, nominal) cell.
- **Seeds:** 3 minimum; 5 if §6.3 allows.

### 6.3 Compute gate
1. Benchmark 300 steps of arms 1, 4 and 6 on the intended host.
2. Project the full grid: 9 arms × 6 cells × seeds, plus the depth sweep.
3. Caps:
   - Deepthought: 72 h wall at the §3.2 worker count;
   - L40S: 48 GPU-hours.
4. If the projection is over the cap: drop to 3 seeds, then to 5 000 steps. If it is still
   over, Gate G2.

Rough estimate for planning: about 0.5M parameters, 8k tokens per batch, about 1e11 FLOP per
step including the target, predictor and noise branches. That is seconds per step on one CPU
thread, so the full grid on CPU is likely near or over the cap, and L40S is the realistic host.
Never drop an arm or cell without the author.

**Accept (WP5):**
- `results/jepa/<date>_E2/` contains `table.csv` and the figures:
  - F5: nuisance R² vs γ_n per arm;
  - F6: physics/CRLB;
  - F7: robustness;
  - F8: depth.
- `PREDICTIONS.md` gives R6–R8 pass/fail.
- An independent reviewer recomputes them from raw per-seed JSONs
  (`docs/jepa/reviews/WP5_REVIEW.md`).

---

## 7. WP6: E3, TIDMAD real noise

### 7.1 Data
TIDMAD (arXiv 2406.04378, CC BY 4.0, Zenodo 10.5281/zenodo.11458076):
- 20 train and 20 val files with injected sinusoids. ch0 is the injection reference; ch1 and
  ch2 are SQUIDs.
- 208 science files with no injection, which are noise-only.
- 10 MS/s, 8-bit.

Repo readers:
- `tidmad_transformer.data.read_channel_pair`
- `tidmad_transformer.n_injection.read_noise_only`
- `tidmad_transformer.psd.estimate_band_psd`
- STFT config in `tidmad_transformer.config` (n_fft 4096, hop 2048, window 198 656 samples →
  96 frames)

First look on Deepthought: `find /ceph/dwong -iname "abra_*.h5" | head`. Otherwise download 4
train + 4 val + 16 science files. Gate G3 if that is over 50 GB.

**ch0 is never a model input.** It is used only to derive the injected amplitude and
frequency targets.

### 7.2 Step 1: cross-view coherence (no training; Gate G4)
- **Tokens:** use **complex** STFT bins. If `tidmad_transformer`'s pipeline exposes only
  magnitudes, compute complex STFTs with the same window and hop in `jepa_noise` and record
  this.
- **Candidate splits:**
  - time: frames [0, 46] → [50, 95] (gap of 3 frames, since frames overlap by 2 048 samples);
  - channel: ch1 → ch2;
  - block.
- **Per band, on noise-only windows (≥ 200 windows):**
  - the noise MSC between views;
  - J(f).
- **Injected windows:** signal coherency at the injection frequencies.
- **Predicted MSC:** |c_s S + c_n J|²/(S + J)², with equal per-view PSDs assumed and checked.
- **Line bands:** J > 5 × the running median over ±50 bands.
- **Output:**
  - F9: predicted MSC vs 1/J, with line bands marked;
  - criterion M's report.

Real lines and injected sinusoids are both coherent across time. Plain JEPA cannot tell them
apart; the anchored penalty can, because only noise lines appear in noise-only files. This is
the real-data test of P4.

### 7.3 Step 2: training
- Arms `mae_chi2`, `ijepa_ema`, `sigreg_jepa`, `sigreg_jepa_masked` and
  `sigreg_jepa_anchored`, on STFT band-frame tokens.
- Use a subclass of the `tidmad_transformer` backbone with fixed per-band standardisation
  (§6, WP4, same reason).
- 3 seeds.

### 7.4 Compute
Benchmark 200 steps. If 5 arms × 3 seeds is projected over 48 h CPU (or 24 GPU-hours on L40S),
reduce the window from 96 to 24 frames and the bands to the 0–500 kHz region that contains the
injections and the strongest lines. Record the choice before training.

### 7.5 Metrics
- **Per-band latent sensitivity:** the mean over windows of ‖∂z/∂X(f, ·)‖, normalised to sum 1.
- **R9:** the fraction of the top-10 sensitivity bands that are line bands, plus the
  injected-amplitude probe R².
- Rank correlation of sensitivity with the predicted MSC and with 1/J.

**Accept (WP6):** `results/jepa/<date>_E3/` contains the coherence table, F9, the sensitivity
table, F10 (sensitivity spectra) and `PREDICTIONS.md` (R9).

---

## 8. Optional WPs (only after WP5 is accepted; only those named in the launch prompt)

### WP7: E4, surprise as a fault monitor
- **Encoders:** the frozen E2 encoders for arms 3, 4 and 6, on the (mid, nominal) and (high,
  nominal) cells.
- **Surprise:** the predictor's L2 error, averaged over target blocks, over 4 random masks per
  event.
- **Breaking faults** (break cross-view consistency):
  - white noise ×2 on channels 6–7;
  - glitch: `EventFamily(extra="glitch")`;
  - dead channel 7;
  - timing jitter: channels 4–7 shifted by ±3 samples using integer shifts with zero padding
    (causal; never an FFT shift).
- **Preserving faults** (keep it): line ×2; common mode ×2; global gain ×1.2.
- **Baseline:** `mae_mse` reconstruction error on the same events.
- **Report** (R10): full AUROC and TPR at FPR 5%, per fault and per family.

### WP8: white-box ToST arm, time-boxed to 2 weeks
- Only if WP5 is accepted by 1 Dec 2026.
- Port ToST's token-statistics attention (github.com/RobinWu218/ToST) in place of the
  backbone attention. Keep the patching, masks and the objectives of arms 4 and 6.
- Measure each layer's subspace against R_c and against η's loading.
- Stop and report if it is not training stably by day 7.

---

## 9. Completion checklist (definition of done)

Paste each command and its output into `docs/jepa/COMPLETION.md`.

- [ ] `git log --oneline 0cae6b0..JEPA` starts with `jepa: register runbook v2` and has at
      least one commit per WP.
- [ ] `git diff 0cae6b0...JEPA --stat` touches only: `src/jepa_noise/`, `docs/jepa/`,
      `results/jepa/`, `docs/INDEX.md`, root `pyproject.toml`, `uv.lock`.
- [ ] `git diff 0cae6b0...JEPA -- <each no-touch path>` is empty.
- [ ] `git log -p -- docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md` shows no change after
      registration, or only changes with an approved G5.
- [ ] `PYTHONPATH=src pytest src/jepa_noise -q` passes on every host used (paste the summary
      lines).
- [ ] `PYTHONPATH=src pytest src/latent_monitor src/noise_module -q` gives the same result as
      at `0cae6b0`.
- [ ] `docs/jepa/THEORY.md` covers P1–P5, M, proofs, the four objections and §Implementation
      deviations.
- [ ] `results/jepa/*_E1a`, `*_E1b`, `*_E1c`, `*_E2` and `*_E3` (unless G4 stopped it) each
      contain `table.csv`, figures, `provenance.json` per run, a README ("development output,
      not reviewed") and `SHA256SUMS`.
- [ ] `python -m jepa_noise.report all` regenerates every `PREDICTIONS.md` byte-identically.
- [ ] R1–R9 (and R10 if WP7 ran) each have pass/fail, numbers and seeds.
- [ ] Independent reviews exist: `reviews/WP2_REVIEW.md`, `WP5_REVIEW.md` and
      `FINAL_REVIEW.md`, each by a fresh agent, each stating what it re-ran.
- [ ] Every fired gate has a `GATE_<id>.md` with the author's answer.
- [ ] `docs/jepa/REPORT.md` (2–4 pages) covers: headline results, figure list, what failed,
      what is not claimed, compute used, and open decisions for the author and Junjie.
- [ ] Raw outputs on Deepthought (`/ceph/dwong/jepa/raw/`) are listed with sizes; no committed
      file is over 50 MB.

---

## 10. Timeline (CPAL Spotlight, 18 Jan 2027)

| by | milestone |
|---|---|
| 24 Oct | WP0–WP2 accepted (G1 cleared) |
| 6 Nov | WP3 (E1) accepted |
| 20 Nov | WP4 accepted; §6.3 benchmark and host decision |
| 11 Dec | WP5 (E2) accepted |
| 1 Jan | WP6 (E3) accepted, or the G4 decision recorded; WP7 if time allows |
| 10 Jan | REPORT.md and final review; the author drafts the manuscript and abstract |
| 18 Jan | submission (author) |

Paper 1 should be on arXiv before submission, because P3 cites its noise-weighted subspace.

---

## Appendix

### A.1 Toy check (16 Oct 2026; reproduce exactly; Gate G1)
`python docs/jepa/toy_check_2026-10-16.py` uses:
- d = 40, k = 3, N = 200 000, rng `default_rng(0)`;
- A_c, A_t, B_c, B_t ~ N(0, 1);
- L ~ N(0, 1)/√d and Σ_n = LLᵀ + 0.3 I;
- θ ~ 0.4·N(0, I), η ~ 3·N(0, 1);
- η is scaled by γ in **both** views (`share="both"`);
- reference = top-k GEVP of (0.16·A_cA_cᵀ, Σ_n + γ²·9·B_cB_cᵀ);
- whitening is **per view**.

Expected output:

| quantity | γ = 0 | γ = 1 |
|---|---|---|
| top canonical correlations | 0.922, 0.889, 0.880, 0.026 | 0.998, 0.920, 0.888, 0.880 |
| max angle CCA top-3 vs reference | 0.9° | 89.3° |
| canonical correlations after per-view whitening | identical | identical |
| max angle noise-anchored vs reference | 1.0° | 0.9° |

Tolerance ±0.01 on correlations and ±0.5° on angles.

### A.2 Read before implementing
- Littwin et al., arXiv 2407.03475: deep linear JEPA dynamics.
- Balestriero & LeCun, *LeJEPA*, arXiv 2511.08544: SIGReg and default λ.
- Maes et al., *LeWorldModel*, arXiv 2603.19312: a small end-to-end SIGReg JEPA.
- Assran et al., *I-JEPA*, arXiv 2301.08243: EMA target, predictor, masks.
- Potapov et al., arXiv 2603.14131, and Huang, *VJEPA*, arXiv 2601.14354: related work to
  differentiate.
- Paper 1 (Wong, *Noise Covariance Defines Reconstruction Geometry*) and
  `src/latent_monitor/estimators/`.

### A.3 Subagent prompt templates

**Implementer:**
> You implement WP<n> of `docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md` (VERITAS, branch `JEPA`).
> Read §1, §2, §4 and WP<n> in full first. Write only under `src/jepa_noise/` and
> `docs/jepa/`. Never edit the no-touch list. Do not commit; the orchestrator does. Make every
> Accept check of WP<n> pass and paste the outputs into `docs/jepa/wp<n>_log.md`. Do not tune
> anything in response to a registered prediction (§2.3). At a gate (§1.5), stop and report.

**Executor (Deepthought):**
> You run frozen configs for WP<n> on Deepthought. Do not edit code or configs, and do not
> commit or push. Clone `origin` at SHA <sha> into `/ceph/dwong/jepa/code/<sha>` if it is
> absent. Use `/ceph/dwong/jepa/venv`. Launch in `tmux -L jepa` as in §3.2 and monitor with
> capture-pane only. On completion, run `python -m jepa_noise.report <exp>`, write SHA256SUMS,
> and put the summary outputs in `/ceph/dwong/jepa/returns/<exp>/` (raw outputs in
> `/ceph/dwong/jepa/raw/<exp>/`). Return the console tail, timings and paths. If there is an
> infrastructure failure before the first result, relaunch once in a fresh directory. On a
> second failure, or anything else, stop and report.

**Independent reviewer:**
> You have not seen how this code was written. Using only the repo at SHA <sha> and the runbook:
> 1. run the WP's tests;
> 2. recompute every number in its `PREDICTIONS.md` from the raw per-seed outputs with your own
>    short script;
> 3. check that the train, probe-train, probe-test and noise-record ids are disjoint;
> 4. check that no registered prediction, threshold, metric or cell changed after registration
>    (`git log -p`);
> 5. for WP2, re-derive P3–P5 on paper and reproduce §A.1.
>
> Write `docs/jepa/reviews/<WP>_REVIEW.md` with ACCEPT / ACCEPT WITH FIXES / REJECT and the
> evidence. Be adversarial.

### A.4 Changelog
- v1 (16 Oct): first draft.
- v2 (16 Oct): independent review applied:
  - fixed per-channel standardisation (per-row z-scoring leaked masked targets);
  - P2 restricted to per-view maps, with P5 joint whitening and an oracle arm added;
  - separate W_c/W_t;
  - γ_load as the knob vs population γ_n as the reported quantity; gaps between views;
  - R1/R4 at n = 1e5; unregularised CCA for R3;
  - overlay penalty (cannot be gamed by detecting "no pulse");
  - SIGReg on token embeddings; `sigreg_jepa` naming;
  - criterion M = ρ_θ,k − ρ_η^noise;
  - CRLB cross-check corrected;
  - E3: complex STFT, gap frames, ch0 excluded, R9 strengthened;
  - git: registration commit, fixed diff base, executors never commit, per-SHA clones;
  - compute: smaller primary config, L40S path.
