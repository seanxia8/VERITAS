# Runbook — Round 2 Full Diagnostics, Probe Repair & Objective/Training Repair (17 Nov 2026)

Date: 17 Nov 2026. Status: Authorised for immediate, unattended end-to-end execution.
Extension: `r2_repair_2026-11-17`, `registered: false`.
Parents:
- `docs/round2/ROOT_CAUSE_ANALYSIS_2026-11-17.md` (technical specification of all 13 root causes and fixes);
- `docs/round2/RUNBOOK_2026-11-17_R2_DIAGNOSTICS.md` (Part A & Part B diagnostics);
- `docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` and its batch/Titan/A100 addenda (`RUNBOOK_2026-11-12_R2_JUNO_TITAN.md`, `RUNBOOK_2026-11-13_R2_HTCONDOR_BATCH.md`, `RUNBOOK_2026-11-13_R2_STAGING_DWONG.md`, `RUNBOOK_2026-11-14_R2_BATCH_REOPEN.md`, `RUNBOOK_2026-11-14_R2_A100_EQUAL.md`).

Branch: `dev` in worktree `~/Documents/VERITAS-worktrees/r2` (remote clone `/work/dwong/r2_repo` on Deepthought and deepthought2).

---

## 0. Authority & Non-Stop Execution Mandate

Quote all three items verbatim in `docs/round2/DECISIONS.md` when starting Stage 0:

1. **Author, 17 Nov 2026, 02:52 CET:**
   > "不错，然后根据这个md写一个runbook，让一个能ssh进deepthought的agent去一步步做，可以用deepthought gpu也可以jobsubmission，要求agent一直做直到做完runbook，它做完我会让你来验收"

   *(Translation: Write a runbook based on `ROOT_CAUSE_ANALYSIS_2026-11-17.md` for an agent that can SSH into Deepthought to execute step-by-step; it may use Deepthought GPUs as well as HTCondor batch job submission; require the agent to work continuously until the entire runbook is finished, after which it will undergo acceptance review.)*
2. **Author, 17 Nov 2026, 01:26 CET (Diagnostics Part A & Part B):**
   Authorises event displays, error anatomy, simulation sanity & HK `2.07x` Gate V reconciliation, representations (PCA/t-SNE/UMAP/CKA), photon-bomb physics ceiling (ML fit + Cramér–Rao bound), JEPA bug checks, and GPU retraining with intermediate step snapshots (`0, 50, 100, 200, 500, 1k, 2k, 5k, 10k`).
3. **Standing Rule (Author, 30 Oct 2026):**
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

   **Strict Unattended Execution Rule:** Do **NOT** stop to ask questions, do **NOT** pause between stages for human confirmation, and do **NOT** exit early. Whenever an engineering, scheduling, memory, or numerical problem arises, choose the best technical solution automatically, record a one-line entry in `docs/round2/DECISIONS.md` and `$CC_STATE/notes/r2-repair.md`, and continue immediately until **all Stages (0 through 6)** are finished, committed, and pushed to `origin/dev`.

### 0.1 Safety & Immutability Constraints (Always Hold)

- **Main Checkout Protection:** Never switch, checkout, stash, reset, clean, add, or commit in `~/Documents/VERITAS` (which is on `pmt/r7-response-v3`). Before every `git commit`, verify `git rev-parse --show-toplevel` equals `/Users/wongdowling/Documents/VERITAS-worktrees/r2`.
- **Round-1 & Round-2 Data/Checkpoint Protection:** Never overwrite, delete, or modify:
  - Round-1 data under `/ceph/dwong/jepa/data/` or Round-1 checkpoints;
  - Round-2 HDF5 files under `/ceph/dwong/jepa/data/round2/<det>/<det>_r2.h5`;
  - Committed Round-2 checkpoints/evals under `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/runs/` or `results/round2/20261112_*`.
- **Where to Write:**
  - Repository outputs: `results/round2/20261117_repair/` (committed to `dev`).
  - Remote scratch/returns on Ceph: `/ceph/dwong/jepa/returns/r2_repair_2026-11-17/`.
  - Local scratch on Deepthought: `/work/dwong/r2_repair/`.
- **Compute Backends Authorised (`R2_repair_gpu` ledger line, cap 100 GPU-h):**
  1. **Deepthought CPU (`dt1`)**: Up to 28 cores total (`nice -n 10`, `TMPDIR=/work/dwong`), keeping at least 4 of the 40 server cores free.
  2. **Deepthought GTX Titans (`dt1`)**: Up to all 4 GTX Titan GPUs (12 GB each), 1 job per free GPU (no other user's process on that GPU), peak VRAM $\le 11\text{ GiB}$. Ideal for WCTE runs, probe MLP training, and differentiable renderer ML fits.
  3. **deepthought2 L40S (`dt2`)**: Up to 2 concurrent jobs via `scripts/jepa_run.sh` (or respecting the $\le 45\%$ VRAM per job and $\le 90\%$ total GPU memory sharing rule).
  4. **ETP HTCondor Batch (`dt1` $\to$ `condor_submit`)**: Using `scripts/r2_batch/stage_code.sh`, `scripts/r2_batch/submit.py`, and `scripts/r2_batch/sync.py` targeting **TOpAS A100-40GB** (`--gpu a100`) or **NEMO2 L40S** (`--gpu l40s`) or **NEMO2 H200** (`--gpu h200`), up to **12 concurrent batch jobs**, `accounting_group = cms`.

---

## 1. Stage 0 — Preflight, Worktree Sync & Infrastructure Check

### Step 0.1 — Synchronise Worktree & Remote Clones
1. In `~/Documents/VERITAS-worktrees/r2` (branch `dev`), pull latest `origin/dev` (`git fetch origin && git merge --ff-only origin/dev`).
2. Copy from `~/Documents/VERITAS/docs/round2/` into `~/Documents/VERITAS-worktrees/r2/docs/round2/`:
   - `ROOT_CAUSE_ANALYSIS_2026-11-17.md`
   - `RUNBOOK_2026-11-17_R2_DIAGNOSTICS.md`
   - `RUNBOOK_2026-11-17_R2_REPAIR.md`
3. Record their SHA256 hashes in `docs/round2/DECISIONS.md` along with the §0 authority quotes, commit them to `dev`, and push to `origin/dev`.
4. Sync `/work/dwong/r2_repo` on both `dt1` and `dt2` to match `origin/dev`.

### Step 0.2 — Initialise Driver State & Directories
1. Write `$CC_STATE/status/r2-repair.json`:
   ```json
   {"phase": "r2-repair", "state": "running", "started": "<UTC-ISO8601>"}
   ```
2. Create `$CC_STATE/notes/r2-repair.md` to track every running/completed sub-step, PID, HTCondor cluster ID, and log path.
3. Create local repo directories under `~/Documents/VERITAS-worktrees/r2/results/round2/20261117_repair/`:
   - `probes/`, `diagnostics/`, `gatev/`, `wcte_p2a/`, `retrain/`, `figures/event_displays/`, `figures/error_anatomy/`, `figures/latent/`, `figures/evolution/`.
4. On `dt1`, create `/ceph/dwong/jepa/returns/r2_repair_2026-11-17/{probes,diagnostics,gatev,wcte_p2a,runs,eval,logs}` and `/work/dwong/r2_repair/`.
5. Add ledger line `"R2_repair_gpu"` (`"cap_h": 100.0`, `"entries": []`, `"cpu_entries": []`) to `docs/round2/RESOURCE_LEDGER.json`.

### Step 0.3 — Check Compute Lanes & Data Availability
1. On `dt1`:
   - Check CPU load (`uptime`), Titan GPUs (`nvidia-smi`), disk & Ceph quotas (`df -h /work /ceph`, Ceph free entries $\ge 30,000$).
   - Check HTCondor queue and unclaimed GPUs:
     ```bash
     condor_q
     condor_status -pool c03-001-140.gridka.de -constraint 'PartitionableSlot && TotalSlotGPUs>0 && State =?= "Unclaimed"' -af:t Name CPUs MEMORY DISK GPUs
     ```
   - Confirm Round-2 HDF5 datasets exist at `/ceph/dwong/jepa/data/round2/{sk/sk_r2.h5,juno/juno_r2.h5,hk/hk_r2.h5}` and staged copies exist under `/ceph/srv/dwong/r2/`. Also check where the WCTE HDF5 or per-run datasets and Round-1/Round-2 checkpoints reside (`/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/` and `/ceph/dwong/jepa/`).
2. On `dt2`:
   - Check `nvidia-smi` and `uptime`.
3. Record available compute lanes in `$CC_STATE/notes/r2-repair.md`.

---

## 2. Stage 1 — Zero-Retrain Probe & Pooling Repair on Existing Checkpoints

**Goal:** Prove how much of the apparent $0.59\text{–}3.54\text{ m}$ vertex error on existing Round-2 (and WCTE) checkpoints comes purely from (a) unmasked all-PMT mean pooling (`pool_tokens` with `mask=None`), and (b) fitting a linear Ridge probe on translation-averaged features instead of using first-order spatial cross-moments (Panda Diplomacy / V2 Appendix D.1) or non-linear readout.

### Step 1.1 — Implement Repaired Pooling & Spatial-Moment Probes (`src/jepa_noise/r2_probe_repair.py`)
Do **not** alter `pool_tokens` in `src/jepa_noise/lucid_arms.py` or `_vertex_fit` in `src/jepa_noise/ext_r2.py` so existing Round-2 regression tests remain untouched. Create a new module `src/jepa_noise/r2_probe_repair.py` (and CLI hooks in `src/jepa_noise/ext_r2.py` or `scripts/jepa_r2_repair.py`) implementing:

1. **Four Token Pooling Modes** (given token embeddings $\mathbf{Z} \in \mathbb{R}^{B \times N \times D}$ and raw input tokens $\mathbf{X} \in \mathbb{R}^{B \times N \times 9}$ where column `6` is scaled `ln_q`, column `7` is scaled `time`, and column `8` is `hit` $\in \{0, 1\}$, optionally restricted to an active mask $M \in \{0, 1\}^{B \times N}$ for context-only vs full-input evaluation):
   - `all_mean`: Baseline unmasked mean over all PMTs in $M$:
     $$\bar{\mathbf{z}}_{\text{all}} = \frac{\sum_{i \in M} \mathbf{z}_i}{|M|}$$
   - `hit_mean`: Masked mean over **hit PMTs only** ($H = M \cap \{i : X_{i,8} > 0.5\}$; fall back to $M$ if $|H|=0$):
     $$\bar{\mathbf{z}}_{\text{hit}} = \frac{1}{\max(|H|, 1)} \sum_{i \in H} \mathbf{z}_i$$
   - `charge_weighted`: Charge-weighted mean over hit PMTs ($w_i = \exp(3.0 \cdot X_{i,6}) \cdot \mathbb{I}[i \in H]$, normalized per event):
     $$\bar{\mathbf{z}}_{q} = \sum_{i \in H} \tilde{w}_i \mathbf{z}_i, \quad \tilde{w}_i = \frac{w_i}{\sum_{j \in H} w_j + \varepsilon}$$
   - `early_time_weighted`: Prompt-wavefront weighted mean over hit PMTs. Let $t_{\min} = \min_{j \in H} X_{j,7}$ (or 5th percentile among hit PMTs to reject dark-noise preamble hits with $X_{j,7} < 0$). Define $u_i = \mathbb{I}[i \in H,\, X_{i,7} \ge 0] \cdot \exp\!\big(-4.0 \cdot (X_{i,7} - t_{\min,+})\big) \cdot w_i$, normalized per event:
     $$\bar{\mathbf{z}}_{t,q} = \sum_{i \in H} \tilde{u}_i \mathbf{z}_i$$

2. **Panda V2 Spatial-Moment Feature Construction (Appendix D.1 Adaptation)**:
   For each event, let $\tilde{\mathbf{p}}_i = X_{i, 0:3} \in \mathbb{R}^3$ be the normalized PMT coordinates and $\mathbf{d}_i = X_{i, 3:6} \in \mathbb{R}^3$ be the PMT orientation. Construct the event feature vector $\Phi_{\text{moment}}(\mathbf{Z}, \mathbf{X}) \in \mathbb{R}^{6D + 15}$:
   - $\bar{\mathbf{z}}_{\text{hit}} \in \mathbb{R}^D$ (0th-order hit mean);
   - $\bar{\mathbf{z}}_{q} \in \mathbb{R}^D$ (0th-order charge-weighted mean);
   - $\bar{\mathbf{z}}_{t,q} \in \mathbb{R}^D$ (0th-order early-time-and-charge-weighted mean);
   - $\text{vec}\!\left(\frac{1}{\max(|H|, 1)} \sum_{i \in H} \tilde{\mathbf{p}}_i \otimes (\mathbf{z}_i - \bar{\mathbf{z}}_{\text{hit}})\right) \in \mathbb{R}^{3D}$ (1st-order spatial cross-moment between PMT position and centered hit token embedding);
   - Explicit summary features $\mathbf{s}_{\text{geom}} \in \mathbb{R}^{15}$: hit fraction $|H|/N$, $\ln(\max(|H|, 1))$, mean hit charge, std hit charge, mean positive hit time, std positive hit time, $\bar{\mathbf{p}}_{\text{hit}} \in \mathbb{R}^3$, $\bar{\mathbf{p}}_{q} \in \mathbb{R}^3$, and $\bar{\mathbf{p}}_{t,q} \in \mathbb{R}^3$.
   - *(Also evaluate `moment_latent_only` $\in \mathbb{R}^{6D}$, which excludes $\mathbf{s}_{\text{geom}}$, so the contribution of the learned latent cross-moments alone is isolated from $\mathbf{s}_{\text{geom}}$!)*

3. **Three Vertex Probe Estimators** (trained on photon-bomb events of `dev_probe_train` $\approx 3,340$ events, evaluated on photon-bomb events of `dev_probe_test` $\approx 3,340$ events):
   - `ridge_all_mean` (Round-2 baseline: Ridge with 5-fold run-grouped CV on $\bar{\mathbf{z}}_{\text{all}} \in \mathbb{R}^D$);
   - `ridge_hit_mean` (Ridge on $[\bar{\mathbf{z}}_{\text{hit}}, \bar{\mathbf{z}}_q, \bar{\mathbf{z}}_{t,q}] \in \mathbb{R}^{3D}$);
   - `ridge_spatial_moment_latent` (PCA/SVD or standardized Ridge with 5-fold run-grouped CV on the $6D$ latent spatial-moment vector);
   - `ridge_spatial_moment_full` (Standardized Ridge with 5-fold run-grouped CV on $\Phi_{\text{moment}} \in \mathbb{R}^{6D+15}$);
   - `mlp_spatial_moment` (2-layer MLP `d_in -> 256 -> 64 -> 3` with LayerNorm, GELU, dropout $0.1$, AdamW $\text{lr}=10^{-3}$, weight decay $10^{-2}$, cosine schedule for 200 epochs on `dev_probe_train` with 10% run-grouped validation early stopping).

4. **Post-LayerNorm & Masked-Target Controls for `token_log_charge_r2` (Root Cause #3)**:
   - Because `PointEncoder` has `final_norm = None` (`src/jepa_noise/point_encoder.py:295, 301`), unmasked tokens pass input column 6 (`ln_q`) linearly through the residual stream (`init` gets $R^2 = 0.999$).
   - Compute and report three versions of `token_log_charge_r2` and `token_time_r2` (new: ridge probe on hit token arrival time $t_i$!):
     1. `raw_unmasked_r2`: Baseline on unmasked `full_input` hit tokens;
     2. `post_ln_unmasked_r2`: After parameter-free `F.layer_norm(z, (D,))`, plus its delta over `init` ($\Delta R^2 = R^2_{\text{arm}} - R^2_{\text{init}}$);
     3. `masked_target_pred_r2`: Probe fit on the **predicted target embeddings** (`arm.predict_targets(batch)`) for masked hit target tokens where the encoder never saw the target token's charge or time!

### Step 1.2 — Unit Tests for Stage 1
Create `src/jepa_noise/tests/test_r2_probe_repair.py`:
- Verify `all_mean` reproduces `pool_tokens(z, None)` bit-for-bit (`atol=1e-7`).
- Verify `hit_mean` ignores unhit tokens completely (modifying unhit token embeddings changes `hit_mean` by $0.0$).
- Verify on synthetic spherical wavefront data that `ridge_spatial_moment_latent` achieves $>5\times$ lower vertex error than `ridge_all_mean`.
- Run `pytest` on `dt1`.

### Step 1.3 — Execute Stage 1 Across All Existing Checkpoints
Run the Stage-1 evaluation on `dt1` (Titans/CPU) or `dt2` across all existing checkpoints:
- Detectors: **WCTE** (Round-1 checkpoints), **SK**, **JUNO**, **HK** (Round-2 checkpoints from `/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/`).
- Arms: `init`, `mae_mse`, `mae_mse_rf` (where present), `sigreg_jepa`, `target_posfree` (seeds 9000 and 9001).
- Save complete JSON results to `results/round2/20261117_repair/probes/probe_repair_results.json` and summary Markdown table `results/round2/20261117_repair/probes/PROBE_REPAIR_TABLE.md` (reporting both mean 3D error in meters and normalized error $\div R_{\text{fid}}$, with $R_{\text{fid}} = 1.262\text{ m}$ for WCTE, $13.570\text{ m}$ for SK, $14.000\text{ m}$ for JUNO, and $25.944\text{ m}$ for HK).
- Commit and push to `origin/dev`.

---

## 3. Stage 2 — Physics Ceiling, Gate V Reconciliation, Event Displays & Latent Diagnostics

**Goal:** Execute all of `RUNBOOK_2026-11-17_R2_DIAGNOSTICS.md` Part A plus the code-level reconciliation of HK Gate V (`2.07x`) and dark-noise trigger shift quantification.

### Step 2.1 — Photon-Bomb Physics Ceiling & Dark-Noise Trigger Impact (`src/jepa_noise/abs_render.py`)
1. Create `src/jepa_noise/abs_render.py` implementing the differentiable forward Cherenkov photon-bomb model in PyTorch for $\theta = (x, y, z, \ln Y, t_0)$:
   - Given PMT positions $\mathbf{p}_i \in \mathbb{R}^3$, normals $\mathbf{d}_i \in \mathbb{R}^3$, photocathode radius $r_{\text{PMT}}$, wavelength-averaged $\overline{\text{QE}} \cdot \text{CE}$, attenuation length $\lambda_0 = 50.0\text{ m}$, and group index $n_g = 1.33$:
     $$d_i(\mathbf{x}) = \|\mathbf{p}_i - \mathbf{x}\|_2, \quad \cos\alpha_i(\mathbf{x}) = \max\!\left(0,\, -\frac{(\mathbf{p}_i - \mathbf{x}) \cdot \mathbf{d}_i}{d_i(\mathbf{x})}\right)$$
     $$\mu_i(\mathbf{x}, Y) = Y \cdot \frac{\pi r_{\text{PMT}}^2 \cos\alpha_i(\mathbf{x})}{4\pi d_i(\mathbf{x})^2} \exp\!\left(-\frac{d_i(\mathbf{x})}{\lambda_0}\right) \cdot \overline{\text{QE}} \cdot \text{CE}$$
     $$t^{\text{pred}}_i(\mathbf{x}, t_0) = t_0 + \frac{n_g d_i(\mathbf{x})}{c} + \Delta t_{\text{walk}}\!\big(\mu_i(\mathbf{x}, Y)\big)$$
2. **Closure Test**: On 500 `dev_probe_test` photon-bomb events per detector (WCTE, SK, JUNO, HK), evaluate $\mu_i(\theta_{\text{true}})$ against the stored `expected_pe` (or generator truth) and verify closure residuals match Poisson + SPE expectations. Record in `results/round2/20261117_repair/diagnostics/physics_ceiling.json`.
3. **Per-Event Maximum Likelihood (ML) Reconstruction & Cramér–Rao Lower Bound (CRLB)**:
   - On 500 test photon-bomb events per detector (WCTE, SK, JUNO, HK), run multi-start L-BFGS / Adam optimization (initializing from the charge-and-early-time centroid plus 4 random perturbations) for three likelihood variants:
     1. **ToF-Only Fit**: Robust Huber / truncated Gaussian loss on hit arrival times $t_i$ (filtering out dark-noise preamble hits with $t_i < 0$ or $|t_i - t^{\text{pred}}_i| > 4\sigma_t$);
     2. **Charge-Only Poisson ML Fit**: Minimizing $\sum_{i=1}^N \big[\mu_i(\mathbf{x}, Y) - q_i \ln(\mu_i(\mathbf{x}, Y) + \mu_{\text{dark}})\big]$;
     3. **Joint Charge + Time ML Fit**: Combining Poisson charge likelihood, hit/no-hit Bernoulli likelihood, and robust first-hit timing likelihood.
   - Compute the **3D Cramér–Rao Lower Bound (CRLB)** $\sqrt{\text{Tr}\big(\mathcal{I}(\theta_{\text{true}})^{-1}_{x,y,z}\big)}$ from the exact Fisher Information Matrix of the joint Poisson-charge + Gaussian-time observation model at the true vertex $\theta_{\text{true}}$.
   - **HK Dark-Noise Trigger Shift Analysis (Root Cause #10)**: Separate the 500 events by whether `dark_moved` occurred (trigger time shifted earlier by dark noise, which happens in $27.8\%$ of HK events). Report ML and probe vertex errors separately on `dark_moved == False` vs `dark_moved == True`.
   - Plot vertex error and CRLB vs $\ln Y$ and vs distance to nearest wall (`results/round2/20261117_repair/figures/physics_ceiling_vs_yield.png`).

### Step 2.2 — HK Gate V `2.07x` Discrepancy Reconciliation (`src/jepa_noise/cherenkov_validation.py`)
1. In `src/jepa_noise/cherenkov_validation.py`, extend `AnalyticConfig` with two backward-compatible fields (defaulting to Round-1 values so all existing callers and tests stay bit-identical):
   - `attenuation_m: float | None = 50.0` (when `None` or `<= 0`, no bulk $\exp(-r/\lambda_0)$ factor is applied, matching `PromptTransport(absorption='off')`);
   - `angular_mode: str = "cos"` (`"cos"` uses flat-disc $\max(\cos\theta, 0)$; `"spherical"` uses isotropic spherical cross-section $1.0$ for all PMTs facing the interior or within acceptance, matching `PromptTransport._first_sphere_hit`).
2. Write a diagnostic script that evaluates the 10 Gate V holdout events (`e500_010..019`) on **WCTE, SK, JUNO, and HK** under the $2 \times 2$ ablation grid:
   - `(attenuation_m=50.0, angular_mode="cos")` — Original Gate V (`HK ratio = 2.073`, `SK = 1.192`, `JUNO = 1.084`);
   - `(attenuation_m=None, angular_mode="cos")` — Matching `PromptTransport(absorption='off')`;
   - `(attenuation_m=50.0, angular_mode="spherical")` — Matching spherical PMT intersection;
   - `(attenuation_m=None, angular_mode="spherical")` — Matching both `PromptTransport` assumptions.
3. Also perform the **Diagnostics A3** checks:
   - Tabulate every detector's simulation parameters (PMT radius, active PMT count, photocoverage $\%$, QE, CE, $\lambda_0$, angular acceptance, prompt window, `time_scale_ns`, `nhits_n`, dark noise rate).
   - Plot per-PMT $\mu_{\text{LUCiD}} / \mu_{\text{analytic}}$ vs distance $d$, incidence angle $\theta$, and $z$ for HK and SK (`results/round2/20261117_repair/figures/gatev_hk_sk_ratio.png`).
   - Plot total photoelectrons and hit fraction vs $Y$ (log-log) and timing residual distribution $t - t_0 - d \cdot n_g / c$ across all 4 detectors (`results/round2/20261117_repair/figures/sim_sanity_summary.png`).
4. Save `results/round2/20261117_repair/gatev/gatev_reconciliation.json` and `GATEV_RECONCILIATION.md`.

### Step 2.3 — Event Displays & Error Anatomy (Diagnostics A1 & A2)
1. **Event Displays (`results/round2/20261117_repair/figures/event_displays/`)**:
   - For **SK, HK, and WCTE**: unrolled cylinder barrel in $(\phi, z)$ with top and bottom caps drawn as discs above and below.
   - For **JUNO**: Mollweide projection of the sphere.
   - **3 panels per event**:
     1. $\ln q$ per hit PMT;
     2. First-hit time $t$ per hit PMT (marking early dark-noise hits $t < 0$ in a distinct marker/color);
     3. Orthographic $(x, y)$ and $(r, z)$ tank cross-sections showing: True vertex $\mathbf{x}_{\text{true}}$, Physics Ceiling ML fit $\mathbf{x}_{\text{ML}}$, Baseline Ridge probe (`all_mean`) for `mae_mse`, `init`, `sigreg_jepa`, `target_posfree`, and Stage-1 Repaired Spatial-Moment probe (`mae_mse` and `target_posfree`).
   - **Events per detector** (15 events each, or unified grid figures):
     - 6 events with largest baseline MAE vertex error;
     - 3 events near baseline MAE median error;
     - 6 events with largest `sigreg_jepa` vertex error.
2. **Error Anatomy (`results/round2/20261117_repair/figures/error_anatomy/`)**:
   - For each detector (WCTE, SK, JUNO, HK), plot binned median vertex error (comparing Baseline `all_mean` Ridge, Repaired Spatial-Moment Probe, and Physics Ceiling ML fit) against:
     1. True radius $r$;
     2. $|z|$;
     3. Distance to nearest wall $d_{\text{wall}}$;
     4. $\ln Y$;
     5. `run_id` (gain drift).

### Step 2.4 — Static Latent Maps (PCA, t-SNE, UMAP, CKA) & Per-PMT Variance (Diagnostics A4 + Panda V1 Check)
1. On 2,000 test photon-bomb events per detector (seed 9000 checkpoints for `init`, `mae_mse`, `mae_mse_rf`, `sigreg_jepa`, `target_posfree`):
   - Compute event embeddings under both `all_mean` and `hit_mean` pooling.
   - Plot 2D **PCA, t-SNE, and UMAP** maps (fixed random seed `42`, identical perplexity `30` and `n_neighbors=15`), colored by true $r$, $z$, $\ln Y$, and `run_id` (`results/round2/20261117_repair/figures/latent/`).
   - Plot **token-level 2D t-SNE/UMAP** for 1 representative event per detector colored by (a) PMT spatial azimuth/height $(\phi, z)$ and (b) PMT hit log-charge $\ln q$.
   - Compute the **Linear CKA similarity matrix** across all arms per detector.
   - **Panda V1 Per-PMT Across-Event Variance Diagnostic**: For 200 test events per detector, track each individual PMT $i$'s token embedding $\mathbf{z}_{e,i}$ across events $e$, decompose total variance into static layout variance $\text{Var}_i(\mathbb{E}_e[\mathbf{z}_{e,i}])$, across-event hit-vs-unhit variance, and across-event charge/time modulation variance within hit events. Save in `results/round2/20261117_repair/diagnostics/latent_variance_cka.json`.
2. Commit all Stage 2 code, figures, and JSONs to `dev` and push to `origin/dev`.

---

## 4. Stage 3 — WCTE Round-1 P2a `R3u` Ceiling Bug Fix (Root Cause #13)

**Goal:** Resolve the `S4` stop from `results/jepa/20261111_p2a/EXECUTION.json` where `R2` (`[z_c, z_a - z_b]`, 256-D) beat the supposed ceiling `R3u` (`[z_c, u_c]`, 233-D) by $5.7\text{–}9.3\text{ mm}$ on 4 drift cells of `fix9001` because `R3u` applied a single shared ridge hyperparameter $\alpha$ across `z_c` (128-D) and `u_c` (105 dead-module indicator columns).

### Step 3.1 — Implement Split-Regularized / Orthogonalized `R3u_split`
1. Create `src/jepa_noise/p2a_ceiling_repair.py` (with unit tests in `src/jepa_noise/tests/test_p2a_ceiling_repair.py`):
   - Given context embedding $\mathbf{Z}_c \in \mathbb{R}^{n \times 128}$ and dead-module indicator matrix $\mathbf{U}_c \in \mathbb{R}^{n \times 105}$:
   - Implement **Two-Stage Split-Penalty Ridge (`R3u_split`)**:
     1. Fit `R0` on $\mathbf{Z}_c$ alone using 5-fold run-grouped CV over the standard $\alpha_z$ grid to obtain optimal $\alpha_z^*$ and out-of-fold / test predictions $\hat{\mathbf{Y}}_{z}$.
     2. Orthogonalize $\mathbf{U}_c$ against $\mathbf{Z}_c$ (or fit a second-stage Ridge regression on the residual $\mathbf{R} = \mathbf{Y} - \hat{\mathbf{Y}}_{z}$ using $[\mathbf{Z}_c, \mathbf{U}_c]$ with separate block scaling $\mathbf{U}_c / \sqrt{\gamma}$, tuning $(\alpha_z, \gamma) \in \mathcal{A}_z \times \{0.01, 0.1, 1.0, 10.0, 100.0, \infty\}$ via 5-fold run-grouped CV). Because $\gamma = \infty$ recovers `R0` ($\mathbf{Z}_c$ only) exactly, CV is mathematically guaranteed never to degrade when $\mathbf{U}_c$ is uninformative for event vertex!
   - Also implement `R3u_full_ceiling` which includes `[z_c, z_a - z_b, u_c]` with block-tuned regularization so `R3u` is a true superset of `R2`.
2. Re-evaluate the WCTE P2a probe features/checkpoints (`fix9000`, `fix9001`, `pf9000`, `pf9001`, `mae9000`, `mae9001`) across all $3 \times 3$ drift cells (`N`, `D1_s015`, `D1_s030` $\times$ `N`, `D2_j25`, `D2_j40`).
3. Verify that `R2` never beats `R3u_split` or `R3u_full_ceiling` beyond the $5\text{ mm}$ tolerance on any cell (`0 / 54` violations), resolving `S4`.
4. Save `results/round2/20261117_repair/wcte_p2a/p2a_repaired_readings.json` and `WCTE_P2A_REPAIR.md`, commit, and push.

---

## 5. Stage 4 — Code Fixes for Training Loop, MAE Time Loss & JEPA Architecture

**Goal:** Fix Root Causes #4, #5, #6a, #6b, #7, and #8 in the codebase cleanly via opt-in config parameters so all Round-1 and Round-2 regression tests continue to pass while enabling the repaired arms (`mae_repaired` and `jepa_repaired`) and the Diagnostics Part B runs.

### Step 4.1 — Fix `R2TrainingLoader` Run Ordering & Full-Batch `SIGReg` (Root Cause #7)
1. **In `src/jepa_noise/r2_store.py` (`R2TrainingLoader.batch_at`)**:
   - Currently, `batch_at(step)` draws 8 runs and appends 4 events per run in contiguous run order (`[r0, r0, r0, r0, r1, r1, r1, r1, ...]`), so `micro_batch = 4` on SK/HK slices 4 events from a single run!
   - Add parameter `shuffle_events_within_batch: bool = False` to `R2TrainingLoader.__init__`. When `True`, after gathering the $B=32$ events in `batch_at(step)`, apply a deterministic permutation `perm = rng.permutation(len(tokens))` (seeded by `self.seed + step + 7919`) to `tokens`, `event_ids`, `run_ids`, and `domains`.
2. **In `src/jepa_noise/lucid_run.py` (`train_loop`)**:
   - Add two improvements to `train_loop`:
     - **Snapshot Saving (`snapshot_steps: tuple[int, ...] = ()`)**: Whenever `step in snapshot_steps` (including `step == 0` before the first optimizer step!), save an immutable snapshot `run_dir / f"snapshot_step_{step:05d}.pt"` containing model state, step, and current metrics, without being overwritten by subsequent periodic checkpoints.
     - **Exact Full-Batch `SIGReg` under Micro-Batching (`full_batch_sigreg: bool = False`)**:
       When `micro_batch < B` and `full_batch_sigreg=True` on a JEPA arm:
       Two-pass or cached-embedding backward so `EppsPulley` is evaluated on the **full $B=32$ batch** rather than on $m=4$ slices:
       - Pass 1 (under `torch.no_grad()`): Run forward on all micro-batches $k = 1 \dots K$ to collect the full-batch context and target embeddings $\mathbf{Z}_{\text{ctx}}^{\text{full}}, \mathbf{Z}_{\text{tgt}}^{\text{full}}$ across all 32 events.
       - Pass 2 (micro-batch backward): In micro-batch $k$, compute the local prediction MSE loss on slice $k$, and compute the `SIGReg` (`EppsPulley`) loss by substituting the active slice $k$ (with gradients) into the detached full-batch tensor $\mathbf{Z}^{\text{full}}$ (multiplied by $K$ so the average over $K$ micro-batches equals the exact gradient of $\mathcal{L}_{\text{SIGReg}}(\mathbf{Z}^{\text{full}})$!).
       - Verify with a unit test that the gradient of this two-pass micro-batch accumulation at `micro_batch=4` equals the single-pass `micro_batch=32` gradient to within float32 precision (`atol=1e-5`)!

### Step 4.2 — Fix `SIGReg` Flattening & Add Event-Level `SIGReg` (Root Cause #8)
1. In `src/jepa_noise/lucid_arms.py`, add support for `sigreg_mode`:
   - `"legacy_flat"` (Round-1/Round-2 default);
   - `"event_and_hit_token"`:
     - Computes `EppsPulley` across the **batch dimension $B=32$** on the **hit-pooled event embeddings** $\bar{\mathbf{z}}_{\text{ctx, hit}} \in \mathbb{R}^{B \times D}$ and $\bar{\mathbf{z}}_{\text{tgt, hit}} \in \mathbb{R}^{B \times D}$ (forcing embeddings to vary across **events**, because static PMT layout averages out per event and cannot pass a batch-level Gaussianity test!);
     - Plus a per-token variance floor / hit-token `EppsPulley` computed **only on hit tokens** (subsampled to at most 256 hit tokens per batch so `EppsPulley` cost and effective scale do not blow up with $N_{\text{PMT}}$).

### Step 4.3 — Fix MAE Time Loss Weighting & Spatial Patch Masking (Root Causes #4 & #5)
1. **Balanced MAE Time Loss (`src/jepa_noise/lucid_arms.py`)**:
   - In `MAEMSEArm`, add config parameter `time_loss_weight: float = 1.0`.
   - For `mae_repaired`, set `time_loss_weight = (time_scale_ns / 50.0) ** 2` (or normalize hit target times inside the MAE loss by dividing by `50.0 / time_scale_ns`, i.e. measuring time residuals in units of $50\text{ ns}$ rather than $450\text{–}767\text{ ns}$). This restores the MAE time loss share from $0.04\%$ to $\sim 25\text{–}35\%$ on SK, JUNO, and HK!
2. **3D Spatial / Hilbert Contiguous Patch Masking (`src/jepa_noise/lucid_views.py`, PoLAr-MAE Adaptation)**:
   - Alongside `view_mode = "within"` (which interleaves odd/even slots inside every 19-PMT module), add `view_mode = "spatial_patch"`:
   - Partition the Hilbert-ordered PMTs into contiguous spatial patches of size $P = \max(32, \lfloor N_{\text{PMT}} / 64 \rfloor)$ (so each detector is divided into 64 contiguous spatial patches on the cylinder/sphere surface).
   - For each event (seeded deterministically by `event_id`), randomly assign 50% of patches (32 patches) to `context` and 50% (32 patches) to `target`. This eliminates trivial nearest-neighbor interpolation inside a module and forces long-range Cherenkov ring completion.

### Step 4.4 — Implement `jepa_repaired` & `sigreg_jepa_all_posfree` (Root Causes #6a, #6b + Diagnostics B2)
1. **Diagnostics B2 Control Arm (`sigreg_jepa_all_posfree`)**:
   - Identical to `sigreg_jepa`, except columns `0..5` (positions and directions) are zeroed in **every** pass (`context`, `label`, and `full` evaluation pass).
2. **Repaired JEPA Arm (`jepa_repaired`)**:
   - Combines the literature-backed fixes from Panda V1, Panda Diplomacy (V2), and I-JEPA:
     1. **EMA Target Encoder with `stop_grad`**: Target encoder is a deep copy updated via Exponential Moving Average ($\tau$ cosine-scheduled from $0.992$ to $0.9995$ over 10,000 steps) under `torch.no_grad()`.
     2. **Position-Free Target Pass + Hit-Only Target Prediction Loss (Panda V2 Observable Distillation principle)**:
        - The EMA target encoder receives `zero_geometry_columns(batch.tokens)` on the target view so target representations depend purely on hit observables $(q_i, t_i, \text{hit}_i)$ and local hit patterns.
        - Crucially, the predictor loss is computed **only on hit target tokens** (`hit_target_mask = batch.target & (batch.tokens[:, :, 8] > 0.5)`), plus a binary cross-entropy / MSE hit-occupancy head on all target tokens (`hit_pred = self.hit_head(pred_z)` predicting `batch.tokens[:, :, 8]`). Because unhit tokens are excluded from the latent vector MSE, the representation never collapses to the all-zero unhit token vector!
     3. **Auxiliary Observable Distillation Head on Hit Targets**:
        - A lightweight linear head `obs_head: Linear(D, 2)` on the predicted hit-target embeddings predicts normalized `(ln_q, time_50ns)` with weight $0.25$, anchoring the latent space to retain both charge and arrival-time precision.
     4. **Event-Level Full-Batch `SIGReg`**: Uses `sigreg_mode = "event_and_hit_token"` with `full_batch_sigreg = True` and `shuffle_events_within_batch = True`.
     5. **Consistent Evaluation Readout**: Because the target encoder was trained on position-free inputs while the online context encoder was trained WITH geometry columns `0..5` (and predicts hit target embeddings from context), `jepa_repaired.encode_tokens` uses the **online context encoder** with `F.layer_norm` applied to the output tokens, and Stage-1 `hit_mean` / `spatial_moment` pooling!
3. **Repaired MAE Arm (`mae_repaired`)**:
   - Uses `EncoderConfig(window=W_rf, shift=W_rf//2)` ($64/32$ on WCTE, $384/192$ on SK, $320/160$ on JUNO, $640/320$ on HK), `time_loss_weight` scaled to $50\text{ ns}$ units, `view_mode = "spatial_patch"`, and `shuffle_events_within_batch = True`.

### Step 4.5 — Unit Tests for Stage 4 (`src/jepa_noise/tests/test_r2_repair.py`)
Write comprehensive unit tests verifying:
1. `R2TrainingLoader(..., shuffle_events_within_batch=True)` contains all 8 runs interleaved across every 4-event micro-batch slice (more than 1 unique run per 4-event slice).
2. `full_batch_sigreg=True` at `micro_batch=4` produces the exact same `SIGReg` loss and parameter gradients (`atol=1e-5`) as `micro_batch=32`.
3. `jepa_repaired` target encoder has zero gradient (`p.grad is None` for all EMA target parameters) and updates via EMA after each step.
4. `jepa_repaired` latent prediction loss ignores unhit target tokens (perturbing unhit target embeddings changes latent MSE by $0.0$).
5. `mae_repaired` initial time-loss share on SK/JUNO/HK synthetic/real batches is $\ge 15\%$ (compared to $<0.5\%$ in baseline `mae_mse`).
6. All existing Round-1 and Round-2 unit tests pass without modification.
Commit Stage 4 and push to `origin/dev`.

---

## 6. Stage 5 — GPU Retraining, Latent Evolution & Multi-Detector Verification

**Goal:** Run the GPU experiments (Diagnostics Part B1/B2 + Repaired MAE & Repaired JEPA across WCTE, SK, JUNO, and HK) using **Deepthought GPUs (`dt1` 4× GTX Titans, `dt2` L40S)** and **ETP HTCondor Batch (`scripts/r2_batch/submit.py` on TOpAS A100 / NEMO2 L40S)**.

### Step 5.1 — Multi-Backend GPU Scheduling Strategy
To finish as fast and reliably as possible without blocking on a single queue:
1. **Stage updated code to `/ceph/srv/dwong/r2/`**:
   On `dt1`, after syncing `/work/dwong/r2_repo` to the Stage-4 commit, run:
   ```bash
   bash /work/dwong/r2_repo/scripts/r2_batch/stage_code.sh
   ```
   Confirm the new `code/r2_<sha8>.tar.gz` is listed in `/ceph/srv/dwong/r2/SHA256SUMS`.
2. **Distribute jobs across all three GPU backends in parallel**:
   - **Deepthought 4× GTX Titans (`dt1`)**: Run all **WCTE** training jobs (`N_PMT = 1,995`, very fast: $\sim 15\text{–}25\text{ min}$ per 10k-step run on a Titan!) across the 4 Titans in parallel. If Titans finish WCTE while larger detectors are still running, use Titans for JUNO runs (`micro_batch=4` or `8`) or probe evaluations.
   - **deepthought2 L40S (`dt2`)**: Run up to **2 concurrent SK or HK jobs** via `scripts/jepa_run.sh` (each $\le 45\%$ VRAM).
   - **ETP HTCondor Batch (`dt1` $\to$ TOpAS A100 / NEMO2 L40S)**: Submit **SK, JUNO, and HK** training jobs via `scripts/r2_batch/submit.py` (`--gpu a100` or `--gpu l40s`, up to 12 jobs at once).
   - **Failover / No-Wait Rule**: Monitor `condor_q` every 15 minutes alongside `dt1` and `dt2` jobs. If a batch job sits idle in `condor_q` while a lane opens up on `dt2` L40S or `dt1` Titans, launch it on the free `dt2`/`dt1` lane and `condor_rm` the duplicate idle batch job so no wall time is wasted waiting!

### Step 5.2 — Experiment Matrix to Run (10,000 Steps Each, Seed 9000)

#### Group 5A — Latent Evolution with 9 Snapshots (`steps = 0, 50, 100, 200, 500, 1000, 2000, 5000, 10000`)
Save all 9 snapshot checkpoints per run (`snapshot_step_00000.pt` ... `snapshot_step_10000.pt`):
1. **WCTE** (6 runs, seed 9000 — run on `dt1` 4× Titans or `dt2` L40S):
   - `wcte__sigreg_jepa__s9000` (Diagnostics B1 baseline fixed-label JEPA)
   - `wcte__target_posfree__s9000` (Diagnostics B1 baseline posfree JEPA)
   - `wcte__mae_mse__s9000` (Diagnostics B1 baseline MAE)
   - `wcte__sigreg_jepa_all_posfree__s9000` (Diagnostics B2 known-answer control: 6 geometry columns zeroed in all passes)
   - `wcte__jepa_repaired__s9000` (Stage-4 repaired JEPA)
   - `wcte__mae_repaired__s9000` (Stage-4 repaired MAE)
2. **SK** (4 runs with 9 snapshots each, seed 9000 — run on `dt2` L40S + Batch A100/L40S):
   - `sk__sigreg_jepa__s9000` (Diagnostics B1 baseline fixed-label evolution)
   - `sk__target_posfree__s9000` (Diagnostics B1 baseline posfree evolution)
   - `sk__jepa_repaired__s9000` (Stage-4 repaired JEPA on SK, $W=384$)
   - `sk__mae_repaired__s9000` (Stage-4 repaired MAE on SK, $W=384$)

#### Group 5B — Multi-Detector Repaired Arms on JUNO & HK (Seed 9000, 10,000 Steps, Final + Key Snapshots)
3. **JUNO** (2 runs, seed 9000, $W=320$ — run on Batch A100/L40S or `dt1` Titans / `dt2` L40S):
   - `juno__jepa_repaired__s9000`
   - `juno__mae_repaired__s9000`
4. **HK** (2 runs, seed 9000, $W=640$ — run on Batch A100/L40S or `dt2` L40S):
   - `hk__jepa_repaired__s9000`
   - `hk__mae_repaired__s9000`

*(Budget & Wall-Time Fallback Rule: Total GPU budget for Group 5A + 5B is $\approx 30\text{–}45\text{ GPU-h}$, well inside the $100\text{ GPU-h}$ `R2_repair_gpu` cap. If HK or SK step times with $W_{\text{rf}}$ on an occupied server threaten to exceed a 6-hour wall window, `jepa_repaired` on SK/HK may use $W=128$ or $W=256$ while `mae_repaired` uses $W_{\text{rf}}$, recording the choice in `DECISIONS.md`.)*

### Step 5.3 — Evaluate All Retrained Runs & Generate Latent Evolution Plots
1. For every completed run in Group 5A and 5B, run both:
   - The standard Round-2 evaluation suite (`vertex` on `all_mean`, `light_yield`, `token_log_charge_r2`, `f_layout`, `f_event`, `effective_rank`, `module_gain_r2`, `mu_e_auc`, `direction_deg`, `deposited_energy_r2`, and MAE loss shares);
   - The **Stage-1 Repaired Probe suite** (`hit_mean` Ridge, `spatial_moment` Ridge & MLP vertex probes, post-LayerNorm `token_log_charge_r2`, `masked_target_pred_r2`, and `token_time_r2`).
2. **Latent Evolution Analysis (Diagnostics B1)**:
   - On WCTE (all 6 arms) and SK (all 4 arms), evaluate the 9 snapshots (`0, 50, 100, 200, 500, 1k, 2k, 5k, 10k`) on 1,000 probe events:
   - Plot curves over training steps (`results/round2/20261117_repair/figures/evolution/metrics_vs_step_{wcte,sk}.png`):
     1. $f_{\text{event}}(\text{step})$ and $f_{\text{layout}}(\text{step})$;
     2. Event-mean `effective_rank(step)`;
     3. Vertex error (`all_mean` Ridge and `spatial_moment` Ridge) vs step;
     4. Training prediction loss, `SIGReg` loss, and MAE time loss share vs step.
   - Plot the **9-step UMAP/t-SNE Evolution Grid** (`results/round2/20261117_repair/figures/evolution/umap_evolution_{wcte,sk}.png`) colored by true vertex radius $r$ and $\ln Y$, showing side-by-side how `sigreg_jepa` collapses, how `target_posfree` compresses unhit tokens, and how `mae_repaired` and `jepa_repaired` organize the manifold over training.
3. Update `docs/round2/RESOURCE_LEDGER.json` with every GPU run's exact hours, hardware model, and host/cluster ID.

---

## 7. Stage 6 — Final Synthesis Report & Acceptance Package

When Stages 1–5 are complete, assemble the final deliverables for acceptance review:

1. **Master Report (`results/round2/20261117_repair/REPAIR_AND_DIAGNOSTICS_REPORT.md` and copy at `$CC_STATE/reports/r2-repair.md`)**:
   Include all quantitative tables and embedded/linked figures:
   - **Section 1 — Executive Summary & 12-Line English Summary for Junjie** (answering Q1–Q4 with exact numbers);
   - **Section 2 — Photon-Bomb Physics Ceiling vs. Baseline Probes vs. Repaired Spatial-Moment Probes vs. Repaired Retrained Models** (complete cross-detector table for WCTE, SK, JUNO, HK in meters and $\div R_{\text{fid}}$, plus CRLB and ML fit);
   - **Section 3 — Simulation Sanity, HK Gate V `2.07x` Reconciliation & Dark-Noise Trigger Analysis** ($2 \times 2$ Gate V table showing `2.07x -> 0.90x/1.15x`, detector parameter comparison table, and `dark_moved` impact on HK);
   - **Section 4 — Event Displays & Error Anatomy Findings** (summary of where worst errors occur and visual comparison of reconstructed vertices);
   - **Section 5 — JEPA Collapse Mechanism, `micro_batch` Bug Verification & Latent Evolution (`0 -> 10k` steps)** (comparing `sigreg_jepa`, `target_posfree`, `sigreg_jepa_all_posfree`, `mae_mse`, `mae_repaired`, and `jepa_repaired` across $f_{\text{event}}$, effective rank, vertex, and particle tasks);
   - **Section 6 — WCTE Round-1 P2a `R3u_split` Repair Table** (showing all 9 drift cells and confirming `S4` is resolved).

2. **Machine-Readable Acceptance Manifest (`docs/round2/ACCEPTANCE_2026-11-17_R2_REPAIR.json` and copy in `results/round2/20261117_repair/ACCEPTANCE_2026-11-17_R2_REPAIR.json`)**:
   Must contain:
   ```json
   {
     "runbook": "docs/round2/RUNBOOK_2026-11-17_R2_REPAIR.md",
     "root_cause_doc": "docs/round2/ROOT_CAUSE_ANALYSIS_2026-11-17.md",
     "verdict": "READY_FOR_ACCEPTANCE",
     "stages_completed": ["stage0_preflight", "stage1_probe_repair", "stage2_diagnostics_ceiling_gatev", "stage3_wcte_p2a_repair", "stage4_code_fixes", "stage5_gpu_retrain_evolution", "stage6_report"],
     "pytest_summary": {"passed": "<N>", "failed": 0},
     "deliverables": {
       "probe_repair_json": "results/round2/20261117_repair/probes/probe_repair_results.json",
       "physics_ceiling_json": "results/round2/20261117_repair/diagnostics/physics_ceiling.json",
       "gatev_reconciliation_json": "results/round2/20261117_repair/gatev/gatev_reconciliation.json",
       "latent_variance_cka_json": "results/round2/20261117_repair/diagnostics/latent_variance_cka.json",
       "wcte_p2a_repair_json": "results/round2/20261117_repair/wcte_p2a/p2a_repaired_readings.json",
       "retrain_eval_json": "results/round2/20261117_repair/retrain/retrain_readings.json",
       "evolution_json": "results/round2/20261117_repair/retrain/latent_evolution.json",
       "master_report": "results/round2/20261117_repair/REPAIR_AND_DIAGNOSTICS_REPORT.md"
     },
     "gpu_hours_used": "<total from R2_repair_gpu>",
     "head_commit": "<git SHA pushed to origin/dev>",
     "completed_utc": "<UTC-ISO8601>"
   }
   ```

3. **Final Git Push & Driver Status Update**:
   - Commit all results, figures, reports, `STATE.md`, `DECISIONS.md`, `RESOURCE_LEDGER.json`, and `ACCEPTANCE_2026-11-17_R2_REPAIR.json` to `dev` and run `git push origin dev`.
   - Verify with `git fetch origin && git ls-tree -r --name-only origin/dev results/round2/20261117_repair/` that all deliverables are on `origin/dev`.
   - Set `$CC_STATE/status/r2-repair.json` to `"state": "done"` with `"report"`, `"commits"`, and `"summary"`.
   - Print `REPAIR_AND_DIAGNOSTICS_REPORT.md` in full as the final message.

---

## 8. Launch Commands (Run on Mac)

### Option A — Interactive / Foreground Unattended Session with `/goal` (Recommended)
```bash
cd ~/Documents/VERITAS-worktrees/r2 && claude --permission-mode auto \
  --settings ~/Documents/VERITAS/docs/round2/claude-code/settings.json \
  --append-system-prompt-file ~/Documents/VERITAS/docs/round2/claude-code/rules_common.md \
  --add-dir ~/Documents/VERITAS/docs/round2 ~/Documents/VERITAS_offrepo/claude-code \
  -n veritas-r2-repair \
  "/goal Read docs/round2/RUNBOOK_2026-11-17_R2_REPAIR.md and docs/round2/ROOT_CAUSE_ANALYSIS_2026-11-17.md (copy them from ~/Documents/VERITAS/docs/round2/ into ~/Documents/VERITAS-worktrees/r2/docs/round2/ first) and execute Stages 0 through 6 end to end without stopping to ask any questions. Use Deepthought CPU/Titans (dt1), deepthought2 L40S (dt2), and ETP HTCondor batch submission (scripts/r2_batch/submit.py) as needed. Solve any routine issue yourself and record decisions in docs/round2/DECISIONS.md. The goal is met only when all Stages 0-6 are complete, docs/round2/ACCEPTANCE_2026-11-17_R2_REPAIR.json and results/round2/20261117_repair/REPAIR_AND_DIAGNOSTICS_REPORT.md are committed and pushed to origin/dev, ~/Documents/VERITAS_offrepo/claude-code/status/r2-repair.json has state done, and the final report is printed in full."
```

### Option B — Detached Background Daemon (Survives Terminal Close, Auto-Resumes on Drop)
```bash
nohup bash ~/Documents/VERITAS/docs/round2/claude-code/run_chain.sh --only r2-repair >/dev/null 2>&1 &
```
*(Monitor with `tail -f ~/Documents/VERITAS_offrepo/claude-code/logs/r2-repair.log` and `cat ~/Documents/VERITAS_offrepo/claude-code/notes/r2-repair.md`.)*
