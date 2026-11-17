# Root-Cause Analysis: Round 2 (SK, JUNO, HK) & Round 1 (WCTE) Results

**Date:** 17 November 2026  
**Scope:** End-to-end audit of simulation design, data/feature scaling, transformer architecture, JEPA/MAE training loops, and evaluation/diagnostic probes across **Round 1 (WCTE — JEPA & PMT R-1..R-7 tracks)** and **Round 2 (SK, JUNO, HK — `r2_multidetector_2026-11-11` on `dev`)**, integrating code-level findings and Deepthought read-only verification numbers.

---

## 1. Executive Summary & Empirical Anomaly Overview

Across Round 1 (WCTE) and Round 2 (SK, JUNO, HK), four counter-intuitive results emerged (`results/round2/20261112_eval/READINGS_SUMMARY.md` on `dev`):

1. **The raw first-hit times in all four datasets support cm-level vertex reconstruction ($1.31\text{–}3.54\text{ cm}$), yet every model's probe error is $10\times\text{–}120\times$ worse ($0.11\text{–}3.54\text{ m}$)**:
   - Independent read-only robust least-squares fitting of $t_i = t_0 + n_g \|\mathbf{p}_i - \mathbf{x}\| / c$ (fitting vertex $\mathbf{x}$ and unknown $t_0$ with $n_g = 1.33$ on the same 128 photon-bomb event IDs per detector, seed `20261117`, multilateration init) achieves **$3.54\text{ cm}$ on WCTE, $1.72\text{ cm}$ on JUNO, $1.68\text{ cm}$ on SK, and $1.31\text{ cm}$ on HK** (128/128 converged, worst event $4.23\text{ cm}$ on HK).
   - Physically, **HK has the best intrinsic timing vertex resolution ($1.31\text{ cm}$)** because single-hit timing resolution $c\sigma_{\text{TTS}}/n_g \approx 34\text{ cm}$ is independent of tank radius and HK has the largest absolute number of hit PMTs ($\sim 4,200$).
   - By contrast, the `mae_mse` probe on those exact 128 events gives **$0.110\text{ m}$ (WCTE), $0.678\text{ m}$ (JUNO), $0.953\text{ m}$ (SK), and $1.591\text{ m}$ (HK)**—scaling linearly with tank radius $R$ because the `Encoder -> all-PMT mean pooling -> linear Ridge probe` chain is performing **tank-scale charge centroiding** rather than time-of-flight (ToF) triangulation.
2. **HK's large meter error is primarily a scale + photocoverage + receptive-field effect, plus an independent Gate V comparison mismatch**:
   - Normalized by fiducial radius $R_{\text{fid}} = 0.8 R$ (or divided by `init`), **HK sits consistently between JUNO and SK for every arm** ($\text{JUNO } 4.2\% < \text{HK } 5.5\% < \text{SK } 6.5\%$ for MAE; `MAE ÷ init` is $0.69$ on SK, $0.62$ on JUNO, $0.69$ on HK; `fixed-label ÷ init` is $1.64$ on SK, $1.37$ on JUNO, $1.73$ on HK).
   - However, HK has **half of SK's active PMT density/photocoverage** ($\sim 0.99\text{ PMT/m}^2$ or $20.0\%$ area coverage vs. SK's $\sim 1.95\text{ PMT/m}^2$ or $39.6\%$; at $320\text{k}$ photons, HK hits $4,208$ PMTs vs. SK's $6,352$ and JUNO's $7,120$), and suffers most from the tiny 64-token attention window (`mae_mse_rf` gains $-0.33\text{ m}$ on HK, twice the $-0.16\text{ m}$ / $-0.17\text{ m}$ gain on SK/JUNO).
   - Independently, **HK fails Gate V (10/10 events fail; SK 9/10 fail; JUNO 1/10 fail)** with a LUCiD/analytic yield ratio of **$2.07$** and module-pattern correlation of **$0.756$** (vs. photon-statistics ceiling $0.993$). Re-scoring the 10 HK Gate V truths proves this $2.07\times$ ratio comes from a **validation-condition mismatch**: `PromptTransport` runs with `absorption='off'` and spherical sensors, whereas `compare_event` scored the analytic model with $\lambda_0 = 50\text{ m}$ attenuation and flat-disc $\times |\cos\theta|$. Turning off the $50\text{ m}$ attenuation alone multiplies the analytic prediction by **$2.30\times$**, reducing the ratio from **$2.07 \to 0.90$** (and switching to spherical acceptance multiplies by **$1.27\times$**).
3. **Both JEPA arms fail on SK, JUNO, and HK due to a geometry shortcut AND a micro-batching implementation bug**:
   - Fixed-label `sigreg_jepa` collapses to an event-invariant representation ($f_{\text{event}} \approx 3\times 10^{-4}$ on SK/HK) with vertex error ($1.31\text{–}3.54\text{ m}$) **worse than random init (`init`) on all four geometries**.
   - Position-free `target_posfree` ($0.99\text{ m}$ JUNO, $1.28\text{ m}$ SK, $2.07\text{ m}$ HK) **fails to beat `init` on vertex** ($0.96\text{ m}$ JUNO, $1.27\text{ m}$ SK, $2.05\text{ m}$ HK), even though it retains directional information ($\mu$ direction error on seed 9000 beats MAE on all three Round 2 detectors: $4.98^\circ$ vs. $6.03^\circ$ on SK, $4.17^\circ$ vs. $6.04^\circ$ on JUNO, $4.59^\circ$ vs. $4.93^\circ$ on HK).
   - In Round 2, `SIGReg` was computed **inside each micro-batch** (`micro_batch = 4` on SK/HK, `8` on JUNO, accumulated to 32) on **unshuffled run-ordered batches** (`4` consecutive events per run), corrupting the nonlinear Epps–Pulley batch statistic and changing the effective regularization weight across detectors.
4. **Round 1 WCTE anomalies**:
   - On the **PMT PointNet track** ([PMT_DECISIONS_LOG.md](file:///Users/wongdowling/Documents/VERITAS/docs/PMT_DECISIONS_LOG.md)), the multi-energy pilot failed initial-energy MAE ($39\text{–}42\%$) because 477/480 muon placements escape the tank, and **R-4, R-4b, and R-4c** failed the vertex gate ($\le 0.80\times$ constant).
   - On the **WCTE JEPA/P2 track** (`docs/jepa/STATE.md` on `dev`), linear latent repair (`EST(latent)`, `LEACE`/`E3u`) failed to achieve drift invariance, and P2 Addendum stopped at `S4` because the `R3u` ceiling was outperformed by `R2` by $>5\text{ mm}$ on 4 cells of `fix9001`.

### Cross-Detector Empirical Summary (Round 1 & Round 2)

| Metric / Property | WCTE (Round 1) | JUNO (Round 2) | SK (Round 2) | HK (Round 2) |
| :--- | :---: | :---: | :---: | :---: |
| **Shape & Dimensions** | Cylinder<br>$R=1.58\text{ m}, H=2.78\text{ m}$ | Sphere (water proxy)<br>$R=17.50\text{ m}$ | Cylinder<br>$R=16.96\text{ m}, H=36.38\text{ m}$ | Cylinder<br>$R=32.43\text{ m}, H=65.93\text{ m}$ |
| **Active PMTs $N_{\text{PMT}}$ ($r_{\text{PMT}}$)** | $1,995$ ($0.04\text{ m}$) | $10,000$ ($0.25\text{ m}$) | $11,096$ ($0.254\text{ m}$) | $19,746$ ($0.254\text{ m}$, half excluded) |
| **PMT Density / Photocoverage** | $\sim 45\text{ /m}^2$ / $\sim 23.0\%$ | $\sim 2.60\text{ /m}^2$ / $\sim 51.0\%$ | $\sim 1.95\text{ /m}^2$ / $\sim 39.6\%$ | **$\sim 0.99\text{ /m}^2$ / $\sim 20.0\%$** |
| **Hit PMTs at $Y = 320\text{k}$ Photons** | $\sim 1,800$ | $7,120$ | $6,352$ | **$4,208$** |
| **Fiducial Radius $R_{\text{fid}} = 0.8 R$** | $1.26\text{ m}$ | $14.00\text{ m}$ | $13.57\text{ m}$ | $25.94\text{ m}$ |
| **Longest Path $L$ / $\lambda_0=50\text{ m}$ survival** | $4.20\text{ m}$ ($92\%$) | $35.00\text{ m}$ ($50\%$) | $49.74\text{ m}$ ($37\%$) | **$92.48\text{ m}$ ($16\%$)** |
| **Default Window $W=64$ (% of $N_{\text{PMT}}$)** | $3.21\%$ (4L: $\sim 8.0\%$) | $0.64\%$ (4L: $\sim 1.6\%$) | $0.58\%$ (4L: $\sim 1.4\%$) | **$0.32\%$ (4L: $\sim 0.8\%$)** |
| **128-Event ToF Fit Baseline vs. MAE (s9000)** | **$3.54\text{ cm}$** vs. $0.110\text{ m}$ ($3.1\times$) | **$1.72\text{ cm}$** vs. $0.678\text{ m}$ ($39\times$) | **$1.68\text{ cm}$** vs. $0.953\text{ m}$ ($57\times$) | **$1.31\text{ cm}$** vs. $1.591\text{ m}$ (**$121\times$**) |
| **`mae_mse` Vertex Error (m / $R_{\text{fid}}$)** | **$0.105\text{ m}$** ($0.083$) | **$0.591\text{ m}$** ($0.042$) | **$0.878\text{ m}$** ($0.065$) | **$1.419\text{ m}$** ($0.055$) |
| **`mae_mse_rf` Vertex Error (m / $R_{\text{fid}}$)** | — | **$0.520\text{ m}$** ($0.037$) | **$0.768\text{ m}$** ($0.057$) | **$1.249\text{ m}$** ($0.048$) |
| **`init` (5-seed floor mean [min, max], m)** | $0.160\text{ m}$ ($0.127$) | $0.930$ [$0.890, 0.968$] | $1.242$ [$1.206, 1.275$] | $1.995$ [$1.946, 2.077$] |
| **`target_posfree` Vertex Error (m / $R_{\text{fid}}$)** | $0.133\text{ m}$ ($0.106$) | $0.993\text{ m}$ ($0.071$) | $1.282\text{ m}$ ($0.095$) | $2.068\text{ m}$ ($0.080$) |
| **`sigreg_jepa` Vertex Error (m / $R_{\text{fid}}$)** | $0.212\text{ m}$ ($0.168$) | $1.312\text{ m}$ ($0.094$) | $2.079\text{ m}$ ($0.153$) | $3.544\text{ m}$ ($0.137$) |
| **`MAE ÷ init` / `fixed-label ÷ init`** | $0.66$ / $1.32$ | $0.62$ / $1.37$ | $0.69$ / $1.64$ | $0.69$ / $1.73$ |
| **$\mu$ Direction Median ($^\circ$): MAE / posfree / init** | $4.93^\circ$ / — / $8.41^\circ$ | $6.04^\circ$ / **$4.17^\circ$** / $6.34^\circ$ | $6.03^\circ$ / **$4.98^\circ$** / $6.54^\circ$ | $4.93^\circ$ / **$4.59^\circ$** / $6.24^\circ$ |
| **`init` Token $\ln q$ $R^2$ (unmasked `full_input`)** | $0.999$ | $0.9993$ | $0.9987$ | $0.9990$ |
| **MAE Time Loss Share (start $\to$ end)** | $0.005 \to 0.002$ | $0.035 \to \mathbf{0.0004}$ | $0.033 \to \mathbf{0.0004}$ | $0.050 \to \mathbf{0.0008}$ |
| **Gate V: Yield Ratio / Module $r$ / Log-$q$ $r$** | $1.01$ / $0.902$ / $0.815$ | $1.08$ / $0.888$ / $0.761$ | $1.19$ / $0.829$ / $0.692$ | **$2.07$ / $0.756$ / $0.563$** |

---

## 2. Summary Table of Where Each Problem Lives & How to Fix It

| Layer / Category | Specific Bug or Design Flaw | Exact File & Line Location (on `dev` unless noted) | Impact on Experimental Results | Concrete Fix |
| :--- | :--- | :--- | :--- | :--- |
| **1A. Evaluation Pooling** | Event embedding is computed via unweighted mean over **all $N_{\text{PMT}}$ tokens (`mask=None`)**, including $50\text{–}90\%$ unhit PMTs. | `src/jepa_noise/ext_r2.py:984`<br>`src/jepa_noise/lucid_arms.py:743–747` | Dilutes event signal by $3\times\text{–}10\times$; wipes out surface spatial distribution; drives $f_{\text{layout}} \to 0.85\text{–}1.00$ and $f_{\text{event}} \le 0.04$. | Pass `mask=hit` (`la._mean_tokens(tokens, hit)`) AND implement **Panda Diplomacy (V2) Appendix D.1 spatial-moment pooling** (first-order cross-moments between PMT coordinates $\mathbf{r}_p$ and token features $z_p$). |
| **1B. Vertex Probe** | 3D vertex is predicted from the 128-D global mean vector using a **linear Ridge regression** (`_vertex_fit`). | `src/jepa_noise/ext_r2.py:1044–1056`<br>`src/jepa_noise/lucid_arms.py:1251–1268` | Linear probe on a global average can only compute a tank-scale charge centroid ($\propto R_{\text{tank}}$, giving $0.6\text{–}1.6\text{ m}$), losing $39\times\text{–}121\times$ relative to the $1.3\text{–}1.7\text{ cm}$ ToF fit baseline. | Compare `mean + Ridge`, `mean + MLP`, **Panda V2 spatial-moment probe**, and a token-level position-aware readout on frozen checkpoints against the per-event ToF likelihood/CRLB baseline. |
| **1C. Misleading Diagnostics (`token_r2` & `f_event`)** | `PointEncoder` has `input_proj(9 -> 128)` + residual stream + `final_norm = None`; `token_r2` is scored on unmasked `full_input`; `f_event` only checks event-mean variance. | `src/jepa_noise/point_encoder.py:295, 301`<br>`src/jepa_noise/ext_r2.py:76, 224`<br>`src/jepa_noise/ext_readout.py:348` | **Random init (`init`) gets `token_r2 = 0.999`** via linear residual pass-through (so `0.94–0.99` on `target_posfree` does *not* prove JEPA is bug-free!); `f_event` misses within-PMT across-event variance. | Score `token_r2` on **masked/propagated** tokens (`context` pass at hidden indices), and log **per-PMT across-event token variance** alongside `f_event`. |
| **2A. MAE Time Loss Crushed** | `_forward_mae_mse` sums `loss_charge + loss_hit + loss_time` unweighted while `time_scale_ns = prompt_hi / 0.6` ($333\text{–}767\text{ ns}$). | `src/jepa_noise/lucid_arms.py:505–509`<br>`src/jepa_noise/r2_geometry.py:493–495` | Scaled time std is $\sim 0.08$; by step 10k, **time loss share is $0.04\%$ (SK/JUNO) and $0.08\%$ (HK)**. Timing patterns are under-weighted during training. | Check whether timing information is lost in the encoder tokens or in pooling; normalize time by within-event std ($\sigma_t \approx 30\text{–}60\text{ ns}$) or up-weight `loss_time` by $\sim 100\text{–}150\times$. |
| **2B. Tiny Receptive Field & Parity Masking** | 4 blocks of 1D Hilbert window $W=64$, shift $32$ covers $0.8\%$ of HK / $1.4\%$ of SK; `within` split masks every odd slot per module (adjacent PMTs always visible). | `src/jepa_noise/point_encoder.py:105–106, 250–270`<br>`src/jepa_noise/lucid_views.py:87–92, 165` | Local interpolation from adjacent PMTs is too easy, while global cross-tank context is absent (`mae_mse_rf` proved $W=64$ costs $0.16\text{–}0.33\text{ m}$). | Use $W_{\text{rf}}$ (or hit-only tokenization + global tokens) AND adopt **PoLAr-MAE spatial-neighborhood patch masking** instead of interleaved odd/even slot parity. |
| **3A. Trainable Target Geometry Shortcut** | `sigreg_jepa` encodes `labels` through the **shared trainable encoder with no `stop_grad` and no EMA**, while `predictor` receives target `(x,y,z)` queries. | `src/jepa_noise/lucid_arms.py:420–422, 467–477`<br>`src/jepa_noise/point_encoder.py:368–372` | Encoder ignores event columns 6–8 and outputs a static Gaussian function of PMT `(pos, dir)`; predictor predicts it from `queries` with $\sim 0$ loss ($f_{\text{event}} \sim 10^{-4}$). | Use an **EMA teacher with `torch.no_grad()`** + target normalization (like `ijepa_ema` / **Panda V1**), or adopt **Panda V2 observable distillation** (distilling only on hit/observable PMTs). |
| **3B. `target_posfree` All-Zero Unhit Collapse & Train/Eval Mismatch** | `target_posfree` zeroes columns 0–5 in the label pass (`~target` masked), leaving $65\text{–}90\%$ of target tokens (unhit PMTs) as `[0,0,0,0,0,0,0,0,0]`, then evaluates with columns 0–5 non-zero and no mask. | `src/jepa_noise/ext_shortcut.py:1173–1205`<br>`src/jepa_noise/ext_r2.py:167–168, 981` | Label pass sees $85\text{–}95\%$ constant tokens (`mask_embed` + `input_proj(0)`), collapsing event-mean rank to $1.54$; `full_input` eval has a train/test feature & mask mismatch. | Distill only on **hit target PMTs** (Panda V2 observable distillation) with relative 3D positional bias in attention, and keep input feature columns consistent between training and evaluation. |
| **3C. `micro_batch` + Unshuffled Runs Breaks `SIGReg`** | `R2TrainingLoader` stacks 4 events from each of 8 runs **in contiguous run order**; `train_loop` slices `micro_batch=4` (SK/HK) or `8` (JUNO) and runs `SIGReg` per micro-batch. | `src/jepa_noise/r2_store.py:250–266`<br>`src/jepa_noise/lucid_run.py:701–714`<br>`src/jepa_noise/sigreg.py:125–155` | `EppsPulley` squares the batch mean and multiplies by $N$, so micro-batch accumulation $\neq$ 32-event batch; on SK/HK (`micro_batch=4`), each pass sees **4 events from 1 run** (rank $\le 3$) and advances `global_step` $16\times$/step. | Shuffle the 32 events across runs before slicing, AND compute `SIGReg` once per step on the full $B=32$ embeddings (two-pass grad caching across micro-batches). |
| **3D. `SIGReg` Target Flattening & $\lambda$ Scaling** | `SIGReg.forward` flattens `target_tokens` to `(B * n_target, d)` ($n_{\text{target}} = 997\text{–}9,873$ PMTs) and `EppsPulley` multiplies by $N$ with fixed $\lambda=0.02$. | `src/jepa_noise/sigreg.py:132, 169–175` | `SIGReg(target_flat)` is satisfied by static variation **across PMT positions** ($f_{\text{layout}} \approx 1, f_{\text{event}} = 0$), and effective regularization weight shifts with $n_{\text{target}}$. | Center each PMT across the batch dimension `dim=0` before `SIGReg` (or apply `SIGReg` to event-pooled target vectors) and normalize `EppsPulley` consistently across detector sizes. |
| **4A. Gate V Attenuation & Sensor-Shape Bug** | `PromptTransport` runs with `absorption='off'` ($\lambda=\infty$) and **spherical** sensors ($\pi r^2$), while `compare_event` uses $\lambda_0 = 50\text{ m}$ and flat-disc obliquity **`|cos θ|`**. | [lucid_transport.py](file:///Users/wongdowling/Documents/VERITAS/src/noise_module_lucid/lucid_transport.py#L61-L70)<br>`src/jepa_noise/cherenkov_validation.py:109–110`<br>`src/jepa_noise/lucid_bombs.py:1722–1724` | On HK's 10 Gate V events, turning off $\lambda_0=50\text{ m}$ multiplies analytic light by **$2.30\times$** (ratio **$2.07 \to 0.90$**); spherical acceptance multiplies by **$1.27\times$**. | Re-run Gate V with matched attenuation (`lb.UNATTENUATED_M`), matched spherical acceptance (no `|cos θ|`), and verify whether HK's module correlation ($0.756$) rises toward its $0.993$ ceiling. |
| **4B. HK Photocoverage, $\lambda_0=50\text{ m}$, & Dark Trigger Jitter** | HK excludes 19,036 `inactive_*` PMTs ($20\%$ coverage vs. SK $39.6\%$, JUNO $51\%$); keeps $\lambda_0=50\text{ m}$ over $L=92.5\text{ m}$; has $19.7$ dark hits/event; JUNO uses water proxy. | `src/jepa_noise/r2_geometry.py:390–392`<br>`src/jepa_noise/r2_generator.py:41`<br>`results/round2/20261112_x0/hk.json:18` | HK far wall loses $84\%$ light; **dark hits move the trigger in $27.8\%$ of HK events**; JUNO water sphere tests spherical layout only, not liquid scintillator physics. | Scale attenuation length to realistic water ($\lambda_0 \approx 90\text{–}100\text{ m}$), de-jitter trigger times, and use liquid-scintillator emission/transport before testing JUNO weight transfer. |
| **5A. WCTE PMT R-1..R-4c Failures** | (i) 477/480 muon placements escape WCTE; (ii) R-4/R-4b window-start time origin adds $+10$ offset on hits vs. $0$ unhit; (iii) only 60 source events + simple max-pool `PointNetBaseline`. | [PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md](file:///Users/wongdowling/Documents/VERITAS/docs/PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md#L19)<br>[PMT_DECISIONS_LOG.md](file:///Users/wongdowling/Documents/VERITAS/docs/PMT_DECISIONS_LOG.md#L447-L497)<br>[multienergy_baseline.py](file:///Users/wongdowling/Documents/VERITAS/src/noise_module_lucid/multienergy_baseline.py) | Initial energy fails ($39\text{–}42\%$ MAE); R-4/R-4b vertex is $0.96\text{–}1.00\times$ constant; R-4c vertex reaches $0.81\text{–}0.88\times$ constant (just above $0.80$ gate). | Use `deposited_energy_in_volume` (`D-E`) or low-energy contained muons ($100\text{–}350\text{ MeV}$), keep `NHits` + prompt-window time origin, and increase physical event count beyond 60. |
| **5B. WCTE P2 / Repair `S4 STOP`** | `R3u` ceiling concatenates 105 dead-module columns `[z_c, u_c]` into `fit_vertex_probe` with a single shared ridge penalty $\alpha$; `EST(latent)` / `LEACE` assume linear low-rank gain shifts. | `docs/jepa/STATE.md:491–520`<br>`src/jepa_noise/ext_p2a.py` | Shared ridge $\alpha$ over-penalizes $z_c$ when 105 $u_c$ columns are added, causing $v_{\text{R2}} < v_{\text{R3u}}$ by $>5\text{ mm}$ (`S4 STOP` on 4 cells of `fix9001`); linear projection removes only $31\text{–}59\%$ of gain shift. | Regularize $z_c$ and $u_c$ with separate ridge penalties $(\alpha_z, \alpha_u)$ or residualize $u_c$ on `R2` predictions so adding side information cannot worsen CV risk. |

---

## 3. Detailed Explanations & How to Fix Each Problem

---

### Problem 1: Diagnostics, Readout Pooling & Probing Method (Why Cm-Level Information in the Data Becomes Meter-Level Error)

#### 1.1 Proof That the Saved Data Supports Cm-Level Vertexing (128-Event ToF Fit)
A critical question (Q2 in [RUNBOOK_2026-11-17_R2_DIAGNOSTICS.md](file:///Users/wongdowling/Documents/VERITAS/docs/round2/RUNBOOK_2026-11-17_R2_DIAGNOSTICS.md)) was whether the large vertex errors ($0.5\text{–}3.5\text{ m}$) were limited by the simulated data itself.  
On Deepthought, an independent robust least-squares fit of the direct-light arrival-time model:
$$t_i = t_0 + \frac{n_g \|\mathbf{p}_i - \mathbf{x}\|}{c}, \qquad n_g = 1.33$$
was run on the **same 128 photon-bomb event IDs** (sample seed `20261117`) across all four detectors, fitting $(\mathbf{x}, t_0)$ directly from saved PMT coordinates and first-hit times without access to true vertex, light yield, or run gain (using multilateration initialization and scaled optimization parameters so $128/128$ events converged):

| Detector | First-Hit Time Fit Only (Mean 3D Error) | Same-128-Event `mae_mse` Probe (Seed 9000) | Ratio (`MAE probe ÷ ToF fit`) |
| :--- | :---: | :---: | :---: |
| **WCTE** | **$3.54\text{ cm}$** ($0.0354\text{ m}$) | $0.110\text{ m}$ | $3.1\times$ |
| **JUNO** | **$1.72\text{ cm}$** ($0.0172\text{ m}$) | $0.678\text{ m}$ | $39.4\times$ |
| **SK** | **$1.68\text{ cm}$** ($0.0168\text{ m}$) | $0.953\text{ m}$ | $56.7\times$ |
| **HK** | **$1.31\text{ cm}$** ($0.0131\text{ m}$, max $4.23\text{ cm}$) | $1.591\text{ m}$ | **$121.5\times$** |

**Why ToF fit gets *better* on larger detectors ($3.54\text{ cm} \to 1.31\text{ cm}$) while the probe gets *worse* ($0.11\text{ m} \to 1.59\text{ m}$):**
- In physical ToF multilateration, single-PMT timing jitter ($\sigma_{\text{TTS}} = 1.5\text{ ns} \iff \sim 34\text{ cm}$) is independent of tank radius $R$, so vertex error scales as $\sim \frac{c\sigma_{\text{TTS}}/n_g}{\sqrt{N_{\text{hit}}}}$. Because HK has the most hit PMTs per event, its physical vertex resolution is the best ($1.31\text{ cm}$).
- In `src/jepa_noise/ext_r2.py:984, 1044`, the evaluation pipeline instead computes an **unmasked arithmetic average over all $N_{\text{PMT}}$ tokens** (`la._mean_tokens(tokens, None)`) and fits a **linear Ridge regression** (`_vertex_fit`) from that 128-D mean vector to $(x, y, z)$.
- Averaging across the entire tank collapses spatial structure into a single 128-D vector where $70\text{–}90\%$ of the averaged tokens are unhit PMTs (`f_layout = 0.81–1.00`), and a linear readout on a global average can only compute a **charge-weighted spatial centroid** $\sum_p w_p \mathbf{r}_p$, whose geometric error scales linearly with tank radius $R_{\text{fid}}$ ($\sim 4\text{–}8\%\,R_{\text{fid}}$).

#### 1.2 Why `token_log_charge_r2 = 0.94–0.99` Does NOT Prove JEPA Is Working, and Why `f_event` Alone Is Insufficient
1. **Pitfall of citing `token_log_charge_r2` on unmasked `full_input`**:
   - `PointEncoder` (`src/jepa_noise/point_encoder.py:295, 301`) projects the 9 input features via `nn.Linear(9, 128)` into a pre-norm residual stream with **`self.final_norm = None`**.
   - Consequently, the raw input column 6 (`log_charge`) sits linearly inside the 128-D token vector even at **random initialization (`init`)**, which achieves **`token_log_charge_r2 = 0.9987` (SK), `0.9993` (JUNO), and `0.9990` (HK)** (`READINGS_SUMMARY.md` lines 136–138).
   - `target_posfree` getting $0.94\text{–}0.99$ is actually *lower* than untrained `init` ($0.999$) and only reflects residual pass-through, not learned representation quality.
2. **Limitation of `f_event` (`src/jepa_noise/ext_readout.py:348`)**:
   - `f_event` measures only the variance of the **PMT-averaged event embedding** across events. If an encoder encodes event information into spatially varying token patterns whose unweighted sum over the tank cancels out, `f_event` will be small even when individual PMT tokens vary across events.
   - Therefore, collapse diagnostics must also measure **same-PMT across-event token variance** ($\text{Var}_e(z_{e,p})$ for hit PMTs $p$) alongside a **position-preserving vertex probe**.

#### 1.3 How to Fix Diagnostics & Probing (Including Panda Diplomacy Appendix D.1)
1. **Port the Spatial-Moment Vertex Probe from Panda Diplomacy (V2, Appendix D.1)**:
   - Instead of only averaging token embeddings $\bar{z} = \frac{1}{\sum h_p}\sum_p h_p z_p$, preserve the spatial distribution of features on the detector surface by forming **first-order spatial cross-moments** between unit-scaled PMT coordinates $\tilde{\mathbf{r}}_p \in \mathbb{R}^3$ and hit-masked token features $z_p \in \mathbb{R}^d$:
     $$M_{\text{spatial}} = \left[\bar{z}_{\text{hit}}, \; \frac{1}{\sum h_p}\sum_{p} h_p z_p \tilde{x}_p, \; \frac{1}{\sum h_p}\sum_{p} h_p z_p \tilde{y}_p, \; \frac{1}{\sum h_p}\sum_{p} h_p z_p \tilde{z}_p\right] \in \mathbb{R}^{4d}$$
   - Run a clean 4-way readout comparison on frozen checkpoints (`seed 9000`) using the same run-grouped split:
     1. `unmasked mean + Ridge` (current baseline),
     2. `hit-masked mean + MLP`,
     3. `Panda V2 spatial-moment probe` (on encoder tokens $z_p$),
     4. `raw-feature spatial-moment baseline` (on raw hit/charge/time features $f_p$, no encoder).
   - This immediately isolates how much error comes from the readout/pooling vs. information lost inside the encoder.
2. **Benchmark Caution Against External Literature (Panda V2 WAND)**:
   - Panda Diplomacy (V2) reports SK-like WAND single-ring $e^-$ vertex $P_{68} = 0.987\text{ m}$ (adapted on 1,000 labeled events in Geant4 + LUCiD with scattering, absorption, and angular acceptance).
   - Do **not** equate WAND's $0.987\text{ m}$ on extended scattered electron showers with our $0.878\text{ m}$ mean 3D error on scatter-free isotropic photon bombs (where ToF fitting reaches $1.68\text{ cm}$). When comparing against WAND or `fiTQun`, evaluate on single-$e^-$ events and report both **mean and $P_{68}$** stratified by energy, hit count, and distance to wall.

---

### Problem 2: Transformer Architecture, Masking & Feature/Loss Scaling

#### 2.1 MAE Time Loss Share ($0.04\text{–}0.08\%$) vs. Timing Preservation
In `src/jepa_noise/lucid_arms.py:505–509`, `_forward_mae_mse` adds `loss_charge + loss_hit + loss_time` with equal weights ($1:1:1$), while `src/jepa_noise/r2_geometry.py:493` scales `time_ns` by `prompt_upper / 0.6` ($333.3\text{ ns}$ JUNO, $450.0\text{ ns}$ SK, $766.7\text{ ns}$ HK).
- Because the scaled time feature has std $\sim 0.08$ (variance $\sim 0.006$), the time term's share of the total MAE loss at the end of training is only **$0.04\%$ on SK and JUNO and $0.08\%$ on HK** (`READINGS_SUMMARY.md` lines 71–90).
- While a small loss share does not automatically equal the exact gradient share (since a well-fitted linear scale has small residual MSE), combining this with the $39\times\text{–}121\times$ gap between the ToF fit ($1.3\text{–}1.7\text{ cm}$) and the MAE probe ($0.68\text{–}1.59\text{ m}$) makes it essential to test whether vertex-relevant timing patterns are preserved in the encoder tokens (`tokens[e, hit]`) or destroyed before pooling.

#### 2.2 Local Window Receptive Field ($W=64$) & Interleaved Parity Masking (`within`)
1. **Receptive field**: 4 blocks of 1D Hilbert window $W=64$, shift $32$ (`src/jepa_noise/point_encoder.py:250, 316`) give a 4-layer receptive field of $160$ tokens ($0.8\%$ of HK, $1.4\%$ of SK). Expanding the window in `mae_mse_rf` ($W=384$ SK, $320$ JUNO, $640$ HK) improved vertex error by **$-0.16\text{ m}$ (SK), $-0.17\text{ m}$ (JUNO), and $-0.33\text{ m}$ (HK)**, though `mae_mse_rf` on HK is still $1.25\text{ m}$ because pooling and time scaling remain unchanged.
2. **Interleaved `within` parity masking is too local**:
   - In `within` split (`src/jepa_noise/lucid_views.py:165`), odd-slot PMTs in each 18–20 PMT pseudo-module are masked targets and even-slot PMTs are visible context.
   - Because Hilbert-contiguous PMTs alternate even/odd (`1, 2, 3, 4...`), **every masked target PMT sits immediately between two visible context PMTs a fewtens of centimeters away**!
   - Both MAE and JEPA can therefore predict a masked PMT's charge/time by trivial **nearest-neighbor interpolation** inside a single window without ever learning global event geometry (such as the 3D vertex or Cherenkov ring cone).

#### 2.3 How to Fix the Transformer, Masking & MAE Loss
1. **Adopt PoLAr-MAE Spatial-Neighborhood Patch Masking**:
   - Instead of interleaved odd/even slot parity (`within`), mask contiguous **3D spatial patches** (e.g., masking whole modules or spherical cap patches of radius $R_{\text{patch}}$ covering $40\text{–}60\%$ of the tank), forcing the encoder and predictor to infer hit, charge, and first-hit time across long distances from global Cherenkov geometry.
2. **Re-scale / Up-weight the Time Objective & Add Global Tokens**:
   - Normalize `time_ns` by the empirical hit-time std ($\sim 35\text{–}65\text{ ns}$) or up-weight `loss_time` by $\sim 100\times$, and add global register/summary tokens (or run attention over hit PMTs only) so tokens communicate across the entire tank.

---

### Problem 3: JEPA Objective & Training-Loop Bugs

#### 3.1 Fixed-Label JEPA (`sigreg_jepa`): The Static Geometry Shortcut
In `src/jepa_noise/lucid_arms.py:408–423, 467–477`:
- Both `context` and `labels` (`label_full = self.jepa.encode(batch.features, hidden_mask=~batch.target)`) use the **shared trainable encoder with gradients enabled on both sides** (no `stop_grad`, no EMA teacher).
- `TargetQueryPredictor` (`src/jepa_noise/point_encoder.py:368–372`) directly receives the target PMT 3D coordinates `queries = self.query_features[target_index]`.
- Because `within` split uses a fixed odd/even PMT partition and `labels` retains static PMT position/direction (columns 0–5), the encoder can minimize $\text{MSE}(\text{prediction}, \text{labels}) + \lambda\,\text{SIGReg}$ by ignoring event columns 6–8 (`ln q, time, hit`) and mapping each PMT's static coordinates to a fixed vector $z_p = g(\mathbf{p}_p, \mathbf{d}_p)$ that is Gaussian across the detector's PMTs ($f_{\text{layout}} \approx 1.000, f_{\text{event}} < 10^{-3}$).
- All Round 1 WCTE architecture/scale/view controls (`window32`, `window128`, `pred4`, `300k/30k scale`) failed to fix this collapse because none of them removed the trainable-target geometry shortcut.

#### 3.2 Position-Free JEPA (`target_posfree`): All-Zero Unhit Target Tokens & Partial Survival of Direction Info
In `src/jepa_noise/ext_shortcut.py:1173–1205`, `target_posfree` zeroes columns 0–5 (`pos, dir`) in the label pass:
- On WCTE, this reduced vertex error from $0.212\text{ m} \to 0.133\text{ m}$. And on Round 2 (seed 9000), `target_posfree` actually **preserved $\mu$ direction information better than `mae_mse` on all three detectors** ($4.98^\circ$ vs. $6.03^\circ$ on SK; $4.17^\circ$ vs. $6.04^\circ$ on JUNO; $4.59^\circ$ vs. $4.93^\circ$ on HK).
- However, on photon-bomb vertexing in Round 2, `target_posfree` failed to beat `init` because:
  1. In the label pass, $50\%$ of tokens are `mask_embed` and **$65\text{–}90\%$ of the remaining target tokens are unhit PMTs** whose columns 6–8 are `(0,0,0)`. Zeroing columns 0–5 turns every unhit target PMT into `[0,0,0,0,0,0,0,0,0]`, so $85\text{–}95\%$ of tokens in every 64-token window are just two constant vectors (`mask_embed` and `input_proj(0)`), collapsing event-mean effective rank to $1.54$.
  2. At evaluation (`full_input`), the encoder is fed non-zero columns 0–5 and $0\%$ `mask_embed`, which mismatches the zero-position label pass it was trained on.

#### 3.3 The `micro_batch` + Unshuffled Run-Order Bug in `SIGReg` (`src/jepa_noise/lucid_run.py:701–714`, `sigreg.py:125`, `r2_store.py:250`)
This is the #1 training-implementation fairness bug in Round 2:
1. **`SIGReg` computed per micro-batch instead of per effective batch (32)**:
   - In `results/round2/20261112_train/runs.json`, SK and HK JEPA ran with **`micro_batch = 4` (`accum_steps = 8`)** and JUNO ran with **`micro_batch = 8` (`accum_steps = 4`)**, whereas WCTE ran with **`micro_batch = 32` (`accum_steps = 1`)**.
   - In `EppsPulley.forward` (`src/jepa_noise/sigreg.py:125–133`), the empirical characteristic function mean $\frac{1}{N}\sum_{i=1}^N e^{i t x_i}$ is **squared and multiplied by $N$**:
     ```python
     cos_mean = x_t.cos().mean(dim=0)
     sin_mean = x_t.sin().mean(dim=0)
     err = (cos_mean - self.phi).square() + sin_mean.square()
     return (err @ self.weights) * n
     ```
   - Accumulating gradients over 8 micro-batches of $N=4$ events does **not** reproduce the 32-event Epps–Pulley statistic (and 4 vectors in $\mathbb{R}^{128}$ span at most a 3-dimensional subspace!).
2. **Unshuffled run-contiguous ordering in `R2TrainingLoader.batch_at` (`src/jepa_noise/r2_store.py:250–266`)**:
   - `batch_at` draws 8 runs and takes 4 events per run, stacking them in contiguous run order (`[run0_ev0..3, run1_ev0..3, ...]`) without shuffling.
   - At `micro_batch = 4` (SK and HK), **every single micro-batch contains 4 events from a single run**!
3. **Target PMT count $n_{\text{target}}$ changes the effective $\lambda = 0.02$ weight across detectors**:
   - `SIGReg` averages `context_value` ($N = B_{\text{micro}}$) and `target_value` ($N = B_{\text{micro}} \times n_{\text{target}}$, where $n_{\text{target}}$ is $997$ on WCTE, $5,000$ on JUNO, $5,548$ on SK, $9,873$ on HK). Because `EppsPulley` multiplies by $N$ while `target_flat` flattens events and PMTs together, static across-PMT variance participates in the regularizer and a fixed $\lambda = 0.02$ does not maintain a constant per-event/per-token regularization weight across geometries.

#### 3.4 How to Fix the JEPA Branch (Including Panda V1 / V2 Ideas)
1. **Fix `SIGReg` batch semantics first**:
   - Shuffle the 32 events across the 8 runs in `R2TrainingLoader.batch_at`, and compute `SIGReg` over the **full 32-event batch** (using two-pass gradient caching across micro-batches) rather than inside each 4-event micro-batch.
   - Center each PMT's embedding across the batch dimension ($\tilde{z}_{b,p} = z_{b,p} - \frac{1}{B}\sum_{b'} z_{b',p}$) before `SIGReg` so static PMT layout variance cannot satisfy the normality test.
2. **Adopt Panda Diplomacy (V2) Observable Distillation & Panda V1 EMA Teacher**:
   - **Observable distillation (Panda V2)**: Remove masked/unhit measurements and distill **only on observable (hit) target PMTs**, which eliminates both the fixed-layout shortcut on unhit PMTs and `target_posfree`'s all-zero unhit token collapse.
   - **EMA teacher + local/global views (Panda V1)**: Use an EMA target encoder with `torch.no_grad()` and prototype/LayerNorm stabilization as a clean control against trainable-target `SIGReg-JEPA`, verifying that per-PMT embeddings vary with charge and time across events.

---

### Problem 4: Simulation Design, Detector Geometry & Gate V Discrepancies

#### 4.1 Exact Breakdown of HK's Gate V $2.07\times$ Discrepancy
In Gate V (`results/round2/20261112_x0/hk_gatev.json`), 10/10 HK holdout events, 9/10 SK events, and 1/10 JUNO events failed the round-1 thresholds. While HK has half the geometric photocoverage of SK ($20.0\%$ vs. $39.6\%$, both using $r = 0.254\text{ m}$ PMTs), **photocoverage does not cause the $2.07\times$ Gate V yield ratio because both sides of Gate V are scored on the exact same active PMT set (`19,746` PMTs)**.

Instead, comparing [lucid_transport.py](file:///Users/wongdowling/Documents/VERITAS/src/noise_module_lucid/lucid_transport.py#L61-L70) (`PromptTransport`) against `compare_event` (`src/jepa_noise/cherenkov_validation.py:109–110`) and re-scoring the 10 HK holdout truths reveals two direct modeling mismatches:
1. **Attenuation (`50 m` vs. `absorption='off'`) — $\mathbf{2.30\times}$ effect on HK**:
   - `compare_event` calls `expected_charge_particle(..., lb.LAMBDA0_M, cfg)` with $\lambda_0 = 50\text{ m}$, whereas `PromptTransport` runs with `absorption='off'` ($\lambda = \infty$).
   - Re-computing the analytic side on the 10 HK holdout truths with attenuation turned off ($\lambda = \infty$) **multiplies the predicted total light by $2.30\times$**, which brings the LUCiD/analytic yield ratio from **$2.07$ down to $\approx 0.90$**!
2. **Angular acceptance (Flat disc $\times |\cos\theta|$ vs. Spherical hard intersection) — $\mathbf{1.27\times}$ effect on HK**:
   - `solid_angle_fraction` (`src/jepa_noise/lucid_bombs.py:1722`) multiplies the solid angle by the flat-disc obliquity factor $|\cos\theta|$, whereas `PromptTransport` intersects rays with spherical sensors (uniform cross-section $\pi r^2$ from all incidence angles).
   - Replacing the flat-disc $|\cos\theta|$ factor with spherical acceptance on the 10 HK truths **multiplies predicted light by $1.27\times$**.
3. **Remaining spatial-pattern check**:
   - HK's module-pattern Pearson correlation is $0.756$ against a photon-statistics ceiling of $0.993$ (`results/round2/20261112_gatev_ceiling/GATEV_CEILING.md`). After matching attenuation ($\lambda=\infty$) and spherical acceptance in `compare_event`, the spatial module/log-charge correlations must be re-evaluated to verify whether `0.756` was also driven by the $\exp(-d/50\text{ m}) \cdot |\cos\theta|$ distortion along the vertical muon ring.

#### 4.2 Detector-Specific Simulation Caveats (HK Coverage, Dark Jitter, JUNO Water Proxy)
1. **HK Photocoverage & Dark-Moved Triggers**:
   - Excluding the 19,036 `inactive_*` sensors leaves $19,746$ active PMTs ($20.0\%$ coverage, $\sim 0.99\text{ PMT/m}^2$). At $1\text{ kHz}$ dark rate over $1\,\mu\text{s}$, HK has $19.7$ dark hits per event, causing **dark coincidences to shift the `NHits` ($N=8$) trigger time in $27.8\%$ of HK events** (`results/round2/20261112_x0/hk.json`).
2. **JUNO Medium Proxy**:
   - Round 2 JUNO uses LUCiD's spherical **water** configuration (`JUNO_geom_config.json`), which is valid for testing spherical PMT layout adaptation, but the real JUNO central detector is **liquid scintillator** ($\sim 10^4\text{ photons/MeV}$ isotropic scintillation + re-emission + slower decay time constants). Any cross-detector weight transfer to real JUNO requires scintillator emission/transport data.

---

### Problem 5: Specific Root Causes in Round 1 WCTE (PMT Track & JEPA/P2 Track)

#### 5.1 WCTE PMT Multi-Energy Track (`R-1` to `R-4c`)
1. **Initial-Energy Gate Failure ([PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md](file:///Users/wongdowling/Documents/VERITAS/docs/PMT_MULTIENERGY_TRACK_REPORT_2026-10-16.md#L19-L47))**:
   - $500\text{–}1,500\text{ MeV}$ muons have water ranges of $2.15\text{–}6.45\text{ m}$ and exit the small WCTE cylinder ($R=1.58\text{ m}, H=2.78\text{ m}$) in **477/480 placements**, making initial kinetic energy physically unidentifiable from inside-tank hits alone. Switching to `deposited_energy_in_volume` (`D-E` in R-4/R-4c) passed the energy gate ($0.57\text{–}0.64\times$ constant).
2. **Vertex Gate Failures in R-4/R-4b ($0.96\text{–}1.00\times$ constant) and R-4c ($0.81\text{–}0.88\times$ constant)**:
   - R-4/R-4b used the readout window start (`-200 ns`) as the time origin (`E33`, `E42`), putting a $+10$-unit constant offset on hit sensors vs. $0$ on unhit sensors, which spoiled `PointNetBaseline`'s conditioning.
   - R-4c (`NHits N=4 + prompt window`, `E46–E47`) fixed the time origin and improved test vertex RMSE to $0.812\text{–}0.883\times$ constant, narrowly missing $\le 0.80\times$ due to the tiny sample size (24 training physical events $\times 8$ rotations = 192 correlated placements) and simple global max-pooling.

#### 5.2 WCTE P2 / Repair `S4 STOP` (`results/jepa/20261111_p2a/EXECUTION.json:238`)
- In `20261111_p2a`, the "fair refit ceiling" `R3u` concatenated 105 dead-module indicator columns $u_c$ to the 128-D embedding $z_c$ inside `fit_vertex_probe` with a **single shared Ridge regularization scalar $\alpha$**.
- In 28 of 195 cells (and 4 cells on `fix9001` with $95\%$ CI excluding 0 by $>5\text{ mm}$: `K5` full $-6.65\text{ mm}$, `K5` exact $-7.34\text{ mm}$, `P10` full $-5.31\text{ mm}$, `P10` exact $-6.61\text{ mm}$), adding 105 noisy columns shifted the shared $\alpha$ and degraded test vertex error relative to `R2` ($z_c$ alone), triggering `S4 STOP`.
- **Fix**: Fit `R3u` with separate ridge hyperparameters $(\alpha_z, \alpha_u)$ for $z_c$ and $u_c$ (or fit $u_c$ on the residuals of `R2`) so adding side information is guaranteed non-increasing in error.

---

## 4. Prioritized Action & Verification Roadmap

1. **Step 1 — Readout & Probe Ablation on Frozen Checkpoints (CPU / Zero Retraining)**:
   - On the existing frozen Round 2 checkpoints (`seed 9000`) and the same run-grouped split, compare:
     1. `all-PMT mean + Ridge` (current report baseline),
     2. `hit-masked mean + MLP`,
     3. **Panda Diplomacy (V2) Appendix D.1 spatial-moment probe** (first-order cross-moments of PMT coordinates and token features),
     4. **Raw-feature spatial-moment baseline** (directly on hit/charge/time without encoder),
     5. **Direct ToF likelihood / least-squares fit & CRLB ceiling** ($1.31\text{–}3.54\text{ cm}$).
   - If switching to the spatial-moment / hit-masked probe recovers most of the gap toward the ToF fit, fix the evaluation probe first before spending GPU hours retraining.
2. **Step 2 — Gate V Closure & Optical Diagnostic on HK (CPU)**:
   - Re-run `compare_event` on HK, SK, and JUNO with matched attenuation ($\lambda = \infty$ matching `PromptTransport(absorption='off')`), matched spherical sensor acceptance (removing $|\cos\theta|$), and matched active-PMT mapping; check whether HK's module-pattern correlation rises from $0.756$ toward its $0.993$ photon-statistics ceiling.
3. **Step 3 — Fix `SIGReg` Batch Semantics, Masking & Time Weighting Before Any Retraining (GPU)**:
   - Fix `R2TrainingLoader` (shuffle events across runs) and compute `SIGReg` over the full 32-event batch with per-PMT batch centering.
   - Test **Panda V2 observable distillation** (hit-only target distillation) + **EMA teacher** (`ijepa_ema` / Panda V1) + **PoLAr-MAE spatial patch masking** + balanced MAE time loss weight.
   - If external **Panda V2 WAND** weights become available, test weight transfer on SK first under matched architecture, inputs, and spatial-moment readout before testing on HK.
