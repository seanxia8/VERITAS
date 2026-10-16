# JEPA execution prelaunch review — 16 October 2026

**Verdict: REJECT for unattended execution; revise the registration documents first.** This is an execution-contract review, not a rejection of the research direction. Review-only scope: no SSH, training, downloads, commits, or runbook edits were performed.

Reviewed snapshot: branch `JEPA`, HEAD `0cae6b061e488e319643cfb0ba820318b8923578`. The JEPA documents were untracked at inspection. References below are to this snapshot; `RUNBOOK` means `docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md` and `PROMPT` means its companion launch prompt. Findings use exact line numbers and quoted premises. Severity P1 means resolve before registration/affected execution; P2 means a concrete implementation or reproducibility repair. Theory validity is assessed separately.

## Findings requiring protocol resolution

### E01 — P1: the Tier-1 full covariance has an exact DC nullspace; the oracle requires an inverse Cholesky

**Premise:** RUNBOOK 508–512: “inverse **lower Cholesky** factor of the cell's full implied noise covariance”; 366–370: “Fisher I(θ) = Jᵀ Σ_n⁻¹ J”.

**Evidence:** `src/latent_monitor/tier1.py:200–207` selects `power_definition="variance"`; `src/noise_module/core/generator.py:203–215` removes DC. `src/latent_monitor/whitening.py:68–71` explicitly handles the zero DC bin by replacing it with a unit scaling for its existing whitener. A bounded local check with `reference_cell().implied_whitener().psd` returned `[0., 0.00041634, 0.00039708, 0.00038988, ...]`. Thus each channel's full temporal noise covariance has a constant-vector nullspace. A full 8×256 covariance has rank at most 8×255. The standard lower Cholesky inverse required by arm 9 does not exist. Copying the whitener's DC floor would create a different covariance, not recover the exact generator covariance. Partial-view covariance may still be positive definite; this finding specifically concerns the full-record oracle and full-record Fisher inverse.

**Consequence:** The mandatory ninth arm cannot implement its stated exact operation; full-record CRLB calculations also need a singular-noise treatment. A jitter silently added only to the inverse changes the oracle and benchmark.

**Minimal remedy:** Author-register a consistent choice: introduce genuine DC variance into the new wrapper's data-generating law and propagate it everywhere, or define a support-restricted/singular model with valid information and whitening formulas and leakage checks. Do not silently use a pseudoinverse or diagonal jitter as if it were the specified full inverse.

### E02 — P1: the E2 gap is not represented by the stated patch mask

**Premise:** RUNBOOK 413–417 defines `[0,112) → [144,256)`; 460–463 specifies patch length/stride 32 and “patches 0–3 → patches 4–7”, with the gap “masking out of the loss the patch straddling the gap”.

**Evidence:** Patches 3 and 4 cover `[96,128)` and `[128,160)`. The gap `[112,144)` straddles **two** patches, including part of a context patch. Excluding a target loss patch cannot remove gap samples from the context. `src/reconstruction_model/model.py:253–257` embeds full patches before temporal attention.

**Consequence:** The realized E2 context, target and gap differ from the sample split used for population γ_n, covariance, and the claimed E1b mapping. The generic zero-gradient test on “masked” inputs need not expose a wrongly declared index set.

**Minimal remedy:** Freeze explicit sample and token ownership for context, gap and target. Choose an exact within-patch masking design or author-approved patch-aligned split; compute γ_n and covariances on those actual indices. Test zero gradients against the declared raw target **and gap** samples, not merely against a model-generated mask.

### E03 — P1: R6 needs the development cell that WP5 excludes

**Premise:** RUNBOOK 242 requires nominal-SNR Spearman “over 3 levels × seeds”; 533–534 says `(mid, nominal)` “is used only for β, λ_J and the smoke run, reported separately and never pooled”.

**Consequence:** Nominal-SNR confirmatory evaluation has only low and high levels if the exclusion is followed. Pooling the tuned mid-level results violates the exclusion; treating smoke/tuning checkpoints as full-grid outcomes violates the equal-training comparison. The declared R6 cannot be evaluated under both instructions.

**Minimal remedy:** Define an independent development dataset/cell and retain all three full-length evaluation levels, or explicitly revise R6 and its aggregation. Freeze paired seeds, aggregation of nuisance targets, and the Spearman inputs before outcomes are generated.

### E04 — P1: criterion M cannot be calculated from noise-only records alone

**Premise:** RUNBOOK 215–219 requires both noise CCA and “the k-th event canonical correlation after subtracting the noise cross-covariance”; 499–501 says “Compute it once from noise-only records **before** training”.

**Consequence:** Noise-only records supply no event cross-covariance. Executors must invent a signal-data source or silently omit half the criterion. Moreover, the text does not specify whether subtraction affects only cross-covariance or also the marginal normalizers. Those choices can select different masks.

**Minimal remedy:** Freeze the exact estimator, event and noise sample counts/IDs, k, normalizers, ridge rule, and tie-break. Supply only training-side event records or a declared population model. Keep all mask selection and normalization estimation off probe-test data.

### E05 — P1: TIDMAD channel identity contradicts the named reader and upstream release

**Premise:** RUNBOOK 579–580: “ch0 is the injection reference; ch1 and ch2 are SQUIDs”; 594–595: “ch0 is never a model input”; 603: “channel: ch1 → ch2”.

**Evidence:** `src/tidmad_transformer/data.py:3–6,96–109` identifies `channel0001` as readout and `channel0002` as reference. The upstream [TIDMAD training loader](https://github.com/jessicafry/TIDMAD/blob/main/train.py#L46-L54), inspected during this review, loads channel0001 as training input and channel0002 as target. The rule excluding ch0 therefore does not exclude the actual injected reference.

**Consequence:** A literal ch1→ch2 split can use the injection reference as a model view, invalidating the real-noise claim. The named reader provides one readout and one reference, not two independently verified SQUIDs.

**Minimal remedy:** Freeze exact HDF5 paths, physical roles and file manifests using release metadata and inspected file headers. Assert exclusion of the reference path in every arm. Remove or redesign the channel split unless two readouts are verified in the selected files.

### E06 — P1: E3 lacks a reproducible training and evaluation contract

**Premise:** RUNBOOK 591–592 selects “4 train + 4 val + 16 science files” without IDs; 619–629 names five arms, a subclass and three seeds, then permits 96→24 frames and a 0–500 kHz crop; 632–634 uses injected-amplitude R².

**Evidence:** No E3-specific file/window assignments, probe-train/test allocation, step count, batch size, mask geometry after reduction, covariance estimator for `mae_chi2`, or hyperparameter transfer/selection rule appears in §7. The E2 event-ID split at 535–540 does not identify TIDMAD files/windows. The existing backbone expects `(B,2F,M)` with real/imaginary rows (`src/tidmad_transformer/model.py:31–35,107–123`), with no independent sensor axis. Existing config uses four-frame patches (`src/tidmad_transformer/config.py:79–81`); frame boundaries 46 and 50 do not both align with that patching. The existing config also records injections above 800 kHz in files 0016–0019 (`config.py:18–21`), so 0–500 kHz cannot be assumed to retain arbitrary selected injections.

**Consequence:** R9 could depend on file selection, overlapping or reused probe windows, arbitrary training budget, or removal of the injected signal. Two implementations can both follow the prose and produce different experiments. Multiple channel inputs require an explicitly defined adapter after E05 is resolved.

**Minimal remedy:** Register an E3 manifest containing checksums, file IDs and guarded sample intervals for training, calibration/mask selection, probe fitting and probe test; freeze model axes, masks for both window sizes, training schedule, inherited hyperparameters and loss covariance. Verify selected injection frequencies remain in the allowed crop before accepting the fallback. Freeze the amplitude extraction and require nonzero held-out target variance: if selected files have effectively fixed injection amplitude, ordinary amplitude R² is degenerate and needs an author-defined alternative or varying-amplitude design. This last issue is a required data validation, not a claim that observed digitized amplitudes are exactly constant.

### E07 — P1: compute limits do not bound the whole permitted execution

**Premise:** RUNBOOK 524–525 requires 500 steps of all nine arms before WP5; 550–556 benchmarks three arms and projects nine-arm grids; 627–629 shrinks E3 inputs when over its cap without specifying a second check or stop.

**Consequence:** The 4,500-step WP4 smoke precedes the compute gate. E2's projection does not explicitly include tuning runs, smoke, calibration, evaluation or the oracle preprocessing, and a benchmark of three primary-depth arms does not establish the cost of the depth sweep. E3 can still exceed its cap after the prescribed reduction, yet has no unambiguous stop condition. A projected cap alone does not prevent overruns.

**Minimal remedy:** Before substantial smoke/tuning, project and allocate the total campaign budget including all phases; benchmark representative depth/oracle work or use justified conservative bounds. Repeat E3 projection after the reduction and stop if still over. Define cumulative runtime accounting and an enforced remaining-budget stop for both campaigns. Keep G2 and GPU authorization independent.

## Concrete implementation and reproducibility repairs

### E08 — P2: the fresh Deepthought environment omits an import dependency

**Premise:** RUNBOOK 262–265 lists the environment packages and imports the backbones through `PYTHONPATH=src` without installing their dependencies.

**Evidence:** `src/reconstruction_model/model.py:9` imports `jaxtyping`; `src/reconstruction_model/pyproject.toml:12` declares it. README 220–224 includes it in the separate CPU environment recipe, but §3.2 does not.

**Consequence:** The stated fresh environment cannot import either backbone if only the listed packages are installed.

**Minimal remedy:** Include `jaxtyping` and perform both explicit backbone imports in preflight. Capture the resolved dependency inventory and CPU/GPU wheel versions. Pin the CPU torch version compatibly with GPU torch 2.5.1; §3.3's “Only the device differs” is not guaranteed by unpinned CPU installs. This is an environment repair, not a scientific redesign.

### E09 — P2: learned mask parameters are omitted by the inherited optimizer

**Premise:** RUNBOOK 469–470 adds a learned `[MASK]`; 480–481 reuses `configure_optimisers` and says “the predictor and heads go in the AdamW group”.

**Evidence:** `src/reconstruction_model/model.py:214–244` enumerates particular child modules rather than all trainable model parameters. Newly added mask/query parameters are not automatically included.

**Consequence:** A direct subclass using the inherited optimizer, with only the explicitly mentioned additions, leaves its learned mask (and potentially positional queries) fixed. Shape and leakage tests can pass anyway.

**Minimal remedy:** Override optimizer construction in `jepa_noise`, assign mask/query/predictor parameters explicitly, and assert every trainable non-EMA parameter occurs in exactly one optimizer group. This is a necessary implementation acceptance test rather than evidence of a bug in unimplemented JEPA code.

### E10 — P2: metric and tuning choices remain open after their claimed registration

**Premise:** RUNBOOK 114–118 specifies ridge CV without an alpha grid; 502–506 chooses β and λ_J “on the dev cell only” without an objective; 542–544 specifies “physics MSE/CRLB per parameter” and “nuisance R²” without aggregation.

**Consequence:** Selection by physics loss, nuisance suppression or a weighted combination can choose different remedies. `mean(error²)/mean(CRLB)` differs from `mean(error²/CRLB)`; separate nuisance scores, a uniform multi-output mean and a variance-weighted mean can change R6/R7. R² may be negative, so multiplicative retention thresholds need a declared interpretation. The problem is not the existence of tunable parameters; it is the absent deterministic rule for selecting and scoring them.

**Minimal remedy:** Freeze the alpha grid, fold assignment and preprocessing; dev selection objective/tie-break; nuisance target aggregation; exact MSE/CRLB formula; and R9 behavior when plain R²≤0 or target variance vanishes. Store per-target/per-seed values so reviewers can recompute every aggregate.

## Confirmed compatible elements and non-blockers

- E2's 8 channels and 8 patches match the inherited positional embedding dimensions (`model.py:149,165–167,261–262`). The required pre-pooling `(B,N,C,d)` endpoint exists (`model.py:265–269`). Returning this grid and replacing per-row normalization locally are feasible.
- The existing TIDMAD pipeline already preserves complex phase (`data.py:28–45,74–82`); no magnitude-only conversion is necessary.
- The CPU-only Mac recipe and no-touch wrappers are coherent. No protected package needs editing to resolve the interface issues.
- `PUSH_JEPA=no` and `GPU_ALLOWED=no` in PROMPT 6–7 are intentional authorization boundaries, not defects. Remote continuation requires G0 if the needed SHA is not published. No remote state was checked here.
- Provenance, independent review and raw-data retention requirements are useful. Recording full dependency versions, data hashes, split hashes and RNG state would strengthen reproducibility; dependency consistency and exact split manifests are already required by E06/E08.
- The log check at RUNBOOK 672–673 should use oldest-first order when checking that registration is first: `git log --reverse --format=%s 0cae6b0..JEPA`. This is a minor checklist wording fix, not a scientific blocker.

## Checks actually performed

Read both launch documents and relevant local source, inspected branch/HEAD/status, checked upstream TIDMAD loader read-only, and ran one bounded CPU inspection:

```text
PYTHONPATH=src .venv/bin/python [inline inspection]
psd[:4] [0.         0.00041634 0.00039708 0.00038988]
DC variance 0.0
temporal rank upper bound 255
patch3 samples 96 127
patch4 samples 128 159
```

No claim is made about live host availability, dataset presence, download size, throughput, torch runtime compatibility, numerical training stability or eventual R1–R9 outcomes. The report applies the markdown-mermaid-writing skill for a source-based Markdown review; a diagram is unnecessary for these localized findings. No prior memory result is used as current evidence.
