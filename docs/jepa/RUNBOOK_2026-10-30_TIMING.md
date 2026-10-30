# RUNBOOK — is the across-split trap trigger timing, and do deep encoders carry it? (30 Oct 2026, reviewed)

_To the JEPA agent on Deepthought (with GitHub push rights). From the author. Branch `JEPA`, at or
after `aaf5b99`. Commit this file unchanged as `docs/jepa/RUNBOOK_2026-10-30_TIMING.md` before any
run.
- Everything needed is in this file and in the repository; it depends on no outside document.
- "§6.1", "§6.2" and "S3" refer to `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md` in the repo.
- An independent review was applied before release. It regenerated the M-dose F1 s = 0.05 level
  locally and matched `F1_0.05.json` to 2.5e-13. Its preliminary numbers are quoted below as
  expectations, not as results._

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 30 Oct 2026, 10:46 CET:
   > "把a+b合成一份runbook，做独立审查"
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

This approves exactly Parts A and B below.

**Status.**
- Every result is an extension: `registered: false`, extension name `timing_2026-10-30`.
- Results go outside the registered result directories, and never replace a verdict, a
  prediction status or a frozen decision.
- **The grid stays paused.**
- **No training.** Part B loads frozen checkpoints and runs them **forward only**; that is
  allowed.
- The repository test suite is not an experiment: run it in full once per deployed SHA, including
  its small synthetic CPU training and forward tests.

### 0.1 Fix it yourself, do not stop to ask

Small fixes and ambiguities are yours to resolve. Do not stop for any of these:
- a bug in your own code;
- a failing test you can fix;
- a blocked tool, path or route;
- an environment or dependency problem;
- a data-format mismatch;
- an unclear scope between two statements;
- a step that is slower or larger than estimated.

For each one:
1. choose the best solution, which is the reading that best serves this runbook's purpose and is
   conservative for registered data and shared resources;
2. write one line in `DECISIONS.md` ("author standing approval, 29–30 Oct");
3. continue.

Report once, at the end (§5). "Not finished" is never a reason to stop.

**The chosen option must meet all four conditions:**
1. **Writes** go only inside `/ceph/dwong/jepa/`, and change only files this extension created.
   That includes archiving and then deleting its own temporary data.
2. It changes no registered prediction, criterion, cell, dataset, frozen decision or verdict, does
   not touch the grid, and **does not edit registered code**: `run.py`, `lucid_bombs.py`,
   `point_encoder.py`, the arms and the frozen configs are imported, called or copied, never
   modified.
   - A **runtime** test stub installed and restored inside a test (§3.4, test 4) is **not** an edit
     of registered code.
3. It stays within the budget below.
4. It writes nothing outside `/ceph/dwong/jepa/`, including `/tmp`, and touches nothing belonging
   to another user.

**Read-only inputs.**
- The repo clone: code, configs and geometry. The generator config hash must match
  `results/jepa/20261025_pregrid_checks/provenance.json`.
- The M-dose `.npz` files in `/ceph/dwong/jepa/returns/ext_m_dose_20261030/`. Verify them against
  `results/jepa/20261030_m_dose/npz_hashes.json`.
- The frozen checkpoints of §3.1:
  - some live under `/work/dwong/jepa_a10_work/`, which the author authorised on 22 Oct; read
    only;
  - the others live under `/ceph/dwong/jepa/returns/`.

**Stop and report only for:**
- a real S3 change;
- an action that fails conditions 1–4;
- **S4**: no route can get a commit to GitHub. Before calling S4, try in order:
  1. `git push origin JEPA` from `~/Documents/VERITAS/jepa-handoff-20261028`. The repo is
     `seanxia8/VERITAS`; the old `oracle` URL redirects.
  2. A Deepthought clone with credentials.
  3. An authenticated `gh`.

  An API or integration 403 is not S4.
- **S8**: the §2.2 reproduction fails after one fix attempt.

**Rules that still hold.**
- §6.1 and §6.2: code reaches the hosts only through GitHub; one shallow or sparse clone per SHA;
  remove a superseded extension clone after the next passes its smoke; the guard; the no-touch
  packages.
- `TMPDIR=/ceph/dwong/jepa/tmp/` for every host command. Pytest uses a `--basetemp` inside
  `<ext>/tmp/` and `-p no:cacheprovider`, removed after a successful run.

**Budget.**
- CPU: ≤ 6 wall-hours on Deepthought, ≤ 16 cores, `nice -n 10`, `OMP_NUM_THREADS=4`. Generation
  also counts against the 48 h generation cap; book it.
- The review's estimate is well under 1 h in total: Part A takes minutes; the de-jitter
  regeneration is about 12 min; Part B generation is about 3 min; 8 forward passes take about
  15 min.
- Titan: ≤ 1 GPU-h through the guard, and only if the forward passes would exceed 3 CPU-h. This is
  not expected.
- L40S: 0.
- Storage: ≤ 1 GB and ≤ 1 000 new entries under
  `/ceph/dwong/jepa/returns/ext_timing_20261030/`, which is `<ext>` below.
- Write nothing if free entries are below 30 000.
- Never store raw events.

---

## 1. Why

The M-dose extension (`aaf5b99`) found two things:
- **In `within`, module-gain drift never enters the kept CCA pairs.** ν\*_gain ≤ 0.098, even under
  unphysical F1x s = 1.50.
- **In `across_fixed`, the weakest kept pair is already about half nuisance with no drift.**
  ν\* ≈ 0.52 at s = 0, almost entirely from the `A` blocks (state fixed): ν\*(A) ≈ 0.51 against
  ν\*_gain ≈ 0.02.

**Hypothesis: trigger-time jitter.**
- Token time is t_digit − t_trigger, and unhit PMTs have time 0. A noise-driven shift δ of the
  NHits trigger therefore moves every hit time by −δ.
- In module-pooled coordinates the shift is −δ·h_{e,g}/100, where h_{e,g} is the event's hit
  fraction in group g. That is a shared nuisance, but its **direction varies from event to
  event** with the hit pattern.
- It displaces the weakest physics pair when each view sees half the detector: σ_k^phys is 0.95
  in `across_fixed` against 0.996 in `within`.

**What the review already saw** on its regenerated F1 s = 0.05 level, as ν\*(A):
- `across`: `all` 0.492, `no_time` 0.0004, de-jittered 0.0002;
- `within`, `time_only`: 0.19, which falls to 0.0000 when de-jittered.

Part A must confirm this on the stored data and across all levels. Part B asks whether the deep
encoders carry δ. In Part D, the deep `across` encoders had the worst vertex resolution:
0.33–0.35 m, against 0.19–0.22 m for `within`.

---

## 2. Part A — attribute the across trap (CPU)

### 2.1 Inputs and layout

**Per-level `.npz` keys** (`scripts/jepa_ext_m_dose.py`, around l.300–309), for split ∈
{`within`, `across_fixed`}:
- `{split}_S`, `{split}_A`, `{split}_B`: full joint covariances, context coordinates first;
- `{split}_null`: only the cross block is filled;
- `{split}_context_dim`, `{split}_target_dim`, `{split}_means`.

N_gain = B − A.

**Layout** (`scripts/jepa_ext_m_dose.py::pool` / `pooled`, l.168–176):
- pooled index = group × 3 + f, with f = 0 for log charge, 1 for time and 2 for hit;
- context groups come first, then target groups;
- dimensions are 315 / 315 for `within` and 156 / 159 for `across_fixed`.

**Verify the layout against the code** and record it.

### 2.2 Reproduction (S8)

Recompute ν\* (`S` with `B`), ν\*(A), ν\*_gain, σ_k^phys, σ_1^noise and r for every level and split.
Use `ext_m_dose.pair_diagnostics` and `ext_m_dose.m_terms`, imported, not reimplemented. They must
equal `results/jepa/20261030_m_dose/*.json` within 1e-9.

### 2.3 Variants on the stored covariances (all 10 levels × both splits)

Each variant is a linear map T per view, applied to every block (T_c·X·T_cᵀ, T_c·X·T_tᵀ and so
on). `pair_diagnostics` recomputes ε on the transformed blocks. Keep k = 4.

| Variant | Map |
|---|---|
| `all` | identity, i.e. the reproduction |
| `time_only` / `charge_only` / `hit_only` | keep only that feature's coordinates |
| `no_time` | drop the time coordinates |
| `time_common_removed` | per view, project the time coordinates onto the complement of the all-ones vector |
| `time_hit_removed` | per view, project out of the time coordinates the top 3 eigenvectors of E[h hᵀ] = (the hit-coordinate block of S) + m_h m_hᵀ, with m_h = the hit part of `{split}_means` averaged over rows. This approximates the event-varying jitter direction h·δ |

For each variant report:
- ν\* under B, A and gain;
- σ_k^phys, σ_1^noise and r;
- the pair index j that attains ν\*.

### 2.4 Exact de-jitter (decisive; small regeneration)

1. Regenerate the **calibration set** of levels `F1_0.00` and `F1_0.05` with the M-dose `collect`
   code path (same IDs, seeds and namespace).
2. Record `trigger_ns` for every realisation. The readout returns a `HitReadout` with `trigger_ns`
   (`lucid_bombs.py` ≈ l.2160). Call the readout directly, as `collect` does.
   `tokenise_realisations` discards the readouts, so do not use it for this.
3. **Check first** that the regenerated `A` and `B` blocks equal the stored ones within 1e-9.
4. Build the **`A_dejitter`** and **`B_dejitter`** blocks: before accumulating, add
   h_{g,r}·(trigger_r − mean_r trigger)/100 to each pooled time coordinate. Here h_{g,r} is that
   realisation's hit fraction in group g; this undoes the per-realisation trigger shift exactly.
5. Report ν\*(A) and ν\*(B) for `all` and `time_only` on both splits.

### 2.5 Alignment and figure

**Alignment.** For each split at s = 0, use the `A` blocks:
- a₁ is the top left singular vector of H_noise;
- the pattern is p = (N_cc + ε)·(N_cc + ε)^(−1/2) a₁;
- report cos(p, 1_time) and cos(p, m_h restricted to the time coordinates).

**Figure.** Redraw as `nu_vs_r_v2.png`:
- toy curve;
- LUCiD `within`;
- **LUCiD `across_fixed`**, which was missing;
- the floor band drawn from the actual ν\*_null values (mean + 2 SD over levels).

Add a second panel with the same data against r^S.

---

## 3. Part B — do the deep encoders carry the jitter? (CPU forward)

### 3.1 Encoders (frozen; forward only)

| Label | Path | How to verify |
|---|---|---|
| `init9000` | random init, seed 9000 | — |
| `fix9000`, `fix9001` | `checkpoint` in `results/jepa/20261029_predictability/probes/fix900*.json` | the sha256 recorded there |
| `mae9000`, `ijepa9000`, `ijepa9001` | the same directory, `probes/<label>.json` | the sha256 recorded there |
| `across9000`, `across9001` | `/ceph/dwong/jepa/returns/ext_pregrid_20261025/train/across_s900X/runs/ext_dev_s005_across__sigreg_jepa__s900X/final.pt` | no sha256 was recorded. Confirm that the sibling `ext_summary.json` matches `results/jepa/20261029_d2_partial/SHA256SUMS`, then compute and record the checkpoint sha256 |

**Loading.** Load exactly as `scripts/jepa_ext_predictability_probes.py` does (l.45–47):
`pregrid.build_loaded(...)`, then `getattr(model, "jepa", model)`, in eval mode.
- `across` checkpoints use `cell = variant_cell(config, "across")`;
- the others use the dev cell.

### 3.2 Data: jitter truth by resampling (photon only)

**Events.** 1 000 photon events at F1 s = 0.05, using the M-dose event-set IDs
(`ext_m_dose/evt/...`), runs 0–99 × 10 events. Build them as the M-dose code does:
`draw_event(..., domain="photon")`, then `make_readout`, then `tokenise`, with `ViewMasker`
n_min = 0.

**Resampling.**
- For each event, R = 16 scheme-A realisations (state fixed, noise redrawn) via
  `ext_predictability.resample_noise`. r = 0 is the original.
- Call the readout directly and keep each `HitReadout`'s `trigger_ns`, `source_trigger_ns` and
  whether a dark coincidence moved the trigger.
- Exclude an event if **any** realisation has `trigger_ns is None`, and report the count. The
  review saw none.

**Targets per event.**
- **δ_r** = trigger_ns(r) − mean over r of trigger_ns. The probe target is δ = δ_0.
- **μ_t** = mean over r of trigger_ns − `draws.emission_offset_ns`.
  - The raw mean trigger is about 99.97% the A9 emission offset (SD 29 ns, against 0.73 ns without
    it).
  - Tokens cancel that offset by design, so it must be subtracted.

**Encoder input.** Realisation r = 0 only, full input, the same tokens for every encoder.

**Dark-moved triggers.** About 3% of realisations have one; |δ| reaches 17 ns there, against a
typical SD of 1.2 ns. Report every probe **with and without** events whose r = 0 trigger was moved
by a dark coincidence.

### 3.3 Probes

**Event vector.** `tokens.mean(1)` over all 1 995 read-out tokens, as in
`jepa_pregrid_checks.event_level` (l.184–185).

**Probe.** `ridge_r2` from `scripts/jepa_pregrid_checks.py`. Train on runs 0–49, test on runs
50–99.

**Targets.** R²(δ), R²(μ_t), R²(vertex) and R²(log yield).

**Ceilings and floors.**
- **Raw-input ceiling:** the same probes on the module-pooled raw features (`pool` layout,
  `within` groups).
- **Hit-token mean time:** the same probes on the mean token time over hit PMTs.
- `init9000` gives the random-init floor. It keeps token time: on 29 Oct its time R² was 0.61,
  against 0.01 for `fix9000`.

### 3.4 Acceptance tests (CPU, before the host run)

Add `src/jepa_noise/tests/test_ext_timing.py`:
1. The §2.3 transform of a synthetic covariance set equals the covariance of the transformed
   samples within 1e-10.
2. `time_common_removed` maps a pure common time shift to zero.
3. Σ_r δ_r = 0 per event within float tolerance.
4. **Jitter → token path.**
   - Install a runtime wrapper in `lb._PMT_TIME_RULE`, the documented test stub, whose
     `nhits_trigger_time` returns the real value + c, with |c| ≤ 5 ns so the prompt cut drops
     nothing. Restore it in `finally`.
   - Compare only PMTs kept in both readouts: their token times shift by exactly −c/100 in the
     `time_100ns` unit.
   - This does not modify registered code.
5. The §2.4 de-jitter on one fixture event sets the pooled time residual's dependence on the
   trigger shift to zero.

Run the full suite once on the deployed SHA.

---

## 4. Pre-declared readings (descriptive, not gates)

Report each one as **supported**, **not supported** or **unclear**, with the numbers. Write any
further tolerance into `DECISIONS.md` **before** computing.

| ID | Reading | Supported if |
|---|---|---|
| T-a | The across trap lives in the time features | `across_fixed`, s = 0: ν\*(A) ≤ 0.1 in `no_time`. The ≥ 0.4 in `all` is just the reproduction; note that |
| T-b | It is the trigger jitter | `across_fixed`, s = 0 **and** s = 0.05: ν\*(A_dejitter) ≤ 0.1 in both `all` and `time_only` |
| T-c | `within` is protected in the full feature set | `within`, s = 0: ν\*(A) ≤ 0.05 in `all`, `no_time` and `time_common_removed`. `time_only` is reported descriptively, because the same jitter shows there (≈ 0.19 in the review) |
| T-d | Deep `across` encoders carry the jitter | In both `across` seeds: R²(δ) ≥ max(`fix` seeds' R²(δ)) + 0.1 **and** ≥ `init9000` R²(δ) + 0.1. **Unclear** if the raw-input ceiling R²(δ) < 0.2. Judge on the set without dark-moved triggers |
| T-e | Jitter content goes with vertex loss | Descriptive only, no criterion: over all encoders, Spearman between R²(δ) and the Part C/D vertex resolution (higher means worse) |

---

## 5. Write-up, commit, report, wait

1. Results go in `results/jepa/20261030_timing/`, with a README marked `registered: false`, plus:
   - `attribution.md`: Part A, levels × splits × variants, with ν\* (B / A / gain), σ_k^phys,
     σ_1^noise, r, the pair index, the alignment cosines and the de-jitter rows;
   - `deep_jitter.md`: Part B, encoders × {R²(δ), R²(μ_t), R²(vertex), R²(log yield)}, with and
     without dark-moved triggers, plus the ceiling and floor rows;
   - `nu_vs_r_v2.png`;
   - JSON summaries, each ≤ 1 MB.
2. Large arrays stay in `<ext>`, with their hashes recorded.
3. `DECISIONS.md` gets one line per engineering choice. `STATE.md` gets a short "timing" entry.
   Book CPU and any GPU in `RESOURCE_LEDGER.json`.
4. Commit and push after Part A, and again at the end.
5. **Report to the author (≤ 1 page, in Chinese):**
   - the two tables, condensed;
   - the figure;
   - the five readings with numbers;
   - resources used;
   - the final SHA.
6. **Wait for the author.** Do not resume the grid.
