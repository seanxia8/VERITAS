# RUNBOOK — does the de-jitter prescription help a model that actually uses time? MAE on LUCiD (31 Oct 2026, reviewed)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the author.
Branch `JEPA`, at or after `2b91f42`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-10-31_MAE_DEJITTER.md` before any run.
- Everything needed is in this file and in the repository; it depends on no outside document.
- "§6.1", "§6.2" and "S3" refer to `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`.
- This runbook reuses the de-jitter extension (`docs/jepa/RUNBOOK_2026-10-30_DEJITTER.md`,
  `scripts/jepa_ext_dejitter.py`, `src/jepa_noise/ext_dejitter.py`).
- An independent review checked it against the code at `2b91f42`. The facts quoted below with
  line numbers come from that review._

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 31 Oct 2026, 22:11 CET:
   > "1. 你写runbook， 2 3你开始动手写"

   Item 1 of the author's list was this MAE de-jitter test, including its training.
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

**Training approved:**
- the eight §3.2 runs;
- one 20-step smoke;
- the synthetic CPU training inside tests.

No other training.

**Status.**
- Every result is an extension: `registered: false`, extension name `mae_dejitter_2026-10-31`.
- Results go outside the registered result directories, and never replace a verdict, a
  prediction status or a frozen decision.
- **Never overwrite** `results/jepa/20261030_dejitter/` or
  `/ceph/dwong/jepa/returns/ext_dejitter_20261030/`.
- **The grid stays paused.**
- Run the full suite once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Small fixes and ambiguities are yours to resolve. Do not stop for any of these:
- a bug in your own code;
- a failing test you can fix;
- a blocked tool, path or route;
- an environment or dependency problem;
- a data-format mismatch;
- an unclear scope;
- a step that is slower or larger than estimated;
- a guard HOLD, during which you wait.

For each one:
1. choose the best solution, which is the reading that best serves this runbook's purpose and is
   conservative for registered data and shared resources;
2. write one line in `DECISIONS.md` ("author standing approval, 29–31 Oct");
3. continue.

Report once, at the end (§5). "Not finished" is never a reason to stop.

**The chosen option must meet all four conditions:**
1. **Writes** go only inside `/ceph/dwong/jepa/`, and change only files this extension created.
   That includes archiving and then deleting its own temporary data.
2. It changes no registered prediction, criterion, cell, dataset, frozen decision or verdict, does
   not touch the grid, and **does not edit registered code or frozen configs**.
   - A runtime wrapper or test stub inside an extension process is not an edit. The de-jitter
     `median_time` wrapper is one.
   - Extension scripts (`scripts/jepa_ext_*.py`, `src/jepa_noise/ext_*.py`) may be extended with
     backward-compatible options. Existing defaults must keep their current behaviour.
3. It stays within the budget below.
4. It writes nothing outside `/ceph/dwong/jepa/`, including `/tmp`, and touches nothing belonging
   to another user.

**Read-only inputs.**
- The repo configs, with hashes per `DECISIONS.md` 2026-10-25 "Exact dev inputs".
- The data root `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`.
- The de-jitter Part 1 arrays, if present. Otherwise see §3.3.
- The de-jitter checkpoints, with hashes in `results/jepa/20261030_dejitter/large_arrays.json`.

**Stop and report only for:**
- a real S3 change;
- an action that fails conditions 1–4;
- **S4**: no route can get a commit to GitHub. Before calling S4, try in order:
  1. `git push origin JEPA` from `~/Documents/VERITAS/jepa-handoff-20261028`. The repo is
     `seanxia8/VERITAS`.
  2. A Deepthought clone with credentials.
  3. An authenticated `gh`.

  An API or integration 403 is not S4.
- **S9**: a §3.4 acceptance test, or the §3.3 regeneration check, fails after one fix attempt.
- **S1**: the L40S ledger plus projections would exceed 160 GPU-h. Today: 97.17 + grid projection
  50.57 + this runbook ≤ 8 = 155.7.

**Rules that still hold.**
- §6.1 and §6.2: GitHub-only code path, one shallow or sparse clone per SHA; the guard; ≤ 2 jobs on
  deepthought2; fp32 with TF32 off; one GPU type per comparison set (all training here on L40S);
  the no-touch packages.
- `TMPDIR=/ceph/dwong/jepa/tmp/`. Pytest uses `--basetemp` inside `<ext>/tmp/` and
  `-p no:cacheprovider`, removed after success.

**Budget.**

| Resource | Cap | Notes |
|---|---|---|
| L40S | ≤ 8 GPU-h, counted as charged wall plus active reservations; ≤ 2 jobs; timeout 90 min per run | MAE trains in 1265–1536 s, plus about 550 s of evaluation, so about 0.5–0.6 h per run and about 4.2–4.9 h in total |
| Titan | 0 expected | the floors are reused (§3.3) |
| CPU | ≤ 5 projected wall-hours on Deepthought, `nice -n 10`, `OMP_NUM_THREADS=4` | the sensitivity path, time probes and tests |
| Storage | ≤ 5 GB and ≤ 3 000 new entries under `/ceph/dwong/jepa/returns/ext_mae_dejitter_20261031/`, which is `<ext>` below | write nothing below 30 000 free entries |

---

## 1. Why

The de-jitter extension (`2b91f42`) found three things:
1. **`median_time` removes the linear `across` trap.** ν\*(A) goes from 0.51 to 0.001.
2. **It does not rescue deep SIGReg-JEPA `across`.** Vertex goes from 0.348 / 0.333 m to
   0.348 / 0.320 m. Those encoders are collapsed (token charge R² about 0) and barely use token
   time.
3. **MAE is the model that does use time.** Its jitter sensitivity is R_jit/phys = 1.64, against
   about 0.01 for SIGReg-JEPA. It keeps per-token charge (R² 0.988), and its `within` vertex is
   0.105 m.

**The question.** Does the prescription help a model whose representation carries the
jitter-bearing feature?

---

## 2. Comparisons

| Question | Comparison |
|---|---|
| Does `across` hurt MAE? | `mae_across_trigger` vs `mae_within_trigger` |
| Does the prescription help MAE `across`? | `mae_across_median` vs `mae_across_trigger` |
| Does it harm MAE `within`? | `mae_within_median` vs `mae_within_trigger` |
| Do the MAE trigger encoders feel the jitter? | their R_jit/phys against the Part 1 references |

---

## 3. Implementation and runs

### 3.1 Code changes (extension scripts only, backward compatible)

In `scripts/jepa_ext_dejitter.py`:

1. **`--arm {sigreg_jepa, mae_mse}`**, default `sigreg_jepa`. Replace the hard-coded
   `arm_name = "sigreg_jepa"` in `mode_train`.
2. **New variant `across_trigger`.**
   - Add it to `TRAIN_VARIANTS`, which is also the argparse `choices` (≈ l.57, l.991).
   - It maps to `cells_for(config)["across"]`, i.e. cell `ext_dev_s005_across`, with default
     transform `none`.
   - `main()` currently leaves the transform None for unknown variants (≈ l.1011–1014); make the
     mapping explicit.
3. **Keep the existing cell IDs; add no suffix.**
   - `run_id` already contains the arm (`{cell}__mae_mse__s9000`, `lucid_run.py` ≈ l.202–204).
   - The output root is new: `--out <ext>/train`.
4. **Guard against auto-resume.** `run_one` resumes whenever `checkpoint.pt` exists
   (`lucid_run.py` ≈ l.811–812).
   - Before each §3.2 launch, **refuse** if `<ext>/train/runs/<run_id>/` already exists.
   - The smoke uses `--out <ext>/smoke`.
5. **Extension name and paths.** `EXT`, `RESULTS` and the summary's `"dejitter_2026-10-30"` are
   hard-coded (≈ l.39, l.43, l.58, l.689).
   - Add `--ext` and `--extension` options whose defaults keep today's values.
   - Pass `<ext>` and `mae_dejitter_2026-10-31` everywhere in this runbook.
6. **Report.** Add a new **`report-mae`** mode that writes only to
   `results/jepa/20261031_mae_dejitter/`. **Never run the existing `report` mode**: it overwrites
   the de-jitter results.

**What already works** (verified in review):
- **MAE targets.** The `mae_mse` loss reads its targets (log charge, time, hit) from
  `batch.features` at the hidden indices (`lucid_arms.py` ≈ l.494–501). Those features come from
  the wrapped `pe.tokenise`, so `median_time` reaches them. No path bypasses it.
- **Loading.** `build_loaded` and `evaluate_forms` work for `mae_mse`, which needs no dark bank
  and has β None. Part C evaluated `mae9000` this way.

| Run label | Command options |
|---|---|
| `mae_across_trigger` | `--arm mae_mse --variant across_trigger` |
| `mae_across_median` | `--arm mae_mse --variant across_median` |
| `mae_within_trigger` | `--arm mae_mse --variant within_trigger` |
| `mae_within_median` | `--arm mae_mse --variant within_median` |

### 3.2 Runs (deepthought2, L40S; the only training approved)

**Configuration.**
- `mae_mse`, the frozen dev config, 10 000 steps.
- The same `--config`, `--generator-config`, `--geometry` and `--data-root` as the de-jitter runs,
  with `--out <ext>/train`.

**Order of priority.** The last pair is dropped first if the budget runs short.

| # | Run | Seeds |
|---|---|---|
| 1 | `mae_across_trigger` | 9000, 9001 |
| 2 | `mae_across_median` | 9000, 9001 |
| 3 | `mae_within_trigger` | 9000, 9001 |
| 4 | `mae_within_median` | 9000, 9001 |

**Smoke first.** A 20-step smoke of `mae_across_median` 9000 into `--out <ext>/smoke`
(`--steps 20 --probe-max-events 200`): no NaN, and `ext_summary.json` is written.

**Running.** Check the guard before every launch; at most 2 at a time; timeout 90 min. Book every
run as an extension line in `RESOURCE_LEDGER.json`.

### 3.3 Evaluation

**For each of the 8 MAE encoders:**
- vertex resolution and module-gain run-mean R² (the `ext_summary.json` protocol, `full_input`);
- effective rank, event SIGReg and hit-token log-charge R² (`evaluate_forms`). `half_pass` is
  **descriptive only** for MAE, because MAE never trains a target-only pass.

**Token time R².**
- Write an extension copy of `event_level`. With the same token picks, it also returns
  `features[..., TIME_INDEX]` for hit tokens, taken after the wrapper where applicable. Then run
  `ridge_r2` on it.
- Report two targets for every encoder:
  - the time in its own convention (trigger-relative or median-relative);
  - median-relative time, as a common target.
- Also compute it, forward-only on CPU, for:
  - `init9000` in both conventions;
  - the six de-jitter SIGReg-JEPA checkpoints, with hashes from `large_arrays.json`.

**Counterfactual sensitivity** (the de-jitter Part 1 path, made reusable):
- **Code.** Add `--ext`, `--checkpoint`, `--arm`, `--cell` and `--transform` options to
  `sens-encode` and `sens-aggregate`. `EXT` is hard-coded, `encoder_specs()` knows only the 8 old
  labels, and `sensitivity.json` is opened with `"xb"`. Write every output under `<ext>`.
- **Inputs.** Read the old Part 1 input arrays (`features_*.npy`, `meta.npz`) read-only if they are
  still present. They are probably gone: the old extension finished at 0.08 GB. If so, regenerate
  them with `sens-gen` into `<ext>/part1/sens` (same IDs and seeds).
- **Regeneration check (S9).** Re-encode `mae9000` and `fix9000` and require R_jit/phys to match
  the recorded 1.6388 and 0.0111 within 1e-3 relative.
- **Median-time encoders.** Apply `ed.median_time_features` to **every** input (x, x_jit±,
  x_noise, x_phys) before encoding. Their R_jit/phys is ≈ 0 by construction, because a constant
  trigger shift is removed exactly; label it "edge effects only" and do not score it.

**Floors** (no rerun is needed). At the same seed the encoder initialisation is bit-identical
across arms, which the review checked:
- trigger time: 0.16007 m (Part C `init9000`);
- median time: 0.159692 m (de-jitter).

### 3.4 Acceptance tests (CPU, before the host runs)

Add to `src/jepa_noise/tests/test_ext_dejitter.py`, keeping its existing tests:
1. **Defaults unchanged.** `--arm` defaults to `sigreg_jepa`; existing variants, cell IDs, run
   directories, `EXT`, `RESULTS` and the extension name are unchanged by default; `across_trigger`
   maps to cell `ext_dev_s005_across` with no transform.
2. **MAE loss is shift-invariant under the wrapper; not vacuous.**
   - Write a synthetic `hits.npz` and tokenise it twice through the wrapped `pe.tokenise`, the
     second time with `time_ns + 37` for every hit.
   - Require `metrics['hidden_hits'] > 0` and |Δloss| ≤ 1e-5 · loss.
   - **Negative control:** without the wrapper, the same shift changes the loss by > 1e-3.
   - **Target identity:** with `arm.head` zeroed, `metrics['time']` equals the mean of the squared
     median-subtracted hidden hit-token time, computed from
     `ed.median_time_features(raw.features)`, to 1e-6.
3. **Wiring.** A 2-step synthetic CPU training of `mae_mse` with the wrapper completes, and the
   wrapper is removed afterwards.
4. **Run-directory guard.** The guard refuses an existing run directory.

Run the full suite once on the deployed SHA.

---

## 4. Pre-declared readings (descriptive, not gates)

Report each one as **supported**, **not supported** or **unclear**, with the numbers. With two
seeds, no seed-floor claim is made; report both seeds. Write any further tolerance into
`DECISIONS.md` **before** computing. Every reading uses `full_input`.

| ID | Reading | Supported if |
|---|---|---|
| M-a | `across` hurts MAE | vertex(`mae_across_trigger`) ≥ 1.2 × vertex(`mae_within_trigger`), same seed, in both seeds |
| M-b | The prescription helps MAE `across` | vertex(`mae_across_median`) ≤ 0.85 × vertex(`mae_across_trigger`), same seed, in both seeds. **If M-a is not supported, report `unclear (no gap)`** |
| M-c | It closes the MAE split gap | vertex(`mae_across_median`) ≤ 1.1 × vertex(`mae_within_trigger`), same seed, in both seeds. **If M-a is not supported, report `unclear (no gap)`** |
| M-d | It does not harm MAE `within` | vertex(`mae_within_median`) ≤ 1.1 × vertex(`mae_within_trigger`), same seed, in both seeds |
| M-e | The MAE trigger encoders feel the jitter | Both `mae_across_trigger` seeds **and** both `mae_within_trigger` seeds have R_jit/phys ≥ 10 × max(fix9000, fix9001, across9000, across9001) as recorded in de-jitter Part 1 = 10 × 0.0495 = **0.495**. Report the whitened version against 10 × 0.0622. Report `mae9000` (1.639) as the prior; the `within` case is close to a replicate of it |
| M-f | Descriptive | Token charge R², token time R² (both targets), effective rank and gain R² for the 8 MAE encoders, against the floors and the de-jitter SIGReg-JEPA runs |

---

## 5. Write-up, commit, report, wait

1. Results go in `results/jepa/20261031_mae_dejitter/` (written by `report-mae` only), with a
   README marked `registered: false`, plus:
   - `mae_prescription.md`: the 8 encoders against references and floors, the time-R² table and
     the sensitivity table;
   - JSON summaries, each ≤ 1 MB.
2. Checkpoints and large arrays stay in `<ext>`, with their hashes in `large_arrays.json`.
3. `DECISIONS.md` gets one line per engineering choice. `STATE.md` gets a short "MAE de-jitter"
   entry. Book every resource.
4. Commit and push after the smoke, and at the end.
5. **Report to the author (≤ 1 page, in Chinese):**
   - the tables, condensed;
   - the six readings with numbers;
   - GPU-h per host and the ledger;
   - the final SHA.
6. **Wait for the author.** Do not resume the grid.
