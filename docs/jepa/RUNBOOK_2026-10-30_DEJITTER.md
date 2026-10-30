# RUNBOOK — do deep encoders feel the trigger jitter, and does removing it fix `across`? (30 Oct 2026, reviewed)

_To the JEPA agent on Deepthought and deepthought2 (with GitHub push rights). From the author.
Branch `JEPA`, at or after `9bded49`. Commit this file unchanged as
`docs/jepa/RUNBOOK_2026-10-30_DEJITTER.md` before any run.
- Everything needed is in this file and in the repository; it depends on no outside document.
- "§6.1", "§6.2" and "S3" refer to `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md` in the repo.
- An independent review checked the code paths, the reference values and the ledger before
  release. Its local measurements are quoted below as expectations, not results._

## 0. Authority, scope, how to work

**Authority.** Quote both verbatim in `DECISIONS.md`.
1. The author, 30 Oct 2026, 19:45 CET:
   > "要，先更新周报，再把 1 和 2 合成一份 runbook，照样先做独立审查。第 2 项要训练，我点头。"
2. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

This approves exactly Parts 1 and 2 below. **Training approved:**
- the six §3.3 runs;
- the one §3.5 smoke;
- the synthetic CPU training inside tests.

No other training.

**Status.**
- Every result is an extension: `registered: false`, extension name `dejitter_2026-10-30`.
- Results go outside the registered result directories, and never replace a verdict, a
  prediction status or a frozen decision.
- **The grid stays paused.**
- The repository test suite is not an experiment: run it in full once per deployed SHA.

### 0.1 Fix it yourself, do not stop to ask

Small fixes and ambiguities are yours to resolve. Do not stop for any of these:
- a bug in your own code;
- a failing test you can fix;
- a blocked tool, path or route;
- an environment or dependency problem;
- a data-format mismatch;
- an unclear scope;
- a step that is slower or larger than estimated;
- a guard HOLD, during which you wait rather than stop.

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
   not touch the grid, and **does not edit registered code or frozen configs**.
   - A **runtime** wrapper or test stub installed inside an extension process is not an edit.
     Examples: wrapping the `jepa_noise.point_encoder.tokenise` module attribute (§3.1), or
     `ext_timing.shifted_time_rule`.
   - Existing extension scripts may be imported, or copied into new extension scripts.
3. It stays within the budget below.
4. It writes nothing outside `/ceph/dwong/jepa/`, including `/tmp`, and touches nothing belonging
   to another user.

**Read-only inputs.**
- The repo clone. The two config hashes must match `DECISIONS.md` 2026-10-25 "Exact dev inputs":
  - `lucid_run_v0_mixed_calibrated.json`: `e6730ac1…`;
  - `lucid_bombs_v2_calibrated.json`: `90ac1601…`.
- The frozen checkpoints recorded by the timing extension. Some of them live under
  `/work/dwong/jepa_a10_work/`, which the author authorised on 22 Oct; read them only.
- The extension dev data root `/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`, reused as Part D
  used it. Nothing regenerates there.
- The M-dose blocks, via `scripts/jepa_ext_timing.py::load_level`.

**Stop and report only for:**
- a real S3 change;
- an action that fails conditions 1–4;
- **S4**: no route can get a commit to GitHub. Before calling S4, try in order:
  1. `git push origin JEPA` from `~/Documents/VERITAS/jepa-handoff-20261028`. The repo is
     `seanxia8/VERITAS`.
  2. A Deepthought clone with credentials.
  3. An authenticated `gh`.

  An API or integration 403 is not S4.
- **S9**: a §2.4 or §3.5 acceptance test fails after one fix attempt.
- **S1**: the L40S ledger plus the projection would exceed 160 GPU-h. Today the charged 91.77 h
  plus the grid projection of 50.57 h plus this runbook's 7 h comes to about 149 h.

**Rules that still hold.**
- §6.1 and §6.2: code reaches the hosts only through GitHub, one shallow or sparse clone per SHA;
  the guard and its start conditions; at most 2 jobs on deepthought2; fp32 with TF32 off; one GPU
  type per comparison set; the no-touch packages.
- `TMPDIR=/ceph/dwong/jepa/tmp/` for every host command. Pytest uses a `--basetemp` inside
  `<ext>/tmp/` and `-p no:cacheprovider`, removed after a successful run.

**Budget.**

| Resource | Cap | Use |
|---|---|---|
| CPU | ≤ 8 projected wall-hours on Deepthought, ≤ 16 cores, `nice -n 10`, `OMP_NUM_THREADS=4` | Part 1 needs about 120 k forward passes, about 3.3 h in one process, so parallelise. Generation also counts against the 48 h cap |
| L40S (deepthought2) | ≤ 7 GPU-h, counted as charged wall time **plus active reservations**. ≤ 2 jobs at a time, **timeout 90 min per run** | the six §3.3 runs, the smoke, and their evaluation. Part D's runs charged 0.72–0.96 GPU-h each; expected use is about 5 h |
| Titan | ≤ 2 GPU-h, through the guard | the `median_time` random-init floors (§3.4); Part 1 forward passes only if CPU would exceed the cap |
| Storage | ≤ 5 GB and ≤ 3 000 new entries under `/ceph/dwong/jepa/returns/ext_dejitter_20261030/`, which is `<ext>` below | write nothing if free entries are below 30 000 |

---

## 1. Why

The timing extension (`9bded49`) found two things:
- **The linear trap in `across_fixed` is NHits trigger-time jitter.** ν\*(A) falls from 0.51 to
  0.0004 without time features, and to 0.0002 when the trigger shift is undone.
- **Whether deep encoders carry it was unclear.** Once dark-moved events are removed, the jitter
  (SD about 1 ns) was not linearly decodable at 1 000 events, even from the raw input: the
  ceiling R² was −0.02.
  - The review traced this partly to the **target definition**. δ used the mean over the
    realisations, and only r = 0 was checked for dark moves. A robust δ, the median over the
    realisations, decodes far better: raw R² 0.035, 0.118 and 0.186 at 1 k, 2 k and 4 k events.

This runbook has two parts:
- **Part 1** measures jitter sensitivity directly, and repeats the probe at 10 000 events with
  the robust target.
- **Part 2** tests the prescription the theory implies: remove the shared readout nuisance at
  tokenisation. Deep `across` vertex is 0.348 / 0.333 m today, against 0.219 / 0.204 m for
  `within`.

---

## 2. Part 1 — jitter sensitivity of the frozen encoders (CPU; never install the §3.1 wrapper here)

### 2.1 Encoders

The timing extension's eight: `init9000`, `fix9000`, `fix9001`, `mae9000`, `ijepa9000`,
`ijepa9001`, `across9000` and `across9001`.
- Use their recorded paths and sha256.
- Load each with its own cell via `build_loaded`, then `getattr(model, "jepa", model)`, in eval
  mode.
- The event vector is the mean over all read-out tokens.

### 2.2 Counterfactual sensitivity (1 000 events)

**Events.** The timing extension's 1 000 photon events: `ext_m_dose/evt`, runs 0–99 × 10, F1
s = 0.05. Exclude an event if **any** of its realisations used here has a dark-moved trigger, and
report the count.

**Four inputs per event**, with common random numbers:

| Input | Construction |
|---|---|
| x | the original readout |
| x_jit± | the same draws, read out with `ext_timing.shifted_time_rule(rule, ±1.0)`, i.e. a trigger shift of ±1 ns |
| x_noise | scheme-A realisation r = 1 via `resample_noise` |
| x_phys | `replace(draws, vertex_m=v′)`, with v′ = vertex + 0.20 m along a keyed unit direction (namespace `ext_dejitter/vertex`). 0.20 m / v_group 0.2172 m/ns ≈ 0.92 ns of light travel |

**Redraws for x_phys.** Redraw the direction (≤ 100 tries) if v′ leaves the 0.8R / 0.8H fiducial
cylinder, or if `count_capacity(expected_charge(…at v′…).max()) > draws.capacity`. Report both
redraw counts. The review saw about 210 fiducial redraws and 8 capacity failures in 1 000 events.

x_phys also changes charges. Its per-PMT time change has an SD of about 0.5 ns, which is reported
for context.

**Quantities per encoder.** With Δ_* = z(x_*) − z(x), and V = trace Cov(z) over the events:
- **sensitivities**: E‖Δ_jit‖² / V (averaged over ±), E‖Δ_noise‖² / V and E‖Δ_phys‖² / V;
- **ratios**: R_jit/phys = E‖Δ_jit‖² / E‖Δ_phys‖², and R_jit/noise;
- **secondary**: the same ratios after whitening z by its covariance (pinv, rank-truncated at
  1e-6), because the encoders' ranks differ widely;
- **raw rows**: the same on the module-pooled raw features, `pooled(features, "within",
  view_groups(geometry))` as in `_raw_ceilings`. These are descriptive only; raw time is in
  units of 100 ns.

### 2.3 Detectability probe at 10 000 events

**Events.** `ext_m_dose/evt`, runs 0–999 × 10, F1 s = 0.05, photon. R = 8 scheme-A realisations
per event.

**Targets.**
- **Primary: robust δ** = trigger(r = 0) − **median** over r of trigger.
- **Continuity: mean δ** = trigger(r = 0) − mean over r of trigger.
- Exclude an event if **any** of its 8 realisations has a dark-moved trigger, and report the
  count.

**Probe.**
- The 8 encoders run on r = 0.
- `ridge_r2`: train on runs 0–499, test on runs 500–999.
- Targets: R²(δ_robust), R²(δ_mean) and R²(vertex).
- Report the raw ceiling and the hit-token mean-time row.

**Storage.** Never store raw events. Keep event vectors (float32) and targets in one `.npz` per
encoder.

### 2.4 Acceptance tests (CPU, before the host run)

Add `src/jepa_noise/tests/test_ext_dejitter.py`:
1. **Shifted trigger.** With `shifted_time_rule(rule, c)`, the token times of PMTs kept in both
   readouts shift by exactly −c/100.
2. **x_phys.** It differs from x only in `vertex_m` and its consequences: `poisson_u`, `charge_z`,
   `transit_z` and the dark draws are identical, and the capacity check is enforced.
3. **The §3.1 transform:**
   - unhit tokens stay 0;
   - hit-token times have median 0 per event;
   - shifting the hit-token times by a constant leaves the output unchanged within 1e-6;
   - an event with no hits is left unchanged and does not produce NaN.
4. **Wrapper wiring.** Under the wrapper, a 2-step synthetic CPU run of the training path
   completes, and the batches it sees are transformed. The wrapper is removed afterwards.

---

## 3. Part 2 — the prescription: de-jittered tokens

### 3.1 The transform `median_time`, and its wrapper

**Definition.** For each event, every **hit** token's time becomes t′ = t − m, where m is the
`np.median` (float64; mean of the two middle values) of that event's hit-token times, in
`time_100ns` units. Unhit tokens stay 0. Events with no hit token are left unchanged.

**Effect.**
- Token time is t_digit − t_trigger, so t′ = t_digit − median(t_digit), and the trigger cancels.
- The median is itself a much smaller shared quantity: about 0.17 ns within-event SD, against
  about 1.9 ns for the trigger (review measurement).
- Prompt-window membership still follows the trigger.
- **So P-1 failing would be a result, not a bug.**

**Leakage note** (record in `DECISIONS.md`).
- Context inputs depend on target-view digits only through one scalar, exactly as with the
  registered NHits trigger. Tensor-level leakage tests cannot see that.
- The review measured, for `within`, SD(median_all − median_context) = 0.21 ns, against
  SD(trigger_all − trigger_context) = 1.41 ns. The median is the less leaky reference.

**Wrapper.**
- In the extension training/inspect process only, replace the module attribute
  `jepa_noise.point_encoder.tokenise` at process start.
- The replacement calls the original, then returns `dataclasses.replace(batch, features=…)` with
  the transform applied to `features[..., TIME_INDEX]` on hit tokens. `LucidBatch` is frozen;
  `target` is a boolean mask and carries no time.
- The review checked that every token path goes through `pe.tokenise`: training loader, twin,
  probe sets, `event_level` and `tokenise_hits`. No subprocesses are involved.

### 3.2 Linear check (CPU, minutes; independent of training)

1. Regenerate M-dose levels F1 s = 0 and s = 0.05 with the M-dose `collect` path. **First** check
   that the untransformed regeneration reproduces the stored blocks (via `load_level`) within
   1e-9.
2. Regenerate the same levels with `median_time` applied to the tokens before pooling.
3. Recompute ν\*(A), ν\*(B) and ν\*_gain for both splits.

### 3.3 Training (deepthought2, L40S; the only training approved)

**Script.** Add `scripts/jepa_ext_dejitter.py`.
- Copy `mode_train` and `mode_inspect` from `scripts/jepa_pregrid_checks.py` into it. Do not edit
  that script.
- Install the §3.1 wrapper at process start for `median_time` runs only.
- Use these cells, so that run IDs are new and no Part D `checkpoint.pt` can be auto-resumed:
  - `replace(variant_cell(config, "across"), cell_id="ext_dev_s005_across_median")`;
  - `variant_cell(config, "median")`, which gives `ext_dev_s005_within_median`;
  - `variant_cell(config, "trigger")`, which gives `ext_dev_s005_within_trigger`, without the
    wrapper.

**Arguments**, the same as Part D:
- `--out <ext>/train`
- `--config src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json`
- `--generator-config src/jepa_noise/configs/lucid_bombs_v2_calibrated.json`
- `--geometry /ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json`
- `--data-root /ceph/dwong/jepa/data/ext_pregrid_dev_20261025/`
- the default arms config, `probe_max_events` None, and 10 000 steps.

Arm `sigreg_jepa` with the fixed target-only labels.

| Run | Cell | Seed | Compare with |
|---|---|---|---|
| `across_median_9000` / `9001` | across + median | 9000 / 9001 | Part D across, trigger time: 0.348113 / 0.333464 m |
| `within_median_9000` / `9001` | within + median | 9000 / 9001 | `within_trigger` (below); also Part C fix: 0.219258 / 0.204281 m |
| `within_trigger_9000` / `9001` | within, **no wrapper** (control) | 9000 / 9001 | its own code path, so P-4 compares like with like |

**Order.** `across_median` first, then `within_median`, then `within_trigger`. The control runs
**last**; if the budget runs short, they are the first to drop, and P-4 then falls back to the
Part C references with a stated caveat.

**Running.** Check the guard before every launch; at most 2 at a time; timeout 90 min each. Book
every run as an extension line in `RESOURCE_LEDGER.json`.

### 3.4 Evaluation

**For each new encoder:**
- vertex resolution and module-gain run-mean R², with the frozen dev step-2 protocol, as written
  to `ext_summary.json`;
- effective rank, event SIGReg and hit-token log-charge R², through the Part C `inspect` path in
  the new script;
- the `median_time` encoders are evaluated with the wrapper installed.

**`median_time` random-init floors**, for both splits: `init9000` through `inspect` with the
wrapper, **on Titan**.

The §2.2 sensitivities of the `median_time` encoders are ≈ 0 by construction for a trigger shift.
Report them labelled "edge effects only", and give no reading for them.

### 3.5 Acceptance (before the six runs)

- The §2.4 tests pass and the full suite passes on the deployed SHA.
- A **20-step smoke** of `across_median_9000` (`--steps 20 --probe-max-events 200`) on
  deepthought2 runs without NaN and writes `ext_summary.json`.

---

## 4. Pre-declared readings (descriptive, not gates)

Report each one as **supported**, **not supported** or **unclear**, with the numbers. Write any
further tolerance into `DECISIONS.md` **before** computing. There are only two seeds, so no
seed-floor claim is made; report both seeds.

| ID | Reading | Supported if |
|---|---|---|
| J-1 | `across` encoders are more jitter-sensitive | min over the `across` seeds of R_jit/phys ≥ 1.5 × max over the `fix` seeds. Report the whitened version beside it |
| J-2 | JEPA training reduces jitter sensitivity | Both `fix` seeds' R_jit/phys ≤ 0.5 × `init9000`'s. The raw rows are descriptive |
| J-3 | The jitter is detectable at 10 000 events | Raw-input ceiling R²(δ_robust) ≥ 0.2, without dark-moved events. **If supported**, re-evaluate T-d on δ_robust: both `across` seeds' R² ≥ max(`fix`) + 0.1 **and** ≥ `init9000` + 0.1. **If not**, T-d stays unclear |
| P-1 | `median_time` removes the linear trap | `across_fixed`, at s = 0 and at s = 0.05: ν\*(A) ≤ 0.05 |
| P-2 | De-jittered tokens help deep `across` | vertex(`across_median`) ≤ 0.85 × the Part D across reference, in both seeds |
| P-3 | They close the split gap | vertex(`across_median`) ≤ 1.1 × vertex(`within_trigger`), same seed, in both seeds. If the control runs were dropped, compare against the Part C references and state that |
| P-4 | They do not harm `within` | vertex(`within_median`) ≤ 1.1 × vertex(`within_trigger`) in both seeds, with the same fallback |
| P-5 | Floors and extras | Status: **descriptive**. Each new encoder against its random-init floor (vertex, gain R², token charge R², effective rank) |

---

## 5. Write-up, commit, report, wait

1. Results go in `results/jepa/20261030_dejitter/`, with a README marked `registered: false`, plus:
   - `sensitivity.md`: Part 1, encoders × {sensitivities, ratios, whitened ratios} plus the raw
     rows, the redraw counts and the 10 000-event probe table;
   - `prescription.md`: the linear check, i.e. ν\* by split and level with and without
     `median_time`, and the six new runs against their references and floors;
   - JSON summaries, each ≤ 1 MB.
2. Checkpoints and large arrays stay in `<ext>`, with their paths and hashes recorded.
3. `DECISIONS.md` gets one line per engineering choice. `STATE.md` gets a short "dejitter" entry.
   Book every resource.
4. Commit and push after Part 1, after the linear check, and at the end.
5. **Report to the author (≤ 1 page, in Chinese):**
   - the condensed tables;
   - the eight readings with numbers;
   - GPU-h per host and the ledger total;
   - the final SHA.
6. **Wait for the author.** Do not resume the grid.
