# RUNBOOK — extensions and pre-grid checks, all parts (25 Oct 2026)

_To the JEPA study agent working on Deepthought and deepthought2. From the author.
Branch `JEPA`, at or after `fecc1f7`. This runbook **replaces** `TASK_2026-10-24_PREGRID_CHECKS.md`.
Commit it unchanged as `docs/jepa/RUNBOOK_2026-10-25_PREGRID_ALL.md` in Part A, before any run._

---

## 0. Authority, scope, stops

**Authority.** The author, 24 Oct 2026, 23:32 CEST (verbatim; quote it in `DECISIONS.md`):
> "我需要你改几个细节，重新做一遍实验，
> 1. 线性模型组，加两组实验，一组是纯噪音没物理，一组是纯物理没噪音，来验证猜想
> 2. 第二，在JEPAtraining 时候泄露是在lucid data上对吧，然后避免泄露不可以用打乱组别对吧。因为组的顺序也是物理信息的一环
> 3. 在toy model的traceset里面，我们要是加上各组arriving time 逐组变晚。加一个这样的causal关系呢？
> 4. 建议恢复 grid 前先做的三项检查，你准备好code和prompt，setup好后我让agent去跑
> 5. 在给俊杰的report里面加上结果table，以及各个量是如何算的，特别是物理量和噪音比
> 6. 我们对JEPA已经做了的修复，sound吗？用两个encoder会不会解决问题，还有别的解决办法吗？双保险"

And 25 Oct 2026, 02:00 CEST (verbatim):
> "全部都可以在deepthought上跑对吧？不如把所有写成一个长runbook，我直接交给deepthought agent"

These approve exactly the work below.

**Status of everything here.** Every result is an **extension**:
- `registered: false`;
- written outside the registered result directories;
- never replaces a verdict, a prediction status or a frozen decision.

**The grid stays paused.** This runbook supersedes:
- step 4 ("resume the grid") of `TASK_2026-10-23_LABEL_SHORTCUT.md`;
- all of `TASK_2026-10-24_PREGRID_CHECKS.md`.

The order is:
1. Let the 23 Oct fixed-label step-3 run finish (Part 1).
2. Do Parts A–F.
3. Report, then wait for the author.

Do not write a grid decision file and do not launch any grid run.

**Rules that still hold.** All of these apply unchanged:
- §6.1 hosts and execution path, and the comparability rule (one GPU type per comparison set);
- §6.2 resource guard;
- the no-touch packages;
- the hard rule on shared storage: never delete, move or modify anything outside `/ceph/dwong/jepa/`;
- never infer author approval.

**Stop and report.** Write one page with lettered options, and stop, only if:
- **S1 budget:**
  - the L40S ledger plus projection would exceed **160 GPU-h**;
  - or Titan use in this runbook exceeds **30 GPU-h**;
  - or LUCiD generation would exceed the CPU generation cap.
- **S2 reproduction:** a reproduction check in Part C fails (`--expect` off by more than 1e-3 m).
- **S3 spec:** something here would change a registered prediction, criterion, cell, dataset or frozen decision.
- **S4 push:** the patch cannot reach GitHub (no push rights from where you are). Do not copy code to the hosts by any other route; §6.1 says code reaches the hosts only through GitHub.
- **S5 storage:** a host is not reachable, or `/ceph/dwong/jepa/` is not mounted on deepthought2, for more than 24 h.

Everything else is engineering: write one line in `DECISIONS.md`, then continue.

---

## 1. What runs where (all through `ssh deepthought` / `ssh deepthought2`)

| Part | Host and lane | GPU / CPU | Estimate |
|---|---|---|---|
| 1. Finish fixed-label dev step 3 (already running) | deepthought2 | L40S | as booked (≤ 15 GPU-h) |
| A. Patch, tests, push, per-SHA clone, linear reproduction | wherever you push from; then Deepthought CPU | CPU | < 1 h |
| B. β re-selection with the fixed plain | any host | CPU | seconds |
| C. Inspect existing checkpoints (no training) | Deepthought | 1 Titan | ≈ 2 GPU-h |
| D0. Smokes of the 5 variants | Deepthought | 1 Titan | ≈ 1 GPU-h |
| D1. Strong-drift data and the two new floors | Deepthought | 1 Titan + CPU | ≈ 1–2 GPU-h + generation |
| D2. 10 training runs | deepthought2 | L40S, ≤ 2 jobs at a time | ≈ 8 GPU-h |
| E. Toy bridge with causal delays | Deepthought | ≤ 3 Titans at a time | ≤ 8 GPU-h |
| F. Write-up and report | any host | none | — |

Why this split:
- Everything that enters a comparison with the fixed LUCiD dev runs is trained on **L40S** (D2), because those runs were L40S.
- Evaluation, smokes, generation and the whole toy bridge run on **Titans** (§6.1 table).
- C is evaluation only, so it runs on a Titan. The `--expect` check guards cross-device reproduction.

**Parallel plan.**
- **Titan lane, in this order:**
  1. C;
  2. D0;
  3. D1 (strong-drift generation must finish before its D2 runs);
  4. E.
  - Never use all four Titans fully loaded; at most 3 concurrent jobs.
- **L40S lane:**
  1. Part 1 runs until it is finished;
  2. then D2, two jobs at a time.
  - Start `across`, `stopgrad`, `lambda` and `ijepa` first. They reuse the existing dev data.
  - `strong_drift` goes last, after D1 has generated its data.
- Check the guard state (`/ceph/dwong/jepa/jobs/guard_<host>.json`) before every launch. Monitor GPU memory and host load while jobs run. At HOLD, launch nothing new. At SHED, follow §6.2.

**Paths.** Use these throughout:
- Code: `/ceph/dwong/jepa/code/<sha>/` (a fresh clone of the pushed SHA).
- Extension root (`<ext>`): `/ceph/dwong/jepa/returns/ext_pregrid_20261025/`. It holds `inspect_dev/`, `train/`, `floors/` and `bridge_delay/`.
- LUCiD data root: the **same** `--data-root` as the 23 Oct step-2b runs, so the dev data is reused and nothing regenerates. Strong-drift data is new and lands under the same root with its own key.
- Toy bridge data root: a new directory, `/ceph/dwong/jepa/data/bridge_delay_fixed001/`.
- `--config`, `--generator-config` and `--geometry`: copy them from the manifest of run `dev_s005_within__sigreg_jepa__s9000` (fixed label, 23 Oct). Record the exact values in `DECISIONS.md` once.

---

## Part 1 — finish the 23 Oct step 3

- The fixed-label `a10_dev_step3_fixed` run is in progress. Let it finish and freeze its decision as already instructed in `TASK_2026-10-23_LABEL_SHORTCUT.md` step 3.
- Its decision uses the **old** plain reference (0.264 m). Part B corrects that.
- Do not start step 4 of that task.

## Part A — patch, tests, push, reproduce

1. **Apply the patch.** The author attaches `0001-jepa-extensions-2026-10-24.patch`. Put it outside the repo tree and run `git am` on `JEPA` (base `fecc1f7`). It adds:
   - `scripts/jepa_e1_ext_pure_cells.py` and `results/jepa/20261024_e1a_ext_pure_cells/` (already run; the README has the table);
   - `src/jepa_noise/delay_views.py`, `scripts/jepa_toy_delay_population.py` and `results/jepa/20261024_toy_delay/`;
   - an opt-in `$JEPA_TOY_DELAY` hook in `bridge_run.py` (unset = registered behaviour, unchanged);
   - `scripts/jepa_pregrid_checks.py`;
   - tests `test_delay_views.py` (4) and `test_pregrid_checks.py` (5).
2. **Commit this runbook and run the tests.**
   - Commit this runbook as `docs/jepa/RUNBOOK_2026-10-25_PREGRID_ALL.md`.
   - Run the full suite. It must pass: 480 tests at `fecc1f7` plus the patch.
3. **Push and clone.**
   - Push `JEPA`. If you cannot, that is S4.
   - Clone the pushed SHA to `/ceph/dwong/jepa/code/<sha>/` on Deepthought and on deepthought2.
   - Use the existing venvs. On both hosts, a CPU smoke must pass before any GPU job:
     `pytest src/jepa_noise/tests/test_pregrid_checks.py src/jepa_noise/tests/test_delay_views.py`.
4. **Reproduce the linear extension once,** on Deepthought CPU, `nice -n 10`, at most 4 threads (`OMP_NUM_THREADS=4`):
   `PYTHONPATH=src python scripts/jepa_e1_ext_pure_cells.py --out <ext>/linear_repro/rows.json`
   - Acceptance: every per-row max angle within 0.1° of the committed `rows.json`.
   - This is not a gate. Record a pass, or report the differences.
5. **Reproduce the population table, read-only:**
   `PYTHONPATH=src python scripts/jepa_toy_delay_population.py --out <ext>/toy_delay_repro/population.json`
   - Compare it with the committed `results/jepa/20261024_toy_delay/population.json`: identical to 0.1°.

## Part B — β re-selection with the fixed plain reference (after Part 1)

```
python scripts/jepa_pregrid_checks.py reselect-beta \
  --decision <a10_dev_step3_fixed .../dev/step3_beta_decision.json> \
  --plain-resolution 0.21925859104673096 \
  --plain-source results/jepa/20261023_label_shortcut_diag/fixed_s9000/probes.json \
  --out configs/decisions/a10_dev_step3_beta_decision_fixedplain_<date>.json
```

- Freeze it beside the other decisions, marked `registered: false` (the script writes that field).
- `anchor_only` follows the anchored arms' β, as already ruled (option A).
- Do **not** write a grid decision file.

## Part C — inspect existing checkpoints (Deepthought, 1 Titan, no training)

Locate the six checkpoints: the `final.pt` of the 23 Oct step-2b runs and of the dev step-3 runs. Record each path and its sha256 in `DECISIONS.md`.

```
python scripts/jepa_pregrid_checks.py inspect --out <ext>/inspect_dev \
  --config <...> --generator-config <...> --geometry <...> --data-root <...> \
  --item old9000=sigreg_jepa:<old-label s9000 final.pt>   --expect old9000=0.264400 \
  --item old9001=sigreg_jepa:<old-label s9001 final.pt>   --expect old9001=0.233759 \
  --item fix9000=sigreg_jepa:<fixed-label s9000 final.pt> --expect fix9000=0.219259 \
  --item fix9001=sigreg_jepa:<fixed-label s9001 final.pt> --expect fix9001=0.204281 \
  --item mae9000=mae_mse:<dev mae_mse s9000 final.pt>     --expect mae9000=0.104949 \
  --item anc100=sigreg_jepa_anchored:<dev anchored beta100 final.pt> \
  --item init9000=sigreg_jepa:init:9000                   --expect init9000=0.160904
```

Each item is evaluated in two forms:
- **full input:** the registered form;
- **half-pass:** every token computed the way training did. Context tokens come from the context-only pass; target tokens come from the target-only pass.

For each form, the script records:
- photon vertex resolution;
- module-gain run-mean R²;
- log-yield R²;
- event-level effective rank and participation ratio;
- the SIGReg statistic of the per-event embeddings;
- a per-PMT ln q probe on hit-PMT tokens.

A failed `--expect` is S2. Stop before Part D2, because the cross-device evaluation would then be suspect.

## Part D — training variants

### D0. Smokes (Deepthought, 1 Titan)

Run one smoke per variant, each with `--steps 20 --probe-max-events 200`:
- variants: `across`, `stopgrad`, `lambda`, `ijepa`;
- skip `strong_drift` until D1 has made its data.

Each smoke must finish without NaN and write `ext_summary.json`.

### D1. Strong-drift data and the new floors (Deepthought)

These two commands generate the strong-drift dev data (module-gain spread 0.30) and give both new random-init floors:

```
python scripts/jepa_pregrid_checks.py inspect --cell-variant strong_drift \
  --out <ext>/floors/strong_drift --config <...> --generator-config <...> --geometry <...> \
  --data-root <...> --item init9000=sigreg_jepa:init:9000
python scripts/jepa_pregrid_checks.py inspect --cell-variant across \
  --out <ext>/floors/across --config <...> --generator-config <...> --geometry <...> \
  --data-root <...> --item init9000=sigreg_jepa:init:9000
```

Generation details:
- The generation of the strong-drift cohort counts against the CPU generation cap.
- Same events and seeds as dev; only the spread changes.

Then run the `strong_drift` smoke from D0.

### D2. Training (deepthought2, L40S, ≤ 2 jobs at a time; seeds 9000 and 9001)

```
python scripts/jepa_pregrid_checks.py train --variant <v> --seed <s> --out <ext>/train \
  --config <...> --generator-config <...> --geometry <...> --data-root <...> [--lam 0.2]
```

| variant | what changes vs the fixed dev run | question |
|---|---|---|
| `across` | split `across` | Is `within` too local a task? |
| `strong_drift` | module-gain spread 0.30 (new dev data, same events) | Is 5% drift too weak per event to be absorbed? |
| `stopgrad` | labels detached in the prediction loss (SIGReg unchanged) | Does label shaping by the shared encoder hurt? |
| `lambda` | SIGReg coefficient 0.2 (10×) | Is the prediction loss overriding SIGReg? |
| `ijepa` | the registered `ijepa_ema` arm on the dev cell | Does an EMA target encoder avoid the problem? |

Each run is about 0.72 L40S GPU-h; the 23 Oct fixed run took 2 576 s. Ten runs plus evaluation is about 8 GPU-h.

Book every run in `RESOURCE_LEDGER.json` as an extension line.

### Pre-declared readings

These are descriptive, not gates. Report each one as **supported**, **not supported** or **unclear**, with the numbers.

| Reading | Supported if |
|---|---|
| Evaluation mismatch (Part C) | Half-pass vertex resolution ≤ 0.9 × full-input, in both fixed seeds. |
| Task too local | `across` vertex resolution ≤ 0.85 × the fixed `within` one, in both seeds. |
| Drift too weak | In `strong_drift`, module-gain run-mean R² exceeds its new random-init floor by ≥ 0.10, in both seeds. |
| Label shaping | `stopgrad` vertex resolution ≤ 0.9 × fixed, in both seeds. |
| SIGReg overridden | With `lambda`, the final SIGReg is ≤ 3 **and** the vertex resolution is better than fixed, in both seeds. |
| EMA | `ijepa` vertex resolution better than the random-init floor (0.161 m), in both seeds. |

Reference numbers:
- Fixed `within` vertex resolution: 0.219259 m (seed 9000) and 0.204281 m (seed 9001).
- Dev random-init floors: vertex 0.160904 m, module-gain run-mean 0.218548.
- For `across`, compare against its own floor from D1 as well.

## Part E — toy bridge with causal delays (Deepthought, Titans)

- Export `JEPA_TOY_DELAY='{"delay_mode":"fixed","delay_step":0.01}'` for every command in this part, and record it in `DECISIONS.md`.
- Run the bridge **exactly** as the registered WP4a grid was run:
  - same command, config, batch 128 and 10 000 steps;
  - the frozen β = 10 decision;
  - arms `sigreg_jepa`, `sigreg_jepa_anchored` and `mae_chi2`, seeds 0–2;
  - new `--out <ext>/bridge_delay` and new `--data-root /ceph/dwong/jepa/data/bridge_delay_fixed001/`.
- Before the full run, do one smoke: `--steps 20 --max-events 2000`, one arm, one seed.
  - It must run, and the cohort `manifest.json` must contain `delay_mode` and `delay_step`.
  - A CPU smoke of exactly this passed off-host on 25 Oct, at 1.6 s/step.
- The cohort cache key differs from the registered one, because the delay enters the key. Never point this run at the registered bridge data root.
- Aggregate with the same aggregate command (`--allow-partial` only if a run failed, and say so).
- Report beside the registered B1 numbers:
  - nuisance R², plain vs anchored;
  - physics MSE/CRLB for amplitude, t0 and tau;
  - the amplitude probe.
- Only the fixed mode is wired. The per-event mode needs a fourth probe target and is out of scope.
- Budget: ≤ 8 Titan GPU-h.

## Part F — write-up, commit, report, wait

1. Results go to `results/jepa/20261025_pregrid_checks/`, each with a README marked `registered: false`:
   - `inspect.json` copies and the floors;
   - the D2 `ext_summary.json` files;
   - the bridge aggregate.
2. `DECISIONS.md` gets one line per engineering choice. `STATE.md` gets a short "pre-grid checks" entry.
3. Commit and push `JEPA`.
4. **Report (≤ 1 page):**
   - **Part A:** reproduction pass/fail.
   - **Part B:** the re-selected β per arm, old vs new plain.
   - **Part C:** one table, items × {full input, half-pass} × metrics.
   - **Part D:** one table, variants × seeds, with the floors, and the six readings.
   - **Part E:** a two-row comparison (registered B1 vs delayed).
   - The GPU-h used per host, and the ledger total.
5. **Wait for the author.** Do not resume the grid, and do not propose a grid change without a gate.