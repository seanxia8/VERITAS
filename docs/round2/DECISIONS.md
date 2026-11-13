# DECISIONS — round 2 (extension `r2_multidetector_2026-11-11`, registered: false)

Runbook: `docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md`, committed unchanged,
sha256 `366f0680a5b262a90774bf725fd6ee75c516766789a605807b22978630572afe`.

## §0 authority (quoted verbatim from the runbook)

1. The author, 11 Nov 2026, 21:30 CET:
   > "根据第一轮的configuration写一个可以follow的runbook，同样的configuration，对于HK，SK，JUNO三种simulation，每种simulation用同样的configuration。有了dataset之后更正后的JEPA，MAR，random init三种结果train后prob，然后做对比"

   ("MAR" means MAE.) This authorises data generation for the three detectors, training of the arms
   in §5, probing and the comparison.
2. The author, 11 Nov 2026, answering the design questions: JEPA arm = **"Fixed-label + position-free
   (Recommended)"**; GPU budget = **"New round-2 ledger, cap 60 h (Recommended)"**.
3. The author, 11 Nov 2026, on the cleanup: **"Tag + fast-forward dev only (Recommended)"** (round 2
   continues on `dev`).
4. The standing rule, 30 Oct 2026:
   > "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

   Fix routine problems yourself (bugs, paths, environment, guard HOLD/SHED, launcher timeouts, ssh
   drops), write one `DECISIONS.md` line each, and continue. Stop only for §9.
5. The author, 11 Nov 2026, 23:05 CET, adding the receptive-field control:
   > "加入这三条建议：只加 MAE 一个 arm、一个 seed；窗口按 64 × N_PMT / 1,995 缩放：SK 约 384，JUNO 约 320，HK 约 640；三个探测器加起来大约多 8–12 L40S h，60 h 的账本还装得下。"

   This authorises the arm `mae_mse_rf` of §5.1 (one seed per detector) inside the same 60 h cap.
6. The author, 11 Nov 2026, 23:18 CET, accepting three design recommendations (keep the time-scale
   rule of §2.2 and log its effect; a pre-declared conditional seed 9002; random init with five seeds):
   > "按照你的建议更新 runbook"

Standing rule for this execution (author, 30 Oct 2026): "别大事小事都问我，自己找解决方案，按最好的解决方案解决".

## Decisions (one line each)

- 2026-11-12 impl: JUNO sensor positions generated once on Deepthought with LUCiD `generate_detector(config/JUNO_geom_config.json)` at `eaf2e89` and frozen to `configs/round2/juno_geometry_lucid_eaf2e89.npz` (sha256 `f91e6bc7…`); LUCiD file hashes pinned in `r2_geometry.LUCID_FILE_SHA256`.
- 2026-11-12 impl: the sphere carries `height_m = 2R`, so every round-1 consumer of `radius_m`/`half_height_m` (Hilbert box, position scaling) sees the §2.2 box `[-R, R]^3` / `(R, R, R)` without touching round-1 code.
- 2026-11-12 impl: sphere fiducial sampling consumes the same three keyed uniforms as the round-1 cylinder sampler (radius, azimuth, third coordinate): `r = 0.8 R u^(1/3)`, `phi ~ U[0, 2pi)`, `cos theta ~ U[-1, 1]`.
- 2026-11-12 impl: `r2_generator.readout` is the round-1 readout with `nhits_trigger_time(n=…, window_ns=…)` and `prompt_window_mask(window_ns=…)` called explicitly; on WCTE it regenerates stored round-1 run `lucid/dev_train/0000` with bit-identical hits (test 2).
- 2026-11-12 impl: the stored round-1 `events.npz` differs from the unchanged round-1 `draw_event` on Deepthought by <= 1 ulp in 11/60 photon yields (host libm); test 2 compares hits bit for bit and truth floats to 1e-12 relative.
- 2026-11-12 impl: round-1 `lucid_run` generated training cohorts with twins; round 2 writes twin hits and per-PMT `expected_pe` only for the four probe cohorts (§3.1); the measured hits are unaffected (separate readout).
- 2026-11-12 impl: training batches come from `r2_store.R2TrainingLoader`, which reuses the round-1 keyed stream `keyed_rng(seed, "lucid/batches", step)` and builds dense tokens only for the drawn events; equality with `lucid_run.TrainingLoader.batch_at` is test 6.
- 2026-11-12 impl: training logs every step (`log_every=1`) so the §6 loss shares use the per-step terms of the first and last 100 steps.
- 2026-11-12 impl: `init`, `target_posfree` and `mae_mse_rf` are built from the round-1 builders: `init` = `sigreg_jepa` at the seed with no training (as round-1 `init9000`); `target_posfree` = `sigreg_jepa` + `ext_shortcut.attach_posfree_target`; `mae_mse_rf` = `mae_mse` with `EncoderConfig(window=W_rf, shift=W_rf/2)`.
- 2026-11-12 impl: the X0 yield range uses the round-1 bisection (10 iterations) with each statistic on the same 1,000 development events (§2.2 wording), parallelised over 16 processes; the round-1 start bracket (6,000–320,000) is doubled until it contains the target, S8 if 30 doublings do not.
- 2026-11-12 impl: Gate V maps transport sensors to the active geometry within the round-1 0.06 m tolerance; photons on sensors absent from the active geometry (HK inactive entries, if the LUCiD transport carries them) are dropped and counted.
- 2026-11-12 impl: seed-9002 planning walks the triggered arms in §5.1 order; a run that does not fit is skipped with its reason and does not block a cheaper later run.
- 2026-11-12 impl: the Mac has no torch/h5py; round-2 tests run on Deepthought (`nice -n 10`, ≤ 8 threads, `TMPDIR=/work/dwong`) from an rsync copy at `/work/dwong/r2_code`.
- 2026-11-12 impl (development check, not the formal regression): through the new HDF5 reader on CPU, init9000 full-input vertex 0.16007164 m (committed 0.16007169) and mae_within_trigger9000 0.10491562 m (committed 0.10491560); work files under /work/dwong/r2_devcheck. The formal §4.2 regression runs after acceptance (`preflight --detector wcte --steps regression`).
- 2026-11-12 impl (development check): gate_v on WCTE reproduces the committed round-1 holdout exactly (yield ratio 1.00611, module 0.90151, log 0.81502, time 0.157 ns, passed); gate_v puts the pinned LUCiD clone first on sys.path (the default anaconda import resolves another LUCiD checkout).
- 2026-11-12 review round 1 BLOCK (B1, B2) fixed: per-reading seed sets for the three-seed version; ledger + S1 pre-check + wall cap + HK fallback record + derived seed-9002 plan + S5/S7 + X0-gated generation.
- 2026-11-12 impl: before X0 measurements exist, budget projections use the runbook §8 design upper ends (SK 17 h, JUNO 15 h, HK 30 h per six main runs; rf 2 / 1.7 / 4 h; evaluation 5 h); each is replaced by the smoke measurement as soon as it exists. Before the HK fallback is recorded, S1 checks for SK/JUNO runs count HK at its smallest plan (§8 step 2).
- 2026-11-12 impl: the HK re-projection takes the larger of the SK/JUNO smoke step times scaled by N_PMT and the HK smoke step time (conservative; runbook §8 says "from the SK and JUNO measured step times").
- 2026-11-12 impl: X0 yield range keeps one Poisson capacity (4 x the current bracket upper yield, as round 1's 4 x y_hi_initial) within each bisection, so trial yields re-map the same random numbers.
- 2026-11-12 §4.3 review round 2 PASS at 8db555f; acceptance recorded. Non-blocking items 1-11 of round 2 are tracked in the review file and handled during r2-run/r2-report.
- 2026-11-12 r2-run (routine, review round-2 items 6, 7, 9): WCTE-regression evaluations book into the main ledger; a crashed run books its GPU time; the 1.5x wall cap counts all invocations of a run; evaluate checks the checkpoint seed.
- 2026-11-12 r2-run: WCTE regression PASS (init9000 0.16007166 m, |d| 3e-8; mae_within_trigger9000 0.10491558 m, |d| 2e-8; tolerance 1e-3).
- 2026-11-12 r2-run: X0 trigger check — §9 defines S3 by the efficiencies only ("the only stops"), so dark-only windows that trigger are recorded and reported (not a stop); SK: 1 of 10,000 dark-only 1 µs windows triggered at N = 6 (1e-4 per window; the §2.2 rule's 50 x P approximation gives 6.6e-6, the sliding-window scan probability is higher); efficiencies 100 %.
- 2026-11-12 r2-run: SK X0 Gate V FAILS (holdout medians: yield ratio 1.186, module-pattern Pearson 0.829 < 0.85, PMT log-charge Pearson 0.692 < 0.70, time 0.78 ns); all 11,096 transport sensors mapped within 6 cm; the same harness reproduces the WCTE holdout exactly. S2: SK stopped, JUNO and HK continue (runbook §9).
- 2026-11-12 r2-run: X0 for JUNO and HK run concurrently with 8 workers each (16 cores in total); the round-1 CPU launcher's module allow-list is a protected file, so X0/generation run directly with nice -n 10 on Deepthought.
- 2026-11-12 r2-run: the ledger records stopped detectors (SK: S2); a stopped detector's runs and evaluation are no longer reserved and its training is refused.
- 2026-11-12 r2-run: JUNO X0 Gate V FAILS with the round-1 harness, which requires every holdout event to pass (round-1 calibration record: "event-level min module Pearson ≥ 0.85 per event"): e500_014 module-pattern Pearson 0.845 < 0.85, while the holdout medians pass (yield 1.077, module 0.888, log 0.761, time 0.78 ns). The gate is applied as fixed (no change after outcomes, §0/§9): S2, JUNO stopped. Dark-only: 1 / 10,000 windows triggered at N = 6 (recorded).

## Addendum 2026-11-12: Gate V informational (`RUNBOOK_2026-11-12_R2_GATEV_CEILING.md`, v2, sha256 `d237ec2b3059f5359b51e20e2047b78454264b214cd73c8699d7d7030cca582c`, committed unchanged at f547174)

Addendum §0, verbatim:
- **Author, 12 Nov 2026, 10:08, after learning that only preflight had run:** "啊？只是preflight？
  能不能一步步往下走，我会handsoff now，直接从生成数据开始".
- **Author, 12 Nov 2026, 10:01** (option B, the ceiling diagnostic): "做B，那现在我把run——chain停掉了".
- **Standing rule (30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".

- 2026-11-12 r2-run (addendum): `x0_clearance` treats Gate V as informational and re-judges S3 from the stored efficiencies (the SK X0 JSON was written before the S3 fix and still says `"stop": "S3"` for 1 dark-only trigger); the ledger's SK and JUNO S2 stops move to `resumed` with the original record kept; `EXECUTION_sk.json` / `EXECUTION_juno.json` carry the resume record.
- 2026-11-12 r2-run: SK generation runs with 6 workers while the HK X0 job (8 workers + the 1-worker Gate V job) still runs, so round-2 CPU use stays ≤ 16 cores.
- 2026-11-12 r2-run: the ceiling diagnostic (`ext_r2_gatev.py`) re-transports the ten holdout events with the exact `gate_v` code and caches them; it reproduces μ via the `compare_event` arithmetic and checks Σμ against the committed `expected_detected` (1e-6 relative). It runs on spare CPU after generation work, never ahead of it.
- 2026-11-12 r2-run: the log-charge ceiling is not monotone in Λ (more photons bring low-μ PMTs into the `keep` set), so the test asserts monotonicity on the module metric only.
- 2026-11-12 r2-run: SK generated (900 runs, 2.01 wall-h on 6 workers) and packed to `sk_r2.h5` (4.95 GB, sha256 `89cde55c…`, every run verified byte for byte); entry in `results/round2/20261112_data/DATA_MANIFEST.json`. JUNO generation started at once with 10 workers (SK gen's 6 freed + HK X0 now in its single-process Gate V step), training order SK → JUNO → HK unchanged.
- 2026-11-12 r2-run: deepthought2's guard was HOLD (other users at 100 % on both L40S). GPU jobs go through `/work/dwong/r2_ops/r2_lane.sh` (two lanes = the two-job limit), which calls `scripts/jepa_run.sh` and retries every 120 s when the guard refuses (exit 3); no bypass.
- 2026-11-12 r2-run: the harness on the Mac blocks a bare foreground `sleep`; waits are bounded polling loops (≤ 28 min per call).
- 2026-11-12 r2-run: results writer `ext_r2_results.py` (runs.json, readings.json, READINGS_SUMMARY.md, figures). The WCTE loss-share reference uses the round-1 MAE log's step-100 and final rows, because round 1 logged every 100 steps (recorded as an approximation).
- 2026-11-12 r2-run: SK smokes (20 steps, deepthought2 L40S GPU 1): `sigreg_jepa` mb8 OOM at the 45 % cap → mb4 3.21 s/step, 13.95 GiB (8.92 h per run); `target_posfree` mb8 OOM → mb4 3.28 s/step, 13.95 GiB (9.12 h); `mae_mse` mb8 3.48 s/step, 9.70 GiB (9.66 h); `mae_mse_rf` mb8 2.30 s/step, 9.70 GiB (6.40 h). Steady-state step times equal the 20-step means. Far above the design projection (≈ 2.8 h per SK run): the predictor's full attention grows with the square of the token count, (11,096 / 1,995)² ≈ 31× WCTE, and window-64 MAE is slower than window-384 MAE (more windows).
- 2026-11-12 r2-run: **S1** (GPU projection over the cap after the fallback). SK's six main runs (55.40 h) + SK evaluation (1.67 h, design) + booked 0.09 h leave 2.84 h of 60. JUNO's six main runs project to 49.93 h (SK step times × 10,000 / 11,096), and the §8 fallback reduces HK only: JUNO stopped (S1). HK fallback (§8) computed from the measured SK step times scaled by N_PMT (no JUNO smoke: it would spend scarce budget and cannot change the result): step 3, S1, HK stopped. `mae_mse_rf` on SK does not fit after the six main runs and the evaluation (63.56 h > 60): that run is stopped (S1); it ranks below every main run (§5.1, §8). Per §9 ("finish what is safe … continue independent items") SK, first in the fixed order, trains its six main runs and is evaluated within the cap.
- 2026-11-12 r2-run: JUNO generation/pack (CPU, running when S1 fired) are finished and kept as data for a later budget; HK X0 (running) is finished; no HK generation (stopped detector).
- 2026-11-12 r2-run: SK main runs on two lanes (A: sigreg 9000, mae 9000, posfree 9001; B: posfree 9000, sigreg 9001, mae 9001); launcher duration = smoke projection (timeout 1.5×). The guard allows one job per GPU and GPU 0 is often held by other users, so the runs may go one at a time.
- 2026-11-12 r2-run: JUNO generated (900 runs, 1.09 wall-h on 10 workers) and packed (`juno_r2.h5`, 4.89 GB, sha256 `a6494ee6…`, verified); shards removed. CPU generation used in round 2: 3.10 wall-h (remaining cap 26.47 h).
- 2026-11-12 r2-run: Gate V ceiling diagnostic for WCTE, SK, JUNO (CPU 1.30 wall-h of 3, one core); fresh-context review of `ext_r2_gatev.py` (base 72c1af7, head 7a16272): PASS, non-blocking notes recorded in `results/round2/20261112_gatev_ceiling/GATEV_CEILING.md`. HK follows when its Gate V rows exist.
- 2026-11-12 r2-run: SK `target_posfree` s9000 trains at 4.61 s/step under GPU co-tenancy (another user's job became active on GPU 1), 1.4× its smoke; the ledger books wall time, so all six SK main runs may not fit in the cap. The runbook does not order runs within a detector, so the queues now put the core comparison first (author line 1: corrected JEPA, MAE, random init → R2-b, R2-c with both seeds): posfree 9000 (running), mae 9000, mae 9001, posfree 9001, then the collapse reference `sigreg_jepa` 9000/9001 (R2-a). Every run still passes the S1 precheck before it starts; a run the cap cannot hold is refused (S1) and reported.
- 2026-11-12 r2-run: the deepthought2 guard shed `target_posfree` s9000 at step 903 (SIGTERM 13:56:32Z, `shed_signal` in `guard_deepthought2.log`); the round-1 loop checkpointed and exited 0. The lane runner is replaced by `r2_lane2.sh`, which relaunches a training run (resume from its checkpoint, same wall cap counting all invocations) until its record is complete or its cap is hit, and never starts a run whose launcher is alive. posfree 9000 resumes first on lane A (A: posfree 9000, mae 9000, posfree 9001, sigreg 9000; B: mae 9001 (running), sigreg 9001).
- 2026-11-12 r2-run: `mae_mse` s9001 trains at 1.49 s/step on a quiet GPU against 3.48 s/step in its smoke: the smokes ran under co-tenancy, so their projections (used by the precheck and the S1 decisions) are conservative. The S1 decisions stand as recorded (fixed rule, no change after outcomes); the actual booked hours decide which SK runs the cap admits.
- 2026-11-12 r2-run: co-tenancy on deepthought2 slowed SK training to 1.5–2× the smoke step times (our jobs got 24–33 % of the SMs; host CPU load 85). The per-run wall cap (1.5 × smoke projection, all invocations) stays as fixed in §5.2. A watchdog (`/work/dwong/r2_ops/watchdog.py`, every 10 min) pauses a run, with SIGTERM and a clean checkpoint as in a guard SHED, when its last-20-min rate cannot finish within the remaining cap. The lane then resumes it once the guard sees a quiet GPU (5-min mean < 60 %). Paused time does not count against the cap or the ledger; a resume loses only start-up time (posfree 9000 resumed exactly at step 903). First pause: posfree 9000 at step 2,336 (6.67 s/step: 14.2 h needed > 10.19 h left).

## Addendum 2026-11-12 23:45: JUNO on Deepthought's GTX Titans (`RUNBOOK_2026-11-12_R2_JUNO_TITAN.md`, sha256 `f381955d64b1a8923f77d236e1074edc4a457acb1a9bf160a3a98cb24935d229`, committed unchanged at f07f1a8)

Addendum §0, verbatim:
> 我同意K 继续在 deepthought2 的 L40S 上跑完；
> * 把 JUNO 整体挪到 Deepthought 的 Titan 上。数据已经生成好了，因为 S1 被停着
> * 然后
> 1. 先花约 10 分钟做可行性测试：查卡、检查 torch 能不能用 CUDA、跑 20 步 smoke 测速度和显存；
> 2. 能用就开始在最多 3 张 Titan 上训练 JUNO；
> 3. 用不了就把原因记下来，第二轮维持只出 SK。

- 2026-11-12 r2-run (Titan): JUNO's S1 lifted (resume records in both ledger lines and `EXECUTION_juno.json`, originals kept). New line `R2_titan_deepthought` (cap 100 Titan GPU-h) in `RESOURCE_LEDGER_R2_TITAN.json`; `docs/round2/RESOURCE_LEDGER.json` now mirrors both lines with totals. Code: GPU profile from `R2_GPU_PROFILE` (default `l40s`, unchanged), `titan` = JUNO only, 11 GiB micro-batch rule, T-S1, Titan evaluation reservation = 12 × measured Titan WCTE eval time × N_PMT/1,995; run/eval records carry the GPU model.
- 2026-11-12 r2-run (Titan feasibility 1–2): 4 × NVIDIA GeForce GTX TITAN X (Maxwell, capability 5.2, 12 GB), idle, no other user's process; the round-2 environment's torch 2.5.1+cu121 includes sm_50 kernels, `cuda.is_available()` True, CUDA matmul and conv run: no new environment needed.
- 2026-11-12 r2-run (Titan launcher): `scripts/jepa_run.sh` supports Deepthought; its guard (profile "deepthought": ≤ 3 training jobs, 4th GPU only for ≤ 30 min jobs, start only on a GPU at < 20 % 5-min mean utilisation with no other user's process > 1 GB, memory fraction 0.80 ≈ 9.6 GiB, stricter than the addendum's 11 GiB) was not running and is started (`jepa_guard.py --host deepthought --interval 120`). Titan jobs run through `/work/dwong/r2_ops/r2_lane2.sh` lanes `titan*` with `R2_GPU_PROFILE=titan`; the guard's SHED gives the clean pause the addendum asks for when another user appears.
- 2026-11-12 r2-run (SK, Titan addendum §1.7): the five SK `init` evaluations run first on L40S lane A, ahead of the paused posfree 9000.
- 2026-11-13 r2-run (Titan addendum v2, sha256 `0a8d76e980e94007699f920dc809a00767e7843f6611840a8c614cdf7d8d277f`, committed unchanged over v1): author 12 Nov 23:59 "titan四张可以全占用" (all four Titans may be fully occupied). The round-1 guard (`scripts/jepa_guard.py`, protected) still admits ≤ 3 training jobs on Deepthought and gives the 4th GPU to jobs ≤ 30 min; all four Titans are used within it: three JUNO training lanes plus the 4th GPU for smokes and evaluations (no guard bypass).
- 2026-11-13 r2-run (Titan feasibility 4): WCTE regression on a GTX TITAN X PASS — init9000 0.16007162 m (|Δ| 7e-8), mae_within_trigger9000 0.10491562 m (|Δ| 2e-8); 171–172 s per WCTE evaluation (3.7× the L40S).
- 2026-11-13 r2-run (Titan feasibility 5–6): JUNO 20-step smokes on GTX TITAN X under the launcher's 9.54 GiB cap — `mae_mse` mb8 14.87 s/step, 8.76 GiB (41.3 h/run); `mae_mse_rf` mb8 14.82 s/step (41.2 h); `sigreg_jepa` mb8 and mb4 OOM → mb2 48.34 s/step, 6.13 GiB (134.3 h); `target_posfree` mb8 and mb4 OOM → mb2 49.55 s/step, 6.13 GiB (137.6 h). Projection: six main runs 626.4 Titan GPU-h + evaluation 2.87 h (+ rf 41.2 h); fallback 1 (drop rf) 629.3 h; fallback 2 (seed 9000 only) 316.1 h — all > 100 h. **T-S1: JUNO stopped** (CUDA, regression and memory all succeeded, so not T-S2). Record: `results/round2/20261113_titan/feasibility.json`. Titan lanes stopped, the Deepthought guard I started is stopped, all four Titans left idle. Round 2 reports SK only (plus the Titan feasibility).
- 2026-11-13 r2-run (HK X0): HK is stopped (S1); the full HK X0 job (17 h, its yield range and trigger check held in memory, now in a Gate V step that duplicates the separate Gate-V-only job) is stopped to free CPU, since its remaining benchmark step is not needed for a stopped detector. The Gate-V-only job continues, so HK's informational Gate V is recorded when it finishes. HK's ceiling diagnostic needs a second HK transport (> 15 CPU-h at the observed speed), beyond the addendum's 3 wall-h ceiling allowance (1.30 h used): HK ceiling skipped, reason recorded.
- 2026-11-13 r2-run (SK): `mae_mse` s9001 complete (10,000 steps, ≈ 11.3 L40S h booked under co-tenancy against a 9.66 h projection; L40S line 14.94 / 60 h). The five remaining SK runs no longer fit at their projections: the S1 precheck refused `sigreg_jepa` s9001 at launch (no hours booked), and that run is recorded as a run-level S1 (lowest priority: second seed of the collapse reference). Lanes rebalanced: A = remaining `init` evaluations, the `mae_mse` s9001 evaluation, posfree 9000 (resume), sigreg 9000; B = mae 9000, posfree 9001. R2-a will be reported with the seeds that exist (the §7 rule needs both, so R2-a reads "not determined: S1").
- 2026-11-13 r2-run (HK Gate V, informational): failed — yield ratio 2.07 (outside 0.5–2), module median 0.756, log-charge median 0.563, time 0.80 ns; recorded in `EXECUTION_hk.json` and the ceiling summary (HK ceiling skipped).
- 2026-11-13 r2-run (SK): the guard shed `mae_mse` s9000 at step 8,880; its resume and the start of posfree 9001 were refused by the S1 precheck, which counted the full 10,000-step projection for partly trained runs. Fixed (accounting bug): a run projects smoke × remaining steps / 10,000, both at launch and in the still-owed reserve. All four remaining SK runs pass (51.04 / 60 h); mae 9000 (resume) and posfree 9001 re-queued on lane B.

## Addendum 2026-11-13 11:40: GPU caps raised (`RUNBOOK_2026-11-13_R2_CAPS.md`, sha256 `d715eee6c417a54b4b36890210ee4f15da4b1d6b04e20c2329c75648c96a1d07`, committed unchanged at d9c4b3f)

Addendum §0, verbatim: **Author, 13 Nov 2026, 11:38:** "raise cap for both, 72 and 160".

- 2026-11-13 r2-run (CAPS): code dff3d9a — `GPU_CAP_H` 72, Titan line 160; resumed runs charged their remaining steps at their measured step time (run's own train log, all invocations; smoke step time before a run starts), with tests; `lift_run_stop` keeps the original record. Both ledger lines carry the new cap and the authority.
- 2026-11-13 r2-run (CAPS, L40S, fixed order): booked 23.55 h + SK main runs still owed 39.45 h (posfree 9000 11.42 h at its measured rate, posfree 9001 9.12, sigreg 9000 8.92, sigreg 9001 8.92, mae 9000 1.07) + SK evaluation 1.38 h = 64.38 h ≤ 72 → `sigreg_jepa` s9001 S1 lifted; + `mae_mse_rf` 6.40 h = 70.78 h ≤ 72 → its S1 lifted as well (it runs last); seed-9002 runs only from what is left after these.
- 2026-11-13 r2-run (CAPS, Titan): JUNO's T-S1 re-checked with the 160 h cap under the Titan addendum §2 fallback: three seed-9000 main runs + evaluation = 316.1 Titan GPU-h (JEPA arms 134–138 h each at mb2, the largest micro-batch the launcher's 9.54 GiB allows; mb4 OOM) > 159.2 h left → **T-S1 kept**; nothing trained on the Titans. The addendum's "expected plan" (three seed-9000 runs in parallel) assumed it would fit; the cap counts wall hours × GPUs, and the parallel plan is ≈ 138 h of wall time but ≈ 316 GPU-h. A cap near 320 Titan GPU-h would hold it.
- 2026-11-13 r2-run: per the CAPS addendum §1.4, `~/Documents/VERITAS/docs/round2/RUNBOOK_*.md` is listed with every poll; new author addenda are committed unchanged and followed.
- 2026-11-13 r2-run (CAPS re-check): with resumed runs charged at their measured co-tenancy step times (posfree 9000 at 5.17 s/step, posfree 9001 at 4.10), the plan reached 72.30 h > 72 and the precheck refused posfree 9001, sigreg 9001 and rf. In the addendum's fixed order `mae_mse_rf` goes back to S1 (original lift kept in the ledger); the four remaining main runs pass (65.90 h). rf is re-checked once the main runs finish.
- 2026-11-13 r2-run: watchdog window and minimum run time 20 → 60 min (step rates swing 2.3–7 s/step within hours; 20-min judgments paused posfree runs after ~300 steps each time).

## Addendum 2026-11-13 23:20: SK run order and GPU sharing (`RUNBOOK_2026-11-13_R2_SK_ORDER_COTENANCY.md`, sha256 `2e9869ea3ac350782ecef664f3131a1820f746e6ae2d0c6b7de10986f2120b72`, committed unchanged at 238976e)

Addendum §0, verbatim: **Author, 13 Nov 2026, 23:17:** "你的建议A+B都采纳" (adopt both suggestions A and B).

- 2026-11-13 r2-run (A): SK training lanes replaced by two shared-priority lanes (`/work/dwong/r2_ops/r2_lane3.sh` P1, P2 over `sk_priority.list`): sigreg 9000 → posfree 9000 (resume) → posfree 9001 (resume) → sigreg 9001 → `mae_mse_rf` (if the 72 h cap holds it). A free lane claims the highest-priority unfinished run (flock); on a guard refusal it releases the claim and re-picks, so sigreg 9000 is always the next launch. The running posfree job was not killed; old lanes A/B stop after it. Evaluations stay on lane E.
- 2026-11-13 r2-run (B, guard, deepthought2 only; code 82972e9, tests added): before — start only on a GPU with 5-min mean utilisation < 60 % and other users' memory + our 45 % cap ≤ 85 % of the card; HOLD when another user holds > 2 GB (or > 30 % utilisation) on a GPU we use. After — start when the card's memory in use + our measured peak (`JEPA_PEAK_MEMORY_MIB`, clipped to the 45 % cap; the cap itself when absent) ≤ 90 % of the card; no foreign-heavy HOLD. Unchanged: ≤ 45 % per process, one job per GPU, ≤ 2 jobs, every SHED rule (GPU memory > 95 %, host RAM, load, entries, our caps), the deepthought (Titan) profile. A fresh-context review gates the use of the new rule (the guard daemon keeps the old code until PASS).
- 2026-11-13 r2-run (B, wall cap): SK per-run wall cap 1.5× → 2.5× the smoke projection, all invocations counted (`wall_cap_factor`); the watchdog uses 2.5×, so runs it paused under the old cap are eligible again.
- 2026-11-13 r2-run (B): fresh-context review of the guard diff (base 238976e, head 82972e9): **PASS**, non-blocking notes — (1) unattributed-memory HOLD still applies (kept, protective); (2) the lane must supply a peak as nvidia-smi sees it → peaks in `sk_priority.list` = smoke `max_memory_allocated` + 1.5 GiB CUDA context/cache (JEPA 15,821 MiB, MAE-RF 11,469 MiB); (3) non-positive candidate peak now ignored (d5ff4df, test); (4) docstring fixed; (5) watchdog 2.5× and this record done outside the delta. The deepthought2 guard daemon (PID 976691, round-1 clone `jepa_work_20261107`) was replaced by the same script from the round-2 clone at d5ff4df (PID 1857090, `--host deepthought2 --interval 120`, default jobs dir). The other guard processes (Deepthought `jepa_a10` jobs dir) are untouched.
- SK runs launched after 2026-11-13 23:20 may train on GPUs shared with other users (reported).

## Addendum 2026-11-13 23:40: HTCondor batch GPUs (`RUNBOOK_2026-11-13_R2_HTCONDOR_BATCH.md`, sha256 `878f5969dcec981499d7d804727a5e86927665646a3cebb381d7db97b762bfc7`, committed unchanged at 140f02f)

Addendum §0, verbatim: **Author, 13 Nov 2026, 23:35:**
> 1. accounting group是cms, 你让他先用 `condor_userprio` 和小作业去试
> 2. 200小时
> 3. 也搬过去

- 2026-11-14 r2-run (batch feasibility): access passes (condor tools on Deepthought; `condor_userprio` shows `cms`, `cms.higgs`, `cms.jet`, `cms.production`; NEMO2 L40S slots listed). The tiny GPU job (cluster 759618, `accounting_group = cms`, docker `mschnepf/slc7-condocker`, NEMO2 L40S requirement, 900 s) found no match in 30 min (the NEMO2 L40S slots advertise `Requirements = false`) and was removed. Staging: `/ceph/srv/dwong` does not exist → **NO-GO, B-S2** (addendum §2.3: the author has to ask the ETP admins). No area created, nothing staged, no batch GPU hours booked. Round 2 continues as it is: SK on deepthought2; JUNO T-S1; HK S1. Record: `results/round2/20261114_batch/feasibility.json`.
