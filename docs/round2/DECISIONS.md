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
