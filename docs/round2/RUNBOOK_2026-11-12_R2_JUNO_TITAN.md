# Runbook addendum — Round 2: JUNO trains on Deepthought's GTX Titans; SK stays on deepthought2

Date: 12 Nov 2026, 23:45 (v2, 00:05 on 13 Nov: all four Titans may be used). Parents:
- `docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` (sha256 `366f0680…`);
- `docs/round2/RUNBOOK_2026-11-12_R2_GATEV_CEILING.md` (v2, sha256 `d237ec2b…`).

All parent rules hold unless changed here.

---

## 0. Authority

**Author, 12 Nov 2026, 23:37, verbatim:**

> 我同意K 继续在 deepthought2 的 L40S 上跑完；
> * 把 JUNO 整体挪到 Deepthought 的 Titan 上。数据已经生成好了，因为 S1 被停着
> * 然后
> 1. 先花约 10 分钟做可行性测试：查卡、检查 torch 能不能用 CUDA、跑 20 步 smoke 测速度和显存；
> 2. 能用就开始在最多 3 张 Titan 上训练 JUNO；
> 3. 用不了就把原因记下来，第二轮维持只出 SK。

**The proposal the author agreed to** (23:35, translated from the Chinese):
- lift JUNO's S1;
- record Titan time in its own ledger, separate from the 60 h L40S ledger, with a cap of about 100 Titan GPU-h (JUNO's six main runs plus evaluation);
- keep each detector on one GPU type: SK on L40S, JUNO on Titan.

**Author, 12 Nov 2026, 23:59** (changes the Titan limit): "titan四张可以全占用", i.e. all four Titans may be
fully occupied.

**Standing rule (30 Oct 2026):** "别大事小事都问我，自己找解决方案，按最好的解决方案解决".

**Machine rule from the author:** Deepthought's four GTX Titans (12 GB each) are shared. Earlier rule: not all four fully
loaded. Since 23:59 on 12 Nov: all four may be fully occupied.

---

## 1. What changes

1. **SK** is unchanged. It trains and evaluates on the deepthought2 L40S under the 60 h ledger `R2_l40s_deepthought2`.
2. **JUNO's S1 is lifted.** Record a resume in the ledger and in `EXECUTION_juno.json`, keeping the original record and citing this addendum.
3. **All JUNO GPU work runs on Deepthought's GTX Titans (up to all four):**
   - smokes;
   - the six main runs: `sigreg_jepa`, `target_posfree`, `mae_mse` × seeds 9000/9001;
   - `mae_mse_rf` if the cap allows (it ranks last);
   - evaluation of every JUNO model, including the five `init` encoders.

   JUNO uses no L40S time.
4. **New ledger line** `R2_titan_deepthought`, cap **100 Titan GPU-h** (wall hours × GPUs, booked like the L40S line).
5. **HK** stays stopped (S1).
6. **Every run and evaluation record carries the GPU model.** The report states "SK: L40S; JUNO: GTX Titan". Cross-detector comparisons are descriptive (parent §7).
7. **SK scheduling** (routine; readings unchanged): evaluate each SK model as soon as a deepthought2 lane is free, with the five `init` encoders first and then each trained model when it finishes. Compute the readings once all SK models are evaluated.

---

## 2. Feasibility test, before any JUNO training (target ≈ 10–20 min, cap 1 h)

On Deepthought, using our own processes only:

1. **GPUs.** `nvidia-smi`: GPU model, memory, current use and other users' processes on each of the four GPUs.
2. **PyTorch.** In the environment the round-2 code uses: `torch.cuda.is_available()`, the device capability, and a small CUDA matmul and conv that must run without a "no kernel image" error.
3. **If that environment's torch cannot run on these GPUs:**
   - create a separate environment under `/work/dwong/envs/r2_titan` with a PyTorch build that supports the architecture. Never modify the base or anaconda environment.
   - Install time ≤ 30 min.
   - Rerun step 2 and the round-2 unit tests that touch the model code.
4. **Equivalence on this hardware.** Rerun the WCTE regression (parent §4.2: `init9000` and `mae_within_trigger9000` full-input vertex through the HDF5 reader) on one Titan. It must match the committed values within 1e-3 m.
5. **JUNO smokes**, 20 steps per arm (`sigreg_jepa`, `target_posfree`, `mae_mse`, `mae_mse_rf`). Use the largest micro-batch in 8/4/2/1 whose peak memory is ≤ 11 GiB. Record step time and peak memory.
6. **Projection.** JUNO's six main runs + evaluation (+ rf) in Titan GPU-h, and the wall time with up to four GPUs in parallel.

**GO** if steps 2 (or 3), 4 and 5 succeed for the three main arms and the six main runs + evaluation fit the 100 h cap.
- If they do not fit: drop `mae_mse_rf`, then train the main arms with seed 9000 only (three runs + evaluation).
- If even that does not fit: T-S1.

**NO-GO** (T-S2): CUDA unusable after step 3, the regression fails after one fix attempt, or a main arm does not fit 12 GB even at micro-batch 1. Record the reason in `results/round2/20261113_titan/feasibility.json` and `DECISIONS.md`. Round 2 then reports SK only, and JUNO keeps its S1.

---

## 3. Running on the Titans

**Limits:**
- Up to **all 4** GPUs at once (author, 23:59).
- One job per GPU, and only on a GPU that has no other user's process when the job launches.
- Peak memory ≤ 11 GiB (practical headroom on a 12 GB card).
- `nice -n 10`, `TMPDIR=/work/dwong`, `setsid` for long jobs.
- Training data-loader threads count against the 28-core Deepthought limit.

**Launcher:** use `scripts/jepa_run.sh` if it supports Deepthought; otherwise use a launcher with the same checks (free GPU, memory cap, logs, run record). Write all job outputs under the parent's returns directory.

**Shared GPUs:** if another user's process appears on a GPU we use, pause our job with a clean checkpoint (as in a deepthought2 guard SHED) and resume when the GPU is free again. Resumed invocations count toward the run's wall cap.

**Per-run wall cap:** 1.5 × the Titan smoke projection, all invocations counted (parent §5.2).

**Queue order:**
1. the core comparison at seed 9000 (`sigreg_jepa`, `target_posfree`, `mae_mse`);
2. the seed-9001 runs;
3. `mae_mse_rf`;
4. evaluation (`init` 9000–9004 and each trained model as it finishes);
5. readings, then the parent's §8 seed-9002 rule for JUNO, under the Titan ledger.

Commit and push JUNO results under `results/round2/`, plus STATE, DECISIONS and both ledger lines.

---

## 4. Stops

- **T-S1:** JUNO's projection exceeds the 100 Titan GPU-h cap after the fallback in §2.
- **T-S2:** NO-GO in the feasibility test.
- All parent stops (S3–S9) apply to JUNO as before. A JUNO stop does not affect SK.

## 5. Report

- **Feasibility:** GPUs, torch, regression on the Titan, smokes and projection.
- **JUNO:** the parent's per-detector readings, with the GPU model.
- **Both ledger lines.**
- Add JUNO to the r2-report's cross-detector table.
