# RUNBOOK — when does the trap appear? Criterion M and a nuisance dose-response (30 Oct 2026, reviewed)

_To the JEPA agent on Deepthought (with GitHub push rights). From the author. Branch `JEPA`, at or
after `6cafd38`. Commit this file unchanged as `docs/jepa/RUNBOOK_2026-10-30_M_DOSE.md` before
any run. Everything this agent needs is in this file and in the repository; it depends on no
outside document. An independent review was applied before release; its population numbers for
the toy are quoted below as acceptance values._

## 0. Authority, scope, how to handle decisions

**Authority.** The author, 30 Oct 2026, 00:51 CET (verbatim; quote it in `DECISIONS.md`):
> "把 A 写成 runbook，然后审查一遍"

"A" was "add criterion M and a nuisance dose-response: CPU-only linear analysis, reusing the
predictability code". This approves exactly the work below.

**Status.**
- Every result is an extension: `registered: false`, extension name `m_dose_2026-10-30`.
- Results go outside the registered result directories, and never replace a verdict, a
  prediction status or a frozen decision.
- **The grid stays paused**: write no grid decision file and launch no grid run.
- **No training and no encoder forward passes.**

**Decisions: do not stop to ask** (author's standing approval, 29 Oct 2026: "你让他按它建议跑，别老停下来问我").
When you face an engineering choice, or would otherwise stop with options, take your own
recommended option, write one line in `DECISIONS.md` ("author standing approval, 29 Oct"), and
continue. The option must meet all four conditions:
1. **Writes** go only inside `/ceph/dwong/jepa/`, and change only files this extension created.
   That includes archiving and then deleting its own temporary data.
2. It changes no registered prediction, criterion, cell, dataset, frozen decision or verdict. It
   does not touch the grid, and it **does not edit registered code**: `run.py`, `lucid_bombs.py`
   and the frozen configs are imported or copied, never modified.
3. It stays within the §0 budget, using the §3.4 drop order.
4. It writes nothing outside `/ceph/dwong/jepa/`, including `/tmp`, and touches nothing belonging
   to another user.
   - **Inputs** come from the repository clone: configs, geometry and code. The repo copy of
     `src/jepa_noise/configs/lucid_bombs_v2_calibrated.json` must match the sha256 recorded in
     `results/jepa/20261025_pregrid_checks/provenance.json`; record both.
   - Nothing under `/work/` is needed.

**Stop and report only for:**
- a real S3 change;
- an action that fails conditions 1–4;
- **S4**: no route can get the commit to GitHub. Before calling S4, try, in order:
  1. `git push origin JEPA` from the local clone at
     `~/Documents/VERITAS/jepa-handoff-20261028`. Its `origin` is
     `https://github.com/seanxia8/oracle.git`, the repo is **`seanxia8/oracle`**, and the earlier
     pushes (`f29137c` … `6cafd38`) went this way.
  2. The same from a Deepthought clone, if git there has credentials.
  3. `gh` authenticated as the author.

  A GitHub API or app-integration 403 ("Resource not accessible by integration") is not S4. Switch
  to the git routes above.
- **S7**: a §2.5 toy acceptance check fails after one fix attempt, which means the measurement
  itself is wrong.

Everything else is engineering. When a tool or route is blocked, find the best workaround within
conditions 1–4, log it in one line, and continue.

**Rules that still hold.**
- §6.1 hosts and execution path. Code reaches the hosts only through GitHub, one clone per SHA.
  Prefer a shallow or sparse clone, and remove a superseded extension clone after the next one
  passes its smoke.
- §6.2 guard.
- The no-touch packages.
- `TMPDIR=/ceph/dwong/jepa/tmp/` for every host command. Pytest uses a `--basetemp` inside
  `<ext>/tmp/` and `-p no:cacheprovider`; remove the basetemp after a successful run.
- Run the full suite **once per deployed SHA**.

**Budget.**
- CPU: ≤ 10 wall-hours on Deepthought, ≤ 16 cores, `nice -n 10`. Generation also counts against
  the 48 h generation cap; see §3.4.
- GPU: 0.
- Storage: ≤ 2 GB and ≤ 2 000 new entries under
  `/ceph/dwong/jepa/returns/ext_m_dose_20261030/`, which is `<ext>` below.
- Write nothing if free entries are below 30 000.
- Never store raw events.

**Host.** Run on Deepthought, CPU only. deepthought2 is not used.

---

## 1. What this answers

The theory (P1) says linear JEPA (CCA) is displaced when a **shared** nuisance's cross-view
canonical correlation competes with the **weakest** physics direction it must keep.

- **The toy:** the third physics mode is weak (canonical correlation ≈ 0.68) and the common mode
  is strong.
- **LUCiD, 29 Oct:** no top-10 CCA direction encodes module gain, at s = 0.05 or s = 0.30.

**Hypothesis.** Both testbeds lie on one curve: the nuisance share of the weakest kept CCA pair is
set by

  **r_k = σ_1^noise / σ_k^phys**,

the ratio of M's own two terms. The natural onset is near r ≈ 1. In the toy, at population level,
the weakest pair's nuisance share crosses 0.5 at g = 0.02, where r_3 = 1.04. LUCiD's registered
drift should sit at small r; pushing the nuisance up should produce the trap near r ≈ 1.

---

## 2. Definitions (identical in both testbeds)

### 2.1 Physics, noise and the conditioning set

For a view pair (context c, target t), the event covariance blocks are S_cc, S_tt and S_ct. The
**noise** blocks N_cc, N_tt and N_ct are covariances of the residual **x − E[x | Φ]**, where Φ is
the conditioning set below. The physics cross block is P_ct = S_ct − N_ct.

S = P + N holds exactly in population, also for multiplicative gain, because a
conditional-expectation residual is uncorrelated with every function of Φ.

| | Toy (A4 Tier-1 law) | LUCiD |
|---|---|---|
| Φ (conditioned) | θ = (amplitude, t0, tau) | vertex, yield, the per-PMT `qe_eff` (wavelength draw) and the emission offset. These are exactly the fields `ext_predictability.resample_noise` keeps fixed |
| How the covariances are obtained | exact population covariances, built as in `docs/jepa/check_v4_levels.py`: linearised fixture, base ρ = 0, E2 θ SD, channel split | estimated from generated photon events, §3.2 |
| k | 3 | 4 |

**LUCiD noise decomposition.** Two schemes:
- **N_B**, scheme B: run state **and** noise redrawn, Φ fixed. This is all non-physics variation.
- **N_A**, scheme A: run state fixed, noise redrawn.

N_gain = N_B − N_A is the part due to the drift state.

N_A contains the event-level trigger-time jitter, which both views share. So ν will **not** be 0
at s = 0; that baseline is a finding, not a bug.

### 2.2 The two terms, the ratio and M

With ε(S) = 1e-3·trace(S)/d:
- H_phys = (S_cc + ε)^(−1/2) · P_ct · (S_tt + ε)^(−1/2), and σ_k^phys = its k-th singular value;
- H_noise = (N_cc + ε)^(−1/2) · N_ct · (N_tt + ε)^(−1/2), and σ_1^noise = its top singular value;
- **M = σ_k^phys − σ_1^noise**, and **r_k = σ_1^noise / σ_k^phys**;
- also **r_k^S = σ_1(H_N^S) / σ_k(H_P^S)**, where both blocks are whitened by S, i.e. the split of
  the actual CCA matrix. In the toy, r^S and r agree; in LUCiD they need not.

Report two forms of M:
- **M_reg**, the registered protocol of v3 §2.3: a random QR projection to d = min(8, d_c, d_t)
  per view. The projection seed comes from `run._seed_entropy(2718, split_label)`, with labels
  `channel`, `within` and `across_fixed`. Apply it to the covariances as QᵀSQ.
- **M_full**: no projection, full feature dimension.

**Code.** `run.py` computes M inline in `run_m_diagnostic`, on its own generated data, so it
cannot be called on covariances. **Copy lines 1399–1411 of `src/jepa_noise/run.py` verbatim** into
`ext_m_dose.m_terms(scc, stt, sct, ncc, ntt, nct, k)`. Test 5 proves the copy is equivalent. Do not
edit `run.py`.

### 2.3 The outcome: nuisance share per canonical pair

Do **not** use `linear.cca` here. Its target rows are an ordered basis of the span, not the
canonical partners of the context rows.

Construct the pairs directly:
1. H_S = (S_cc + ε)^(−1/2) · S_ct · (S_tt + ε)^(−1/2) = A Σ Bᵀ (SVD).
2. u_j = (S_cc + ε)^(−1/2) a_j and v_j = (S_tt + ε)^(−1/2) b_j.
3. The nuisance share of pair j is
   **ν_(j) = a_jᵀ [(S_cc + ε)^(−1/2) N_ct (S_tt + ε)^(−1/2)] b_j / σ_j**.
4. The **primary outcome** is **ν\* = max over j ≤ k of ν_(j)**: the most-displaced kept pair.
   In the toy only pair 3 moves, so ν\* = ν_(3).
5. Also report the aggregate ν_k = Σ_{j≤k} ν_(j) σ_j / Σ_{j≤k} σ_j. Do not use it for readings:
   in the toy it saturates near 0.33, because pairs 1–2 are pure physics.

**Report** for each pair: ν_(j) from N_B, from N_A and from N_gain.
- Values are **unclipped**. Flag any value outside [0, 1]; that can happen legitimately when
  a_jᵀ·H_P·b_j < 0.
- Identity check: Σ_j a_jᵀ H_S b_j = Σ_j σ_j to 1e-10.

**Interpretable cross-checks** (report them; they carry no criterion):
- **LUCiD:** for each of the top 4 pairs, the held-out R²_gain. Use `ridge_r2`
  (`scripts/jepa_pregrid_checks.py`, line 137) as is, from the run's 105 ln gain_m to the
  run-mean projection u_jᵀx̄_run, split by run 50/50.
- **Toy:** D, the population displacement (`views.displacement`).

### 2.4 Sampling floors (LUCiD)

Estimate a shuffled-pair null by pairing each context residual with the target residual of the
**next** calibration event. Report:
- σ_1^noise,null and r_null;
- ν\*_null, the same ν construction with the shuffled N_ct.

A LUCiD level counts as **at floor** if σ_1^noise ≤ 1.2 × σ_1^noise,null.

### 2.5 Toy acceptance values (S7)

These are population values, computed by the reviewer and reproduced by the author's session.

| g | ν_(3) | r_3 | D | M_full | M_reg |
|---|---:|---:|---:|---:|---:|
| 0 | 0 (within 1e-6) | 0 | 0.00° | +0.682 | +0.0099 |
| 0.005 | 0.043 | 0.530 | 11.27° | +0.317 | −0.0025 |
| 0.02 | **0.525 ± 0.005** | **1.036** | 47.96° | **−0.024** | −0.038 |
| 0.05 | 0.918 | 1.281 | 76.95° | −0.186 | −0.102 |

Tolerances: D within 0.5°; r and M within 0.005.

**Note on the registered E1b value.** At g = 0.02 the population M is **negative**. The registered
E1b development value (+0.0135, channel split) is a finite-sample estimate: n = 4096 and d = 8, and
its noise singular values (0.05–0.08) sit at the sampling floor. A mismatch between the population
M and that number is **expected and is not an S7 failure**. The pairing "positive M with D > 0"
does hold at population level for g ≤ 0.01: M_full = +0.14 at g = 0.01, with D = 22°.

---

## 3. Data and computation

### 3.1 Toy (minutes)

Use the 9 g levels {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9} at line 0, channel split,
k = 3, exact population. For each level compute:
- M_reg, M_full;
- r_3, r_3^S;
- ν_(1..3), ν\*, ν_3;
- D.

Check §2.5 before going on.

### 3.2 LUCiD dose levels (generation, photon only, common random numbers)

**Inputs.** The repo copies of the frozen A10 generator config (`lucid_bombs_v2_calibrated.json`,
after the hash check) and geometry, and readout `readout_pmt_r6`.
- **Do not edit the config.** It is mixed photon/muon/electron, and `generate_run` follows its
  domain plan.
- Instead call `draw_event(..., domain="photon")`, then `make_readout`, then `tokenise`, as
  `ext_predictability.tokenise_realisations` does. Use `ViewMasker` with n_min = 0, and report the
  count of events below N_MIN.

**IDs.** Use the **same run and event IDs at every level**. Then the levels differ only in s,
because u_m is keyed by run_id. Encode the namespace in the run IDs:
- `ext_m_dose/evt/<run>` for the event set;
- `ext_m_dose/cal/<run>` for the calibration set.

**Levels.** The knob is the module-gain spread. F3 is at its registered s_p in every level.

| Law | s values | What it is |
|---|---|---|
| **F1** (registered law, clipped to [0.5, 1.5]) | 0, 0.02, 0.05, 0.10, 0.20, 0.30 | via `state_for_run('F1', s, …)`; report the clipped fraction |
| **F1x** (extension only) | 0.50, 0.75, 1.00, 1.50 | lognormal, **no clipping**, label `F1x_stress_unphysical` |

F1x is built without editing `lucid_bombs.py`:
1. `st = state_for_run('F1', 0.0, run_id, …)`, which gives F3 with module gain 1.
2. `u = keyed_rng(seed, NS_F1, run_id)`, the same draw F1 uses.
3. `replace(st, law='F1x', setting=s, module_gain=exp(s·u), gains=exp(s·u)[module_of_pmt]·st.pmt_gain)`.

Gain is applied after the Poisson draw (`lucid_bombs.py` ≈ l.2105), so F1x cannot overflow the
count capacity. F1x exists only to reach large r; it is not a detector model.

**Per level:**
1. **Event set:** 1 000 runs × 10 events = 10 000 events, for S, the CCA pairs and R²_gain.
2. **Calibration set:** 1 000 further events.
   - For each event, R = 8 scheme-A realisations: the run's state, fresh noise.
   - Also R = 8 scheme-B realisations: fresh state and fresh noise. Reuse
     `ext_predictability.resample_noise`. For scheme B, take a fresh state seed per realisation
     exactly as `ext_predictability.scheme_b_state` does, then build the state with that seed:
     F1 through `state_for_run`, F1x through the recipe above.
   - Residuals = realisation − mean over that event's 8, multiplied by √(8/7). Accumulate N_A and
     N_B, plus the shuffled null, in float64.

**Features.** Module-pooled, as in Part 2 of 29 Oct: per module and per view, the mean over that
view's PMTs of (log charge, time, hit flag).
- `within`: 315 + 315 dimensions.
- `across_fixed`: the first half of the module Hilbert order as context; 156 + 159 dimensions.
- `pool` and the view groups live in `scripts/jepa_ext_predictability_linear.py`. Import them from
  there or make a minimal copy. Its `extract` hard-codes 400 runs; do not reuse that part.

**Storage.** Stream events into float64 running sums. Per level, store one `.npz` holding:
- S, N_A, N_B and the null blocks;
- per-run feature means (1000 × 630);
- per-run ln gain_m (1000 × 105);
- the clip counts.

About 10 MB per level.

**Per level and split, compute:**
- M_reg, M_full;
- r_4, r_4^S;
- ν_(1..4) under N_B, N_A and N_gain;
- ν\*, ν_4;
- the floors;
- R²_gain for the top 4 pairs.

### 3.3 Order and smoke

1. The toy (§3.1).
2. F1 s = 0.05, as the smoke. Pass if:
   - it runs end to end;
   - the top-4 `within` canonical correlations are all > 0.8;
   - the held-out R²_gain of the top 4 pairs is < 0.1, consistent with 29 Oct.

   If it fails, investigate once. Record the outcome and continue; flag it in the report.
3. The remaining levels, in ascending s, F1 then F1x.

### 3.4 Budget rule and drop order (never stop for budget)

**Remaining.**
- The remaining generation budget = 48 h − the CPU generation hours already booked in
  `RESOURCE_LEDGER.json` (`v4_caps` and the extension lines).
- The allowance = min(10 h, remaining).

**Projection.** After the smoke, project the total from the measured per-level time. The
reviewer's estimate is about 0.35 h per level and about 4 h serial.

**Drop order.** If the projection exceeds the allowance, drop levels in this order and record each
drop: F1x 1.50, F1 0.02, F1x 1.00, F1 0.20.

---

## 4. Pre-declared readings (descriptive, not gates)

Report each one as **supported**, **not supported** or **unclear**, with the numbers. LUCiD uses
the `within` split and N_B unless stated. Write any further tolerance into `DECISIONS.md` before
computing.

| ID | Reading | Supported if |
|---|---|---|
| M-1 | On LUCiD, the share follows the ratio | Spearman(r_4, ν\*) ≥ 0.8 over LUCiD's nonzero levels, F1 and F1x together. The toy curve is reported beside it; it is monotone by construction and carries no criterion |
| M-2 | LUCiD's registered drift is in the safe regime | At F1 s = 0.05 and at s = 0.10: (r_4 < 0.530, the toy's r_3 at g = 0.005, **or** the level is at floor) **and** ν\* ≤ 0.1. Report ν\* under N_gain beside it |
| M-3 | The trap appears on LUCiD when the nuisance is strong enough | Some scanned LUCiD level has ν\* ≥ 0.5, and its ν\* exceeds ν\*_null by ≥ 0.2 |
| M-4 | Same onset in both testbeds | Evaluable only if M-3 is supported. Take the first LUCiD level (ascending s) with ν\* ≥ 0.5. Supported if its r_4 lies in [0.6, 1.2]. The toy onset is r_3 = 1.036 at g = 0.02. Report r^S beside it |
| M-5 | ν is interpretable on LUCiD | At every level with ν\*_gain ≥ 0.5, at least one of the top 4 pairs has held-out R²_gain ≥ 0.5. At every level with ν\*_gain ≤ 0.1, none does |
| M-6 (sanity check) | The split behaves as constructed | At levels where ν\*(within) ≥ 0.2: ν\*_gain(across_fixed) < ν\*_gain(within). Expected by construction, because F1 gains are independent per module and `across_fixed` views share no module |
| M-7 | The registered M protocol ranks like the full one | Spearman(M_reg, M_full) ≥ 0.8, computed **separately** for the toy and for LUCiD |

---

## 5. Code and tests

**Code.** Add `src/jepa_noise/ext_m_dose.py` and `scripts/jepa_ext_m_dose.py`. Import from
`ext_predictability`, `lucid_bombs`, `point_encoder`, `metrics`, `views` and `run`
(`_seed_entropy` only). Copy only what §2.2 and §3.2 allow.

**Tests** in `src/jepa_noise/tests/test_ext_m_dose.py`:
1. On a 6 + 6-dimensional Gaussian fixture with known P and N at n = 20 000, the estimated ν_(j)
   and r_k match the population values within 0.02.
2. With N_ct = 0 (N_cc and N_tt positive definite), every ν_(j) = 0 and r = 0. With P_ct = 0 and
   N_ct ≠ 0, every ν_(j) = 1.
3. The pairing identity: Σ a_jᵀ H_S b_j = Σ σ_j to 1e-10. The ν construction does **not** call
   `linear.cca`.
4. F1x: gain_m = exp(s·u_m) with no clipping, and the same u_m as F1 for the same run_id. F1 at
   s = 0.30 through `state_for_run` is unchanged bit for bit.
5. `m_terms` equivalence. Regenerate the E1b development channel-split covariances exactly as
   `run_m_diagnostic` does (same generator calls, seeds and projection) and feed them to
   `m_terms`. Its score must equal the registered +0.013519688568 within 1e-9.
6. The §2.5 toy acceptance values.

The full suite must pass once on the deployed SHA; record the count.

---

## 6. Write-up, commit, report, wait

1. Results go in `results/jepa/20261030_m_dose/`, with a README marked `registered: false`, plus:
   - `m_dose.md`: one table, rows = every toy and LUCiD level and split, columns = M_reg,
     M_full, σ_k^phys, σ_1^noise, r_k, r_k^S, ν\* (N_B / N_A / N_gain), ν\*_null, ν_k, R²_gain
     (top 4) or D, and the clipped fraction;
   - `nu_vs_r.png`: ν\* against r, both testbeds, marked by testbed and law, with the floor band;
   - JSON summaries, each ≤ 1 MB.
2. The `.npz` files stay in `<ext>`, with their hashes recorded.
3. `DECISIONS.md` gets one line per engineering choice. `STATE.md` gets a short "M dose" entry.
   Book CPU wall-hours in `RESOURCE_LEDGER.json`.
4. Commit and push `JEPA` after the toy and the smoke, and again at the end.
5. **Report to the author (≤ 1 page, in Chinese):**
   - the condensed table;
   - the figure;
   - the seven readings with their numbers;
   - resources used;
   - the final pushed SHA.
6. **Wait for the author.** Do not resume the grid, and do not propose a grid change without a
   gate.
