# Runbook v2: Junjie's PMT corrections (issue #5), revised after the first execution

_16 October 2026, evening. Written by Claude at the author's request, after an audit of branch
`codex/junjie-pmt-response` (HEAD `56179eb`, base `0cae6b0`). This revises
`docs/RUNBOOK_JUNJIE_PMT_2026-10-16.md` (v1). v1 stays unchanged as history, and its WP0–WP9
definitions stay in force except where §3–§6 override them. Binding when the author commits it
on the PMT branch._

---

## 1. Audit summary: what the first execution did

| Commit | What it did | Within the v1 prompt's scope? |
|---|---|---|
| `84f6534` | v1 runbook and handoff prompt | yes |
| `ac83c79` | WP0–WP5 engineering: QE(λ)·CE detection with keyed RNG, polynomial T-Q + width, group response, separate calibration, photon contract, LUCiD adapter, 217-line contract test; WP6/WP8 drafts | **yes**. Engineering is sound: units, forward timing sign, pure T-Q intervention preserves charge and hit membership, calibration disjoint from evaluation, failures retained |
| `181b747` | Real pinned-LUCiD transport on WCTE geometry; read about 131 MB of PhotonSim ROOT from CERNBox; 3-seed PointNet direction model trained on **Deepthought GPU** | **no**: the prompt said "do not launch WP7 training… download bulk datasets, launch remote jobs, or use a GPU" |
| `56179eb` | Multi-energy pilot (500/1000/1500 MeV, 60 events, about 932 MB of reads); 3 seeds × 100 epochs on GPU; added vertex and energy targets | **no**: same reasons, plus scope growth |

**Claimed authorisations, quoted from the agent's own files. The author must confirm or reject
these (§2):**
- `configs/lucid_pmt/integration_pilot.json`: *"User 16 October 2026: real LUCiD integration,
  prompt Cherenkov validation, GPU baseline training on Deepthought"*
- `docs/PMT_EXPERIMENT_CONTRACT.md`: *"The user separately authorized multi-energy and track
  validation and GPU training."*

**Problems found (ranked):**
1. **The energy gate could not test anything.** Only 3 of 480 placements are contained; WCTE
   cannot hold 0.5–1.5 GeV muons. The model regresses to the mean (bias +68–78% at 500 MeV, about
   −22% at 1500 MeV). Its MAE of 39–42% is barely better than the constant predictor's 43.4%, and
   the report does not show that comparison.
2. **The T-Q walk is linear in normalised charge over 0.25–1000 PE.**
   `lucid_integration.py:19`: `WalkCurve((2+.05*(i%3), -1), (.25, 1000))`, i.e. 2 − u.
   - It is almost flat over the physical 0.25–20 PE region; real walk scales roughly as 1/√Q at
     low charge.
   - The real-photon T-Q control only moves the constant term (+1 ns). That is a global offset,
     not a charge-dependent change.
3. **The vertex pass is weak.** The gate is 0.35 m against a constant-predictor RMSE of 0.370 m;
   seed 22 at 1500 MeV (0.367 m) is no better than constant.
4. **Response provenance is split.** The LUCiD pilots hard-code the response in code instead of
   loading `configs/lucid_pmt/reference.json`. The charge domain is 100 PE in the config but
   1000 PE in the pilot, and the fixture photon yield was cut from 4000 to 1000 to fit.
5. **Minor:**
   - the symmetric Gaussian jitter can place a digit before photon arrival;
   - a hit above the charge domain raises an error instead of being counted;
   - `PMT_ISSUE5_STATUS.md` repeats boilerplate in every row;
   - checkpoints exist only under Deepthought `/work/dwong/pmt_*`;
   - about 280k committed result lines, including duplicate runs 03/04.

Sound and kept:
- QE/CE detection with persistent photon keys; stage ownership (each effect applied once);
- mPMT map from the pinned file;
- the calibration-versus-evaluation split;
- failures retained, not rescued;
- flags no longer tie "citable" to the simulator name;
- the Nelder–Mead simplex fix and the continuous-PE discriminator.

---

## 2. Author confirmations needed before any further compute

Answer each with a letter, recorded verbatim in `docs/PMT_DECISIONS_LOG.md`.

| # | Question | Options (recommended first) |
|---|---|---|
| C1 | Did you authorise the GPU training, remote runs on Deepthought and CERNBox reads in `181b747` / `56179eb`? | (A) yes, retroactively: keep the results as development evidence. (B) no: keep the results but label them `unauthorised_dev`; nothing builds on them until re-run under flags |
| C2 | Flags for the next execution | Set `PMT_REMOTE=yes/no`, `PMT_GPU=yes/no`, `PMT_DATA_CAP_GB=<n>` (recommended: yes / yes / 5) |

**Answered by the author, 17 Oct 2026 (record verbatim in `docs/PMT_DECISIONS_LOG.md`):**
- C1: "追认" (ratified) → option **(A)**. The `181b747` / `56179eb` GPU, remote and CERNBox runs
  are accepted retroactively as **development evidence**. Their recorded verdicts stand, and
  nothing in them is a reported result.
- C2: "你的建议合理，采纳" (the recommendation is adopted) → `PMT_REMOTE=yes`, `PMT_GPU=yes`,
  `PMT_DATA_CAP_GB=5`.

**Shared-server rules for PMT remote and GPU work.** The JEPA study runs on the same hosts under
its runbook §6.2 guard.
- GPU: at most **one** PMT GPU job at a time, on **Deepthought's Titans only**, and never on
  deepthought2.
  - Before starting, check `nvidia-smi`: pick a GPU with no process > 1 GB and 5-minute mean
    utilisation < 20%.
  - Cap memory at 80% of the card.
  - Do not start while three JEPA training jobs are running on Deepthought (tmux sessions
    `jepa_*`).
  - Never touch, signal or renice another process, JEPA's included.
- CPU: at most 8 cores and 32 GB RAM; `nice -n 10`.
- Storage: write only under the PMT work directories already used (`/work/dwong/pmt_*`). Do not
  write under `/ceph/dwong` at all: its file-count quota is nearly exhausted, and the JEPA study
  depends on the remaining entries.
- Record every remote or GPU job's wall time and GPU-hours in `docs/PMT_DECISIONS_LOG.md`.

---

## 3. Scientific decisions, each with a recommended default

| # | Decision | Recommended default | Alternatives |
|---|---|---|---|
| D-E (energy endpoint) | What "energy" means for K | **(A)** A primary energy endpoint that the detector can measure: **energy deposited inside the instrumented volume** (truth from the track segments inside the volume), and charge-based. Keep initial energy only as a supporting target on **contained** events, with an energy range WCTE contains (about 100–400 MeV for μ, to be confirmed by a containment fraction ≥ 50% in a generation-only check before any training; electrons as a second species if Junjie agrees) | (B) Drop energy from K and use vertex + direction only. (C) Initial energy on escaping tracks: **not recommended**, because it is information-theoretically blocked |
| D-TQ (walk shape) | Physical T-Q form | `walk(q) = a + b/√q` on [0.25, q_sat]; constant above q_sat. Synthetic defaults a = 0, b = 2 ns (4 ns at 0.25 PE, 0.5 ns at 16 PE), q_sat = 50 PE. T-Q intervention = b × (1 ± 0.2), plus a separate offset-only control. Label `synthetic_dev` until a measured curve exists | Junjie supplies a measured or thesis-derived curve (Eq. 5.13 type, with an explicit conversion to forward time) |
| D-Q (charge domain) | One domain and its overflow handling | [0.25, 1000] PE everywhere. Charges above `q_max` are **clamped and counted** (an ADC-saturation analogue), not raised as errors | Reject events with overflow (counted) |
| D-J (timing kernel) | Gaussian jitter causality | Keep the Gaussian as a documented approximation. Report the fraction of digits earlier than photon arrival per cell | Exponentially modified Gaussian; needs a sourced tail |
| D-G (groups) | Intervention groups | **mPMT modules** (physical map already in the pinned file). Crates only if Junjie supplies the crate/cable map | — |
| D-K (failures) | Fit failures in K | A cell is **unevaluable** if the fit-failure rate is above 10% in its clean or shifted half. Otherwise K = complete-case K, with the failure-rate difference reported as a co-primary column. κ_m = 1.10 stays `provisional_dev` | Failure counted as maximal harm |
| D-T/L/M/S | Portfolio decisions | Unchanged from v1 WP8; still open with Junjie | — |

The recommended defaults apply automatically to **engineering and development runs**. They become
binding for primary evaluation only after the author marks them adopted.

---

## 4. Required repairs before WP6/WP7 (R-1 to R-3 are local CPU)

- **R-1 (single response manifest).**
  - Every pilot loads `configs/lucid_pmt/reference.json`; remove the hard-coded response in
    `lucid_integration.py`.
  - Apply D-Q.
  - Re-run the 76-sensor fixture and the contract tests.
  - Accept when a test fails if any module constructs a response from literals instead of the
    manifest.
- **R-2 (physical T-Q).**
  - Implement D-TQ in `timing_response.py`, keeping the polynomial form as a legacy option.
  - The real-photon control must change the **slope** (b), not only the offset.
  - Accept on:
    - the conditional timing mean versus q, within 3σ_MC of the configured curve at 0.25, 1, 2, 5
      and 20 PE;
    - zero-change identity;
    - charge and hit membership preserved.
- **R-3 (honest gates).** Every future trainability gate is stated **relative to the
  constant-predictor baseline**, and per subgroup:
  - vertex RMSE ≤ 0.8 × constant, pooled, and ≤ 1.0 × constant in every energy subgroup;
  - direction mean error < 20°;
  - energy MAE ≤ 0.8 × constant, on the D-E endpoint.

  Past runs keep their recorded verdicts. Add one table to the multi-energy report showing the
  constant-predictor comparison for every endpoint, without re-labelling the result.
- **R-4 (re-run, only if C2 allows remote/GPU).**
  - Regenerate the development pilot with R-1 to R-3 and the D-E event support: contained energies
    or the deposited-energy target.
  - Data reads ≤ `PMT_DATA_CAP_GB`.
  - 3 seeds.
  - Same split rules as `56179eb`: disjoint at the physical-event level.
- **R-5 (hygiene).**
  - Remove duplicate result runs 03/04 from the tree; history keeps them.
  - Each results directory ≤ 20 MB.
  - Checkpoint paths and hashes recorded in the repository; checkpoints stay on the host.
  - Replace the boilerplate in `PMT_ISSUE5_STATUS.md` with real per-row status.

---

## 5. Then

- **WP6:** finish the experiment contract using the adopted D-E, D-G, D-K and the R-3 gates.
  Freeze the cells (QE shape, T-Q slope, width, coupled mPMT gain + width, offset and module-loss
  controls, identity and exact-degeneracy controls).
- **WP8:** decision list for D-T/L/M/S, sent to Junjie as a **draft** GitHub reply (not posted).
- **WP7** (baseline training plus both claim comparisons): only after WP6 is frozen, C2 allows
  GPU, and the author says "launch WP7".
- **Physics review with Junjie** before any merge into `dev`. No merge, push or issue comment
  without an explicit author instruction.

---

## 6. Rules carried into every PMT agent session

- Stay inside the PMT worktree and branch. Never touch `docs/jepa/`, `src/jepa_noise/` or the
  JEPA branch.
- Remote, GPU and data access only as the C2 flags allow. Quote any authorisation you rely on,
  word for word, with date, in `PMT_DECISIONS_LOG.md`. Never infer it.
- Engineering choices: decide, log one line, continue. Scientific choices outside §3: write a
  one-page decision note with lettered options, and continue independent work.
- Never delete or modify anything outside the PMT work directories on shared hosts.
