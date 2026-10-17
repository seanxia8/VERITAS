# PMT Decisions Log

## Author confirmations (runbook v2 §2)

### C1 -- Retroactive authorisation of GPU/remote/CERNBox runs

**Answer (17 Oct 2026):** "追认" (ratified) -- option **(A)**. The `181b747` / `56179eb` GPU, remote and CERNBox runs are accepted retroactively as **development evidence**. Their recorded verdicts stand, and nothing in them is a reported result.

### C2 -- Flags for the next execution

**Answer (17 Oct 2026):** "你的建议合理，采纳" (the recommendation is adopted) -- `PMT_REMOTE=yes`, `PMT_GPU=yes`, `PMT_DATA_CAP_GB=5`.

Authorisation relied on for remote/GPU work, quoted from the launch prompt `docs/AGENT_PROMPT_JUNJIE_PMT_v2_2026-10-16.md` (committed `c2b0a3a`, 18 Oct 2026): "PMT_REMOTE=yes # C2, author, 17 Oct 2026; within runbook v2 §2 shared-server rules", "PMT_GPU=yes # one Titan job at a time on Deepthought only (§2)", "PMT_DATA_CAP_GB=5".

---

## Scientific decisions: development settings (runbook v2 §3 recommended defaults)

These apply to engineering and development runs only. None is adopted for primary evaluation until the author marks it adopted (v2 §3, last paragraph).

| # | Decision | Development setting | Status |
|---|---|---|---|
| D-TQ | T-Q walk shape | `walk(q) = a + b/sqrt(q)` on [0.25, q_sat]; constant above q_sat. a=0, b=2 ns, q_sat=50 PE. Intervention b × (1 ± 0.2), plus a separate offset-only control | `synthetic_dev`, not adopted |
| D-Q | Charge domain | [0.25, 1000] PE everywhere; overflow clamped and counted | development setting, not adopted |
| D-J | Timing kernel | Gaussian approximation; fraction of digits before photon arrival reported per cell | development setting, not adopted |
| D-G | Intervention groups | mPMT modules (physical map from pinned file) | development setting, not adopted |
| D-K | Fit failures | Unevaluable if failure rate > 10%; otherwise complete-case K with failure-rate difference reported; kappa_m = 1.10 `provisional_dev` | development setting, not adopted |
| D-E | Energy endpoint | (A) Deposited energy inside instrumented volume; initial energy supporting on contained events only; containment preflight before training | development setting for R-4; open with Junjie |
| D-T/L/M/S | Portfolio | Unchanged from v1 WP8 | open with Junjie |

---

## Engineering decisions

_One line each, logged as made (18 Oct 2026)._

- E1. The pre-existing draft of this log (untracked, 18 Oct 00:29) called the §3 defaults "adopted". That word was changed to "development setting" to match v2 §3.
- E2. Interpreter: `/Users/wongdowling/Documents/VERITAS/.venv/bin/python` (Python 3.12.7, NumPy 2.5.3, SciPy 1.18.1, pytest 9.1.1), the same one the 16 Oct runs recorded. It is only executed, with `-B` and `-p no:cacheprovider`, and never written to. It has no torch, so the two torch tests skip locally.
- E3. `src/noise_module_lucid/endpoint_identifiability.py` was found untracked (mtime 16 Oct 21:48, after `56179eb`), and no prompt mentions it. It was left uncommitted and unused pending author review.
- E4 (R-1). Manifest schema `pmt-response-v2`. Per-channel factors are a rule on a channel-to-group map (`base + delta × (group mod 2)`), so one manifest serves the 76-sensor fixture and WCTE. The values are unchanged from v1 (gain +3%, width +5% on odd groups).
- E5 (R-1). The integration and control pilots now group by physical mPMT module (D-G). Before, `lucid_integration.response` used `channel // 19`, which is not the module map. The multi-energy pilot already used mPMT.
- E6 (R-1). One QE table everywhere: the pilots' 5-anchor table (200–800 nm, zero at the ends). It equals the v1 fixture table on 300–600 nm, the toy spectrum's range.
- E7 (R-1). Fixture photon yield restored to the v1-declared 4000 (it was cut to 1000 only to fit the old 100 PE domain). Seeds, closure bounds and tolerances are unchanged.
- E8 (R-1). The literal-response scan flags a response constructor or a `replace` of a response field whose argument contains a numeric constant and no attribute of an existing response (`np.*` does not count). Bare `ReadoutParams()` is allowed only in `dataset.py`/`readout.py`, as the legacy toy chain with no QE or T-Q.
- E9 (R-2). The walk uses a = 0 on every channel, a single curve rather than the old per-channel 2 + 0.05·(i mod 3) constants, which cancelled in calibrated time anyway.
- E10 (R-2). Fixture T-Q cells: `fixed_tq_slope_up` (b × 1.2), `fixed_tq_slope_down` (b × 0.8) and `tq_offset_control` (a + 1 ns), on all channels. They replace the old odd-channel polynomial cell.
- E11 (D-Q). Clamping acts on the measured, gain-scaled charge before the discriminator, walk and width. The reported digit charge is the clamped value. The count is `overflow_clamped`. Without a declared `charge_domain_pe` (legacy), out-of-domain charge still raises.
- E12 (D-J). `digits_before_arrival` counts digits whose raw time minus first photon arrival is negative: cell walk plus jitter in T-Q mode, the template walk in legacy mode, before any calibration subtraction or offsets.
- E13 (R-2). The T-Q validation uses one photon per channel, flat QE, spe_sigma = 0 and gain = q, with a zero reference walk to expose the raw curve; N = 20000 and seed 71. Each charge point has its own event ID (1000 + 100·q). Jitter is keyed per (seed, event, channel), so the five points are independent tests. The first version reused one event and gave one draw five times; the independent review caught it.
- E14 (R-2). The charge-dependence floor in the paired control is a peak-to-peak spread above 1e-9 ns, because floating-point rounding gives a constant shift a spread of about 1e-15.
- E15–E23 (R-3, implemented by a subagent and reviewed):
  - E15. The constant predictor is the pooled training-split mean, evaluated unchanged in every subgroup.
  - E16. The vertex and energy gates use ≤; the direction gate uses a strict < 20°, both pooled and in every energy subgroup.
  - E17. Energy ratios per subgroup are reported, not gated.
  - E18. An undefined denominator, a non-finite metric or an empty or missing subgroup fails the gate, with a reason.
  - E19. Expected subgroups are `cfg['energies_MeV']`.
  - E20. The energy endpoint is chosen by column and name, and must be encoded as log(E/1000 MeV).
  - E21. The old absolute gates are removed for future runs, and the MSE-below-constant gates are kept.
  - E22. The past run's constant was reconstructed from the seeded placements and the +z gun axis, and it reproduces every recorded constant metric exactly.
  - E23. Thresholds are in `configs/lucid_pmt/trainability_gates.json`.
- E25. Independent review fixes:
  - D-Q/D-J counts are recorded per event and per run in both LUCiD pilots.
  - `overflow_clamped` counts final digits only.
  - In T-Q mode the raw delay includes the template residual when the threshold differs from calibration.
  - The literal scan is now builder-only for response constructors. Literal-valued field replacements are allowed only inside `ResponseCellSpec(...)` or a declared construction. It flags all four HEAD pilots. Known blind spot: a literal bound to a name first.
  - The exact QE×2/CE÷2 control moves to `response_manifest.exact_product_control`.
  - The offset control gets its own family, `tq_offset_control`.
  - The R-3 energy gate fails closed unless the run declares the endpoint `deposited_energy_in_volume`.
  - The past-run reproduction test skips when `results/` is not shipped.
  - Note added to the integration results README on the mPMT regrouping.
- E26. Known, not fixed: on the manifest response, a threshold below 0.25 PE raises (charge below the walk domain) instead of a counted rejection. This is latent, because the legacy `threshold_walk` grid is ≥ 0.35 and no T-Q cell moves the threshold.
- E24 (R-5). Runs 03/04 were removed from the tree (history `56179eb`). Checkpoint paths and hashes are in `results/PMT_CHECKPOINTS.json`. The integration-pilot root is inferred from the README and is re-verified on the host in R-4.

---

## Remote/GPU job log

| Date | Host | Job | Wall time | GPU-hours | Data read |
|---|---|---|---|---|---|
| 18 Oct 2026 | local | R-1 fixture re-run (`results/pmt_r1_r3_dev_2026-10-18/fixture_run_01`, pre-review) | 6.4 s | 0 | 0 |
| 18 Oct 2026 | local | R-1 fixture re-run after review fixes (`fixture_run_02`, final) | 5.3 s | 0 | 0 |

_No remote or GPU job for R-1 to R-3; they ran on local CPU._
