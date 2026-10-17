# DECISIONS — JEPA study (append-only)

Runbook v4: `docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`. One line per engineering decision
(v4 §5). Author approvals are quoted verbatim with date; nothing is inferred.

**Format.** `YYYY-MM-DD — decision — basis/authority.` Entries are never edited; a reversal is a
new entry that cites the old one.

---

## Author decisions and instructions received (verbatim, dated)

- **2026-10-17 — Launch prompt v4 issued.** Flags: `PUSH_JEPA=yes`, `GPU_ALLOWED=yes`,
  `DEEPTHOUGHT_GPUS=4`, `DEEPTHOUGHT2_GPUS=2`, `CONDOR_NEMO2=no`, `OPTIONAL_WPS=WP7`.
  Authority: "If this conflicts with those files, they win." (launch message to the orchestrator).
- **2026-10-17 — PMT readout, not yet named.** Author: "I have not yet named the PMT-branch SHA.
  Develop on the unrepaired readout, labelled `readout_synthetic_unrepaired`, and produce no
  reported LUCiD results until I add the SHA to `DECISIONS.md`." → §3.5.2 [A8] development path.
- **2026-10-17 — Drift ranges pending Junjie.** Author: "s, s_p and a are still to be confirmed
  with Junjie. Use the runbook defaults, logged in `DECISIONS.md`, until I change them, and change
  them only before training." → defaults listed below.
- **2026-10-17 — Ambiguity rule.** Author: "When unsure whether something is an engineering choice
  or a specification change, treat it as a specification change: write a gate with lettered
  options, and keep working on everything that does not depend on it."
- **2026-10-17 — Runbook/companion files are the author's.** "Commit them unchanged."
  "Leave `docs/LUCID_MODEL_LINEUP_2026-10-09.md` untouched; it is mine."

## Registered specification (WP0-v4)

- **2026-10-17 — Runbook v4 registered unchanged**, commit `151f26e`
  `jepa: register runbook v4 (author revision)`. SHA-256:
  - `RUNBOOK_JEPA_NOISE_v4_2026-10-16.md` = `3a489d305df6679db234e24f5b76ebb33a0e6740185e3d9e2a458c6830d8c428`
  - `AGENT_PROMPT_JEPA_v4_2026-10-16.md` = `ca68c61879b381feed36fa7f47725a9ea549372ea71df0a81584d2e0a2d0882f`
  - `check_v4_levels.py` = `4fa55821f7714c522bc3cace7a6eabfba1e51c9b7af6af3355556a325d93faf5`
  v3 stays byte-identical and its kept sections remain in force (v4 §2).
- **2026-10-17 — CPAL 2027 dates verified (WP0-v4 step 2).** Recent Spotlight Track submission
  deadline **18 January 2027, 23:59 AoE**; notification 1 February 2027; conference 23–26 March
  2027, Hitotsubashi Hall, Tokyo. Sources: <https://cpal.cc/call_for_papers/> (Important Dates) and
  <https://cpal.cc/deadlines/>, fetched 2026-10-17. This confirms runbook §9's planned
  "18 Jan 2027"; the plan's last milestone (8 Jan 2027) keeps 10 days of margin.
- **2026-10-17 — WP3 E1 budget projection recorded**, commit `dab95d8`. Hygiene (v4 §7): 56 KB
  JSON < 1 MB, 13 KB script, no raw arrays, provenance hashes match the registered v3 runbook.
  Status in the file itself: PROJECTION ONLY — it clears no gate.

## Drift-law parameter defaults in force (runbook §3.5.3; change only before training)

Until the author or Junjie changes them, these are the values every LUCiD cell uses:

- **F1 module gain:** s ∈ {0.02, 0.05, 0.10} (main cells); gain_m = 1 + s·u_m.
- **F3 independent per-PMT gain:** s_p = 0.05 in every LUCiD cell (may be raised to ≤ 0.10 only by
  §3.5.3a step 3 — that is the registered fallback, not a new choice).
- **F2 attenuation:** λ = λ0·exp(a·v), a from {0.10, 0.20, 0.50}; **the value is set by the
  §3.5.3a detectability check**, not chosen here. F2′ fallback b ∈ {0.05, 0.10}.
- **Law R11b (co-varying):** s = 0.05. **Law T (aging):** s_T = 0.10, R = 100 training runs.
- **Gain clipping:** each factor clipped to [0.5, 1.5] separately; clipped counts reported (A8).
- **Tier-1 bridge:** A4 law at g = 0.02, nominal SNR, channel split; θ0 = (4.5, 0.35, 0.08);
  E1b θ SD = (3, 0.1, 0.04)/√12 = (0.8660254, 0.0288675, 0.0115470).

## Engineering decisions (v4 §5 "decision default")

- **2026-10-17 — SSH targets.** The bare names `deepthought` / `deepthought2` do not resolve on this
  Mac (DNS). Use `dwong@deepthought.etp.kit.edu` and `dwong@deepthought2.etp.kit.edu` with
  `-o BatchMode=yes -o ConnectTimeout=20`. Verified 2026-10-17T11:39Z (deepthought: 40 cores;
  deepthought2: 128 cores). Same hosts as the `~/.ssh/config` entries.
- **2026-10-17 — Local interpreter.** `/private/tmp/jepa-venv-cpu311/bin/python` (3.11.11,
  numpy 1.26.4, scipy 1.17.1, torch 2.5.1 CPU, pytest 9.1.1, sklearn 1.9.1, jaxtyping present).
  This is the already-accepted local environment from v3; 75 existing tests pass in 5.9 s. No new
  local environment.
- **2026-10-17 — E1 worker count.** 3 = min(physical cores − 2, floor(free GB / 4)) at 10 cores /
  16 GB (v4 §6). Recorded per run; peak per worker < 4 GB.
- **2026-10-17 — Dirty-tree rule.** The v4 registration commit contains exactly the three v4 files,
  unchanged. The only file kept deliberately untracked is `docs/LUCID_MODEL_LINEUP_2026-10-09.md`
  (the author's). Every outcome-generating run records the code SHA and the working-tree status.
- **2026-10-17 — Budget ledger gains a `v4_caps` block.** Titan 300 GPU-h, L40S 80 GPU-h, LUCiD
  generation CPU 48 wall-h (§3.5.6/§6.1), with v3's E1 24-wall-hour cap kept. The v3 campaign rows
  are unchanged; the new caps live in a separate block so no v3 accounting is rewritten.

## Open questions (not gates yet)

- **PMT-branch SHA** not named (blocks reported LUCiD results, not development; G-PMT only if the
  repairs are not accepted by 31 Oct 2026).
- **Drift ranges s, s_p, a** pending Junjie; defaults above in force.
- **`docs/LUCID_MODEL_LINEUP_2026-10-09.md`** is untracked and untouched, per the author.

## Verification records

- **2026-10-17 — Appendix A reproduced at registration SHA.** Ran the registered script
  `PYTHONPATH=src python docs/jepa/check_v4_levels.py` at `151f26e` with the accepted v3 views
  (the script builds the A4 covariance directly from `base_rho = 0` and the E2-matched θ SD, so it
  does not need the WP2-v4 code). A4 channel-split angle k = 3: 0.0, 11.3, 22.3, 48.0, **76.9**,
  84.1, 87.5, 88.8, 89.4° for g = 0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9 — §A's 77.0°
  at g = 0.05 differs by 0.1°, within the 0.5° tolerance; γ_n matches §A exactly (0.000, 0.357,
  0.526, 0.690, 0.847, 0.917, 0.962, 0.980, 0.990); interleaved agrees within 0.5°. The v3 table
  (base ρ = 0.4, v3 θ SD, channel) reproduces exactly: 18.4, 74.0, 84.6, 89.2°. Output:
  `results/jepa/20261017_appendixA_check/check_v4_levels_151f26e.txt`.
