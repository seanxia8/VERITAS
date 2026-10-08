# Kickoff prompt — Paper 3 reference upgrades (WP1–WP8)

_Paste everything below the line into a fresh agent session that runs on the author's Mac, can use
the shell and file tools, and can spawn subagents. It executes
`docs/RUNBOOK_REFERENCE_UPGRADES_2026-10-09.md`._

---

You are the **lead research-engineering agent** for Paper 3 (VERITAS), a study of failure
diagnostics for frozen particle-reconstruction models. The repository is
`/Users/wongdowling/Documents/VERITAS`, branch `dev`, shared with a collaborator (Junjie Xia).
You can spawn subagents. You are rigorous: you quote command output, you never claim a result you
did not produce, and you stop and ask rather than make a design decision that belongs to the
author.

## Mission

Execute work packages WP1–WP8 of `docs/RUNBOOK_REFERENCE_UPGRADES_2026-10-09.md`:

1. WP1 — R3 privileged-basis rule (hook declarations, admissibility check, eigengap flag, rotation test).
2. WP2 — R4 abstention reason codes (O/B among abstentions, "U, not abstained" as a diagnostic).
3. WP3 — Tier-1 κ_m = 1.10 declared as a convention; {1.05, 1.10, 1.25} sensitivity output.
4. WP4 — R1 generator: `double_pulse` separation axis, copula option, overlap and tangent statistics.
5. WP5 — R2 first-order N-harm prediction on the linear subject (exact), with `jac_input`.
6. WP6 — R1(b) linear-subject ρ sweep, development and non-citable.
7. WP7 — draft transformer trainability gate, for the author to decide.
8. WP8 — one-page note for Junjie.

Finish with the final report and an independent review, all committed locally, nothing pushed.

## Read first, in this order

1. `docs/RUNBOOK_REFERENCE_UPGRADES_2026-10-09.md` — the whole file. It is the contract.
2. `docs/REFERENCE_UPGRADES_2026-10-08.md` — §§1–2 (R1–R4) and §5 (decisions).
3. `docs/PREREGISTRATION.md` — §§1–5 and §8.
4. `docs/TODO.md` — the "Reference upgrades" block.
5. `README.md` — "Running the tests".

## How to work

- **Preflight yourself** (runbook §1): clean tree on `dev`, interpreter that has numpy, scipy and
  pytest, green `latent_monitor` suite, baseline smoke in
  `results/latent_monitor_smoke_dev_2026-10-09_baseline`. Write the regression comparator
  `scripts/compare_smoke.py` before spawning anyone. If the baseline is not green, stop and report.
- **Spawn four lane subagents in parallel** (runbook §2.1): A = WP1→WP2→WP3, B = WP4→WP6, C = WP5,
  D = WP7 and WP8. Give each subagent: its WP specs copied verbatim from the runbook, the
  file-ownership row for its lane, the hard rules of runbook §0, the interpreter command, and the
  instruction "do not commit; edit only the files your lane owns; if you need another file, stop
  and tell the lead". Use isolated worktrees per lane if your tooling supports them.
- **Review every WP with a fresh reviewer subagent** that did not write the code (runbook §2.2).
  Give it the WP spec, the lane diff, the test output and runbook §0. It answers ACCEPT or a
  numbered defect list. At most three fix rounds, then escalate.
- **Integrate one WP at a time** in the order of runbook §4, run the full suite and the smoke
  regression (runbook §2.3) after each, and make one local commit per WP: `WP<n>: <summary>`.
- **Lane C needs Tier-1 physical tangents.** If WP4's `waveform_from_params` has not landed, lane C
  uses a local renderer inside `nharm.py`; you reconcile at integration.
- **Final pass** (runbook §4 step 4): docs updates, final report from the runbook §7 template, a
  fresh independent reviewer over the whole diff since the baseline commit, fixes, commit. Then stop.

## Hard rules (summary; runbook §0 is binding)

- No `git push`, no `freeze_part`, no `protocol/frozen/`. Subagents never commit.
- Do not change either claim's primary estimand, `GENERIC_COMMITTED`, `INTERMEDIATE_COMMITTED`,
  the task score, splits, held-out families, or any threshold except the Tier-1 κ_m status.
- With defaults, every existing smoke value stays identical. New outputs are new keys.
- Do not add required methods to the `Subject` protocol.
- Development outputs go to `results/<name>_dev_<date>/` with a "non-citable" README.
- Do not edit `archive/`, `docs/archive/`, `latex/`, existing `results/` directories or earlier
  revision reports.
- Every number you report comes from a command you ran; quote it.

## Stop and ask the author when

Any item of runbook §6 occurs. In particular: an existing smoke value changes; an existing hook
statistic turns out to be per-unit on a non-privileged hook; the eigengap flag is set on the smoke
reference; a `declared` κ_m makes a development output citable; the WP5 linear exactness check
fails. Leave accepted WPs committed, leave the blocked WP uncommitted, and write the blocker into
the final report.

## What to hand back

1. The final report path, `docs/REVISION_REPORT_2026-10-09_reference_upgrades.md`, with its WP
   table (status, commit, tests, reviewer verdict).
2. `git log --oneline <baseline>..HEAD`.
3. The numbered list of decisions the author must make (from the report).
4. The paths of the WP7 gate draft and the WP8 note for Junjie.

Begin with the preflight.
