# STATE — JEPA study (single source of truth)

Updated: 2026-10-17 15:20Z by Lead. Registered at `151f26e`; WP2-v4 at `10693eb`.

## Where we are

- **WP0-v4: complete.** Registration `151f26e`; WP3 artifacts `dab95d8`; hygiene `6399a4d`;
  decisions/status/state `6f41b52`; CPAL dates verified (Recent Spotlight **18 Jan 2027 23:59
  AoE**); prediction status v4 written; pushed (`origin/JEPA` = registration chain).
- **Phase 0 preflight: complete** (`PREFLIGHT_A8.md` + the two host reports, `38c750f`). Both
  hosts reachable (FQDN `*.etp.kit.edu`), `/ceph` mount OK, GitHub fetch of the pushed SHA OK.
  Environments chosen read-only (§6.1 step 1): deepthought `/home/dwong/anaconda3`
  (torch 2.5.1+cu121, sm_50 cubin → Titan sm_52 OK); deepthought2 `/ceph/dwong/transformer/.venv`
  (torch 2.5.1+cu124, sm_89). No new env (quota). LUCiD generation → Deepthought CPUs (no JAX GPU).
- **Quota: G-QUOTA open** (`docs/jepa/GATE_G_QUOTA.md`, `786b6c7`). Shared `/ceph/dwong`: 500 000
  entries, 468 056 used, **31 944 free**, write-stop at 30 000 → 1 944 usable. Blocks only the
  storage volume of generation/E2 returns; E1, code, guard, probes continue. No deletion performed.
- **WP2-v4: committed** (`10693eb`): A4 law (base ρ = 0, A4 θ SD, 13-cell grid, channel primary,
  displacement D), population tables 13×4; D matches §A within 0.051°, γ_n exact, v3 law reproduces
  v3 §A. Fresh review (review-wp2v4) in progress.
- **WP3 E1a: full grid done** (630 rows, 5 seeds, n up to 1e5, 3 workers, ~4.4 min). Preliminary
  aggregate (unreviewed): R1 E1a clause passes (all 6 cells ≤5°); R2 E1a clause fails 2/6 cells
  (saturation of the angle at high γ_load); R3 rows not implemented yet; R3b fails (sample 32°
  vs population 1e-15 rad); R4a 32/42 cells ≤5°; R4b 33/42 at β=10. NOT final: R1/R2 need their
  E1b clauses; no result is reported until the light independent recompute after E1.
- **WP4a bridge**: committed (`9d59762`, fixes `7f98b86`). 26 tests; fresh config review ACCEPT
  WITH FIXES -> all 11 findings fixed (GPU device bug, nonlinear signal, frozen schedule/beta
  cohort, overlay bank 1024, N_actual sha verification). Only the guard-gated 1-step CUDA smoke
  and the beta development cohort remain before the bridge run.
- **Guard: deployed and live on both hosts.** Two fresh reviews (final: all findings resolved),
  137 tests. Dry runs 12:18-12:29Z on both hosts passed (level OK, cephfs_xattr, unattributed
  <20 MiB, window warm, no status JSON, no SHED). Live guards in tmux `jepa_guard` since
  12:29-12:30Z. One <=10-min GPU probe per host ran through `jepa_run.sh` with no bypass:
  Titan 10.55 s (0.0029 GPU-h), L40S 126.07 s cold-cache (0.035 GPU-h); torch 2.5.1+cu121 /
  +cu124, capabilities (5,2) / (8,9), TF32 disabled, determinism 0.0.
- **E1b running locally** (attempt_001, W=2, 65 rows, ~5 h projected, started 12:37Z). E1c done
  (`not_evaluable`, frozen deep modes; dynamics init check requested). E1a final = attempt_007.
  Next after E1b: regenerate PREDICTIONS.md, then the E1 light independent recompute (fresh agent).
- **WP4b: scaffold in progress** (detector-state laws F1/F3/F2/F2'/R11b/T, nominal-twin contract,
  run/ID/storage structure, §3.5.3a Fisher/ceiling numerics), labelled
  `readout_synthetic_unrepaired`. **Two dependency findings**: the pinned LUCiD revision
  `eaf2e89` is not available locally (only `86a1871` in /ceph/dwong/ORACLE/external/LUCiD), and the
  PMT branch is still at synthetic 76-sensor fixture geometry, so the physical WCTE 105x19 map
  required by §3.5.3 does not exist yet. Physical binding waits for the author-named PMT SHA;
  G-PMT by 31 Oct if it has not arrived.

## Running jobs

None on the hosts. Local: impl-guard (code + tests), impl-bridge (code + tests), impl-e1 (E1b
projection + runner fixes), review-wp2v4 (independent recompute). CPU is oversubscribed while
they run; start nothing new that needs >1 core until they finish.

## Queue

- deepthought: guard dry run (10 min) → live guard → ≤10-min probe job (`scripts/jepa_probe_env.py`)
  → 10-batch equivalence comparison vs the Mac.
- deepthought2: same, after deepthought proves the path.
- Then: bridge smoke on a Titan, scaling benchmark, LUCiD generator.

## Guard status at last check

- Not deployed. No `jobs/` dir yet. No HOLD/SHED events. `INFRA_LOG.md` not created.

## Next three actions

1. Guard fixes → delta review → commit → sparse clone on both hosts → 10-min dry run each → live
   guard → probe job each → equivalence comparison.
2. Merge WP2-v4 review verdict; merge WP4a when its tests pass; merge E1b when projected and run.
3. First report to the author (access/registration, preflight, gates, two-week plan), then continue.

## Open gates

- **G-QUOTA** (storage volume; options A–D, A recommended: delete our own failed venvs).
  Blocks: LUCiD generation data volume and E2 returns when the write-stop is reached.
- Awaiting (not gates): PMT SHA (blocks reported LUCiD results); drift ranges from Junjie
  (defaults in DECISIONS.md).

## Budget

- E1 CPU: 0.995 h charged + 20261017_wp3_e1a (~0.06 h at 3 workers); cap 24 h.
- LUCiD/Titan: 0 of 300 GPU-h. LUCiD/L40S: 0 of 80 GPU-h. LUCiD generation CPU: 0 of 48 wall-h.
- GPU-hours used so far: 0.

## Last verified SHA on hosts

- Both hosts: `6f41b52` visible via `git ls-remote` at 11:43 UTC. No clone of the current chain on
  either host yet (only the old skeleton `72add60e` on deepthought).

## Environment / ledger

- Local: `/private/tmp/jepa-venv-cpu311/bin/python` (3.11.11, torch 2.5.1 CPU).
- Hosts: see above; guard and jobs must use `/home/dwong/anaconda3/bin/python3` (3.12.2).
- Ledger: `RESOURCE_LEDGER.json` (+ `v4_caps` block: Titan 300, L40S 80, LUCiD gen 48 CPU-h).
