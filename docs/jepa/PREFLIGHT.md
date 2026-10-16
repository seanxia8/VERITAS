# JEPA v3 preflight

_16 October 2026; partial acceptance, remote environment blocked._

## Registration and isolation

The initial branch was JEPA and HEAD exactly `0cae6b061e488e319643cfb0ba820318b8923578`. No tracked changes existed. The received docs/jepa tree and unrelated docs/LUCID_MODEL_LINEUP_2026-10-09.md were untracked. The unrelated file was not staged. PMT worktree `/Users/wongdowling/.codex/worktrees/junjie-pmt-response/VERITAS` was clean at `ac83c79` on `codex/junjie-pmt-response`.

Full v3 runbook SHA256 is `5f8de710a39e74a6f76235eeaf46c4df0907bfbe5145b1d3aa88e3a9fff1e6cd`, matching both final review dispositions. Both reviews conclude ACCEPT of the specification and no remaining P1. Historical findings remain in their original reports. The launch prompt's current SHA256 is `5165b48af039a63281060503f96091c5e250f09fc4679e2201973e1b608d4340`; the user explicitly superseded historical permission flags and their review hash. Registered flags are PUSH_JEPA=yes, GPU_ALLOWED=yes, OPTIONAL_WPS=WP7. WP8 is not selected.

`f36c179ad5efca3305379ed3143ddd545b02c760` is the first commit after the fixed base, with subject `jepa: register runbook v3`. Push succeeded and `git ls-remote origin refs/heads/JEPA` independently returned that full SHA. Remote executor also verified publication. The package skeleton was published as `72add60e0e92ad4ec344bb309adbfc023b486455`. Author-approved waveform amendment A1 was subsequently registered as `d28739d`; it precedes any affected outcome grid. Only the orchestrator committed.

## Checks and boundaries

[PREFLIGHT_LOCAL.md](PREFLIGHT_LOCAL.md) records CPU Python3.11.11 / torch2.5.1 and required imports. Registered toy output reproduced correlation and angle tolerances. P5 annihilation was correctly recovery-ineligible. The complete local execution inventory and checksums are under `results/jepa/20261016_wp0_local`. These are reproduced prior development evidence, not prospective R1–R10 observations.

[PREFLIGHT_REMOTE.md](PREFLIGHT_REMOTE.md) records GitHub transfer at pinned SHA, resource observations, Condor transfer pattern and the dedicated environment failure. Installation exited2 with Disk quota exceeded. The failed directory remains intact, the log was returned with rsync, and its hash matches the remote stdout digest. No remote import pass or GPU readiness is claimed. No new outcome grids ran.

Protected-source hashes are in PROTECTED_BASELINE.json. Waveform covariance/eligibility and fixed linear configuration now have source-bound independent acceptance. Actual TIDMAD manifest, remaining numerical configurations and whole-campaign projections remain independent gates. No blanket WP0 acceptance is declared while remote imports are blocked.

Authorized source-backed cache cleanup restored file-count quota headroom. The sole600s remote environment retry then timed out (outer exit124) before imports. Both attempts remain preserved; STORAGE_RECOVERY.md and its verified inventories give the terminal evidence. E2 fixed-grid G6 is independently confirmed; A3 revision is pending author instructions.
