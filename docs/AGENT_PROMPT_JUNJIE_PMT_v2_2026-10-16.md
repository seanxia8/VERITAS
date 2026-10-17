# Launch prompt — PMT corrections, runbook v2

```
PMT_REMOTE=yes         # C2, author, 17 Oct 2026; within runbook v2 §2 shared-server rules
PMT_GPU=yes            # one Titan job at a time on Deepthought only (§2)
PMT_DATA_CAP_GB=5
```

---

Work in `/Users/wongdowling/.codex/worktrees/junjie-pmt-response/VERITAS`, branch
`codex/junjie-pmt-response` (HEAD `56179eb` or later). Read
`docs/RUNBOOK_JUNJIE_PMT_v2_2026-10-16.md` in full, then v1
(`docs/RUNBOOK_JUNJIE_PMT_2026-10-16.md`) for any work-package definition that v2 does not
override.

1. **First**, commit the v2 runbook and this prompt unchanged:
   `docs: PMT runbook v2 (author revision)`. Record the C1/C2 answers (runbook §2) verbatim in
   `docs/PMT_DECISIONS_LOG.md`.
2. **R-1, R-2, R-3 (the report table) and R-5**, local CPU. Use the v2 §3 recommended defaults as
   development settings, and log each in `docs/PMT_DECISIONS_LOG.md`.
3. **Report** with the R-1 to R-3 acceptance evidence, and push. The author reviews it and names
   the accepted SHA, which the JEPA study then merges (its G-PMT deadline is 31 Oct). Do not wait
   for that: continue with steps 4 and 5. Commits after the named SHA do not change what JEPA
   merges.
4. **Then R-4** (re-run, under the C2 flags and the §2 shared-server rules), and WP6 (freeze the
   experiment contract).
5. **WP8:** draft the decision list for Junjie as a GitHub reply in
   `docs/PMT_ISSUE5_REPLY_DRAFT.md`. Do not post it.
6. **WP7 only after the author says "launch WP7".**

Rules:
- Push only `codex/junjie-pmt-response`. No merge into `dev`/`main`, and no issue comment.
- GPU, remote jobs and data downloads only as the flags and runbook v2 §2 allow.
- Do not touch the JEPA branch or files.
- Only you commit.

Report:
- tests run and their results;
- the new T-Q validation numbers;
- the constant-baseline table;
- the open decisions, as a list.
