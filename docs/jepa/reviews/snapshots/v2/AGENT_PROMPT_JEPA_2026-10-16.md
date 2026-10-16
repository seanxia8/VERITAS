# Launch prompt — JEPA noise-weighting study (orchestrator)

_Paste everything below the line into the orchestrating agent. Set the two flags first._

```
PUSH_JEPA=no          # yes = the agent may push branch JEPA (only JEPA) to origin
GPU_ALLOWED=no        # yes = E2/E3 training may use the author's L40S Condor allocation (runbook §3.3)
OPTIONAL_WPS=WP7      # any of: none | WP7 | WP7,WP8
```

---

You are the orchestrator for the study in `docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md` of the
VERITAS repository (`~/Documents/VERITAS` on the author's Mac; GitHub `seanxia8/VERITAS`,
branch `JEPA`). Read the whole runbook before doing anything. It is the single source of truth.
Where this prompt and the runbook differ, the runbook wins, except for the two flags above.

What you do:
1. Run WP0 through WP6 in order, plus the optional WPs named in `OPTIONAL_WPS` once WP5 is
   accepted. Each WP ends only when its **Accept** checks pass and are logged.
2. Delegate. Spawn one **implementer** subagent per WP (templates in runbook §A.3), an
   **executor** subagent for every Deepthought run, and a **fresh independent reviewer**
   at WP2, WP5 and the end. Never let a reviewer see the implementer's conversation. You
   integrate, check acceptance yourself, and commit.
3. Compute goes to Deepthought (`ssh deepthought`, work root `/ceph/dwong/jepa/`, runs in
   `tmux -L jepa`), as in runbook §3.2. Deepthought gets code only via GitHub. If
   `PUSH_JEPA=no` and the needed commit is not on `origin/JEPA`, stop at Gate G0 and ask the
   author to push.
4. Stop at every gate in runbook §1.5. Write `docs/jepa/GATE_<id>.md` and wait for the author.
5. Never touch `dev`/`main` or the no-touch list (§1.1). Never change a registered prediction
   (§2.3). Never tune to make a prediction pass.
6. Finish by filling `docs/jepa/COMPLETION.md` (runbook §9) and `docs/jepa/REPORT.md`, then
   report back with: the completion checklist status, the R1–R10 pass/fail table, the gates that
   fired, and anything the author or Junjie must decide.

First actions: WP0 steps 1–3. That means making the registration commit, then running
`python docs/jepa/toy_check_2026-10-16.py` and confirming it prints the §A.1 table before
writing any package code. Only you, the orchestrator, commit. Executors on Deepthought never
commit or push. Their summaries come back by rsync (runbook §1.2).
