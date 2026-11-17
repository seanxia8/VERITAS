# Round 2 diagnostics: event displays, simulation sanity, physics ceiling, latent evolution (17 Nov 2026)

Extension `r2_diagnostics_2026-11-17`, `registered: false`. Parent: `RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md` and its addenda. Round 2 is closed (report `results/round2/R2_RUN_REPORT.md`); nothing here changes a round-2 reading, dataset or checkpoint.

## 0. Authority

**Author, 17 Nov 2026, 01:26**, relaying Junjie Xia's review of the round-2 results:
- Dow: "你有event display吗 看看他simulation有没有问题。 HK怎么差这么多？"
- Junjie (Slack, 16 Nov, 23:42–23:44):
  - "这些是 vertex resolution？那感觉结果很差啊。。SK->HK结果变差是因为什么？更低的photo coverage吗？"
  - "我觉得JEPA的branch fundamentally有些问题，可能是bug？"
  - "我们先从simulation event display看，找几个resolution最差的查原因。然后看latent space training evolution"
  - "用tsne或者umap 跟MAE的同epoch参照比对 然后random init应该是混成一片的"

Part A is authorised now: it uses existing data and checkpoints, CPU only. Part B needs GPU hours outside every existing ledger and runs only after the author authorises a cap (§4).

## 1. Questions

- **Q1.** Is the simulation sane on every detector, and on HK in particular? Facts so far:
  - HK's Gate V yield ratio is 2.07: LUCiD transport detects about twice the light the analytic model predicts. SK is at 1.19 and JUNO at 1.08.
  - HK's module-pattern correlation is 0.756, against 0.83–0.89 elsewhere.
- **Q2.** Why are the absolute vertex errors so large (0.5–3.5 m), and how far are they from what the data allow? No physics ceiling for the photon-bomb vertex has been computed yet.
- **Q3.** Is there a bug in the JEPA branch, or is the fixed-label collapse a property of the objective?
- **Q4.** How do the representations of fixed-label JEPA, position-free JEPA, MAE and random init differ, and how do they evolve during training?

## 2. Part A (CPU only, existing data and checkpoints)

Run on Deepthought (CPU) with at most 16 workers. Write only under `results/round2/20261117_diag/` (repo) and `/ceph/dwong/jepa/returns/r2_diagnostics_2026-11-17/` (scratch). Read round-2 data and returns; never modify them.

### A1. Event displays

- **Layout:**
  - SK and HK: the barrel unrolled in (φ, z), with the two caps as discs above and below.
  - JUNO: a Mollweide projection of the sphere.
  - WCTE (round 1): the same unrolled layout, as a reference.
- **Panels per event:**
  1. log charge per PMT;
  2. first-hit time per PMT;
  3. a top or side view of the tank showing the true vertex and the probe-predicted vertices of MAE, init, fixed-label JEPA and position-free JEPA (seed 9000; full-input probe).
- **Event selection per detector:**
  - the 6 events with the largest MAE vertex error;
  - 3 events near the MAE median;
  - the 6 events with the largest fixed-label vertex error.
  - Take errors, run ids and event ids from each evaluation's `per_event.npz`.
- **Caption per event:** the true vertex, ln Y, number of hit PMTs, total photoelectrons, distance to the nearest wall, and each arm's error.

### A2. Error anatomy

For each detector and arm (seed 9000), plot the binned median vertex error against:
- true radius r;
- |z|;
- distance to the nearest wall;
- ln Y;
- run (the injected gain drift).

Report whether the worst events cluster, for example near a wall, at low yield or in particular runs.

### A3. Simulation sanity

1. **Light versus yield.** Total detected photoelectrons and hit fraction against Y, on log-log axes, per detector. Fit the slope and compare across detectors.
2. **Closure against the analytic model.** Per-PMT mean photoelectrons against distance and incidence angle, binned, compared with the analytic expectation computed from the generator's own code (same parameters).
3. **Timing.** Distribution of t − t₀ − d·n/c for hit PMTs. Its width should match the TTS model used by the generator; check for offsets and tails.
4. **HK yield discrepancy.** For the Gate V holdout events, compare analytic μ and LUCiD-transported μ per PMT. Plot the ratio against distance, incidence angle, z and position on the wall. Find which ingredient gives the factor of about 2: photocathode radius, QE, angular acceptance, attenuation, or the active-PMT selection.
   - List the parameters each detector uses (PMT radius, QE, attenuation length, angular acceptance), from the configs, in one table.
   - Do the same comparison for SK (ratio 1.19) as the reference.

### A4. Representations at init and at the end of training (no training)

- **Event-level maps.** For each detector, take event-mean embeddings of 2,000 test photon-bomb events for init, MAE, fixed-label JEPA and position-free JEPA, using the existing final checkpoints (seed 9000; init = step 0). Draw PCA, t-SNE and UMAP 2-D maps, coloured by true r, z, ln Y and run.
- **Token-level map.** For one event per detector, plot tokens coloured by PMT position and, separately, by the PMT's own charge. This shows directly whether the layout dominates.
- **Similarity.** Linear CKA between arms (event level).
- Fix the seeds for t-SNE and UMAP, and use the same perplexity and neighbours across arms.

### A5. Physics ceiling for the photon-bomb vertex

1. **Renderer.** Write the analytic model as a differentiable PyTorch function of θ = (x, y, z, ln Y, t₀), reusing the generator's code and parameters.
2. **Closure test.** At the true θ, the residuals must be consistent with Poisson fluctuations. If not, stop and report the discrepancy.
3. **Per-event fits.** Fit θ per event by maximum likelihood: Poisson charge, plus hit/no-hit, plus first-hit time. Use 500 test events per detector and several initial values.
4. **Report:**
   - the vertex resolution, as the mean 3-D error in m and divided by R_fid;
   - the median Cramér–Rao bound from the Fisher information at the truth;
   - both against yield.
   - This puts every round-2 vertex number on an absolute scale.
- This is also steps 1–2 of the analysis-by-synthesis plan (Docs: "Analysis-by-synthesis MAE"). Keep the code reusable (`src/jepa_noise/abs_render.py`).

### A6. Cheap bug checks on the JEPA branch (no GPU)

1. Diff the code paths of fixed-label and position-free JEPA. Confirm they differ only in the six zeroed label-pass columns.
2. From the training logs, plot over steps: the prediction loss, the SIGReg term and its context and target parts, and any logged rank or variance metric. Do this for every round-2 run.
3. Recompute SIGReg (Epps–Pulley) on final embeddings of held-out events and compare it with the last logged value.
4. Check that the predictor receives the target positions it is meant to receive, and that the label pass uses `~batch.target` as the hidden mask.
5. Run the existing unit tests of the JEPA objective and the round-1 linear-limit tests (CCA). List which pass.

## 3. Deliverables (Part A)

- `results/round2/20261117_diag/DIAGNOSTICS.md`: one page answering Q1–Q4 as far as Part A can, with figures under `figures/`.
- A short English summary for Junjie (at most 12 lines), at the end of `DIAGNOSTICS.md`.
- Commit to `dev` as in round 2. No stops are expected. If the closure test in A5 fails, or A3 finds a generator bug, record it and continue with the other items.

## 4. Part B (GPU; only after the author authorises a cap)

- **B1. Latent evolution.**
  - Retrain fixed-label JEPA, position-free JEPA and MAE (seed 9000) on WCTE and SK with snapshot checkpoints at steps 0, 50, 100, 200, 500, 1k, 2k, 5k and 10k. Save each snapshot to its own file, not overwriting.
  - Draw A4's maps per snapshot, with the same projection settings, and track f_event, f_layout, effective rank and the vertex probe over steps.
  - Expected cost: about 20–25 A100-h on TOpAS (SK runs about 6 h each; WCTE much less).
- **B2. Known-answer control:** fixed-label JEPA on WCTE with the six geometry columns zeroed in every pass (context, label and full), one seed. If the collapse is the geometry shortcut, f_event should rise above the fixed-label value. About 2–3 A100-h.
- **Ledger:** `R2_diag_gpu`, cap set by the author. Without that authorisation, Part B does not run.

## 5. Launch

From the Mac:

```
cd ~/Documents/VERITAS-worktrees/r2 && claude --permission-mode auto \
  --settings ~/Documents/VERITAS/docs/round2/claude-code/settings.json \
  --append-system-prompt-file ~/Documents/VERITAS/docs/round2/claude-code/rules_common.md \
  --add-dir ~/Documents/VERITAS/docs/round2 ~/Documents/VERITAS_offrepo/claude-code \
  -n veritas-r2-diag \
  "Read docs/round2/RUNBOOK_2026-11-17_R2_DIAGNOSTICS.md (copy it into the worktree's docs/round2 first if it is not there) and execute Part A end to end without stopping to ask; use your own recommended option at every decision point. Part B only if a later addendum authorises a GPU cap."
```
