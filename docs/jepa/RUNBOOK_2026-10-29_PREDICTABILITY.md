# RUNBOOK — predictability budget and linear CCA on LUCiD (29 Oct 2026)

_To the JEPA agent on Deepthought (with GitHub push rights). From the author. Branch `JEPA`, at or
after `0caea48`. Commit this file unchanged as `docs/jepa/RUNBOOK_2026-10-29_PREDICTABILITY.md`
before any run._

## 0. Authority, scope, stops

**Authority.** The author, 29 Oct 2026, 14:15 CET (verbatim; quote it in `DECISIONS.md`):
> "我选方案b，写runbook，我交给agent"

Option B was "keep the grid paused; first run the predictability budget and linear CCA on LUCiD,
CPU only, then decide on the grid". This approves exactly the work below.

**Status.**
- Every result is an extension: `registered: false`, extension name `predictability_2026-10-29`.
- Results go outside the registered result directories, and never replace a verdict, a
  prediction status or a frozen decision.
- **The grid stays paused.** Write no grid decision file and launch no grid run.

**No training.** Encoders are only loaded and run forward. L40S use is zero.

**Rules that still hold.** All of these apply unchanged:
- §6.1 hosts and execution path. Code reaches the hosts only through GitHub, one clone per SHA.
- §6.2 resource guard.
- The no-touch packages.
- The hard rule on shared storage: never delete, move or modify anything outside
  `/ceph/dwong/jepa/`. **This includes `/tmp`.**
  - Every command on a host sets `TMPDIR=/ceph/dwong/jepa/tmp/`, and so does pytest
    (`--basetemp` and `-p no:cacheprovider`).
  - This is the 29 Oct `/tmp/pytest-of-dwong` lesson.
- Never infer author approval.

**Budget.**
- CPU: ≤ 12 wall-hours, ≤ 16 cores, `nice -n 10`. This counts against the 48 h generation cap.
- Titan: ≤ 2 GPU-h, used only if the encoder forward passes on CPU would exceed 4 wall-hours.
  Go through the guard: one job, ≤ 30 min each.
- Storage: ≤ 2 GB and ≤ 3 000 new entries under
  `/ceph/dwong/jepa/returns/ext_predictability_20261029/`, which is `<ext>` below.
- Write nothing if free entries are below 30 000.

**Stop and report** (one page, lettered options) only if:
- **S1:** a budget line above would be exceeded.
- **S3:** a step would change a registered prediction, criterion, cell, dataset or frozen decision.
- **S4:** GitHub push fails.
- **S6:** a §2 acceptance test fails after one engineering fix attempt.

Everything else is engineering: write one line in `DECISIONS.md`, then continue.

---

## 1. Why, in one paragraph

Part C and Part D (29 Oct) showed the following:
- Every trained SIGReg-JEPA variant is worse than its random-init floor on vertex resolution
  (0.19–0.35 m against 0.161 m).
- It is also worse on module-gain run-mean R²: about 0 against 0.22. In strong drift it is
  0.10 / 0.22 against 0.84.
- It nearly erases per-PMT log charge: hit-PMT token R² is 0.02–0.05, against 0.998 at random
  init and 0.988 for MAE.
- None of the six pre-declared causes is supported.

**Working hypothesis H.** JEPA keeps what is predictable across views, not what is physical:
- Shared nuisance is predictable, so it is kept. This is the linear theory and the toy.
- On LUCiD, a hit PMT's charge is about 1 PE and dominated by private Poisson and SPE
  fluctuation. It is unpredictable from context, so the charge channel is dropped, and with it
  information the vertex needs.

This runbook measures **how predictable each token feature is** and **whether the encoders kept
the predictable part**. It also checks **whether the linear theory predicts a gain trap on LUCiD
at all**. Every test can refute H.

---

## 2. Part 1 — predictability budget (CPU)

### 2.1 Data

- Use the A10 dev cell with the frozen config and generator config from
  `results/jepa/20261025_pregrid_checks/provenance.json`:
  - `within` split, s = 0.05, photon domain, readout `readout_pmt_r6`.
- Base events: the **first 1 000 events of the dev probe-test cohort**, which is 20 runs of 50.
  Use the same IDs and the same run states.
- Strong drift: the same 1 000 IDs under the `strong_drift` cell (s = 0.30) that Part D1
  generated. Use the same keyed IDs; only the state spread differs.

### 2.2 Resampling

Write `src/jepa_noise/ext_predictability.py`. Import from `lucid_bombs`, `point_encoder` and
`lucid_run`; never reimplement the readout or the tokenisation.

1. **The noise resampler** `resample_noise(draws, geometry, cfg, r)` returns a copy of a
   `SensorDraws`:
   - **Kept:** vertex, yield, `qe_eff`, emission offset and capacity.
   - **Redrawn:** `poisson_u`, `charge_z`, `transit_z`, `dark_counts` and `dark_times`, with the
     **same shapes** as `draw_event`.
   - Keyed namespace: `keyed_rng(seed, "ext_predictability", run_id, event_id, r, role)`.
   - r = 0 must return the original draws unchanged.
2. **Scheme A** (state fixed): for each base event, R = 16 noise realisations (r = 0…15) under its
   run's own `DetectorState`.
3. **Scheme B** (state also redrawn): R = 16 realisations, each with a fresh F1 + F3 state.
   - Same law and s as the cell, keyed under `"ext_predictability_state"`.
   - Draw a new noise realisation for each state as well.
4. **Readout and tokens.** Run every realisation through the cell's own readout
   (`make_readout`). Tokenise with `point_encoder.tokenise` on a temporary run directory under
   `<ext>/tmp/`; delete it after use (it is inside `<ext>`, so deletion is allowed).
   - Use the same scaling and the same `within` mask as the frozen cell.
   - Keep only target (odd) PMT positions.
5. **Per (event, target PMT) statistics.** For each scheme and each token feature
   f ∈ {log charge, time, hit flag}, store the mean and variance over the 16 realisations.
   - Store float32 in one `.npz` per cell and scheme, under `<ext>/budget/`.
   - Never store the raw realisations.

### 2.3 Quantities

Compute these on target tokens exactly as the encoder sees them (unhit → 0).

| Quantity | Definition |
|---|---|
| ρ_total(f) | 1 − mean over (event, PMT) of Var_A(f) / Var over all (event, PMT, r) of f |
| ρ_phys(f) | 1 − mean over (event, PMT) of Var_B(f) / Var over all of f |
| ρ_state(f) | ρ_total − ρ_phys: the shared detector-state part |

- Report all three:
  - for all target tokens;
  - for hit-only tokens, i.e. (event, PMT, r) where the PMT is hit;
  - by yield tercile.
- ρ_total is an **upper bound** on what any context can predict, because a context knows at most
  the event's physics and its run state.

### 2.4 Encoder probes (CPU forward; Titan only under the §0 rule)

**Encoders.** Use the frozen checkpoints from Part C and Part D. Record each path and sha256 in
`DECISIONS.md`.
- `init9000` (random init);
- `fix9000`, `fix9001`;
- `anc100`;
- `mae9000`;
- `ijepa` 9000 and 9001, `stopgrad` 9000 and 9001, `lambda` 9000 and 9001;
- `strong_drift` 9000 and 9001, evaluated on the strong-drift budget events.

`across` encoders are out of scope here, because the budget uses the `within` mask.

**Input.** Run each encoder on the **r = 0** realisation of the 1 000 events, as full input.
Take the read-out tokens at target PMT positions.

**Probes.** Ridge probes from token to target, using `ridge_r2` from `jepa_pregrid_checks.py`:
- Train and test split: 10 runs and 10 runs, grouped by run.
- Targets, per feature f:
  1. **observed f** (r = 0);
  2. **predictable part** E_A[f], the scheme-A mean;
  3. **physics part** E_B[f], the scheme-B mean.
- For log charge and time, fit on hit-only tokens. For the hit flag, fit on all target tokens.

**Output.** `<ext>/probes/<encoder>.json` and one table `budget_probes.md`, with encoders as rows
and (feature × target) as columns, plus ρ_total, ρ_phys and ρ_state per feature.

### 2.5 Acceptance tests (CPU, before any host run)

Add `src/jepa_noise/tests/test_ext_predictability.py`:
1. `resample_noise(r=0)` reproduces the original draws bit-for-bit. For r ≥ 1, vertex, yield and
   `qe_eff` are unchanged and every noise field changes.
2. On a fixture feature that is deterministic given (vertex, state), ρ_total = 1 within 1e-9.
   On a pure-noise feature, ρ_total ≈ 0, within 0.02 at R = 16 and n = 2 000.
3. The scheme-B states follow the cell's law: the SD of ln gain_m is within 5% of the configured
   value.
4. A probe of E_A[f] on tokens that equal E_A[f] gives R² = 1.

The full suite must still pass; record the count. Push, clone the SHA, and run the CPU smoke on
the host with the TMPDIR rule.

---

## 3. Part 2 — linear CCA on LUCiD (CPU)

**Features.** Use module-pooled features, which keep the module-gain signal and keep the
covariances well conditioned:
- Per module and per view, take the mean over that view's PMTs of the three token features.
- `within`: context = even PMTs, target = odd PMTs, each 105 × 3 = 315 dimensions.
- `across-fixed`: a **fixed** module half (the first half of the Hilbert order is the context).
  This is a linear analogue of `across`; the deep `across` redraws the half per event. Label it.

**Data.**
- The dev probe-train plus probe-test cohorts at s = 0.05 (20 000 events, 400 states).
- The same at s = 0.30 (strong drift).
- The Part 1 scheme-B means of the 1 000 events, for the reference.

**Computation.** Use `linear.cca`, `linear.gevp` and `metrics.principal_angles`, imported, not
reimplemented, with the THEORY regulariser ε = 1e-3·trace/d.
1. **CCA** on context versus target, k ∈ {1, 4, 10, 30}.
2. **What each direction encodes.** For each of the top 30 canonical context directions j:
   - R²_gain(j): ridge from the run's 105 ln gain_m to the run-mean projection. Use
     cross-validation grouped by run; this is run-level.
   - R²_phys(j): ridge from (vertex, log yield) with quadratic terms to the projection.
     Event-level, grouped by run.
3. **Displacement D.** The maximum principal angle between the top-4 CCA context directions and
   R_c = the top-4 GEVP of (S_phys, N_cc):
   - S_phys = covariance over events of the scheme-B means;
   - N_cc = the mean within-event scheme-B covariance, which includes state and noise.
   - Report D for both splits at s = 0.05 and s = 0.30.

**Output.** `<ext>/linear/` (JSON plus `linear_cca.md`).

---

## 4. Pre-declared readings (descriptive, not gates)

Report each one as **supported**, **not supported** or **unclear**, with the numbers. Declare any
further tolerance in `DECISIONS.md` before computing.

| ID | Reading | Supported if |
|---|---|---|
| H-a | Charge is shot-noise dominated | Hit-only ρ_total(log charge) ≤ 0.5 at the median-yield tercile, **and** it is the smallest of the three features' hit-only ρ_total |
| H-b | JEPA keeps the predictable part of charge | In both `fix` seeds: R²(token → E_A[log charge]) ≥ 0.5 **and** R²(token → observed log charge) ≤ 0.2 |
| H-b′ | JEPA drops charge entirely | In both `fix` seeds: R²(token → E_A[log charge]) < 0.2. **H-b and H-b′ are mutually exclusive. If H-b′ holds, H is refuted for charge:** the predictable part is lost too, which points to collapse rather than predictability |
| H-c | Retention follows predictability | In both `fix` seeds, the order of the three features by R²(token → E_A[f]) matches their order by ρ_total (all target tokens). MAE and init are reported beside it without a criterion |
| H-d | The linear trap exists on LUCiD | At s = 0.05: in `within`, at least one of the top-4 directions has R²_gain ≥ 0.5; in `across-fixed`, none of the top-4 has R²_gain ≥ 0.2; and D_within − D_across ≥ 20° |
| H-e | Strong drift deepens it | D_within(s = 0.30) > D_within(s = 0.05) |
| H-f | The deep failure is not the linear trap | Evaluable only if H-d is supported. Both `fix` seeds' module-gain run-mean R² is below the random-init floor (Part C values, no new run) |

---

## 5. Part 0 — a small read-only check (do it first; minutes)

The original **WP4a β decision file** was not found at its recorded path (29 Oct report).
- Search read-only under `/ceph/dwong/jepa/` and in the git history of `JEPA`.
  `git log --all -- '*beta*decision*'` and a `find` by name and by the sha256 in the WP4a
  manifests are enough.
- If found, record its path and sha256 in `DECISIONS.md` beside the extension descriptor.
- If not found, write one line saying so. No other action.

---

## 6. Write-up, commit, report, wait

1. Results go in `results/jepa/20261029_predictability/`, with a README marked
   `registered: false`. Include `budget_probes.md`, `linear_cca.md` and the JSON summaries, all
   ≤ 1 MB each. The large `.npz` files stay in `<ext>`, with path and hash recorded.
2. `DECISIONS.md` gets one line per engineering choice. `STATE.md` gets a short "predictability"
   entry. Book CPU wall-hours and any Titan GPU-h in `RESOURCE_LEDGER.json`.
3. Commit and push `JEPA` after Part 1 and again after Part 2.
4. **Report to the author (≤ 1 page, in Chinese):**
   - the ρ table (3 features × total/phys/state, all and hit-only, by tercile);
   - the probe table;
   - the per-direction table for the top 10 CCA directions in both splits, and D;
   - the seven readings;
   - resources used;
   - the final pushed SHA.
5. **Wait for the author.** Do not resume the grid and do not propose a grid change without a
   gate.