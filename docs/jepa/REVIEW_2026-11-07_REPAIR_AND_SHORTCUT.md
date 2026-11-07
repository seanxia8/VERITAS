**Verdict: PASS**

**Final status (frozen SHA `51999d20315bf097a7846a61b8bde25e708a76a7`, round 5): PASS.** All 16
checklist items (Part A (i)–(viii), Part B (i)–(vi), front matter (F-i)/(F-ii)) PASS. The round-1
blockers B1–B9, the round-2 blocker B10, the four round-4 execution-defect fixes, and the round-5
X3-b raw-family particle-truth fix are all fixed and independently re-verified on this SHA. The
round-1/2/3/4 sections below are preserved as the historical record; the round-5 re-review of the
X3-b truth fix, the new validation evidence and the final reviewed object are in “Re-review
(round 5) — X3-b raw-family truth fix” at the end of this file.

The implementation is reviewed at `184cd2f` against the binding runbook
`docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` (sha256 `45e5ff98…119b`, verified equal to the
base-commit copy). Part A's checklist items (i), (v), (vii) and (viii) are sound, and Part B's variant,
manipulation check, training guard, gate, evaluation path and cost projection are sound. But large
parts of Part A cannot execute at all: `ext_repair.py` references five names it never defines
(`_x2c`, `_stack_coefficients`, `_standardisation`, `_X3_ROBUST_VERTEX_M`, `_X3_ROBUST_ENERGY_REL`)
and one function parameter it never accepts (`device`), so X2, the X1 particle half and X3-b fail;
the X0-b reproduction/identity code uses a hardcoded split that contradicts §3.2 for the four
`across` models and builds its identity condition from the test set only; and X3-b's readings
additionally crash on a type error and read the embedding cache under keys the pilot never writes.
Part B's R-1 reads only one of the pilot's two per-model payloads, so R²_eq and ρ_LR — the two
quantities its fixed reading needs — are always `None`, and its response-size S is taken from the
pilot's particle cache. The focused tests (37 + 21 passed) and the dry runs do not exercise the mode
bodies, so none of this is caught by the supplied validation logs. The required full-suite run is not
green: 2 failed, 725 passed, 2 skipped, 3 deselected — one failure is this diff's own
`test_forced_import_classification`, which asserts a whole-process import closure that the full suite
(by importing other extensions' modules and scripts) cannot satisfy (B9), and the other is an
untouched, data/numerics-dependent `test_ext_followup` test (N16). Verdict: **BLOCK**, with 9
blocking findings and 16 nonblocking findings.

## Checklist

| # | Subject | Verdict | Main file:line evidence |
|---|---|---|---|
| A-i | condition factors and the §2.5 context (twin redirect, recomputed states, hash and content checks, failing closed) | **PASS** | `ext_repair.py:519-524` (source/file map), `:571-607` (factors), `:653-697` (scaled store, fail-closed), `:681-697` (both bindings restored in `finally`), `:492-513` (RunSpec identical to `lucid_run.py:346-355`), `:1354-1390` (states/twin/events), `:1402-1484` (twin and SWtw content), `:1487-1535` (μ_p), tests `test_ext_repair.py:261-399` |
| A-ii | the X0-b reproduction and the identity check | **BLOCK** | reproduction table correct (`:214-226`, `:1633-1697`) but identity condition built from the test set only (`:1720-1736`) and split hardcoded `"within"` (`:1621`, `:1730/1734`); see B2, B3 |
| A-iii | probes, metrics, pairing and bootstrap | **BLOCK** | `stream_condition_particles` uses undefined `device` (`:918-937`, error at `:928`) so the six §5 particle models fail in `_x1_particle` (`:1901-1919`, called `:1882-1885`); bootstrap/pairing itself correct (`:1789-1797`, `:1052-1109`); X1-g secondary split not implemented (`:1976-1999`); see B1, N1 |
| A-iv | the X2 fits (run-disjoint train and test; no probe-test truth in any fit or hyperparameter choice) | **BLOCK** | fit code is run-disjoint (`:2291-2321` raises on overlap) and CV uses probe-train truth only (`:2416-2510`), but `_x2_model` references undefined `_stack_coefficients`/`_x2c` (`:2812`, `:2816`) and `_standardisation` (`:2447`, `:2497`); X2 cannot produce readings; see B1 |
| A-v | the X3-a Fisher accounting, P0 and the C-var term | **PASS** | `:2991-3044` (inputs/μ Jacobian), `:3024-3079` (C-mean/C-var/occupancy blocks), `:3102-3151` (CRLB ∞ rule, P0), `:3179-3254` (reporting/σ90); tests `test_ext_repair.py:661-722` |
| A-vi | LEACE (reduced rank, permutation rule, random control) and the invariant transforms | **BLOCK** | LEACE core is right (`:3497-3604`) and the invariant token transform is right (`:701-767`, test `:400-432`), but the fixed readings crash (`:3986-3993` vs `:4033-4034`, `:4089-4090`), the X1 cache keys do not match (`:1264-1267`/`:1879` vs `:3878-3880`, `:3915`), and several §7.2 protocol items are missing (`:3809-3810`, `:3837-3858`, `:3738-3765`, `:3727`); see B4, B5, B6 |
| A-vii | the gate, the import closure and the forbidden call paths | **PASS** | gate `:353-455` reads record sha/verdict and both `git diff`/`ls-files` over files **and** directories (`:391-416`); pre-acceptance pending/`SystemExit(2)` (`:426-443`); library entry points re-assert the closure at their end (`:1597`, `:1695`, `:1897`, `:2749`, `:3252`, `:3823`, `:4538`); no forbidden call in `ext_repair` (only §2.4 helpers); sentinel test `test_ext_repair.py:792-842`; see N5, N6 |
| A-viii | the cost projection against the caps | **PASS** | `:1271-1299` GPU (core 2.4747 h, with optional 3.0662 h ≤ 4.0), `:1302-1305` CPU 5.70 ≤ 6, `:1308-1322` S1, `:252-264`; see N7 |
| B-i | design: R-2 changes only the label-pass input (§5.1), and the mechanism argument matches the code | **PASS** | `ext_shortcut.py:536-568` is `lucid_arms.py:405-420` with `posfree_features` in the label pass only; test `test_ext_shortcut.py:102-150`; see N8 for the runbook's target-set/Hilbert wording |
| B-ii | design: the R-1 families and reading are fixed and cannot be tuned after the pilot outcomes | **BLOCK** | families/reading are fixed (`:1150-1186`, `:1220-1236`, `:1376-1428`), but `r1_payload` reads only one of x1.json/x2.json (`:1612`), so r2_eq/ρ_LR are always `None` and the fixed reading is never computable; S is taken from the pilot's particle cache (`:1544-1552`, `:1679-1689`); see B7, B8 |
| B-iii | the variant installation, the manipulation check and the training guard | **PASS** | `:571-609` (build-arm patch restored in `finally`), `:678-784` (manipulation checks), `:493-532` (guard), `:891-916` (fresh run dir/no resume); tests `test_ext_shortcut.py:173-215`; see N9 |
| B-iv | the reproduction, the evaluation path and the readings | **PASS** | `:993-1006` (`ext_h12.reproduction`, targets `ext_h12.py:51-52`), `:1010-1043` (full/half + label-pass 64 events), `:1113-1144` (R2-a/b/c), `:1052-1100` (recovery rule); tests `test_ext_shortcut.py:333-379`; see N10 |
| B-v | the gate, the import closure, the forbidden paths and the write scope | **PASS** | gate `:380-490` (incl. §0 author line, `:391-406`), closure `:336-377`, forbidden-path tests `test_ext_shortcut.py:384-449`, write scope `<ext>`/results/TMPDIR only (`:127-139`, CLI `:239-548`); see N11, N12 |
| B-vi | the cost projection against the cap and the S1 rule | **PASS** | `:1725-1733` 1.34 h ≤ 1.5, `:1693-1722` S1, `:1814-1838` per-job caps; S1 called before GPU jobs in CLI (`jepa_ext_shortcut.py:307,350,399,465`) |
| F-i | the two parts share no mutable state, and neither writes into the other's `<ext>` or results directory | **PASS** | no Part B reference in Part A (`grep`); Part B reads the pilot only (`ext_shortcut.py:125-127`, `:1555-1658`) and writes only `ext_shortcut_20261107`/`20261107_shortcut` |
| F-ii | the combined budgets and S1 rule of §F3 are enforced in both CLIs | **PASS** | Part A `:1308-1322` + preflight stops (`:1592-1595`); Part B `:1693-1722` + `assert_gpu_budget` with the CLI calls; see N7 |

## Blocking findings

### B1 — Part A: five undefined names and one undefined parameter make X2, the X1 particle half and X3-b unrunnable

`src/jepa_noise/ext_repair.py` references names it never defines. Verified on the read-only host
clone at `184cd2f` (no GPU, `nice -n 10`):

```
$ python -m pyflakes src/jepa_noise/ext_repair.py
ext_repair.py:928:40: undefined name 'device'
ext_repair.py:2447:25: undefined name '_standardisation'
ext_repair.py:2497:21: undefined name '_standardisation'
ext_repair.py:2812:30: undefined name '_stack_coefficients'
ext_repair.py:2816:19: undefined name '_x2c'
ext_repair.py:4089:43: undefined name '_X3_ROBUST_VERTEX_M'
ext_repair.py:4090:53: undefined name '_X3_ROBUST_ENERGY_REL'
```

`import jepa_noise.ext_repair as rp; hasattr(rp, "_x2c")` is `False` (likewise the other four), and
`grep -n "_x2c\|_stack_coefficients\|_standardisation"` finds only the call sites, never a `def`.
Consequences, all reachable from the shipped CLI:

- **X2**: `x2()` calls `_x2_model` (`ext_repair.py:2737`), which raises `NameError` at `:2812`
  (occupancy `A_hat`) after streaming the model's embeddings. `x2()` catches it per model
  (`:2741-2746`) and marks every model `unavailable`, so `x2.json` has no readings and
  `mode_x2` exits 2 (`scripts/jepa_ext_repair.py:239-253`). X2-d's latent stream additionally needs
  `_standardisation` (`:2447`, `:2497`).
- **X1 particle half**: `stream_condition_particles` (`:918-924`) has no `device` parameter, but its
  producer uses `device` (`:928`). `_x1_particle` (`:1901-1919`) is called for the six §5 particle
  models (`:1882-1885`); the `NameError` is caught at `:1889-1892`, so those six models are dropped
  from X1 entirely (their photon embeddings are cached but unused), and the electron-energy readouts
  of §5/§3.2 never exist.
- **X3-b**: `_x3_robustness` (`:4069-4091`) raises `NameError` on the two missing constants when a
  variant supplies both vertex and energy readings.

Demonstration of the two X3 failures (same host clone):

```
>>> rp._ratio_and_ci(0.10, 0.12)
TypeError: object of type 'float' has no len()
>>> rp._x3_robustness({"vertex": {"N": {"mean_error_m": 0.10}, "D": {"mean_error_m": 0.12}},
...                    "energy": {"N": {"resolution": 0.10}, "D": {"resolution": 0.12}}}, {})
NameError: name '_X3_ROBUST_VERTEX_M' is not defined
```

The supplied focused tests (37 passed) and the dry runs (which call `_mode_setup_objects`, not the
mode bodies) never reach these paths, which is why the validation logs look green.

### B2 — Part A: the probe tokenisation hardcodes `split="within"`, contradicting §3.2 for the four `across` models

§3.2 pins the photon embeddings to `ext_followup.stream_event_rows` with "split and mask_seed from
the row and the config". `ext_repair.py` hardcodes `split="within"` at every streaming call:
`:1621` (X0-b), `:1730/1734` (identity check), `:1878` (X1 photon), `:2734` (X2), `:2899` (X2-e),
`:2970` (unused helper), `:3842/3846/3850/3854` (X3-b init9000). For example:

```
1621:            model, dataset, probes["geometry"], condition["condition"], split="within",
```

The inventory rows carry `split`, and `ext_followup`'s own A-2 evaluation uses
`row.get("split", "within")` (`ext_followup.py:1598,1604`). The committed inventory has
`results/jepa/20261103_followup/inventory.json` → `mae_across_trigger9000.split == "across"` (likewise
`mae_across_trigger9001`, `t1_s030_across_trigger9000/9001`). The X0-b reproduction targets are the
A-2 values measured with that split (`ext_followup.py:2739-2743`, runbook Appendix F), so the X0-b
gate cannot reproduce the `mae_across_trigger9000`/`9001` and `t1_s030_across_trigger9000`/`9001`
targets (0.120958 / 0.119626 / 0.125605 / 0.119031 m and their D columns) with a `within`
tokenisation; and every X1/X2/X3-b embedding of those four models is computed under the wrong view
convention. This is a blocking defect for A-ii and infects A-iii/iv/vi.

### B3 — Part A: the identity check builds its condition from the test set only and applies it to the train set

`identity_check` builds `build_condition("N", probes005["test"], geometry)` and
`build_condition("D", probes030["test"], geometry)` (`ext_repair.py:1720-1721`), so each condition's
`run_id_by_dir` and directory cover the **test** set only, but the function then loops over **both**
train and test roles (`:1726-1736`). For a train run directory the wrapper's set check fails and
`_transformed_store` raises `RuntimeError: hit_transform_context: … is outside the N set …`
(`:660-668`; the out-of-set check at `:662-664` fires first). The exception propagates out
of `identity_check` (no handler) and out of `reproduction`, so `reproduce` (X0-b) can never write
`reproduction.json` after doing its GPU work. `identity_check` is part of X0-b's pass
(`:1690-1694`), so this alone blocks the gate.

### B4 — Part A: X3-b's readings crash and use a degenerate bootstrap

- `_variant_reading` calls `_ratio_and_ci(left["resolution"], right["resolution"])`
  (`:4033-4034`), i.e. two floats, while `_ratio_and_ci` (`:3986-3993`) treats them as per-event
  arrays (`np.abs(left)`, `len(left)`, `paired_cluster_bootstrap`). The result is
  `TypeError: object of type 'float' has no len()`, raised from `x3_readings` (`:3821`) — `x3` mode
  dies with a traceback, so `x3.json` is never written.
- Both X3-1 and X3-4 bootstraps pass `np.zeros(n, dtype=object)` as run ids (`:3991`, `:4112-4114`),
  i.e. one cluster; every resample is the same cluster and the "CI" collapses to the observed
  difference — not the §7.2 run-cluster CI. X3-4 has no combined "calibration dominates / invariance
  dominates / mixed" verdict at all (`:4094-4100`).
- `_gain_specific` (`:4121-4145`) implements only the first §7.2 condition (LEACE ρ_E above the 95th
  percentile of the 20 control ratios); the required run-cluster CI of
  `res_LEACE − mean(res_control)` excluding 0 is absent.
- `_x3_robustness` uses the undefined constants (B1); the §7.2 thresholds (2 mm, 5 % relative) are
  not defined anywhere.

### B5 — Part A: X3-b reads the X1 embedding cache under keys the pilot never writes

X1/X2 write cache keys `<role>__<condition>` via `embedding_cache_key` (`:1264-1267`), e.g.
`train__N`, `test__D`, `test__SWunit`. X3-b fetches different keys (`:3878-3880`, `:3915`):
`N__train`, `D__train`, `N__test`, `D__test`, `SWunit__test`. Demonstrated on the host clone:

```
wrote: ['train__N']
has train__N: True   has N__train: False
```

Consequently `_x3_leace` (`:3861-3926`) puts all ten planned trained-encoder models into
`unavailable` ("X1 embeddings are not cached"), so no LEACE variant exists, and the calibrated
variant for each is never attached (`:3915-3921`).

### B6 — Part A: several fixed X3-b protocol items are not implemented

- **Raw calibrated variant**: §7.2 requires "raw-std features on SWunit D charges". The code computes
  it from plain D charges (`:3809-3810`) and `raw_std_features` ignores its `condition` argument
  (`:3400-3410`: neither `ef.raw_event_features` nor `raw_photon_features` receives it). So the
  "calibrated" raw variant is numerically the standard D variant, and X3-4 is meaningless for it.
- **init9000**: §7.2 asks for invariant tokens on "photon N (train, test) and D (test), and particle
  N (train, test) and D (test)" plus a calibrated variant equal to X1's SWunit reading. `_x3_init9000`
  (`:3827-3858`) streams photon rows only and returns no `calibrated` block, so the init9000
  electron-energy reading and its X3-4 comparison do not exist.
- **Erasure sanity**: `leace_sanity` (`:3738-3765`) computes the unerased/erased R² but no code
  enforces the protocol ("unerased ≤ 0.02 → not applicable; erased > 0.02 → unavailable, erasure
  incomplete"): `_leace_variant` (`:3701-3735`) reports variant readings regardless of
  `sanity["applicable"]`/`sanity["complete"]`.
- **Random-control key**: §7.2 keys the 20 draws as `keyed_rng(20261106, "paper3/x3/leace_rand",
  f"{label}/{domain}/{concept}/{i}")`; the code keys one stream without the index and draws 20 bases
  from it (`:3727-3730`). Reproducible, but not the pinned stream.
- **X2-a additivity check**: `x2_decomposition` computes the SWtw probe cost from the **S** rows
  (`:2363`), not from SWtw rows, so the §6 "v_D − v_N against (v_S − v_N) + (v_SWtw − v_N)" check is
  wrong by construction.
- **Occupancy design**: §6 X2-b defines `n_e,m` from the **N** hits, X2-c from the D hits; the code
  builds one occupancy from the D store (`:2708-2709`) and uses it for both X2-b (`:2780-2783`) and
  the latent designs.

### B7 — Part B: R-1 reads only one of the pilot's two per-model payloads, so its fixed reading is never computable

`r1_payload` selects a single block per model:

```
1612:        block = find_model_block(x1, label) or find_model_block(x2, label)
1617:            for name, candidates in R1_FIELDS.items():
1618:                row[name] = _get(block, *candidates)
```

Part A writes `r2_eq`, `class` and `rho_lr` only into x2.json (`ext_repair.py:1940-1946,2861`), and
only X1 fields into x1.json (`:1922-1937`). Because the x1 block always exists when the pilot has
completed X1, the R²_eq, X2-b class and ρ_LR columns are always `None`. Reproduced with the actual
functions (a fake gate root, `dry_run=True`, and Part-A-shaped payloads):

```
R-1 row: {'r2_eq': None, 'x2_class': None, 'rho_lr': None, 'rho_swtw': 0.1, 'share_cs': 0.5, 'delta_d': 0.03}
reading verdict: mixed | comparisons: [None, None]
```

The R-1 reading ("compare the generative median with each frozen_target seed on R²_eq and ρ_LR",
§3) therefore degenerates to `mixed (incomplete)` for every real pilot output. This is a blocking
defect for B-ii, and the same reader cannot be judged as "genuinely reading the fields R-1 requires".

### B8 — Part B: R-1's response size S is taken from the pilot's particle cache, not the photon cache

`cache_files` classifies any `*.npz` whose name contains `__N`/`__D` (`:1430-1437`, `:1544-1552`);
the pilot's particle caches are named `particle__test__N.npz`, and `_first_verified` (`:1679-1689`)
picks the first row of the sorted list. Demonstrated:

```
N candidates: ['particle__test__N.npz', 'test__N.npz']
_first_verified picks: particle__test__N.npz
embeddings shape used for S: (5, 16)
```

For the six particle models (`X1_PARTICLE_MODELS`) S would be computed from particle-event
embeddings, contrary to §3 ("the pilot's cached N and D **photon** probe-test embeddings"), and the
"invariant" taxonomy row (`S < 0.01 and Δ_D not eligible`) would use a different quantity. This is
blocking for B-ii's fixed table.

### B9 — The required full `src/jepa_noise/tests` suite is not green: a new Part B test fails in the full-suite context

The detached run (`run_final.sh`, Deepthought, no GPU) produced:

```
2 failed, 725 passed, 2 skipped, 3 deselected in 1172.28s (0:19:32)
FAILED src/jepa_noise/tests/test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows
FAILED src/jepa_noise/tests/test_ext_shortcut.py::test_forced_import_classification
```

`test_forced_import_classification` (`test_ext_shortcut.py:565-586`) calls the whole-process
`xs.import_closure(ROOT)` and asserts that every violation is under `src/jepa_noise/tests/`; in the
full suite, other extensions' tests have imported 47 repository modules outside Part B's `<P>`
(e.g. `scripts/jepa_cpu_launch.py`), so the assertion fails:

```
E       AssertionError: assert [{'module': '...get.py'}, ...] == []
E         Left contains 47 more items, first extra item:
E         {'module': 'scripts.jepa_cpu_launch', 'path': 'scripts/jepa_cpu_launch.py'}
```

This is a defect in the new test (the assertion is only valid in a one-file focused run) rather than
in the modes' closure — the two dry runs both report `closure pass: true`, `violations: []` — but it
means the §9.3/§F4 requirement ("the focused tests plus the full `src/jepa_noise/tests` suite run
once on the final SHA, deselecting only the three known `test_guard` launcher tests") is not met at
the reviewed SHA, so the supplied validation evidence cannot support acceptance until the test is
made suite-safe (e.g. assert only on the modules the setup actually imported).

### Additional Part B cross-contract defect (nonblocking; recorded under B-ii/B-iv)

`pilot_status` looks for `section.get("complete")` in readings.json (`ext_shortcut.py:1576-1582`),
but Part A's report writes `"completed"` (`ext_repair.py:4509-4514`). R-1 therefore runs even if the
pilot's X1/X2 sections are incomplete, where §3 requires `not_run (pilot outputs missing)`. (Missing
files are still handled as `not_run`, so this is an incompleteness-detection gap, not a crash.)

## Nonblocking findings

- **N1 — X1-g secondary reading missing.** §5's secondary (descriptive) unit-gain split
  `v_D(refit) − v_HW(refit) = [v_D − v_SWunit](refit) + [v_SWunit − v_HW](refit)`, beside
  `v_N − v_HW(refit)`, is not implemented: `_x1_model` does compute the HW refit readout
  (`PHOTON_REFIT_VERTEX` includes `HW`, `ext_repair.py:1763`, `:1962-1966`) but no comparison uses
  it — only D/SWtw/SWunit refit comparisons are built (`:1976-1999`). The primary N-referenced X1-g
  split is correct.
- **N2 — X2-a SWtw rows** (see B6) and the `d_test`/`aligned_d` row-order assumption
  (`ext_repair.py:2350-2353` uses unaligned `d_test["events"]` for Δz_ni/Δz_tot although
  `aligned_d` is computed) — latent only if the two probe sets' row orders differ, but it is a
  correctness risk.
- **N3 — Over-strict X0-a stops; HW-unavailable not propagated.** A
  `twin_content`/`swtw_content`/`mu_reconstruction` failure is turned into a global preflight STOP
  (`ext_repair.py:1576-1578`), whereas §4/§2.2 scope those to specific readings ("the other items
  continue"; a μ_p failure makes the X2-d source stream `unavailable`, not the whole preflight).
  Conversely §4's "if `twin_hits.npz` is missing … the HW condition is `unavailable` for that set" is
  recorded only as a global `hw_condition` (`:1579-1582`) and never consulted by X1, which would
  instead fail the whole model on the missing twin. Not triggered by the pinned data (twins present).
- **N4 — `gpu_projection`/`cpu_projection`/optional-drop are not STOPs.** Preflight appends only
  `entries` and `s1` failures to `stops` (`ext_repair.py:1592-1596`), so a projection above the
  Part A cap would not halt anything, and `with_optional_pass` (`:1296`) is never used to drop the
  optional models as §8.1 requires. Currently unreachable (projection 2.47/3.07 h), hence
  nonblocking. Part A also has no per-job hour accounting at launch (unlike Part B's
  `record_job`/`assert_gpu_budget`), so §8.1's "stop before a job that would cross the cap" is not
  mechanically enforced; F-ii is passed on the preflight S1 check plus the recorded cap flags.
- **N5 — The Part A dry run does not execute the actual mode bodies.** It calls
  `_mode_setup_objects` for every mode (`scripts/jepa_ext_repair.py:119-129`), but no CLI mode calls
  `_mode_setup_objects`/`mode_setup` (`mode_setup` has no caller), so the closure is asserted over a
  mirror of the setup, not the real paths. This is why B1/B2/B3 are invisible to the green dry run.
  The executed-import closure itself is complete (module imports are all at module level).
- **N6 — Read-only `ext_followup` helpers beyond §2.4's list.** `data_status`, `hardware_and_software`,
  `entries_snapshot`, `entries_projection`, `_git`, `section_zero`, `EXT_ROOT`, `PINNED_DATA` are
  imported from `ext_followup` but not enumerated in §2.4 ("`ext_followup`, called: …"). All are
  read-only; §4 nevertheless requires a hardware/software and entries record, so this is a
  documentation gap, not a capability leak.
- **N7 — GPU projection under-counts X1 by six sets per model.** `X1_PHOTON_SETS_PER_MODEL = 16`
  (`ext_repair.py:273`) follows §5's list, but `x1()` streams all 11 condition names for **both**
  train and test roles (`:1871-1880`), i.e. 22 sets per model (extra `SWmod`, `SWunitN` and four
  `NOISY` train sets). Actual projection is about 2.84 h core / 3.46 h with optional (still ≤ 4.0).
- **N8 — Runbook mechanism wording vs the code (Part B).** §5.1 says the target set changes per
  event; the training cell is `within`, whose mask is a fixed parity split
  (`lucid_views.py:205-219`: even `pmt_id_in_mpmt` context, odd target, event-id only recorded in the
  key), while the Hilbert ordering used for windows is applied inside the encoder regardless of split
  (`point_encoder.py:356`). The variant itself (columns 0–5 zeroed in the label pass only; everything
  else untouched) matches §5.1 exactly; the mismatch is in the runbook's mechanism paragraph, so it
  is recorded, not blocked.
- **N9 — Part B per-run wall cap is reported, not enforced.** `RUN_WALL_CAP_H = 1.0` is only checked
  after the run (`ext_shortcut.py:982-985`, `scripts/jepa_ext_shortcut.py:423-427`); a run over 1 h
  is flagged (`wall_cap_exceeded`) rather than stopped.
- **N10 — `reproduce`/`evaluate` do not re-run the data-hash preflight.** `mode_reproduce` uses
  `eh.load_probes` and `mode_evaluate` uses `xs.open_datasets` without `eh.preflight_data`
  (`scripts/jepa_ext_shortcut.py:304-339`, `:462-498`); §5.3's order (preflight first) covers this in
  practice, and `ensure_dataset` still re-verifies run states.
- **N11 — R-1 "generative median over the eight models".** `r1_reading` includes the optional
  `mae_within_median` rows in the generative median when they ran (`ext_shortcut.py:1384-1391` with
  `R1_ALL_LABELS`), while §3 says "the median over the eight generative models". The optional rows
  are selected by the cap before outcomes, so this is not outcome-tuning, but it is a deviation from
  the fixed wording.
- **N12 — R2-a fallback to the label-pass `f_event`.** `r2_readings` falls back to
  `label_fractions["f_event"]` if the full-input layout is missing (`ext_shortcut.py:1117-1120`);
  §4's class must use the full-input `f_event`. The fallback is currently unreachable because
  `ext_h12.evaluate_run` always emits the full-input layout.
- **N13 — Part B dry run does not exercise the reproduce/evaluate/train setup**. It exercises
  preflight, cell, variant install, manipulation and R-1 status (`scripts/jepa_ext_shortcut.py:162-210`)
  but not `eh.load_probes`, `eh.evaluate_run`, `prepare_run_dir` or `run_one`. All imports are
  module-level, so the closure report is unaffected.
- **N14 — Fisher ∞-count granularity.** §7.1 asks for "the number of events with CRLB = ∞ … per
  model, case and parameter"; `fisher_accounting` reports only a total per case
  (`ext_repair.py:3199-3221`).
- **N15 — README key typo.** `_report_readme` prints `row.get("theory_lnY")` while the rows carry
  `theory_ratio_lnY` (`ext_repair.py:4152-4160` vs `:4580-4582`), so X3-5's theory values render as
  "—".
- **N16 — Second full-suite failure is an untouched data/numerics-dependent test.**
  `test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows` (unchanged by
  the diff, runs before any new test file) fails on Deepthought with
  `('y': {'checked': True, 'equal': False, 'shapes': [[64000, 128], [64000, 128]]})` — i.e. the
  recomputed fix9000 label rows are not bit-equal to the committed cache. The diff modifies no file
  that test imports (`git diff --stat 062a8d1..184cd2f` = six new files only), so this cannot be
  attributed to the implementation; it is most likely host/BLAS-dependent (the 3 Nov precedent
  records the same test selection as green on deepthought2). Recorded as the observed state — the
  suite is not green — and not rounded into a pass; it is not one of the three authorised
  deselections.
- **N17 — Report completeness flags overstate X2 (and X3).** `assemble_report` marks a section
  `completed` from `payload["readings"] is not None` (`ext_repair.py:4511-4514`), which is true even
  when every model was `unavailable` (as the X2 mode currently produces, B1). The per-mode `x1`
  flag uses the stricter `complete` key (`:4509`), so the two sections are inconsistent. Same for the
  report's overall `complete` (`:4535-4536`).
- **N18 — One test's cut-specific assertions are vacuous.** `test_c_var_mu_cut_applies_only_to_the_pmts_it_names`
  ends with `np.allclose(cut[4:, 4:] * 0.0, np.zeros_like(cut[4:, 4:] * 0.0))` and a shape assertion
  that cannot fail (`test_ext_repair.py:699-705`); the meaningful comparison is the preceding
  manual-formula check. No silent pass is at stake (the earlier assertion is real), but the test does
  not actually demonstrate the "below the cut contributes nothing" claim it names.

## Declared deviations — judgement

1. **Part B's extra forced transitive imports (`ext_arch.py`, `ext_dejitter.py`).** Part B's `<P>`
   contains `ext_followup.py`, which imports `ext_arch`/`ext_dejitter` at module level
   (`ext_followup.py:44-45`); neither is in Part B's §9.2 `<P>`, and neither is in
   `ext_followup.FORCED_IMPORT_EXCEPTIONS` (`ext_followup.py:129`). Read literally, §9.2's closure
   rule makes every Part B mode a closure STOP, i.e. the runbook is internally inconsistent. The
   implementation records the two paths in a separate
   `PART_B_FORCED_TRANSITIVE_IMPORTS` list (`ext_shortcut.py:171-174`) and excludes them from
   violations (`:317-321`, `:364-369`), exactly the precedent of `ext_followup`'s own exception set.
   The diff modifies no file outside the six new ones (`git diff --stat 062a8d1..184cd2f`), so the
   weakening is unexercised. **Judgement: legitimate, nonblocking**, recorded as a runbook
   inconsistency rather than an implementation BLOCK. A stricter reading would require the runbook to
   be amended (which is outside this review's scope).
2. **`geometry.module_of_pmt` vs `DetectorGeometry.module_index`.** Equivalent: `module_of_pmt` is a
   property returning `module_index` (`lucid_bombs.py:1125-1126`), and the helper falls back to
   `module_index` when the attribute is absent (`ext_repair.py:486-489`). Not a deviation.
3. **Part A's gate has no hard-coded runbook sha pre-acceptance.** Part B hard-codes
   `RUNBOOK_SHA256` (`ext_shortcut.py:131-135`); Part A checks only the record's sha
   (`ext_repair.py:360-372`). Post-acceptance behaviour is correct: (a) requires
   `computed == record.runbook_sha256`, and pre-acceptance without a record is `gate_pending_acceptance`
   for dry-run and `SystemExit(2)` otherwise (`:426-443`), exactly §9.2/§F5. Not a deviation.
4. **Part B's R-1 JSON-schema reader.** The runbook does not pin the pilot JSON schema. The reader's
   field candidates (`ext_shortcut.py:1171-1186`) do include the keys Part A actually writes
   (`delta_d_m`, `delta_d_eligible`, `rho_swtw`, `rho_swunit`, `share_cs`, `r2_eq`, `class`,
   `rho_lr`), and the families/reading are fixed. But because it only consults one payload per model
   (B7), it does **not** genuinely read R²_eq/ρ_LR; that is a BLOCK, not a schema ambiguity. The
   schema-tolerance itself cannot be tuned after outcomes (no thresholds in the reader).

## Reviewed object

- Implementation commit (reviewed object): `184cd2f67618c87851c08722bd650fcd96c0480d`
  ("jepa: implement repair-pilot (Part A) and shortcut-followup (Part B) extensions"); base
  (runbook) commit `062a8d1`. `git diff --stat 062a8d1..184cd2f` = 6 files, 9267 insertions, no
  existing file modified.
- File sha256 (`shasum -a 256` in `/private/tmp/veritas-jepa-20261107`):
  - `src/jepa_noise/ext_repair.py` `8eb9619c925c8740317c65f7b9dcaf77580509cf0f51dd6944ec0e3ecd55d861`
  - `scripts/jepa_ext_repair.py` `dd9906e336abb056969fbc8246b8e1ee0fa048563b35aa176a8ae96c800073db`
  - `src/jepa_noise/tests/test_ext_repair.py` `a53dafe2f7cf0ca146a8119eb747c9fbbea5b007da688f223c5a5857c442f392`
  - `src/jepa_noise/ext_shortcut.py` `4d4cc858cf094428c75cdddd1aaac0b54ba95511f37ec34277c3f588b9644c0d`
  - `scripts/jepa_ext_shortcut.py` `616797f301ac7aa960877085a61eecb581acbd69f05d2e819d79e08d62feeeaf`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `32257ad4e2e7a4820caf4dd709a10d9858b32e90a6995ff9e230f0283af6ee92`
- Runbook sha256: `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b` (matches the
  required value; identical at `062a8d1` and `184cd2f`).
- Full diff sha256 (`git diff 062a8d1..184cd2f | sha256sum`):
  `fe9abde750f738b0dbf8d6a00a210837fc8fe4b04ec323f885481b9ba68bfc0f`.
- Host clone `/ceph/dwong/jepa/code/jepa_work_20261107` at `184cd2f`, clean; all read-only checks
  ran there with `TMPDIR=/ceph/dwong/jepa/tmp/jepa_rb_20261107/review`, `PYTHONPATH=src`,
  `nice -n 10`, no GPU.

## Evidence (detached validation on Deepthought, `/ceph/dwong/jepa/tmp/jepa_rb_20261107/final/`)

`summary.txt` (complete, ends with `DONE`):

```
SHA=184cd2f67618c87851c08722bd650fcd96c0480d
focused_a_exit=0
focused_b_exit=0
full_suite_exit=1
dryrun_a_exit=0
dryrun_b_exit=0
DONE
```

- `focused_a.log`: `37 passed in 138.77s` (exit 0).
- `focused_b.log`: `21 passed in 19.48s` (exit 0).
- `full_suite.log`: `2 failed, 725 passed, 2 skipped, 3 deselected in 1172.28s (0:19:32)`, exit 1.
  Failures: `test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows`
  (untouched by the diff; N16) and `test_ext_shortcut.py::test_forced_import_classification`
  (new file; B9). The single deselection set is exactly the runbook's three known `test_guard`
  launcher tests (`--deselect` ×3 in `run_final.sh`, matching `test_guard.py:1457,1508,1526`); no
  other test is deselected. The 2 skips are pre-existing in-test `pytest.skip`/`skipif` sites
  (`test_ext_dejitter.py:31,113,297`, `test_ext_arch.py:153`; the 3 Nov record at `59c0a0d` also has
  "2 skipped"), not deselections.
- `dryrun_a.log` / `repair_dryrun/dry_run.json`: exit 0; `gate_pending_acceptance`; closure
  `pass: true`, `violations: []`; forced imports only `src/noise_module/*` and
  `src/reconstruction_model/*`; `setup_errors: []`; setup blocks for data, objects, preflight,
  reproduce, x1, x2, x3, x3-fisher, report and the sample stream; wall 8.921 s. As noted in N5 this
  executes the mirror setup path, not the mode bodies.
- `dryrun_b.log` / `shortcut_dryrun/plan.json`: exit 0; `gate_pending_acceptance` with
  `checks {a: true, b: false, c: true}` (pre-acceptance, expected); closure `pass: true`,
  `violations: []`; forced exceptions 27 × `src/noise_module/*`/`src/reconstruction_model/*` plus
  `part_b_forced_transitive_imports = [src/jepa_noise/ext_arch.py, src/jepa_noise/ext_dejitter.py]`;
  `setup_errors: []`; manipulation check `pass: true`; R-1 `not_run (pilot outputs missing)`
  (the normal, non-STOP pre-pilot state); wall 90.468 s.

---

# Re-review (round 2)

**Round-2 verdict: BLOCK retained.** Frozen reviewed object
`0cef73afc98d3ee6001c99fecff0632c4d216cda` (round-3 polish, parent `9d7d992`, parent `184cd2f`,
base runbook `062a8d1`). The round-1 blockers **B1–B9 are fixed**; the five round-1 BLOCKed
checklist items re-issue as four **PASS** (Part A ii, iii, iv; Part B ii) and one **BLOCK** (Part A
vi) on the remaining finding **B10** below. Eight of the round-1 nonblocking findings are fixed;
seven are partial/unchanged (none blocking).

## Frozen object and round-3 delta

- Frozen SHA `0cef73afc98d3ee6001c99fecff0632c4d216cda`; `git diff --stat 9d7d992..0cef73a` is
  exactly 2 files, +54/−2:
  - `src/jepa_noise/ext_repair.py` `EmbeddingCache.write` now also casts `particle`/`domain` to
    fixed-width `str` (`ext_repair.py:886-889`), so particle caches round-trip under
    `allow_pickle=False` (verified on the clean clone: an object-dtype `particle` cache writes and
    re-reads as `reused` with the string values intact);
  - `src/jepa_noise/tests/test_ext_repair.py` adds `test_x3_init9000_executes_photon_particle_and_calibrated`
    (`:1392-1443`), which executes the whole `_x3_init9000` body. This closes round-1 B6's
    test-coverage gap (the committed test set is now 53 tests).
- The two CLIs are byte-identical to round 1; the runbook sha256 is unchanged; the full diff
  `062a8d1..0cef73a` is still the same six added files, 10 353 insertions, with no existing file
  modified.
- The earlier `final2*/final3*` logs are **ignored**: the Lead's validation scratch exhausted the
  `/ceph/dwong` quota (Errno 122), so those runs contain setup errors and a killed suite
  (exit 143). Every piece of evidence below is from the clean lean runs `v/` (Deepthought) and
  `v_dt2/` (deepthought2) on the frozen SHA.

## Re-issued round-1 BLOCK items (frozen SHA)

| # | Subject | Verdict | Frozen-SHA evidence |
|---|---|---|---|
| A-ii | the X0-b reproduction and the identity check | **PASS** | `row_split` (`ext_repair.py:493`) feeds both X0-b environments (`:1699`, `:1655`) and the identity check (`:1729`, `:1765`, `:1782-1786`); the identity conditions are built per role on the role's own dataset (`:1771-1781`), so train and test run directories are both in-set; tests `test_row_split_propagates_into_streaming` (`test_ext_repair.py:993`) and `test_identity_check_runs_on_both_roles` (`:1016`) pass; focused run 53 passed |
| A-iii | probes, metrics, pairing and bootstrap | **PASS** | `stream_condition_particles` now takes `device` (`:927`) and `_x1_particle` passes it (`:1965-1983`); the six §5 particle models execute; the X1-g secondary unit-gain split is implemented (`:2076-2085`, `_x1d` `:2258-2274`); tests `test_stream_condition_particles_executes_and_caches` (`:1035`), `test_x1_particle_executes_with_all_conditions` (`:1057`) pass |
| A-iv | the X2 fits (run-disjoint train and test; no probe-test truth in any fit or hyperparameter choice) | **PASS** | `_standardisation` (`:2512`), `_stack_coefficients` (`:2780`), `_x2c` (`:2796`) are defined and wired into `_x2_model` (`:2941-3021`); the fit remains run-disjoint (`equivariance_fit` raises on overlapping run sets) and the latent λ CV still uses probe-train truth only (`latent_cv`); X2-b uses N-hit occupancy while X2-c and the latent stream use D-hit occupancy (`:2855-2858`, `:2939-2941`, `:2974`, `:2985-2989`, `:3019`); test `test_x2_model_executes_and_exposes_the_canonical_keys` (`:1082`) passes |
| A-vi | LEACE (reduced rank, permutation rule, random control) and the invariant transforms | **BLOCK** | the LEACE core, the reduced-rank permutation rule, the per-index random controls (`:3857-3864`), the invariant-token transform and the X3-b cache keys (`:4100`, `:4143-4144`) are correct and tested, but the fixed X3-b readings are still incomplete on two counts — see **B10** |
| B-ii | design: the R-1 families and reading are fixed and cannot be tuned after the pilot outcomes | **PASS** | `r1_payload` now collects **both** pilot payloads per model (`ext_shortcut.py:1666-1701`) with per-field payload preference (`R1_FIELD_PAYLOAD` `:1182-1189`, `_field_from_payloads` `:1276-1299`); `R1_GENERATIVE_CORE` pins the median to the eight core generative models (`:1175`, `:1416-1445`); response size excludes particle caches and prefers photon `test__N`/`test__D` (`:1486-1509`, `:1597-1625`); tests `test_r1_merges_both_pilot_payloads…` (`:684`), `test_r1_response_size_uses_the_photon_cache…` (`:733`), `test_r1_median_uses_exactly_the_eight_core_generative_models` (`:792`) pass; demonstrated with Part-A-shaped payloads (row carries `r2_eq=0.9`, `x2_class='equivariant'`, `rho_lr=0.9`, reading `supports`, 8 core models) |

## B1–B9 verification on the frozen SHA

- **B1 fixed.** `pyflakes` on the four frozen files exits 0 with an empty log (`v/pyflakes.log`,
  `pyflakes_exit=0`; independently re-run on the committed blobs, exit 0). The five names and the
  `device` parameter exist: `_X3_ROBUST_VERTEX_M`/`_X3_ROBUST_ENERGY_REL` (`ext_repair.py:239-240`),
  `_standardisation` (`:2512`), `_stack_coefficients` (`:2780`), `_x2c` (`:2796`),
  `stream_condition_particles(..., device="cpu", ...)` (`:927`). X2, the X1 particle half and X3-b now
  execute (`test_x2_model_executes…`, `test_x1_particle_executes…`,
  `test_x3_readings_execute_with_run_cluster_bootstraps`).
- **B2 fixed.** `row_split` (`:493`) is used for every reported photon stream: X0-b (`:1655`,
  `:1699`), identity check (`:1729`, `:1765`, `:1782-1786`), X1 (`:1924`, `:1933`), X2 (`:2877`,
  `:2886`), X2-e (`:3009`, `:3061`), X3-b init9000 (`:4017`, `:4031`, `:4035`). The only remaining
  hardcoded `split="within"` in `ext_repair.py` is the `_x2e` default (`:3042`, always overridden by
  the caller) and the diagnostic 8-event CLI dry-run sample (`scripts/jepa_ext_repair.py:140`,
  unchanged from round 1 and not a reported reading).
- **B3 fixed.** `identity_check` builds the N/D condition inside the role loop on the role's own
  dataset (`:1771-1781`) and uses the row's split (`:1765`); the new
  `test_identity_check_runs_on_both_roles` passes.
- **B4 fixed except B10.** `_metric_rows` now carries per-event `errors`/`run_ids`/`event_ids`
  (`:4216-4239`); `_ratio_and_ci` takes metric blocks and the paired run-cluster bootstrap uses real
  run ids (`:4245-4280`); `_compare_at_d` (`:4319-4337`) bootstraps the calibrated-vs-invariant
  difference with real run clusters; X3-4 now emits a combined verdict (`:4366-4384`); the §7.2
  robustness thresholds are `0.002` m / `0.05` relative (`:239-240`); the gain-specific flag now
  requires both the q95 and the control-mean-CI conditions (`:4386-4423`); the zero-run-id
  bootstraps are gone. The log-yield halves of X3-1 and gain-specific remain incomplete — B10.
- **B5 fixed.** All X3-b cache reads use `embedding_cache_key` (`:4100`, `:4143-4144`), exactly the
  keys X1 writes (`train__N`, `test__D`, `test__SWunit`, …); no `N__train`-style key remains; test
  `test_x3_leace_reads_the_keys_x1_writes` (`:1172`) passes.
- **B6 fixed (verified).** The raw calibrated variant is `raw_std_features(..., condition=swunit)`
  and `raw_std_features` honours the condition for both photon and particle rows (`:3548-3566`,
  `:3988`); `_x3_init9000` covers photon **and** particle invariant tokens plus the calibrated
  SWunit block (`:4006-4084`), and the new committed test executes it (`:1392-1443`, passes);
  erasure sanity is enforced (`not_applicable` / `unavailable (erasure incomplete)`, `:3867-3905`);
  random controls use one keyed stream per draw index (`:3857-3864`,
  `test_random_control_keys_are_per_index` `:1234`); X2-a uses the SWtw cached rows and aligned D
  rows (`:2438-2461`, `:2925-2927`, `test_x2_additivity_uses_the_swtw_rows` `:1269`); X2-b uses
  N-hit occupancy and X2-c/latent use D-hit occupancy (`:2855-2858`, `:2939-2941`, `:2974`,
  `:2985-2989`).
- **B7 fixed.** `r1_payload` reads both payloads (`ext_shortcut.py:1666-1701`) and picks each field
  from its own payload with a fallback (`:1182-1189`, `:1276-1299`); demonstrated on Part-A-shaped
  `x1.json`/`x2.json`: the row carries `r2_eq`, `x2_class`, `rho_lr` and the fixed reading returns
  `supports`/`contradicts` as expected; tests `:684-732` pass.
- **B8 fixed.** `is_particle_cache`/`_photon_cache_rank` (`:1486-1509`) and `cache_files`
  (`:1597-1625`) exclude particle/muon/electron caches and rank photon `test__N`/`test__D` first;
  demonstrated (both `particle__test__*.npz` excluded, `_first_verified` picks `test__N.npz`, S from
  the 4-D photon embeddings, not the 16-D particle ones); test `:733-791` passes.
- **B9 fixed.** `test_forced_import_classification` (`test_ext_shortcut.py:566-640`) tests the
  classifier itself with synthetic modules and no longer asserts a whole-process closure, so it is
  suite-context independent; the frozen full suite is green on deepthought2 (evidence below).

## B10 — (remaining blocking) X3-b's X3-1 reading is not produced/complete, and the raw-family X3-4 is not produced

All three points are in the §7.2 readings that round-1 item A-vi covers through B4; they are
demonstrated on the clean frozen clone (no GPU):

1. **The X3-1 "price in energy" class is never emitted.** `_variant_reading`
   (`ext_repair.py:4293-4315`) stores `rho_E` and a bootstrap/CI but never applies §7.2's fixed
   thresholds ("price present: ρ_E ≥ 1.2 **and** the run-cluster CI of res_inv − res_std excludes 0;
   no price: ρ_E ≤ 1.05; otherwise unclear"). A search of the frozen module for `1.2`/`1.05`/price
   class labels finds no classification code, and the README's price table prints only `rho_E`
   (`:4852-4856`).
2. **The log-yield X3-1 CI is absent for the init9000 and all LEACE variants.** `log_yield_row`
   (`:1863-1877`) returns `resolution`/`ci95`/`r2`/`n_events`/`errors` but **no**
   `run_ids`/`event_ids`; `_metric_rows` needs them (`:4216-4239`) and `_align_metric_pair`
   returns `None` without them, so `_ratio_and_ci` leaves `ci95 = None`. Demonstrated on the frozen
   clone: a `log_yield_row`-shaped block yields `run_ids` size 0 and
   `_ratio_and_ci(...)['ci95'] = None`. Consequences: (a) the §7.2 "price present" rule cannot be
   evaluated for log-yield on 11 of the 12 variant pairs (only the raw-feature family's log-yield
   blocks carry ids, via `feature_probe_metrics:3620-3622`); (b) `_gain_specific`'s log-yield
   `condition_ci_excludes_zero` is always `False` and the flag is reported `False` rather than
   `not available` (`:4410-4423`).
3. **The raw-family X3-4 is `not available`.** `_feature_variant` stores `calibrated` as flat
   per-metric blocks (`:3846-3852`), while `_compare_at_d` expects `{"metric": {"D": block}}`
   (`:4319-4337`). Demonstrated: `_x3_dominance(flat_calibrated, invariant)` returns
   `verdict='not available', status='not available'`, whereas the dict-shaped (LEACE/init9000)
   form returns a verdict. So the §7.2 X3-4 reading does not exist for the raw variant pair even
   though its calibrated SWunit data are now computed correctly.

The fixes are small (add `run_ids`/`event_ids` to `log_yield_row`; nest the raw calibrated block
under `"D"`; add the two X3-1 classification helpers). Until then Part A item (vi) and round-1 B4
are not fully resolved.

## Leftover nonblocking items (round 2)

- **Fixed:** N1 (X1-g secondary), N2 (aligned D rows and SWtw additivity), N7
  (`X1_PHOTON_SETS_PER_MODEL = 22`, `:274`), N14 (per-case/per-parameter ∞ counts, `:3339-3390`),
  N15 (README `theory_ratio_lnY`), N17 (`complete`/`completed` + `completeness`,
  `:4763-4821`), N18 (the C-var cut test now perturbs the excluded PMTs instead of asserting a
  tautology).
- **N3 partial:** content-mismatch scope and per-set HW availability are now recorded
  (`:1583-1605`), but the modes do not consume `readings_stopped`, and HW availability is reported
  per set while X1 still gates HW globally.
- **N4 partial:** `gpu_projection`/`cpu_projection` are now STOP entries (`:1617-1621`), but the
  §8.1 "drop the optional models" branch is dead code: it compares `gpu_projection()["pass"]`
  (core-only) with `gpu_projection_core()["pass"]`, which are the same dict (`:1613-1627`). It is
  currently unreachable (core 2.84 h, with optional 3.46 h ≤ 4.0 h).
- **N5 unchanged:** the Part A dry run still exercises `_mode_setup_objects`, not the mode bodies.
- **N6 not addressed:** the extra read-only `ext_followup` helpers beyond §2.4's list are still used
  and undocumented.
- **N8–N13 unchanged** as recorded in round 1 (mechanism wording, wall-cap reporting, data-preflight
  recheck in reproduce/evaluate, R-2-a fallback, Part B dry-run scope); none is a checklist blocker.
- **N16 (new characterisation):** the clean Deepthought full suite has exactly one failure, the
  untouched, data/numerics-dependent `test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows`
  (`'y': {'checked': True, 'equal': False, 'shapes': [[64000, 128], [64000, 128]]}`); the same test
  passes on deepthought2 (748 passed there). It is not caused by the diff (the file and its inputs
  are untouched and it runs before any new test) and is recorded as the observed pre-existing host
  dependence, not rounded into a pass.
- **N19 (new):** `x3.json` never carries the X3-1 class labels even where the inputs exist (part of
  B10 item 1); once fixed this is a reporting-layer issue.
- **N20 (new):** X3-4 is `not available` for the six photon-only LEACE models (T-1 ×4,
  frozen_target ×2) because the dominance verdict requires both vertex and energy; §7.2 says those
  LEACE readings are photon-only, so the runbook is ambiguous about whether a vertex-only X3-4
  verdict is required. Nonblocking ambiguity.
- **N21 (new):** the earlier `final2*/final3*` evidence trees and logs were contaminated by the
  `/ceph` quota exhaustion (Errno 122) and by files being synced while a suite was running; they are
  excluded. The frozen worktree (except this review file) and the host clone are clean at `0cef73a`.

## Evidence (clean lean runs on the frozen SHA)

`/ceph/dwong/jepa/tmp/jepa_rb_20261107/v/summary.txt` (Deepthought, complete, `DONE`):

```
SHA=0cef73afc98d3ee6001c99fecff0632c4d216cda HOST=deepthought
pyflakes_exit=0
focused_a_exit=0
focused_b_exit=0
full_suite_exit=1
dryrun_a_exit=0
dryrun_b_exit=0
DONE
```

- `v/pyflakes.log`: empty (`pyflakes_exit=0`).
- `v/focused_a.log`: `53 passed in 159.32s` (exit 0) — includes the new `_x3_init9000` body test.
- `v/focused_b.log`: `26 passed in 15.72s` (exit 0).
- `v/full_suite.log`: `1 failed, 747 passed, 2 skipped, 3 deselected in 1134.43s (0:18:54)`, exit 1;
  the single failure is `test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows`
  (N16), untouched by the diff; only the three known `test_guard` launcher tests are deselected.
- `v/dryrun_a.log` / `v/repair_dryrun/dry_run.json`: exit 0, `gate_pending_acceptance`, closure
  `pass: true`, `violations: []`, `errors: []`, wall 9.85 s; the repair sample and the ten setup
  blocks (data/objects/preflight/reproduce/x1/x2/x3/x3-fisher/report/sample) are present.
- `v/dryrun_b.log` / `v/shortcut_dryrun/plan.json`: exit 0, `gate_pending_acceptance` (checks
  `{a: true, b: false, c: true}`, as expected pre-acceptance), closure `pass: true`,
  `violations: []`, `setup_errors: []`, manipulation check `pass: true`, R-1
  `not_run (pilot outputs missing)` (the normal non-STOP pre-pilot state), wall 85.413 s. The log's
  preflight line `data=True s1=True caps=True entries=False` reflects the live guard/free-entries
  state (≈27k free at that moment, below the §F3 floor plus projection), not a dry-run setup error.
- `/ceph/dwong/jepa/tmp/jepa_rb_20261107/v_dt2/summary.txt` (deepthought2):
  `full_suite_dt2_exit=0`; `v_dt2/full_suite_dt2.log`:
  `748 passed, 2 skipped, 3 deselected in 486.82s (0:08:06)` — **green** on the same three
  deselections.

## Final reviewed object (frozen SHA)

- Implementation SHA (reviewed object): `0cef73afc98d3ee6001c99fecff0632c4d216cda` ("jepa: execute
  the X3-b init9000 body and fix particle-cache string arrays (round 3 final impl)"), parent
  `9d7d992`, base (runbook) `062a8d1`.
- File sha256 (`shasum -a 256`, frozen blobs):
  - `src/jepa_noise/ext_repair.py` `28c28c056dcb9845665af4bedf303471a3025b1fe55a0de4b36d87328dddeec8`
  - `src/jepa_noise/tests/test_ext_repair.py` `f74ff44c9d63e8c63771083901649b8045207a39e12cc2879d74de1c1b4e9a0d`
  - `src/jepa_noise/ext_shortcut.py` `537f6f1f2700f318a7f65ae0fbea19eb13e1f6989e577f8fdb2375d14344334c`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `746c03aef7415c5003e126389a9fe247668b0bf84852cfa78d7ad3ccd2202801`
  - `scripts/jepa_ext_repair.py` `dd9906e336abb056969fbc8246b8e1ee0fa048563b35aa176a8ae96c800073db`
  - `scripts/jepa_ext_shortcut.py` `616797f301ac7aa960877085a61eecb581acbd69f05d2e819d79e08d62feeeaf`
- Runbook sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b` (unchanged).
- Diff sha256: `9d7d992..0cef73a` `6a5404b7bae919db6b1f29305326c186c3c0c2cd179ee5b853aca0ede520e39d`;
  full `062a8d1..0cef73a` `3cc8de4a9077f68ed4da15c746bffb6d4c32ce980f0bb8c2b82ae55139f20864`.
- Worktree `/private/tmp/veritas-jepa-20261107` at `0cef73a` with only this review file untracked;
  host clone `/ceph/dwong/jepa/code/jepa_work_20261107` clean at `0cef73a`. All re-review checks
  were read-only on that clone (`TMPDIR` under `/ceph/dwong/jepa/tmp/jepa_rb_20261107/review`,
  `PYTHONPATH=src`, `nice -n 10`), no GPU, no commits/checkouts/resets, no implementation edits.

---

# Re-review (round 3)

**Round-3 verdict: PASS.** Frozen reviewed object
`320602ee49eed3339d3611c53d2de35d30dd5af6` (round-3 fix, parent `0cef73a`, base runbook `062a8d1`).
The round-2 blocker **B10** is fixed: the three sub-points plus the inverted X3-4 dominance sign are
re-verified below, Part A item (vi) re-issues as **PASS**, and the delta does not touch any other
reviewed surface. All 16 checklist items now PASS.

## Delta and scope

- `git diff --stat 0cef73a..320602e` is exactly 2 files, +212/−25: `src/jepa_noise/ext_repair.py`
  (9 hunks) and `src/jepa_noise/tests/test_ext_repair.py` (one append of 128 lines / four tests).
  The two CLIs are byte-identical (`dd9906e3…`/`616797f3…`), `ext_shortcut.py` and its tests are
  unchanged, the runbook sha256 is unchanged, and the full diff `062a8d1..320602e` is still the same
  six added files (10 540 insertions) with no existing file modified.
- The `ext_repair.py` hunks are confined to the B10 surface: the two price constants
  (`:242-243`), `log_yield_row` (`:1866-1888`), `_feature_variant`'s calibrated block
  (`:3850-3863`), the new `_price_class` (`:4262-4275`), `_variant_reading` (`:4318-4359`),
  `_compare_at_d` (`:4363-4384`), `_gain_specific` (`:4453-4479`) and the README X3 table
  (`:4898-4916`). No gate/closure, condition/§2.5, X0-b/identity, X1/X2, Fisher, LEACE-core or
  CLI code is touched, so there is no regression surface for B1–B9 or the round-1 PASS items.

## B10 re-verification (frozen SHA)

1. **X3-1's fixed price class is now emitted.** `_X3_PRICE_HIGH = 1.20` and `_X3_PRICE_NONE = 1.05`
   (`ext_repair.py:242-243`); `_price_class(rho, ci95)` (`:4262-4275`) implements §7.2 exactly
   (`price present` iff ρ_E ≥ 1.2 **and** the run-cluster CI excludes 0; `no price` iff ρ_E ≤ 1.05;
   otherwise `unclear`; a missing CI at ρ_E ≥ 1.2 is `not available (CI missing)`).
   `_variant_reading` emits per metric `rho_E`, `ci95`, `ci_excludes_zero`, `class`, `gain_specific`,
   `rho_controls`, `rho_control_q95`, `gain_specific_conditions` and `X3-1["classes"]`
   (`:4318-4359`); the README X3 table shows ρ_E + class + gain-specific (`:4898-4916`).
   Demonstrated on the frozen clone: `_price_class(1.30,(0.1,0.3))='price present'`,
   `(1.30,None)='not available (CI missing)'`, `(1.02,None)='no price'`,
   `(1.10,(-0.1,0.2))='unclear'`, `(None,None)='not available'`; a real `_variant_reading` pair
   returns `class='price present'` and `X3-1["classes"]={'log_yield': 'price present', …}`.
   Test `test_x3_price_class_uses_the_fixed_thresholds` passes.
2. **The log-yield X3-1 CI and gain-specific condition are now real.** `log_yield_row` returns the
   finite-masked `errors` **and** the matching `run_ids`/`event_ids` (`:1872-1884`), so
   `_metric_rows`/`_align_metric_pair`/`_ratio_and_ci` compute the paired run-cluster CI for the
   init9000 and all LEACE variants. Demonstrated: a real fit/score pair yields 20/20 pairing keys, a
   non-`None` `ci95`, and `class='price present'`; `_gain_specific` now reports a missing control CI
   as `status="not available (CI missing)"` with `gain_specific=None` and
   `condition_ci_excludes_zero=None` (`:4453-4466`) instead of a failed condition. Tests
   `test_log_yield_row_returns_pairing_keys_and_a_real_ci` and
   `test_missing_log_yield_ci_is_not_available_not_a_failed_condition` pass.
3. **The raw-family X3-4 now has a real verdict (and the sign is correct).** `_feature_variant`
   nests the calibrated block per metric under `"D"` (`:3850-3863`), so `_compare_at_d`/
   `_x3_dominance` (`:4363-4430`) evaluate. Demonstrated: `_x3_dominance` with a calibrated block
   whose per-event errors are half the invariant's returns `status='evaluated'`,
   `verdict='calibration dominates'`, `calibration_better=True`. The sign convention is now right
   (`difference = res_invariant − res_calibrated`; calibration better iff the CI's lower bound is
   > 0, `:4376-4382`) — the round-2 code had `calibration_better`/`invariance_better` swapped, and
   the fix is confirmed by the demo and by `test_raw_feature_family_x3_4_has_a_real_verdict`.
- The four new tests pass in isolation on the clean frozen clone (`4 passed, 53 deselected`), and
  `pyflakes` is clean on both modules and both CLIs (`pyflakes_exit=0`, empty log).

## Re-issued round-1/round-2 BLOCK item

| # | Subject | Verdict | Frozen-SHA evidence |
|---|---|---|---|
| A-vi | LEACE (reduced rank, permutation rule, random control) and the invariant transforms | **PASS** | LEACE core, reduced-rank permutation rule, per-index random controls and invariant transforms unchanged from the round-2 PASS state; the three remaining X3-b reading defects fixed above (`:242-243`, `:1866-1888`, `:3850-3863`, `:4262-4275`, `:4318-4359`, `:4363-4384`, `:4453-4479`, `:4898-4916`); four new tests + focused 57 passed; B10 resolved |

## Regression check

- The delta touches only the B10 surface (9 `ext_repair.py` hunks listed above + one test append);
  no other file changed (CLIs byte-identical, `ext_shortcut.py` and its tests unchanged, runbook
  unchanged).
- Fresh validation on the frozen SHA: focused Part A **57 passed**, focused Part B **26 passed**,
  pyflakes clean, both dry runs exit 0 with `gate_pending_acceptance`, import closure `pass: true`,
  `closures violations: []`, no setup errors (details below). Part B's suite is unchanged and still
  green on depth2 (752 passed). B1–B9 code paths are untouched by the delta and the round-2 tests
  (`test_row_split_propagates…`, `test_identity_check_runs_on_both_roles`,
  `test_x2_model_executes…`, `test_x3_leace_reads_the_keys_x1_writes`,
  `test_r1_merges_both_pilot_payloads…`, `test_r1_response_size_uses_the_photon_cache…`, B9's
  `test_forced_import_classification`, etc.) all pass in the 57/26 focused runs and the full suite.

## Evidence (frozen SHA `320602e`, clean lean runs)

`/ceph/dwong/jepa/tmp/jepa_rb_20261107/w/summary.txt` (Deepthought, complete, `DONE`):

```
SHA=320602ee49eed3339d3611c53d2de35d30dd5af6 HOST=deepthought
pyflakes_exit=0
focused_a_exit=0
focused_b_exit=0
full_suite_exit=1
dryrun_a_exit=0
dryrun_b_exit=0
DONE
```

- `w/pyflakes.log`: empty (`pyflakes_exit=0`).
- `w/focused_a.log`: `57 passed in 169.76s` (exit 0).
- `w/focused_b.log`: `26 passed in 15.57s` (exit 0).
- `w/full_suite.log`: `1 failed, 751 passed, 2 skipped, 3 deselected in 1107.56s (0:18:27)`, exit 1;
  the single failure is the untouched, data/numerics-dependent
  `test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows` (N16,
  `'y': {'checked': True, 'equal': False, ...}`), recorded as observed and not rounded into a pass;
  only the three known `test_guard` launcher tests are deselected.
- `w/dryrun_a.log` / `w/repair_dryrun/dry_run.json`: exit 0, `gate_pending_acceptance`, closure
  `pass: true`, `violations: []`, `setup_errors: []`, wall 8.989 s.
- `w/dryrun_b.log` / `w/shortcut_dryrun/plan.json`: exit 0, `gate_pending_acceptance` (checks
  `{a: true, b: false, c: true}`, as expected pre-acceptance), closure `pass: true`,
  `violations: []`, `setup_errors: []`, manipulation check `pass: true`, R-1
  `not_run (pilot outputs missing)` (normal, non-STOP pre-pilot), wall 84.986 s.
- `/ceph/dwong/jepa/tmp/jepa_rb_20261107/w_dt2/summary.txt` (deepthought2):
  `full_suite_dt2_exit=0`; `w_dt2/full_suite_dt2.log`: `752 passed, 2 skipped, 3 deselected in
  494.45s (0:08:14)` — **green**, same three deselections (this is the same suite/selection as
  Deepthought and shows the one Deepthought failure is host-dependent/pre-existing N16).

## Final reviewed object (frozen SHA)

- Implementation SHA (reviewed object): `320602ee49eed3339d3611c53d2de35d30dd5af6` ("jepa: fix
  round-2 blocker B10 (X3-b readings, final impl)"), parent `0cef73a`, base (runbook) `062a8d1`.
- File sha256 (`shasum -a 256`, frozen blobs):
  - `src/jepa_noise/ext_repair.py` `053396eb98dcd77857752b5e31eafe2fffdce4a185ac451b40de75c5f38dd21b`
  - `src/jepa_noise/tests/test_ext_repair.py` `0c626b847a5478dd51946d185f54b66b0df4e8b2ddffe86853ebc87381af076a`
  - `src/jepa_noise/ext_shortcut.py` `537f6f1f2700f318a7f65ae0fbea19eb13e1f6989e577f8fdb2375d14344334c`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `746c03aef7415c5003e126389a9fe247668b0bf84852cfa78d7ad3ccd2202801`
  - `scripts/jepa_ext_repair.py` `dd9906e336abb056969fbc8246b8e1ee0fa048563b35aa176a8ae96c800073db`
  - `scripts/jepa_ext_shortcut.py` `616797f301ac7aa960877085a61eecb581acbd69f05d2e819d79e08d62feeeaf`
- Runbook sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b` (unchanged).
- Diff sha256: `0cef73a..320602e` `470c5b20a001b81621a04f67cbb126cc53b94693a3ae8a67ee4450a436b6815a`;
  full `062a8d1..320602e` `fff70bde9859d440a9f19b3dc5614939c14c139610c70ac8b7246b25c63238ad`.
- Worktree `/private/tmp/veritas-jepa-20261107` at `320602e` with only this review file untracked;
  host clone `/ceph/dwong/jepa/code/jepa_work_20261107` clean at `320602e`. All round-3 checks were
  read-only on that clone (`TMPDIR` under `/ceph/dwong/jepa/tmp/jepa_rb_20261107/review`,
  `PYTHONPATH=src`, `nice -n 10`), no GPU, no commits/checkouts/resets, no implementation edits; the
  reviewer scratch was removed afterwards.

---

# Re-review (round 4) — execution-defect fixes

**Round-4 verdict: PASS.** Frozen reviewed object
`7b9352c0760358375a0ec8ef4604a45115510e06` (parent `d9f71a1`, grandparent `01537c0` = docs/results
only, base runbook `062a8d1`); the previously PASSed implementation SHA was `320602e`. The three
execution-time defects in Part A and the post-acceptance test-state defect in Part B are fixed; the
delta contains nothing else; no round-3 PASS item regressed. All 16 checklist items remain **PASS**.

**Record of the round-3 miss.** The round-3 PASS did not catch these defects because they live in
code paths that no test executed at `320602e`: the X0-b `_frozen_refit_vertex` body (its only test
encoded the same wrong call shape as the bug), the JSON writer with the §2.3 inventory's nested
`Path` rows, and `identity_check`'s encode on a GPU encoder. They were exposed only by the first
real X0-b run. This round records that, verifies the fixes, and checks the updated/new tests now
exercise the repaired paths.

## Delta and scope

- Code delta `320602e..7b9352c` = exactly 3 files, +44/−11 (`src/jepa_noise/ext_repair.py`,
  `src/jepa_noise/tests/test_ext_repair.py`, `src/jepa_noise/tests/test_ext_shortcut.py`). It
  decomposes as the three Part A fixes in `01537c0..d9f71a1` (2 files, +38/−9) plus the Part B
  test-state fix in `d9f71a1..7b9352c` (1 file, +6/−2). `320602e..01537c0` changes **no**
  implementation file (`git diff --stat 320602e..01537c0 -- src scripts` is empty; only the
  acceptance record, review, decisions, ledger, state and results artifacts). The two CLIs are
  byte-identical (`dd9906e3…`/`616797f3…`), `ext_shortcut.py` is unchanged (`537f6f1f…`), the
  runbook sha256 is unchanged, and relative to the base only the same six source files were added
  (no existing source file modified).
- The `ext_repair.py` delta is confined to three functions: `_jsonable` (`:787-792`),
  `_frozen_refit_vertex` (`:1653-1676`) and `identity_check` (`:1806-1811`). The
  `test_ext_shortcut.py` delta is one test-only change (`:433-441`).

## The four changes (file:line + demos on the frozen clone)

1. **`_jsonable` stringifies `os.PathLike`** (`ext_repair.py:787-792`; `isinstance(value, os.PathLike)
   → str(value)`). This is the fix for the X0-a/X0-b write crash: `write_json` serialises with
   `json.dumps(..., default=float)`, and the §2.3 inventory rows carry `Path` `source` values.
   Demonstrated on the frozen clone: a nested-Path inventory-like payload now round-trips through
   `_jsonable`/`write_json` (values become strings), while the pre-fix call reproduces exactly
   `TypeError: float() argument must be a string or a real number, not 'PosixPath'`. New test
   `test_jsonable_and_write_json_handle_paths` (`test_ext_repair.py:1024`) passes.
2. **`_frozen_refit_vertex` uses the production call shape and builds the §2.5 condition per role**
   (`ext_repair.py:1653-1676`): `geometry = probes["geometry"]`; the loop is
   `("train", probes["train"]), ("test", probes["test"])`; for each role
   `role_condition = build_condition(condition["name"], dataset, geometry)` (`:1665`) is built on the
   dataset about to be streamed, so probe-train run directories are inside their own condition's set.
   For X0-b the factor is 1 (N and D), so the identity semantics and the cached embeddings are
   unchanged (same cache keys `embedding_cache_key(role, condition["name"])`). Demonstrated:
   production-shape call streams `[('tr', 'across'), ('te', 'across')]` and builds conditions
   `[('D', 'tr'), ('D', 'te')]`; the updated `test_row_split_propagates_into_streaming`
   (`:993-1022`) asserts exactly the production shape and the per-role condition build.
3. **`identity_check` moves the CPU probe batches to the encoder's device before `encode`**
   (`ext_repair.py:1806-1811`): `encode_device = torch.device(device)` (or the passed
   `torch.device`) and `model.encode(plain.features.to(encode_device))` (likewise wrapped). The
   caller passes the model's device (`reproduction` → `identity_check(..., device=device)`, `:1739`),
   and the CPU-only path is a no-op move. Verified by code inspection plus the CPU
   `test_identity_check_runs_on_both_roles`; the CUDA branch cannot be exercised here (no GPU
   allowed), but the moved tensor's device is exactly the encoder's device by construction.
4. **Part B forbidden-path test is acceptance-state independent** (test-only,
   `test_ext_shortcut.py:433-441`): after the acceptance record was committed (`impl_commit=320602e`),
   the Part B gate legitimately *grants* on its own `<P>` (which contains no file changed since
   `320602e`), so the block that expected `SystemExit(2)` from the library entry points failed
   post-acceptance. The test now pins the pending state with
   `monkeypatch.setattr(xs, "acceptance_record", lambda root=None: None)` (`:438`); the real
   grant/pending behaviour remains covered by `test_gate_positive_and_pending_acceptance`
   (`:488-505`, fake repos, verified: granted with a matching record, pending+SystemExit without).
   This is test-only; the production gate code is byte-identical (`ext_shortcut.py` hash unchanged).
   On the clean frozen tree the gate states are observed as Part A **denied** (the committed record
   still names `impl_commit=320602e`, and `ext_repair.py` is in Part A's `<P>` and differs) and
   Part B **granted** (Part B's `<P>` is unchanged since `320602e`) — both are correct per §F5
   ("each part checks its own `<P>`"); after the new acceptance record with `impl_commit=7b9352c`
   both will grant.

## Regression check

- The delta touches only these three `ext_repair.py` functions (X0-b-only plus the strictly additive
  path serialisation) and two test files; no reading, threshold, item, gate, spec or CLI change.
- Fresh frozen-SHA validation: focused Part A **58 passed**, focused Part B **26 passed**, pyflakes
  clean; Deepthought full suite **1 failed / 752 passed / 2 skipped / 3 deselected** (the single
  failure is the untouched, data/numerics-dependent N16
  `test_ext_followup::test_label_collector_reproduces_the_committed_fix9000_rows`, recorded as
  observed, not rounded into a pass); deepthought2 full suite **753 passed / 2 skipped /
  3 deselected, exit 0** (green). All round-2/round-3 tests (B1–B10 paths) still pass.
- Both dry runs exit 0 with the import closure `pass: true`, `violations: []`, `setup_errors: []`
  (Part A gate observed `denied` and Part B `granted` as explained above; Part A 33.0 s, Part B
  116.795 s).

## Evidence (frozen SHA `7b9352c`, clean runs)

`/work/dwong/jepa_tmp_20261107/v5/summary.txt` (Deepthought, complete, `DONE`):

```
SHA=7b9352c0760358375a0ec8ef4604a45115510e06 HOST=deepthought
pyflakes_exit=0
focused_a_exit=0
focused_b_exit=0
full_suite_exit=1
dryrun_a_exit=0
dryrun_b_exit=0
DONE
```

- `v5/pyflakes.log`: empty (`pyflakes_exit=0`).
- `v5/focused_a.log`: `58 passed in 166.99s`; `v5/focused_b.log`: `26 passed in 22.45s`.
- `v5/full_suite.log`: `1 failed, 752 passed, 2 skipped, 3 deselected in 1179.42s (0:19:39)`, exit 1;
  only the N16 failure; only the three known `test_guard` launcher tests deselected.
- `v5/dryrun_a.log` / `v5/repair_dryrun/dry_run.json`: exit 0, gate `denied` (stale record, see
  above), closure `pass: true`, `violations: []`, `errors: []`, wall 33.0 s.
- `v5/dryrun_b.log` / `v5/shortcut_dryrun/plan.json`: exit 0, gate `granted` (see above), closure
  `pass: true`, `violations: []`, `setup_errors: []`, manipulation check `pass: true`, wall 116.795 s.
- `/work/dwong/jepa_tmp_20261107/v5_dt2/summary.txt` (deepthought2): `full_suite_dt2_exit=0`;
  `v5_dt2/full_suite_dt2.log`: `753 passed, 2 skipped, 3 deselected in 620.58s (0:10:20)` — green,
  same three deselections.

Note: the acceptance record currently committed (`impl_commit=320602e`) is the previous round's; the
new record with `impl_commit=7b9352c` is to be written after this PASS, and the committed
execution-stage artifacts (`results/jepa/20261106_repair/*`, `results/jepa/20261107_shortcut/*`) were
produced by the earlier adapted run. Neither is part of this code review object; the resumed run
will regenerate them with the fixed code and the new record.

## Final reviewed object (frozen SHA)

- Implementation SHA (reviewed object): `7b9352c0760358375a0ec8ef4604a45115510e06` ("jepa: make the
  Part B forbidden-path gate assertion acceptance-state independent"), parent `d9f71a1`, base
  (runbook) `062a8d1`; previously PASSed implementation SHA `320602e`.
- File sha256 (`shasum -a 256`, frozen blobs):
  - `src/jepa_noise/ext_repair.py` `b19face959415f304a6e89344a7b5a592ae3188d9c488f7b9be222552cd320e3`
  - `src/jepa_noise/tests/test_ext_repair.py` `d6230d768150e4877a959a471c0422eb254b414e40ee59a6e2a4bceacf47be08`
  - `src/jepa_noise/ext_shortcut.py` `537f6f1f2700f318a7f65ae0fbea19eb13e1f6989e577f8fdb2375d14344334c`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `1bd0da27e86ac4779cfd3aed1aa9c5a4aa424cb3f0b394dcde1483c385b7764d`
  - `scripts/jepa_ext_repair.py` `dd9906e336abb056969fbc8246b8e1ee0fa048563b35aa176a8ae96c800073db`
  - `scripts/jepa_ext_shortcut.py` `616797f301ac7aa960877085a61eecb581acbd69f05d2e819d79e08d62feeeaf`
- Runbook sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b` (unchanged).
- Diff sha256: `320602e..7b9352c` `0965a88dcea3ada68ea6ac979b4ad461da334ea030d4ab4df39d554c806bb0d5`;
  full `062a8d1..7b9352c` `3a9ce4610a5bb1b783d77efa55ab5ccfc9762f8947f9a03c9eae462dc5add57a`. For the
  decomposition: `01537c0..d9f71a1` `0545791e89198cdb9d811f9b26cc1eab579cdf67fcf29cba38037d32ba046190`
  and `d9f71a1..7b9352c` `de8b148fe36723f0aa758dd90c8789f102a370cbd3d93f10f06a281f19d8715e`.
- Worktree `/private/tmp/veritas-jepa-20261107` at `7b9352c`; host clone
  `/ceph/dwong/jepa/code/jepa_work_20261107` clean at `7b9352c`. All round-4 checks were read-only on
  that clone (`TMPDIR` under `/work/dwong/jepa_tmp_20261107/review`, `PYTHONPATH=src`,
  `nice -n 10`), no GPU, no commits/checkouts/resets, no implementation edits; the reviewer scratch
  was removed afterwards.

---

# Re-review (round 5) — X3-b raw-family truth fix

**Round-5 verdict: PASS.** Frozen reviewed object
`51999d20315bf097a7846a61b8bde25e708a76a7` (parent `1b30ad6` = the round-4 PASS acceptance/docs
commit; base runbook `062a8d1`). The X3-b raw-family defect (a `KeyError: 'drift'` when X3-b first
ran on GPU) is fixed, the delta contains nothing else, and no B1–B10 or round-4 item regressed. All
16 checklist items remain **PASS**.

## Delta and scope

- `git diff --stat 1b30ad6..51999d2` is exactly 2 files, +63/−7
  (`src/jepa_noise/ext_repair.py` 16 lines, `src/jepa_noise/tests/test_ext_repair.py` 54 lines);
  `git diff --stat 7b9352c..1b30ad6 -- src scripts` is empty (the round-4 PASS acceptance/docs commit
  changed no implementation file). The two CLIs are byte-identical (`dd9906e3…`/`616797f3…`),
  `ext_shortcut.py` and its tests are unchanged (`537f6f1f…`/`1bd0da27…`), the runbook sha256 is
  unchanged, and relative to the base only the same six source files were added (no existing source
  file modified).
- The `ext_repair.py` delta is one hunk in `x3()` (`:3995-4004`); the test delta is one appended
  regression test (`test_ext_repair.py:1589-1638`).

## The fix (file:line + demos on the frozen clone)

- **Defect.** `x3()` loads the particle truth per probe split only:
  `truth = {role: ef.load_particle_truth(data_root, PROBE_KEYS_S005[role]) for role in
  ("train", "test")}` (`ext_repair.py:3986-3987`), while the pre-fix standard raw-family loop did
  `truth[role]` for roles `train`, `test`, **`drift`** → `KeyError: 'drift'`. Demonstrated: with
  `truth = {"train": …, "test": …}`, the pre-fix call shape `truth["drift"]` raises exactly
  `KeyError('drift')`.
- **Fix.** The standard raw-family loop (`:3995-4004`) now maps
  `("train", probes005["train"], "train"), ("test", probes005["test"], "test"),
   ("drift", probes030["test"], "test")` and calls
  `raw_std_features(..., truth=truth[truth_role])` — the drift (s = 0.30) probe-test set reuses the
  s = 0.05 probe-test particle truth. This is the convention already used by the calibrated raw
  variant (`:4008-4009`) and by `_x3_init9000` (`truth_key = "test" if role == "drift" else role`).
  No feature, reading, threshold or item changed; the datasets and conditions are identical.
- **Validity of the convention (checked on the real probe data, read-only).** The S005 and S030
  probe-test sets share their run-id grid (200/200 run ids equal) and the first-run event ids are
  equal; the particle-truth fields that `ef.raw_event_features` consumes agree exactly or to float
  round-off across settings (`event_ids`, `vertex_m`, `particle`, `contained` identical;
  `yield_photons` max |Δ| = 1.2e-9, `deposited_MeV` 4.5e-13, `direction_m` 2.2e-16 — far below any
  reading tolerance). So the drift particle labels are correct and look up by run id.
- **Regression test.** `test_x3_raw_family_uses_the_s005_truth_for_the_drift_role`
  (`test_ext_repair.py:1589`) stubs `x3()`'s openers/family bodies and asserts the truth used for
  all three roles (`TRUTH_TRAIN`, `TRUTH_TEST`, `TRUTH_TEST`), the datasets
  (`TRAIN0.05`, `TEST0.05`, `TEST0.3`), that the standard drift call carries no calibration
  condition, and that the calibrated call uses the SWunit condition built on the D probe-test set
  (`COND:SWunit` for `TEST0.3`). It passes (focused Part A now 59 tests).

## Regression check

- Only the raw standard drift truth mapping changed (plus one new test); the raw train/test rows,
  the invariant and calibrated variants, all other X3-b families, X1/X2, the gate/closure and the
  round-4 fixes are untouched. No reading/threshold/item/gate/spec change.
- Fresh frozen-SHA validation: focused Part A **59 passed**, focused Part B **26 passed**, pyflakes
  clean; Deepthought full suite **1 failed / 753 passed / 2 skipped / 3 deselected** (the single
  failure is the untouched, data/numerics-dependent N16
  `test_ext_followup::test_label_collector_reproduces_the_committed_fix9000_rows`, recorded as
  observed, not rounded into a pass); deepthought2 full suite **754 passed / 2 skipped /
  3 deselected, exit 0** (green). All round-2/3/4 tests (B1–B10 + execution-defect fixes) still pass.
- Both dry runs exit 0 with the import closure `pass: true`, `violations: []`, `setup_errors: []`
  (Part A 48.093 s, Part B 84.175 s). As in round 4 the observed gate statuses with the still-committed
  round-4 acceptance record (`impl_commit=7b9352c`) are Part A **denied** (its `<P>` contains the
  changed `ext_repair.py`) and Part B **granted** (its `<P>` is unchanged since `7b9352c`); this is
  correct §F5 per-part behaviour, and both will grant after the new acceptance record with
  `impl_commit=51999d2`.

## Evidence (frozen SHA `51999d2`, clean runs)

`/work/dwong/jepa_tmp_20261107/v6/summary.txt` (Deepthought, complete, `DONE`):

```
SHA=51999d20315bf097a7846a61b8bde25e708a76a7 HOST=deepthought
pyflakes_exit=0
focused_a_exit=0
focused_b_exit=0
full_suite_exit=1
dryrun_a_exit=0
dryrun_b_exit=0
DONE
```

- `v6/pyflakes.log`: empty (`pyflakes_exit=0`).
- `v6/focused_a.log`: `59 passed in 182.58s`; `v6/focused_b.log`: `26 passed in 24.74s`.
- `v6/full_suite.log`: `1 failed, 753 passed, 2 skipped, 3 deselected in 1143.28s (0:19:03)`, exit 1;
  only the N16 failure; only the three known `test_guard` launcher tests deselected.
- `v6/dryrun_a.log` / `v6/repair_dryrun/dry_run.json`: exit 0, gate `denied` (stale record, see
  above), closure `pass: true`, `violations: []`, `errors: []`, wall 48.093 s.
- `v6/dryrun_b.log` / `v6/shortcut_dryrun/plan.json`: exit 0, gate `granted` (see above), closure
  `pass: true`, `violations: []`, `setup_errors: []`, manipulation check `pass: true`, wall 84.175 s.
- `/work/dwong/jepa_tmp_20261107/v6_dt2/summary.txt` (deepthought2): `full_suite_dt2_exit=0`;
  `v6_dt2/full_suite_dt2.log`: `754 passed, 2 skipped, 3 deselected in 831.35s (0:13:51)` — green,
  same three deselections.

Context recorded, not code: the first X3-b attempt exited 1 after 36 s (GPU hours booked, no output
written) on this defect; the rerun follows the new acceptance. The worktree also carries the resumed
run's execution outputs under `results/jepa/…` (modified `plan.json`, new
`reproduction/x1/x2/manipulation` payloads) — none is in `<P>` and no code differs there; the
`results/` artifacts are deliverables of the resumed execution, not part of this code review object.

## Final reviewed object (frozen SHA)

- Implementation SHA (reviewed object): `51999d20315bf097a7846a61b8bde25e708a76a7` ("jepa: fix the
  X3-b raw-family particle-truth lookup (drift role)"), parent `1b30ad6`, base (runbook) `062a8d1`;
  previously PASSed implementation SHA `7b9352c`.
- File sha256 (`shasum -a 256`, frozen blobs):
  - `src/jepa_noise/ext_repair.py` `191e96237fe582473bf5b1a5958b4a498a1e007deb7dc46ce203f387ac2be2cf`
  - `src/jepa_noise/tests/test_ext_repair.py` `78bd65aaa4b04f133a8c519f87cb6e9db395dd3d8d64b11b1f5aff8fccba8043`
  - `src/jepa_noise/ext_shortcut.py` `537f6f1f2700f318a7f65ae0fbea19eb13e1f6989e577f8fdb2375d14344334c`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `1bd0da27e86ac4779cfd3aed1aa9c5a4aa424cb3f0b394dcde1483c385b7764d`
  - `scripts/jepa_ext_repair.py` `dd9906e336abb056969fbc8246b8e1ee0fa048563b35aa176a8ae96c800073db`
  - `scripts/jepa_ext_shortcut.py` `616797f301ac7aa960877085a61eecb581acbd69f05d2e819d79e08d62feeeaf`
- Runbook sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b` (unchanged).
- Diff sha256: `1b30ad6..51999d2` `b59eaef74b1bc50e9b67ce7585d2ced14d25180856f85c1643ae8dfb321dab8c`;
  full `062a8d1..51999d2` `ed9a015c9f20762a3531239164d26925d0e92ec6024789ab436ba25916456d63`.
- Worktree `/private/tmp/veritas-jepa-20261107` at `51999d2` (its review file is the only
  reviewer-written change; the resumed run's `results/` outputs are as noted above); host clone
  `/ceph/dwong/jepa/code/jepa_work_20261107` clean at `51999d2`. All round-5 checks were read-only on
  that clone (`TMPDIR` under `/work/dwong/jepa_tmp_20261107/review`, `PYTHONPATH=src`,
  `nice -n 10`), no GPU, no commits/checkouts/resets, no implementation edits; the reviewer scratch
  was removed afterwards.
