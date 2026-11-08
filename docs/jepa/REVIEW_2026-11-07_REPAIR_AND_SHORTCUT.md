**Verdict: PASS**

**Final status (branch `jepa-runbook-20261107`, HEAD `90f190a8090661236383dc5fb9168c307814ac0b` =
`origin/JEPA`, addendum final round): PASS.** The combined-runbook implementation keeps its round-6
PASS at the frozen SHA `0cf61ff476bc8f3463b4595816ba40508cba239c` (rounds 1–6 below are the
historical record and are unchanged). The 7 Nov addendum's first review (`923276a`) BLOCKed A1–A4
and the fix round (`ffe3390`) BLOCKed A5/A6; both reports are preserved as history. The final commit
`90f190a` (delta `ffe3390..90f190a`: the three Part B source files plus `docs/jepa/DECISIONS.md` and
`docs/jepa/ARCHIVE_2026-11-08_MANIFEST.jsonl`; the binding runbooks are unchanged) fixes A5 with an
order-independent R-1 source chain (D1 overlay → corrected-results marker → 7 Nov payload, with the
basis recorded) and resolves A6 operationally (free entries 25,878 → 41.5k+, guard OK; the archive
manifest verified), as reviewed in the “## Addendum review — final (A5, A6)” section at the end of
this file. All eight addendum checklist items (i)–(viii) PASS, with no new blocking findings (four
nonblocking notes N1–N4 plus carried legacy notes). The acceptance record can be re-issued with
`addendum_path` / `addendum_sha256` and `impl_commit = 90f190a`; the addendum §8 execution may then
start. The implementation is reviewed at `184cd2f` against the binding runbook
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

---

# Re-review (round 6) — X3-b raw-std merged-table alignment

**Round-6 verdict: PASS.** Frozen reviewed object
`0cf61ff476bc8f3463b4595816ba40508cba239c` (parent `7f1221d` = the round-5 PASS acceptance/docs
commit; base runbook `062a8d1`). The second X3-b execution defect (the raw-std particle/photon merged
table was ragged, so the per-domain metric masks could not broadcast) is fixed; the delta contains
nothing else; no B1–B10 or rounds-4/5 item regressed. All 16 checklist items remain **PASS**.

## Delta and scope

- `git diff --stat 7f1221d..0cf61ff` is exactly 2 files, +55/−6
  (`src/jepa_noise/ext_repair.py` 30 lines, `src/jepa_noise/tests/test_ext_repair.py` 31 lines);
  `git diff --stat 51999d2..7f1221d -- src scripts` is empty (the round-5 PASS acceptance/docs commit
  changed no implementation file). The two CLIs are byte-identical (`dd9906e3…`/`616797f3…`),
  `ext_shortcut.py` and its tests are unchanged (`537f6f1f…`/`1bd0da27…`), the runbook sha256 is
  unchanged, and relative to the base only the same six source files were added (no existing source
  file modified).
- The `ext_repair.py` delta is confined to `_neutral_row_array` (`:3539-3554`) and
  `merge_standard_rows` (`:3556-3583`); the test delta is one appended regression test
  (`test_ext_repair.py:1643-1673`).

## The fix (file:line + demos on the frozen clone)

- **Defect.** `merge_standard_rows` concatenated field by field, and for a field present on only one
  side it kept that side's shorter array. `ef.raw_event_features` (muon/electron rows) has no
  `log_yield`, while the photon raw-std table has one, so the merged table had `features` of length
  10 000 but `log_yield` of length 3 400; `feature_probe_metrics(kind="log_yield")` then evaluated
  `np.isfinite(y) & (domain == "photon")` on mismatched lengths and raised
  `ValueError: operands could not be broadcast together with shapes (3400,) (10000,)`.
- **Fix.** `merge_standard_rows` (`:3556-3583`) now computes `particle_len`/`photon_len` from the two
  tables' `run_ids` (`:3563-3564`) and, for a field missing on one side, inserts a neutral column of
  the right shape/dtype via `_neutral_row_array` (`:3539-3554`: `False` for bool, `"none"` object for
  string/object, `NaN` float otherwise) before concatenating (`:3575`). Every merged field therefore
  has length = particle rows + photon rows; the photon `log_yield`/`domain` masks then select exactly
  the photon rows, and particle rows get a neutral `log_yield = NaN`. This affects only the
  raw-feature family's table construction; no feature formula, reading, threshold or item changed.
- **Demonstrated.** Pre-fix ragged merge of a 2-row particle table (no `log_yield`) and a 5-row
  photon table gives lengths `features 7, log_yield 5, domain 7`, and the real
  `feature_probe_metrics(kind="log_yield")` on it raises
  `ValueError: operands could not be broadcast together with shapes (5,) (7,)` — the same failure
  pattern as the GPU run. Post-fix, `merge_standard_rows` returns all fields of length 7, the two
  particle rows' `log_yield` are `NaN`, the photon `log_yield` values are intact, `domain` is
  `["muon","electron","photon","photon","photon","photon","photon"]`, the photon log-yield mask
  selects exactly 5 rows, and `feature_probe_metrics(kind="log_yield")` returns `status="evaluated"`
  (no broadcast error).
- **Regression test.** `test_merge_standard_rows_aligns_every_column_for_domain_masks`
  (`test_ext_repair.py:1643`) builds a 2-particle + 3-photon table, asserts all eleven merged fields
  have length 5, that `log_yield` is NaN for the particle rows and `[7,8,9]` for the photon rows,
  that `domain` is `["muon","electron","photon","photon","photon"]`, and that the photon mask selects
  3 rows; it passes (focused Part A now 60 tests).

## Regression check

- Only the raw-std table construction changed (plus one test); `merge_standard_rows` has a single
  production call site (`raw_std_features` → the raw-feature family), and the change is limited to
  filling missing columns — the per-source tables and their orders are unchanged, so the merged
  row order and the existing pairwise keys are preserved.
- Recorded honestly: the earlier rounds did not execute this path end to end — round-5's test stubs
  `raw_std_features`/`_variant_metric_rows`/`_feature_variant`, and round-3's raw-family test
  monkeypatches `feature_probe_metrics` — so the ragged merge only appeared on the real GPU run.
  The new test now covers the construction directly.
- Fresh frozen-SHA validation: focused Part A **60 passed**, focused Part B **26 passed**, pyflakes
  clean; Deepthought full suite **1 failed / 754 passed / 2 skipped / 3 deselected** (the single
  failure is the untouched, data/numerics-dependent N16
  `test_ext_followup::test_label_collector_reproduces_the_committed_fix9000_rows`, recorded as
  observed, not rounded into a pass); deepthought2 full suite **755 passed / 2 skipped /
  3 deselected, exit 0** (green). All round-2..5 tests still pass.
- Both dry runs exit 0 with the import closure `pass: true`, `violations: []`,
  `setup_errors: []` (Part A 25.753 s, Part B 83.978 s). As in rounds 4–5 the observed gate statuses
  with the still-committed round-5 acceptance record (`impl_commit=51999d2`) are Part A **denied**
  (its `<P>` contains the changed `ext_repair.py`) and Part B **granted** (its `<P>` is unchanged
  since `51999d2`); this is correct §F5 per-part behaviour, and both will grant after the new
  acceptance record with `impl_commit=0cf61ff`.

## Evidence (frozen SHA `0cf61ff`, clean runs)

`/work/dwong/jepa_tmp_20261107/v7/summary.txt` (Deepthought, complete, `DONE`):

```
SHA=0cf61ff476bc8f3463b4595816ba40508cba239c HOST=deepthought
pyflakes_exit=0
focused_a_exit=0
focused_b_exit=0
full_suite_exit=1
dryrun_a_exit=0
dryrun_b_exit=0
DONE
```

- `v7/pyflakes.log`: empty (`pyflakes_exit=0`).
- `v7/focused_a.log`: `60 passed in 148.17s`; `v7/focused_b.log`: `26 passed in 17.35s`.
- `v7/full_suite.log`: `1 failed, 754 passed, 2 skipped, 3 deselected in 1119.29s (0:18:39)`, exit 1;
  only the N16 failure; only the three known `test_guard` launcher tests deselected.
- `v7/dryrun_a.log` / `v7/repair_dryrun/dry_run.json`: exit 0, gate `denied` (stale record, see
  above), closure `pass: true`, `violations: []`, `errors: []`, wall 25.753 s.
- `v7/dryrun_b.log` / `v7/shortcut_dryrun/plan.json`: exit 0, gate `granted` (see above), closure
  `pass: true`, `violations: []`, `setup_errors: []`, manipulation check `pass: true`, wall 83.978 s.
- `/work/dwong/jepa_tmp_20261107/v7_dt2/summary.txt` (deepthought2): `full_suite_dt2_exit=0`;
  `v7_dt2/full_suite_dt2.log`: `755 passed, 2 skipped, 3 deselected in 637.46s (0:10:37)` — green,
  same three deselections.

Context recorded, not code: the first X3-b attempt (round-5 drift-truth defect) exited 1 after 36 s
and this second attempt (this defect) after 46 s, both booked with no output; the X3-b rerun follows
the new acceptance. The worktree also carries the resumed run's execution outputs under
`results/jepa/…` (modified `plan.json`, new reproduction/x1/x2/manipulation payloads) — none is in
`<P>` and no code differs there.

## Final reviewed object (frozen SHA)

- Implementation SHA (reviewed object): `0cf61ff476bc8f3463b4595816ba40508cba239c` ("jepa: align the
  X3-b raw-std merged table across particle/photon rows"), parent `7f1221d`, base (runbook) `062a8d1`;
  previously PASSed implementation SHA `51999d2`.
- File sha256 (`shasum -a 256`, frozen blobs):
  - `src/jepa_noise/ext_repair.py` `6845c3019f6eb67a53fd3bb16339a46e15131a90e3753f4217520720de622fd3`
  - `src/jepa_noise/tests/test_ext_repair.py` `67b2d50f5581954902ca89c9574b12cd32cd69126c8596b248ca3c3b71e8b62b`
  - `src/jepa_noise/ext_shortcut.py` `537f6f1f2700f318a7f65ae0fbea19eb13e1f6989e577f8fdb2375d14344334c`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `1bd0da27e86ac4779cfd3aed1aa9c5a4aa424cb3f0b394dcde1483c385b7764d`
  - `scripts/jepa_ext_repair.py` `dd9906e336abb056969fbc8246b8e1ee0fa048563b35aa176a8ae96c800073db`
  - `scripts/jepa_ext_shortcut.py` `616797f301ac7aa960877085a61eecb581acbd69f05d2e819d79e08d62feeeaf`
- Runbook sha256 `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b` (unchanged).
- Diff sha256: `7f1221d..0cf61ff` `77c5f897ea321c24ae5aaf5e30e9a8daf095e38cc926f369e082c492ebfc16e7`;
  full `062a8d1..0cf61ff` `b14f0b1e0f174544e651484454b99c02e1263b8c10477523a1aea58885a3171a`.
- Worktree `/private/tmp/veritas-jepa-20261107` at `0cf61ff` (its review file is the only
  reviewer-written change; the resumed run's `results/` outputs are as noted above); host clone
  `/ceph/dwong/jepa/code/jepa_work_20261107` clean at `0cf61ff`. All round-6 checks were read-only on
  that clone (`TMPDIR` under `/work/dwong/jepa_tmp_20261107/review`, `PYTHONPATH=src`,
  `nice -n 10`), no GPU, no commits/checkouts/resets, no implementation edits; the reviewer scratch
  was removed afterwards.

# Addendum review (D1–D5)

**Verdict: BLOCK.** Fresh-context round-7 review of the 7 Nov 2026 repair addendum delta
`0cf61ff..923276a`. The combined-runbook implementation at `0cf61ff` keeps its round-6 PASS (history
above unchanged). The D1–D3 core, the D4 report generator body and the D5 R-2 evaluation path are
sound. But the D1→R-1 hand-off is not wired (A1/A2), the addendum S1 is enforced on only one side
(A3), and the only entries-preflight measurement in the supplied evidence fails (A4). Items (vi) and
(viii) are therefore **BLOCK**; items (i)–(v) and (vii) PASS. Findings: 4 blocking (A1–A4) and 9
nonblocking (N1–N9).

## Reviewed object, SHAs and file sha256s

- Worktree `/private/tmp/veritas-jepa-20261107`, branch `jepa-runbook-20261107`, HEAD
  `923276afa6c48d79751818019d9a4a6612b8d4f4` = `origin/JEPA`. Host clone
  `/ceph/dwong/jepa/code/jepa_work_20261107` at the same SHA; its tracked tree is clean (the only
  untracked files are the execution's 7 Nov `results/` payloads, none in `<P>`).
- Binding documents, verified unchanged:
  - combined runbook `45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b`
    (`git diff 0cf61ff..HEAD -- docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md` empty);
  - addendum `abe0904af0ed0b5a6586cee459c507ae0fa2a7c60c65ffb58192c71b7180621a`
    (`git diff 48d25a1..HEAD -- docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md` empty, and the blob
    added by `48d25a1` hashes to the same value).
- Delta: `0cf61ff..923276a` over the six files, +4528/−102; the other commits in the range are
  bookkeeping (`48d25a1` addendum, `9f5579e` labels, `9d55a82` 7 Nov results, `72a1ba2` round-6
  review/acceptance). No source file outside the six changed:
  `git diff --name-only 0cf61ff..HEAD -- 'src/**/*.py' 'scripts/**/*.py'` returns exactly the six.
- File sha256 (`shasum -a 256`, frozen blobs):
  - `src/jepa_noise/ext_repair.py` `c4fcdcdae0747185d114020b5625a89ad3e1c99680de4f608d7f2b95d566fb50`
  - `src/jepa_noise/tests/test_ext_repair.py` `63256db762223c7d32fd5333a959fd4cdc5875ee371e0096c88742335d0a409b`
  - `src/jepa_noise/ext_shortcut.py` `d67f963ab98660931b72a519c430d4448bfeda0d9e5b8c398bb311ef2d1024c4`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `fcffd36fa8825aaa7fae98e44a2a806a78b5cab38f03e0e6690ac1bd3b879b45`
  - `scripts/jepa_ext_repair.py` `cc7c4a9f9da53a71e069e7d514d0e744054b0b2dcbcef0227d8080f1565d2aa1`
  - `scripts/jepa_ext_shortcut.py` `c9f4c58a801f40c7d8895d827e6ef14cb6c19c9f3b7bab198cc408cbb63023b6`
- Diff sha256 (`git diff 0cf61ff..923276a -- <the six> | shasum -a 256`):
  `4b4d339725aebf72a138ba843eb4ccf1ca3d3d82a34ba30e7ea250d248583cf3`; the `0cf61ff..HEAD` value is
  identical (HEAD == `923276a`).
- Acceptance record at review time (not the re-issued one):
  `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json`,
  sha256 `750a7f6578a6997bad89f1be2a33703ec0f024344b55e94fc9a80dad50ae4ef1`; it carries the six 7 Nov
  keys only (`impl_commit` `0cf61ff`), so both gates correctly DENY (not pending) and the lead must
  re-issue it with `addendum_path`/`addendum_sha256` (checked in both gates, below).

## Checklist — addendum §7 (i)–(viii)

| # | Subject | Verdict | Evidence |
|---|---|---|---|
| (i) | `raw_map` correct incl. constant columns; only route to Â | **PASS** | `ext_repair.py:2902-2941` (Â_raw = coefficients/scale; only `.coefficients` read at `:2921`); `probes.py:34-54`, constant-column rule at `:44-46` (scale 1 and an exactly-zero standardised column, so the coefficient is 0); tests `test_ext_repair.py:1745-1790` (1e-10 identity incl. constant column, 1e-6 recovery, AST route check asserting the only owner of `.coefficients` is `raw_map` at `:1771-1785`) |
| (ii) | every Â consumer converted; source/raw streams unchanged | **PASS** | producers `_fit_equivariance_occupancy:2751-2769` (`raw_map` at `:2767`), `_stack_coefficients:2943-2960`; consumers `_x2c:2962-2980`, `latent_estimate:2606-2638`, `latent_cv:2645-2699` (raw map from `:2671`), `latent_stream_estimates:2701-2746` (`:2718`), `_latent_repair_rows:3244-3252`, `identifiability_spectrum:2983-2998`; addendum paths `_addendum_x2_model:5542-5594`, `addendum_x2e:5656-5765`; source/raw take no Â (`source_estimate:2784-2832`, `raw_estimate:2834-2878`) and the summary keeps the 7 Nov values (test `test_ext_repair.py:2129-2160`); repair is still `z_D − u Â_rawᵀ`, no intercept (`:2968-2969`, `:3250-3251`) |
| (iii) | D1 self-check and S9 rule | **PASS** | `d1_identity_check:5410-5423` (default tolerance 1e-8), `_d1_check_model:5424-5468` (removed share, recovered fraction with P_N, `r2_eq ≥ 0.4 ∩ recovered < 0` flag), `addendum_selfcheck:5470-5539` (13 core models, `identity_pass`, S9 text on failure, exit 2 via `scripts/jepa_ext_repair.py:354-371`); tests `test_ext_repair.py:1842-1921` |
| (iv) | D2 statistics in every comparison of both modules | **PASS** (note N3) | `metric_statistic:4569-4581`; `_compare_at_d:4582-4620` uses `ef.paired_run_bootstrap` (mean point **and** mean interval) for vertex and `paired_cluster_bootstrap(statistic=rms)` for log-yield/energy on the same aligned pairs (`_align_metric_pair:4450-4466`); `_x1c:2343-2362` mean/mean; `_ratio_and_ci:4483-4500` and `_gain_specific:4668-4712` RMS/RMS; X3-3 is a point-only robustness difference; `ext_shortcut` reports no point/interval pair, and `statistic_audit:1506-1603` checks any interval's declared statistic against its point (tests `test_ext_repair.py:1922-1985`, `test_ext_shortcut.py:1120-1155`). N3: X3-2 has no CI at all — see notes |
| (v) | D3 folded into `<P>`; adapter retired | **PASS** | `_crlb_from_json:4714-4719` + `_x3_five:4721-4757` (JSON `null` efficient CRLB → ∞); `_INTERNAL_KEYS:917-918` + `_strip_internal:919-930` drop `fitted`/`prediction`/error arrays, and `x3()` strips before writing (`:4190`); tests `test_ext_repair.py:1986-2016`. The runtime adapter is no longer needed; its retirement is recorded in the code docstrings (`:915-918`, `_addendum_outcome` note) and the DECISIONS/OUTCOME line is a §10 post-execution deliverable |
| (vi) | D4 completeness, superseded table, `ADDENDUM_2026-11-07.json`, figure labels/legends | **BLOCK** | Report body is complete: `addendum_report:6387-6520` writes `SUMMARY.json`, `READINGS_SUMMARY.md`, `ADDENDUM_2026-11-07.json`, `OUTCOME.json` and the figures; the markdown carries X1-c/d/e/f, the labelled X2-d triple (`source (known light)` / `raw` / `latent`), X2-e, X2-f, X3-b per family incl. LEACE rank/sanity-R²/ρ_E/control-q95/gain-specific and X3-2/3/4 (`_addendum_summary_markdown:6165-6350`); `render_figures:4948-5250` gives every figure axis labels and a legend and names the three identification streams (`:5046`); test `test_ext_repair.py:2345-2440`. **Blocked by A1/A2**: the R-1 row of the superseded table cannot obtain the replaced value (the R-1 rerun is not D1-corrected and the picker misses Part B's output), so the §5 list item “R-1 … the new value” is not met. Also N8 (X2-f singular-value spectrum missing from the markdown) |
| (vii) | R-2 evaluation path and `not_run` label | **PASS** | `r2_readings:1353-1397` filters runs with no evaluation and returns `r2_not_run_readings:1326-1352` (`class/verdict = not_run`, never “single-seed”); `evaluate_run_dirs:1211-1250` does full input + half pass via `eh.evaluate_run`; `mode_addendum_evaluate` (`scripts/jepa_ext_shortcut.py:578-642`) is one job over `_run_dirs:466-481` with `label_batches:988-1008` (§5.2 64-event selection); `r2_evaluation_record:2647-2680`; tests `test_ext_shortcut.py:1057-1118`, `1255-1345`. N7: pass `--run` twice in execution rather than relying on the default directory scan |
| (viii) | gates, import closure, 0.35 h guard, S1, entries, dry-run coverage | **BLOCK** | Gates PASS: addendum sha/author-line/record fields checked in `ext_repair.py:432-460` and `ext_shortcut.py:459-497`; 8-key schema `ext_repair.py:141-143` / `ext_shortcut.py:200`; pending only when the record is absent; tests `test_ext_repair.py:2052-2090`, `test_ext_shortcut.py:900-940`; dry-run evidence shows Part A `gate_checks {a: true, a2: false, b: false, c: false}`, `pending_acceptance: false`, `expected_addendum_sha256: null` and Part B `addendum_sha256_pass: false` — the required deny-not-pending state. Closure PASS: `_assert_closure:534-543` via `_addendum_finish:5361-5369` and `addendum_init9000:6674`; Part B `_closure_end` (`scripts/jepa_ext_shortcut.py:138-148`) called at `:618`, `:677`, `:716`. Ledger/caps PASS: rows byte-identical and mutually readable (reviewer demo), caps 0.35/3.0 in `ext_repair.py:5274-5277,5307-5358` and `ext_shortcut.py:196-197,660-684`, closed caps not consulted by addendum modes (tests `test_ext_repair.py:2114-2130`, `test_ext_shortcut.py:1320-1345`). Dry-run coverage PASS: Part A's dry run iterates every mode, so `addendum-init9000/report/selfcheck/x2/x2e/x3` appear with the addendum plan (sha, caps, projections, shared ledger) and no setup errors; the only encode is the pre-existing CPU §2.5 sample (8 events, `dry_run.json:setup.sample`), no addendum GPU encode/evaluate. **Blocked by A3 (S1) and A4 (entries)** |

## Blocking findings

### A1 — the D1-corrected ρ_LR never reaches the R-1 rerun (addendum §2, D4 list item “R-1”)

`ext_shortcut.pilot_status:2078-2115` and `r1_payload:2117-2205` build R-1 from the pilot's
`results/jepa/20261106_repair/{readings.json,x1.json,x2.json}`. The corrected X2-c value is written by
the D1 rerun only to `<ext>/addendum/x2_addendum.json` (`ext_repair.addendum_x2` at
`ext_repair.py:5596-5654`; CLI `scripts/jepa_ext_repair.py:373-390`), and Part A's `addendum_report`
writes only `SUMMARY.json` / `READINGS_SUMMARY.md` / `ADDENDUM_2026-11-07.json` / `OUTCOME.json`
(`ext_repair.py:6505-6510`) — it never writes `results/jepa/20261106_repair/x2.json`, and the overlay
is never merged into the pilot payload. Demonstrated on the read-only host clone (TMPDIR under
`/work/dwong/jepa_tmp_20261107/review_add`, removed afterwards):

```
pilot x2 7 Nov rho_LR (mae_within_trigger9000) = 0.9
D1 overlay (corrected) rho_LR                 = -0.42
r1_payload(..., mode="addendum-r1") row       = 0.9     <-- the 7 Nov defective value
```

Consequence: the addendum §2 row “Part B R-1, with the corrected ρ_LR” is not delivered; R-1 is
computed from the D1-defective payload (the same failure mode as the historical B7, but now from a
stale payload instead of a single payload). The unit test
`test_ext_shortcut.py:1157-1175` only proves the reader accepts an already-corrected `x2.json`; it
writes the correction into the fixture by hand and does not exercise the hand-off.

### A2 — the “Superseded 7 Nov values” table cannot show the R-1 new value

`ext_repair._part_b_r1:6072-6084` looks for `r1_addendum.json` / `ADDENDUM_2026-11-07.json` in
`results/jepa/20261107_shortcut/` and only reads a top-level `r1`, `R-1`, `reading` or
`reruns.R-1`. Part B's `addendum-r1` writes `<ext>/r1_addendum.json` by default
(`scripts/jepa_ext_shortcut.py:673`), and `assemble_addendum_report` writes the results file as
`{"items": {"R1_rerun": ...}}` (`ext_shortcut.py:2996`), which the picker does not read.
Demonstrated with the real output structure:

```
results/.../ADDENDUM_2026-11-07.json (Part B output) present -> _part_b_r1 = (old, None)
R-1 row -> {"item": "R-1 reading", "defect": "D1", "old": {...}, "new": null, "status": "not_run"}
```

so `_superseded_rows:6137-6138` marks the required R-1 row `not_run`, and `addendum_report:6433`
adds “R-1 (Part B) not_run” even when the Part B rerun completed. The D4 test
`test_ext_repair.py:2403` only asserts the row key exists, so it passes with `new=None`. (If Part B's
`addendum-report` runs before Part A's, `assemble_addendum_report` has already overwritten
`readings.json`'s `r1` with the new reading — `ext_shortcut.py:2897` — so the “old” value is lost as
well; the two reports' order is unspecified in §8.)

### A3 — addendum S1 is enforced for Part B only, and not where §8 places the GPU jobs

`ext_repair.s1_projection:1414-1429` still computes the combined front-matter S1 from the closed
Part A (4.0 h) + Part B (1.5 h) caps; there is no addendum-S1 helper in `ext_repair`, and the two
Part A GPU modes check only the 0.35 h cap guard: `addendum_x2e:5656-5765` (`gate`, then
`_addendum_budget_for`) and `addendum_init9000:6567-6675` (same). The Part A dry-run addendum plan
contains the sha, caps, projections and ledger but no S1 (`va/repair_dryrun/dry_run.json`). Part B
does implement the exact rule — `addendum_s1_check(_or_stop)` (`ext_shortcut.py:705-735`:
ledger charged + 0.35 ≤ 160) — and calls it in `addendum-evaluate`
(`scripts/jepa_ext_shortcut.py:582`). But §8 orders init9000/x2e before the evaluation, so the STOP
rule of §6/§9 (S1) is checked after the Part A GPU jobs it is meant to gate. Numerically it passes
today (ledger 129.7461 + 0.35 ≪ 160), so the practical impact is nil, but the requirement is not
implemented where the spec puts it.

### A4 — the binding entries precondition fails in the supplied evidence, and no addendum mode checks it

The Part B dry-run preflight prints `[shortcut] preflight failed: data=True s1=True caps=True
entries=False` (`va/dryrun_b.log:1`) and the plan records `free_entries = 35,770`,
`projected_after = 28,886 < floor 30,000` (`va/shortcut_dryrun/plan.json`,
`setup.preflight.entries`; projection = clone 3,059 + suite 1,825 + Part A 1,000 + Part B 1,000).
Free entries (35,770) still exceed 30,000 and the addendum's own writes are a few dozen files, so
this may be the runbook projection over-counting clone/suite writes that the addendum does not
perform — but the addendum's §6 figure “44 286 at the last check” is stale, the §9 STOP list names an
entries-preflight failure, and no addendum mode runs an entries preflight at all
(`entries_preflight:1441-1450` is called only from `preflight`/setup with `build_model=True`,
`:1712,4865`). The lead must clear/free entries or record a revised entries basis (with its
projection) before execution; the supplied evidence cannot support a PASS on “entries ≥ 30 000”.

## Nonblocking notes

- **N1 — dead duplicated block.** `ext_shortcut.addendum_superseded:2744-2761` repeats the
  R2-a/R2-b/R2-c block *after* `return rows` (`:2744`). The live block (`:2729-2744`) is correct and
  the tests pass; pyflakes does not diagnose unreachable code, so this is sloppiness, not a behaviour
  change.
- **N2 — “helper call-shape parity” is inaccurate.** The ledger JSON is identical and mutually
  readable (reviewer demo: Part A `record_addendum_job(mode, kind, seconds, part=)` and Part B
  `record_addendum_job(part, *, mode, kind, seconds)` rows are `{utc, part, mode, kind,
  wall_seconds, hours}`, and each module's `addendum_hours` reads both; `ext_repair.py:5331-5350`,
  `ext_shortcut.py:686-704`). The call shapes themselves are not parity, and the S1 contract is
  single-sided (A3). The “shared ledger + identical caps + parity rows” reading of the 0.35 h guard
  is legitimate; “call-shape parity” and “S1 on both sides” are not what the code does.
- **N3 — X3-2 has no CI.** The runbook §7.2 defines X3-2 “v_inv(N) − v_std(N), with its CI”, but
  `_variant_reading:4543-4545` emits only `difference_m`, the markdown table prints only Δv
  (`:6307-6320`), and `test_ext_repair.py:1980-1984` explicitly asserts the absence of `ci95`. This
  is pre-existing at `0cf61ff` (`git show 0cf61ff:src/jepa_noise/ext_repair.py:4368-4370`), so it is
  not a D2 statistic *mismatch* (there is no interval to mismatch) and it is outside the addendum's
  D1–D5 defect list; but the D2 audit of X3-2 is then vacuous and a runbook-required CI is still
  missing. Flag for the next review round, not this addendum.
- **N4 — `_stack_coefficients` fallback width.** `ext_repair.py:2949-2952` falls back to
  `fit["train_mean"].size` (embedding dimensions) when no probe fit exists, but the matrix width is
  `n_modules`; a degenerate all-unfitted model would return a wrong-width zero map instead of failing
  cleanly. Not exercised by the evidence.
- **N5 — Part A closure-before-write.** `_addendum_finish:5361-5369` calls `_assert_closure:534-543`,
  which raises `SystemExit(2)` inside the library before the CLI writes the payload, so a closure
  violation leaves no quarantined/marked output (Part B's `_closure_end:138-148` writes then returns
  2). The runbook's quarantine wording (“kept, marked, never used”) is only met on the Part B side.
- **N6 — `addendum-report` exit code.** `mode_addendum_report` (`scripts/jepa_ext_repair.py:454-469`)
  always returns 0, unlike the other addendum modes' exit-2-on-incomplete convention; an incomplete
  D4 report (missing stages) exits 0 and only the payload records `not_run`.
- **N7 — evaluation run-dir default.** `_run_dirs:466-481` evaluates every directory under
  `<ext>/train/runs`; the execution should pass `--run` twice to the two authorised run dirs rather
  than rely on the scan, so a stale/archived run directory cannot be evaluated.
- **N8 — X2-f in the markdown.** `SUMMARY.json` carries the X2-f singular values
  (`_x2_seed_summary:5984`), but the `READINGS_SUMMARY.md` X2-f table lists only max singular value,
  condition number, median CRLB and the per-module CRLB (`:6280-6292`); addendum §5 asks both files
  to gain “the singular values of W Â_raw and the per-module CRLB”. Minor D4 completeness gap (item
  (vi) is already BLOCK for A1/A2).
- **N9 — figures count.** `_figure_names:4922-4925` lists seven figures (adding `rho_by_repair.png`,
  which the runbook §9.6 requires and the 7 Nov record omitted) while the addendum says “all six
  figures”; the extra figure is labelled and legended, so this is an improvement, not a defect.

## The 7 Nov payloads, the in-place update and the superseded table

- The D1 reruns write overlays only — `<ext>/addendum/d1_selfcheck.json`, `x2_addendum.json`,
  `x2e_addendum.json`, `x3_addendum.json`, `init9000/` — and `addendum_x2`'s docstring records that
  the 7 Nov `x2.json` stays intact for the superseded table (`ext_repair.py:5603-5606`); the 7 Nov
  `<ext>/x2/x2.json`, `x3/x3.json` are read, not overwritten. This satisfies “saved pre-fix payloads
  preserved”.
- `addendum_report` then updates the committed 7 Nov results in place as §10 requires
  (`SUMMARY.json`, `READINGS_SUMMARY.md`, `OUTCOME.json`, figures, the new
  `ADDENDUM_2026-11-07.json`; `:6505-6510`) and builds the superseded rows from the untouched 7 Nov
  payloads (`_superseded_rows:6087-6139`). The design cannot silently overwrite the pre-fix payloads
  — the design flaw is the opposite: A1/A2 mean the R-1 replacement cannot be propagated/listed.
- The Part B report updates its own results in place (`assemble_addendum_report:2846-3053`) and
  rewrites `readings.json["r1"]` with the rerun payload; combined with A1 (rerun not corrected) this
  would publish an R-1 “superseded” row whose new value equals the old value.

## Evidence (detached runs, `/work/dwong/jepa_tmp_20261107/va/` and `va_dt2/`)

Both hosts at SHA `923276af…`; `python3 -m pyflakes` clean; the three known `test_guard` launcher
deselections only; no GPU used by the reviewer.

- `va/summary.txt`: `SHA=923276afa6c48d79751818019d9a4a6612b8d4f4 HOST=deepthought`,
  `pyflakes_exit=0`, `focused_a_exit=0`, `focused_b_exit=0`, `full_suite_exit=1`,
  `dryrun_a_exit=0`, `dryrun_b_exit=0`, `DONE`. `va/pyflakes.log` is empty (0 lines).
- `va/focused_a.log`: `87 passed in 146.63s (0:02:26)`.
- `va/focused_b.log`: `43 passed in 16.42s`.
- `va/full_suite.log`: `1 failed, 798 passed, 2 skipped, 3 deselected in 1036.54s (0:17:16)`; the
  failure is `test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows`, the
  known pre-existing data/numerics-dependent N16 failure (same failure in the round-6 evidence on
  `0cf61ff`; `ext_followup.py` is not one of the six files), so the delta adds no new suite failure;
  the 3 deselections are the three known `test_guard` launcher tests.
- `va_dt2/summary.txt`: `SHA=923276af… HOST=deepthought2`, `full_suite_dt2_exit=0`, `DONE`;
  `va_dt2/full_suite_dt2.log`: `799 passed, 2 skipped, 3 deselected in 697.98s (0:11:37)`.
- `va/dryrun_a.log` + `va/repair_dryrun/dry_run.json` (Part A): exit 0, wall 8.923 s, gate
  `denied` with `gate_checks {a: true, a2: false, b: false, c: false}` and
  `addendum_gate {pending_acceptance: false, expected_addendum_sha256: null}`, addendum sha
  `abe0904a…` matched, all six addendum modes in `setup` with the addendum plan (sha, 0.35/3.0
  caps, projections, shared ledger), `closure.pass: true`, `violations: []`, `errors: []`; the only
  encode is `setup.sample` (`init9000` N, 8 events, CPU, `reused: uncached`).
- `va/dryrun_b.log` + `va/shortcut_dryrun/plan.json` (Part B): exit 0, wall 78.347 s, gate
  `denied` (`a`/`b`/`c` false, `addendum_sha256_pass: false`), `closure_pass: true`,
  `setup_errors: []`, `manipulation_pass: true`; preflight `data=True s1=True caps=True
  entries=False` (A4).
- Reviewer-only checks were read-only on `/ceph/dwong/jepa/code/jepa_work_20261107` with `TMPDIR`
  under `/work/dwong/jepa_tmp_20261107/review_add`; no GPU, no `/ceph` writes, no
  commits/checkouts/resets; the scratch was removed afterwards.

## Process / acceptance notes

- The addendum's §7 requirement of “one fresh-context focused review of the delta from `0cf61ff`” is
  this section; the D4 deliverables that are written by the report modes are reviewed as code and
  tests, not as executed results (nothing outcome-generating may run before the re-issued
  acceptance).
- Re-issue `docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json` with the new `impl_commit`
  and the two addendum fields after A1–A4 are dispositioned; both gates then require the committed
  addendum sha `abe0904a…0621a`, its §0 author line and the record's
  `addendum_path`/`addendum_sha256` (verified in code and in the pre-reissue dry-run denial above).
- The shared-contract decision (same ledger file/JSON rows and identical 0.35/3.0 caps; no shared
  helper module) is a legitimate reading of the “0.35 h guard” requirement, but only once the S1
  check is enforced on both sides (A3) and the entries precondition is cleared (A4).

## Final reviewed object (addendum round)

- Reviewed SHA `923276afa6c48d79751818019d9a4a6612b8d4f4`; delta `0cf61ff..923276a` over the six
  files; file sha256s and diff sha256 `4b4d3397…` above.
- Runbooks unchanged: combined `45e5ff98…d119b`, addendum `abe0904a…0621a`.
- Verdict: **BLOCK** (items (i)–(v), (vii) PASS; (vi), (viii) BLOCK; 4 blocking findings A1–A4,
  9 nonblocking notes N1–N9).

# Addendum review — fix round (A1-A4)

**Verdict: BLOCK.** The A1–A4 fixes are real and verified in code and tests, but two new blocking
findings remain: **A5** — the A1 fix is execution-order-dependent and `r1_payload` still silently
reads the D1-defective pilot `x2.json` when the binding addendum §8 order is followed; **A6** — the
live entries precondition fails (free 25,878 < 30,000, guard HOLD), so every addendum mode's
preflight refuses before work and nothing can be executed until entries are freed. The first BLOCK
report above (at `923276a`) is kept as history; this section supersedes its verdict for the fix
round at `ffe3390`.

## Reviewed object (fix delta)

- Worktree `/private/tmp/veritas-jepa-20261107`, branch `jepa-runbook-20261107`, HEAD
  `ffe3390c249f1094f9aa8abddd4726017438bd9a` = `origin/JEPA`. Host clone
  `/ceph/dwong/jepa/code/jepa_work_20261107` at the same SHA (tracked tree clean; only the
  execution's untracked 7 Nov `results/` payloads, none in `<P>`).
- Fix delta `923276a..ffe3390` over exactly the six source files, +1350/−71; docs unchanged in the
  fix commit (`git diff --name-only 923276a..ffe3390 | grep -v '^src/\|^scripts/'` is empty). No
  source outside the six changed.
- File sha256 (`shasum -a 256`, frozen blobs at `ffe3390`):
  - `src/jepa_noise/ext_repair.py` `9453f8c33f302b17d7ccf8b37c47c56226fd3c7c24f85cc8463e85fdd49f222d`
  - `src/jepa_noise/tests/test_ext_repair.py` `0c06e22d8d75fafbb339ea53736a8c4fa40d09f29f4f1dfb0147ae4a7ccf0a75`
  - `src/jepa_noise/ext_shortcut.py` `af39f965d4aec6f9acbd917af7d42bfb1a8b7d3674600059d34315026b505ed3`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `6f5f12f941bb7ebc598b3df2993ab267cbb6db01fe80fb2786d144d154abc47a`
  - `scripts/jepa_ext_repair.py` `f3dbd415e9ec048827342a6810399a9536619ffc5ec6e56b415fa9ae6f787c6d`
  - `scripts/jepa_ext_shortcut.py` `4c932b404dd0fcdf7eb907cdc8fb70b6c8aa1266bbdc1554f256777271c52e46`
- Diff sha256: `0cf61ff..ffe3390` over the six files
  `c982fa408389430d444e6581daa9921ee2beeaaf1eafc17b67e2d2bd01fc2b3d`; `923276a..ffe3390` over the
  six files `f26e2f4727c7eb9b8d94bf2669af72338e4263ec4f6a0de33251d7cc33a23db9`.
- Binding documents unchanged: combined runbook `45e5ff98…d119b`, addendum `abe0904a…0621a`.
- Acceptance record at review time is still the old 6-key record (`impl_commit` `0cf61ff`), so both
  dry runs correctly show the pre-reissue deny-not-pending gate (below).

## Per-finding disposition (A1–A4)

### A1 — FIXED in code (in-place corrected payloads + write-once preservation); order gap is A5

- `_preserve_7nov_payloads:5395-5436` copies `results/jepa/20261106_repair/{x1,x2,x3,readings}.json`
  write-once to `<ext_repair>/addendum/preserved_7nov/` with sha256 sidecars before any rewrite;
  `_preserved_7nov_payload:5439-5452` refuses a mismatched sidecar.
- `_corrected_x2_models:6644-6687` merges the D1 rerun into the pilot records (X2-c `rho_LR`/
  `class`, X2-d latent, X2-f, the raw-map `A_hat` into X2-b occupancy, X2-e latent) and
  `_corrected_results_payloads:6689-6738` rebuilds the x1/x2/x3/readings payloads; `addendum_report`
  writes them back into `results/jepa/20261106_repair/` in place after the originals are preserved
  (`:6883-6892`). The superseded table reads old values only from the preserved copies
  (`:6759-6788`).
- Demonstration (read-only host clone, TMPDIR under `/work/dwong/jepa_tmp_20261107/review_add2`,
  removed afterwards): 7 Nov pilot `x2.json` (`rho_lr` 0.9) → `r1_payload` row 0.9;
  `_corrected_results_payloads` gives `rho_lr = -0.42`; after writing that x2 into the pilot results
  → `r1_payload` row **−0.42**. So the corrected value does reach R-1 — but only after
  `addendum_report` has run (A5).
- Tests: `test_addendum_report_updates_results_in_place_and_preserves_7nov`
  (`test_ext_repair.py:2630-2683`; includes the second-run preservation check) and
  `test_superseded_r1_row_is_populated_from_part_b_payloads` (`:2685-2728`).

### A2 — FIXED

- `_r1_reading_from:6275-6298` accepts all documented shapes; `_part_b_r1:6300-6330` now reads
  `PART_B_EXT_ROOT/r1_addendum.json`, `results/jepa/20261107_shortcut/ADDENDUM_2026-11-07.json ->
  items.R1_rerun`, and a results `r1_addendum.json`; the old R-1 is preserved write-once
  (`<ext_repair>/addendum/preserved_7nov/r1_reading.json`, else `git show HEAD` of the committed 7
  Nov readings). Part B writes both hand-offs (`scripts/jepa_ext_shortcut.py:718`) via
  `results_r1_payload:788-808` / `write_results_r1:810-819`, and its report preserves the 7 Nov
  `readings.json` write-once (`ext_shortcut.py:3129-3138`, `preserve_7nov_readings:826-866`,
  `seven_nov_readings:869-880`).
- Demonstrations: `_part_b_r1` with the Part B ext hand-off returns old
  `generative_median.rho_lr = -0.67786`, new `-0.42`; same through the results ADDENDUM
  `items.R1_rerun`; after the results `readings.json` is rewritten with 999, the preserved old value
  is still returned; the D4 test now asserts the new value is populated (`new.verdict`), not just
  the row key (`test_ext_repair.py:2706-2728`). Part B tests:
  `test_results_r1_handoff_has_the_documented_stable_keys:1342-1359`,
  `test_addendum_report_preserves_the_7nov_readings_and_orders_the_handoff:1361-1418`,
  `test_addendum_r1_cli_writes_the_ext_and_stable_results_handoffs:1511-1590`.

### A3 — FIXED

- `addendum_s1_check:1444-1467` and `addendum_s1_check_or_stop:1469-1499` implement the addendum
  rule exactly (`ledger charged + 0.35 <= 160`; the closed Part A/Part B caps are not consulted).
  They run before any work in both Part A GPU modes — `addendum_x2e:5867-5868` and
  `addendum_init9000:6983-6984` — and are recorded on every path including `already_cached`
  (`:6983-6993`); the dry-run plan records `plan.addendum.s1` (below).
- Demonstration: ledger 129.7461 → `sum_hours` 130.0961, pass; ledger 159.7 → 160.05, stop. Tests:
  `test_ext_repair.py:2730-2770`; dry-run plan `s1.pass = true` (below). Part B keeps its own
  `addendum_s1_check(_or_stop)` (`ext_shortcut.py:747-779`, called before the evaluation
  `scripts/jepa_ext_shortcut.py:613`).

### A4 — FIXED in code (addendum-only basis); the live precondition fails, see A6

- `ADDENDUM_ENTRIES_PROJECTION = 200` (`ext_repair.py:143-149`), `addendum_entries_preflight:1503-1535`
  (free ≥ 30,000 and free − 200 ≥ 30,000; guard snapshot as the free-entry source) and
  `_or_stop:1537-1552`. It is called before work in every Part A addendum mode
  (`:5679, :5804, :5868, :6039, :6744, :6984`).
- Part B uses the same rule with a live-source chain: `_entry_xattr_reading:883-910` (live CephFS
  quota xattrs: `ceph.quota.max_files − ceph.dir.rfiles − ceph.dir.rsubdirs`),
  `addendum_entries_snapshot:912-928` (live first, guard fallback),
  `addendum_entries_preflight:930-968`, `assert_addendum_entries_preflight:970-980`; called in all
  three addendum modes and the dry run (`scripts/jepa_ext_shortcut.py:240, 616, 683, 743`).
- Demonstrations: free 35,770 → pass (projected 35,570); free 30,050 → fail; free 29,999 → fail.
  Tests: `test_ext_repair.py:2772-2846`, `test_ext_shortcut.py:1434-1509`. The combined-preflight
  entries block remains in the plan as descriptive history only (`ext_repair.py:4963-4971`).
  **The live values now fail: see A6.**

## Checklist re-assessment (addendum §7 (i)–(viii), fix round)

| # | Subject | Verdict | Evidence |
|---|---|---|---|
| (i) | `raw_map` correct; only route to Â | **PASS** | unchanged from the first report (`ext_repair.py:2902-2941`, tests `test_ext_repair.py:1745-1790`); the fix delta does not touch it |
| (ii) | every Â consumer converted; source/raw unchanged | **PASS** | unchanged consumers; the fix exposes the raw-unit `A_hat` into the corrected pilot X2-b (`_corrected_x2_models:6671-6673`) while source/raw remain untouched; test unchanged |
| (iii) | D1 self-check and S9 rule | **PASS** | unchanged (`ext_repair.py:5410-5539`); the fix adds the entries/S1 preflights around it |
| (iv) | D2 statistics in both modules | **PASS** | unchanged; the pre-existing X3-2-CI note (first report N3) remains |
| (v) | D3 folded in; adapter retired | **PASS** | unchanged (`_x3_five`, `_strip_internal`); the new post-write closure also marks quarantined payloads (`_addendum_closure:5540-5553`, `_finalise_addendum` `scripts/jepa_ext_repair.py:354-373`) |
| (vi) | D4 completeness, superseded table, `ADDENDUM_2026-11-07.json`, figures | **PASS (conditional on A5)** | the X2-f singular-value column is now in `READINGS_SUMMARY.md` (`ext_repair.py:6537-6552`, test `test_ext_repair.py:2848-2867`); the dead duplicated R2 block is gone (one `return rows`); the R-1 row is now populated when Part B's rerun is present (A2 demos/tests). Under the binding §8 order the R-1 *reading* is still stale (A5) |
| (vii) | R-2 evaluation path and `not_run` label | **PASS** | unchanged (`ext_shortcut.py:1318-1397`; tests `1057-1118`, `1255-1345`) |
| (viii) | gates, closure, 0.35 h guard, S1, entries, dry-run coverage | **BLOCK** | gates/closure/S1/coverage pass (dry runs below: gate deny-not-pending, closure pass, no setup errors, six addendum modes, plan `s1.pass=true`); **entries fails in the live evidence** (A6), so the §9 entries-preflight STOP is live |

## New blocking findings (fix round)

### A5 — the A1 fix is execution-order-dependent; under addendum §8 the R-1 rerun still silently reads the D1-defective payload

`r1_payload`/`pilot_status` were not changed by the fix (`ext_shortcut.py:2078-2115, 2117-2205`) and
still read the pilot's `results/jepa/20261106_repair/x2.json`. The corrected payloads are written
only by **Part A's `addendum_report`** (`ext_repair.py:6883-6892`), i.e. in the “both reports” step.
The binding addendum §8 order is “X2-e (latent) → Part B R-1 rerun → both reports”, so following
the spec R-1 runs *before* the file is corrected. Demonstrated on the read-only host clone:

```
7 Nov pilot x2 -> r1_payload row rho_lr        = 0.9
(no addendum_report run; `r1_payload` never reads the D1 overlay)  -> R-1 row still 0.9
after addendum_report in-place rewrite         -> R-1 row -0.42
```

There is no overlay read and no check that the pilot `x2.json` carries the `corrected_7nov` /
`addendum_sha256` markers that `_corrected_results_payloads` writes; the stale read is silent.
Consequently the fixed chain requires running Part A `addendum-report` **before** Part B
`addendum-r1` (and again after it, so the Part A superseded table's R-1 row is populated), which is
a deviation from §8 that the fix commit does not document, and a future executor following §8
reproduces the original A1 defect. Disposition: either make `r1_payload` order-independent (read
the corrected `X2-c.rho_LR` from `<ext_repair>/addendum/x2_addendum.json`, or STOP when the pilot
`x2.json` lacks the corrected marker), or commit the reordered/two-run execution to the execution
record (STATE/DECISIONS) and note the §8 deviation. No reading or threshold changes either way.

### A6 — the live entries precondition fails; the §9 entries-preflight STOP is active

The A4 code is correct and fails closed, but on the live system (both sources agree) the addendum
cannot start: **free CephFS entries = 25,878 < 30,000**. The guard snapshot is fresh
(`/ceph/dwong/jepa/jobs/guard_deepthought2.json`, timestamp `2026-11-07T23:38:33Z`, level `HOLD`,
reason “free entries 25878 < 30000; write no new data”); the Part B live CephFS xattrs give
`quota_max_files 500000 − rfiles 401305 − rsubdirs 72817 = 25,878`. Both modules' preflights return
`pass=false`, `projected_after=25,678 < floor 30000`:

- Part A dry-run plan `addendum.addendum_entries`: `pass=false`, free 25,878, projected 25,678,
  floor 30,000, snapshot guard `HOLD` (`vf/repair_dryrun/dry_run.json`).
- Part B dry-run summary: `addendum_entries_pass=false`, `addendum_entries_basis="live CephFS quota
  xattrs"`, `addendum_s1_pass=true` (`vf/dryrun_b.log:48-50`); plan snapshot live free 25,878
  (`vf/shortcut_dryrun/plan.json`).

Because every addendum mode calls `addendum_entries_preflight_or_stop` / `assert_addendum_entries_preflight`
before work, every addendum mode would `SystemExit(2)` (“not_run (entries)”) at execution. The
addendum §6 figure “44 286 at the last check” is long stale; the environment must be freed
(archive/cleanup on `/ceph`, as done before the 7 Nov execution) until free − 200 ≥ 30,000, and the
dry-run evidence re-run, before the addendum is executable. No code change is implied by A6.

## Nonblocking notes (fix round)

- **N1 — `_r1_reading_from` treats a `not_run` hand-off as a reading.** A results hand-off with
  `{"status": "not_run", "reading": null, "table": {}}` is returned as the “new” reading (the
  `table` key triggers the `any(key in payload ...)` branch at `ext_repair.py:6275-6298`);
  `_part_b_r1` then reports R-1 “superseded” with a not-run dict instead of `not_run` (reviewer
  demo: `new keys = ['reading','reason','status','table']`). Only reachable when Part B R-1 is
  not_run but writes its hand-off; the intended execution has a complete R-1.
- **N2 — entries-source asymmetry.** Part B prefers live CephFS quota xattrs and falls back to the
  guard snapshot (`ext_shortcut.py:912-968`); Part A uses the guard snapshot only
  (`addendum_entries_preflight:1503-1535` via `entries_snapshot`). At the review check both agreed
  (25,878, guard timestamp fresh), but Part A has no live-xattr fallback; worth aligning for parity.
- **N3 — results hand-off not declared.** `results/jepa/20261107_shortcut/r1_addendum.json` is now
  written by both `addendum-r1` and `addendum-report`, but it is not listed in the Part B ADDENDUM
  `outputs` (`ext_shortcut.py:3275-3276`) nor in addendum §10; declare/commit it so the stable
  hand-off survives.
- **N4 — quarantined hand-off.** `mode_addendum_r1` writes the stable results hand-off after
  `_closure_end` returns (`scripts/jepa_ext_shortcut.py:718`) even when the closure code is 2, so a
  quarantined R-1 value is still written to results and could be consumed by Part A. Write it only
  when the closure passes, or mark the hand-off quarantined.
- **N5 — large corrected payloads in `results/`.** `addendum_report` now writes corrected
  `x1.json`/`x2.json`/`x3.json`/`readings.json` into `results/jepa/20261106_repair/` (`:6883-6892`)
  as the Part B R-1 data bus; these are per-event payloads (large) and must not be committed. The
  committed record remains `SUMMARY.json`, `READINGS_SUMMARY.md`, `OUTCOME.json`, the figures and
  `ADDENDUM_2026-11-07.json`; keep the corrected payloads untracked (they are written on /ceph).
- **N6 — two-report-run requirement.** Following A5, the execution must run Part A
  `addendum-report` twice (before and after the Part B R-1 rerun); the preserved copies are
  write-once and the second run is covered by `test_addendum_report_updates_results_in_place_and_preserves_7nov`.
- **N7 — X3-2 CI still absent** (pre-existing at `0cf61ff`; unchanged by the fix; first-report note
  N3 still stands).
- **N8 — nonblocking fixes verified.** Dead duplicated R2 block removed (one `return rows`,
  `ext_shortcut.py:2941-2989`; test `test_ext_shortcut.py:1502-1509`); X2-f singular values in the
  markdown (`:6537-6552`); Part A closure is now post-write/quarantined (`_finalise_addendum`,
  `scripts/jepa_ext_repair.py:354-373`, test `test_ext_repair.py:2869-2895`).

## Evidence (vf/ on deepthought, vf_dt2/ on deepthought2; SHA `ffe3390`)

- `vf/summary.txt`: `SHA=ffe3390c249f1094f9aa8abddd4726017438bd9a HOST=deepthought`,
  `pyflakes_exit=0`, `focused_a_exit=0`, `focused_b_exit=0`, `full_suite_exit=1`,
  `dryrun_a_exit=0`, `dryrun_b_exit=0`, `DONE`. `vf/pyflakes.log` empty (0 lines).
- `vf/focused_a.log`: `96 passed in 142.84s (0:02:22)`; `vf/focused_b.log`:
  `52 passed in 18.69s` (+18 tests vs the `923276a` round, consistent with the full-suite count).
- `vf/full_suite.log`: `1 failed, 816 passed, 2 skipped, 3 deselected in 1069.73s (0:17:49)`; the
  failure is `src/jepa_noise/tests/test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows`,
  the same known pre-existing data/numerics N16 failure recorded for `0cf61ff` and `923276a`
  (`ext_followup.py` is not one of the six files), not a fix-delta failure; the 3 deselections are
  the three known `test_guard` launcher tests.
- `vf_dt2/summary.txt`: `SHA=ffe3390… HOST=deepthought2`, `full_suite_dt2_exit=0`, `DONE`;
  `vf_dt2/full_suite_dt2.log`: `817 passed, 2 skipped, 3 deselected in 702.36s (0:11:42)`.
- `vf/dryrun_a.log` + `vf/repair_dryrun/dry_run.json` (Part A): exit 0, wall 9.172 s, gate
  `denied` with `gate_checks {a: true, a2: false, b: false, c: false}`, addendum gate
  `pending_acceptance: false`, `expected_addendum_sha256: null` (correct pre-reissue deny);
  `addendum_modes`: the six; addendum plan `s1.pass = true` (`sum_hours 130.0961`) and
  `addendum_entries.pass = false` (free 25,878, projected 25,678, floor 30,000, guard `HOLD`);
  `closure.pass = true`, `violations []`, `errors []`; the only encode is the pre-existing CPU
  §2.5 sample (8 events).
- `vf/dryrun_b.log` + `vf/shortcut_dryrun/plan.json` (Part B): exit 0, wall 76.414 s, gate
  `denied`, `closure_pass: true`, `setup_errors: []`, `manipulation_pass: true`,
  `addendum_entries_pass: false`, `addendum_entries_basis: "live CephFS quota xattrs"`,
  `addendum_s1_pass: true`; plan `addendum.entries_preflight.snapshot.live` gives
  `quota_max_files 500000`, `rfiles 401305`, `rsubdirs 72817`, free 25,878. (The first log line
  `preflight failed: … entries=False` is the old combined-preflight block, descriptive history
  only; the addendum STOP basis is the `addendum_entries` block above.)
- Reviewer-only checks were read-only on `/ceph/dwong/jepa/code/jepa_work_20261107` with `TMPDIR`
  under `/work/dwong/jepa_tmp_20261107/review_add2` (removed afterwards); no GPU, no `/ceph` writes,
  no commits/checkouts/resets. Live re-check at 00:38 UTC: Part A 25,878 (guard) and Part B 25,878
  (live xattrs) both `pass=false`.

## Final reviewed object (fix round)

- Reviewed SHA `ffe3390c249f1094f9aa8abddd4726017438bd9a` (= `origin/JEPA`); first BLOCK report at
  `923276a` preserved above. File sha256s and diff sha256s above.
- Runbooks unchanged: combined `45e5ff98…d119b`, addendum `abe0904a…0621a`.
- Verdict: **BLOCK** — A1–A4 fixed in code (A4’s live precondition failing is A6); items (i)–(v) and
  (vii) PASS, (vi) PASS conditional on A5, (viii) BLOCK on A6; 2 new blocking findings A5–A6 and
  8 nonblocking notes N1–N8.

# Addendum review — final (A5, A6)

**Verdict: PASS.** A5 is fixed by an order-independent R-1 source chain (the D1 overlay is read
first, so the §8 order “X2-e → Part B R-1 rerun → both reports” now yields the corrected ρ_LR), and
A6 is resolved operationally (live free entries 41.5k ≥ 30,000, guard OK; the archive manifest is
verified). All eight addendum checklist items PASS; no new blocking findings, four nonblocking notes
(N1–N4) plus carried legacy notes. The first two BLOCK reports above (`923276a`, `ffe3390`) are kept
as history; this section is the final review of the addendum as delivered at `90f190a`.

## Reviewed object (final)

- Worktree `/private/tmp/veritas-jepa-20261107`, branch `jepa-runbook-20261107`, HEAD
  `90f190a8090661236383dc5fb9168c307814ac0b` = `origin/JEPA`. Host clone
  `/ceph/dwong/jepa/code/jepa_work_20261107` at the same SHA (tracked tree clean; only the
  execution's untracked 7 Nov `results/` payloads).
- Final delta `ffe3390..90f190a`: three Part B source files (`src/jepa_noise/ext_shortcut.py`,
  `src/jepa_noise/tests/test_ext_shortcut.py`, `scripts/jepa_ext_shortcut.py`) plus
  `docs/jepa/DECISIONS.md` and the new `docs/jepa/ARCHIVE_2026-11-08_MANIFEST.jsonl`; no other source
  changed; the binding runbooks are unchanged (`45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b`,
  `abe0904af0ed0b5a6586cee459c507ae0fa2a7c60c65ffb58192c71b7180621a`).
- File sha256 (`shasum -a 256`, frozen blobs at `90f190a`):
  - `src/jepa_noise/ext_repair.py` `9453f8c33f302b17d7ccf8b37c47c56226fd3c7c24f85cc8463e85fdd49f222d`
  - `src/jepa_noise/tests/test_ext_repair.py` `0c06e22d8d75fafbb339ea53736a8c4fa40d09f29f4f1dfb0147ae4a7ccf0a75`
  - `src/jepa_noise/ext_shortcut.py` `11fe739a6a14f06ac8bb26431ce3acf11dbf505b46d37421cd6434e6d518f2f2`
  - `src/jepa_noise/tests/test_ext_shortcut.py` `1d6f8ba82f6ef8f861b242c821557d068b61382844e27e015c94ae09b548da9a`
  - `scripts/jepa_ext_repair.py` `f3dbd415e9ec048827342a6810399a9536619ffc5ec6e56b415fa9ae6f787c6d`
  - `scripts/jepa_ext_shortcut.py` `294fe153413653437a3cb2d1b94b48036d2d79beb036e2296de3a005c8960ff2`
  - `docs/jepa/ARCHIVE_2026-11-08_MANIFEST.jsonl` `91cd1e22a6097831d67fa00cef2c3cf8ff2fbb09115dd15475d37ac4903e2eee`
- Diff sha256: six-source-file diff `0cf61ff..90f190a` `430bacf6bf00daa396904d1897a9157f3922d83cf47eb68c6f7dea91a59d7667`;
  six-source-file diff `ffe3390..90f190a` `57cfde550ac47754a15bcf4fc7c009317d65e41cc72f510fa68f2d6a5b86eff2`.
  Whole-tree diffs: `0cf61ff..90f190a` `9c2bd391f92ef771a6d3a294d2fcfee591aec6d5b3ff04afe6eaf8cfdb27be68`;
  `ffe3390..90f190a` `46403406fdd80d1e92ab8e5d9a8d01c21ce0237f499bc5f8a5a3593f7cf70c8c`.

## A5 disposition — PASS

- The order-independent source chain is `resolve_r1_x2_sources:2401-2479`: (1) the Part A D1 overlay
  `<PILOT_EXT_ROOT>/addendum/x2_addendum.json` (`x2_overlay_path:2339-2343`) with per-model
  `X2-c.rho_LR`/class (`overlay_corrections:2372-2391`, tolerant label lookup
  `_lookup_model_block:2393-2398`); (2) the pilot `results/jepa/20261106_repair/x2.json` only when it
  carries both Part A markers — `corrected_7nov` naming `X2-c` and `addendum_sha256 ==
  abe0904a…0621a` (`corrected_results_marker:2345-2370`); (3) otherwise the 7 Nov payload. The basis
  is recorded in `source`/`readings_basis` ∈ {`d1_overlay`, `corrected_results`, `7nov_results`} plus
  the `x2_source` detail (overlay presence/completeness, marker status, `model_sources`,
  `fallbacks`, and the `source_order` in the reading). `pilot_status:2504-2550` makes the same
  decision as the reading (`r1_payload:2552-2585`), and the Part B report records the basis
  (`:3247-3255`, `:3387`, `:3473-3476`). The R-1 rule, families and thresholds are unchanged;
  per-model fallbacks are recorded and no value is invented.
- Demonstration under the binding §8 order (host clone, read-only; TMPDIR
  `/work/dwong/jepa_tmp_20261107/review_add3`, removed afterwards). With the pilot results `x2.json`
  still the stale 7 Nov payload (`rho_lr 0.9`, no markers) and **no Part A `addendum-report` run**,
  only the D1 overlay present:

```
1) overlay-only: rho_lr = -0.42 | readings_basis = d1_overlay | source = d1_overlay
   x2_source overlay present: True | model source: d1_overlay
2) no overlay:               rho_lr = 0.9   | basis = 7nov_results
3) corrected-results marker: rho_lr = -0.42 | basis = corrected_results
4) wrong-sha marker:         basis = 7nov_results | marker.corrected = False
5) per-model fallback:       overlay model -0.42 | fallback model 0.9 | basis d1_overlay |
                             fallbacks recorded
6) not_run hand-off:         table absent | Part A `_r1_reading_from` -> None
```

  The first line is the §8-order case: R-1 now reads the D1 overlay before the Part A report, so the
  original A5 defect (0.9 silently used) is gone. The test
  `test_r1_reads_the_d1_overlay_when_the_results_x2_is_stale:1622-1648` also covers the
  `frozen_target9000` overlay row (−0.12); the other tests are `:1650-1663` (7 Nov basis),
  `:1665-1685` (corrected-results marker incl. wrong-sha not trusted), `:1687-1716` (per-model
  fallback), `:1718-1745` (not_run hand-off sanitised; `sanitise_r1_handoff:2481-2502`, applied in
  `results_r1_payload:805-819` and `scripts/jepa_ext_shortcut.py:696`).

## A6 disposition — PASS (operational)

- No code change was needed; the addendum entries basis remains the addendum's own projected writes
  (200 entries; floor 30,000; `ext_shortcut.py:235-240`, `:930-980`; `ext_repair.py:143-149`,
  `:1503-1552`).
- Live check at review time: **Part A free 41,556** with guard `OK`, timestamp
  `2026-11-07T23:56:36Z`, no reasons; **Part B free 41,557 from live CephFS quota xattrs**
  (`ceph.quota.max_files − ceph.dir.rfiles − ceph.dir.rsubdirs`); projected after the 200-entry
  addendum writes 41,356 / 41,357 ≥ 30,000, so `pass=true` in both modules; addendum S1
  `130.0961 ≤ 160` pass.
- The operational relief is recorded and verifiable: `docs/jepa/ARCHIVE_2026-11-08_MANIFEST.jsonl`
  has 7 rows (2 verified `tar.gz` result archives and 5 `git bundle` checkouts) under
  `/work/dwong/jepa_archive_20261108/`. I re-verified all seven sha256 values against the manifest
  (all OK), the two tars' file/member counts (6 and 7 members), and `git bundle verify` on
  `checkout_02283a6.bundle` (“The bundle records a complete history”); the five
  `/ceph/dwong/jepa/code/{02283a6,f5b7434,444cf58,eafbce6,080fb58}` scratch checkouts are absent.
  `DECISIONS.md:2500-2501` records the entries basis and the cleanup; the runbook-listed data,
  checkpoints and committed results were not touched (manifest reasons). Free entries
  25,878 → 41.5k, guard OK.
- The `vg` dry runs confirm the addendum entries preflight now passes (below), so the §9 entries
  STOP is clear and the addendum is executable under §8.

## Checklist re-assessment (addendum §7 (i)–(viii), final)

| # | Subject | Verdict | Evidence |
|---|---|---|---|
| (i) | `raw_map` correct; only route to Â | **PASS** | unchanged (`ext_repair.py:2902-2941`; tests `test_ext_repair.py:1745-1790`) |
| (ii) | every Â consumer converted; source/raw unchanged | **PASS** | unchanged consumers (A1/A2 fix exposes the raw-unit `A_hat` into the corrected pilot payloads); source/raw untouched |
| (iii) | D1 self-check and S9 rule | **PASS** | unchanged (`ext_repair.py:5410-5539`); entries/S1 preflights around it |
| (iv) | D2 statistics in both modules | **PASS** | unchanged; the pre-existing X3-2-CI note remains (nonblocking, outside the addendum D1–D5 defect list) |
| (v) | D3 folded in; adapter retired | **PASS** | unchanged; retirement recorded in the fix-round code and now in `DECISIONS.md` |
| (vi) | D4 completeness, superseded table, `ADDENDUM_2026-11-07.json`, figures | **PASS** | A1/A2 verified in the fix round; the R-1 row is populated under the §8 order (A5); X2-f singular values in the markdown; figures labelled/legended |
| (vii) | R-2 evaluation path and `not_run` label | **PASS** | unchanged (`ext_shortcut.py:1318-1397`; tests `1057-1118`, `1255-1345`) |
| (viii) | gates, closure, 0.35 h guard, S1, entries, dry-run coverage | **PASS** | gates deny-not-pending pre-reissue; closure pass; S1 pass in both modules (130.0961); entries pass in both modules (41.5k); the dry runs exercise all six Part A addendum setup paths and do not encode/evaluate (only the pre-existing CPU §2.5 sample) |

## Nonblocking notes (final)

- **N1 — the “7 Nov payload” fallback is a basis label, not a distinct value source.** With no
  overlay and an invalid marker, `resolve_r1_x2_sources` still uses the pilot `x2.json` values (there
  is no read of Part A's preserved `<ext_repair>/addendum/preserved_7nov/x2.json`); the demo's
  wrong-sha case reports basis `7nov_results` while the row still carries the file's value (−0.42).
  In the real flow the marker is written by `addendum_report` with the correct sha, so this is a
  corruption-tolerance edge, not an expected state; a future hardening could read the preserved copy
  or STOP on a mismatched marker.
- **N2 — the D1 overlay is trusted without an integrity check.** Unlike the embedding caches (sha256
  sidecars) and the results marker (sha constant), `<PILOT_EXT_ROOT>/addendum/x2_addendum.json` is
  read as-is; the overlay payload carries `addendum_sha256` but the resolver does not verify it. A
  stale/corrupt overlay would silently change R-1; consider verifying it against the addendum sha on
  the next round.
- **N3 — metadata asymmetry.** `r1_payload` records `source_order` at top level; `pilot_status`'s
  `x2_source` does not include `source_order` (it has overlay/marker/model_sources/fallbacks). The
  reported order is available in the R-1 payload and the README, so this is cosmetic.
- **N4 — present-but-empty overlay basis.** A present overlay with no usable correction yields a
  `model_sources` fallback for every model but the overall `readings_basis` remains `d1_overlay`
  (the per-model fallbacks are recorded). Only relevant to a broken overlay; the values are correct
  (7 Nov) and the fallbacks are explicit.
- **Carried legacy notes** (unchanged from the fix round): the Part B results hand-off
  `results/jepa/20261107_shortcut/r1_addendum.json` is not itemised in the ADDENDUM `outputs` /
  §10 deliverables (commit it with the results); `addendum-r1` writes the results hand-off after a
  closure violation could still expose a quarantined value (low risk); the large corrected
  `x1/x2/x3` payloads written into `results/jepa/20261106_repair/` must stay untracked (the committed
  record is SUMMARY/READINGS/OUTCOME/ADDENDUM/figures); Part A's entries preflight uses the guard
  snapshot only while Part B prefers live xattrs (both agreed at 41.5k); X3-2's CI remains absent
  (pre-existing at `0cf61ff`, outside the addendum D1–D5 defects). The shared-contract arbitration
  (call-shape parity only; identical ledger JSON rows and 0.35/3.0 caps) is recorded in
  `DECISIONS.md:2500` and remains a legitimate reading once S1/entries are enforced on both sides,
  which they are.

## Evidence (vg/ on deepthought, vg_dt2/ on deepthought2; SHA `90f190a`)

- `vg/summary.txt`: `SHA=90f190a8090661236383dc5fb9168c307814ac0b HOST=deepthought`,
  `pyflakes_exit=0`, `focused_a_exit=0`, `focused_b_exit=0`, `full_suite_exit=1`,
  `dryrun_a_exit=0`, `dryrun_b_exit=0`, `DONE`. `vg/pyflakes.log` empty (0 lines).
- `vg/focused_a.log`: `96 passed in 156.51s (0:02:36)`; `vg/focused_b.log`:
  `57 passed in 18.13s` (+5 A5 tests; both expected counts match the lead).
- `vg/full_suite.log`: `1 failed, 821 passed, 2 skipped, 3 deselected in 1023.54s (0:17:03)`; the
  failure is `src/jepa_noise/tests/test_ext_followup.py::test_label_collector_reproduces_the_committed_fix9000_rows`,
  the same known pre-existing data/numerics N16 failure recorded at `0cf61ff`, `923276a` and
  `ffe3390` (`ext_followup.py` is untouched by the addendum), not a fix-delta failure; the three
  deselections are the known `test_guard` launcher tests.
- `vg_dt2/summary.txt`: `SHA=90f190a… HOST=deepthought2`, `full_suite_dt2_exit=0`, `DONE`;
  `vg_dt2/full_suite_dt2.log`: `822 passed, 2 skipped, 3 deselected in 706.88s (0:11:46)`.
- `vg/dryrun_a.log` + `vg/repair_dryrun/dry_run.json` (Part A): exit 0, gate `denied` with
  `gate_checks {a: true, a2: false, b: false, c: false}`, `addendum_gate.pending_acceptance: false`,
  `expected_addendum_sha256: null` (correct pre-reissue deny); all six addendum modes in `setup`;
  addendum plan `s1.pass = true` (`sum_hours 130.0961`), `addendum_entries.pass = true`
  (`free_entries 41,522`, `projected_after 41,322`, floor 30,000); `closure.pass = true`,
  `violations []`, `errors []`; the only encode is the pre-existing CPU §2.5 sample (8 events).
- `vg/dryrun_b.log` + `vg/shortcut_dryrun/plan.json` (Part B): exit 0, wall 78.657 s, gate
  `denied`, `closure_pass: true`, `setup_errors: []`, `manipulation_pass: true`,
  `addendum_entries_pass: true` (`free_before 41,517`, `projected_after 41,317`,
  `projected_writes 200`, basis “live CephFS quota xattrs”), `addendum_s1_pass: true` (130.0961).
  (The first log line `preflight failed: … entries=False` is the old combined-preflight block,
  history only; the addendum STOP basis is the entries block above, which passes.)
- Reviewer checks were read-only on `/ceph/dwong/jepa/code/jepa_work_20261107` with `TMPDIR` under
  `/work/dwong/jepa_tmp_20261107/review_add3` (removed afterwards); no GPU, no `/ceph` writes, no
  commits/checkouts/resets. Live re-check at 01:0x UTC: Part A 41,556 (guard OK) and Part B 41,557
  (live xattrs), both `pass=true`; archive manifest 7/7 sha256 verified and the scratch checkouts
  removed.

## Final reviewed object (final)

- Reviewed SHA `90f190a8090661236383dc5fb9168c307814ac0b` (= `origin/JEPA`); previous BLOCK reports at
  `923276a` and `ffe3390` preserved above. File sha256s and diff sha256s above.
- Binding runbooks unchanged; final delta is Part B source/tests/script plus the DECISIONS/archive
  manifest docs.
- Verdict: **PASS** — A1–A5 fixed and A6 resolved operationally; all addendum checklist items
  (i)–(viii) PASS; 0 blocking findings, 4 nonblocking notes N1–N4 plus the carried legacy notes. The
  acceptance record can be re-issued with `impl_commit = 90f190a`, `addendum_path =
  docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md` and `addendum_sha256 = abe0904a…0621a`; the
  addendum §8 execution may then start.
