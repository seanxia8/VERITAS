# WP4b §3.5.3a detectability check — implementation and frozen-decision procedure

Owner: WP4b implementer. Scope: `src/jepa_noise/lucid_detectability.py`,
`src/jepa_noise/tests/test_lucid_detectability.py`,
`src/jepa_noise/configs/lucid_detectability_v0.json`, this file. Binding:
`docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md` §3.5.3, §3.5.3a, §3.5.4 and §3.5.6.

**Status: development-only, `readout_synthetic_unrepaired`.** The module runs on the real
WCTE geometry (host CPU, geometry read from LUCiD's `config/WCTE_geom_config.json` / `.npz`)
but the local numbers below are from the **synthetic schema fixture** (`NOT WCTE`) and are
not reportable. The host run must be executed once and its JSON frozen before training;
until then every value here is a placeholder or a smoke-test illustration.

## The registered text (quoted verbatim, runbook v4 §3.5.3a)

> **3.5.3a Detectability check [A8]** (before any training, on development data; re-run on the
> merged readout).
> - For each factor and setting (F1 at each s, F3, F2 or F2′, law T), compute the **Fisher information**
>   for the factor from the generator's true expected charges: Poisson statistics inflated by
>   (1 + σ_spe²), with the vertex, the yield, all other module gains and the other factors treated as
>   unknown (profiled).
> - Report two ceilings, with σ_f the factor's spread: per event, R²_max = σ_f² / (σ_f² + 1/I_event);
>   per run, the same with I_run = the sum of I_event over a run's 50 events. Both are lower bounds:
>   profiling ignores the known yield prior, which run-mean probes can exploit.
> - Every factor used by a registered criterion needs a **run-level ceiling ≥ 0.6** at the median
>   yield. If it falls short, adjust in this order, recording each step in `DECISIONS.md`:
>   1. raise Y_hi to its bound (b) above;
>   2. F2: the next a in {0.10, 0.20, 0.50}, then switch to F2′ (b = 0.05, then 0.10);
>   3. F3: s_p up to 0.10.
> - F1 cells whose ceiling is still below 0.6 stay in the grid, because they show the trap's lower
>   limit, but they are excluded from the R6 and R7 criteria. If no F1 cell reaches 0.6, or F2/F2′
>   cannot, write G6 for the dependent predictions. If F3 cannot, R12 is `not_evaluable`; no gate.
> - Freeze the ceilings in `DECISIONS.md` and report them beside every detector-state R².

The yield bound (b) from §3.5.3 is the lower of: (a) the yield at which the median module
charge reaches 300 PE and (b) the yield at which the median hit fraction reaches 0.95. Y_lo is
the yield at which the median is about 200 hit PMTs; N_min = 20 (§3.5.4).

## Model and formulas

**True expected charge (generator, before threshold).** For every event and every PMT,

```
mu_p = Y * QE(lambda_p) * CE * Omega_p/(4 pi) * exp(-d_p/lambda) * gain_m(p) * gain_p * grad_p
```

with `Y` the event yield, `QE` the configured per-PMT quantum efficiency, `CE` the collection
efficiency, `Omega_p/(4 pi)` the generator's disc solid angle times obliquity, `d_p` the
vertex-sensor distance, `lambda` the run's attenuation length, and the state gains the product
of the F1, F3 and (when present) F2′ factors. This is exactly
`lucid_bombs.expected_charge(...) * state.gains`; **the fixed-PE threshold is not applied to
mu** and dark counts are not added. The threshold censors the readout (it removes low-charge
hits and turns a gain change into a hit/unhit switch), so the continuous-charge Fisher used
here is optimistic at low `mu`; this partly offsets the downward bias of profiling.

**Fisher information.** With log-derivative columns `L_{p,a} = d ln mu_p / d theta_a` and
weights

```
w_p = mu_p^2 / (mu_p (1 + sigma_spe^2) + v_dark),     v_dark = 0
I_ab = sum_p w_p L_pa L_pb,
```

the profiled (effective) information for a scalar signal direction `c` is

```
I_eff(c) = 1 / (c^T I^+ c) = I_ff - I_fv I_vv^+ I_vf   (single coordinate f),
```

where `I^+` is the Moore-Penrose pseudo-inverse and `v` is the nuisance block. For F1 and F3
the signal is a zero-sum **contrast** (module pattern relative to the mean; per-PMT pattern
within its module), because the uniform gain mode is exactly degenerate with the yield. For
F2/F2′ the signal is the N(0,1) variable `v`; for law T it is the in-range age `t = r/R`.

**Ceilings.** With the factor's spread `sigma_f`:

```
R2_event = sigma_f^2 / (sigma_f^2 + 1/I_event)
R2_run   = sigma_f^2 / (sigma_f^2 + 1/I_run),    I_run = sum over the run's 50 events of I_event.
```

For a vector factor the reported value is the median over its coordinates (modules for F1,
PMTs for F3) after the zero-sum-contrast correction: the contrast variance is
`sigma_f^2 * n/(n-1)` and the unidentifiable uniform mode contributes `1/n` of the target
variance, so the F1/F3 rows multiply by `(1 - 1/n)` and use the contrast variance. The effect
is below 1 % for F1 and below 0.1 % for F3; the F3 within-module contrast uses
`sigma_c^2 = s_p^2 * 19/18` and `(1 - 105/1995)` for the mode absorbed by F1's module gains.

**Nuisance policy (the profiling approximation).**

| Factor | Profiled nuisances | Signal |
|---|---|---|
| F1 | vertex (3), yield (1), other module gains (104) | one module-gain zero-sum contrast per module |
| F3 | vertex (3), yield (1), F1 module gains (105) | within-module per-PMT contrast (one representative PMT per module) |
| F2 | vertex (3), yield (1), F1 module gains (105) | `v` |
| F2′ | vertex (3), yield (1), F1 module gains (105) | `v` |
| T | vertex (3), yield (1) | `t = r/R` at the in-range median 1/2 |

**Direction of bias.** Profiling treats the yield, vertex and module-gain nuisances as free
per event (no prior, no run-level sharing). Ignoring the known yield/module-gain priors can
only *remove* information, so both ceilings are **lower bounds** on what a run-mean probe can
achieve; this is the runbook's stated direction. Two approximations go the other way and are
acknowledged: no threshold censoring and no dark counts (optimistic), and the vertex cusp
`|cos theta|` contributes a large derivative for near-perpendicular PMTs (the generator's
model; it is integrable and not truncated). The F3 per-PMT gains are part of the true `mu` but
are **not** profiled: one free gain per PMT is an incidental-parameter block that spans every
per-PMT direction and would make every other factor's profiled information identically zero.
Profiling F1 module gains for F2/F2′ is the runbook's explicit joint profiling; it absorbs the
smooth module-mean part of the attenuation/gradient and leaves only the within-module
variation, which is the conservative and intended reading.

**F3 details.** The F3 signal is the within-module contrast `+1` on one representative PMT and
`-1/(19-1)` on the other 18 PMTs of the module; the median over the 105 modules is reported.
One representative PMT per module (the first in the module) is used to keep the Fisher matrix
at 214 columns; the per-PMT spread is recorded in the JSON through `r2_run_min/max`. This is
an engineering choice; the full 19-contrast-per-module variant is a possible refinement.

## Yield range and occupancy (§3.5.3/§3.5.4)

- **Y_lo**: bisection on the median hit PMT count at a fixed yield, target 200. The occupancy
  is evaluated with the nominal state and the generator readout on `bound_events` events.
- **Y_hi**: bisection on the median mean-module charge (target 300 PE) and the median hit
  fraction (target 0.95); the lower yield wins, and the source is recorded. "Median module
  charge" is the per-event mean over modules, then the median over events.
- **Yield draws** use a fixed Poisson capacity (4 × the config Y_hi) so the same Poisson
  uniforms, vertices and QE are used at every yield; the objective is therefore smooth and
  the bounds are reproducible.
- **Median yield for the ceilings**: the population median of the registered log-uniform law,
  `sqrt(Y_lo * Y_hi)`, not the realised median of a finite cohort. The chosen policy is
  recorded as `population_median_of_log_uniform` in the JSON. (The realised median of a
  50-event run fluctuates; the population median is deterministic and is what "at the median
  yield" means here.)
- **N_min**: at Y_lo, the within target (odd `pmt_id_in_mpmt`) and an across target proxy
  (random half of modules) must both have ≥ 20 hits. The real across view is a contiguous
  Hilbert half, but `view_masks` is still a documented WP4c stub; the proxy is labelled. The
  across redraw estimate after 3 redraws is `p^4` for an initial below-N_min rate `p`.

## Acceptance and fallback logic (implemented verbatim)

The code evaluates the table at the **initial** Y_hi from the config, then:

1. If any factor is below 0.6, set Y_hi to the computed bound (recording `from`/`to` and
   `changed`; the bound can be above or below the placeholder) and re-evaluate.
2. If F2 is below 0.6, try the next `a` in `{0.20, 0.50}` (the initial `a = 0.10` is the first
   attempt); if none reaches 0.6, switch to F2′ with `b = 0.05`, then `0.10`. Every attempted
   setting is recorded, including failures.
3. If F3 is below 0.6, raise `s_p` from 0.05 to 0.10 and re-evaluate.
4. F1 cells below 0.6 remain in the grid but are excluded from R6/R7; if **no** F1 cell
   reaches 0.6, the branches write **G6** for R6/R7 (and R11b, which depends on R11). If
   F2/F2′ cannot reach 0.6, G6 is written for R11/R11b. If F3 cannot, **R12 is
   `not_evaluable`, no gate**.
5. The chosen `a`/`b`, F2 vs F2′, the eligible F1 cells, the excluded F1 cells, the fallback
   steps and the branch reason are frozen in the decision JSON and in the `DECISIONS.md`
   block.

The code can take all of these branches; the local synthetic smoke test (below) takes
`F1 = {0.05, 0.10}`, `F2 a = 0.50`, `F3 not_evaluable` (R12 not_evaluable), no G6. The host
numbers decide the real branch.

## Host command (real WCTE geometry, CPU)

The pinned LUCiD clone is on the host at `/ceph/dwong/jepa/external/LUCiD`; the module reads
the geometry from the npz path and never imports LUCiD. Run from the JEPA checkout that
contains this code, read-only on LUCiD:

```sh
ssh deepthought
cd /ceph/dwong/jepa/code/<JEPA_SHA>
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<host Python 3.11 with numpy from runbook §6.1>
sha256sum /ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json
# must print 9943d6028e02bc7534a809b70e37893fedd1cd6f5e258babcbacc12613d5ed5c
$PY -m jepa_noise.lucid_detectability run \
  --geometry /ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json \
  --config src/jepa_noise/configs/lucid_detectability_v0.json \
  --generator-config src/jepa_noise/configs/lucid_bombs_v0.json \
  --seed 20261017 \
  --out /ceph/dwong/jepa/jobs/wp4b_detectability_wcte.json
```

The loader verifies the pointer sha256 (named `WCTE_geom_config.json`) and the npz prefix
`0a75b035`. The command computes Y_lo, Y_hi, the N_min check and the full decision, writes the
schema-versioned JSON and prints the `DECISIONS.md`-ready block. A synthetic geometry is
refused unless `--allow-synthetic` is passed, and it is never frozen.

**Expected runtime and memory (real geometry, single-threaded BLAS).** The 1000-event
generator benchmark is 202–207 events/s at 270–274 MiB (`WP4B_GENERATOR.md`). This check does
`3 bound searches x ~10 bisections x 48 events` plus the N_min check: about 1.5–2.5 k readout
events, plus 50 events x ~6–10 table evaluations of pure `mu` and 110–214 dimensional Fisher
matrices. Expected **20–60 s wall time** and **< 500 MiB peak RSS** on the host (the
synthetic-fixture smoke run is 45–60 s on the Mac; the bound bisection dominates and scales
with `bound_events * bound_iterations`). No GPU, no LUCiD import, no benchmark of the
generator itself (that is separate).

**Re-benchmark input.** The decision JSON's `rebenchmark` block carries the frozen
`[Y_lo, Y_hi]`, the median yield, the expected volume `1 070 000` events and the breakdown
(main-grid 3×100k, secondary 5×100k, dev 100k, probe/robustness 17×10k), the 16-core cap and
the 48 h generation cap. Re-run `lucid_bombs.benchmark_generation` at these yields; the
generation cost is dominated by the per-event Poisson capacity, so the frozen Y_hi must be
used. This check does **not** run the host benchmark.

## Decision table template (freeze these values in `DECISIONS.md`)

The JSON's `decisions_md` block is ready to copy. Its table is:

| factor | setting | R²_run (lower bound) | pass |
|---|---|---|---|
| F1 | 0.02 | … | … |
| F1 | 0.05 | … | … |
| F1 | 0.10 | … | … |
| F3 | 0.05 or 0.10 | … | … |
| F2 or F2′ | chosen a/b | … | … |
| T | 0.5 | … | … |

plus the frozen lines: geometry digest and synthetic flag, readout label, code SHA,
`Y_lo`, `Y_hi` and its bound source, median yield and policy, N_min occupancy, fallback steps,
F1 eligible/excluded, F2/F2′ choice and attempts, F3 choice, R12 status, law-T status, G6
dependent predictions, and the one-line branch reason.

**Synthetic-fixture smoke (NOT WCTE; illustration only, not a result).** With the schema
fixture, `Y_lo = 6011.7`, `Y_hi = 330729` (hit-fraction 0.95 bound), median yield 44 590,
N_min within/across medians 114/106 hits and zero exclusions:

| factor | setting | R²_run |
|---|---|---|
| F1 | 0.02 | 0.243 (excluded) |
| F1 | 0.05 | 0.659 (eligible) |
| F1 | 0.10 | 0.875 (eligible) |
| F3 | 0.10 | 0.303 → R12 not_evaluable |
| F2 | 0.50 | 0.911 (chosen; 0.10→0.20→0.50) |
| T | 0.5 | 0.982 |

This is a development smoke on synthetic arrays; the real-geometry numbers are pending.

**Joint F1/F2′ finding (synthetic fixture).** Because F2′ shares the F1 module-gain
nuisance block, the joint F1/F2′ ceilings are very small on the fixture: the joint F1
module-contrast median is `r2_run ≈ 1.5e-2` (mean 0.081, range 9e-3–0.75) and F2′ at
`b = 0.10` is `r2_run ≈ 4.4e-2` (the F2′ gradient is nearly degenerate with the module
pattern). If R11's F1 leg were required to use the joint ceiling, R11 would fail on this
fixture. The implementation uses the main-cell (F2-free) F1 ceiling for R6/R7 and records
the joint row as `f1_joint` for the Lead; this is the main methodological decision to
confirm before freezing.

## Pending

- **The actual real-geometry numbers.** The host command above must be run and the JSON
  frozen; no number in this file is reportable.
- **Merged PMT readout.** §3.5.2: after the PMT repairs the label changes and the check is
  re-run; the synthetic readout's QE curve, CE, threshold and walk are placeholders.
- **`lambda0`.** 50 m is the module constant placeholder; the PMT reference has absorption off.
  F2's absolute strength and even the F2 vs F2′ branch depend on it. If `lambda0` changes, both
  the generator and this check must be re-run (the config enforces the module constant).
- **`Y_lo`, `Y_hi`, N_min.** Only this check may change them; the frozen JSON is the source.
- **Views.** `view_masks` is a WP4c stub; the across N_min proxy is a random module half, not
  the contiguous Hilbert half.
- **The threshold censoring** is not in the Fisher model; a future refinement could use the
  generator readout's hit probability and the observed-charge distribution.

## Ambiguities (for the lead)

1. **F3 nuisance vs incidental parameters.** The runbook says "per-PMT ... parameters as
   appropriate (profiled)". Profiling one free gain per PMT makes every other factor's
   information exactly zero; the implementation therefore keeps F3 in `mu` but not in the
   nuisance block and profiles only the smooth factors. If the Lead wants the literal
   all-PMT profiling, the check becomes vacuous (all ceilings zero) and the decision is G6 /
   R12 not_evaluable by construction. This is the main methodological gate.
2. **`DetectorGeometry.from_synthetic`.** The actual API is the module-level
   `lucid_bombs.from_synthetic(schema)`; there is no classmethod. Tests use the module-level
   function.
3. **F1 acceptance cell.** F1's acceptance uses the main-cell (F2-free) ceiling; the
   joint F1/F2′ row (required by §3.5.3 for the F2′ cell) is computed and recorded in the F2′
   row's `f1_joint`, but is not used to exclude F1 cells. If the Lead wants the joint F1/F2′
   ceiling to govern R11's F1 leg, the branch can change.
4. **Median yield.** Population median `sqrt(Y_lo Y_hi)` is used; the realised median of a
   finite cohort is a different (noisier) choice. The runbook allows either; this is recorded.
5. **Across N_min proxy.** Random module half instead of the contiguous Hilbert half;
   `view_masks` is a stub.
6. **F3 representative PMT.** One PMT per module (the first) rather than all 19 contrasts;
   the median over modules is reported and the min/max are kept.
7. **Yield bound "median module charge".** Per-event mean over modules, then median over
   events, matching the generator benchmark's tercile table. A per-module median would give a
   different Y_hi.
8. **F2/F2′ joint profiling.** Only the within-module part of the attenuation/gradient
   survives after the F1 module-gain nuisance; this is the runbook's explicit joint profiling
   but it makes the F2 family much harder to detect than a no-F1-profile estimate. The JSON
   records the profiled list.

## Engineering decisions (one line each)

- Deterministic keyed event draws at a fixed yield; the vertex/QE draw is independent of the
  yield, so comparisons across yields are paired.
- A fixed Poisson capacity (4 × initial Y_hi) keeps the same uniforms across yields.
- Fisher matrices use the analytic generator derivatives (vertex derivative checked against
  finite differences) and pinv with `rcond = 1e-10` for the exact gain/yield degeneracies.
- Zero-sum contrasts remove the uniform gain mode that is exactly degenerate with the yield.
- F1/F3 vector ceilings use the median coordinate and record min/max; the contrast variance
  correction is explicit.
- The F2 family is profiled jointly with F1 module gains, as the runbook requires; only the
  within-module variation remains.
- The decision JSON is self-hashed (`content_sha256`) and schema-versioned; the
  `DECISIONS.md` block is embedded so the Lead copies it without editing this module.
- The CLI refuses a synthetic geometry unless explicitly allowed and never freezes one.
- The module imports only `lucid_bombs`; no LUCiD import and nothing redistributed.

## Tests

```sh
export PYTHONPATH=src
/private/tmp/jepa-venv-cpu311/bin/python -m pytest src/jepa_noise/tests/test_lucid_detectability.py -q
# 14 passed in ~1.3 s
```

Coverage: the analytic single-PMT Fisher `I = (dmu/df)^2/(mu(1+sigma_spe^2))` and the
`R2_max` limits; the profiling reduction to the naive information for an orthogonal/zero
nuisance; monotonicity in `sigma_spe` and in the number of runs; the vertex derivative against
finite differences; the median-yield policy and config constants; the decision logic for
pass, F1 shortfall/exclusion, no-F1 G6, the exact F2 fallback sequence, the F2′ fallback and
all-fail G6, and the F3 shortfall → `raise_F3_spread` → R12 not_evaluable; the frozen JSON
round trip and `content_sha256`; determinism for the same config/seed; the npz-path geometry
load with no LUCiD import; and the N_min occupancy check at low yield.
