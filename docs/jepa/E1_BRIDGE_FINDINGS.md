# E1 / B1 post-hoc diagnostics — finite-sample floors, R3 conditioning, amplitude ceiling

Date: 2026-10-17. Author: bounded analysis session (post hoc, unreviewed). Code: branch `JEPA`,
working tree; no `src/jepa_noise/*.py` file was modified and no training was run.

## 0. Scope: this file changes no registered verdict

The verdicts below are the ones registered when this analysis was requested (they are also the
verdicts this file changes nothing about). While the note was being written the detached E1b
continuation completed all 65 rows and `report E1` regenerated `docs/jepa/PREDICTIONS.md`; the only
status change is R2 (`not_evaluable` → **fail**, see the note after the table). All other rows are
unchanged.

| prediction | registered status (at request time) | registered evidence |
|---|---|---|
| R1 | **fail** | E1b clean cell: sample median max angle 17.309° > 10°; population design value 0° |
| R2 | **not_evaluable** | E1a Spearman < 0.8 in 2 of 6 cells (the reason code `fail_4of6_spearman` counts the 4 passing cells); E1b g-sweep clause not yet evaluated |
| R3 | **not_evaluable** | 2 of 210 rows raise the reviewed solver's tolerance check |
| R4a | **fail** | 32/42 cells ≤ 5° |
| R4b | **fail** | 33/42 at β = 10 |
| R5 | **not_evaluable** | no eligible depth-2/3 modes |
| B1 | **fail** | plain nuisance R² 0.5251 vs anchored 0.5157; needs plain > anchored in every seed |

**E1b completion note (2026-10-17 21:51 local, outside this analysis).** All 13 E1b cells now have 5 seeds. The
E1b half of R2 passes (Spearman over g = 1.000; seed-mean angle at g = 0.05 is 77.33° ≥ 45°) but
the E1a half still fails (4 of 6 cells reach Spearman ≥ 0.8), so `report E1` records R2 as **fail**
instead of `not_evaluable`. R1 stays fail (E1b clean-cell median 17.309° > 10°), R3 stays
`not_evaluable`, R4a/R4b/R5/B1 are unchanged. The new E1b g-sweep medians (20.5° at g = 0.005 …
77.4° at g = 0.05 … 89.6° at g = 0.9) track the population displacement of the same cells
(11.3°, 76.9°, 89.4°) closely, which is the same saturation phenomenon diagnosed in §1.5; those
numbers are the Lead's and are only noted here.

Every number below is either (a) recomputed from the stored tables/row files, or (b) computed
from *new independent draws* of the same registered fixtures. Nothing is re-tuned, no cell or
seed is re-selected, and no registered statistic is replaced. Where a computation supports a
registered reading it is stated as support; where it does not, that is stated too.

**Summary of what is found.**

1. **R1 (E1b)** — supported as a *finite-sample* effect. The population angle is 1.90 × 10⁻¹¹ rad;
   at n = 10⁵ two independent samples are 18.26° and 15.09° from R_c and 24.06° from each other
   (floor/angle = 1.44 ≈ √2, the independent-draw value), the registered seed-0 angle is 16.31° and
   the 5-seed median 17.31°. The angle scales as n^(−0.48) and reaches 4.97°/5.98° at n = 10⁶,
   i.e. below the 10° threshold (crossing at n ≈ 2.8 × 10⁵). The n = 10⁵ failure is the
   estimator's own sampling noise, not a population bias.
2. **R4a** — supported the same way. The population recovery error is 3.02 × 10⁻¹⁴ rad (recorded;
   recomputed max 1.66 × 10⁻¹³ rad over the 42 cells: float64 zeros). At n = 10⁵ the GEVP angle is
   4.0°–6.3° (mean 5.05°) with a two-sample floor of 5.9°–9.7° (mean 7.14°), and 1.43°–1.51° at
   n = 10⁶. The 5° threshold sits *inside* the sampling distribution at the registered n.
3. **R2** — the two failing cells are the most saturated ones; the registered Spearman is being
   computed over a curve with 4–5 of 7 points on the 81–89° plateau (mean 86.4°). This is a property of the statistic,
   not evidence against displacement: the population angle itself is already 70–89° at γ ≥ 0.1.
   The diagnostic recomputation without saturated points is reported as diagnostic only.
4. **R3** — the 208 clean rows are invariant to 8.20 × 10⁻¹⁰ (tolerance 10⁻⁸). The two
   non-evaluable rows are the two worst-conditioned transformed problems in the grid
   (cond 1.47 × 10¹⁰ and 1.59 × 10¹¹, ranked #2 and #1 of 210); their transformed isotropy
   residual is 2.11 × 10⁻⁸ / 3.18 × 10⁻⁸ against the fixed 10⁻⁸ solver tolerance, with the
   objective gap still at 10⁻¹²/10⁻¹¹. A float64 conditioning symptom of the frozen transform.
5. **Amplitude ceiling** — the 0.9 threshold does **not** lie above the information ceiling:
   R²_max = 0.9393 (joint, nuisance-marginalised) and 0.9911 (channel-local). Tokens from the
   sigreg arms are at 95–97% of that ceiling; post-time (0.808–0.817) and the mae_chi2/parallel
   arms (0.75–0.80) are at 80–87%, i.e. representation-limited. The recorded "criterion not met"
   stands and the CRLB does **not** explain it.

Evidence, scripts and JSON: `results/jepa/20261017_e1_findings/`. Figure:
`figures/fig_sampling_floor.png` (finite-sample), `figures/fig_r2_saturation.png` (R2).

## 1. Finite-sample check (R1, R4a, R2)

### 1.1 What was computed, and the replay check

Two fixtures are used, both already used by the E1 runners:

* **E1a** — `views.ExactModel(d=40, k=3, seed, theta_scale, eta_scale, share='target')`, with the
  registered `E1a_subject` namespace (`run.run_e1a_row`). Cells: θ_scale = 0.1, η ∈ {1, 3},
  γ ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1} (14 cells).
* **E1b** — `views.Tier1Views(signal_mode='linearized', split='channel', line_multiplier=0,
  common_strength=0, base_rho=0, law=A4)`, i.e. the clean cell (g = 0, line = 0), with the
  registered θ SDs and the `channel` split (context = channels 0–3, target = 4–7, 1024 + 1024
  dimensions).

The estimator is the registered `r1_cca` row: `linear.cca` on 10⁻⁶-ridge marginals with the
unridged cross term, angle against the population `R_c = span(N_cc⁻¹ A_c)`.

**Replay validation was run before any new draw** (`validation.json`): with the registered
namespaces the E1a sample-covariance SHA-256 and the E1b sample-covariance SHA-256 are reproduced
**bit-exactly** (identical digests), and the CCA angles reproduce the stored values to 0.0°:

| replay | sample-covariance SHA-256 | registered angle | recomputed |
|---|---|---|---|
| E1a `th0.1_eta1_g0_n100000_s0` | matches | 2.7799886° | 2.7799886° |
| E1b `g0_line0_s0` | matches | 16.3067552° | 16.3067552° |

For n = 10⁶ the E1a and E1b draws are chunked (streaming sufficient statistics, float64, ≤ 20k /
5k events per block; 10⁶ × 2048 is never materialised). The E1b chunked generator draws i.i.d.
Gaussian records with the same spectral shaping and channel weights as `Tier1Views.noise_record`
(one bulk RNG instead of the per-event namespace RNG): same law, independent realisation. The
bit-exact replay above validates the estimator and the fixture; the n = 10⁵ vectorised result
(18.26°/15.09°) brackets the registered 5-seed range (15.90° … 18.24°).

### 1.2 Population values (the theory is exact)

| quantity | value |
|---|---|
| E1b clean cell, population CCA angle vs R_c | **1.90 × 10⁻¹¹ rad** (1.09 × 10⁻⁹ deg) |
| E1b clean cell, population canonical correlations (top 3) | 0.99938, 0.99860, 0.68241 |
| E1a θ = 0.1, γ = 0, population CCA angle vs R_c | 4.1 × 10⁻¹³ / 2.4 × 10⁻¹² deg |
| E1a θ = 0.1, γ = 0, population canonical correlations | 0.4255, 0.3350, 0.3246 |
| R4a population recovery error (GEVP), recorded | **3.0246 × 10⁻¹⁴ rad** |
| R4a population recovery error, recomputed max over 42 cells | 1.661 × 10⁻¹³ rad |

So the population displacement is zero to float64 precision and any nonzero sample angle is
sampling error. (The whitened-PCA path, which is not R4a's estimator, gives the same order of
magnitude, ≤ 2.1 × 10⁻¹³ rad.) The E1b population canonical spectrum is strongly non-degenerate
(0.9994/0.9986/0.6824), which is why the E1b subspace does not suffer the near-tie problem of the
E1a θ = 0.1 cells (0.3350 vs 0.3246).

### 1.3 Sampling floor and 1/√n scaling

"Floor" = maximum principal angle between the top-3 context canonical subspaces of **two
independent samples** of the same law at the same n (registered estimator, both times).

| fixture | n | replicate angles (deg) | floor (deg) | floor / angle |
|---|---|---|---|---|
| E1b clean cell (channel) | 10,000 | 48.88, 51.25 | 67.72 | 1.35 |
| E1b clean cell (channel) | 100,000 | **18.26, 15.09** | **24.06** | 1.44 |
| E1b clean cell (channel) | 1,000,000 | 4.97, 5.98 | 7.53 | 1.37 |
| E1a th0.1_eta1_g0 | 10,000 | 10.21, 10.20 | 15.38 | 1.51 |
| E1a th0.1_eta1_g0 | 100,000 | 2.78, 4.07 | 5.41 | 1.58 |
| E1a th0.1_eta1_g0 | 1,000,000 | 1.11, 1.25 | 1.56 | 1.32 |
| E1a th0.1_eta3_g0 | 10,000 | 10.47, 11.74 | 13.59 | 1.22 |
| E1a th0.1_eta3_g0 | 100,000 | 4.18, 4.13 | 5.65 | 1.36 |
| E1a th0.1_eta3_g0 | 1,000,000 | 1.19, 1.00 | 1.56 | 1.43 |

Log-log slopes (n = 10⁴ → 10⁶):

| fixture | slope of angle | slope of floor |
|---|---|---|
| E1b clean cell | **−0.481** | −0.477 |
| E1a th0.1_eta1_g0 | −0.468 | −0.497 |
| E1a th0.1_eta3_g0 | −0.503 | −0.470 |

Both the CCA angle and its floor follow n^(−1/2) to within ~4% over two decades, and the floor is
1.2–1.6 × the angle. For two *independent* estimates the expected floor is √2 ≈ 1.41 × the
estimate-vs-truth error, which is the range measured: the registered angle at n = 10⁵ is at the
estimator's noise scale.

**Reading for R1.** The registered E1b failure (median 17.31°, seed 0 16.31°) is reproduced by two
fresh independent draws (16.67° mean) and sits on the two-sample floor. At n = 10⁶ the same
estimator gives 4.97°/5.98° (mean 5.48°), inside the 10° criterion; the crossing is at
n ≈ 10⁵ · (16.67/10)² ≈ 2.8 × 10⁵. The E1a half of R1 is far from its 10° threshold
(3.1°/4.2° medians at n = 10⁵). So: the R1 verdict fails at the registered n = 10⁵ as recorded,
and the failure is consistent with estimation noise at that sample size — it is not evidence that
the population identity fails (population angle 1.9 × 10⁻¹¹ rad). The verdict itself is unchanged.

### 1.4 R4a (GEVP) finite-sample floor

R4a's estimator is `r5_gevp` (GEVP of the unregularised subject covariance against an independent
`E1a_noise_cal` covariance, reference R_c). Recomputing its floor requires an independent
calibration draw for the second replicate; with that, for the 14 θ = 0.1 cells at n = 10⁵:

| quantity | value |
|---|---|
| GEVP angle at n = 10⁵, 14 cells | mean **5.05°**, range 4.04°–6.31° (6/14 above 5°) |
| two-sample GEVP floor at n = 10⁵ | mean **7.14°**, range 5.85°–9.66° (floor/angle = 1.42) |
| GEVP angle at n = 10⁴ → 10⁶ (γ = 0 cells) | 13.9° → 4.4° → 1.43°; 14.3° → 4.0° → 1.51° |
| log-log slope (γ = 0 cells) | −0.487, −0.518 |

The registered R4a criterion (population error ≤ 10⁻⁶ rad **and** median sample angle ≤ 5°) is
therefore met or missed at n = 10⁵ depending on the seed, exactly as a distribution centred at
~5° with a ~2° spread would behave: 32 of 42 cells pass. The population half is exact
(3 × 10⁻¹⁴ rad), the sample half is at the floor, and at n = 10⁶ the same estimator is at 1.4–1.5°.
R4a's failure is a sample-size statement, not a theory failure.

### 1.5 R2: the failures sit on the saturated plateau

Registered statistic (unchanged): Spearman(gamma, seed-mean max angle) ≥ 0.8 per cell. Seed means
are read from `aggregate_E1a.json`; nothing is recomputed for the registered value.

| cell | registered Spearman | pass | points ≥ 80° | diagnostic Spearman without ≥ 80° points | points kept |
|---|---|---|---|---|---|
| theta0.1_eta1 | +0.8571 | pass | 5/7 | — (2 points: undefined) | 2/7 |
| theta0.1_eta3 | +0.6429 | **fail** | 5/7 | — (2 points: undefined) | 2/7 |
| theta0.4_eta1 | +0.8571 | pass | 3/7 | 1.000 | 4/7 |
| theta0.4_eta3 | +0.7143 | **fail** | 4/7 | 1.000 | 3/7 |
| theta1_eta1 | +1.0000 | pass | 0/7 | 1.000 | 7/7 |
| theta1_eta3 | +0.8571 | pass | 4/7 | 1.000 | 3/7 |

Of the 42 (cell, γ) seed means, **21 are ≥ 80°** (mean 86.35°, range 80.8°–89.2°): the plateau is
near 87°, as the author's reading states. The two failing cells are the two whose curves spend
most of their range on the plateau. Dropping the saturated points makes the rank correlation 1.0
where it is still defined (2–4 points) — a **diagnostic only**; it is not a replacement for the
registered Spearman, which remains the verdict.

Why this is a statistic limitation and not evidence against displacement: the *population* angle
against R_c is already 70.3°/74.6° at γ = 0.1 and 87–89° at γ ≥ 0.5 for these cells, so the
plateau is a population property of the nuisance displacement (P1). The sample angle at n = 10⁶
tracks the population value to 0.4–4.4° for γ ≥ 0.25 (12.9–14.5° at the near-tie γ = 0.1 cells). Two genuine caveats, reported without
changing anything: (i) near the transition (η = 3, γ = 0.1) the population canonical correlations
are close (0.920 then 0.418/0.333) and the top-3 subspace remains ill-determined even at
n = 10⁶ (deviation 12.9°), so an angle-at-γ diagnostic is intrinsically unstable there;
(ii) with 7 points and 4–5 near-ties, the Spearman outcome flips between pass and fail on tie
count alone (θ0.1_η1 passes at 5/7 saturated; θ0.4_η3 fails at 4/7).

`R2`'s E1b clause is no longer the blocker: the completed g-sweep gives Spearman 1.000 and a
seed-mean angle of 77.33° at g = 0.05 (≥ 45°), so the clause passes. R2 is recorded as **fail**
because of the E1a clause analysed here (4/6 cells reach Spearman ≥ 0.8), i.e. exactly the
saturation effect above, not a displacement failure. The E1b g-sweep shows the same plateau: the
sample medians (20.5°, 27.4°, 50.4°, 77.4°, 84.5°, 87.6°, 89.0°, 89.6° for g = 0.005 … 0.9) track
the population displacement of the same cells (11.3°, 22.3°, 48.0°, 76.9°, 84.1°, 87.5°, 88.8°,
89.4°).

## 2. R3: the 208 clean rows and the two non-evaluable rows

Recomputed from the 630 stored row files of `attempt_007` (210 rows carry an R3 diagnostic at
n = 10⁵):

| quantity | value |
|---|---|
| rows with an R3 diagnostic | 210 |
| evaluated rows | 208 |
| max observed spectrum difference (208 rows) | **8.204 × 10⁻¹⁰** |
| median / p99 observed difference | 2.34 × 10⁻¹³ / 2.30 × 10⁻¹⁰ |
| rows outside the fixed 10⁻⁸ tolerance | 2 |

The maximum is the same value the aggregate records (`8.203891455060841e-10`), so the 10⁻⁸
invariance check holds on all 208 computable rows with three orders of magnitude of margin.

The two failures (`th0.4_eta1_g0_s3`, `th0.4_eta3_g1_s2`) are reproduced exactly (same samples,
same frozen transform) and are the two worst-conditioned transformed problems in the grid:

| failing row | cond(S_cc) | cond(S_tt) | cond(T_c S_cc T_cᵀ) | cond(T_t S_tt T_tᵀ) | cond(T_c) | cond(T_t) |
|---|---|---|---|---|---|---|
| th0.4_eta1_g0_s3 | 145.4 | 25.9 | **1.468 × 10¹⁰** (#2 of 210) | 4.81 × 10⁴ | 2.86 × 10⁴ | 1.00 × 10² |
| th0.4_eta3_g1_s2 | 802.6 | 990.8 | 4.30 × 10⁶ | **1.591 × 10¹¹** (#1 of 210) | 2.07 × 10² | 3.43 × 10⁴ |

Which check trips: the reviewed solver raises when either the isotropy residual or the
objective/reduced-objective gap exceeds 10⁻⁸. Recomputing both on the transformed problems:

| failing row | base isotropy | transformed isotropy | TOL | transformed objective gap | eps × cond |
|---|---|---|---|---|---|
| th0.4_eta1_g0_s3 | 5.6 × 10⁻¹⁵ | **2.11 × 10⁻⁸** | 10⁻⁸ | 1.21 × 10⁻¹² | 3.3 × 10⁻⁶ |
| th0.4_eta3_g1_s2 | 3.5 × 10⁻¹⁴ | **3.18 × 10⁻⁸** | 10⁻⁸ | 3.81 × 10⁻¹¹ | 3.5 × 10⁻⁵ |

The failing check is the isotropy residual, exceeded by only 2–3×, on a problem whose condition
number is 10¹⁰–10¹¹ (so the float64 precision of the product is 10⁻⁶–10⁻⁵ — far above the fixed
10⁻⁸ tolerance). The invariance identity itself is not contradicted: the base problems pass at
10⁻¹⁴–10⁻¹⁵ and the p99 spectrum difference over all other rows is 2.3 × 10⁻¹⁰. R3 therefore
stays **not_evaluable** for the same reason recorded: 2 of 210 rows cannot be certified at the
reviewed solver's tolerance.

## 3. Amplitude-probe information ceiling

Fixture: the WP4a bridge cell exactly as configured (`signal_mode='nonlinear'`, `split='channel'`,
`common_strength=0.02`, `line_multiplier=0`, A4 law, `base_rho=0`, seed 0, cell `bridge`). The
materialised `N_actual` matches the committed reference target marginal
(`configs/bridge_v1_n_actual_reference.npz`) with relative Frobenius error **5.1 × 10⁻¹⁷**, i.e.
it is the same matrix the runs used.

Target: the probe target of the bridge cohort, `A_c = max_t views.waveform(theta)[c]`, with the
registered nonlinear prior (amplitude U[3, 6], t0 U[.30, .40], tau_t U[.06, .10]); the bridge
geometry has **unit channel gains**, so A_c is the same random variable for all eight channels.

`R²_max(c) = Var(A_c) / (Var(A_c) + CRLB_c)`, with `CRLB_c` the Cramér–Rao variance of the
amplitude under `N_actual`:

* **joint** — `g_cᵀ (Jᵀ N⁻¹ J)⁻¹ g_c`, `g_c = ∂A_c/∂θ`, i.e. all 2048 samples (the probe reaches
  every channel through attention); Fisher from `jepa_noise.crlb` (cross-checked against the
  accelerated triangular path to 4.8 × 10⁻¹⁴ relative);
* **channel-local** — `1/(u_cᵀ N_cc,c⁻¹ u_c)`, i.e. channel c's own 256 samples, a conservative
  reference ceiling.

| channel | Var(A_c) | CRLB joint (prior mean) | CRLB joint at θ₀ | CRLB channel-local | R²_max joint | R²_max local |
|---|---|---|---|---|---|---|
| 0 | 0.4496 | 0.02908 | 0.02955 | 0.004043 | **0.9393** | 0.9911 |
| 1 | 0.4496 | 0.02908 | 0.02955 | 0.004027 | 0.9393 | 0.9911 |
| 2 | 0.4496 | 0.02908 | 0.02955 | 0.004037 | 0.9393 | 0.9911 |
| 3 | 0.4496 | 0.02908 | 0.02955 | 0.004046 | 0.9393 | 0.9911 |
| 4 | 0.4496 | 0.02908 | 0.02955 | 0.004019 | 0.9393 | 0.9911 |
| 5 | 0.4496 | 0.02908 | 0.02955 | 0.004007 | 0.9393 | 0.9912 |
| 6 | 0.4496 | 0.02908 | 0.02955 | 0.004052 | 0.9393 | 0.9911 |
| 7 | 0.4496 | 0.02908 | 0.02955 | 0.004046 | 0.9393 | 0.9911 |

The θ-CRLB diagonal at θ₀ is (0.07335, 2.649 × 10⁻⁷, 2.092 × 10⁻⁵); the amplitude-only
known-other-parameters bound is 1/I_aa = 8.53 × 10⁻⁴, 86× tighter than (I⁻¹)_aa, which is why the
nuisance-marginalised bound is the right ceiling for a single event. Prior mean vs θ₀ CRLB differ
by 1.6%, so the local Fisher is not a limitation.

Measured amplitude R² (per-channel mean over the 8 channels) divided by R²_max (joint):

| arm | seed | tokens R² | tokens / R²_max | post-time R² | post-time / R²_max |
|---|---|---|---|---|---|
| sigreg_jepa | 0 | 0.9122 | **0.971** | 0.8076 | 0.860 |
| sigreg_jepa | 1 | 0.9004 | 0.959 | 0.8100 | 0.862 |
| sigreg_jepa | 2 | 0.9097 | 0.969 | 0.8111 | 0.864 |
| sigreg_jepa_anchored | 0 | 0.8953 | 0.953 | 0.8164 | 0.869 |
| sigreg_jepa_anchored | 1 | 0.9048 | 0.963 | 0.8087 | 0.861 |
| sigreg_jepa_anchored | 2 | 0.9052 | 0.964 | 0.8169 | 0.870 |
| mae_chi2 | 0 | 0.7718 | 0.822 | 0.7741 | 0.824 |
| mae_chi2 | 1 | 0.7976 | 0.849 | 0.7768 | 0.827 |
| mae_chi2 | 2 | 0.7955 | 0.847 | 0.7741 | 0.824 |
| sigreg_jepa_parallel | 0 | 0.7775 | 0.828 | 0.7799 | 0.830 |
| sigreg_jepa_parallel | 1 | 0.7515 | 0.800 | 0.7790 | 0.829 |
| sigreg_jepa_parallel | 2 | 0.7973 | 0.849 | 0.7777 | 0.828 |

(random-init floor for both read-outs: 0.7376.)

**Is the 0.9 threshold above the information ceiling? No.** 0.9 < 0.9393 (joint) < 0.9911
(channel-local). The threshold is demanding — it asks for ~96% of the joint information ceiling —
but it is achievable, and the plain `sigreg_jepa` token read-out reaches 96–97% of the ceiling in
all three seeds (0.9004, 0.9097, 0.9122 — all three above 0.9, as the recorded per-seed table shows). The recorded "criterion not met" therefore stands **without** an
information-theoretic excuse: post-time (0.808–0.817, 86–87% of the ceiling), `mae_chi2` and
`sigreg_jepa_parallel` (0.75–0.80, 80–85%) are representation/probe limited, not CRLB limited.

Consistency check with the runs' own numbers: the plain arm's seed-0 amplitude MSE/CRLB is 1.1138
and its token R² is 0.9122, so MSE/Var = 0.0878, CRLB/Var = 0.0789 and the implied ceiling is
1/1.0789 = 0.927 — within 1.3% of the 0.9393 computed here from `N_actual` and the prior.

## 4. B1 diagnostics

The detailed additions are appended to `docs/jepa/WP4A_BRIDGE_RESULTS.md` as a **Diagnostics**
section (2026-10-17, post hoc). In brief, with values verified from the combined aggregate and
the 12 per-run `probes.json` files:

* **Positive half of B1 (P1 support).** Plain `sigreg_jepa` nuisance R² = **0.5251** against a
  random-init floor of **−0.00358** (identical in all 12 runs; per-target common_0 0.5812,
  common_1 0.5034 in seed 0). The deep transformer does encode the shared common mode; the probe
  is not fitting noise. Anchored: 0.5157.
* **Anchor may have been too weak (hypothesis).** β = 10 was the top of the registered grid
  {0.1, 1, 10} and the registered β-selection score was still improving at that boundary
  (0.104152 → 0.104057 → 0.097206). R4b independently fails at β = 10 in the exact linear model
  (9 of 42 E1a cells > 5°). No retuning is done or implied here; the author has separately
  approved a labelled β ∈ {10, 100, 1000} follow-up under the LUCiD selection rule.

## 5. Limits of this note

* No registered verdict, threshold, cell, seed or statistic is changed. The follow-up in §4 is
  separately approved and is not run here; no training was run.
* The two-sample floors are single pairs of independent draws per (cell, n); a two-sample angle is
  a noisy quantity (its own relative spread is ~20–30%). The conclusions are "at the noise scale",
  not exact floor values, and they hold across three n and five fixtures.
* The E1b n = 10⁶ and E1a n = 10⁶ draws use bulk/chunked RNG streams of the registered laws rather
  than the per-event namespace RNG (documented in `e1f_common.py`). The registered path is
  reproduced bit-exactly at n = 10⁵ (§1.1), so the law and estimator are verified; the n = 10⁶
  numbers are a new independent realisation.
* Where my numbers differ from a stored number: the R4a population recovery recorded by
  `report.compute_r4a` is the last cell's value (3.0246 × 10⁻¹⁴ rad); the max over all 42 cells is
  1.661 × 10⁻¹³ rad. Both are float64 zeros and neither changes any reading.
* The one place where the requested framing does not survive contact with the numbers: the 0.9
  amplitude threshold is **below** the information ceiling, not above it (§3). The "criterion not
  met" is therefore not explained by the CRLB.

## 6. Commands, files and runtime

All runs used `/private/tmp/jepa-venv-cpu311/bin/python` with `PYTHONPATH=src` and
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`, from
`results/jepa/20261017_e1_findings/`:

```sh
python s00_validate.py        # bit-exact replay of the two registered fixtures (~50 s)
python s01_finite_sample.py   # floors + n-scaling, E1a + E1b up to n=1e6 (~7 min, 1.2 GB peak)
python s02_r2_saturation.py   # R2 seed-mean curves + saturation diagnostic (<5 s)
python s03_r3_conditioning.py # 210-row R3 evidence + condition numbers (~90 s)
python s03b_r3_residuals.py   # which solver check trips on the 2 rows (<5 s)
python s04_amplitude_ceiling.py  # Fisher, CRLB, R2_max, measured ratios (~90 s)
python s08_gevp_floor.py      # R4a GEVP sample angles and floor (~15 s)
python s05_tables.py          # print the tables of this document from the JSON
python s07_figure.py          # regenerate figures/fig_sampling_floor.png
```

Files: `e1f_common.py` (shared fixtures/estimators), the nine scripts above, `validation.json`,
`finite_sample.json`, `r2_saturation.json`, `r3_conditioning.json`, `r3_residuals.json`,
`amplitude_ceiling.json`, `gevp_floor.json`, `figures/fig_sampling_floor.png` (122 KB),
`figures/fig_r2_saturation.png` (86 KB) — 19 new files in total, all under the two reported
paths.
Total wall time ≈ 17 min of compute (including one discarded s01 rerun after an accumulator bug in
my own script, and the repeated figure and GEVP passes), one process at a time,
peak RSS 1.21 GB (the E1b n = 10⁶ streaming run), well inside the 2-process / 3 GB envelope; no `src/jepa_noise` file was modified, and the
concurrently running E1b row generation was not touched.
