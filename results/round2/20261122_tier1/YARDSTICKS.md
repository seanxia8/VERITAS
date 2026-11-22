# YARDSTICKS — what LUCiD's own model says about the vertex (r2-tier1 T6, exploratory)

Events: `dev_probe_test` photon bombs (Y1: 300 per detector; Y2: the first 200), θ = (x, y, z, t₀, ln N_γ). Model: LUCiD's expected (non-sampled) isotropic-source response (`setup_event_simulator(..., transport='expected', detection='expected')`) plus the dark expectation; time term = LUCiD's `lucid/losses.py` `first_arrival_window_nll` (σ = 2.1 ns, LUCiD's reconstruction working point). Code `src/jepa_noise/tier1_yardstick.py`.

**Y1 label:** surrogate Fisher from LUCiD's expectation model; omits SPE smearing, charge-dependent jitter and threshold; an optimistic bound.

## SK

### Y1 — surrogate Fisher split (σ_vertex = √trace of the vertex block of F⁻¹)

| Channel | median σ_vertex (m) | 16–84 % |
|---|---:|---|
| counts | 0.0125 | 0.0051–0.0366 |
| time | 0.0219 | 0.0119–0.0640 |
| joint | 0.0105 | 0.0046–0.0312 |
| time share σ_c²/(σ_c²+σ_t²) | 0.236 | 0.155–0.300 |

| Tertile | radius range (m) | joint | counts | time | time share | yield range | joint | time share |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.26–7.96 | 0.0097 | 0.0111 | 0.0187 | 0.239 | 2440–17267 | 0.0308 | 0.266 |
| 2 | 7.96–11.10 | 0.0116 | 0.0144 | 0.0238 | 0.234 | 17267–109204 | 0.0105 | 0.242 |
| 3 | 11.10–13.52 | 0.0101 | 0.0124 | 0.0215 | 0.234 | 109204–603589 | 0.0047 | 0.164 |

### Y2 — achieved resolution (minimise LUCiD's loss on the sampled events)

| Start | Channel | n | mean 3-D (m) | median 3-D (m) | converged | bias x / y / z (mm) ± SE | z-scores x / y / z / t₀ / ln N |
|---|---|---:|---:|---:|---:|---|---|
| truth | counts | 200 | 0.1779 | 0.1704 | 0.995 | +1.3±8.7 / -1.6±10.1 / +7.1±10.5 | +0.1 / -0.2 / +0.7 / – / +1.0 |
| truth | time | 200 | 0.1774 | 0.1634 | 0.995 | -1.5±8.5 / -7.3±9.6 / +8.3±7.9 | -0.2 / -0.8 / +1.1 / +16.6 / – |
| truth | joint | 200 | 0.1830 | 0.1781 | 1.000 | +4.7±8.9 / -12.0±9.9 / +10.7±8.4 | +0.5 / -1.2 / +1.3 / +14.8 / +4.2 |
| centroid | counts | 50 | 3.3950 | 3.3855 | 1.000 | +46.1±272.8 / -115.7±322.1 / -307.1±294.4 | +0.2 / -0.4 / -1.0 / – / +4.8 |
| centroid | time | 50 | 7.5373 | 4.1990 | 1.000 | -1099.0±977.3 / -192.3±927.2 / -1881.6±1090.9 | -1.1 / -0.2 / -1.7 / -10.7 / – |
| centroid | joint | 50 | 4.0510 | 3.7121 | 1.000 | -79.3±439.9 / -156.8±349.0 / -12.4±411.6 | -0.2 / -0.4 / -0.0 / -13.3 / +1.7 |

Frozen readouts on the same 200 events (validation split):

| Run | matched | Ridge spatial mean / median (m) | raw-feature mean (m) | attentive mean (m) |
|---|---:|---|---:|---:|
| `sk__mae_repaired__s9000` | – | pending | – | – |
| `sk__init__s9000` | – | pending | – | – |

Diagnostic (not a gate input): Y2 truth start re-fitted with the time-term σ = 3.49 ns (the detector's 1-PE jitter):

| Channel | n | mean 3-D (m) | t₀ bias (ns) ± SE | z-scores x / y / z / t₀ / ln N |
|---|---:|---:|---|---|
| counts | 50 | 0.1531 | +0.000±0.000 | -0.1 / -1.4 / +0.4 / – / +2.1 |
| time | 50 | 0.2154 | +1.307±0.136 | +0.5 / -1.2 / +0.8 / +9.6 / – |
| joint | 50 | 0.2159 | +1.156±0.148 | +0.7 / -0.8 / +0.2 / +7.8 / +3.2 |

## WCTE

### Y1 — surrogate Fisher split (σ_vertex = √trace of the vertex block of F⁻¹)

| Channel | median σ_vertex (m) | 16–84 % |
|---|---:|---|
| counts | 0.0047 | 0.0019–0.0121 |
| time | 0.0549 | 0.0193–0.1747 |
| joint | 0.0047 | 0.0019–0.0121 |
| time share σ_c²/(σ_c²+σ_t²) | 0.007 | 0.004–0.010 |

| Tertile | radius range (m) | joint | counts | time | time share | yield range | joint | time share |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.03–0.75 | 0.0046 | 0.0046 | 0.0536 | 0.007 | 2421–13847 | 0.0119 | 0.004 |
| 2 | 0.75–1.05 | 0.0069 | 0.0069 | 0.0899 | 0.007 | 13847–96327 | 0.0047 | 0.007 |
| 3 | 1.05–1.26 | 0.0036 | 0.0036 | 0.0407 | 0.007 | 96327–604071 | 0.0019 | 0.009 |

### Y2 — achieved resolution (minimise LUCiD's loss on the sampled events)

| Start | Channel | n | mean 3-D (m) | median 3-D (m) | converged | bias x / y / z (mm) ± SE | z-scores x / y / z / t₀ / ln N |
|---|---|---:|---:|---:|---:|---|---|
| truth | counts | 200 | 0.0304 | 0.0164 | 0.990 | -0.9±2.1 / -1.2±1.8 / +3.5±1.9 | -0.4 / -0.7 / +1.8 / – / +5.1 |
| truth | time | 200 | 0.1560 | 0.1517 | 1.000 | +5.8±7.0 / +9.1±7.1 / +0.4±6.5 | +0.8 / +1.3 / +0.1 / +61.5 / – |
| truth | joint | 200 | 0.1306 | 0.1084 | 1.000 | -4.6±6.4 / +0.1±6.4 / -0.6±6.1 | -0.7 / +0.0 / -0.1 / +58.1 / +13.4 |
| centroid | counts | 50 | 0.1299 | 0.1028 | 1.000 | +8.1±10.9 / -6.8±14.2 / -1.8±13.5 | +0.7 / -0.5 / -0.1 / – / +9.9 |
| centroid | time | 50 | 0.3283 | 0.1709 | 1.000 | +6.3±45.7 / +38.2±52.1 / +44.7±81.0 | +0.1 / +0.7 / +0.6 / +2.8 / – |
| centroid | joint | 50 | 0.1313 | 0.1102 | 1.000 | +0.3±14.3 / -6.4±14.1 / +0.8±11.3 | +0.0 / -0.5 / +0.1 / +3.5 / +4.4 |

Frozen readouts on the same 200 events (validation split):

| Run | matched | Ridge spatial mean / median (m) | raw-feature mean (m) | attentive mean (m) |
|---|---:|---|---:|---:|
| `wcte__mae_repaired__s9000` | 200 | 0.0552 / 0.0457 | 0.0782 | 0.0587 |
| `wcte__init__s9000` | 200 | 0.0699 / 0.0575 | 0.0782 | 0.0880 |

Diagnostic (not a gate input): Y2 truth start re-fitted with the time-term σ = 0.95 ns (the detector's 1-PE jitter):

| Channel | n | mean 3-D (m) | t₀ bias (ns) ± SE | z-scores x / y / z / t₀ / ln N |
|---|---:|---:|---|---|
| counts | 50 | 0.0212 | +0.000±0.000 | +0.7 / +0.4 / +0.3 / – / +3.4 |
| time | 50 | 0.1017 | +1.720±0.037 | +1.1 / +1.0 / +0.3 / +46.9 / – |
| joint | 50 | 0.0884 | +1.696±0.040 | +0.1 / +0.7 / -0.4 / +42.3 / +6.3 |

## Reading (K4(b) and the information share)

- **K4(b)** (joint fit started at truth, 200 events): the vertex coordinates are unbiased on both detectors (largest |bias|/SE: SK 1.3, WCTE 0.7). The nuisance t₀ is biased (SK +0.66 ns, WCTE +2.7 ns), and the bias moves with the time-term resolution assumed by the surrogate (σ = detector 1-PE jitter: SK +1.16 ns, WCTE +1.70 ns; vertex still unbiased), so it is a property of the first-arrival time term's Gaussian resolution model (no charge-dependent jitter or TDC), not a stable data inconsistency. On the vertex reading recorded in `docs/round2/DECISIONS.md`, K4(b) does **not** fire; on a strict reading over every component of θ it would. `GATE_G1.md` states the outcome under both.
- **Information share (Y1):** under LUCiD's expectation model the time channel carries a minority of the vertex information on SK (median time share 0.24) and almost none on WCTE (0.007) at the LUCiD working point σ = 2.1 ns. This is an optimistic surrogate (label above), not a bound on what a learned encoder can reach.
- **Achieved (Y2):** started at truth, LUCiD's loss gives 0.18 m (SK) and 0.03 m counts-only / 0.13 m joint (WCTE) mean 3-D error, far above the Fisher numbers (≈1 cm): the expectation model omits SPE smearing, charge-dependent jitter and threshold. On WCTE the time term degrades the fit, consistent with its mis-specified resolution. Fits started at the charge-weighted centroid stay in a wrong basin on SK (≈3.4–7.5 m) within 40 iterations. Yardsticks are exploratory (§2.6).
