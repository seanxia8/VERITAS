# H1/H2 causal tests: E2 geometry feature ablation + E3 frozen target — 1 Nov 2026

`registered: false`. Extension `h12_causal_2026-11-01`.
Runbook: `docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md` (sha256 `aab229a08cd9dca6288dd0143ff0b75efa4d5e4cea0330830d56727a2c51436f`).
Author-approved training: exactly E2 seeds 9000/9001, E3 seeds 9000/9001 and one 20-step smoke per variant; no other training.

- final code SHA: `7952cb65a12b3599305648738720f7c5d88574fd`
- §4.2 reproduction: PASS

## §6 verdict

- **E2-r** (geometry feature ablation): **not supported**
- **E3-r** (frozen random target): **supported**
- **decision**: **H2 dominant (trainable-target trivial solution)**

## Phase 0 priors (E0/E1)

| reading | value |
|---|---|
| E0-a `f_layout` | 0.9970192836471821 |
| E0-b `f_event` | 7.89195390170171e-05 |
| E0-c position-only label ridge R2 | 0.0020612391143502544 |
| E1 invariant fraction | 1.007205900908509 |
| E1 position-explained R2 | 0.0011217516180519888 |
| E1 label-variance ratio vs init | 0.5220993129209447 |
| E1 n label rows / events | 64000 / 2000 |

Prior readings: {"E0-1": "unclear", "E1-1": "unclear"}

## E2/E3 runs

| run | variant | seed | wall s | full vertex m | full token R2 | half vertex m | half token R2 | zero geometry |
|---|---|---:|---:|---:|---:|---:|---:|---|
| geomblind_9000 | geomblind | 9000 | 2313.678 | 0.8296332662924278 | 0.9999148946975726 | 0.27908881792204737 | 0.6480924930901957 | True |
| geomblind_9001 | geomblind | 9001 | 2671.474 | 0.8421279557839693 | 0.9998290378350404 | 0.2772831081740512 | 0.585994071982691 | True |
| frozen_target_9000 | frozen_target | 9000 | 1383.772 | 0.1384592831431598 | 0.9237341684234989 | 0.13165235748531384 | 0.8875579852696529 | False |
| frozen_target_9001 | frozen_target | 9001 | 1711.494 | 0.1314360211884925 | 0.9328444878479647 | 0.13760177059503667 | 0.8907825711580486 | False |

Training-log NaN check (§4.6): geomblind_9000=True, geomblind_9001=True, frozen_target_9000=True, frozen_target_9001=True (last logged metrics).


Baselines: `init9000` full_input vertex 0.16007169092238768 m, half_pass vertex 0.17482124528478005 m; `fix9000` full_input vertex 0.21925859104673262 m, `fix9001` 0.20428123360020733 m (Q1/Q2 committed matrix).

## §6 recovery detail

- **geomblind**: not supported (floors {"full_input_m": 0.16007169092238768, "half_pass_m": 0.17482124528478005})
  - seed 9000: full vertex 0.8296332662924278 m (ok=False), full token R2 0.9999148946975726 (ok=True), half vertex 0.27908881792204737 m / half R2 0.6480924930901957 (half ok=False)
  - seed 9001: full vertex 0.8421279557839693 m (ok=False), full token R2 0.9998290378350404 (ok=True), half vertex 0.2772831081740512 m / half R2 0.585994071982691 (half ok=False)
- **frozen_target**: supported (floors {"full_input_m": 0.16007169092238768, "half_pass_m": 0.17482124528478005})
  - seed 9000: full vertex 0.1384592831431598 m (ok=True), full token R2 0.9237341684234989 (ok=True), half vertex 0.13165235748531384 m / half R2 0.8875579852696529 (half ok=True)
  - seed 9001: full vertex 0.1314360211884925 m (ok=True), full token R2 0.9328444878479647 (ok=True), half vertex 0.13760177059503667 m / half R2 0.8907825711580486 (half ok=True)

E2 is a **feature ablation** (token order, parity mask and predictor position queries remain geometry-defined), so a positive E2 reads "consistent with H1". E3 also changes the SIGReg target term (the frozen label feeds both the prediction MSE and the SIGReg target term).

## §4.2 reproduction

| item | expected m | recomputed m | |diff| m | pass |
|---|---:|---:|---:|---|
| fix9000 | 0.219258 | 0.21925859104673262 | 5.910467326120106e-07 | True |
| init9000 | 0.160072 | 0.16007169092238768 | 3.090776123093608e-07 | True |

## Outputs

- `e0e1.json` — Phase 0 priors (E0-a/b/c, E1) and their readings.
- `runs.json` — per run: variant, seed, wall, full_input/half_pass rows, comparison with fix/init, recovery detail.
- `decision.json` — §6 E2-r/E3-r with the actual values and the author's conjunction.
- `reproduction.json` — §4.2 check (exit 2 on mismatch).
- `README.md` — this file.

## Code-origin addendum for the per-PMT invariant component (2 Nov 2026, 0 training)

Question: what is the origin of the `fix9000` fixed per-PMT code (the E1 invariant component
`m_p`)? The position-probe addendum already ruled out the PMT position; this addendum tests the
PMT's other fixed attributes: hit rate, mean hit ln q, within-target parity (`pmt_id_in_mpmt % 2`),
module id (one-hot, 105 modules), scaled position, and the mean raw-token feature vector
(9 = `FEATURES`), plus the combined 120-feature set. Code SHA `d6d013f` (backward-compatible
`e0e1 --code-origin`; `e0e1.json` unchanged). Inputs: the same 64,000 label rows / 2,000 photon
probe-test events / seed 0 as `e0e1`; per-PMT features are read-only from the same probe events
(tokenisation only, no encoder forward pass); 80/20 split by PMT exactly as the position-probe
addendum (1,050 present PMTs → 840 train / 210 test, 84 validation PMTs for MLP early stopping;
kNN k = 5). Target: `m_p` (128-d, invariant fraction 1.0072). File: `e0e1_code_origin.json`.

| feature group | `fix9000` m_p R² (linear / MLP / kNN5) | `init9000` m_p R² (linear / MLP / kNN5) |
|---|---:|---:|
| `position` | -0.0085 / -0.0085 / 0.3050 | 0.6815 / 0.7932 / 0.8855 |
| `hit_rate` | -0.0011 / -0.0014 / -0.1864 | -0.0006 / -0.0034 / -0.1945 |
| `charge` | -0.0021 / -0.0012 / -0.1792 | 0.0015 / -0.0028 / -0.1832 |
| `parity` | -0.0049 / -0.0054 / -0.3656 | -0.0028 / -0.0052 / -0.7585 |
| `module` | 0.0875 / 0.0702 / -0.0069 | 0.8378 / 0.8389 / 0.8291 |
| `raw_token` | 0.0023 / 0.1074 / 0.1865 | 0.7635 / 0.8529 / 0.8412 |
| `all` | 0.1043 / 0.4335 / 0.2177 | 0.8931 / 0.9067 / 0.8656 |

Reading (descriptive): the `fix9000` per-PMT code is **not identified by any single fixed PMT
attribute** — every single-group probe is at or below |R²| 0.09 linearly (module 0.0875), at or
below 0.11 with the MLP (module 0.0702, raw-token 0.1074), and at or below 0.31 with kNN (position
0.305, the local-neighbour structure already seen in the position addendum); hit rate, mean hit
ln q and parity are at zero, and the **combined** 120-feature probe reaches linear 0.1043, MLP
0.4335 and kNN 0.2177. So at most ~43 % of the invariant component's variance is predictable from
the PMT's fixed attributes, and none of the individual physics/geometry/readout attributes explains
it. The `init9000` reference behaves completely differently: module 0.8378 / 0.8389 / 0.8291,
raw-token 0.7635 / 0.8529 / 0.8412 and combined 0.8931 / 0.9067 / 0.8656 are all high (position
0.68 / 0.79 / 0.89), while hit rate, charge and parity stay at zero — i.e. the probes do have the
capacity to identify fixed per-PMT codes, and the `fix9000` code is a learned per-PMT code of other
origin, not a function of the PMT's fixed attributes. This is consistent with the H12 H2-dominant
verdict and with the position-probe addendum.
