# LUCiD-arm detector-response cells on the toy detector — DEVELOPMENT, NOT CITABLE

**12 October 2026.** Produced by

```
PYTHONPATH=src .venv/bin/python -m noise_module_lucid --response-toy \
    --out <scratch> --n-events 200 --toy-photons 800
```

(`summary.json` is copied here; per-cell digits, ~21 MB, are regenerable and not committed.)

**What this is.** The five detector-response N families (`noise_module_lucid/response.py`) applied
through the declared readout chain (`readout.py`) to **toy** flasher events (`toy.py`: 482-PMT
WCTE-scale cylinder, isotropic point source, Poisson light yield with mean 800 photons, 1.5 ns TTS,
15 % diffuse late light). Each cell moves one readout parameter and is paired with the reference
event by event. K comes from the frozen, independent endpoints of `consequence.py`: a time-of-flight
vertex fit and a reference-calibrated total-charge energy, each the ratio of the mean error to the
reference on the same 200 events (paired bootstrap 95 % intervals). Draft primary
`K = max(K_vertex, K_energy)`; κ_m = 1.10 `provisional_dev` (decision D-K, `PREREGISTRATION.md` §8
item 17). Reference: mean vertex error 0.177 m, mean relative energy error 0.028.

**What this is not.** Not LUCiD (gate A0 open), not a detector simulation (no Cherenkov cone,
absorption or reflections), not a monitor result — no model was trained and neither claim was
evaluated. Crates are the package's contiguous grouping of 64 channels (8 groups on this toy, the
last 34 channels; group 6 spans both caps) — not a physical crate map; `module_loss@4` kills groups
{0, 2, 4, 6}, the same set the per-crate drifts act on (~53 % of the channels).

| cell | K_vertex [95 %] | K_energy [95 %] | K_energy, dead-map corrected [95 %] | primary K (uncorr. / corr.) | mean digits |
|---|---|---|---|---|---|
| `reference` | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 / 1.00 | 356 |
| `N_struct:efficiency_loss@0.95` | 1.00 [0.96, 1.04] | 1.33 [1.19, 1.47] | 1.33 [1.19, 1.47] | 1.33 / 1.33 | 352 |
| `N_struct:efficiency_loss@0.9` | 0.98 [0.93, 1.02] | 2.07 [1.82, 2.35] | 2.07 [1.82, 2.35] | 2.07 / 2.07 | 348 |
| `N_struct:efficiency_loss@0.8` | 1.02 [0.97, 1.07] | 3.93 [3.46, 4.40] | 3.93 [3.46, 4.40] | 3.93 / 3.93 | 338 |
| `N_struct:efficiency_loss@0.6` | 1.00 [0.96, 1.05] | 7.83 [6.99, 8.69] | 7.83 [6.99, 8.69] | 7.83 / 7.83 | 314 |
| `N_struct:gain_drift@0.8` | 0.99 [0.96, 1.02] | 3.91 [3.44, 4.36] | 3.91 [3.44, 4.36] | 3.91 / 3.91 | 356 |
| `N_struct:gain_drift@0.9` | 0.99 [0.97, 1.01] | 2.04 [1.78, 2.32] | 2.04 [1.78, 2.32] | 2.04 / 2.04 | 356 |
| `N_struct:gain_drift@1.1` | 1.00 [0.98, 1.02] | 2.02 [1.79, 2.26] | 2.02 [1.79, 2.26] | 2.02 / 2.02 | 356 |
| `N_struct:gain_drift@1.25` | 0.99 [0.97, 1.02] | 4.89 [4.36, 5.48] | 4.89 [4.36, 5.48] | 4.89 / 4.89 | 356 |
| `N_struct:timing_drift@0.5` | 1.02 [0.97, 1.07] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.02 / 1.02 | 356 |
| `N_struct:timing_drift@1` | 1.03 [0.98, 1.08] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.03 / 1.03 | 356 |
| `N_struct:timing_drift@2` | 1.17 [1.10, 1.25] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.17 / 1.17 | 356 |
| `N_struct:timing_drift@5` | 2.06 [1.93, 2.20] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 2.06 / 2.06 | 356 |
| `N_struct:threshold_walk@0.35` | 1.01 [0.97, 1.05] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.01 / 1.01 | 355 |
| `N_struct:threshold_walk@0.5` | 1.04 [0.99, 1.08] | 1.00 [0.99, 1.02] | 1.00 [0.99, 1.02] | 1.04 / 1.04 | 351 |
| `N_struct:threshold_walk@0.75` | 1.06 [1.00, 1.12] | 1.17 [1.08, 1.26] | 1.17 [1.08, 1.26] | 1.17 / 1.17 | 331 |
| `N_struct:threshold_walk@1` | 1.09 [1.03, 1.14] | 2.27 [1.98, 2.56] | 2.27 [1.98, 2.56] | 2.27 / 2.27 | 291 |
| `N_struct:module_loss@1` | 1.03 [0.98, 1.09] | 3.67 [3.24, 4.15] | 1.79 [1.58, 2.00] | 3.67 / 1.79 | 314 |
| `N_struct:module_loss@2` | 1.03 [0.97, 1.09] | 8.71 [7.76, 9.66] | 1.30 [1.18, 1.45] | 8.71 / 1.30 | 265 |
| `N_struct:module_loss@4` | 1.13 [1.07, 1.21] | 19.54 [17.53, 21.62] | 1.62 [1.43, 1.84] | 19.54 / 1.62 | 165 |

**Readings (development, with their limits).**

- *Timing drift* moves only the vertex (K_vertex 1.17 at 2 ns, 2.06 at 5 ns; energy exactly 1). Its
  grid straddles κ_m (0.5 and 1 ns below).
- *Threshold walk* moves both endpoints and loses small hits (291 vs 356 digits at 1 pe); its grid
  straddles κ_m (0.35 and 0.5 pe below).
- *Gain drift and efficiency loss* move only the energy scale, and **every severity is above κ_m** —
  the grids do **not** straddle κ_m on this toy, contrary to `PREREGISTRATION.md` §8 item 3. Reason:
  K_energy ≈ |scale bias| / resolution, and at 2.8 % resolution a 2.5 % bias (5 % on half the
  crates) already doubles the error. The grids need weaker points (sub-percent) once the
  resolution of the real arm is known.
- *Module loss* is dominated by the **uncorrected** charge deficit of the dead crates (K_energy 3.7,
  8.7, 19.5). With the known-dead-map correction (charge scaled by the reference alive share — what
  an experiment would apply) it is 1.8, 1.3, 1.6: still harmful, but **no longer monotone in
  severity**, so its harm ranking depends on D-K item 17(c). It also moves the vertex at 4 crates
  (K_vertex 1.14 [1.07, 1.21]).
- **Caveat for Claim 2.** For gain drift and efficiency loss, K_energy is a pure global charge-scale
  bias (one rescale returns it to ≈ 1); any monitor that sees the summed charge will rank these
  cells. They make Claim 2 *non-empty* on this arm but not *hard*; the timing and threshold families
  are where a representation score has to work.
- Whether a generic monitor already sees these changes (Claim 1) is not tested here.
