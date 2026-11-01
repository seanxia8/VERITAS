# GATE G-DESCOPE — what to do with the paused registered LUCiD grid (31 Oct 2026)

_Author decision required. One page. Commit as `docs/jepa/GATE_GRID_DESCOPE_2026-10-31.md`._

## State

**Grid status.**
- The registered A10 grid has 165 runs at ladder rung 1 (3 seeds).
- It was paused on 23 Oct for the label-shortcut diagnosis.
- The 25 runs completed with the old labels are superseded.
- The ledger counts 18 valid completed runs and 147 remaining. The projection for the remainder is
  **50.6 L40S GPU-h**, at the measured 1 238 s per run.
- The L40S ledger stands at **97.2 / 170**.

**Predictions the grid would evaluate.** R6, R7, R7b, R8, R11 and R11b are `not_run`. R12 is
`not_evaluable` by the frozen detectability check (F3 ceiling 0.32 < 0.6).

## What the extensions now predict for each

All of the following evidence is `registered: false`. Sources: `20261029_*`, `20261030_*` and
`2b91f42`.

| Prediction | Requires | Evidence | Expected registered outcome |
|---|---|---|---|
| R6 (the split decides the gain trap) | `within` propagated gain R² clearly above its floor and above `across` | Deep gain R² ≈ 0, **below** the random-init floor (0.20–0.22) in both splits (Part C/D). Linear CCA: no gain direction. ν\*_gain ≤ 0.0006 at s = 0.05 and 0.10 | fail |
| R7 (the anchor removes drift and keeps physics) | plain encodes gain, and the anchor lowers it | plain gain R² ≈ 0, so there is nothing to remove. The anchor needs β at the edge of the grid to stay within 1.1× physics | fail or uninformative |
| R7b, R8 | absorbed drift in plain | none is absorbed | uninformative |
| R11, R11b | F1 R² ≥ 0.5 from the encoder | trained encoders destroy gain information (strong drift: 0.10 / 0.22 against a floor of 0.84) | R11 fail; R11b `not_evaluable` |

## Options

**A. Descope completely.**
- Run no grid.
- Record R6–R11b as `not_run`, with reason "author descope (gate G-DESCOPE, date)".
- Report the extension evidence beside them.
- Cost: 0 GPU-h.
- Risk: the pre-registered core predictions get no registered verdict.

**B. Run a minimal registered core** — *recommended*.
- **Scope:**
  - `sigreg_jepa` at `within` and `across`, each at s = 0.05 and 0.10;
  - `sigreg_jepa_anchored` at `within`, s = 0.10;
  - 3 seeds each, with the frozen grid config and fixed labels.

  That is 15 runs, fewer if any of the 18 valid runs match.
- **What it evaluates:** R6 and R7, the two central predictions, under their registered criteria.
  Everything else is recorded as in A.
- **Cost:** about **5–6 L40S GPU-h**, plus data generation for any cell not already on disk.
- **Why recommended.** It turns the paper's central claim into a pre-registered test. The
  diagnostics predicted, **before** these runs, that R6 and R7 fail. A registered failure then
  becomes evidence that the diagnostics work, not just a null.

**C. Run the full grid as registered.**
- 147 runs, about 50.6 L40S GPU-h. The ledger would end near 148 / 170.
- On current evidence it adds little beyond B, at about 9× the cost.

## What follows from each option

- In every option, the 25 superseded old-label runs stay as history, and nothing registered is
  overwritten.
- Under B, the extension predictions must be written into `DECISIONS.md` **before** the first run,
  quoting this gate:
  - R6 and R7 fail;
  - plain gain R² is below its random-init floor in every cell.

Reply with **A**, **B** or **C**, plus any change in scope.
