# Part A - reading-level summary (2026-11-07 execution)

Full machine-readable payloads (x1.json 101 MB, x2.json 34 MB, x3.json 84 MB, readings.json 15 MB, auto README.md 10.6 MB) stay on `/ceph/dwong/jepa/returns/ext_repair_20261106/{x0,x1,x2,x3}` with the sha256s recorded in `LARGE_ARRAYS.json`; this committed summary and `SUMMARY.json` carry the reading-level results.

## X1 - drift cost, charge-scale share and post-refit residual

| family | seed | delta_D (m) [CI] | rho_SWtw | X1-b class | X1-g share_cs | X1-g class |
|---|---|---|---|---|---|---|
| fix | 9000 | 0.01521 [0.0135, 0.0169] | 1.0028801074287108 | charge-scale dominated | 0.9893564988616376 | None |
| fix | 9001 | 0.01659 [0.0149, 0.0184] | 1.0114164761011544 | charge-scale dominated | 0.9924389264110064 | None |
| frozen_target | 9000 | 0.02181 [0.0197, 0.0240] | 1.003287712761013 | charge-scale dominated | 1.0022306694579364 | None |
| frozen_target | 9001 | 0.01792 [0.0161, 0.0198] | 0.9978333079740663 | charge-scale dominated | 0.9957832489936345 | None |
| ijepa | 9000 | 0.01833 [0.0167, 0.0201] | 1.018468345107505 | charge-scale dominated | 1.0219646070675326 | None |
| ijepa | 9001 | 0.02165 [0.0196, 0.0238] | 1.0064984896166693 | charge-scale dominated | 0.9924721430909993 | None |
| init | 9000 | 0.01657 [0.0149, 0.0184] | 0.9944793343149229 | charge-scale dominated | 0.9899615679795581 | None |
| mae_across_trigger | 9000 | 0.00857 [0.0075, 0.0097] | 0.994802357054155 | charge-scale dominated | 1.0020802975842615 | None |
| mae_across_trigger | 9001 | 0.01719 [0.0159, 0.0187] | 0.998604994464385 | charge-scale dominated | 1.0184234714597573 | None |
| mae_within_median | 9000 | 0.01029 [0.0091, 0.0115] | 1.001883441034805 | charge-scale dominated | 1.011899161471919 | None |
| mae_within_median | 9001 | 0.01223 [0.0111, 0.0136] | 0.9905227521812597 | charge-scale dominated | 0.9820518669671636 | None |
| mae_within_trigger | 9000 | 0.00999 [0.0088, 0.0112] | 0.9874654998601331 | charge-scale dominated | 0.9655219635770047 | None |
| mae_within_trigger | 9001 | 0.01658 [0.0151, 0.0182] | 0.996049578965043 | charge-scale dominated | 0.9887533404812484 | None |
| static_target | 9000 | 0.02526 [0.0227, 0.0280] | 0.9993866577894799 | charge-scale dominated | 0.9898998304925308 | None |
| static_target | 9001 | 0.01729 [0.0157, 0.0190] | 1.0041322381609692 | charge-scale dominated | 1.0011457408911153 | None |
| t1_s030_across_trigger | 9000 | 0.00843 [0.0075, 0.0095] | 0.9956197637651626 | charge-scale dominated | 0.9895513318801225 | None |
| t1_s030_across_trigger | 9001 | 0.01186 [0.0107, 0.0132] | 1.0018275442790123 | charge-scale dominated | 1.0094562235253006 | None |
| t1_s030_within_trigger | 9000 | 0.01091 [0.0098, 0.0119] | 0.9872548986898794 | charge-scale dominated | 0.9654578177262231 | None |
| t1_s030_within_trigger | 9001 | 0.00985 [0.0088, 0.0110] | 1.0004712544379688 | charge-scale dominated | 1.0114691021373188 | None |

## X2 - equivariance and latent repair

| family | seed | R2_eq (occupancy, dz_cs) | X2-b class | rho_LR |
|---|---|---|---|---|
| fix | 9000 | 0.2514129671783629 | None | -0.8887818262789069 |
| fix | 9001 | 0.2880371343196978 | None | -0.8354556194642937 |
| frozen_target | 9000 | 0.7077025052837376 | None | -0.533840048787172 |
| frozen_target | 9001 | 0.727779335545238 | None | -0.5864968707636384 |
| init | 9000 | 0.7711081469884691 | None | -0.7046875991796347 |
| mae_across_trigger | 9000 | 0.1901406644641499 | None | -0.7192959824175895 |
| mae_across_trigger | 9001 | 0.22328999694044527 | None | -1.0228357203803662 |
| mae_within_trigger | 9000 | 0.6633901129102977 | None | -0.6364232658920607 |
| mae_within_trigger | 9001 | 0.43284828854879354 | None | -0.6304387006267047 |
| t1_s030_across_trigger | 9000 | 0.19230379889089821 | None | -1.2290302549342118 |
| t1_s030_across_trigger | 9001 | 0.20491442510576785 | None | -0.5486774170471186 |
| t1_s030_within_trigger | 9000 | 0.46685642084042944 | None | -0.500949910210284 |
| t1_s030_within_trigger | 9001 | 0.43235082144648507 | None | -0.7885105605081398 |

## X3-b - variants

| variant | rho_E log-yield | X3-1 log-yield | X3-1 energy | X3-4 verdict |
|---|---|---|---|---|
| raw | 3.5588538217086123 | price present | price present | calibration dominates |
| init9000 | 1.1887997172884603 | unclear | price present | mixed |

## X3-5 theory rows

```json
{
 "empirical_rho_E_log_yield": 3.5588538217086123,
 "theory_rows": {
  "C-mean": {
   "theory_ratio_lnY": null
  },
  "C-mean+occupancy": {
   "theory_ratio_lnY": 2.1238765687471672
  },
  "C-var_all": {
   "theory_ratio_lnY": 3.156166995051316
  },
  "C-var_ge1": {
   "theory_ratio_lnY": 3.3732389173806103
  },
  "C-var_ge3": {
   "theory_ratio_lnY": 4.133497536007928
  }
 },
 "rule": "C-mean alone gives \u221e by P0 and is listed as the charge-only limit, not as a comparison point"
}
```

