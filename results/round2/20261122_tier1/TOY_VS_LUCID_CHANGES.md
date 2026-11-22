# TOY_VS_LUCID rebuild (r2-tier1 T1.5, defects D1 and D7)

Rebuilt with the fixed `scripts/lucid_toy_vs_lucid.py` (runs keyed by full run id; main tables take default-recipe runs only; suffixed runs in a separate sweep table) from the unchanged r2-lucid `final.json` records. The `20261121_lucid` copy is left as committed. No ToF line was requested in this rebuild; the builder's optional line is now labelled "crude fit, not a bound".

Only the WCTE `mae_repaired` seed-9000 row changed: the old row was the `__ts5` sweep run (key collision, §0.3 fact 8).

| Cell (WCTE, mae_repaired, s9000, LUCiD) | Old (= ts5 run) | Rebuilt (default run) |
|---|---:|---:|
| Spatial Ridge (m) | 0.0501 | **0.0521** |
| Gain over init (m) | 0.0155 | 0.0135 |
| Spatial MLP (m) | 0.0513 | 0.0465 |
| All-mean Ridge (m) | 0.1218 | 0.1410 |
| Token time R² | 0.983 | 0.971 |
| f_event | 0.1594 | 0.1852 |
| f_layout | 0.639 | 0.548 |
| μ direction MLP (°) | 10.29 | 10.20 |
| e direction MLP (°) | 7.78 | 8.34 |

New sweep table rows: `__ts450` 0.0541, `__ts15` 0.0512, `__ts5` 0.0501, `__ts1p5` 0.0485 m (spatial Ridge).
