# Sealed access log (r2-tier1)

Every read of a sealed pack, written by `scripts/tier1_eval.py` before the pack is opened.

| UTC | Commit | Run id @ step | Purpose | Host |
|---|---|---|---|---|
| 2026-11-22T19:46:51Z | 3bfb0888b6e9 | sk__mae_repaired__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T19:46:51Z | 3bfb0888b6e9 | sk__mae_repaired__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T19:47:43Z | 3bfb0888b6e9 | wcte__mae_repaired__s9001__mrandom@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T19:47:43Z | 3bfb0888b6e9 | wcte__mae_repaired__s9000__mrandom@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T19:47:43Z | 3bfb0888b6e9 | wcte__mae_repaired__s9000__mrandom__ts450@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T19:47:43Z | 3bfb0888b6e9 | wcte__mae_repaired__s9000__mrandom__ts1p5@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T19:47:43Z | 3bfb0888b6e9 | wcte__mae_repaired__s9000__mrandom__ts5@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T19:47:44Z | 3bfb0888b6e9 | wcte__mae_repaired__s9001__mrandom__ts450@10000 | T7_E-cell_score | deepthought |
| 2026-11-22T19:47:44Z | 3bfb0888b6e9 | wcte__mae_repaired__s9001__mrandom__ts5@10000 | T7_E-cell_score | deepthought |
| 2026-11-22T20:15:18Z | 87bca71e95e0 | sk__mae_repaired__s9000__mrandom__ts450@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T20:16:20Z | 87bca71e95e0 | sk__mae_repaired__s9000__ts450@10000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T20:16:20Z | 87bca71e95e0 | sk__mae_repaired__s9000__ts450@05000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T20:16:31Z | 87bca71e95e0 | sk__mae_repaired__s9000@05000 | F5 snapshot score [batch job; read happens in the job] | deepthought |
| 2026-11-22T20:23:24Z | 87bca71e95e0 | wcte__mae_repaired__s9001__ts450@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T20:23:24Z | 87bca71e95e0 | wcte__mae_repaired__s9001__ts5@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T20:27:04Z | 87bca71e95e0 | wcte__mae_repaired__s9002__ts450@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T20:27:05Z | 87bca71e95e0 | wcte__mae_repaired__s9000@10000 | T7_E-cell_score_and_T3_rescore | portal1 |
| 2026-11-22T20:27:05Z | 87bca71e95e0 | wcte__mae_repaired__s9002__ts5@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T20:28:03Z | 87bca71e95e0 | wcte__mae_repaired__s9001@10000 | T7_E-cell_score_and_T3_rescore | deepthought |
| 2026-11-22T20:28:27Z | 87bca71e95e0 | wcte__mae_repaired__s9000__ts450@10000 | T7_E-cell_score_and_T3_rescore | deepthought |
| 2026-11-22T21:01:45Z | 48a15550a53d | wcte__mae_repaired__s9000__ts5@10000 | T7_E-cell_score_and_T3_rescore | portal1 |
| 2026-11-22T21:02:35Z | 48a15550a53d | wcte__mae_repaired__s9000__ts1p5@10000 | T7_E-cell_score_and_T3_rescore | portal1 |
| 2026-11-22T21:03:39Z | 48a15550a53d | wcte__mae_repaired__s9000__ts15@10000 | T7_E-cell_score_and_T3_rescore | portal1 |
| 2026-11-22T21:03:40Z | 48a15550a53d | wcte__mae_mse__s9000@10000 | T3_rescore_manifest_A | portal1 |
| 2026-11-22T21:04:02Z | 48a15550a53d | wcte__mae_mse__s9001@10000 | T3_rescore_manifest_A | portal1 |
| 2026-11-22T21:06:31Z | 48a15550a53d | sk__mae_repaired__s9000__mrandom@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:10:34Z | 48a15550a53d | sk__mae_repaired__s9001__mrandom__ts450@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:10:50Z | 48a15550a53d | wcte__jepa_repaired__s9000@10000 | T3_rescore_manifest_A | deepthought |
| 2026-11-22T21:11:07Z | d6d5d0be50eb | wcte__jepa_repaired__s9001@10000 | T3_rescore_manifest_A | deepthought |
| 2026-11-22T21:16:07Z | d6d5d0be50eb | sk__mae_repaired__s9001__ts450@10000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:16:07Z | d6d5d0be50eb | sk__mae_repaired__s9001__ts450@05000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:17:40Z | d6d5d0be50eb | sk__mae_repaired__s9001__mrandom@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:20:48Z | d6d5d0be50eb | sk__mae_repaired__s9001@10000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:20:49Z | d6d5d0be50eb | sk__mae_repaired__s9001@05000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:01Z | d6d5d0be50eb | sk__mae_mse__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:01Z | d6d5d0be50eb | sk__mae_mse__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:02Z | d6d5d0be50eb | sk__jepa_repaired__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:03Z | d6d5d0be50eb | sk__jepa_repaired__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:04Z | d6d5d0be50eb | sk__sigreg_jepa__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:04Z | d6d5d0be50eb | sk__sigreg_jepa__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:06Z | d6d5d0be50eb | sk__init__s9000@00000 | T3 rescore manifest A legacy init [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:21:07Z | d6d5d0be50eb | sk__init__s9001@00000 | T3 rescore manifest A legacy init [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:22Z | d6d5d0be50eb | juno__mae_repaired__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:22Z | d6d5d0be50eb | juno__mae_repaired__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:25Z | d6d5d0be50eb | juno__jepa_repaired__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:25Z | d6d5d0be50eb | juno__jepa_repaired__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:27Z | d6d5d0be50eb | juno__init__s9000@00000 | T3 rescore manifest A legacy init [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:29Z | d6d5d0be50eb | juno__init__s9001@00000 | T3 rescore manifest A legacy init [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:31Z | d6d5d0be50eb | hk__mae_repaired__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:31Z | d6d5d0be50eb | hk__mae_repaired__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:33Z | d6d5d0be50eb | hk__jepa_repaired__s9000@10000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:33Z | d6d5d0be50eb | hk__jepa_repaired__s9000@00000 | T3 rescore manifest A (matched init) [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:36Z | d6d5d0be50eb | hk__init__s9000@00000 | T3 rescore manifest A legacy init [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:26:37Z | d6d5d0be50eb | hk__init__s9001@00000 | T3 rescore manifest A legacy init [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:37:35Z | d6d5d0be50eb | wcte__sigreg_jepa__s9000@10000 | T3_rescore_manifest_A | portal1 |
| 2026-11-22T21:37:36Z | d6d5d0be50eb | wcte__init__s9000@00000 | T3_rescore_manifest_A_legacy_init | portal1 |
| 2026-11-22T21:44:45Z | d6d5d0be50eb | wcte__init__s9001@00000 | T3_rescore_manifest_A_legacy_init | portal1 |
| 2026-11-22T21:45:20Z | d6d5d0be50eb | wcte__mae_repaired__s9000@00000 | T3_rescore_manifest_A_matched_init | portal1 |
| 2026-11-22T21:45:40Z | d6d5d0be50eb | wcte__mae_repaired__s9001@00000 | T3_rescore_manifest_A_matched_init | portal1 |
| 2026-11-22T21:52:06Z | d6d5d0be50eb | sk__mae_repaired__s9000__ts5@10000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:52:06Z | d6d5d0be50eb | sk__mae_repaired__s9000__ts5@05000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:54:50Z | d6d5d0be50eb | wcte__mae_mse__s9000@00000 | T3_rescore_manifest_A_matched_init | deepthought |
| 2026-11-22T21:55:08Z | d6d5d0be50eb | wcte__mae_mse__s9001@00000 | T3_rescore_manifest_A_matched_init | deepthought |
| 2026-11-22T21:56:47Z | d6d5d0be50eb | sk__mae_repaired__s9000__mwithin__ts450@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:59:02Z | d6d5d0be50eb | sk__mae_repaired__s9001__ts5@10000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T21:59:03Z | d6d5d0be50eb | sk__mae_repaired__s9001__ts5@05000 | T7 E-cell score and F5 [batch job; read happens in the job] | deepthought |
| 2026-11-22T22:09:11Z | d6d5d0be50eb | sk__mae_repaired__s9000__ts15@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T22:10:48Z | d6d5d0be50eb | sk__mae_repaired__s9000__ts150@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T22:15:44Z | d6d5d0be50eb | sk__mae_repaired__s9000__cw100@10000 | T7 E-cell score [batch job; read happens in the job] | deepthought |
| 2026-11-22T22:19:43Z | d6d5d0be50eb | wcte__jepa_repaired__s9000@00000 | T3_rescore_manifest_A_matched_init | portal1 |
| 2026-11-22T22:25:06Z | 21986a7e9710 | wcte__jepa_repaired__s9001@00000 | T3_rescore_manifest_A_matched_init | portal1 |
| 2026-11-22T22:25:07Z | 21986a7e9710 | wcte__sigreg_jepa__s9000@00000 | T3_rescore_manifest_A_matched_init | portal1 |
| 2026-11-22T22:36:31Z | 21986a7e9710 | wcte__mae_repaired__s9000__mwithin__ts5@10000 | T7_E-cell_score | deepthought |
| 2026-11-22T22:37:03Z | 21986a7e9710 | wcte__mae_repaired__s9001__mrandom__ts1p5@10000 | T7_E-cell_score | deepthought |
| 2026-11-22T22:50:02Z | 21986a7e9710 | wcte__mae_repaired__s9001__ts1p5@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T23:00:52Z | 21986a7e9710 | wcte__mae_repaired__s9002__mrandom@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T23:00:56Z | 21986a7e9710 | wcte__mae_repaired__s9002__ts15@10000 | T7_E-cell_score | portal1 |
| 2026-11-22T23:18:16Z | 21986a7e9710 | wcte__mae_repaired__s9002__ts1p5@10000 | T7_E-cell_score | deepthought |
| 2026-11-23T18:38:28Z | 21986a7e9710 | wcte__mae_repaired__s9000__trmedian@10000 | T7_E-cell_score | deepthought |
| 2026-11-23T18:38:28Z | 21986a7e9710 | wcte__mae_repaired__s9001__ts15@10000 | T7_E-cell_score | deepthought |
| 2026-11-23T18:39:39Z | 21986a7e9710 | wcte__mae_repaired__s9000__cw100@10000 | T7_E-cell_score | portal1 |
| 2026-11-23T18:39:39Z | 21986a7e9710 | wcte__mae_repaired__s9000__mwithin__ts450@10000 | T7_E-cell_score | portal1 |
| 2026-11-23T18:49:02Z | 7678d3c402ca | wcte__mae_repaired__s9002@10000 | T7_E-cell_score | portal1 |
| 2026-11-23T18:49:02Z | 7678d3c402ca | wcte__mae_repaired__s9002__mrandom__ts1p5@10000 | T7_E-cell_score | portal1 |
| 2026-11-23T18:49:02Z | 7678d3c402ca | wcte__mae_repaired__s9002__mrandom__ts450@10000 | T7_E-cell_score | portal1 |
| 2026-11-23T19:16:45Z | 79a08ee69597 | wcte__mae_repaired__s9002__mrandom__ts5@10000 | T7_E-cell_score | portal1 |
