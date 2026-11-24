# SEALED_ACCESS_LOG — r2-mask

Copy of the canonical log `/ceph/dwong/jepa/returns/r2_mask_2026-11-23/SEALED_ACCESS_LOG.md` (its header still says r2-tier1 because `scripts/tier1_eval.py` writes that header; the file itself belongs to r2-mask). Every line was written by `scripts/tier1_eval.py` (pinned, READOUT_SPEC_v1) before a sealed pack was opened — on the CPU host for WCTE, and on Deepthought (`--log-only`) before submission for the SK A100 jobs, whose own local copies are filed under `batch_access_logs/` in the returns. No other code read a sealed pack. First sealed read: 2026-11-23T19:54:18Z (after the PREREG push at 2026-11-23T19:50Z).

| UTC | Commit | Run id @ step | Purpose | Host |
|---|---|---|---|---|
| 2026-11-23T19:54:18Z | e280fb73d75f | wcte__mae_repaired__s9001__mrandom@10000 | r2-mask M2 score wcte_rnd_B | deepthought |
| 2026-11-23T19:54:39Z | e280fb73d75f | wcte__mae_repaired__s9001@10000 | r2-mask M2 score wcte_def_B | portal1 |
| 2026-11-23T19:54:39Z | e280fb73d75f | wcte__mae_repaired__s9002@10000 | r2-mask M2 score wcte_def_C | portal1 |
| 2026-11-23T19:54:39Z | e280fb73d75f | wcte__mae_repaired__s9000@10000 | r2-mask M2 score wcte_def_A | portal1 |
| 2026-11-23T19:54:40Z | e280fb73d75f | wcte__mae_repaired__s9000__mrandom@10000 | r2-mask M2 score wcte_rnd_A | portal1 |
| 2026-11-23T20:35:37Z | 30ce5d22b9f3 | wcte__mae_repaired__s9001__mrandom@05000 | r2-mask M2 score wcte_rnd_B | deepthought |
| 2026-11-23T20:47:11Z | 30ce5d22b9f3 | wcte__mae_repaired__s9001@05000 | r2-mask M2 score wcte_def_B | portal1 |
| 2026-11-23T20:47:15Z | 30ce5d22b9f3 | wcte__mae_repaired__s9000@05000 | r2-mask M2 score wcte_def_A | portal1 |
| 2026-11-23T20:47:49Z | 30ce5d22b9f3 | wcte__mae_repaired__s9000__mrandom@05000 | r2-mask M2 score wcte_rnd_A | portal1 |
| 2026-11-23T20:49:07Z | 30ce5d22b9f3 | wcte__mae_repaired__s9002@05000 | r2-mask M2 score wcte_def_C | portal1 |
| 2026-11-23T21:16:36Z | 30ce5d22b9f3 | wcte__mae_repaired__s9002__mrandom@10000 | r2-mask M2 score wcte_rnd_C | deepthought |
| 2026-11-23T21:39:29Z | 30ce5d22b9f3 | wcte__mae_repaired__s9001@00000 | r2-mask M2 score wcte_def_B | portal1 |
| 2026-11-23T21:39:30Z | 30ce5d22b9f3 | wcte__mae_repaired__s9000@00000 | r2-mask M2 score wcte_def_A | portal1 |
| 2026-11-23T21:41:29Z | 30ce5d22b9f3 | wcte__mae_repaired__s9000__ps8@10000 | r2-mask M2 score wcte_ps8_A | portal1 |
| 2026-11-23T21:43:02Z | 30ce5d22b9f3 | wcte__mae_repaired__s9002@00000 | r2-mask M2 score wcte_def_C | portal1 |
| 2026-11-23T21:57:50Z | 30ce5d22b9f3 | wcte__mae_repaired__s9002__mrandom@05000 | r2-mask M2 score wcte_rnd_C | deepthought |
| 2026-11-23T22:33:31Z | 87628ef6eca3 | wcte__mae_repaired__s9001__ps8@10000 | r2-mask M2 score wcte_ps8_B | portal1 |
| 2026-11-23T22:33:31Z | 87628ef6eca3 | wcte__mae_repaired__s9001__ps128@10000 | r2-mask M2 score wcte_ps128_B | portal1 |
| 2026-11-23T22:34:16Z | 87628ef6eca3 | wcte__mae_repaired__s9000__ps8@05000 | r2-mask M2 score wcte_ps8_A | portal1 |
| 2026-11-23T22:36:01Z | 87628ef6eca3 | wcte__mae_repaired__s9001__ps256@10000 | r2-mask M2 score wcte_ps256_B | portal1 |
| 2026-11-23T22:39:13Z | 87628ef6eca3 | wcte__mae_repaired__s9000__ps128@10000 | r2-mask M2 score wcte_ps128_A | deepthought |
| 2026-11-23T23:20:16Z | 87628ef6eca3 | wcte__mae_repaired__s9000__ps128@05000 | r2-mask M2 score wcte_ps128_A | deepthought |
| 2026-11-23T23:27:01Z | 87628ef6eca3 | wcte__mae_repaired__s9001__ps8@05000 | r2-mask M2 score wcte_ps8_B | portal1 |
| 2026-11-23T23:27:30Z | 87628ef6eca3 | wcte__mae_repaired__s9001__ps128@05000 | r2-mask M2 score wcte_ps128_B | portal1 |
| 2026-11-23T23:27:38Z | 87628ef6eca3 | wcte__mae_repaired__s9002__ps256@10000 | r2-mask M2 score wcte_ps256_C | portal1 |
| 2026-11-23T23:28:18Z | 87628ef6eca3 | wcte__mae_repaired__s9001__ps256@05000 | r2-mask M2 score wcte_ps256_B | portal1 |
| 2026-11-24T00:01:48Z | 87628ef6eca3 | wcte__mae_repaired__s9000__ps256@10000 | r2-mask M2 score wcte_ps256_A | deepthought |
| 2026-11-24T00:19:34Z | 87628ef6eca3 | wcte__mae_repaired__s9002__ps8@10000 | r2-mask M2 score wcte_ps8_C | portal1 |
| 2026-11-24T00:20:05Z | 87628ef6eca3 | wcte__mae_repaired__s9002__ps128@10000 | r2-mask M2 score wcte_ps128_C | portal1 |
| 2026-11-24T00:23:30Z | 87628ef6eca3 | wcte__mae_repaired__s9002__ps256@05000 | r2-mask M2 score wcte_ps256_C | portal1 |
| 2026-11-24T00:44:30Z | 87628ef6eca3 | wcte__mae_repaired__s9000__ps256@05000 | r2-mask M2 score wcte_ps256_A | deepthought |
| 2026-11-24T00:50:01Z | 87628ef6eca3 | sk__mae_repaired__s9000@10000 | r2-mask_M2_SK_default_A [batch job; read happens in the job] | deepthought |
| 2026-11-24T00:50:01Z | 87628ef6eca3 | sk__mae_repaired__s9000@05000 | r2-mask_M2_SK_default_A [batch job; read happens in the job] | deepthought |
| 2026-11-24T00:50:01Z | 87628ef6eca3 | sk__mae_repaired__s9000@00000 | r2-mask_M2_SK_default_A [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:08:45Z | 87628ef6eca3 | sk__mae_repaired__s9000__mrandom@10000 | r2-mask_M2_SK_random_A [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:08:45Z | 87628ef6eca3 | sk__mae_repaired__s9000__mrandom@05000 | r2-mask_M2_SK_random_A [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:15:07Z | 87628ef6eca3 | sk__mae_repaired__s9000__ps43@10000 | r2-mask_M2_SK_ps43_A [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:15:21Z | 87628ef6eca3 | wcte__mae_repaired__s9002__ps8@05000 | r2-mask M2 score wcte_ps8_C | portal1 |
| 2026-11-24T01:16:16Z | 87628ef6eca3 | wcte__mae_repaired__s9002__ps128@05000 | r2-mask M2 score wcte_ps128_C | portal1 |
| 2026-11-24T01:23:29Z | 87628ef6eca3 | sk__mae_repaired__s9001@10000 | r2-mask_M2_SK_default_B [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:23:29Z | 87628ef6eca3 | sk__mae_repaired__s9001@05000 | r2-mask_M2_SK_default_B [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:23:29Z | 87628ef6eca3 | sk__mae_repaired__s9001@00000 | r2-mask_M2_SK_default_B [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:49:52Z | 87628ef6eca3 | sk__mae_repaired__s9001__mrandom@10000 | r2-mask_M2_SK_random_B [batch job; read happens in the job] | deepthought |
| 2026-11-24T01:49:53Z | 87628ef6eca3 | sk__mae_repaired__s9001__mrandom@05000 | r2-mask_M2_SK_random_B [batch job; read happens in the job] | deepthought |
| 2026-11-24T02:06:16Z | 87628ef6eca3 | sk__mae_repaired__s9000__ps692@10000 | r2-mask_M2_SK_ps692_A [batch job; read happens in the job] | deepthought |
| 2026-11-24T02:16:37Z | 87628ef6eca3 | sk__mae_repaired__s9001__ps43@10000 | r2-mask_M2_SK_ps43_B [batch job; read happens in the job] | deepthought |
| 2026-11-24T02:36:59Z | 87628ef6eca3 | sk__mae_repaired__s9001__ps692@10000 | r2-mask_M2_SK_ps692_B [batch job; read happens in the job] | deepthought |

47 reads. Batch-job local logs: `meval_sk_def_A_759995.md`, `meval_sk_def_B_760001.md`, `meval_sk_ps43_A_759999.md`, `meval_sk_ps43_B_760008.md`, `meval_sk_ps692_A_760007.md`, `meval_sk_ps692_B_760011.md`, `meval_sk_rnd_A_759998.md`, `meval_sk_rnd_B_760004.md`.
