# Sealed access log (r2-sparse)

Canonical copy: `/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/SEALED_ACCESS_LOG.md` (appended under a lock by the writing script; tier1_eval.py writes the metric reads). Passes before `shards_w2b` were superseded (no metrics in any).

- 2026-11-23T23:38:10Z | c3a00e463fb5 | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-23T23:44:43Z | 3d6cfd5b824e | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-23T23:53:24Z | 63ea330ec583 | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-24T00:01:01Z | 78c7b2244de6 | wcte sealed_test runs 0:600 | W2 re-pack from the lucid_v1 LUCiD cache, window [-110, +50] ns (packing only, no metrics) | scripts/sparse_w2_repack.py
- 2026-11-24T00:06:18Z | ddde5ca2990d | wcte sealed_test | W2 structural validation v2 vs v1 shards and cache (event ids, truths, hit identity, triggers, dark counts; no model metrics) | scripts/sparse_w2_pack.py
- 2026-11-24T00:10:44Z | 1a75f3b47e8f | wcte sealed_test | pack lucid_v1_w2 sealed shards into wcte_lucid_v1_w2_sealed.h5 (packing + byte verification only, no metrics) | scripts/sparse_w2_pack.py
| 2026-11-24T00:48:18Z | 9bfc364a5fb9 | wcte__mae_repaired__s9000@00000 | r2-sparse_T7_C0_score | portal1 |
| 2026-11-24T00:48:18Z | 9bfc364a5fb9 | wcte__mae_repaired__s9001@10000 | r2-sparse_T7_C0_score | portal1 |
| 2026-11-24T00:48:18Z | 9bfc364a5fb9 | wcte__mae_repaired__s9000@10000 | r2-sparse_T7_C0_score | portal1 |
| 2026-11-24T01:22:03Z | 9bfc364a5fb9 | wcte__mae_repaired__s9001@00000 | r2-sparse_T7_C0_score | deepthought |
| 2026-11-24T01:44:56Z | 9bfc364a5fb9 | wcte__mae_repaired__s9002@00000 | r2-sparse_T7_C0_score | portal1 |
| 2026-11-24T01:44:56Z | 9bfc364a5fb9 | wcte__mae_repaired__s9002@10000 | r2-sparse_T7_C0_score | portal1 |
| 2026-11-24T01:44:56Z | 9bfc364a5fb9 | wcte__mae_repaired__s9000__dw2__lfixed__psent@10000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T01:53:47Z | 9bfc364a5fb9 | wcte__mae_repaired__s9000__dw2__lfixed__psent@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T02:08:40Z | 9bfc364a5fb9 | wcte__mae_repaired__s9001__dw2__lfixed__psent@10000 | r2-sparse_T7_score | deepthought |
| 2026-11-24T02:46:07Z | 9bfc364a5fb9 | wcte__mae_repaired__s9001__dw2__lfixed__psent@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T02:46:07Z | 9bfc364a5fb9 | wcte__mae_repaired__s9001__dw2__lfixed__psent__efourier__disol__cnt@10000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T02:46:23Z | 9bfc364a5fb9 | wcte__mae_repaired__s9001__dw2__lfixed__psent__efourier__disol__cnt@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T02:51:59Z | 9bfc364a5fb9 | wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier__disol__cnt@10000 | r2-sparse_T7_score | deepthought |
| 2026-11-24T02:53:32Z | 9bfc364a5fb9 | wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier__disol__cnt@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T03:36:24Z | 9bfc364a5fb9 | wcte__sparse_mae__s9001__dw2__qall@10000 | r2-sparse_T7_score | deepthought |
- 2026-11-24T03:39:51Z | bbff85a4d088 | wcte sealed_test | stage byte copy of wcte_lucid_v1_w2_sealed.h5 to /ceph/srv/dwong/r2/data/lucid_v1_w2/ for batch scoring (copy only, no read of contents) | cp
| 2026-11-24T03:41:00Z | bbff85a4d088 | wcte__sparse_mae__s9001__dw2__qall@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T03:42:19Z | bbff85a4d088 | wcte__sparse_time__s9001__dw2@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T03:42:20Z | bbff85a4d088 | wcte__sparse_time__s9001__dw2@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T03:46:04Z | bbff85a4d088 | wcte__mae_repaired__s9002__dw2__lfixed__psent@10000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T03:46:46Z | bbff85a4d088 | wcte__mae_repaired__s9002__dw2__lfixed__psent@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T03:51:18Z | bbff85a4d088 | wcte__mae_repaired__s9002__dw2__lfixed__psent__efourier__disol__cnt@10000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T04:08:55Z | bbff85a4d088 | wcte__mae_repaired__s9002__dw2__lfixed__psent__efourier__disol__cnt@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:08:56Z | bbff85a4d088 | wcte__sparse_mae__s9000__dw2__qall@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:08:59Z | bbff85a4d088 | wcte__sparse_mae__s9000__dw2__qall@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:09:00Z | bbff85a4d088 | wcte__sparse_mae__s9002__dw2__qall@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:15:31Z | bbff85a4d088 | wcte__sparse_mae__s9002__dw2__qall@00000 | r2-sparse_T7_score | deepthought |
| 2026-11-24T04:34:32Z | bbff85a4d088 | wcte__sparse_time__s9002__dw2@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:34:33Z | bbff85a4d088 | wcte__sparse_time__s9002__dw2@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:34:35Z | bbff85a4d088 | wcte__mae_repaired__s9001__dw2@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:34:36Z | bbff85a4d088 | wcte__mae_repaired__s9001__dw2@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T04:38:23Z | bbff85a4d088 | wcte__sparse_mae__s9001__dw2__qpos@10000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T04:43:17Z | bbff85a4d088 | wcte__sparse_mae__s9001__dw2__qpos@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T04:43:18Z | bbff85a4d088 | wcte__sparse_time__s9000__dw2@10000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T04:45:44Z | bbff85a4d088 | wcte__sparse_time__s9000__dw2@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T04:54:44Z | bbff85a4d088 | wcte__mae_repaired__s9002__dw2@10000 | r2-sparse_T7_score | deepthought |
| 2026-11-24T05:26:51Z | bbff85a4d088 | wcte__mae_repaired__s9002__dw2@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T05:26:53Z | bbff85a4d088 | wcte__sparse_mae__s9002__dw2__qpos@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T05:26:56Z | bbff85a4d088 | wcte__sparse_mae__s9002__dw2__qpos@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T05:26:57Z | bbff85a4d088 | wcte__mae_repaired__s9000__dw2@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T05:39:30Z | bbff85a4d088 | wcte__mae_repaired__s9000__dw2@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T05:39:47Z | bbff85a4d088 | wcte__sparse_mae__s9001__dw2__qneg@10000 | r2-sparse_T7_score | deepthought |
| 2026-11-24T05:42:03Z | bbff85a4d088 | wcte__sparse_mae__s9001__dw2__qneg@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T05:44:34Z | bbff85a4d088 | wcte__sparse_mae__s9002__dw2__qneg@10000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T05:46:11Z | bbff85a4d088 | wcte__sparse_mae__s9002__dw2__qneg@00000 | r2-sparse_T7_score | portal1 |
| 2026-11-24T05:54:20Z | bbff85a4d088 | wcte__sparse_mae__s9000__dw2__qpos@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T05:54:22Z | bbff85a4d088 | wcte__sparse_mae__s9000__dw2__qpos@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T06:24:34Z | 06f9567d91bb | wcte__sparse_mae__s9000__dw2__qneg@10000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-24T06:24:37Z | 06f9567d91bb | wcte__sparse_mae__s9000__dw2__qneg@00000 | r2-sparse_T7_score [batch job; read happens in the job] | deepthought |
