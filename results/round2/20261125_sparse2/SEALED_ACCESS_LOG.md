# Sealed access log (r2-tier1)

Every read of a sealed pack, written by `scripts/tier1_eval.py` before the pack is opened.

| UTC | Commit | Run id @ step | Purpose | Host |
|---|---|---|---|---|
| 2026-11-25T13:35:07Z | 166e388e6a45 | wcte__sparse_mae__s9000__dw2__elin__dleg__qall@00000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T13:35:08Z | 166e388e6a45 | wcte__sparse_mae__s9000__dw2__elin__dleg__qall@10000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T14:00:44Z | f6f48f6d3f34 | wcte__sparse_mae__s9001__dw2__elin__dleg__qall@10000 | r2-sparse2_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-25T14:00:46Z | f6f48f6d3f34 | wcte__sparse_mae__s9001__dw2__elin__dleg__qall@00000 | r2-sparse2_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-25T14:01:49Z | 166e388e6a45 | wcte__sparse_mae__s9002__dw2__elin__dleg__qall@10000 | r2-sparse2_T7_score | deepthought |
| 2026-11-25T14:02:45Z | 166e388e6a45 | wcte__sparse_mae__s9002__dw2__elin__dleg__qall@00000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T14:16:59Z | 166e388e6a45 | wcte__sparse_time__s9000__dw2__elin__dfull@00000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T14:16:59Z | 166e388e6a45 | wcte__sparse_time__s9000__dw2__elin__dfull@10000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T14:32:41Z | 166e388e6a45 | wcte__sparse_time__s9000__dw2__elin__dfull__bt5@10000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T14:37:20Z | 166e388e6a45 | wcte__sparse_time__s9000__dw2__elin__dfull__bt5@00000 | r2-sparse2_T7_score | deepthought |
| 2026-11-25T14:57:51Z | f6f48f6d3f34 | wcte__sparse_time__s9001__dw2__elin__dfull@10000 | r2-sparse2_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-25T14:57:52Z | f6f48f6d3f34 | wcte__sparse_time__s9001__dw2__elin__dfull@00000 | r2-sparse2_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-25T14:57:53Z | f6f48f6d3f34 | wcte__sparse_time__s9001__dw2__elin__dfull__bt5@10000 | r2-sparse2_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-25T14:59:48Z | 166e388e6a45 | wcte__sparse_time__s9001__dw2__elin__dfull__bt5@00000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T15:01:02Z | 166e388e6a45 | wcte__sparse_time__s9002__dw2__elin__dfull@10000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T15:08:37Z | 166e388e6a45 | wcte__sparse_time__s9002__dw2__elin__dfull@00000 | r2-sparse2_T7_score | portal1 |
| 2026-11-25T15:25:20Z | f6f48f6d3f34 | wcte__sparse_time__s9002__dw2__elin__dfull__bt5@10000 | r2-sparse2_T7_score [batch job; read happens in the job] | deepthought |
| 2026-11-25T15:25:21Z | f6f48f6d3f34 | wcte__sparse_time__s9002__dw2__elin__dfull__bt5@00000 | r2-sparse2_T7_score [batch job; read happens in the job] | deepthought |
