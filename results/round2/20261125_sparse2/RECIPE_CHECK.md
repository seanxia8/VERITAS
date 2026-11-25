# r2-sparse2 T0 — reused runs (read-only) and T1.6 recipe check

Reused runs from `/ceph/dwong/jepa/returns/r2_sparse_2026-11-24/train/runs/` (read in place; never written). All trained by code 9bfc364a (staged batch code `code/sparse_9bfc364a.tar.gz`; dt2 lane runs from the r2-sparse clone at 9bfc364).

| Cell | Seed | Run directory | config SHA-256 (16) | code SHA | host | step | complete |
|---|---|---|---|---|---|---|---|
| C2 | 9000 | `wcte__mae_repaired__s9000__dw2__lfixed__psent` | 7dc00f4f227aeaf8 | 9bfc364a5fb9 | deepthought2 | 10000 | True |
| C2 | 9001 | `wcte__mae_repaired__s9001__dw2__lfixed__psent` | 609b7b732228d039 | 9bfc364a | dwong-759989.0-c03-001-167.gridka.de | 10000 | True |
| C2 | 9002 | `wcte__mae_repaired__s9002__dw2__lfixed__psent` | a42c3c05f51b68d1 | 9bfc364a | dwong-760002.0-c03-001-175.gridka.de | 10000 | True |
| S1 | 9000 | `wcte__sparse_mae__s9000__dw2__qall` | 6a69a2ebeb78d916 | 9bfc364a5fb9 | deepthought2 | 10000 | True |
| S1 | 9001 | `wcte__sparse_mae__s9001__dw2__qall` | cd0984bc8e109029 | 9bfc364a | dwong-759991.0-c03-001-175.gridka.de | 10000 | True |
| S1 | 9002 | `wcte__sparse_mae__s9002__dw2__qall` | d51bea9c34ad8206 | 9bfc364a | dwong-760005.0-c03-001-175.gridka.de | 10000 | True |
| S2 | 9000 | `wcte__sparse_time__s9000__dw2` | 1eb8318bfb40b71d | 9bfc364a5fb9 | deepthought2 | 10000 | True |
| S2 | 9001 | `wcte__sparse_time__s9001__dw2` | b953931cb156db4f | 9bfc364a | dwong-759992.0-c03-001-175.gridka.de | 10000 | True |
| S2 | 9002 | `wcte__sparse_time__s9002__dw2` | b0aa1c8d9ac347bf | 9bfc364a | dwong-760006.0-c03-001-175.gridka.de | 10000 | True |

## Recipe check (`scripts/sparse2_recipe_check.sh`, code 11bbaf2 + script fix, dt1)

Each existing recipe was rebuilt from its submission CLI with this phase's code and compared with the reference run's `recipe.json` (every field) and its `snapshot_step_00000.pt` (every state-dict key, bitwise). Existing cells: config hash equal, no field difference, initial weights identical. New cells (S1L vs C2, S2L/S2Lt vs S2 of the same seed): differences only in `identity` and `r2_sparse_flags`; encoder, scaling, training, data identity, mask seed, window and LUCiD SHA are identical. Reference runs for C3e and C2dc are the r2-follow runs (`r2_follow_2026-11-24/train/runs`).

| Check | rebuilt config SHA (16) | reference config SHA (16) | equal | field diffs | step-0 weights equal | init state SHA (12) |
|---|---|---|---|---|---|---|
| C2_s9000 | 7dc00f4f227aeaf8 | 7dc00f4f227aeaf8 | True | 0 | True | a11d74bbb539 |
| C2_s9001 | 609b7b732228d039 | 609b7b732228d039 | True | 0 | True | 8ea2d8570921 |
| C2_s9002 | a42c3c05f51b68d1 | a42c3c05f51b68d1 | True | 0 | True | cb40efeebe96 |
| C2dc_s9000 | bf342fd0d7f50a7d | bf342fd0d7f50a7d | True | 0 | True | 20538c0f71de |
| C3_s9000 | 416356d62dbbed38 | 416356d62dbbed38 | True | 0 | True | b55dbbb2a8d8 |
| C3e_s9000 | a24882979061eac3 | a24882979061eac3 | True | 0 | True | 916be130cca4 |
| S1L_s9000 | effb5f2b8b47522c | 7dc00f4f227aeaf8 | False | 7 | n/a | 8a5b650460cc |
| S1L_s9001 | d3660f4954cd2175 | 609b7b732228d039 | False | 7 | n/a | bce87b385221 |
| S1L_s9002 | 81c73be98bf02226 | a42c3c05f51b68d1 | False | 7 | n/a | eaaf7b4962ef |
| S1_s9000 | 6a69a2ebeb78d916 | 6a69a2ebeb78d916 | True | 0 | True | 31a26259fb6c |
| S1_s9001 | cd0984bc8e109029 | cd0984bc8e109029 | True | 0 | True | abc141057ed2 |
| S1_s9002 | d51bea9c34ad8206 | d51bea9c34ad8206 | True | 0 | True | 117a386a1c26 |
| S1n_s9000 | 2e44f086beb3fac1 | 2e44f086beb3fac1 | True | 0 | True | 31a26259fb6c |
| S1p_s9000 | b43aa2d52c9b917d | b43aa2d52c9b917d | True | 0 | True | 31a26259fb6c |
| S2L_s9000 | d9c88bd379e51d1e | 1eb8318bfb40b71d | False | 3 | n/a | c35ecd380332 |
| S2L_s9001 | 05ec056657f7a928 | b953931cb156db4f | False | 3 | n/a | c791efbd787f |
| S2L_s9002 | 92f8cbdc1ac6400f | b0aa1c8d9ac347bf | False | 3 | n/a | 93db7548a46f |
| S2Lt_s9000 | fc3593350bbe04fd | 1eb8318bfb40b71d | False | 5 | n/a | 4426844100cb |
| S2Lt_s9001 | 71c0b673ba5601f7 | b953931cb156db4f | False | 5 | n/a | ffdfb19c0c5a |
| S2Lt_s9002 | 850e9603dca627dd | b0aa1c8d9ac347bf | False | 5 | n/a | 605affe0668b |
| S2_s9000 | 1eb8318bfb40b71d | 1eb8318bfb40b71d | True | 0 | True | 4968b9bc016b |
| S2_s9001 | b953931cb156db4f | b953931cb156db4f | True | 0 | True | 860e8ee73968 |
| S2_s9002 | b0aa1c8d9ac347bf | b0aa1c8d9ac347bf | True | 0 | True | 6403ee1461d8 |
