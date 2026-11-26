# r2-sparse-sk T1.4 — recipe check and leakage test (e)

`scripts/sparse_sk_recipe_check.sh` (code bdc4e8a, dt1, nice 15): each recipe rebuilt from its submission CLI with this phase's code and compared with the reference run's `recipe.json` (every field) and `snapshot_step_00000.pt` (every state-dict key, bitwise). Reference runs are read-only.

## Earlier recipes (must be unchanged)

| Check | rebuilt SHA (16) | reference SHA (16) | equal | field diffs | step-0 weights equal | init SHA (12) |
|---|---|---|---|---|---|---|
| C2_s9000 | 7dc00f4f227aeaf8 | 7dc00f4f227aeaf8 | True | 0 | True | a11d74bbb539 |
| C2_s9001 | 609b7b732228d039 | 609b7b732228d039 | True | 0 | True | 8ea2d8570921 |
| C2_s9002 | a42c3c05f51b68d1 | a42c3c05f51b68d1 | True | 0 | True | cb40efeebe96 |
| C2dc_s9000 | bf342fd0d7f50a7d | bf342fd0d7f50a7d | True | 0 | True | 20538c0f71de |
| C3_s9000 | 416356d62dbbed38 | 416356d62dbbed38 | True | 0 | True | b55dbbb2a8d8 |
| C3e_s9000 | a24882979061eac3 | a24882979061eac3 | True | 0 | True | 916be130cca4 |
| S1L_s9000 | effb5f2b8b47522c | effb5f2b8b47522c | True | 0 | True | 8a5b650460cc |
| S1L_s9001 | d3660f4954cd2175 | d3660f4954cd2175 | True | 0 | True | bce87b385221 |
| S1L_s9002 | 81c73be98bf02226 | 81c73be98bf02226 | True | 0 | True | eaaf7b4962ef |
| S1_s9000 | 6a69a2ebeb78d916 | 6a69a2ebeb78d916 | True | 0 | True | 31a26259fb6c |
| S1_s9001 | cd0984bc8e109029 | cd0984bc8e109029 | True | 0 | True | abc141057ed2 |
| S1_s9002 | d51bea9c34ad8206 | d51bea9c34ad8206 | True | 0 | True | 117a386a1c26 |
| S1n_s9000 | 2e44f086beb3fac1 | 2e44f086beb3fac1 | True | 0 | True | 31a26259fb6c |
| S1p_s9000 | b43aa2d52c9b917d | b43aa2d52c9b917d | True | 0 | True | 31a26259fb6c |
| S2_s9000 | 1eb8318bfb40b71d | 1eb8318bfb40b71d | True | 0 | True | 4968b9bc016b |
| S2_s9001 | b953931cb156db4f | b953931cb156db4f | True | 0 | True | 860e8ee73968 |
| S2_s9002 | b0aa1c8d9ac347bf | b0aa1c8d9ac347bf | True | 0 | True | 6403ee1461d8 |
| SKD0_s9000 | fc4918ec31dee88b | fc4918ec31dee88b | True | 0 | True | a11d74bbb539 |
| SKD0_s9001 | b25dad2374f03a62 | b25dad2374f03a62 | True | 0 | True | 8ea2d8570921 |
| SKD0_s9002 | 945da2a56c424970 | 945da2a56c424970 | True | 0 | True | cb40efeebe96 |

## New recipes (for the record; diffs expected only in identity and flags)

| Check | reference | field diffs | differing fields | shared step-0 keys equal | init SHA (12) |
|---|---|---|---|---|---|
| SKC2_s9000 | `sk__mae_repaired__s9000` | 7 | identity.run_id, r2_sparse_flags.count_head, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.loss_mode, r2_sparse_flags.placeholder, r2_sparse_flags.sentinel_time | 83/83 | a11d74bbb539 |
| SKC2_s9001 | `sk__mae_repaired__s9001` | 7 | identity.run_id, r2_sparse_flags.count_head, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.loss_mode, r2_sparse_flags.placeholder, r2_sparse_flags.sentinel_time | 83/83 | 8ea2d8570921 |
| SKC2_s9002 | `sk__mae_repaired__s9002` | 7 | identity.run_id, r2_sparse_flags.count_head, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.loss_mode, r2_sparse_flags.placeholder, r2_sparse_flags.sentinel_time | 83/83 | cb40efeebe96 |
| SKS1A_s9000 | `sk__mae_repaired__s9000` | 7 | identity.arm, identity.run_id, r2_sparse_flags.absence_patch_size, r2_sparse_flags.absence_tokens, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.queries | 2/83 | 34c2d97e4069 |
| SKS1A_s9001 | `sk__mae_repaired__s9001` | 7 | identity.arm, identity.run_id, r2_sparse_flags.absence_patch_size, r2_sparse_flags.absence_tokens, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.queries | 2/83 | 34ae49d09a7f |
| SKS1A_s9002 | `sk__mae_repaired__s9002` | 7 | identity.arm, identity.run_id, r2_sparse_flags.absence_patch_size, r2_sparse_flags.absence_tokens, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.queries | 2/83 | a305d71037c6 |
| SKS1L_s9000 | `sk__mae_repaired__s9000` | 5 | identity.arm, identity.run_id, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.queries | 2/83 | 8a5b650460cc |
| SKS1L_s9001 | `sk__mae_repaired__s9001` | 5 | identity.arm, identity.run_id, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.queries | 2/83 | bce87b385221 |
| SKS1L_s9002 | `sk__mae_repaired__s9002` | 5 | identity.arm, identity.run_id, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.queries | 2/83 | eaaf7b4962ef |
| SKS2L_s9000 | `sk__mae_repaired__s9000` | 5 | identity.arm, identity.run_id, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.mask_seed | 0/83 | c35ecd380332 |
| SKS2L_s9001 | `sk__mae_repaired__s9001` | 5 | identity.arm, identity.run_id, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.mask_seed | 0/83 | c791efbd787f |
| SKS2L_s9002 | `sk__mae_repaired__s9002` | 5 | identity.arm, identity.run_id, r2_sparse_flags.decoder, r2_sparse_flags.embed, r2_sparse_flags.mask_seed | 0/83 | 93db7548a46f |
| WS1A_s9000 | `wcte__sparse_mae__s9000__dw2__elin__dleg__qall` | 3 | identity.run_id, r2_sparse_flags.absence_patch_size, r2_sparse_flags.absence_tokens | 82/82 | 34c2d97e4069 |
| WS1A_s9001 | `wcte__sparse_mae__s9001__dw2__elin__dleg__qall` | 3 | identity.run_id, r2_sparse_flags.absence_patch_size, r2_sparse_flags.absence_tokens | 82/82 | 34ae49d09a7f |
| WS1A_s9002 | `wcte__sparse_mae__s9002__dw2__elin__dleg__qall` | 3 | identity.run_id, r2_sparse_flags.absence_patch_size, r2_sparse_flags.absence_tokens | 82/82 | a305d71037c6 |

**Leakage test (e):** SK-C2 differs from SK-D0 (same seed) only in `identity.run_id` and the `r2_sparse_flags` block that `--loss fixed --placeholder sentinel` creates (loss_mode, placeholder, sentinel_time = 0.7111, embed/decoder/count_head at their legacy values); encoder, scaling, training, data identity, mask seed and view mode are identical, and the initial weights are bitwise identical (all keys). SK-S1L/S1A/S2L vs SK-D0 differ in `identity.arm`, run id and `r2_sparse_flags` only (step-0 weights not comparable: different module names). W-S1A vs WCTE S1L: only run id and the two absence flags; every S1L key equal, plus the three absence parameters.
