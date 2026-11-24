# Recipe rebuilds and field-by-field diffs (r2-follow T0)

Built with `scripts/follow_recipes.py` from the exact submission CLI by the code that trains (WCTE: `code/follow_760c3d5d.tar.gz` = 9bfc364 + the §2.1 patch; SK: `code/mask_87628ef6.tar.gz`), on dt1, CPU, nothing trained. `snapshot_steps` (tuple in memory, list in JSON, equal values and hashes) is excluded. Full outputs in `recipes/`.

| Rebuild | vs reference | config SHA new / ref | fields that differ | initial weights vs reference step 0 | init state SHA |
|---|---|---|---|---|---|
| C2_s9000_vsC2 | `wcte__mae_repaired__s9000__dw2__lfixed__psent` (code 9bfc364a) | 7dc00f4f22 / 7dc00f4f22 | none | `jepa.*` equal (81/81) | a11d74bbb5 |
| C2dc_s9000_vsC2 | `wcte__mae_repaired__s9000__dw2__lfixed__psent` (code 9bfc364a) | bf342fd0d7 / 7dc00f4f22 | `r2_sparse_flags.count_head`: False → True, `r2_sparse_flags.decoder`: legacy → isolated (+ run_id) | `jepa.encoder.*` equal (51/51) | 20538c0f71 |
| C2dc_s9001_vsC2 | `wcte__mae_repaired__s9001__dw2__lfixed__psent` (code 9bfc364a) | 140805c4be / 609b7b7322 | `r2_sparse_flags.count_head`: False → True, `r2_sparse_flags.decoder`: legacy → isolated (+ run_id) | `jepa.encoder.*` equal (51/51) | f865c5daaa |
| C2dc_s9002_vsC2 | `wcte__mae_repaired__s9002__dw2__lfixed__psent` (code 9bfc364a) | f2fd2183ee / a42c3c05f5 | `r2_sparse_flags.count_head`: False → True, `r2_sparse_flags.decoder`: legacy → isolated (+ run_id) | `jepa.encoder.*` equal (51/51) | 597f35016f |
| C3_s9000_vsC3 | `wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier__disol__cnt` (code 9bfc364a) | 416356d62d / 416356d62d | none | `jepa.*` equal (84/84) | b55dbbb2a8 |
| C3e_s9000_vsC2 | `wcte__mae_repaired__s9000__dw2__lfixed__psent` (code 9bfc364a) | a248829790 / 7dc00f4f22 | `r2_sparse_flags.embed`: linear → fourier (+ run_id) | — | 916be130cc |
| C3e_s9000_wC3 | `wcte__mae_repaired__s9000__dw2__lfixed__psent__efourier__disol__cnt` (code 9bfc364a) | a248829790 / 416356d62d | `r2_sparse_flags.count_head`: True → False, `r2_sparse_flags.decoder`: isolated → legacy (+ run_id) | `jepa.encoder.*` equal (53/53) | 916be130cc |
| C3e_s9001_vsC2 | `wcte__mae_repaired__s9001__dw2__lfixed__psent` (code 9bfc364a) | 66f8b5c70e / 609b7b7322 | `r2_sparse_flags.embed`: linear → fourier (+ run_id) | — | 2ad3bbb7b5 |
| C3e_s9001_wC3 | `wcte__mae_repaired__s9001__dw2__lfixed__psent__efourier__disol__cnt` (code 9bfc364a) | 66f8b5c70e / 3e6ddcde10 | `r2_sparse_flags.count_head`: True → False, `r2_sparse_flags.decoder`: isolated → legacy (+ run_id) | `jepa.encoder.*` equal (53/53) | 2ad3bbb7b5 |
| C3e_s9002_vsC2 | `wcte__mae_repaired__s9002__dw2__lfixed__psent` (code 9bfc364a) | 8053985d52 / a42c3c05f5 | `r2_sparse_flags.embed`: linear → fourier (+ run_id) | — | c7b97c5568 |
| C3e_s9002_wC3 | `wcte__mae_repaired__s9002__dw2__lfixed__psent__efourier__disol__cnt` (code 9bfc364a) | 8053985d52 / bfc6737ad1 | `r2_sparse_flags.count_head`: True → False, `r2_sparse_flags.decoder`: isolated → legacy (+ run_id) | `jepa.encoder.*` equal (53/53) | c7b97c5568 |
| SKdef_s9000_vsLU | `sk__mae_repaired__s9000` (code b1cc7637) | fc4918ec31 / fc4918ec31 | none | `jepa.*` equal (81/81) | a11d74bbb5 |
| SKdef_s9001_vsT1 | `sk__mae_repaired__s9001` (code c26288f3) | b25dad2374 / b25dad2374 | none | `jepa.*` equal (81/81) | 8ea2d85709 |
| SKp173_s9002_vsA | `sk__mae_repaired__s9000` (code b1cc7637) | 945da2a56c / fc4918ec31 | `identity.seed`: 9000 → 9002 (+ run_id) | `jepa.*` differ (52/81) | cb40efeebe |
| SKp173_s9002_vsT1 | `sk__mae_repaired__s9001` (code c26288f3) | 945da2a56c / b25dad2374 | `identity.seed`: 9001 → 9002 (+ run_id) | `jepa.*` differ (52/81) | cb40efeebe |
| SKp43_s9000_vsMK | `sk__mae_repaired__s9000__ps43` (code ed3a047f) | 408e813024 / 408e813024 | none | `jepa.*` equal (81/81) | a11d74bbb5 |
| SKp43_s9002_vsA | `sk__mae_repaired__s9000__ps43` (code ed3a047f) | 4914809913 / 408e813024 | `identity.seed`: 9000 → 9002 (+ run_id) | `jepa.*` differ (52/81) | cb40efeebe |
| SKp692_s9000_vsMK | `sk__mae_repaired__s9000__ps692` (code ed3a047f) | 5313b388f4 / 5313b388f4 | none | `jepa.*` equal (81/81) | a11d74bbb5 |
| SKp692_s9002_vsA | `sk__mae_repaired__s9000__ps692` (code ed3a047f) | 10817f4f6d / 5313b388f4 | `identity.seed`: 9000 → 9002 (+ run_id) | `jepa.*` differ (52/81) | cb40efeebe |
| SKrnd_s9000_vsT1 | `sk__mae_repaired__s9000__mrandom` (code c26288f3) | 027314f1a5 / 027314f1a5 | none | `jepa.*` equal (81/81) | a11d74bbb5 |
| SKrnd_s9001_vsT1 | `sk__mae_repaired__s9001__mrandom` (code c26288f3) | 3de17e1c31 / 3de17e1c31 | none | `jepa.*` equal (81/81) | 8ea2d85709 |
| SKrnd_s9002_vsA | `sk__mae_repaired__s9000__mrandom` (code c26288f3) | 3fc9e94f14 / 027314f1a5 | `identity.seed`: 9000 → 9002 (+ run_id) | `jepa.*` differ (52/81) | cb40efeebe |
