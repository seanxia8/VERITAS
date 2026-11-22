# INTERIM_E1 — r2-tier1 registered contrasts (sealed_test, step 10,000)

Primary metric: mean 3-D vertex error on sealed photon bombs, frozen spatial-moment Ridge readout (READOUT_SPEC_v1). T = e(cell 2)/e(cell 1) − 1 per seed; T̄ its seed mean; SE_boot paired bootstrap (2,000 resamples of sealed events, shared by all runs); SE_seed = SD_pooled/√n_seeds; N_σ = |T̄|/max(SE_boot, SE_seed); clear = N_σ ≥ 2 and one sign in every seed.

| Contrast | T̄ | per seed | SE_boot | SE_seed | N_σ | clear | missing runs |
|---|---:|---|---:|---:|---:|---|---|
| CT-p50 | -3.3 % | A -0.8 %, B -5.9 % | 0.0033 | 0.0237 | 1.40 | no | – |
| CT-r50 | -2.0 % | A +1.9 %, B -5.8 % | 0.0027 | 0.0237 | 0.83 | no | – |
| CM-450 | -12.7 % | A -11.7 %, B -13.8 % | 0.0038 | 0.0237 | 5.37 | yes | – |
| CM-50 | -13.9 % | A -14.0 %, B -13.8 % | 0.0036 | 0.0237 | 5.87 | yes | – |

SD_pooled per detector: {"sk": 0.0335}

## Secondary outcomes and probe variants (same contrasts)

| Contrast | μ direction (median angle) | e direction (median angle) | e energy 68 % half-width | μ energy 68 % half-width | μ mis-ID at 99 % e eff. | vertex, MLP (spatial_moment_full, 3 seeds) | vertex, MLP (latent, 3 seeds) | vertex, attentive (3 seeds) |
|---|---|---|---|---|---|---|---|---|
| CT-p50 | -4.2 % (4.6*) | -7.2 % (2.9*) | -3.4 % (0.9) | +1.6 % (0.4) | – | -8.8 % (20.9*) | -7.1 % (2.6*) | -9.6 % (4.8*) |
| CT-r50 | +3.2 % (3.5*) | +1.2 % (0.5) | +2.7 % (0.7) | +4.6 % (1.3) | +180.2 % (0.8) | +0.3 % (0.6) | -0.1 % (0.0) | -8.5 % (4.2*) |
| CM-450 | -13.3 % (14.5*) | -18.0 % (7.2*) | +4.6 % (1.2) | +1.0 % (0.3) | -100.0 % (0.7) | -6.3 % (14.9*) | -6.8 % (2.5*) | -17.6 % (8.7*) |
| CM-50 | -19.5 % (21.3*) | -24.8 % (10.0*) | -1.5 % (0.4) | -1.8 % (0.5) | -100.0 % (0.7) | -14.7 % (35.0*) | -13.5 % (4.9*) | -18.6 % (9.2*) |

Cells: T̄ (N_σ); * = clear.


## Status at this interim (22 Nov, 22:30 UTC)

SK-1 … SK-7 and the existing (patch, 50, A) run are scored on `sealed_test` (step 10,000). CT-p5 waits for SK-8/9 (trained, scoring). No registered SK contrast is in the §2.2 borderline band (1.5 ≤ N_σ < 2.5) at this point; the band is re-checked on the final table. The mask contrasts are clear (patch better than random by ≈13 % at both 450 and 50 ns); the 450 → 50 ns contrasts are small and not clear, with opposite seed signs for the random mask. Nothing here is a gate verdict.
