# PREDICTIONS_UPDATE — WP4a Tier-1 bridge

Generated 2026-10-29T07:54:53Z by `jepa_noise.bridge_run aggregate`; config sha256 `2f7b3c5eb0321e27dbb589a10fe84bfeda91ff0ec66eb26ad6c86bf5dc043a5f`; code `888596caaf5cbb5d89888807e1bc0a0c42adad8c`.

Registration: runbook v4 §3.5.9 B1 and the bridge half of R7. `PREDICTION_STATUS.json` is deliberately **not** edited here; the Lead merges this fragment.

**Labelled follow-up:** `toy_delay_20261025` (registered: false). These verdicts are reported beside the registered result and never replace it.

**Registered run:** `False`.  A registered bridge run is A4 data, full cohort ID ranges, 10000 scheduled steps and batch 128 (v4 §3.5.7, §6.1).  Smoke/truncated runs are marked `provisional` below.

Run outcomes: `wp4a_eval_sigreg_jepa_s0` = complete, `wp4a_eval_sigreg_jepa_s1` = complete, `wp4a_eval_sigreg_jepa_s2` = complete, `wp4a_eval_sigreg_jepa_anchored_s0` = complete, `wp4a_eval_sigreg_jepa_anchored_s1` = complete, `wp4a_eval_sigreg_jepa_anchored_s2` = complete, `wp4a_eval_mae_chi2_s0` = complete, `wp4a_eval_mae_chi2_s1` = complete, `wp4a_eval_mae_chi2_s2` = complete.

| ID | Comparison | Value | Criterion | Status |
|---|---|---|---|---|
| B1 | `sigreg_jepa` nuisance R² > `sigreg_jepa_anchored` | plain 0.432, anchored 0.6221, delta -0.1901, seed floor 0.03453 (2x = 0.06906) | every seed, |delta| > 2 x seed floor | **follow_up:fail** (fail; not_every_seed) |
| R7 | bridge half: anchored nuisance R² < plain and physics kept | nuisance delta -0.1901; ratios amplitude 0.843x, t0 2.01x, tau_t 0.861x | every paired seed, physics ratio <= 1.1 | **follow_up:fail** (fail; not_every_seed) |

Per-parameter physics MSE/CRLB (seed mean, anchored / plain):

| parameter | anchored | plain | ratio |
|---|---|---|---|
| amplitude | 1.079 | 1.281 | 0.843 |
| t0 | 98.72 | 49.22 | 2.01 |
| tau_t | 1.015 | 1.179 | 0.861 |

Arm seed means (nuisance R² over the defined targets):

| arm | seeds | nuisance R² mean | SD | random-init floor |
|---|---|---|---|---|
| `sigreg_jepa` | 3 | 0.432 | 0.01441 | -0.003142 |
| `sigreg_jepa_anchored` | 3 | 0.6221 | 0.03453 | -0.003142 |
| `mae_chi2` | 3 | 0.5146 | 0.04675 | -0.003142 |

Random-init floor per probe (same architecture, init seed = the arm's first registered seed; one model per arm, identical across its seeds):

| arm | nuisance R² | physics MSE/CRLB (amp / t0 / tau_t) | amplitude R² (post-time / tokens) |
|---|---|---|---|
| `sigreg_jepa` | -0.003142 | 4.044 / 1374 / 3.593 | 0.7383 / 0.7383 |
| `sigreg_jepa_anchored` | -0.003142 | 4.044 / 1374 / 3.593 | 0.7383 / 0.7383 |
| `mae_chi2` | -0.003142 | 4.044 / 1374 / 3.593 | 0.7383 / 0.7383 |

```json
{
  "B1": {
    "all_seeds_plain_greater": false,
    "anchored_mean": 0.6221167726959131,
    "delta_mean": -0.19008775187602017,
    "exceeds_floor": false,
    "exceeds_two_floors": false,
    "id": "B1",
    "metric": "nuisance_r2",
    "paired_difference_mean": -0.19008775187602014,
    "paired_difference_sd": 0.022369417857608154,
    "paired_differences": {
      "0": -0.17799561856888813,
      "1": -0.21590061112095488,
      "2": -0.1763670259382174
    },
    "per_seed_plain_greater": {
      "0": false,
      "1": false,
      "2": false
    },
    "plain_mean": 0.432029020819893,
    "reason": "not_every_seed",
    "registered": false,
    "replaces_registered_verdict": false,
    "rule": "v4 \u00a73.5.9 B1: plain nuisance R2 > anchored in every seed; v4 \u00a73.5.8 requires |delta mean| > 2 x the seed floor (the conservative reading of 'exceeds the seed-floor rule')",
    "seed_floor": 0.034530686567806335,
    "seeds": [
      0,
      1,
      2
    ],
    "status": "fail",
    "verdict_role": "descriptive_extension"
  },
  "R7_bridge": {
    "all_seeds_anchored_lower": false,
    "anchored_mean": 0.6221167726959131,
    "half": "bridge",
    "id": "R7",
    "metric": "nuisance_r2 (down) and physics_MSE_CRLB (kept)",
    "nuisance_delta": -0.19008775187602017,
    "paired_difference_mean": -0.19008775187602014,
    "paired_difference_sd": 0.022369417857608154,
    "paired_differences": {
      "0": -0.17799561856888813,
      "1": -0.21590061112095488,
      "2": -0.1763670259382174
    },
    "per_seed_anchored_lower": {
      "0": false,
      "1": false,
      "2": false
    },
    "physics": {
      "amplitude": {
        "anchored_seed_mean": 1.0789957540415285,
        "paired_ratio_mean": 0.842666039921481,
        "per_seed_ratio": {
          "0": 0.8223015030317266,
          "1": 0.8363139608346926,
          "2": 0.8693826558980239
        },
        "plain_seed_mean": 1.2806873019106084,
        "ratio": 0.8425130415768283,
        "within_tolerance": true
      },
      "t0": {
        "anchored_seed_mean": 98.71912721175612,
        "paired_ratio_mean": 2.0113675522408982,
        "per_seed_ratio": {
          "0": 2.5297238175721164,
          "1": 1.8981835669779124,
          "2": 1.606195272172666
        },
        "plain_seed_mean": 49.21604899431103,
        "ratio": 2.0058320248983668,
        "within_tolerance": false
      },
      "tau_t": {
        "anchored_seed_mean": 1.0150063647395744,
        "paired_ratio_mean": 0.8605902167522693,
        "per_seed_ratio": {
          "0": 0.8535798224311497,
          "1": 0.8589160856304014,
          "2": 0.8692747421952571
        },
        "plain_seed_mean": 1.1794360095292042,
        "ratio": 0.8605862094584807,
        "within_tolerance": true
      }
    },
    "physics_all_within_tolerance": false,
    "physics_tolerance": 1.1,
    "plain_mean": 0.432029020819893,
    "reason": "not_every_seed",
    "registered": false,
    "replaces_registered_verdict": false,
    "rule": "v4 \u00a74 R7 (bridge half): anchored nuisance R2 < plain in every paired seed and anchored seed-mean physics_MSE_CRLB per parameter <= 1.1 x plain",
    "seed_floor": 0.034530686567806335,
    "seeds": [
      0,
      1,
      2
    ],
    "status": "fail",
    "verdict_role": "descriptive_extension"
  },
  "registered_run": false
}
```

