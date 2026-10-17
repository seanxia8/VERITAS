# PREDICTIONS_UPDATE — WP4a Tier-1 bridge

Generated 2026-10-17T18:23:20Z by `jepa_noise.bridge_run grid`; config sha256 `2f7b3c5eb0321e27dbb589a10fe84bfeda91ff0ec66eb26ad6c86bf5dc043a5f`; code `cd33ff11c2ea056e5da543e9b264c9d59a56d533`.

Registration: runbook v4 §3.5.9 B1 and the bridge half of R7. `PREDICTION_STATUS.json` is deliberately **not** edited here; the Lead merges this fragment.

**Registered run:** `True`.  A registered bridge run is A4 data, full cohort ID ranges, 10000 scheduled steps and batch 128 (v4 §3.5.7, §6.1).  Smoke/truncated runs are marked `provisional` below.

Run outcomes: `wp4a_eval_mae_chi2_s0` = complete, `wp4a_eval_mae_chi2_s1` = complete, `wp4a_eval_mae_chi2_s2` = complete, `wp4a_eval_sigreg_jepa_anchored_s0` = complete, `wp4a_eval_sigreg_jepa_anchored_s1` = complete, `wp4a_eval_sigreg_jepa_anchored_s2` = complete, `wp4a_eval_sigreg_jepa_parallel_s0` = complete, `wp4a_eval_sigreg_jepa_parallel_s1` = complete, `wp4a_eval_sigreg_jepa_parallel_s2` = complete, `wp4a_eval_sigreg_jepa_s0` = complete, `wp4a_eval_sigreg_jepa_s1` = complete, `wp4a_eval_sigreg_jepa_s2` = complete.

| ID | Comparison | Value | Criterion | Status |
|---|---|---|---|---|
| B1 | `sigreg_jepa` nuisance R² > `sigreg_jepa_anchored` | plain 0.5251, anchored 0.5157, delta 0.009422, seed floor 0.05127 (2x = 0.1025) | every seed, |delta| > 2 x seed floor | **fail** (fail; not_every_seed) |
| R7 | bridge half: anchored nuisance R² < plain and physics kept | nuisance delta 0.009422; ratios amplitude 0.984x, t0 0.337x, tau_t 0.981x | every paired seed, physics ratio <= 1.1 | **fail** (fail; not_every_seed) |

Per-parameter physics MSE/CRLB (seed mean, anchored / plain):

| parameter | anchored | plain | ratio |
|---|---|---|---|
| amplitude | 1.147 | 1.165 | 0.984 |
| t0 | 3.159 | 9.374 | 0.337 |
| tau_t | 1.079 | 1.1 | 0.981 |

Arm seed means (nuisance R² over the defined targets):

| arm | seeds | nuisance R² mean | SD | random-init floor |
|---|---|---|---|---|
| `sigreg_jepa` | 3 | 0.5251 | 0.01569 | -0.003583 |
| `sigreg_jepa_anchored` | 3 | 0.5157 | 0.05127 | -0.003583 |
| `mae_chi2` | 3 | 0.4506 | 0.04854 | -0.003583 |
| `sigreg_jepa_parallel` | 3 | 0.5618 | 0.03847 | -0.003579 |

Random-init floor per probe (same architecture, init seed = the arm's first registered seed; one model per arm, identical across its seeds):

| arm | nuisance R² | physics MSE/CRLB (amp / t0 / tau_t) | amplitude R² (post-time / tokens) |
|---|---|---|---|
| `sigreg_jepa` | -0.003583 | 3.938 / 302.3 / 3.463 | 0.7376 / 0.7376 |
| `sigreg_jepa_anchored` | -0.003583 | 3.938 / 302.3 / 3.463 | 0.7376 / 0.7376 |
| `mae_chi2` | -0.003583 | 3.938 / 302.3 / 3.463 | 0.7376 / 0.7376 |
| `sigreg_jepa_parallel` | -0.003579 | 3.938 / 302.3 / 3.463 | 0.7376 / 0.7376 |

```json
{
  "B1": {
    "all_seeds_plain_greater": false,
    "anchored_mean": 0.5157053533004471,
    "delta_mean": 0.009422173573236714,
    "exceeds_floor": false,
    "exceeds_two_floors": false,
    "id": "B1",
    "metric": "nuisance_r2",
    "paired_difference_mean": 0.009422173573236789,
    "paired_difference_sd": 0.06449415861765509,
    "paired_differences": {
      "0": 0.0828000034753611,
      "1": -0.016254609207331017,
      "2": -0.038278873548319714
    },
    "per_seed_plain_greater": {
      "0": true,
      "1": false,
      "2": false
    },
    "plain_mean": 0.5251275268736838,
    "reason": "not_every_seed",
    "rule": "v4 \u00a73.5.9 B1: plain nuisance R2 > anchored in every seed; v4 \u00a73.5.8 requires |delta mean| > 2 x the seed floor (the conservative reading of 'exceeds the seed-floor rule')",
    "seed_floor": 0.051267563623404795,
    "seeds": [
      0,
      1,
      2
    ],
    "status": "fail"
  },
  "R7_bridge": {
    "all_seeds_anchored_lower": false,
    "anchored_mean": 0.5157053533004471,
    "half": "bridge",
    "id": "R7",
    "metric": "nuisance_r2 (down) and physics_MSE_CRLB (kept)",
    "nuisance_delta": 0.009422173573236714,
    "paired_difference_mean": 0.009422173573236789,
    "paired_difference_sd": 0.06449415861765509,
    "paired_differences": {
      "0": 0.0828000034753611,
      "1": -0.016254609207331017,
      "2": -0.038278873548319714
    },
    "per_seed_anchored_lower": {
      "0": true,
      "1": false,
      "2": false
    },
    "physics": {
      "amplitude": {
        "anchored_seed_mean": 1.1465412449627694,
        "paired_ratio_mean": 0.9872114013847845,
        "per_seed_ratio": {
          "0": 1.0561930444644176,
          "1": 0.9192670753962496,
          "2": 0.9861740842936865
        },
        "plain_seed_mean": 1.164606056139135,
        "ratio": 0.9844884791031798,
        "within_tolerance": true
      },
      "t0": {
        "anchored_seed_mean": 3.1591141399093705,
        "paired_ratio_mean": 0.33896526681679356,
        "per_seed_ratio": {
          "0": 0.34581957541548347,
          "1": 0.3697208216823289,
          "2": 0.30135540335256844
        },
        "plain_seed_mean": 9.374028039320576,
        "ratio": 0.3370071144078145,
        "within_tolerance": true
      },
      "tau_t": {
        "anchored_seed_mean": 1.0794992949095228,
        "paired_ratio_mean": 0.9805820859253259,
        "per_seed_ratio": {
          "0": 0.989417018415786,
          "1": 0.9956651296085022,
          "2": 0.9566641097516895
        },
        "plain_seed_mean": 1.1002702995788773,
        "ratio": 0.981121907337402,
        "within_tolerance": true
      }
    },
    "physics_all_within_tolerance": true,
    "physics_tolerance": 1.1,
    "plain_mean": 0.5251275268736838,
    "reason": "not_every_seed",
    "rule": "v4 \u00a74 R7 (bridge half): anchored nuisance R2 < plain in every paired seed and anchored seed-mean physics_MSE_CRLB per parameter <= 1.1 x plain",
    "seed_floor": 0.051267563623404795,
    "seeds": [
      0,
      1,
      2
    ],
    "status": "fail"
  },
  "registered_run": true
}
```

