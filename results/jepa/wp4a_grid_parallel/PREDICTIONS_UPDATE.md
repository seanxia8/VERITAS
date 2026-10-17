# PREDICTIONS_UPDATE — WP4a Tier-1 bridge

Generated 2026-10-17T18:04:30Z by `jepa_noise.bridge_run grid`; config sha256 `2f7b3c5eb0321e27dbb589a10fe84bfeda91ff0ec66eb26ad6c86bf5dc043a5f`; code `f10c0347ead6ca0262ef217a8687b6afeedd3594`.

Registration: runbook v4 §3.5.9 B1 and the bridge half of R7. `PREDICTION_STATUS.json` is deliberately **not** edited here; the Lead merges this fragment.

**Registered run:** `True`.  A registered bridge run is A4 data, full cohort ID ranges, 10000 scheduled steps and batch 128 (v4 §3.5.7, §6.1).  Smoke/truncated runs are marked `provisional` below.

Run outcomes: `wp4a_eval_sigreg_jepa_parallel_s0` = complete, `wp4a_eval_sigreg_jepa_parallel_s1` = complete, `wp4a_eval_sigreg_jepa_parallel_s2` = complete.

| ID | Comparison | Value | Criterion | Status |
|---|---|---|---|---|
| B1 | `sigreg_jepa` nuisance R² > `sigreg_jepa_anchored` | plain n/a, anchored n/a, delta n/a, seed floor n/a (2x = n/a) | every seed, |delta| > 2 x seed floor | **not_evaluable** (not_evaluable; missing_required_nuisance_target_in_the_declared_set) |
| R7 | bridge half: anchored nuisance R² < plain and physics kept | nuisance delta n/a; ratios amplitude n/ax, t0 n/ax, tau_t n/ax | every paired seed, physics ratio <= 1.1 | **not_evaluable** (not_evaluable; missing_required_nuisance_target_in_the_declared_set) |

Per-parameter physics MSE/CRLB (seed mean, anchored / plain):

| parameter | anchored | plain | ratio |
|---|---|---|---|
| amplitude | n/a | n/a | n/a |
| t0 | n/a | n/a | n/a |
| tau_t | n/a | n/a | n/a |

Arm seed means (nuisance R² over the defined targets):

| arm | seeds | nuisance R² mean | SD | random-init floor |
|---|---|---|---|---|
| `sigreg_jepa_parallel` | 3 | 0.5618 | 0.03847 | -0.003579 |

Random-init floor per probe (same architecture, init seed = the arm's first registered seed; one model per arm, identical across its seeds):

| arm | nuisance R² | physics MSE/CRLB (amp / t0 / tau_t) | amplitude R² (post-time / tokens) |
|---|---|---|---|
| `sigreg_jepa_parallel` | -0.003579 | 3.938 / 302.3 / 3.463 | 0.7376 / 0.7376 |

```json
{
  "B1": {
    "all_seeds_plain_greater": false,
    "anchored_mean": null,
    "delta_mean": null,
    "exceeds_floor": false,
    "exceeds_two_floors": false,
    "id": "B1",
    "metric": "nuisance_r2",
    "paired_difference_mean": null,
    "paired_difference_sd": null,
    "paired_differences": {},
    "per_seed_plain_greater": {},
    "plain_mean": null,
    "reason": "missing_required_nuisance_target_in_the_declared_set",
    "rule": "v4 \u00a73.5.9 B1: plain nuisance R2 > anchored in every seed; v4 \u00a73.5.8 requires |delta mean| > 2 x the seed floor (the conservative reading of 'exceeds the seed-floor rule')",
    "seed_floor": null,
    "seeds": [],
    "status": "not_evaluable"
  },
  "R7_bridge": {
    "all_seeds_anchored_lower": false,
    "anchored_mean": null,
    "half": "bridge",
    "id": "R7",
    "metric": "nuisance_r2 (down) and physics_MSE_CRLB (kept)",
    "nuisance_delta": null,
    "paired_difference_mean": null,
    "paired_difference_sd": null,
    "paired_differences": {},
    "per_seed_anchored_lower": {},
    "physics": {
      "amplitude": {
        "anchored_seed_mean": null,
        "per_seed_ratio": {},
        "plain_seed_mean": null,
        "ratio": null,
        "within_tolerance": null
      },
      "t0": {
        "anchored_seed_mean": null,
        "per_seed_ratio": {},
        "plain_seed_mean": null,
        "ratio": null,
        "within_tolerance": null
      },
      "tau_t": {
        "anchored_seed_mean": null,
        "per_seed_ratio": {},
        "plain_seed_mean": null,
        "ratio": null,
        "within_tolerance": null
      }
    },
    "physics_all_within_tolerance": false,
    "physics_tolerance": 1.1,
    "plain_mean": null,
    "reason": "missing_required_nuisance_target_in_the_declared_set",
    "rule": "v4 \u00a74 R7 (bridge half): anchored nuisance R2 < plain in every paired seed and anchored seed-mean physics_MSE_CRLB per parameter <= 1.1 x plain",
    "seed_floor": 0.03847147964412725,
    "seeds": [],
    "status": "not_evaluable"
  },
  "registered_run": true
}
```

