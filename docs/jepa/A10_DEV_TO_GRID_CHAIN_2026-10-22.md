# A10 development -> grid chain (22 October 2026)

Operational chain installed on deepthought2 in tmux session `a10_chain`:

1. Wait for `a10_dev_step2` to write
   `/work/dwong/jepa_a10_work/dev_step2/dev/step2_probe_decision.json`.
2. Run `dev --step 3` (β audit) on deepthought2 with the guard and the step-2
   `sigreg_jepa` probes.json as `--plain-reference`.
3. Freeze `step2_probe_decision.json` and `step3_beta_decision.json` under
   `/work/dwong/jepa_a10_work/frozen_decisions/` with SHA256SUMS.
4. Launch two grid jobs with the guard, split by cell:

| job | cells | runs |
|---|---|---:|
| `a10_grid_gpu0` | `main_s002_across, main_s002_within, main_s005_across, r11_f1f2f3_within, r11b_law_within` | 87 |
| `a10_grid_gpu1` | `main_s005_within, main_s010_across, main_s010_within, dark_anchor_s010_within, law_t_aging_within, occupancy_s005_within_balanced` | 78 |

The jobs use separate `--out` and `--data-root` trees to avoid cache races, and
`--beta-decision` / `--physics-probe-decision` from the frozen dev decisions.
The chain log is `/work/dwong/jepa_a10_work/a10_chain.log`; the grid dry run
(`165` runs across `11` cells) is in `/work/dwong/jepa_a10_work/grid_dry_run.txt`.

The grid is a multi-day run; the chain reports completion in the log.
