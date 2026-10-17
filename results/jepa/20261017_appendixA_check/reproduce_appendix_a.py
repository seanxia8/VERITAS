#!/usr/bin/env python3
"""Reproduce runbook v4 Appendix A from the registered API (A4 law and v3 law).

Why this file exists
--------------------
`docs/jepa/check_v4_levels.py` is the registered design-check script that produced Appendix A.
It was committed unchanged in `151f26e` as the author's revision, and it builds the population
covariance *by hand* from the v3 `views.py` internals (`v.shared_weight`, `v.private_weight`,
`v.epsilon_dc`, `v.theta_sd`).  After WP2-v4 those attributes describe the A4 law, so running that
script on the A4 tree silently produces wrong numbers (the "base_rho = 0.4" rows collapse, the v3
table is not reproduced, and one row trips the runbook's 1 degree STOP).

The registered record of Appendix A is therefore the archived run at 151f26e:
`results/jepa/20261017_appendixA_check/check_v4_levels_151f26e.txt`.
On the delivered tree, this script is the supported reproduction: it uses the public API
(`Tier1Views(law=...)` + `views.displacement`) and checks both Appendix A tables.

Run:
    PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        /private/tmp/jepa-venv-cpu311/bin/python \
        results/jepa/20261017_appendixA_check/reproduce_appendix_a.py

Exit 0 = reproduced within the runbook tolerance (0.5 degrees); exit 2 = bug check (a value
differs by more than 1 degree, the runbook STOP; or by more than 0.5 degrees, the tolerance).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Runbook v4 Appendix A, channel split, line = 0.
A4_D = [0.0, 11.3, 22.3, 48.0, 77.0, 84.1, 87.5, 88.8, 89.4]
A4_GAMMA = [0.000, 0.357, 0.526, 0.690, 0.847, 0.917, 0.962, 0.980, 0.990]
G_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9]
# v3 Appendix A table (base rho = 0.4, v3 theta SD, channel split).
V3_D = {0.0: 18.4, 0.005: 74.0, 0.05: 84.6, 0.45: 89.2}
TOL_DEG = 0.5
STOP_DEG = 1.0


def main() -> int:
    from jepa_noise import views

    results = {'a4': {}, 'v3': {}, 'checks': {}, 'status': 'ok'}
    worst = 0.0
    worst_v3 = 0.0
    for split in ('channel', 'interleaved'):
        for g in G_VALUES:
            for law, key in ((views.A4_LAW, 'a4'), (views.V3_LAW, 'v3')):
                v = views.Tier1Views(split=split, common_strength=g, line_multiplier=0.,
                                     law=law)
                pop = v.population()
                d = views.displacement(pop)
                results[key][f'{split}/g={g}'] = {
                    'D_deg': d['angle_deg'],
                    'gamma_n': pop['gamma_n'],
                    'law': v.law,
                    'base_rho': v.base_rho,
                    'theta_fixture': v.theta_fixture,
                }

    for i, g in enumerate(G_VALUES):
        row = results['a4'][f'channel/g={g}']
        dd = abs(row['D_deg'] - A4_D[i])
        dg = abs(row['gamma_n'] - A4_GAMMA[i])
        worst = max(worst, dd)
        results['checks'][f'a4_channel_g={g}'] = {
            'D_deg': row['D_deg'], 'appendix_a_D_deg': A4_D[i], 'delta_deg': dd,
            'gamma_n': row['gamma_n'], 'appendix_a_gamma_n': A4_GAMMA[i],
            'delta_gamma_n': dg, 'within_tolerance': dd <= TOL_DEG,
        }
    for g, expected in V3_D.items():
        dd = abs(results['v3'][f'channel/g={g}']['D_deg'] - expected)
        worst_v3 = max(worst_v3, dd)
        results['checks'][f'v3_channel_g={g}'] = {
            'D_deg': results['v3'][f'channel/g={g}']['D_deg'],
            'appendix_a_v3_D_deg': expected, 'delta_deg': dd,
            'within_tolerance': dd <= TOL_DEG,
        }
    inter = max(abs(results['a4'][f'interleaved/g={g}']['D_deg']
                    - results['a4'][f'channel/g={g}']['D_deg']) for g in G_VALUES)
    results['checks']['interleaved_vs_channel_max_deg'] = inter

    results['worst_a4_delta_deg'] = worst
    results['worst_v3_delta_deg'] = worst_v3
    results['interleaved_channel_max_deg'] = inter
    ok = worst <= TOL_DEG and worst_v3 <= TOL_DEG and inter <= TOL_DEG
    stop = worst > STOP_DEG or worst_v3 > STOP_DEG
    results['status'] = 'ok' if ok else ('bug_check_stop' if stop else 'bug_check_tolerance')
    results['tolerance_deg'] = TOL_DEG
    results['stop_deg'] = STOP_DEG

    out_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(out_dir, 'reproduce_appendix_a.json'), 'w') as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    print(json.dumps({k: results[k] for k in
                      ('status', 'worst_a4_delta_deg', 'worst_v3_delta_deg',
                       'interleaved_channel_max_deg')}, indent=1))
    if not ok:
        print('MISMATCH: see reproduce_appendix_a.json', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
