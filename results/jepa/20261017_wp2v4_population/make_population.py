"""WP2-v4 population tables (runbook v4 §3.2 A4, Appendix A): 13 cells x 4 splits.

Deterministic population report; no sampling, no tuning, no mask selection.
Writes ``population_v4.json`` (compact, < 1 MB) next to this script.

Run from the repository root:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    /private/tmp/jepa-venv-cpu311/bin/python \
    results/jepa/20261017_wp2v4_population/make_population.py

D is the A4.5 displacement: population maximum principal angle in degrees between
the top-3 context CCA functionals (linearised fixture) and R_c = top-3 GEVP of
(A_c Sigma_theta A_c^T, N_cc).  gamma_n is reported beside it.
"""
import datetime
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'src'))

from jepa_noise.views import (  # noqa: E402
    A4_BASE_RHO, A4_CANDIDATE_ORDER, A4_CELL_GRID, A4_LAW, A4_THETA_SD,
    SPLIT_ROLES, Tier1Views, V3_BASE_RHO, V3_LAW, V3_THETA_SD, displacement,
    validate_a4_cells)

OUTPUT = HERE/'population_v4.json'
# Runbook v4 Appendix A, A4 design, channel split: g -> (angle k=3 deg, gamma_n)
APPENDIX_A_A4 = {0.: (0.0, 0.000), .005: (11.3, 0.357), .01: (22.3, 0.526),
                 .02: (48.0, 0.690), .05: (77.0, 0.847), .1: (84.1, 0.917),
                 .225: (87.5, 0.962), .45: (88.8, 0.980), .9: (89.4, 0.990)}
# Runbook v3 Appendix A design, channel split: g -> angle k=3 deg
APPENDIX_A_V3 = {0.: 18.4, .005: 74.0, .05: 84.6, .45: 89.2}
D_TOLERANCE_DEG = .5
D_BUG_CHECK_DEG = 1.0


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()


def population_row(common_strength, line_multiplier, split, base_rho, theta_sd, law):
    model = Tier1Views(signal_mode='linearized', split=split,
                       line_multiplier=line_multiplier, common_strength=common_strength,
                       base_rho=base_rho, theta_sd=theta_sd, seed=0,
                       cell='wp2v4_population_20261017', model_seed=0)
    pop = model.population()
    d = displacement(pop, 3)
    return {'common_strength': float(common_strength),
            'line_multiplier': float(line_multiplier), 'split': split,
            'split_role': SPLIT_ROLES[split], 'law': law,
            'gamma_n': float(pop['gamma_n']),
            'displacement_deg': float(d['angle_deg']),
            'angles_deg': {str(k): float(v) for k, v in d['angles_deg'].items()},
            'canonical_correlations': [float(x) for x in d['canonical_correlations']],
            'split_hash': model.split.digest,
            'noise_covariance_sha256': model.manifest()['noise_covariance_sha256']}


def constants(base_rho, theta_sd):
    model = Tier1Views(signal_mode='linearized', split='channel',
                       line_multiplier=0., common_strength=0., base_rho=base_rho,
                       theta_sd=theta_sd, seed=0, cell='wp2v4_population_20261017')
    return {'law': model.law, 'base_rho': model.base_rho,
            'theta_sd': model.theta_sd.tolist(), 'theta_fixture': model.theta_fixture,
            'v_white': model.v_white, 'epsilon_dc': model.epsilon_dc,
            'j0_sha256': model.manifest()['j0_sha256'],
            'temporal_covariance_sha256': digest(model.temporal_covariance),
            'common_basis_sha256': model.manifest()['common_basis_sha256'],
            'reference_channel_covariance_sha256': digest(model.channel_covariance),
            'base_channel_covariance_sha256': digest(model.base_channel_covariance)}


def main():
    start = time.monotonic()
    cells = validate_a4_cells(A4_CELL_GRID)
    splits = list(A4_CANDIDATE_ORDER)
    a4_rows = [population_row(g, line, split, A4_BASE_RHO, A4_THETA_SD, A4_LAW)
               for g, line in cells for split in splits]
    v3_rows = [population_row(g, 0., 'channel', V3_BASE_RHO, V3_THETA_SD, V3_LAW)
               for g in APPENDIX_A_V3]
    # --- registered checks ---------------------------------------------------
    channel = {(row['common_strength'], row['line_multiplier']): row
               for row in a4_rows if row['split'] == 'channel'}
    a4_angle = {g: channel[(g, 0.)]['displacement_deg'] for g in APPENDIX_A_A4}
    a4_gamma = {g: channel[(g, 0.)]['gamma_n'] for g in APPENDIX_A_A4}
    a4_delta = {g: a4_angle[g]-APPENDIX_A_A4[g][0] for g in APPENDIX_A_A4}
    gamma_delta = {g: a4_gamma[g]-APPENDIX_A_A4[g][1] for g in APPENDIX_A_A4}
    interleaved = {(row['common_strength'], row['line_multiplier']): row
                   for row in a4_rows if row['split'] == 'interleaved'}
    interleaved_delta = {g: interleaved[(g, 0.)]['displacement_deg']-a4_angle[g]
                         for g in APPENDIX_A_A4}
    v3_angle = {row['common_strength']: row['displacement_deg'] for row in v3_rows}
    v3_delta = {g: v3_angle[g]-APPENDIX_A_V3[g] for g in APPENDIX_A_V3}
    max_a4 = max(abs(x) for x in a4_delta.values())
    max_gamma = max(abs(x) for x in gamma_delta.values())
    max_interleaved = max(abs(x) for x in interleaved_delta.values())
    max_v3 = max(abs(x) for x in v3_delta.values())
    bug = max(max_a4, max_v3, max_interleaved) > D_BUG_CHECK_DEG
    checks = {
        'appendix_a_a4': {'expected_angle_deg': APPENDIX_A_A4,
                          'computed_angle_deg': a4_angle,
                          'delta_deg': a4_delta, 'max_abs_delta_deg': max_a4,
                          'tolerance_deg': D_TOLERANCE_DEG, 'pass': max_a4 <= D_TOLERANCE_DEG},
        'appendix_a_a4_gamma_n': {'expected': {str(g): v[1] for g, v in APPENDIX_A_A4.items()},
                                  'computed': {str(g): a4_gamma[g] for g in APPENDIX_A_A4},
                                  'max_abs_delta': max_gamma, 'tolerance': .005,
                                  'pass': max_gamma <= .005},
        'interleaved_vs_channel': {'delta_deg': {str(g): interleaved_delta[g] for g in interleaved_delta},
                                   'max_abs_delta_deg': max_interleaved,
                                   'tolerance_deg': D_TOLERANCE_DEG,
                                   'pass': max_interleaved <= D_TOLERANCE_DEG},
        'v3_appendix_a': {'expected_angle_deg': APPENDIX_A_V3,
                          'computed_angle_deg': v3_angle, 'delta_deg': v3_delta,
                          'max_abs_delta_deg': max_v3, 'tolerance_deg': D_TOLERANCE_DEG,
                          'pass': max_v3 <= D_TOLERANCE_DEG},
        'bug_check_threshold_deg': D_BUG_CHECK_DEG, 'bug_check_triggered': bug,
    }
    evidence = {
        'schema': 'jepa-wp2v4-population-v1',
        'evidence_level': 'development output, not reviewed',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host': platform.node(), 'platform': platform.platform(),
        'executable': sys.executable, 'python': sys.version,
        'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
        'environment': {key: __import__('os').environ.get(key) for key in
                        ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        'command': ('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
                    'PYTHONPATH=src /private/tmp/jepa-venv-cpu311/bin/python '
                    'results/jepa/20261017_wp2v4_population/make_population.py'),
        'law': A4_LAW, 'base_rho': A4_BASE_RHO, 'theta_sd': A4_THETA_SD.tolist(),
        'v3_law': V3_LAW, 'v3_base_rho': V3_BASE_RHO, 'v3_theta_sd': V3_THETA_SD.tolist(),
        'primary_split': 'channel', 'split_roles': SPLIT_ROLES,
        'cells': [[g, line] for g, line in cells], 'splits': splits,
        'constants_a4': constants(A4_BASE_RHO, A4_THETA_SD),
        'constants_v3': constants(V3_BASE_RHO, V3_THETA_SD),
        'a4_population': a4_rows,
        'v3_reproduction': v3_rows,
        'checks': checks,
        'status': 'bug_check' if bug else 'ok',
        'runtime_seconds': time.monotonic()-start,
    }
    OUTPUT.write_text(json.dumps(evidence, sort_keys=True, separators=(',', ':')) + '\n')
    print(f'wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)')
    for name, check in checks.items():
        if isinstance(check, dict) and 'pass' in check:
            print(f'{name}: pass={check["pass"]} max_abs_delta={check.get("max_abs_delta_deg", check.get("max_abs_delta"))}')
    if bug:
        print('BUG CHECK: a registered value differs from Appendix A by more than '
              f'{D_BUG_CHECK_DEG} deg; STOP and report, do not tune.')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
