"""Pre-grid population eligibility for all four fixed E1b splits.

Implementation only: execute only after orchestrator review, commit and explicit
ledger allocation. No sample grid, mask selection or parameter reselection.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def validate_cell(cfg, line_multiplier, common_strength):
    """A1 specifies two one-factor sweeps, not their Cartesian product."""
    in_grids = (line_multiplier in cfg['line_multipliers']
                and common_strength in cfg['common_strengths'])
    in_sweeps = (common_strength == cfg['E1b_line_sweep_fixed_common_strength']
                 or line_multiplier == cfg['E1b_common_sweep_fixed_line_multiplier'])
    if not in_grids or not in_sweeps:
        raise ValueError('cell is outside the two reviewed A1 sweeps')


def verify_source_hashes(initial, root):
    """Run even after STOP; missing/changed input cannot receive acceptance."""
    final = {}
    for name in initial:
        path = root/name
        try:
            final[name] = sha(path)
        except OSError:
            final[name] = None
    changed = [name for name in initial if final[name] != initial[name]]
    return final, changed


def population_recovery_angle(context, reference):
    """Euclidean row-space angle after oracle pullback, in radians."""
    from scipy.linalg import subspace_angles
    return float(max(subspace_angles(context.T, reference)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-code-sha', required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--review-report', type=Path, required=True)
    parser.add_argument('--allocation-seconds', type=int, required=True)
    parser.add_argument('--line-multiplier', type=float, required=True)
    parser.add_argument('--common-strength', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if not 1 <= args.allocation_seconds <= 600:
        parser.error('allocation must be within the registered 600-second benchmark ceiling')
    if git('rev-parse', 'HEAD') != args.expected_code_sha:
        parser.error('HEAD differs from pinned code SHA')
    cfg_path = ROOT/'src/jepa_noise/configs/waveform_v1.json'
    if sha(cfg_path) != args.expected_config_sha256:
        parser.error('configuration hash differs from reviewed hash')
    cfg = json.loads(cfg_path.read_text())
    try:
        validate_cell(cfg, args.line_multiplier, args.common_strength)
    except ValueError as error:
        parser.error(str(error))
    review = args.review_report.resolve(strict=True)
    watched = ['src/jepa_noise', 'src/latent_monitor', 'src/noise_module',
               'docs/jepa/check_waveform_population.py', 'docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md',
               'docs/jepa/GATE_G5.md']
    if git('status', '--porcelain=v1', '--untracked-files=all', '--', *watched):
        parser.error('uncommitted scientific inputs; commit and review before execution')
    git('ls-files', '--error-unmatch', 'docs/jepa/check_waveform_population.py', 'src/jepa_noise/configs/waveform_v1.json')
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        if os.environ.get(key) != '1':
            parser.error(f'{key}=1 is required')
    args.output.mkdir(parents=True, exist_ok=False)
    args.raw_dir.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), cfg_path, ROOT/'src/jepa_noise/configs/linear_solver_v1.json',
             ROOT/'docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md', ROOT/'docs/jepa/GATE_G5.md']
    for package in ('jepa_noise', 'latent_monitor', 'noise_module'):
        paths.extend(sorted((ROOT/'src'/package).rglob('*.py')))
    record = dict(status='running', evidence_level='development output, not reviewed',
                  code_sha=args.expected_code_sha, host=platform.node(), python=sys.version,
                  executable=sys.executable, start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
                  configuration_sha256=sha(cfg_path), review_report=str(review), review_sha256=sha(review),
                  line_multiplier=args.line_multiplier, common_strength=args.common_strength,
                  allocation_seconds=args.allocation_seconds, workers=1, rng='fixed reference structure seed11; no outcome samples',
                  raw_directory=f'{platform.node()}:{args.raw_dir.resolve()}',
                  environment={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                  git_status=git('status', '--porcelain=v1'), splits=[])
    start = time.monotonic()
    write(args.output/'provenance.json', record)

    def timeout(signum, frame):
        raise TimeoutError('preallocated population-check time exhausted; no retry authorized')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.allocation_seconds)
    exit_code = 1
    try:
        import numpy as np
        from jepa_noise.views import Tier1Views
        from jepa_noise.linear import oracle
        first_hash = None
        for split in cfg['split_candidates']:
            model = Tier1Views(signal_mode='linearized', split=split,
                               line_multiplier=args.line_multiplier, common_strength=args.common_strength,
                               seed=0, cell='population_eligibility_only', model_seed=0)
            manifest = model.manifest()
            if first_hash is None:
                first_hash = manifest['noise_covariance_sha256']
                np.save(args.raw_dir/'noise_covariance.npy', model.noise_covariance)
                np.save(args.raw_dir/'J0.npy', model.j0)
            elif first_hash != manifest['noise_covariance_sha256']:
                raise RuntimeError('split construction changed population noise law')
            p = model.population()
            r = oracle(*(p[x] for x in ['scc', 'stt', 'sct', 'ncc', 'ntt', 'nct', 'ac', 'at']), 3)
            spectra = {name: np.linalg.svd(a, compute_uv=False).tolist()
                       for name, a in [('context', p['ac']), ('target', p['at']), ('residual_target', r['residual_target'])]}
            joint = np.block([[p['ncc'], p['nct']], [p['nct'].T, p['ntt']]])
            # This is an explicit population report; ineligible splits remain.
            from scipy.linalg import eigvalsh
            minimum = float(eigvalsh(joint, subset_by_index=[0, 0])[0])
            item = dict(split=split, status=r['status'], gamma_n=p['gamma_n'],
                        singular_values=spectra, ranks={key: int(np.linalg.matrix_rank(a)) for key, a in
                          [('context', p['ac']), ('target', p['at']), ('residual_target', r['residual_target'])]},
                        context_ratio=r['context_ratio'], residual_target_ratio=r['residual_target_ratio'],
                        target_ratio=float(spectra['target'][-1]/spectra['target'][0]) if spectra['target'][0] else 0.,
                        kth_correlation=r['kth_correlation'], correlation_gap=r['correlation_gap'],
                        joint_noise_min_eigenvalue=minimum, manifest=manifest,
                        elapsed_seconds=time.monotonic()-start)
            item['population_recovery_angle_rad'] = (population_recovery_angle(r['context'], p['reference'])
                                                       if r['status'] == 'eligible' else None)
            item['population_recovery_status'] = ('not_evaluable' if r['status'] != 'eligible' else
                                                   ('pass' if item['population_recovery_angle_rad'] <= 1e-6 else 'fail'))
            record['splits'].append(item)
            write(args.output/f'{split}.json', item)
            write(args.output/'provenance.json', record)
            if item['population_recovery_status'] == 'fail':
                record['gate'] = 'G1'
                raise RuntimeError('eligible P5 population recovery exceeds 1e-6 rad; preserve row and STOP')
        record['status'] = 'complete'
        record['all_splits_oracle_ineligible'] = all(s['status'] != 'eligible' for s in record['splits'])
        exit_code = 0
    except BaseException as error:
        record.update(status='stopped', error_type=type(error).__name__, error=str(error))
    finally:
        signal.alarm(0)
        record.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-start, exit_status=exit_code)
        final_hashes, changed = verify_source_hashes(record['source_sha256'], ROOT)
        record['final_source_sha256'] = final_hashes
        record['changed_scientific_inputs'] = changed
        try:
            record['final_review_sha256'] = sha(review)
        except OSError:
            record['final_review_sha256'] = None
        record['final_code_sha'] = git('rev-parse', 'HEAD')
        if (changed or record['final_review_sha256'] != record['review_sha256']
                or record['final_code_sha'] != record['code_sha']):
            exit_code = 1
            record.update(status='stopped', exit_status=1,
                          acceptance_blocker='scientific_source_or_review_or_head_changed')
        record['raw_sha256'] = {str(p.resolve()): sha(p) for p in args.raw_dir.iterdir() if p.is_file()}
        write(args.output/'provenance.json', record)
        (args.output/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(args.output.iterdir()) if p.name != 'SHA256SUMS'))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
