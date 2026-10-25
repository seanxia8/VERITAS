#!/usr/bin/env python3
"""Guard-waiting Part C evaluation only. No training or grid continuation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {'old9000': .264400, 'old9001': .233759, 'fix9000': .219259,
            'fix9001': .204281, 'mae9000': .104949}
LABELS = (*EXPECTED, 'anc100')
ARMS = {label: ('mae_mse' if label == 'mae9000' else
               'sigreg_jepa_anchored' if label == 'anc100' else 'sigreg_jepa')
        for label in LABELS}
BASE_HOURS = .1125 + 785.11 / 3600  # retained failure + accepted init retry
PLANNED_MINUTES = 100
RUN_ID = 'pregrid_C_checkpoints_20261026'


def verdict(report):
    """Never infer scientific success from the inspector's zero exit code."""
    if report.get('registered') is not False or set(report.get('items', {})) != set(LABELS):
        return 'ENGINEERING_FAILURE'
    for label, expected in EXPECTED.items():
        row = report['items'][label]
        check = row.get('reproduction', {})
        got = (row.get('full_input') or {}).get('vertex_resolution_m')
        if check.get('expected_m') != expected or not isinstance(got, (float, int)):
            return 'ENGINEERING_FAILURE'
        if check.get('pass') is not True or abs(got - expected) > 1e-3:
            return 'STOP_S2'
    return 'PART_C_COMPLETE'


def within_budget(base_hours, planned_minutes):
    return base_hours + 1.5 * planned_minutes / 60 <= 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default='/home/dwong/anaconda3/bin/python3')
    parser.add_argument('--root', type=Path, default=Path('/ceph/dwong/jepa'))
    args = parser.parse_args()
    root = args.root.resolve()
    if root != Path('/ceph/dwong/jepa'):
        parser.error('this bounded execution uses only /ceph/dwong/jepa')
    ext = root / 'returns/ext_pregrid_20261025'
    queue = ext / 'part_c_queue_20261026'
    queue.mkdir(exist_ok=True)
    # Lock this queue only; never kill or replace another coordinator.
    import fcntl
    lock = (queue / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = threading.Event()
    child = None

    def stop(signum, _frame):
        stopped.set()
        if child is not None and child.poll() is None:
            child.send_signal(signum)  # jepa_launch forwards to its own job only

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    def record(status, **fields):
        payload = dict(registered=False, status=status, grid='paused', run_id=RUN_ID,
                       code_sha=code_sha, base_titan_gpu_hours=BASE_HOURS,
                       utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)
        temporary = queue / 'status.json.tmp'
        temporary.write_text(json.dumps(payload, indent=2) + '\n')
        os.replace(temporary, queue / 'status.json')
        with (queue / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(payload) + '\n')

    provenance = json.loads((ROOT / 'results/jepa/20261025_pregrid_checks/provenance.json').read_text())
    output = ext / 'inspect_dev/checkpoints_20261026'
    job = root / 'jobs' / RUN_ID
    guard = root / 'jobs/guard_deepthought.json'
    if not within_budget(BASE_HOURS, PLANNED_MINUTES):
        record('STOP_S1', reason='Titan runbook cap would be exceeded')
        return 2
    # Configuration, source checkpoints and nominal caches are frozen inputs.
    expected_files = {value['path']: value['sha256'] for value in provenance['checkpoints'].values()}
    for key in ('config', 'generator_config'):
        expected_files[provenance[key]['path']] = provenance[key]['sha256']
    for value in provenance['datasets'].values():
        expected_files[str(Path(value['path']) / 'manifest.json')] = value['sha256']
    for filename, expected in expected_files.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            record('INPUT_MISMATCH', path=filename)
            return 2
    job_args = ['inspect', '--out', str(output), '--config', provenance['config']['path'],
                '--generator-config', provenance['generator_config']['path'],
                '--geometry', provenance['geometry'], '--data-root',
                str(root / 'data/ext_pregrid_dev_20261025'), '--device', 'auto']
    for label in LABELS:
        job_args += ['--item', f"{label}={ARMS[label]}:{provenance['checkpoints'][label]['path']}"]
        if label in EXPECTED:
            job_args += ['--expect', f'{label}={EXPECTED[label]}']
    # Recover only a finished attempt; a partly existing attempt is never overwritten.
    if not (job / 'run_status.json').exists():
        if job.exists() or output.exists():
            record('EXISTING_ATTEMPT_REQUIRES_REVIEW', reason='no overwrite or blind retry')
            return 2
        while not stopped.is_set():
            check = subprocess.run([args.python, str(ROOT / 'scripts/jepa_guard.py'),
                    '--status-for-run', str(guard), '--host', 'deepthought',
                    '--job-kind', 'evaluation', '--planned-minutes', str(PLANNED_MINUTES)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if check.returncode == 0:
                break
            record('WAITING_GUARD', reason=check.stdout.strip(),
                   planned_minutes=PLANNED_MINUTES, max_gpu_hours=1.5 * PLANNED_MINUTES / 60)
            stopped.wait(120)
        if stopped.is_set():
            record('CANCELLED_BEFORE_START')
            return 143
        env = os.environ.copy()
        env.update(JEPA_PYTHON=args.python, JEPA_JOB_MODULE='scripts.jepa_pregrid_checks',
                   JEPA_JOB_KIND='evaluation', JEPA_JOBS_DIR=str(root / 'jobs'),
                   PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT))
        command = ['bash', str(ROOT / 'scripts/jepa_run.sh'), RUN_ID,
                   '--duration-minutes', str(PLANNED_MINUTES), '--host', 'deepthought', '--', *job_args]
        record('LAUNCHING', command=command, max_gpu_hours=2.5)
        with (queue / 'launcher.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            child.wait()
        # A race to HOLD is an operational refusal, not a scientific run or retry.
        if not (job / 'run_status.json').exists():
            record('LAUNCH_REFUSED_OR_ENGINEERING_FAILURE', returncode=child.returncode)
            return 2
    run = json.loads((job / 'run_status.json').read_text())
    hours = run['wall_seconds'] / 3600
    if run.get('exit_status') != 0 or not (output / 'inspect.json').exists():
        record('ENGINEERING_FAILURE', run_status=run, titan_gpu_hours=hours)
        return 2
    report = json.loads((output / 'inspect.json').read_text())
    result = verdict(report)
    record(result, run_status=run, inspect_path=str(output / 'inspect.json'),
           titan_gpu_hours=hours, total_runbook_titan_gpu_hours=BASE_HOURS + hours,
           next_action='Stop for author at S2; otherwise review Part C before Part D. No automatic D or grid.')
    if result == 'STOP_S2':
        rows = ['| Item | Expected m | Observed m | Absolute difference m |', '|---|---:|---:|---:|']
        for label, expected in EXPECTED.items():
            observed = report['items'][label]['full_input']['vertex_resolution_m']
            rows.append(f'| {label} | {expected} | {observed} | {abs(observed - expected)} |')
        (queue / 'STOP_S2.md').write_text('# Part C — S2 reproduction stop\n\n'
            'One or more unchanged --expect checks failed at 1e-3 m. Preserve all outputs; no retry, D or grid.\n\n'
            + '\n'.join(rows) + '\n\n'
            'A. Audit input/metric provenance read-only (recommended).\n'
            'B. Author supplies a prospective protocol clarification after that audit.\n'
            'C. Leave the extension stopped.\n\nSee inspect.json and status.json for the exact numbers.\n')
    return 0 if result == 'PART_C_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
