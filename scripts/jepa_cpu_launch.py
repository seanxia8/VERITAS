#!/usr/bin/env python3
"""Guard-gated CPU-only JEPA job front door; never creates a CUDA context."""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BOOTSTRAP = 'import runpy,sys; m=sys.argv[1]; sys.argv=[m]+sys.argv[3:]; runpy.run_module(m,run_name="__main__")'


def validate_guard(status, host, now):
    if status.get('host') != host or status.get('level') != 'OK':
        raise RuntimeError('guard is not OK for this host')
    if not 0 <= now - float(status['timestamp_epoch']) < 120:
        raise RuntimeError('guard status is stale')
    if status.get('guard', {}).get('dry_run') is not False:
        raise RuntimeError('guard must be live, not dry-run')
    if status.get('guard', {}).get('process_metrics_available') is not True:
        raise RuntimeError('guard process metrics are unavailable')
    caps = status['caps']
    if float(caps['cpu_cores_used']) + 4 > 16:
        raise RuntimeError('CPU core cap would be exceeded')


def storage_check(ext, quota):
    free = (int(os.getxattr(quota, 'ceph.quota.max_files')) -
            int(os.getxattr(quota, 'ceph.dir.rfiles')) -
            int(os.getxattr(quota, 'ceph.dir.rsubdirs')))
    if free < 30000:
        raise RuntimeError('S1: free entries below 30000')
    size, count = 0, 0
    for base, dirs, files in os.walk(ext):
        count += len(dirs) + len(files)
        for name in files:
            try:
                size += (Path(base) / name).stat().st_size
            except FileNotFoundError:
                continue
    if count >= 2998 or size >= 1_900_000_000:
        raise RuntimeError('S1: extension storage safety margin exhausted')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('run_id')
    p.add_argument('--module', required=True)
    p.add_argument('--timeout-seconds', type=int, required=True)
    args, args.job_args = p.parse_known_args(argv)
    return args


def main():
    args = parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.run_id):
        raise ValueError('invalid run ID')
    if not 0 < args.timeout_seconds <= 43200:
        raise ValueError('timeout must fit within the 12-hour cap')
    if not args.module.startswith('scripts.jepa_ext_predictability_'):
        raise ValueError('CPU launcher restricted to predictability extension modules')
    root = Path('/ceph/dwong/jepa')
    ext = root / 'returns/ext_predictability_20261029'
    if not ext.is_dir():
        raise FileNotFoundError('extension root must already exist')
    jobs = root / 'jobs'
    guard_path = jobs / 'guard_deepthought.json'
    validate_guard(json.loads(guard_path.read_text()), 'deepthought', time.time())
    storage_check(ext, '/ceph/dwong')
    run_dir = jobs / args.run_id
    run_dir.mkdir(exist_ok=False)
    tag = 'jepa_' + args.run_id
    env = dict(os.environ)
    repo = Path(__file__).resolve().parents[1]
    env['PYTHONPATH'] = str(repo / 'src') + os.pathsep + str(repo)
    env.update(TMPDIR=str(root / 'tmp'), MPLCONFIGDIR=str(root / 'tmp'),
               XDG_CACHE_HOME=str(root / 'tmp'), XDG_CONFIG_HOME=str(root / 'tmp'),
               TORCH_HOME=str(root / 'tmp'), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4')
    job_args = args.job_args[1:] if args.job_args[:1] == ['--'] else args.job_args
    argv = [sys.executable, '-c', BOOTSTRAP, args.module, tag, *job_args]
    start = time.time(); started = time.monotonic(); cause = None
    with (run_dir / 'job.log').open('xb') as log:
        child = subprocess.Popen(argv, cwd=repo, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 preexec_fn=lambda: os.nice(10))
        info = {'pid': child.pid, 'pgid': child.pid, 'tag': tag, 'host': 'deepthought',
                'run_id': args.run_id, 'gpu_index': None, 'job_kind': 'other',
                'started_at_utc': datetime.now(timezone.utc).isoformat(), 'module': args.module,
                'timeout_seconds': args.timeout_seconds, 'registered': False}
        (run_dir / 'pid.json').write_text(json.dumps(info) + '\n')
        def forward(signum, _frame):
            nonlocal cause
            cause = 'signal_' + str(signum)
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        while child.poll() is None:
            try:
                if time.monotonic() - started >= args.timeout_seconds:
                    raise RuntimeError('S1: CPU wall-time deadline')
                storage_check(ext, '/ceph/dwong')
                state = json.loads(guard_path.read_text())
                if state.get('level') == 'SHED':
                    raise RuntimeError('guard SHED')
                if state.get('host') != 'deepthought' or time.time() - float(state['timestamp_epoch']) >= 240:
                    raise RuntimeError('guard unavailable or stale')
            except Exception as exc:
                cause = str(exc)
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                break
            time.sleep(5)
        try:
            rc = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            rc = child.wait()
            cause = (cause or 'child did not stop') + '; SIGKILL after 30 s'
    result = {'registered': False, 'run_id': args.run_id, 'exit_status': rc,
              'reason': cause, 'wall_seconds': time.time() - start, 'gpu_hours': 0}
    if cause != 'S1: free entries below 30000':
        (run_dir / 'run_status.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 1 if cause else rc


if __name__ == '__main__':
    sys.exit(main())
