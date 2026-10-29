"""Safety checks for the CPU-only guarded extension launcher."""
import time

import pytest

from scripts.jepa_cpu_launch import storage_check, validate_guard


def test_cpu_guard_refuses_stale_dry_hold_and_excess_cores():
    now = time.time()
    good = {'host': 'deepthought', 'level': 'OK', 'timestamp_epoch': now,
            'guard': {'dry_run': False, 'process_metrics_available': True}, 'caps': {'cpu_cores_used': 12}}
    validate_guard(good, 'deepthought', now)
    for patch in ({'level': 'HOLD'}, {'timestamp_epoch': now - 121},
                  {'guard': {'dry_run': True}}, {'caps': {'cpu_cores_used': 13}}):
        with pytest.raises(RuntimeError):
            validate_guard(dict(good, **patch), 'deepthought', now)


def test_cpu_storage_guard_stops_before_entry_floor(tmp_path, monkeypatch):
    values = {'ceph.quota.max_files': 500000, 'ceph.dir.rfiles': 400000,
              'ceph.dir.rsubdirs': 69999}
    from scripts.jepa_cpu_launch import os
    monkeypatch.setattr(os, 'getxattr', lambda _p, key: str(values[key]).encode(), raising=False)
    storage_check(tmp_path, '/not-read')
    values['ceph.dir.rsubdirs'] = 70001
    with pytest.raises(RuntimeError, match='S1'):
        storage_check(tmp_path, '/not-read')


def test_front_door_cpu_dispatch_does_not_enter_gpu_launcher(tmp_path):
    import os
    from pathlib import Path
    import subprocess

    python_stub = tmp_path / 'python-stub'
    python_stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python_stub.chmod(0o755)
    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ, JEPA_PYTHON=str(python_stub))
    result = subprocess.run(['bash', str(root / 'scripts/jepa_run.sh'), 'probe1', '--cpu',
                             '--module', 'scripts.jepa_ext_predictability_budget'],
                            env=env, text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    assert lines[0] == str(root / 'scripts/jepa_cpu_launch.py')
    assert lines[1:] == ['probe1', '--module', 'scripts.jepa_ext_predictability_budget']
    assert 'jepa_launch.py' not in result.stdout


def test_cpu_front_door_options_after_run_id_are_parsed():
    from scripts.jepa_cpu_launch import parse_args
    args = parse_args(['run1', '--module', 'scripts.jepa_ext_predictability_budget',
                       '--timeout-seconds', '10', '--', '--help'])
    assert args.run_id == 'run1'
    assert args.module == 'scripts.jepa_ext_predictability_budget'
    assert args.timeout_seconds == 10
    assert args.job_args == ['--', '--help']
