"""Runner infrastructure checks only; no waveform eligibility execution."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('population_runner', ROOT/'docs/jepa/check_waveform_population.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
CFG = json.loads((ROOT/'src/jepa_noise/configs/waveform_v1.json').read_text())


def test_a1_only_two_authorized_sweeps():
    for line in CFG['line_multipliers']:
        runner.validate_cell(CFG, line, .45)
    for common in CFG['common_strengths']:
        runner.validate_cell(CFG, 0., common)
    for line in CFG['line_multipliers'][1:]:
        for common in [0., .225, .675, .9]:
            with pytest.raises(ValueError, match='two reviewed A1 sweeps'):
                runner.validate_cell(CFG, line, common)
    with pytest.raises(ValueError):
        runner.validate_cell(CFG, 999., .45)


def test_completion_and_stop_hash_guard_detect_change_and_removal(tmp_path):
    for name in ['source.py', 'config.json']:
        (tmp_path/name).write_text('original')
    initial = {name: runner.sha(tmp_path/name) for name in ['source.py', 'config.json']}
    final, changed = runner.verify_source_hashes(initial, tmp_path)
    assert final == initial and changed == []
    (tmp_path/'source.py').write_text('modified')
    (tmp_path/'config.json').unlink()
    final, changed = runner.verify_source_hashes(initial, tmp_path)
    assert changed == ['source.py', 'config.json']
    assert final['config.json'] is None


def test_recovery_angle_matches_raw_coordinate_reference():
    reference = np.array([[1., 0.], [0., 2.], [.5, .2]])
    transformed_basis = np.array([[2., .1], [0., 3.]])@reference.T
    assert runner.population_recovery_angle(transformed_basis, reference) <= 1e-6
    assert runner.population_recovery_angle(np.array([[0., 0., 1.]]), np.array([[1.], [0.], [0.]])) > 1.


def test_stop_path_rehashes_inputs_and_blocks_acceptance(tmp_path, monkeypatch):
    import sys
    import jepa_noise.views
    config_dir = tmp_path/'src/jepa_noise/configs'
    config_dir.mkdir(parents=True)
    docs = tmp_path/'docs/jepa'
    docs.mkdir(parents=True)
    config = config_dir/'waveform_v1.json'
    config.write_text(json.dumps(CFG))
    (config_dir/'linear_solver_v1.json').write_text('{}')
    for name in ['check_waveform_population.py', 'RUNBOOK_JEPA_NOISE_2026-10-16.md', 'GATE_G5.md', 'review.md']:
        (docs/name).write_text('fixture-only source')
    scientific = tmp_path/'src/jepa_noise/fixture.py'
    scientific.write_text('original')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, '__file__', str(docs/'check_waveform_population.py'))
    monkeypatch.setattr(runner, 'git', lambda *args: 'pinned' if args[0] == 'rev-parse' else '')
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
        monkeypatch.setenv(name, '1')
    def stop_before_population(**kwargs):
        scientific.write_text('changed during failed attempt')
        raise RuntimeError('infrastructure fixture STOP; no population instantiated')
    monkeypatch.setattr(jepa_noise.views, 'Tier1Views', stop_before_population)
    monkeypatch.setattr(sys, 'argv', ['check', '--expected-code-sha', 'pinned',
        '--expected-config-sha256', runner.sha(config), '--review-report', str(docs/'review.md'),
        '--allocation-seconds', '10', '--line-multiplier', '0', '--common-strength', '.45',
        '--output', str(tmp_path/'output'), '--raw-dir', str(tmp_path/'raw')])
    monkeypatch.chdir(tmp_path)
    assert runner.main() == 1
    evidence = json.loads((tmp_path/'output/provenance.json').read_text())
    assert evidence['status'] == 'stopped'
    assert evidence['splits'] == []
    assert evidence['changed_scientific_inputs'] == ['src/jepa_noise/fixture.py']
    assert evidence['acceptance_blocker'] == 'scientific_source_or_review_or_head_changed'
