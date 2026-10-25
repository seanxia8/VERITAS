# SPDX-License-Identifier: MIT
"""Extension tests (registered: false): channel-ordered arrival delays."""
import numpy as np
import pytest

from jepa_noise.delay_views import DelayedTier1Views
from jepa_noise.views import Tier1Views

KW = dict(signal_mode='nonlinear', split='channel', line_multiplier=0., common_strength=0.02, base_rho=0.)


def _onset_index(row, frac=0.5):
    return int(np.argmax(row >= frac * row.max()))


def test_zero_delay_matches_registered_waveform():
    base, ext = Tier1Views(**KW), DelayedTier1Views(**KW, delay_mode='fixed', delay_step=0.0)
    theta = np.array([4.5, .35, .08])
    assert np.allclose(base.waveform(theta), ext.waveform(theta))
    assert np.allclose(base.noise_covariance, ext.noise_covariance)


def test_fixed_delay_orders_channels_and_keeps_three_parameters():
    v = DelayedTier1Views(**KW, delay_mode='fixed', delay_step=0.01)
    w = v.waveform([4.5, .35, .08])
    onsets = [_onset_index(r) for r in w]
    assert all(b > a for a, b in zip(onsets, onsets[1:]))
    assert v.j0.shape == (2048, 3) and v.n_physics == 3
    s = v.sample([0, 1, 2])
    assert s['theta'].shape == (3, 3) and s['raw'].shape == (3, 8, 256)


def test_per_event_delay_adds_fourth_parameter():
    v = DelayedTier1Views(**dict(KW, signal_mode='linearized'), delay_mode='per_event',
                          delay_range=(0.005, 0.015))
    assert v.j0.shape == (2048, 4) and v.theta_sd.shape == (4,)
    assert np.linalg.matrix_rank(v.j0) == 4
    nl = DelayedTier1Views(**KW, delay_mode='per_event', delay_range=(0.005, 0.015))
    th = nl.sample([5, 6])['theta']
    assert th.shape == (2, 4) and np.all((th[:, 3] >= 0.005) & (th[:, 3] <= 0.015))


def test_bad_delays_rejected():
    with pytest.raises(ValueError):
        DelayedTier1Views(**KW, delay_mode='fixed', delay_step=0.05)
    with pytest.raises(ValueError):
        DelayedTier1Views(**KW, delay_mode='sideways')


def test_delay_run_and_aggregate_are_explicitly_nonregistered(monkeypatch):
    pytest.importorskip("torch")
    from jepa_noise import bridge_run as br
    monkeypatch.setenv("JEPA_TOY_DELAY", '{"delay_mode":"fixed","delay_step":0.01}')
    run = br.RunSpec("delay", "sigreg_jepa", 0, "evaluation", 10000, 128)
    status = br._run_status(run, "complete", started_at="test", started_epoch=0,
                            steps_completed=10000)
    assert status["registered"] is False
    assert status["toy_delay"]["delay_step"] == .01
    aggregate = br.build_aggregate([], config=br.load_config(), config_sha="test",
                                   code_sha="test", status_text="", cohorts={}, fixture="a4")
    assert aggregate["registered"] is False
    assert aggregate["replaces_registered_verdict"] is False
    assert all(v["registered"] is False for v in aggregate["verdicts"].values())
    with pytest.raises(RuntimeError, match="missing or mixed delay provenance"):
        br.build_aggregate([{"manifest": {}}], config=br.load_config(), config_sha="test",
                           code_sha="test", status_text="", cohorts={}, fixture="a4")
