"""r2-mask M1.3: unit tests of the network-free shortcut budget (synthetic fields, no physics)."""
from __future__ import annotations

import numpy as np
import pytest

from jepa_noise import shortcut_budget as sb
from jepa_noise.tests.r2_fixtures import fibonacci_sphere


def _random_hidden(n_events, n, rng):
    hidden = np.zeros((n_events, n), bool)
    for e in range(n_events):
        hidden[e, rng.permutation(n)[: n // 2]] = True
    return hidden


def _linear(pos, n_events, rng):
    """Every channel exactly linear in position with per-event coefficients; all PMTs hit."""
    a = rng.normal(0, 1, (n_events, 3))
    c = a @ pos.T + rng.normal(0, 1, (n_events, 1))
    b = rng.normal(0, 1, (n_events, 3))
    t = b @ pos.T + rng.normal(0, 1, (n_events, 1))
    return sb.Targets(np.ones_like(c, np.float32), c.astype(np.float32), t.astype(np.float32))


def test_linear_field_is_explained_by_the_local_predictor_under_random():
    rng = np.random.default_rng(0)
    pos = fibonacci_sphere(800, 3.0)
    fit, score = _linear(pos, 120, rng), _linear(pos, 120, rng)
    r = sb.shortcut_budget(pos, fit, score, _random_hidden(120, 800, rng),
                           _random_hidden(120, 800, rng), k=8)
    for ch in ("charge", "time"):
        assert r["channels"][ch]["E_L"] > 0.97, (ch, r["channels"][ch])
        assert r["channels"][ch]["E_S"] < 0.1
    assert np.isnan(r["channels"]["hit"]["E_S"])  # constant hit flag: no variance, not combined
    assert r["E_L"] > 0.97 and r["U"] < 0.03
    assert 0.45 < r["hidden_fraction"] < 0.55


def test_per_pmt_constant_field_is_explained_by_the_static_predictor():
    rng = np.random.default_rng(1)
    pos = fibonacci_sphere(500, 2.0)
    hit = (rng.random(500) < 0.5).astype(np.float32)
    c = (rng.normal(0, 1, 500) * hit).astype(np.float32)
    t = (rng.normal(0, 1, 500) * hit).astype(np.float32)

    def tg(n):
        return sb.Targets(np.tile(hit, (n, 1)), np.tile(c, (n, 1)), np.tile(t, (n, 1)))

    r = sb.shortcut_budget(pos, tg(60), tg(60), _random_hidden(60, 500, rng),
                           _random_hidden(60, 500, rng))
    for ch in sb.CHANNELS:
        assert r["channels"][ch]["E_S"] == pytest.approx(1.0, abs=1e-9)
    assert r["E_S"] == pytest.approx(1.0, abs=1e-9) and r["static"] == pytest.approx(1.0, abs=1e-9)
    assert r["E_L"] >= 1.0 - 1e-6


def test_avg_dist_parity_vs_contiguous_hemisphere():
    rng = np.random.default_rng(2)
    pos = fibonacci_sphere(600, 2.0)
    t = _linear(pos, 20, rng)
    spread = np.tile(np.arange(600) % 2 == 0, (20, 1))     # interleaved
    block = np.tile(pos[:, 2] > 0, (20, 1))                # one contiguous hemisphere
    r1 = sb.shortcut_budget(pos, t, t, spread, spread)
    r2 = sb.shortcut_budget(pos, t, t, block, block)
    assert r2["avg_dist_m"] > 3 * r1["avg_dist_m"]
    assert r2["U"] > r1["U"]
    assert r1["never_hidden_fraction"] == pytest.approx(0.5)


def test_correlation_length_grows_with_field_scale():
    rng = np.random.default_rng(3)
    pos = fibonacci_sphere(1500, 5.0)

    def field(scale, n=300):
        centres = fibonacci_sphere(64, 5.0)
        amp = rng.normal(0, 1, (n, 64))
        d2 = ((pos[:, None, :] - centres[None]) ** 2).sum(-1)
        c = amp @ np.exp(-d2 / (2 * scale ** 2)).T + 0.05 * rng.normal(0, 1, (n, len(pos)))
        return sb.Targets(np.ones_like(c, np.float32), c.astype(np.float32), np.zeros_like(c, np.float32))

    small = sb.correlation_length(pos, field(0.6))
    large = sb.correlation_length(pos, field(2.0))
    assert np.isfinite(small["l_c_m"]) and np.isfinite(large["l_c_m"])
    assert large["l_c_m"] > 1.5 * small["l_c_m"]
