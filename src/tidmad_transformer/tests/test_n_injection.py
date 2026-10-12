# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Controlled N injection on TIDMAD noise-only records (12 Oct 2026, WP-C).

Torch-free: imports only numpy, noise_module and (for one test) h5py.
"""
from __future__ import annotations

import numpy as np
import pytest

from tidmad_transformer import n_injection as nj
from tidmad_transformer.n_injection import (
    FAMILIES,
    NCell,
    ZERO_LEVEL,
    added_coloured_noise,
    band_line,
    baseline_drift,
    default_cells,
    dropout,
    gain_change,
    glitches,
    inject,
    requantise_int8,
    sampling_jitter,
)

FS = nj.FS_TIDMAD


def _record(n=65_536, sigma=3.0, seed=0):
    return ZERO_LEVEL + np.random.default_rng(seed).normal(0.0, sigma, n)


def test_added_noise_variance_matches_declared_psd_integral():
    x = _record()
    target = (0.5 * np.std(x)) ** 2
    realised = []
    for seed in range(20):
        _, meta = added_coloured_noise(x, 0.5, FS, seed)
        assert meta["declared_added_variance"] == pytest.approx(target, rel=1e-6)
        realised.append(meta["realised_added_variance"])
    # pink noise: per-draw variance fluctuates; the ensemble mean must match
    assert np.mean(realised) == pytest.approx(target, rel=0.1)


def test_band_line_power_lands_at_the_line():
    x = _record(n=2 ** 17)
    y, meta = band_line(x, 0.5, FS, 3, frequency_hz=2.0e5, width_hz=2.0e3)
    add = y - x
    spec = np.abs(np.fft.rfft(add)) ** 2
    f = np.fft.rfftfreq(add.size, 1 / FS)
    near = np.abs(f - 2.0e5) < 2.0e4
    assert spec[near].sum() / spec.sum() > 0.95
    assert meta["declared_added_variance"] == pytest.approx((0.5 * np.std(x)) ** 2, rel=1e-6)
    with pytest.raises(ValueError):
        band_line(x, 0.1, FS, 0, frequency_hz=6e6)


def test_gain_scales_about_the_analogue_zero():
    x = _record()
    y, _ = gain_change(x, 1.25)
    np.testing.assert_allclose(y - ZERO_LEVEL, 1.25 * (x - ZERO_LEVEL))
    y0, _ = gain_change(x, 1.0)
    np.testing.assert_allclose(y0, x)


def test_drift_is_slow_and_has_declared_span():
    x = _record(n=2 ** 16)
    y, meta = baseline_drift(x, 2.0, FS, 5)
    d = y - x
    assert meta["peak_to_peak"] == pytest.approx(2.0 * np.std(x), rel=1e-9)
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    spec = np.abs(np.fft.rfft(d - d.mean())) ** 2
    assert spec[1:65].sum() / spec[1:].sum() > 0.99          # power in the lowest ~0.1 % of the band
    y_lin, m_lin = baseline_drift(x, 1.0, FS, 0, kind="linear")
    assert np.all(np.diff(y_lin - x) >= 0) and m_lin["peak_to_peak"] == pytest.approx(np.std(x))


def test_glitch_count_follows_the_contract():
    x = _record(n=200_000)                                    # 20 ms at 10 MS/s
    counts = [glitches(x, 1000.0, FS, s)[1]["count"] for s in range(30)]
    _, meta = glitches(x, 1000.0, FS, 0)
    assert meta["expected_count"] == pytest.approx(20.0)
    assert np.mean(counts) == pytest.approx(20.0, rel=0.2)
    y0, m0 = glitches(x, 0.0, FS, 0)
    assert m0["count"] == 0 and np.array_equal(y0, x)


def test_jitter_scales_with_slope():
    t = np.arange(50_000) / FS
    x = ZERO_LEVEL + 10.0 * np.sin(2 * np.pi * 1e5 * t)
    y, _ = sampling_jitter(x, 0.1, FS, 1)
    slope_rms = np.std(np.gradient(x))
    assert np.std(y - x) == pytest.approx(0.1 * slope_rms, rel=0.05)
    np.testing.assert_allclose(sampling_jitter(x, 0.0, FS, 1)[0], x)


def test_dropout_exact_fraction_and_fill():
    x = _record(n=10_000)
    y, meta = dropout(x, 0.1, FS, 2, n_segments=3)
    assert meta["fraction"] == pytest.approx(0.1)
    assert sum(L for _, L in meta["segments"]) == 1000
    gap = np.zeros(x.size, dtype=bool)
    for s, L in meta["segments"]:
        gap[s:s + L] = True
    assert np.all(y[gap] == ZERO_LEVEL) and np.array_equal(y[~gap], x[~gap])
    with pytest.raises(ValueError):
        dropout(x, 1.0)


def test_requantise_round_trip_and_saturation():
    ints = np.arange(-128, 128, dtype=float) + ZERO_LEVEL
    q, meta = requantise_int8(ints)
    np.testing.assert_array_equal(q, ints)
    assert meta["n_saturated"] == 0
    q2, meta2 = requantise_int8(np.array([ZERO_LEVEL + 300.0, ZERO_LEVEL - 300.0, ZERO_LEVEL]))
    np.testing.assert_array_equal(q2, [ZERO_LEVEL + 127, ZERO_LEVEL - 128, ZERO_LEVEL])
    assert meta2["saturation_fraction"] == pytest.approx(2 / 3)


def test_inject_is_deterministic_and_records_provenance():
    x = _record()
    cell = NCell("added_pink_noise", 0.25, seed=7)
    y1, p1 = inject(x, cell)
    y2, p2 = inject(x, cell)
    np.testing.assert_array_equal(y1, y2)
    assert p1["origin"] == "N_cov" and p1["cell"] == "N_cov:added_pink_noise@0.25"
    assert p1["requantised"] and "saturation_fraction" in p1
    assert np.all(np.mod(y1, 1.0) == 0.0)                     # back on the integer grid
    y3, _ = inject(x, NCell("added_pink_noise", 0.25, seed=8))
    assert not np.array_equal(y1, y3)
    with pytest.raises(KeyError):
        inject(x, NCell("angular_acceptance", 1.0))


def test_registry_and_default_cells():
    assert set(nj.describe().values()) <= {"N_cov", "N_struct"}
    cells = default_cells()
    assert len(cells) == sum(len(s["grid"]) for s in FAMILIES.values())
    x = _record(n=16_384)
    for c in cells:                                           # every cell runs on a real-sized record
        y, p = inject(x, c)
        assert y.shape == x.shape and p["family"] == c.family


def test_inject_file_reads_a_noise_only_window(tmp_path):
    h5py = pytest.importorskip("h5py")
    raw = np.random.default_rng(0).integers(-20, 20, size=50_000).astype(np.int8)
    path = tmp_path / "abra_science_0000.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("timeseries/channel0001/timeseries", data=raw)
    y, prov = nj.inject_file(path, "timeseries/channel0001/timeseries", 1000, 8192,
                             NCell("gain_change", 1.1))
    expected = ZERO_LEVEL + np.round(1.1 * raw[1000:9192].astype(float))
    np.testing.assert_allclose(y, np.clip(expected, 0, 255))
    assert prov["start"] == 1000 and prov["length"] == 8192 and prov["file"].endswith(".h5")
    with pytest.raises(ValueError, match="holds only"):
        nj.read_noise_only(path, "timeseries/channel0001/timeseries", 49_000, 8192)
