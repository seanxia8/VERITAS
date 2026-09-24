# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Multiplicative transfer functions (``Filtered``) and frequency-dependent
crate coherence (``spectral_shared_private``) — the two fixes to the PMT
front-end preset of 11 Sep 2026 (src/noise_module_lucid/docs/LUCID_NOISE_REVIEW_2026-09-11.md)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import csd, welch

from noise_module import Filtered, MultiChannelNoiseGenerator, NoiseGenerator, Peaking, Reflection
from noise_module.spectral.models import component_from_config

FS, N = 1e9, 512


def _base(components, power=0.64):
    return dict(noise_type="composite", sampling_frequency=FS, noise_power=power,
                power_definition="variance", composite_psd_scaling="normalize", components=components)


# --------------------------------------------------------------- Filtered

def test_filtered_is_product_of_source_and_transfer_functions() -> None:
    f = np.fft.rfftfreq(N, 1 / FS)
    comp = component_from_config({
        "type": "filtered",
        "source": {"type": "white", "scale": 2.0},
        "filters": [{"type": "rolloff", "corner_hz": 2.5e8, "order": 2.0},
                    {"type": "peaking", "center_hz": 1.5e8, "half_width_hz": 2e7, "gain": 0.5}],
    })
    expected = 2.0 / (1 + (f / 2.5e8) ** 2) * (1 + 0.5 / (1 + ((f - 1.5e8) / 2e7) ** 2))
    assert isinstance(comp, Filtered)
    assert np.allclose(comp.shape(f), expected)


def test_additive_rolloff_leaves_floor_flat_but_filtered_rolls_off() -> None:
    """The defect the review found: an additive low-pass is a second source."""
    additive = _base([{"type": "white", "scale": 1.0},
                      {"type": "rolloff", "scale": 1.0, "corner_hz": 2.5e8, "order": 2.0}])
    filtered = _base([{"type": "filtered", "source": {"type": "white", "scale": 1.0},
                       "filters": [{"type": "rolloff", "corner_hz": 2.5e8, "order": 2.0}]}])
    _, s_add = NoiseGenerator(additive).build_psd_density(N)
    f, s_fil = NoiseGenerator(filtered).build_psd_density(N)
    nyq = np.argmin(np.abs(f - FS / 2))
    assert s_add[nyq] / s_add[1] > 0.5           # flat to Nyquist: 1.2 / 2.0 = 0.6
    assert s_fil[nyq] / s_fil[1] == pytest.approx(1 / 5, rel=1e-3)   # 1/(1+(500/250)^2)
    assert np.sum(s_fil[1:]) * FS / N == pytest.approx(0.64, rel=1e-9)   # normalisation unaffected


def test_peaking_and_reflection_are_unit_gain_responses() -> None:
    f = np.linspace(0, 5e8, 1001)
    assert np.all(Peaking(center_hz=1.5e8, half_width_hz=2e7, gain=0.5).shape(f) >= 1.0)
    r = Reflection(delay_s=10e-9, reflection=0.2).shape(f)
    assert r.min() == pytest.approx((1 - 0.2) ** 2, rel=1e-3)
    assert r.max() == pytest.approx((1 + 0.2) ** 2, rel=1e-3)
    # comb period = 1 / delay = 100 MHz
    peaks = f[1:-1][(r[1:-1] > r[:-2]) & (r[1:-1] > r[2:])]
    assert np.allclose(np.diff(peaks), 1e8, rtol=2e-2)


def test_filtered_metadata_records_the_filters_and_rejects_nesting() -> None:
    cfg = _base([{"type": "filtered", "name": "floor", "source": {"type": "white"},
                  "filters": [{"type": "rolloff", "corner_hz": 2.5e8}]}])
    _, _, meta = NoiseGenerator(cfg).build_psd_density(N, return_metadata=True)
    detail = meta["component_contributions"][0]["detail"]
    assert detail["source"]["type"] == "White" and detail["filters"][0]["type"] == "RollOff"
    with pytest.raises(ValueError, match="nest"):
        component_from_config({"type": "filtered", "source": {"type": "filtered", "source": {"type": "white"}}})
    with pytest.raises(ValueError, match="source"):
        component_from_config({"type": "filtered"})


# ------------------------------------------------ spectral_shared_private

SHARED = [{"type": "line", "scale": 6.0, "frequency_hz": 6.25e7, "width_hz": 2e6, "name": "clock"}]
PRIVATE = [{"type": "filtered", "name": "floor", "source": {"type": "white", "scale": 1.0},
            "filters": [{"type": "rolloff", "corner_hz": 2.5e8, "order": 2.0}]}]


def _gen(C=8, seed=0, **extra):
    cfg = {"mode": "spectral_shared_private", "n_channels": C, "shared_components": SHARED,
           "private_components": PRIVATE, "channel_gain_jitter": 0.0,
           "private_strength_range": [1.0, 1.0], "normalize_channel_variance": False, **extra}
    return MultiChannelNoiseGenerator(_base([{"type": "white"}]), cfg, seed=seed)


def test_spectral_mode_needs_both_component_lists() -> None:
    with pytest.raises(ValueError, match="shared_components"):
        MultiChannelNoiseGenerator(_base([{"type": "white"}]),
                                   {"mode": "spectral_shared_private", "private_components": PRIVATE})


def test_per_channel_variance_is_the_base_noise_power() -> None:
    X, meta = _gen(C=4).generate(N, return_metadata=True)
    assert X.shape == (4, N)
    diag = np.diag(meta["implied_covariance"])
    assert np.allclose(diag, 0.64)
    assert meta["implied_spectra"]["shared_power"] + meta["implied_spectra"]["private_power"] == pytest.approx(0.64)
    assert meta["kronecker_separable"] is False


def test_correlation_spectrum_is_one_at_the_line_and_zero_on_the_floor() -> None:
    _, meta = _gen(C=4).generate(N, return_metadata=True)
    f, rho = MultiChannelNoiseGenerator.implied_correlation_spectrum(meta, 0, 1)
    at_line = np.argmin(np.abs(f - 6.25e7))
    sp = meta["implied_spectra"]
    expected = sp["shared_psd"][at_line] / (sp["shared_psd"][at_line] + sp["private_psd"][at_line])
    assert rho[at_line] == pytest.approx(expected)      # 0.86 with these placeholder scales
    assert rho[at_line] > 0.8
    assert rho[np.argmin(np.abs(f - 1e7))] < 0.01
    assert rho[np.argmin(np.abs(f - 3e8))] < 0.01
    # the shared_private mode, by contrast, is flat at corr_strength
    _, meta_flat = MultiChannelNoiseGenerator(_base([{"type": "white"}]),
                                              {"mode": "shared_private", "n_channels": 4, "corr_strength": 0.3,
                                               "channel_gain_jitter": 0.0, "private_strength_range": [1.0, 1.0],
                                               "normalize_channel_variance": False}, seed=0).generate(N, return_metadata=True)
    assert np.allclose(meta_flat["implied_correlation"][0, 1], 0.3)


def test_realized_coherence_matches_the_implied_correlation_spectrum() -> None:
    """Welch coherence of a long record follows rho_ij(f): ~1 at the clock, ~0 on the floor."""
    n_long = 1 << 15
    gen = _gen(C=2, seed=3)
    X, meta = gen.generate(n_long, return_metadata=True)
    f_w, s01 = csd(X[0], X[1], fs=FS, nperseg=512)
    _, s00 = welch(X[0], fs=FS, nperseg=512)
    _, s11 = welch(X[1], fs=FS, nperseg=512)
    rho_hat = np.real(s01) / np.sqrt(s00 * s11)
    f_i, rho = MultiChannelNoiseGenerator.implied_correlation_spectrum(meta, 0, 1)
    rho_on_welch = np.interp(f_w, f_i, rho)
    at_line = np.argmin(np.abs(f_w - 6.25e7))
    assert rho_hat[at_line] == pytest.approx(rho_on_welch[at_line], abs=0.08)
    floor = (f_w > 1e7) & (f_w < 4e7)
    assert abs(rho_hat[floor].mean()) < 0.1
    assert rho_hat[at_line] - rho_hat[floor].mean() > 0.7   # the contrast is the point


def test_implied_csd_is_hermitian_psd_and_matches_covariance_trace() -> None:
    gen = _gen(C=3)
    _, meta = gen.generate(N, return_metadata=True)
    S = gen.implied_csd(3, N, meta)
    assert S.shape == (N // 2 + 1, 3, 3)
    assert np.allclose(S, np.swapaxes(S, 1, 2))
    assert np.all(np.linalg.eigvalsh(S[1:]) >= -1e-15)
    df = FS / N
    assert np.allclose(np.sum(S[1:], axis=0) * df, meta["implied_covariance"], rtol=1e-9)


def test_frozen_structure_is_reused_in_spectral_mode() -> None:
    gen = _gen(C=5, seed=1, freeze_channel_structure=True, channel_gain_jitter=0.1, private_strength_range=[0.8, 1.2])
    _, m1 = gen.generate(N, return_metadata=True)
    _, m2 = gen.generate(N, return_metadata=True)
    assert np.allclose(m1["gains"], m2["gains"]) and np.allclose(m1["implied_covariance"], m2["implied_covariance"])
    gen.reset_channel_structure()
    _, m3 = gen.generate(N, return_metadata=True)
    assert not np.allclose(m1["gains"], m3["gains"])
