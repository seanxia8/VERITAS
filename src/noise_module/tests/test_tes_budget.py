# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""TESNoiseBudget: the closed form and the composite agree; provenance is honest."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import rfftfreq

from noise_module import HERALD_V1_PLACEHOLDER, MultiChannelNoiseGenerator, NoiseGenerator, TESNoiseBudget

FS, N = 2.5e5, 16384  # the QPSimulator / DELight grid


def test_closed_form_equals_composite_on_the_hest_grid() -> None:
    f = rfftfreq(N, d=1.0 / FS)
    b = HERALD_V1_PLACEHOLDER
    closed = b.component_psds(f)["total"]
    composite, meta = b.to_composite().evaluate(f, FS / N, zero_dc=False)
    # DC differs only in the 1/f term, which the closed form zeroes at f=0.
    np.testing.assert_allclose(composite[1:], closed[1:], rtol=1e-12)
    names = {m["name"] for m in meta}
    assert {"tfn", "tes_johnson_floor", "tes_johnson_etf", "shunt_johnson", "squid_white", "squid_1_f"} <= names
    assert not any(n.startswith("er") for n in names), "no paramagnetic-spin term in a TES budget"


def test_tes_johnson_form_is_suppressed_by_loop_gain_then_white() -> None:
    b = HERALD_V1_PLACEHOLDER
    low = b.component_psds(np.array([b.f_el_hz * 1e-3]))["TES_Johnson"][0]
    high = b.component_psds(np.array([b.f_el_hz * 1e3]))["TES_Johnson"][0]
    assert low == pytest.approx(b.tes_johnson_psd / b.loop_gain**2, rel=1e-3)
    assert high == pytest.approx(b.tes_johnson_psd, rel=1e-3)


def test_mains_and_vibration_lines_are_in_band_at_hest_sampling() -> None:
    f = rfftfreq(N, d=1.0 / FS)
    df = FS / N
    assert df == pytest.approx(15.2587890625)
    for freq, _, _ in HERALD_V1_PLACEHOLDER.line_triples():
        assert freq >= 2 * df and freq < FS / 2, (freq, 'a line below two bins is a drift, not a line')
    # off-grid: a 0.5 Hz-wide mains line falls between 15 Hz bins and is lost
    raw = HERALD_V1_PLACEHOLDER.component_psds(f)["lines"]
    assert raw[np.argmin(np.abs(f - 50.0))] < 1e-6
    # on_grid widens it at fixed power, so it is the strongest line where it should be
    on = HERALD_V1_PLACEHOLDER.on_grid(FS, N)
    lines = on.component_psds(f)["lines"]
    assert abs(f[int(np.argmax(lines))] - 50.0) <= df
    # integrated line power is preserved by the widening (Gaussian: peak*width*sqrt(2pi))
    p_raw = sum(pk * w for _, pk, w in HERALD_V1_PLACEHOLDER.line_triples())
    p_on = sum(pk * w for _, pk, w in on.line_triples())
    assert p_on == pytest.approx(p_raw, rel=1e-12)
    with pytest.raises(ValueError, match="drift"):
        TESNoiseBudget(name="x", tfn_psd=1, tau_eff_s=1e-3, tes_johnson_psd=1, loop_gain=2, tau_el_s=1e-5,
                       shunt_johnson_psd=1, squid_white_psd=1, squid_knee_hz=1,
                       vibration_lines=((1.4, 1.0, 0.1),)).on_grid(FS, N)


def test_noise_config_round_trips_through_the_generators() -> None:
    cfg = HERALD_V1_PLACEHOLDER.to_noise_config(FS, N)
    g = NoiseGenerator(cfg)
    f, S = g.build_psd_density(N)
    df = FS / N
    assert np.sum(S) * df == pytest.approx(cfg["noise_power"], rel=1e-9)
    mc = MultiChannelNoiseGenerator(cfg, {"mode": "shared_private", "n_channels": 24,
                                          "corr_strength": 0.3, "freeze_channel_structure": True}, seed=0)
    X, meta = mc.generate(N, return_metadata=True)
    assert X.shape == (24, N)
    assert meta["implied_covariance"].shape == (24, 24)
    assert np.isfinite(meta["realized_covariance"]).all()


def test_provenance_defaults_to_placeholder_and_is_recorded() -> None:
    rec = HERALD_V1_PLACEHOLDER.provenance_record()
    assert rec["tfn_psd"] == "placeholder"
    assert rec["tau_eff_s"] == "design"
    assert HERALD_V1_PLACEHOLDER.has_placeholders
    d = HERALD_V1_PLACEHOLDER.to_dict()
    assert d["has_placeholders"] is True and d["provenance"] == rec
    with pytest.raises(ValueError, match="provenance"):
        TESNoiseBudget(name="x", tfn_psd=1, tau_eff_s=1e-3, tes_johnson_psd=1, loop_gain=2, tau_el_s=1e-5,
                       shunt_johnson_psd=1, squid_white_psd=1, squid_knee_hz=1, provenance={"tfn_psd": "measured"})
    with pytest.raises(ValueError, match="loop_gain"):
        TESNoiseBudget(name="x", tfn_psd=1, tau_eff_s=1e-3, tes_johnson_psd=1, loop_gain=0.5, tau_el_s=1e-5,
                       shunt_johnson_psd=1, squid_white_psd=1, squid_knee_hz=1)


def test_signal_response_shares_the_qp_simulator_time_constants() -> None:
    b = HERALD_V1_PLACEHOLDER
    assert b.rise_s == pytest.approx(50e-6) and b.decay_s == pytest.approx(3e-3)
    f = np.array([0.0, 1.0 / (2 * np.pi * b.decay_s)])
    m = b.signal_magnitude(f)
    assert m[0] == pytest.approx(b.signal_amplitude)
    assert m[1] == pytest.approx(b.signal_amplitude / np.sqrt(2.0) / np.sqrt(1 + (b.rise_s / b.decay_s) ** 2), rel=1e-6)
