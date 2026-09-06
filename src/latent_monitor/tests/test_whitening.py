# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
from __future__ import annotations

import numpy as np
import pytest

from latent_monitor.whitening import KroneckerWhitener, estimate_kronecker
from noise_module import MultiChannelNoiseGenerator, NoiseGenerator

FS, N, C = 1e4, 256, 8
BASE = dict(noise_type="composite", sampling_frequency=FS, noise_power=4.0, power_definition="variance",
            composite_psd_scaling="normalize",
            components=[{"type": "white", "scale": 1.0}, {"type": "rolloff", "scale": 1.0, "corner_hz": 1500.0, "order": 2.0, "kind": "lowpass"}])


def _gen(corr: float, seed: int = 0) -> MultiChannelNoiseGenerator:
    return MultiChannelNoiseGenerator(BASE, {"mode": "shared_private", "n_channels": C, "corr_strength": corr,
                                             "freeze_channel_structure": True, "normalize_channel_variance": False}, seed=seed)


def _whitener(gen: MultiChannelNoiseGenerator) -> KroneckerWhitener:
    _, meta = gen.generate(N, return_metadata=True)
    Sc = meta["implied_covariance"]
    _, psd = NoiseGenerator(BASE).build_psd_density(N)
    return KroneckerWhitener(Sc / np.mean(np.diag(Sc)), psd, FS, N)


def test_whitened_noise_has_unit_variance_and_no_channel_correlation() -> None:
    gen = _gen(0.5); W = _whitener(gen)
    recs = np.stack([gen.generate(N) for _ in range(300)])
    Xw = W.whiten(recs)
    assert np.var(Xw) == pytest.approx(1.0, abs=0.08)
    corr = np.corrcoef(Xw.transpose(1, 0, 2).reshape(C, -1))
    assert np.abs(corr[np.triu_indices(C, 1)]).mean() < 0.03


def test_round_trip_and_kappa() -> None:
    gen = _gen(0.5); W = _whitener(gen)
    x = gen.generate(N)
    np.testing.assert_allclose(W.unwhiten(W.whiten(x)), x, atol=1e-10)
    assert W.kappa_against(W.channel_cov, W.psd)["kappa"] == pytest.approx(1.0, abs=1e-9)
    other = _whitener(_gen(0.0))
    assert W.kappa_against(other.channel_cov)["kappa_channel"] > 3.0


def test_for_channels_carries_the_correlation_assumption() -> None:
    W = _whitener(_gen(0.5))
    W4 = W.for_channels(4)
    assert W4.n_channels == 4
    assert W4.mean_offdiag_correlation == pytest.approx(W.mean_offdiag_correlation, abs=1e-9)
    assert W.for_channels(C) is W
    assert W.whiten(np.zeros((3, 4, N))).shape == (3, 4, N)


def test_estimate_kronecker_recovers_the_generator_and_zeroes_dc() -> None:
    gen = _gen(0.5, seed=3); W = _whitener(gen)
    recs = np.stack([gen.generate(N) for _ in range(400)])
    Sc, psd = estimate_kronecker(recs, FS, psd_smoothing=5)
    assert psd[0] == 0.0
    k = W.kappa_against(Sc, psd)
    assert k["kappa_channel"] < 1.15 and k["kappa_temporal"] < 1.6
