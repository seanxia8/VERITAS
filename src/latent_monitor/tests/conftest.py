# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Shared fixtures: one fitted linear subject and reference cell per session (≈2 s)."""

from __future__ import annotations

import numpy as np
import pytest

from latent_monitor import LinearSubject, calibrate, fit_reference
from latent_monitor.tier1 import TARGET_NAMES, reference_cell

FIT_IDS = np.arange(0, 160)
EVAL_IDS = np.arange(1000, 1050)
NOISE_IDS = np.arange(5000, 5080)


@pytest.fixture(scope="session")
def ref_cell():
    return reference_cell(n_channels=8, n_samples=256)


@pytest.fixture(scope="session")
def subject(ref_cell):
    X, T = ref_cell.batch(FIT_IDS)
    return LinearSubject.fit(X, T, ref_cell.geometry, ref_cell.implied_whitener(), latent_dim=6, seed=0, target_names=TARGET_NAMES)


@pytest.fixture(scope="session")
def reference(subject, ref_cell):
    return fit_reference(subject, ref_cell, EVAL_IDS, NOISE_IDS, seed=0)


@pytest.fixture(scope="session")
def thresholds(reference):
    return calibrate(reference)
