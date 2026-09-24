# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Validation: confidence-aware checks and reference-dataset calibration."""
from .calibration import CalibrationPreset, ReferenceDataset, calibrate_dataset
from .checks import (
    ValidationConfig,
    ValidationResult,
    bootstrap_interval,
    validate_artifacts,
    validate_csd_ensemble,
    validate_local_nonstationarity,
    validate_stationary_gaussian,
)

__all__ = [
    "CalibrationPreset", "ReferenceDataset", "calibrate_dataset",
    "ValidationConfig", "ValidationResult", "bootstrap_interval", "validate_artifacts",
    "validate_csd_ensemble", "validate_local_nonstationarity", "validate_stationary_gaussian",
]
