# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Composable noise generation modules for detector-physics waveform studies.

The package is organised into subpackages; this module re-exports the public
API so existing ``from noise_module import X`` calls keep working.

| subpackage | role |
|---|---|
| `core` | single-channel generator, config schema, utilities, templates, streaming |
| `spectral` | composable one-sided PSD components and the multiplicative grammar |
| `multichannel` | correlated channels with implied and realized covariance |
| `artifacts` | injected transients and non-Gaussian innovations |
| `temporal` | non-stationarity, drift, piecewise stationarity |
| `resampling` | alias folding and in-band PSD resampling |
| `validation` | confidence-aware checks and reference-dataset calibration |
| `budgets` | closed-form detector noise budgets (Al2O3 athermal, HeRALD TES) |
"""

from .artifacts.injector import ArtifactInjector
from .artifacts.non_gaussian import NonGaussianNoiseGenerator
from .budgets.al2o3_athermal import (
    DEFAULT_SAMPLES as AL2O3_DEFAULT_SAMPLES,
    DEFAULT_SAMPLING_FREQUENCY as AL2O3_DEFAULT_SAMPLING_FREQUENCY,
    OptimalFilter,
    PulseFit,
    build_optimal_filter,
    fit_reference_pulse,
    load_composite as load_al2o3_athermal_composite,
    noise_generator as al2o3_athermal_noise_generator,
    recommend_record_length,
    validate_reference_noise,
)
from .budgets.reference_budget import (
    AL2O3_AL_ATHERMAL,
    AthermalNoiseBudget,
    BudgetGrid,
    write_reference_asd,
)
from .budgets.tes_budget import HERALD_V1_PLACEHOLDER, TESNoiseBudget
from .core.config import (
    CONFIG_SCHEMA_VERSION,
    ArtifactConfig,
    MultiChannelConfig,
    NoiseConfig,
    TemporalNoiseConfig,
    migrate_config,
)
from .core.generator import NoiseGenerator
from .core.streaming import StreamingNoiseGenerator, benchmark_generation
from .core.templates import pulse_template_2
from .core.utils import to_jsonable
from .multichannel.generator import MultiChannelNoiseGenerator
from .resampling.psd import (
    alias_fold_psd_density,
    inband_resample_psd_density,
    load_psd_density,
    make_target_psd_density,
    save_psd_density,
    synthetic_resample_psd_density,
)
from .spectral.models import (
    BandLimited,
    CompositeSpectrum,
    Filtered,
    Line,
    Lorentzian,
    Peaking,
    PowerLaw,
    Reflection,
    Resonance,
    RollOff,
    SpectralComponent,
    White,
)
from .temporal.wrapper import TemporalNoiseWrapper
from .validation.calibration import CalibrationPreset, ReferenceDataset, calibrate_dataset
from .validation.checks import (
    ValidationConfig,
    ValidationResult,
    bootstrap_interval,
    validate_artifacts,
    validate_csd_ensemble,
    validate_local_nonstationarity,
    validate_stationary_gaussian,
)

__all__ = [
    "AL2O3_AL_ATHERMAL",
    "al2o3_athermal_noise_generator",
    "AL2O3_DEFAULT_SAMPLES",
    "AL2O3_DEFAULT_SAMPLING_FREQUENCY",
    "alias_fold_psd_density",
    "ArtifactConfig",
    "ArtifactInjector",
    "AthermalNoiseBudget",
    "BandLimited",
    "benchmark_generation",
    "bootstrap_interval",
    "BudgetGrid",
    "build_optimal_filter",
    "calibrate_dataset",
    "CalibrationPreset",
    "CompositeSpectrum",
    "CONFIG_SCHEMA_VERSION",
    "fit_reference_pulse",
    "HERALD_V1_PLACEHOLDER",
    "inband_resample_psd_density",
    "Filtered",
    "Line",
    "load_al2o3_athermal_composite",
    "load_psd_density",
    "Lorentzian",
    "make_target_psd_density",
    "migrate_config",
    "MultiChannelConfig",
    "MultiChannelNoiseGenerator",
    "NoiseConfig",
    "NoiseGenerator",
    "NonGaussianNoiseGenerator",
    "OptimalFilter",
    "Peaking",
    "PowerLaw",
    "pulse_template_2",
    "PulseFit",
    "recommend_record_length",
    "ReferenceDataset",
    "Reflection",
    "Resonance",
    "RollOff",
    "save_psd_density",
    "SpectralComponent",
    "StreamingNoiseGenerator",
    "synthetic_resample_psd_density",
    "TemporalNoiseConfig",
    "TemporalNoiseWrapper",
    "TESNoiseBudget",
    "to_jsonable",
    "validate_artifacts",
    "validate_csd_ensemble",
    "validate_local_nonstationarity",
    "validate_reference_noise",
    "validate_stationary_gaussian",
    "ValidationConfig",
    "ValidationResult",
    "White",
    "write_reference_asd",
]

__version__ = "0.3.0"
