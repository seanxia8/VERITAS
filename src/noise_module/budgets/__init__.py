# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Budgets: closed-form detector noise budgets (Al2O3 athermal, HeRALD TES)."""
from .al2o3_athermal import (
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
from .reference_budget import (
    AL2O3_AL_ATHERMAL,
    AthermalNoiseBudget,
    BudgetGrid,
    write_reference_asd,
)
from .tes_budget import HERALD_V1_PLACEHOLDER, TESNoiseBudget

__all__ = [
    "AL2O3_DEFAULT_SAMPLES", "AL2O3_DEFAULT_SAMPLING_FREQUENCY", "OptimalFilter", "PulseFit",
    "build_optimal_filter", "fit_reference_pulse", "load_al2o3_athermal_composite",
    "al2o3_athermal_noise_generator", "recommend_record_length", "validate_reference_noise",
    "AL2O3_AL_ATHERMAL", "AthermalNoiseBudget", "BudgetGrid", "write_reference_asd",
    "HERALD_V1_PLACEHOLDER", "TESNoiseBudget",
]
