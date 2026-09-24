# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Core: the single-channel generator, config schema, utilities and templates."""
from .config import (
    CONFIG_SCHEMA_VERSION,
    ArtifactConfig,
    ConfigModel,
    MultiChannelConfig,
    NoiseConfig,
    TemporalNoiseConfig,
    migrate_config,
)
from .generator import NoiseGenerator
from .streaming import StreamingNoiseGenerator, benchmark_generation
from .templates import generate_burst_template, generate_glitch_template, pulse_template_2
from .utils import (
    concatenate_with_crossfade,
    match_target_std,
    mean_offdiag_corrcoef,
    resolve_rng,
    sample_range,
    spawn_rng,
    to_jsonable,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION", "ArtifactConfig", "ConfigModel", "MultiChannelConfig",
    "NoiseConfig", "TemporalNoiseConfig", "migrate_config",
    "NoiseGenerator", "StreamingNoiseGenerator", "benchmark_generation",
    "generate_burst_template", "generate_glitch_template", "pulse_template_2",
    "concatenate_with_crossfade", "match_target_std", "mean_offdiag_corrcoef",
    "resolve_rng", "sample_range", "spawn_rng", "to_jsonable",
]
