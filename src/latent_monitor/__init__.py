# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Controlled-variable latent-space monitoring (docs/LATENT_MONITORING_PLAN_2026-09-05.md)."""

from .adjust import activation_patch, damage_patch, refit_stage, rewhiten
from .designed import DesignedCell, DesignedFamily
from .linear_subject import LinearSubject
from .lookup import ADJUSTMENT, Attribution, Thresholds, attribute, calibrate
from .reference import ReferenceCell, fit_reference
from .statistics import CellStatistics, cell_statistics, participation_ratio, rank_auroc
from .subject import HOOKS, Geometry, Subject
from .whitening import KroneckerWhitener, estimate_kronecker

__all__ = [
    "ADJUSTMENT", "Attribution", "CellStatistics", "DesignedCell", "DesignedFamily", "Geometry", "HOOKS",
    "KroneckerWhitener", "LinearSubject", "ReferenceCell", "Subject", "Thresholds",
    "activation_patch", "attribute", "damage_patch", "calibrate", "cell_statistics", "estimate_kronecker",
    "fit_reference", "participation_ratio", "rank_auroc", "refit_stage", "rewhiten",
]
__version__ = "0.1.0"
