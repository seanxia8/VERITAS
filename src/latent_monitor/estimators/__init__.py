# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The four linear representation classes of Paper 1, as ORACLE subject stages.

    OF  →  CW-PCA / EMPCA (+ IsoPCA)  →  tied linear AE  →  NFPA (+ Iso-MPCA)

Every class is the same Σ̂⁻¹-orthogonal projection restricted to a different
admissible set; nothing here is a new architecture. Tentative plan for their
use as linear subjects: ``docs/EXPERIMENT_DESIGN.md`` §IV. Provenance of each
module is in its docstring (Paper 1 experiment repository
``noise-weighted-subspace-reconstruction`` @ ``ea076ff``, 2026-09-10).
"""

from .cw_pca import SubspaceFit, fit_pca, fit_weighted_pca, principal_angles, project_onto_basis
from .identification import MixingReport, align_to_tangent, mixing_matrix, tangent_basis
from .nfpa import NFPAFit, als_fit, fit_iso_mpca, fit_nfpa, kron_basis, kron_basis_whitened, recon_kron, unwhiten_separable, whiten_separable
from .of import (
    OptimalFilter,
    gls_amplitude,
    gls_amplitude_metric,
    gls_amplitude_variance_metric,
    matched_filter_score,
    project_rank1,
    psd_amplitude_variance,
)
from .tied_linear_ae import (
    TrainedAEResult,
    empca_optimal_loss,
    tied_linear_ae_closed_form,
    train_weighted_linear_ae,
    train_whitened_linear_ae,
    weighted_reconstruction_loss,
)

__all__ = [
    # class 1
    "OptimalFilter", "gls_amplitude", "gls_amplitude_metric", "gls_amplitude_variance_metric",
    "matched_filter_score", "project_rank1", "psd_amplitude_variance",
    # class 2
    "SubspaceFit", "fit_pca", "fit_weighted_pca", "principal_angles", "project_onto_basis",
    # class 3
    "TrainedAEResult", "empca_optimal_loss", "tied_linear_ae_closed_form",
    "train_weighted_linear_ae", "train_whitened_linear_ae", "weighted_reconstruction_loss",
    # class 4
    "NFPAFit", "als_fit", "fit_iso_mpca", "fit_nfpa", "kron_basis", "kron_basis_whitened",
    "recon_kron", "unwhiten_separable", "whiten_separable",
    # identification
    "MixingReport", "align_to_tangent", "mixing_matrix", "tangent_basis",
]
