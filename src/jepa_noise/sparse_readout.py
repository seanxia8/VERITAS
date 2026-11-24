# SPDX-License-Identifier: MIT
"""r2-sparse T1.6: readout adapter for hit-token encoders (frozen readout unchanged).

READOUT_SPEC_v1 applies unchanged: the raw features (l = ln q, tau = t / T, hit flag) come from the
pack in the standard dense encoding; only the token embeddings z come from the model. A sparse
encoder's hit tokens are scattered into a (B, N_pmt, d) array with zeros at unhit PMTs and the
standard dense features (hit flag 0 at unhit PMTs), with active = all PMTs, and the unchanged
``r2_probe_repair.moment_features`` / ``pool_tokens`` are called on that. (A padded hits-only array
with active = hits would be wrong: the raw summary hit_count / active_count would be 1 in every
event.) Pooling modes that average over all tokens use the hit tokens for sparse models.
"""
from __future__ import annotations

import numpy as np

from . import lucid_arms as la
from .r2_probe_repair import moment_features

__all__ = ["adapter_moment_features", "all_token_mean", "moment_latent_all_hits"]


def adapter_moment_features(hit_tokens: np.ndarray, hit_index: np.ndarray, hit_valid: np.ndarray,
                            features: np.ndarray):
    """``moment_features`` of padded hit tokens (B, L, d) scattered onto the dense PMT grid."""
    b, _, d = hit_tokens.shape
    n = features.shape[1]
    z = np.zeros((b, n, d), dtype=hit_tokens.dtype)
    for row in range(b):
        sel = hit_valid[row]
        z[row, hit_index[row, sel]] = hit_tokens[row, sel]
    return moment_features(z, features)


def all_token_mean(z: np.ndarray, features: np.ndarray, sparse: bool) -> np.ndarray:
    """``all_mean`` pooling: every token for dense models, the hit tokens for sparse ones."""
    if not sparse:
        return la._mean_tokens(z, None)
    return la._mean_tokens(z, features[..., 8] > 0.5)


def moment_latent_all_hits(z: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Exploratory readout variant (runbook §0.2): the early-time weight over P = all hits in the
    window instead of the prompt hits tau >= 0. Implemented by shifting each event's hit times so
    the earliest hit is at 0 (the early weight depends on delays only, so the latent block equals
    the frozen formula with P = all hits); only the latent output is used."""
    shifted = features.copy()
    hit = shifted[..., 8] > 0.5
    tau = shifted[..., 7]
    first = np.min(np.where(hit, tau, np.inf), axis=1, keepdims=True)
    first = np.where(np.isfinite(first), first, 0.0)
    shifted[..., 7] = np.where(hit, tau - first, tau)
    return moment_features(z, shifted)[0]
