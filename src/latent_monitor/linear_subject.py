# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The linear analytic subject (plan §2.3): every stage is a matrix.

    x̃_c   = W x_c                          whitening, Σ̂ a named parameter
    h_c    = A x̃_c                         per-channel encoder, shared A (k × N)
    tok_c  = [h_c ‖ E pos_c]               geometry embedding E (d_e × 3), FIXED at init
    z      = Σ_c a_c h_c / Σ_c a_c,  a_c = 1 + pos_c·u      geometry-weighted pooling (u = 0 at init)
    x̂̃_c   = D z                            decoder in whitened coordinates, D (N × k)
    y      = O z + o₀                       least-squares output head to the physics targets

At fit time D = Aᵀ (tied) with A the top-k PCA basis of the whitened
per-channel training traces — the exact minimiser of the tied-linear whitened
objective (Paper 1) — so the whitened reconstruction Jacobian is Aᵀ and the
pullback Fisher is A Aᵀ = I_k. The *signal model* the subject carries is the
raw-coordinate basis S = W⁻¹ D.

**Re-whitening** (:meth:`with_sigma_hat`) is the whitening lemma applied as an
adjustment: keep S (the physics), replace Σ̂, and re-derive the encoder as the
Σ̂'-weighted projection onto S — the GLS / optimal-filter estimate — so z keeps
its meaning (coefficients on the same raw signal basis) and the output head O
stays valid. No gradient step; D and A are no longer transposes of each other
afterwards, which is the honest general form.

**Geometry** enters z through the pooling weights only. A symmetric sensor
layout has zero mean position, so a *pooled linear embedding* of positions
carries no information at z — the token hook still exposes it per channel —
and the stage-restricted refit for a geometry cell fits the three pooling
parameters u, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares

from .subject import HOOKS, Geometry
from .whitening import KroneckerWhitener


@dataclass
class LinearSubject:
    whitener: KroneckerWhitener
    A: np.ndarray                     # (k, N) encoder
    D: np.ndarray                     # (N, k) decoder in whitened coordinates
    E: np.ndarray                     # (d_e, 3) token embedding (fixed)
    u: np.ndarray                     # (3,) position-dependent pooling weights
    O: np.ndarray                     # (n_targets, k)
    o0: np.ndarray                    # (n_targets,)
    channel_weights: np.ndarray | None = None   # (C_ref,) GLS channel weights set by re-whitening; None = uniform
    pool_gain: float = 1.0            # global scale after normalised pooling; re-whitening sets it so O stays calibrated
    pos_scale: float = 1.0
    target_names: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- protocol
    @property
    def latent_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def sigma_hat(self) -> KroneckerWhitener:
        return self.whitener

    def signal_basis_raw(self) -> np.ndarray:
        """S = W⁻¹ D: the raw-coordinate signal basis, one channel, (N, k)."""
        return self.whitener.for_channels(1).unwhiten(self.D.T[:, None, :])[:, 0, :].T

    def _effective_channel_gain(self) -> float:
        """Gain of the pooled z on a unit shared signal through the current channel whitening + pooling."""
        C = self.whitener.n_channels
        w = self.pool_weights(Geometry(np.zeros((C, 3))))                 # position-free weights (uniform × channel_weights)
        return float(self.pool_gain * (w @ self.whitener._c_isqrt @ np.ones(C)))

    def with_sigma_hat(self, sigma_hat: KroneckerWhitener) -> "LinearSubject":
        """Re-whitening (the whitening lemma as an adjustment).

        Keep the raw signal basis S = W⁻¹D and the shared-waveform model; under
        Σ̂' = Σ_c' ⊗ T' the GLS estimate of the coefficients on S factorises into
        (i) a temporal GLS per channel — the new encoder A' = pinv(T'^{-1/2} S) —
        and (ii) GLS channel pooling with weights ∝ Σ_c'^{-1} 1, realised here as
        ``channel_weights = Σ_c'^{-1/2} 1`` in front of the channel-whitened
        traces, normalised so the pooled gain on a unit shared signal equals the
        reference's, so the output head O keeps its calibration. No gradient step.
        """
        S = self.signal_basis_raw()                                       # (N, k)
        W2 = sigma_hat.for_channels(1)
        S2 = W2.whiten(S.T[:, None, :])[:, 0, :].T                        # (N, k) in the new temporal whitening
        A2 = np.linalg.pinv(S2)                                           # (k, N): GLS coefficients on S
        gain_ref = self._effective_channel_gain()
        C = sigma_hat.n_channels
        cw = sigma_hat._c_isqrt @ np.ones(C)                              # Σ_c'^{-1/2} 1  (GLS pooling, up to scale)
        cw = np.clip(cw / cw.sum() * C, 1e-3, None)
        new = replace(self, whitener=sigma_hat, A=A2, D=S2, channel_weights=cw, pool_gain=1.0)
        gain_new = new._effective_channel_gain()
        return replace(new, pool_gain=self.pool_gain * gain_ref / gain_new)

    def embed(self, geometry: Geometry) -> np.ndarray:
        return (geometry.positions / self.pos_scale) @ self.E.T           # (C, d_e)

    def pool_weights(self, geometry: Geometry, u: np.ndarray | None = None) -> np.ndarray:
        """Normalised pooling weights: position term (1 + pos·u) × GLS channel weights (when C matches)."""
        u = self.u if u is None else u
        a = 1.0 + (geometry.positions / self.pos_scale) @ u
        a = np.clip(a, 1e-3, None)
        if self.channel_weights is not None and self.channel_weights.shape[0] == geometry.n_channels:
            a = a * self.channel_weights
        return a / a.sum()

    def _pool(self, h: np.ndarray, geometry: Geometry, u: np.ndarray | None = None) -> np.ndarray:
        w = self.pool_weights(geometry, u)
        return self.pool_gain * np.einsum("c,bck->bk", w, h)

    def represent(self, X: np.ndarray, geometry: Geometry) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 2
        if squeeze:
            X = X[None]
        xw = self.whitener.whiten(X)                                      # (n, C, N)
        h = np.einsum("kn,bcn->bck", self.A, xw)                          # (n, C, k)
        e = self.embed(geometry)                                          # (C, d_e)
        tok = np.concatenate([h, np.broadcast_to(e, (h.shape[0],) + e.shape)], axis=-1)
        z = self._pool(h, geometry)                                       # (n, k)
        pre = z @ self.O.T
        y = pre + self.o0
        out = {"whitened": xw, "channel": h, "token": tok, "z": z, "pre_output": pre, "output": y}
        if squeeze:
            out = {k: v[0] for k, v in out.items()}
        return out

    def outputs(self, X: np.ndarray, geometry: Geometry) -> np.ndarray:
        return self.represent(X, geometry)["output"]

    def decode(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        C = geometry.n_channels
        xw_hat = z @ self.D.T                                             # (n, N)
        return np.broadcast_to(xw_hat[:, None, :], (xw_hat.shape[0], C, xw_hat.shape[1])).copy()

    def jac_recon(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        """Whitened ∂g/∂z, stacked over channels: (C·N, k). Constant for a linear subject."""
        return np.tile(self.D, (geometry.n_channels, 1))

    def jac_output(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        return self.O.copy()

    def noise_variance_scale(self, reference: Geometry, geometry: Geometry) -> float:
        """Weighted pooling of whitened (decorrelated) channels: Var[z] ∝ Σ_c w_c²."""
        return float(np.sum(self.pool_weights(geometry) ** 2) / np.sum(self.pool_weights(reference) ** 2))

    def forward_from_stage(self, stage: str, value: np.ndarray, X: np.ndarray, geometry: Geometry) -> dict[str, np.ndarray]:
        """Overwrite hook ``stage`` with ``value`` and finish the pass (activation patching)."""
        if stage not in HOOKS:
            raise ValueError(f"unknown stage {stage!r}")
        rep = self.represent(X, geometry)
        if stage == "whitened":
            xw = np.asarray(value, dtype=float)
            h = np.einsum("kn,bcn->bck", self.A, xw)
        elif stage == "channel":
            xw = rep["whitened"]
            h = np.asarray(value, dtype=float)
        elif stage == "token":
            xw = rep["whitened"]
            h = np.asarray(value, dtype=float)[..., : self.latent_dim]
        else:
            xw, h = rep["whitened"], rep["channel"]
        e = self.embed(geometry)
        if stage in ("whitened", "channel", "token"):
            tok = np.concatenate([h, np.broadcast_to(e, h.shape[:-1] + (e.shape[1],))], axis=-1)
            z = self._pool(h, geometry)
        else:
            tok = rep["token"]
            z = np.asarray(value, dtype=float) if stage == "z" else rep["z"]
        pre = z @ self.O.T if stage != "pre_output" else np.asarray(value, dtype=float)
        y = pre + self.o0 if stage != "output" else np.asarray(value, dtype=float)
        return {"whitened": xw, "channel": h, "token": tok, "z": z, "pre_output": pre, "output": y}

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        targets: np.ndarray,
        geometry: Geometry,
        whitener: KroneckerWhitener,
        latent_dim: int,
        embed_dim: int = 4,
        seed: int = 0,
        target_names: tuple[str, ...] = (),
    ) -> "LinearSubject":
        """A by PCA of whitened per-channel traces (D = Aᵀ); E by initialisation; u = 0; O by least squares."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=float)
        n, C, N = X.shape
        xw = whitener.whiten(X).reshape(n * C, N)
        _, s, Vt = np.linalg.svd(xw, full_matrices=False)
        A = Vt[:latent_dim]
        pos_scale = float(np.max(np.abs(geometry.positions))) or 1.0
        E = rng.normal(0.0, 1.0, size=(embed_dim, 3))
        sub = cls(whitener=whitener, A=A, D=A.T.copy(), E=E, u=np.zeros(3),
                  O=np.zeros((targets.shape[1], latent_dim)), o0=np.zeros(targets.shape[1]),
                  pos_scale=pos_scale, target_names=tuple(target_names))
        sub = sub.refit_stage("output", X, targets, geometry)
        sub.meta = {"singular_values": s[: latent_dim + 4].tolist(), "n_train": int(n),
                    "explained": float(np.sum(s[:latent_dim] ** 2) / np.sum(s**2))}
        return sub

    def refit_stage(self, stage: str, X: np.ndarray, targets: np.ndarray, geometry: Geometry) -> "LinearSubject":
        """Stage-restricted refit — the linear analogue of a LoRA on one stage.

        ``channel`` refits the encoder/decoder pair (A, D = Aᵀ) by PCA on the new
        data and then the head; ``token`` refits the three pooling weights u only;
        ``output`` refits the head only. Everything else is untouched.
        """
        X = np.asarray(X, dtype=float)
        n, C, N = X.shape
        T = np.asarray(targets, dtype=float)
        if stage == "channel":
            xw = self.whitener.whiten(X).reshape(n * C, N)
            _, _, Vt = np.linalg.svd(xw, full_matrices=False)
            A = Vt[: self.latent_dim]
            return replace(self, A=A, D=A.T.copy()).refit_stage("output", X, T, geometry)
        if stage == "token":
            h = self.represent(X, geometry)["channel"]                    # (n, C, k)

            def resid(u: np.ndarray) -> np.ndarray:
                z = self._pool(h, geometry, u)
                return ((z @ self.O.T + self.o0) - T).ravel()

            # keep every pooling weight positive: |pos·u| ≤ 0.9 with |pos| ≤ pos_scale
            bound = 0.9 / np.sqrt(3.0)
            sol = least_squares(resid, x0=np.clip(self.u, -bound, bound), method="trf",
                                bounds=(-bound, bound), max_nfev=300)
            return replace(self, u=sol.x)
        if stage == "output":
            z = self.represent(X, geometry)["z"]
            Z1 = np.concatenate([z, np.ones((n, 1))], axis=1)
            coef, *_ = np.linalg.lstsq(Z1, T, rcond=None)
            return replace(self, O=coef[:-1].T, o0=coef[-1])
        raise ValueError(f"refit_stage: unsupported stage {stage!r}")
