# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""S1: the compact transformer (``reconstruction_model.models.current_compact``) as a ``Subject``.

The adapter makes the four latent-facing pieces explicit around the frozen
network (plan §2.2) without changing its trained weights:

* **W** — a :class:`~latent_monitor.whitening.KroneckerWhitener` applied *before*
  the model; Σ̂ is a named, replaceable parameter. (The model's own
  ``normalise_input_sequence`` is a per-channel standardisation, not a
  covariance assumption; it stays inside the model.)
* **S1** — the temporal blocks, per channel; the ``channel`` hook is the
  patch-averaged per-channel token. Patching a pooled hook shifts every patch
  by the pooled delta, so a self-patch is exact and a clean-patch moves the
  summary the monitor actually reads.
* **P** — the model's ``abs_pos_embd`` embeds the *channel index*, not a
  position. The adapter adds an explicit, position-dependent additive
  embedding ``pos @ E_pos`` (zero at init, so the trained model is unchanged)
  — the named geometry stage the plan asks for, and the only thing
  ``refit_stage("token")`` touches.
* **S2** — the spatial blocks + mean pooling → ``z`` (d_model).
* **g** — the network has no reconstruction head, so ``decode`` / ``jac_recon``
  use a ridge linear *probe* decoder fitted on reference data
  (:meth:`fit_probe_decoder`). It is a probe of the excited support, not a
  trained decoder, and is labelled as such in ``meta``.
* **y** — the spatial (3) and energy (1) heads concatenated, ``(n, 4)``.

Everything is float64 NumPy at the boundary and torch inside; Jacobians come
from autograd (``jac_output``) or from the probe (``jac_recon``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .subject import HOOKS, Geometry
from .whitening import KroneckerWhitener

try:  # torch is optional for the package; required for this module
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("latent_monitor.torch_subject needs torch") from exc


def _np(t: "torch.Tensor") -> np.ndarray:
    return t.detach().cpu().numpy().astype(float)


@dataclass
class TransformerSubject:
    model: "nn.Module"
    whitener: KroneckerWhitener
    E_pos: np.ndarray                              # (3, d_model) position embedding, zero at init
    pos_scale: float = 1.0
    probe_D: np.ndarray | None = None              # (N, d_model) probe decoder in whitened coordinates (shared waveform)
    device: str = "cpu"
    #: which head components form ``output`` and in what order — the heads are linear probes on z,
    #: so any consistent selection is valid; training and monitoring use the same one.
    output_select: tuple[int, ...] = (0, 1, 2, 3)
    target_names: tuple[str, ...] = ("x", "y", "z", "energy")
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- protocol
    @property
    def latent_dim(self) -> int:
        return int(self.model.config.d_model)

    @property
    def sigma_hat(self) -> KroneckerWhitener:
        return self.whitener

    def with_sigma_hat(self, sigma_hat: KroneckerWhitener) -> "TransformerSubject":
        """Replace Σ̂ in W. For a nonlinear S1 there is no closed-form encoder re-derivation:
        the re-whitening adjustment here is the layer swap alone, and its effect is measured."""
        return replace(self, whitener=sigma_hat)

    @classmethod
    def wrap(cls, model: "nn.Module", whitener: KroneckerWhitener, pos_scale: float = 1.0, device: str = "cpu",
             output_select: tuple[int, ...] = (0, 1, 2, 3), target_names: tuple[str, ...] | None = None) -> "TransformerSubject":
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        names = target_names or tuple(("x", "y", "z", "energy")[i] for i in output_select)
        return cls(model=model, whitener=whitener, E_pos=np.zeros((3, model.config.d_model)), pos_scale=pos_scale,
                   device=device, output_select=tuple(output_select), target_names=names,
                   meta={"kind": "TransformerSubject", "decoder": "none"})

    def _heads(self, z: "torch.Tensor") -> "torch.Tensor":
        full = torch.cat([self.model.spatial_head(z), self.model.energy_head(z)], dim=-1)
        return full[..., list(self.output_select)]

    # ------------------------------------------------------------- staged forward
    def _stages(self, xw: "torch.Tensor", geometry: Geometry, override: tuple[str, "torch.Tensor"] | None = None
                ) -> dict[str, "torch.Tensor"]:
        """Mirror ``Transformer.forward`` with named stages; ``override`` substitutes one stage."""
        from reconstruction_model.models.current_compact import normalise_input_sequence, patch_input_sequence

        m, cfg = self.model, self.model.config
        B, C, L = xw.shape
        st: dict[str, torch.Tensor] = {"whitened": xw}
        name, val = override if override is not None else (None, None)
        if name == "whitened":
            xw = val
            st["whitened"] = xw
        x = normalise_input_sequence(xw.reshape(B * C, L))
        x = m.patch_embedding(patch_input_sequence(x, cfg.patch_len, cfg.patch_stride))
        for layer in m.temporal_layers:
            x = layer(x, m._rope_cache)
        x = x.view(B, C, -1, cfg.d_model)                                  # (B, C, P, d)
        if name == "channel":                                              # (B, C, d) given: shift every patch by the
            x = x + (val - x.mean(dim=2)).unsqueeze(2)                     # pooled delta (exact when val is the pass's own)
        st["channel"] = x.mean(dim=2)
        P = x.shape[2]
        x = x.transpose(1, 2).reshape(-1, C, cfg.d_model)                  # (B·P, C, d)
        x = m.abs_pos_embd(x)
        pos = torch.as_tensor(geometry.positions / self.pos_scale, dtype=x.dtype, device=x.device)
        x = x + (pos @ torch.as_tensor(self.E_pos, dtype=x.dtype, device=x.device)).unsqueeze(0)
        tok = x.view(B, P, C, cfg.d_model).mean(dim=1)                     # (B, C, d)
        if name == "token":
            delta = (val - tok).unsqueeze(1)                               # (B, 1, C, d)
            x = (x.view(B, P, C, cfg.d_model) + delta).reshape(-1, C, cfg.d_model)
            tok = val
        st["token"] = tok
        for layer in m.spatial_layers:
            x = layer(x, None)
        x = m.final_proj_norm(x)
        z = x.view(B, P, C, cfg.d_model).mean(dim=(1, 2))                  # (B, d)
        if name == "z":
            z = val
        st["z"] = z
        pre = self._heads(z)
        if name == "pre_output":
            pre = val
        st["pre_output"] = pre
        st["output"] = val if name == "output" else pre
        return st

    def _to_tensor(self, X: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)

    def represent(self, X: np.ndarray, geometry: Geometry) -> dict[str, np.ndarray]:
        X = np.asarray(X, dtype=float)
        squeeze = X.ndim == 2
        if squeeze:
            X = X[None]
        xw = self.whitener.whiten(X)
        with torch.no_grad():
            st = self._stages(self._to_tensor(xw), geometry)
        out = {k: _np(v) for k, v in st.items()}
        out["whitened"] = xw
        if squeeze:
            out = {k: v[0] for k, v in out.items()}
        return out

    def outputs(self, X: np.ndarray, geometry: Geometry) -> np.ndarray:
        return self.represent(X, geometry)["output"]

    def forward_from_stage(self, stage: str, value: np.ndarray, X: np.ndarray, geometry: Geometry) -> dict[str, np.ndarray]:
        if stage not in HOOKS:
            raise ValueError(stage)
        xw = self.whitener.whiten(np.asarray(X, dtype=float))
        with torch.no_grad():
            st = self._stages(self._to_tensor(xw), geometry, override=(stage, self._to_tensor(value)))
        out = {k: _np(v) for k, v in st.items()}
        out["whitened"] = xw if stage != "whitened" else np.asarray(value, dtype=float)
        return out

    def jac_output(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        """∂y/∂z by autograd at ``z`` — (len(output_select), d_model)."""
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device).reshape(1, -1).requires_grad_(True)
        y = self._heads(zt)[0]
        rows = [torch.autograd.grad(y[i], zt, retain_graph=True)[0][0] for i in range(y.shape[0])]
        return _np(torch.stack(rows))

    # ------------------------------------------------------------- probe decoder
    def fit_probe_decoder(self, X: np.ndarray, geometry: Geometry, ridge: float = 1e-2) -> "TransformerSubject":
        """Ridge linear probe from z to the channel-mean whitened trace: g(z) ≈ D z (shared waveform)."""
        rep = self.represent(X, geometry)
        z, xw = rep["z"], rep["whitened"].mean(axis=1)                    # (n, d), (n, N)
        if not (np.isfinite(z).all() and np.isfinite(xw).all()):
            raise ValueError("non-finite representation: the network is not in a usable state (diverged training?)")
        G = z.T @ z + ridge * max(np.trace(z.T @ z) / z.shape[1], 1e-12) * np.eye(z.shape[1])
        D = np.linalg.solve(G, z.T @ xw).T                                 # (N, d)
        return replace(self, probe_D=D, meta={**self.meta, "decoder": "linear_probe", "probe_ridge": ridge, "probe_n": int(z.shape[0])})

    def decode(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        if self.probe_D is None:
            raise RuntimeError("no decoder: call fit_probe_decoder on reference data first")
        z = np.atleast_2d(np.asarray(z, dtype=float))
        xw = z @ self.probe_D.T
        return np.broadcast_to(xw[:, None, :], (xw.shape[0], geometry.n_channels, xw.shape[1])).copy()

    def jac_recon(self, z: np.ndarray, geometry: Geometry) -> np.ndarray:
        if self.probe_D is None:
            raise RuntimeError("no decoder: call fit_probe_decoder on reference data first")
        return np.tile(self.probe_D, (geometry.n_channels, 1))

    def noise_variance_scale(self, reference: Geometry, geometry: Geometry) -> float:
        return reference.n_channels / geometry.n_channels               # mean pooling over channels

    # ------------------------------------------------------------- stage refit
    def refit_stage(self, stage: str, X: np.ndarray, targets: np.ndarray, geometry: Geometry,
                    steps: int = 200, lr: float = 1e-2) -> "TransformerSubject":
        """Stage-restricted refit. ``token`` fits E_pos only (the geometry stage); ``output`` fits the heads."""
        xw = self._to_tensor(self.whitener.whiten(np.asarray(X, dtype=float)))
        T = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=self.device)
        if stage == "token":
            E = torch.as_tensor(self.E_pos, dtype=torch.float32, device=self.device).clone().requires_grad_(True)
            opt = torch.optim.Adam([E], lr=lr)
            new = replace(self)
            for _ in range(steps):
                opt.zero_grad()
                new.E_pos = E                                              # type: ignore[assignment]
                y = new._stages(xw, geometry)["output"]
                loss = torch.mean((y - T) ** 2)
                loss.backward()
                opt.step()
            return replace(self, E_pos=_np(E))
        if stage == "output":
            import copy
            model = copy.deepcopy(self.model)
            params = list(model.spatial_head.parameters()) + list(model.energy_head.parameters())
            for p in params:
                p.requires_grad_(True)
            opt = torch.optim.Adam(params, lr=lr)
            sub = replace(self, model=model)
            for _ in range(steps):
                opt.zero_grad()
                y = sub._stages(xw, geometry)["output"]
                loss = torch.mean((y - T) ** 2)
                loss.backward()
                opt.step()
            for p in params:
                p.requires_grad_(False)
            return sub
        raise ValueError(f"refit_stage: unsupported stage {stage!r} for TransformerSubject (token | output)")


def train_reference(subject: TransformerSubject, X: np.ndarray, targets: np.ndarray, geometry: Geometry,
                    steps: int = 300, lr: float = 1e-3, batch: int = 32, seed: int = 0) -> TransformerSubject:
    """Train the whole network on the reference cell's physics targets (the model's own objective). CPU-scale helper."""
    import copy
    torch.manual_seed(seed)
    model = copy.deepcopy(subject.model).train()
    for p in model.parameters():
        p.requires_grad_(True)
    sub = replace(subject, model=model)
    xw = sub._to_tensor(subject.whitener.whiten(np.asarray(X, dtype=float)))
    T = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=subject.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = xw.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = torch.as_tensor(rng.choice(n, size=min(batch, n), replace=False))
        opt.zero_grad()
        y = sub._stages(xw[idx], geometry)["output"]
        loss = torch.mean((y - T[idx]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return replace(sub, meta={**subject.meta, "trained_steps": steps, "final_loss": float(loss.item())})
