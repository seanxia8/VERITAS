"""LUCiD point encoder and tokeniser, runbook v4 §3.5.5 (WP4c).

One token per PMT, all 1995 PMTs hit and unhit, features in this order:
position / (R, R, H/2); unit direction / 1.0; ln(q / 1 PE) for hit PMTs, 0
unhit; first-hit time / 100 ns, 0 unhit; hit flag.  Scaling is fixed and
data-independent (no per-event normalisation).

A *serialised* point transformer: tokens are Hilbert-sorted (see
``lucid_views``), attention runs inside windows of 64 serialised tokens with the
offset shifted by 32 on alternate layers, 4 pre-norm blocks, d = 128, 4 heads,
and **no final normalisation** -- the read-out used by every loss and probe is
the block-4 output (``final_norm is None``; the tests assert that no norm runs
after the last block).

``encode(features, hidden_mask)`` replaces hidden tokens with one learned
``mask_embed`` before block 1, so their raw features reach nothing downstream
(exactly zero gradient, exact invariance).  The predictor (2 blocks, d = 64,
full attention) reads the context read-out plus position queries at the target
PMTs.  ``tokenise(run, geometry)`` reads a generated run's ``hits.npz`` into the
``(B, N_pmt, N_features)`` tensor, the masks and the per-event keys.

Engineering choices (one line each in ``WP4C_ENCODER.md``): direction is
dimensionless so its divisor is 1.0; charge is floored at 1e-6 PE before the
log; one mask vector for all positions; full predictor attention; position-only
queries; local muP init (the source package is no-touch).  ``smoke_loss`` is not
a registered arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .lucid_bombs import DetectorGeometry, read_sparse_hits
from .lucid_views import TOKEN_HILBERT_BITS, ViewMasker, token_hilbert_order

__all__ = ["FEATURES", "N_FEATURES", "LOG_CHARGE_INDEX", "TIME_INDEX", "HIT_INDEX",
           "FeatureScaling", "EncoderConfig", "LucidBatch", "tokenise", "PointEncoder",
           "TargetQueryPredictor", "LucidPointJepa", "smoke_loss", "ridge_r2", "amplitude_probe"]

FEATURES = ("pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z",
            "log_charge", "time_100ns", "hit")
N_FEATURES = len(FEATURES)
LOG_CHARGE_INDEX, TIME_INDEX, HIT_INDEX = 6, 7, 8
QUERY_FEATURES = 3  # scaled position only (registered "position queries")


def init_mup_linears(module: nn.Module) -> None:
    """Spectral muP init (``reconstruction_model.model`` rule; that package is no-touch)."""
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            d_out, d_in = sub.weight.size(0), sub.weight.size(1)
            sigma = min(1.0, (d_out / d_in) ** 0.5) / (d_in ** 0.5)
            nn.init.normal_(sub.weight, mean=0.0, std=sigma)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)


@dataclass(frozen=True)
class FeatureScaling:
    """Fixed, data-independent token scaling (no per-event normalisation)."""

    radial_scale_m: float
    axial_scale_m: float
    direction_scale: float = 1.0
    time_scale_ns: float = 100.0
    charge_reference_pe: float = 1.0
    charge_floor_pe: float = 1e-6

    def __post_init__(self):
        for name in ("radial_scale_m", "axial_scale_m", "direction_scale", "time_scale_ns",
                     "charge_reference_pe", "charge_floor_pe"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"scaling: {name} must be positive and finite")

    @classmethod
    def from_geometry(cls, geometry: DetectorGeometry) -> "FeatureScaling":
        return cls(radial_scale_m=float(geometry.radius_m),
                   axial_scale_m=float(geometry.half_height_m))


def geometry_features(geometry: DetectorGeometry, scaling: FeatureScaling) -> np.ndarray:
    """Fixed (N_pmt, 6) position/direction block; geometry only, never the data."""
    positions = np.asarray(geometry.positions_m, dtype=np.float64) / np.array(
        [scaling.radial_scale_m, scaling.radial_scale_m, scaling.axial_scale_m])
    directions = np.asarray(geometry.directions, dtype=np.float64) / scaling.direction_scale
    return np.concatenate([positions, directions], axis=1).astype(np.float32)


@dataclass(frozen=True)
class EncoderConfig:
    """Frozen architecture of §3.5.5; optimiser numerics are not set here."""

    d_model: int = 128
    n_blocks: int = 4
    n_heads: int = 4
    window: int = 64
    shift: int = 32
    d_ff: int | None = None
    norm_eps: float = 1e-6
    mask_embed_init: float = 0.02
    d_pred: int = 64
    n_pred_blocks: int = 2
    n_pred_heads: int = 4
    d_pred_ff: int | None = None
    predictor_window: int | None = None
    n_features: int = N_FEATURES

    def __post_init__(self):
        if self.d_model % self.n_heads or self.d_pred % self.n_pred_heads:
            raise ValueError("d_model and d_pred must be divisible by their head counts")
        if self.window <= 0 or not 0 <= self.shift < self.window:
            raise ValueError("window must be positive and 0 <= shift < window")
        if self.predictor_window is not None and self.predictor_window <= 0:
            raise ValueError("predictor_window must be positive or None (full attention)")

    @property
    def d_ff_resolved(self) -> int:
        return 4 * self.d_model if self.d_ff is None else int(self.d_ff)

    @property
    def d_pred_ff_resolved(self) -> int:
        return 4 * self.d_pred if self.d_pred_ff is None else int(self.d_pred_ff)


@dataclass(frozen=True)
class LucidBatch:
    """A tokenised batch: features, view masks, indices and provenance keys."""

    features: Tensor          # (B, N_pmt, N_features) float32
    target: Tensor            # (B, N_pmt) bool, target view
    hidden: Tensor            # (B, N_pmt) bool, replaced by mask_embed
    context_index: Tensor     # (B, n_context) long
    target_index: Tensor      # (B, n_target) long
    hit_flags: np.ndarray     # (B, N_pmt) bool
    log_charge: np.ndarray    # (B, N_pmt) float32, 0 for unhit
    event_ids: np.ndarray     # (B,)
    keys: tuple               # per-event mask keys (JSON-serialisable)
    report: dict              # §3.5.4 counters and hit fractions
    scaling: FeatureScaling
    split: str
    occupancy: str


def _hits_path(source) -> Path:
    path = Path(source)
    return (path / "hits.npz") if path.is_dir() else path


def _event_features(store, index: int, geometry: DetectorGeometry,
                    scaling: FeatureScaling, base: np.ndarray):
    """One event's (N_pmt, N_features) features and its hit pattern."""
    span = slice(int(store.offsets[index]), int(store.offsets[index + 1]))
    pmt = np.asarray(store.pmt_id[span], dtype=np.int64)
    charge = np.asarray(store.charge_pe[span], dtype=np.float64)
    features, hit = base.copy(), np.zeros(geometry.n_pmt, dtype=bool)
    hit[pmt] = True
    features[pmt, LOG_CHARGE_INDEX] = np.log(
        np.maximum(charge, scaling.charge_floor_pe) / scaling.charge_reference_pe)
    features[pmt, TIME_INDEX] = np.asarray(store.time_ns[span], dtype=np.float64) / scaling.time_scale_ns
    features[pmt, HIT_INDEX] = 1.0
    return features, hit


def tokenise(source, geometry: DetectorGeometry, *, event_ids=None, limit: int | None = None,
             split: str = "across", seed: int = 0, replicate: int = 0,
             occupancy: str = "natural", scaling: FeatureScaling | None = None,
             masker: ViewMasker | None = None, device="cpu") -> LucidBatch:
    """Read a generated run (``hits.npz`` + geometry) into a tokenised batch.

    Events failing the §3.5.4 N_min rule are dropped and counted in the report;
    one split per batch, so the context/target index arrays are rectangular.
    """
    store = read_sparse_hits(_hits_path(source))
    scaling = scaling or FeatureScaling.from_geometry(geometry)
    base = np.zeros((geometry.n_pmt, N_FEATURES), dtype=np.float32)
    base[:, :6] = geometry_features(geometry, scaling)
    masker = masker or ViewMasker(geometry, split, seed=seed, replicate=replicate,
                                  occupancy=occupancy)
    wanted = None if event_ids is None else {int(x) for x in np.asarray(event_ids).ravel()}
    features, targets, hidden, contexts, target_indices = [], [], [], [], []
    hits, event_kept, keys = [], [], []
    for index, event_id in enumerate(store.event_ids):
        if wanted is not None and int(event_id) not in wanted:
            continue
        if limit is not None and len(event_kept) >= int(limit):
            break
        event, hit = _event_features(store, index, geometry, scaling, base)
        mask = masker.mask(int(event_id), hit)
        if not mask.accepted:
            continue
        features.append(event)
        targets.append(mask.target)
        hidden.append(mask.hidden)
        contexts.append(mask.context)
        target_indices.append(mask.target)
        hits.append(hit)
        event_kept.append(int(event_id))
        keys.append(mask.key)
    if not features:
        raise RuntimeError("tokenise: every event was excluded by the N_min rule")
    if len({target.size for target in targets}) != 1 or len({context.size for context in contexts}) != 1:
        raise ValueError("tokenise: ragged views; one split per batch only")
    target_np = np.zeros((len(features), geometry.n_pmt), dtype=bool)
    hidden_np = np.zeros_like(target_np)
    for row, (target, hide) in enumerate(zip(targets, hidden)):
        target_np[row, target] = True
        hidden_np[row, hide] = True
    features_np = np.stack(features)
    return LucidBatch(
        features=torch.as_tensor(features_np, dtype=torch.float32, device=device),
        target=torch.as_tensor(target_np, dtype=torch.bool, device=device),
        hidden=torch.as_tensor(hidden_np, dtype=torch.bool, device=device),
        context_index=torch.as_tensor(np.stack(contexts), dtype=torch.long, device=device),
        target_index=torch.as_tensor(np.stack(target_indices), dtype=torch.long, device=device),
        hit_flags=np.stack(hits), log_charge=features_np[..., LOG_CHARGE_INDEX].copy(),
        event_ids=np.asarray(event_kept, dtype=np.int64), keys=tuple(keys),
        report=masker.report(), scaling=scaling, split=masker.split,
        occupancy=masker.occupancy)


class WindowAttention(nn.Module):
    """Pre-norm attention inside serialised windows; ``window=None`` is full."""

    def __init__(self, d_model: int, n_heads: int, window: int | None, norm_eps: float):
        super().__init__()
        self.n_heads, self.head_dim = int(n_heads), int(d_model) // int(n_heads)
        self.window = None if window is None else int(window)
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        batch, n_tokens, d_model = x.shape
        n_heads, head_dim = self.n_heads, self.head_dim
        qkv = self.qkv(self.norm(x)).view(batch, n_tokens, 3, n_heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        if self.window is None or self.window >= n_tokens:
            attended = F.scaled_dot_product_attention(query, key, value)
            attended = attended.transpose(1, 2).reshape(batch, n_tokens, d_model)
            return self.proj(attended)
        window = self.window
        n_windows = (n_tokens + window - 1) // window
        pad = n_windows * window - n_tokens
        order = torch.roll(torch.arange(n_tokens, device=x.device), shifts=-int(offset))
        index = torch.cat([order, order.new_zeros(pad)]) if pad else order
        valid = torch.arange(n_windows * window, device=x.device) < n_tokens
        mask = valid.view(n_windows, window).view(1, n_windows, 1, 1, window)
        mask = mask.expand(batch, -1, -1, -1, -1).reshape(batch * n_windows, 1, 1, window)
        rows = batch * n_windows

        def windows(tensor):
            return tensor.index_select(2, index).view(batch, n_heads, n_windows, window, head_dim) \
                .permute(0, 2, 1, 3, 4).reshape(rows, n_heads, window, head_dim)

        attended = F.scaled_dot_product_attention(windows(query), windows(key), windows(value),
                                                  attn_mask=mask)
        attended = attended.view(batch, n_windows, n_heads, window, head_dim)
        attended = attended.permute(0, 1, 3, 2, 4).reshape(batch, n_windows * window, d_model)
        restored = torch.zeros_like(x)
        restored.index_copy_(1, order, attended[:, :n_tokens])
        return self.proj(restored)


class PointBlock(nn.Module):
    """Pre-norm transformer block: windowed attention then FFN, no post-norm."""

    def __init__(self, d_model: int, n_heads: int, window: int | None, d_ff: int,
                 norm_eps: float):
        super().__init__()
        self.attn = WindowAttention(d_model, n_heads, window, norm_eps)
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        x = x + self.attn(x, offset=offset)
        return x + self.ffn(self.norm(x))


class PointEncoder(nn.Module):
    """Serialised point transformer returning the pre-final-norm read-out."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.n_features, config.d_model)
        self.mask_embed = nn.Parameter(torch.zeros(config.d_model))
        self.blocks = nn.ModuleList([
            PointBlock(config.d_model, config.n_heads, config.window,
                       config.d_ff_resolved, config.norm_eps)
            for _ in range(config.n_blocks)])
        self.final_norm = None   # registered: read-out before any final norm
        init_mup_linears(self)
        nn.init.normal_(self.mask_embed, std=config.mask_embed_init)

    def forward(self, features: Tensor, hidden_mask: Tensor | None = None,
                order: Tensor | None = None) -> Tensor:
        if (features.ndim != 3 or features.size(-1) != self.config.n_features
                or (hidden_mask is not None and hidden_mask.shape != features.shape[:2])):
            raise ValueError(f"features must be (B, N, {self.config.n_features}) with a (B, N) mask")
        x = self.input_proj(features)
        if hidden_mask is not None:
            x = torch.where(hidden_mask.unsqueeze(-1), self.mask_embed, x)
        if order is not None:
            x = x.index_select(1, order)
        for level, block in enumerate(self.blocks):
            x = block(x, offset=(self.config.shift if level % 2 else 0))
        if order is not None:
            x = x.index_select(1, torch.argsort(order))
        return x  # before any final normalisation (registered read-out)


class TargetQueryPredictor(nn.Module):
    """Two d = 64 blocks over context read-out tokens plus position queries."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_pred)
        self.query_proj = nn.Linear(QUERY_FEATURES, config.d_pred)
        self.blocks = nn.ModuleList([
            PointBlock(config.d_pred, config.n_pred_heads, config.predictor_window,
                       config.d_pred_ff_resolved, config.norm_eps)
            for _ in range(config.n_pred_blocks)])
        self.out_proj = nn.Linear(config.d_pred, config.d_model)
        init_mup_linears(self)

    def forward(self, context_tokens: Tensor, queries: Tensor) -> Tensor:
        if context_tokens.ndim != 3 or queries.ndim != 3:
            raise ValueError("predictor needs (B, n_context, d_model) and (B, n_target, 3)")
        n_target = queries.size(1)
        x = torch.cat([self.in_proj(context_tokens), self.query_proj(queries)], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x[:, x.size(1) - n_target:])


class LucidPointJepa(nn.Module):
    """Encoder + predictor bound to one geometry (Hilbert order and queries)."""

    def __init__(self, config: EncoderConfig, geometry: DetectorGeometry,
                 scaling: FeatureScaling | None = None,
                 token_bits: int = TOKEN_HILBERT_BITS):
        super().__init__()
        self.config = config
        self.scaling = scaling or FeatureScaling.from_geometry(geometry)
        self.geometry_digest = geometry.digest
        self.encoder = PointEncoder(config)
        self.predictor = TargetQueryPredictor(config)
        order = torch.as_tensor(token_hilbert_order(geometry, token_bits), dtype=torch.long)
        queries = geometry_features(geometry, self.scaling)[:, :QUERY_FEATURES]
        self.register_buffer("token_order", order, persistent=False)
        self.register_buffer("query_features", torch.as_tensor(queries, dtype=torch.float32),
                             persistent=False)

    def encode(self, features: Tensor, hidden_mask: Tensor | None = None) -> Tensor:
        return self.encoder(features, hidden_mask=hidden_mask, order=self.token_order)

    def predict(self, readout: Tensor, context_index: Tensor, target_index: Tensor) -> Tensor:
        width = readout.size(-1)
        context = readout.gather(1, context_index.unsqueeze(-1).expand(-1, -1, width))
        queries = self.query_features[target_index]
        return self.predictor(context, queries)

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


def smoke_loss(model: LucidPointJepa, batch: LucidBatch) -> Tensor:
    """MSE between predicted and stop-grad target read-outs; smoke tests only.

    Not a registered arm: the §3.5.6 arm losses and the optimiser conventions
    are WP5 work.
    """
    context_readout = model.encode(batch.features, hidden_mask=batch.hidden)
    with torch.no_grad():
        full_readout = model.encode(batch.features)
    prediction = model.predict(context_readout, batch.context_index, batch.target_index)
    width = full_readout.size(-1)
    label = full_readout.gather(1, batch.target_index.unsqueeze(-1).expand(-1, -1, width))
    return F.mse_loss(prediction, label)


def ridge_r2(x_train, y_train, x_test, y_test, alpha: float = 1.0) -> dict:
    """Standardised ridge with an intercept; fixture probe only (the registered
    §3.5.6 probe is the v3 §1.3 ridge with run-grouped CV, WP5 work)."""
    xt = np.asarray(x_train, dtype=np.float64)
    xs = np.asarray(x_test, dtype=np.float64)
    yt = np.asarray(y_train, dtype=np.float64).ravel()
    ys = np.asarray(y_test, dtype=np.float64).ravel()
    if xt.ndim != 2 or xt.shape != (yt.size, xt.shape[1]) or xs.shape[0] != ys.size:
        raise ValueError("ridge_r2 needs (n, d) features with matching targets")
    centre = xt.mean(axis=0)
    scale = xt.std(axis=0)
    scale[scale == 0.0] = 1.0
    zt, zs = (xt - centre) / scale, (xs - centre) / scale
    offset = float(yt.mean())
    gram = zt.T @ zt + float(alpha) * np.eye(zt.shape[1])
    coef = np.linalg.solve(gram, zt.T @ (yt - offset))
    prediction = zs @ coef + offset
    residual = float(((ys - prediction) ** 2).sum())
    total = float(((ys - ys.mean()) ** 2).sum())
    return {"r2": 1.0 - residual / total, "alpha": float(alpha),
            "n_train": int(yt.size), "n_test": int(ys.size), "n_features": int(zt.shape[1])}


def amplitude_probe(model: LucidPointJepa, batch: LucidBatch, *, alpha: float = 1.0,
                    train_fraction: float = 0.5) -> dict:
    """Per-PMT log charge from the unmasked read-out tokens; **fixture** probe.

    Only hit PMTs enter.  This is not the registered R² >= 0.9 criterion, which
    is measured after training and on the final encoders.
    """
    with torch.no_grad():
        readout = model.encode(batch.features).detach().cpu().numpy()
    n_events = int(batch.features.size(0))
    if n_events < 2:
        raise ValueError("amplitude_probe needs at least two events")
    cut = max(1, min(n_events - 1, int(round(train_fraction * n_events))))
    is_train = np.zeros(n_events, dtype=bool)
    is_train[:cut] = True
    hit = np.asarray(batch.hit_flags, dtype=bool)
    train = is_train[:, None] & hit
    test = (~is_train)[:, None] & hit
    result = ridge_r2(readout[train], batch.log_charge[train],
                      readout[test], batch.log_charge[test], alpha=alpha)
    result.update({"train_events": cut, "test_events": n_events - cut,
                   "status": "fixture_not_the_registered_criterion"})
    return result
