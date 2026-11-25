# SPDX-License-Identifier: MIT
"""r2-sparse T1: new embedding, query-isolated decoder, fixed loss, sparse hit-token arms.

Runbook ``RUNBOOK_2026-11-24_R2_SPARSE_WCTE.md`` §2 (pre-registered) and §T1.

* :func:`fourier_code` -- fixed (not trained) positional code of the PMT positions: x, y scaled by
  the detector radius and z by the half-height; per axis 21 wavelengths log-spaced from twice the
  detector size to the median nearest-neighbour PMT spacing (in that axis's scaled units); sin and
  cos of each, 3 x 21 x 2 = 126 dimensions plus 2 zeros = 128.
* :class:`ContentMLP` -- two-layer MLP (width 128, GELU) on the content features.
* :class:`DenseFourierJepa` -- the dense C3 encoder (token = content + positional code, hidden =
  mask embedding + positional code, the unchanged serialised blocks) with the query-isolated
  predictor; a drop-in for ``LucidPointJepa`` inside ``LucidArm``.
* :class:`IsolatedPredictor` -- the existing predictor shape (2 blocks, d_pred = 64) whose queries
  attend to the context tokens and to themselves only, never to other queries.
* :func:`fixed_loss` -- class-balanced BCE hit term, ln q MSE and 2 x Huber time term on hit PMTs.
* :class:`SparseMaeArm` (S1 / S1n / S1p) and :class:`SparseTimeArm` (S2) on hit tokens only.

Tokens are built from the standard dense batch (``r2_store.tokenise_sparse``), so every arm sees
the same events and the same patch mask; hit tokens carry position (for the positional code),
ln q and tau only (no direction, no hit flag).

r2-sparse2 (runbook ``RUNBOOK_2026-11-25_R2_SPARSE2.md`` §2, §T1) adds, behind flags whose defaults
keep every existing cell unchanged:

* ``embed="linear"`` for both sparse arms: C2's input projection (``Linear(9, d_model)``) over the hit
  tokens' 9 standard features (scaled position, direction, ln(1 + q), tau, hit flag = 1), built from
  a seed-matched C2 model so the initial weights equal C2's at the same seed;
* ``decoder="legacy"`` for ``sparse_mae`` (S1L): C2's ``TargetQueryPredictor`` (queries = linear map
  of the scaled xyz, full attention over context and queries; padding keys masked), no count head,
  ``queries="all"`` only;
* ``decoder="full"`` for ``sparse_time`` (S2L, S2Lt): the S2 predictor with full attention among
  context and queries;
* ``blocks="time"`` with ``block_width_ns`` for ``sparse_time`` (S2Lt): Delta-wide time bins from
  -110 ns, a keyed half of the non-empty bins hidden, loss averaged per bin then over bins;
* :meth:`SparseTimeArm.block_mass_stats`: the own-block mass fraction f_own and its chance level.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import lucid_bombs as lb
from .lucid_views import token_hilbert_order
from .point_encoder import (HIT_INDEX, LOG_CHARGE_INDEX, TIME_INDEX, EncoderConfig, FeatureScaling,
                            PointBlock, init_mup_linears)

__all__ = ["N_FREQ", "PE_DIM", "fourier_code", "median_nn_spacing", "ContentMLP", "SparseBlock",
           "SparseEncoder", "IsolatedPredictor", "DenseFourierJepa", "fixed_loss", "gather_tokens",
           "FactorJepa", "SparseMaeArm", "SparseTimeArm", "SPARSE_ARMS", "QUERY_MODES", "COUNT_WEIGHT",
           "sentinel_features", "scatter_tokens", "legacy_predict", "linear_token_features",
           "EMBED_MODES", "TIME_BLOCK_MODES", "S2_WINDOW_NS"]

N_FREQ = 21
PE_DIM = 128
COUNT_WEIGHT = 0.1
HUBER_DELTA = 0.2          # in units of 50 ns (10 ns)
TIME_LOSS_UNIT_NS = 50.0
POS_WEIGHT_CLIP = (0.05, 20.0)
SPARSE_ARMS = ("sparse_mae", "sparse_time")
QUERY_MODES = ("all", "neg", "pos")
S2_BLOCK = 32
S2_MAX_BLOCKS = 64
S2_EPS = 1e-4
NS_TIME_BLOCKS = "r2-sparse/time_blocks"
EMBED_MODES = ("fourier", "linear")          # r2-sparse2: "fourier" = the unchanged S1/S2 embedding
TIME_BLOCK_MODES = ("count32", "time")       # r2-sparse2: "count32" = the unchanged S2 blocks
S2_WINDOW_NS = (-110.0, 50.0)                # lucid_v1_w2 window; S2Lt bins start at -110 ns


# ============================================================================ positional code
def median_nn_spacing(positions_m) -> float:
    p = np.asarray(positions_m, dtype=np.float64)
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def fourier_code(geometry, scaling: FeatureScaling) -> tuple[np.ndarray, dict]:
    """(N_pmt, 128) fixed Fourier code and its recorded parameters."""
    scale = np.array([scaling.radial_scale_m, scaling.radial_scale_m, scaling.axial_scale_m])
    pos = np.asarray(geometry.positions_m, dtype=np.float64) / scale
    spacing = median_nn_spacing(geometry.positions_m)
    blocks, meta = [], {"median_nn_spacing_m": spacing, "axes": {}}
    for axis, name in enumerate("xyz"):
        lam_max = 2.0 * 2.0  # twice the detector size; the size is 2 in scaled units ([-1, 1])
        lam_min = spacing / scale[axis]
        lam = np.geomspace(lam_max, lam_min, N_FREQ)
        omega = 2.0 * np.pi / lam
        arg = pos[:, axis:axis + 1] * omega[None, :]
        blocks += [np.sin(arg), np.cos(arg)]
        meta["axes"][name] = {"wavelength_max_scaled": lam_max, "wavelength_min_scaled": float(lam_min),
                              "scale_m": float(scale[axis])}
    code = np.concatenate(blocks + [np.zeros((pos.shape[0], PE_DIM - 6 * N_FREQ))], axis=1)
    meta["rms_per_dim"] = float(np.sqrt((code ** 2).mean()))
    return code.astype(np.float32), meta


class ContentMLP(nn.Module):
    """Linear -> GELU -> Linear, width 128; muP init (output RMS recorded at step 0)."""

    def __init__(self, n_in: int, d_model: int = PE_DIM, width: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(n_in, width)
        self.fc2 = nn.Linear(width, d_model)
        init_mup_linears(self)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def sentinel_features(features: Tensor, sentinel_time: float) -> Tensor:
    """§2.2 placeholders on standard dense features: ln(1 + q) for the charge (0 = no hit) and
    tau = (A + 50 ns) / T on unhit PMTs; the hit flag is kept."""
    out = features.clone()
    hit = features[..., HIT_INDEX] > 0.5
    q = torch.exp(features[..., LOG_CHARGE_INDEX])
    out[..., LOG_CHARGE_INDEX] = torch.where(hit, torch.log1p(q), torch.zeros_like(q))
    out[..., TIME_INDEX] = torch.where(hit, features[..., TIME_INDEX],
                                       torch.full_like(q, float(sentinel_time)))
    return out


# ============================================================================ attention blocks
class _MaskedAttention(nn.Module):
    """Pre-norm multi-head attention with a boolean (B, 1, L, L) or key mask."""

    def __init__(self, d_model: int, n_heads: int, norm_eps: float):
        super().__init__()
        self.n_heads, self.head_dim = int(n_heads), int(d_model) // int(n_heads)
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class _MaskedBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, norm_eps: float):
        super().__init__()
        self.attn = _MaskedAttention(d_model, n_heads, norm_eps)
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(x, mask)
        return x + self.ffn(self.norm(x))


class SparseBlock(nn.Module):
    """Windowed attention over a padded per-event token order (windows of ``window`` positions;
    odd layers shift the windows by ``shift`` with front padding, no wrap-around)."""

    def __init__(self, d_model: int, n_heads: int, window: int | None, d_ff: int, norm_eps: float):
        super().__init__()
        self.window = None if window is None else int(window)
        self.attn = _MaskedAttention(d_model, n_heads, norm_eps)
        self.norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def _attend(self, x: Tensor, valid: Tensor, offset: int) -> Tensor:
        b, n, d = x.shape
        if self.window is None or self.window >= n + offset:
            mask = valid[:, None, None, :] | torch.eye(n, dtype=torch.bool, device=x.device)[None, None]
            return self.attn(x, mask)
        w = self.window
        total = offset + n
        n_win = (total + w - 1) // w
        back = n_win * w - total
        xp = F.pad(x, (0, 0, offset, back))
        vp = F.pad(valid, (offset, back), value=False)
        xw = xp.view(b * n_win, w, d)
        vw = vp.view(b * n_win, w)
        mask = vw[:, None, None, :] | torch.eye(w, dtype=torch.bool, device=x.device)[None, None]
        out = self.attn(xw, mask).view(b, n_win * w, d)
        return out[:, offset:offset + n]

    def forward(self, x: Tensor, valid: Tensor, offset: int = 0) -> Tensor:
        x = x + self._attend(x, valid, offset) * valid[..., None]
        return x + self.ffn(self.norm(x))


class SparseEncoder(nn.Module):
    """``EncoderConfig`` width, depth, heads and window over padded hit tokens; no final norm."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([SparseBlock(config.d_model, config.n_heads, config.window,
                                                 config.d_ff_resolved, config.norm_eps)
                                     for _ in range(config.n_blocks)])
        init_mup_linears(self)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        for level, block in enumerate(self.blocks):
            x = block(x, valid, offset=(self.config.shift if level % 2 else 0))
        return x * valid[..., None]


class IsolatedPredictor(nn.Module):
    """2 blocks at d_pred over [context; queries]; queries see the context and themselves only.

    Every row may attend to the valid context tokens and to itself; no row attends to a query
    other than itself, so a query's output does not depend on which other queries are present.
    """

    def __init__(self, config: EncoderConfig, d_out: int, attention: str = "isolated"):
        super().__init__()
        if attention not in ("isolated", "full"):
            raise ValueError("attention must be 'isolated' or 'full'")
        self.config = config
        self.attention = attention  # r2-sparse2: "full" lets every row attend to all valid queries too
        self.in_proj = nn.Linear(config.d_model, config.d_pred)
        self.blocks = nn.ModuleList([_MaskedBlock(config.d_pred, config.n_pred_heads,
                                                  config.d_pred_ff_resolved, config.norm_eps)
                                     for _ in range(config.n_pred_blocks)])
        self.out_proj = nn.Linear(config.d_pred, d_out)
        init_mup_linears(self)

    def forward(self, context: Tensor, context_valid: Tensor, queries: Tensor,
                query_valid: Tensor | None = None) -> Tensor:
        b, lc, _ = context.shape
        lq = queries.size(1)
        x = torch.cat([self.in_proj(context), queries], dim=1)
        n = lc + lq
        if self.attention == "full":
            q_ok = (torch.ones(b, lq, dtype=torch.bool, device=x.device) if query_valid is None
                    else query_valid.to(torch.bool))
            key_ok = torch.cat([context_valid, q_ok], 1)
        else:
            key_ok = torch.cat([context_valid, torch.zeros(b, lq, dtype=torch.bool, device=x.device)], 1)
        mask = key_ok[:, None, None, :] | torch.eye(n, dtype=torch.bool, device=x.device)[None, None]
        for block in self.blocks:
            x = block(x, mask)
        return self.out_proj(x[:, lc:])


class QueryEmbedding(nn.Module):
    """Query = learned query embedding + linear projection of the positional code (§2.2a)."""

    def __init__(self, d_pred: int, init: float = 0.02):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(d_pred))
        self.proj = nn.Linear(PE_DIM, d_pred)
        init_mup_linears(self)
        nn.init.normal_(self.embed, std=init)

    def forward(self, pe: Tensor) -> Tensor:
        return self.embed + self.proj(pe)


def legacy_predict(predictor, context: Tensor, context_valid: Tensor, queries: Tensor,
                   query_valid: Tensor | None = None) -> Tensor:
    """r2-sparse2 S1L: C2's ``TargetQueryPredictor`` (its own weights, statement for statement) on a
    padded context. Full attention over [context; queries] as in C2, except that padding keys are
    masked out (every row may attend to the valid context tokens, the valid queries and itself).
    With an all-valid context this equals ``predictor(context, queries)`` (focused test)."""
    if predictor.config.predictor_window is not None:
        raise ValueError("the legacy decoder of S1L is the full-attention TargetQueryPredictor")
    b, lc, _ = context.shape
    lq = queries.size(1)
    x = torch.cat([predictor.in_proj(context), predictor.query_proj(queries)], dim=1)
    n = lc + lq
    q_ok = (torch.ones(b, lq, dtype=torch.bool, device=x.device) if query_valid is None
            else query_valid.to(torch.bool))
    key_ok = torch.cat([context_valid.to(torch.bool), q_ok], 1)
    mask = key_ok[:, None, None, :] | torch.eye(n, dtype=torch.bool, device=x.device)[None, None]
    for block in predictor.blocks:
        a = block.attn
        q, k, v = a.qkv(a.norm(x)).view(b, n, 3, a.n_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + a.proj(att.transpose(1, 2).reshape(b, n, x.size(-1)))
        x = x + block.ffn(block.norm(x))
    return predictor.out_proj(x[:, lc:])


def linear_token_features(features: Tensor) -> Tensor:
    """r2-sparse2: C2's encoder input for hit tokens: the 9 standard features with ln(1 + q) as the
    charge (as C2's sentinel placeholder does for hit PMTs) and the hit flag set to 1."""
    out = features.clone()
    out[..., LOG_CHARGE_INDEX] = torch.log1p(torch.exp(features[..., LOG_CHARGE_INDEX]))
    out[..., HIT_INDEX] = 1.0
    return out


# ============================================================================ dense C3
class DenseFourierEncoder(nn.Module):
    def __init__(self, config: EncoderConfig, pe: Tensor):
        super().__init__()
        self.config = config
        self.content = ContentMLP(3, config.d_model)
        self.mask_embed = nn.Parameter(torch.zeros(config.d_model))
        self.blocks = nn.ModuleList([PointBlock(config.d_model, config.n_heads, config.window,
                                                config.d_ff_resolved, config.norm_eps)
                                     for _ in range(config.n_blocks)])
        self.final_norm = None
        init_mup_linears(self.blocks)
        nn.init.normal_(self.mask_embed, std=config.mask_embed_init)
        self.register_buffer("pe", pe, persistent=False)

    def content_tokens(self, features: Tensor) -> Tensor:
        return self.content(features[..., [LOG_CHARGE_INDEX, TIME_INDEX, HIT_INDEX]])

    def forward(self, features: Tensor, hidden_mask: Tensor | None = None,
                order: Tensor | None = None) -> Tensor:
        x = self.content_tokens(features) + self.pe
        if hidden_mask is not None:
            x = torch.where(hidden_mask.unsqueeze(-1), self.mask_embed + self.pe, x)
        if order is not None:
            x = x.index_select(1, order)
        for level, block in enumerate(self.blocks):
            x = block(x, offset=(self.config.shift if level % 2 else 0))
        if order is not None:
            x = x.index_select(1, torch.argsort(order))
        return x


class DenseFourierJepa(nn.Module):
    """C3: drop-in for ``LucidPointJepa`` (``encode``, ``predict``) with the §2.2a design."""

    def __init__(self, config: EncoderConfig, geometry, scaling: FeatureScaling):
        super().__init__()
        self.config, self.scaling, self.geometry_digest = config, scaling, geometry.digest
        code, self.pe_meta = fourier_code(geometry, scaling)
        pe = torch.as_tensor(code)
        self.encoder = DenseFourierEncoder(config, pe)
        self.query = QueryEmbedding(config.d_pred)
        self.predictor = IsolatedPredictor(config, config.d_model)
        self.register_buffer("token_order", torch.as_tensor(token_hilbert_order(geometry), dtype=torch.long),
                             persistent=False)
        self.register_buffer("pe", pe, persistent=False)

    def encode(self, features: Tensor, hidden_mask: Tensor | None = None) -> Tensor:
        return self.encoder(features, hidden_mask=hidden_mask, order=self.token_order)

    def predict(self, readout: Tensor, context_index: Tensor, target_index: Tensor) -> Tensor:
        width = readout.size(-1)
        context = readout.gather(1, context_index.unsqueeze(-1).expand(-1, -1, width))
        valid = torch.ones(context.shape[:2], dtype=torch.bool, device=readout.device)
        return self.predictor(context, valid, self.query(self.pe[target_index]))

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


class FactorJepa(nn.Module):
    """r2-follow §2.1: one C3 component on the C2 model, the other kept as in C2.

    ``embed="fourier", decoder="legacy"`` (C3e): the C3 encoder (content MLP + fixed positional
    code, mask embedding + code) with the C2 predictor (``TargetQueryPredictor`` on the scaled
    PMT positions, full attention). ``embed="linear", decoder="isolated"`` (C2dc): the C2 encoder
    (``PointEncoder``, linear input projection) with the C3 predictor (``QueryEmbedding`` of the
    fixed positional code, query-isolated attention). Each part is the exact module class of the
    cell it comes from and is built first in the same order, so at a given seed the encoder's
    initial weights equal those of the parent cell (C3 for C3e, C2 for C2dc).
    """

    def __init__(self, config: EncoderConfig, geometry, scaling: FeatureScaling, *, embed: str,
                 decoder: str):
        super().__init__()
        from .point_encoder import QUERY_FEATURES, PointEncoder, TargetQueryPredictor, geometry_features
        if (embed, decoder) not in (("fourier", "legacy"), ("linear", "isolated")):
            raise ValueError("FactorJepa builds only C3e (fourier, legacy) and C2dc (linear, isolated)")
        self.config, self.scaling, self.geometry_digest = config, scaling, geometry.digest
        self.embed, self.decoder = embed, decoder
        code, self.pe_meta = fourier_code(geometry, scaling)
        pe = torch.as_tensor(code)
        self.encoder = DenseFourierEncoder(config, pe) if embed == "fourier" else PointEncoder(config)
        if decoder == "isolated":
            self.query = QueryEmbedding(config.d_pred)
            self.predictor = IsolatedPredictor(config, config.d_model)
        else:
            self.predictor = TargetQueryPredictor(config)
            queries = geometry_features(geometry, scaling)[:, :QUERY_FEATURES]
            self.register_buffer("query_features", torch.as_tensor(queries, dtype=torch.float32),
                                 persistent=False)
        self.register_buffer("token_order", torch.as_tensor(token_hilbert_order(geometry), dtype=torch.long),
                             persistent=False)
        self.register_buffer("pe", pe, persistent=False)

    def encode(self, features: Tensor, hidden_mask: Tensor | None = None) -> Tensor:
        return self.encoder(features, hidden_mask=hidden_mask, order=self.token_order)

    def predict(self, readout: Tensor, context_index: Tensor, target_index: Tensor) -> Tensor:
        width = readout.size(-1)
        context = readout.gather(1, context_index.unsqueeze(-1).expand(-1, -1, width))
        if self.decoder == "isolated":
            valid = torch.ones(context.shape[:2], dtype=torch.bool, device=readout.device)
            return self.predictor(context, valid, self.query(self.pe[target_index]))
        return self.predictor(context, self.query_features[target_index])

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))


# ============================================================================ loss
def fixed_loss(out: Tensor, ln_q: Tensor, tau: Tensor, hit: Tensor, valid: Tensor, *,
               time_scale_ns: float, hit_term: bool = True) -> dict:
    """§2.2: class-balanced BCE on the queried set, ln q MSE and 2 x Huber_0.2 time on hit PMTs.

    ``out[..., 0]`` = ln q prediction, ``out[..., 1]`` = tau prediction, ``out[..., 2]`` = hit
    logit; ``valid`` marks real (non-padding) queries. Positives get w = clip(n_neg / n_pos, 0.05,
    20) over the queried set of the (micro-)batch, negatives 1; the term is divided by the weight sum.
    """
    hit = hit & valid
    zero = out.sum() * 0.0
    terms = {}
    if hit_term:
        n_pos = hit.sum().float()
        n_neg = (valid & ~hit).sum().float()
        w_pos = torch.clamp(n_neg / n_pos.clamp_min(1.0), *POS_WEIGHT_CLIP)
        weights = torch.where(hit, w_pos, torch.ones_like(w_pos)) * valid
        bce = F.binary_cross_entropy_with_logits(out[..., 2], hit.float(), reduction="none")
        terms["hit"] = (bce * weights).sum() / weights.sum().clamp_min(1e-12)
        terms["pos_weight"] = w_pos.detach()
    if bool(hit.any()):
        terms["charge"] = ((out[..., 0] - ln_q)[hit] ** 2).mean()
        r = (out[..., 1] - tau)[hit] * (float(time_scale_ns) / TIME_LOSS_UNIT_NS)
        terms["time"] = 2.0 * F.huber_loss(r, torch.zeros_like(r), delta=HUBER_DELTA, reduction="mean")
    else:
        terms["charge"], terms["time"] = zero, zero
    terms["loss"] = terms["charge"] + terms["time"] + (terms["hit"] if hit_term else zero)
    return terms


def poisson_count_loss(log_rate: Tensor, count: Tensor) -> Tensor:
    return F.poisson_nll_loss(log_rate, count.float(), log_input=True, full=True)  # Stirling term: constant, readable logs


# ============================================================================ tokens
def gather_tokens(select: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
    """Indices (B, L) of the selected PMTs sorted by ``key`` (stable: ties keep PMT-index order),
    padded at the end; and the (B, L) validity mask."""
    b, n = select.shape
    inf = torch.finfo(torch.float32).max
    k = torch.where(select, key.float().expand(b, n), torch.full((b, n), inf, device=select.device))
    order = torch.sort(k, dim=1, stable=True).indices
    counts = select.sum(1)
    length = max(int(counts.max()), 1)
    idx = order[:, :length]
    valid = torch.arange(length, device=select.device)[None, :] < counts[:, None]
    return idx, valid


def scatter_tokens(tokens: Tensor, index: Tensor, valid: Tensor, n_pmt: int) -> Tensor:
    """(B, L, d) hit tokens -> (B, N_pmt, d) with zeros at PMTs without a token (readout adapter)."""
    b, _, d = tokens.shape
    out = tokens.new_zeros((b, n_pmt, d))
    src = tokens * valid[..., None]
    safe = torch.where(valid, index, torch.zeros_like(index))
    out.scatter_add_(1, safe[..., None].expand(-1, -1, d), src)
    return out


def _mark(shape, index: Tensor, valid: Tensor) -> Tensor:
    """(B, N) bool mask that is True at ``index[valid]`` (padding never overwrites a real entry)."""
    out = torch.zeros(shape, dtype=torch.int32, device=index.device)
    out.scatter_add_(1, torch.where(valid, index, torch.zeros_like(index)), valid.to(torch.int32))
    return out > 0


def _gather(x: Tensor, index: Tensor) -> Tensor:
    if x.ndim == 2:
        return x.gather(1, index)
    return x.gather(1, index[..., None].expand(-1, -1, x.size(-1)))


# ============================================================================ sparse arms
class _SparseBase(nn.Module):
    """Shared parts: positional code, content MLP over (ln q, tau), sparse encoder, Muon groups."""

    def __init__(self, arm: str, geometry, *, encoder_config: EncoderConfig, scaling: FeatureScaling,
                 seed: int, embed: str = "fourier"):
        super().__init__()
        if embed not in EMBED_MODES:
            raise ValueError(f"embed must be one of {EMBED_MODES}")
        self.arm, self.seed = arm, int(seed)
        self.embed = embed
        self.n_pmt = int(geometry.n_pmt)
        torch.manual_seed(self.seed)
        self.config = encoder_config
        self.scaling = scaling
        self.d_model = int(encoder_config.d_model)
        code, self.pe_meta = fourier_code(geometry, scaling)
        self.register_buffer("pe", torch.as_tensor(code), persistent=False)
        rank = np.empty(self.n_pmt, dtype=np.int64)
        rank[token_hilbert_order(geometry)] = np.arange(self.n_pmt)
        self.register_buffer("hilbert_rank", torch.as_tensor(rank), persistent=False)
        self._c2_parts = None
        if embed == "fourier":   # the unchanged S1 / S2 construction (same RNG draws)
            self.content = ContentMLP(2, self.d_model)
            self.encoder = SparseEncoder(encoder_config)
        else:
            self._c2_parts = self._build_c2_linear(geometry, encoder_config, scaling)
        self._generator = torch.Generator().manual_seed(self.seed + 100003)

    def _build_c2_linear(self, geometry, encoder_config: EncoderConfig, scaling: FeatureScaling) -> dict:
        """r2-sparse2: C2's modules at this seed, drawn in C2's order (``LucidArm``:
        ``torch.manual_seed(seed)``, ``LucidPointJepa``, then the ``Linear(d_model, 3)`` head), so the
        input projection, the encoder blocks, the legacy predictor and the head start from C2's
        initial weights. The encoder blocks are copied into the padded-token ``SparseEncoder`` (same
        parameter names and shapes). Returns the parts a subclass may use (predictor, head, queries)."""
        from .point_encoder import LucidPointJepa
        torch.manual_seed(self.seed)
        donor = LucidPointJepa(encoder_config, geometry, scaling)
        head = nn.Linear(self.d_model, 3)
        self.input_proj = donor.encoder.input_proj
        self.encoder = SparseEncoder(encoder_config)
        self.encoder.blocks.load_state_dict(donor.encoder.blocks.state_dict())
        return {"predictor": donor.predictor, "head": head,
                "query_features": donor.query_features.detach().clone()}

    # optimiser contract of LucidArm
    def parameter_groups(self) -> dict:
        muon, adamw = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (muon if ".blocks." in f".{name}" and parameter.ndim >= 2 else adamw).append(parameter)
        return {"muon": muon, "adamw": adamw}

    def check_parameter_coverage(self) -> dict:
        grouped = self.parameter_groups()
        seen = [id(p) for g in grouped.values() for p in g]
        trainable = [id(p) for p in self.parameters() if p.requires_grad]
        if len(seen) != len(set(seen)) or set(seen) != set(trainable):
            raise RuntimeError("optimizer groups do not partition the trainable parameters")
        return {"n_trainable": len(trainable), "n_muon": len(grouped["muon"]), "n_adamw": len(grouped["adamw"])}

    def token_inputs(self, features: Tensor, index: Tensor) -> Tensor:
        """Token = content MLP(ln q, tau) + positional code, for the PMTs in ``index``; with
        ``embed="linear"``, C2's ``Linear(9, d_model)`` over the 9 standard features (hit flag 1)."""
        if self.embed == "linear":
            return self.input_proj(linear_token_features(_gather(features, index)))
        content = _gather(features[..., [LOG_CHARGE_INDEX, TIME_INDEX]], index)
        return self.content(content) + self.pe[index]

    def encode_hits(self, features: Tensor, select: Tensor, order: str) -> tuple[Tensor, Tensor, Tensor]:
        key = self.hilbert_rank[None, :] if order == "hilbert" else features[..., TIME_INDEX]
        idx, valid = gather_tokens(select, key)
        z = self.encoder(self.token_inputs(features, idx) * valid[..., None], valid)
        return z, idx, valid

    def encode_tokens(self, features: Tensor, hidden_mask: Tensor | None = None) -> Tensor:
        """Readout: all hit tokens (none hidden) scattered to (B, N_pmt, d), zeros at unhit PMTs."""
        hit = features[..., HIT_INDEX] > 0.5
        if hidden_mask is not None:
            hit = hit & ~hidden_mask
        z, idx, valid = self.encode_hits(features, hit, self.token_order)
        return scatter_tokens(z, idx, valid, self.n_pmt)

    def content_rms(self, features: Tensor) -> dict:
        hit = features[..., HIT_INDEX] > 0.5
        idx, valid = gather_tokens(hit, self.hilbert_rank[None, :])
        if self.embed == "linear":
            c = self.token_inputs(features, idx)[valid]
            return {"content_rms": float(c.pow(2).mean().sqrt()), "pe_rms": 0.0}
        c = self.content(_gather(features[..., [LOG_CHARGE_INDEX, TIME_INDEX]], idx))[valid]
        return {"content_rms": float(c.pow(2).mean().sqrt()), "pe_rms": float(self.pe.pow(2).mean().sqrt())}


class SparseMaeArm(_SparseBase):
    """S1 / S1n / S1p: hit-only encoder (masked hits removed), isolated queries, count head."""

    token_order = "hilbert"

    def __init__(self, geometry, *, encoder_config, scaling, seed, queries: str = "all",
                 neg_ratio: float | None = None, embed: str = "fourier", decoder: str = "isolated"):
        if decoder not in ("isolated", "legacy"):
            raise ValueError("sparse_mae decoder must be 'isolated' or 'legacy'")
        if decoder == "legacy" and queries != "all":
            # r2-sparse2 T1.2: neg / pos query sets depend on the hits; full attention among queries
            # would expose them, so the legacy decoder is refused with them
            raise ValueError("--decoder legacy needs --queries all (neg/pos query sets depend on hits)")
        if decoder == "legacy" and embed != "linear":
            raise ValueError("--decoder legacy is registered with --embed linear only (S1L)")
        super().__init__("sparse_mae", geometry, encoder_config=encoder_config, scaling=scaling, seed=seed,
                         embed=embed)
        if queries not in QUERY_MODES:
            raise ValueError(f"queries must be one of {QUERY_MODES}")
        if queries == "neg" and not (neg_ratio and neg_ratio > 0):
            raise ValueError("S1n needs a positive neg_ratio (fixed in W2)")
        self.queries, self.neg_ratio = queries, (float(neg_ratio) if queries == "neg" else None)
        self.decoder = decoder
        if decoder == "legacy":
            # S1L: C2's predictor and head (C2 init at this seed), queries = scaled xyz, no count head
            parts, self._c2_parts = self._c2_parts, None
            self.predictor = parts["predictor"]
            self.head = parts["head"]
            self.count_head = None
            self.register_buffer("query_features", parts["query_features"], persistent=False)
            return
        self._c2_parts = None
        self.query = QueryEmbedding(encoder_config.d_pred)
        self.predictor = IsolatedPredictor(encoder_config, self.d_model)
        self.head = nn.Linear(self.d_model, 3)
        self.count_head = nn.Linear(self.d_model, 1)
        init_mup_linears(self.head)
        init_mup_linears(self.count_head)
        nn.init.constant_(self.count_head.bias, math.log(max(self.n_pmt / 4.0, 1.0)))
        spacing = median_nn_spacing(geometry.positions_m)
        p = np.asarray(geometry.positions_m, dtype=np.float64)
        near = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)) <= 2.0 * spacing
        self.register_buffer("near", torch.as_tensor(near), persistent=False)
        self.hard_radius_m = 2.0 * spacing

    def sample_negatives(self, hidden: Tensor, hit: Tensor) -> Tensor:
        """S1n query set: every masked hit + k = min(n_unhit_masked, ceil(r n_hit_masked)) masked
        unhit PMTs, half uniform and half within two median spacings of a masked hit ("hard";
        topped up uniformly when fewer exist). Draws use the arm's generator (checkpointed)."""
        b, n = hidden.shape
        out = (hidden & hit).clone()
        near = self.near.float()
        for row in range(b):
            mh = hidden[row] & hit[row]
            mu = hidden[row] & ~hit[row]
            n_u = int(mu.sum())
            k = min(n_u, int(math.ceil(self.neg_ratio * int(mh.sum()))))
            if k <= 0:
                continue
            hard = mu & ((near @ mh.float()) > 0)
            cand_h = torch.nonzero(hard, as_tuple=False)[:, 0].cpu()
            n_hard = min(int(cand_h.numel()), k // 2)
            chosen = torch.zeros(n, dtype=torch.bool)
            if n_hard:
                pick = torch.randperm(cand_h.numel(), generator=self._generator)[:n_hard]
                chosen[cand_h[pick]] = True
            rest = torch.nonzero(mu.cpu() & ~chosen, as_tuple=False)[:, 0]
            pick = torch.randperm(rest.numel(), generator=self._generator)[:k - n_hard]
            chosen[rest[pick]] = True
            out[row] |= chosen.to(out.device)
        return out

    def query_set(self, hidden: Tensor, hit: Tensor) -> Tensor:
        if self.queries == "all":
            return hidden.clone()
        if self.queries == "pos":
            return hidden & hit
        return self.sample_negatives(hidden, hit)

    def predict_queries(self, z: Tensor, valid: Tensor, q_idx: Tensor, q_valid: Tensor | None = None) -> Tensor:
        if self.decoder == "legacy":
            return self.head(legacy_predict(self.predictor, z, valid, self.query_features[q_idx], q_valid))
        return self.head(self.predictor(z, valid, self.query(self.pe[q_idx])))

    def query_inputs(self, q_idx: Tensor) -> Tensor:
        """The raw per-query input (leakage test (b)): scaled xyz for S1L, the positional code else."""
        return self.query_features[q_idx] if self.decoder == "legacy" else self.pe[q_idx]

    def count_rate(self, z: Tensor, valid: Tensor) -> Tensor:
        pooled = (z * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.count_head(pooled)[:, 0]

    def forward(self, batch, *, oracle: bool = False, twin_batch=None, step: int = 0) -> dict:
        if oracle or twin_batch is not None:
            raise PermissionError("the nominal twin is not used by sparse arms")
        x = batch.features
        hit = x[..., HIT_INDEX] > 0.5
        hidden = batch.hidden
        z, idx, valid = self.encode_hits(x, hit & ~hidden, "hilbert")
        qsel = self.query_set(hidden, hit)
        q_idx, q_valid = gather_tokens(qsel, self.hilbert_rank[None, :])
        out = self.predict_queries(z, valid, q_idx, q_valid)
        terms = fixed_loss(out, _gather(x[..., LOG_CHARGE_INDEX], q_idx), _gather(x[..., TIME_INDEX], q_idx),
                           _gather(hit, q_idx), q_valid, time_scale_ns=self.scaling.time_scale_ns,
                           hit_term=self.queries != "pos")
        n_masked_hits = (hidden & hit).sum(1)
        metrics = {"charge": float(terms["charge"].detach()), "time": float(terms["time"].detach()),
                   "hidden_hits": int(n_masked_hits.sum()), "n_queries": int(q_valid.sum()),
                   "n_context_tokens": int(valid.sum())}
        if self.count_head is not None:
            count = poisson_count_loss(self.count_rate(z, valid), n_masked_hits)
            loss = terms["loss"] + COUNT_WEIGHT * count
            metrics["count"] = float(count.detach())
        else:
            loss = terms["loss"]
        metrics = {"loss": float(loss.detach()), **metrics}
        if "hit" in terms:
            metrics.update(hit=float(terms["hit"].detach()), pos_weight=float(terms["pos_weight"]))
        return {"loss": loss, "metrics": metrics, "prediction": out}


class SparseTimeArm(_SparseBase):
    """S2: time-ordered hit tokens, hidden time blocks, position prediction by soft many-to-one
    assignment onto the fixed positional code (§2.3). No count head.

    r2-sparse2: ``embed="linear"`` (C2's input projection), ``decoder="full"`` (full attention among
    context and queries) and ``blocks="time"`` (Delta-wide time bins, ``block_width_ns``); the
    defaults build the unchanged S2."""

    token_order = "time"

    def __init__(self, geometry, *, encoder_config, scaling, seed, mask_seed: int = lb.DEFAULT_SEED,
                 embed: str = "fourier", decoder: str = "isolated", blocks: str = "count32",
                 block_width_ns: float | None = None):
        if decoder not in ("isolated", "full"):
            raise ValueError("sparse_time decoder must be 'isolated' or 'full'")
        if blocks not in TIME_BLOCK_MODES:
            raise ValueError(f"blocks must be one of {TIME_BLOCK_MODES}")
        if (blocks == "time") != (block_width_ns is not None):
            raise ValueError("--block-width-ns goes with --blocks time (and only with it)")
        if block_width_ns is not None and not (math.isfinite(block_width_ns) and block_width_ns > 0):
            raise ValueError("block_width_ns must be positive and finite")
        super().__init__("sparse_time", geometry, encoder_config=encoder_config, scaling=scaling, seed=seed,
                         embed=embed)
        self._c2_parts = None   # S2L / S2Lt use C2's input projection and encoder init only
        self.mask_seed = int(mask_seed)
        self.decoder, self.blocks = decoder, blocks
        self.block_width_ns = None if block_width_ns is None else float(block_width_ns)
        if blocks == "time":
            span = S2_WINDOW_NS[1] - S2_WINDOW_NS[0]
            self.n_blocks = int(math.ceil(span / self.block_width_ns)) + 1   # + 1: a hit exactly at +50 ns
        else:
            self.n_blocks = S2_MAX_BLOCKS
        d_pred = encoder_config.d_pred
        self.mask_embed = nn.Parameter(torch.zeros(d_pred))
        self.time_mlp = nn.Sequential(nn.Linear(1, d_pred), nn.GELU(), nn.Linear(d_pred, d_pred))
        self.block_embed = nn.Embedding(self.n_blocks, d_pred)
        self.predictor = (IsolatedPredictor(encoder_config, PE_DIM) if decoder == "isolated"
                          else IsolatedPredictor(encoder_config, PE_DIM, attention="full"))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        init_mup_linears(self.time_mlp)
        nn.init.normal_(self.mask_embed, std=0.02)
        nn.init.normal_(self.block_embed.weight, std=0.02)

    def time_blocks(self, event_ids, n_hits: Tensor) -> list:
        """Per event: the hidden block indices (a keyed half of the full 32-token blocks; the
        trailing partial block is always visible)."""
        out = []
        for eid, n in zip(np.asarray(event_ids).tolist(), n_hits.tolist()):
            n_full = int(n) // S2_BLOCK
            rng = lb.keyed_rng(self.mask_seed, NS_TIME_BLOCKS, int(eid))
            out.append(np.sort(rng.permutation(n_full)[:(n_full + 1) // 2]) if n_full else np.empty(0, int))
        return out

    def time_bin(self, tau: Tensor) -> Tensor:
        """S2Lt bin index of trigger-relative times (tau = t / T): floor((t + 110 ns) / Delta),
        clipped to [0, n_blocks - 1]."""
        t_ns = tau * float(self.scaling.time_scale_ns)
        b = torch.floor((t_ns - S2_WINDOW_NS[0]) / self.block_width_ns).to(torch.long)
        return b.clamp(0, self.n_blocks - 1)

    def hidden_bins(self, event_ids, bins: Tensor, valid: Tensor) -> list:
        """Per event: the hidden bin indices: among the non-empty bins (ascending), a keyed half
        (ceil(n / 2)), drawn with the r2-sparse time-block RNG namespace."""
        out = []
        for row, eid in enumerate(np.asarray(event_ids).tolist()):
            nonempty = torch.unique(bins[row][valid[row]]).cpu().numpy()
            n = int(nonempty.size)
            rng = lb.keyed_rng(self.mask_seed, NS_TIME_BLOCKS, int(eid))
            out.append(np.sort(nonempty[rng.permutation(n)[:(n + 1) // 2]]) if n else np.empty(0, int))
        return out

    def layout(self, batch):
        """Time-ordered hit list, hidden/visible split and query descriptors (predictor inputs)."""
        x = batch.features
        hit = x[..., HIT_INDEX] > 0.5
        order_idx, order_valid = gather_tokens(hit, x[..., TIME_INDEX])   # time order, ties by PMT index
        n_hits = order_valid.sum(1)
        pos = torch.arange(order_idx.size(1), device=x.device)
        hidden_pos = torch.zeros_like(order_valid)
        if self.blocks == "time":
            block_of_pos = self.time_bin(_gather(x[..., TIME_INDEX], order_idx))          # (B, L)
            blocks = self.hidden_bins(batch.event_ids, block_of_pos, order_valid)
            for row, hb in enumerate(blocks):
                if hb.size:
                    hidden_pos[row] = (torch.isin(block_of_pos[row], torch.as_tensor(hb, device=x.device))
                                       & order_valid[row])
        else:
            blocks = self.time_blocks(batch.event_ids, n_hits)
            block_of_pos = pos // S2_BLOCK
            for row, hb in enumerate(blocks):
                if hb.size:
                    hidden_pos[row] = torch.isin(block_of_pos, torch.as_tensor(hb, device=x.device)) & order_valid[row]
        visible_pos = order_valid & ~hidden_pos
        # visible tokens keep time order (stable gather of positions)
        vis_sel = _mark(hit.shape, order_idx, visible_pos)
        v_idx, v_valid = gather_tokens(vis_sel, x[..., TIME_INDEX])
        # queries: one per hidden hit, in time order
        h_pos_idx, h_valid = gather_tokens(hidden_pos, pos[None, :].float())
        q_pmt = order_idx.gather(1, h_pos_idx)
        q_block = block_of_pos.gather(1, h_pos_idx) if block_of_pos.ndim == 2 else block_of_pos[h_pos_idx]
        q_tau = _gather(x[..., TIME_INDEX], q_pmt)
        return {"v_idx": v_idx, "v_valid": v_valid, "vis_sel": vis_sel, "q_pmt": q_pmt, "q_block": q_block,
                "q_tau": q_tau, "q_valid": h_valid, "hit": hit}

    def predictor_inputs(self, batch) -> dict:
        lay = self.layout(batch)
        x = batch.features
        tokens = self.token_inputs(x, lay["v_idx"]) * lay["v_valid"][..., None]
        q = (self.mask_embed + self.time_mlp(lay["q_tau"][..., None])
             + self.block_embed(lay["q_block"].clamp_max(self.n_blocks - 1)))
        return dict(lay, tokens=tokens, queries=q * lay["q_valid"][..., None])

    def run_predictor(self, p: dict) -> Tensor:
        z = self.encoder(p["tokens"], p["v_valid"])
        if self.decoder == "full":
            return self.predictor(z, p["v_valid"], p["queries"], p["q_valid"])
        return self.predictor(z, p["v_valid"], p["queries"])

    def assignment_mass(self, e_hat: Tensor, p: dict) -> Tensor:
        scores = torch.einsum("bqd,nd->bqn", e_hat, self.pe) / torch.exp(self.log_temperature)
        cand = ~p["vis_sel"]                                                     # (B, N)
        scores = scores.masked_fill(~cand[:, None, :], float("-inf"))
        return torch.softmax(scores, dim=-1) * p["q_valid"][..., None]           # (B, Q, N)

    def forward(self, batch, *, oracle: bool = False, twin_batch=None, step: int = 0) -> dict:
        if oracle or twin_batch is not None:
            raise PermissionError("the nominal twin is not used by sparse arms")
        p = self.predictor_inputs(batch)
        e_hat = self.run_predictor(p)                                            # (B, Q, 128)
        mass = self.assignment_mass(e_hat, p)
        b, q, n = mass.shape
        blk = p["q_block"].clamp_max(self.n_blocks - 1)
        lam = mass.new_zeros((b, self.n_blocks, n))
        lam.scatter_add_(1, blk[..., None].expand(-1, -1, n), mass)
        own = lam.gather(1, blk[..., None].expand(-1, -1, n))                    # (B,Q,N)
        lam_at_target = own.gather(2, p["q_pmt"][..., None])[..., 0]            # λ_k(p) at each hidden hit
        valid = p["q_valid"]
        nll = -torch.log(lam_at_target + S2_EPS)
        if self.blocks == "time":
            # §2.3: per hidden bin, divided by that bin's number of hidden hits, then averaged over bins
            vf = valid.to(nll.dtype)
            sums = nll.new_zeros((b, self.n_blocks)).scatter_add_(1, blk, nll * vf)
            counts = nll.new_zeros((b, self.n_blocks)).scatter_add_(1, blk, vf)
            present = counts > 0
            loss = (sums[present] / counts[present]).sum() / present.sum().clamp_min(1)
        else:
            loss = (nll * valid).sum() / valid.sum().clamp_min(1)
        with torch.no_grad():
            hit = p["hit"]
            denom = valid.sum().clamp_min(1)
            on_unhit = (mass * (~hit)[:, None, :]).sum() / denom
            hidden_hit = _mark(hit.shape, p["q_pmt"], valid)
            on_hidden = (mass * hidden_hit[:, None, :]).sum() / denom
        metrics = {"loss": float(loss.detach()), "assign_nll": float(loss.detach()),
                   "mass_on_unhit": float(on_unhit), "mass_on_hidden_hits": float(on_hidden),
                   "temperature": float(torch.exp(self.log_temperature).detach()),
                   "hidden_hits": int(valid.sum()), "n_context_tokens": int(p["v_valid"].sum())}
        return {"loss": loss, "metrics": metrics, "prediction": e_hat}

    @torch.no_grad()
    def block_mass_stats(self, batch) -> dict:
        """r2-sparse2 §2.4 diagnostic (sums over the batch; divide at the end).

        Per query: ``own`` = mass on hidden hits of its own block / bin, ``other`` = mass on hidden
        hits of other blocks, ``unhit`` = mass on unhit PMTs (total mass 1 per query). Chance level:
        per hidden block k of an event, n_k / (all hidden hits of the event); summed over blocks
        (``chance_block_sum`` / ``n_blocks``, the registered W1 definition) and over queries
        (``chance_query_sum`` / ``queries``)."""
        p = self.predictor_inputs(batch)
        mass = self.assignment_mass(self.run_predictor(p), p)
        valid, blk, hit = p["q_valid"], p["q_block"], p["hit"]
        b, q, n = mass.shape
        # block of each hidden-hit PMT (-1 elsewhere); padding queries write to a dropped extra column
        blk_of_pmt = torch.full((b, n + 1), -1, dtype=torch.long, device=mass.device)
        blk_of_pmt.scatter_(1, torch.where(valid, p["q_pmt"], torch.full_like(p["q_pmt"], n)), blk)
        blk_of_pmt = blk_of_pmt[:, :n]
        is_hidden_hit = blk_of_pmt >= 0
        same = blk_of_pmt[:, None, :] == blk[..., None]                         # (B, Q, N)
        own = (mass * (same & is_hidden_hit[:, None, :])).sum(-1)
        other = (mass * (~same & is_hidden_hit[:, None, :])).sum(-1)
        unhit = (mass * (~hit)[:, None, :]).sum(-1)
        vf = valid.to(mass.dtype)
        counts = mass.new_zeros((b, self.n_blocks)).scatter_add_(1, blk.clamp_max(self.n_blocks - 1), vf)
        n_hidden = vf.sum(1, keepdim=True)
        ratio = torch.where(counts > 0, counts / n_hidden.clamp_min(1), torch.zeros_like(counts))
        per_query_chance = ratio.gather(1, blk.clamp_max(self.n_blocks - 1)) * vf
        return {"own": float((own * vf).sum()), "other": float((other * vf).sum()),
                "unhit": float((unhit * vf).sum()), "queries": float(vf.sum()),
                "chance_block_sum": float(ratio.sum()), "n_blocks": float((counts > 0).sum()),
                "chance_query_sum": float(per_query_chance.sum()),
                "own_over_hidden_sum": float((own / (own + other).clamp_min(1e-12) * vf).sum())}


def build_sparse_arm(arm: str, geometry, seed: int, *, encoder_config: EncoderConfig, scaling: FeatureScaling,
                     queries: str = "all", neg_ratio: float | None = None, mask_seed: int = lb.DEFAULT_SEED,
                     embed: str = "fourier", decoder: str = "isolated", blocks: str | None = None,
                     block_width_ns: float | None = None):
    """r2-sparse2 flags (embed, decoder, blocks, block_width_ns) default to the unchanged S1 / S2."""
    if arm == "sparse_mae":
        if blocks is not None or block_width_ns is not None:
            raise ValueError("--blocks / --block-width-ns apply to sparse_time only")
        return SparseMaeArm(geometry, encoder_config=encoder_config, scaling=scaling, seed=seed,
                            queries=queries, neg_ratio=neg_ratio, embed=embed, decoder=decoder)
    if arm == "sparse_time":
        return SparseTimeArm(geometry, encoder_config=encoder_config, scaling=scaling, seed=seed,
                             mask_seed=mask_seed, embed=embed, decoder=decoder,
                             blocks=blocks or "count32", block_width_ns=block_width_ns)
    raise ValueError(f"unknown sparse arm {arm!r}")
