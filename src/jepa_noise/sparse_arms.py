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
           "sentinel_features", "scatter_tokens"]

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

    def __init__(self, config: EncoderConfig, d_out: int):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_pred)
        self.blocks = nn.ModuleList([_MaskedBlock(config.d_pred, config.n_pred_heads,
                                                  config.d_pred_ff_resolved, config.norm_eps)
                                     for _ in range(config.n_pred_blocks)])
        self.out_proj = nn.Linear(config.d_pred, d_out)
        init_mup_linears(self)

    def forward(self, context: Tensor, context_valid: Tensor, queries: Tensor) -> Tensor:
        b, lc, _ = context.shape
        lq = queries.size(1)
        x = torch.cat([self.in_proj(context), queries], dim=1)
        n = lc + lq
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
                 seed: int):
        super().__init__()
        self.arm, self.seed = arm, int(seed)
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
        self.content = ContentMLP(2, self.d_model)
        self.encoder = SparseEncoder(encoder_config)
        self._generator = torch.Generator().manual_seed(self.seed + 100003)

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
        """Token = content MLP(ln q, tau) + positional code, for the PMTs in ``index``."""
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
        c = self.content(_gather(features[..., [LOG_CHARGE_INDEX, TIME_INDEX]], idx))[valid]
        return {"content_rms": float(c.pow(2).mean().sqrt()), "pe_rms": float(self.pe.pow(2).mean().sqrt())}


class SparseMaeArm(_SparseBase):
    """S1 / S1n / S1p: hit-only encoder (masked hits removed), isolated queries, count head."""

    token_order = "hilbert"

    def __init__(self, geometry, *, encoder_config, scaling, seed, queries: str = "all",
                 neg_ratio: float | None = None):
        super().__init__("sparse_mae", geometry, encoder_config=encoder_config, scaling=scaling, seed=seed)
        if queries not in QUERY_MODES:
            raise ValueError(f"queries must be one of {QUERY_MODES}")
        if queries == "neg" and not (neg_ratio and neg_ratio > 0):
            raise ValueError("S1n needs a positive neg_ratio (fixed in W2)")
        self.queries, self.neg_ratio = queries, (float(neg_ratio) if queries == "neg" else None)
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

    def predict_queries(self, z: Tensor, valid: Tensor, q_idx: Tensor) -> Tensor:
        return self.head(self.predictor(z, valid, self.query(self.pe[q_idx])))

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
        out = self.predict_queries(z, valid, q_idx)
        terms = fixed_loss(out, _gather(x[..., LOG_CHARGE_INDEX], q_idx), _gather(x[..., TIME_INDEX], q_idx),
                           _gather(hit, q_idx), q_valid, time_scale_ns=self.scaling.time_scale_ns,
                           hit_term=self.queries != "pos")
        n_masked_hits = (hidden & hit).sum(1)
        count = poisson_count_loss(self.count_rate(z, valid), n_masked_hits)
        loss = terms["loss"] + COUNT_WEIGHT * count
        metrics = {"loss": float(loss.detach()), "charge": float(terms["charge"].detach()),
                   "time": float(terms["time"].detach()), "count": float(count.detach()),
                   "hidden_hits": int(n_masked_hits.sum()), "n_queries": int(q_valid.sum()),
                   "n_context_tokens": int(valid.sum())}
        if "hit" in terms:
            metrics.update(hit=float(terms["hit"].detach()), pos_weight=float(terms["pos_weight"]))
        return {"loss": loss, "metrics": metrics, "prediction": out}


class SparseTimeArm(_SparseBase):
    """S2: time-ordered hit tokens, hidden time blocks, position prediction by soft many-to-one
    assignment onto the fixed positional code (§2.3). No count head."""

    token_order = "time"

    def __init__(self, geometry, *, encoder_config, scaling, seed, mask_seed: int = lb.DEFAULT_SEED):
        super().__init__("sparse_time", geometry, encoder_config=encoder_config, scaling=scaling, seed=seed)
        self.mask_seed = int(mask_seed)
        d_pred = encoder_config.d_pred
        self.mask_embed = nn.Parameter(torch.zeros(d_pred))
        self.time_mlp = nn.Sequential(nn.Linear(1, d_pred), nn.GELU(), nn.Linear(d_pred, d_pred))
        self.block_embed = nn.Embedding(S2_MAX_BLOCKS, d_pred)
        self.predictor = IsolatedPredictor(encoder_config, PE_DIM)
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

    def layout(self, batch):
        """Time-ordered hit list, hidden/visible split and query descriptors (predictor inputs)."""
        x = batch.features
        hit = x[..., HIT_INDEX] > 0.5
        order_idx, order_valid = gather_tokens(hit, x[..., TIME_INDEX])   # time order, ties by PMT index
        n_hits = order_valid.sum(1)
        blocks = self.time_blocks(batch.event_ids, n_hits)
        pos = torch.arange(order_idx.size(1), device=x.device)
        block_of_pos = pos // S2_BLOCK
        hidden_pos = torch.zeros_like(order_valid)
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
        q_block = block_of_pos[h_pos_idx]
        q_tau = _gather(x[..., TIME_INDEX], q_pmt)
        return {"v_idx": v_idx, "v_valid": v_valid, "vis_sel": vis_sel, "q_pmt": q_pmt, "q_block": q_block,
                "q_tau": q_tau, "q_valid": h_valid, "hit": hit}

    def predictor_inputs(self, batch) -> dict:
        lay = self.layout(batch)
        x = batch.features
        tokens = self.token_inputs(x, lay["v_idx"]) * lay["v_valid"][..., None]
        q = (self.mask_embed + self.time_mlp(lay["q_tau"][..., None])
             + self.block_embed(lay["q_block"].clamp_max(S2_MAX_BLOCKS - 1)))
        return dict(lay, tokens=tokens, queries=q * lay["q_valid"][..., None])

    def forward(self, batch, *, oracle: bool = False, twin_batch=None, step: int = 0) -> dict:
        if oracle or twin_batch is not None:
            raise PermissionError("the nominal twin is not used by sparse arms")
        p = self.predictor_inputs(batch)
        z = self.encoder(p["tokens"], p["v_valid"])
        e_hat = self.predictor(z, p["v_valid"], p["queries"])                    # (B, Q, 128)
        scores = torch.einsum("bqd,nd->bqn", e_hat, self.pe) / torch.exp(self.log_temperature)
        cand = ~p["vis_sel"]                                                     # (B, N)
        scores = scores.masked_fill(~cand[:, None, :], float("-inf"))
        mass = torch.softmax(scores, dim=-1) * p["q_valid"][..., None]           # (B, Q, N)
        b, q, n = mass.shape
        lam = mass.new_zeros((b, S2_MAX_BLOCKS, n))
        lam.scatter_add_(1, p["q_block"].clamp_max(S2_MAX_BLOCKS - 1)[..., None].expand(-1, -1, n), mass)
        own = lam.gather(1, p["q_block"].clamp_max(S2_MAX_BLOCKS - 1)[..., None].expand(-1, -1, n))  # (B,Q,N)
        lam_at_target = own.gather(2, p["q_pmt"][..., None])[..., 0]            # λ_k(p) at each hidden hit
        valid = p["q_valid"]
        nll = -torch.log(lam_at_target + S2_EPS)
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


def build_sparse_arm(arm: str, geometry, seed: int, *, encoder_config: EncoderConfig, scaling: FeatureScaling,
                     queries: str = "all", neg_ratio: float | None = None, mask_seed: int = lb.DEFAULT_SEED):
    if arm == "sparse_mae":
        return SparseMaeArm(geometry, encoder_config=encoder_config, scaling=scaling, seed=seed,
                            queries=queries, neg_ratio=neg_ratio)
    if arm == "sparse_time":
        return SparseTimeArm(geometry, encoder_config=encoder_config, scaling=scaling, seed=seed,
                             mask_seed=mask_seed)
    raise ValueError(f"unknown sparse arm {arm!r}")
