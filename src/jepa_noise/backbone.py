"""WP4a bridge backbone: fixed-scale, pre-final-norm token-grid encoder.

Subclasses (never edits) ``reconstruction_model.model.Transformer`` and keeps its
factorised time-then-channel block structure.  Two contractual changes from the
package ``forward`` are made, both required by runbook v4 §3.5.7 / A8:

1. ``normalise_input_sequence`` (the per-event z-score) is **replaced by fixed
   per-channel scaling**.  One scalar mean and one scalar standard deviation per
   channel are fitted once on disjoint calibration-noise records, stored as
   buffers, and never refitted (robustness shifts use the frozen values).  No
   per-event mean or standard deviation is computed anywhere in the forward
   pass.
2. The representation returned by every loss and probe is the token grid
   ``(B, N_patch, C, d_model)`` taken **after the spatial (channel) layers and
   before** ``final_proj_norm``.  ``final_proj_norm``, ``spatial_head`` and
   ``energy_head`` are unused here and frozen (``requires_grad=False``).

Masks for the bridge are A4's channel masks: context = channels 0-3, target =
channels 4-7, all patches, no gap.  The context encoder replaces the
non-context tokens with a learned mask embedding *before* attention; the target
encoder masks the context tokens.  Masking zeroes the raw samples of
non-visible tokens *and* replaces their patch-embedded tokens with the learned
mask embedding, so no raw masked sample can reach the visible outputs through
any path (tests/test_bridge.py checks this with gradients and finite
perturbations).  The same machinery supports patch-wise (gap) masks, which the
leakage fixture uses; the registered bridge split has an empty gap.

``parallel=True`` builds the registered ablation: at each level a time branch
and a channel branch run on the same tokens, their outputs are concatenated and
projected, instead of running the two factorisations serially.

Engineering choices inside the WP4a scope (each logged in the report):
- scalar per-channel scaling, not per-sample-position scaling;
- one learned mask embedding vector per channel, shared across patches;
- token masks built from (channel set x patch set) products, in channel-major
  order so target ordering matches the raw channel-major split;
- channel positional identity comes from the package's
  ``AbsolutePositionalEmbedding`` applied on the channel axis, exactly as the
  wrapped model does;
- predictor d_ff = 4*d_pred, 4 heads, output projected back to d_model;
- the parallel merge is one linear projection per level (Muon group);
- implicit torch.compile disabled for CPU checks (runbook v3 §3.1).
"""

from __future__ import annotations

import os

# Runbook v3 §3.1: "Disable implicit compilation for CPU checks where the
# repository supports it".  The no-touch Muon module decides at *import* time
# whether to wrap Newton-Schulz in torch.compile, so this must be set before
# reconstruction_model.model (which imports it) is imported.
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from reconstruction_model.model import (
    Transformer,
    TransformerBlock,
    TransformerConfig,
    patch_input_sequence,
)

__all__ = [
    "TokenGridEncoder",
    "TargetQueryPredictor",
    "PredictorConfig",
    "init_mup_linears",
]


def init_mup_linears(module: nn.Module) -> None:
    """Apply the wrapped package's muP linear initialisation to ``module``.

    The rule is the one in
    ``reconstruction_model.model.Transformer._init_weights`` (spectral muP:
    ``sigma = min(1, sqrt(d_out/d_in)) / sqrt(d_in)``).  It is repeated here
    because that package is no-touch and the helper is private.
    """
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            d_out, d_in = sub.weight.size(0), sub.weight.size(1)
            sigma = min(1.0, (d_out / d_in) ** 0.5) / (d_in ** 0.5)
            nn.init.normal_(sub.weight, mean=0.0, std=sigma)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)


@dataclass
class PredictorConfig:
    """Predictor numerics (v3 WP4: d_pred=64, two full-attention blocks)."""

    d_model: int = 128
    d_pred: int = 64
    n_layers: int = 2
    n_head: int = 4
    # v3 does not pin the predictor FFN width; frozen here at 4*d_pred.
    d_ff: Optional[int] = None
    norm_eps: float = 1e-6
    n_target: int = 32
    query_init_std: float = 0.02

    def resolved_d_ff(self) -> int:
        return 4 * self.d_pred if self.d_ff is None else int(self.d_ff)


class TargetQueryPredictor(nn.Module):
    """Reads context outputs plus learned target-positional queries only.

    Input: context token embeddings ``(B, n_context, d_model)``.  Output:
    predictions at the ``n_target`` target tokens, ``(B, n_target, d_model)``, in
    the encoder's channel-major target-token order.  The predictor never sees
    target or gap raw samples or target-view tokens; all cross-view information
    must arrive through the context outputs.
    """

    def __init__(self, config: PredictorConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_pred, bias=True)
        # One learned positional query per target token: the index encodes the
        # (channel, patch) position of that target token.  These are the
        # "target positional queries" of v3 WP4.
        self.queries = nn.Parameter(
            torch.empty(config.n_target, config.d_pred)
        )
        block_cfg = TransformerConfig(
            d_model=config.d_pred,
            d_ff=config.resolved_d_ff(),
            max_seq_len=max(1, config.n_target),
            patch_len=1,
            patch_stride=1,
            n_head=config.n_head,
            n_time_layers=0,
            n_channel_layers=0,
            rope_base=10000.0,
            norm_eps=config.norm_eps,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(block_cfg, use_rope=False)
                for _ in range(config.n_layers)
            ]
        )
        self.out_proj = nn.Linear(config.d_pred, config.d_model, bias=True)

    def forward(self, context_tokens: Tensor) -> Tensor:
        if context_tokens.ndim != 3:
            raise ValueError("context tokens must be (B, n_context, d_model)")
        batch = context_tokens.size(0)
        n_context = context_tokens.size(1)
        sequence = torch.cat(
            [
                self.in_proj(context_tokens),
                self.queries.unsqueeze(0).expand(batch, -1, -1),
            ],
            dim=1,
        )
        for block in self.blocks:
            sequence = block(sequence, None)
        return self.out_proj(sequence[:, n_context:, :])


class TokenGridEncoder(Transformer):
    """Fixed-scale factorised encoder returning the pre-final-norm token grid.

    Parameters
    ----------
    config:
        The wrapped package ``TransformerConfig`` (C=8, N=256, patch 32,
        d_model=128, 4 heads, depths (1,1) for the registered bridge; the data
        shape is passed separately through ``n_channels``/``n_samples``).
    channel_mean, channel_std:
        Fixed per-channel scaling, shape ``(C,)``, fitted once on calibration
        noise with :meth:`fit_scaling` / :meth:`from_calibration_noise`.
    n_channels, n_samples:
        Data-law dimensions (8 channels, 256 samples for the bridge).
    context_channels, target_channels:
        A4 channel split: ``(0,1,2,3) -> (4,5,6,7)`` for the bridge.
    context_patches, target_patches:
        Patch sets of the two views; default all patches (the channel split).
        Other registered raw splits are expressible as products; the bridge uses
        the defaults.  Tokens in neither view are the gap and are masked from
        both views.
    parallel:
        Build the parallel factorised-attention ablation.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        channel_mean,
        channel_std,
        n_channels: int = 8,
        n_samples: int = 256,
        context_channels: Sequence[int] = (0, 1, 2, 3),
        target_channels: Sequence[int] = (4, 5, 6, 7),
        context_patches: Optional[Sequence[int]] = None,
        target_patches: Optional[Sequence[int]] = None,
        parallel: bool = False,
        mask_embed_init: float = 0.02,
    ):
        super().__init__(config)
        if config.patch_len != config.patch_stride:
            raise NotImplementedError(
                "token-level raw masking requires patch_stride == patch_len"
            )
        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples)
        self.patch_len = int(config.patch_len)
        self.patch_stride = int(config.patch_stride)
        if self.n_samples <= 0 or self.n_samples % self.patch_len:
            raise ValueError("n_samples must be a positive multiple of patch_len")
        self.n_patch = self.n_samples // self.patch_len
        if self.n_patch > self.max_patches:
            raise ValueError("n_samples exceeds the wrapped model's max_seq_len")
        self.parallel = bool(parallel)
        self._mask_embed_init = float(mask_embed_init)

        def _channels(name: str, values: Sequence[int]) -> tuple:
            vals = tuple(int(v) for v in values)
            if not vals or len(set(vals)) != len(vals):
                raise ValueError(f"{name} must be a non-empty unique channel set")
            if min(vals) < 0 or max(vals) >= self.n_channels:
                raise ValueError(f"{name} outside 0..{self.n_channels - 1}")
            return vals

        self.context_channels = _channels("context_channels", context_channels)
        self.target_channels = _channels("target_channels", target_channels)
        # Channel sets may overlap (a time split uses all channels in both
        # views); the token sets below are the registered partition and must be
        # disjoint.
        partial = set(range(self.n_channels)) - (
            set(self.context_channels) | set(self.target_channels)
        )
        if partial:
            raise ValueError(
                "every channel must belong to the context or target view; "
                f"unassigned: {sorted(partial)}"
            )

        def _patches(name: str, values: Optional[Sequence[int]]) -> tuple:
            vals = (
                tuple(range(self.n_patch))
                if values is None
                else tuple(int(v) for v in values)
            )
            if not vals or len(set(vals)) != len(vals):
                raise ValueError(f"{name} must be a non-empty unique patch set")
            if min(vals) < 0 or max(vals) >= self.n_patch:
                raise ValueError(f"{name} outside 0..{self.n_patch - 1}")
            return vals

        self.context_patches = _patches("context_patches", context_patches)
        self.target_patches = _patches("target_patches", target_patches)

        ctx_tok = torch.zeros(self.n_patch, self.n_channels, dtype=torch.bool)
        tgt_tok = torch.zeros(self.n_patch, self.n_channels, dtype=torch.bool)
        for n in self.context_patches:
            ctx_tok[n, list(self.context_channels)] = True
        for n in self.target_patches:
            tgt_tok[n, list(self.target_channels)] = True
        if (ctx_tok & tgt_tok).any():
            raise ValueError("context and target token sets must be disjoint")
        gap_tok = ~(ctx_tok | tgt_tok)

        # Tokens replaced/zeroed when encoding each view.
        context_view_tok = tgt_tok | gap_tok
        target_view_tok = ctx_tok | gap_tok

        def _token_indices(mask: Tensor) -> Tensor:
            idx = []
            for c in range(self.n_channels):  # channel-major
                for n in range(self.n_patch):
                    if bool(mask[n, c]):
                        idx.append(n * self.n_channels + c)
            return torch.tensor(idx, dtype=torch.long)

        def _raw_indices(mask: Tensor) -> Tensor:
            idx = []
            for c in range(self.n_channels):  # channel-major, time ascending
                for n in range(self.n_patch):
                    if bool(mask[n, c]):
                        start = c * self.n_samples + n * self.patch_len
                        idx.extend(range(start, start + self.patch_len))
            return torch.tensor(idx, dtype=torch.long)

        def _raw_sample_mask(mask: Tensor) -> Tensor:
            out = torch.zeros(self.n_channels, self.n_samples, dtype=torch.bool)
            for c in range(self.n_channels):
                for n in range(self.n_patch):
                    if bool(mask[n, c]):
                        start = n * self.patch_len
                        out[c, start : start + self.patch_len] = True
            return out

        # Public, non-persistent masks/indices so tests can address raw samples.
        self.register_buffer("context_token_bool", ctx_tok, persistent=False)
        self.register_buffer("target_token_bool", tgt_tok, persistent=False)
        self.register_buffer("gap_token_bool", gap_tok, persistent=False)
        self.register_buffer("context_view_token_mask", context_view_tok, persistent=False)
        self.register_buffer("target_view_token_mask", target_view_tok, persistent=False)
        self.register_buffer(
            "context_view_raw_mask", _raw_sample_mask(context_view_tok), persistent=False
        )
        self.register_buffer(
            "target_view_raw_mask", _raw_sample_mask(target_view_tok), persistent=False
        )
        self.register_buffer("context_token_index", _token_indices(ctx_tok), persistent=False)
        self.register_buffer("target_token_index", _token_indices(tgt_tok), persistent=False)
        self.register_buffer("gap_token_index", _token_indices(gap_tok), persistent=False)
        self.register_buffer(
            "all_token_index", torch.arange(self.n_patch * self.n_channels), persistent=False
        )
        self.register_buffer("context_raw_index", _raw_indices(ctx_tok), persistent=False)
        self.register_buffer("target_raw_index", _raw_indices(tgt_tok), persistent=False)
        self.register_buffer("gap_raw_index", _raw_indices(gap_tok), persistent=False)

        self.n_context_tokens = int(ctx_tok.sum())
        self.n_target_tokens = int(tgt_tok.sum())
        self.n_gap_tokens = int(gap_tok.sum())

        self._set_scaling(channel_mean, channel_std)

        self.mask_embedding = nn.Parameter(
            torch.empty(self.n_channels, config.d_model)
        )
        # Safe default so a forward pass before initialize() is not garbage;
        # initialize(seed) redraws it.
        nn.init.normal_(self.mask_embedding, mean=0.0, std=self._mask_embed_init)

        # Parallel ablation: replace the serial time/channel stacks by parallel
        # branches plus one merge projection per level.
        if self.parallel:
            n_levels = max(config.n_time_layers, config.n_channel_layers, 1)
            self.temporal_layers = nn.ModuleList(
                [TransformerBlock(config, use_rope=True) for _ in range(n_levels)]
            )
            self.spatial_layers = nn.ModuleList(
                [TransformerBlock(config, use_rope=False) for _ in range(n_levels)]
            )
            self.merge_layers = nn.ModuleList(
                [
                    nn.Linear(2 * config.d_model, config.d_model)
                    for _ in range(n_levels)
                ]
            )
        else:
            self.merge_layers = nn.ModuleList()

        # The read-out is pre-final-norm; the pooling heads are unused.  Freeze
        # them so "trainable" means "used" and optimizer coverage is exact.
        for module in (self.final_proj_norm, self.spatial_head, self.energy_head):
            for param in module.parameters():
                param.requires_grad_(False)

    # ------------------------------------------------------------------ scaling
    @staticmethod
    def fit_scaling(records, min_std: float = 1e-6) -> tuple:
        """Fit fixed per-channel mean/std on ``(n_records, C, T)`` noise records.

        Called once, on disjoint calibration-noise records; the returned arrays
        are stored as buffers and never refitted by this package.
        """
        r = torch.as_tensor(records, dtype=torch.float32)
        if r.ndim != 3:
            raise ValueError("calibration records must be (n, C, T)")
        if not torch.isfinite(r).all():
            raise ValueError("calibration records must be finite")
        mean = r.mean(dim=(0, 2))
        std = r.std(dim=(0, 2), correction=1)
        if not torch.isfinite(std).all() or float(std.min()) <= 0:
            raise ValueError("calibration std must be finite and positive")
        return mean, std.clamp_min(float(min_std))

    @classmethod
    def from_calibration_noise(cls, config: TransformerConfig, records, **kwargs):
        mean, std = cls.fit_scaling(records)
        encoder = cls(config, channel_mean=mean, channel_std=std, **kwargs)
        encoder.initialize()
        return encoder

    def _set_scaling(self, channel_mean, channel_std) -> None:
        mean = torch.as_tensor(channel_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(channel_std, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.n_channels or std.numel() != self.n_channels:
            raise ValueError(
                f"fixed scaling must have one value per channel ({self.n_channels})"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("fixed scaling must be finite")
        if bool((std <= 0).any()):
            raise ValueError("fixed scaling std must be positive")
        self.register_buffer("channel_mean", mean, persistent=True)
        self.register_buffer("channel_std", std, persistent=True)

    # -------------------------------------------------------------------- init
    def initialize(self, seed: Optional[int] = None) -> None:
        """Initialise the wrapped backbone and the WP4a additions.

        ``AbsolutePositionalEmbedding`` draws its parameter in the package
        ``__init__`` (before any model-level seeding); it is redrawn here with
        the package's own N(0, 0.02) rule so that a seeded initialisation is
        complete and reproducible.
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        self.init_weights()  # muP linear init, zero residual outputs/heads
        nn.init.normal_(self.abs_pos_embd.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(
            self.mask_embedding, mean=0.0, std=self._mask_embed_init
        )

    # ------------------------------------------------------------------- encode
    def _forward_tokens(self, x: Tensor, view: str) -> Dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError("input must be (B, C, n_samples)")
        batch, channels, samples = x.shape
        if channels != self.n_channels or samples != self.n_samples:
            raise ValueError(
                f"expected (B, {self.n_channels}, {self.n_samples}), got "
                f"{tuple(x.shape)}"
            )
        if view == "context":
            token_mask = self.context_view_token_mask
            raw_mask = self.context_view_raw_mask
        elif view == "target":
            token_mask = self.target_view_token_mask
            raw_mask = self.target_view_raw_mask
        elif view == "unmasked":
            token_mask = None
            raw_mask = None
        else:
            raise ValueError("view must be 'context', 'target' or 'unmasked'")

        # Fixed per-channel scaling -- the only input normalisation.
        scaled = (x - self.channel_mean.view(1, -1, 1)) / self.channel_std.view(
            1, -1, 1
        )
        if raw_mask is not None:
            # Zero the raw samples of non-visible tokens: by construction the
            # gradient with respect to those raw samples is exactly zero.
            scaled = scaled.masked_fill(raw_mask.unsqueeze(0), 0.0)

        flat = patch_input_sequence(
            scaled.reshape(batch * channels, samples),
            self.patch_len,
            self.patch_stride,
        )
        tokens = self.patch_embedding(flat)  # (B*C, N, d)
        tokens = tokens.view(
            batch, channels, self.n_patch, self.config.d_model
        )
        if token_mask is not None:
            replacement = self.mask_embedding[:, None, :].expand(
                channels, self.n_patch, self.config.d_model
            )
            tokens = torch.where(
                token_mask.transpose(0, 1).unsqueeze(-1).unsqueeze(0),
                replacement.unsqueeze(0),
                tokens,
            )

        if self.parallel:
            grid, post_time = self._parallel_blocks(tokens)
        else:
            grid, post_time = self._serial_blocks(tokens)
        return {"grid": grid, "post_time": post_time}

    def _serial_blocks(self, tokens: Tensor) -> tuple:
        batch, channels, n_patch, d_model = tokens.shape
        x = tokens.reshape(batch * channels, n_patch, d_model)
        for layer in self.temporal_layers:
            x = layer(x, self._rope_cache)
        post_time = x.view(batch, channels, n_patch, d_model)
        x = post_time.transpose(1, 2).reshape(batch * n_patch, channels, d_model)
        x = self.abs_pos_embd(x)
        for layer in self.spatial_layers:
            x = layer(x, None)
        # Read-out point: after the spatial layers, BEFORE final_proj_norm.
        return x.view(batch, n_patch, channels, d_model), post_time

    def _parallel_blocks(self, tokens: Tensor) -> tuple:
        batch, channels, n_patch, d_model = tokens.shape
        x = tokens.reshape(batch * channels, n_patch, d_model)
        post_time = None
        for time_block, channel_block, merge in zip(
            self.temporal_layers, self.spatial_layers, self.merge_layers
        ):
            time_out = time_block(x, self._rope_cache)  # (B*C, N, d)
            channel_in = x.view(batch, channels, n_patch, d_model).transpose(
                1, 2
            ).reshape(batch * n_patch, channels, d_model)
            channel_in = self.abs_pos_embd(channel_in)
            channel_out = channel_block(channel_in, None)  # (B*N, C, d)
            time_grid = time_out.view(batch, channels, n_patch, d_model).transpose(
                1, 2
            )  # (B, N, C, d)
            channel_grid = channel_out.view(batch, n_patch, channels, d_model)
            merged = merge(torch.cat([time_grid, channel_grid], dim=-1))
            x = merged.transpose(1, 2).reshape(batch * channels, n_patch, d_model)
            post_time = time_out.view(batch, channels, n_patch, d_model)
        grid = x.view(batch, channels, n_patch, d_model).transpose(1, 2)
        return grid, post_time

    def select_tokens(self, grid: Tensor, which: str) -> Tensor:
        """Select context/target/gap/all tokens from ``(B, N, C, d)`` in order."""
        if which == "context":
            index = self.context_token_index
        elif which == "target":
            index = self.target_token_index
        elif which == "gap":
            index = self.gap_token_index
        elif which == "all":
            index = self.all_token_index
        else:
            raise ValueError("which must be context/target/gap/all")
        return grid.flatten(1, 2).index_select(1, index)

    def encode(self, x: Tensor, view: str = "context") -> Dict[str, Tensor]:
        """Encode one view.

        Returns ``grid`` (B, N, C, d; pre-final-norm read-out), ``post_time``
        (B, C, N, d; after the time block) and ``pooled`` (B, d; mean over the
        view's visible tokens).
        """
        out = self._forward_tokens(x, view)
        grid = out["grid"]
        if view == "context":
            pooled = self.select_tokens(grid, "context").mean(dim=1)
        elif view == "target":
            pooled = self.select_tokens(grid, "target").mean(dim=1)
        else:
            pooled = grid.mean(dim=(1, 2))
        return {"grid": grid, "post_time": out["post_time"], "pooled": pooled}

    def forward(self, x: Tensor) -> Tensor:
        """Context-view token grid ``(B, N_patch, C, d_model)``."""
        return self.encode(x, "context")["grid"]

    def encode_target(self, x: Tensor) -> Tensor:
        """Target-view token grid (context tokens masked)."""
        return self.encode(x, "target")["grid"]

    def encode_unmasked(self, x: Tensor) -> Dict[str, Tensor]:
        """Unmasked encoding: used by probes and by the anchor overlay."""
        return self.encode(x, "unmasked")

    def encode_after_time(self, x: Tensor, view: str = "unmasked") -> Tensor:
        """Post-time-block per-channel representation ``(B, C, N_patch, d_model)``.

        This is the representation "after the time block" used by the
        amplitude-preservation probe; it precedes the channel layers.
        """
        return self.encode(x, view)["post_time"]
