"""WP4a Tier-1 bridge objectives (runbook v4 §3.5.7; v3 WP4 stays in force).

Arms
----
``sigreg_jepa``
    v3 arm 4: shared weights, no stop-gradient.  Two masked views of the same
    event are encoded by the same encoder; the predictor reads context outputs
    plus target positional queries and predicts the target-view token
    embeddings with an L2 loss.  SIGReg is applied to the context pooled
    embeddings and to the target token embeddings (runbook v3 WP4), averaged.
``sigreg_jepa_anchored``
    v3 arm 6: ``sigreg_jepa`` plus ``beta`` times the independent-noise overlay
    squared embedding change divided by ``max(trace(Cov(z)), 1e-6)``.  The
    overlay uses *unmasked* pooled embeddings and independent calibration-noise
    draws; gradients flow through both the numerator and the trace denominator.
    ``beta`` is not selected here: the registered grid is {0.1, 1, 10} and the
    choice happens on the independent Tier-1 development cohort (v3 §1.3).
``mae_chi2``
    v3 arm 2 for the bridge: context-masked reconstruction with the exact target
    marginal ``N_actual`` weighting, ``0.5 * r^T N_tt^{-1} r``.  The log
    determinant term is omitted (it is constant in the parameters for a fixed
    covariance).  This arm **uses oracle noise information** and is labelled so
    in its return value.
``sigreg_jepa_parallel``
    The registered ablation: ``sigreg_jepa`` with the parallel factorised
    attention of :class:`jepa_noise.backbone.TokenGridEncoder` (time branch ||
    channel branch, concatenated then projected).

All arms share one backbone class, one data law, one schedule, and explicit
optimizer groups in which every trainable non-EMA parameter appears exactly
once.  Optimizer numerics follow the repository's
``reconstruction_model.model.Transformer.configure_optimisers`` conventions
(AdamW for embeddings/heads/norms/queries, Muon for the 2-D block matrices),
frozen in ``configs/bridge_v1.json``.  ``final_proj_norm``, ``spatial_head`` and
``energy_head`` are unused by the read-out and are frozen.

The bridge data law is A4 at g=0.02, nominal SNR, channel split.  This module
takes synthetic arrays or caller-supplied calibration products (fixed scaling,
target noise covariance, noise bank); it does not import the data generator, so
WP4a is testable while the WP2-v4 A4 base-rho integration is pending.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Runbook v3 §3.1: no implicit compilation for CPU checks.  This must run
# before reconstruction_model.model (which imports the no-touch Muon module and
# decides there whether to wrap Newton-Schulz in torch.compile).
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW

from reconstruction_model.model import Muon, TransformerConfig

from .backbone import (
    PredictorConfig,
    TargetQueryPredictor,
    TokenGridEncoder,
    init_mup_linears,
)
from .sigreg import SIGReg, sigreg_config_from_dict

__all__ = [
    "CONFIG_PATH",
    "load_config",
    "JEPA_ARMS",
    "BRIDGE_ARMS",
    "ABLATION_ARM",
    "ALL_ARMS",
    "BridgeModel",
    "build_model",
    "build_optimizers",
    "check_optimizer_coverage",
]

JEPA_ARMS = ("sigreg_jepa", "sigreg_jepa_anchored")
BRIDGE_ARMS = ("sigreg_jepa", "sigreg_jepa_anchored", "mae_chi2")
ABLATION_ARM = "sigreg_jepa_parallel"
ALL_ARMS = BRIDGE_ARMS + (ABLATION_ARM,)

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "bridge_v1.json"


def load_config(path: Optional[os.PathLike | str] = None) -> dict:
    """Load the frozen bridge numeric config (``configs/bridge_v1.json``)."""
    target = CONFIG_PATH if path is None else Path(path)
    with open(target, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "backbone" not in config or "data" not in config:
        raise ValueError("bridge config must declare 'data' and 'backbone'")
    return config


def transformer_config(config: dict) -> TransformerConfig:
    """Build the wrapped package config from the bridge ``backbone`` block."""
    b = config["backbone"]
    d = config["data"]
    return TransformerConfig(
        d_model=int(b["d_model"]),
        d_ff=int(b["d_ff"]),
        max_seq_len=int(b["max_seq_len"]),
        patch_len=int(d["patch_len"]),
        patch_stride=int(d["patch_stride"]),
        n_head=int(b["n_head"]),
        n_time_layers=int(b["n_time_layers"]),
        n_channel_layers=int(b["n_channel_layers"]),
        rope_base=float(b.get("rope_base", 10000.0)),
        norm_eps=float(b.get("norm_eps", 1e-6)),
    )


class BridgeModel(nn.Module):
    """One WP4a bridge arm on the shared TokenGridEncoder backbone."""

    def __init__(
        self,
        config: dict,
        arm: str,
        *,
        channel_mean,
        channel_std,
        target_noise_covariance=None,
        noise_bank=None,
        beta: Optional[float] = None,
        seed: int = 0,
    ):
        super().__init__()
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown bridge arm {arm!r}; expected {ALL_ARMS}")
        self.config = config
        self.arm = arm
        self.seed = int(seed)
        data = config["data"]
        backbone = config["backbone"]
        self.n_channels = int(data["channels"])
        self.n_samples = int(data["n_samples"])
        self.model_config = transformer_config(config)

        self.encoder = TokenGridEncoder(
            self.model_config,
            channel_mean=channel_mean,
            channel_std=channel_std,
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            context_channels=data["context_channels"],
            target_channels=data["target_channels"],
            context_patches=data.get("context_patches"),
            target_patches=data.get("target_patches"),
            parallel=(arm == ABLATION_ARM),
        )
        self.parallel = arm == ABLATION_ARM

        self.predictor: Optional[TargetQueryPredictor] = None
        self.sigreg: Optional[SIGReg] = None
        self.recon_head: Optional[nn.Linear] = None
        if arm in JEPA_ARMS or arm == ABLATION_ARM:
            predictor_cfg = config["predictor"]
            self.predictor = TargetQueryPredictor(
                PredictorConfig(
                    d_model=int(backbone["d_model"]),
                    d_pred=int(predictor_cfg["d_pred"]),
                    n_layers=int(predictor_cfg["n_layers"]),
                    n_head=int(predictor_cfg["n_head"]),
                    d_ff=predictor_cfg.get("d_ff"),
                    norm_eps=float(backbone.get("norm_eps", 1e-6)),
                    n_target=self.encoder.n_target_tokens,
                )
            )
            self.sigreg = SIGReg(sigreg_config_from_dict(config["sigreg"]))
            self.sigreg_lambda = float(
                sigreg_config_from_dict(config["sigreg"]).coefficient_lambda
            )

        self.beta: Optional[float] = None
        if arm == "mae_chi2":
            if target_noise_covariance is None:
                raise ValueError(
                    "mae_chi2 requires the target marginal of N_actual "
                    "(oracle noise information); pass target_noise_covariance"
                )
            self.register_buffer(
                "target_precision_chol",
                self._prepare_target_covariance(target_noise_covariance),
                persistent=True,
            )
            self.recon_head = nn.Linear(
                int(backbone["d_model"]), self.encoder.patch_len
            )
        if arm == "sigreg_jepa_anchored":
            if noise_bank is None:
                raise ValueError(
                    "sigreg_jepa_anchored requires independent calibration-noise "
                    "draws; pass noise_bank of shape (n_records, C, T)"
                )
            if beta is None:
                raise ValueError(
                    "sigreg_jepa_anchored requires beta; the registered grid is "
                    "{0.1, 1, 10} and beta is selected on the independent "
                    "development cohort (v3 §1.3), not here"
                )
            if float(beta) < 0:
                raise ValueError("beta must be non-negative")
            self.beta = float(beta)
            bank = torch.as_tensor(noise_bank, dtype=torch.float32)
            if bank.ndim != 3 or bank.shape[1:] != (
                self.n_channels,
                self.n_samples,
            ):
                raise ValueError(
                    "noise_bank must be (n_records, C, T) = "
                    f"(n, {self.n_channels}, {self.n_samples})"
                )
            if not torch.isfinite(bank).all():
                raise ValueError("noise_bank must be finite")
            self.register_buffer("overlay_bank", bank, persistent=False)
            self._overlay_generator = torch.Generator(
                device=bank.device
            ).manual_seed(self.seed + 100003)
        self.initialize(self.seed)

    # ------------------------------------------------------------- construction
    def _prepare_target_covariance(self, full_covariance) -> Tensor:
        """Slice and factorise the target marginal of ``N_actual``.

        ``full_covariance`` is the full raw ``(C*T, C*T)`` covariance.  The
        target marginal is taken at ``encoder.target_raw_index`` (channel-major,
        the registered raw ordering) -- never the gap -- and factorised with a
        lower Cholesky, so no jitter is added and a non-SPD marginal is
        rejected.
        """
        cov = torch.as_tensor(full_covariance, dtype=torch.float64)
        expected = self.n_channels * self.n_samples
        if cov.shape != (expected, expected):
            raise ValueError(
                f"target_noise_covariance must be the full raw ({expected}, "
                f"{expected}) N_actual; got {tuple(cov.shape)}"
            )
        if not torch.isfinite(cov).all():
            raise ValueError("target_noise_covariance must be finite")
        if not torch.allclose(cov, cov.transpose(0, 1), rtol=1e-6, atol=1e-8):
            raise ValueError("target_noise_covariance must be symmetric")
        index = self.encoder.target_raw_index
        marginal = cov.index_select(0, index).index_select(1, index)
        try:
            chol = torch.linalg.cholesky(marginal)
        except RuntimeError as exc:  # torch raises on non-PD input
            raise ValueError(
                "target marginal of N_actual is not positive definite"
            ) from exc
        return chol.to(torch.float32)

    def initialize(self, seed: Optional[int] = None) -> None:
        """Deterministic initialisation of the whole arm for ``seed``."""
        if seed is not None:
            self.seed = int(seed)
        torch.manual_seed(self.seed)
        self.encoder.initialize()
        if self.predictor is not None:
            init_mup_linears(self.predictor)
            nn.init.normal_(
                self.predictor.queries,
                mean=0.0,
                std=self.predictor.config.query_init_std,
            )
        if self.recon_head is not None:
            init_mup_linears(self.recon_head)

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> Dict[str, object]:
        """Compute the arm loss on a raw batch ``(B, C, T)``."""
        if x.ndim != 3:
            raise ValueError("batch must be (B, C, T)")
        if self.arm == "mae_chi2":
            return self._forward_mae_chi2(x)
        if self.arm == "sigreg_jepa_anchored":
            return self._forward_anchored(x)
        return self._forward_sigreg(x)

    def _view_tokens(self, x: Tensor):
        context = self.encoder.encode(x, "context")
        target = self.encoder.encode(x, "target")
        context_tokens = self.encoder.select_tokens(context["grid"], "context")
        target_tokens = self.encoder.select_tokens(target["grid"], "target")
        return context, target, context_tokens, target_tokens

    def _forward_sigreg(self, x: Tensor) -> Dict[str, object]:
        context, target, context_tokens, target_tokens = self._view_tokens(x)
        prediction = self.predictor(context_tokens)
        pred_loss = F.mse_loss(prediction, target_tokens)
        reg = self.sigreg(context["pooled"], target_tokens)
        loss = pred_loss + self.sigreg_lambda * reg["value"]
        metrics = {
            "loss": float(loss.detach()),
            "prediction": float(pred_loss.detach()),
            "sigreg": float(reg["value"].detach()),
            "sigreg_context": float(reg["context"].detach()),
            "sigreg_target": float(reg["target"].detach()),
        }
        return {
            "loss": loss,
            "metrics": metrics,
            "prediction": prediction,
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
        }

    @staticmethod
    def _trace_cov(z: Tensor) -> Tensor:
        """``trace(Cov(z))`` over the batch (unbiased B-1 divisor), differentiable."""
        if z.size(0) < 2:
            raise ValueError("trace(Cov(z)) needs at least two embeddings")
        centered = z - z.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / (z.size(0) - 1)
        return covariance.diagonal().sum()

    def _sample_overlay(self, batch_size: int) -> Tensor:
        index = torch.randint(
            0,
            self.overlay_bank.size(0),
            (int(batch_size),),
            generator=self._overlay_generator,
        )
        return self.overlay_bank.index_select(0, index)

    def anchor_penalty(self, z: Tensor, z_tilde: Tensor) -> Tensor:
        """``beta * mean_b ||z_tilde - z||^2 / max(trace(Cov(z)), 1e-6)``.

        Exposed for tests: the trace denominator keeps its gradient (no detach).
        """
        numerator = (z_tilde - z).pow(2).sum(dim=-1).mean()
        denominator = torch.clamp(self._trace_cov(z), min=1e-6)
        return self.beta * numerator / denominator

    def _forward_anchored(self, x: Tensor) -> Dict[str, object]:
        out = self._forward_sigreg(x)
        z = self.encoder.encode_unmasked(x)["pooled"]
        overlay = self._sample_overlay(x.size(0))
        z_tilde = self.encoder.encode_unmasked(x + overlay)["pooled"]
        penalty = self.anchor_penalty(z, z_tilde)
        loss = out["loss"] + penalty
        metrics = dict(out["metrics"])
        metrics.update(
            {
                "loss": float(loss.detach()),
                "anchor": float(penalty.detach()),
                "anchor_beta": self.beta,
                "anchor_trace": float(self._trace_cov(z).detach()),
            }
        )
        result = dict(out)
        result["loss"] = loss
        result["metrics"] = metrics
        result["anchor"] = penalty
        return result

    def _forward_mae_chi2(self, x: Tensor) -> Dict[str, object]:
        context = self.encoder.encode(x, "context")
        tokens = self.encoder.select_tokens(context["grid"], "target")
        patch_prediction = self.recon_head(tokens)  # (B, n_target, patch_len)
        prediction = patch_prediction.reshape(x.size(0), -1)
        # Channel-major token order then samples inside a patch is exactly the
        # registered raw target ordering (encoder.target_raw_index).
        label = x.flatten(1).index_select(1, self.encoder.target_raw_index)
        residual = prediction - label
        whitened = torch.linalg.solve_triangular(
            self.target_precision_chol,
            residual.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        # 0.5 * r^T N_tt^{-1} r; the log-determinant is parameter-free and
        # omitted.
        chi2 = 0.5 * whitened.pow(2).sum(dim=1).mean()
        metrics = {
            "loss": float(chi2.detach()),
            "chi2": float(chi2.detach()),
            "uses_oracle_noise_information": True,
        }
        return {
            "loss": chi2,
            "metrics": metrics,
            "prediction": prediction,
            "label": label,
        }


def build_model(
    config: dict,
    arm: str,
    *,
    channel_mean,
    channel_std,
    target_noise_covariance=None,
    noise_bank=None,
    beta: Optional[float] = None,
    seed: int = 0,
) -> BridgeModel:
    """Build one bridge arm; see :class:`BridgeModel` for the required inputs."""
    return BridgeModel(
        config,
        arm,
        channel_mean=channel_mean,
        channel_std=channel_std,
        target_noise_covariance=target_noise_covariance,
        noise_bank=noise_bank,
        beta=beta,
        seed=seed,
    )


def _block_modules(model: BridgeModel) -> List[nn.Module]:
    modules: List[nn.Module] = list(model.encoder.temporal_layers)
    modules += list(model.encoder.spatial_layers)
    if model.encoder.parallel:
        modules += list(model.encoder.merge_layers)
    if model.predictor is not None:
        modules += list(model.predictor.blocks)
    return modules


def build_optimizers(model: BridgeModel, config: Optional[dict] = None) -> list:
    """Explicit AdamW + Muon groups; every trainable parameter exactly once."""
    config = model.config if config is None else config
    optimizer_cfg = config["optimizer"]
    adamw_cfg = optimizer_cfg["adamw"]
    muon_cfg = optimizer_cfg["muon"]

    muon_params: List[nn.Parameter] = []
    aux_params: List[nn.Parameter] = []
    for module in _block_modules(model):
        for param in module.parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                aux_params.append(param)

    encoder = model.encoder
    adamw_groups = [
        dict(params=list(encoder.patch_embedding.parameters())),
        dict(params=list(encoder.abs_pos_embd.parameters())),
        dict(params=[encoder.mask_embedding]),
    ]
    if aux_params:
        adamw_groups.append(dict(params=aux_params))
    if model.predictor is not None:
        adamw_groups.append(dict(params=list(model.predictor.in_proj.parameters())))
        adamw_groups.append(dict(params=[model.predictor.queries]))
        adamw_groups.append(dict(params=list(model.predictor.out_proj.parameters())))
    if model.recon_head is not None:
        adamw_groups.append(dict(params=list(model.recon_head.parameters())))

    adamw = AdamW(
        adamw_groups,
        lr=float(adamw_cfg["lr"]),
        betas=tuple(float(b) for b in adamw_cfg["betas"]),
        weight_decay=float(adamw_cfg["weight_decay"]),
        fused=bool(adamw_cfg.get("fused", False)),
    )
    muon = Muon(
        muon_params,
        lr=float(muon_cfg["lr"]),
        momentum=float(muon_cfg["momentum"]),
        nesterov=bool(muon_cfg["nesterov"]),
        ns_steps=int(muon_cfg["ns_steps"]),
    )
    optimizers = [adamw, muon]
    check_optimizer_coverage(model, optimizers)
    return optimizers


def check_optimizer_coverage(model: nn.Module, optimizers: Sequence) -> Dict[str, int]:
    """Raise unless every trainable parameter occurs in exactly one group."""
    name_by_id = {id(p): name for name, p in model.named_parameters()}
    trainable = {id(p) for _, p in model.named_parameters() if p.requires_grad}
    seen: Dict[int, str] = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for param in group["params"]:
                key = id(param)
                if key in seen:
                    raise RuntimeError(
                        "parameter appears in more than one optimizer group: "
                        f"{seen[key]}"
                    )
                seen[key] = name_by_id.get(key, "<not a named parameter>")
    missing = [name_by_id[key] for key in trainable if key not in seen]
    extra = [name for key, name in seen.items() if key not in trainable]
    if missing or extra:
        raise RuntimeError(
            f"optimizer coverage broken: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    return {
        "n_trainable": len(trainable),
        "n_optimizer": len(seen),
        "n_adamw": sum(len(p["params"]) for p in optimizers[0].param_groups),
        "n_muon": sum(len(p["params"]) for p in optimizers[1].param_groups),
    }
