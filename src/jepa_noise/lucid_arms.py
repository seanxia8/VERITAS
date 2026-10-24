"""LUCiD WP5 arm objectives, anchor perturbation and probe forms (§3.5.6).

Binding: runbook v4 §3.5.6, §3.5.5 (encoder/read-out/masking), §3.5.3 (nominal
twin, noise-only records, gains before threshold), §3.5.4 (views), §3.5.8 (seed
floor) and §3.5.9 (R6-R8, R7b, R11, R11b, R12).  Objective/penalty and probe
layer only: no cohorts, no checkpoints, no trainer.

The seven §3.5.6 arms (``ARMS``) plus the secondary ``sigreg_jepa_anchored_dark``
variant; per-arm definitions are in ``docs/jepa/WP5_ARMS.md``.  ``mae_chi2`` uses
the registered covariance below and is labelled oracle; ``ijepa_ema`` uses a
cosine EMA target (0.996 -> 1); the SIGReg arms use :mod:`jepa_noise.sigreg`.

``mae_chi2`` covariance: for the hidden PMTs of module ``m``, ``r = q - mu`` and::

    C_m = diag(mu_p (1 + sigma_spe^2) + v_dark + s_p^2 mu_p^2) + s^2 mu_m mu_m^T
    loss = 0.5 sum_m (r^T C_m^-1 r + ln det C_m)
    C^-1 = D^-1 - s^2 D^-1 u (1 + s^2 u^T D^-1 u)^-1 u^T D^-1
    ln det C = sum ln D + log1p(s^2 u^T D^-1 u)

with ``D`` the diagonal part, ``u = mu_m``, ``s``/``s_p`` the cell's true spreads
and ``sigma_spe``/``v_dark`` from the reference generator config (target-marginal
second-moment approximation; threshold censoring ignored).  ``v_dark`` is the
R-6 dark-charge variance per window, ``lambda_dark (1 + sigma_spe^2)``
(1000 Hz x 1000 ns x 1.09 = 0.00109 PE^2), which is positive, so ``C_m`` is
positive definite even for a hidden unhit PMT with ``mu = 0`` and no variance
floor is needed.  ``variance_floor_pe2`` stays 0 in the frozen config; a floor
would be required only if a (non-registered) config set ``v_dark = 0``.

Anchor penalty: per event one perturbed copy ``x~`` with (1) an independent
dark-noise-only window overlaid (union of hits; charges add where a PMT is hit in
both; earlier time kept) and (2) every module's charges multiplied by an
independent ``N(1, 0.01^2)`` factor.  Penalty = mean over modules of
``||z_m(x~) - z_m(x)||^2`` over the trace of the module-token covariance, ``z_m``
the module-pooled **unmasked** read-out token (Tier-1 pattern).  The trace keeps
its gradient; the rescale is applied in the log-charge domain so a unit factor is
bit-exact and the penalty is exactly zero for ``x~ == x``.

Probes: four forms -- ``per_event``, ``run_mean`` (primary), ``propagated``
(context-encoder outputs at masked positions with the training mask, run-mean
over events in which the position is masked) and ``raw_input`` (ceiling, raw
token features, no encoder).  Ridge follows v3 §1.3 via
:mod:`jepa_noise.probes`; targets are vertex (x, y, z), log yield, ``ln gain_m``,
``ln gain_p``, the F2/F2' variable ``v`` and age ``r/R``.  The random-init floor
is :func:`random_init_floor`.

Optimiser numerics are **pending_review**; only the AdamW/Muon *groups* are exposed.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import probes as ridge
from .lucid_bombs import (LAMBDA0_M, ORACLE_ARM, V_DARK_READOUT, DetectorState,
                          read_sparse_hits, state_for_run)
from .point_encoder import (HIT_INDEX, LOG_CHARGE_INDEX, N_FEATURES, TIME_INDEX,
                            EncoderConfig, FeatureScaling, LucidBatch, LucidPointJepa)
from .sigreg import SIGReg, SIGRegConfig

__all__ = [
    "CONFIG_PATH", "ARMS", "ALL_ARMS", "ANCHOR_ARMS", "BETA_ARMS", "SIGREG_ARMS",
    "PROBE_KINDS", "PROBE_FORMS", "PRIMARY_PROBE_FORM", "ArmSettings", "load_config",
    "LucidArm", "build_arm", "DarkBank", "dark_bank_from_store", "chi2_module_terms",
    "chi2_dense_module", "simulate_module_readout", "charge_matrix", "RunProbeData",
    "ProbeDataset", "probe_data", "truth_from_state", "build_probe_dataset", "fit_probe",
    "probe_predictions", "probe_r2", "random_init_floor", "squared_correlation",
    "probe_rows", "ProbeAccumulator", "StreamedRidgeFit", "probe_r2_streamed",
]

#: The seven §3.5.6 arms, in table order.
ARMS = ("mae_mse", "mae_chi2", "ijepa_ema", "sigreg_jepa", "sigreg_jepa_anchored",
        "anchor_only", "oracle_nominal_twin")
DARK_VARIANT = "sigreg_jepa_anchored_dark"
ALL_ARMS = ARMS + (DARK_VARIANT,)
ANCHOR_ARMS = ("sigreg_jepa_anchored", DARK_VARIANT)
#: Arms whose beta is chosen on the §3.5.6 development cohort.
BETA_ARMS = ANCHOR_ARMS + ("anchor_only",)
SIGREG_ARMS = ("sigreg_jepa", "anchor_only") + ANCHOR_ARMS + (ORACLE_ARM,)

PROBE_KINDS = ("vertex", "log_yield", "module_gain", "pmt_gain", "f2_variable", "age")
PROBE_FORMS = ("per_event", "run_mean", "propagated", "raw_input")
#: §3.5.6: run-mean is the primary form for every registered criterion.
PRIMARY_PROBE_FORM = "run_mean"
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "lucid_arms_v0.json"
_TINY = 1e-6


# --------------------------------------------------------------------- config
def load_config(path: Path | str | None = None) -> dict:
    """Load ``configs/lucid_arms_v0.json`` and check its required blocks."""
    target = CONFIG_PATH if path is None else Path(path)
    config = json.loads(Path(target).read_text())
    for block in ("arms", "beta", "ema", "sigreg", "mae_chi2", "anchor", "cell", "probes"):
        if block not in config:
            raise ValueError(f"{target}: missing the {block!r} block")
    if tuple(config["arms"]) != ARMS:
        raise ValueError(f"{target}: arms must be the seven §3.5.6 arms {ARMS}")
    if config["cell"]["s"] <= 0 or config["cell"]["s_p"] < 0:
        raise ValueError(f"{target}: cell spreads must be positive (s) / non-negative (s_p)")
    if config["training"]["optimizer"] != "pending_review":
        raise ValueError(f"{target}: the optimizer must stay pending_review")
    return config


@dataclass(frozen=True)
class ArmSettings:
    """Typed view of the config: every number the arms read, no optimiser numerics."""

    beta_grid: tuple
    tau_start: float
    tau_end: float
    total_steps: int
    sigreg: SIGRegConfig
    coefficient_lambda: float
    sigma_spe: float
    v_dark: float
    variance_floor_pe2: float
    time_sigma_ns: float
    rescale_sd: float
    s: float; s_p: float

    @classmethod
    def from_config(cls, config: dict | None = None, *, cell=None, beta=None) -> "ArmSettings":
        config = load_config() if config is None else config
        ema, sig, chi2 = config["ema"], config["sigreg"], config["mae_chi2"]
        anchor = config["anchor"]
        s, s_p = ((float(config["cell"]["s"]), float(config["cell"]["s_p"])) if cell is None
                  else (float(cell[0]), float(cell[1])))
        if s <= 0 or s_p < 0:
            raise ValueError("cell spreads: s > 0 and s_p >= 0")
        grid = tuple(float(x) for x in config["beta"]["grid"])
        if beta is not None and float(beta) not in grid:
            raise ValueError(f"beta {beta!r} is not in the registered grid {grid}")
        if not 0.0 <= ema["tau_start"] < 1.0 or float(ema["tau_end"]) != 1.0:
            raise ValueError("EMA tau must run from tau_start < 1 to exactly 1")
        if int(ema["total_steps"]) != 10_000:
            raise ValueError("EMA schedule length must be the §3.5.5 10 000 steps")
        settings = cls(
            beta_grid=grid, tau_start=float(ema["tau_start"]), tau_end=float(ema["tau_end"]),
            total_steps=int(ema["total_steps"]),
            sigreg=SIGRegConfig(num_slices=int(sig["num_slices"]), t_max=float(sig["t_max"]),
                                n_points=int(sig["n_points"]),
                                coefficient_lambda=float(sig["coefficient_lambda"])),
            coefficient_lambda=float(sig["coefficient_lambda"]),
            sigma_spe=float(chi2["sigma_spe"]), v_dark=float(chi2["v_dark"]),
            variance_floor_pe2=float(chi2["variance_floor_pe2"]),
            time_sigma_ns=float(chi2["time_sigma_ns"]), rescale_sd=float(anchor["rescale_sd"]),
            s=s, s_p=s_p)
        if (settings.sigma_spe <= 0 or settings.v_dark < 0 or settings.time_sigma_ns <= 0
                or settings.variance_floor_pe2 < 0 or settings.rescale_sd <= 0):
            raise ValueError("mae_chi2 noise settings and rescale SD must be non-negative/positive")
        # readout_pmt_r6 (author 2026-10-18): the dark-charge variance is the R-6 value
        # lambda (1 + sigma_spe^2) = 0.001 x 1.09 = 0.00109 PE^2, derived from
        # configs/lucid_pmt/reference.json and never typed in; the variance floor stays 0
        # because v_dark > 0 makes C_m positive definite even at mu = 0.
        if float(settings.v_dark) != float(V_DARK_READOUT):
            raise ValueError(f"mae_chi2.v_dark {settings.v_dark!r} != the R-6 dark-charge "
                             f"variance {V_DARK_READOUT!r} derived from the reference manifest")
        if settings.variance_floor_pe2 != 0.0:
            raise ValueError("the variance floor stays 0: v_dark > 0 keeps C_m positive definite")
        if config["probes"]["primary_form"] != PRIMARY_PROBE_FORM:
            raise ValueError("run_mean must stay the primary probe form")
        return settings

    def ema_tau(self, step: int) -> float:
        """Cosine tau 0.996 -> 1 over the registered 10 000 steps."""
        fraction = min(max(float(step), 0.0), float(self.total_steps)) / float(self.total_steps)
        return self.tau_end - (self.tau_end - self.tau_start) * 0.5 * (
            1.0 + math.cos(math.pi * fraction))


# --------------------------------------------------------------- noise helpers
def charge_matrix(features: Tensor) -> Tensor:
    """Recorded charge in PE from token features: ``exp(log q)`` for hit PMTs, 0 else."""
    return torch.exp(features[..., LOG_CHARGE_INDEX]) * (features[..., HIT_INDEX] > 0.5)


@dataclass(frozen=True)
class DarkBank:
    """Dark-noise-only windows: per-record charge (PE), time (100 ns) and hit mask."""

    charge_pe: np.ndarray   # (R, N) float32
    time_100ns: np.ndarray  # (R, N) float32
    hit: np.ndarray         # (R, N) bool

    def __post_init__(self):
        arrays = (np.asarray(self.charge_pe, dtype=np.float32),
                  np.asarray(self.time_100ns, dtype=np.float32),
                  np.asarray(self.hit, dtype=bool))
        if arrays[0].ndim != 2 or arrays[0].shape != arrays[1].shape != arrays[2].shape:
            raise ValueError("dark bank arrays must share one (records, n_pmt) shape")
        if not np.isfinite(arrays[0]).all() or not np.isfinite(arrays[1]).all():
            raise ValueError("dark bank charge/time must be finite")
        if np.any(arrays[0] < 0) or np.any(arrays[2] != (arrays[0] > 0)):
            raise ValueError("dark bank: the hit mask must match the positive charges")
        for name, array in zip(("charge_pe", "time_100ns", "hit"), arrays):
            object.__setattr__(self, name, np.ascontiguousarray(array))

    @classmethod
    def empty(cls, n_pmt: int, n_records: int = 1) -> "DarkBank":
        """All-zero records: overlaying one is the identity (tests)."""
        return cls(np.zeros((n_records, n_pmt), np.float32),
                   np.zeros((n_records, n_pmt), np.float32),
                   np.zeros((n_records, n_pmt), bool))


def dark_bank_from_store(store, geometry, *, n_records: int | None = None) -> DarkBank:
    """One dark-noise-only record per stored (no-source) event."""
    store = read_sparse_hits(store) if isinstance(store, (str, Path)) else store
    ids = np.arange(store.event_ids.size) if n_records is None else np.arange(int(n_records))
    if ids.size == 0 or ids.max() >= store.event_ids.size:
        raise ValueError("dark bank: no records, or n_records exceeds the store")
    charge = np.zeros((ids.size, geometry.n_pmt), np.float32)
    time_100ns = np.zeros_like(charge)
    hit = np.zeros((ids.size, geometry.n_pmt), bool)
    for row, index in enumerate(ids):
        span = slice(int(store.offsets[index]), int(store.offsets[index + 1]))
        pmt = np.asarray(store.pmt_id[span], dtype=np.int64)
        charge[row, pmt] = np.asarray(store.charge_pe[span], np.float32)
        time_100ns[row, pmt] = np.asarray(store.time_ns[span], np.float32) / 100.0
        hit[row, pmt] = True
    return DarkBank(charge, time_100ns, hit)


# ------------------------------------------------------------------ mae_chi2
def chi2_module_terms(q: Tensor, mu: Tensor, module: Tensor, slot: Tensor, *,
                      n_modules: int, pmts_per_module: int, s: float, s_p: float,
                      sigma_spe: float, v_dark: float,
                      floor: float = _TINY) -> Tensor:
    """Per-module ``0.5 (r^T C_m^-1 r + ln det C_m)`` by Woodbury; shape ``(B, M)``.

    ``q``/``mu`` are ``(B, k)`` at the hidden PMTs; ``module``/``slot`` are their
    ``(k,)`` or ``(B, k)`` positions.  Non-hidden slots are neutral (``D = 1``,
    ``u = r = 0``) and contribute exactly zero.
    """
    if q.ndim != 2 or q.shape != mu.shape or module.shape != slot.shape \
            or module.shape not in ((q.size(1),), q.shape):
        raise ValueError("chi2_module_terms needs (B, k) q/mu and (k,) or (B, k) module/slot")
    if s <= 0 or s_p < 0 or sigma_spe < 0 or v_dark < 0 or floor < 0:
        raise ValueError(
            "chi2 spreads must be s > 0, s_p/sigma_spe/v_dark >= 0, floor >= 0"
        )
    # v_dark = 0.00109 (R-6) with floor = 0 keeps C_m positive definite, even at mu = 0.
    batch, k = q.shape
    module, slot = module.to(torch.long), slot.to(torch.long)
    module = module.expand(batch, k) if module.ndim == 1 else module
    slot = slot.expand(batch, k) if slot.ndim == 1 else slot
    if int(module.min()) < 0 or int(module.max()) >= n_modules \
            or int(slot.min()) < 0 or int(slot.max()) >= pmts_per_module:
        raise ValueError("module/slot index out of range")

    def scatter(values: Tensor, fill: float) -> Tensor:
        out = values.new_full((batch, n_modules * pmts_per_module), float(fill))
        out.scatter_(1, module * pmts_per_module + slot, values)
        return out.view(batch, n_modules, pmts_per_module)

    hidden = scatter(torch.ones_like(mu), 0.0)
    mu_mat, q_mat = scatter(mu, 0.0), scatter(q, 0.0)
    u = mu_mat * hidden
    residual = (q_mat - mu_mat) * hidden
    variance = (mu_mat * (1.0 + float(sigma_spe) ** 2) + float(v_dark)
                + float(s_p) ** 2 * mu_mat ** 2)
    diagonal = torch.where(hidden > 0.5, variance.clamp_min(float(floor)),
                           torch.ones_like(variance))
    inverse = 1.0 / diagonal
    a = (u * u * inverse).sum(-1)                       # u^T D^-1 u
    b = (u * residual * inverse).sum(-1)                # u^T D^-1 r
    quadratic = (residual * residual * inverse).sum(-1) - b * b / (1.0 / s ** 2 + a)
    log_det = torch.log(diagonal).sum(-1) + torch.log1p(s ** 2 * a)
    return 0.5 * (quadratic + log_det)


def chi2_dense_module(r, mu, *, s: float, s_p: float, sigma_spe: float, v_dark: float,
                      floor: float = 0.0) -> float:
    """Reference dense ``0.5 (r^T C^-1 r + ln det C)`` for one module's hidden PMTs."""
    r, mu = np.asarray(r, dtype=np.float64), np.asarray(mu, dtype=np.float64)
    if r.shape != mu.shape or r.ndim != 1:
        raise ValueError("dense chi2 needs one module's (k,) residual and mean")
    diagonal = mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark) + float(s_p) ** 2 * mu ** 2
    diagonal = np.maximum(diagonal, float(floor)) if floor else diagonal
    covariance = np.diag(diagonal) + float(s) ** 2 * np.outer(mu, mu)
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("dense module covariance is not positive definite")
    return float(0.5 * (r @ np.linalg.solve(covariance, r) + log_det))


def simulate_module_readout(mu, s: float, s_p: float, sigma_spe: float, v_dark: float,
                            *, n_samples: int, seed: int = 0) -> np.ndarray:
    """Draw ``q_p = mu_p (1 + s u_m + s_p w_p) + sqrt(mu_p (1 + sigma_spe^2) + v_dark) z_p``.

    Independent standard normals; the covariance is exactly ``C_m`` (threshold
    censoring is not part of the registered second-moment model).
    """
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 1 or np.any(mu < 0) or not np.isfinite(mu).all():
        raise ValueError("simulate_module_readout needs one module's finite mu >= 0")
    if int(n_samples) < 2:
        raise ValueError("Monte-Carlo covariance needs at least two samples")
    rng = np.random.default_rng(int(seed))
    shared = rng.standard_normal((int(n_samples), 1))
    pmt = rng.standard_normal((int(n_samples), mu.size))
    noise = rng.standard_normal((int(n_samples), mu.size))
    spread = np.sqrt(mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark))
    return mu * (1.0 + float(s) * shared + float(s_p) * pmt) + spread * noise


# ------------------------------------------------------------------- the arms
class LucidArm(nn.Module):
    """One §3.5.6 arm: encoder + predictor (+ head / EMA target / SIGReg / anchor)."""

    def __init__(self, arm: str, geometry, *, settings: ArmSettings,
                 encoder_config: EncoderConfig | None = None,
                 scaling: FeatureScaling | None = None, beta: float | None = None,
                 seed: int = 0, dark_bank: DarkBank | None = None):
        super().__init__()
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown LUCiD arm {arm!r}; expected {ALL_ARMS}")
        if arm in BETA_ARMS:
            if beta is None:
                raise ValueError(f"{arm} requires beta from the registered grid "
                                 f"{settings.beta_grid}; the development cohort selects it")
            if float(beta) not in settings.beta_grid:
                raise ValueError(f"beta {beta!r} is not in the registered grid "
                                 f"{settings.beta_grid}")
            if arm in ANCHOR_ARMS and dark_bank is None:
                raise ValueError(f"{arm} requires dark-noise-only windows; pass dark_bank")
        elif beta is not None:
            raise ValueError(f"{arm} takes no beta")
        self.arm, self.settings, self.seed = arm, settings, int(seed)
        self.beta = float(beta) if beta is not None else None
        self.dark_only = arm == DARK_VARIANT
        self.n_modules = int(geometry.n_modules)
        self.pmts_per_module = int(geometry.pmts_per_module)
        self.n_pmt = int(geometry.n_pmt)
        torch.manual_seed(self.seed)
        self.jepa = LucidPointJepa(encoder_config or EncoderConfig(), geometry, scaling)
        self.d_model = int(self.jepa.config.d_model)
        self.register_buffer("module_of_pmt",
                             torch.as_tensor(geometry.module_index, dtype=torch.long),
                             persistent=False)
        self.register_buffer("slot_of_pmt",
                             torch.as_tensor(geometry.pmt_id_in_mpmt - 1, dtype=torch.long),
                             persistent=False)
        self.charge_floor = float(self.jepa.scaling.charge_floor_pe)
        sizes = {"mae_mse": 3, "mae_chi2": 2}
        self.head = nn.Linear(self.d_model, sizes[arm]) if arm in sizes else None
        self.sigreg = SIGReg(settings.sigreg) if arm in SIGREG_ARMS else None
        self.target_encoder, self.target_norm = None, None
        if arm == "ijepa_ema":
            import copy
            self.target_encoder = copy.deepcopy(self.jepa.encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.target_norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        self._generator = torch.Generator().manual_seed(self.seed + 100003)
        bank = dark_bank if dark_bank is not None else DarkBank.empty(self.n_pmt)
        if int(bank.hit.shape[1]) != self.n_pmt:
            raise ValueError("dark bank and geometry disagree on the PMT count")
        self.register_buffer("dark_charge", torch.as_tensor(bank.charge_pe), persistent=False)
        self.register_buffer("dark_time", torch.as_tensor(bank.time_100ns), persistent=False)
        self.register_buffer("dark_hit", torch.as_tensor(bank.hit), persistent=False)

    @property
    def coefficient_lambda(self) -> float:
        return float(self.settings.coefficient_lambda)

    def parameter_groups(self) -> dict:
        """Registered AdamW/Muon groups; learning rates and schedules are pending_review."""
        muon, adamw = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad or name.startswith("target_encoder."):
                continue
            (muon if ".blocks." in name and parameter.ndim >= 2 else adamw).append(parameter)
        return {"muon": muon, "adamw": adamw}

    def check_parameter_coverage(self) -> dict:
        """Every trainable non-EMA parameter in exactly one group."""
        grouped = self.parameter_groups()
        seen = [id(p) for group in grouped.values() for p in group]
        trainable = [id(p) for name, p in self.named_parameters()
                     if p.requires_grad and not name.startswith("target_encoder.")]
        if len(seen) != len(set(seen)) or set(seen) != set(trainable):
            raise RuntimeError("optimizer groups do not partition the trainable parameters")
        return {"n_trainable": len(trainable), "n_muon": len(grouped["muon"]),
                "n_adamw": len(grouped["adamw"])}

    @staticmethod
    def _hidden_index(batch: LucidBatch) -> Tensor:
        counts = batch.hidden.sum(1)
        if int(counts.min()) != int(counts.max()):
            raise ValueError("arm losses need rectangular hidden views (natural occupancy)")
        index = torch.nonzero(batch.hidden, as_tuple=False)[:, 1]
        return index.reshape(batch.hidden.size(0), int(counts[0]))

    def _encode_views(self, batch: LucidBatch, *, need_full: bool = True):
        hidden_index = self._hidden_index(batch)
        context = self.jepa.encode(batch.features, hidden_mask=batch.hidden)
        prediction = self.jepa.predict(context, batch.context_index, hidden_index)
        if not need_full:
            return context, prediction, None, hidden_index, None
        full = self.jepa.encode(batch.features)
        # WP4 label definition (v3, kept by v4 §2): labels come from a target-only
        # pass; the full unmasked pass is kept only for the anchor and evaluation.
        # A label read-out on the full event can otherwise copy context through
        # the shared encoder (A10 audit / 2026-10-23 label-shortcut diagnostic).
        label_full = self.jepa.encode(batch.features, hidden_mask=~batch.target)
        gather = hidden_index.unsqueeze(-1).expand(-1, -1, self.d_model)
        return context, prediction, label_full.gather(1, gather), hidden_index, full

    @staticmethod
    def _context_pooled(context: Tensor, context_index: Tensor) -> Tensor:
        gather = context_index.unsqueeze(-1).expand(-1, -1, context.size(-1))
        return context.gather(1, gather).mean(1)

    def _module_tokens(self, readout: Tensor) -> Tensor:
        """``z_m``: mean read-out token over each module's PMTs, ``(B, M, d)``."""
        sums = readout.new_zeros((readout.size(0), self.n_modules, readout.size(-1)))
        sums.index_add_(1, self.module_of_pmt, readout)
        return sums / float(self.pmts_per_module)

    @staticmethod
    def _trace_cov(z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.size(-1))
        if flat.size(0) < 2:
            raise ValueError("trace(Cov(z)) needs at least two module tokens")
        centered = flat - flat.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / (flat.size(0) - 1)
        return covariance.diagonal().sum()

    # ---------------------------------------------------------------- forward
    def forward(self, batch: LucidBatch, *, oracle: bool = False,
                twin_batch: LucidBatch | None = None, step: int = 0) -> dict:
        """Arm loss on one tokenised batch; returns the loss tensor and float metrics."""
        if self.arm == ORACLE_ARM:
            if not oracle or twin_batch is None:
                raise PermissionError(
                    "oracle_nominal_twin reads the nominal twin only; pass oracle=True and "
                    "the twin batch (lucid_bombs.nominal_twin guard)")
            batch = twin_batch
        elif oracle or twin_batch is not None:
            raise PermissionError("the nominal twin is private to oracle_nominal_twin")
        if self.arm == "mae_mse":
            return self._forward_mae_mse(batch)
        if self.arm == "mae_chi2":
            return self._forward_mae_chi2(batch)
        if self.arm == "ijepa_ema":
            return self._forward_ijepa_ema(batch, step)
        if self.arm == "anchor_only":
            return self._forward_anchor_only(batch)
        out = self._forward_sigreg(batch)
        return self._add_anchor(batch, out) if self.arm in ANCHOR_ARMS else out

    def _forward_sigreg(self, batch: LucidBatch) -> dict:
        context, prediction, labels, _, full = self._encode_views(batch)
        prediction_loss = F.mse_loss(prediction, labels)
        regulariser = self.sigreg(self._context_pooled(context, batch.context_index), labels)
        loss = prediction_loss + self.coefficient_lambda * regulariser["value"]
        metrics = {"loss": float(loss.detach()), "prediction": float(prediction_loss.detach()),
                   "sigreg": float(regulariser["value"].detach()),
                   "sigreg_context": float(regulariser["context"].detach()),
                   "sigreg_target": float(regulariser["target"].detach())}
        return {"loss": loss, "metrics": metrics, "prediction": prediction,
                "context_tokens": context, "target_tokens": labels, "full_readout": full}

    def _forward_ijepa_ema(self, batch: LucidBatch, step: int) -> dict:
        context, prediction, _, hidden_index, _ = self._encode_views(batch, need_full=False)
        with torch.no_grad():
            # v3 WP4 (kept by v4 §2): the EMA target encoder sees **target-only**
            # tokens with every other position masked, for the clean two-view
            # comparison.  `batch.target` marks the target view, so the target
            # encoder masks its complement.
            full = self.target_encoder(batch.features, hidden_mask=~batch.target,
                                       order=self.jepa.token_order)
            gather = hidden_index.unsqueeze(-1).expand(-1, -1, self.d_model)
            labels = self.target_norm(full.gather(1, gather))
        prediction_loss = F.mse_loss(prediction, labels)
        metrics = {"loss": float(prediction_loss.detach()),
                   "prediction": float(prediction_loss.detach()),
                   "ema_tau": self.settings.ema_tau(step), "step": int(step)}
        return {"loss": prediction_loss, "metrics": metrics, "prediction": prediction,
                "context_tokens": context, "target_tokens": labels}

    def _forward_mae_mse(self, batch: LucidBatch) -> dict:
        _, prediction, _, hidden_index, _ = self._encode_views(batch)
        columns = torch.tensor([LOG_CHARGE_INDEX, TIME_INDEX, HIT_INDEX],
                               device=batch.features.device)
        gather = hidden_index.unsqueeze(-1).expand(-1, -1, N_FEATURES)
        label = batch.features.gather(1, gather)[..., columns]
        out = self.head(prediction)
        hit = label[..., 2] > 0.5
        loss_charge = F.mse_loss(out[..., 0], label[..., 0])
        loss_hit = F.mse_loss(out[..., 2], label[..., 2])
        loss_time = (((out[..., 1] - label[..., 1])[hit] ** 2).mean() if bool(hit.any())
                     else out.sum() * 0.0)
        loss = loss_charge + loss_hit + loss_time
        metrics = {"loss": float(loss.detach()), "charge": float(loss_charge.detach()),
                   "hit": float(loss_hit.detach()), "time": float(loss_time.detach()),
                   "hidden_hits": int(hit.sum())}
        return {"loss": loss, "metrics": metrics, "prediction": out}

    def _forward_mae_chi2(self, batch: LucidBatch) -> dict:
        _, prediction, _, hidden_index, _ = self._encode_views(batch)
        out = self.head(prediction)
        mu = F.softplus(out[..., 0])
        module, slot = self.module_of_pmt[hidden_index], self.slot_of_pmt[hidden_index]
        q = charge_matrix(batch.features).gather(1, hidden_index)
        terms = chi2_module_terms(q, mu, module, slot, n_modules=self.n_modules,
                                  pmts_per_module=self.pmts_per_module,
                                  s=self.settings.s, s_p=self.settings.s_p,
                                  sigma_spe=self.settings.sigma_spe,
                                  v_dark=self.settings.v_dark,
                                  floor=self.settings.variance_floor_pe2)
        loss_charge = terms.sum(-1).mean()   # terms already carry the registered 1/2
        gather = hidden_index.unsqueeze(-1).expand(-1, -1, N_FEATURES)
        truth = batch.features.gather(1, gather)
        hit = truth[..., HIT_INDEX] > 0.5
        sigma_t = self.settings.time_sigma_ns / 100.0
        loss_time = ((0.5 * ((out[..., 1] - truth[..., TIME_INDEX])[hit] / sigma_t) ** 2).mean()
                     if bool(hit.any()) else out.sum() * 0.0)
        variance = (mu * (1.0 + self.settings.sigma_spe ** 2) + self.settings.v_dark
                    + self.settings.s_p ** 2 * mu ** 2)
        loss = loss_charge + loss_time
        metrics = {"loss": float(loss.detach()), "chi2": float(loss_charge.detach()),
                   "time_nll": float(loss_time.detach()),
                   "chi2_floor_fraction": float((variance < self.settings.variance_floor_pe2)
                                                .float().mean()),
                   "hidden_hits": int(hit.sum()), "uses_oracle_noise_information": True}
        return {"loss": loss, "metrics": metrics, "prediction": out, "mu": mu}

    def _forward_anchor_only(self, batch: LucidBatch) -> dict:
        hidden_index = self._hidden_index(batch)
        context = self.jepa.encode(batch.features, hidden_mask=batch.hidden)
        full = self.jepa.encode(batch.features)
        gather = hidden_index.unsqueeze(-1).expand(-1, -1, self.d_model)
        labels = full.gather(1, gather)
        regulariser = self.sigreg(self._context_pooled(context, batch.context_index), labels)
        penalty = self.anchor_penalty(self._module_tokens(full),
                                      self._module_tokens(self._perturbed_readout(batch.features)))
        loss = self.coefficient_lambda * regulariser["value"] + penalty
        metrics = {"loss": float(loss.detach()), "sigreg": float(regulariser["value"].detach()),
                   "anchor": float(penalty.detach()), "anchor_beta": self.beta, "prediction": 0.0}
        return {"loss": loss, "metrics": metrics, "context_tokens": context,
                "target_tokens": labels}

    # ----------------------------------------------------------------- anchor
    def anchor_penalty(self, z: Tensor, z_tilde: Tensor) -> Tensor:
        """``beta * mean_{b,m} ||z~(m) - z(m)||^2 / trace(Cov(z))`` (trace kept)."""
        if z.shape != z_tilde.shape or z.ndim != 3:
            raise ValueError("anchor penalty needs two equal (B, M, d) module-token tensors")
        numerator = (z_tilde - z).pow(2).sum(-1).mean()
        return self.beta * numerator / self._trace_cov(z).clamp_min(_TINY)

    def perturb(self, features: Tensor, *, with_dark: bool = True, with_rescale: bool = True,
                dark_index: Tensor | None = None, rescale: Tensor | None = None) -> Tensor:
        """One perturbed copy ``x~``: dark-window overlay then per-module rescale.

        ``dark_index``/``rescale`` are injectable for fixed draws; the rescale is
        applied in the log-charge domain, so a unit factor is bit-exact.
        """
        batch, n_pmt = features.shape[:2]
        if n_pmt != self.n_pmt:
            raise ValueError("perturb: feature tensor and geometry disagree")
        log_charge = features[..., LOG_CHARGE_INDEX].clone()
        time = features[..., TIME_INDEX].clone()
        hit = features[..., HIT_INDEX] > 0.5
        if with_dark:
            if dark_index is None:
                dark_index = torch.randint(0, int(self.dark_charge.size(0)), (batch,),
                                           generator=self._generator)
            dark_index = dark_index.to(device=self.dark_charge.device, dtype=torch.long)
            dark_q, dark_t, dark_hit = (self.dark_charge[dark_index],
                                        self.dark_time[dark_index],
                                        self.dark_hit[dark_index])
            both, only = hit & dark_hit, dark_hit & ~hit
            with torch.no_grad():
                merged = torch.log(torch.exp(log_charge) + dark_q)
                dark_log_q = torch.log(dark_q.clamp_min(self.charge_floor))
            log_charge = torch.where(only, dark_log_q.to(log_charge.dtype),
                                     torch.where(both, merged, log_charge))
            time = torch.where(both, torch.minimum(time, dark_t),
                               torch.where(only, dark_t, time))
            hit = hit | dark_hit
        if with_rescale:
            if rescale is None:
                rescale = 1.0 + self.settings.rescale_sd * torch.randn(
                    (batch, self.n_modules), generator=self._generator)
            if rescale.shape != (batch, self.n_modules):
                raise ValueError("perturb: rescale must be (B, n_modules)")
            gain = rescale.to(device=features.device, dtype=features.dtype)[:, self.module_of_pmt]
            log_charge = torch.where(hit, log_charge + torch.log(gain.clamp_min(self.charge_floor)),
                                     log_charge)
        out = features.clone()
        out[..., LOG_CHARGE_INDEX] = log_charge
        out[..., TIME_INDEX] = torch.where(hit, time, torch.zeros_like(time))
        out[..., HIT_INDEX] = hit.to(features.dtype)
        return out

    def _perturbed_readout(self, features: Tensor) -> Tensor:
        """Encoder read-out of the perturbed copy (draws per the arm's variant)."""
        return self.jepa.encode(self.perturb(features, with_dark=True,
                                             with_rescale=not self.dark_only))

    def _add_anchor(self, batch: LucidBatch, out: dict) -> dict:
        z = self._module_tokens(out["full_readout"])
        z_tilde = self._module_tokens(self._perturbed_readout(batch.features))
        penalty = self.anchor_penalty(z, z_tilde)
        loss = out["loss"] + penalty
        metrics = dict(out["metrics"])
        metrics.update({"loss": float(loss.detach()), "anchor": float(penalty.detach()),
                        "anchor_beta": self.beta, "anchor_draws": 1 if self.dark_only else 2,
                        "anchor_trace": float(self._trace_cov(z).detach())})
        return dict(out, loss=loss, metrics=metrics, anchor=penalty)

    # -------------------------------------------------------------------- EMA
    def update_target(self, step: int) -> float:
        """One cosine EMA step of the ``ijepa_ema`` target encoder; returns tau."""
        if self.target_encoder is None:
            raise ValueError("only ijepa_ema has a target encoder")
        tau = self.settings.ema_tau(step)
        with torch.no_grad():
            for target, online in zip(self.target_encoder.parameters(),
                                      self.jepa.encoder.parameters()):
                target.mul_(tau).add_(online, alpha=1.0 - tau)
        return tau


def build_arm(arm: str, geometry, *, config: dict | None = None,
              settings: ArmSettings | None = None, cell=None, beta: float | None = None,
              seed: int = 0, dark_bank: DarkBank | None = None,
              encoder_config: EncoderConfig | None = None,
              scaling: FeatureScaling | None = None) -> LucidArm:
    """Build one arm; ``seed`` fixes the init (the floor hook uses the first seed)."""
    settings = settings or ArmSettings.from_config(config, cell=cell, beta=beta)
    return LucidArm(arm, geometry, settings=settings, encoder_config=encoder_config,
                    scaling=scaling, beta=beta, seed=seed, dark_bank=dark_bank)


# ---------------------------------------------------------------------- probes
@dataclass(frozen=True)
class RunProbeData:
    """One run's frozen embeddings, raw features, masks and detector-state truth."""

    run_id: str
    event_ids: np.ndarray
    tokens: np.ndarray          # (E, N, d) full-input read-out
    context_tokens: np.ndarray  # (E, N, d) context-encoder read-out (masked input)
    raw: np.ndarray             # (E, N, N_FEATURES) token features
    hit: np.ndarray             # (E, N) bool
    hidden: np.ndarray          # (E, N) bool, the training mask
    module_index: np.ndarray    # (N,)
    truth: dict = None

    def __post_init__(self):
        n_events, n_pmt = self.hit.shape
        # The streaming evaluator builds one representation per form and leaves the
        # others None (it never holds the full (E, N, d) token table); every present
        # array is still validated against (E, N).
        if self.tokens is not None and (self.tokens.shape[:2] != (n_events, n_pmt)
                                        or (self.context_tokens is not None
                                            and self.context_tokens.shape != self.tokens.shape)):
            raise ValueError("probe data: tokens/context tokens must share (E, N)")
        if self.context_tokens is not None and self.context_tokens.shape[:2] != (n_events, n_pmt):
            raise ValueError("probe data: context tokens must be (E, N, d)")
        if self.raw is not None and (self.raw.shape[:2] != (n_events, n_pmt)
                                     or self.raw.shape[-1] != N_FEATURES):
            raise ValueError("probe data: raw features must be (E, N, N_FEATURES)")
        if self.hidden.shape != self.hit.shape or self.module_index.shape != (n_pmt,):
            raise ValueError("probe data: masks and module index disagree with (E, N)")
        object.__setattr__(self, "truth", dict(self.truth or {}))


def _select_truth(truth: dict, event_ids: np.ndarray) -> dict:
    """Align per-event truth to the events a tokenised batch actually kept."""
    stored = truth.get("event_ids")
    if stored is None:
        return dict(truth)
    index = np.searchsorted(np.asarray(stored, dtype=np.int64), np.asarray(event_ids))
    if np.any(index >= len(stored)) or not np.array_equal(
            np.asarray(stored, dtype=np.int64)[index], np.asarray(event_ids)):
        raise ValueError("probe truth does not cover the tokenised event ids")
    out = dict(truth)
    for key in ("vertex", "log_yield"):
        if truth.get(key) is not None:
            out[key] = np.asarray(truth[key])[index]
    return out


def probe_data(model: LucidPointJepa, batch: LucidBatch, truth: dict, geometry, *,
               run_id: str = "") -> RunProbeData:
    """Freeze one run's embeddings (full input and masked context) plus its truth."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        tokens = model.encode(batch.features).cpu().numpy().astype(np.float64)
        context = model.encode(batch.features,
                               hidden_mask=batch.hidden).cpu().numpy().astype(np.float64)
    if was_training:
        model.train()
    return RunProbeData(run_id=str(run_id), event_ids=np.asarray(batch.event_ids),
                        tokens=tokens, context_tokens=context,
                        raw=batch.features.cpu().numpy().astype(np.float64),
                        hit=np.asarray(batch.hit_flags, bool),
                        hidden=np.asarray(batch.hidden.cpu(), bool),
                        module_index=np.asarray(geometry.module_index, np.int64),
                        truth=_select_truth(truth, np.asarray(batch.event_ids)))


def truth_from_state(geometry, state: DetectorState, *, vertex_m, yield_photons) -> dict:
    """Detector-state truth targets of §3.5.6 for one run's events."""
    vertex = np.asarray(vertex_m, dtype=float).reshape(-1, 3)
    yield_photons = np.asarray(yield_photons, dtype=float).reshape(-1)
    if vertex.shape[0] != yield_photons.size:
        raise ValueError("truth_from_state needs one vertex per yield")
    truth = {"vertex": vertex, "log_yield": np.log(yield_photons),
             "module_gain": np.log(np.asarray(state.module_gain, dtype=float)),
             "pmt_gain": np.log(np.asarray(state.pmt_gain, dtype=float)),
             "f2_variable": np.nan, "age": np.nan}
    setting = None if state.setting is None else float(state.setting)
    if state.law == "F2" and setting:
        truth["f2_variable"] = float(np.log(state.attenuation_length_m / LAMBDA0_M) / setting)
    elif state.law == "F2p" and setting:
        z = np.asarray(geometry.positions_m, float)[:, 2]
        v = (np.asarray(state.gradient, float) - 1.0) * float(geometry.half_height_m) \
            / np.where(np.abs(z) > 1e-9, setting * z, 1.0)
        valid = np.isfinite(v) & (np.abs(z) > 1e-9)
        truth["f2_variable"] = float(np.median(v[valid])) if valid.any() else np.nan
    if state.law == "T" and setting is not None:
        truth["age"] = float(setting)
    return truth


def _mean_tokens(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return x.mean(axis=1)
    weights = mask.astype(np.float64)
    return (x * weights[..., None]).sum(axis=1) / np.maximum(weights.sum(1), 1.0)[:, None]


def _module_pool(x: np.ndarray, module_index: np.ndarray, n_modules: int,
                 mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-event module means ``(E, M, d)`` and their token counts ``(E, M)``."""
    order = np.argsort(module_index, kind="stable")
    counts_all = np.bincount(module_index, minlength=n_modules)
    starts = np.concatenate([[0], np.cumsum(counts_all)[:-1]])
    ordered = x[:, order]
    if mask is None:
        pooled = np.add.reduceat(ordered, starts, axis=1) / counts_all[None, :, None]
        return pooled, np.broadcast_to(counts_all.astype(np.float64), pooled.shape[:2]).copy()
    weights = mask[:, order].astype(np.float64)
    sums = np.add.reduceat(ordered * weights[..., None], starts, axis=1)
    counts = np.add.reduceat(weights, starts, axis=1)
    return sums / np.maximum(counts, 1.0)[..., None], counts


@dataclass(frozen=True)
class ProbeDataset:
    """Aligned probe rows: features, targets, run groups and row provenance."""

    kind: str
    form: str
    x: np.ndarray
    y: np.ndarray
    run_ids: np.ndarray
    target_names: tuple
    meta: dict

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])


def _rows(run_id, x, y, key, out):
    out["x"].append(np.asarray(x, float))
    out["y"].append(np.asarray(y, float).reshape(len(x), -1))
    out["run"].append(np.full(len(x), run_id, dtype=object))
    out["key"].append(np.asarray(key, dtype=np.int64))


def _form_representation(data: RunProbeData, form: str):
    """The one input representation a probe form reads, plus its masking rule."""
    if form == "raw_input":
        return data.raw, None
    if form == "propagated":
        return data.context_tokens, data.hidden
    return data.tokens, None


def probe_rows(data: RunProbeData, kind: str, form: str | None = None):
    """One run's ``(x, y, key, target_names)`` rows for one §3.5.6 probe form.

    The single-run row builder shared by :func:`build_probe_dataset` (materialised,
    debug/test path) and :class:`ProbeAccumulator` (streaming sufficient
    statistics), so both see byte-identical rows.  ``x`` is returned in the
    array's natural precision (float32 for raw/token slices).
    """
    form = PRIMARY_PROBE_FORM if form is None else str(form)
    if kind not in PROBE_KINDS:
        raise ValueError(f"unknown probe kind {kind!r}; expected {PROBE_KINDS}")
    if form not in PROBE_FORMS:
        raise ValueError(f"unknown probe form {form!r}; expected {PROBE_FORMS}")
    if form == "run_mean" and kind in ("vertex", "log_yield"):
        raise ValueError(f"run-mean is undefined for the per-event target {kind!r}")
    x, mask = _form_representation(data, form)
    if x is None:
        raise ValueError(f"probe data {data.run_id!r} has no {form!r} representation")
    events = int(data.hit.shape[0])
    if kind in ("vertex", "log_yield", "f2_variable", "age"):
        if kind == "vertex":
            target, names = np.asarray(data.truth["vertex"], float), ("x", "y", "z")
        elif kind == "log_yield":
            target = np.asarray(data.truth["log_yield"], float)[:, None]
            names = ("log_yield",)
        else:
            value = data.truth.get(kind, np.nan)
            if value is None or not np.isfinite(value):
                raise ValueError(f"{data.run_id}: target {kind!r} is unavailable")
            target, names = np.full((events, 1), float(value)), (kind,)
        features = _mean_tokens(x, mask)
        if form == "run_mean":
            return (features.mean(0, keepdims=True), target[:1], np.zeros(1, dtype=np.int64), names)
        return features, target, np.zeros(events, dtype=np.int64), names
    if kind == "module_gain":
        target, names = np.asarray(data.truth["module_gain"], float), ("log_gain_m",)
        pooled, counts = _module_pool(x, data.module_index, target.size, mask)
        if form == "run_mean":
            pooled, counts = pooled.mean(0, keepdims=True), counts.sum(0, keepdims=True)
        event_row, module_col = np.nonzero(counts > 0)
        return (pooled[event_row, module_col], target[module_col][:, None],
                module_col.astype(np.int64), names)
    # pmt_gain
    target, names = np.asarray(data.truth["pmt_gain"], float), ("log_gain_p",)
    selected = data.hidden if form == "propagated" else data.hit
    event_row, pmt_col = np.nonzero(selected)
    values = x[event_row, pmt_col]
    if form == "run_mean":
        sums = np.zeros((target.size, values.shape[-1]))
        np.add.at(sums, pmt_col, values)
        counts = np.bincount(pmt_col, minlength=target.size)
        keep = np.flatnonzero(counts > 0)
        return sums[keep] / counts[keep][:, None], target[keep][:, None], keep.astype(np.int64), names
    return values, target[pmt_col][:, None], pmt_col.astype(np.int64), names


def probe_run_mean_partial(data: RunProbeData, kind: str):
    """Partial (chunk-level) sums for one run's ``run_mean`` row.

    ``run_mean`` aggregates over *all* events of a run, so a streamed run must not
    be reduced chunk-by-chunk.  This returns the additive per-chunk pieces; merge
    them across chunks and call :func:`finalise_run_mean` once per run.
    """
    if kind not in PROBE_KINDS:
        raise ValueError(f"unknown probe kind {kind!r}; expected {PROBE_KINDS}")
    if kind in ("vertex", "log_yield"):
        raise ValueError(f"run-mean is undefined for the per-event target {kind!r}")
    x, mask = _form_representation(data, PRIMARY_PROBE_FORM)
    events = int(data.hit.shape[0])
    if kind in ("f2_variable", "age"):
        value = data.truth.get(kind, np.nan)
        if value is None or not np.isfinite(value):
            raise ValueError(f"{data.run_id}: target {kind!r} is unavailable")
        features = _mean_tokens(x, mask)
        return {"mode": "scalar", "sums": features.sum(0, keepdims=True), "events": int(events),
                "counts": int(events), "target": np.full(1, float(value)), "key": None,
                "names": (kind,)}
    if kind == "module_gain":
        target = np.asarray(data.truth["module_gain"], float)
        pooled, counts = _module_pool(x, data.module_index, target.size, mask)
        return {"mode": "module", "sums": pooled.sum(0), "events": int(events),
                "counts": counts.sum(0), "target": target, "key": None,
                "names": ("log_gain_m",)}
    target = np.asarray(data.truth["pmt_gain"], float)
    event_row, pmt_col = np.nonzero(data.hit)
    values = x[event_row, pmt_col]
    sums = np.zeros((target.size, values.shape[-1]), dtype=np.float64)
    np.add.at(sums, pmt_col, values)
    counts = np.bincount(pmt_col, minlength=target.size)
    return {"mode": "pmt", "sums": sums, "events": int(events), "counts": counts,
            "target": target, "key": None, "names": ("log_gain_p",)}


def _merge_run_mean_partial(left: dict, right: dict) -> dict:
    merged = dict(left)
    merged["sums"] = left["sums"] + right["sums"]
    merged["counts"] = left["counts"] + right["counts"]
    merged["events"] = int(left.get("events", 0)) + int(right.get("events", 0))
    return merged


def finalise_run_mean(partial: dict):
    """Turn merged per-run ``run_mean`` partials into ``(x, y, key, names)`` rows."""
    if partial["mode"] == "scalar":
        x = partial["sums"] / max(int(partial["counts"]), 1)
        y = partial["target"][:, None]
        return x, y, np.zeros(x.shape[0], dtype=np.int64), partial["names"]
    target, counts = partial["target"], partial["counts"]
    keep = counts > 0
    if partial["mode"] == "module":
        x = partial["sums"] / max(int(partial["events"]), 1)
        return x[keep], target[keep][:, None], np.flatnonzero(keep).astype(np.int64), partial["names"]
    x = partial["sums"][keep] / counts[keep][:, None]
    return x, target[keep][:, None], np.flatnonzero(keep).astype(np.int64), partial["names"]


def build_probe_dataset(runs: Sequence[RunProbeData], kind: str,
                        form: str | None = None) -> ProbeDataset:
    """Assemble one §3.5.6 probe form for one target kind (materialised).

    ``run_mean`` is primary; ``vertex``/``log_yield`` are per-event truths, so a
    run-mean target is undefined for them and is refused.  This is the legacy
    full-matrix path, kept for tests and ``legacy_materialised=True``; the runner
    evaluates with :class:`ProbeAccumulator` so no full table is materialised.
    """
    form = PRIMARY_PROBE_FORM if form is None else str(form)
    if kind not in PROBE_KINDS:
        raise ValueError(f"unknown probe kind {kind!r}; expected {PROBE_KINDS}")
    if form not in PROBE_FORMS:
        raise ValueError(f"unknown probe form {form!r}; expected {PROBE_FORMS}")
    if form == "run_mean" and kind in ("vertex", "log_yield"):
        raise ValueError(f"run-mean is undefined for the per-event target {kind!r}")
    out = {"x": [], "y": [], "run": [], "key": []}
    names: tuple = ()
    for data in runs:
        x, y, key, names = probe_rows(data, kind, form)
        _rows(data.run_id, x, y, key, out)
    if not out["x"]:
        raise ValueError(f"probe dataset {kind!r}/{form!r} has no rows")
    return ProbeDataset(kind=kind, form=form, x=np.concatenate(out["x"]),
                        y=np.concatenate(out["y"]), run_ids=np.concatenate(out["run"]),
                        target_names=names, meta={"key": np.concatenate(out["key"])})


def _fold_labels(run_ids: np.ndarray, n_folds: int = 5) -> np.ndarray:
    """Five grouped folds by run: sorted runs assigned round-robin (v3 §1.3 CV)."""
    unique = np.unique(run_ids)
    if unique.size < n_folds:
        raise ValueError("five run-grouped folds need at least five runs")
    return unique[np.arange(unique.size) % n_folds][np.searchsorted(unique, run_ids)]


# --------------------------------------------------- streaming probe statistics
@dataclass
class _RunStats:
    """One run's raw sufficient statistics for a (kind, form) probe table.

    Everything is stored relative to the accumulator's fixed reference row
    (``c`` for features, ``d`` for targets), which keeps the shifted moments
    well conditioned: no raw first/second moment of a large-offset feature is
    accumulated directly.
    """

    n: int
    b: np.ndarray        # (d,)   sum(x - c)
    gram: np.ndarray     # (d, d) sum (x - c)(x - c)^T
    sy: np.ndarray       # (t,)   sum(y - d)
    sxy: np.ndarray      # (d, t) sum (x - c)(y - d)^T
    syy: np.ndarray      # (t,)   sum (y - d)^2
    xmin: np.ndarray     # (d,)
    xmax: np.ndarray     # (d,)
    ymin: np.ndarray     # (t,)
    ymax: np.ndarray     # (t,)


def _empty_run_stats(dim: int, targets: int) -> _RunStats:
    return _RunStats(n=0, b=np.zeros(dim), gram=np.zeros((dim, dim)),
                     sy=np.zeros(targets), sxy=np.zeros((dim, targets)),
                     syy=np.zeros(targets), xmin=np.full(dim, np.inf),
                     xmax=np.full(dim, -np.inf), ymin=np.full(targets, np.inf),
                     ymax=np.full(targets, -np.inf))


def _sum_run_stats(stats: Sequence[_RunStats]) -> _RunStats:
    total = None
    for part in stats:
        if total is None:
            total = _empty_run_stats(part.b.size, part.sy.size)
        total.n += part.n
        total.b += part.b
        total.gram += part.gram
        total.sy += part.sy
        total.sxy += part.sxy
        total.syy += part.syy
        np.minimum(total.xmin, part.xmin, out=total.xmin)
        np.maximum(total.xmax, part.xmax, out=total.xmax)
        np.minimum(total.ymin, part.ymin, out=total.ymin)
        np.maximum(total.ymax, part.ymax, out=total.ymax)
    if total is None:
        raise ValueError("no probe statistics to combine")
    return total


def _standardisation(total: _RunStats, reference: np.ndarray):
    """Fold-train mean/std (population) with the §1.3 constant-column rule."""
    n = total.n
    mean = reference + total.b / n
    centred = total.gram - np.outer(total.b, total.b) / n
    variance = np.diag(centred) / n
    constant = total.xmin == total.xmax
    scale = np.sqrt(np.maximum(variance, 0.0))
    mean = np.array(mean, dtype=np.float64)
    mean[constant] = total.xmin[constant]      # exactly x[0], as in fit_ridge
    scale[constant] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("feature scaling numerically invalid")
    return mean, scale, constant


def _standardised_moments(stats: _RunStats, mean: np.ndarray, scale: np.ndarray,
                          reference: np.ndarray, target_reference: np.ndarray):
    """``ZᵀZ``, ``Zᵀy'`` and ``Zᵀ1`` for rows standardised by a *train* fit.

    ``u = mean - reference``; the statistics are centred at ``reference`` so the
    transform is a stable rank-one correction rather than a raw moment difference.
    """
    n, u, inv = stats.n, mean - reference, 1.0 / scale
    centred = stats.gram - np.multiply.outer(u, stats.b) - np.multiply.outer(stats.b, u) \
        + n * np.outer(u, u)
    gram = centred * inv[:, None] * inv[None, :]
    sxy = (stats.sxy - np.multiply.outer(u, stats.sy)) * inv[:, None]
    b = (stats.b - n * u) * inv
    return gram, sxy, b


def _regression_sse(gram, sxy_column, z1, n: int, sy: float, syy: float,
                    coefficients: np.ndarray, intercept_shift: float) -> float:
    """SSE of ``z·β + a`` against shifted targets, from sufficient statistics only."""
    return float(coefficients @ gram @ coefficients
                 + 2.0 * intercept_shift * float(coefficients @ z1)
                 - 2.0 * float(coefficients @ sxy_column)
                 + n * intercept_shift ** 2 - 2.0 * intercept_shift * float(sy) + float(syy))


@dataclass
class StreamedRidgeFit:
    """Ridge fit solved from sufficient statistics; ``predict`` for small tables."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float

    def predict(self, features):
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError("feature dimension mismatch")
        return ((features - self.mean) / self.scale) @ self.coefficients + self.intercept


class ProbeAccumulator:
    """Streaming sufficient statistics for one ``(kind, form)`` probe table.

    The §1.3 protocol is unchanged: the registered alpha grid, five folds grouped
    by whole run, fold-internal standardisation and the same R² definition.  Rows
    arrive in chunks of events (``add``) and are never all held at once: per run we
    keep ``sum(x-c)``, ``sum (x-c)(x-c)^T``, ``sum(y-d)``, ``sum (x-c)(y-d)^T`` and
    ``sum (y-d)^2`` in float64 (``block_rows`` bounds the transient float64 copy of
    a chunk).  Validation and test losses are evaluated from the *same* fold-held
    sufficient statistics, which is algebraically identical to predicting every
    held row.
    """

    def __init__(self, kind: str, form: str | None = None, target_names: tuple = (),
                 *, n_folds: int = 5, block_rows: int = 8192):
        if kind not in PROBE_KINDS:
            raise ValueError(f"unknown probe kind {kind!r}; expected {PROBE_KINDS}")
        self.form = PRIMARY_PROBE_FORM if form is None else str(form)
        if self.form not in PROBE_FORMS:
            raise ValueError(f"unknown probe form {self.form!r}; expected {PROBE_FORMS}")
        self.kind, self.target_names = str(kind), tuple(target_names)
        self.n_folds, self.block_rows = int(n_folds), int(block_rows)
        if self.block_rows < 1:
            raise ValueError("block_rows must be positive")
        self.reference = None
        self.target_reference = None
        self.runs: dict[str, _RunStats] = {}
        self.n_rows = 0

    def add(self, run_id, features, targets) -> None:
        """Accumulate one chunk of rows (all from ``run_id``)."""
        x = np.asarray(features)
        if x.ndim != 2:
            raise ValueError("probe chunk features must be a 2-D matrix")
        y = np.asarray(targets, dtype=np.float64)
        n = int(x.shape[0])
        if n == 0:
            return
        y = y.reshape(n, -1)
        if y.shape[0] != n:
            raise ValueError("probe chunk features/targets disagree")
        if self.reference is None:
            self.reference = np.asarray(x[0], dtype=np.float64)
            self.target_reference = np.array(y[0], dtype=np.float64)
        if x.shape[1] != self.reference.size or y.shape[1] != self.target_reference.size:
            raise ValueError("probe chunk dimension changed")
        stats = self.runs.get(run_id)
        if stats is None:
            stats = _empty_run_stats(self.reference.size, self.target_reference.size)
            self.runs[run_id] = stats
        c, d = self.reference, self.target_reference
        for start in range(0, n, self.block_rows):
            stop = min(start + self.block_rows, n)
            xb = np.asarray(x[start:stop], dtype=np.float64)
            yb = y[start:stop]
            if not np.isfinite(xb).all() or not np.isfinite(yb).all():
                raise ValueError("probe chunk features and targets must be finite")
            shifted_x, shifted_y = xb - c, yb - d
            stats.gram += shifted_x.T @ shifted_x
            stats.b += shifted_x.sum(0)
            stats.sxy += shifted_x.T @ shifted_y
            stats.sy += shifted_y.sum(0)
            stats.syy += np.einsum("ij,ij->j", shifted_y, shifted_y)
            stats.n += stop - start
            np.minimum(stats.xmin, xb.min(0), out=stats.xmin)
            np.maximum(stats.xmax, xb.max(0), out=stats.xmax)
            np.minimum(stats.ymin, yb.min(0), out=stats.ymin)
            np.maximum(stats.ymax, yb.max(0), out=stats.ymax)
        self.n_rows += n

    # -- fold-wise ridge ---------------------------------------------------
    def fit(self, *, fit_intercept: bool = True) -> dict:
        """Five grouped-fold ridge per target; same output shape as ``fit_probe``."""
        if type(fit_intercept) is not bool:
            raise ValueError("explicit boolean intercept required")
        if self.n_rows == 0 or not self.runs:
            raise ValueError(f"probe dataset {self.kind!r}/{self.form!r} has no rows")
        unique = np.array(sorted(self.runs))
        if unique.size < self.n_folds:
            raise ValueError("five run-grouped folds need at least five runs")
        fold_of = {str(run): index % self.n_folds for index, run in enumerate(unique.tolist())}
        total = _sum_run_stats(list(self.runs.values()))
        reference, target_reference = self.reference, self.target_reference
        folds = []
        for fold in range(self.n_folds):
            held_ids = [rid for rid in unique.tolist() if fold_of[str(rid)] == fold]
            held = _sum_run_stats([self.runs[rid] for rid in held_ids])
            train = _sum_run_stats([self.runs[rid] for rid in unique.tolist()
                                    if fold_of[str(rid)] != fold])
            mean, scale, _ = _standardisation(train, reference)
            gram, sxy, z1 = _standardised_moments(train, mean, scale, reference,
                                                  target_reference)
            held_gram, held_sxy, held_z1 = _standardised_moments(held, mean, scale, reference,
                                                                 target_reference)
            folds.append({"mean": mean, "scale": scale, "gram": gram, "sxy": sxy, "z1": z1,
                          "held_gram": held_gram, "held_sxy": held_sxy, "held_z1": held_z1,
                          "held": held, "train_mean_y": train.sy / train.n})
        mean, scale, _ = _standardisation(total, reference)
        gram, sxy, z1 = _standardised_moments(total, mean, scale, reference, target_reference)
        raw_mean_y = target_reference + total.sy / total.n
        results = []
        for column in range(int(total.sy.size)):
            if total.ymin[column] == total.ymax[column]:
                results.append({"status": "not_evaluable",
                                "reason": "constant_probe_train_target", "alpha": None,
                                "fit": None, "validation_mse": None, "folds": []})
                continue
            squared_errors = {alpha: 0.0 for alpha in ridge.ALPHAS}
            for fold in folds:
                # mean(y) - target-reference, i.e. the shift the fold model's intercept
                # is applied at (0 - d when the intercept is disabled).
                intercept_shift = (float(fold["train_mean_y"][column]) if fit_intercept
                                   else -float(target_reference[column]))
                for alpha in ridge.ALPHAS:
                    coefficients = np.linalg.solve(
                        fold["gram"] + alpha * np.eye(gram.shape[0]),
                        fold["sxy"][:, column] - intercept_shift * fold["z1"])
                    squared_errors[alpha] += _regression_sse(
                        fold["held_gram"], fold["held_sxy"][:, column], fold["held_z1"],
                        fold["held"].n, float(fold["held"].sy[column]),
                        float(fold["held"].syy[column]), coefficients, intercept_shift)
            scores = {alpha: value / total.n for alpha, value in squared_errors.items()}
            chosen = ridge.select_alpha(scores)
            intercept = float(raw_mean_y[column]) if fit_intercept else 0.0
            coefficients = np.linalg.solve(
                gram + chosen * np.eye(gram.shape[0]),
                sxy[:, column] - (intercept - float(target_reference[column])) * z1)
            fit = StreamedRidgeFit(mean.copy(), scale.copy(), coefficients, intercept,
                                   float(chosen))
            results.append({"status": "fit", "reason": None, "alpha": chosen, "fit": fit,
                            "validation_mse": scores, "folds": []})
        return {"kind": self.kind, "form": self.form, "target_names": self.target_names,
                "fits": results,
                "status": "evaluated" if any(r["status"] == "fit" for r in results)
                else "not_evaluable"}

    # -- memory accounting -------------------------------------------------
    def memory_mb(self) -> float:
        """Approximate bytes held by the sufficient statistics, in MiB."""
        total = 0
        for stats in self.runs.values():
            total += (stats.b.nbytes + stats.gram.nbytes + stats.sy.nbytes
                      + stats.sxy.nbytes + stats.syy.nbytes + stats.xmin.nbytes
                      + stats.xmax.nbytes + stats.ymin.nbytes + stats.ymax.nbytes)
        return total / (1024.0 * 1024.0)


def _test_squared_error(fit: StreamedRidgeFit, column: int, test: ProbeAccumulator) -> float:
    """SSE of a fitted ridge probe on a streamed test table, from its statistics."""
    total = _sum_run_stats(list(test.runs.values()))
    mean, scale = fit.mean, fit.scale
    gram, sxy, z1 = _standardised_moments(total, mean, scale, test.reference,
                                          test.target_reference)
    return _regression_sse(gram, sxy[:, column], z1, total.n, float(total.sy[column]),
                           float(total.syy[column]), fit.coefficients,
                           fit.intercept - float(test.target_reference[column]))


def probe_r2_streamed(train: ProbeAccumulator, test: ProbeAccumulator, *,
                      fit_intercept: bool = True) -> dict:
    """Fit on probe-train and score probe-test from sufficient statistics only.

    Same schema and R² definition as :func:`probe_r2`; the alpha choice is the
    registered :func:`jepa_noise.probes.select_alpha` over the same fold-wise MSEs.
    """
    if (train.kind, train.form) != (test.kind, test.form):
        raise ValueError("probe train and test sets must share kind and form")
    base = {"kind": train.kind, "form": train.form, "target_names": train.target_names}
    if train.n_rows == 0 or test.n_rows == 0:
        return dict(base, status="not_evaluable", reason="empty_dataset", r2=None,
                    r2_mean=None, alphas=None)
    if test.n_rows < 2:
        raise ValueError("paired target matrices with at least two rows required")
    fitted = train.fit(fit_intercept=fit_intercept)
    total = _sum_run_stats(list(test.runs.values()))
    rows = []
    for column, probe in enumerate(fitted["fits"]):
        if probe["status"] != "fit":
            rows.append({"status": "not_evaluable", "reason": probe["reason"]})
            continue
        if total.ymin[column] == total.ymax[column]:
            rows.append({"status": "not_evaluable", "reason": "constant_probe_test_target"})
            continue
        sse = _test_squared_error(probe["fit"], column, test)
        sst = float(total.syy[column]) - float(total.sy[column]) ** 2 / total.n
        if not np.isfinite(sse) or not np.isfinite(sst) or sst <= 0:
            raise ValueError("nuisance R2 numerically invalid")
        rows.append({"status": "evaluated", "reason": None, "r2": float(1.0 - sse / sst)})
    scores = [row["r2"] for row in rows if row["status"] == "evaluated"]
    if not scores:
        return dict(base, status="not_evaluable", reason=rows[0]["reason"], r2=None,
                    r2_mean=None, alphas=None)
    return dict(base, status="evaluated", reason=None, r2=scores,
                r2_mean=float(np.mean(scores)),
                alphas=[row["alpha"] for row in fitted["fits"] if row["status"] == "fit"],
                n_train=train.n_rows, n_test=test.n_rows)


def fit_probe(train: ProbeDataset, *, fit_intercept: bool = True) -> dict:
    """Five-fold run-grouped ridge per target (v3 §1.3); returns fits and status."""
    results = ridge.grouped_ridge(train.x, train.y, _fold_labels(train.run_ids),
                                  fit_intercept=fit_intercept)
    return {"kind": train.kind, "form": train.form, "target_names": train.target_names,
            "fits": results,
            "status": "evaluated" if any(r["status"] == "fit" for r in results)
            else "not_evaluable"}


def probe_predictions(fitted: dict, dataset: ProbeDataset) -> np.ndarray:
    """Predictions aligned with ``dataset`` rows, nan where a target is undefined."""
    out = np.full(dataset.y.shape, np.nan)
    for column, probe in enumerate(fitted["fits"]):
        if probe["status"] == "fit":
            out[:, column] = probe["fit"].predict(dataset.x)
    return out


def probe_r2(train: ProbeDataset, test: ProbeDataset, *, fit_intercept: bool = True) -> dict:
    """Fit on probe-train and score probe-test; R² per target and the mean."""
    if (train.kind, train.form) != (test.kind, test.form):
        raise ValueError("probe train and test sets must share kind and form")
    base = {"kind": train.kind, "form": train.form, "target_names": train.target_names}
    if train.n_rows == 0 or test.n_rows == 0:
        return dict(base, status="not_evaluable", reason="empty_dataset", r2=None,
                    r2_mean=None, alphas=None)
    fitted = fit_probe(train, fit_intercept=fit_intercept)
    rows = ridge.evaluate(fitted["fits"], test.x, test.y)
    scores = [row["r2"] for row in rows if row["status"] == "evaluated"]
    if not scores:
        return dict(base, status="not_evaluable", reason=rows[0]["reason"], r2=None,
                    r2_mean=None, alphas=None)
    return dict(base, status="evaluated", reason=None, r2=scores,
                r2_mean=float(np.mean(scores)),
                alphas=[row["alpha"] for row in fitted["fits"] if row["status"] == "fit"],
                n_train=train.n_rows, n_test=test.n_rows)


def random_init_floor(arm: str, geometry, *, encoder_config: EncoderConfig | None = None,
                      scaling: FeatureScaling | None = None, seed: int = 0) -> LucidPointJepa:
    """§3.5.6 random-init floor: the same encoder architecture at the arm's first seed."""
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown LUCiD arm {arm!r}")
    torch.manual_seed(int(seed))
    return LucidPointJepa(encoder_config or EncoderConfig(), geometry, scaling)


def squared_correlation(a, b) -> float:
    """Squared Pearson correlation: R11's cross-leakage statistic (nan if degenerate)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError("squared_correlation needs two equally sized vectors")
    a, b = a - a.mean(), b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float("nan") if denominator <= 0.0 else float((a @ b) / denominator) ** 2
