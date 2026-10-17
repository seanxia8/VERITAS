# SPDX-License-Identifier: MIT
"""Runbook v4 §3.5.3a detectability check for the LUCiD photon bombs (WP4b).

Development-only (``readout_synthetic_unrepaired``, §3.5.2).  The WCTE geometry is
read from LUCiD's ``config/WCTE_geom_config.json``/``.npz`` path; no LUCiD import,
nothing redistributed.  True expected charge before threshold, no dark counts or
threshold censoring: ``mu_p = Y QE CE Omega_p/(4pi) exp(-d_p/lambda) gains_p``.
Poisson variance is inflated by ``(1+sigma_spe^2)``; the factor information is the
profiled/effective ``1/(c^T I^+ c)`` with the vertex (3), yield (1) and smooth
co-varying factors profiled.  Per-PMT F3 gains are part of the true ``mu`` but are
not profiled (an incidental-parameter block would zero every other factor).
``R2_max = sigma_f^2/(sigma_f^2+1/I)`` per event and per run (50 events,
``I_run = sum I_event``); profiling ignores the yield/module-gain priors, so the
ceilings are lower bounds.  See ``docs/jepa/WP4B_DETECTABILITY.md``.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from . import lucid_bombs as lb

SCHEMA = "jepa-lucid-detectability-v0"
DECISION_SCHEMA = "jepa-lucid-detectability-decision-v0"
THRESHOLD = 0.6
N_MIN, RUN_EVENTS, MODULE_PMTS = lb.N_MIN, lb.RUN_EVENTS, lb.WCTE_PMTS_PER_MODULE
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "lucid_detectability_v0.json"
DEFAULT_GENERATOR_CONFIG = Path(__file__).resolve().parent / "configs" / "lucid_bombs_v0.json"
NUISANCE_POLICY = {"F1": ("vertex", "yield", "other module gains"),
                   "F3": ("vertex", "yield", "F1 module gains"),
                   "F2": ("vertex", "yield", "F1 module gains"),
                   "F2p": ("vertex", "yield", "F1 module gains"),
                   "T": ("vertex", "yield")}
AMBIGUITIES = (
    "F3 per-PMT gains are part of the true mu but are not profiled: one free gain per PMT "
    "is an incidental-parameter block and would make every other factor's profiled "
    "information exactly zero.",
    "F2/F2' share the F1 module-gain nuisance block (the runbook's explicit joint "
    "profiling), so only their within-module variation survives; conservative.",
    "The across view used for the N_min check is a random module half, not the contiguous "
    "Hilbert half (view_masks is still a WP4c stub).",
    "F1 acceptance uses the main-cell (F2-free) ceiling; the joint F1/F2' row is recorded "
    "beside it for R11.",
    "Profiling is per event, not run-jointly; I_run = sum I_event is the registered "
    "approximation and biases the ceilings down.")
def _sha(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def _median(values) -> float:
    return float(np.median(np.asarray(values, dtype=float)))
@dataclass(frozen=True)
class CheckConfig:
    sigma_spe: float = lb.SIGMA_SPE
    v_dark: float = lb.V_DARK
    run_events: int = RUN_EVENTS
    n_events: int = 50
    seed: int = lb.DEFAULT_SEED
    threshold: float = THRESHOLD
    f1_spreads: tuple = lb.F1_SPREADS
    f3_spread: float = lb.F3_SPREAD
    f3_max_spread: float = 0.10
    f2_spreads: tuple = lb.F2_SPREADS
    f2p_spreads: tuple = lb.F2P_SPREADS
    t_spread: float = lb.T_SPREAD
    t_ratio: float = 0.5
    y_lo: float = lb.Y_LO_PHOTONS
    y_hi_initial: float = lb.Y_HI_PHOTONS
    y_lo_target_hits: float = 200.0
    y_hi_target_module_charge_pe: float = 300.0
    y_hi_target_hit_fraction: float = 0.95
    bound_events: int = 48
    bound_iterations: int = 10
    occupancy_events: int = 64
    n_min: int = N_MIN
    median_policy: str = "population_median_of_log_uniform"
    lambda0_m: float = lb.LAMBDA0_M
    def __post_init__(self):
        object.__setattr__(self, "f1_spreads", tuple(float(x) for x in self.f1_spreads))
        object.__setattr__(self, "f2_spreads", tuple(float(x) for x in self.f2_spreads))
        object.__setattr__(self, "f2p_spreads", tuple(float(x) for x in self.f2p_spreads))
        ok = (self.sigma_spe >= 0 and self.v_dark >= 0 and self.run_events > 0 and self.n_events > 0
              and 0 < self.threshold < 1 and all(s > 0 for s in self.f1_spreads)
              and 0 < self.f3_spread <= self.f3_max_spread
              and all(a > 0 for a in self.f2_spreads) and all(b > 0 for b in self.f2p_spreads)
              and self.t_spread > 0 and 0 <= self.t_ratio <= 1 and 0 < self.y_lo < self.y_hi_initial
              and self.bound_events > 0 and self.bound_iterations > 0 and self.n_min > 0
              and float(self.lambda0_m) == float(lb.LAMBDA0_M)
              and self.median_policy == "population_median_of_log_uniform")
        if not ok:
            raise ValueError("detectability config invalid")
    @property
    def digest(self) -> str:
        return _sha(asdict(self))
def load_check_config(path: Path | str = DEFAULT_CONFIG) -> CheckConfig:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {payload.get('schema')!r} != {SCHEMA!r}")
    block = payload.get("check")
    if not isinstance(block, dict):
        raise ValueError(f"{path}: missing the 'check' block")
    unknown = sorted(set(block) - set(CheckConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"{path}: unknown check keys {unknown}")
    block = dict(block)
    for key in ("f1_spreads", "f2_spreads", "f2p_spreads"):
        block[key] = tuple(block[key])
    return CheckConfig(**block)
def median_yield(y_lo: float, y_hi: float) -> float:
    """Population median of the log-uniform yield law: sqrt(Y_lo Y_hi)."""
    if not (0.0 < float(y_lo) < float(y_hi)):
        raise ValueError("median yield needs 0 < Y_lo < Y_hi")
    return float(math.sqrt(float(y_lo) * float(y_hi)))
@dataclass(frozen=True)
class EventTruth:
    event_id: int
    vertex_m: np.ndarray
    qe_eff: np.ndarray
    yield_photons: float
def sample_events(geometry, gen_cfg, yield_photons: float, n_events: int, seed: int) -> tuple:
    y = float(yield_photons)
    if not np.isfinite(y) or y <= 0.0:
        raise ValueError("fixed yield must be positive and finite")
    out = []
    for i in range(int(n_events)):
        rv = lb.keyed_rng(seed, lb.NS_DRAWS, "detectability/events", i, "vertex")
        rq = lb.keyed_rng(seed, lb.NS_DRAWS, "detectability/events", i, "wavelength")
        out.append(EventTruth(i, lb.fiducial_vertex(geometry, rv, gen_cfg.fiducial_fraction),
                              lb.sample_qe(geometry, rq, gen_cfg), y))
    return tuple(out)
def true_expected_charge(geometry, gen_cfg, state, event: EventTruth) -> np.ndarray:
    mu = lb.expected_charge(geometry, event.vertex_m, event.yield_photons, event.qe_eff,
                            state.attenuation_length_m, gen_cfg)
    return mu * np.asarray(state.gains, dtype=float)
def poisson_weights(mu, sigma_spe: float = lb.SIGMA_SPE, v_dark: float = lb.V_DARK) -> np.ndarray:
    """Fisher weights mu^2/(mu(1+sigma_spe^2)+v_dark) for log-derivatives."""
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 1 or not np.isfinite(mu).all() or np.any(mu < 0.0) or sigma_spe < 0 or v_dark < 0:
        raise ValueError("invalid mu/sigma_spe/v_dark")
    var = mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark)
    return np.divide(mu * mu, var, out=np.zeros_like(mu), where=var > 0)
def vertex_log_derivatives(geometry, vertex_m, lambda_m: float) -> np.ndarray:
    """d ln mu / d vertex for Omega exp(-d/lambda), generator solid angle."""
    delta = np.asarray(geometry.positions_m, float) - np.asarray(vertex_m, float)
    dist = np.linalg.norm(delta, axis=1)
    if np.any(dist <= 0.0):
        raise ValueError("a sensor coincides with the vertex")
    u = delta / dist[:, None]
    normal = np.asarray(geometry.directions, float)
    cos_raw = np.einsum("ij,ij->i", delta, normal)
    q = np.sqrt(dist * dist + float(geometry.sensor_radius_m) ** 2)
    out = ((q + dist) / q ** 2)[:, None] * u
    out -= normal / np.where(np.abs(cos_raw) > 0.0, cos_raw, np.inf)[:, None]
    out += u / dist[:, None] + u / float(lambda_m)
    return out
def fisher_matrix(columns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """I = L^T diag(w) L; columns are log-derivatives."""
    cols, w = np.asarray(columns, float), np.asarray(weights, float)
    if cols.ndim != 2 or w.shape != (cols.shape[0],) or not np.isfinite(cols).all() \
            or not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("columns must be (n_pmt, n_param), weights (n_pmt,), all finite")
    return cols.T @ (w[:, None] * cols)
def profiled_information(matrix: np.ndarray, direction: np.ndarray, *, rcond: float = 1e-10) -> float:
    """Profiled information for a signal direction: 1/(c^T I^+ c)."""
    mat, c = np.asarray(matrix, float), np.asarray(direction, float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or c.shape != (mat.shape[0],) \
            or not np.isfinite(mat).all() or not np.isfinite(c).all():
        raise ValueError("need a finite square matrix and a matching direction")
    value = float(c @ np.linalg.pinv(mat, rcond=rcond) @ c)
    return 0.0 if value <= 0.0 else 1.0 / value
def r2_max(information: float, sigma_f: float) -> float:
    """R2_max = sigma_f^2/(sigma_f^2+1/I); 0 at I=0, 1 at I=inf."""
    if sigma_f <= 0 or not np.isfinite(sigma_f):
        raise ValueError("sigma_f must be positive and finite")
    information = float(information)
    if information < 0 or np.isnan(information):
        raise ValueError("information must be non-negative")
    if information == 0.0:
        return 0.0
    if np.isinf(information):
        return 1.0
    return float(sigma_f ** 2 / (sigma_f ** 2 + 1.0 / information))
def _unit(size: int, index: int) -> np.ndarray:
    out = np.zeros(size, dtype=float)
    out[index] = 1.0
    return out
def _contrast(size: int, indices, position: int) -> np.ndarray:
    indices = list(indices)
    if len(indices) < 2 or position not in indices:
        raise ValueError("contrast needs >=2 indices and the position among them")
    out = np.zeros(size, dtype=float)
    out[position] = 1.0
    for j in indices:
        if j != position:
            out[j] = -1.0 / (len(indices) - 1)
    return out
def _smooth_columns(geometry, event: EventTruth, lambda_m: float, include_module_gains: bool) -> np.ndarray:
    n, m = geometry.n_pmt, geometry.n_modules
    cols = np.zeros((n, 4 + (m if include_module_gains else 0)))
    cols[:, :3] = vertex_log_derivatives(geometry, event.vertex_m, lambda_m)
    cols[:, 3] = 1.0
    if include_module_gains:
        cols[np.arange(n), 4 + geometry.module_index] = 1.0
    return cols
def _event_fisher(geometry, state, event: EventTruth, gen_cfg, check: CheckConfig,
                  include_module_gains: bool) -> tuple:
    mu = true_expected_charge(geometry, gen_cfg, state, event)
    return poisson_weights(mu, check.sigma_spe, check.v_dark), \
        _smooth_columns(geometry, event, state.attenuation_length_m, include_module_gains)
def _info_from_inverse(inverse: np.ndarray, direction: np.ndarray) -> float:
    value = float(direction @ inverse @ direction)
    return 0.0 if value <= 0.0 else 1.0 / value
def _ceiling_row(factor: str, setting: float, law: str, *, n_events: int, sigma_f: float,
                 i_event, i_run, r2_event, r2_run, threshold: float, extra: dict) -> dict:
    i_event, i_run = np.asarray(i_event, float), np.asarray(i_run, float)
    r2_event, r2_run = np.asarray(r2_event, float), np.asarray(r2_run, float)
    return {"factor": factor, "setting": float(setting), "law": law, "sigma_f": float(sigma_f),
            "n_events": int(n_events), "i_event_median": _median(i_event),
            "i_run_median": _median(i_run), "r2_event_median": _median(r2_event),
            "r2_run_median": _median(r2_run), "r2_run_min": float(np.min(r2_run)),
            "r2_run_max": float(np.max(r2_run)), "passes": bool(_median(r2_run) >= float(threshold)),
            "threshold": float(threshold), **extra}
def f1_ceiling(geometry, gen_cfg, check: CheckConfig, events, s: float, *, seed: int) -> dict:
    """F1 module gain; other module gains, vertex and yield profiled."""
    state = lb.state_for_run("F1", s, f"lucid/detectability/F1/{s:g}", geometry, seed=seed)
    m = geometry.n_modules
    matrices = []
    for event in events:
        w, cols = _event_fisher(geometry, state, event, gen_cfg, check, True)
        matrices.append(fisher_matrix(cols, w))
    total = np.sum(matrices, axis=0)
    inv_total = np.linalg.pinv(total, rcond=1e-10)
    i_run = [_info_from_inverse(inv_total, _contrast(4 + m, range(4, 4 + m), 4 + j)) for j in range(m)]
    i_event = []
    for mat in matrices:
        inv = np.linalg.pinv(mat, rcond=1e-10)
        i_event.extend(_info_from_inverse(inv, _contrast(4 + m, range(4, 4 + m), 4 + j)) for j in range(m))
    sigma_c2, fraction = s * s * m / (m - 1.0), 1.0 - 1.0 / m
    return _ceiling_row("F1", s, "F1", n_events=len(events), sigma_f=s, i_event=i_event,
                        i_run=i_run, r2_event=[fraction * r2_max(i, math.sqrt(sigma_c2)) for i in i_event],
                        r2_run=[fraction * r2_max(i, math.sqrt(sigma_c2)) for i in i_run],
                        threshold=check.threshold,
                        extra={"profiled": list(NUISANCE_POLICY["F1"]),
                               "signal": "module-gain zero-sum contrast",
                               "sigma_eff": math.sqrt(sigma_c2), "identifiable_fraction": fraction})
def _f2_matrix(geometry, state, event: EventTruth, gen_cfg, check: CheckConfig, law: str, setting: float):
    w, cols = _event_fisher(geometry, state, event, gen_cfg, check, True)
    extra = np.zeros((geometry.n_pmt, 1))
    if law == "F2":
        dist = np.linalg.norm(geometry.positions_m - event.vertex_m, axis=1)
        extra[:, 0] = dist * float(setting) / state.attenuation_length_m
    else:
        extra[:, 0] = (float(setting) * geometry.z_pmt_m / geometry.half_height_m
                       / np.asarray(state.gradient, float))
    return np.concatenate([cols, extra], axis=1), w
def _scalar_ceiling(geometry, state, events, gen_cfg, check, law: str, setting: float,
                    sigma_f: float, extra: dict, *, with_f1_joint: bool = False):
    matrices, i_event = [], []
    for event in events:
        cols, w = _f2_matrix(geometry, state, event, gen_cfg, check, law, setting)
        mat = fisher_matrix(cols, w)
        matrices.append(mat)
        i_event.append(_info_from_inverse(np.linalg.pinv(mat, rcond=1e-10), _unit(mat.shape[0], -1)))
    total = np.sum(matrices, axis=0)
    i_run = [_info_from_inverse(np.linalg.pinv(total, rcond=1e-10), _unit(total.shape[0], -1))]
    row = _ceiling_row(law, setting, law, n_events=len(events), sigma_f=sigma_f, i_event=i_event,
                       i_run=i_run, r2_event=[r2_max(i, sigma_f) for i in i_event],
                       r2_run=[r2_max(i, sigma_f) for i in i_run], threshold=check.threshold,
                       extra=extra)
    if with_f1_joint:
        inv = np.linalg.pinv(total, rcond=1e-10)
        m = geometry.n_modules
        sigma_c2 = 0.05 ** 2 * m / (m - 1.0)
        row["f1_joint"] = [{"module": j, "r2_run": (1.0 - 1.0 / m) * r2_max(
            _info_from_inverse(inv, _contrast(5 + m, range(4, 4 + m), 4 + j)), math.sqrt(sigma_c2))}
                           for j in range(m)]
    return row
def f2_ceiling(geometry, gen_cfg, check: CheckConfig, events, law: str, setting: float, *, seed: int,
               with_f1_joint: bool = False):
    """F2/F2' N(0,1) variable v; F1 module gains, vertex and yield profiled."""
    law = "F2p" if law in ("F2p", "F2'") else law
    state = lb.state_for_run(law, setting, f"lucid/detectability/{law}/{setting:g}", geometry, seed=seed)
    key = "F2" if law == "F2" else "F2p"
    return _scalar_ceiling(geometry, state, events, gen_cfg, check, law, setting, 1.0,
                           {"profiled": list(NUISANCE_POLICY[key]),
                            "signal": "N(0,1) attenuation/gradient variable v"},
                           with_f1_joint=with_f1_joint)
def f3_ceiling(geometry, gen_cfg, check: CheckConfig, events, s_p: float, *, seed: int) -> dict:
    """F3 within-module per-PMT contrast; F1 module gains, vertex and yield profiled."""
    state = lb.state_for_run("F1", 0.05, f"lucid/detectability/F3/{s_p:g}", geometry, seed=seed)
    m, n = geometry.n_modules, geometry.n_pmt
    n_param = 4 + 2 * m
    matrices = []
    for event in events:
        w, cols = _event_fisher(geometry, state, event, gen_cfg, check, True)
        cols = np.concatenate([cols, np.zeros((n, m))], axis=1)
        for j in range(m):
            inside = np.flatnonzero(geometry.module_index == j)
            cols[inside, 4 + m + j] = -1.0 / (inside.size - 1)
            cols[inside[0], 4 + m + j] = 1.0
        matrices.append(fisher_matrix(cols, w))
    total = np.sum(matrices, axis=0)
    inv_total = np.linalg.pinv(total, rcond=1e-10)
    index = [4 + m + j for j in range(m)]
    i_run = [_info_from_inverse(inv_total, _unit(n_param, j)) for j in index]
    i_event = []
    for mat in matrices:
        inv = np.linalg.pinv(mat, rcond=1e-10)
        i_event.extend(_info_from_inverse(inv, _unit(n_param, j)) for j in index)
    sigma_c2 = s_p * s_p * MODULE_PMTS / (MODULE_PMTS - 1.0)
    fraction = 1.0 - m / n
    return _ceiling_row("F3", s_p, "F3", n_events=len(events), sigma_f=s_p, i_event=i_event,
                        i_run=i_run, r2_event=[fraction * r2_max(i, math.sqrt(sigma_c2)) for i in i_event],
                        r2_run=[fraction * r2_max(i, math.sqrt(sigma_c2)) for i in i_run],
                        threshold=check.threshold,
                        extra={"profiled": list(NUISANCE_POLICY["F3"]),
                               "signal": "within-module per-PMT zero-sum contrast",
                               "sigma_eff": math.sqrt(sigma_c2), "identifiable_fraction": fraction})
def t_ceiling(geometry, gen_cfg, check: CheckConfig, events, *, seed: int) -> dict:
    """Law T for t=r/R at its median 1/2 (in-range uniform prior)."""
    state = lb.state_for_run("T", check.t_ratio, "lucid/detectability/T", geometry, seed=seed)
    k = lb.k_T(geometry, seed)
    matrices, i_event = [], []
    for event in events:
        w, cols = _event_fisher(geometry, state, event, gen_cfg, check, False)
        trend = np.zeros((geometry.n_pmt, 1))
        trend[:, 0] = (check.t_spread * k[geometry.module_index]
                       / np.asarray(state.module_gain, float)[geometry.module_index])
        cols = np.concatenate([cols, trend], axis=1)
        mat = fisher_matrix(cols, w)
        matrices.append(mat)
        i_event.append(_info_from_inverse(np.linalg.pinv(mat, rcond=1e-10), _unit(mat.shape[0], -1)))
    total = np.sum(matrices, axis=0)
    i_run = [_info_from_inverse(np.linalg.pinv(total, rcond=1e-10), _unit(total.shape[0], -1))]
    sigma_f = math.sqrt(1.0 / 12.0)
    return _ceiling_row("T", check.t_ratio, "T", n_events=len(events), sigma_f=sigma_f,
                        i_event=i_event, i_run=i_run, r2_event=[r2_max(i, sigma_f) for i in i_event],
                        r2_run=[r2_max(i, sigma_f) for i in i_run], threshold=check.threshold,
                        extra={"profiled": list(NUISANCE_POLICY["T"]),
                               "signal": "in-range age t=r/R, median 1/2", "sigma_f_sq": 1.0 / 12.0})
@dataclass(frozen=True)
class CeilingTable:
    median_yield: float
    f1: dict
    f3: dict
    f2: dict
    t: dict
    f2_law: str
    f2_setting: float
    n_events: int
    def rows(self) -> list:
        return [self.f1[s] for s in sorted(self.f1)] + [self.f3, self.f2, self.t]
    def short_labels(self, threshold: float) -> list:
        labels = [f"F1 s={s:g}" for s in sorted(self.f1) if self.f1[s]["r2_run_median"] < threshold]
        if self.f3["r2_run_median"] < threshold:
            labels.append(f"F3 s_p={self.f3['setting']:g}")
        if self.f2["r2_run_median"] < threshold:
            labels.append(f"{self.f2_law} {self.f2_setting:g}")
        if self.t["r2_run_median"] < threshold:
            labels.append("T")
        return labels
def evaluate_table(geometry, gen_cfg, check: CheckConfig, yield_photons: float, *, f2_law: str = "F2",
                   f2_setting: float | None = None, f3_spread: float | None = None,
                   seed: int | None = None, events=None) -> CeilingTable:
    seed = int(check.seed if seed is None else seed)
    f2_setting = float(check.f2_spreads[0] if f2_setting is None else f2_setting)
    f3_spread = float(check.f3_spread if f3_spread is None else f3_spread)
    if events is None:
        events = sample_events(geometry, gen_cfg, yield_photons, check.n_events, seed)
    f1 = {s: f1_ceiling(geometry, gen_cfg, check, events, s, seed=seed) for s in check.f1_spreads}
    f3 = f3_ceiling(geometry, gen_cfg, check, events, f3_spread, seed=seed)
    if f2_law in ("F2p", "F2'"):
        f2 = f2_ceiling(geometry, gen_cfg, check, events, "F2p", f2_setting, seed=seed, with_f1_joint=True)
        f2_law = "F2p"
    elif f2_law == "F2":
        f2 = f2_ceiling(geometry, gen_cfg, check, events, "F2", f2_setting, seed=seed)
    else:
        raise ValueError(f"unknown F2 law {f2_law!r}")
    t = t_ceiling(geometry, gen_cfg, check, events, seed=seed)
    return CeilingTable(float(yield_photons), f1, f3, f2, t, f2_law, f2_setting, len(events))
def _branch_reason(f1_eligible, f2_ok, f3_ok) -> str:
    if not f1_eligible and not f2_ok:
        return "no F1 cell and no F2/F2' setting reach 0.6: G6 for R6/R7/R11/R11b"
    if not f1_eligible:
        return "no F1 cell reaches 0.6: G6 for R6/R7"
    if not f2_ok:
        return "F2 and F2' cannot reach 0.6: G6 for R11/R11b"
    if not f3_ok:
        return "F3 cannot reach 0.6: R12 not_evaluable, no gate"
    return "all registered factors reach 0.6 at the frozen yield range"
def apply_fallbacks(table_fn: Callable[[float, str, float, float], CeilingTable], check: CheckConfig,
                    *, y_lo: float, y_hi_initial: float, y_hi_bound: float) -> tuple:
    """§3.5.3a acceptance/fallback rules verbatim; returns (table, steps, branches)."""
    steps, f2_attempted = [], []
    y_hi, f2_law, f2_setting, f3_spread = (float(y_hi_initial), "F2", float(check.f2_spreads[0]),
                                           float(check.f3_spread))
    table = table_fn(y_hi, f2_law, f2_setting, f3_spread)
    if table.short_labels(check.threshold):
        target = float(y_hi_bound)
        steps.append({"step": 1, "action": "raise_y_hi", "from": y_hi, "to": target,
                      "changed": abs(target - y_hi) > 1e-9 * max(abs(target), abs(y_hi), 1.0),
                      "reason": "factor(s) below the 0.6 run ceiling",
                      "short_before": table.short_labels(check.threshold)})
        y_hi = target
        table = table_fn(y_hi, f2_law, f2_setting, f3_spread)
    if table.f2["r2_run_median"] < check.threshold:
        f2_attempted.append(f2_setting)
        for a in check.f2_spreads[1:]:
            steps.append({"step": 2, "action": "set_F2_a", "law": "F2", "from": f2_setting,
                          "to": float(a), "reason": "F2 below the 0.6 run ceiling"})
            f2_law, f2_setting = "F2", float(a)
            f2_attempted.append(f2_setting)
            table = table_fn(y_hi, f2_law, f2_setting, f3_spread)
            if table.f2["r2_run_median"] >= check.threshold:
                break
        if table.f2["r2_run_median"] < check.threshold:
            for b in check.f2p_spreads:
                steps.append({"step": 2, "action": "switch_F2p_b", "law": "F2p",
                              "from": f2_setting, "to": float(b),
                              "reason": "F2 exhausted; switch to the F2' fallback"})
                f2_law, f2_setting = "F2p", float(b)
                f2_attempted.append(f2_setting)
                table = table_fn(y_hi, f2_law, f2_setting, f3_spread)
                if table.f2["r2_run_median"] >= check.threshold:
                    break
    if table.f3["r2_run_median"] < check.threshold and f3_spread < check.f3_max_spread:
        steps.append({"step": 3, "action": "raise_F3_spread", "from": f3_spread,
                      "to": float(check.f3_max_spread), "reason": "F3 below the 0.6 run ceiling"})
        f3_spread = float(check.f3_max_spread)
        table = table_fn(y_hi, f2_law, f2_setting, f3_spread)
    table.f2["attempted_settings"] = f2_attempted or [f2_setting]
    f1_eligible = sorted(s for s in table.f1 if table.f1[s]["r2_run_median"] >= check.threshold)
    f2_ok, f3_ok = bool(table.f2["r2_run_median"] >= check.threshold), bool(table.f3["r2_run_median"] >= check.threshold)
    g6 = []
    if not f1_eligible:
        g6 += ["R6", "R7"]
    if not f2_ok:
        g6 += ["R11"]
    if not f1_eligible or not f2_ok:
        g6 += ["R11b"]
    branches = {"f1_eligible_settings": f1_eligible,
                "f1_excluded_settings": sorted(s for s in table.f1 if s not in f1_eligible),
                "f1_all_below_threshold": not f1_eligible, "f2_ok": f2_ok,
                "f2_choice": ({"law": table.f2_law, "setting": table.f2["setting"]} if f2_ok else None),
                "f2_attempted": table.f2["attempted_settings"], "f3_ok": f3_ok,
                "f3_choice": ({"spread": table.f3["setting"]} if f3_ok else None),
                "f3_attempted": sorted({check.f3_spread, table.f3["setting"]}),
                "r12_status": "evaluable" if f3_ok else "not_evaluable", "r12_gate": None,
                "law_T_ok": bool(table.t["r2_run_median"] >= check.threshold), "g6": bool(g6),
                "g6_dependent_predictions": sorted(set(g6)),
                "r6_status": "evaluable" if f1_eligible else "g6",
                "r7_status": ("evaluable" if 0.10 in f1_eligible
                              else ("g6" if not f1_eligible else "not_evaluable_at_s010")),
                "reason": _branch_reason(f1_eligible, f2_ok, f3_ok)}
    return table, steps, branches
# --------------------------------------------------------------------------- yields / occupancy
def _fixed_yield_draws(geometry, gen_cfg, y: float, event_id: int, seed: int, capacity_y: float):
    cfg_cap = replace(gen_cfg, y_lo=capacity_y * (1.0 - 1e-12), y_hi=capacity_y)
    return replace(lb.draw_event(geometry, cfg_cap, "detectability/occupancy", event_id, seed=seed),
                   yield_photons=float(y))
def occupancy_stats(geometry, gen_cfg, y: float, n_events: int, seed: int, capacity_y: float) -> dict:
    """Median hit count, mean-module charge and hit fraction at a fixed yield."""
    state, readout = lb.nominal_state(geometry, "detectability/occupancy"), lb.make_readout(geometry, gen_cfg)
    hits_n, module_charge, fractions = [], [], []
    for i in range(int(n_events)):
        hits = readout(state, _fixed_yield_draws(geometry, gen_cfg, y, i, seed, capacity_y))
        hits_n.append(len(hits))
        fractions.append(len(hits) / geometry.n_pmt)
        module_charge.append(float(np.bincount(geometry.module_index[np.asarray(hits.pmt_id, int)],
                                               weights=hits.charge_pe, minlength=geometry.n_modules).mean()))
    return {"yield_photons": float(y), "n_events": int(n_events), "median_hit_pmts": _median(hits_n),
            "median_module_charge_pe": _median(module_charge), "median_hit_fraction": _median(fractions)}
def _bisect_log(objective, target: float, lo: float, hi: float, iterations: int) -> float:
    lo, hi = float(lo), float(hi)
    for _ in range(int(iterations)):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if objective(mid) < target else (lo, mid)
    return math.sqrt(lo * hi)
def compute_yield_range(geometry, gen_cfg, check: CheckConfig, *, seed: int | None = None) -> dict:
    """Y_lo at ~200 median hit PMTs; Y_hi at the lower of 300 PE / 0.95 hit fraction."""
    seed = int(check.seed if seed is None else seed)
    capacity_y = 4.0 * check.y_hi_initial
    stats = lambda y: occupancy_stats(geometry, gen_cfg, y, check.bound_events, seed, capacity_y)
    y_lo = _bisect_log(lambda y: stats(y)["median_hit_pmts"], check.y_lo_target_hits,
                       check.y_lo, check.y_hi_initial, check.bound_iterations)
    upper = max(check.y_hi_initial, 2.0 * y_lo)
    for _ in range(20):
        s = stats(upper)
        if s["median_module_charge_pe"] >= check.y_hi_target_module_charge_pe and \
                s["median_hit_fraction"] >= check.y_hi_target_hit_fraction:
            break
        upper *= 2.0
    y_pe = _bisect_log(lambda y: stats(y)["median_module_charge_pe"], check.y_hi_target_module_charge_pe,
                       y_lo, upper, check.bound_iterations)
    y_hf = _bisect_log(lambda y: stats(y)["median_hit_fraction"], check.y_hi_target_hit_fraction,
                       y_lo, upper, check.bound_iterations)
    return {"y_lo": float(y_lo), "y_hi": float(min(y_pe, y_hf)),
            "y_hi_bound_source": "module_charge_300PE" if y_pe <= y_hf else "hit_fraction_0.95",
            "y_lo_check": stats(y_lo), "y_hi_check": stats(min(y_pe, y_hf)),
            "y_at_module_charge_300pe": float(y_pe), "y_at_hit_fraction_0.95": float(y_hf),
            "bound_events": int(check.bound_events), "bound_iterations": int(check.bound_iterations),
            "targets": {"median_hit_pmts": check.y_lo_target_hits,
                        "module_charge_pe": check.y_hi_target_module_charge_pe,
                        "hit_fraction": check.y_hi_target_hit_fraction}}
def occupancy_n_min(geometry, gen_cfg, check: CheckConfig, y_lo: float, *, seed: int | None = None) -> dict:
    """§3.5.4 N_min check; across uses a random module half (view_masks is a WP4c stub)."""
    seed = int(check.seed if seed is None else seed)
    state, readout = lb.nominal_state(geometry, "detectability/nmin"), lb.make_readout(geometry, gen_cfg)
    rng, capacity_y = lb.keyed_rng(seed, "detectability/nmin/half"), 4.0 * check.y_hi_initial
    within, across = [], []
    for i in range(int(check.occupancy_events)):
        hits = readout(state, _fixed_yield_draws(geometry, gen_cfg, y_lo, i, seed, capacity_y))
        pmt = np.asarray(hits.pmt_id, int)
        within.append(int(np.count_nonzero(geometry.pmt_id_in_mpmt[pmt] % 2 == 1)))
        half = rng.permutation(geometry.n_modules)[: geometry.n_modules // 2]
        across.append(int(np.count_nonzero(np.isin(geometry.module_index[pmt], half))))
    within, across = np.asarray(within), np.asarray(across)
    p_within, p_across = float(np.mean(within < check.n_min)), float(np.mean(across < check.n_min))
    return {"n_min": int(check.n_min), "n_events": int(check.occupancy_events),
            "within_target_median_hits": _median(within), "within_target_min_hits": int(within.min()),
            "within_exclusion_rate": p_within, "across_target_median_hits": _median(across),
            "across_target_min_hits": int(across.min()), "across_initial_below_nmin_rate": p_across,
            "across_exclusion_rate_after_3_redraws_estimate": p_across ** 4,
            "note": "across split is a random module half (proxy); view_masks is a WP4c stub"}
# --------------------------------------------------------------------------- payload / CLI
def _table_rows(table: CeilingTable) -> list:
    keep = ("factor", "setting", "law", "sigma_f", "n_events", "i_event_median", "i_run_median",
            "r2_event_median", "r2_run_median", "r2_run_min", "r2_run_max", "passes", "threshold",
            "profiled", "signal", "f1_joint")
    return [{k: row[k] for k in keep if k in row} for row in table.rows()]
def _decisions_md(payload: dict) -> str:
    geo, yields, occ, br = payload["geometry"], payload["yields"], payload["occupancy"], payload["branches"]
    lines = [f"### §3.5.3a detectability check — {payload['status']} ({geo['name']})", "",
             f"- Geometry: `{geo['name']}`, digest `{geo['geometry_sha256']}`, "
             f"synthetic={str(geo['synthetic']).lower()}; readout `{payload['readout_label']}`; "
             f"code `{payload['code_sha']}`.",
             f"- Yield range: `Y_lo = {yields['y_lo']:.6g}`, `Y_hi = {yields['y_hi']:.6g}` photons "
             f"({yields['y_hi_bound_source']}); median yield `{yields['median_yield']:.6g}` "
             f"({yields['median_policy']}).",
             f"- N_min = {occ['n_min']} at Y_lo: within target median {occ['within_target_median_hits']:.0f} "
             f"hits (exclusion {occ['within_exclusion_rate']:.3g}); across proxy median "
             f"{occ['across_target_median_hits']:.0f} hits (below-N_min "
             f"{occ['across_initial_below_nmin_rate']:.3g}).", "",
             "| factor | setting | R²_run (lower bound) | pass |", "|---|---|---|---|"]
    lines += [f"| {row['factor']} | {row['setting']:.3g} | {row['r2_run_median']:.3f} | "
              f"{str(row['passes']).lower()} |" for row in payload["table"]]
    lines += ["", f"- Fallback steps: {len(payload['fallback_steps'])} "
              f"({', '.join(s['action'] for s in payload['fallback_steps']) or 'none'}).",
              f"- F1 eligible: {br['f1_eligible_settings']}; excluded: {br['f1_excluded_settings']}.",
              f"- F2 choice: {br['f2_choice']} (attempted {br['f2_attempted']}); "
              f"F3 choice: {br['f3_choice']}.",
              f"- R12: {br['r12_status']} (no gate); law T >= 0.6: {str(br['law_T_ok']).lower()}.",
              f"- G6: {str(br['g6']).lower()} for {br['g6_dependent_predictions']}.",
              f"- One line: {br['reason']}."]
    return "\n".join(lines)
def _freeze(payload: dict) -> dict:
    out = dict(payload)
    out.pop("content_sha256", None)
    out["content_sha256"] = _sha(out)
    return out
def build_decision_payload(geometry, gen_cfg, check: CheckConfig, table: CeilingTable, steps, branches,
                           yields, occupancy, *, seed: int | None = None, code_sha: str | None = None,
                           frozen: bool = True) -> dict:
    seed = int(check.seed if seed is None else seed)
    payload = {
        "schema": DECISION_SCHEMA, "status": "frozen" if frozen else "development_only",
        "readout_label": lb.READOUT_LABEL, "reported_results": False, "frozen": bool(frozen),
        "code_sha": code_sha or lb.current_code_sha(), "seed": seed,
        "config_digest": check.digest, "generator_config_digest": gen_cfg.digest,
        "sigma_spe": check.sigma_spe, "v_dark": check.v_dark, "run_events": check.run_events,
        "n_events": check.n_events, "geometry": {**geometry.summary(), "geometry_sha256": geometry.digest},
        "yields": {**yields, "median_yield": table.median_yield, "median_policy": check.median_policy},
        "occupancy": occupancy,
        "median_yield_policy": {"policy": check.median_policy, "value_photons": table.median_yield},
        "nuisance_policy": {k: list(v) for k, v in NUISANCE_POLICY.items()},
        "table": _table_rows(table), "fallback_steps": list(steps), "branches": dict(branches),
        "rebenchmark": {"y_lo": float(yields["y_lo"]), "y_hi": float(yields["y_hi"]),
                        "median_yield": table.median_yield, "volume_events": lb.REGISTERED_VOLUME_EVENTS,
                        "volume_breakdown": dict(lb.REGISTERED_VOLUME_BREAKDOWN),
                        "cores_at_most": lb.BENCHMARK_CORES, "cpu_cap_wall_hours": lb.CPU_CAP_WALL_HOURS,
                        "note": "repeat benchmark_generation at the frozen yields"},
        "pending": ["real-geometry numbers", "merged PMT readout re-run (§3.5.2)"],
        "ambiguities": AMBIGUITIES,
    }
    payload["decisions_md"] = _decisions_md(payload)
    return _freeze(payload)
def validate_decision(payload: Mapping) -> None:
    if payload.get("schema") != DECISION_SCHEMA:
        raise ValueError(f"decision schema {payload.get('schema')!r} != {DECISION_SCHEMA!r}")
    required = ("status", "readout_label", "reported_results", "frozen", "geometry", "yields",
                "occupancy", "table", "fallback_steps", "branches", "rebenchmark", "content_sha256",
                "decisions_md")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"decision payload missing {missing}")
    if payload["reported_results"] is not False or not payload["table"]:
        raise ValueError("invalid development-only decision payload")
    if _freeze({k: v for k, v in payload.items() if k != "content_sha256"})["content_sha256"] \
            != payload["content_sha256"]:
        raise ValueError("decision content_sha256 mismatch")
def write_decision(path: Path | str, payload: dict) -> Path:
    validate_decision(payload)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return Path(path)
def read_decision(path: Path | str) -> dict:
    payload = json.loads(Path(path).read_text())
    validate_decision(payload)
    return payload
def run_check(geometry, gen_cfg, check: CheckConfig, *, seed: int | None = None, y_lo: float | None = None,
              y_hi: float | None = None, y_hi_bound: float | None = None, frozen: bool = True,
              code_sha: str | None = None):
    seed = int(check.seed if seed is None else seed)
    if y_lo is not None and y_hi is not None:
        bounds = {"y_lo": float(y_lo), "y_hi": float(y_hi), "y_hi_bound_source": "provided"}
    else:
        bounds = compute_yield_range(geometry, gen_cfg, check, seed=seed)
    y_lo = float(bounds["y_lo"] if y_lo is None else y_lo)
    if y_hi is not None:
        y_hi_bound = float(y_hi)
    else:
        y_hi_bound = float(bounds["y_hi"] if y_hi_bound is None else y_hi_bound)
    occ = occupancy_n_min(geometry, gen_cfg, check, y_lo, seed=seed)
    cache = {}
    def table_fn(y_hi_value, f2_law, f2_setting, f3_spread):
        y_med = median_yield(y_lo, y_hi_value)
        key = (round(y_med, 9), f2_law, round(float(f2_setting), 12), round(float(f3_spread), 12))
        if key not in cache:
            events = sample_events(geometry, gen_cfg, y_med, check.n_events, seed)
            cache[key] = evaluate_table(geometry, gen_cfg, check, y_med, f2_law=f2_law,
                                        f2_setting=f2_setting, f3_spread=f3_spread, seed=seed,
                                        events=events)
        return cache[key]
    table, steps, branches = apply_fallbacks(table_fn, check, y_lo=y_lo,
                                             y_hi_initial=check.y_hi_initial, y_hi_bound=y_hi_bound)
    yields = {"y_lo": y_lo, "y_hi": y_hi_bound, "y_hi_initial": float(check.y_hi_initial),
              "y_hi_bound": y_hi_bound, "y_hi_bound_source": bounds.get("y_hi_bound_source"),
              "y_hi_bounds": bounds, "median_yield": table.median_yield,
              "median_policy": check.median_policy}
    return build_decision_payload(geometry, gen_cfg, check, table, steps, branches, yields, occ,
                                  seed=seed, code_sha=code_sha, frozen=frozen)
def main(argv=None) -> int:  # pragma: no cover - host CLI
    parser = argparse.ArgumentParser(prog="python -m jepa_noise.lucid_detectability")
    parser.add_argument("command", choices=("run",), nargs="?", default="run")
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--generator-config", default=str(DEFAULT_GENERATOR_CONFIG))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--y-lo", type=float, default=None)
    parser.add_argument("--y-hi", type=float, default=None)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--no-freeze", action="store_true")
    args = parser.parse_args(argv)
    check, gen_cfg = load_check_config(args.config), lb.load_generator_config(args.generator_config)
    geometry = lb.load_wcte_geometry(args.geometry)
    if geometry.synthetic and not args.allow_synthetic:
        raise SystemExit("refusing a synthetic geometry without --allow-synthetic")
    frozen = (not args.no_freeze) and (not geometry.synthetic)
    payload = run_check(geometry, gen_cfg, check, seed=args.seed, y_lo=args.y_lo, y_hi=args.y_hi,
                        frozen=frozen)
    write_decision(args.out, payload)
    print(payload["decisions_md"])
    print(json.dumps({k: payload[k] for k in ("status", "frozen", "content_sha256")}, sort_keys=True))
    return 0
if __name__ == "__main__":  # pragma: no cover - host CLI
    raise SystemExit(main())
