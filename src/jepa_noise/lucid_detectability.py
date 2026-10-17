# SPDX-License-Identifier: MIT
"""Runbook v4 §3.5.3a detectability check for the LUCiD photon bombs (WP4b).

Development-only (``readout_dark_off``, §3.5.2).  The WCTE geometry is read from
LUCiD's ``config/WCTE_geom_config.json``/``.npz`` path; no LUCiD import, nothing
redistributed.  True expected charge before threshold, no threshold censoring:
``mu_p = Y QE CE Omega_p/(4pi) exp(-d_p/lambda) gains_p`` **times the fixed parity
charge gain** of the merged PMT reference response (odd dense module index x1.03).
Dark counts are off (``v_dark = 0``): the anchor's dark window is not physical
until the PMT R-6 SHA lands.  F3 is treated as random, not known: each PMT's
variance is ``mu_p (1 + sigma_spe^2) + v_dark + s_p^2 mu_p^2`` (the §3.5.6
``mae_chi2`` covariance term), so a zero-mean PMT has zero variance and carries no
information (skipped, no NaN).  The factor information is the profiled/effective
``1/(c^T I^+ c)`` with the vertex (3), yield (1) and smooth co-varying factors
profiled.  ``R2_max = sigma_f^2/(sigma_f^2+1/I)`` per event and per run (50 events,
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
#: The single per-PMT variance formula (F3 marginalised as a random gain, §3.5.6).
VARIANCE_FORMULA = lb.VARIANCE_FORMULA
READOUT_LABEL = lb.READOUT_LABEL
THRESHOLD = 0.6
N_MIN, RUN_EVENTS, MODULE_PMTS = lb.N_MIN, lb.RUN_EVENTS, lb.WCTE_PMTS_PER_MODULE
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "lucid_detectability_v0.json"
DEFAULT_GENERATOR_CONFIG = Path(__file__).resolve().parent / "configs" / "lucid_bombs_v0.json"
#: The previously frozen ceilings (2026-10-17, pre-merge readout): read-only input.
PREVIOUS_DECISION_PATH = (Path(lb.__file__).resolve().parents[2] / "results" / "jepa"
                          / "20261017_wp4b_detectability" / "wp4b_detectability_wcte_v2.json")
NUISANCE_POLICY = {"F1": ("vertex", "yield", "other module gains"),
                   "F3": ("vertex", "yield", "F1 module gains"),
                   "F2": ("vertex", "yield", "F1 module gains"),
                   "F2p": ("vertex", "yield", "F1 module gains"),
                   "T": ("vertex", "yield")}
AMBIGUITIES = (
    "F3 per-PMT gains are marginalised as a random gain (s_p^2 mu_p^2 added to every "
    "PMT variance) rather than profiled: one free gain per PMT is an incidental-parameter "
    "block that would make every other factor's profiled information exactly zero. s_p is "
    "the spread in force (lb.F3_SPREAD = 0.10), not the candidate spread of an F3 ceiling "
    "scan, because that is the F3 actually present in the simulated data.",
    "F2/F2' share the F1 module-gain nuisance block (the runbook's explicit joint "
    "profiling), so only their within-module variation survives; conservative.",
    "The across view used for the N_min check is a random module half, not the contiguous "
    "Hilbert half (view_masks is still a WP4c stub).",
    "F1 acceptance uses the main-cell (F2-free) ceiling; the joint F1/F2' row is recorded "
    "beside it for R11.",
    "Profiling is per event, not run-jointly; I_run = sum I_event is the registered "
    "approximation and biases the ceilings down.",
    "Dark counts are off for readout_dark_off (v_dark = 0): the anchor's dark-window draw "
    "and the private dark noise are not physical until the PMT R-6 SHA lands, so no "
    "dark-bank information enters any ceiling.",
    "The fixed parity charge gain is detector response: it is part of mu here and in the "
    "nominal twin, so the twin cancels only the injected F1/F3 (and F2/F2'/R11b/T) gains.",
)
def _sha(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def _sha_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _median(values) -> float:
    return float(np.median(np.asarray(values, dtype=float)))
@dataclass(frozen=True)
class CheckConfig:
    sigma_spe: float = lb.SIGMA_SPE
    #: readout_dark_off: no dark counts are drawn, so v_dark = 0 (PMT R-6 SHA pending).
    v_dark: float = lb.V_DARK_READOUT
    #: F3 is marginalised at the spread in force, not at an F3 ceiling candidate.
    f3_marginalised_spread: float = lb.F3_SPREAD
    run_events: int = RUN_EVENTS
    n_events: int = 50
    seed: int = lb.DEFAULT_SEED
    threshold: float = THRESHOLD
    f1_spreads: tuple = lb.F1_SPREADS
    f3_spread: float = lb.F3_SPREAD_DEFAULT
    f3_max_spread: float = 0.10
    f2_spreads: tuple = lb.F2_SPREADS
    f2p_spreads: tuple = lb.F2P_SPREADS
    t_spread: float = lb.T_SPREAD
    t_ratio: float = 0.5
    y_lo: float = lb.Y_LO_PHOTONS_PLACEHOLDER
    y_hi_initial: float = lb.Y_HI_PHOTONS_PLACEHOLDER
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
              and 0 <= self.f3_marginalised_spread <= self.f3_max_spread
              and all(a > 0 for a in self.f2_spreads) and all(b > 0 for b in self.f2p_spreads)
              and self.t_spread > 0 and 0 <= self.t_ratio <= 1 and 0 < self.y_lo < self.y_hi_initial
              and self.bound_events > 0 and self.bound_iterations > 0 and self.n_min > 0
              and float(self.lambda0_m) == float(lb.LAMBDA0_M)
              and float(self.v_dark) == float(lb.V_DARK_READOUT)
              and self.median_policy == "population_median_of_log_uniform")
        if not ok:
            raise ValueError(
                f"detectability config invalid (readout {READOUT_LABEL} needs v_dark = "
                f"{lb.V_DARK_READOUT}, 0 <= f3_marginalised_spread <= {self.f3_max_spread})")
    @property
    def digest(self) -> str:
        return _sha(asdict(self))
def load_check_config(path: Path | str = DEFAULT_CONFIG) -> CheckConfig:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {payload.get('schema')!r} != {SCHEMA!r}")
    if payload.get("readout_label", READOUT_LABEL) != READOUT_LABEL:
        raise ValueError(f"{path}: readout_label {payload.get('readout_label')!r} != "
                         f"{READOUT_LABEL!r} (the merged readout is dark-off)")
    if payload.get("dark_counts", lb.DARK_COUNTS_OFF) != lb.DARK_COUNTS_OFF:
        raise ValueError(f"{path}: dark_counts must be {lb.DARK_COUNTS_OFF!r}")
    if payload.get("f3_marginalised") is not True:
        raise ValueError(f"{path}: f3_marginalised must be true")
    if payload.get("variance_formula", VARIANCE_FORMULA) != VARIANCE_FORMULA:
        raise ValueError(f"{path}: variance_formula must be {VARIANCE_FORMULA!r}")
    block = payload.get("check")
    if not isinstance(block, dict):
        raise ValueError(f"{path}: missing the 'check' block")
    unknown = sorted(set(block) - set(CheckConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"{path}: unknown check keys {unknown}")
    block = dict(block)
    for key in ("f1_spreads", "f2_spreads", "f2p_spreads"):
        block[key] = tuple(block[key])
    fixed = payload.get("fixed_response")
    if fixed is not None and fixed.get("sha256") != lb.reference_response(
            fixed.get("path")).sha256:
        raise ValueError(f"{path}: fixed_response.sha256 does not match the manifest")
    previous = payload.get("previous_decision")
    if previous is not None and previous.get("path"):
        recorded = Path(previous["path"])
        if not recorded.is_absolute():
            recorded = Path(lb.__file__).resolve().parents[2] / recorded
        if recorded.resolve() != PREVIOUS_DECISION_PATH.resolve():
            raise ValueError(f"{path}: previous_decision.path does not name the frozen "
                             "ceiling file the check records")
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
def true_expected_charge(geometry, gen_cfg, state, event: EventTruth, *,
                         reference=None) -> np.ndarray:
    """``mu_p`` with the fixed parity charge gain and the injected (drifted) gains."""
    ref = lb.generator_reference(gen_cfg) if reference is None else reference
    mu = lb.expected_charge(geometry, event.vertex_m, event.yield_photons, event.qe_eff,
                            state.attenuation_length_m, gen_cfg)
    return mu * lb.fixed_charge_gain(geometry, ref) * np.asarray(state.gains, dtype=float)
def per_pmt_variance(mu, sigma_spe: float = lb.SIGMA_SPE, v_dark: float = lb.V_DARK_READOUT,
                     s_p: float = lb.F3_SPREAD) -> np.ndarray:
    """``mu_p (1 + sigma_spe^2) + v_dark + s_p^2 mu_p^2`` (F3 marginalised, §3.5.6)."""
    mu = np.asarray(mu, dtype=float)
    if mu.ndim != 1 or not np.isfinite(mu).all() or np.any(mu < 0.0) \
            or sigma_spe < 0 or v_dark < 0 or s_p < 0:
        raise ValueError("invalid mu/sigma_spe/v_dark/s_p")
    return mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark) + float(s_p) ** 2 * mu ** 2
def poisson_weights(mu, sigma_spe: float = lb.SIGMA_SPE, v_dark: float = lb.V_DARK_READOUT,
                    s_p: float = lb.F3_SPREAD) -> np.ndarray:
    """Fisher weights ``mu^2/(mu(1+sigma_spe^2)+v_dark+s_p^2 mu^2)`` for log-derivatives.

    A zero-mean PMT has zero variance when ``v_dark = 0`` and carries no information,
    so exactly-zero-``mu`` PMTs are skipped (weight 0, no NaN, no division by zero).
    """
    mu = np.asarray(mu, dtype=float)
    var = per_pmt_variance(mu, sigma_spe, v_dark, s_p)
    weights = np.divide(mu * mu, var, out=np.zeros_like(mu), where=(var > 0.0) & (mu > 0.0))
    return weights
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
    return poisson_weights(mu, check.sigma_spe, check.v_dark,
                           check.f3_marginalised_spread), \
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
        # v4 §3.5.3a profiles "the other factors"; the F1/attenuation joint
        # ceiling must therefore be recorded on the F2 branch as well as on F2′.
        f2 = f2_ceiling(geometry, gen_cfg, check, events, "F2", f2_setting,
                        seed=seed, with_f1_joint=True)
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
def _setting_key(value) -> float:
    return round(float(value), 9)
def previous_ceilings(path: Path | str | None = None) -> dict:
    """The previously frozen ceilings, recorded read-only beside the new ones.

    Path: ``results/jepa/20261017_wp4b_detectability/wp4b_detectability_wcte_v2.json``
    by default.  ``"omitted": true`` with a reason when it is absent, so the Lead can
    always tell "no comparison" from "no change".
    """
    target = Path(PREVIOUS_DECISION_PATH if path is None else path)
    if not target.is_file():
        return {"omitted": True, "path": str(target),
                "reason": "previous frozen decision JSON absent; no side-by-side ceilings"}
    payload = json.loads(target.read_text())
    yields, branches = payload.get("yields", {}), payload.get("branches", {})
    rows = [{k: row.get(k) for k in ("factor", "setting", "law", "sigma_f",
                                     "r2_run_median", "r2_run_min", "r2_run_max", "passes")}
            for row in payload.get("table", [])]
    return {"omitted": False, "path": str(target), "sha256": _sha_file(target),
            "schema": payload.get("schema"), "status": payload.get("status"),
            "recorded_as": lb.SUPERSEDED_PENDING_MERGED_READOUT,
            "content_sha256": payload.get("content_sha256"),
            "readout_label": payload.get("readout_label"),
            "sigma_spe": payload.get("sigma_spe"), "v_dark": payload.get("v_dark"),
            "y_lo": yields.get("y_lo"), "y_hi": yields.get("y_hi"),
            "median_yield": yields.get("median_yield"),
            "f2_choice": branches.get("f2_choice"), "f3_choice": branches.get("f3_choice"),
            "rows": rows}
def ceiling_comparison(table: CeilingTable, previous: dict) -> list:
    """Pair each new ceiling with the previously frozen one on (factor, setting)."""
    if previous.get("omitted"):
        return []
    old = {(row.get("factor"), _setting_key(row.get("setting"))): row
           for row in previous.get("rows", [])}
    out = []
    for row in _table_rows(table):
        before = old.get((row["factor"], _setting_key(row["setting"])))
        out.append({"factor": row["factor"], "law": row["law"], "setting": row["setting"],
                    "previous_r2_run_median": None if before is None else before["r2_run_median"],
                    "new_r2_run_median": row["r2_run_median"],
                    "delta_r2_run_median": (None if before is None else
                                            row["r2_run_median"] - before["r2_run_median"]),
                    "previous_passes": None if before is None else before["passes"],
                    "new_passes": row["passes"], "matched": before is not None,
                    "previous_readout_label": previous.get("readout_label")})
    return out
def _decisions_md(payload: dict) -> str:
    geo, yields, occ, br = payload["geometry"], payload["yields"], payload["occupancy"], payload["branches"]
    fixed = payload["fixed_response"]
    lines = [f"### §3.5.3a detectability check — {payload['status']} ({geo['name']})", "",
             f"- Geometry: `{geo['name']}`, digest `{geo['geometry_sha256']}`, "
             f"synthetic={str(geo['synthetic']).lower()}; readout `{payload['readout_label']}`; "
             f"code `{payload['code_sha']}`.",
             f"- Response: `{fixed['relpath']}` sha256 `{fixed['sha256']}` "
             f"(charge gain {fixed['gain_rule']}, timing width {fixed['timing_width_factor_rule']}); "
             f"dark counts {payload['dark_counts']} (v_dark = {payload['v_dark']:g}).",
             f"- F3 marginalised: {str(payload['f3_marginalised']).lower()} at s_p = "
             f"{payload['f3_marginalised_spread']:g}; variance `{payload['variance_formula']}`.",
             f"- Yield range: `Y_lo = {yields['y_lo']:.6g}`, `Y_hi = {yields['y_hi']:.6g}` photons "
             f"({yields['y_hi_bound_source']}); median yield `{yields['median_yield']:.6g}` "
             f"({yields['median_policy']}).",
             f"- N_min = {occ['n_min']} at Y_lo: within target median {occ['within_target_median_hits']:.0f} "
             f"hits (exclusion {occ['within_exclusion_rate']:.3g}); across proxy median "
             f"{occ['across_target_median_hits']:.0f} hits (below-N_min "
             f"{occ['across_initial_below_nmin_rate']:.3g}).", "",
             "| factor | setting | R²_run (lower bound) | previous frozen R²_run | Δ | pass |",
             "|---|---|---|---|---|---|"]
    previous_rows = {(row["factor"], _setting_key(row["setting"])): row
                     for row in payload.get("ceiling_comparison", [])}
    for row in payload["table"]:
        before = previous_rows.get((row["factor"], _setting_key(row["setting"])))
        before_text = "—" if not before or before["previous_r2_run_median"] is None else \
            f"{before['previous_r2_run_median']:.3f}"
        delta_text = "—" if not before or before["delta_r2_run_median"] is None else \
            f"{before['delta_r2_run_median']:+.3f}"
        lines.append(f"| {row['factor']} | {row['setting']:.3g} | {row['r2_run_median']:.3f} | "
                     f"{before_text} | {delta_text} | {str(row['passes']).lower()} |")
    previous = payload.get("previous_decision", {})
    if previous.get("omitted"):
        lines.append("")
        lines.append(f"- Previous frozen ceilings omitted: {previous['reason']} "
                     f"(`{previous['path']}`).")
    else:
        lines.append("")
        lines.append(f"- Previous frozen ceilings (`{previous['path']}`, sha256 "
                     f"`{previous['sha256']}`, status {previous['status']}, recorded as "
                     f"{previous['recorded_as']}): `Y_lo = {previous['y_lo']:.6g}`, "
                     f"`Y_hi = {previous['y_hi']:.6g}`, F2 {previous['f2_choice']}, "
                     f"F3 {previous['f3_choice']}, v_dark {previous['v_dark']}, readout "
                     f"`{previous['readout_label']}`.")
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
                           frozen: bool = False, previous_path: Path | str | None = None) -> dict:
    """The decision JSON; ``frozen=False`` (the default) because the author has not
    frozen the merged-readout values yet.  Freezing is an explicit opt-in (``--freeze``)."""
    seed = int(check.seed if seed is None else seed)
    previous = previous_ceilings(previous_path)
    payload = {
        "schema": DECISION_SCHEMA, "status": "frozen" if frozen else "development_only",
        "readout_label": lb.READOUT_LABEL, "reported_results": False, "frozen": bool(frozen),
        "decision_freeze_status": ("frozen_by_this_re_run" if frozen else
                                   lb.SUPERSEDED_PENDING_MERGED_READOUT),
        "dark_counts": lb.DARK_COUNTS_OFF,
        "fixed_response": lb.reference_provenance(gen_cfg),
        "f3_marginalised": True, "variance_formula": VARIANCE_FORMULA,
        "f3_marginalised_spread": check.f3_marginalised_spread,
        "code_sha": code_sha or lb.current_code_sha(), "seed": seed,
        "config_digest": check.digest, "generator_config_digest": gen_cfg.digest,
        "sigma_spe": check.sigma_spe, "v_dark": check.v_dark, "run_events": check.run_events,
        "n_events": check.n_events, "geometry": {**geometry.summary(), "geometry_sha256": geometry.digest},
        "yields": {**yields, "median_yield": table.median_yield, "median_policy": check.median_policy},
        "occupancy": occupancy,
        "median_yield_policy": {"policy": check.median_policy, "value_photons": table.median_yield},
        "nuisance_policy": {k: list(v) for k, v in NUISANCE_POLICY.items()},
        "table": _table_rows(table), "fallback_steps": list(steps), "branches": dict(branches),
        "previous_decision": previous,
        "ceiling_comparison": ceiling_comparison(table, previous),
        "rebenchmark": {"y_lo": float(yields["y_lo"]), "y_hi": float(yields["y_hi"]),
                        "median_yield": table.median_yield, "volume_events": lb.REGISTERED_VOLUME_EVENTS,
                        "volume_breakdown": dict(lb.REGISTERED_VOLUME_BREAKDOWN),
                        "cores_at_most": lb.BENCHMARK_CORES, "cpu_cap_wall_hours": lb.CPU_CAP_WALL_HOURS,
                        "note": "repeat benchmark_generation at the frozen yields"},
        "pending": ["the frozen 2026-10-17 yields, F2-vs-F2' and s_p are "
                    "superseded_pending_merged_readout until this merged-readout re-run is "
                    "accepted; the author has not frozen the new values"],
        "ambiguities": AMBIGUITIES,
    }
    payload["decisions_md"] = _decisions_md(payload)
    return _freeze(payload)
def validate_decision(payload: Mapping) -> None:
    if payload.get("schema") != DECISION_SCHEMA:
        raise ValueError(f"decision schema {payload.get('schema')!r} != {DECISION_SCHEMA!r}")
    required = ("status", "readout_label", "reported_results", "frozen", "geometry", "yields",
                "occupancy", "table", "fallback_steps", "branches", "rebenchmark", "content_sha256",
                "decisions_md", "dark_counts", "fixed_response", "f3_marginalised",
                "variance_formula", "previous_decision", "ceiling_comparison")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"decision payload missing {missing}")
    if payload["readout_label"] != lb.READOUT_LABEL:
        raise ValueError(f"decision readout_label must be {lb.READOUT_LABEL!r}")
    if payload["dark_counts"] != lb.DARK_COUNTS_OFF or float(payload["v_dark"]) != 0.0:
        raise ValueError("decision is not a dark-off readout")
    if payload["f3_marginalised"] is not True or payload["variance_formula"] != VARIANCE_FORMULA:
        raise ValueError("decision does not marginalise F3 with the registered variance")
    if payload["fixed_response"].get("sha256") != lb.reference_response().sha256:
        raise ValueError("decision fixed-response hash mismatch")
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
              y_hi: float | None = None, y_hi_bound: float | None = None, frozen: bool = False,
              code_sha: str | None = None, previous_path: Path | str | None = None):
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
                                  seed=seed, code_sha=code_sha, frozen=frozen,
                                  previous_path=previous_path)
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
    parser.add_argument("--previous-decision", default=None,
                        help="previously frozen decision JSON to record beside the new ceilings")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--freeze", action="store_true",
                        help="write a frozen decision (only after the author accepts the re-run)")
    parser.add_argument("--no-freeze", action="store_true",
                        help="kept for compatibility; not freezing is the default")
    args = parser.parse_args(argv)
    check, gen_cfg = load_check_config(args.config), lb.load_generator_config(args.generator_config)
    geometry = lb.load_wcte_geometry(args.geometry)
    if geometry.synthetic and not args.allow_synthetic:
        raise SystemExit("refusing a synthetic geometry without --allow-synthetic")
    if args.freeze and args.no_freeze:
        raise SystemExit("--freeze and --no-freeze are mutually exclusive")
    frozen = bool(args.freeze) and not geometry.synthetic
    payload = run_check(geometry, gen_cfg, check, seed=args.seed, y_lo=args.y_lo, y_hi=args.y_hi,
                        frozen=frozen, previous_path=args.previous_decision)
    write_decision(args.out, payload)
    print(payload["decisions_md"])
    print(json.dumps({k: payload[k] for k in ("status", "frozen", "content_sha256")}, sort_keys=True))
    return 0
if __name__ == "__main__":  # pragma: no cover - host CLI
    raise SystemExit(main())
