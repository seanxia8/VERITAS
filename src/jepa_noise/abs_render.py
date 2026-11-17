"""Differentiable analytic renderer of the round-2 photon bomb (analysis-by-synthesis, steps 1–2).

Runbook ``RUNBOOK_2026-11-17_R2_DIAGNOSTICS.md`` §A5.  The forward model is the generator's own
code path, written in PyTorch with the same parameters (``r2_generator.readout`` /
``lucid_bombs.expected_charge``):

* ``mu_p = Y · QE_p · CE · Ω_p/(4π) · exp(-d_p/λ) · g^fix_p``, Ω the disc solid angle times the
  obliquity (``lucid_bombs.solid_angle_fraction``), ``g^fix`` the fixed group-parity gain;
* ``QE_p = QE(λ_p)`` with one wavelength ``λ_p ~ U(300, 600) nm`` per PMT per event
  (``n_wavelength_samples = 1``) – unknown to a fit, so it is marginalised by quadrature;
* photo-electrons ``n = Poisson(mu_p) + Poisson(dark)`` (dark = 1 kHz × 1 µs, bypasses QE);
* charge ``q = g_p (n + sqrt(n) σ_spe z)``, the hit fires when ``q`` reaches the discriminator;
* first-hit time ``t = d_p n_g / c + t0 + σ_t,p z`` (the walk residual is zero because the
  walk calibration charge equals the threshold); a dark-only hit has a uniform time.

θ = (x, y, z, ln Y, t0).  The likelihood is Poisson–Gaussian charge for hit PMTs, the no-hit
probability for the others, and the Gaussian (plus dark-uniform) first-hit time.  The run's
detector state (gains, attenuation length) is taken as known: this is a ceiling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import lucid_bombs as lb

__all__ = ["PhotonBombRenderer", "qe_quadrature", "discriminator_threshold_pe", "fit_event",
           "fisher_crb"]

N_EXACT = 64          # Poisson–Gaussian sum up to this many PE
GAUSS_ABOVE_PE = 36.0  # above this charge (in PE) the Poisson–Gaussian is replaced by a Gaussian
C_M_PER_NS = lb.C_M_PER_S * 1e-9


def qe_quadrature(cfg, per_segment: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights of E_λ[f(QE(λ))] for λ ~ U(spectrum): Gauss–Legendre on each linear piece
    of the QE curve (exact for polynomials in QE of degree 2·per_segment − 1)."""
    lo, hi = (float(v) for v in cfg.spectrum_nm)
    knots = np.asarray(cfg.qe_curve_nm, float)
    edges = np.unique(np.clip(np.concatenate([[lo, hi], knots]), lo, hi))
    x, w = np.polynomial.legendre.leggauss(int(per_segment))
    lam, wt = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        lam.append(0.5 * (b - a) * x + 0.5 * (b + a))
        wt.append(0.5 * (b - a) * w / (hi - lo))
    lam, wt = np.concatenate(lam), np.concatenate(wt)
    return lb.qe_at(lam, cfg), wt


def discriminator_threshold_pe(cfg) -> float:
    """Smallest charge (PE × gain) whose leading edge is finite (the readout's fired rule)."""
    thr_mv = float(cfg.threshold_pe) * float(cfg.mv_per_pe)
    grid = np.linspace(0.01, 2.0, 19901)
    t = lb.leading_edge_time(grid, thr_mv, mv_per_pe=float(cfg.mv_per_pe))
    return float(grid[np.isfinite(t)][0])


@dataclass
class EventData:
    hit: np.ndarray       # (n_hit,) PMT indices
    charge: np.ndarray    # (n_hit,) stored charge (PE × gain)
    time: np.ndarray      # (n_hit,) stored token time (ns, digit − trigger)


class PhotonBombRenderer(torch.nn.Module):
    def __init__(self, geometry, cfg, state, reference=None, *, qe_per_segment: int = 4,
                 dtype=torch.float64):
        super().__init__()
        ref = lb.generator_reference(cfg) if reference is None else reference
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=dtype)
        self.n_pmt = int(geometry.n_pmt)
        self.register_buffer("pos", t(geometry.positions_m))
        self.register_buffer("dirs", t(geometry.directions))
        self.sensor_radius = float(geometry.sensor_radius_m)
        self.register_buffer("gain", t(state.gains))
        self.register_buffer("fixed_gain", t(lb.fixed_charge_gain(geometry, ref)))
        self.register_buffer("sigma_t", t(float(cfg.tts_sigma_ns)
                                          * lb.fixed_timing_width_factor(geometry, ref)))
        self.lambda_att = float(state.attenuation_length_m)
        self.ce = float(cfg.ce)
        self.n_group = float(cfg.n_group)
        start, end = (float(v) for v in cfg.readout_window_ns)
        self.window_ns = end - start
        self.dark = float(cfg.dark_rate_hz) * self.window_ns * 1e-9
        self.sigma_spe = float(cfg.sigma_spe)
        self.q_thr = discriminator_threshold_pe(cfg)
        nodes, weights = qe_quadrature(cfg, qe_per_segment)
        self.register_buffer("qe", t(nodes))
        self.register_buffer("log_w", t(np.log(weights)))
        self.register_buffer("qe_mean", t([np.sum(nodes * weights)]))
        self.register_buffer("qe_var", t([np.sum(nodes ** 2 * weights)
                                          - np.sum(nodes * weights) ** 2]))
        # P(not fired | n) for n = 1, 2 (threshold in PE before the gain)
        thr = self.q_thr / torch.clamp(self.gain, min=1e-6)
        ns = torch.tensor([1.0, 2.0], dtype=dtype)
        z = (thr[:, None] - ns[None, :]) / (self.sigma_spe * torch.sqrt(ns)[None, :])
        self.register_buffer("p_low", 0.5 * torch.erfc(-z / math.sqrt(2.0)))  # (P, 2)
        self.register_buffer("n_grid", torch.arange(1, N_EXACT + 1, dtype=dtype))
        self.register_buffer("lgam", torch.lgamma(self.n_grid + 1.0))
        self.dtype = dtype

    # ------------------------------------------------------------------ forward model
    def geometry_terms(self, x):
        delta = self.pos - x
        d = torch.linalg.norm(delta, dim=1)
        r = self.sensor_radius
        cos = torch.abs((delta * self.dirs).sum(1)) / d
        omega = 0.5 * (1.0 - d / torch.sqrt(d * d + r * r)) * torch.clamp(cos, 0.0, 1.0)
        return d, omega

    def mu_noqe(self, theta):
        """Per-PMT mean PE divided by QE (Y · CE · Ω/4π · e^{-d/λ} · g^fix) and distances."""
        d, omega = self.geometry_terms(theta[:3])
        base = torch.exp(theta[3]) * self.ce * omega * torch.exp(-d / self.lambda_att) * self.fixed_gain
        return base, d

    def predicted_time(self, theta, d):
        return d * (self.n_group / C_M_PER_NS) + theta[4]

    def moments(self, theta):
        """E[q], Var[q] per PMT (charge in stored units) and P(hit), at θ."""
        base, d = self.mu_noqe(theta)
        mean_n = base * self.qe_mean + self.dark
        var_n = mean_n + base * base * self.qe_var  # Poisson + QE spread
        g = self.gain
        eq = g * mean_n
        vq = g * g * (var_n + self.sigma_spe ** 2 * mean_n)
        mu_k = base[:, None] * self.qe[None, :] + self.dark
        p_unhit = (torch.exp(self.log_w)[None, :] * self._p_unhit_k(mu_k)).sum(1)
        return eq, vq, 1.0 - p_unhit, base, d

    def _p_unhit_k(self, mu_k, sel=None):
        pl = self.p_low if sel is None else self.p_low[sel]
        e = torch.exp(-mu_k)
        return e * (1.0 + mu_k * pl[:, :1] + 0.5 * mu_k * mu_k * pl[:, 1:2])

    # ------------------------------------------------------------------ likelihood
    def nll(self, theta, ev: EventData, *, terms=("charge", "hit", "time"), per_pmt=False):
        base, d = self.mu_noqe(theta)
        mu_k = base[:, None] * self.qe[None, :] + self.dark            # (P, K)
        log_w = self.log_w[None, :]
        hit = torch.as_tensor(ev.hit, dtype=torch.long)
        unhit_mask = torch.ones(self.n_pmt, dtype=torch.bool)
        unhit_mask[hit] = False
        total = theta.new_zeros(())
        out = {}
        if "hit" in terms:
            lp0 = torch.logsumexp(log_w + torch.log(self._p_unhit_k(mu_k[unhit_mask], unhit_mask)), dim=1)
            out["hit"] = -lp0.sum()
            total = total + out["hit"]
        mu_h = mu_k[hit]                                                 # (H, K)
        g = self.gain[hit]
        x = torch.as_tensor(ev.charge, dtype=self.dtype) / g            # charge in PE
        if "charge" in terms:
            small = x < GAUSS_ABOVE_PE
            lf = torch.empty_like(x)
            if bool(small.any()):
                xs, ms = x[small], mu_h[small]                           # (S,), (S, K)
                n = self.n_grid
                lpois = n[None, None, :] * torch.log(ms[:, :, None]) - ms[:, :, None] - self.lgam
                var = (self.sigma_spe ** 2) * n
                lgau = (-0.5 * (xs[:, None, None] - n) ** 2 / var
                        - 0.5 * torch.log(2 * math.pi * var))
                lf[small] = torch.logsumexp(log_w[:, :, None] + lpois + lgau, dim=(1, 2))
            big = ~small
            if bool(big.any()):
                xb, mb = x[big], mu_h[big]
                var = mb * (1.0 + self.sigma_spe ** 2)
                lgau = -0.5 * (xb[:, None] - mb) ** 2 / var - 0.5 * torch.log(2 * math.pi * var)
                lf[big] = torch.logsumexp(log_w + lgau, dim=1)
            lf = lf - torch.log(g)
            out["charge"] = -lf.sum()
            total = total + out["charge"]
        if "time" in terms:
            th = torch.as_tensor(ev.time, dtype=self.dtype)
            st = self.sigma_t[hit]
            tp = self.predicted_time(theta, d[hit])
            p_ph0 = torch.exp(torch.logsumexp(log_w - (mu_h - self.dark), dim=1))
            p_hit = 1.0 - torch.exp(torch.logsumexp(log_w + torch.log(self._p_unhit_k(mu_h, hit)), dim=1))
            eps = torch.clamp(p_ph0 * (1.0 - math.exp(-self.dark)) / torch.clamp(p_hit, min=1e-300),
                              0.0, 1.0)
            lg = -0.5 * ((th - tp) / st) ** 2 - torch.log(st * math.sqrt(2 * math.pi))
            lt = torch.logaddexp(torch.log1p(-eps.clamp(max=1 - 1e-12)) + lg,
                                 torch.log(eps.clamp(min=1e-300)) - math.log(self.window_ns))
            out["time"] = -lt.sum()
            total = total + out["time"]
        out["total"] = total
        return out if per_pmt else total


# ====================================================================== fitting
def profile_t0(renderer, theta_xyzY, ev: EventData) -> float:
    """t0 that maximises the time term for a fixed (x, y, z, ln Y): weighted median then
    Gaussian mean of the residuals (robust start for fits and for the truth)."""
    with torch.no_grad():
        th = torch.cat([theta_xyzY, theta_xyzY.new_zeros(1)])
        _, d = renderer.mu_noqe(th)
        res = np.asarray(ev.time) - (d[torch.as_tensor(ev.hit)] * renderer.n_group / C_M_PER_NS).numpy()
    med = float(np.median(res))
    keep = np.abs(res - med) < 10.0
    return float(np.mean(res[keep])) if keep.any() else med


def _initial_points(renderer, ev: EventData, geometry) -> list:
    pos = renderer.pos.numpy()
    q = np.asarray(ev.charge, float)
    w = q / q.sum()
    centroid = (pos[ev.hit] * w[:, None]).sum(0)
    early = np.argsort(ev.time)[:max(5, len(ev.time) // 20)]
    early_c = pos[ev.hit][early].mean(0)
    pts = [0.5 * centroid, np.zeros(3), 0.5 * early_c]
    out = []
    for p in pts:
        x = torch.as_tensor(p, dtype=renderer.dtype)
        with torch.no_grad():
            base, _ = renderer.mu_noqe(torch.cat([x, x.new_zeros(2)]))
            ly = math.log(max(q.sum(), 1.0) / float((base * renderer.qe_mean * renderer.gain).sum()))
        xy = torch.cat([x, x.new_tensor([ly])])
        out.append(torch.cat([xy, xy.new_tensor([profile_t0(renderer, xy, ev)])]))
    return out


def fit_event(renderer, ev: EventData, geometry, *, terms=("charge", "hit", "time"),
              max_iter: int = 200, extra_inits=()) -> dict:
    """Maximum-likelihood θ from several initial values (none uses the truth)."""
    scale = torch.tensor([geometry.radius_m, geometry.radius_m, max(geometry.radius_m,
                                                                     geometry.half_height_m),
                          1.0, 10.0], dtype=renderer.dtype)
    best = None
    inits = _initial_points(renderer, ev, geometry) + list(extra_inits)
    for k, theta0 in enumerate(inits):
        u = (theta0 / scale).clone().requires_grad_(True)
        opt = torch.optim.LBFGS([u], lr=1.0, max_iter=max_iter, tolerance_grad=1e-9,
                                tolerance_change=1e-12, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            v = renderer.nll(u * scale, ev, terms=terms)
            v.backward()
            return v
        try:
            opt.step(closure)
            opt.step(closure)
        except RuntimeError:
            continue
        with torch.no_grad():
            val = float(renderer.nll(u * scale, ev, terms=terms))
        if np.isfinite(val) and (best is None or val < best["nll"]):
            best = {"theta": (u * scale).detach().numpy().copy(), "nll": val, "init": k}
    return best


def fisher_crb(renderer, theta, ev: EventData, *, terms=("charge", "hit", "time")) -> dict:
    """Observed Fisher information (Hessian of the NLL) at θ and the Cramér–Rao covariance."""
    th = torch.as_tensor(theta, dtype=renderer.dtype)
    h = torch.autograd.functional.hessian(lambda v: renderer.nll(v, ev, terms=terms), th)
    h = 0.5 * (h + h.T)
    if "time" not in terms:  # t0 carries no information without the time term
        h = h[:4, :4]
    try:
        cov = torch.linalg.inv(h)
        ok = bool(torch.all(torch.linalg.eigvalsh(h) > 0))
    except RuntimeError:
        return {"ok": False}
    c = cov.numpy()
    return {"ok": ok, "sigma_xyz_m": float(np.sqrt(max(np.trace(c[:3, :3]), 0.0))),
            "sigma_x_m": float(np.sqrt(max(c[0, 0], 0))), "sigma_ln_y": float(np.sqrt(max(c[3, 3], 0))),
            "sigma_t0_ns": float(np.sqrt(max(c[4, 4], 0))) if c.shape[0] > 4 else None}


def reconstruction_fisher(renderer, theta) -> dict:
    """Expected joint Poisson-count and Gaussian-time Fisher information.

    A time observation exists when the Poisson count is positive. Its score
    has conditional mean zero, so the joint information has no cross term
    between the count score and the conditional time score.
    """
    theta = torch.as_tensor(theta, dtype=renderer.dtype, device=renderer.pos.device)

    def predict(parameters):
        base, distance = renderer.mu_noqe(parameters)
        return torch.stack((base * renderer.qe_mean + renderer.dark,
                            renderer.predicted_time(parameters, distance)), dim=0)

    jacobian = torch.func.jacfwd(predict)(theta)
    mean = predict(theta)[0].clamp_min(1e-12)
    charge, arrival = jacobian.unbind(0)
    count_fisher = charge.T @ (charge / mean[:, None])
    time_weights = -torch.expm1(-mean) / renderer.sigma_t.square()
    fisher = count_fisher + arrival.T @ (time_weights[:, None] * arrival)
    eigenvalues = torch.linalg.eigvalsh(fisher)
    if not bool(torch.isfinite(eigenvalues).all() and eigenvalues.min() > 0):
        return {"ok": False, "sigma_xyz_m": None}
    cov = torch.linalg.inv(fisher)
    return {"ok": True, "sigma_xyz_m": float(cov[:3, :3].trace().sqrt()),
            "fisher_condition": float(eigenvalues.max() / eigenvalues.min())}


def fit_reconstruction(renderer, ev: EventData, geometry, *, variant: str,
                       seed: int, n_starts: int = 5) -> dict:
    """Fit the runbook's simpler Poisson-count and Huber timing objectives."""
    from scipy.optimize import minimize

    if variant not in ("tof", "charge", "joint"):
        raise ValueError(f"unknown reconstruction objective {variant}")
    if not len(ev.hit):
        return {"converged": False, "theta": None, "n_starts": 0}

    dev, dtype = renderer.pos.device, renderer.dtype
    hit = torch.as_tensor(ev.hit, dtype=torch.long, device=dev)
    time = torch.as_tensor(ev.time, dtype=dtype, device=dev)
    prompt = time >= 0
    timing_hit, observed = hit[prompt], time[prompt]
    if variant != "charge" and len(timing_hit) < 4:
        return {"converged": False, "theta": None, "n_starts": 0}
    q = torch.zeros(renderer.n_pmt, dtype=dtype, device=dev)
    q[hit] = torch.as_tensor(ev.charge, dtype=dtype, device=dev) / renderer.gain[hit]
    dimensions = torch.tensor([geometry.radius_m, geometry.radius_m, geometry.half_height_m],
                              dtype=dtype, device=dev)
    scale = torch.cat([dimensions, dimensions.new_tensor([1.0, 20.0])])
    n_params = 4 if variant == "charge" else 5
    if variant == "tof":
        # ln Y does not enter the ToF-only objective.
        indices = [0, 1, 2, 4]
    else:
        indices = list(range(n_params))
    rng = np.random.default_rng(seed)
    early = torch.argsort(time)[:max(4, len(time) // 10)]
    centroid = (renderer.pos[hit] * q[hit, None]).sum(0) / q.sum().clamp_min(1.0)
    initial_vertex = (centroid + renderer.pos[hit[early]].mean(0)) * 0.25

    def objective(parameters):
        p = torch.tensor(parameters, device=dev, dtype=dtype, requires_grad=True)
        x = p * scale[indices]
        loss = x.new_zeros(())
        if variant == "tof":
            delta = renderer.pos[timing_hit] - x[:3]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            arrival = x[3] + renderer.n_group * distance / C_M_PER_NS
        else:
            base, distance = renderer.mu_noqe(x)
            mean = (base * renderer.qe_mean + renderer.dark).clamp_min(1e-10)
            if variant == "charge":
                loss = (mean - q * mean.log()).sum()
            else:
                # Factor count likelihood into occupancy and charge given a
                # positive count, keeping the Poisson information counted once.
                unhit = torch.ones_like(q, dtype=torch.bool)
                unhit[hit] = False
                mean_hit = mean[hit]
                log_phit = torch.log(-torch.expm1(-mean_hit))
                occupancy = mean[unhit].sum() - log_phit.sum()
                charge_given_hit = (mean_hit - q[hit] * mean_hit.log() + log_phit).sum()
                loss = occupancy + charge_given_hit
                arrival = renderer.predicted_time(x, distance[timing_hit])
        if variant != "charge":
            residual = (observed - arrival) / renderer.sigma_t[timing_hit]
            loss = loss + torch.nn.functional.huber_loss(
                residual, torch.zeros_like(residual), reduction="sum", delta=2.5
            )
        gradient, = torch.autograd.grad(loss, p)
        return float(loss.detach()), gradient.detach().cpu().numpy()

    bounds = [(-0.995, 0.995)] * 3 + ([(-50.0, 50.0)] if variant == "tof" else [(0.0, 25.0)])
    if variant == "joint":
        bounds.append((-50.0, 50.0))
    best = None
    for start in range(n_starts):
        perturbation = 0.0 if start == 0 else torch.as_tensor(
            rng.normal(0.0, 0.15, 3), dtype=dtype, device=dev)
        vertex = (initial_vertex + perturbation * dimensions).clamp(
            min=-0.8 * dimensions, max=0.8 * dimensions)
        with torch.no_grad():
            base, distance = renderer.mu_noqe(torch.cat([vertex, vertex.new_zeros(1)]))
            log_yield = torch.log(q.sum().clamp_min(1.0) /
                                  (base * renderer.qe_mean).sum().clamp_min(1e-10))
            t0 = (observed - renderer.n_group / C_M_PER_NS * distance[timing_hit]).median()
            initial = torch.cat([vertex, log_yield.reshape(1), t0.reshape(1)])
        normalized = (initial[indices] / scale[indices]).cpu().numpy()
        outcome = minimize(objective, normalized, method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"maxiter": 125, "ftol": 1e-12, "gtol": 1e-6})
        if np.isfinite(outcome.fun) and (best is None or (not best.success, best.fun) >
                                         (not outcome.success, outcome.fun)):
            best = outcome
    if best is None:
        return {"converged": False, "theta": None, "n_starts": n_starts}
    fitted = list(best.x * scale[indices].cpu().numpy())
    if variant == "tof":
        fitted.insert(3, None)
    elif variant == "charge":
        fitted.append(None)
    return {"converged": bool(best.success), "theta": fitted, "n_starts": n_starts,
            "nll_up_to_constant": float(best.fun)}
