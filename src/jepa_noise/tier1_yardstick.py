"""r2-tier1 T6 yardsticks with LUCiD's own models (runs in the LUCiD JAX environment, no torch).

Parameters theta = (x, y, z, t0, ln N) of a photon bomb (isotropic point source of N photons).

* Expected response: LUCiD ``setup_event_simulator(source="light_source", transport="expected",
  detection="expected", hits="photon_list")`` with ``isotropic_source`` (the expectation mode of
  validation gate G3, ``scripts/lucid_generate.py`` selfcheck), averaged over a fixed set of keys.
  mu_p(theta) = expected PE per PMT + dark expectation in the prompt window.
* Counts term: LUCiD ``losses.counts_loss(..., normalize=False)`` (Poisson).
* Time term: LUCiD ``losses.first_arrival_window_nll`` (the reconstruction time term; weights
  and totals detached inside, so it constrains vertex and t0 only).

Y1 (surrogate Fisher): F_counts = J^T diag(1/mu) J with J = d mu / d theta (forward-mode
autodiff); F_time = covariance over 32 sampled replicates of the same truth of the score of the
time term at the truth; joint = sum. Nuisances a channel cannot constrain are dropped (t0 for
counts, ln N for time). Y2 (achieved resolution): L-BFGS-B minimisation of the counts, time and
joint losses on the sampled events, started at the truth (and at the charge centroid for a subset).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

PARAMS = ("x", "y", "z", "t0", "lnN")
SIGMA_TTS_NS = 2.1   # LUCiD published reconstruction working point (examples/seed_reconstruct.py)
DELTA_NS = 1.0       # LUCiD recon window default
C_WATER_M_PER_NS = 0.299792458 / 1.33


class Model:
    def __init__(self, detector, n_rays=65536, n_keys=4, temperature=0.1, K=12, window_ns=(-20, 270),
                 dark_rate_khz=4.2, sigma_tts_ns=SIGMA_TTS_NS):
        import jax
        jax.config.update("jax_enable_x64", False)
        from . import lucid_mc_generator as mg
        self.jax = jax
        self.det = detector
        self.kernel = mg.LucidKernel(detector, n_rays)
        spec = mg.SPECS[detector]
        cwd = os.getcwd()
        os.chdir(mg.LUCID_ROOT)
        try:
            from lucid.simulation import setup_event_simulator
            self.esim = setup_event_simulator(str(self.kernel.geom_cfg), n_rays, source="light_source",
                                              transport="expected", detection="expected",
                                              hits="photon_list", K=K, temperature=temperature,
                                              physics_config=str(self.kernel.phys_cfg),
                                              default_detector_params=True,
                                              wavelength_mode=bool(spec["wavelength_mode"]))
        finally:
            os.chdir(cwd)
        jnp = jax.numpy
        from lucid.losses import counts_loss, first_arrival_window_nll
        from lucid.sources.calibration_sources import IsotropicSource
        # the pytree directly (the isotropic_source factory casts to float and is not traceable)
        self.counts_loss, self.time_nll = counts_loss, first_arrival_window_nll
        self.iso = lambda pos, n: IsotropicSource(position=pos.astype(jnp.float32), intensity=n.astype(jnp.float32))
        self.n_det = int(self.kernel.n_sensors)
        self.dark = float(dark_rate_khz) * 1e3 * (window_ns[1] - window_ns[0]) * 1e-9
        self.keys = [jax.random.PRNGKey(7000 + i) for i in range(n_keys)]
        self.config = {"n_rays": n_rays, "n_keys": n_keys, "temperature": temperature, "K": K,
                       "sigma_tts_ns": float(sigma_tts_ns), "delta_ns": DELTA_NS, "dark_pe_per_pmt": self.dark}
        jnp = jax.numpy

        def forward(theta, key):
            res = self.esim(self.iso(theta[:3], jnp.exp(theta[4])), key)
            return res.log_w, res.times + theta[3], res.indices, res.charge

        def mu(theta):
            return sum(forward(theta, k)[3] for k in self.keys) / len(self.keys) + self.dark

        def loss_counts(theta, q):
            return self.counts_loss(q, mu(theta), normalize=False)

        def loss_time(theta, t_obs, n_obs):
            total = 0.0
            for k in self.keys:
                lw, tt, idx, ch = forward(theta, k)
                total = total + self.time_nll(lw, tt, idx, t_obs, ch + self.dark, n_obs, self.n_det,
                                              sigma=float(sigma_tts_ns), delta=DELTA_NS).sum()
            return total / len(self.keys)

        self.mu = jax.jit(mu)
        self.jac_mu = jax.jit(jax.jacfwd(mu))
        self.loss_counts = jax.jit(loss_counts)
        self.grad_counts = jax.jit(jax.value_and_grad(loss_counts))
        self.loss_time = jax.jit(loss_time)
        self.grad_time = jax.jit(jax.value_and_grad(loss_time))

    # ------------------------------------------------------------------ data helpers
    def observed(self, pmt_lucid, q, t_abs):
        """Per-sensor charge, first time (absolute clock) and count for the time term."""
        qq = np.zeros(self.n_det)
        tt = np.zeros(self.n_det)
        nn = np.zeros(self.n_det)
        np.add.at(qq, pmt_lucid, q)
        order = np.argsort(t_abs)
        first = {}
        for i in order:
            first.setdefault(int(pmt_lucid[i]), float(t_abs[i]))
        idx = np.fromiter(first.keys(), int)
        tt[idx] = np.fromiter(first.values(), float)
        nn[idx] = np.maximum(1.0, np.round(qq[idx]))
        return qq, tt, nn

    def replicate(self, theta_true, seed):
        """One sampled replicate of the same truth through LUCiD's production kernel (no
        electronics: first deposit time per sensor and deposit count)."""
        from . import lucid_bombs as lb
        from . import lucid_mc_generator as mg
        truth = mg.Truth(f"yardstick/{self.det}", int(seed), lb.DOMAIN_PHOTON,
                         np.asarray(theta_true[:3], float), float(np.exp(theta_true[4])), 0.0)
        dep = self.kernel.simulate(truth, int(seed))
        fin = np.isfinite(dep["t_reco"])
        s, t = np.asarray(dep["sensor"])[fin].astype(int), np.asarray(dep["t_reco"], float)[fin] + theta_true[3]
        nn = np.bincount(s, minlength=self.n_det).astype(float)
        tt = np.full(self.n_det, np.inf)
        np.minimum.at(tt, s, t)
        tt[~np.isfinite(tt)] = 0.0
        return tt, nn

    # ------------------------------------------------------------------ Y1
    def fisher(self, theta_true, n_rep=32, seed0=0):
        theta = np.asarray(theta_true, np.float32)
        jac = np.asarray(self.jac_mu(theta), float)            # (n_det, 5)
        mu = np.asarray(self.mu(theta), float)
        f_counts = (jac / mu[:, None]).T @ jac
        scores = []
        for r in range(n_rep):
            tt, nn = self.replicate(theta_true, seed0 + r)
            _, g = self.grad_time(theta, tt.astype(np.float32), nn.astype(np.float32))
            scores.append(np.asarray(g, float))
        scores = np.asarray(scores)
        f_time = np.cov(scores.T)
        out = {"F_counts": f_counts.tolist(), "F_time": f_time.tolist(), "mu_total": float(mu.sum())}
        keep_c, keep_t, keep_j = [0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3, 4]
        for name, f, keep in (("counts", f_counts, keep_c), ("time", f_time, keep_t),
                              ("joint", f_counts + f_time, keep_j)):
            sub = f[np.ix_(keep, keep)]
            try:
                cov = np.linalg.inv(sub)
                out[f"sigma_vertex_{name}_m"] = float(np.sqrt(np.trace(cov[:3, :3])))
            except np.linalg.LinAlgError:
                out[f"sigma_vertex_{name}_m"] = None
        sc, st = out["sigma_vertex_counts_m"], out["sigma_vertex_time_m"]
        out["time_share"] = (sc ** 2 / (sc ** 2 + st ** 2)) if sc and st else None
        return out

    # ------------------------------------------------------------------ Y2
    def fit(self, mode, theta_start, theta_true, obs, maxiter=40):
        from scipy.optimize import minimize
        qq, tt, nn = (o.astype(np.float32) for o in obs)
        free = {"counts": [0, 1, 2, 4], "time": [0, 1, 2, 3], "joint": [0, 1, 2, 3, 4]}[mode]
        base = np.asarray(theta_true, np.float64).copy()
        base[free] = np.asarray(theta_start, np.float64)[free]
        scale = np.array([0.2, 0.2, 0.2, 0.5, 0.05])[free]

        def fg(u):
            th = base.copy()
            th[free] = u * scale
            th32 = th.astype(np.float32)
            v, g = 0.0, np.zeros(5)
            if mode in ("counts", "joint"):
                vc, gc = self.grad_counts(th32, qq)
                v, g = v + float(vc), g + np.asarray(gc, float)
            if mode in ("time", "joint"):
                vt, gt = self.grad_time(th32, tt, nn)
                v, g = v + float(vt), g + np.asarray(gt, float)
            return v, g[free] * scale

        t0 = time.time()
        res = minimize(fg, base[free] / scale, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter})
        th = base.copy()
        th[free] = res.x * scale
        return {"theta": th.tolist(), "success": bool(res.success), "nit": int(res.nit),
                "fun": float(res.fun), "wall_s": time.time() - t0,
                "vertex_error_m": float(np.linalg.norm(th[:3] - np.asarray(theta_true[:3])))}


def run(args):
    """lucid_generate.py yardstick entry: Y1 and/or Y2 for events [start, stop) of the input file."""
    data = np.load(args.input, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    det = meta["detector"]
    model = Model(det, n_rays=args.n_rays, n_keys=args.n_keys, window_ns=tuple(meta["prompt_window_ns"]),
                  sigma_tts_ns=getattr(args, "sigma_tts_ns", None) or SIGMA_TTS_NS)
    off = data["offsets"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = min(args.stop, len(data["yield"]))
    for i in range(args.start, stop):
        path = out_dir / f"{det}_event_{i:04d}.json"
        if path.exists():
            continue
        sl = slice(off[i], off[i + 1])
        theta_true = np.r_[data["vertex"][i], data["t0"][i], np.log(data["yield"][i])]
        obs = model.observed(data["pmt_lucid"][sl], data["q"][sl], data["t_abs"][sl])
        row = {"index": i, "event_id": int(data["event_id"][i]), "theta_true": theta_true.tolist(),
               "config": model.config}
        t0 = time.time()
        if "y1" in args.parts:
            row["y1"] = model.fisher(theta_true, n_rep=args.n_rep, seed0=1_000_000 + 100 * i)
        if "y2" in args.parts and i < meta["n_y2"]:
            row["y2"] = {m: model.fit(m, theta_true, theta_true, obs, args.maxiter)
                         for m in ("counts", "time", "joint")}
            if i < args.n_centroid:
                d = np.linalg.norm(data["positions_lucid_order"][data["pmt_lucid"][sl]] - data["centroid"][i], axis=1)
                start = np.r_[data["centroid"][i], float(np.min(data["t_abs"][sl] - d / C_WATER_M_PER_NS)),
                              np.log(data["yield"][i])]
                row["y2_centroid"] = {m: model.fit(m, start, theta_true, obs, args.maxiter)
                                      for m in ("counts", "time", "joint")}
        row["wall_s"] = time.time() - t0
        path.write_text(json.dumps(row) + "\n")
        print(json.dumps({"event": i, "wall_s": round(row["wall_s"], 1),
                          "y1_joint": row.get("y1", {}).get("sigma_vertex_joint_m"),
                          "y2_joint": row.get("y2", {}).get("joint", {}).get("vertex_error_m")}), flush=True)


# ---------------------------------------------------------------------- r2-mask M3 seed test
SEED_STARTS = ("truth", "centroid", "readout", "init", "raw")


def seed_start(data, starts, i, name, pos_lucid, sl, radius_m, half_height_m, margin_m=0.05):
    """theta_start of start ``name`` for event i (runbook 2026-11-23 §M3.2).

    ``truth``: the full true theta (as Y2). Otherwise (x, y, z) from the starts file, projected into
    the tank (``margin_m`` inside the wall) if it lies outside; ln N from the starts file (linear fit
    to total charge on dev_probe_train); t0 = median over hit PMTs of (t_abs - |r_p - x| / c_water).
    """
    theta_true = np.r_[data["vertex"][i], data["t0"][i], np.log(data["yield"][i])]
    if name == "truth":
        return theta_true, False
    xyz = np.asarray(starts[f"start_{name}"][i], np.float64).copy()
    clipped = False
    r = float(np.hypot(xyz[0], xyz[1]))
    if r > radius_m - margin_m:
        xyz[:2] *= (radius_m - margin_m) / r
        clipped = True
    if abs(xyz[2]) > half_height_m - margin_m:
        xyz[2] = np.sign(xyz[2]) * (half_height_m - margin_m)
        clipped = True
    d = np.linalg.norm(pos_lucid[data["pmt_lucid"][sl]] - xyz, axis=1)
    t0 = float(np.median(data["t_abs"][sl] - d / C_WATER_M_PER_NS))
    return np.r_[xyz, t0, float(starts["lnN_start"][i])], clipped


def run_seedtest(args):
    """lucid_generate.py seedtest entry: Y2-style fits of the first n events from every start."""
    data = np.load(args.input, allow_pickle=False)
    starts = np.load(args.starts, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    det = meta["detector"]
    n = len(starts["event_id"])
    if not np.array_equal(np.asarray(starts["event_id"]), np.asarray(data["event_id"][:n])):
        raise ValueError("starts file and yardstick input disagree on the events")
    radius = float(meta.get("radius_m") or np.hypot(data["positions_lucid_order"][:, 0],
                                                     data["positions_lucid_order"][:, 1]).max())
    half = float(meta.get("half_height_m") or np.abs(data["positions_lucid_order"][:, 2]).max())
    model = Model(det, n_rays=args.n_rays, n_keys=args.n_keys, window_ns=tuple(meta["prompt_window_ns"]))
    off = data["offsets"]
    pos = data["positions_lucid_order"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.start, min(args.stop, n)):
        path = out_dir / f"{det}_seed_{i:04d}.json"
        if path.exists():
            continue
        sl = slice(off[i], off[i + 1])
        theta_true = np.r_[data["vertex"][i], data["t0"][i], np.log(data["yield"][i])]
        obs = model.observed(data["pmt_lucid"][sl], data["q"][sl], data["t_abs"][sl])
        row = {"index": i, "event_id": int(data["event_id"][i]), "theta_true": theta_true.tolist(),
               "config": dict(model.config, maxiter=args.maxiter), "starts": {}}
        t0 = time.time()
        for name in args.start_names:
            th0, clipped = seed_start(data, starts, i, name, pos, sl, radius, half)
            row["starts"][name] = {
                "theta_start": th0.tolist(), "clipped": clipped,
                "start_error_m": float(np.linalg.norm(th0[:3] - theta_true[:3])),
                "fits": {m: model.fit(m, th0, theta_true, obs, args.maxiter) for m in args.modes}}
        row["wall_s"] = time.time() - t0
        path.write_text(json.dumps(row) + "\n")
        print(json.dumps({"event": i, "wall_s": round(row["wall_s"], 1),
                          **{f"{s}_{m}": round(row["starts"][s]["fits"][m]["vertex_error_m"], 3)
                             for s in args.start_names for m in args.modes}}), flush=True)
