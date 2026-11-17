"""Reusable transport, plotting and JEPA checks from the independent Part A diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lucid_bombs as lb
from . import r2_geometry as rg
from .ext_r2_repair_probes import R2 as R2_RETURNS, checkpoint_candidates, dataset_path

CEILING_SEED = 20261117
SEED = 9000
HELDOUT_STEPS = 8
GATEV_CACHE = R2_RETURNS / "gatev_ceiling/cache"
DET_COLORS = {"wcte": "#7f7f7f", "sk": "#1f77b4", "juno": "#2ca02c", "hk": "#d62728"}


def trace_spheres(geometry, origins, directions, *, chunk: int = 1024, eps: float = 1e-6):
    """Straight-ray first forward hit on the PMT spheres (radius ``sensor_radius_m``, centred on
    the PMT positions) with the entry point inside the detector volume; no absorption, no
    reflection — the semantics of the pinned ``PromptTransport``.  Returns (photon index,
    sensor index, distance) of the photons that hit.  Validated against the cached LUCiD
    transport of the SK/JUNO/WCTE Gate V events before use on HK."""
    c = np.asarray(geometry.positions_m, float)
    r2 = float(geometry.sensor_radius_m) ** 2
    o_all, d_all = np.asarray(origins, float), np.asarray(directions, float)
    d_all = d_all / np.linalg.norm(d_all, axis=1, keepdims=True)
    R, H = float(geometry.radius_m), float(geometry.half_height_m)
    sphere = getattr(geometry, "shape", rg.SHAPE_CYLINDER) == rg.SHAPE_SPHERE
    out_i, out_s, out_t = [], [], []
    for lo in range(0, o_all.shape[0], chunk):
        o, d = o_all[lo:lo + chunk], d_all[lo:lo + chunk]
        oc = c[None, :, :] - o[:, None, :]                       # (n, P, 3)
        b = np.einsum("npk,nk->np", oc, d)
        c2 = np.einsum("npk,npk->np", oc, oc) - b * b
        disc = r2 - c2
        ok = (disc >= 0) & (b > 0)
        t = np.where(ok, b - np.sqrt(np.where(ok, disc, 0.0)), np.inf)
        t = np.where(t > 0, t, np.inf)
        # entry point inside the volume
        n_i, p_i = np.nonzero(np.isfinite(t))
        if n_i.size:
            pts = o[n_i] + t[n_i, p_i][:, None] * d[n_i]
            if sphere:
                inside = np.linalg.norm(pts, axis=1) <= R * (1 + eps)
            else:
                inside = (np.hypot(pts[:, 0], pts[:, 1]) <= R * (1 + eps)) & (np.abs(pts[:, 2]) <= H * (1 + eps))
            t[n_i[~inside], p_i[~inside]] = np.inf
        first = np.argmin(t, axis=1)
        tf = t[np.arange(t.shape[0]), first]
        hit = np.isfinite(tf)
        out_i.append(lo + np.flatnonzero(hit))
        out_s.append(first[hit])
        out_t.append(tf[hit])
    return np.concatenate(out_i), np.concatenate(out_s), np.concatenate(out_t)


def trace_event(detector: str, name: str, cfg, *, fraction: float = 1.0):
    """A Gate V holdout event through :func:`trace_spheres` (ValidationEvent, as the cache)."""
    from types import SimpleNamespace
    from . import cherenkov_validation as cv
    from . import ext_r2 as x
    geometry = rg.load_r2_geometry(detector)
    index = list(x.GATE_V_HOLDOUT).index(name)
    with np.load(Path(x.PHOTONSIM_DIR) / name, allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    raw = cv.contain_event(raw, geometry)
    keep = np.ones(np.asarray(raw["photon_id"]).size, bool)
    if fraction < 1.0:
        rng = lb.keyed_rng(CEILING_SEED, "r2_diag/gatev_subsample", detector, name)
        keep = rng.random(keep.size) < float(fraction)
    o = np.asarray(raw["origins_m"], float)[keep]
    d = np.asarray(raw["directions"], float)[keep]
    i, sid, dist = trace_spheres(geometry, o, d)
    t = np.asarray(raw["emission_ns"], float)[keep][i] + dist * float(cfg.n_group) / (lb.C_M_PER_S * 1e-9)
    kept = SimpleNamespace(sensor_id=sid, arrival_time_ns=t,
                           wavelength_nm=np.asarray(raw["wavelength_nm"], float)[keep][i])
    return cv.event_from_transport(raw, kept, event_id=index)


def heldout_objective(model, geometry, scaling, cohort, micro_batch: int, *,
                      steps: int = HELDOUT_STEPS) -> dict:
    """The training objective on held-out (probe-test) batches drawn as in training (A6.3)."""
    import torch
    from . import ext_r2 as x
    from . import lucid_run as lr
    from . import r2_store as rs
    cfg = x.run_config()
    loader = rs.R2TrainingLoader(cohort, geometry, seed=SEED, scaling=scaling,
                                 n_min=int(cfg["views"]["n_min"]), batch=x.BATCH)
    rows = []
    pooled = []
    if getattr(model, "sigreg", None) is not None:
        model.sigreg.sliced.global_step.zero_()
    for step in range(int(steps)):
        batch, _ = loader.batch_at(10_000 + step)
        n = int(batch.features.size(0))
        for b0 in range(0, n, int(micro_batch)):
            micro = lr._slice(batch, b0, min(n, b0 + int(micro_batch)))
            with torch.no_grad():
                out = model.forward(micro)
            rows.append({k: float(v) for k, v in out["metrics"].items()})
            if "context_tokens" in out:
                pooled.append(model._context_pooled(out["context_tokens"],
                                                    micro.context_index).cpu())
    keys = sorted({k for r in rows for k in r})
    res = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    res["n_micro_batches"] = len(rows)
    res["micro_batch"] = int(micro_batch)
    if pooled and getattr(model, "sigreg", None) is not None:
        allp = torch.cat(pooled)
        sl = model.sigreg.sliced
        big = {}
        for size in (4, 8, 32, int(allp.size(0))):
            vals = []
            for b0 in range(0, int(allp.size(0)) - size + 1, size):
                sl.global_step.zero_()
                vals.append(float(sl(allp[b0:b0 + size])))
            big[str(size)] = float(np.mean(vals))
        res["sigreg_context_by_batch_size"] = big
    return res


def bugcheck(detector: str, out: Path, *, model_loader) -> dict:
    """A6.1 and A6.4 on real batches: fixed-label vs position-free code paths with identical
    weights, the label mask, the predictor's queries, and leak tests of both masked passes."""
    import difflib
    import inspect
    import torch
    from . import ext_r2 as x
    from . import ext_shortcut as es
    from . import lucid_arms as la
    from . import lucid_run as lr
    from . import r2_store as rs
    torch.manual_seed(0)
    fixed, geometry, rule, scaling, meta_f = model_loader(detector, "sigreg_jepa")
    posfree, _, _, _, meta_p = model_loader(detector, "target_posfree")
    cfg = x.run_config()
    cohort = rs.open_cohort(dataset_path(detector), "dev_probe_test")
    loader = rs.R2TrainingLoader(cohort, geometry, seed=SEED, scaling=scaling,
                                 n_min=int(cfg["views"]["n_min"]), batch=x.BATCH)
    batch = lr._slice(loader.batch_at(10_000)[0], 0, 4)
    res = {"detector": detector, "checkpoints": {"fixed": meta_f["checkpoint"],
                                                 "posfree": meta_p["checkpoint"]}}
    # (1) source diff of the two label passes
    a = inspect.getsource(la.LucidArm._encode_views).splitlines()
    b = inspect.getsource(es._encode_views).splitlines()
    res["source_diff"] = [l for l in difflib.unified_diff(a, b, "lucid_arms.LucidArm._encode_views",
                                                         "ext_shortcut._encode_views", lineterm="", n=0)]
    res["position_columns"] = [es.POSITION_COLUMNS.start, es.POSITION_COLUMNS.stop]
    res["state_dict_keys_equal"] = sorted(fixed.state_dict()) == sorted(posfree.state_dict())
    res["encoder_config_equal"] = fixed.jepa.config == posfree.jepa.config
    res["settings_equal"] = repr(fixed.settings) == repr(posfree.settings)
    # identical weights in both arms: only the bound label pass may differ
    fixed.load_state_dict(posfree.state_dict())
    with torch.no_grad():
        cf, pf_, lf, hf, ff = fixed._encode_views(batch)
        cp, pp, lp, hp, fp = posfree._encode_views(batch)
        same = lambda u, v: bool(torch.equal(u, v))
        res["same_weights_outputs"] = {"context": same(cf, cp), "prediction": same(pf_, pp),
                                       "full": same(ff, fp), "hidden_index": same(hf, hp),
                                       "label_differs": not same(lf, lp)}
        zf = batch.features.clone()
        zf[..., 0:6] = 0.0
        gather = hp.unsqueeze(-1).expand(-1, -1, posfree.d_model)
        manual_p = posfree.jepa.encode(zf, hidden_mask=~batch.target).gather(1, gather)
        manual_f = fixed.jepa.encode(batch.features, hidden_mask=~batch.target).gather(1, gather)
        res["label_pass"] = {"posfree_equals_zeroed_cols_0_5_with_hidden_not_target": same(manual_p, lp),
                             "fixed_equals_hidden_not_target": same(manual_f, lf)}
        mf, mp = fixed.forward(batch)["metrics"], posfree.forward(batch)["metrics"]
        res["forward_metric_keys_equal"] = sorted(mf) == sorted(mp)
        # (4) predictor queries are the scaled target positions; hidden == target (natural occupancy)
        tgt = [torch.nonzero(batch.target[i]).flatten() for i in range(batch.target.size(0))]
        res["hidden_equals_target"] = same(batch.hidden, batch.target)
        res["hidden_index_is_target_index"] = all(same(hp[i], tgt[i]) for i in range(len(tgt)))
        geo = torch.as_tensor(np.asarray(batch.features[0, :, :3]))
        res["query_features_are_scaled_positions"] = bool(torch.allclose(
            posfree.jepa.query_features.to(geo.dtype), geo, atol=1e-6))
        res["context_disjoint_from_hidden"] = all(
            not bool(batch.hidden[i, batch.context_index[i]].any()) for i in range(batch.target.size(0)))
        # leak tests: hidden features must not reach the context pass; non-target features must
        # not reach the label pass
        noisy = batch.features.clone()
        noisy[batch.hidden] = torch.randn_like(noisy[batch.hidden]) * 5
        c2 = posfree.jepa.encode(noisy, hidden_mask=batch.hidden)
        gi = batch.context_index.unsqueeze(-1).expand(-1, -1, posfree.d_model)
        res["context_pass_ignores_hidden_features_max_abs"] = float(
            (c2.gather(1, gi) - cp.gather(1, gi)).abs().max())
        noisy2 = batch.features.clone()
        noisy2[~batch.target] = torch.randn_like(noisy2[~batch.target]) * 5
        l2 = posfree.jepa.encode(es.posfree_features(noisy2), hidden_mask=~batch.target).gather(1, gather)
        res["label_pass_ignores_non_target_features_max_abs"] = float((l2 - lp).abs().max())
        res["n_target_tokens"] = int(hp.numel())
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{detector}__bugcheck.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float), flush=True)
    return res


def _cyl_coords(geometry, pos):
    R, H = float(geometry.radius_m), float(geometry.half_height_m)
    surf = np.asarray(geometry.surfaces).astype(str)
    x, y = np.zeros(len(pos)), np.zeros(len(pos))
    b = surf == "barrel"
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    x[b], y[b] = R * phi[b], pos[b, 2]
    t = surf == "top"
    x[t], y[t] = pos[t, 0], H + R + pos[t, 1]
    m = surf == "bottom"
    x[m], y[m] = pos[m, 0], -H - R - pos[m, 1]
    return x, y


def _display(ax, geometry, values, hit, title, cmap, vmin=None, vmax=None):
    pos = np.asarray(geometry.positions_m)
    s = 1.0 if geometry.n_pmt > 5000 else 4.0
    if geometry.shape == rg.SHAPE_SPHERE:
        lon = np.arctan2(pos[:, 1], pos[:, 0])
        lat = np.arcsin(np.clip(pos[:, 2] / np.linalg.norm(pos, axis=1), -1, 1))
        ax.scatter(lon[~hit], lat[~hit], s=s * 0.5, c="#e6e6e6", lw=0)
        sc = ax.scatter(lon[hit], lat[hit], s=s, c=values[hit], cmap=cmap, vmin=vmin, vmax=vmax, lw=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        x, y = _cyl_coords(geometry, pos)
        ax.scatter(x[~hit], y[~hit], s=s * 0.5, c="#e6e6e6", lw=0)
        sc = ax.scatter(x[hit], y[hit], s=s, c=values[hit], cmap=cmap, vmin=vmin, vmax=vmax, lw=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    return sc


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "figure.dpi": 110})
    return plt


def _tank(ax, geometry, view):
    import matplotlib.patches as mp
    R, H = float(geometry.radius_m), float(geometry.half_height_m)
    if geometry.shape == rg.SHAPE_SPHERE or view == "top":
        ax.add_patch(mp.Circle((0, 0), R, fill=False, lw=0.8))
        ax.add_patch(mp.Circle((0, 0), 0.8 * R, fill=False, lw=0.5, ls=":"))
        ax.set_xlim(-1.05 * R, 1.05 * R)
        ax.set_ylim(-1.05 * R, 1.05 * R)
    else:
        ax.add_patch(mp.Rectangle((-R, -H), 2 * R, 2 * H, fill=False, lw=0.8))
        ax.add_patch(mp.Rectangle((-0.8 * R, -0.8 * H), 1.6 * R, 1.6 * H, fill=False, lw=0.5, ls=":"))
        ax.set_xlim(-1.05 * R, 1.05 * R)
        ax.set_ylim(-1.05 * H, 1.05 * H)
    ax.set_aspect("equal")


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0)
    y = y - y.mean(0)
    num = np.linalg.norm(x.T @ y, "fro") ** 2
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(num / den) if den > 0 else float("nan")


def training_logs(out: Path, summary: dict) -> None:
    plt = _plt()
    runs = []
    for root in (R2_RETURNS / "runs", R2_RETURNS / "batch/runs"):
        if root.is_dir():
            runs += sorted(p for p in root.iterdir() if (p / "train_log.jsonl").is_file())
    for arm in ("sigreg_jepa", "target_posfree"):
        for checkpoint in checkpoint_candidates("wcte", arm, SEED):
            if (checkpoint.parent / "train_log.jsonl").is_file():
                runs.append(checkpoint.parent)
    keys_seen = set()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    last = {}
    for p in runs:
        lines = [json.loads(x) for x in (p / "train_log.jsonl").read_text().splitlines() if x.strip()]
        steps = np.array([l.get("step", 0) for l in lines])
        met = [l.get("metrics", l) for l in lines]
        keys = sorted({k for m in met for k in m if isinstance(m.get(k), (int, float))})
        keys_seen |= set(keys)
        name = p.name if "dev_s005" not in p.name else f"wcte__{'posfree' if 'shortcut' in str(p) else 'sigreg_jepa'}__s9000"
        det = name.split("__")[0]
        arm = name.split("__")[1]
        last[name] = {k: met[-1].get(k) for k in keys}
        col = DET_COLORS.get(det, "k")
        ls = "-" if name.endswith("s9000") else ("--" if name.endswith("s9001") else ":")
        if arm in ("sigreg_jepa", "posfree", "target_posfree"):
            row = 0 if arm == "sigreg_jepa" else 1
            for c, k in enumerate(("prediction", "sigreg", "sigreg_context", "sigreg_target")):
                if k in keys:
                    axes[row, c].plot(steps, [m.get(k, np.nan) for m in met], ls, color=col, lw=0.7,
                                      label=name)
                    axes[row, c].set_title(f"{'fixed-label' if row == 0 else 'position-free'}: {k}")
                    axes[row, c].set_yscale("log")
                    axes[row, c].set_xlabel("step")
    axes[0, 0].legend(fontsize=5)
    axes[1, 0].legend(fontsize=5)
    fig.suptitle("A6.2 training logs of every round-2 JEPA run (+ WCTE round 1); colour = detector, "
                 "line = seed (9000 solid, 9001 dashed, 9002 dotted)")
    fig.tight_layout()
    fig.savefig(out / "A6_training_logs_jepa.png", bbox_inches="tight")
    plt.close(fig)
    # MAE loss terms
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for p in runs:
        if "mae" not in p.name:
            continue
        lines = [json.loads(x) for x in (p / "train_log.jsonl").read_text().splitlines() if x.strip()]
        steps = np.array([l.get("step", 0) for l in lines])
        met = [l.get("metrics", l) for l in lines]
        det = p.name.split("__")[0]
        ls = "-" if "rf" not in p.name else ":"
        for c, k in enumerate(("loss", "charge", "hit", "time")):
            if k in met[-1]:
                axes[c].plot(steps, [m.get(k, np.nan) for m in met], ls, color=DET_COLORS[det], lw=0.7,
                             label=p.name)
                axes[c].set_title(f"MAE: {k}")
                axes[c].set_yscale("log")
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(out / "A6_training_logs_mae.png", bbox_inches="tight")
    plt.close(fig)
    summary["A6_logs"] = {"logged_metric_keys": sorted(keys_seen), "last_logged": last,
                          "rank_or_variance_metric_logged": any(("rank" in k or "var" in k or "std" in k)
                                                                for k in keys_seen)}
