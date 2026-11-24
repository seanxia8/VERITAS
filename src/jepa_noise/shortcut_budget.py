"""r2-mask M1 (runbook 2026-11-23 §1.2): network-free shortcut budget of a mask.

Targets and scalings are those of the default ``mae_repaired`` loss
(``lucid_arms._forward_mae_mse``): per hidden PMT the hit flag h, the charge channel
c = ln(max(q, 1e-6 PE) / 1 PE) for hit PMTs and 0 for unhit ones (both scored on every hidden
PMT), and the time t / 50 ns scored on hidden *hit* PMTs only; unit weights.

* **Static predictor S**: each PMT's mean target over the fitting events (time: mean over the
  events in which the PMT is hit; the global hit-time mean if it is never hit).
* **Local predictor L**: one Ridge regression per channel, shared across PMTs, from the same
  event's k nearest *visible* PMTs (their h, c, t and distances, nearest first) plus the PMT's
  static means of the three channels.
* **Explained fraction** E = 1 - MSE / Var on the scoring events, per channel and combined as
  1 - sum(MSE) / sum(Var) (the loss's unit weights).
* **Budget**: static = E_S, local = E_L - E_S, remainder U = 1 - E_L.
* **AvgDist**: mean distance from a hidden PMT to its nearest visible PMT (metres and in units of
  the median nearest-neighbour spacing of the array).
* **Correlation length**: E_1(d) = mean over PMT pairs at distance d of the squared correlation
  (across events) of their charge channels; l_c is where E_1 falls to half its value in the
  nearest-neighbour bin (linear interpolation).

No neural network, no physics: numpy, scipy's cKDTree and scikit-learn's Ridge only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

CHANNELS = ("charge", "hit", "time")
TIME_UNIT_NS = 50.0
CHARGE_FLOOR_PE = 1e-6
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)


@dataclass
class Targets:
    """Dense per-event targets, shape (E, N): hit flag, charge channel, time / 50 ns (0 if unhit)."""

    hit: np.ndarray
    charge: np.ndarray
    time: np.ndarray

    def subset(self, rows) -> "Targets":
        return Targets(self.hit[rows], self.charge[rows], self.time[rows])

    @property
    def n_events(self) -> int:
        return int(self.hit.shape[0])


def targets_from_store(store, indices, n_pmt: int) -> Targets:
    """Targets of the events ``indices`` of a sparse hit store (``lucid_bombs.SparseHitStore``)."""
    indices = np.asarray(indices, dtype=np.int64)
    hit = np.zeros((indices.size, n_pmt), np.float32)
    charge = np.zeros_like(hit)
    time = np.zeros_like(hit)
    for row, i in enumerate(indices):
        span = slice(int(store.offsets[i]), int(store.offsets[i + 1]))
        pmt = np.asarray(store.pmt_id[span], dtype=np.int64)
        hit[row, pmt] = 1.0
        charge[row, pmt] = np.log(np.maximum(np.asarray(store.charge_pe[span], np.float64),
                                             CHARGE_FLOOR_PE))
        time[row, pmt] = np.asarray(store.time_ns[span], np.float64) / TIME_UNIT_NS
    return Targets(hit, charge, time)


def static_means(fit: Targets) -> np.ndarray:
    """(N, 3) per-PMT means of (charge, hit, time | hit) over the fitting events."""
    n_hit = fit.hit.sum(0)
    t_hit = (fit.time * fit.hit).sum(0)
    global_t = float(t_hit.sum() / max(n_hit.sum(), 1.0))
    time = np.where(n_hit > 0, t_hit / np.maximum(n_hit, 1.0), global_t)
    return np.stack([fit.charge.mean(0), fit.hit.mean(0), time], axis=1).astype(np.float64)


def median_nn_spacing(positions) -> float:
    d, _ = cKDTree(positions).query(positions, k=2)
    return float(np.median(d[:, 1]))


def _event_rows(positions, targets: Targets, row: int, hidden: np.ndarray, static, k: int,
                pick=None):
    """Features (len(pick), 4k + 3), labels (len(pick), 3), hit mask and nearest distance for the
    hidden PMTs ``pick`` (default: all hidden PMTs) of one event."""
    hidden_idx = np.flatnonzero(hidden)
    visible_idx = np.flatnonzero(~hidden)
    if pick is not None:
        hidden_idx = hidden_idx[pick]
    kk = min(int(k), visible_idx.size)
    dist, nb = cKDTree(positions[visible_idx]).query(positions[hidden_idx], k=kk)
    dist = dist.reshape(len(hidden_idx), kk)
    nb = visible_idx[nb.reshape(len(hidden_idx), kk)]
    h, c, t = targets.hit[row], targets.charge[row], targets.time[row]
    feats = np.concatenate([h[nb], c[nb], t[nb], dist, static[hidden_idx]], axis=1)
    labels = np.stack([c[hidden_idx], h[hidden_idx], t[hidden_idx]], axis=1)
    return feats, labels, h[hidden_idx] > 0.5, dist[:, 0]


def _fit_ridge(x, y):
    from sklearn.linear_model import RidgeCV
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    model = RidgeCV(alphas=ALPHAS).fit((x - mu) / sd, y)
    return model, mu, sd


def shortcut_budget(positions, fit: Targets, score: Targets, hidden_fit, hidden_score, *,
                    k: int = 8, rows_per_fit_event: int = 400, seed: int = 0,
                    spacing: float | None = None) -> dict:
    """The §1.2 budget of one mask configuration.

    ``hidden_fit`` / ``hidden_score`` are boolean (E, N) arrays of the hidden PMTs of each event.
    The static means come from all fitting events; L is fitted on up to ``rows_per_fit_event``
    hidden PMTs per fitting event (time: hidden hit PMTs only) and scored on every hidden PMT of
    the scoring events.
    """
    positions = np.asarray(positions, np.float64)
    hidden_fit, hidden_score = np.asarray(hidden_fit, bool), np.asarray(hidden_score, bool)
    static = static_means(fit)
    rng = np.random.default_rng(seed)
    xs, ys, hs = [], [], []
    for e in range(fit.n_events):
        n_hidden = int(hidden_fit[e].sum())
        pick = None
        if n_hidden > rows_per_fit_event:
            # keep every hidden hit PMT (time rows are scarce) plus a uniform subset of the rest
            hidden_idx = np.flatnonzero(hidden_fit[e])
            is_hit = fit.hit[e, hidden_idx] > 0.5
            hit_pos = np.flatnonzero(is_hit)
            if hit_pos.size > rows_per_fit_event:
                hit_pos = rng.choice(hit_pos, rows_per_fit_event, replace=False)
            rest = np.flatnonzero(~is_hit)
            n_rest = min(rest.size, rows_per_fit_event)
            pick = np.sort(np.concatenate([hit_pos, rng.choice(rest, n_rest, replace=False)]))
        x, y, h, _ = _event_rows(positions, fit, e, hidden_fit[e], static, k, pick)
        xs.append(x), ys.append(y), hs.append(h)
    x, y, h = np.concatenate(xs), np.concatenate(ys), np.concatenate(hs)
    # charge and hit rows: re-weight so that hit/unhit proportions match the hidden population
    models = {}
    hit_rate = float((fit.hit * hidden_fit).sum() / max(hidden_fit.sum(), 1))
    w = np.where(h, hit_rate / max(h.mean(), 1e-12), (1 - hit_rate) / max(1 - h.mean(), 1e-12))
    for j, name in enumerate(CHANNELS):
        if name == "time":
            if h.sum() < 10:
                models[name] = None
                continue
            m, mu, sd = _fit_ridge(x[h], y[h, j])
        else:
            from sklearn.linear_model import RidgeCV
            mu, sd = x.mean(0), x.std(0)
            sd = np.where(sd > 0, sd, 1.0)
            m = RidgeCV(alphas=ALPHAS).fit((x - mu) / sd, y[:, j], sample_weight=w)
        models[name] = (m, mu, sd)
    # score on every hidden PMT of the scoring events (streaming sums)
    acc = {name: {"n": 0, "sum": 0.0, "sum2": 0.0, "se_s": 0.0, "se_l": 0.0} for name in CHANNELS}
    nn_sum, nn_n = 0.0, 0
    for e in range(score.n_events):
        x, y, h, d0 = _event_rows(positions, score, e, hidden_score[e], static, k)
        nn_sum += float(d0.sum())
        nn_n += d0.size
        s_pred = static[np.flatnonzero(hidden_score[e])]
        for j, name in enumerate(CHANNELS):
            rows = h if name == "time" else slice(None)
            yy = y[rows, j]
            if yy.size == 0:
                continue
            a = acc[name]
            a["n"] += yy.size
            a["sum"] += float(yy.sum())
            a["sum2"] += float((yy ** 2).sum())
            a["se_s"] += float(((yy - s_pred[rows, j]) ** 2).sum())
            if models[name] is None:
                a["se_l"] += float(((yy - s_pred[rows, j]) ** 2).sum())
            else:
                m, mu, sd = models[name]
                a["se_l"] += float(((yy - m.predict((x[rows] - mu) / sd)) ** 2).sum())
    out = {"channels": {}}
    tot_var = tot_s = tot_l = 0.0
    for name in CHANNELS:
        a = acc[name]
        n = max(a["n"], 1)
        var = a["sum2"] / n - (a["sum"] / n) ** 2
        mse_s, mse_l = a["se_s"] / n, a["se_l"] / n
        e_s = 1.0 - mse_s / var if var > 0 else float("nan")
        e_l = 1.0 - mse_l / var if var > 0 else float("nan")
        out["channels"][name] = {"var": var, "mse_S": mse_s, "mse_L": mse_l, "E_S": e_s,
                                 "E_L": e_l, "n_rows": a["n"],
                                 "alpha": (None if models[name] is None
                                           else float(models[name][0].alpha_))}
        if var > 0:
            tot_var, tot_s, tot_l = tot_var + var, tot_s + mse_s, tot_l + mse_l
    e_s, e_l = 1.0 - tot_s / tot_var, 1.0 - tot_l / tot_var
    spacing = median_nn_spacing(positions) if spacing is None else float(spacing)
    avg = nn_sum / max(nn_n, 1)
    out.update(E_S=e_s, E_L=e_l, static=e_s, local=e_l - e_s, U=1.0 - e_l,
               avg_dist_m=avg, avg_dist_spacing=avg / spacing, nn_spacing_m=spacing,
               hidden_fraction=float(hidden_score.mean()),
               never_hidden_fraction=float((~hidden_score.any(0)).mean()))
    return out


def correlation_length(positions, targets: Targets, *, n_anchor: int = 1500, k_near: int = 48,
                       n_far: int = 48, n_bins: int = 30, seed: int = 0) -> dict:
    """l_c of the charge channel: E_1(d) = mean squared across-event correlation of PMT pairs at
    distance d; l_c = distance at which E_1 falls to half its nearest-neighbour-bin value."""
    positions = np.asarray(positions, np.float64)
    n = len(positions)
    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, min(n_anchor, n), replace=False)
    tree = cKDTree(positions)
    _, near = tree.query(positions[anchors], k=min(k_near + 1, n))
    pairs = []
    for a, row in zip(anchors, near):
        partners = np.concatenate([row[1:], rng.choice(n, n_far, replace=False)])
        partners = partners[partners != a]
        pairs.append(np.stack([np.full(partners.size, a), partners], axis=1))
    pairs = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)
    d = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    c = targets.charge.astype(np.float64)
    c = c - c.mean(0)
    sd = c.std(0)
    ok = (sd[pairs[:, 0]] > 0) & (sd[pairs[:, 1]] > 0)
    pairs, d = pairs[ok], d[ok]
    cov = (c[:, pairs[:, 0]] * c[:, pairs[:, 1]]).mean(0)
    r2 = (cov / (sd[pairs[:, 0]] * sd[pairs[:, 1]])) ** 2
    spacing = median_nn_spacing(positions)
    edges = np.geomspace(0.5 * spacing, max(d.max(), spacing) * 1.0001, n_bins + 1)
    idx = np.digitize(d, edges) - 1
    centres, e1, counts = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 20:
            centres.append(float(np.sqrt(edges[b] * edges[b + 1])))
            e1.append(float(r2[sel].mean()))
            counts.append(int(sel.sum()))
    centres, e1 = np.asarray(centres), np.asarray(e1)
    # nearest-neighbour value: the bin containing the median spacing (first populated bin otherwise)
    ref_bin = int(np.argmin(np.abs(np.log(centres / spacing))))
    ref = float(e1[ref_bin])
    l_c = float("nan")
    for b in range(ref_bin, len(e1) - 1):
        if e1[b + 1] <= ref / 2 < e1[b] or e1[b + 1] <= ref / 2 <= e1[b]:
            f = (e1[b] - ref / 2) / (e1[b] - e1[b + 1])
            l_c = float(centres[b] + f * (centres[b + 1] - centres[b]))
            break
    return {"l_c_m": l_c, "l_c_spacing": l_c / spacing, "E1_nn": ref, "nn_spacing_m": spacing,
            "curve": {"d_m": centres.tolist(), "E1": e1.tolist(), "n_pairs": counts}}
