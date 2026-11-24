#!/usr/bin/env python3
"""r2-sparse T7.4 diagnostics (exploratory), on the validation cohort (dev_probe_test, lucid_v1_w2) only.

* S2 predicted-mass maps for 6 photon bombs (one figure) and S1 hit-probability maps against truth for
  the same events (training masks: spatial patch for S1, keyed time blocks for S2);
* count-head calibration (C3, S1, S1n, S1p): predicted vs true number of masked hits;
* S2: fraction of predicted mass on unhit PMTs, and on hidden hits of the query's own block / other blocks.
  LUCiD's direct/indirect flag is not in the cache (digits carry only a physics flag), so the split by it
  is not available.

    sparse_diagnostics.py --train-root <returns>/train/runs --seed 9000 --out-dir results/round2/20261124_sparse
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402
from jepa_noise import sparse_arms as sa  # noqa: E402
from jepa_noise.lucid_views import ViewMasker  # noqa: E402
from jepa_noise.point_encoder import HIT_INDEX  # noqa: E402

RUN = {"C3": "wcte__mae_repaired__s{s}__dw2__lfixed__psent__efourier__disol__cnt",
       "S1": "wcte__sparse_mae__s{s}__dw2__qall", "S1n": "wcte__sparse_mae__s{s}__dw2__qneg",
       "S1p": "wcte__sparse_mae__s{s}__dw2__qpos", "S2": "wcte__sparse_time__s{s}__dw2"}


def load(train_root, rid, step=10000):
    import tier1_eval as te
    arm, geometry, scaling, meta = te.load_model(Namespace(legacy_init=False, run_dir=str(Path(train_root) / rid),
                                                          step=step))
    if meta["data_kind"] != dr.LUCID_V1_W2:
        raise ValueError("diagnostics expect lucid_v1_w2 runs")
    return arm.eval(), geometry, scaling


def batches(geometry, scaling, n_events, photons_only):
    pack = dr.pack("wcte", dr.LUCID_V1_W2)
    dr.check_pack_kind(pack, dr.LUCID_V1_W2)
    cohort = rs.open_cohort(pack, "dev_probe_test")
    masker = ViewMasker(geometry, "spatial_patch", seed=lb.DEFAULT_SEED, n_min=0)
    done = 0
    for i in range(cohort.n_runs):
        truth = cohort.run_truth(i)
        dom = np.asarray(truth["domain"]).astype(str)
        ids = truth["event_ids"][dom == "photon"] if photons_only else truth["event_ids"]
        if not ids.size:
            continue
        b = rs.tokenise_sparse(cohort.run_store(i), geometry, event_ids=ids[:8], scaling=scaling, masker=masker)
        yield b
        done += len(b.event_ids)
        if done >= n_events:
            return


def count_calibration(arm, geometry, scaling, n_events=1000):
    pred, true = [], []
    for b in batches(geometry, scaling, n_events, photons_only=False):
        with torch.no_grad():
            x, hidden = b.features, b.hidden
            hit = x[..., HIT_INDEX] > 0.5
            if isinstance(arm, sa.SparseMaeArm):
                z, _, valid = arm.encode_hits(x, hit & ~hidden, "hilbert")
                rate = arm.count_rate(z, valid)
            else:
                ctx = arm.jepa.encode(arm._encoder_features(x), hidden_mask=hidden)
                rate = arm.count_head(arm._context_pooled(ctx, b.context_index))[:, 0]
        pred.append(torch.exp(rate).numpy())
        true.append((hidden & hit).sum(1).numpy())
    p, t = np.concatenate(pred), np.concatenate(true).astype(float)
    return {"n_events": int(t.size), "median_pred_over_true": float(np.median(p / np.maximum(t, 1))),
            "pearson": float(np.corrcoef(p, t)[0, 1]), "mean_abs_rel_error": float(np.mean(np.abs(p - t) / np.maximum(t, 1))),
            "slope_pred_on_true": float(np.polyfit(t, p, 1)[0])}


def s2_mass(arm, b):
    with torch.no_grad():
        p = arm.predictor_inputs(b)
        z = arm.encoder(p["tokens"], p["v_valid"])
        e = arm.predictor(z, p["v_valid"], p["queries"])
        s = torch.einsum("bqd,nd->bqn", e, arm.pe) / torch.exp(arm.log_temperature)
        s = s.masked_fill(p["vis_sel"][:, None, :], float("-inf"))
        m = torch.softmax(s, -1) * p["q_valid"][..., None]
    return p, m


def s2_stats(arm, geometry, scaling, n_events=1000):
    tot = {"unhit": 0.0, "own_block_hidden": 0.0, "other_block_hidden": 0.0, "queries": 0.0}
    for b in batches(geometry, scaling, n_events, photons_only=False):
        p, m = s2_mass(arm, b)
        hit = p["hit"]
        for row in range(m.size(0)):
            qv = p["q_valid"][row]
            if not qv.any():
                continue
            mm = m[row][qv]
            blk = p["q_block"][row][qv]
            pmt = p["q_pmt"][row][qv]
            tot["queries"] += float(qv.sum())
            tot["unhit"] += float(mm[:, ~hit[row]].sum())
            for k in blk.unique():
                own = pmt[blk == k]
                other = pmt[blk != k]
                tot["own_block_hidden"] += float(mm[blk == k][:, own].sum())
                tot["other_block_hidden"] += float(mm[blk == k][:, other].sum())
    q = max(tot["queries"], 1.0)
    return {"fraction_mass_on_unhit": tot["unhit"] / q, "fraction_mass_on_own_block_hidden_hits": tot["own_block_hidden"] / q,
            "fraction_mass_on_other_block_hidden_hits": tot["other_block_hidden"] / q,
            "direct_indirect_split": "not available: the LUCiD cache keeps only a physics flag per digit",
            "n_queries": int(q)}


def maps(s1, s2, geometry, scaling, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = np.asarray(geometry.positions_m)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    z = pos[:, 2]
    b = next(batches(geometry, scaling, 6, photons_only=True))
    p, m = s2_mass(s2, b)
    lam = m.sum(1).numpy()
    x, hidden = b.features, b.hidden
    hit = (x[..., HIT_INDEX] > 0.5).numpy()
    with torch.no_grad():
        zz, _, valid = s1.encode_hits(x, torch.as_tensor(hit) & ~hidden, "hilbert")
        q_idx, q_valid = sa.gather_tokens(hidden, s1.hilbert_rank[None, :])
        prob = torch.sigmoid(s1.predict_queries(zz, valid, q_idx)[..., 2]).numpy()
    n = min(6, x.size(0))
    fig, ax = plt.subplots(n, 3, figsize=(13, 2.6 * n), squeeze=False)
    for r in range(n):
        hid_hits = np.zeros(geometry.n_pmt, bool)
        qp = p["q_pmt"][r][p["q_valid"][r]].numpy()
        hid_hits[qp] = True
        ax[r, 0].scatter(phi, z, c=lam[r], s=4, cmap="viridis")
        ax[r, 0].scatter(phi[hid_hits], z[hid_hits], s=1, c="r")
        ax[r, 0].set_title(f"S2 mass (red: hidden hits) ev {b.event_ids[r]}", fontsize=7)
        pr = np.full(geometry.n_pmt, np.nan)
        pr[q_idx[r].numpy()] = prob[r]
        ax[r, 1].scatter(phi, z, c=pr, s=4, cmap="magma", vmin=0, vmax=1)
        ax[r, 1].set_title("S1 hit probability (masked PMTs)", fontsize=7)
        tr = np.where(hidden[r].numpy(), hit[r].astype(float), np.nan)
        ax[r, 2].scatter(phi, z, c=tr, s=4, cmap="magma", vmin=0, vmax=1)
        ax[r, 2].set_title("truth: masked PMT hit", fontsize=7)
        for a in ax[r]:
            a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return [int(e) for e in b.event_ids[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = {"cohort": "dev_probe_test (lucid_v1_w2, validation only)", "seed": a.seed, "count_calibration": {}}
    arms = {}
    for cell in RUN:
        rid = RUN[cell].format(s=a.seed)
        if not (Path(a.train_root) / rid / "snapshot_step_10000.pt").is_file():
            out.setdefault("missing", []).append(rid)
            continue
        arms[cell], geometry, scaling = load(a.train_root, rid)
    for cell in ("C3", "S1", "S1n", "S1p"):
        if cell in arms:
            out["count_calibration"][cell] = count_calibration(arms[cell], geometry, scaling)
    if "S2" in arms:
        out["s2_mass"] = s2_stats(arms["S2"], geometry, scaling)
    if "S1" in arms and "S2" in arms:
        out["map_events"] = maps(arms["S1"], arms["S2"], geometry, scaling, a.out_dir / "DIAG_MAPS.png")
    (a.out_dir / "DIAGNOSTICS.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "map_events"}, indent=1))


if __name__ == "__main__":
    main()
