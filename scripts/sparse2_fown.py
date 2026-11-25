#!/usr/bin/env python3
"""r2-sparse2 §2.4 registered diagnostic: own-block mass fraction f_own and its chance level (PREREG appendix B).

For S2 (reused, read-only), S2L and S2Lt x seeds 9000/9001/9002 at step 10,000, on ``dev_probe_test`` (validation
only; the r2-sparse diagnostic event set: the first 8 events of every run, 1,000 events), each model with its own
training mask (keyed 32-hit blocks or 5 ns bins). Sums of ``SparseTimeArm.block_mass_stats`` over the events:

* f_own = mean over queries of own / (mass on all hidden hits)  (P6 statistic);
* chance = mean over queries of n_own_block / n_hidden;  ratio = f_own / chance;
* also: own / total mass (the r2-sparse "8 %" statistic), mass on hidden hits, on unhit PMTs, and the literal
  mean-over-blocks chance.

Exploratory: one figure with the summed predicted mass of S2 and S2Lt (seed 9000) for 6 photon bombs.

    sparse2_fown.py --s2-root <r2-sparse train/runs> --new-root <r2-sparse2 train/runs> --out FOWN.json [--fig F.png]
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
from jepa_noise.lucid_views import ViewMasker  # noqa: E402

RUNS = {"S2": ("s2", "wcte__sparse_time__s{s}__dw2"), "S2L": ("new", "wcte__sparse_time__s{s}__dw2__elin__dfull"),
        "S2Lt": ("new", "wcte__sparse_time__s{s}__dw2__elin__dfull__bt5")}
SEEDS = {"A": 9000, "B": 9001, "C": 9002}
KEYS = ("own", "other", "unhit", "queries", "chance_block_sum", "n_blocks", "chance_query_sum", "own_over_hidden_sum")


def load(run_dir, step=10000):
    import tier1_eval as te
    arm, geometry, scaling, meta = te.load_model(Namespace(legacy_init=False, run_dir=str(run_dir), step=step))
    if meta["data_kind"] != dr.LUCID_V1_W2:
        raise ValueError("expected a lucid_v1_w2 run")
    return arm.eval(), geometry, scaling, meta


def event_batches(geometry, scaling, n_events=1000, photons_only=False, per_run=8):
    pack = dr.pack("wcte", dr.LUCID_V1_W2)
    dr.check_pack_kind(pack, dr.LUCID_V1_W2)
    cohort = rs.open_cohort(pack, "dev_probe_test")
    masker = ViewMasker(geometry, "spatial_patch", seed=lb.DEFAULT_SEED, n_min=0)
    done = 0
    for i in range(cohort.n_runs):
        truth = cohort.run_truth(i)
        ids = truth["event_ids"]
        if photons_only:
            ids = ids[np.asarray(truth["domain"]).astype(str) == "photon"]
        ids = ids[:per_run]
        if not ids.size:
            continue
        yield rs.tokenise_sparse(cohort.run_store(i), geometry, event_ids=ids, scaling=scaling, masker=masker)
        done += len(ids)
        if done >= n_events:
            return


def stats(arm, geometry, scaling):
    tot = dict.fromkeys(KEYS, 0.0)
    n_ev = 0
    for b in event_batches(geometry, scaling):
        st = arm.block_mass_stats(b)
        for k in KEYS:
            tot[k] += st[k]
        n_ev += len(b.event_ids)
    q = max(tot["queries"], 1.0)
    f_own = tot["own_over_hidden_sum"] / q
    chance = tot["chance_query_sum"] / q
    return {"n_events": n_ev, "n_queries": int(tot["queries"]), "f_own": f_own, "chance": chance,
            "ratio": f_own / chance if chance > 0 else None,
            "own_over_total": tot["own"] / q, "mass_on_hidden_hits": (tot["own"] + tot["other"]) / q,
            "mass_on_unhit": tot["unhit"] / q,
            "chance_mean_over_blocks": tot["chance_block_sum"] / max(tot["n_blocks"], 1.0),
            "ratio_total_mass_vs_block_chance": (tot["own"] / q) / (tot["chance_block_sum"] / max(tot["n_blocks"], 1.0)),
            "sums": tot}


def figure(arms, geometry, scaling, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = np.asarray(geometry.positions_m)
    phi, z = np.arctan2(pos[:, 1], pos[:, 0]), pos[:, 2]
    b = next(event_batches(geometry, scaling, 6, photons_only=True, per_run=6))
    n = min(6, len(b.event_ids))
    fig, ax = plt.subplots(n, len(arms), figsize=(4.4 * len(arms), 2.4 * n), squeeze=False)
    for col, (cell, arm) in enumerate(arms.items()):
        with torch.no_grad():
            p = arm.predictor_inputs(b)
            lam = arm.assignment_mass(arm.run_predictor(p), p).sum(1).numpy()
        for r in range(n):
            hid = np.zeros(geometry.n_pmt, bool)
            hid[p["q_pmt"][r][p["q_valid"][r]].numpy()] = True
            ax[r, col].scatter(phi, z, c=lam[r], s=4, cmap="viridis")
            ax[r, col].scatter(phi[hid], z[hid], s=1, c="r")
            ax[r, col].set_title(f"{cell} summed mass (red: hidden hits), event {b.event_ids[r]}", fontsize=7)
            ax[r, col].set_xticks([]), ax[r, col].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return [int(e) for e in b.event_ids[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s2-root", type=Path, required=True)
    ap.add_argument("--new-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig", type=Path)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    root = {"s2": a.s2_root, "new": a.new_root}
    out = {"cohort": "dev_probe_test (lucid_v1_w2, validation only)", "step": 10000,
           "definition": "PREREG_2026-11-25_R2_SPARSE2.md appendix B", "cells": {}, "missing": [],
           "direct_indirect_split": "not available: the LUCiD cache keeps only a physics flag per digit (r2-sparse T7.4)"}
    fig_arms = {}
    for cell, (which, pat) in RUNS.items():
        out["cells"][cell] = {}
        for lab, s in SEEDS.items():
            d = root[which] / pat.format(s=s)
            if not (d / "snapshot_step_10000.pt").is_file():
                out["missing"].append(str(d))
                continue
            arm, geometry, scaling, meta = load(d)
            res = stats(arm, geometry, scaling)
            res.update(run=str(d), checkpoint_sha256=meta["checkpoint_sha256"])
            out["cells"][cell][lab] = res
            print(cell, lab, json.dumps({k: res[k] for k in ("f_own", "chance", "ratio", "own_over_total")}), flush=True)
            if lab == "A" and cell in ("S2", "S2Lt"):
                fig_arms[cell] = arm
    if a.fig and len(fig_arms) == 2:
        out["figure"] = {"path": a.fig.name, "events": figure(fig_arms, geometry, scaling, a.fig)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
