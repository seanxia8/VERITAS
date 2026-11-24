#!/usr/bin/env python3
"""r2-follow §2.4 Part A diagnostics (registered as exploratory, CPU), on dev_probe_test (lucid_v1_w2) only.

For C2, C3e, C2dc, C3 at seed 9000, steps 0 and 10,000 (every event fully visible, as in the frozen readouts):

* token composition at the encoder input: RMS per dimension of the positional part and of the content part.
  Fourier encoders (C3e, C3): positional part = the fixed Fourier code, content part = the content-MLP output.
  Linear encoders (C2, C2dc): the input projection splits exactly into W_geo x_geo (the six geometry features)
  and W_content x_content + b (ln q, tau, hit; bias included);
* block-4 tokens (n_blocks = 4, so the encoder read-out before any final norm): fraction of their variance
  explained out of sample by a linear map of the fixed Fourier code of the PMT (geometry share, 128 regressors,
  the same code for every model), by (ln q, tau, hit) (content share) and by both; all tokens and hit tokens;
* geometry vs event: out-of-sample R^2 of a linear probe from block-4 tokens to the PMT position (scaled x, y, z),
  next to the R^2 of the event vertex through the frozen readout (spatial-moment Ridge, READOUT_SPEC_v1 A.2),
  computed from the evaluator's per-event validation errors and the dev_probe_test truth:
  R^2 = 1 - sum(err^2) / sum(|v - mean v|^2).

Fits use the first ``--n-fit`` photon bombs of dev_probe_test and are scored on the next ``--n-test`` (disjoint).
No sealed pack is opened. Run with ``FOLLOW_CODE_TREE=<760c3d5 tree>`` (FactorJepa is needed to load C3e/C2dc).

    follow_diagnostics.py --out results/round2/20261124_follow/DIAGNOSTICS_A.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

TREE = Path(os.environ.get("FOLLOW_CODE_TREE", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(TREE / "src"))
sys.path.insert(0, str(TREE / "scripts"))
from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402
from jepa_noise import sparse_arms as sa  # noqa: E402
from jepa_noise.lucid_views import ViewMasker  # noqa: E402

SPARSE = Path("/ceph/dwong/jepa/returns/r2_sparse_2026-11-24")
FOLLOW = Path("/ceph/dwong/jepa/returns/r2_follow_2026-11-24")
CELLS = {"C2": (SPARSE, "wcte__mae_repaired__s{s}__dw2__lfixed__psent"),
         "C3e": (FOLLOW, "wcte__mae_repaired__s{s}__dw2__lfixed__psent__efourier"),
         "C2dc": (FOLLOW, "wcte__mae_repaired__s{s}__dw2__lfixed__psent__disol__cnt"),
         "C3": (SPARSE, "wcte__mae_repaired__s{s}__dw2__lfixed__psent__efourier__disol__cnt")}


class Normal:
    """Streaming least squares: train-split normal equations, test-split sufficient statistics."""

    def __init__(self, p, q):
        self.xtx = {s: np.zeros((p, p)) for s in ("fit", "test")}
        self.xty = {s: np.zeros((p, q)) for s in ("fit", "test")}
        self.ysum = {s: np.zeros(q) for s in ("fit", "test")}
        self.yy = {s: 0.0 for s in ("fit", "test")}
        self.n = {s: 0 for s in ("fit", "test")}

    def add(self, split, x, y):
        x, y = x.astype(np.float64), y.astype(np.float64)
        self.xtx[split] += x.T @ x
        self.xty[split] += x.T @ y
        self.ysum[split] += y.sum(0)
        self.yy[split] += float((y ** 2).sum())
        self.n[split] += x.shape[0]

    def r2(self, ridge=1e-6):
        a = self.xtx["fit"]
        b = np.linalg.solve(a + ridge * np.trace(a) / a.shape[0] * np.eye(a.shape[0]), self.xty["fit"])
        xtx, xty, n = self.xtx["test"], self.xty["test"], self.n["test"]
        sse = self.yy["test"] - 2.0 * np.sum(b * xty) + np.sum(b * (xtx @ b))
        sst = self.yy["test"] - float((self.ysum["test"] ** 2).sum()) / n
        return float(1.0 - sse / sst)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def input_parts(arm, x):
    enc = arm.jepa.encoder
    if isinstance(enc, sa.DenseFourierEncoder):
        content = enc.content_tokens(x)
        return enc.pe.expand_as(content), content, "fourier"
    w, b = enc.input_proj.weight, enc.input_proj.bias
    return x[..., :6] @ w[:, :6].T, x[..., 6:9] @ w[:, 6:9].T + b, "linear"


def load(cell, seed, step):
    import tier1_eval as te
    root, pat = CELLS[cell]
    rid = pat.format(s=seed)
    arm, geometry, scaling, meta = te.load_model(Namespace(legacy_init=False, run_dir=str(root / "train/runs" / rid),
                                                          step=step))
    if meta["data_kind"] != dr.LUCID_V1_W2:
        raise ValueError("Part A diagnostics expect lucid_v1_w2 runs")
    return arm.eval(), geometry, scaling, rid, meta


def photon_batches(cohort, geometry, scaling, n_total, chunk=16):
    masker = ViewMasker(geometry, "within", seed=0, n_min=0)
    done = 0
    for i in range(cohort.n_runs):
        truth = cohort.run_truth(i)
        dom = np.asarray(truth["domain"]).astype(str)
        ids = truth["event_ids"][dom == lb.DOMAIN_PHOTON]
        for first in range(0, ids.size, chunk):
            take = ids[first:first + min(chunk, n_total - done)]
            if not take.size:
                return
            yield rs.tokenise_sparse(cohort.run_store(i), geometry, event_ids=take, scaling=scaling, masker=masker)
            done += take.size
            if done >= n_total:
                return


def vertex_r2(base, rid, step, cohort):
    root = base / "eval" / rid / f"step_{step:05d}"
    if not (root / "validation.npz").is_file():
        return None
    with np.load(root / "validation.npz") as z:
        ids, dom, err = z["event_ids"], z["domain"].astype(str), z["err_ridge_spatial_moment_latent"]
    ph_ids = ids[dom == lb.DOMAIN_PHOTON]
    vtx = {}
    for i in range(cohort.n_runs):
        t = cohort.run_truth(i)
        for e, v in zip(t["event_ids"], np.asarray(t["vertex_m"], float)):
            vtx[int(e)] = v
    v = np.array([vtx[int(e)] for e in ph_ids])
    return {"r2": float(1.0 - np.sum(err ** 2) / np.sum((v - v.mean(0)) ** 2)), "mean_err_m": float(err.mean()),
            "n_events": int(err.size), "eval_dir": str(root)}


def diagnose(cell, seed, step, cohort, n_fit, n_test):
    arm, geometry, scaling, rid, meta = load(cell, seed, step)
    code = torch.as_tensor(sa.fourier_code(geometry, scaling)[0])  # (N, 128), the same for every model
    acc = {"pos_rms": [], "content_rms": [], "pos_rms_hit": [], "content_rms_hit": []}
    sets = {"geometry": 129, "content": 4, "joint": 132}
    fits = {f"{k}_{t}": Normal(p, 128) for k, p in sets.items() for t in ("all", "hit")}
    probe = Normal(129, 3)
    seen = 0
    for b in photon_batches(cohort, geometry, scaling, n_fit + n_test):
        split = "fit" if seen < n_fit else "test"
        with torch.no_grad():
            x = arm._encoder_features(b.features)
            pos, content, kind = input_parts(arm, x)
            z = arm.encode_tokens(b.features)
        hit = (b.features[..., 8] > 0.5).numpy()
        acc["pos_rms"].append(pos.numpy()), acc["content_rms"].append(content.numpy())
        acc["pos_rms_hit"].append(pos.numpy()[hit]), acc["content_rms_hit"].append(content.numpy()[hit])
        zz = z.numpy().reshape(-1, 128)
        feat = b.features.numpy()
        one = np.ones((zz.shape[0], 1))
        geo = np.concatenate([np.broadcast_to(code.numpy(), (b.features.size(0),) + tuple(code.shape)).reshape(-1, 128),
                              one], 1)
        con = np.concatenate([feat[..., 6:9].reshape(-1, 3), one], 1)
        regs = {"geometry": geo, "content": con, "joint": np.concatenate([geo[:, :128], con], 1)}
        hh = hit.reshape(-1)
        for k, r in regs.items():
            fits[f"{k}_all"].add(split, r, zz)
            fits[f"{k}_hit"].add(split, r[hh], zz[hh])
        probe.add(split, np.concatenate([zz, one], 1), feat[..., 0:3].reshape(-1, 3))
        seen += b.features.size(0)
    out = {"run_id": rid, "step": step, "embedding": kind, "checkpoint_sha256": meta["checkpoint_sha256"],
           "n_fit_events": n_fit, "n_test_events": seen - n_fit,
           "input_rms": {k: rms(np.concatenate([a.reshape(-1) for a in v])) for k, v in acc.items()},
           "block4_variance_explained": {k: f.r2() for k, f in fits.items()},
           "probe_pmt_position_r2": probe.r2(),
           "frozen_readout_vertex": vertex_r2(CELLS[cell][0], rid, step, cohort)}
    out["input_rms"]["pos_over_content"] = out["input_rms"]["pos_rms"] / out["input_rms"]["content_rms"]
    out["input_rms"]["pos_over_content_hit"] = out["input_rms"]["pos_rms_hit"] / out["input_rms"]["content_rms_hit"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 10000])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--n-fit", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=128)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    pack = dr.pack("wcte", dr.LUCID_V1_W2)
    dr.check_pack_kind(pack, dr.LUCID_V1_W2)
    cohort = rs.open_cohort(pack, "dev_probe_test")
    res = {"phase": "r2-follow", "stage": "§2.4 Part A diagnostics (exploratory)", "cohort": "dev_probe_test",
           "pack": str(pack), "code_tree": str(TREE), "seed": a.seed, "results": {}}
    for cell in a.cells:
        for step in a.steps:
            r = diagnose(cell, a.seed, step, cohort, a.n_fit, a.n_test)
            res["results"].setdefault(cell, {})[str(step)] = r
            print(cell, step, json.dumps({"rms": r["input_rms"], "ve": r["block4_variance_explained"],
                                          "pos": r["probe_pmt_position_r2"],
                                          "vtx": (r["frozen_readout_vertex"] or {}).get("r2")}), flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
