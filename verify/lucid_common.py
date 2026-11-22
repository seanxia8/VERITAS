"""r2-tier1v helpers for the LUCiD (lucid_v1) data, written for this phase (runbook §0.2, §1).

Independent pieces: the PMT table, the scales and the 9 input features are rebuilt here from
the LUCiD geometry files, the frozen lucid_v1 geometry files and the pack attributes; the
photon cohorts are read straight from the HDF5 pack.  The repository is used only as a
black box: ``point_encoder.LucidPointJepa`` for the encoder forward, bound to the geometry the
data were generated with (``data_roots.geometry``), which fixes the Hilbert token order.
"""
from __future__ import annotations

import hashlib
import json
import os

import h5py
import numpy as np

CEPH = "/ceph/dwong/jepa"
REPO = os.environ.get("R2V_REPO", "/work/dwong/r2_tier1v_repo")
LUCID_CFG = f"{CEPH}/external/LUCiD/config"
DETS = ("wcte", "sk", "juno", "hk")
PACK = {d: f"{CEPH}/data/lucid_v1/{d}/{d}_lucid_v1.h5" for d in DETS}
FROZEN = {d: f"{REPO}/src/jepa_noise/configs/lucid_v1/{d}_geometry_e651e836.npz" for d in DETS}
# LUCiD sensor tables (directions, radius, height); JUNO is a sphere (inward normals)
LUCID_NPZ = {"wcte": f"{LUCID_CFG}/wcte_geometry.npz", "sk": f"{LUCID_CFG}/sk_geometry.npz",
             "hk": f"{LUCID_CFG}/hk_geometry.npz",
             "juno": f"{REPO}/configs/round2/juno_geometry_lucid_eaf2e89.npz"}
# WCTE/SK: LUCiD snaps PMTs onto the wall, the frozen lucid_v1 positions are the data's.
# HK/JUNO: the LUCiD table itself (the frozen file documents < 1 mm agreement).
SNAPPED = {"wcte": True, "sk": True, "juno": False, "hk": False}
PROBE_SPLITS = ("dev_probe_train", "dev_probe_test")


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


class LGeometry:
    """PMT table in pack ``pmt_id`` order, scales (R, R, H/2) and the pack's time scale."""

    def __init__(self, det: str, pack: str | None = None):
        self.det = det
        z = np.load(LUCID_NPZ[det], allow_pickle=False)
        if "positions_mm" in z.files:
            pos = np.asarray(z["positions_mm"], float) / 1000.0
        else:
            pos = np.asarray(z["positions_m"], float)
        self.radius = float(np.asarray(z["radius"]).reshape(-1)[0])
        if "directions" in z.files:
            self.dirs = np.asarray(z["directions"], float)
            self.half_h = 0.5 * float(np.asarray(z["height"]).reshape(-1)[0])
            self.shape = "cylinder"
        else:
            self.dirs = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
            self.half_h = self.radius
            self.shape = "sphere"
        if SNAPPED[det]:
            fz = np.load(FROZEN[det], allow_pickle=False)
            pos = np.asarray(fz["positions_m"], float)
        self.pos = pos
        self.n = pos.shape[0]
        with h5py.File(pack or PACK[det], "r") as f:
            tr = json.loads(f.attrs["time_rule"])
        self.time_scale_ns = float(tr["time_scale_ns"])
        self.scale = np.array([self.radius, self.radius, self.half_h])
        self.unit_pos = self.pos / self.scale


class LCohort:
    """Photon-bomb (or other domain) events of one cohort of a pack, sparse, in RAM."""

    def __init__(self, pack: str, group: str, domain=b"photon", truth=("vertex_m",)):
        with h5py.File(pack, "r") as f:
            g = f[group]
            dom = g["truth/domain"][:]
            self.idx = np.flatnonzero(dom == domain) if domain is not None else np.arange(len(dom))
            self.offsets = g["offsets"][:]
            self.pmt_id = g["pmt_id"][:]
            self.charge = g["charge_pe"][:]
            self.time = g["time_ns"][:]
            self.event_ids = g["event_ids"][:][self.idx]
            self.truth = {k: g["truth"][k][:][self.idx] for k in truth}
            run_events = g["run_events"][:]
        self.vertex = self.truth.get("vertex_m")
        self.run = (np.searchsorted(run_events, np.arange(len(dom)), side="right") - 1)[self.idx]
        self.n = len(self.idx)

    def hits(self, k):
        e = self.idx[k]
        s = slice(self.offsets[e], self.offsets[e + 1])
        return self.pmt_id[s].astype(np.int64), self.charge[s].astype(np.float64), self.time[s].astype(np.float64)


def build_features(geo: LGeometry, coh: LCohort, ks) -> np.ndarray:
    """(len(ks), N, 9) float32: r/(R,R,H/2), direction, ln max(q,1e-6), t/T, hit (§1)."""
    out = np.zeros((len(ks), geo.n, 9), np.float32)
    out[:, :, 0:3] = geo.unit_pos
    out[:, :, 3:6] = geo.dirs
    for i, k in enumerate(ks):
        p, q, t = coh.hits(k)
        out[i, p, 6] = np.log(np.maximum(q, 1e-6))
        out[i, p, 7] = t / geo.time_scale_ns
        out[i, p, 8] = 1.0
    return out


# ------------------------------------------------------------------ encoder (black box)
def load_encoder(det: str, ckpt: str | None, *, window: int = 64, shift: int = 32, seed: int | None = None,
                 device: str = "cpu"):
    """``LucidPointJepa`` on the LUCiD data geometry: checkpoint weights (the ``jepa.`` sub-dict
    of ``model``), or the seed-fixed random init (manual_seed right before construction)."""
    import sys
    import torch
    sys.path.insert(0, f"{REPO}/src")
    from jepa_noise import data_roots
    from jepa_noise.point_encoder import EncoderConfig, LucidPointJepa
    rgeo = data_roots.geometry(det)
    cfg = EncoderConfig(window=int(window), shift=int(shift))
    if ckpt is None:
        torch.manual_seed(int(seed))
        model = LucidPointJepa(cfg, rgeo)
    else:
        model = LucidPointJepa(cfg, rgeo)
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = blob["model"] if "model" in blob else blob
        sub = {k[len("jepa."):]: v for k, v in state.items() if k.startswith("jepa.")}
        missing, unexpected = model.load_state_dict(sub, strict=False)
        assert not unexpected, unexpected
        assert all(m.startswith(("token_order", "query_features")) for m in missing), missing
    model.eval().to(device)
    return model, rgeo
