"""r2-verify shared helpers, written from scratch (runbook §0.2).

Only the encoder forward is taken from the repository (``point_encoder.LucidPointJepa``,
the class that ``LucidArm.jepa`` holds, plus ``r2_geometry.load_r2_geometry`` for the
Hilbert token order it needs).  Geometry, cohorts and the 9 input features are rebuilt
here directly from the LUCiD geometry files and the HDF5 attributes.
"""
from __future__ import annotations

import hashlib
import json
import os

import h5py
import numpy as np

CEPH = "/ceph/dwong/jepa"
REPO = os.environ.get("R2V_REPO", "/work/dwong/r2_verify_repo")
DET = {
    "wcte": dict(h5=f"{CEPH}/returns/r2_multidetector_2026-11-11/wcte_regression/wcte_probe_r2.h5",
                 geom=f"{CEPH}/external/LUCiD/config/wcte_geometry.npz"),
    "sk": dict(h5=f"{CEPH}/data/round2/sk/sk_r2.h5", geom=f"{CEPH}/external/LUCiD/config/sk_geometry.npz"),
    "juno": dict(h5=f"{CEPH}/data/round2/juno/juno_r2.h5",
                 geom=f"{REPO}/configs/round2/juno_geometry_lucid_eaf2e89.npz"),
    "hk": dict(h5=f"{CEPH}/data/round2/hk/hk_r2.h5", geom=f"{CEPH}/external/LUCiD/config/hk_geometry.npz"),
}
# WCTE carries no time_scale_ns attribute; the round-1 run records (recipe.json "scaling")
# give 100 ns.  The other detectors read it from the HDF5 ``time_rule`` attribute.
WCTE_TIME_SCALE_NS = 100.0
FIDUCIAL_FRACTION_DEFAULT = 0.8
SPLITS = {"train": "dev_probe_train", "test": "dev_probe_test"}


def sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


class Geometry:
    """PMT table in HDF5 ``pmt_id`` order (row index of the LUCiD npz; checked in K0)."""

    def __init__(self, det: str):
        self.det = det
        z = np.load(DET[det]["geom"], allow_pickle=False)
        if "positions_mm" in z.files:
            self.pos = np.asarray(z["positions_mm"], float) / 1000.0
        else:
            self.pos = np.asarray(z["positions_m"], float)
        self.radius = float(np.asarray(z["radius"]).reshape(-1)[0])
        if "directions" in z.files:
            self.dirs = np.asarray(z["directions"], float)
            self.shape = "cylinder"
            self.half_h = 0.5 * float(np.asarray(z["height"]).reshape(-1)[0])
        else:  # sphere: sensors face the centre (inward unit normal)
            self.dirs = -self.pos / np.linalg.norm(self.pos, axis=1, keepdims=True)
            self.shape = "sphere"
            self.half_h = self.radius
        self.n = self.pos.shape[0]
        with h5py.File(DET[det]["h5"], "r") as f:
            attrs = dict(f.attrs)
        dv = json.loads(attrs["derived_values"]) if "derived_values" in attrs else {}
        tr = json.loads(attrs["time_rule"]) if "time_rule" in attrs else {}
        self.time_scale_ns = float(tr.get("time_scale_ns", WCTE_TIME_SCALE_NS))
        self.fid_frac = float(dv.get("fiducial_fraction", FIDUCIAL_FRACTION_DEFAULT))
        if dv:  # cross-check npz against the pack's own derived values
            assert abs(dv["radius_m"] - self.radius) < 1e-9, (dv["radius_m"], self.radius)
            assert abs(dv["half_extent_m"][2] - self.half_h) < 1e-9
            assert dv["n_pmt"] == self.n
        self.r_fid = self.fid_frac * self.radius
        self.scale = np.array([self.radius, self.radius, self.half_h])
        self.unit_pos = self.pos / self.scale  # PMT coordinates in [-1, 1]


class Cohort:
    """Photon-bomb events of one probe split (sparse hits held in RAM)."""

    def __init__(self, det: str, split: str, domain=b"photon"):
        with h5py.File(DET[det]["h5"], "r") as f:
            g = f[SPLITS[split]]
            dom = g["truth/domain"][:]
            self.idx = np.flatnonzero(dom == domain)
            self.offsets = g["offsets"][:]
            self.pmt_id = g["pmt_id"][:]
            self.charge = g["charge_pe"][:]
            self.time = g["time_ns"][:]
            self.event_ids = g["event_ids"][:][self.idx]
            self.vertex = g["truth/vertex_m"][:][self.idx]
            run_events = g["run_events"][:]
        run_of_event = np.searchsorted(run_events, np.arange(len(dom)), side="right") - 1
        self.run = run_of_event[self.idx]
        self.n = len(self.idx)

    def hits(self, k):
        """(pmt, charge, time) of the k-th photon event."""
        e = self.idx[k]
        s = slice(self.offsets[e], self.offsets[e + 1])
        return self.pmt_id[s].astype(np.int64), self.charge[s].astype(np.float64), self.time[s].astype(np.float64)


def build_features(geo: Geometry, coh: Cohort, ks) -> np.ndarray:
    """Independent 9-column encoder input, (len(ks), N_pmt, 9) float32.

    cols 0-2 position / (R, R, H/2); 3-5 unit direction; 6 ln(max(q, 1e-6 PE) / 1 PE) on hit
    PMTs else 0; 7 (digit - trigger) time / T_det on hit PMTs else 0; 8 hit flag.
    """
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
def load_encoder(det: str, ckpt: str | None, seed: int | None, window: int = 64, shift: int = 32):
    """Repository encoder as a black box: ``LucidPointJepa`` with checkpoint weights, or the
    seed-fixed random init (``torch.manual_seed(seed)`` immediately before construction,
    which is what ``LucidArm.__init__`` / ``build_arm`` do)."""
    import sys
    import torch
    sys.path.insert(0, f"{REPO}/src")
    from jepa_noise.point_encoder import EncoderConfig, LucidPointJepa
    from jepa_noise.r2_geometry import load_r2_geometry
    rgeo = load_r2_geometry(det)
    cfg = EncoderConfig(window=window, shift=shift)
    if ckpt is None:
        torch.manual_seed(int(seed))
        model = LucidPointJepa(cfg, rgeo)
    else:
        model = LucidPointJepa(cfg, rgeo)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        sub = {k[len("jepa."):]: v for k, v in state.items() if k.startswith("jepa.")}
        missing, unexpected = model.load_state_dict(sub, strict=False)
        assert not unexpected, unexpected
        assert all(m.startswith(("token_order", "query_features")) for m in missing), missing
    model.eval()
    return model, rgeo
