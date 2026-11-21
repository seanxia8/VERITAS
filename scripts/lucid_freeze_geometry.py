#!/usr/bin/env python
"""L3.0: compare LUCiD's sensors with the frozen round-2 geometry and freeze the LUCiD map.

Writes ``src/jepa_noise/configs/lucid_v1/<det>_geometry_<sha8>.npz`` with ``lucid_to_r2`` (round-2
index of each LUCiD sensor), ``positions_m`` (LUCiD sensor positions in round-2 order) and the
per-PMT deviation from the round-2 positions, plus a JSON comparison record. Runs in the LUCiD env.
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_mc_generator as mg  # noqa: E402
from jepa_noise import r2_geometry as rg  # noqa: E402


def main():
    out_dir = REPO / "src/jepa_noise/configs/lucid_v1"
    record = {}
    os.chdir(mg.LUCID_ROOT)
    from lucid.simulation import setup_event_simulator
    for det, spec in mg.SPECS.items():
        sim = setup_event_simulator(f"config/{spec['geom']}_geom_config.json", 16, source="photons",
                                    K=1, temperature=0.0,
                                    physics_config=f"config/{spec['phys']}_physics_config.json",
                                    default_detector_params=True, hits="segment_truth",
                                    wavelength_mode=spec["wavelength_mode"])
        d = sim.det_geom.detector
        pts = np.asarray(sim.det_geom.sensor_points, float)
        r2 = rg.load_r2_geometry(det)
        order = (np.asarray(d._reorder_indices, np.int64) if hasattr(d, "_reorder_indices")
                 else np.arange(pts.shape[0]))
        lucid_to_r2 = order  # LUCiD sensor j is npz row order[j]; round-2 index == npz row
        pos_r2 = np.empty_like(pts)
        pos_r2[lucid_to_r2] = pts
        dev = np.linalg.norm(pos_r2 - r2.positions_m, axis=1)
        raw = getattr(d, "raw_positions", None)
        raw_dev = None
        if raw is not None:
            raw_r2 = np.empty_like(raw); raw_r2[lucid_to_r2] = raw
            raw_dev = float(np.abs(raw_r2 - r2.positions_m).max())
        reuse = bool(dev.max() < 1e-3)
        path = out_dir / f"{det}_geometry_{mg.LUCID_SHA[:8]}.npz"
        with open(path, "wb") as fh:
            np.savez(fh, lucid_to_r2=lucid_to_r2, positions_m=(r2.positions_m if reuse else pos_r2),
                     lucid_positions_r2_order=pos_r2, deviation_m=dev)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cfg_bytes = Path(f"config/{spec['geom']}_geom_config.json").read_bytes()
        record[det] = {"frozen_npz": str(path.relative_to(REPO)), "sha256": sha,
                       "reuse_round2_positions": reuse, "n_pmt_lucid": int(pts.shape[0]),
                       "n_pmt_round2": int(r2.n_pmt), "max_dev_mm": float(dev.max() * 1e3),
                       "median_dev_mm": float(np.median(dev) * 1e3),
                       "n_dev_over_1mm": int((dev > 1e-3).sum()),
                       "raw_positions_max_abs_dev_m_vs_round2": raw_dev,
                       "lucid_radius_m": float(getattr(d, "r", np.nan)),
                       "lucid_height_m": float(getattr(d, "H", np.nan)),
                       "round2_radius_m": float(r2.radius_m), "round2_height_m": float(r2.height_m),
                       "sensor_radius_m": float(d.S_radius) if hasattr(d, "S_radius") else float(r2.sensor_radius_m),
                       "lucid_geom_config": f"config/{spec['geom']}_geom_config.json",
                       "lucid_geom_config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
                       "snap_to_wall": bool(getattr(d, "snap_to_wall", False))}
        print(det, json.dumps(record[det]), flush=True)
    (out_dir / "geometry_comparison.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
