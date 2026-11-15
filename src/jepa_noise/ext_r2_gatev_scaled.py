"""HK Gate V photon-statistics ceiling without a second LUCiD transport (r2-report, CPU only).

Addendum ``RUNBOOK_2026-11-12_R2_GATEV_CEILING.md`` §2 needs μ per PMT, Λ and w̄. In
``ext_r2_gatev.expected_mu`` the transport enters μ only through ``qe_mean`` (the mean QE over the
detected photons' wavelengths), which multiplies every PMT by the same scalar: μ is linear in it.
So μ's shape follows from the PhotonSim truth alone and its scale is fixed by Σμ = the committed
``expected_detected``. ``--validate`` checks this against the exact (transported) μ on the cached
SK and JUNO events and compares the resulting ceilings with the committed ones. The split-half
reliability and photons per hit module need the transported hits and are not computed here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import lucid_bombs as lb
from . import ext_r2_gatev as g


def truth_event(raw: dict, event_id: int):
    """The fields ``expected_mu`` reads, exactly as ``cv.event_from_transport`` sets them; the
    wavelengths are the emitted ones (they only set the uniform QE scalar, removed by rescaling)."""
    direction = np.asarray(raw["primary_direction"], dtype=float)
    path = np.asarray(raw.get("primary_path_m", np.zeros((2, 3))), dtype=float)
    track_length = float(np.linalg.norm(path[-1] - path[0])) if path.shape[0] >= 2 else float("nan")
    return SimpleNamespace(
        event_id=int(event_id), domain=lb.DOMAIN_MUON,
        vertex_m=np.asarray(raw["vertex_m"], dtype=float),
        wavelength_nm=np.asarray(raw["wavelength_nm"], dtype=float),
        direction_m=direction / np.linalg.norm(direction),
        energy_MeV=float(raw["energy_MeV"]), deposited_MeV=float(raw["energy_MeV"]),
        contained=bool(np.isfinite(track_length)), track_length_m=track_length)


def scaled_mu(geometry, cfg, raw: dict, event_id: int, expected_detected: float) -> np.ndarray:
    mu = g.expected_mu(geometry, cfg, truth_event(raw, event_id))
    return mu * (float(expected_detected) / float(mu.sum()))


def load_raw(detector: str, index: int, geometry) -> dict:
    from . import cherenkov_validation as cv
    from . import ext_r2 as x
    with np.load(Path(x.PHOTONSIM_DIR) / x.GATE_V_HOLDOUT[index], allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    return cv.contain_event(raw, geometry)


def detector_ceiling_scaled(detector: str, rows: list, cfg, *, k: int = g.K_DRAWS) -> dict:
    from . import r2_geometry as rg
    geometry = rg.load_r2_geometry(detector)
    module = np.asarray(geometry.module_index, np.int64)
    out_rows = []
    for i, row in enumerate(rows):
        mu = scaled_mu(geometry, cfg, load_raw(detector, i, geometry), i, row["expected_detected"])
        lam = int(row["n_photons"])
        w_bar = float(row["observed_detected"]) / lam
        c = g.event_ceiling(mu, lam, w_bar, module, geometry.n_modules, detector=detector,
                            event=i, k=k)
        r = {"event": i, "lambda_photons": lam, "w_bar": w_bar,
             "photons_per_hit_pmt": lam / max(1, int(row["n_hit_pmts"])), "metrics": {}}
        for m in g.METRICS:
            obs = float(row[m])
            r["metrics"][m] = {**c[m], "observed": obs, "normalised": obs / c[m]["c"]}
        out_rows.append(r)
    out = {"detector": detector, "skipped": False, "method": "scaled_mu_no_transport",
           "n_pmt": int(geometry.n_pmt), "n_modules": int(geometry.n_modules), "events": out_rows,
           "C": {}, "normalised_median": {}, "observed_median": {}}
    for m in g.METRICS:
        out["C"][m] = float(np.median([r["metrics"][m]["c"] for r in out_rows]))
        out["normalised_median"][m] = float(np.median([r["metrics"][m]["normalised"] for r in out_rows]))
        out["observed_median"][m] = float(np.median([r["metrics"][m]["observed"] for r in out_rows]))
    return out


def validate(detector: str, rows: list, cfg, cache: Path, committed: dict, k: int) -> dict:
    """Scaled μ against the exact μ of the cached transported events, and C against committed C."""
    from . import cherenkov_validation as cv
    from . import ext_r2 as x
    from . import r2_geometry as rg
    geometry = rg.load_r2_geometry(detector)
    worst = 0.0
    for i, row in enumerate(rows):
        exact = g.expected_mu(geometry, cfg, cv.load_event_npz(cache / f"{detector}__{x.GATE_V_HOLDOUT[i]}"))
        approx = scaled_mu(geometry, cfg, load_raw(detector, i, geometry), i, row["expected_detected"])
        nz = exact > 0
        if not np.array_equal(nz, approx > 0):
            return {"detector": detector, "ok": False, "reason": f"event {i}: support differs"}
        worst = max(worst, float(np.max(np.abs(approx[nz] / exact[nz] - 1.0))) if nz.any() else 0.0)
    res = detector_ceiling_scaled(detector, rows, cfg, k=k)
    dC = {m: abs(res["C"][m] - committed["C"][m]) for m in g.METRICS}
    return {"detector": detector, "ok": bool(worst < 1e-9), "max_rel_mu_diff": worst,
            "abs_C_diff_vs_committed": dC}


def main(argv=None) -> int:
    from . import ext_r2 as x
    from . import r2_generator as gen
    p = argparse.ArgumentParser()
    p.add_argument("--detector", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=g.K_DRAWS)
    p.add_argument("--validate-cache", help="ceiling cache dir: run the validation mode")
    p.add_argument("--committed", help="committed exact ceiling JSON (validation mode)")
    a = p.parse_args(argv)
    doc = json.loads(Path(a.rows).read_text())
    rows = (doc.get("gate_v") or doc)["rows"]
    cfg = gen.load_r2_config()
    t0 = time.time()
    if a.validate_cache:
        res = validate(a.detector, rows, cfg, Path(a.validate_cache),
                       json.loads(Path(a.committed).read_text()), a.k)
    else:
        res = detector_ceiling_scaled(a.detector, rows, cfg, k=a.k)
    res.update({"addendum_sha256": g.ADDENDUM_SHA256, "seed": g.CEILING_SEED, "k": a.k,
                "rows_source": a.rows, "wall_s": time.time() - t0, "code_sha": lb.current_code_sha()})
    Path(x.check_write_path(a.out)).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res.get(k) for k in ("detector", "ok", "max_rel_mu_diff",
                                              "abs_C_diff_vs_committed", "C", "normalised_median",
                                              "wall_s")}, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
