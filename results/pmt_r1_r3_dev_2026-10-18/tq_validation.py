"""R-2 T-Q validation numbers (runbook v2 §4). Same construction and tolerances as tests/test_physical_tq.py.

Each charge point uses its own event ID (1000 + 100*q), so the points' jitter draws are independent.
Run from the repository root: PYTHONPATH=src python -B results/pmt_r1_r3_dev_2026-10-18/tq_validation.py
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from scipy.stats import norm
from noise_module_lucid import response_manifest
from noise_module_lucid.photon_contract import PhotonBatch
from noise_module_lucid.quantum_efficiency import QECurve
from noise_module_lucid.readout import readout_chain

N, SEED = 20000, 71
base = response_manifest.reference_response(N, np.zeros(N, dtype=int))
base = replace(base, qe_curve=QECurve(base.qe_curve.wavelength_nm, (1.0,)*len(base.qe_curve.wavelength_nm)),
               collection_efficiency=1.0, spe_sigma=0.0, dark_rate_hz=0.0)  # T-Q in isolation; R-6 dark counts off


def read(p, q):
    batch = PhotonBatch(1000 + int(round(100*q)), np.arange(N), np.arange(N), np.zeros(N), np.full(N, 400.0))
    return readout_chain(batch.sensor_id, batch.arrival_time_ns, N, p, SEED, photon_batch=batch)


rows = []
for q in (0.25, 1, 2, 5, 20):
    p = replace(base, gain=float(q))
    raw = replace(p, reference_walk=replace(p.reference_walk, b_ns=0.0))
    d = read(raw, q)
    configured, sigma = float(p.walk_curve(q)), float(p.timing_width(q))
    sigma_mc = sigma / np.sqrt(len(d))
    ref, up, down = read(p, q), read(replace(p, walk_curve=p.walk_curve.scaled(1.2)), q), read(replace(p, walk_curve=p.walk_curve.scaled(0.8)), q)
    early = ref.counts["digits_before_arrival"] / len(ref)
    rows.append(dict(q_pe=q, n_digits=len(d), configured_walk_ns=configured, measured_mean_ns=float(d.time.mean()),
                     sigma_mc_ns=sigma_mc, z=float((d.time.mean() - configured) / sigma_mc), within_3_sigma_mc=bool(abs(d.time.mean() - configured) < 3*sigma_mc),
                     slope_x1_2_shift_ns=float(np.mean(up.time - ref.time)), slope_x0_8_shift_ns=float(np.mean(down.time - ref.time)),
                     expected_slope_shift_ns=float(0.2*p.walk_curve.b_ns/np.sqrt(min(q, p.walk_curve.q_sat_pe))),
                     charge_membership_preserved=bool(np.array_equal(ref.sensor, up.sensor) and np.array_equal(ref.charge, up.charge)
                                                      and np.array_equal(ref.sensor, down.sensor) and np.array_equal(ref.charge, down.charge)),
                     digits_before_arrival_fraction=early, expected_before_arrival_fraction=float(norm.cdf(-configured/sigma))))
out = dict(walk=dict(form="inverse_sqrt", a_ns=base.walk_curve.a_ns, b_ns=base.walk_curve.b_ns, q_sat_pe=base.walk_curve.q_sat_pe,
                     domain_pe=list(base.walk_curve.domain_pe)),
           acceptance="conditional mean within 3 sigma_MC at 0.25, 1, 2, 5, 20 PE (runbook v2 R-2)",
           passed=all(r["within_3_sigma_mc"] and r["charge_membership_preserved"] for r in rows), rows=rows,
           response_manifest=response_manifest.provenance(), evidence_stage="development", non_citable=True, status="synthetic_dev")
Path(__file__).with_name("tq_validation.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["rows"], indent=1)); print("passed", out["passed"])
