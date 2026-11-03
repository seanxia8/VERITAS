"""R-7 v2 fixture comparison: response v2 vs the thesis cross-check vs the WCSim 3-inch default.

Run (CPU only, no WCTE data, no network)::

    PYTHONPATH=src python -B scripts/pmt_r7v2_fixture_comparison.py

Fixed before the run, as in the 23 Oct comparison (``scripts/pmt_r7_v2_v3_fixture_comparison.py``):

* detector and events from ``noise_module_lucid.pmt_development.fixture()`` (synthetic
  76-sensor cylinder, 4000 photons/event, NOT WCTE geometry); the 24 evaluation events
  ``30000..30023`` read out as ``calibration.event_digits`` does, per-event seed ``event_id``;
* only the response manifest changes: ``v2`` = ``configs/lucid_pmt/reference.json`` (frozen),
  ``v1_thesis`` = ``configs/lucid_pmt/reference_v3_thesis_xcheck.json`` (the first R-7, 20-inch
  proxies), ``v3_wcsim`` = the default ``configs/lucid_pmt/reference_v3.json`` (R-7 v2).
  All three carry the R-6 dark-count stage at 1 kHz.

Reported per evaluation event and response: digit (hit-PMT) count, source-only digit count,
detected source photoelectrons, mean digit time and within-event digit-time SD (ddof=1).

R-7 v2 also asks for the **1-PE hit efficiency**: the fraction of single-photoelectron
channels that produce a digit. The fixture events are high-occupancy (most PMTs see many
PE), so it is measured separately: 20000 channels, one photon each, every photon detected
(QE = CE = 1), unit gain, dark counts off, seed 20261103; the analytic value is
P(q >= 0.25 PE) under each response's SPE model. Development evidence only, not citable.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.special import ndtr

from noise_module_lucid import response_manifest
from noise_module_lucid.photon_contract import PhotonBatch
from noise_module_lucid.pmt_development import fixture
from noise_module_lucid.quantum_efficiency import QECurve
from noise_module_lucid.readout import readout_chain

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "pmt_r7v2_dev_2026-11-03" / "fixture_comparison.json"
EVAL_EVENT_IDS = tuple(range(30000, 30024))
ONE_PE_CHANNELS = 20000
ONE_PE_SEED = 20261103
TZ = ZoneInfo("Europe/Berlin")
RESPONSES = {"v2": response_manifest.V2_PATH, "v1_thesis": response_manifest.THESIS_XCHECK_PATH,
             "v3_wcsim": response_manifest.DEFAULT_PATH}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, cwd=REPO_ROOT).strip()


def _row(event, n_channels, params):
    d = readout_chain(event.photon_sensor, event.photon_time, n_channels, params,
                      seed=int(event.event_id), photon_batch=event.photon_batch)
    t = np.asarray(d.time, dtype=float)
    if d.n_dark_true is None or t.size < 2:
        raise RuntimeError(f"event {event.event_id}: dark stage off or < 2 digits")
    src = d.n_pe_true - d.n_dark_true
    return {"event_id": int(event.event_id), "n_digits": int(len(d)), "n_source_digits": int(np.count_nonzero(src > 0)),
            "source_pe_in_digits": int(src.sum()), "photons_detected": int(d.counts["photons_detected"]),
            "lost_threshold": int(d.counts["lost_threshold"]),
            "mean_digit_time_ns": float(t.mean()), "digit_time_spread_ns": float(t.std(ddof=1))}


def _totals(rows):
    f = lambda k: [r[k] for r in rows]
    return {"n_events": len(rows), "mean_digits_per_event": float(np.mean(f("n_digits"))),
            "mean_source_digits_per_event": float(np.mean(f("n_source_digits"))),
            "mean_detected_pe_per_event": float(np.mean(f("photons_detected"))),
            "mean_digit_time_ns": float(np.mean(f("mean_digit_time_ns"))),
            "mean_digit_time_spread_ns": float(np.mean(f("digit_time_spread_ns"))),
            "sd_digit_time_spread_ns": float(np.std(f("digit_time_spread_ns"), ddof=1))}


def one_pe_efficiency(path):
    n = ONE_PE_CHANNELS
    p = response_manifest.reference_response(n, np.zeros(n, dtype=int), manifest=path)
    p = replace(p, qe_curve=QECurve(p.qe_curve.wavelength_nm, (1.0,) * len(p.qe_curve.wavelength_nm)),
                collection_efficiency=1.0, dark_rate_hz=0.0)
    b = PhotonBatch(ONE_PE_SEED, np.arange(n), np.arange(n), np.zeros(n), np.full(n, 400.0))
    d = readout_chain(b.sensor_id, b.arrival_time_ns, n, p, ONE_PE_SEED, photon_batch=b)
    thr = p.threshold_pe
    if p.spe_charge is not None:
        analytic, model = 1.0 - float(p.spe_charge.cdf_at(thr)), "WCSim Getqpe table (mean 1 PE)"
    else:
        analytic, model = float(ndtr((1.0 - thr) / p.spe_sigma)), f"Gaussian 1 + {p.spe_sigma} z"
    eff = len(d) / n
    return {"spe_model": model, "threshold_pe": thr, "channels": n, "digits": int(len(d)), "efficiency": eff,
            "binomial_se": float(np.sqrt(eff * (1 - eff) / n)), "analytic": analytic,
            "lost_threshold": int(d.counts["lost_threshold"]), "below_charge_domain": int(d.counts.get("below_charge_domain", 0))}


def run(out=DEFAULT_OUT):
    start = time.monotonic()
    det, mapping, _ref, _fit, _closure, events = fixture()
    if [int(e.event_id) for e in events] != list(EVAL_EVENT_IDS):
        raise RuntimeError("fixture evaluation events changed")
    group = np.asarray(mapping.channel_group)
    refs = {k: response_manifest.reference_response(det.n_channels, group, manifest=v) for k, v in RESPONSES.items()}
    rows = {k: [_row(ev, det.n_channels, p) for ev in events] for k, p in refs.items()}
    totals = {k: _totals(r) for k, r in rows.items()}
    one_pe = {k: one_pe_efficiency(v) for k, v in RESPONSES.items()}
    tw = refs["v3_wcsim"].timing_width
    report = {
        "title": "R-7 v2 fixture comparison: PMT response v2 vs v1 thesis cross-check vs v3 WCSim 3-inch default",
        "repair": "R-7 v2 (author message, 23 Oct 2026; decisions 3 Nov 2026)",
        "evidence_stage": "development", "non_citable": True, "source_kind": "synthetic_prompt_flasher",
        "head": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_clean_for_inputs": _git("status", "--porcelain", "--", "src", "configs", "scripts") == "",
        "timestamp": datetime.now(TZ).isoformat(timespec="seconds"), "timezone": "Europe/Berlin",
        "python_version": platform.python_version(), "numpy": np.__version__, "command": sys.argv,
        "manifest": {k: response_manifest.provenance(v) for k, v in RESPONSES.items()},
        "moved_fields": {"v1_thesis_vs_v2": refs["v1_thesis"].moved_from(refs["v2"]),
                         "v3_wcsim_vs_v2": refs["v3_wcsim"].moved_from(refs["v2"]),
                         "v3_wcsim_vs_v1_thesis": refs["v3_wcsim"].moved_from(refs["v1_thesis"])},
        "fixture": {"description": "synthetic 76-sensor cylinder (NOT WCTE geometry); isotropic prompt flasher, no "
                                   "scatter/reflection/upstream TTS; uniform 300..600 nm synthetic spectrum",
                    "builder": "noise_module_lucid.pmt_development.fixture()", "n_channels": int(det.n_channels),
                    "photons_per_event": 4000, "evaluation_event_ids": list(EVAL_EVENT_IDS)},
        "per_event": rows, "totals": totals,
        "one_pe_hit_efficiency": one_pe,
        "kernel_sigma_ns_v3_wcsim": {str(q): float(tw(q)) for q in (0.25, 0.5, 1.0, 4.0, 100.0)},
        "elapsed_seconds": None,
    }
    report["elapsed_seconds"] = time.monotonic() - start
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    for k in RESPONSES:
        t, e = totals[k], one_pe[k]
        print(f"{k:10s} hit PMTs/event {t['mean_digits_per_event']:.2f}  detected PE/event {t['mean_detected_pe_per_event']:.1f}  "
              f"digit-time spread {t['mean_digit_time_spread_ns']:.3f} ns  1-PE eff {e['efficiency']:.4f} (analytic {e['analytic']:.4f})")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="R-7 v2 fixture comparison")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(ap.parse_args().out)
