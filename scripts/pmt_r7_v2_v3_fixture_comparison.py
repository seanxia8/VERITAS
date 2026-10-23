"""R-7 fixture comparison: frozen response v2 vs the new default v3.

Run (CPU only, no WCTE data, no network):

    PYTHONPATH=src .venv/bin/python -B scripts/pmt_r7_v2_v3_fixture_comparison.py

Everything is fixed before the run:
* detector and events come from ``noise_module_lucid.pmt_development.fixture()``
  (synthetic 76-sensor cylinder, 4000 photons/event, NOT WCTE geometry);
* the 24 evaluation events ``30000..30023`` are read out exactly as
  ``calibration.event_digits`` does, with the per-event seed ``event_id``;
* only the response manifest changes: ``v2`` = ``configs/lucid_pmt/reference.json``
  (pinned, frozen) through ``response_manifest.V2_PATH``, ``v3`` = the default
  ``configs/lucid_pmt/reference_v3.json``. Both manifests carry the R-6 dark-count
  stage at the same 1 kHz, so the dark-count stage is not a variable here.

Reported per evaluation event and response: digit (hit-PMT) count, source-only
digit count (``n_pe_true > n_dark_true``), mean digit time, and the within-event
standard deviation of digit times (ddof=1). Development evidence only, not citable.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from noise_module_lucid import response_manifest
from noise_module_lucid.pmt_development import fixture
from noise_module_lucid.readout import readout_chain

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "pmt_r7_dev_2026-10-23" / "fixture_comparison.json"
EVAL_EVENT_IDS = tuple(range(30000, 30024))
TZ = ZoneInfo("Europe/Berlin")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _digits_row(event, n_channels: int, params) -> dict:
    """One event x one response, read out through the declared PMT chain."""
    digits = readout_chain(event.photon_sensor, event.photon_time, n_channels, params,
                           seed=int(event.event_id), photon_batch=event.photon_batch)
    times = np.asarray(digits.time, dtype=float)
    if digits.n_dark_true is None:
        raise RuntimeError("dark-count stage is off; both manifests must keep it on")
    if times.size < 2:
        raise RuntimeError(f"event {event.event_id}: {times.size} digits, within-event spread undefined")
    n_digits = int(len(digits))
    n_source_digits = int(np.count_nonzero(digits.n_pe_true > digits.n_dark_true))
    spread = float(times.std(ddof=1))
    if not (n_digits > 0 and n_source_digits > 0 and np.isfinite(spread) and spread > 0.0):
        raise RuntimeError(f"event {event.event_id}: degenerate fixture metric "
                           f"(n_digits={n_digits}, n_source_digits={n_source_digits}, spread={spread})")
    return {"event_id": int(event.event_id), "n_digits": n_digits, "n_source_digits": n_source_digits,
            "mean_digit_time_ns": float(times.mean()), "digit_time_spread_ns": spread}


def _totals(rows: list[dict]) -> dict:
    n_digits = [r["n_digits"] for r in rows]
    n_source = [r["n_source_digits"] for r in rows]
    spreads = [r["digit_time_spread_ns"] for r in rows]
    means = [r["mean_digit_time_ns"] for r in rows]
    return {"n_events": len(rows),
            "total_digits": int(sum(n_digits)),
            "total_source_digits": int(sum(n_source)),
            "mean_digits_per_event": float(np.mean(n_digits)),
            "mean_source_digits_per_event": float(np.mean(n_source)),
            "mean_digit_time_ns": float(np.mean(means)),
            "mean_digit_time_spread_ns": float(np.mean(spreads))}


def _summary(events_v2: list[dict], events_v3: list[dict]) -> dict:
    t2, t3 = _totals(events_v2), _totals(events_v3)
    return {
        "hit_pmts_per_event_v2": t2["mean_digits_per_event"],
        "hit_pmts_per_event_v3": t3["mean_digits_per_event"],
        "hit_pmts_per_event_delta": t3["mean_digits_per_event"] - t2["mean_digits_per_event"],
        "source_hit_pmts_per_event_v2": t2["mean_source_digits_per_event"],
        "source_hit_pmts_per_event_v3": t3["mean_source_digits_per_event"],
        "mean_digit_time_spread_ns_v2": t2["mean_digit_time_spread_ns"],
        "mean_digit_time_spread_ns_v3": t3["mean_digit_time_spread_ns"],
        "mean_digit_time_spread_delta_ns": t3["mean_digit_time_spread_ns"] - t2["mean_digit_time_spread_ns"],
        "mean_digit_time_ns_v2": t2["mean_digit_time_ns"],
        "mean_digit_time_ns_v3": t3["mean_digit_time_ns"],
    }


def comparison_line(n_channels: int, summary: dict) -> str:
    return ("R-7 fixture ({n_events} eval events, {n_channels} sensors): hit PMTs/event "
            "{hit_v2:.2f} -> {hit_v3:.2f} (delta {hit_delta:+.2f}); mean digit time spread "
            "{spread_v2:.3f} -> {spread_v3:.3f} ns (delta {spread_delta:+.3f} ns)").format(
        n_events=len(EVAL_EVENT_IDS), n_channels=n_channels,
        hit_v2=summary["hit_pmts_per_event_v2"], hit_v3=summary["hit_pmts_per_event_v3"],
        hit_delta=summary["hit_pmts_per_event_delta"],
        spread_v2=summary["mean_digit_time_spread_ns_v2"], spread_v3=summary["mean_digit_time_spread_ns_v3"],
        spread_delta=summary["mean_digit_time_spread_delta_ns"])


def run(out: Path = DEFAULT_OUT) -> dict:
    start = time.monotonic()
    det, mapping, _fixture_ref, _fit, _closure, events = fixture()
    if [int(e.event_id) for e in events] != list(EVAL_EVENT_IDS):
        raise RuntimeError(f"fixture evaluation events changed: {[int(e.event_id) for e in events]}")

    group = np.asarray(mapping.channel_group)
    ref_v2 = response_manifest.reference_response(det.n_channels, group, manifest=response_manifest.V2_PATH)
    ref_v3 = response_manifest.reference_response(det.n_channels, group)
    moved = ref_v2.moved_from(ref_v3)
    if not moved:
        raise RuntimeError("v2 and v3 reference responses are identical; nothing to compare")

    rows = {"v2": [_digits_row(ev, det.n_channels, ref_v2) for ev in events],
            "v3": [_digits_row(ev, det.n_channels, ref_v3) for ev in events]}
    if all(a == b for a, b in zip(rows["v2"], rows["v3"])):
        raise RuntimeError("v2 and v3 fixture rows are identical; the response change is not reaching the readout")

    summary = _summary(rows["v2"], rows["v3"])
    report = {
        "title": "R-7 fixture comparison: frozen PMT response v2 vs default v3",
        "repair": "R-7 (author message, 23 Oct 2026)",
        "evidence_stage": "development",
        "non_citable": True,
        "source_kind": "synthetic_prompt_flasher",
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "timestamp": datetime.now(TZ).isoformat(timespec="seconds"),
        "timezone": "Europe/Berlin",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "numpy": np.__version__,
        "command": sys.argv,
        "manifest": {"v2": response_manifest.provenance(response_manifest.V2_PATH),
                     "v3": response_manifest.provenance()},
        "response_difference_moved_fields": moved,
        "fixture": {
            "description": "synthetic 76-sensor cylinder (4 barrel rings x 12 + 2 cap rings, NOT WCTE geometry); "
                           "isotropic prompt flasher, no scatter/reflection/upstream TTS; "
                           "uniform 300..600 nm synthetic spectrum",
            "builder": "noise_module_lucid.pmt_development.fixture()",
            "geometry_call": "toy.cylinder_detector(4, 12, 2)",
            "n_channels": int(det.n_channels),
            "photons_per_event": int(response_manifest.load_manifest()["fixture"]["photons_per_event"]),
            "evaluation_event_ids": list(EVAL_EVENT_IDS),
            "readout": "readout_chain(event.photon_sensor, event.photon_time, det.n_channels, params, "
                       "seed=int(event.event_id), photon_batch=event.photon_batch)",
            "dark_count_stage": "on in both manifests (1 kHz per PMT, readout window [-200, 800] ns); "
                                "not varied by this comparison",
        },
        "per_event": {"event_ids": list(EVAL_EVENT_IDS), "v2": rows["v2"], "v3": rows["v3"]},
        "totals": {"v2": _totals(rows["v2"]), "v3": _totals(rows["v3"])},
        "summary": summary,
        "elapsed_seconds": time.monotonic() - start,
    }

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(comparison_line(det.n_channels, summary))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R-7 fixture comparison: response v2 vs v3")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().out)
