"""The single PMT response manifest (runbook v2 R-1).

Every pilot builds its reference response from ``configs/lucid_pmt/reference.json``
through :func:`reference_response`; no module constructs a response from literals.
Per-channel factors are rules on a declared channel-to-group map (mPMT modules on
WCTE, D-G; synthetic contiguous groups on the fixture), so one manifest serves any
detector size. Every parameter carries source, status and a value hash.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .quantum_efficiency import QECurve
from .readout import ReadoutParams
from .timing_response import InverseSqrtWalk, WidthCurve

SCHEMA = "pmt-response-v2"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "lucid_pmt" / "reference.json"
PARAMETERS = ("survival", "timing_offsets_ns", "reference_offsets_ns", "threshold_pe", "walk_calibration_pe",
              "spe_sigma", "dead_channels", "mv_per_pe", "qe_curve", "collection_efficiency", "charge_domain_pe",
              "walk", "timing_width", "gain", "timing_width_factor", "dark_rate_hz", "readout_window_ns")


def value_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_manifest(path=None):
    path = Path(path or DEFAULT_PATH)
    m = json.loads(path.read_text())
    if m.get("schema") != SCHEMA:
        raise ValueError(f"response manifest schema {m.get('schema')!r} is not {SCHEMA}")
    params = m.get("parameters", {})
    if set(params) != set(PARAMETERS):
        raise ValueError(f"manifest parameters must be exactly {sorted(PARAMETERS)}")
    for name, entry in params.items():
        if not entry.get("source") or not entry.get("status"):
            raise ValueError(f"parameter {name} lacks source/status provenance")
        if value_sha256(entry["value"]) != entry.get("sha256"):
            raise ValueError(f"parameter {name} value does not match its recorded hash")
    return m


def provenance(path=None):
    """Path and file hash recorded by every pilot that builds a response."""
    path = Path(path or DEFAULT_PATH)
    m = load_manifest(path)
    return {"path": str(path), "schema": m["schema"], "status": m["status"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def module_groups(module_id):
    """Dense group index per channel from a physical module ID (e.g. WCTE ``mpmt_id``)."""
    module_id = np.asarray(module_id)
    if module_id.ndim != 1 or not module_id.size:
        raise ValueError("one module ID per channel required")
    return np.unique(module_id, return_inverse=True)[1]


def _group_rule(rule, group):
    if rule.get("rule") != "group_parity":
        raise ValueError(f"unknown per-channel rule {rule.get('rule')!r}")
    return tuple(float(rule["base"] + rule["odd_group_delta"] * (g % 2)) for g in group)


def exact_product_control(ref, factor=2.0):
    """Exact observable-degeneracy control: QE x factor and CE / factor, same detection product."""
    return replace(ref, qe_curve=QECurve(ref.qe_curve.wavelength_nm, tuple(factor*np.asarray(ref.qe_curve.efficiency)), ref.qe_curve.source),
                   collection_efficiency=ref.collection_efficiency/factor)


def reference_response(n_channels, channel_group, manifest=None):
    """The manifest's reference ``ReadoutParams`` for ``n_channels`` mapped to ``channel_group``."""
    m = manifest if isinstance(manifest, dict) else load_manifest(manifest)
    v = {k: e["value"] for k, e in m["parameters"].items()}
    group = np.asarray(channel_group)
    if group.shape != (n_channels,) or not np.issubdtype(group.dtype, np.integer) or np.any(group < 0):
        raise ValueError(f"one nonnegative integer group per channel ({n_channels}) required")
    domain = tuple(float(x) for x in v["charge_domain_pe"])
    walk = v["walk"]
    if walk.get("form") != "inverse_sqrt":
        raise ValueError(f"unsupported walk form {walk.get('form')!r}")
    curve = InverseSqrtWalk(float(walk["a_ns"]), float(walk["b_ns"]), float(walk["q_sat_pe"]), domain)
    width = WidthCurve(tuple(v["timing_width"]["charge_pe"]), tuple(v["timing_width"]["sigma_ns"]))
    if (width.charge_pe[0], width.charge_pe[-1]) != domain:
        raise ValueError("timing-width table must span the declared charge domain")
    return ReadoutParams(
        survival=v["survival"], timing_offsets_ns=v["timing_offsets_ns"], reference_offsets_ns=v["reference_offsets_ns"],
        threshold_pe=v["threshold_pe"], walk_calibration_pe=v["walk_calibration_pe"], spe_sigma=v["spe_sigma"],
        dead_channels=tuple(v["dead_channels"]), mv_per_pe=v["mv_per_pe"],
        qe_curve=QECurve(tuple(v["qe_curve"]["wavelength_nm"]), tuple(v["qe_curve"]["efficiency"]), m["parameters"]["qe_curve"]["status"]),
        collection_efficiency=v["collection_efficiency"], charge_domain_pe=domain,
        dark_rate_hz=float(v["dark_rate_hz"]), readout_window_ns=tuple(float(x) for x in v["readout_window_ns"]),
        walk_curve=curve, reference_walk=curve, timing_width=width,
        gain=_group_rule(v["gain"], group), timing_width_factor=_group_rule(v["timing_width_factor"], group))
