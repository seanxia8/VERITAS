"""s00: bit-exact replay checks for the two fixtures used by the diagnostics.

1. E1a: reproduce the registered row th0.1_eta1_g0_n100000_s0 scc digest and r1
   CCA angle from the stored table (ExactModel + registered namespaces).
2. E1b: reproduce the registered clean-cell row g0_line0_s0 scc digest and r1
   CCA angle (Tier1Views + per-event namespace RNG, streamed as the runner does).

No ``src`` edits.  Writes ``validation.json`` only.  Runtime ~3-4 min (E1b is the
registered per-event generator, 1e5 events in 1e4-event chunks).
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

import e1f_common as C
from jepa_noise import linear
from jepa_noise.metrics import principal_angles
from jepa_noise.masks import namespace_rng, raw_split
from jepa_noise.views import ExactModel, Tier1Views


def digest(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def e1a_replay():
    out = {}
    cell = {"theta_scale": 0.1, "eta_scale": 1.0, "gamma_load": 0.0, "n": 100000}
    label = "th0.1_eta1_g0"
    model = ExactModel(d=40, k=3, seed=0, theta_scale=0.1, eta_scale=1.0,
                       share="target")
    model.rng = namespace_rng(0, "E1a_subject", cell=f"{label}/n{cell['n']}")
    t = time.time()
    sample = model.sample(cell["n"], cell["gamma_load"])
    out["sample_seconds"] = time.time() - t
    population = sample["population"]
    reference = np.linalg.solve(population["ncc"], population["ac"]).T
    scc, stt, sct = linear.sample_covariances(sample["context"], sample["target"],
                                              ridge_relative=0.0)
    out["scc_unregularized_sha256"] = digest(scc)
    out["scc_expected_sha256"] = "79e8d97b0f423917d130101d505eac0e240835750f30e177874937eeda0094f6"
    out["scc_digest_matches"] = out["scc_unregularized_sha256"] == out["scc_expected_sha256"]
    angle, _ = C.cca_angle(scc, stt, sct, reference)
    out["cca_max_angle_deg"] = angle
    out["cca_expected_deg"] = 2.7799885516166336
    out["cca_abs_diff_deg"] = abs(angle - 2.7799885516166336)
    out["population_cca_deg"] = C.population_cca_angle(population, reference)[0]
    out["status"] = "pass" if (out["scc_digest_matches"]
                               and out["cca_abs_diff_deg"] < 1e-9) else "fail"
    return out


def e1b_replay():
    out = {}
    n, chunk = 100000, 10000
    view = Tier1Views(signal_mode="linearized", split="channel",
                      line_multiplier=0., common_strength=0., seed=0,
                      cell="E1b/g0_line0_s0")
    split = raw_split("channel")
    acc = {"sum_c": np.zeros(len(split.context)),
           "sum_t": np.zeros(len(split.target)),
           "xx_c": np.zeros((len(split.context),) * 2),
           "xx_t": np.zeros((len(split.target),) * 2),
           "ct": np.zeros((len(split.context), len(split.target)))}
    t = time.time()
    for start in range(0, n, chunk):
        ids = np.arange(start, min(start + chunk, n), dtype=np.int64)
        raw = view.sample(ids, role="E1b_subject")["raw"]
        values = raw.reshape(len(ids), 2048)
        context, target = values[:, split.context], values[:, split.target]
        acc["sum_c"] += context.sum(0)
        acc["sum_t"] += target.sum(0)
        acc["xx_c"] += context.T @ context
        acc["xx_t"] += target.T @ target
        acc["ct"] += context.T @ target
        del raw, values, context, target
    out["stream_seconds"] = time.time() - t
    scc = (acc["xx_c"] - np.outer(acc["sum_c"], acc["sum_c"]) / n) / (n - 1)
    stt = (acc["xx_t"] - np.outer(acc["sum_t"], acc["sum_t"]) / n) / (n - 1)
    sct = (acc["ct"] - np.outer(acc["sum_c"], acc["sum_t"]) / n) / (n - 1)
    population = C.e1b_population(view)
    reference = np.linalg.solve(population["ncc"], population["ac"]).T
    out["scc_sha256"] = digest(scc)
    out["scc_expected_sha256"] = "eb6a6770c6a13331854c4de9ca21cf1182cb08c553666d2827c919722bdfacc7"
    out["scc_digest_matches"] = out["scc_sha256"] == out["scc_expected_sha256"]
    angle, _ = C.cca_angle(scc, stt, sct, reference)
    out["cca_max_angle_deg"] = angle
    out["cca_expected_deg"] = 16.306755154100735
    out["cca_abs_diff_deg"] = abs(angle - 16.306755154100735)
    out["status"] = "pass" if (out["scc_digest_matches"]
                               and out["cca_abs_diff_deg"] < 1e-9) else "fail"
    return out


if __name__ == "__main__":
    payload = {"e1a_replay": e1a_replay(), "e1b_replay": e1b_replay()}
    payload["status"] = ("pass" if all(payload[k]["status"] == "pass"
                                       for k in ("e1a_replay", "e1b_replay"))
                         else "fail")
    C.dump(C.OUT / "validation.json", payload)
    print(json.dumps(payload, indent=1))
