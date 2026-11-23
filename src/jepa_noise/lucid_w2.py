# SPDX-License-Identifier: MIT
"""r2-sparse W2: re-pack WCTE from the cached LUCiD digits with a wider readout window.

Runbook ``RUNBOOK_2026-11-24_R2_SPARSE_WCTE.md`` §W2. The cached LUCiD digits of ``lucid_v1``
(LUCiD e651e836, tier A1) are re-read with the VERITAS readout of
:func:`lucid_mc_generator.veritas_readout` and a new prompt window ``[-P, +A]``; the NHits trigger
rule, the detector state, run order, event order and pack layout are unchanged. Nothing here
simulates physics.

**Dark-noise extension.** The cache holds LUCiD dark noise only over
``[first deposit - 100 ns, last deposit + 100 ns]``; deposit times are not cached, so the earliest
and latest physics digit are their proxies. Every part of ``[trigger - P, trigger + A]`` outside
that span gets dark photoelectrons from LUCiD's ``generate_dark_noise`` (the digitizer's rate, all
PMTs), passed through the same LUCiD chain as :func:`lucid_mc_generator.electronics`
(``digitize_event``, ``apply_readout_resolution`` without TDC, the per-PMT width factor, TDC
rounding via ``_sample_time_jitter``, ``apply_discriminator``), with the random stream keyed by
``event_seed(seed, run_id, event_id, "dark_ext")``. Added digits are dark (``physics = False``).
Where an added digit lies within one integration window of a cached digit on the same PMT, LUCiD
would have formed one digit: the earlier time is kept and the charges are summed (counted).

The LUCiD import is lazy (only :func:`dark_extension` needs it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import lucid_bombs as lb
from . import lucid_mc_generator as mg
from noise_module_lucid.readout import nhits_trigger_time, prompt_window_mask

__all__ = ["DATASET_KIND_W2", "W2_ROOT", "window_rule", "uncovered_intervals", "dark_extension",
           "merge_digits", "veritas_readout_w2", "write_run_w2", "EXTRA_FIELDS"]

DATASET_KIND_W2 = "lucid_v1_w2"
W2_ROOT = Path("/ceph/dwong/jepa/data/lucid_v1_w2")
EXTRA_FIELDS = ("n_later", "q_total_window", "physics_first", "dark_ext_first")


def window_rule(base_rule: dict, pre_ns: float, post_ns: float) -> dict:
    """The detector's ``time_rule`` with the prompt window replaced by ``[-pre_ns, +post_ns]``."""
    if not (pre_ns > 0 and post_ns > 0):
        raise ValueError("window sides must be positive")
    rule = dict(base_rule)
    rule["prompt_window_ns"] = [-float(pre_ns), float(post_ns)]
    rule["window_source"] = "r2-sparse W1 rule (RUNBOOK_2026-11-24_R2_SPARSE_WCTE.md)"
    return rule


def _first_digit_times(dig: dict) -> np.ndarray:
    s, t = np.asarray(dig["sensor"]), np.asarray(dig["t"], float)
    order = np.lexsort((t, s))
    s, t = s[order], t[order]
    first = np.ones(s.size, bool)
    first[1:] = s[1:] != s[:-1]
    return t[first]


def uncovered_intervals(dig: dict, rule: dict, pad_ns: float) -> tuple[list, dict]:
    """Parts of ``[trigger - P, trigger + A]`` outside the cached dark span (trigger on cached digits)."""
    n, w = int(rule["n"]), float(rule["trigger_window_ns"])
    lo, hi = (float(v) for v in rule["prompt_window_ns"])
    phys = np.asarray(dig["physics"], bool)
    t = np.asarray(dig["t"], float)
    trig = nhits_trigger_time(_first_digit_times(dig), n=n, window_ns=w)
    info = {"trigger_cached": trig, "pre_uncovered_ns": 0.0, "post_uncovered_ns": 0.0}
    if trig is None or not phys.any():
        return [], info
    cov_lo, cov_hi = float(t[phys].min()) - pad_ns, float(t[phys].max()) + pad_ns
    win_lo, win_hi = trig + lo, trig + hi
    out = []
    if win_lo < cov_lo:
        out.append((win_lo, min(cov_lo, win_hi)))
        info["pre_uncovered_ns"] = float(min(cov_lo, win_hi) - win_lo)
    if win_hi > cov_hi:
        out.append((max(cov_hi, win_lo), win_hi))
        info["post_uncovered_ns"] = float(win_hi - max(cov_hi, win_lo))
    return out, info


def merge_digits(dig: dict, added: dict, window_ns: float) -> tuple[dict, dict]:
    """Add dark-extension digits to cached digits; merge within one integration window (same PMT).

    An added digit less than ``window_ns`` before a cached digit opens that digit's window: the
    added time is kept and the charges are summed. An added digit at most ``window_ns`` after a
    cached digit falls inside the cached digit's window: its charge is added there. The cached
    digit's physics flag is kept. Returns the new digit dict and counters.
    """
    s = np.asarray(dig["sensor"], np.int64).copy()
    pe = np.asarray(dig["pe"], float).copy()
    t = np.asarray(dig["t"], float).copy()
    phys = np.asarray(dig["physics"], bool).copy()
    ext = np.zeros(s.size, bool)
    new_s, new_pe, new_t = [], [], []
    n_before, n_after = 0, 0
    for a_s, a_pe, a_t in zip(np.asarray(added["sensor"], np.int64), np.asarray(added["pe"], float),
                              np.asarray(added["t"], float)):
        same = np.flatnonzero(s == a_s)
        if same.size:
            dt = t[same] - a_t
            before = same[(dt >= 0) & (dt < window_ns)]
            after = same[(dt < 0) & (-dt <= window_ns)]
            if before.size:
                k = before[np.argmin(t[before])]
                t[k], pe[k], ext[k] = a_t, pe[k] + a_pe, True
                n_before += 1
                continue
            if after.size:
                k = after[np.argmax(t[after])]
                pe[k] += a_pe
                n_after += 1
                continue
        new_s.append(a_s), new_pe.append(a_pe), new_t.append(a_t)
    k = len(new_s)
    out = dict(dig)
    out.update(sensor=np.concatenate([s, np.asarray(new_s, np.int64)]),
               pe=np.concatenate([pe, np.asarray(new_pe, float)]),
               t=np.concatenate([t, np.asarray(new_t, float)]),
               physics=np.concatenate([phys, np.zeros(k, bool)]),
               dark_ext=np.concatenate([ext, np.ones(k, bool)]))
    return out, {"n_ext_digits": int(np.asarray(added["sensor"]).size), "n_ext_new": k,
                 "n_merge_before": n_before, "n_merge_after": n_after}


def dark_extension(dig: dict, rule: dict, digitizer: dict, width_factor: np.ndarray, n_pmt: int,
                   rng: np.random.Generator, pad_ns: float) -> tuple[dict, dict]:
    """LUCiD dark noise over the uncovered parts of the new window, through LUCiD's chain."""
    from lucid.simulation.digitizer import (apply_discriminator, apply_readout_resolution,
                                            digitize_event, generate_dark_noise,
                                            _sample_time_jitter)
    intervals, info = uncovered_intervals(dig, rule, pad_ns)
    rate = float(digitizer.get("dark_rate_khz", 0.0))
    parts = [generate_dark_noise(n_pmt, rate, a, b, rng) for a, b in intervals if b > a]
    if parts:
        ds = np.concatenate([p[0] for p in parts])
        dt = np.concatenate([p[1] for p in parts])
        dq = np.concatenate([p[2] for p in parts])
    else:
        ds, dt, dq = np.empty(0, np.int64), np.empty(0), np.empty(0)
    info["n_dark_ext_pe"] = int(ds.size)
    if ds.size:
        res = digitize_event(ds, dt, dq, n_pmt, digitizer)
        no_tdc = dict(digitizer, tdc_ns=0.0)
        pe, t_jit = apply_readout_resolution(res.digit_pe_true, res.digit_time, no_tdc, rng)
        sens = res.digit_sensor_idx.astype(np.int64)
        t_scaled = res.digit_time + np.asarray(width_factor, float)[sens] * (t_jit - res.digit_time)
        t_dig = _sample_time_jitter(t_scaled, pe, {"time_model": "none",
                                                   "tdc_ns": float(digitizer.get("tdc_ns", 0.0))}, rng)
        keep = apply_discriminator(pe, digitizer)
        added = {"sensor": sens[keep], "pe": np.asarray(pe, float)[keep], "t": np.asarray(t_dig, float)[keep]}
    else:
        added = {"sensor": np.empty(0, np.int64), "pe": np.empty(0), "t": np.empty(0)}
    window = digitizer.get("integration_window_ns")
    window = np.inf if window is None else float(window)
    merged, counts = merge_digits(dig, added, window)
    info.update(counts)
    return merged, info


@dataclass
class ReadoutW2:
    hits: lb.HitReadout
    extra: dict


def veritas_readout_w2(dig: dict, n_pmt: int, gains: np.ndarray, rule: dict, readout_start_ns: float,
                       record: dict | None = None) -> ReadoutW2:
    """:func:`lucid_mc_generator.veritas_readout` with the window of ``rule`` plus extra per-hit fields.

    The hit arrays are computed statement for statement as in ``veritas_readout`` (earliest digit
    per PMT, detector-state charge, NHits token time, prompt window). Extra fields per kept hit:
    the number of later digits on the PMT inside the window, the total (gain-applied) charge of
    the PMT's digits inside the window, the first digit's physics flag and whether the first
    digit came from the dark-noise extension.
    """
    s, pe, t = dig["sensor"], dig["pe"], dig["t"]
    order = np.lexsort((t, s))
    s, pe, t, phys = s[order], pe[order], t[order], dig["physics"][order]
    ext = np.asarray(dig.get("dark_ext", np.zeros(len(order), bool)))[order]
    first = np.ones(s.size, bool)
    first[1:] = s[1:] != s[:-1]
    all_s, all_t, all_q = s, t, pe * np.asarray(gains, float)[s]
    s, pe, t, phys, ext = s[first], pe[first], t[first], phys[first], ext[first]
    q = pe * np.asarray(gains, float)[s]
    n, w = int(rule["n"]), float(rule["trigger_window_ns"])
    trigger = nhits_trigger_time(t, n=n, window_ns=w)
    source_trigger = nhits_trigger_time(t[phys], n=n, window_ns=w) if phys.any() else None
    origin = readout_start_ns if trigger is None else float(trigger)
    keep = np.asarray(prompt_window_mask(t, trigger, window_ns=tuple(rule["prompt_window_ns"])))
    if record is not None:
        record["n_events"] += 1
        record["n_no_trigger"] += int(trigger is None)
        record["n_prompt_dropped"] += int(np.count_nonzero(~keep))
        moved = source_trigger is not None and (trigger is None or abs(trigger - source_trigger) > 1e-9)
        record["n_dark_moved_trigger"] += int(moved)
    # extra fields: digits of each kept PMT inside the window
    in_win = np.asarray(prompt_window_mask(all_t, trigger, window_ns=tuple(rule["prompt_window_ns"])))
    starts = np.flatnonzero(first)
    group = np.cumsum(first) - 1
    n_in = np.bincount(group, weights=in_win.astype(float), minlength=starts.size)
    q_in = np.bincount(group, weights=np.where(in_win, all_q, 0.0), minlength=starts.size)
    hits = lb.HitReadout(s[keep].astype(np.int32), q[keep].astype(np.float32),
                         (t[keep] - origin).astype(np.float32), absolute_time_ns=t[keep],
                         trigger_ns=None if trigger is None else float(trigger),
                         source_trigger_ns=None if source_trigger is None else float(source_trigger))
    extra = {"n_later": (n_in[keep] - 1).astype(np.int16), "q_total_window": q_in[keep].astype(np.float32),
             "physics_first": phys[keep].astype(bool), "dark_ext_first": ext[keep].astype(bool)}
    return ReadoutW2(hits, extra)


def _extra_store(readouts) -> dict:
    out = {k: np.concatenate([r.extra[k] for r in readouts]) if readouts else np.empty(0)
           for k in EXTRA_FIELDS}
    offsets = np.concatenate([[0], np.cumsum([len(r.hits) for r in readouts])]).astype(np.int64)
    return dict(out, offsets=offsets)


def write_run_w2(directory, run, event_ids, truths, stats, readouts, twin_readouts, state_sha256,
                 geometry, cfg: dict, rule: dict, provenance: dict, record: dict, ext_stats: list) -> dict:
    """Per-run shard in the ``lucid_v1`` layout (hits.npz, twin_hits.npz, events.npz, manifest.json,
    SHA256SUMS) plus ``hits_extra.npz``; written to ``<dir>.tmp`` and renamed. Never overwrites."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        for p in tmp.iterdir():
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    lb.write_sparse_hits(tmp / "hits.npz", lb.sparse_from_readouts(event_ids, [r.hits for r in readouts]))
    with open(tmp / "hits_extra.npz", "wb") as handle:
        np.savez_compressed(handle, **_extra_store(readouts))
    if twin_readouts is not None:
        lb.write_sparse_hits(tmp / "twin_hits.npz",
                             lb.sparse_from_readouts(event_ids, [r.hits for r in twin_readouts]))
    lrun = mg.LucidRun(run, event_ids, truths, [r.hits for r in readouts], None, stats, record, state_sha256)
    payload = mg.truth_payload(lrun)
    with open(tmp / "events.npz", "wb") as handle:
        np.savez_compressed(handle, **payload)
    ext = {k: int(sum(e.get(k, 0) for e in ext_stats)) for k in
           ("n_dark_ext_pe", "n_ext_digits", "n_ext_new", "n_merge_before", "n_merge_after")}
    ext["pre_uncovered_ns_mean"] = float(np.mean([e["pre_uncovered_ns"] for e in ext_stats])) if ext_stats else 0.0
    ext["post_uncovered_ns_mean"] = float(np.mean([e["post_uncovered_ns"] for e in ext_stats])) if ext_stats else 0.0
    ext["n_trigger_changed"] = int(sum(e.get("trigger_changed", 0) for e in ext_stats))
    manifest = {
        "schema": "jepa-lucid-mc-manifest-v1", "dataset_kind": DATASET_KIND_W2,
        "source_dataset_kind": mg.DATASET_KIND, "source_data": "lucid_v1 cached LUCiD digits",
        "sampling_tier": mg.SAMPLING_TIER, "registered": False, "reported_results": False,
        "readout_label": f"lucid_{mg.LUCID_SHA[:8]}_{provenance['kernel']['digitizer']['model']}",
        "dataset_seed": int(run.seed), "geometry_name": geometry.name,
        "geometry_sha256": geometry.digest, "n_runs": 1, "n_events": int(event_ids.size),
        "runs": [{"run_id": run.run_id, "event_namespace": run.event_namespace,
                  "event_id_start": run.event_start, "event_id_stop": run.event_stop,
                  "n_events": run.n_events, "law": run.law, "setting": run.setting,
                  "state_sha256": state_sha256, "event_ids_sha256": lb._event_ids_sha(run)}],
        "detector": provenance["detector"], "lucid": provenance["kernel"],
        "lucid_v1_config_sha256": cfg["_sha256"], "time_rule": rule,
        "veritas_steps": list(mg.VERITAS_STEPS) + [
            "r2-sparse W2: prompt window from the W1 rule; LUCiD dark noise (generate_dark_noise + "
            "LUCiD digitizer chain) over the parts of the window outside the cached dark span, keyed "
            "event_seed(seed, run_id, event_id, 'dark_ext'), merged within one integration window"],
        "truth_field_status": mg.TRUTH_FIELD_STATUS, "code_sha": provenance.get("code_sha"),
        "repack_code_sha": provenance.get("repack_code_sha"),
        "twin_present": twin_readouts is not None, "expected_pe_pmt_present": False,
        "n_hits": int(sum(len(r.hits) for r in readouts)), "extra_fields": list(EXTRA_FIELDS),
        "dark_extension": ext, **record,
        "regime": {"true_pe_quantiles_5_50_95": [float(np.quantile([s["true_pe"] for s in stats], q))
                                                  for q in (0.05, 0.5, 0.95)],
                   "n_lucid_untriggered": int(sum(not s["lucid_triggered"] for s in stats)),
                   "n_thinning_overflow_rays": int(sum(s["n_overflow"] for s in stats)),
                   "kernel_calls": int(sum(s["n_calls"] for s in stats)),
                   "photons": int(sum(s["n_photons"] for s in stats))},
        "event_truth_sha256": lb._sha((tmp / "events.npz").read_bytes()),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    lb.write_sha256sums(tmp)
    tmp.rename(directory)
    return manifest
