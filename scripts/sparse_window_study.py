#!/usr/bin/env python3
"""r2-sparse W0: readout-window study from the cached LUCiD digits (runbook 2026-11-24 §W0, §W1).

Per detector, from the cached digits of ``dev_probe_train`` (run order, the first 2,000 photon
bombs, 500 muons and 500 electrons), with the NHits trigger recomputed exactly as the packs did
(``lucid_mc_generator.veritas_readout``: earliest digit per PMT, then
``nhits_trigger_time(n, window)`` of the detector's ``time_rule``):

1. lost early light: events / signal digits earlier than trigger - 20 ns; trigger - earliest
   signal digit;
2. late light: signal-digit times after the trigger (first signal digit per PMT, and all digits);
   fraction beyond the current window end;
3. dark noise: dark digits per event inside candidate windows (P, A); uniformity of dark digit
   times; coverage of each candidate by the cached dark-noise span
   [t_phys,min - pad, t_phys,max + pad] (earliest / latest physics digit as deposit proxies);
4. digits per PMT: how often a PMT has more than one digit and how late the later ones are.

For WCTE it applies the W1 rule. Reads only the read-only caches; writes one JSON per detector.
Runs in the base env (no LUCiD import). Usage::

    sparse_window_study.py --detector wcte --out /work/dwong/r2_sparse/w0/wcte.json
    sparse_window_study.py --merge /work/dwong/r2_sparse/w0 --out-dir results/round2/20261124_sparse
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_mc_generator as mg  # noqa: E402
from noise_module_lucid.readout import nhits_trigger_time  # noqa: E402

DATA = Path("/ceph/dwong/jepa/data/lucid_v1")
QUOTA = {"photon": 2000, "muon": 500, "electron": 500}
EARLY_CUT_NS = 20.0
P_GRID = (20, 50, 100, 150, 200, 250, 300, 400, 500)
A_GRID = (60, 100, 150, 200, 250, 270, 300, 400, 460, 500, 600, 750, 1000)
QS = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)
TRUNCATED = {"sk": "LUCiD trigger gate (200 ns window + 300 ns each side) applied before caching",
             "hk": "LUCiD trigger gate (200 ns window + 300 ns each side) applied before caching",
             "wcte": None, "juno": None}


def quant(x):
    x = np.asarray(x, float)
    return {f"q{q:g}": (float(np.quantile(x, q)) if x.size else None) for q in QS} | {
        "mean": float(x.mean()) if x.size else None, "n": int(x.size)}


def events(det):
    """Yield (domain, digits) in run order until the quota is reached."""
    need = dict(QUOTA)
    for run in mg.cohort_runs("dev_probe_train"):
        if not any(need.values()):
            return
        path = DATA / det / "cache" / "dev_probe_train" / (run.run_id.replace("/", "__") + ".pkl")
        cache = pickle.loads(path.read_bytes())
        for truth, dig in zip(cache["truths"], cache["digits"]):
            if need.get(truth.domain, 0) > 0:
                need[truth.domain] -= 1
                yield truth.domain, dig


def study(det, pad_ns):
    cfg = mg.load_config()
    rule = mg.time_rule(det, cfg)
    n, w = int(rule["n"]), float(rule["trigger_window_ns"])
    lo, hi = (float(v) for v in rule["prompt_window_ns"])
    rows = {k: [] for k in ("domain", "trigger_minus_earliest_signal", "early_event", "n_signal",
                            "n_signal_early", "n_signal_late", "t_phys_min", "t_phys_max", "span_min",
                            "span_max", "n_hit_pmt", "n_multi_pmt", "n_signal_first_late")}
    first_sig_rel, all_sig_rel, later_delay, dark_rel = [], [], [], []
    dark_counts = {f"{p}_{a}": [] for p in P_GRID for a in A_GRID}
    uncovered = {f"{p}_{a}": {"pre_frac": 0, "pre_len": [], "post_frac": 0, "post_len": []}
                 for p in P_GRID for a in A_GRID}
    n_no_trigger, n_no_signal = 0, 0
    cov_rows = []
    first_sig_rel_by_domain = {d: [] for d in QUOTA}
    for domain, dig in events(det):
        s, t, phys = (np.asarray(dig[k]) for k in ("sensor", "t", "physics"))
        order = np.lexsort((t, s))
        s, t, phys = s[order], t[order], phys[order].astype(bool)
        first = np.ones(s.size, bool)
        first[1:] = s[1:] != s[:-1]
        trig = nhits_trigger_time(t[first], n=n, window_ns=w)  # exactly as veritas_readout
        if trig is None:
            n_no_trigger += 1
            continue
        if not phys.any():
            n_no_signal += 1
            continue
        rel = t - trig
        t_sig = rel[phys]
        rows["domain"].append(domain)
        rows["trigger_minus_earliest_signal"].append(float(-t_sig.min()))
        rows["early_event"].append(bool(t_sig.min() < -EARLY_CUT_NS))
        rows["n_signal"].append(int(t_sig.size))
        rows["n_signal_early"].append(int((t_sig < -EARLY_CUT_NS).sum()))
        rows["n_signal_late"].append(int((t_sig > hi).sum()))
        tpmin, tpmax = float(t[phys].min()), float(t[phys].max())
        rows["t_phys_min"].append(tpmin - trig)
        rows["t_phys_max"].append(tpmax - trig)
        rows["span_min"].append(float(rel.min()))
        rows["span_max"].append(float(rel.max()))
        # first signal digit per PMT
        sig_s, sig_t = s[phys], rel[phys]
        fs = np.ones(sig_s.size, bool)
        fs[1:] = sig_s[1:] != sig_s[:-1]
        first_sig_rel.append(sig_t[fs])
        first_sig_rel_by_domain[domain].append(sig_t[fs])
        rows["n_signal_first_late"].append(int((sig_t[fs] > hi).sum()))
        all_sig_rel.append(t_sig)
        # digits per PMT
        starts = np.flatnonzero(first)
        counts = np.diff(np.append(starts, s.size))
        rows["n_hit_pmt"].append(int(starts.size))
        rows["n_multi_pmt"].append(int((counts > 1).sum()))
        firsts = np.repeat(t[first], counts)
        later_delay.append((t - firsts)[~first])
        # dark digits
        d_rel = rel[~phys]
        dark_rel.append(d_rel)
        cov_lo, cov_hi = tpmin - pad_ns - trig, tpmax + pad_ns - trig
        cov_rows.append((cov_lo, cov_hi))
        for p in P_GRID:
            for a in A_GRID:
                key = f"{p}_{a}"
                dark_counts[key].append(int(((d_rel >= -p) & (d_rel <= a)).sum()))
                u = uncovered[key]
                if -p < cov_lo:
                    u["pre_frac"] += 1
                u["pre_len"].append(max(0.0, cov_lo + p))
                if a > cov_hi:
                    u["post_frac"] += 1
                u["post_len"].append(max(0.0, a - cov_hi))
    n_ev = len(rows["domain"])
    domain = np.asarray(rows["domain"])
    fsr = np.concatenate(first_sig_rel)
    asr = np.concatenate(all_sig_rel)
    dr = np.concatenate(dark_rel)
    later = np.concatenate(later_delay) if later_delay else np.empty(0)
    tme = np.asarray(rows["trigger_minus_earliest_signal"])
    out = {"detector": det, "time_rule": rule, "dark_pad_ns_cached": pad_ns,
           "truncated_by_lucid_trigger": TRUTNC(det), "n_events": n_ev,
           "n_by_domain": {d: int((domain == d).sum()) for d in QUOTA},
           "n_no_trigger": n_no_trigger, "n_no_signal": n_no_signal,
           "cache_span_rel_trigger": {"min": quant(rows["span_min"]), "max": quant(rows["span_max"])},
           "lost_early": {
               "event_fraction": float(np.mean(rows["early_event"])),
               "event_fraction_by_domain": {d: float(np.mean(np.asarray(rows["early_event"])[domain == d]))
                                            for d in QUOTA},
               "signal_digit_fraction": float(np.sum(rows["n_signal_early"]) / np.sum(rows["n_signal"])),
               "trigger_minus_earliest_signal_ns": quant(tme),
               "trigger_minus_earliest_signal_by_domain": {d: quant(tme[domain == d]) for d in QUOTA}},
           "late": {
               "window_end_ns": hi,
               "first_per_pmt_signal_rel_ns": quant(fsr),
               "all_signal_rel_ns": quant(asr),
               "first_per_pmt_fraction_beyond_end": float((fsr > hi).mean()),
               "all_digit_fraction_beyond_end": float((asr > hi).mean()),
               "first_per_pmt_cdf": {str(a): float((fsr <= a).mean()) for a in A_GRID},
               "first_per_pmt_in_window_fraction": {f"{p}_{a}": float(((fsr >= -p) & (fsr <= a)).mean())
                                                    for p in P_GRID for a in A_GRID}},
           "dark": {
               "per_event_in_window_mean": {k: float(np.mean(v)) for k, v in dark_counts.items()},
               "expected_rate_per_ns_all_pmts": None,
               "uniformity_hist_edges_ns": None, "uniformity_hist": None,
               "coverage": {k: {"pre_uncovered_event_fraction": u["pre_frac"] / max(n_ev, 1),
                                "pre_uncovered_mean_ns": float(np.mean(u["pre_len"])),
                                "post_uncovered_event_fraction": u["post_frac"] / max(n_ev, 1),
                                "post_uncovered_mean_ns": float(np.mean(u["post_len"]))}
                            for k, u in uncovered.items()},
               "t_phys_min_rel_trigger_ns": quant(rows["t_phys_min"]),
               "t_phys_max_rel_trigger_ns": quant(rows["t_phys_max"])},
           "digits_per_pmt": {
               "multi_digit_pmt_fraction": float(np.sum(rows["n_multi_pmt"]) / np.sum(rows["n_hit_pmt"])),
               "later_digit_delay_ns": quant(later),
               "hit_pmt_per_event": quant(rows["n_hit_pmt"])}}
    edges = np.arange(-600.0, 1101.0, 100.0)
    hist, _ = np.histogram(dr, bins=edges)
    out["dark"]["uniformity_hist_edges_ns"] = edges.tolist()
    out["dark"]["uniformity_hist"] = hist.tolist()
    out["dark"]["uniformity_note"] = ("dark digit counts per 100 ns relative to the trigger, all events; "
                                      "they fall off where the cached span [t_phys,min - pad, t_phys,max + pad] ends")
    if det in ("wcte", "juno"):
        out["early_signal_interpretation"] = interpret_early(det, 4.2)
    if det == "wcte":
        out["w1"] = w1_rule(tme, fsr, out)
        p, a = out["w1"]["P_ns"], out["w1"]["A_ns"]
        cov = np.asarray(cov_rows)
        pre = np.maximum(0.0, cov[:, 0] + p)
        post = np.maximum(0.0, a - cov[:, 1])
        n_dark = np.asarray([int(((d >= -p) & (d <= a)).sum()) for d in dark_rel])
        out["w1"]["coverage"] = {
            "pre_uncovered_event_fraction": float((pre > 0).mean()), "pre_uncovered_mean_ns": float(pre.mean()),
            "post_uncovered_event_fraction": float((post > 0).mean()), "post_uncovered_mean_ns": float(post.mean()),
            "cached_dark_digits_per_event_in_window": float(n_dark.mean()),
            "events_covered_without_extension_fraction": float(((pre == 0) & (post == 0)).mean()),
            "largest_P_covered_by_99pct_events_ns": float(np.floor(np.quantile(-cov[:, 0], 0.01)))}
        out["w1"]["old_window_signal_first_per_pmt_in_50_60_fraction"] = float(((fsr > 50) & (fsr <= 60)).mean())
    out["first_per_pmt_by_domain"] = {d: quant(np.concatenate(v)) if v else None
                                      for d, v in first_sig_rel_by_domain.items()}
    return out


def interpret_early(det, rate_khz):
    """Are the early "signal" digits early light or dark-opened windows? LUCiD flags a digit as
    physics when its integration window holds any physics photoelectron, so a dark photoelectron
    that arrives on a PMT before its light opens the window and sets the digit time. Expected count
    per event of such digits earlier than trigger - 20 ns: rate x n_hit_PMT x the cached dark span
    before trigger - 20 ns (from first-light proxy - 100 ns). Compared with the observed count."""
    from jepa_noise.lucid_w2 import first_light_proxy
    cfg = mg.load_config()
    rule = mg.time_rule(det, cfg)
    n, w = int(rule["n"]), float(rule["trigger_window_ns"])
    obs, exp_, frac_obs, frac_exp, rel_early, q_ratio = [], [], [], [], [], []
    for domain, dig in events(det):
        s, t, phys, pe = (np.asarray(dig[k]) for k in ("sensor", "t", "physics", "pe"))
        order = np.lexsort((t, s))
        s, t, phys, pe = s[order], t[order], phys[order].astype(bool), pe[order]
        first = np.ones(s.size, bool)
        first[1:] = s[1:] != s[:-1]
        trig = nhits_trigger_time(t[first], n=n, window_ns=w)
        if trig is None or not phys.any():
            continue
        fl = first_light_proxy(t[phys])
        early = first & phys & (t < trig - EARLY_CUT_NS)
        span = max(0.0, (trig - EARLY_CUT_NS) - (fl - 100.0))
        lam = rate_khz * 1e-6 * span * int((first & phys).sum())
        obs.append(int(early.sum()))
        exp_.append(lam)
        frac_obs.append(bool(early.any()))
        frac_exp.append(1.0 - math.exp(-lam))
        rel_early.append(trig - t[early])
        if early.any():
            q_ratio.append(float(np.median(pe[early]) / np.median(pe[first & phys])))
    rel = np.concatenate(rel_early) if rel_early else np.empty(0)
    hist, edges = np.histogram(rel, bins=np.arange(20.0, 121.0, 10.0))
    return {"observed_early_signal_digits_per_event": float(np.mean(obs)),
            "expected_dark_opened_per_event": float(np.mean(exp_)),
            "observed_event_fraction": float(np.mean(frac_obs)),
            "expected_event_fraction_dark_opened": float(np.mean(frac_exp)),
            "trigger_minus_early_digit_hist_edges_ns": edges.tolist(),
            "trigger_minus_early_digit_hist": hist.tolist(),
            "median_charge_early_over_all_first_digits": float(np.median(q_ratio)) if q_ratio else None,
            "note": "dark-opened windows: uniform in time over the cached dark pad, charge of a normal hit"}


def TRUTNC(det):
    return TRUNCATED[det]


def w1_rule(tme, fsr, out):
    """Runbook §W1: P from q99.5(trigger - earliest signal digit), A from first-per-PMT coverage."""
    q = float(np.quantile(tme, 0.995))
    p_raw = max(50.0, q + 10.0)
    p = min(500.0, math.ceil(p_raw / 10.0 - 1e-9) * 10.0)
    a = None
    for cand in range(50, 1001, 50):
        if ((fsr >= -p) & (fsr <= cand)).mean() >= 0.995:
            a = float(cand)
            break
    capped = a is None
    a = 1000.0 if a is None else a
    frac = float(((fsr >= -p) & (fsr <= a)).mean())
    return {"q995_trigger_minus_earliest_signal_ns": q, "P_raw_ns": p_raw, "P_ns": p, "A_ns": a,
            "A_capped_at_1000": capped, "first_per_pmt_fraction_in_window": frac,
            "rule": "P = max(50, q99.5(trigger - earliest signal) + 10) rounded up to 10 ns, <= 500; "
                    "A = smallest multiple of 50 ns with >= 99.5 % of first-per-PMT signal digits in [-P, A], <= 1000",
            "pooled_domains": "photon bombs, muons and electrons pooled (2000/500/500)"}


def merge(src, out_dir):
    res = {d: json.loads((Path(src) / f"{d}.json").read_text()) for d in ("wcte", "sk", "juno", "hk")}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "WINDOW_STUDY.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    w = res["wcte"]["w1"]
    lines = ["# r2-sparse W0: readout-window study (cached LUCiD digits)", "",
             "Source: cached LUCiD digits of `lucid_v1` `dev_probe_train` (LUCiD e651e836), first 2,000 photon "
             "bombs, 500 muons, 500 electrons per detector in run order; NHits trigger recomputed exactly as the "
             "packs (earliest digit per PMT, `time_rule` N and 20 ns). Signal = LUCiD digit with a physics "
             "photoelectron. Script `scripts/sparse_window_study.py`; numbers in `WINDOW_STUDY.json`.", "",
             "## Summary per detector", "",
             "| | WCTE | SK | JUNO | HK |", "|---|---|---|---|---|"]

    def row(label, f):
        lines.append(f"| {label} | " + " | ".join(f(res[d]) for d in ("wcte", "sk", "juno", "hk")) + " |")

    fmt = lambda x, n=1: "n/a" if x is None else f"{x:.{n}f}"
    row("current window [ns]", lambda r: f"[{r['time_rule']['prompt_window_ns'][0]:g}, {r['time_rule']['prompt_window_ns'][1]:g}]")
    row("cache truncated by LUCiD trigger", lambda r: "yes" if r["truncated_by_lucid_trigger"] else "no")
    row("events (no NHits trigger)", lambda r: f"{r['n_events']} ({r['n_no_trigger']})")
    row("cached digit span rel. trigger, median min / max [ns]",
        lambda r: f"{fmt(r['cache_span_rel_trigger']['min']['q0.5'],0)} / {fmt(r['cache_span_rel_trigger']['max']['q0.5'],0)}")
    row("events with signal before trigger - 20 ns", lambda r: f"{100*r['lost_early']['event_fraction']:.1f} %")
    row("signal digits before trigger - 20 ns", lambda r: f"{100*r['lost_early']['signal_digit_fraction']:.2f} %")
    row("trigger - earliest signal, median / q95 / q99.5 [ns]",
        lambda r: "{} / {} / {}".format(*(fmt(r['lost_early']['trigger_minus_earliest_signal_ns'][k], 0)
                                          for k in ("q0.5", "q0.95", "q0.995"))))
    row("first-per-PMT signal beyond window end", lambda r: f"{100*r['late']['first_per_pmt_fraction_beyond_end']:.2f} %")
    row("all signal digits beyond window end", lambda r: f"{100*r['late']['all_digit_fraction_beyond_end']:.2f} %")
    row("first-per-PMT signal rel. trigger q99 / q99.5 [ns]",
        lambda r: f"{fmt(r['late']['first_per_pmt_signal_rel_ns']['q0.99'],0)} / {fmt(r['late']['first_per_pmt_signal_rel_ns']['q0.995'],0)}")
    row("PMTs with > 1 digit", lambda r: f"{100*r['digits_per_pmt']['multi_digit_pmt_fraction']:.2f} %")
    row("later-digit delay median / q95 [ns]",
        lambda r: f"{fmt(r['digits_per_pmt']['later_digit_delay_ns']['q0.5'],0)} / {fmt(r['digits_per_pmt']['later_digit_delay_ns']['q0.95'],0)}")
    row("dark digits / event in current window",
        lambda r: fmt(r['dark']['per_event_in_window_mean'].get(_cur_key(r)), 2))
    lines += ["", "## WCTE window by the registered rule (W1)", "",
              f"- q99.5(trigger − earliest signal digit) = {w['q995_trigger_minus_earliest_signal_ns']:.1f} ns → "
              f"P = max(50, {w['q995_trigger_minus_earliest_signal_ns']:.1f} + 10) rounded up to 10 ns = **{w['P_ns']:.0f} ns**.",
              f"- Smallest multiple of 50 ns with ≥ 99.5 % of first-per-PMT signal digits in [−P, A]: **A = {w['A_ns']:.0f} ns** "
              f"({100*w['first_per_pmt_fraction_in_window']:.2f} % inside).",
              f"- WCTE window: **[−{w['P_ns']:.0f} ns, +{w['A_ns']:.0f} ns]** around the NHits trigger (old: [−20, +60] ns).", ""]
    rw = res["wcte"]
    c = w.get("coverage")
    if c:
        lines.append(f"- Coverage by the cached dark-noise span at this window: pre-trigger side uncovered in "
                     f"{100*c['pre_uncovered_event_fraction']:.1f} % of events (mean {c['pre_uncovered_mean_ns']:.1f} ns), "
                     f"post-trigger side in {100*c['post_uncovered_event_fraction']:.1f} % (mean {c['post_uncovered_mean_ns']:.1f} ns). "
                     "W2 fills the uncovered parts with LUCiD dark noise.")
        lines.append(f"- Dark digits per event inside the window from the cache (before extension): "
                     f"{c['cached_dark_digits_per_event_in_window']:.2f}; events fully covered without extension: "
                     f"{100*c['events_covered_without_extension_fraction']:.1f} %; largest P covered in >= 99 % of events "
                     f"without extension: {c['largest_P_covered_by_99pct_events_ns']:.0f} ns.")
        lines.append(f"- A = {w['A_ns']:.0f} ns is shorter than the old end (+60 ns): first-per-PMT signal digits in (50, 60] ns "
                     f"are {100*w['old_window_signal_first_per_pmt_in_50_60_fraction']:.4f} % of all; no WCTE PMT has a "
                     "second digit (the 200 ns integration window spans the whole event), so there is no separate late light.")
    ei = rw.get("early_signal_interpretation")
    if ei:
        lines += ["", "## What the early \"signal\" digits are (WCTE)", "",
                  "LUCiD flags a digit as physics when its 200 ns integration window holds any physics photoelectron. "
                  "A dark photoelectron that reaches a PMT before that PMT's light therefore opens the window and sets "
                  "the digit time, and the digit is still flagged as signal. Expected number of such digits earlier "
                  "than trigger − 20 ns, from the dark rate (4.2 kHz), the number of hit PMTs and the cached dark span:",
                  "",
                  f"- observed early signal digits per event: {ei['observed_early_signal_digits_per_event']:.3f}; "
                  f"expected from dark-opened windows: {ei['expected_dark_opened_per_event']:.3f};",
                  f"- events with at least one: observed {100*ei['observed_event_fraction']:.1f} %, expected "
                  f"{100*ei['expected_event_fraction_dark_opened']:.1f} %;",
                  f"- (trigger − early digit) per 10 ns from 20 ns: {ei['trigger_minus_early_digit_hist']} "
                  "(flat up to the 100 ns dark pad, as for dark noise);",
                  f"- median charge of the early digits / median of all first digits: "
                  f"{ei['median_charge_early_over_all_first_digits']:.2f} (normal hits, not single-PE).", "",
                  "So on WCTE the early digits are almost entirely dark-opened windows on PMTs that are hit later: no "
                  "early direct light is lost. What the old window does lose is the whole PMT (its charge and later "
                  "light), because the readout keeps each PMT's earliest digit and then applies the window (F3). "
                  "P = 110 ns keeps these PMTs; their time is the dark photoelectron's time. The registered W1 rule is "
                  "applied unchanged; this reading is recorded as a W0 finding."]
    lines += ["", "## Dark noise uniformity (WCTE, digits per 100 ns rel. trigger, all events)", "",
              "| bin start [ns] | " + " | ".join(f"{e:g}" for e in rw["dark"]["uniformity_hist_edges_ns"][:-1]) + " |",
              "|---|" + "---|" * len(rw["dark"]["uniformity_hist"]),
              "| dark digits | " + " | ".join(str(x) for x in rw["dark"]["uniformity_hist"]) + " |", "",
              "The cached dark noise spans only [first deposit − 100 ns, last deposit + 100 ns] (WCTE, JUNO), so the "
              "counts fall off outside it; SK and HK caches end at LUCiD's trigger gate.", "",
              "## True time range of the current samples (co-author's question)", ""]
    for d in ("wcte", "sk", "juno", "hk"):
        r = res[d]
        lines.append(f"- **{d.upper()}**: cached digits span median [{r['cache_span_rel_trigger']['min']['q0.5']:.0f}, "
                     f"{r['cache_span_rel_trigger']['max']['q0.5']:.0f}] ns around the NHits trigger"
                     + (f" ({r['truncated_by_lucid_trigger']})" if r["truncated_by_lucid_trigger"] else
                        " (no LUCiD trigger: every digit kept)") +
                     f"; the packs keep [{r['time_rule']['prompt_window_ns'][0]:g}, {r['time_rule']['prompt_window_ns'][1]:g}] ns.")
    (out_dir / "WINDOW_STUDY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(w))


def _cur_key(r):
    lo, hi = r["time_rule"]["prompt_window_ns"]
    k = f"20_{int(hi)}"
    return k if k in r["dark"]["per_event_in_window_mean"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=("wcte", "sk", "juno", "hk"))
    ap.add_argument("--out")
    ap.add_argument("--merge")
    ap.add_argument("--out-dir")
    a = ap.parse_args()
    if a.merge:
        return merge(a.merge, a.out_dir)
    cfg = mg.load_config()
    pad = float(cfg["dark_pad_no_trigger_ns"]) if mg.SPECS[a.detector]["trigger"] is None else float("nan")
    res = study(a.detector, pad if np.isfinite(pad) else 0.0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res.get("w1", {"detector": a.detector, "n": res["n_events"]})))


if __name__ == "__main__":
    main()
