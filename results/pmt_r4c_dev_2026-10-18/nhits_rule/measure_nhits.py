"""Measured properties of the NHits trigger rule (author decision 18 Oct 2026, fifth message).

Synthetic prompt events (the test-suite generator) and keyed dark-only windows on the manifest
response. Run from the repository root:
  PYTHONPATH=src python -B results/pmt_r4c_dev_2026-10-18/nhits_rule/measure_nhits.py
"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import beta, poisson
sys.path.insert(0, str(Path("src/noise_module_lucid/tests").resolve()))
from test_nhits_trigger import N, prompt_event, read, reference, true_cluster
from noise_module_lucid.readout import dark_noise_window, nhits_trigger_time


def ci(k, n):  # 95% Clopper-Pearson
    return [float(beta.ppf(.025, k, n-k+1)) if k else 0.0, float(beta.ppf(.975, k+1, n-k)) if k < n else 1.0]


ref = reference(); out = {}
W = 20000
times = [dark_noise_window(ref, N, seed=9, run=77, window=w).time for w in range(W)]
mu20 = ref.dark_rate_hz * 20e-9 * N; ndig = float(np.mean([len(t) for t in times]))
out["dark_only"] = dict(windows=W, mean_digits_per_window=ndig, expected_dark_digits_per_20ns=mu20)
for n in (2, 3, 4, 5):
    k = sum(nhits_trigger_time(t, n=n) is not None for t in times)
    out["dark_only"][f"n{n}"] = dict(triggers=int(k), rate=k/W, ci95=ci(k, W),
                                    fixed_bin_estimate=50*float(poisson.sf(n-1, mu20)), sliding_estimate=ndig*float(poisson.sf(n-2, mu20)))
nojit = replace(ref, dark_rate_hz=0.0, timing_width=None, timing_width_factor=1.0); jit = replace(ref, dark_rate_hz=0.0)
out["prompt"] = {}
for npmt, pe in ((300, 20), (100, 5), (30, 50), (1000, 3)):
    ev = [prompt_event(k, npmt, pe) for k in range(200)]
    row = {}
    for name, p in (("no_jitter", nojit), ("reference_jitter", jit)):
        d = np.array([nhits_trigger_time(read(b, p)) - true_cluster(b) for b in ev])
        row[name] = dict(median_ns=float(np.median(d)), p05_ns=float(np.quantile(d, .05)), p95_ns=float(np.quantile(d, .95)),
                         max_abs_ns=float(np.abs(d).max()), within_1ns=float(np.mean(np.abs(d) <= 1)))
    out["prompt"][f"{npmt}pmt_{pe}pe"] = row
ch = []; big = 0; explained = 0
for k in range(600):
    b = prompt_event(k); a, d = read(b, jit), read(b, ref)
    t0, t1 = nhits_trigger_time(a), nhits_trigger_time(d)
    if t1 != t0:
        ch.append(t1 - t0)
        if t1 < t0 - 1:
            big += 1; dk = d.time[d.n_dark_true > 0]; explained += bool(np.any((dk >= t0 - 20) & (dk < t0)))
out["dark_invariance"] = dict(events=600, changed=len(ch), changed_fraction=len(ch)/600, ci95=ci(len(ch), 600), moved_earlier_by_more_than_1ns=big,
                              of_which_dark_digit_in_preceding_20ns=explained, shifts_ns=sorted(float(x) for x in ch))
Path(__file__).with_name("nhits_measured.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: (v if k != "dark_invariance" else {kk: vv for kk, vv in v.items() if kk != "shifts_ns"}) for k, v in out.items()}, indent=1))
