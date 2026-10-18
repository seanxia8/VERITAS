"""NHits trigger and prompt window (author decisions, 18 Oct 2026, fifth and sixth messages).

origin = the earliest digit time t with >= 4 digits in [t, t + 20 ns]; fallback to the window
start, counted. N = 3 is superseded (sixth message). Digits outside [-20, +60] ns of the trigger
are out-of-time and dropped from the model input; with no trigger nothing is dropped.
The requested properties are tested as measured, not as hoped:

- dark-only windows: the dark-coincidence bound (< 1e-3 per 1 us window) holds at N = 4: no
  trigger in 20000 windows (acceptable evidence, sixth message). N = 3 gave 1.2e-3, because the
  rule slides from every digit; the proposal's estimate counted fixed 20 ns bins.
- prompt events: within 1 ns of the true first prompt cluster with the timing width off; with
  the reference width the trigger is early by a median of 0.6-3.3 ns (the earliest of many
  jittered digits), so that case is bounded, not held to 1 ns.
- dark counts: the trigger is unchanged in most events; a dark digit up to 20 ns before the
  prompt cluster can complete a cluster and move the trigger earlier.
"""
from dataclasses import replace
import numpy as np
import pytest
from noise_module_lucid import response_manifest
from noise_module_lucid.photon_contract import PhotonBatch
from noise_module_lucid.readout import (Digits, NHITS_N, NHITS_WINDOW_NS, PROMPT_WINDOW_NS, dark_noise_window,
                                       nhits_trigger_time, prompt_window_mask, readout_chain, select_digits)

N = 1995


def reference():
    return response_manifest.reference_response(N, response_manifest.module_groups(np.arange(N)//19))


def prompt_event(k, npmt=300, pe=20):
    rng = np.random.default_rng(1000 + k)
    pmts = rng.choice(N, npmt, replace=False); n = rng.poisson(pe, npmt) + 1
    sensor = np.repeat(pmts, n)
    t = np.repeat(rng.uniform(0, 8, npmt), n) + rng.exponential(0.5, sensor.size) + 100
    o = np.argsort(t, kind="stable")
    return PhotonBatch(k, np.arange(sensor.size), sensor[o], t[o], rng.uniform(300, 500, sensor.size))


def read(b, p, seed=5):
    return readout_chain(b.sensor_id, b.arrival_time_ns, N, p, seed, photon_batch=b)


def true_cluster(b):
    first = np.full(N, np.inf); np.minimum.at(first, b.sensor_id, b.arrival_time_ns)
    return nhits_trigger_time(first[np.isfinite(first)])


def test_rule_constants_and_definition():
    assert (NHITS_N, NHITS_WINDOW_NS) == (4, 20.0)
    assert nhits_trigger_time([50.0, 0.0, 30.0, 45.0, 49.0]) == 30.0   # 0 is isolated; [30, 50] holds 4
    assert nhits_trigger_time([0.0, 5.0, 10.0, 20.0]) == 0.0           # closed window
    assert nhits_trigger_time([0.0, 5.0, 10.0, 20.5]) is None
    assert nhits_trigger_time([0.0, 10.0, 20.0]) is None and nhits_trigger_time([0.0, 10.0, 20.0], n=3) == 0.0
    assert nhits_trigger_time([1.0, 2.0]) is None and nhits_trigger_time([]) is None
    assert nhits_trigger_time([5.0, 7.0], n=2, window_ns=2.0) == 5.0
    d = Digits(sensor=np.arange(4), charge=np.ones(4), time=np.array([9.0, 3.0, 4.0, 22.0]), n_pe_true=np.ones(4, int))
    assert nhits_trigger_time(d) == 3.0
    for bad in (dict(n=0), dict(n=2.5), dict(n=True), dict(window_ns=-1.0), dict(window_ns=np.inf)):
        with pytest.raises(ValueError): nhits_trigger_time([1.0, 2.0, 3.0], **bad)
    with pytest.raises(ValueError): nhits_trigger_time([1.0, np.nan, 3.0])


def test_dark_only_windows_never_trigger_at_n4():
    # The bound, asserted: no dark-only trigger in 20000 windows (95% upper limit 1.5e-4 < 1e-3).
    p = reference(); windows = 20000
    times = [dark_noise_window(p, N, seed=9, run=78, window=w).time for w in range(windows)]
    assert sum(nhits_trigger_time(t) is not None for t in times) == 0
    assert sum(nhits_trigger_time(t, n=3) is not None for t in times) / windows > 2e-4   # N = 3 does see dark triples


def test_prompt_event_triggers_within_1ns_of_true_cluster_without_jitter():
    p = replace(reference(), dark_rate_hz=0.0, timing_width=None, timing_width_factor=1.0)
    for npmt, pe in ((300, 20), (100, 5), (1000, 3)):
        d = np.array([nhits_trigger_time(read(b, p)) - true_cluster(b) for b in (prompt_event(k, npmt, pe) for k in range(30))])
        assert np.all(np.abs(d) <= 1.0), (npmt, pe, d.min(), d.max())


def test_prompt_event_with_reference_jitter_is_early_by_a_few_ns():
    p = replace(reference(), dark_rate_hz=0.0)
    d = np.array([nhits_trigger_time(read(b, p)) - true_cluster(b) for b in (prompt_event(k) for k in range(30))])
    # The earliest of ~300 jittered digits sits in the low tail of the 0.5-2.5 ns widths.
    assert np.all(d < 0.5) and np.all(d > -6.0)
    assert -3.0 < np.median(d) < -0.5


def test_dark_counts_rarely_move_the_trigger_and_only_via_an_early_dark_digit():
    p = reference(); off = replace(p, dark_rate_hz=0.0)
    changed = 0
    for k in range(60):
        b = prompt_event(k)
        a, d = read(b, off), read(b, p)
        t0, t1 = nhits_trigger_time(a), nhits_trigger_time(d)
        if t1 != t0:
            changed += 1
            if t1 < t0 - 1.0:   # a big move: a dark digit completed an earlier cluster
                dark = d.time[d.n_dark_true > 0]
                assert np.any((dark >= t0 - NHITS_WINDOW_NS) & (dark < t0))
    assert changed / 60 <= 0.15   # measured 26 of 600 events (4.3%)


def test_dark_digits_after_the_cluster_never_move_it_and_an_early_one_can():
    prompt = np.array([100.0, 100.4, 101.0, 102.0, 105.0])
    assert nhits_trigger_time(np.r_[prompt, 300.0, 650.0, -150.0]) == 100.0
    assert nhits_trigger_time(np.r_[prompt, 85.0]) == 85.0   # 85 + the first two prompt digits within 20 ns


def test_prompt_window_rule_and_basis():
    assert PROMPT_WINDOW_NS == (-20.0, 60.0)
    # Basis: direct time of flight across WCTE (~4.2 m at 0.225 m/ns), walk, jitter and the
    # early-trigger tail all fit, with margin.
    assert PROMPT_WINDOW_NS[1] > 4.2 / 0.225 + 4.0 + 6.0 and PROMPT_WINDOW_NS[0] < -6.0
    t = np.array([79.0, 80.0, 100.0, 160.0, 160.5, -300.0])
    np.testing.assert_array_equal(prompt_window_mask(t, 100.0), [False, True, True, True, False, False])  # closed
    assert prompt_window_mask(t, None).all()                       # fallback: drop nothing
    assert prompt_window_mask(np.array([]), 1.0).size == 0
    for bad in (dict(trigger_ns=np.nan), dict(trigger_ns=0.0, window_ns=(5.0, 60.0)), dict(trigger_ns=0.0, window_ns=(-20.0, np.inf))):
        with pytest.raises(ValueError): prompt_window_mask(t, **bad)
    with pytest.raises(ValueError): prompt_window_mask(np.array([np.nan]), 0.0)


def test_select_digits_keeps_fields_aligned():
    d = Digits(sensor=np.array([4, 7, 9]), charge=np.array([1.0, 2.0, 3.0]), time=np.array([0.0, 1.0, 2.0]),
               n_pe_true=np.array([1, 2, 3]), counts={"x": 1}, n_dark_true=np.array([0, 2, 0]))
    s = select_digits(d, np.array([True, False, True]))
    np.testing.assert_array_equal(s.sensor, [4, 9]); np.testing.assert_array_equal(s.n_dark_true, [0, 0])
    np.testing.assert_array_equal(s.charge, [1.0, 3.0]); assert s.counts == {"x": 1} and s.counts is not d.counts
    with pytest.raises(ValueError): select_digits(d, np.array([True]))


def test_prompt_window_drops_dark_digits_not_source_and_keeps_the_trigger():
    p = reference(); dropped_dark = kept_dark = 0
    for k in range(40):
        d = read(prompt_event(k), p)
        trig = nhits_trigger_time(d); keep = prompt_window_mask(d, trig)
        source = d.n_pe_true > d.n_dark_true
        # No pure-source digit is out-of-time. A source digit can be, when an earlier dark PE on the
        # same PMT set its time (measured: 2 of 40 events, at -250 and -56 ns).
        assert not np.any(source & (d.n_dark_true == 0) & ~keep)
        assert np.all(d.time[source & ~keep] < trig)
        dropped_dark += int((~source & ~keep).sum()); kept_dark += int((~source & keep).sum())
        assert nhits_trigger_time(select_digits(d, keep)) == trig   # the window never moves the trigger
    # A 1 us window keeps about 80/1000 of the uniform dark digits.
    assert dropped_dark > 30 and kept_dark / (kept_dark + dropped_dark) < 0.2
