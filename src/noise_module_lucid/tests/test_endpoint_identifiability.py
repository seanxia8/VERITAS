"""Engineering checks for the dev_unreviewed D-E preflight diagnostics (analytic cases only)."""
import numpy as np
import pytest
from noise_module_lucid import endpoint_identifiability as ei

R, H = 1.0, 2.0  # cylinder x^2+y^2 < 1, |z| < 1


def test_status_label():
    assert ei.STATUS == "dev_unreviewed"


@pytest.mark.parametrize("p0,p1,length", [
    ((0, 0, -.5), (0, 0, .5), 1.0),        # fully inside
    ((-2, 0, 0), (2, 0, 0), 2.0),          # crosses the barrel twice
    ((0, 0, -3), (0, 0, 3), 2.0),          # crosses both end caps
    ((0, 0, 0), (0, 0, 5), 1.0),           # starts inside, leaves through the cap
    ((2, 2, 0), (3, 3, 0), 0.0),           # outside
    ((0, 0, 0), (0, 0, 0), 0.0),           # zero length inside counts zero
    ((.5, 0, 0), (.5, 0, 4), 1.0),         # axial segment inside the barrel
])
def test_clip_segment_length_analytic(p0, p1, length):
    assert ei.clip_segment_length(p0, p1, R, H) == pytest.approx(length, abs=1e-12)


def test_clip_rejects_bad_input():
    with pytest.raises(ValueError): ei.clip_segment_length((0, 0), (0, 0, 1), R, H)
    with pytest.raises(ValueError): ei.clip_segment_length((0, 0, np.nan), (0, 0, 1), R, H)
    with pytest.raises(ValueError): ei.clip_segment_length((0, 0, 0), (0, 0, 1), 0, H)


def test_primary_path_containment():
    inside = ei.primary_path_lengths([(0, 0, -.5), (0, 0, 0)], [(0, 0, 0), (0, 0, .5)], R, H)
    assert inside["contained"] and inside["inside_m"] == pytest.approx(1) and inside["total_m"] == pytest.approx(1)
    escaping = ei.primary_path_lengths([(0, 0, 0)], [(0, 0, 3)], R, H)
    assert not escaping["contained"] and escaping["inside_m"] == pytest.approx(1) and escaping["total_m"] == pytest.approx(3)


def test_identifiability_and_saturation():
    e = np.repeat([500., 1000., 1500.], 4)
    perfect = ei.endpoint_identifiability(e, e)
    assert perfect["model_relative_mae"] == 0 and perfect["improvement_over_constant"] == 1
    flat = ei.endpoint_identifiability(np.full(12, e.mean()), e)
    assert flat["improvement_over_constant"] == pytest.approx(0) and flat["prediction_span_fraction"] == 0
    # The constant is the evaluated set's mean energy (not equal-weight): 3 x 500 + 1 x 1500 -> 750.
    skew = np.array([500., 500., 500., 1500.])
    r = ei.endpoint_identifiability(np.full(4, 750.), skew)
    assert r["constant_relative_mae"] == pytest.approx(r["model_relative_mae"]) and r["improvement_over_constant"] == pytest.approx(0)
    assert r["by_energy"][500.0]["bias_fraction"] == pytest.approx(.5) and r["by_energy"][1500.0]["bias_fraction"] == pytest.approx(-.5)
    flat_light = np.ones(12) + 1e-3*(np.arange(12) % 2)
    sat = ei.light_yield_saturation(e, flat_light)
    assert sat["levels"] == [500., 1000., 1500.] and sat["marginal_span_fraction"] < 1e-2
    assert sat["ratios_vs_lowest"][1500.0] == pytest.approx(1.0)
    rising = ei.light_yield_saturation(e, e/500 + 1e-3*(np.arange(12) % 2))
    assert rising["ratios_vs_lowest"][1500.0] == pytest.approx(3.0, rel=1e-3) and rising["r2_energy_label"] > .99 and rising["separation"] > 100


def test_preflight_verdict_uses_declared_criteria_only():
    stats = {"a": {"b": 0.6}, "c": 2.0}
    ok = ei.preflight_verdict(stats, {"a.b": {"op": "ge", "value": .5}, "c": {"op": "lt", "value": 3}})
    assert ok["passed"] and not ok["reasons"]
    bad = ei.preflight_verdict(stats, {"a.b": {"op": "ge", "value": .7}})
    assert not bad["passed"] and "a.b" in bad["reasons"][0]
    with pytest.raises(KeyError): ei.preflight_verdict(stats, {"x": {"op": "ge", "value": 0}})
    with pytest.raises(ValueError): ei.preflight_verdict(stats, {"c": {"op": "eq", "value": 0}})
