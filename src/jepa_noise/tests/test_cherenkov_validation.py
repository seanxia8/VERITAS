"""A10 PhotonSim-vs-analytic validation harness (synthetic fixture)."""
import numpy as np

from jepa_noise import cherenkov_validation as cv, lucid_bombs as lb

GEOM = lb.from_synthetic({"module_ids": tuple(range(4)),
                          "surface_counts": {"top": 16, "barrel": 44, "bottom": 16}})
CFG = lb.load_generator_config("src/jepa_noise/configs/lucid_bombs_v1_mixed.json")


def _photon_event(seed=3):
    rng = np.random.default_rng(seed)
    vertex = np.array([0.1, -0.2, 0.05])
    yield_photons = 20_000.0
    qe = np.full(GEOM.n_pmt, 0.2)
    mu = lb.expected_charge(GEOM, vertex, yield_photons, qe, lb.LAMBDA0_M, CFG)
    counts = rng.poisson(mu)
    sensor = np.repeat(np.arange(GEOM.n_pmt), counts)
    time = np.repeat(lb.propagation_time_ns(GEOM, vertex, CFG), counts) + rng.normal(0, 0.3, sensor.size)
    wavelength = np.full(sensor.size, 450.0)
    return cv.ValidationEvent(event_id=1, domain=lb.DOMAIN_PHOTON, vertex_m=vertex,
                              sensor_id=sensor, time_ns=time, wavelength_nm=wavelength,
                              yield_photons=yield_photons)


def test_validation_event_roundtrip_and_compare(tmp_path):
    event = _photon_event()
    path = tmp_path / "event_001.npz"
    np.savez(path, event_id=event.event_id, domain=event.domain, vertex_m=event.vertex_m,
             sensor_id=event.sensor_id, time_ns=event.time_ns,
             wavelength_nm=event.wavelength_nm, yield_photons=event.yield_photons)
    loaded = cv.load_event_npz(path)
    row = cv.compare_event(GEOM, CFG, loaded)
    assert row["domain"] == "photon" and row["n_photons"] == event.sensor_id.size
    assert np.isfinite(row["yield_ratio"]) and np.isfinite(row["pattern_pearson"])
    result = cv.validate(GEOM, CFG, [loaded])
    assert result["events"] == 1 and "summary" in result


def test_transport_sensor_map_recovers_a_shuffled_indexing():
    rng = np.random.default_rng(0)
    perm = rng.permutation(GEOM.n_pmt)
    mapping = cv.sensor_id_map(GEOM, GEOM.positions_m[perm])
    assert np.array_equal(mapping, perm)
