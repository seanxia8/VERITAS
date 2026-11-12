# SPDX-License-Identifier: MIT
"""Round-2 generator: the round-1 LUCiD generator applied to an :mod:`r2_geometry` detector.

Binding: ``docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md`` §2, §3, §4.1.

Everything physical is the round-1 code, imported from :mod:`jepa_noise.lucid_bombs`
(QE, solid angle, expected charge, particle ring model, Poisson-from-uniform, dark bank,
detector state, fixed response, walk primitive, sparse store).  Only three things are
geometry-dependent and are passed explicitly here instead of read from WCTE constants:

* the fiducial vertex and the distance to the boundary (cylinder = the round-1 functions;
  sphere = :mod:`r2_geometry`), drawn from the same keyed namespaces and roles;
* the NHits trigger ``n`` and the prompt window, passed explicitly to the imported
  ``noise_module_lucid.readout.nhits_trigger_time(n=..., window_ns=...)`` and
  ``prompt_window_mask(window_ns=...)``;
* the per-detector photon-bomb yield range (from the X0 detectability procedure).

On WCTE with the round-1 time rule (n 4, 20 ns, (-20, 60) ns) and yield range, the readout
below is the round-1 ``readout_synthetic_unrepaired`` statement for statement, so a stored
round-1 run regenerates bit for bit (test 2).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import lucid_bombs as lb
from . import r2_geometry as rg
from noise_module_lucid.readout import nhits_trigger_time, prompt_window_mask

__all__ = ["TimeRule", "time_rule_for", "R2_CONFIG_RELPATH", "load_r2_config", "draw_event",
           "readout", "generate_run", "write_generated_run", "COHORTS", "cohort_runs"]

R2_CONFIG_RELPATH = "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
DATASET_SEED = lb.DEFAULT_SEED  # 20261017 (§2.1)
F1_TRAIN, F1_DRIFT = 0.05, 0.30


@dataclass(frozen=True)
class TimeRule:
    """The per-detector A9 rule: NHits n in a 20 ns window, the prompt window, time scale."""

    n: int
    trigger_window_ns: float
    prompt_window_ns: tuple
    time_scale_ns: float

    def record(self) -> dict:
        return {"trigger": "NHits", "n": int(self.n),
                "trigger_window_ns": float(self.trigger_window_ns),
                "prompt_window_ns": [float(v) for v in self.prompt_window_ns],
                "no_trigger": "window_start",
                "emission_offset_ns": [float(v) for v in lb.EMISSION_OFFSET_NS],
                "token_time": "digit - trigger", "time_scale_ns": float(self.time_scale_ns),
                "source": "round-2 runbook §2.2 rules; imported nhits_trigger_time / "
                          "prompt_window_mask with explicit arguments"}


def time_rule_for(geometry) -> TimeRule:
    v = rg.derived_values(geometry)
    return TimeRule(int(v["nhits_n"]), float(v["nhits_window_ns"]),
                    tuple(float(x) for x in v["prompt_window_ns"]), float(v["time_scale_ns"]))


ROUND1_TIME_RULE = TimeRule(int(lb.TIME_RULE["n"]), float(lb.TIME_RULE["trigger_window_ns"]),
                            tuple(float(x) for x in lb.TIME_RULE["prompt_window_ns"]), 100.0)


def load_r2_config(y_lo: float | None = None, y_hi: float | None = None) -> lb.GeneratorConfig:
    """The round-1 calibrated generator config (§2.1), with a per-detector yield range."""
    cfg = lb.load_generator_config(lb._repo_root() / R2_CONFIG_RELPATH)
    changes = {}
    if y_lo is not None:
        changes["y_lo"] = float(y_lo)
    if y_hi is not None:
        changes["y_hi"] = float(y_hi)
    return dataclasses.replace(cfg, **changes) if changes else cfg


# ======================================================================= draws
def draw_particle_truth(geometry, cfg, run_id: str, event_id: int, domain: str,
                        seed: int = DATASET_SEED) -> lb.ParticleTruth:
    """The round-1 particle truth with the geometry's fiducial sampler and boundary."""
    if domain not in lb.PARTICLE_DOMAINS:
        raise ValueError(f"draw_particle_truth needs a particle domain, got {domain!r}")

    def rng(role):
        return lb.keyed_rng(seed, lb.NS_DRAWS, run_id, event_id, role)
    vertex = rg.fiducial_vertex(geometry, rng("vertex"), cfg.fiducial_fraction)
    direction = lb._unit_direction(rng("direction"))
    lo, hi = (float(x) for x in (cfg.muon_energy_MeV if domain == lb.DOMAIN_MUON
                                 else cfg.electron_energy_MeV))
    energy = float(np.exp(rng("energy").uniform(np.log(lo), np.log(hi))))
    boundary = rg.distance_to_boundary(geometry, vertex, direction)
    full_length = (float(cfg.muon_range_m_per_MeV) * energy if domain == lb.DOMAIN_MUON
                   else float(cfg.electron_shower_length_m))
    track_length = min(full_length, boundary)
    contained = bool(full_length <= boundary)
    deposited = float(energy if contained else energy * (track_length / max(full_length, 1e-12)))
    return lb.ParticleTruth(domain=domain, vertex_m=vertex, direction=direction,
                            energy_MeV=energy, deposited_MeV=deposited, contained=contained,
                            track_length_m=track_length)


def draw_event(geometry, cfg, run_id: str, event_id: int, seed: int = DATASET_SEED,
               domain: str | None = None) -> lb.SensorDraws:
    """The round-1 ``draw_event`` with the geometry's fiducial sampler (same keyed roles)."""
    def rng(role):
        return lb.keyed_rng(seed, lb.NS_DRAWS, run_id, event_id, role)
    if domain is not None and domain != lb.DOMAIN_PHOTON:
        truth = draw_particle_truth(geometry, cfg, run_id, event_id, domain, seed=seed)
        qe_eff = lb.sample_qe(geometry, rng("wavelength"), cfg)
        bound, _ = lb.expected_charge_particle(geometry, truth, qe_eff, lb.UNATTENUATED_M, cfg)
        capacity = lb.count_capacity(float(bound.max()))
        poisson_u = np.clip(rng("poisson").random((geometry.n_pmt, capacity)),
                            np.finfo(float).tiny, 1.0)
        dark_counts, dark_times = lb.dark_photoelectrons(geometry, cfg, rng("dark"),
                                                         rng("dark_time"))
        offset = float(rng(lb.NS_EMISSION_OFFSET).uniform(*lb.EMISSION_OFFSET_NS))
        return lb.SensorDraws(
            key=f"{lb.NS_DRAWS}/{run_id}/{event_id}", run_id=run_id, event_id=int(event_id),
            vertex_m=truth.vertex_m,
            yield_photons=float(cfg.cherenkov_yield_per_MeV) * float(truth.deposited_MeV),
            qe_eff=qe_eff, poisson_u=poisson_u,
            charge_z=rng("charge").standard_normal(geometry.n_pmt),
            transit_z=rng("transit").standard_normal(geometry.n_pmt),
            dark_counts=dark_counts, dark_times=dark_times, window_ns=float(cfg.window_ns),
            capacity=capacity, emission_offset_ns=offset, domain=domain, truth=truth)
    if cfg.dark_counts != lb.DARK_COUNTS_ON:
        raise ValueError("dark counts must be on (§2.1)")
    vertex = rg.fiducial_vertex(geometry, rng("vertex"), cfg.fiducial_fraction)
    yield_photons = float(np.exp(rng("yield").uniform(np.log(cfg.y_lo), np.log(cfg.y_hi))))
    qe_eff = lb.sample_qe(geometry, rng("wavelength"), cfg)
    bound = lb.expected_charge(geometry, vertex, yield_photons, qe_eff, lb.UNATTENUATED_M, cfg)
    capacity = lb.count_capacity(float(bound.max()))
    poisson_u = np.clip(rng("poisson").random((geometry.n_pmt, capacity)),
                        np.finfo(float).tiny, 1.0)
    dark_counts, dark_times = lb.dark_photoelectrons(geometry, cfg, rng("dark"), rng("dark_time"))
    offset = float(rng(lb.NS_EMISSION_OFFSET).uniform(*lb.EMISSION_OFFSET_NS))
    return lb.SensorDraws(key=f"{lb.NS_DRAWS}/{run_id}/{event_id}", run_id=run_id,
                          event_id=int(event_id), vertex_m=vertex, yield_photons=yield_photons,
                          qe_eff=qe_eff, poisson_u=poisson_u,
                          charge_z=rng("charge").standard_normal(geometry.n_pmt),
                          transit_z=rng("transit").standard_normal(geometry.n_pmt),
                          dark_counts=dark_counts, dark_times=dark_times,
                          window_ns=float(cfg.window_ns), capacity=capacity,
                          emission_offset_ns=offset)


# ======================================================================= readout
def expected_pe(state, draws, geometry, cfg, fixed_gain):
    """Per-PMT mean PE (fixed response included) and the source arrival time."""
    if draws.domain == lb.DOMAIN_PHOTON:
        mu = (lb.expected_charge(geometry, draws.vertex_m, draws.yield_photons, draws.qe_eff,
                                 state.attenuation_length_m, cfg) * fixed_gain)
        return mu, lb.propagation_time_ns(geometry, draws.vertex_m, cfg)
    if draws.domain in lb.PARTICLE_DOMAINS:
        mu_base, t = lb.expected_charge_particle(geometry, draws.truth, draws.qe_eff,
                                                 state.attenuation_length_m, cfg)
        return mu_base * fixed_gain, t
    raise ValueError(f"readout: unknown domain {draws.domain!r}")


def readout(state, draws, geometry, cfg, rule: TimeRule, reference=None, *,
            accounting: lb.ReadoutRecord | None = None) -> lb.HitReadout:
    """Round-1 ``readout_synthetic_unrepaired`` with the trigger and prompt window explicit."""
    if cfg.dark_counts != lb.DARK_COUNTS_ON:
        raise ValueError("dark counts must be on (§2.1)")
    n_dark = np.asarray(draws.dark_counts, dtype=np.int64)
    if n_dark.shape != (geometry.n_pmt,) or np.any(n_dark < 0):
        raise ValueError("dark_counts must be one non-negative count per PMT")
    dark_sensor = np.repeat(np.arange(geometry.n_pmt, dtype=np.int64), n_dark)
    dark_times = np.asarray(draws.dark_times, dtype=float)
    emission_offset_ns = float(draws.emission_offset_ns)
    start, end = (float(v) for v in cfg.readout_window_ns)
    ref = lb.generator_reference(cfg) if reference is None else reference
    fixed_gain = lb.fixed_charge_gain(geometry, ref)
    mu, source_time = expected_pe(state, draws, geometry, cfg, fixed_gain)
    source_time = np.asarray(source_time, dtype=float) + emission_offset_ns
    n_photon = lb.poisson_counts_from_uniform(draws.poisson_u, mu)
    if int(n_photon.max()) >= int(draws.capacity):
        raise RuntimeError("photon count reached the shared-draw capacity; enlarge count_capacity")
    n = n_photon + n_dark
    charge = np.clip(n + np.sqrt(n) * float(cfg.sigma_spe) * draws.charge_z, 0.0, None)
    q = charge * np.asarray(state.gains, dtype=float)
    thr_mv = float(cfg.threshold_pe) * float(cfg.mv_per_pe)
    cal_mv = float(cfg.walk_calibration_pe) * float(cfg.mv_per_pe)
    walk = lb.leading_edge_time(q, thr_mv, mv_per_pe=float(cfg.mv_per_pe))
    walk_cal = lb.leading_edge_time(q, cal_mv, mv_per_pe=float(cfg.mv_per_pe))
    fired = np.isfinite(walk)
    residual = np.where(np.isfinite(walk_cal),
                        walk - np.where(np.isfinite(walk_cal), walk_cal, 0.0), 0.0)
    sigma_t = float(cfg.tts_sigma_ns) * lb.fixed_timing_width_factor(geometry, ref)
    first = np.where(n_photon > 0, source_time, np.inf)
    dark_first = np.full(geometry.n_pmt, np.inf)
    if dark_sensor.size:
        np.minimum.at(dark_first, dark_sensor, dark_times)
    first = np.where(np.isfinite(dark_first), np.minimum(first, dark_first), first)
    times = first + sigma_t * draws.transit_z + residual
    index = np.flatnonzero(fired)
    digit_times = times[index]
    n_trig, w_trig = int(rule.n), float(rule.trigger_window_ns)
    trigger = nhits_trigger_time(digit_times, n=n_trig, window_ns=w_trig)
    source_trigger = trigger
    if dark_sensor.size:
        source_first = np.where(n_photon > 0, source_time, np.inf)
        source_times = source_first + sigma_t * draws.transit_z + residual
        source_digits = source_times[index[n_photon[index] > 0]]
        source_trigger = nhits_trigger_time(source_digits, n=n_trig, window_ns=w_trig)
        if source_trigger is not None:
            moved = trigger is None or abs(float(trigger) - float(source_trigger)) > 1e-9
            if moved and accounting is not None:
                accounting.n_dark_moved_trigger += 1
    if trigger is None:
        origin = start
        if accounting is not None:
            accounting.n_no_trigger += 1
    else:
        origin = float(trigger)
    token = digit_times - origin
    keep = np.asarray(prompt_window_mask(digit_times, None if trigger is None else float(trigger),
                                         window_ns=tuple(rule.prompt_window_ns)))
    if accounting is not None:
        accounting.n_events += 1
        accounting.n_prompt_dropped += int(np.count_nonzero(~keep))
    kept = index[keep]
    return lb.HitReadout(kept, q[kept], token[keep], absolute_time_ns=digit_times[keep],
                         trigger_ns=(None if trigger is None else float(trigger)),
                         source_trigger_ns=(None if source_trigger is None
                                            else float(source_trigger)))


# ======================================================================= cohorts and runs
#: §3.1 cohorts (event-ID ranges as round 1; the s = 0.30 sets reuse the probe IDs).
COHORTS = {
    "dev_train": ("dev_train", F1_TRAIN, False),
    "dev_probe_train": ("dev_probe_train", F1_TRAIN, True),
    "dev_probe_test": ("dev_probe_test", F1_TRAIN, True),
    "dev_probe_train_s030": ("dev_probe_train", F1_DRIFT, True),
    "dev_probe_test_s030": ("dev_probe_test", F1_DRIFT, True),
}


def cohort_runs(cohort: str, seed: int = DATASET_SEED):
    """The round-1 run specs of a §3.1 cohort (``lucid_bombs.COHORTS`` ID ranges)."""
    base, setting, _ = COHORTS[cohort]
    return lb.make_runs(lb.COHORTS[base], "F1", setting=setting, seed=int(seed))


def cohort_has_twin(cohort: str) -> bool:
    return bool(COHORTS[cohort][2])


@dataclass(frozen=True, eq=False)
class R2Run:
    run: lb.RunSpec
    state: lb.DetectorState
    event_ids: np.ndarray
    vertices_m: np.ndarray
    yields_photons: np.ndarray
    expected_pe: np.ndarray       # (n_events,) total expected PE
    expected_pe_pmt: np.ndarray | None  # (n_events, n_pmt) float32, probe cohorts only
    emission_offsets_ns: np.ndarray
    hits: tuple
    twin_hits: tuple | None
    readout_record: lb.ReadoutRecord
    domains: np.ndarray
    particle_truth: tuple

    @property
    def n_events(self) -> int:
        return int(self.event_ids.size)

    @property
    def n_hits(self) -> int:
        return int(sum(len(h) for h in self.hits))


def generate_run(geometry, cfg, run: lb.RunSpec, rule: TimeRule, *, with_twin: bool = False,
                 store_expected_pmt: bool = False) -> R2Run:
    """One run with a single detector state, as round-1 ``generate_run``."""
    seed = int(run.seed)
    state = run.state(geometry)
    reference = lb.generator_reference(cfg)
    fixed_gain = lb.fixed_charge_gain(geometry, reference)
    record = lb.ReadoutRecord()
    event_ids = np.arange(run.event_start, run.event_stop, dtype=np.int64)
    domains = lb.domain_plan(event_ids.size, cfg)
    n = event_ids.size
    vertices, yields = np.zeros((n, 3)), np.zeros(n)
    expected, offsets = np.zeros(n), np.zeros(n)
    expected_pmt = np.zeros((n, geometry.n_pmt), dtype=np.float32) if store_expected_pmt else None
    hits, twin_hits, truths = [], [], []
    nominal = lb.nominal_state(geometry, run.run_id) if with_twin else None
    for i, event_id in enumerate(event_ids):
        draws = draw_event(geometry, cfg, run.run_id, int(event_id), seed=seed, domain=domains[i])
        vertices[i], yields[i] = draws.vertex_m, draws.yield_photons
        offsets[i] = float(draws.emission_offset_ns)
        mu, _ = expected_pe(state, draws, geometry, cfg, fixed_gain)
        expected[i] = float(mu.sum())
        if expected_pmt is not None:
            expected_pmt[i] = mu.astype(np.float32)
        truths.append(draws.truth)
        hits.append(readout(state, draws, geometry, cfg, rule, reference, accounting=record))
        if with_twin:
            twin_hits.append(readout(nominal, draws, geometry, cfg, rule, reference))
    return R2Run(run=run, state=state, event_ids=event_ids, vertices_m=vertices,
                 yields_photons=yields, expected_pe=expected, expected_pe_pmt=expected_pmt,
                 emission_offsets_ns=offsets, hits=tuple(hits),
                 twin_hits=tuple(twin_hits) if with_twin else None, readout_record=record,
                 domains=np.asarray(domains, dtype="<U8"), particle_truth=tuple(truths))


def truth_payload(generated: R2Run) -> dict:
    """The round-1 ``events.npz`` fields (mixed domains)."""
    truths = generated.particle_truth
    payload = dict(event_ids=generated.event_ids,
                   vertex_m=generated.vertices_m.astype(np.float64),
                   yield_photons=generated.yields_photons.astype(np.float64),
                   expected_pe=generated.expected_pe.astype(np.float32),
                   emission_offset_ns=generated.emission_offsets_ns.astype(np.float64),
                   domain=np.asarray(generated.domains, dtype="<U8"),
                   particle=np.asarray(["none" if t is None else t.domain for t in truths],
                                       dtype="<U8"),
                   direction_m=np.asarray([[np.nan] * 3 if t is None else t.direction
                                           for t in truths], dtype=np.float64),
                   energy_MeV=np.asarray([np.nan if t is None else t.energy_MeV for t in truths],
                                         dtype=np.float64),
                   deposited_MeV=np.asarray([np.nan if t is None else t.deposited_MeV
                                             for t in truths], dtype=np.float64),
                   contained=np.asarray([False if t is None else t.contained for t in truths]),
                   track_length_m=np.asarray([np.nan if t is None else t.track_length_m
                                              for t in truths], dtype=np.float64))
    return payload


def write_generated_run(directory, generated: R2Run, geometry, cfg, rule: TimeRule, *,
                        code_sha: str | None = None, derived: dict | None = None) -> dict:
    """The round-1 per-run layout: hits.npz, twin_hits.npz, events.npz, manifest, SHA256SUMS."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lb.write_sparse_hits(directory / "hits.npz",
                         lb.sparse_from_readouts(generated.event_ids, generated.hits))
    if generated.twin_hits is not None:
        lb.write_sparse_hits(directory / "twin_hits.npz",
                             lb.sparse_from_readouts(generated.event_ids, generated.twin_hits))
    payload = truth_payload(generated)
    if generated.expected_pe_pmt is not None:
        payload["expected_pe_pmt"] = generated.expected_pe_pmt
    truth = directory / "events.npz"
    with open(truth, "wb") as handle:
        np.savez_compressed(handle, **payload)
    record = generated.readout_record
    manifest = lb.write_manifest(
        directory / "manifest.json", (generated.run,), geometry,
        seed=int(generated.run.seed), code_sha=code_sha,
        extra={"generator_config_sha256": cfg.digest,
               "round2": True, "detector": getattr(geometry, "detector", "wcte"),
               "geometry_source_path": geometry.source_path,
               "geometry_source_sha256": geometry.source_sha256,
               "geometry_config_sha256": geometry.config_sha256,
               "geometry_summary": geometry.summary(),
               "derived_values": derived or rg.derived_values(geometry),
               "yield_range_photons": [float(cfg.y_lo), float(cfg.y_hi)],
               "fiducial_fraction": float(cfg.fiducial_fraction),
               "state_clipped": dict(generated.state.clipped), "n_hits": generated.n_hits,
               "twin_present": generated.twin_hits is not None,
               "expected_pe_pmt_present": generated.expected_pe_pmt is not None,
               "time_rule": rule.record(),
               "store_time_ns": "digit - trigger (A9 token time, after the prompt-window cut)",
               **{k: v for k, v in record.as_dict().items() if k != "n_events"},
               "n_readout_events": int(record.n_events),
               "event_truth_sha256": lb._sha(truth.read_bytes()),
               "domains": {str(d): int(np.count_nonzero(generated.domains == d))
                           for d in cfg.domains},
               "domain_fractions": dict(cfg.domain_fractions),
               "generator_mode": cfg.generator_mode})
    lb.write_sha256sums(directory)
    return manifest
