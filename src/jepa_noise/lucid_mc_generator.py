# SPDX-License-Identifier: MIT
"""LUCiD Monte Carlo generator adapter (r2-lucid, runbook ``RUNBOOK_2026-11-21_R2_LUCID_REGEN.md``).

LUCiD does the physics (P1). This module only

* defines the event source draws (vertex, yield, particle kinematics, emission offset) with the
  round-2 keyed namespaces, so a LUCiD event and its toy counterpart share their truth draws;
* hands photons to LUCiD's production kernel (``setup_event_simulator(source='photons',
  hits='segment_truth')``, sampled transport and detection, K = 12, hard sensor assignment: the
  ``lucid-run-job`` settings) and runs LUCiD's own digitizer (SPE charge, charge-dependent PMT
  jitter, TDC, discriminator, dark noise) and, where the detector config carries one, LUCiD's
  sliding-window trigger;
* applies the VERITAS-side steps listed in ``VERITAS_STEPS`` (detector state, token-time
  convention, prompt window, one hit per PMT) and converts to the round-2 shard/pack schema.

Photon bombs use LUCiD's isotropic ray generator (``get_isotropic_rays_random``). Muons and
electrons use LUCiD's SIREN Cherenkov emitter (``make_cherenkov_surrogate_fn``) and its emission
time fit (``predict_t0``); its weighted rays are turned into unit photons by thinning with
probability ``intensity / n_chunks`` and rays emitted outside the tank are dropped (the track has
left the water). Nothing here computes expected charge, solid angle or an analytic arrival time:
``lucid_bombs`` / ``r2_generator`` physics is never imported into this path (gate G6; the
allowed ``lucid_bombs`` helpers are ``ALLOWED_LB_NAMES``).

The LUCiD import is lazy (:class:`LucidKernel`), so the conversion functions run and are tested
without JAX.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import lucid_bombs as lb
from . import r2_geometry as rg
from noise_module_lucid.readout import nhits_trigger_time, prompt_window_mask

__all__ = ["LUCID_SHA", "LUCID_ROOT", "DATASET_KIND", "SAMPLING_TIER", "SPECS", "COHORTS",
           "VERITAS_STEPS", "ALLOWED_LB_NAMES", "FORBIDDEN_LB_NAMES", "load_config",
           "lucid_geometry", "cohort_runs", "draw_truth", "LucidKernel", "electronics",
           "veritas_readout", "generate_run", "write_run", "LucidRun"]

LUCID_SHA = "e651e836b420f541b02672d9866e6219f2c6d66d"
LUCID_ROOT = Path(os.environ.get("LUCID_ROOT", f"/ceph/dwong/jepa/external/LUCiD_main_{LUCID_SHA[:8]}"))
DATASET_KIND = "lucid_mc_native"
SAMPLING_TIER = "A1"
CONFIG_RELPATH = "src/jepa_noise/configs/lucid_v1.json"
DATASET_SEED = lb.DEFAULT_SEED  # 20261017, the round-2 keyed namespace seed
NS_LUCID = "lucid_mc"          # namespace of the LUCiD kernel / digitizer keys
F1_TRAIN, F1_DRIFT = 0.05, 0.30

#: The production-kernel settings of ``lucid.production.dataprod.run_job._simulator``.
KERNEL = {"source": "photons", "K": 12, "temperature": 0.0, "charge_resolution": None,
          "default_detector_params": True, "hits": "segment_truth",
          "transport": "sampled (default for source='photons')",
          "detection": "sampled (default for source='photons')"}

#: Per detector: LUCiD config pair, wavelength mode, digitizer and LUCiD trigger block.
#: ``digitizer=None`` / ``trigger='config'`` read the block from the physics config (WAND).
SPECS = {
    "wcte": {"geom": "WCTE", "phys": "WCTE", "wavelength_mode": False, "digitizer": "hk",
             "trigger": None},
    "sk": {"geom": "SK_WAND", "phys": "SK_WAND", "wavelength_mode": True, "digitizer": None,
           "trigger": "config"},
    "hk": {"geom": "HK_WAND", "phys": "HK_WAND", "wavelength_mode": True, "digitizer": None,
           "trigger": "config"},
    "juno": {"geom": "JUNO", "phys": "JUNO", "wavelength_mode": True, "digitizer": "hk",
             "trigger": None},
}

#: Round-2 cohorts (same names, ID ranges and F1 settings). ``_s030`` cohorts reuse the probe
#: IDs (as in round 2), so their LUCiD photon-level output is shared with the s = 0.05 cohort.
COHORTS = {
    "dev_train": ("dev_train", F1_TRAIN, False),
    "dev_probe_train": ("dev_probe_train", F1_TRAIN, True),
    "dev_probe_test": ("dev_probe_test", F1_TRAIN, True),
    "dev_probe_train_s030": ("dev_probe_train", F1_DRIFT, True),
    "dev_probe_test_s030": ("dev_probe_test", F1_DRIFT, True),
}

#: Every VERITAS-side step (DATA_LINEAGE.md mirrors this list).
VERITAS_STEPS = (
    "source draws: round-2 keyed RNG namespaces (vertex, yield, domain plan, particle "
    "direction/energy, emission offset) - lucid_bombs.keyed_rng / r2_geometry.fiducial_vertex",
    "photon bombs: LUCiD get_isotropic_rays_random directions, emission at t = 0; "
    "chunked into the kernel buffer with LUCiD's N mask",
    "particles: LUCiD SIREN rays thinned to unit photons with p = intensity / n_chunks; "
    "rays with origins outside the tank dropped",
    "sensor index map LUCiD -> round-2 PMT index (LUCiD reorders barrel/top/bottom)",
    "per-event emission offset (uniform [-50, 50] ns, truth only) added before digitizing, "
    "as LUCiD's local-frame production digitize adds t0",
    "per-PMT timing-width factor of the round-2 fixed response multiplies LUCiD's sampled "
    "PMT jitter (before LUCiD's TDC rounding)",
    "one hit per PMT: the earliest surviving LUCiD digit",
    "charge x round-2 fixed parity gain x per-run F1 gain (detector state; twin: nominal gains)",
    "VERITAS NHits trigger time and prompt window (noise_module_lucid.readout) for the token "
    "time digit - trigger; window-start fallback when no NHits trigger",
)

#: lucid_bombs names the generation path may use (geometry, RNG, detector state, storage).
ALLOWED_LB_NAMES = frozenset({
    "DEFAULT_SEED", "keyed_rng", "NS_DRAWS", "NS_EMISSION_OFFSET", "EMISSION_OFFSET_NS",
    "COHORTS", "make_runs", "RunSpec", "nominal_state", "generator_reference",
    "fixed_charge_gain", "fixed_timing_width_factor", "load_generator_config", "_repo_root",
    "domain_plan", "_unit_direction", "sparse_from_readouts", "write_sparse_hits",
    "write_sha256sums", "HitReadout", "DOMAIN_PHOTON", "DOMAIN_MUON", "DOMAIN_ELECTRON",
    "PARTICLE_DOMAINS", "current_code_sha", "_sha", "fiducial_vertex",
    "_distance_to_boundary", "_event_ids_sha", "ReadoutRecord", "state_for_run",
})
#: lucid_bombs / r2_generator physics that must never run in the generation path (G6).
FORBIDDEN_LB_NAMES = ("expected_charge", "expected_charge_particle", "solid_angle",
                      "propagation_time_ns", "sample_qe", "poisson_counts_from_uniform",
                      "dark_photoelectrons", "leading_edge_time", "count_capacity",
                      "readout_synthetic_unrepaired", "transmission")


# ======================================================================= config
def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path is not None else lb._repo_root() / CONFIG_RELPATH
    cfg = json.loads(Path(path).read_text())
    cfg["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    cfg["_path"] = str(path)
    return cfg


def toy_generator_config():
    """The round-2 calibrated toy config, read only for the domain plan and energy ranges."""
    return lb.load_generator_config(lb._repo_root() / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json")


# ======================================================================= geometry
def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lucid_geometry(detector: str, cfg: dict, *, repo_root: Path | None = None):
    """The VERITAS geometry for LUCiD data and the LUCiD -> round-2 index map.

    HK and JUNO reuse the frozen round-2 geometry (LUCiD positions agree within 1 mm); WCTE and
    SK use the frozen ``configs/lucid_v1/<det>_geometry_<sha8>.npz`` (LUCiD snaps PMTs onto the
    wall), which keeps the round-2 PMT order, module map and directions and replaces positions.
    Returns ``(geometry, lucid_to_r2)`` with ``lucid_to_r2[j]`` the round-2 index of LUCiD sensor j.
    """
    root = Path(repo_root) if repo_root is not None else lb._repo_root()
    entry = cfg["geometry"][detector]
    base = rg.load_r2_geometry(detector)
    frozen = root / entry["frozen_npz"]
    data_bytes = frozen.read_bytes()
    if _sha_bytes(data_bytes) != entry["sha256"]:
        raise ValueError(f"{frozen}: sha256 mismatch with lucid_v1.json")
    with np.load(frozen, allow_pickle=False) as data:
        lucid_to_r2 = np.asarray(data["lucid_to_r2"], dtype=np.int64)
        positions = np.asarray(data["positions_m"], dtype=float)
    if entry["reuse_round2_positions"]:
        geometry = base
    else:
        mi, m = base.module_index, base.module_ids.size
        centres = np.stack([positions[mi == i].mean(axis=0) for i in range(m)])
        geometry = dataclasses.replace(
            base, name=f"{detector.upper()} LUCiD {LUCID_SHA[:8]} snapped", positions_m=positions,
            module_centres_m=centres, source_path=str(frozen), source_sha256=entry["sha256"])
    if sorted(lucid_to_r2.tolist()) != list(range(geometry.n_pmt)):
        raise ValueError("lucid_to_r2 must be a permutation of the PMT indices")
    return geometry, lucid_to_r2


def time_rule(detector: str, cfg: dict) -> dict:
    return dict(cfg["time_rule"][detector])


# ======================================================================= runs and draws
def cohort_runs(cohort: str, seed: int = DATASET_SEED):
    base, setting, _ = COHORTS[cohort]
    return lb.make_runs(lb.COHORTS[base], "F1", setting=setting, seed=int(seed))


def cohort_has_twin(cohort: str) -> bool:
    return bool(COHORTS[cohort][2])


@dataclass(frozen=True)
class Truth:
    run_id: str
    event_id: int
    domain: str
    vertex_m: np.ndarray
    yield_photons: float        # photons emitted (bomb) / SIREN n_photons(E) (particle)
    emission_offset_ns: float
    direction: np.ndarray | None = None
    energy_MeV: float = float("nan")
    deposited_MeV: float = float("nan")
    contained: bool = False
    track_length_m: float = float("nan")


def draw_truth(geometry, cfg: dict, detector: str, run_id: str, event_id: int, domain: str,
               seed: int = DATASET_SEED, emitters=None) -> Truth:
    """Source truth with the round-2 keyed roles (same vertex/direction/energy/offset draws)."""
    def rng(role):
        return lb.keyed_rng(seed, lb.NS_DRAWS, run_id, event_id, role)
    vertex = rg.fiducial_vertex(geometry, rng("vertex"), rg.FIDUCIAL_FRACTION)
    offset = float(rng(lb.NS_EMISSION_OFFSET).uniform(*lb.EMISSION_OFFSET_NS))
    if domain == lb.DOMAIN_PHOTON:
        lo, hi = (float(v) for v in cfg["yield_range_photons"][detector])
        y = float(np.exp(rng("yield").uniform(np.log(lo), np.log(hi))))
        return Truth(run_id, int(event_id), domain, vertex, y, offset)
    direction = lb._unit_direction(rng("direction"))
    lo, hi = (float(v) for v in cfg["energy_MeV"][domain])
    energy = float(np.exp(rng("energy").uniform(np.log(lo), np.log(hi))))
    boundary = rg.distance_to_boundary(geometry, vertex, direction)
    if emitters is None:
        n_ph, full = float("nan"), float("nan")
    else:
        n_ph, full = emitters.yield_and_range(domain, energy)
    track = min(full, boundary) if np.isfinite(full) else float("nan")
    contained = bool(np.isfinite(full) and full <= boundary)
    deposited = energy if contained else (energy * track / full if np.isfinite(full) else float("nan"))
    return Truth(run_id, int(event_id), domain, vertex, n_ph, offset, direction, energy,
                 float(deposited), contained, float(track))


def event_seed(seed: int, run_id: str, event_id: int, role: str) -> int:
    return int(lb.keyed_rng(seed, NS_LUCID, run_id, event_id, role).integers(0, 2**31 - 1))


# ======================================================================= the LUCiD kernel
class LucidKernel:
    """LUCiD's production kernel for one detector, plus its emitters (lazy JAX import)."""

    def __init__(self, detector: str, buffer: int, *, lucid_root: Path | str = LUCID_ROOT):
        import jax
        import jax.numpy as jnp
        root = Path(lucid_root)
        cwd = os.getcwd()
        os.chdir(root)  # LUCiD reads config/ and data/ relative to the checkout
        try:
            from lucid.simulation import setup_event_simulator
            from lucid.simulation.digitizer import resolve_model_config
            from lucid.simulation.trigger import TriggerConfig
            from lucid.sources.calibration_sources import get_isotropic_rays_random
            from lucid.siren.emitter import load_emitter
            from lucid.sources.siren_rays import make_cherenkov_surrogate_fn, predict_t0
            spec = SPECS[detector]
            self.geom_cfg = root / "config" / f"{spec['geom']}_geom_config.json"
            self.phys_cfg = root / "config" / f"{spec['phys']}_physics_config.json"
            self.sim = setup_event_simulator(
                str(self.geom_cfg), int(buffer), source="photons", K=KERNEL["K"],
                temperature=KERNEL["temperature"], charge_resolution=None,
                physics_config=str(self.phys_cfg), default_detector_params=True,
                hits="segment_truth", wavelength_mode=bool(spec["wavelength_mode"]))
            self.emitters = {d: load_emitter(d, "water", "cherenkov") for d in ("muon", "electron")}
        finally:
            os.chdir(cwd)
        phys = json.loads(self.phys_cfg.read_text())
        self.digitizer = resolve_model_config(phys.get("digitizer") if spec["digitizer"] is None
                                              else spec["digitizer"])
        block = phys.get("trigger") if spec["trigger"] == "config" else None
        self.trigger = TriggerConfig.from_block(block)
        self.detector, self.buffer, self.jax, self.jnp = detector, int(buffer), jax, jnp
        det = self.sim.det_geom.detector
        self.n_sensors = int(self.sim.det_geom.num_sensors)
        self.lucid_index_order = (np.asarray(det._reorder_indices, dtype=np.int64)
                                  if hasattr(det, "_reorder_indices") else
                                  np.arange(self.n_sensors, dtype=np.int64))
        self.sensor_points = np.asarray(self.sim.det_geom.sensor_points, dtype=float)
        self.shape = "sphere" if hasattr(det, "r") and not hasattr(det, "H") else "cylinder"
        self.radius = float(getattr(det, "r", np.nan))
        self.height = float(getattr(det, "H", np.nan))
        B = self.buffer
        sim = self.sim

        @jax.jit
        def _run(key, origins, directions, times, n):
            seg = jnp.where(jnp.arange(B) < n, 0, -1).astype(jnp.int32)
            out = sim(key, {"photon_origins": origins, "photon_directions": directions,
                            "photon_times": times, "N": n, "photon_segment_index": seg})
            w = out.deposit_weight
            idx = jnp.nonzero(w > 0, size=B, fill_value=-1)[0]
            ok = idx >= 0
            safe = jnp.where(ok, idx, 0)
            return (ok, out.deposit_sensor[safe], out.deposit_time_true[safe],
                    out.deposit_time_reco[safe], w[safe], jnp.sum(w > 0))
        self._run = _run

        @jax.jit
        def _iso(key, position):
            d, o, _ = get_isotropic_rays_random(position, 1.0, B, key)
            return o, d
        self._iso = _iso
        def _jit_siren(fn):
            return jax.jit(lambda o, d, e, params, key: fn(o, d, e, B, params, key))
        self._siren = {d: _jit_siren(make_cherenkov_surrogate_fn(e)) for d, e in self.emitters.items()}
        self._predict_t0 = jax.vmap(predict_t0, in_axes=(0, None, None, None))
        self.version = {"lucid_sha": LUCID_SHA, "kernel": KERNEL,
                        "geom_config_sha256": _sha_bytes(self.geom_cfg.read_bytes()),
                        "physics_config_sha256": _sha_bytes(self.phys_cfg.read_bytes()),
                        "wavelength_mode": bool(spec["wavelength_mode"]),
                        "digitizer": {k: v for k, v in self.digitizer.items() if k != "spe"}
                        | ({"spe": self.digitizer["spe"]} if "spe" in self.digitizer else {}),
                        "trigger": (None if self.trigger is None
                                    else dataclasses.asdict(self.trigger)),
                        "buffer": B, "jax": jax.__version__,
                        "device": str(jax.devices()[0])}

    # ---------------------------------------------------------------- emitters
    def yield_and_range(self, domain: str, energy: float) -> tuple:
        e = self.emitters[domain]
        n_ph = float(e.n_photons(self.jnp.float32(energy)))
        full = float(e.along_track_m(self.jnp.float32(1.0), self.jnp.float32(energy))) / 1.1
        return n_ph, full

    def _inside(self, points):
        p = np.asarray(points)
        if self.shape == "sphere":
            return np.linalg.norm(p, axis=1) < self.radius
        return (np.hypot(p[:, 0], p[:, 1]) < self.radius) & (np.abs(p[:, 2]) < 0.5 * self.height)

    # ---------------------------------------------------------------- photons
    def _propagate(self, key, origins, directions, times):
        """Run host photon arrays through the kernel in buffer-sized chunks."""
        jnp, B = self.jnp, self.buffer
        n_total = origins.shape[0]
        parts, n_calls = [], 0
        for c, start in enumerate(range(0, max(n_total, 1), B)):
            n = min(B, n_total - start)
            if n <= 0:
                break
            o = np.zeros((B, 3), np.float32); d = np.tile(np.float32([0, 0, 1]), (B, 1))
            t = np.zeros(B, np.float32)
            o[:n], d[:n], t[:n] = origins[start:start + n], directions[start:start + n], times[start:start + n]
            parts.append(self._run(self.jax.random.fold_in(key, c), jnp.asarray(o), jnp.asarray(d),
                                   jnp.asarray(t), jnp.int32(n)))
            n_calls += 1
        return self._collect(parts), n_calls

    def _collect(self, parts):
        cols = [[], [], [], []]
        for ok, s, tt, tr, w, n_pos in parts:
            ok = np.asarray(ok)
            if int(n_pos) > ok.size:
                raise RuntimeError(f"{int(n_pos)} deposits exceed the compaction buffer {ok.size}")
            for col, arr in zip(cols, (s, tt, tr, w)):
                col.append(np.asarray(arr)[ok])
        cat = [np.concatenate(c) if c else np.empty(0) for c in cols]
        return {"sensor": cat[0].astype(np.int64), "t_true": cat[1].astype(np.float64),
                "t_reco": cat[2].astype(np.float64), "weight": cat[3].astype(np.float64)}

    def simulate(self, truth: Truth, seed: int) -> dict:
        """LUCiD deposits (LUCiD sensor index, times in ns before the emission offset)."""
        jax, jnp, B = self.jax, self.jnp, self.buffer
        key = jax.random.PRNGKey(seed)
        if truth.domain == lb.DOMAIN_PHOTON:
            n = int(round(truth.yield_photons))
            parts, n_calls = [], 0
            pos = jnp.asarray(truth.vertex_m, jnp.float32)
            for c, start in enumerate(range(0, n, B)):
                k_dir, k_sim = jax.random.split(jax.random.fold_in(key, c))
                o, d = self._iso(k_dir, pos)
                parts.append(self._run(k_sim, o, d, jnp.zeros(B), jnp.int32(min(B, n - start))))
                n_calls += 1
            dep = self._collect(parts)
            dep.update(n_photons=n, n_calls=n_calls, n_thinned_overflow=0, n_outside=0)
            return dep
        # particle: SIREN rays -> unit photons
        emitter = self.emitters[truth.domain]
        n_mean = float(truth.yield_photons)
        n_chunks = max(1, int(math.ceil(n_mean / (0.8 * B))))
        k_rays, k_thin, k_sim = jax.random.split(key, 3)
        rng = np.random.default_rng(seed)
        pos = jnp.asarray(truth.vertex_m, jnp.float32)
        direc = jnp.asarray(truth.direction, jnp.float32)
        energy = jnp.float32(truth.energy_MeV)
        keep_o, keep_d, keep_t = [], [], []
        overflow = outside = 0
        chunks, c = [], 0
        while c < n_chunks:
            d, o, inten = self._siren[truth.domain](pos, direc, energy, emitter.params,
                                                   jax.random.fold_in(k_rays, c))
            t = self._predict_t0(jnp.linalg.norm(o - pos[None, :], axis=1), energy,
                                 emitter.t0, emitter.c_mm_per_ns)
            chunks.append(tuple(np.asarray(x) for x in (o, d, t, inten)))
            if c == 0:
                # enough chunks that the largest ray weight thins with p <= 1 (cap 64 calls)
                need = int(math.ceil(1.05 * float(chunks[0][3].max())))
                n_chunks = int(min(max(n_chunks, need), max(n_chunks, 64)))
            c += 1
        for o, d, t, inten in chunks:
            p = inten / n_chunks
            overflow += int(np.count_nonzero(p > 1.0))
            accept = rng.random(B) < np.clip(p, 0.0, 1.0)
            inside = self._inside(o)
            outside += int(np.count_nonzero(accept & ~inside))
            accept &= inside
            keep_o.append(o[accept]); keep_d.append(d[accept]); keep_t.append(t[accept])
        o, d, t = (np.concatenate(x) for x in (keep_o, keep_d, keep_t))
        dep, n_calls = self._propagate(k_sim, o.astype(np.float32), d.astype(np.float32),
                                       t.astype(np.float32))
        dep.update(n_photons=int(o.shape[0]), n_calls=n_calls, n_thinned_overflow=overflow,
                   n_outside=outside, n_siren_chunks=n_chunks)
        return dep


# ======================================================================= electronics
def electronics(sensor_r2, t_reco, charge, n_pmt: int, digitizer: dict, trigger, rng,
                width_factor: np.ndarray, dark_pad_ns: float) -> dict:
    """LUCiD digitizer and trigger on round-2-indexed deposits (times already offset).

    LUCiD functions do every step: ``generate_dark_noise`` over the readout span,
    ``digitize_event`` (integration windows), ``apply_readout_resolution`` (SPE charge and the
    charge-dependent jitter; its TDC rounding is applied after the per-PMT width factor),
    ``apply_discriminator`` and ``find_trigger_gates`` / ``hits_in_gates``.
    """
    from lucid.simulation.digitizer import (apply_discriminator, apply_readout_resolution,
                                            digitize_event, generate_dark_noise,
                                            _sample_time_jitter)
    from lucid.simulation.trigger import find_trigger_gates, hits_in_gates
    from lucid.simulation.digitizer import _MAX_DIGIT_TIME_NS
    s = np.asarray(sensor_r2, np.int64)
    t = np.asarray(t_reco, np.float64)
    q = np.asarray(charge, np.float64)
    # LUCiD digitize_and_decompose: drop non-finite deposits, cap late light at 100 us after
    # the earliest deposit (t_ref=None, the single-vertex production path).
    valid = np.isfinite(t) & (q > 0)
    if valid.any():
        valid &= (t - t[valid].min()) < _MAX_DIGIT_TIME_NS
    s, t, q = s[valid], t[valid], q[valid]
    is_dark = np.zeros(s.size, bool)
    rate = float(digitizer.get("dark_rate_khz", 0.0))
    if rate > 0.0 and t.size:
        ds, dt, dq = generate_dark_noise(n_pmt, rate, float(t.min()) - dark_pad_ns,
                                         float(t.max()) + dark_pad_ns, rng)
        s, t, q = np.concatenate([s, ds]), np.concatenate([t, dt]), np.concatenate([q, dq])
        is_dark = np.concatenate([is_dark, np.ones(ds.size, bool)])
    res = digitize_event(s, t, q, n_pmt, digitizer)
    no_tdc = dict(digitizer, tdc_ns=0.0)
    pe, t_jit = apply_readout_resolution(res.digit_pe_true, res.digit_time, no_tdc, rng)
    sens = res.digit_sensor_idx.astype(np.int64)
    t_scaled = res.digit_time + np.asarray(width_factor, float)[sens] * (t_jit - res.digit_time)
    t_dig = _sample_time_jitter(t_scaled, pe, {"time_model": "none",
                                               "tdc_ns": float(digitizer.get("tdc_ns", 0.0))}, rng)
    keep = apply_discriminator(pe, digitizer)
    # digit -> has a physics (non-dark) photoelectron
    phys = np.zeros(res.n_digits, bool)
    pdi = res.photon_digit_idx
    m = (pdi >= 0) & ~is_dark
    phys[pdi[m]] = True
    sens, pe, t_dig, phys = sens[keep], pe[keep], t_dig[keep], phys[keep]
    lucid_triggered, n_gates = True, 0
    if trigger is not None:
        gates = find_trigger_gates(t_dig, trigger)
        n_gates = int(gates.shape[0])
        if n_gates:
            g = hits_in_gates(t_dig, gates)
            sens, pe, t_dig, phys = sens[g], pe[g], t_dig[g], phys[g]
        else:
            lucid_triggered = False  # LUCiD would drop the event; kept, flagged (decision L3)
    return {"sensor": sens, "pe": pe.astype(np.float64), "t": t_dig.astype(np.float64),
            "physics": phys, "lucid_triggered": lucid_triggered, "n_gates": n_gates,
            "n_dark_pe": int(is_dark.sum())}


def veritas_readout(dig: dict, n_pmt: int, gains: np.ndarray, rule: dict, readout_start_ns: float,
                    record: dict | None = None) -> lb.HitReadout:
    """One hit per PMT (earliest digit), detector-state charge, NHits token time, prompt window."""
    s, pe, t = dig["sensor"], dig["pe"], dig["t"]
    order = np.lexsort((t, s))
    s, pe, t, phys = s[order], pe[order], t[order], dig["physics"][order]
    first = np.ones(s.size, bool)
    first[1:] = s[1:] != s[:-1]
    s, pe, t, phys = s[first], pe[first], t[first], phys[first]
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
    return lb.HitReadout(s[keep].astype(np.int32), q[keep].astype(np.float32),
                         (t[keep] - origin).astype(np.float32), absolute_time_ns=t[keep],
                         trigger_ns=None if trigger is None else float(trigger),
                         source_trigger_ns=None if source_trigger is None else float(source_trigger))


# ======================================================================= runs
@dataclass(eq=False)
class LucidRun:
    run: object
    event_ids: np.ndarray
    truths: list
    hits: list
    twin_hits: list | None
    stats: list
    record: dict
    state_sha256: str


def run_from_cache(cache: dict, run, geometry, cfg: dict, rule: dict, *, with_twin: bool,
                   seed: int = DATASET_SEED) -> LucidRun:
    """Apply a run's detector state to cached LUCiD digits (lets s = 0.30 reuse s = 0.05)."""
    state = run.state(geometry)
    ref = lb.generator_reference(toy_generator_config())
    fixed_gain = lb.fixed_charge_gain(geometry, ref)
    gains = fixed_gain * np.asarray(state.gains, float)
    record = {"n_events": 0, "n_no_trigger": 0, "n_prompt_dropped": 0, "n_dark_moved_trigger": 0}
    hits, twins = [], ([] if with_twin else None)
    for dig in cache["digits"]:
        hits.append(veritas_readout(dig, geometry.n_pmt, gains, rule, cfg["readout_start_ns"], record))
        if with_twin:
            twins.append(veritas_readout(dig, geometry.n_pmt, fixed_gain, rule, cfg["readout_start_ns"]))
    return LucidRun(run, cache["event_ids"], cache["truths"], hits, twins, cache["stats"], record,
                    state.digest)


def simulate_run_digits(kernel: LucidKernel, geometry, lucid_to_r2, cfg: dict, detector: str,
                        run, seed: int = DATASET_SEED, limit: int | None = None,
                        select=None) -> dict:
    """LUCiD deposits -> LUCiD digits for every event of a run (state-independent)."""
    ref = lb.generator_reference(toy_generator_config())
    width = lb.fixed_timing_width_factor(geometry, ref)
    domains = lb.domain_plan(run.n_events, toy_generator_config())
    event_ids = np.arange(run.event_start, run.event_stop, dtype=np.int64)
    if select is not None:
        sel = np.asarray(select, dtype=np.int64)
        event_ids, domains = event_ids[sel], tuple(np.asarray(domains)[sel])
    if limit is not None:
        event_ids, domains = event_ids[:limit], domains[:limit]
    pad = (max(kernel.trigger.pad_before_ns, kernel.trigger.pad_after_ns) + kernel.trigger.window_ns
           if kernel.trigger is not None else float(cfg["dark_pad_no_trigger_ns"]))
    truths, digits, stats = [], [], []
    for eid, dom in zip(event_ids, domains):
        truth = draw_truth(geometry, cfg, detector, run.run_id, int(eid), str(dom), seed=seed,
                           emitters=kernel)
        dep = kernel.simulate(truth, event_seed(seed, run.run_id, int(eid), "kernel"))
        fin = np.isfinite(dep["t_reco"])
        for k in ("sensor", "t_true", "t_reco", "weight"):
            dep[k] = dep[k][fin]
        s_r2 = lucid_to_r2[dep["sensor"]]
        t = dep["t_reco"] + truth.emission_offset_ns
        rng = np.random.default_rng(event_seed(seed, run.run_id, int(eid), "digitizer"))
        dig = electronics(s_r2, t, np.ones(s_r2.size), geometry.n_pmt, kernel.digitizer,
                          kernel.trigger, rng, width, pad)
        true_pe = float(dep["weight"].size)
        late = float(np.mean((dep["t_reco"] - _direct_time(geometry, truth, s_r2)) > 10.0)) \
            if (truth.domain == lb.DOMAIN_PHOTON and s_r2.size) else float("nan")
        truths.append(dataclasses.replace(truth))
        digits.append(dig)
        stats.append({"event_id": int(eid), "domain": truth.domain, "true_pe": true_pe,
                      "n_photons": dep["n_photons"], "n_calls": dep["n_calls"], "n_siren_chunks": dep.get("n_siren_chunks", 0),
                      "n_outside": dep["n_outside"], "n_overflow": dep["n_thinned_overflow"],
                      "n_digits": int(dig["sensor"].size), "digit_pe": float(dig["pe"].sum()),
                      "lucid_triggered": dig["lucid_triggered"], "n_gates": dig["n_gates"],
                      "late_frac_10ns": late})
    return {"event_ids": event_ids, "truths": truths, "digits": digits, "stats": stats}


def _direct_time(geometry, truth: Truth, sensors) -> np.ndarray:
    """Straight-line time for the G2 late-light diagnostic only (never enters the data)."""
    d = np.linalg.norm(geometry.positions_m[sensors] - truth.vertex_m[None, :], axis=1)
    return d * 1.33 / 0.299792458


def truth_payload(lrun: LucidRun) -> dict:
    t = lrun.truths
    nan3 = [np.nan] * 3
    return dict(
        event_ids=lrun.event_ids.astype(np.int64),
        vertex_m=np.asarray([x.vertex_m for x in t], np.float64),
        yield_photons=np.asarray([x.yield_photons for x in t], np.float64),
        expected_pe=np.asarray([s["true_pe"] for s in lrun.stats], np.float32),
        emission_offset_ns=np.asarray([x.emission_offset_ns for x in t], np.float64),
        domain=np.asarray([x.domain for x in t], dtype="<U8"),
        particle=np.asarray(["none" if x.domain == lb.DOMAIN_PHOTON else x.domain for x in t], dtype="<U8"),
        direction_m=np.asarray([nan3 if x.direction is None else x.direction for x in t], np.float64),
        energy_MeV=np.asarray([x.energy_MeV for x in t], np.float64),
        deposited_MeV=np.asarray([x.deposited_MeV for x in t], np.float64),
        contained=np.asarray([bool(x.contained) for x in t]),
        track_length_m=np.asarray([x.track_length_m for x in t], np.float64))


#: Truth-field status for LUCiD packs (every round-2 field filled or marked not applicable).
TRUTH_FIELD_STATUS = {
    "vertex_m": "filled: bomb position / track start (round-2 keyed draw)",
    "yield_photons": "filled: emitted photons (bomb) / LUCiD SIREN n_photons(E) (particle)",
    "expected_pe": "filled with a CHANGED meaning: LUCiD-sampled detected photoelectrons before "
                   "electronics (no expectation exists in MC data)",
    "emission_offset_ns": "filled: uniform [-50, 50] ns, truth only",
    "domain": "filled", "particle": "filled",
    "direction_m": "filled (particles; NaN for bombs as in round 2)",
    "energy_MeV": "filled (particles)",
    "deposited_MeV": "filled (particles): E if contained else E x in-tank fraction of the LUCiD "
                     "emission range",
    "contained": "filled (particles): LUCiD emission range (s_max / 1.1) <= distance to wall",
    "track_length_m": "filled (particles): min(LUCiD emission range, distance to wall)",
    "expected_pe_pmt": "not applicable: no per-PMT expectation in MC data (optional field)",
}


def write_run(directory, lrun: LucidRun, geometry, cfg: dict, rule: dict, provenance: dict) -> dict:
    """Round-1/2 per-run shard: hits.npz, twin_hits.npz, events.npz, manifest.json, SHA256SUMS.

    Written into ``<directory>.tmp`` and renamed atomically after validation (L6).
    """
    directory = Path(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        for p in tmp.iterdir():
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    lb.write_sparse_hits(tmp / "hits.npz", _sparse(lrun.event_ids, lrun.hits))
    if lrun.twin_hits is not None:
        lb.write_sparse_hits(tmp / "twin_hits.npz", _sparse(lrun.event_ids, lrun.twin_hits))
    payload = truth_payload(lrun)
    with open(tmp / "events.npz", "wb") as handle:
        np.savez_compressed(handle, **payload)
    run = lrun.run
    stats = lrun.stats
    manifest = {
        "schema": "jepa-lucid-mc-manifest-v1", "dataset_kind": DATASET_KIND,
        "sampling_tier": SAMPLING_TIER, "registered": False, "reported_results": False,
        "readout_label": f"lucid_{LUCID_SHA[:8]}_{provenance['kernel']['digitizer']['model']}",
        "dataset_seed": int(run.seed), "geometry_name": geometry.name,
        "geometry_sha256": geometry.digest, "n_runs": 1, "n_events": int(lrun.event_ids.size),
        "runs": [{"run_id": run.run_id, "event_namespace": run.event_namespace,
                  "event_id_start": run.event_start, "event_id_stop": run.event_stop,
                  "n_events": run.n_events, "law": run.law, "setting": run.setting,
                  "state_sha256": lrun.state_sha256, "event_ids_sha256": lb._event_ids_sha(run)}],
        "detector": provenance["detector"], "lucid": provenance["kernel"],
        "lucid_v1_config_sha256": cfg["_sha256"], "time_rule": rule,
        "veritas_steps": list(VERITAS_STEPS), "truth_field_status": TRUTH_FIELD_STATUS,
        "code_sha": provenance.get("code_sha"), "twin_present": lrun.twin_hits is not None,
        "expected_pe_pmt_present": False, "n_hits": int(sum(len(h) for h in lrun.hits)),
        **lrun.record,
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
    check = lb.read_sparse_hits(tmp / "hits.npz")
    if check.event_ids.size != run.n_events and check.event_ids.size != lrun.event_ids.size:
        raise RuntimeError("run shard event count mismatch")
    if directory.exists():
        raise FileExistsError(directory)
    tmp.rename(directory)
    return manifest


def _sparse(event_ids, hits):
    return lb.sparse_from_readouts(event_ids, hits)
