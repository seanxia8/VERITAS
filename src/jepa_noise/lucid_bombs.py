# SPDX-License-Identifier: MIT
"""Development-only LUCiD photon-bomb generator contract (WP4b).

Binding: ``docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`` §3.5.2--§3.5.4,
§3.5.3a and v3 §6 WP5.  Everything is labelled
``readout_synthetic_unrepaired``; the physical WCTE geometry (105 mPMT × 19
PMTs, physical ``mpmt_id`` map) and pinned LUCiD revision are unavailable, so
nothing produced here is a reported result.

Implemented: detector-state laws F1/F3/F2/F2'/R11b/T with keyed RNG
namespaces; the ``readout(state, draws) -> hits`` contract and nominal twin;
run structure, v3 §6 ID ranges under the ``lucid/`` prefix, manifest and
sparse-hit storage; N_min exclusion accounting (§3.5.4); and §3.5.3a
Fisher/ceiling numerics.  The built-in 8×7 synthetic cylinder is *not* WCTE.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

READOUT_LABEL = "readout_synthetic_unrepaired"
DEVELOPMENT_ONLY = True
DEFAULT_SEED = 20261017
#: Placeholder: §3.5.3 fixes F2's form but no lambda0, and the PMT reference
#: fixture has absorption off.
LAMBDA0_M = 50.0
GAIN_CLIP = (0.5, 1.5)
F1_SPREADS = (0.02, 0.05, 0.10)
F3_SPREAD = 0.05
F2_SPREADS = (0.10, 0.20, 0.50)
F2P_SPREADS = (0.05, 0.10)
R11B_SPREAD = 0.05
T_SPREAD = 0.10
T_RUNS = 100
N_MIN = 20
MAX_REDRAWS = 3
RUN_EVENTS = 50
SIGMA_SPE = 0.3
V_DARK = 0.0
ORACLE_ARM = "oracle_nominal_twin"

NS_F1, NS_F3, NS_F2, NS_F2P = "lucid/F1", "lucid/F3", "lucid/F2", "lucid/F2p"
NS_R11B, NS_K_R11B, NS_K_T = "lucid/R11b", "lucid/k_R11b", "lucid/k_T"
NS_T_RATIOS, NS_DRAWS, NS_NOMINAL = "lucid/T_ratios", "lucid/draws", "lucid/nominal"

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _array_sha(array, dtype="<f8") -> str:
    return _sha(np.ascontiguousarray(array, dtype=dtype).tobytes())

def _event_ids_sha(run: "RunSpec") -> str:
    return _array_sha(np.arange(run.event_start, run.event_stop, dtype=np.int64), "<i8")

def keyed_rng(seed: int, *parts) -> np.random.Generator:
    """Seed-keyed namespace, never an integer offset assumption."""
    key = json.dumps([int(seed), *[str(p) for p in parts]], separators=(",", ":")).encode()
    entropy = np.frombuffer(hashlib.sha256(key).digest(), dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))

def current_code_sha(root: Path | str | None = None) -> str:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - host without git
        return "unknown"

# --------------------------------------------------------------------------- geometry interface
@dataclass(frozen=True)
class Geometry:
    """Explicit layout; the physical WCTE map binds here once available."""

    name: str
    n_modules: int
    pmts_per_module: int
    module_of_pmt: np.ndarray  # (n_pmt,)
    pmt_id_in_module: np.ndarray  # (n_pmt,)
    z_pmt_m: np.ndarray  # (n_pmt,), for F2'
    half_height_m: float  # H of §3.5.3

    def __post_init__(self):
        n = int(self.n_modules) * int(self.pmts_per_module)
        object.__setattr__(self, "module_of_pmt",
                           np.ascontiguousarray(self.module_of_pmt, dtype=np.int64))
        object.__setattr__(self, "pmt_id_in_module",
                           np.ascontiguousarray(self.pmt_id_in_module, dtype=np.int64))
        object.__setattr__(self, "z_pmt_m", np.ascontiguousarray(self.z_pmt_m, dtype=float))
        for name, shape in (("module_of_pmt", (n,)), ("pmt_id_in_module", (n,)),
                            ("z_pmt_m", (n,))):
            if getattr(self, name).shape != shape:
                raise ValueError(f"geometry {name} must have shape {shape}")
        if (self.n_modules <= 0 or self.pmts_per_module <= 0 or self.half_height_m <= 0
                or self.module_of_pmt.min() < 0 or self.module_of_pmt.max() >= self.n_modules
                or self.pmt_id_in_module.min() < 0
                or self.pmt_id_in_module.max() >= self.pmts_per_module):
            raise ValueError("invalid geometry dimensions or indices")
        if not np.all(np.bincount(self.module_of_pmt, minlength=self.n_modules)
                      == self.pmts_per_module):
            raise ValueError("every module must carry exactly pmts_per_module PMTs")
        if not np.isfinite(self.z_pmt_m).all():
            raise ValueError("geometry z coordinates must be finite")

    @property
    def n_pmt(self) -> int:
        return int(self.n_modules) * int(self.pmts_per_module)

    @property
    def digest(self) -> str:
        meta = json.dumps({"name": str(self.name), "n_modules": int(self.n_modules),
                           "pmts_per_module": int(self.pmts_per_module),
                           "half_height_m": float(self.half_height_m)},
                          sort_keys=True, separators=(",", ":")).encode()
        h = hashlib.sha256(meta)
        for a in (self.module_of_pmt, self.pmt_id_in_module, self.z_pmt_m):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

def synthetic_cylinder_geometry(n_modules: int = 8, pmts_per_module: int = 7, *,
                                half_height_m: float = 4.0, name: str =
                                "synthetic_cylinder_dev (NOT WCTE)") -> Geometry:
    """Small synthetic fixture for tests only; not WCTE and not a physics model."""
    z_m = -half_height_m + (np.arange(n_modules) + 0.5) * 2.0 * half_height_m / n_modules
    return Geometry(name=name, n_modules=n_modules, pmts_per_module=pmts_per_module,
                    module_of_pmt=np.repeat(np.arange(n_modules), pmts_per_module),
                    pmt_id_in_module=np.tile(np.arange(pmts_per_module), n_modules),
                    z_pmt_m=z_m[np.repeat(np.arange(n_modules), pmts_per_module)],
                    half_height_m=half_height_m)

# --------------------------------------------------------------------------- law primitives
def _clip_gain(raw):
    raw = np.asarray(raw, dtype=float)
    clipped = np.clip(raw, *GAIN_CLIP)
    return clipped, int(np.count_nonzero((raw < GAIN_CLIP[0]) | (raw > GAIN_CLIP[1])))

def f1_module_gain(u, s: float):
    """F1: ``gain_m = 1 + s u_m``, clipped; returns (gains, n_clipped)."""
    if s < 0 or not np.isfinite(s):
        raise ValueError("F1 spread must be finite and non-negative")
    return _clip_gain(1.0 + float(s) * np.asarray(u, dtype=float))

def f3_pmt_gain(w, s_p: float):
    """F3: ``gain_p = 1 + s_p w_p``, clipped; returns (gains, n_clipped)."""
    if s_p < 0 or not np.isfinite(s_p):
        raise ValueError("F3 spread must be finite and non-negative")
    return _clip_gain(1.0 + float(s_p) * np.asarray(w, dtype=float))

def f2_attenuation_length(v, a: float, lambda0_m: float = LAMBDA0_M):
    """F2: ``lambda = lambda0 exp(a v)`` (always positive)."""
    if lambda0_m <= 0 or a < 0 or not np.isfinite(a):
        raise ValueError("lambda0 must be positive and a finite and non-negative")
    return float(lambda0_m) * np.exp(float(a) * np.asarray(v, dtype=float))

def f2p_gradient(v, b: float, z_m, half_height_m: float):
    """F2': ``1 + b v z_p/H``, clipped; returns (factor, n_clipped)."""
    if b < 0 or not np.isfinite(b) or half_height_m <= 0:
        raise ValueError("F2' needs finite non-negative b and a positive half-height")
    return _clip_gain(1.0 + float(b) * np.asarray(v, dtype=float)
                      * np.asarray(z_m, dtype=float) / float(half_height_m))

def k_R11b(geometry: Geometry, seed: int = DEFAULT_SEED) -> np.ndarray:
    """``k_m`` drawn once from ``lucid/k_R11b`` for this geometry and seed."""
    return keyed_rng(seed, NS_K_R11B, geometry.digest).standard_normal(geometry.n_modules)

def k_T(geometry: Geometry, seed: int = DEFAULT_SEED) -> np.ndarray:
    """``k'_m`` from ``lucid/k_T``, independent of ``lucid/k_R11b``."""
    return keyed_rng(seed, NS_K_T, geometry.digest).standard_normal(geometry.n_modules)

def law_T_training_ratios(n_runs: int = T_RUNS) -> np.ndarray:
    """Law T training ratios ``r/R`` for ``r = 0..R-1``."""
    return np.arange(int(n_runs), dtype=float) / int(n_runs)


def law_T_ratios(n: int, *, kind: str = "in_range", seed: int = DEFAULT_SEED) -> np.ndarray:
    """Law T probe ratios: in-range ``U[0,1)`` or extrapolation ``U[1.9,2.1]``."""
    if kind == "in_range":
        return keyed_rng(seed, NS_T_RATIOS, kind).uniform(0.0, 1.0, int(n))
    if kind == "extrapolation":
        return keyed_rng(seed, NS_T_RATIOS, kind).uniform(1.9, 2.1, int(n))
    raise ValueError("kind must be 'in_range' or 'extrapolation'")

# --------------------------------------------------------------------------- detector state
LAWS = ("F1", "F3", "F2", "F2p", "R11b", "T")

@dataclass(frozen=True)
class DetectorState:
    """One run's state; the truth target of each event is its run's values."""

    law: str
    setting: float | None
    run_id: str
    seed: int
    namespace: str
    gains: np.ndarray  # (n_pmt,) gain_m * gain_p * F2' gradient
    module_gain: np.ndarray  # (n_modules,)
    pmt_gain: np.ndarray  # (n_pmt,)
    gradient: np.ndarray  # (n_pmt,), ones unless F2'
    attenuation_length_m: float
    clipped: dict

    @property
    def digest(self) -> str:
        meta = json.dumps({k: getattr(self, k) for k in
                           ("law", "setting", "run_id", "seed", "namespace",
                            "attenuation_length_m", "clipped")},
                          sort_keys=True, separators=(",", ":")).encode()
        h = hashlib.sha256(meta)
        for a in (self.gains, self.module_gain, self.pmt_gain, self.gradient):
            h.update(np.ascontiguousarray(a, dtype="<f8").tobytes())
        return h.hexdigest()

def nominal_state(geometry: Geometry, run_id: str = "nominal",
                  seed: int = DEFAULT_SEED) -> DetectorState:
    """All gains 1 and ``lambda = lambda0``; the twin's detector state."""
    m, p = np.ones(geometry.n_modules), np.ones(geometry.n_pmt)
    return DetectorState("nominal", None, run_id, int(seed), NS_NOMINAL, p.copy(), m,
                         p.copy(), p.copy(), LAMBDA0_M,
                         {"module": 0, "pmt": 0, "gradient": 0})

def state_for_run(law: str, setting, run_id: str, geometry: Geometry, *,
                  seed: int = DEFAULT_SEED, include_f3: bool = True,
                  include_f1_for_f2: bool = True) -> DetectorState:
    """Build a run's state.

    ``setting``: F1/F3 spread, F2 ``a``, F2' ``b``, R11b attenuation ``a``,
    T ratio ``r/R``.  F3 is included in every cell; F2/F2' also carry F1 at
    s = 0.05 so R11 has all three factors varying independently.
    """
    law = {"F2'": "F2p", "F2P": "F2p", "f2p": "F2p"}.get(law, law)
    if law not in LAWS:
        raise ValueError(f"unknown detector-state law {law!r}")
    s = float(setting)
    if not np.isfinite(s) or not isinstance(run_id, str) or not run_id:
        raise ValueError("finite setting and non-empty run_id required")
    m_gain, p_gain, grad = (np.ones(geometry.n_modules), np.ones(geometry.n_pmt),
                            np.ones(geometry.n_pmt))
    m_clip = p_clip = g_clip = 0
    lam, ns = LAMBDA0_M, {"F1": NS_F1, "F3": NS_F3, "F2": NS_F2, "F2p": NS_F2P,
                          "R11b": NS_R11B, "T": NS_K_T}[law]

    def f3():
        nonlocal p_gain, p_clip
        w = keyed_rng(seed, NS_F3, run_id).standard_normal(geometry.n_pmt)
        p_gain, p_clip = f3_pmt_gain(w, F3_SPREAD)

    if law == "F1":
        u = keyed_rng(seed, NS_F1, run_id).standard_normal(geometry.n_modules)
        m_gain, m_clip = f1_module_gain(u, s)
        if include_f3:
            f3()
    elif law == "F3":
        w = keyed_rng(seed, NS_F3, run_id).standard_normal(geometry.n_pmt)
        p_gain, p_clip = f3_pmt_gain(w, s)
    elif law in ("F2", "F2p"):
        if include_f1_for_f2:
            u = keyed_rng(seed, NS_F1, run_id).standard_normal(geometry.n_modules)
            m_gain, m_clip = f1_module_gain(u, 0.05)
        if law == "F2":
            v = float(keyed_rng(seed, NS_F2, run_id).standard_normal())
            lam = float(f2_attenuation_length(v, s))
        else:
            v = float(keyed_rng(seed, NS_F2P, run_id).standard_normal())
            grad, g_clip = f2p_gradient(v, s, geometry.z_pmt_m, geometry.half_height_m)
        if include_f3:
            f3()
    elif law == "R11b":
        c = float(keyed_rng(seed, NS_R11B, run_id).standard_normal())
        m_gain, m_clip = _clip_gain(1.0 + R11B_SPREAD * c * k_R11b(geometry, seed))
        lam = float(f2_attenuation_length(c, s))
        if include_f3:
            f3()
    else:  # T
        m_gain, m_clip = _clip_gain(1.0 + T_SPREAD * k_T(geometry, seed) * s)
        if include_f3:
            f3()
    return DetectorState(law, s, run_id, int(seed), ns,
                         m_gain[geometry.module_of_pmt] * p_gain * grad, m_gain, p_gain,
                         grad, lam, {"module": m_clip, "pmt": p_clip, "gradient": g_clip})

# --------------------------------------------------------------------------- readout contract
@dataclass(frozen=True)
class Draws:
    """Shared draws for one event; the twin must reuse them."""

    key: str
    run_id: str
    event_id: int
    photon_sensor: np.ndarray  # (n_photons,)
    photon_time_ns: np.ndarray  # (n_photons,)
    charge_z: np.ndarray  # (n_pmt,)
    transit_z: np.ndarray  # (n_pmt,)
    dark_counts: np.ndarray  # (n_pmt,)
    window_ns: float = 1000.0

@dataclass(frozen=True)
class HitReadout:
    """Observable hits for one event, sorted by PMT index."""

    pmt_id: np.ndarray
    charge_pe: np.ndarray
    time_ns: np.ndarray

    def __len__(self) -> int:
        return int(self.pmt_id.size)

ReadoutCallable = Callable[[DetectorState, Draws], HitReadout]

def nominal_twin(draws: Draws, geometry: Geometry, readout: ReadoutCallable, *,
                 arm: str) -> HitReadout:
    """Nominal-gain readout on the same draws; only the oracle arm may call this."""
    if arm != ORACLE_ARM:
        raise PermissionError(f"nominal twin is private to {ORACLE_ARM!r}, not {arm!r}")
    return readout(nominal_state(geometry, draws.run_id), draws)

# --------------------------------------------------------------------------- runs and IDs
@dataclass(frozen=True)
class CohortSpec:
    """A registered set: v3 §6 ID range, run length and namespace."""

    event_namespace: str
    event_start: int
    n_events: int
    run_length: int

    def __post_init__(self):
        if self.event_start < 0 or self.n_events <= 0 or self.run_length <= 0:
            raise ValueError("invalid cohort sizes")
        if self.n_events % self.run_length:
            raise ValueError("n_events must be a positive multiple of run_length")

    @property
    def n_runs(self) -> int:
        return self.n_events // self.run_length

COHORTS = {
    "train": CohortSpec("lucid/train", 0, 100_000, 1000),
    "probe_train": CohortSpec("lucid/probe_train", 200_000, 10_000, 50),
    "probe_test": CohortSpec("lucid/probe_test", 210_000, 10_000, 50),
    # v3 §6 has no robustness range; this is a WP4b extension, reported.
    "robustness": CohortSpec("lucid/robustness", 220_000, 10_000, 50),
    "dev_train": CohortSpec("lucid/dev_train", 1_000_000, 100_000, 1000),
    "dev_probe_train": CohortSpec("lucid/dev_probe_train",
                                  1_200_000, 10_000, 50),
    "dev_probe_test": CohortSpec("lucid/dev_probe_test",
                                 1_210_000, 10_000, 50),
    "dev_robustness": CohortSpec("lucid/dev_robustness",
                                 1_220_000, 10_000, 50),
}

@dataclass(frozen=True)
class RunSpec:
    """One run: a contiguous event block sharing one detector state."""

    run_id: str
    event_namespace: str
    law: str
    setting: float
    event_start: int
    n_events: int
    seed: int

    @property
    def event_stop(self) -> int:
        return self.event_start + self.n_events

    def state(self, geometry: Geometry) -> DetectorState:
        return state_for_run(self.law, self.setting, self.run_id, geometry, seed=self.seed)

    def state_sha256(self, geometry: Geometry) -> str:
        return self.state(geometry).digest

def make_runs(cohort: CohortSpec, law: str, setting=None, *, seed: int = DEFAULT_SEED,
              settings: Sequence[float] | None = None,
              event_id_start: int | None = None) -> tuple[RunSpec, ...]:
    """Build a cohort's runs; ``settings`` gives one setting per run (e.g. law T)."""
    per_run = None if settings is None else [float(x) for x in settings]
    if per_run is not None and len(per_run) != cohort.n_runs:
        raise ValueError("settings must have one entry per run")
    if per_run is None and setting is None:
        raise ValueError("either setting or settings is required")
    start = cohort.event_start if event_id_start is None else int(event_id_start)
    return tuple(RunSpec(f"{cohort.event_namespace}/{i:04d}", cohort.event_namespace, law,
                         float(per_run[i]) if per_run is not None else float(setting),
                         start + i * cohort.run_length, cohort.run_length, int(seed))
                 for i in range(cohort.n_runs))

def write_manifest(path: Path | str, runs: Sequence[RunSpec], geometry: Geometry,
                   seed: int = DEFAULT_SEED, *, code_sha: str | None = None) -> dict:
    """Write the manifest: run/event spans, law, state hashes, geometry hash, label, code SHA."""
    rows = [{"run_id": r.run_id, "event_namespace": r.event_namespace,
             "event_id_start": r.event_start, "event_id_stop": r.event_stop,
             "n_events": r.n_events, "law": r.law, "setting": r.setting,
             "state_sha256": r.state_sha256(geometry), "event_ids_sha256": _event_ids_sha(r)}
            for r in runs]
    payload = {"schema": "jepa-lucid-bombs-manifest-v0",
               "status": "development_only" if DEVELOPMENT_ONLY else "production",
               "readout_label": READOUT_LABEL, "reported_results": False,
               "geometry_name": geometry.name, "geometry_sha256": geometry.digest,
               "dataset_seed": int(seed), "code_sha": code_sha or current_code_sha(),
               "n_runs": len(rows), "n_events": int(sum(r.n_events for r in runs)),
               "runs": rows}
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload

def verify_manifest(payload: dict, runs: Sequence[RunSpec], geometry: Geometry,
                    seed: int = DEFAULT_SEED) -> None:
    """Recompute every state/event hash and raise on the first mismatch."""
    if payload.get("readout_label") != READOUT_LABEL:
        raise ValueError("manifest readout label mismatch")
    if payload.get("geometry_sha256") != geometry.digest:
        raise ValueError("manifest geometry hash mismatch")
    if int(payload.get("dataset_seed", -1)) != int(seed):
        raise ValueError("manifest seed mismatch")
    rows = payload.get("runs", [])
    if len(rows) != len(runs):
        raise ValueError("manifest run count mismatch")
    for row, run in zip(rows, runs):
        if (row.get("run_id") != run.run_id
                or int(row.get("event_id_start", -1)) != run.event_start
                or int(row.get("event_id_stop", -1)) != run.event_stop):
            raise ValueError(f"manifest run mismatch for {run.run_id}")
        if row.get("state_sha256") != run.state_sha256(geometry):
            raise ValueError(f"state hash mismatch for {run.run_id}")
        if row.get("event_ids_sha256") != _event_ids_sha(run):
            raise ValueError(f"event-id hash mismatch for {run.run_id}")

# --------------------------------------------------------------------------- sparse storage
@dataclass(frozen=True)
class SparseHitStore:
    """CSR-like hits: ``offsets`` indexes the hit arrays per event."""

    event_ids: np.ndarray
    offsets: np.ndarray
    pmt_id: np.ndarray
    charge_pe: np.ndarray
    time_ns: np.ndarray
    readout_label: str = READOUT_LABEL

def sparse_from_readouts(event_ids, hits: Sequence[HitReadout]) -> SparseHitStore:
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if event_ids.shape != (len(hits),):
        raise ValueError("one event id per readout required")
    if event_ids.size > 1 and not np.all(np.diff(event_ids) > 0):
        raise ValueError("event ids must be strictly increasing")
    offsets = np.zeros(len(hits) + 1, dtype=np.int64)
    pmt, charge, time_ns = [], [], []
    for i, readout in enumerate(hits):
        p = np.asarray(readout.pmt_id, dtype=np.int32)
        q = np.asarray(readout.charge_pe, dtype=np.float32)
        t = np.asarray(readout.time_ns, dtype=np.float32)
        if not (p.shape == q.shape == t.shape) or p.ndim != 1:
            raise ValueError("hit arrays must be one-dimensional and equally shaped")
        if np.any(q < 0) or not np.isfinite(q).all() or not np.isfinite(t).all():
            raise ValueError("hit charges/times must be finite and charges non-negative")
        if p.size > 1 and not np.all(np.diff(p) > 0):
            raise ValueError("hits must be sorted and unique within an event")
        offsets[i + 1] = offsets[i] + p.size
        pmt.append(p), charge.append(q), time_ns.append(t)
    cat = lambda parts, dtype: (np.concatenate(parts).astype(dtype) if parts
                                else np.empty(0, dtype))
    return SparseHitStore(event_ids, offsets, cat(pmt, np.int32), cat(charge, np.float32),
                          cat(time_ns, np.float32))

def write_sparse_hits(path: Path | str, store: SparseHitStore) -> Path:
    path = Path(path)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, event_ids=store.event_ids, offsets=store.offsets,
                            pmt_id=store.pmt_id, charge_pe=store.charge_pe,
                            time_ns=store.time_ns,
                            readout_label=np.asarray(store.readout_label))
    return path

def read_sparse_hits(path: Path | str) -> SparseHitStore:
    with np.load(Path(path), allow_pickle=False) as data:
        return SparseHitStore(data["event_ids"], data["offsets"], data["pmt_id"],
                              data["charge_pe"], data["time_ns"],
                              str(data["readout_label"]))

def write_sha256sums(directory: Path | str) -> Path:
    directory = Path(directory)
    lines = [f"{_sha(p.read_bytes())}  {p.name}" for p in sorted(directory.iterdir())
             if p.is_file() and p.name != "SHA256SUMS"]
    path = directory / "SHA256SUMS"
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path

def verify_sha256sums(directory: Path | str) -> None:
    directory = Path(directory)
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            target = directory / name
            if not target.is_file() or _sha(target.read_bytes()) != digest:
                raise ValueError(f"SHA256SUMS mismatch for {name}")

# --------------------------------------------------------------------------- N_min accounting
@dataclass
class ViewExclusionCounter:
    """§3.5.4 exclusion/redraw accounting; the mask itself is geometry-blocked."""

    n_min: int = N_MIN
    max_redraws: int = MAX_REDRAWS
    n_across: int = 0
    n_within: int = 0
    across_redraws: int = 0
    across_excluded: int = 0
    within_excluded: int = 0
    across_accepted: int = 0
    within_accepted: int = 0

    def record_across(self, target_hit_counts: Sequence[int]) -> bool:
        """Initial draw plus at most ``max_redraws``; returns whether the view is usable."""
        counts = [int(c) for c in target_hit_counts]
        if not counts or len(counts) > self.max_redraws + 1:
            raise ValueError("give the initial draw plus at most max_redraws redraws")
        self.n_across += 1
        for attempt, count in enumerate(counts):
            if count >= self.n_min:
                self.across_accepted += 1
                self.across_redraws += attempt
                return True
        self.across_excluded += 1
        self.across_redraws += len(counts) - 1
        return False

    def record_within(self, target_hits: int) -> bool:
        self.n_within += 1
        if int(target_hits) >= self.n_min:
            self.within_accepted += 1
            return True
        self.within_excluded += 1
        return False

    def rates(self) -> dict:
        return {"n_min": self.n_min, "max_redraws": self.max_redraws,
                "n_across": self.n_across, "n_within": self.n_within,
                "across_redraw_rate": self.across_redraws / self.n_across if self.n_across else 0.0,
                "across_exclusion_rate": self.across_excluded / self.n_across if self.n_across else 0.0,
                "within_exclusion_rate": self.within_excluded / self.n_within if self.n_within else 0.0,
                "across_accepted": self.across_accepted,
                "within_accepted": self.within_accepted}

def view_masks(geometry: Geometry, split: str, rng=None):
    """Documented stub: both views need the physical module map/centres."""
    raise NotImplementedError(
        "view masks require the physical WCTE geometry and module centres; "
        "N_min counters and exclusion accounting are implemented independently")

# --------------------------------------------------------------------------- detectability
@dataclass(frozen=True)
class Detectability:
    """§3.5.3a ceilings; lower bounds because nuisances are profiled away."""

    i_event: np.ndarray
    r2_event: np.ndarray
    i_run: float
    r2_run: float
    n_events: int
    sigma_f: float
    profiled: bool

def fisher_event(mu, dmu, *, sigma_spe: float = SIGMA_SPE, v_dark: float = V_DARK) -> float:
    """``sum_p (dmu_p/df)^2 / (mu_p (1 + sigma_spe^2) + v_dark)``."""
    mu, dmu = np.asarray(mu, float), np.asarray(dmu, float)
    if mu.ndim != 1 or mu.shape != dmu.shape:
        raise ValueError("mu and dmu must be one-dimensional and equally shaped")
    if not np.isfinite(mu).all() or not np.isfinite(dmu).all():
        raise ValueError("mu and dmu must be finite")
    if np.any(mu < 0) or v_dark < 0 or sigma_spe < 0:
        raise ValueError("mu, v_dark and sigma_spe must be non-negative")
    variance = mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark)
    if np.any((dmu != 0.0) & (variance <= 0.0)):
        raise ValueError("non-zero derivative where the Poisson variance vanishes")
    info = float(np.sum(np.divide(dmu ** 2, variance, out=np.zeros_like(mu),
                                  where=variance > 0)))
    if not np.isfinite(info):
        raise ValueError("non-finite Fisher information")
    return info

def r2_max(information: float, sigma_f: float) -> float:
    """``R^2_max = sigma_f^2 / (sigma_f^2 + 1/I)``; 0 when the information vanishes."""
    if sigma_f <= 0 or not np.isfinite(sigma_f):
        raise ValueError("sigma_f must be positive and finite")
    information = float(information)
    if information < 0 or np.isnan(information):
        raise ValueError("information must be non-negative")
    if information == 0.0:
        return 0.0
    if np.isinf(information):
        return 1.0
    return float(sigma_f ** 2 / (sigma_f ** 2 + 1.0 / information))

def detectability(mu_events, dmu_events, sigma_f: float, *, sigma_spe: float = SIGMA_SPE,
                  v_dark: float = V_DARK,
                  profiled_information: float | None = None) -> Detectability:
    """Per-event and per-run ceilings; ``profiled_information`` replaces the naive sum.

    The naive sum treats the vertex, yield and other factors as known, so it
    is not the registered profiling; the caller supplies the profiled
    information here once the profiling solver exists.
    """
    mu_events, dmu_events = list(mu_events), list(dmu_events)
    if not mu_events or len(mu_events) != len(dmu_events):
        raise ValueError("one mu/dmu pair per event is required")
    i_event = np.asarray([fisher_event(m, d, sigma_spe=sigma_spe, v_dark=v_dark)
                          for m, d in zip(mu_events, dmu_events)])
    if profiled_information is None:
        i_run, profiled = float(i_event.sum()), False
    else:
        i_run = float(profiled_information)
        if i_run < 0 or not np.isfinite(i_run):
            raise ValueError("profiled_information must be finite and non-negative")
        profiled = True
    return Detectability(i_event, np.asarray([r2_max(i, sigma_f) for i in i_event]),
                         i_run, r2_max(i_run, sigma_f), len(i_event), float(sigma_f), profiled)
