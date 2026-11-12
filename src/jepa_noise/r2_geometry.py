# SPDX-License-Identifier: MIT
"""Round-2 detector geometries: LUCiD cylinders (with or without modules) and the sphere.

Binding: ``docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md`` §2.2 and §4.1.

:class:`R2Geometry` is duck-type compatible with the round-1
:class:`jepa_noise.lucid_bombs.DetectorGeometry` (same attribute names, same layout
digest formula), so the round-1 generator primitives, view masks
(:mod:`jepa_noise.lucid_views`) and encoder binding work on it unchanged.  On WCTE it
returns the round-1 geometry exactly (test 1).

Every §2.2 rule lives here as a function of the geometry file, never as a typed value:

* modules: the real ``mpmt_id`` where present; otherwise **pseudo-modules** of size
  ``k`` = the divisor of N_PMT closest to 19 (ties -> larger), cut from the PMTs sorted in
  token Hilbert order (12 bits); ``pmt_id_in_mpmt`` = 1-based rank in the group;
* fiducial volume 0.8 x each dimension; position scaling and the Hilbert bounding box
  from the tank's bounding box: (R, R, H/2) for a cylinder, (R, R, R) for the sphere
  (the sphere carries ``height_m = 2R`` so every round-1 consumer of ``radius_m`` /
  ``half_height_m`` sees the right box);
* longest direct path, crossing time, prompt window, time-feature scale, NHits N and the
  receptive-field window of the ``mae_mse_rf`` control (:func:`derived_values`).

The JUNO sensor positions were generated once by LUCiD's own ``generate_detector`` on
``config/JUNO_geom_config.json`` at the pin and frozen to
``configs/round2/juno_geometry_lucid_eaf2e89.npz`` (sha256 pinned below).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import lucid_bombs as lb
from . import lucid_views as lv

__all__ = ["R2Geometry", "DETECTORS", "DESIGN_TABLE", "load_r2_geometry", "pseudo_module_size",
           "pseudo_modules", "fiducial_vertex", "distance_to_boundary", "derived_values",
           "nhits_n_rule", "prompt_window_rule", "rf_window_rule", "longest_path_m",
           "lucid_config_dir", "JUNO_FROZEN_RELPATH", "JUNO_FROZEN_SHA256"]

SHAPE_CYLINDER = "cylinder"
SHAPE_SPHERE = "sphere"
SPHERE_SURFACE = "sphere"
CYLINDER_SURFACES = ("top", "barrel", "bottom")

#: LUCiD files at the pin ``eaf2e89`` (sha256 of the files as read on Deepthought, 12 Nov 2026).
LUCID_FILE_SHA256 = {
    "WCTE_geom_config.json": "9943d6028e02bc7534a809b70e37893fedd1cd6f5e258babcbacc12613d5ed5c",
    "wcte_geometry.npz": "0a75b0352bd6e926d37e111c33f691ef2291a3a7d638f84cf618f91ad46bb9a8",
    "SK_geom_config.json": "35d609c3ca9203a0e796bd1c6e0ef6644d3be00dde4f6e39a27b16009de9e692",
    "sk_geometry.npz": "0eba275abaecb66e35a8aefcb3aa9cea2fe5c758eb99706b607ced68aed45433",
    "HK_geom_config.json": "d673cdbf4b67d137c4165b19517ad4e081e1a551ae1a7ecf71f400bc7b05d5c3",
    "hk_geometry.npz": "28c4482fb975df105170a2b909a7486d7112d53c58be8601ce904bccb6f7069b",
    "JUNO_geom_config.json": "9cacfe6a80615d3acdb749757f4fd9db3ded6588429bf82baee30588a1fd7413",
}
JUNO_FROZEN_RELPATH = "configs/round2/juno_geometry_lucid_eaf2e89.npz"
JUNO_FROZEN_SHA256 = "f91e6bc702d93b2a7f50847851517d21803ee994367d1bd10a72bb626db6b4f3"

#: Detector -> (LUCiD config file, shape).  The npz is named by the config.
DETECTORS = {
    "wcte": ("WCTE_geom_config.json", SHAPE_CYLINDER),
    "sk": ("SK_geom_config.json", SHAPE_CYLINDER),
    "hk": ("HK_geom_config.json", SHAPE_CYLINDER),
    "juno": ("JUNO_geom_config.json", SHAPE_SPHERE),
}

#: The §2.2 table as printed in the runbook (design time).  X0 recomputes every entry from
#: the files and stops (S4) on any difference; this table is only what it is compared with.
DESIGN_TABLE = {
    "wcte": {"n_pmt": 1995, "k": 19, "n_modules": 105, "pseudo_modules": False,
             "L_m": 4.20, "t_cross_ns": 18.6, "prompt_window_ns": (-20.0, 60.0),
             "time_scale_ns": 100.0, "nhits_n": 4, "w_rf": 64, "shift_rf": 32},
    "sk": {"n_pmt": 11096, "k": 19, "n_modules": 584, "pseudo_modules": True,
           "L_m": 49.74, "t_cross_ns": 220.7, "prompt_window_ns": (-20.0, 270.0),
           "time_scale_ns": 450.0, "nhits_n": 6, "w_rf": 384, "shift_rf": 192},
    "hk": {"n_pmt": 19746, "k": 18, "n_modules": 1097, "pseudo_modules": True,
           "L_m": 92.48, "t_cross_ns": 410.3, "prompt_window_ns": (-20.0, 460.0),
           "time_scale_ns": 766.7, "nhits_n": 8, "w_rf": 640, "shift_rf": 320},
    "juno": {"n_pmt": 10000, "k": 20, "n_modules": 500, "pseudo_modules": True,
             "L_m": 35.00, "t_cross_ns": 155.3, "prompt_window_ns": (-20.0, 200.0),
             "time_scale_ns": 333.3, "nhits_n": 6, "w_rf": 320, "shift_rf": 160},
}
#: Printed precision of each table column (the X0 comparison rounds to it).
DESIGN_DECIMALS = {"L_m": 2, "t_cross_ns": 1, "time_scale_ns": 1}

#: Round-1 constants the rules reproduce (quoted in runbook §2.1/§2.2).
FIDUCIAL_FRACTION = 0.8
K_TARGET = 19
WCTE_N_PMT = 1995
BASE_WINDOW, BASE_SHIFT = 64, 32
TIME_FEATURE_FRACTION = 0.6
PROMPT_LOW_NS = -20.0
PROMPT_MARGIN_NS = 40.0
NHITS_MIN = 4
NHITS_RUNS_PER_US = 50.0  # the rule's factor: 50 x P(Poisson(lambda_20) >= N) <= 1e-5
NHITS_FALSE_RATE = 1e-5
NHITS_WINDOW_NS = 20.0
DARK_RATE_HZ = lb.R6_DARK_RATE_HZ  # 1 kHz, bound from the merged manifest (§2.1)
N_GROUP = 1.33


def lucid_config_dir() -> Path:
    """Directory holding LUCiD's ``config/`` at the pin (``JEPA_LUCID_CONFIG`` overrides)."""
    import os
    for candidate in (os.environ.get("JEPA_LUCID_CONFIG"),
                      "/ceph/dwong/jepa/external/LUCiD/config",
                      str(Path.home() / "Documents/VERITAS_offrepo/lucid_config")):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError("LUCiD config directory not found; set JEPA_LUCID_CONFIG")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ======================================================================= the geometry type
@dataclass(frozen=True, eq=False)
class R2Geometry:
    """A LUCiD cylinder or sphere in the round-1 geometry schema.

    ``module_index`` is dense 0..n_modules-1; ``pmt_id_in_mpmt`` is 1-based.  For a
    pseudo-module detector ``mpmt_id == module_index`` and ``module_ids == arange``.
    """

    name: str
    detector: str
    shape: str
    positions_m: np.ndarray
    directions: np.ndarray
    surfaces: np.ndarray
    pmt_id: np.ndarray
    mpmt_id: np.ndarray
    pmt_id_in_mpmt: np.ndarray
    module_index: np.ndarray
    module_ids: np.ndarray
    module_centres_m: np.ndarray
    radius_m: float
    height_m: float
    sensor_radius_m: float
    pseudo_modules: bool
    source_path: str = ""
    source_sha256: str = ""
    config_sha256: str | None = None
    synthetic: bool = False

    def __post_init__(self):
        if self.shape not in (SHAPE_CYLINDER, SHAPE_SPHERE):
            raise ValueError(f"r2 geometry: unknown shape {self.shape!r}")
        pos = np.ascontiguousarray(self.positions_m, dtype=float)
        dirs = np.ascontiguousarray(self.directions, dtype=float)
        n = pos.shape[0]
        surf = np.asarray(self.surfaces).astype("<U8")
        arrays = {"positions_m": pos, "directions": dirs, "surfaces": surf,
                  "pmt_id": np.ascontiguousarray(self.pmt_id, dtype=np.int64),
                  "mpmt_id": np.ascontiguousarray(self.mpmt_id, dtype=np.int64),
                  "pmt_id_in_mpmt": np.ascontiguousarray(self.pmt_id_in_mpmt, dtype=np.int64),
                  "module_index": np.ascontiguousarray(self.module_index, dtype=np.int64),
                  "module_ids": np.ascontiguousarray(self.module_ids, dtype=np.int64),
                  "module_centres_m": np.ascontiguousarray(self.module_centres_m, dtype=float)}
        m = arrays["module_ids"].size
        for label, shape in (("positions_m", (n, 3)), ("directions", (n, 3)), ("surfaces", (n,)),
                             ("pmt_id", (n,)), ("mpmt_id", (n,)), ("pmt_id_in_mpmt", (n,)),
                             ("module_index", (n,)), ("module_centres_m", (m, 3))):
            if arrays[label].shape != shape:
                raise ValueError(f"r2 geometry: {label} must have shape {shape}")
        mi = arrays["module_index"]
        counts = np.bincount(mi, minlength=m)
        if mi.min() < 0 or mi.max() >= m or np.any(counts != counts[0]):
            raise ValueError("r2 geometry: every module must carry the same number of PMTs")
        k = int(counts[0])
        order = np.lexsort((arrays["pmt_id_in_mpmt"], mi))
        if not np.array_equal(arrays["pmt_id_in_mpmt"][order], np.tile(np.arange(1, k + 1), m)):
            raise ValueError("r2 geometry: pmt_id_in_mpmt must be 1..k within every module")
        if np.any(np.abs(np.linalg.norm(dirs, axis=1) - 1.0) > 1e-6):
            raise ValueError("r2 geometry: directions must be unit vectors")
        allowed = CYLINDER_SURFACES if self.shape == SHAPE_CYLINDER else (SPHERE_SURFACE,)
        if set(surf.tolist()) - set(allowed):
            raise ValueError(f"r2 geometry: surfaces must be in {allowed}")
        if self.shape == SHAPE_SPHERE and abs(float(self.height_m) - 2.0 * float(self.radius_m)) > 0:
            raise ValueError("r2 geometry: a sphere carries height_m = 2 R (bounding box)")
        for label, value in arrays.items():
            object.__setattr__(self, label, value)
        for label in ("radius_m", "height_m", "sensor_radius_m"):
            object.__setattr__(self, label, float(getattr(self, label)))

    # -- the round-1 interface ---------------------------------------------
    @property
    def n_sensors(self) -> int:
        return int(self.positions_m.shape[0])

    @property
    def n_pmt(self) -> int:
        return self.n_sensors

    @property
    def n_modules(self) -> int:
        return int(self.module_ids.size)

    @property
    def pmts_per_module(self) -> int:
        return self.n_sensors // self.n_modules

    @property
    def half_height_m(self) -> float:
        return 0.5 * float(self.height_m)

    @property
    def module_of_pmt(self) -> np.ndarray:
        return self.module_index

    @property
    def z_pmt_m(self) -> np.ndarray:
        return self.positions_m[:, 2]

    @property
    def surface_counts(self) -> tuple:
        labels, counts = np.unique(self.surfaces, return_counts=True)
        return tuple(sorted((str(a), int(b)) for a, b in zip(labels, counts)))

    @property
    def module_ids_contiguous(self) -> bool:
        return bool(np.array_equal(self.module_ids, np.arange(self.n_modules)))

    @property
    def half_extent_m(self) -> np.ndarray:
        """Position scaling: the half-extent of the tank's bounding box (§2.2)."""
        return np.array([self.radius_m, self.radius_m, self.half_height_m], dtype=float)

    @property
    def digest(self) -> str:
        """The round-1 layout digest formula (identical value on WCTE)."""
        return lb._geom_sha(
            {"n_sensors": int(self.n_sensors), "n_modules": int(self.n_modules),
             "pmts_per_module": int(self.pmts_per_module), "radius_m": float(self.radius_m),
             "height_m": float(self.height_m), "sensor_radius_m": float(self.sensor_radius_m),
             "surface_counts": dict(self.surface_counts)},
            ((self.positions_m, "<f8"), (self.directions, "<f8"), (self.surfaces, "<U8"),
             (self.pmt_id, "<i8"), (self.mpmt_id, "<i8"), (self.pmt_id_in_mpmt, "<i8"),
             (self.module_index, "<i8"), (self.module_ids, "<i8"),
             (self.module_centres_m, "<f8")))

    def summary(self) -> dict:
        return {"name": self.name, "detector": self.detector, "shape": self.shape,
                "synthetic": bool(self.synthetic), "source_path": self.source_path,
                "geometry_sha256": self.digest, "source_sha256": self.source_sha256,
                "config_sha256": self.config_sha256, "pseudo_modules": bool(self.pseudo_modules),
                "module_ids_contiguous": self.module_ids_contiguous,
                "surface_counts": dict(self.surface_counts), "n_sensors": int(self.n_sensors),
                "n_modules": int(self.n_modules), "pmts_per_module": int(self.pmts_per_module),
                "radius_m": float(self.radius_m), "height_m": float(self.height_m),
                "sensor_radius_m": float(self.sensor_radius_m)}


# ======================================================================= pseudo-modules
def pseudo_module_size(n_pmt: int, target: int = K_TARGET) -> int:
    """The divisor of ``n_pmt`` closest to ``target``; ties -> the larger (§2.2)."""
    n = int(n_pmt)
    if n <= 0:
        raise ValueError("pseudo_module_size needs a positive PMT count")
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    return int(min(divisors, key=lambda d: (abs(d - target), -d)))


class _Box:
    """The bounding-box view the round-1 Hilbert curve reads (radius, half-height)."""

    def __init__(self, radius_m: float, half_height_m: float):
        self.radius_m, self.half_height_m = float(radius_m), float(half_height_m)


def token_order_for_box(positions_m, radius_m: float, half_height_m: float,
                        bits: int = lv.TOKEN_HILBERT_BITS) -> np.ndarray:
    """The round-1 token Hilbert order (``lucid_views._curve_order``) on a bounding box."""
    return lv._curve_order(np.asarray(positions_m, dtype=float), _Box(radius_m, half_height_m),
                           int(bits))


def pseudo_modules(positions_m, radius_m: float, half_height_m: float, k: int | None = None):
    """Pseudo-module index and 1-based slot per PMT from the token Hilbert order (§2.2).

    Returns ``(module_index, pmt_id_in_mpmt, k)``: consecutive groups of ``k`` PMTs along the
    12-bit token Hilbert order; the slot is the rank inside the group (1-based).
    """
    pos = np.asarray(positions_m, dtype=float)
    n = pos.shape[0]
    k = pseudo_module_size(n) if k is None else int(k)
    if n % k:
        raise ValueError(f"pseudo-modules: {k} does not divide {n}")
    order = token_order_for_box(pos, radius_m, half_height_m)
    module_index = np.empty(n, dtype=np.int64)
    slot = np.empty(n, dtype=np.int64)
    rank = np.arange(n, dtype=np.int64)
    module_index[order] = rank // k
    slot[order] = rank % k + 1
    return module_index, slot, k


def _centres(positions_m, module_index, n_modules) -> np.ndarray:
    return np.stack([positions_m[module_index == i].mean(axis=0) for i in range(n_modules)])


# ======================================================================= loaders
def _from_round1_wcte(source: Path) -> R2Geometry:
    g = lb.load_wcte_geometry(source)
    return R2Geometry(name=g.name, detector="wcte", shape=SHAPE_CYLINDER,
                      positions_m=g.positions_m, directions=g.directions, surfaces=g.surfaces,
                      pmt_id=g.pmt_id, mpmt_id=g.mpmt_id, pmt_id_in_mpmt=g.pmt_id_in_mpmt,
                      module_index=g.module_index, module_ids=g.module_ids,
                      module_centres_m=g.module_centres_m, radius_m=g.radius_m,
                      height_m=g.height_m, sensor_radius_m=g.sensor_radius_m,
                      pseudo_modules=False, source_path=g.source_path,
                      source_sha256=g.source_sha256, config_sha256=g.config_sha256,
                      synthetic=bool(g.synthetic))


def build_cylinder(name: str, detector: str, positions_m, directions, surfaces, radius_m: float,
                   height_m: float, sensor_radius_m: float, *, pmt_id=None, source_path: str = "",
                   source_sha256: str = "", config_sha256: str | None = None,
                   synthetic: bool = False) -> R2Geometry:
    """A cylinder without modules: pseudo-modules from the token Hilbert order."""
    pos = np.asarray(positions_m, dtype=float)
    mi, slot, _ = pseudo_modules(pos, radius_m, 0.5 * float(height_m))
    m = int(mi.max()) + 1
    return R2Geometry(name=name, detector=detector, shape=SHAPE_CYLINDER, positions_m=pos,
                      directions=np.asarray(directions, dtype=float),
                      surfaces=lb._surface_array(surfaces),
                      pmt_id=(np.arange(1, pos.shape[0] + 1) if pmt_id is None else pmt_id),
                      mpmt_id=mi, pmt_id_in_mpmt=slot, module_index=mi,
                      module_ids=np.arange(m, dtype=np.int64),
                      module_centres_m=_centres(pos, mi, m), radius_m=radius_m,
                      height_m=height_m, sensor_radius_m=sensor_radius_m, pseudo_modules=True,
                      source_path=source_path, source_sha256=source_sha256,
                      config_sha256=config_sha256, synthetic=synthetic)


def build_sphere(name: str, detector: str, positions_m, radius_m: float, sensor_radius_m: float,
                 *, source_path: str = "", source_sha256: str = "",
                 config_sha256: str | None = None, synthetic: bool = False) -> R2Geometry:
    """A sphere: inward unit directions, ``surfaces = 'sphere'``, pseudo-modules."""
    pos = np.asarray(positions_m, dtype=float)
    radius = float(radius_m)
    dirs = -pos / np.linalg.norm(pos, axis=1, keepdims=True)
    mi, slot, _ = pseudo_modules(pos, radius, radius)
    m = int(mi.max()) + 1
    return R2Geometry(name=name, detector=detector, shape=SHAPE_SPHERE, positions_m=pos,
                      directions=dirs, surfaces=np.full(pos.shape[0], SPHERE_SURFACE),
                      pmt_id=np.arange(1, pos.shape[0] + 1), mpmt_id=mi, pmt_id_in_mpmt=slot,
                      module_index=mi, module_ids=np.arange(m, dtype=np.int64),
                      module_centres_m=_centres(pos, mi, m), radius_m=radius,
                      height_m=2.0 * radius, sensor_radius_m=sensor_radius_m,
                      pseudo_modules=True, source_path=source_path,
                      source_sha256=source_sha256, config_sha256=config_sha256,
                      synthetic=synthetic)


def load_r2_geometry(detector: str, config_dir: Path | str | None = None, *,
                     verify: bool = True) -> R2Geometry:
    """Load a round-2 detector from LUCiD's files at the pin (hashes verified)."""
    detector = str(detector).lower()
    if detector not in DETECTORS:
        raise ValueError(f"unknown detector {detector!r}; known {sorted(DETECTORS)}")
    cfg_name, shape = DETECTORS[detector]
    cdir = Path(config_dir) if config_dir is not None else lucid_config_dir()
    cfg_path = cdir / cfg_name
    cfg_sha = _sha_file(cfg_path)
    if verify and cfg_sha != LUCID_FILE_SHA256[cfg_name]:
        raise ValueError(f"{cfg_name}: sha256 {cfg_sha} != pinned {LUCID_FILE_SHA256[cfg_name]}")
    payload = json.loads(cfg_path.read_text())
    if payload.get("detector_type") != shape:
        raise ValueError(f"{cfg_name}: detector_type {payload.get('detector_type')!r} != {shape!r}")
    if detector == "wcte":
        return _from_round1_wcte(cfg_path)
    name = f"{detector.upper()} LUCiD {lb.LUCID_PIN[:8]} config/{cfg_name}"
    if shape == SHAPE_CYLINDER:
        npz_name = payload["geometry_definitions"]["npz_file_path"]
        npz_path = cdir / npz_name
        npz_sha = _sha_file(npz_path)
        if verify and npz_sha != LUCID_FILE_SHA256[npz_name]:
            raise ValueError(f"{npz_name}: sha256 {npz_sha} != pinned {LUCID_FILE_SHA256[npz_name]}")
        with np.load(npz_path, allow_pickle=False) as data:
            # Active PMTs only: the ``inactive_*`` entries (HK) are excluded (§2.2).
            arrays = {key: data[key] for key in ("positions_mm", "directions", "surfaces",
                                                 "pmt_id", "radius", "height", "sensor_radius")}
            if "mpmt_id" in data.files:
                raise ValueError(f"{npz_name}: a real module map exists; use it (not implemented "
                                 "for round 2 beyond WCTE)")
        scalar = lambda key: float(np.asarray(arrays[key]).reshape(-1)[0])
        return build_cylinder(name, detector, arrays["positions_mm"] / 1000.0,
                              arrays["directions"], arrays["surfaces"], scalar("radius"),
                              scalar("height"), scalar("sensor_radius"),
                              pmt_id=np.asarray(arrays["pmt_id"], dtype=np.int64),
                              source_path=str(npz_path), source_sha256=npz_sha,
                              config_sha256=cfg_sha)
    frozen = lb._repo_root() / JUNO_FROZEN_RELPATH
    frozen_sha = _sha_file(frozen)
    if verify and frozen_sha != JUNO_FROZEN_SHA256:
        raise ValueError(f"{frozen.name}: sha256 {frozen_sha} != pinned {JUNO_FROZEN_SHA256}")
    geo = payload["geometry_definitions"]
    with np.load(frozen, allow_pickle=False) as data:
        positions = np.asarray(data["positions_m"], dtype=float)
        if str(data["config_sha256"]) != cfg_sha:
            raise ValueError("frozen JUNO positions were generated from a different config")
        if (positions.shape[0] != int(geo["n_sensors"])
                or float(data["radius"]) != float(geo["radius"])
                or float(data["sensor_radius"]) != float(geo["sensor_radius"])):
            raise ValueError("frozen JUNO positions disagree with JUNO_geom_config.json")
    return build_sphere(name, detector, positions, float(geo["radius"]),
                        float(geo["sensor_radius"]), source_path=str(frozen),
                        source_sha256=frozen_sha, config_sha256=cfg_sha)


# ======================================================================= fiducial sampling
def fiducial_vertex(geometry, rng: np.random.Generator,
                    fraction: float = FIDUCIAL_FRACTION) -> np.ndarray:
    """Uniform-in-volume vertex in the fiducial volume (0.8 x each dimension).

    Cylinders call the round-1 :func:`lucid_bombs.fiducial_vertex` (bit-identical on WCTE).
    The sphere consumes the same three keyed uniforms in the same order (radius, azimuth,
    third coordinate), so a given event ID draws the same random numbers on every detector:
    ``r = 0.8 R u^(1/3)``, ``phi ~ U[0, 2 pi)``, ``cos theta ~ U[-1, 1]``.
    """
    if getattr(geometry, "shape", SHAPE_CYLINDER) == SHAPE_CYLINDER:
        return lb.fiducial_vertex(geometry, rng, fraction)
    radius = float(fraction) * float(geometry.radius_m)
    r = radius * np.cbrt(rng.random())
    phi = rng.uniform(0.0, 2.0 * np.pi)
    cos_t = rng.uniform(-1.0, 1.0)
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    return np.array([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t])


def distance_to_boundary(geometry, vertex_m, direction) -> float:
    """First positive intersection with the tank (cylinder: round-1 function; sphere: exact)."""
    if getattr(geometry, "shape", SHAPE_CYLINDER) == SHAPE_CYLINDER:
        return lb._distance_to_boundary(geometry, vertex_m, direction)
    p = np.asarray(vertex_m, dtype=float)
    d = np.asarray(direction, dtype=float)
    a = float(d @ d)
    b = 2.0 * float(p @ d)
    c = float(p @ p) - float(geometry.radius_m) ** 2
    disc = b * b - 4.0 * a * c
    if a <= 0.0 or disc < 0.0:
        raise ValueError("particle truth: no positive boundary intersection from the vertex")
    t = (-b + math.sqrt(disc)) / (2.0 * a)
    if t <= 1e-9:
        raise ValueError("particle truth: no positive boundary intersection from the vertex")
    return float(t)


def inside_fiducial(geometry, points_m, fraction: float = FIDUCIAL_FRACTION) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points_m, dtype=float))
    if getattr(geometry, "shape", SHAPE_CYLINDER) == SHAPE_SPHERE:
        return np.linalg.norm(pts, axis=1) <= fraction * geometry.radius_m
    return ((np.hypot(pts[:, 0], pts[:, 1]) <= fraction * geometry.radius_m)
            & (np.abs(pts[:, 2]) <= fraction * geometry.half_height_m))


def fiducial_radius_m(geometry, fraction: float = FIDUCIAL_FRACTION) -> float:
    """The fiducial radius used to normalise the vertex error (R2 readings)."""
    return float(fraction) * float(geometry.radius_m)


# ======================================================================= §2.2 derived values
def longest_path_m(geometry) -> float:
    """Cylinder sqrt((2R)^2 + H^2); sphere 2R."""
    if getattr(geometry, "shape", SHAPE_CYLINDER) == SHAPE_SPHERE:
        return 2.0 * float(geometry.radius_m)
    return math.hypot(2.0 * float(geometry.radius_m), float(geometry.height_m))


def crossing_time_ns(length_m: float, n_group: float = N_GROUP) -> float:
    return float(length_m) * float(n_group) / lb.C_M_PER_S * 1e9


def ceil10(value: float) -> float:
    return float(math.ceil(float(value) / 10.0 - 1e-12) * 10.0)


def prompt_window_rule(t_cross_ns: float) -> tuple:
    """(-20 ns, ceil_10(t_cross + 40 ns))."""
    return (PROMPT_LOW_NS, ceil10(float(t_cross_ns) + PROMPT_MARGIN_NS))


def time_scale_rule(prompt_window_ns) -> float:
    """Time-feature scale = prompt upper edge / 0.6 (100 ns on WCTE)."""
    return float(prompt_window_ns[1]) / TIME_FEATURE_FRACTION


def nhits_n_rule(n_pmt: int, dark_rate_hz: float = DARK_RATE_HZ,
                 window_ns: float = NHITS_WINDOW_NS) -> int:
    """Smallest N >= 4 with 50 P(Poisson(lambda_20) >= N) <= 1e-5."""
    from scipy.stats import poisson
    lam = float(dark_rate_hz) * int(n_pmt) * float(window_ns) * 1e-9
    n = NHITS_MIN
    while NHITS_RUNS_PER_US * float(poisson.sf(n - 1, lam)) > NHITS_FALSE_RATE:
        n += 1
    return int(n)


def rf_window_rule(n_pmt: int) -> tuple:
    """W_rf = the multiple of 64 nearest 64 N_PMT / 1995; shift W_rf / 2."""
    target = BASE_WINDOW * int(n_pmt) / WCTE_N_PMT
    lower = max(BASE_WINDOW, int(math.floor(target / BASE_WINDOW)) * BASE_WINDOW)
    upper = lower + BASE_WINDOW
    w = lower if abs(target - lower) <= abs(upper - target) else upper
    return int(w), int(w // 2)


def derived_values(geometry) -> dict:
    """Every §2.2 quantity recomputed from the geometry."""
    length = longest_path_m(geometry)
    t_cross = crossing_time_ns(length)
    prompt = prompt_window_rule(t_cross)
    w_rf, shift_rf = rf_window_rule(geometry.n_pmt)
    return {"detector": getattr(geometry, "detector", "wcte"),
            "shape": getattr(geometry, "shape", SHAPE_CYLINDER),
            "n_pmt": int(geometry.n_pmt), "k": int(geometry.pmts_per_module),
            "n_modules": int(geometry.n_modules),
            "pseudo_modules": bool(getattr(geometry, "pseudo_modules", False)),
            "radius_m": float(geometry.radius_m), "height_m": float(geometry.height_m),
            "half_extent_m": [float(geometry.radius_m), float(geometry.radius_m),
                              float(geometry.half_height_m)],
            "fiducial_fraction": FIDUCIAL_FRACTION,
            "L_m": length, "t_cross_ns": t_cross, "prompt_window_ns": prompt,
            "time_scale_ns": time_scale_rule(prompt), "nhits_n": nhits_n_rule(geometry.n_pmt),
            "nhits_window_ns": NHITS_WINDOW_NS, "w_rf": w_rf, "shift_rf": shift_rf}


def compare_with_design(values: dict, detector: str) -> list:
    """Differences between recomputed values and the printed §2.2 table (S4 if non-empty)."""
    design = DESIGN_TABLE[detector]
    diffs = []
    for key, expected in design.items():
        found = values[key]
        if key in DESIGN_DECIMALS:
            ok = round(float(found), DESIGN_DECIMALS[key]) == float(expected)
        elif isinstance(expected, tuple):
            ok = tuple(float(v) for v in found) == tuple(float(v) for v in expected)
        else:
            ok = found == expected
        if not ok:
            diffs.append({"key": key, "design": expected, "recomputed": found})
    return diffs


def readout_window_check(values: dict, readout_window_ns=lb.R6_READOUT_WINDOW_NS,
                         emission_offset_ns=lb.EMISSION_OFFSET_NS) -> dict:
    """X0: the readout window must contain the prompt window plus the emission offset."""
    lo, hi = (float(v) for v in readout_window_ns)
    e_lo, e_hi = (float(v) for v in emission_offset_ns)
    p_lo, p_hi = (float(v) for v in values["prompt_window_ns"])
    need = (min(p_lo, 0.0) + e_lo, p_hi + e_hi)
    return {"readout_window_ns": [lo, hi], "needed_ns": list(need),
            "ok": bool(lo <= need[0] and need[1] <= hi)}
