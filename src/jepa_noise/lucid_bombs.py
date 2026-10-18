# SPDX-License-Identifier: MIT
"""Development-only LUCiD photon-bomb generator contract (WP4b).

Binding: ``docs/jepa/RUNBOOK_JEPA_NOISE_v4_2026-10-16.md`` §3.5.2--§3.5.4,
§3.5.3a and v3 §6 WP5.  Everything is labelled ``readout_pmt_r6`` (the merged
PMT R-6 response, dark counts **on**, author decision 2026-10-18); the physical
WCTE geometry (105 mPMT × 19 PMTs, physical ``mpmt_id`` map) and pinned LUCiD
revision are unavailable locally, so nothing produced here is a reported result.

Implemented: the WCTE geometry reader (``load_wcte_geometry``: the physical
``mpmt_id`` map from LUCiD's own ``config/WCTE_geom_config.json`` at the pinned
revision, with a schema-identical local fixture from ``from_synthetic``);
detector-state laws F1/F3/F2/F2'/R11b/T with keyed RNG namespaces; the
photon-bomb event generator (§3.5.3: isotropic point source, log-uniform yield,
config spectrum, prompt transport with ``exp(-d/lambda)``, per-PMT expected
charge); the synthetic unrepaired readout (Poisson, SPE smearing, transit-time
spread, fixed-PE threshold, the ``noise_module_lucid.response`` walk primitive)
with the merged fixed detector response from
``configs/lucid_pmt/reference.json`` (per-module parity charge gain and digit
timing-width factor, present in the measured readout **and** the nominal twin)
and the R-6 **dark bank**: a keyed Poisson count per PMT (mean
``rate x window = 0.001``, so ~2 dark hits per WCTE-sized event), times uniform
in ``readout_window_ns``, joined to the detected PE before digitisation and then
sharing the SPE charge, gain, discriminator and T-Q walk/width; the nominal twin
on shared keyed draws; run structure, v3 §6 ID ranges under the ``lucid/``
prefix, manifest and sparse-hit storage; N_min exclusion accounting (§3.5.4);
§3.5.3a Fisher/ceiling numerics; and the 1000-event benchmark.  The synthetic
schema fixture is *not* WCTE; no LUCiD data is redistributed.

**A9 time convention** (author 2026-10-18): the token time is
``digit - NHits trigger`` after the prompt-window cut; the true emission time is
truth only (a keyed per-event offset uniform in [-50, +50] ns, recorded in
``events.npz`` and never in a feature).  The NHits rule is **imported lazily**
from the PMT readout (``noise_module_lucid.readout``) and never reimplemented
here; until the author names the SHA that module is unmerged, so the import
raises :class:`RuntimeError` and **no dataset may be generated**.  Tests inject
the documented interface stub; nothing falls back to another convention.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

#: The merged PMT reference response is bound by repo-relative path (never a literal copy).
FIXED_RESPONSE_RELPATH = "configs/lucid_pmt/reference.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_reference_path(path: Path | str | None = None) -> Path:
    """Resolve the reference-response path; a repo-relative path stays portable."""
    if path is None:
        return _repo_root() / FIXED_RESPONSE_RELPATH
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


try:  # the hash is read from the file, never typed in
    REFERENCE_RESPONSE_SHA256 = hashlib.sha256(
        resolve_reference_path().read_bytes()).hexdigest()
except OSError:  # pragma: no cover - checkout without the manifest
    REFERENCE_RESPONSE_SHA256 = ""


def reference_parameter_values(path: Path | str | None = None) -> dict:
    """The manifest's declared parameter values, keyed by name (R-6 binds these here).

    Read from the merged manifest itself (``configs/lucid_pmt/reference.json``), so the
    generator's dark-count, SPE, threshold and window values are **derived, not typed**.
    The full per-value hash validation happens in :func:`reference_response`.
    """
    payload = json.loads(resolve_reference_path(path).read_text())
    try:
        return {name: entry["value"] for name, entry in payload["parameters"].items()}
    except (KeyError, TypeError) as error:  # pragma: no cover - malformed manifest
        raise ValueError(f"{path or FIXED_RESPONSE_RELPATH}: malformed parameters block: "
                         f"{error}")


MANIFEST_PARAMETERS = reference_parameter_values()
#: PMT R-6 (merged): keyed Poisson dark photoelectrons per PMT, uniform in
#: ``readout_window_ns``, bypassing QE/CE, then the same SPE charge, gain,
#: discriminator, T-Q walk and width as every other photoelectron.
R6_DARK_RATE_HZ = float(MANIFEST_PARAMETERS["dark_rate_hz"])
R6_READOUT_WINDOW_NS = tuple(float(v) for v in MANIFEST_PARAMETERS["readout_window_ns"])
#: Window duration: ``end - start`` (the 1 us [-200, 800] ns window, never a literal).
R6_WINDOW_NS = float(R6_READOUT_WINDOW_NS[1] - R6_READOUT_WINDOW_NS[0])
R6_SPE_SIGMA = float(MANIFEST_PARAMETERS["spe_sigma"])
R6_THRESHOLD_PE = float(MANIFEST_PARAMETERS["threshold_pe"])

#: This readout is the merged PMT **R-6** response with dark counts **on**; the label
#: travels with every config, manifest, benchmark and decision payload.
READOUT_LABEL = "readout_pmt_r6"
#: Legacy labels.  Kept as constants so a stale artefact is recognisable; **never written**.
READOUT_LABEL_LEGACY_DARK_OFF = "readout_dark_off"
READOUT_LABEL_LEGACY = "readout_synthetic_unrepaired"
DARK_COUNTS_ON = "on"
DARK_COUNTS_OFF = "off"  # legacy: only a stale pre-R-6 artefact carries it
DARK_RATE_HZ_ON = R6_DARK_RATE_HZ
DARK_RATE_HZ_OFF = 0.0  # legacy: the pre-R-6 dark-off rate
#: A frozen decision that the R-6 re-run must replace.  The old values stay recorded;
#: they are never re-frozen as new values without the author.
SUPERSEDED_PENDING_R6_FREEZE = "superseded_pending_r6_freeze"
#: The intermediate ``28e24c8`` detectability run was never frozen; R-6 supersedes it.
SUPERSEDED_BY_R6_DARK_COUNTS = "superseded_by_r6_dark_counts"
#: Every artefact of this module is a development artefact, never a reported result.
DEVELOPMENT_ONLY = True

DEFAULT_SEED = 20261017
#: Placeholder: §3.5.3 fixes F2's form but no lambda0, and the PMT reference
#: fixture has absorption off.
LAMBDA0_M = 50.0
GAIN_CLIP = (0.5, 1.5)
F1_SPREADS = (0.02, 0.05, 0.10)
#: Registered default per v4 §3.5.3 (``s_p = 0.05``), adjustable only through §3.5.3a step 3.
F3_SPREAD_DEFAULT = 0.05
#: Restarted from the registered default on 2026-10-18 (author correction): the 0.10 value
#: came from the superseded run that treated F3 as *known*; with F3 marginalised the §3.5.3a
#: The registered cells use the value the frozen §3.5.3a ladder ended on, s_p = 0.10 (the ladder
#: starts at the default 0.05 and may raise s_p to ``F3_SPREAD_MAX`` only at step 3).
#: The §3.5.3a step-3 cap: s_p may rise to 0.10 only after the Y_hi and F2 steps.
F3_SPREAD_MAX = 0.10
F3_SPREAD = F3_SPREAD_MAX
F2_SPREADS = (0.10, 0.20, 0.50)
F2P_SPREADS = (0.05, 0.10)
R11B_SPREAD = 0.05
T_SPREAD = 0.10
T_RUNS = 100
N_MIN = 20
MAX_REDRAWS = 3
RUN_EVENTS = 50
#: The merged manifest's SPE resolution (0.3), not a literal.
SIGMA_SPE = R6_SPE_SIGMA


def dark_charge_variance(dark_rate_hz: float, window_ns: float,
                         sigma_spe: float) -> float:
    """Per-PMT dark-charge variance in one readout window (PE^2).

    Compound-Poisson: a dark count rate over the window gives lambda counts, each
    smeared by the SPE charge resolution, so Var = lambda * (1 + sigma_spe^2).
    v4 §3.5.6/§3.5.3 define ``v_dark`` as this reference dark-charge variance; under
    PMT R-6 ``lambda = 0.001`` per PMT-window, so ``v_dark = 0.001 x 1.09 = 0.00109``.
    """
    lam = float(dark_rate_hz) * float(window_ns) * 1e-9
    return lam * (1.0 + float(sigma_spe) ** 2)


#: Legacy history: the 4200 Hz / 1000 ns / sigma_spe 0.3 scaffold variance (0.004578 PE^2).
#: It was never the merged reference response's variance and is recorded here only so an
#: old artefact is recognisable; it must not be used for a reported ceiling.
V_DARK = round(dark_charge_variance(4200.0, 1000.0, 0.3), 6)  # 0.004578 PE^2
#: ``v_dark`` for **readout_pmt_r6**, derived from the merged manifest (never a literal):
#: ``dark_charge_variance(1000 Hz, 1000 ns, 0.3) = 0.001 x 1.09 = 0.00109 PE^2``.
V_DARK_READOUT = round(dark_charge_variance(R6_DARK_RATE_HZ, R6_WINDOW_NS, R6_SPE_SIGMA), 12)
#: The registered per-PMT variance with F3 marginalised as a random gain (§3.5.6).
VARIANCE_FORMULA = "mu_p * (1 + sigma_spe^2) + v_dark + s_p^2 * mu_p^2"
ORACLE_ARM = "oracle_nominal_twin"

NS_F1, NS_F3, NS_F2, NS_F2P = "lucid/F1", "lucid/F3", "lucid/F2", "lucid/F2p"
NS_R11B, NS_K_R11B, NS_K_T = "lucid/R11b", "lucid/k_R11b", "lucid/k_T"
NS_T_RATIOS, NS_DRAWS, NS_NOMINAL = "lucid/T_ratios", "lucid/draws", "lucid/nominal"
#: New keyed role for the A9 leak guard: one emission-time offset per event.  It is its own
#: keyed stream, so it perturbs no existing draw (vertex, yield, Poisson, dark, charge, transit).
NS_EMISSION_OFFSET = "emission_offset"

# --------------------------------------------------------------------------- A9 time convention
#: The author's registered convention (2026-10-18, part of A9), recorded by the config, the
#: manifest and the benchmark.  The **values used are the imported PMT readout's**; these are
#: only the record they are validated against (no second implementation of the rule).
TIME_RULE = {
    "trigger": "NHits",
    "n": 4,
    "trigger_window_ns": 20.0,
    "prompt_window_ns": (-20.0, 60.0),
    "no_trigger": "window_start",
    "emission_offset_ns": (-50.0, 50.0),
    "token_time": "digit - trigger",
    "source": "author 2026-10-18, A9; imported from the PMT readout, not reimplemented",
}
#: The emission-time offset interval (truth only): uniform per event, keyed RNG.
EMISSION_OFFSET_NS = tuple(float(v) for v in TIME_RULE["emission_offset_ns"])
#: Raised until the author names and merges the PMT SHA: an unvalidated time rule must not
#: silently become the generator's convention.
PENDING_TIME_RULE_MESSAGE = (
    "the PMT NHits time rule is pending the author-named R-6.5 SHA; merge it before generating")

#: Cached ``(nhits_trigger_time, n, trigger_window_ns, prompt_window_ns)`` from the PMT readout.
#: Tests install the documented interface stub here; production code only ever fills it through
#: the lazy import below, so a missing merge raises instead of falling back to a local rule.
_PMT_TIME_RULE = None


def pmt_time_rule_available() -> bool:
    """True iff the author-named PMT SHA (the NHits time rule) is merged and importable.

    Until then the generator must not produce digits (the convention would be undefined);
    dataset-generating tests skip with this predicate rather than silently using another rule.
    """
    try:
        pmt_time_rule()
        return True
    except RuntimeError:
        return False


def pmt_time_rule():
    """The merged PMT readout's NHits time rule, imported lazily (never reimplemented).

    Returns ``(nhits_trigger_time, prompt_window_mask, n, trigger_window_ns, prompt_window_ns)``.
    The prompt-window **cut** is the imported function, never reimplemented here.  Raises a clear
    :class:`RuntimeError` while the author-named PMT SHA is unmerged, so no code path can silently
    use a different time convention.
    """
    global _PMT_TIME_RULE
    if _PMT_TIME_RULE is not None:
        return _PMT_TIME_RULE
    try:  # the merged PMT branch's public interface; do not copy its code
        from noise_module_lucid.readout import (NHITS_N, NHITS_WINDOW_NS, PROMPT_WINDOW_NS,
                                                nhits_trigger_time, prompt_window_mask)
    except Exception as error:  # ImportError, or a module that does not yet expose the rule
        raise RuntimeError(PENDING_TIME_RULE_MESSAGE) from error
    rule = (nhits_trigger_time, prompt_window_mask, int(NHITS_N), float(NHITS_WINDOW_NS),
            tuple(float(v) for v in PROMPT_WINDOW_NS))
    # No silent convention drift: the imported rule must be the registered one.
    registered = (int(TIME_RULE["n"]), float(TIME_RULE["trigger_window_ns"]),
                  tuple(float(v) for v in TIME_RULE["prompt_window_ns"]))
    if rule[2:] != registered:
        raise RuntimeError(
            f"the PMT readout's time rule (n={rule[2]}, window_ns={rule[3]}, "
            f"prompt_window_ns={rule[4]}) differs from the registered A9 convention "
            f"{registered}; an author decision is required before generating")
    _PMT_TIME_RULE = rule
    return _PMT_TIME_RULE


def time_rule_record() -> dict:
    """The A9 convention as JSON-ready provenance (config, manifest, benchmark)."""
    return {"trigger": str(TIME_RULE["trigger"]), "n": int(TIME_RULE["n"]),
            "trigger_window_ns": float(TIME_RULE["trigger_window_ns"]),
            "prompt_window_ns": [float(v) for v in TIME_RULE["prompt_window_ns"]],
            "no_trigger": str(TIME_RULE["no_trigger"]),
            "emission_offset_ns": [float(v) for v in TIME_RULE["emission_offset_ns"]],
            "token_time": str(TIME_RULE["token_time"]),
            "source": str(TIME_RULE["source"])}


def validate_time_rule(block) -> dict:
    """Check a recorded ``time_rule`` block against the registered A9 convention.

    The readout still calls the **imported** PMT rule; this only guarantees that the recorded
    convention cannot drift from the one the code is written against.
    """
    if not isinstance(block, dict):
        raise ValueError("config: the 'time_rule' block is missing (author 2026-10-18, A9)")
    for key, expected in TIME_RULE.items():
        if key not in block:
            raise ValueError(f"config: time_rule.{key} is missing (A9 convention)")
        value = block[key]
        try:
            if isinstance(expected, tuple):
                value = tuple(float(v) for v in value)
            elif isinstance(expected, int):
                number = float(value)
                value = int(number) if number.is_integer() else number
            elif isinstance(expected, float):
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError):
            raise ValueError(f"config: time_rule.{key}={value!r} has the wrong shape") from None
        if value != expected:
            raise ValueError(f"config: time_rule.{key}={value!r} != the registered {expected!r}")
    return dict(block)

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
                  seed: int = DEFAULT_SEED, lambda0_m: float = LAMBDA0_M) -> DetectorState:
    """All gains 1 and ``lambda = lambda0``; the twin's detector state."""
    if not np.isfinite(lambda0_m) or float(lambda0_m) <= 0.0:
        raise ValueError("nominal_state needs a positive finite lambda0_m")
    m, p = np.ones(geometry.n_modules), np.ones(geometry.n_pmt)
    return DetectorState("nominal", None, run_id, int(seed), NS_NOMINAL, p.copy(), m,
                         p.copy(), p.copy(), float(lambda0_m),
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
    window_ns: float = R6_WINDOW_NS

@dataclass
class ReadoutRecord:
    """Per-run readout accounting for the A9 time convention (mutable counter).

    ``n_events`` counts readouts, ``n_no_trigger`` the events where
    :func:`nhits_trigger_time` returned ``None`` (origin falls back to the readout-window
    start), ``n_prompt_dropped`` the digits dropped by the prompt-window cut, and
    ``n_dark_moved_trigger`` the events in which a dark coincidence moves the trigger (the
    full-digit trigger differs from the trigger computed on the source-derived digits alone).
    It is written to the run manifest, never into a feature.
    """

    n_events: int = 0
    n_no_trigger: int = 0
    n_prompt_dropped: int = 0
    n_dark_moved_trigger: int = 0

    def as_dict(self) -> dict:
        return {"n_events": int(self.n_events), "n_no_trigger": int(self.n_no_trigger),
                "n_prompt_dropped": int(self.n_prompt_dropped),
                "n_dark_moved_trigger": int(self.n_dark_moved_trigger),
                "dark_moved_trigger_rate": (float(self.n_dark_moved_trigger) / float(self.n_events)
                                            if self.n_events else 0.0)}


@dataclass(frozen=True)
class HitReadout:
    """Observable hits for one event, sorted by PMT index.

    ``time_ns`` is the A9 **token time** ``digit - trigger`` after the prompt-window cut (the
    per-digit time stored in the sparse hit store).  ``absolute_time_ns`` (additive) carries
    the same digits' absolute readout times, so a ToF estimator that needs absolute times is
    unaffected; no estimator is changed here (it estimates t0, so trigger-relative times are
    admissible too).
    """

    pmt_id: np.ndarray
    charge_pe: np.ndarray
    time_ns: np.ndarray
    absolute_time_ns: np.ndarray | None = None
    #: The imported-rule trigger used for this event (``None`` = window-start fallback).
    trigger_ns: float | None = None
    #: The trigger the rule would return on the **source-derived** digits alone; equal to
    #: ``trigger_ns`` unless a dark coincidence moved it (see ``n_dark_moved_trigger``).
    source_trigger_ns: float | None = None

    def __len__(self) -> int:
        return int(self.pmt_id.size)

ReadoutCallable = Callable[[DetectorState, Draws], HitReadout]

def nominal_twin(draws: Draws, geometry: Geometry, readout: ReadoutCallable, *,
                 arm: str, lambda0_m: float = LAMBDA0_M) -> HitReadout:
    """Nominal-gain readout on the same draws; only the oracle arm may call this."""
    if arm != ORACLE_ARM:
        raise PermissionError(f"nominal twin is private to {ORACLE_ARM!r}, not {arm!r}")
    return readout(nominal_state(geometry, draws.run_id, lambda0_m=lambda0_m), draws)

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
                   seed: int = DEFAULT_SEED, *, code_sha: str | None = None,
                   extra: dict | None = None) -> dict:
    """Write the manifest: run/event spans, law, state hashes, geometry hash, label, code SHA.

    ``extra`` adds provenance (geometry source/config hashes, generator digest, yields);
    it may not shadow a reserved key.
    """
    rows = [{"run_id": r.run_id, "event_namespace": r.event_namespace,
             "event_id_start": r.event_start, "event_id_stop": r.event_stop,
             "n_events": r.n_events, "law": r.law, "setting": r.setting,
             "state_sha256": r.state_sha256(geometry), "event_ids_sha256": _event_ids_sha(r)}
            for r in runs]
    payload = {"schema": "jepa-lucid-bombs-manifest-v0",
               "status": "development_only" if DEVELOPMENT_ONLY else "production",
               "readout_label": READOUT_LABEL, "reported_results": False,
               "dark_counts": DARK_COUNTS_ON,
               "dark_rate_hz": R6_DARK_RATE_HZ,
               "readout_window_ns": list(R6_READOUT_WINDOW_NS),
               "window_ns": R6_WINDOW_NS,
               "v_dark": V_DARK_READOUT,
               "fixed_response": reference_provenance(),
               "yield_range_decision_status": YIELD_RANGE_DECISION_STATUS_IN_FORCE,
               "geometry_name": geometry.name, "geometry_sha256": geometry.digest,
               "dataset_seed": int(seed), "code_sha": code_sha or current_code_sha(),
               "n_runs": len(rows), "n_events": int(sum(r.n_events for r in runs)),
               "runs": rows}
    if extra:
        clash = sorted(set(extra) & set(payload))
        if clash:
            raise ValueError(f"manifest extra keys collide with reserved keys: {clash}")
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return payload

def verify_manifest(payload: dict, runs: Sequence[RunSpec], geometry: Geometry,
                    seed: int = DEFAULT_SEED) -> None:
    """Recompute every state/event hash and the R-6 dark provenance; raise on a mismatch."""
    if payload.get("readout_label") != READOUT_LABEL:
        raise ValueError("manifest readout label mismatch")
    if payload.get("dark_counts") != DARK_COUNTS_ON:
        raise ValueError(f"manifest dark_counts must be {DARK_COUNTS_ON!r} (PMT R-6)")
    if float(payload.get("dark_rate_hz", -1.0)) != R6_DARK_RATE_HZ:
        raise ValueError("manifest dark_rate_hz mismatch")
    if tuple(float(v) for v in payload.get("readout_window_ns", ())) != R6_READOUT_WINDOW_NS:
        raise ValueError("manifest readout_window_ns mismatch")
    if float(payload.get("window_ns", -1.0)) != R6_WINDOW_NS:
        raise ValueError("manifest window_ns (end - start) mismatch")
    if float(payload.get("v_dark", -1.0)) != V_DARK_READOUT:
        raise ValueError("manifest v_dark mismatch (it is derived from the R-6 manifest)")
    if payload.get("fixed_response", {}).get("sha256") != reference_response().sha256:
        raise ValueError("manifest fixed-response hash mismatch")
    if "time_rule" in payload and payload["time_rule"] != time_rule_record():
        raise ValueError("manifest time_rule mismatch (A9 convention)")
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
        "view masks are WP4c encoder work: the geometry now supplies module_ids, "
        "module_centres_m and the 1-based pmt_id_in_mpmt; N_min counters and exclusion "
        "accounting are implemented independently")

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
    s_p: float = F3_SPREAD

def fisher_event(mu, dmu, *, sigma_spe: float = SIGMA_SPE, v_dark: float = V_DARK_READOUT,
                 s_p: float = F3_SPREAD) -> float:
    """``sum_p (dmu_p/df)^2 / (mu_p (1 + sigma_spe^2) + v_dark + s_p^2 mu_p^2)``.

    F3 is marginalised as a random per-PMT gain: ``s_p^2 mu_p^2`` is the same
    covariance term as the §3.5.6 ``lucid_arms.chi2_dense_module`` diagonal.  A
    zero-mean PMT has zero variance when ``v_dark = 0`` and carries no information,
    so exactly-zero-``mu`` PMTs are skipped rather than dividing by zero.
    """
    mu, dmu = np.asarray(mu, float), np.asarray(dmu, float)
    if mu.ndim != 1 or mu.shape != dmu.shape:
        raise ValueError("mu and dmu must be one-dimensional and equally shaped")
    if not np.isfinite(mu).all() or not np.isfinite(dmu).all():
        raise ValueError("mu and dmu must be finite")
    if np.any(mu < 0) or v_dark < 0 or sigma_spe < 0 or s_p < 0:
        raise ValueError("mu, v_dark, sigma_spe and s_p must be non-negative")
    variance = (mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark)
                + float(s_p) ** 2 * mu ** 2)
    keep = (variance > 0.0) & (mu > 0.0)
    info = float(np.sum(np.divide(dmu ** 2, variance, out=np.zeros_like(mu), where=keep)))
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
                  v_dark: float = V_DARK_READOUT, s_p: float = F3_SPREAD,
                  profiled_information: float | None = None) -> Detectability:
    """Per-event and per-run ceilings; ``profiled_information`` replaces the naive sum.

    The naive sum treats the vertex, yield and other factors as known, so it
    is not the registered profiling; the caller supplies the profiled
    information here once the profiling solver exists.
    """
    mu_events, dmu_events = list(mu_events), list(dmu_events)
    if not mu_events or len(mu_events) != len(dmu_events):
        raise ValueError("one mu/dmu pair per event is required")
    i_event = np.asarray([fisher_event(m, d, sigma_spe=sigma_spe, v_dark=v_dark, s_p=s_p)
                          for m, d in zip(mu_events, dmu_events)])
    if profiled_information is None:
        i_run, profiled = float(i_event.sum()), False
    else:
        i_run = float(profiled_information)
        if i_run < 0 or not np.isfinite(i_run):
            raise ValueError("profiled_information must be finite and non-negative")
        profiled = True
    return Detectability(i_event, np.asarray([r2_max(i, sigma_f) for i in i_event]),
                         i_run, r2_max(i_run, sigma_f), len(i_event), float(sigma_f), profiled,
                         s_p=float(s_p))

# ======================================================================= WCTE geometry (WP4b)
#: Pinned LUCiD revision and geometry facts, DECISIONS.md 2026-10-17.
LUCID_PIN = "eaf2e89f579d6679e90020c732ac2c846895d691"
WCTE_CONFIG_RELPATH = "config/WCTE_geom_config.json"
WCTE_NPZ_RELPATH = "config/wcte_geometry.npz"
WCTE_CONFIG_SHA256 = "9943d6028e02bc7534a809b70e37893fedd1cd6f5e258babcbacc12613d5ed5c"
WCTE_NPZ_SHA256_PREFIX = "0a75b035"
WCTE_N_SENSORS, WCTE_N_MODULES, WCTE_PMTS_PER_MODULE = 1995, 105, 19
WCTE_SURFACE_NAMES = ("top", "barrel", "bottom")
WCTE_SURFACE_COUNTS = {"top": 399, "barrel": 1197, "bottom": 399}
WCTE_RADIUS_M, WCTE_HEIGHT_M = 1.5771308488246254, 2.77697099384
WCTE_SENSOR_RADIUS_M = 0.04
C_M_PER_S = 299792458.0
#: Effectively infinite attenuation length: the capacity bound (attenuation only removes light).
UNATTENUATED_M = 1.0e9

def _geom_sha(meta: dict, arrays) -> str:
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    for array, dtype in arrays:
        h.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return h.hexdigest()

def _surface_array(values, surface_map=None) -> np.ndarray:
    out = []
    for raw in np.asarray(values).reshape(-1).tolist():
        label = str(raw).strip().lower()
        out.append(surface_map.get(label, label) if surface_map else label)
    return np.array(out, dtype="<U8")

@dataclass(frozen=True, eq=False)
class DetectorGeometry:
    """Validated WCTE sensor map, or a schema-identical synthetic fixture.

    ``module_index`` is the dense 0..n_modules-1 index; ``mpmt_id`` keeps the raw
    module ids (105 distinct values over 0..105 with id 45 absent), so nothing
    may assume a dense 0..104 block.  ``pmt_id_in_mpmt`` is **1-based** (1..19),
    exactly as the geometry file gives it.
    """

    name: str
    positions_m: np.ndarray        # (n_sensors, 3)
    directions: np.ndarray         # (n_sensors, 3), inward-facing unit vectors
    surfaces: np.ndarray           # (n_sensors,) 'top' | 'barrel' | 'bottom'
    pmt_id: np.ndarray             # (n_sensors,) 1..n_sensors
    mpmt_id: np.ndarray            # (n_sensors,) raw module id
    pmt_id_in_mpmt: np.ndarray     # (n_sensors,) 1..19
    module_index: np.ndarray       # (n_sensors,) dense module index
    module_ids: np.ndarray         # (n_modules,) sorted raw ids
    module_centres_m: np.ndarray   # (n_modules, 3)
    radius_m: float
    height_m: float
    sensor_radius_m: float
    source_path: str = "<synthetic schema fixture>"
    source_sha256: str = ""
    config_sha256: str | None = None
    n_sensors: int = WCTE_N_SENSORS
    n_modules: int = WCTE_N_MODULES
    pmts_per_module: int = WCTE_PMTS_PER_MODULE
    surface_counts: tuple = tuple(sorted(WCTE_SURFACE_COUNTS.items()))
    synthetic: bool = False

    def __post_init__(self):
        n, m, k = int(self.n_sensors), int(self.n_modules), int(self.pmts_per_module)
        if n <= 0 or m <= 0 or k <= 0:
            raise ValueError("geometry: n_sensors, n_modules and pmts_per_module must be positive")
        pos = np.ascontiguousarray(self.positions_m, dtype=float)
        dirs = np.ascontiguousarray(self.directions, dtype=float)
        surf = _surface_array(self.surfaces)
        p_id = np.ascontiguousarray(self.pmt_id, dtype=np.int64)
        m_id = np.ascontiguousarray(self.mpmt_id, dtype=np.int64)
        p_in = np.ascontiguousarray(self.pmt_id_in_mpmt, dtype=np.int64)
        mi = np.ascontiguousarray(self.module_index, dtype=np.int64)
        mids = np.ascontiguousarray(self.module_ids, dtype=np.int64)
        centres = np.ascontiguousarray(self.module_centres_m, dtype=float)
        for label, array, shape in (("positions_m", pos, (n, 3)),
                                    ("directions", dirs, (n, 3)),
                                    ("surfaces", surf, (n,)),
                                    ("pmt_id", p_id, (n,)), ("mpmt_id", m_id, (n,)),
                                    ("pmt_id_in_mpmt", p_in, (n,)),
                                    ("module_index", mi, (n,)),
                                    ("module_ids", mids, (m,)),
                                    ("module_centres_m", centres, (m, 3))):
            if array.shape != shape:
                raise ValueError(f"geometry: {label} must have shape {shape}, got {array.shape}")
        if mids.size > 1 and np.any(np.diff(mids) <= 0):
            raise ValueError("geometry: module_ids must be strictly increasing and unique")
        if mi.min() < 0 or mi.max() >= m:
            raise ValueError("geometry: module_index out of range for module_ids")
        counts = np.bincount(mi, minlength=m)
        bad = np.flatnonzero(counts != k)
        if bad.size:
            j = int(bad[0])
            raise ValueError(f"geometry: module mpmt_id {int(mids[j])} carries {int(counts[j])} "
                             f"PMTs, expected {k} (a module with missing PMTs is refused)")
        declared = dict(self.surface_counts)
        found = {s: int(np.count_nonzero(surf == s)) for s in WCTE_SURFACE_NAMES}
        extra = sorted(set(surf.tolist()) - set(WCTE_SURFACE_NAMES))
        if extra:
            raise ValueError(f"geometry: unknown surface labels {extra}; pass surface_map to name them "
                             f"as {list(WCTE_SURFACE_NAMES)}")
        if found != declared:
            raise ValueError(f"geometry: surface counts {found} != declared {declared}")
        if np.unique(p_id).size != n or p_id.min() < 1 or p_id.max() > n:
            raise ValueError(f"geometry: pmt_id must be the unique integers 1..{n}")
        order = np.lexsort((p_in, mi))
        if not np.array_equal(p_in[order], np.tile(np.arange(1, k + 1), m)):
            raise ValueError(f"geometry: pmt_id_in_mpmt must be 1..{k} within every module "
                             "(the file is 1-based; 0-based ids are refused)")
        if not (np.isfinite(pos).all() and np.isfinite(dirs).all() and np.isfinite(centres).all()):
            raise ValueError("geometry: positions, directions and module centres must be finite")
        norms = np.linalg.norm(dirs, axis=1)
        if np.any(np.abs(norms - 1.0) > 1e-6):
            raise ValueError(f"geometry: directions must be unit vectors (max |n|-1 = "
                             f"{np.abs(norms - 1.0).max():.3g})")
        scalars = (float(self.radius_m), float(self.height_m), float(self.sensor_radius_m))
        if not all(np.isfinite(s) and s > 0 for s in scalars):
            raise ValueError("geometry: radius_m, height_m and sensor_radius_m must be positive and finite")
        if np.abs(pos[:, 2]).max() > 0.5 * scalars[1] * (1.0 + 1e-6) + 1e-9:
            raise ValueError("geometry: sensor z exceeds half the detector height")
        for label, value in (("positions_m", pos), ("directions", dirs), ("surfaces", surf),
                             ("pmt_id", p_id), ("mpmt_id", m_id), ("pmt_id_in_mpmt", p_in),
                             ("module_index", mi), ("module_ids", mids),
                             ("module_centres_m", centres)):
            object.__setattr__(self, label, value)
        object.__setattr__(self, "surface_counts", tuple(sorted(declared.items())))

    # -- derived views -----------------------------------------------------
    @property
    def n_pmt(self) -> int:
        return int(self.n_sensors)

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
    def module_ids_contiguous(self) -> bool:
        """False for the real map (id 45 absent); never assume a dense id block."""
        return bool(np.array_equal(self.module_ids, np.arange(self.n_modules)))

    @property
    def digest(self) -> str:
        """Layout-only hash: identical arrays give identical digests across files."""
        return _geom_sha({"n_sensors": int(self.n_sensors), "n_modules": int(self.n_modules),
                          "pmts_per_module": int(self.pmts_per_module),
                          "radius_m": float(self.radius_m), "height_m": float(self.height_m),
                          "sensor_radius_m": float(self.sensor_radius_m),
                          "surface_counts": dict(self.surface_counts)},
                         ((self.positions_m, "<f8"), (self.directions, "<f8"),
                          (self.surfaces, "<U8"), (self.pmt_id, "<i8"), (self.mpmt_id, "<i8"),
                          (self.pmt_id_in_mpmt, "<i8"), (self.module_index, "<i8"),
                          (self.module_ids, "<i8"), (self.module_centres_m, "<f8")))

    def summary(self) -> dict:
        out = {"name": self.name, "synthetic": bool(self.synthetic),
               "source_path": self.source_path, "geometry_sha256": self.digest,
               "source_sha256": self.source_sha256, "config_sha256": self.config_sha256,
               "module_ids_contiguous": self.module_ids_contiguous,
               "surface_counts": dict(self.surface_counts)}
        out.update(n_sensors=int(self.n_sensors), n_modules=int(self.n_modules),
                   pmts_per_module=int(self.pmts_per_module), radius_m=float(self.radius_m),
                   height_m=float(self.height_m), sensor_radius_m=float(self.sensor_radius_m))
        return out

def from_synthetic(schema=None) -> DetectorGeometry:
    """Schema-identical synthetic fixture, generated here; **not** WCTE and never LUCiD data.

    ``schema`` may override the counts (tests only): module ids, PMTs per module, surface
    counts/names, the three scalars, ``pmt_id_in_mpmt_start`` and
    ``drop_last_pmt_of_module``.
    """
    s = {"module_ids": tuple(x for x in range(106) if x != 45),
         "pmts_per_module": WCTE_PMTS_PER_MODULE,
         "surface_counts": dict(WCTE_SURFACE_COUNTS), "surface_names": WCTE_SURFACE_NAMES,
         "radius_m": WCTE_RADIUS_M, "height_m": WCTE_HEIGHT_M,
         "sensor_radius_m": WCTE_SENSOR_RADIUS_M, "pmt_id_in_mpmt_start": 1,
         "drop_last_pmt_of_module": None}
    s.update(dict(schema or {}))
    module_ids = np.asarray(s["module_ids"], dtype=np.int64)
    mid, per, names = int(module_ids.size), int(s["pmts_per_module"]), tuple(s["surface_names"])
    counts = {label: int(s["surface_counts"][label]) for label in names}
    radius, height = float(s["radius_m"]), float(s["height_m"])
    golden, ratio = np.pi * (3.0 - np.sqrt(5.0)), 0.5 * (2.0 ** 0.5 - 1.0)
    pos, dirs, surf = [], [], []
    for label in names:
        c = counts[label]
        j = np.arange(c, dtype=float)
        if label in ("top", "bottom"):
            r = radius * np.sqrt((j + ratio) / max(c, 1))
            phi = 2.0 * np.pi * j * golden
            z = np.full(c, (0.5 if label == "top" else -0.5) * height)
            pos.append(np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1))
            dirs.append(np.tile([0.0, 0.0, -1.0 if label == "top" else 1.0], (c, 1)))
        else:
            phi = 2.0 * np.pi * j / max(c, 1)
            z = -0.5 * height + (j + 0.5) * height / max(c, 1)
            pos.append(np.stack([radius * np.cos(phi), radius * np.sin(phi), z], axis=1))
            dirs.append(np.stack([-np.cos(phi), -np.sin(phi), np.zeros(c)], axis=1))
        surf.append(np.full(c, label, dtype="<U8"))
    pos, dirs, surf = np.concatenate(pos), np.concatenate(dirs), np.concatenate(surf)
    index = np.arange(sum(counts.values()), dtype=np.int64)
    drop = s["drop_last_pmt_of_module"]
    if drop is not None:
        victim = int(drop) * per + per - 1
        if not 0 <= victim < index.size:
            raise ValueError("from_synthetic: drop_last_pmt_of_module out of range")
        keep = index != victim
        pos, dirs, surf, index = pos[keep], dirs[keep], surf[keep], index[keep]
    mi = index // per
    centres = np.stack([pos[mi == i].mean(axis=0) for i in range(mid)])
    return DetectorGeometry(
        name="synthetic_wcte_schema_fixture (NOT WCTE)", positions_m=pos, directions=dirs,
        surfaces=surf, pmt_id=np.arange(1, pos.shape[0] + 1, dtype=np.int64),
        mpmt_id=module_ids[mi],
        pmt_id_in_mpmt=(index % per + int(s["pmt_id_in_mpmt_start"])).astype(np.int64),
        module_index=mi, module_ids=module_ids, module_centres_m=centres,
        radius_m=radius, height_m=height, sensor_radius_m=float(s["sensor_radius_m"]),
        n_sensors=int(pos.shape[0]), n_modules=mid, pmts_per_module=per,
        surface_counts=tuple(sorted(counts.items())), synthetic=True)

def load_wcte_geometry(source, *, surface_map=None, expected_config_sha256=None,
                       expected_npz_sha256_prefix=None, name=None,
                       synthetic: bool | None = None) -> DetectorGeometry:
    """Read LUCiD's WCTE geometry (``.json`` pointer or the ``.npz`` it points at).

    The JSON pointer is verified against the pinned SHA-256 (``WCTE_CONFIG_SHA256``)
    when it is named ``WCTE_geom_config.json`` or when ``expected_config_sha256`` is
    given; the npz is checked against ``WCTE_NPZ_SHA256_PREFIX``.  No LUCiD code or
    data is imported or redistributed.  ``synthetic`` marks a local fixture (a config
    with ``"synthetic": true`` does the same) so a benchmark can never be mistaken
    for the real-geometry run.
    """
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"geometry: source not found: {path}")
    config_sha = None
    if path.suffix.lower() == ".json":
        raw = path.read_bytes()
        config_sha = _sha(raw)
        pinned = expected_config_sha256
        if pinned is None and path.name == Path(WCTE_CONFIG_RELPATH).name:
            pinned = WCTE_CONFIG_SHA256
        if pinned is not None and config_sha != pinned:
            raise ValueError(f"geometry: {path.name} sha256 {config_sha} != pinned {pinned}")
        payload = json.loads(raw.decode())
        if synthetic is None:
            synthetic = bool(payload.get("synthetic", False))
        rel = payload.get("geometry_definitions", {}).get("npz_file_path")
        if not rel:
            raise ValueError("geometry: config has no geometry_definitions.npz_file_path")
        npz_path = (path.parent / rel).resolve()
        if expected_npz_sha256_prefix is None and path.name == Path(WCTE_CONFIG_RELPATH).name:
            expected_npz_sha256_prefix = WCTE_NPZ_SHA256_PREFIX
    elif path.suffix.lower() == ".npz":
        npz_path = path
    else:
        raise ValueError("geometry: source must be a .json config or a .npz file")
    if not npz_path.is_file():
        raise FileNotFoundError(f"geometry: npz not found: {npz_path}")
    npz_sha = _sha(npz_path.read_bytes())
    if expected_npz_sha256_prefix and not npz_sha.startswith(expected_npz_sha256_prefix):
        raise ValueError(f"geometry: {npz_path.name} sha256 {npz_sha} does not start with the "
                         f"pinned {expected_npz_sha256_prefix}")
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    required = ("positions_mm", "directions", "surfaces", "pmt_id", "mpmt_id", "pmt_id_in_mpmt",
                "radius", "height", "sensor_radius")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"geometry: {npz_path.name} is missing {missing}; present: {sorted(arrays)}")
    scalar = lambda key: float(np.asarray(arrays[key]).reshape(-1)[0])
    positions = np.asarray(arrays["positions_mm"], dtype=float) / 1000.0
    mpmt_id = np.asarray(arrays["mpmt_id"], dtype=np.int64)
    module_ids = np.unique(mpmt_id)
    module_index = np.searchsorted(module_ids, mpmt_id).astype(np.int64)
    centres = np.stack([positions[module_index == i].mean(axis=0) for i in range(module_ids.size)])
    name = name or (f"synthetic schema fixture (NOT WCTE) {npz_path.name}" if synthetic
                    else f"WCTE LUCiD {LUCID_PIN[:8]} {WCTE_CONFIG_RELPATH}")
    return DetectorGeometry(
        name=name,
        positions_m=positions, directions=np.asarray(arrays["directions"], dtype=float),
        surfaces=_surface_array(arrays["surfaces"], surface_map),
        pmt_id=np.asarray(arrays["pmt_id"], dtype=np.int64), mpmt_id=mpmt_id,
        pmt_id_in_mpmt=np.asarray(arrays["pmt_id_in_mpmt"], dtype=np.int64),
        module_index=module_index, module_ids=module_ids, module_centres_m=centres,
        radius_m=scalar("radius"), height_m=scalar("height"),
        sensor_radius_m=scalar("sensor_radius"), source_path=str(npz_path),
        source_sha256=npz_sha, config_sha256=config_sha, synthetic=bool(synthetic))

# ======================================================================= photon-bomb generator
NS_EVENT = "lucid/event"
#: Placeholder yields: the runbook fixes their *definitions* (about 200 hit PMTs at
#: Y_lo; the lower of 300 PE median module charge / 0.95 median hit fraction at Y_hi)
#: but the values come from the §3.5.3a detectability check on development data.
#: Pre-check placeholder range (what §3.5.3a starts from and raises).
Y_LO_PHOTONS_PLACEHOLDER, Y_HI_PHOTONS_PLACEHOLDER = 6000.0, 320_000.0
#: In force after the §3.5.3a detectability check of 2026-10-17 on the real WCTE geometry
#: (Y_lo at ~200 median hit PMTs; Y_hi at the 0.95 hit-fraction bound; fallback step 1).
Y_LO_PHOTONS, Y_HI_PHOTONS = 6298.436828, 539_433.964527
#: Placeholder before §3.5.3a has run; "frozen_by_detectability_check" afterwards.  Both
#: are valid states of the generator contract; the frozen state must carry the decision record.
YIELD_RANGE_STATUS = "placeholder_pending_detectability_check"
FROZEN_YIELD_RANGE_STATUS = "frozen_by_detectability_check"
VALID_YIELD_RANGE_STATUSES = (YIELD_RANGE_STATUS, FROZEN_YIELD_RANGE_STATUS)
#: The status in force for manifests and benchmarks (the frozen check has been run).
YIELD_RANGE_STATUS_IN_FORCE = FROZEN_YIELD_RANGE_STATUS
#: The frozen 2026-10-17 range predates the R-6 dark-on readout, so it is *superseded*:
#: recorded everywhere, re-frozen only after the R-6 §3.5.3a re-run.
YIELD_RANGE_DECISION_STATUS_IN_FORCE = SUPERSEDED_PENDING_R6_FREEZE
CPU_CAP_WALL_HOURS = 48.0
BENCHMARK_CORES = 16
#: §3.5.3 volume enumerated for the 48 h projection (see WP4B_GENERATOR.md).
REGISTERED_VOLUME_EVENTS = 1_070_000
REGISTERED_VOLUME_BREAKDOWN = {"main_grid_train_3x100k": 300_000,
                               "secondary_train_5x100k": 500_000,
                               "dev_train_100k": 100_000,
                               "probe_and_robustness_17x10k": 170_000}

_readout_primitive = None

def leading_edge_time(charges, threshold_mv, **kw):
    """The walk primitive from ``noise_module_lucid.response`` (imported, not forked)."""
    global _readout_primitive
    if _readout_primitive is None:  # pragma: no cover - import cost only
        from noise_module_lucid.response import leading_edge_time as primitive
        _readout_primitive = primitive
    return _readout_primitive(charges, threshold_mv, **kw)

# ======================================================================= fixed detector response
#: The merged PMT reference response (PMT R-6): path + hash bound at the top of the module.


@dataclass(frozen=True)
class ReferenceResponse:
    """The fixed detector response shared by the measured readout and the nominal twin.

    It is detector response, not injected drift: it is present in the measured readout
    **and** in the nominal twin, which therefore cancels only the injected F1/F3 (and
    F2/F2'/R11b/T) gains.  Every number is reproduced from the manifest, never a
    literal: ``factor = base + odd_group_delta * (g % 2)`` on the **dense** module
    index ``g`` (``response_manifest.module_groups(mpmt_id)`` over the sparse ids), and
    the R-6 dark-count settings (rate, window, SPE sigma, threshold) are bound from the
    same manifest.
    """

    path: str
    relpath: str
    sha256: str
    schema: str
    status: str
    gain_rule: dict
    timing_width_rule: dict
    dark_counts: str
    parameters: dict
    dark_stage_owner: str = ""

    @property
    def dark_rate_hz(self) -> float:
        return float(self.parameters["dark_rate_hz"])

    @property
    def readout_window_ns(self) -> tuple:
        return tuple(float(v) for v in self.parameters["readout_window_ns"])

    @property
    def window_ns(self) -> float:
        start, end = self.readout_window_ns
        return float(end - start)

    @property
    def spe_sigma(self) -> float:
        return float(self.parameters["spe_sigma"])

    @property
    def threshold_pe(self) -> float:
        return float(self.parameters["threshold_pe"])

    def gain(self, module_index) -> np.ndarray:
        """Per-PMT fixed charge gain on the dense module index."""
        return _parity_factor(self.gain_rule, module_index)

    def timing_width_factor(self, module_index) -> np.ndarray:
        """Per-PMT digit Gaussian timing-width factor on the dense module index."""
        return _parity_factor(self.timing_width_rule, module_index)

    @property
    def provenance(self) -> dict:
        return {"path": str(self.path), "relpath": self.relpath, "sha256": self.sha256,
                "schema": self.schema, "status": self.status,
                "dark_counts": self.dark_counts, "dark_rate_hz": self.dark_rate_hz,
                "readout_window_ns": list(self.readout_window_ns),
                "window_ns": self.window_ns, "spe_sigma": self.spe_sigma,
                "threshold_pe": self.threshold_pe,
                "v_dark": round(dark_charge_variance(self.dark_rate_hz, self.window_ns,
                                                     self.spe_sigma), 12),
                "dark_stage_owner": self.dark_stage_owner,
                "gain_rule": dict(self.gain_rule),
                "timing_width_factor_rule": dict(self.timing_width_rule)}


def _parity_factor(rule, module_index) -> np.ndarray:
    """``base + odd_group_delta * (g % 2)``, the merged per-module rule."""
    rule = dict(rule)
    if rule.get("rule") != "group_parity":
        raise ValueError(f"unknown per-module rule {rule.get('rule')!r}")
    group = np.asarray(module_index, dtype=np.int64)
    if group.ndim != 1 or np.any(group < 0):
        raise ValueError("one non-negative dense module index per PMT required")
    return float(rule["base"]) + float(rule["odd_group_delta"]) * (group % 2)


def dense_module_index(geometry) -> np.ndarray:
    """Dense 0..n_modules-1 module index per PMT; never a raw (sparse) ``mpmt_id``."""
    index = np.asarray(geometry.module_of_pmt, dtype=np.int64)
    n_pmt, n_modules = int(geometry.n_pmt), int(geometry.n_modules)
    if index.shape != (n_pmt,) or index.size == 0 or index.min() < 0 or index.max() >= n_modules:
        raise ValueError("geometry must give a dense module index per PMT")
    return index


def dense_module_index_from_mpmt_id(mpmt_id) -> np.ndarray:
    """Dense index from raw module ids through the merged manifest helper.

    ``response_manifest.module_groups`` takes ``np.unique(..., return_inverse=True)``,
    so the sparse WCTE ids (0..105 with id 45 absent) map to dense 0..104.
    """
    from noise_module_lucid.response_manifest import module_groups
    return module_groups(mpmt_id)


@lru_cache(maxsize=8)
def _reference_response_cached(path: str) -> ReferenceResponse:
    from noise_module_lucid.response_manifest import load_manifest
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"fixed response not found: {resolved}")
    manifest = load_manifest(resolved)  # schema + every recorded parameter value hash
    values = {name: entry["value"] for name, entry in manifest["parameters"].items()}
    try:
        relpath = str(resolved.resolve().relative_to(_repo_root()))
    except ValueError:  # pragma: no cover - response outside the repo (tests)
        relpath = str(resolved)
    stage_owner = str(manifest.get("stage_owners", {}).get("dark_counts", ""))
    # R-6 declares dark counts as a *stage*, not an on/off flag: the rate is the truth.
    dark_counts = DARK_COUNTS_ON if float(values["dark_rate_hz"]) > 0.0 else DARK_COUNTS_OFF
    if dark_counts == DARK_COUNTS_ON and not stage_owner:
        raise ValueError(f"{resolved}: dark_rate_hz > 0 without a dark_counts stage owner")
    return ReferenceResponse(
        path=str(resolved.resolve()), relpath=relpath,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(), schema=manifest["schema"],
        status=manifest["status"], gain_rule=dict(values["gain"]),
        timing_width_rule=dict(values["timing_width_factor"]),
        dark_counts=dark_counts, parameters=values, dark_stage_owner=stage_owner)


def reference_response(path: Path | str | None = None) -> ReferenceResponse:
    """Read the fixed response once per path (cached; validated against the manifest)."""
    return _reference_response_cached(str(resolve_reference_path(path)))


def generator_reference(cfg: "GeneratorConfig") -> ReferenceResponse:
    """The response a generator config is bound to (path recorded in the config)."""
    return reference_response(cfg.reference_response_path)


def reference_provenance(cfg: "GeneratorConfig" | None = None) -> dict:
    """Source path + sha256 recorded by every manifest, benchmark and decision."""
    return (generator_reference(cfg) if cfg is not None else reference_response()).provenance


def fixed_charge_gain(geometry, reference: ReferenceResponse | None = None) -> np.ndarray:
    """Fixed per-PMT charge gain from the reference response (odd dense module +3%)."""
    ref = reference_response() if reference is None else reference
    return ref.gain(dense_module_index(geometry))


def fixed_timing_width_factor(geometry, reference: ReferenceResponse | None = None) -> np.ndarray:
    """Fixed per-PMT digit Gaussian width factor (odd dense module +5%)."""
    ref = reference_response() if reference is None else reference
    return ref.timing_width_factor(dense_module_index(geometry))


@dataclass(frozen=True)
class GeneratorConfig:
    """All event-generation settings; science values that are pending are parameters, not constants."""

    y_lo: float = Y_LO_PHOTONS
    y_hi: float = Y_HI_PHOTONS
    fiducial_fraction: float = 0.8
    spectrum_kind: str = "uniform_synthetic"
    spectrum_nm: tuple = (300.0, 600.0)
    qe_curve_nm: tuple = (300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0)
    qe_curve: tuple = (0.10, 0.18, 0.24, 0.25, 0.22, 0.16, 0.08)
    ce: float = 0.95
    n_wavelength_samples: int = 1
    sigma_spe: float = SIGMA_SPE
    tts_sigma_ns: float = 1.5
    #: PMT R-6 (merged): dark counts are **on**.  Rate, window and the response values are
    #: bound to the manifest (``configs/lucid_pmt/reference.json``), never typed again.
    dark_counts: str = DARK_COUNTS_ON
    dark_rate_hz: float = DARK_RATE_HZ_ON
    #: ``[start, end]`` ns, event time origin; dark hits are uniform in it and their times
    #: are measured from the window start.  The duration (``end - start``) is ``window_ns``.
    readout_window_ns: tuple = R6_READOUT_WINDOW_NS
    threshold_pe: float = R6_THRESHOLD_PE
    walk_calibration_pe: float = R6_THRESHOLD_PE
    mv_per_pe: float = 4.0
    n_group: float = 1.33
    lambda0_m: float = LAMBDA0_M
    twin: bool = True
    spectrum_label: str = "synthetic_uniform_300_600_nm (PMT reference stand-in)"
    #: The fixed detector response is bound by path + hash (both recorded in manifests).
    reference_response_path: str = FIXED_RESPONSE_RELPATH
    reference_response_sha256: str = REFERENCE_RESPONSE_SHA256

    @property
    def window_ns(self) -> float:
        """Readout-window duration ``end - start`` (1 us for R-6), derived not stored."""
        start, end = (float(v) for v in self.readout_window_ns)
        return float(end - start)

    def __post_init__(self):
        object.__setattr__(self, "readout_window_ns",
                           tuple(float(v) for v in self.readout_window_ns))
        pairs = list(zip(self.qe_curve_nm, self.qe_curve))
        window = self.readout_window_ns
        checks = [("y_lo", self.y_lo > 0), ("y_hi > y_lo", self.y_hi > self.y_lo),
                  ("0 < fiducial_fraction <= 1", 0.0 < self.fiducial_fraction <= 1.0),
                  ("qe curve lengths", len(self.qe_curve_nm) == len(self.qe_curve) >= 2),
                  ("qe curve wavelengths increasing", all(a[0] < b[0] for a, b in zip(pairs, pairs[1:]))),
                  ("qe in [0, 1]", all(0.0 <= q <= 1.0 for q in self.qe_curve)),
                  ("0 < ce <= 1", 0.0 < self.ce <= 1.0),
                  ("n_wavelength_samples >= 1", int(self.n_wavelength_samples) >= 1),
                  ("sigma_spe >= 0", self.sigma_spe >= 0.0),
                  ("tts_sigma_ns >= 0", self.tts_sigma_ns >= 0.0),
                  ("dark_rate_hz >= 0", self.dark_rate_hz >= 0.0),
                  ("dark_counts on", self.dark_counts == DARK_COUNTS_ON),
                  ("dark on needs a positive rate",
                   self.dark_counts != DARK_COUNTS_ON or float(self.dark_rate_hz) > 0.0),
                  ("readout_window_ns is a finite [start, end] pair",
                   len(window) == 2 and all(np.isfinite(window)) and window[0] < window[1]),
                  ("window_ns > 0", self.window_ns > 0.0),
                  ("threshold_pe > 0", self.threshold_pe > 0.0),
                  ("walk_calibration_pe > 0", self.walk_calibration_pe > 0.0),
                  ("mv_per_pe > 0", self.mv_per_pe > 0.0),
                  ("n_group >= 1", self.n_group >= 1.0),
                  ("spectrum_kind", self.spectrum_kind == "uniform_synthetic"),
                  ("spectrum_nm", 0.0 < self.spectrum_nm[0] < self.spectrum_nm[1])]
        bad = [name for name, ok in checks if not ok]
        if bad:
            raise ValueError(f"generator config invalid: {bad}")
        if float(self.lambda0_m) != float(LAMBDA0_M):
            raise ValueError(f"lambda0_m is frozen at the module constant {LAMBDA0_M} m pending "
                             "§3.5.3a; change both together")
        fixed_path = resolve_reference_path(self.reference_response_path)
        if not fixed_path.is_file():
            raise ValueError(f"generator config: fixed response {fixed_path} not found")
        if _sha(fixed_path.read_bytes()) != str(self.reference_response_sha256):
            raise ValueError(f"generator config: fixed response sha256 does not match "
                             f"{fixed_path}; re-read the file instead of editing the hash")
        # R-6 provenance: the dark stage and its window are the manifest's, not this file's.
        ref = reference_response(fixed_path)
        if ref.dark_counts != self.dark_counts:
            raise ValueError(f"generator config: dark_counts {self.dark_counts!r} but the "
                             f"manifest declares {ref.dark_counts!r}")
        if float(self.dark_rate_hz) != ref.dark_rate_hz:
            raise ValueError(f"generator config: dark_rate_hz {self.dark_rate_hz!r} != the "
                             f"manifest's {ref.dark_rate_hz!r}")
        if tuple(self.readout_window_ns) != tuple(ref.readout_window_ns):
            raise ValueError(f"generator config: readout_window_ns {self.readout_window_ns!r} "
                             f"!= the manifest's {ref.readout_window_ns!r}")

    @property
    def digest(self) -> str:
        fields = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return _sha(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())

def load_generator_config(path: Path | str) -> GeneratorConfig:
    """Read the ``generator`` block of ``lucid_bombs_v0.json``."""
    payload = json.loads(Path(path).read_text())
    block = payload.get("generator")
    if not isinstance(block, dict):
        raise ValueError(f"{path}: missing the 'generator' block")
    kwargs = dict(block)
    status = kwargs.pop("yield_range_status", YIELD_RANGE_STATUS)
    if status not in VALID_YIELD_RANGE_STATUSES:
        raise ValueError(
            f"config: yield_range_status {status!r} not in "
            f"{VALID_YIELD_RANGE_STATUSES!r}"
        )
    decision_block = kwargs.pop("yield_range_decision", None)
    if status == FROZEN_YIELD_RANGE_STATUS and not decision_block:
        raise ValueError(
            "config: a frozen yield range must carry a yield_range_decision block"
        )
    if decision_block is not None and not decision_block.get("status"):
        raise ValueError("config: the yield_range_decision block must record its status")
    allowed = set(GeneratorConfig.__dataclass_fields__) | {"window_ns"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(f"config: unknown generator keys {unknown}")
    kwargs["qe_curve_nm"] = tuple(kwargs["qe_curve_nm"])
    kwargs["qe_curve"] = tuple(kwargs["qe_curve"])
    kwargs["spectrum_nm"] = tuple(kwargs["spectrum_nm"])
    #: ``window_ns`` is the derived duration ``end - start``; the config records the pair.
    if "window_ns" in kwargs:
        recorded_window = float(kwargs.pop("window_ns"))
        derived = (float(kwargs["readout_window_ns"][1])
                   - float(kwargs["readout_window_ns"][0]))
        if abs(recorded_window - derived) > 1e-9:
            raise ValueError(f"config: generator.window_ns {recorded_window} != the "
                             f"readout_window_ns duration {derived}")
    kwargs["readout_window_ns"] = tuple(kwargs["readout_window_ns"])
    if payload.get("lambda0_m") is not None and float(payload["lambda0_m"]) != float(kwargs["lambda0_m"]):
        raise ValueError("config: top-level lambda0_m and generator.lambda0_m disagree")
    if payload.get("readout_label") != READOUT_LABEL:
        raise ValueError(f"config: readout_label {payload.get('readout_label')!r} != "
                         f"{READOUT_LABEL!r} (the merged R-6 readout has dark counts on)")
    if payload.get("dark_counts") != DARK_COUNTS_ON:
        raise ValueError(f"config: dark_counts {payload.get('dark_counts')!r} must be "
                         f"{DARK_COUNTS_ON!r} (PMT R-6)")
    validate_time_rule(payload.get("time_rule"))
    cfg = GeneratorConfig(**kwargs)
    #: The response values are bound from the manifest, not copied into the config.
    manifest = generator_reference(cfg)
    for config_name, manifest_name in (("sigma_spe", "spe_sigma"),
                                       ("threshold_pe", "threshold_pe")):
        if float(getattr(cfg, config_name)) != float(getattr(manifest, manifest_name)):
            raise ValueError(f"config: generator.{config_name} {getattr(cfg, config_name)!r} != "
                             f"the manifest's {getattr(manifest, manifest_name)!r}")
    fixed = payload.get("fixed_response")
    if fixed is not None:
        ref = manifest
        if not fixed.get("path") or \
                resolve_reference_path(fixed["path"]).resolve() != Path(ref.path):
            raise ValueError("config: fixed_response.path does not name the recorded response")
        if fixed.get("sha256") != ref.sha256:
            raise ValueError(f"config: fixed_response.sha256 {fixed.get('sha256')!r} != "
                             f"the manifest's {ref.sha256!r}")
        if fixed.get("gain") != ref.gain_rule or \
                fixed.get("timing_width_factor") != ref.timing_width_rule:
            raise ValueError("config: fixed_response rules disagree with the manifest")
        if fixed.get("dark_counts") != DARK_COUNTS_ON:
            raise ValueError(f"config: fixed_response.dark_counts must be {DARK_COUNTS_ON!r}")
        if fixed.get("readout_window_ns") not in (None, list(ref.readout_window_ns)):
            raise ValueError("config: fixed_response.readout_window_ns disagrees with the manifest")
    return cfg

def qe_at(wavelength_nm, cfg: GeneratorConfig) -> np.ndarray:
    """Interpolate the (synthetic) QE curve; wavelengths outside the table are clipped."""
    w = np.asarray(wavelength_nm, dtype=float)
    return np.interp(w, np.asarray(cfg.qe_curve_nm, float), np.asarray(cfg.qe_curve, float))

def sample_qe(geometry, rng: np.random.Generator, cfg: GeneratorConfig) -> np.ndarray:
    """Per-PMT effective QE from the config spectrum (uniform synthetic stand-in)."""
    lo, hi = cfg.spectrum_nm
    total = np.zeros(geometry.n_pmt)
    for _ in range(int(cfg.n_wavelength_samples)):
        total += qe_at(rng.uniform(lo, hi, geometry.n_pmt), cfg)
    return total / float(cfg.n_wavelength_samples)

def fiducial_vertex(geometry, rng: np.random.Generator, fraction: float = 0.8) -> np.ndarray:
    """Uniform-in-volume vertex in the 80% fiducial cylinder."""
    radius = float(fraction) * geometry.radius_m
    half_height = float(fraction) * geometry.half_height_m
    r = radius * np.sqrt(rng.random())
    phi = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([r * np.cos(phi), r * np.sin(phi), rng.uniform(-half_height, half_height)])

def solid_angle_fraction(geometry, vertex_m, sensor_radius_m: float | None = None) -> np.ndarray:
    """Disc solid angle / 4π per sensor.

    Approximation: the exact on-axis disc solid angle ``2π(1 - d/√(d²+r²))`` times the
    obliquity ``|cosθ|``, i.e. ``π r² cosθ / d²`` for ``d >> r``.  The tilt correction is
    O((r/d)²); with the vertex inside the 80% fiducial, ``d >= 0.2 R = 0.315 m`` and
    ``r = 0.04 m``, so the error is below ~2%.
    """
    delta = geometry.positions_m - np.asarray(vertex_m, dtype=float)
    dist = np.linalg.norm(delta, axis=1)
    if np.any(dist <= 0.0):
        raise ValueError("expected charge: a sensor coincides with the vertex")
    r = float(geometry.sensor_radius_m if sensor_radius_m is None else sensor_radius_m)
    cos = np.abs(np.einsum("ij,ij->i", delta, geometry.directions)) / dist
    omega = 2.0 * np.pi * (1.0 - dist / np.sqrt(dist * dist + r * r)) * np.clip(cos, 0.0, 1.0)
    return omega / (4.0 * np.pi)

def expected_charge(geometry, vertex_m, yield_photons, qe_eff, lambda_m, cfg: GeneratorConfig,
                    *, sensor_radius_m: float | None = None) -> np.ndarray:
    """``mu_p = Y * QE(lambda) * CE * Omega_p/(4pi) * exp(-d_p/lambda)``."""
    lam = float(lambda_m)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("expected charge: attenuation length must be positive and finite")
    if not np.isfinite(yield_photons) or yield_photons < 0.0:
        raise ValueError("expected charge: yield must be finite and non-negative")
    dist = np.linalg.norm(geometry.positions_m - np.asarray(vertex_m, dtype=float), axis=1)
    omega = solid_angle_fraction(geometry, vertex_m, sensor_radius_m)
    return (float(yield_photons) * np.clip(np.asarray(qe_eff, float), 0.0, 1.0) * float(cfg.ce)
            * omega * np.exp(-dist / lam))

def propagation_time_ns(geometry, vertex_m, cfg: GeneratorConfig) -> np.ndarray:
    """Prompt direct-light time from the vertex, in ns (group velocity ``c / n_group``)."""
    dist = np.linalg.norm(geometry.positions_m - np.asarray(vertex_m, dtype=float), axis=1)
    return dist * (float(cfg.n_group) / C_M_PER_S) * 1e9

def count_capacity(max_mu: float, *, n_sigma: float = 12.0) -> int:
    """Uniform columns needed by :func:`poisson_counts_from_uniform` (12σ + 10 margin)."""
    m = float(max_mu)
    if not np.isfinite(m) or m < 0.0:
        raise ValueError("count capacity needs a finite non-negative mean")
    return int(np.ceil(m + n_sigma * np.sqrt(m + 1.0) + 10.0))

def poisson_counts_from_uniform(poisson_u, mu) -> np.ndarray:
    """Exact Poisson draw from shared uniforms: arrivals of a unit-rate process.

    ``counts_i = #{k : sum_{j<=k} -ln u_ij <= mu_i}``.  The *same* ``poisson_u`` mapped
    through two different ``mu`` vectors gives correlated counts, which is how the nominal
    twin shares the photon draw across a different attenuation length.
    """
    u = np.asarray(poisson_u, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if u.ndim != 2 or mu.shape != (u.shape[0],):
        raise ValueError("poisson_counts_from_uniform needs u (n_pmt, n_cap) and mu (n_pmt,)")
    if not np.isfinite(mu).all() or np.any(mu < 0.0):
        raise ValueError("poisson counts need finite non-negative means")
    tiny = np.finfo(float).tiny
    arrivals = np.cumsum(-np.log(np.clip(u, tiny, 1.0)), axis=1)
    return np.count_nonzero(arrivals <= mu[:, None], axis=1).astype(np.int64)

@dataclass(frozen=True, eq=False)
class SensorDraws:
    """One event's shared draws: the twin re-maps these through the nominal state."""

    key: str
    run_id: str
    event_id: int
    vertex_m: np.ndarray        # (3,)
    yield_photons: float
    qe_eff: np.ndarray          # (n_pmt,)
    poisson_u: np.ndarray       # (n_pmt, n_cap)
    charge_z: np.ndarray        # (n_pmt,)
    transit_z: np.ndarray       # (n_pmt,)
    dark_counts: np.ndarray     # (n_pmt,), keyed Poisson dark photoelectrons per PMT
    #: (n_dark_total,) ns, event time origin: uniform in ``readout_window_ns`` and ordered
    #: like ``np.repeat(np.arange(n_pmt), dark_counts)``.  Shared by the nominal twin.
    dark_times: np.ndarray
    window_ns: float
    capacity: int
    #: A9 truth: per-event emission-time offset in [-50, +50] ns (keyed RNG).  It shifts every
    #: **source photon** arrival time (never a dark time) and is never a feature; the readout
    #: token time is trigger-relative, so the offset cancels.
    emission_offset_ns: float = 0.0


def dark_photoelectrons(geometry, cfg: GeneratorConfig, rng_counts, rng_times) -> tuple:
    """Keyed PMT R-6 dark bank: Poisson(rate x duration) per PMT, times uniform in the window.

    ``rate x duration = 1000 Hz x 1 us = 0.001`` per PMT, so a zero count is common and a
    WCTE-sized event carries **~2 dark hits in total** (1995 x 0.001).  Dark photoelectrons
    bypass QE/CE; they join the detected PE before digitisation.  Times are measured from
    the window start and returned as absolute event times in ``[start, end]``.
    """
    start, end = (float(v) for v in cfg.readout_window_ns)
    mean = float(cfg.dark_rate_hz) * (end - start) * 1e-9
    if mean < 0.0 or not np.isfinite(mean):
        raise ValueError("dark count mean must be finite and non-negative")
    from scipy.stats import poisson
    # The keyed uniform can round to exactly 1.0 (p = 2**-53), where the Poisson ppf is inf.
    uniform = np.minimum(np.asarray(rng_counts.random(geometry.n_pmt), dtype=float),
                         np.nextafter(1.0, 0.0))
    counts = poisson.ppf(uniform, mean).astype(np.int64)
    total = int(counts.sum())
    times = start + (end - start) * np.asarray(rng_times.random(total), dtype=float)
    return counts, times


def draw_event(geometry, cfg: GeneratorConfig, run_id: str, event_id: int,
               seed: int = DEFAULT_SEED) -> SensorDraws:
    """Vertex, log-uniform yield, spectrum, and every shared normal/Poisson draw.

    PMT R-6 turns the dark bank back on: this consumes the keyed ``dark`` (per-PMT
    Poisson) and ``dark_time`` (uniform-in-window) namespaces and returns real
    ``dark_counts`` and ``dark_times``.  On a WCTE-sized geometry (~1995 PMTs) the
    expected number of dark hits per event is **~2** (1995 x 1000 Hz x 1 us), with a
    zero count on most PMTs; a small fixture scales with its PMT count.

    A9: the event also draws an **emission-time offset** uniform in [-50, +50] ns from its
    own ``emission_offset`` keyed role (truth only, recorded in ``events.npz``).
    """
    if cfg.dark_counts != DARK_COUNTS_ON:
        raise ValueError(f"{READOUT_LABEL}: dark counts must be on, got {cfg.dark_counts!r}")

    def rng(role):
        return keyed_rng(seed, NS_DRAWS, run_id, event_id, role)

    vertex = fiducial_vertex(geometry, rng("vertex"), cfg.fiducial_fraction)
    yield_photons = float(np.exp(rng("yield").uniform(np.log(cfg.y_lo), np.log(cfg.y_hi))))
    qe_eff = sample_qe(geometry, rng("wavelength"), cfg)
    bound = expected_charge(geometry, vertex, yield_photons, qe_eff, UNATTENUATED_M, cfg)
    capacity = count_capacity(float(bound.max()))
    poisson_u = np.clip(rng("poisson").random((geometry.n_pmt, capacity)),
                        np.finfo(float).tiny, 1.0)
    dark_counts, dark_times = dark_photoelectrons(geometry, cfg, rng("dark"), rng("dark_time"))
    emission_offset_ns = float(rng(NS_EMISSION_OFFSET).uniform(*EMISSION_OFFSET_NS))
    return SensorDraws(key=f"{NS_DRAWS}/{run_id}/{event_id}", run_id=run_id, event_id=int(event_id),
                       vertex_m=vertex, yield_photons=yield_photons, qe_eff=qe_eff,
                       poisson_u=poisson_u, charge_z=rng("charge").standard_normal(geometry.n_pmt),
                       transit_z=rng("transit").standard_normal(geometry.n_pmt),
                       dark_counts=dark_counts, dark_times=dark_times,
                       window_ns=float(cfg.window_ns), capacity=capacity,
                       emission_offset_ns=emission_offset_ns)

def readout_synthetic_unrepaired(state: DetectorState, draws: SensorDraws, geometry,
                                 cfg: GeneratorConfig,
                                 reference: ReferenceResponse | None = None, *,
                                 accounting: ReadoutRecord | None = None) -> HitReadout:
    """The unrepaired PMT chain with the merged fixed R-6 response and dark counts on.

    QE/CE -> Poisson source PE (mean carries the fixed per-module parity gain) **joined
    with the keyed dark PE before digitisation** -> SPE smear -> injected gain ->
    threshold+walk.  Per the merged manifest, dark photoelectrons bypass QE/CE and then
    get the *same* SPE charge smearing, the injected gains, the fixed parity gain, the
    discriminator and the T-Q walk/width as every other photoelectron; their times are
    uniform in the readout window.  Because one digit is recorded per PMT, a dark PE
    earlier than the source light sets that digit's time, and a dark PE can make an
    otherwise-unhit PMT fire.  Injected gains act on the charge **before** the fixed-PE
    threshold, so a gain change can move a PMT between hit and unhit.  The fixed response
    is applied here, not in :class:`DetectorState`, so the nominal twin keeps both the
    parity charge gain and the timing-width factor and cancels only the injected F1/F3
    (and F2/F2'/R11b/T) gains; the twin also shares the same dark draw.

    The digit Gaussian width is ``tts_sigma_ns * timing_width_factor`` per PMT (odd
    dense module +5%).  The walk curve is ``response.leading_edge_time`` at the actual
    threshold, corrected by the same curve at the training-time calibration threshold;
    with ``threshold_pe == walk_calibration_pe`` the residual is exactly zero.

    **A9 time convention.**  The source photon arrival time is
    ``d*n_group/c + emission_offset_ns`` (the truth offset shifts every source photon, never a
    dark time).  After digitisation, the trigger is
    ``nhits_trigger_time(digit_times, n=NHITS_N, window_ns=NHITS_WINDOW_NS)`` imported from
    the PMT readout; ``None`` falls back to the readout-window start (counted).  The token
    time is ``t_token = t_digit - t_trigger`` and digits outside the imported prompt window
    (default [-20, +60] ns) are **dropped and counted**.  ``HitReadout.time_ns`` holds the
    tokens; ``HitReadout.absolute_time_ns`` keeps the same digits' absolute times additively.
    No truth (vertex, emission offset) enters the token computation.
    """
    if cfg.dark_counts != DARK_COUNTS_ON:
        raise ValueError(f"{READOUT_LABEL}: dark counts must be on, got {cfg.dark_counts!r}")
    (nhits_trigger_time, prompt_window_mask, nhits_n, nhits_window_ns,
     prompt_window_ns) = pmt_time_rule()
    n_dark = np.asarray(draws.dark_counts, dtype=np.int64)
    if n_dark.shape != (geometry.n_pmt,) or np.any(n_dark < 0):
        raise ValueError("dark_counts must be one non-negative count per PMT")
    dark_sensor = np.repeat(np.arange(geometry.n_pmt, dtype=np.int64), n_dark)
    dark_times = np.asarray(draws.dark_times, dtype=float)
    if dark_times.shape != (dark_sensor.size,) or not np.all(np.isfinite(dark_times)):
        raise ValueError("dark_times must be one finite time per dark photoelectron")
    emission_offset_ns = float(draws.emission_offset_ns)
    if not np.isfinite(emission_offset_ns):
        raise ValueError("emission_offset_ns must be finite (it is truth, not a feature)")
    start, end = (float(v) for v in cfg.readout_window_ns)
    if dark_times.size and (dark_times.min() < start - 1e-9 or dark_times.max() > end + 1e-9):
        raise ValueError("dark times must lie in the readout window")
    ref = generator_reference(cfg) if reference is None else reference
    mu = (expected_charge(geometry, draws.vertex_m, draws.yield_photons, draws.qe_eff,
                          state.attenuation_length_m, cfg)
          * fixed_charge_gain(geometry, ref))
    n_photon = poisson_counts_from_uniform(draws.poisson_u, mu)
    if int(n_photon.max()) >= int(draws.capacity):
        raise RuntimeError("photon count reached the shared-draw capacity; enlarge count_capacity")
    # Dark PE join the detected PE *before* digitisation (R-6): the same SPE charge smear
    # acts on the total count, so a dark PE adds charge like any other photoelectron.
    n = n_photon + n_dark
    charge = np.clip(n + np.sqrt(n) * float(cfg.sigma_spe) * draws.charge_z, 0.0, None)
    q = charge * np.asarray(state.gains, dtype=float)
    thr_mv = float(cfg.threshold_pe) * float(cfg.mv_per_pe)
    cal_mv = float(cfg.walk_calibration_pe) * float(cfg.mv_per_pe)
    walk = leading_edge_time(q, thr_mv, mv_per_pe=float(cfg.mv_per_pe))
    walk_cal = leading_edge_time(q, cal_mv, mv_per_pe=float(cfg.mv_per_pe))
    fired = np.isfinite(walk)
    residual = np.where(np.isfinite(walk_cal), walk - np.where(np.isfinite(walk_cal), walk_cal, 0.0), 0.0)
    sigma_t = float(cfg.tts_sigma_ns) * fixed_timing_width_factor(geometry, ref)
    # Earliest *photoelectron* arrival per PMT: the prompt source light (only where a source
    # PE was detected, shifted by the truth emission offset) or an earlier dark hit; the digit
    # width/jitter is applied afterwards.  Dark times are not shifted by the emission offset.
    first = np.where(n_photon > 0,
                     propagation_time_ns(geometry, draws.vertex_m, cfg) + emission_offset_ns,
                     np.inf)
    dark_first = np.full(geometry.n_pmt, np.inf)
    if dark_sensor.size:
        np.minimum.at(dark_first, dark_sensor, dark_times)
    first = np.where(np.isfinite(dark_first), np.minimum(first, dark_first), first)
    times = first + sigma_t * draws.transit_z + residual
    index = np.flatnonzero(fired)
    digit_times = times[index]  # absolute readout digit times, before any time-origin change
    # A9: the trigger comes from the imported PMT rule, on the digit times only (no truth).
    trigger = nhits_trigger_time(digit_times, n=nhits_n, window_ns=nhits_window_ns)
    # The author's invariance refinement: with dark counts on, the trigger must be identical on
    # the source-derived digits alone; when a dark coincidence moves it, count it and report the
    # rate.  Source-derived digits use only the prompt light (n_photon > 0) through the same
    # width/jitter chain, so a constant photon-time shift moves them exactly.
    source_trigger = trigger  # no dark digit exists: the source trigger is the trigger
    if dark_sensor.size:
        source_first = np.where(n_photon > 0,
                                propagation_time_ns(geometry, draws.vertex_m, cfg) + emission_offset_ns,
                                np.inf)
        source_times = source_first + sigma_t * draws.transit_z + residual
        # A dark-only digit has no source time: the counterfactual uses the source-digit PMTs
        # only, never an infinite time (the imported rule requires finite digit times).
        source_digits = source_times[index[n_photon[index] > 0]]
        source_trigger = nhits_trigger_time(source_digits, n=nhits_n, window_ns=nhits_window_ns)
        if source_trigger is not None:
            moved = trigger is None or abs(float(trigger) - float(source_trigger)) > 1e-9
            if moved and accounting is not None:
                accounting.n_dark_moved_trigger += 1
    if trigger is not None and not np.isfinite(trigger):
        raise ValueError("nhits_trigger_time returned a non-finite trigger")
    if trigger is None:
        # A9 fallback: the readout-window start is the origin.  Counted, never silent.
        origin = start
        if accounting is not None:
            accounting.n_no_trigger += 1
    else:
        origin = float(trigger)
    token = digit_times - origin
    # A9: the prompt-window cut is the **imported** mask (it keeps every digit when there is no
    # trigger and validates finite digit times); never reimplemented here.
    keep = np.asarray(prompt_window_mask(digit_times, None if trigger is None else float(trigger)))
    if accounting is not None:
        accounting.n_events += 1
        accounting.n_prompt_dropped += int(np.count_nonzero(~keep))
    kept = index[keep]
    return HitReadout(kept, q[kept], token[keep], absolute_time_ns=digit_times[keep],
                      trigger_ns=(None if trigger is None else float(trigger)),
                      source_trigger_ns=(None if source_trigger is None
                                         else float(source_trigger)))

def make_readout(geometry, cfg: GeneratorConfig, accounting: ReadoutRecord | None = None) -> ReadoutCallable:
    """Bind the real readout to a geometry/config, for :func:`nominal_twin`.

    ``accounting`` is the run's :class:`ReadoutRecord`; the nominal twin gets its own readout
    (without this counter) so the record describes the measured readout only.
    """
    reference = generator_reference(cfg)  # read once, then shared by every event
    def readout(state, draws):
        return readout_synthetic_unrepaired(state, draws, geometry, cfg, reference,
                                            accounting=accounting)
    readout.accounting = accounting
    return readout

# ======================================================================= generated runs/storage
@dataclass(frozen=True, eq=False)
class GeneratedRun:
    """One run: a contiguous event block sharing one detector state (and its nominal twin)."""

    run: RunSpec
    state: DetectorState
    event_ids: np.ndarray
    vertices_m: np.ndarray
    yields_photons: np.ndarray
    expected_pe: np.ndarray
    emission_offsets_ns: np.ndarray  # A9 truth, one per event; never a feature
    hits: tuple
    twin_hits: tuple | None = None
    readout_record: ReadoutRecord | None = None

    @property
    def n_events(self) -> int:
        return int(self.event_ids.size)

    @property
    def n_hits(self) -> int:
        return int(sum(len(h) for h in self.hits))

def generate_run(geometry, cfg: GeneratorConfig, run: RunSpec, *, seed: int | None = None,
                 with_twin: bool | None = None) -> GeneratedRun:
    """Generate one run with a single detector state (a run property, §3.5.3)."""
    seed = int(run.seed if seed is None else seed)
    with_twin = bool(cfg.twin if with_twin is None else with_twin)
    state = run.state(geometry)
    reference = generator_reference(cfg)
    fixed_gain = fixed_charge_gain(geometry, reference)
    record = ReadoutRecord()
    readout = make_readout(geometry, cfg, accounting=record)
    twin_readout = make_readout(geometry, cfg)  # its own counter: the record is the measured path
    event_ids = np.arange(run.event_start, run.event_stop, dtype=np.int64)
    vertices = np.zeros((event_ids.size, 3))
    yields = np.zeros(event_ids.size)
    expected = np.zeros(event_ids.size)
    offsets = np.zeros(event_ids.size)
    hits, twin_hits = [], []
    for i, event_id in enumerate(event_ids):
        draws = draw_event(geometry, cfg, run.run_id, int(event_id), seed=seed)
        vertices[i], yields[i] = draws.vertex_m, draws.yield_photons
        offsets[i] = float(draws.emission_offset_ns)
        expected[i] = float((expected_charge(geometry, draws.vertex_m, draws.yield_photons,
                                             draws.qe_eff, state.attenuation_length_m, cfg)
                             * fixed_gain).sum())
        hits.append(readout(state, draws))
        if with_twin:
            twin_hits.append(nominal_twin(draws, geometry, twin_readout, arm=ORACLE_ARM))
    return GeneratedRun(run, state, event_ids, vertices, yields, expected, offsets,
                        tuple(hits), tuple(twin_hits) if with_twin else None, record)

def generate_events(geometry, cfg: GeneratorConfig, runs, *, seed: int | None = None,
                    with_twin: bool | None = None):
    """Iterate ``generate_run`` over a cohort; memory holds one run at a time."""
    for run in runs:
        yield generate_run(geometry, cfg, run, seed=seed, with_twin=with_twin)

def write_generated_run(directory, generated: GeneratedRun, geometry, cfg: GeneratorConfig, *,
                        seed: int | None = None, code_sha: str | None = None) -> dict:
    """Write sparse hits, the twin (if any), event truth, the manifest and ``SHA256SUMS``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    store = sparse_from_readouts(generated.event_ids, generated.hits)
    write_sparse_hits(directory / "hits.npz", store)
    if generated.twin_hits is not None:
        write_sparse_hits(directory / "twin_hits.npz",
                          sparse_from_readouts(generated.event_ids, generated.twin_hits))
    truth = directory / "events.npz"
    with open(truth, "wb") as handle:
        np.savez_compressed(handle, event_ids=generated.event_ids,
                            vertex_m=generated.vertices_m.astype(np.float64),
                            yield_photons=generated.yields_photons.astype(np.float64),
                            expected_pe=generated.expected_pe.astype(np.float32),
                            # A9 truth: the per-event emission-time offset.  Never a feature.
                            emission_offset_ns=generated.emission_offsets_ns.astype(np.float64))
    record = generated.readout_record or ReadoutRecord()
    payload = write_manifest(
        directory / "manifest.json", (generated.run,), geometry,
        seed=int(generated.run.seed if seed is None else seed), code_sha=code_sha,
        extra={"generator_config_sha256": cfg.digest,
               "geometry_source_path": geometry.source_path,
               "geometry_source_sha256": geometry.source_sha256,
               "geometry_config_sha256": geometry.config_sha256,
               "geometry_synthetic": bool(geometry.synthetic),
               "geometry_summary": geometry.summary(),
               "yield_range_photons": [float(cfg.y_lo), float(cfg.y_hi)],
               "yield_range_status": YIELD_RANGE_STATUS_IN_FORCE,
               "expected_pe_includes_fixed_response": True,
               "fiducial_fraction": float(cfg.fiducial_fraction),
               "threshold_pe": float(cfg.threshold_pe), "spectrum_label": cfg.spectrum_label,
               "state_clipped": dict(generated.state.clipped), "n_hits": generated.n_hits,
               "twin_present": generated.twin_hits is not None,
               # A9: the convention and the counted no-trigger / prompt-window drops.
               "time_rule": time_rule_record(),
               "store_time_ns": "digit - trigger (A9 token time, after the prompt-window cut)",
               "n_no_trigger": int(record.n_no_trigger),
               "n_prompt_dropped": int(record.n_prompt_dropped),
               "n_dark_moved_trigger": int(record.n_dark_moved_trigger),
               "dark_moved_trigger_rate": (float(record.n_dark_moved_trigger) / record.n_events
                                           if record.n_events else 0.0),
               "n_readout_events": int(record.n_events),
               "emission_offset_truth": "events.npz:emission_offset_ns (truth only; never a feature)",
               "event_truth_sha256": _sha(truth.read_bytes())})
    write_sha256sums(directory)
    return payload

# ======================================================================= 1000-event benchmark
def peak_rss_mb() -> float:
    """Process peak RSS in MiB (macOS reports bytes, Linux KiB)."""
    import resource
    import sys
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 ** 2) if sys.platform == "darwin" else rss / 1024.0

def _tercile_table(geometry, generated: GeneratedRun) -> list[dict]:
    """§3.5.3 per-yield-tercile physics table."""
    def module_charge(hit):
        return float(np.bincount(geometry.module_index[np.asarray(hit.pmt_id, int)],
                                 weights=hit.charge_pe, minlength=geometry.n_modules).mean())
    rows = []
    for rank, idx in enumerate(np.array_split(np.argsort(generated.yields_photons), 3)):
        hits = [generated.hits[i] for i in idx]
        counts = np.array([len(h) for h in hits], dtype=float)
        rows.append({
            "tercile": ("low", "mid", "high")[rank], "n_events": int(idx.size),
            "yield_photons_median": float(np.median(generated.yields_photons[idx])),
            "mean_hit_pmts": float(counts.mean()), "median_hit_pmts": float(np.median(counts)),
            "mean_hit_fraction": float((counts / geometry.n_pmt).mean()),
            "mean_module_charge_pe": float(np.mean([module_charge(h) for h in hits])),
            "total_charge_pe_median": float(np.median([h.charge_pe.sum() for h in hits]))})
    return rows

def benchmark_generation(geometry, cfg: GeneratorConfig, *, n_events: int = 1000,
                         seed: int = DEFAULT_SEED, with_twin: bool | None = None,
                         law: str = "F1", setting: float = 0.05) -> dict:
    """Time/memory for ``n_events`` and the §3.5.3 per-yield-tercile table."""
    import time
    with_twin = bool(cfg.twin if with_twin is None else with_twin)
    run = RunSpec(f"benchmark/{law}/{setting:g}", "benchmark", law, float(setting), 0,
                  int(n_events), int(seed))
    t0 = time.perf_counter()
    generated = generate_run(geometry, cfg, run, with_twin=with_twin)
    seconds = time.perf_counter() - t0
    stores = [sparse_from_readouts(generated.event_ids, generated.hits)]
    if generated.twin_hits is not None:
        stores.append(sparse_from_readouts(generated.event_ids, generated.twin_hits))
    bytes_hits = int(sum(a.nbytes for st in stores
                         for a in (st.event_ids, st.offsets, st.pmt_id, st.charge_pe, st.time_ns)))
    record = generated.readout_record or ReadoutRecord()
    rate = float(generated.n_events) / seconds if seconds > 0 else float("inf")
    hours_1 = REGISTERED_VOLUME_EVENTS / rate / 3600.0 if rate else float("inf")
    return {"benchmark_events": int(generated.n_events), "with_twin": with_twin,
            "seconds": seconds, "events_per_s": rate, "peak_rss_mb": peak_rss_mb(),
            "sparse_store_mib": bytes_hits / (1024.0 ** 2),
            "hits_per_event": float(generated.n_hits) / max(generated.n_events, 1),
            "is_synthetic_geometry": bool(geometry.synthetic),
            "geometry_name": geometry.name, "geometry_sha256": geometry.digest,
            "readout_label": READOUT_LABEL, "dark_counts": DARK_COUNTS_ON,
            "dark_rate_hz": float(cfg.dark_rate_hz),
            "readout_window_ns": list(cfg.readout_window_ns),
            "window_ns": float(cfg.window_ns), "v_dark": V_DARK_READOUT,
            "time_rule": time_rule_record(),
            "n_no_trigger": int(record.n_no_trigger),
            "n_prompt_dropped": int(record.n_prompt_dropped),
            "n_dark_moved_trigger": int(record.n_dark_moved_trigger),
            "fixed_response": reference_provenance(cfg),
            "yield_range_photons": [float(cfg.y_lo), float(cfg.y_hi)],
            "yield_range_status": YIELD_RANGE_STATUS_IN_FORCE,
            "yield_range_decision_status": YIELD_RANGE_DECISION_STATUS_IN_FORCE,
            "law": law, "setting": float(setting),
            "terciles": _tercile_table(geometry, generated),
            "volume_events": REGISTERED_VOLUME_EVENTS,
            "volume_breakdown": dict(REGISTERED_VOLUME_BREAKDOWN),
            "projected_hours_one_process": hours_1,
            "projected_hours_16_cores": hours_1 / BENCHMARK_CORES,
            "cpu_cap_wall_hours": CPU_CAP_WALL_HOURS,
            "fraction_of_cap_16_cores": (hours_1 / BENCHMARK_CORES) / CPU_CAP_WALL_HOURS}

def main(argv=None) -> int:  # pragma: no cover - host CLI
    """``python -m jepa_noise.lucid_bombs benchmark --geometry ...``."""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m jepa_noise.lucid_bombs")
    parser.add_argument("command", choices=("benchmark",), nargs="?", default="benchmark")
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--expected-config-sha256", default=None)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs"
                                                / "lucid_bombs_v0.json"))
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    cfg = load_generator_config(args.config)
    geometry = load_wcte_geometry(args.geometry,
                                  expected_config_sha256=args.expected_config_sha256)
    result = benchmark_generation(geometry, cfg, n_events=args.n_events)
    result["geometry_summary"] = geometry.summary()
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0

if __name__ == "__main__":  # pragma: no cover - host CLI
    raise SystemExit(main())
