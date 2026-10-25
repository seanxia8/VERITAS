"""WP4a Tier-1 bridge execution driver (runbook v4 §3.5.7, §3.5.8, §6.1).

This is the WP5 trainer that ``jepa_noise.train`` deliberately does not
implement.  It runs the registered A4 bridge cell (g = 0.02, nominal, channel
split, nonlinear) end to end:

* **Cohorts** -- the v3 WP5 ID table is fixed here (evaluation and independent
  development, seed 100) and never re-derived.  Events are generated once per
  host from the frozen ``Tier1Views`` API, cached under the data root as
  float16 ``.npy`` shards plus float32 targets, and read by every arm and seed
  (so all paired runs see identical events).  ``--max-events`` truncates each
  registered range from its start and only exists for smokes/tests; the full
  registered ranges are always recorded in the provenance.
* **Training** -- the frozen numerics of ``configs/bridge_v1.json`` (constant
  lr, no schedule, AdamW + Muon via ``jepa.build_optimizers``), fp32 with TF32
  disabled, 10 000 scheduled steps at batch 128 (v3 §6 WP4, the registered
  Tier-1 E2 batch), checkpointing by
  wall clock (default 30 min) plus one guaranteed mid-run checkpoint, and
  ``--resume``.  The encoder is frozen at the final scheduled step; no best
  checkpoint is ever used.
* **Evaluation** -- float16 probe-set read-out embeddings (pre-final-norm
  token grid and the post-time-block representation), ridge probes with the
  v3 §1.3 protocol (alphas {1e-6..100}, five contiguous grouped folds on
  probe-train, per-fold standardisation, final scaler on probe-train only,
  exact ties to the larger alpha), physics and nuisance targets,
  ``physics_MSE_CRLB`` from ``jepa_noise.crlb`` and the random-init floor.
* **Verdicts** -- the v4 §3.5.8 seed floor (largest of the two seed SDs and
  the cell's median arm SD) and the bridge halves of B1 and R7.

Everything the driver writes lives under ``--out`` or ``--data-root``.  It is
runnable as a guard-gated one-shot job (``python -m jepa_noise.bridge_run``)
and deterministic given the seeds.

Modes
-----
``beta-dev``
    Trains ``sigreg_jepa_anchored`` on the independent development cohort for
    beta in the frozen grid, one model seed (100), and applies the v3 §1.3
    rule: minimise the mean over physics parameters of validation MSE divided
    by the development probe-training variance; ties within 1e-8 relative
    favour the smaller beta.  Writes ``beta_decision.json``; the anchored
    grid refuses to start without one.
``grid``
    The bridge grid: ``sigreg_jepa``, ``sigreg_jepa_anchored``, ``mae_chi2``
    and the parallel-attention ablation, 3 seeds each, evaluation cohort.
    Writes per-run manifests/status/checkpoints/embeddings/probes, a compact
    ``aggregate.json`` and a ``PREDICTIONS_UPDATE`` fragment.  It never edits
    ``PREDICTION_STATUS.json``.
``aggregate``
    Read-only aggregation over completed runs spread across several out roots.
    It discovers ``runs/<run_id>/probes.json`` (and ``embeddings.npz`` for
    seed CKA when present), requires the full registered arm x seed set unless
    ``--allow-partial``, and writes only ``aggregate.json``,
    ``PREDICTIONS_UPDATE.md``, ``run_status.json`` and an aggregate-level
    ``manifest.jsonl`` recording each source run dir and probes sha256.  It
    never trains, evaluates, re-encodes, repairs or writes inside a run
    directory.

Engineering choices are reported one line each with the WP4a hand-off.
"""

from __future__ import annotations

import os

# Runbook v3 §3.1: no implicit torch compilation for CPU checks; this must
# precede any ``reconstruction_model`` import (see backbone.py).
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")
# Registered single-threaded BLAS/OMP environment contract (v3 §3.1, as in
# ``run.py``), well inside the guard's <= 4 threads.  The local numpy/OpenBLAS
# build thrashes badly on the small ridge solves with > 1 BLAS thread
# (128x128 solve: 87 ms at 4 threads vs 0.1 ms at 1 thread), so the runner
# pins one thread and enforces it at runtime in ``configure_runtime``.
for _thread_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import train as data_products
from .crlb import crlb as crlb_matrix
from .jepa import (
    ALL_ARMS,
    CONFIG_PATH,
    build_model,
    build_optimizers,
    load_config,
)
from .probes import ALPHAS, evaluate as evaluate_probes, grouped_ridge

__all__ = [
    "COHORT_IDS",
    "PHYSICS_TARGETS",
    "NUISANCE_TARGETS",
    "DEFINED_NUISANCE_TARGETS",
    "REGISTERED_SEEDS",
    "CohortSpec",
    "CohortData",
    "RunSpec",
    "build_cohort_specs",
    "contiguous_groups",
    "crlb_variances",
    "encode_events",
    "ensure_cohort",
    "fit_probe_suite",
    "fit_ridge_probes",
    "linear_cka",
    "load_beta_decision",
    "prepare_products",
    "run_grid",
    "run_beta_dev",
    "run_aggregate",
    "seed_floor",
    "select_beta",
    "select_beta_lucid_physics",
    "read_plain_physics_reference",
    "B1_EXTENSION",
    "B1_EXT_BETA_GRID",
    "B1_EXT_PHYSICS_TOLERANCE",
    "b1_verdict",
    "r7_bridge_verdict",
    "write_beta_decision",
    "main",
]

#: Registered physics targets, in the generator's (amplitude, t0, tau_t) order.
PHYSICS_TARGETS = ("amplitude", "t0", "tau_t")
#: Registered nuisance coefficients, in ``Tier1Views.sample()['nuisance']`` order.
NUISANCE_TARGETS = ("line_I", "line_Q", "common_0", "common_1")
#: Nuisance coefficients that are undefined at line = 0 (they are recorded as
#: undefined, never as zero; v3 §1.3, v4 §3.2.8).
LINE_TARGETS = ("line_I", "line_Q")
#: Nuisance targets that are defined for the bridge cell (line = 0); B1/R7 use
#: this set, and a target recorded as undefined is never silently dropped
#: (v4 §3.2.8).  The per-run ``probes.json`` records the same set as
#: ``nuisance.declared`` and ``defined_target_set_complete``.
DEFINED_NUISANCE_TARGETS = ("common_0", "common_1")
#: The registered arm x seed set of the WP4a bridge grid.
REGISTERED_SEEDS = (0, 1, 2)

SCHEMA_RUN = "jepa_bridge_run_v1"
SCHEMA_PROBES = "jepa_bridge_probes_v1"
SCHEMA_CHECKPOINT = "jepa_bridge_checkpoint_v1"
SCHEMA_BETA = "jepa_bridge_beta_decision_v1"
SCHEMA_AGGREGATE = "jepa_bridge_aggregate_v1"
SCHEMA_AGGREGATE_MANIFEST = "jepa_bridge_aggregate_manifest_v1"

#: A9 labelled follow-up B1-ext: bridge beta extension.  It is opt-in only and
#: never changes the registered v3 §1.3 grid {0.1, 1, 10} or the registered
#: B1 verdict.  The development cohort uses the LUCiD selection rule: the
#: largest beta whose per-parameter physics MSE/CRLB stays within 1.1 x the
#: plain (sigreg_jepa) arm's.  Budget <= 4 Titan GPU-hours.
B1_EXTENSION = "B1-ext"
B1_EXT_BETA_GRID = (10.0, 100.0, 1000.0)
B1_EXT_PHYSICS_TOLERANCE = 1.1
B1_EXT_BUDGET_GPU_HOURS = 4.0
B1_EXT_TIE_RULE = (
    "primary rule: the largest beta whose per-parameter physics MSE/CRLB is "
    "<= 1.1 x the plain reference's; if no beta qualifies the selection is "
    "'none_qualified' and no beta is frozen.  Tie rule for qualified betas "
    "whose physics vectors are numerically equal: choose the smaller beta "
    "(no physical gain from the larger coefficient), unless the physics "
    "values are bit-identical, in which case choose the larger beta (the "
    "LUCiD rule's preference for an active anchor at identical physics)"
)

#: v3 WP5 ID table, inclusive ranges.  The bridge cell is A4 g = 0.02,
#: nominal amplitude, channel split; the development cohort is independent
#: (seed 100) and is never an evaluation cohort.
COHORT_IDS = {
    "evaluation": {
        "subject_training": (0, 199999),
        "probe_training": (200000, 209999),
        "probe_test": (210000, 219999),
    },
    "development": {
        "subject_training": (1000000, 1199999),
        "probe_training": (1200000, 1209999),
        "probe_validation": (1210000, 1219999),
    },
}

#: Role namespaces: one independent seed namespace per role/cell/model seed
#: (v3 WP5).  These strings are part of the data cache key.
COHORT_ROLE_NAMES = {
    "evaluation": {
        "subject_training": "evaluation_subject_training",
        "probe_training": "evaluation_probe_training",
        "probe_test": "evaluation_probe_test",
    },
    "development": {
        "subject_training": "development_subject_training",
        "probe_training": "development_probe_training",
        "probe_validation": "development_probe_validation",
    },
}

#: Registered batch size for the bridge: v3 §6 WP4 fixes ``batch = 128`` for the
#: Tier-1 E2, and v4 §3.5.1 defines the bridge as that Tier-1 E2.  v4 §6.1's
#: "batch 32 must fit ... same effective batch of 32" is the LUCiD encoder's
#: registered batch (§3.5.5), not a re-specification of the bridge; that
#: reading is recorded in every aggregate (``config_batch_conflict``) and no
#: config value is edited here.
REGISTERED_BATCH = 128
#: Registered primary step count (config ``training.steps_primary``).
REGISTERED_STEPS = 10000

#: Fixed conversion from float64 numpy arrays to canonical bytes for hashing.
_DATA_DTYPE = "float16"


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:  # pragma: no cover - git is present in this checkout
        return "unknown"


def git_status(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def host_name() -> str:
    return platform.node() or "unknown-host"


def _jsonable(value):
    """Recursively convert numpy values and reject non-finite floats."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite float in result record")
        return number
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported result type {type(value)!r}")


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), separators=(",", ":"), sort_keys=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_sha256sums(directory: Path) -> dict:
    """Write ``SHA256SUMS`` for the flat files of ``directory`` (excluding it)."""
    directory = Path(directory)
    entries = []
    total_bytes = 0
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        total_bytes += path.stat().st_size
        entries.append(f"{sha256_file(path)}  {path.name}")
    atomic_write_text(directory / "SHA256SUMS", "\n".join(entries) + "\n")
    return {
        "file_count": len(entries) + 1,
        "bytes": total_bytes,
        "sha256": sha256_file(directory / "SHA256SUMS"),
    }


def _atomic_savez(path: Path, **arrays) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npz")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _atomic_save_numpy(path: Path, array) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


#: Runtime handle for the single-BLAS-thread contract (kept alive for the
#: process lifetime because the limiter restores the old pool on exit).
_THREADPOOL_LIMIT = None


def configure_runtime(device: str) -> torch.device:
    """fp32, TF32 disabled, deterministic cuDNN, single BLAS thread (v4 §6.1)."""
    global _THREADPOOL_LIMIT
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMIT = threadpool_limits(limits=1, user_api="blas")
    except Exception:  # pragma: no cover - threadpoolctl is a dependency
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:  # pragma: no cover - older builds
        pass
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is unavailable")
    return resolved


# ---------------------------------------------------------------------------
# cohorts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CohortSpec:
    name: str
    views_constructor: dict
    roles: Dict[str, Tuple[int, int]]
    role_names: Dict[str, str]
    crlb_role: Optional[str]

    @property
    def probe_train_role(self) -> str:
        return "probe_training"

    @property
    def probe_test_role(self) -> str:
        return "probe_test" if "probe_test" in self.roles else "probe_validation"


@dataclass
class RoleData:
    name: str
    namespace: str
    ids: np.ndarray
    raw: np.ndarray
    theta: np.ndarray
    nuisance: np.ndarray
    nuisance_valid: np.ndarray
    event_ids: np.ndarray
    crlb_variances: Optional[np.ndarray] = None
    amplitude_targets: Optional[np.ndarray] = None


@dataclass
class CohortData:
    spec: CohortSpec
    key: str
    directory: Path
    roles: Dict[str, RoleData]
    manifest: dict

    @property
    def views_manifest(self) -> dict:
        return self.manifest["views_manifest"]


TOY_DELAY_ENV = "JEPA_TOY_DELAY"


def toy_delay_extension() -> dict:
    """Read the opt-in channel-delay extension from ``$JEPA_TOY_DELAY`` (JSON).

    Only ``{"delay_mode": "fixed", "delay_step": <float>}`` is accepted here: the
    bridge probes and CRLB assume three physics parameters, so the per-event
    (fourth-parameter) mode needs further work and is refused.
    """
    raw = os.environ.get(TOY_DELAY_ENV)
    if not raw:
        return {}
    value = json.loads(raw)
    if set(value) != {"delay_mode", "delay_step"} or value["delay_mode"] != "fixed":
        raise RuntimeError(f"{TOY_DELAY_ENV} must be {{'delay_mode': 'fixed', 'delay_step': x}}")
    return {"delay_mode": "fixed", "delay_step": float(value["delay_step"])}


def build_cohort_specs(config: dict) -> Dict[str, CohortSpec]:
    """The registered evaluation and development cohorts, with no drift."""
    anchor = config["objective"]["anchor"]
    development = dict(anchor["development_cohort"]["views_constructor"])
    evaluation = dict(data_products.BRIDGE_VIEWS)
    frozen = (config.get("data", {}).get("generator") or {}).get(
        "views_constructor"
    )
    delay = toy_delay_extension()
    if delay:
        # Extension (registered: false, author request 2026-10-24): channel-ordered
        # arrival delays.  Merged into both cohorts so the cohort cache key and every
        # run manifest record it; the registered BRIDGE_VIEWS check is not applied.
        evaluation.update(delay)
        development.update(delay)
        frozen = None
    if frozen and dict(frozen) != evaluation:
        raise RuntimeError(
            "configs bridge generator views_constructor "
            f"{frozen} differs from the registered BRIDGE_VIEWS {evaluation}"
        )
    return {
        "evaluation": CohortSpec(
            name="evaluation",
            views_constructor=evaluation,
            roles=dict(COHORT_IDS["evaluation"]),
            role_names=dict(COHORT_ROLE_NAMES["evaluation"]),
            crlb_role="probe_test",
        ),
        "development": CohortSpec(
            name="development",
            views_constructor=development,
            roles=dict(COHORT_IDS["development"]),
            role_names=dict(COHORT_ROLE_NAMES["development"]),
            crlb_role="probe_validation",
        ),
    }


def _cohort_key(spec: CohortSpec, max_events: Optional[int], fixture: str) -> str:
    payload = {
        "schema": "jepa_bridge_cohort_v2",
        "name": spec.name,
        "includes_amplitude_targets": True,
        "views_constructor": spec.views_constructor,
        "roles": {k: list(v) for k, v in spec.roles.items()},
        "role_names": spec.role_names,
        "crlb_role": spec.crlb_role,
        "max_events": max_events,
        "fixture": fixture,
        "dtype": _DATA_DTYPE,
    }
    return canonical_sha256(payload)[:32]


def _registered_views_manifest(spec: CohortSpec) -> dict:
    """The required Tier1Views provenance (law/base/split/noise-cov hashes)."""
    constructor = spec.views_constructor
    return {
        "fixture_requested_views": {
            "signal_mode": constructor.get("signal_mode"),
            "split": constructor.get("split"),
            "law": constructor.get("law"),
            "base_rho": constructor.get("base_rho"),
            "theta_fixture": "v4-a4-e1b",
            "seed": constructor.get("seed"),
            "cell": constructor.get("cell"),
            "model_seed": constructor.get("model_seed"),
            "common_strength": constructor.get("common_strength"),
            "line_multiplier": constructor.get("line_multiplier"),
            "amplitude_scale": constructor.get("amplitude_scale"),
        },
        "note": (
            "the full Tier1Views.manifest() (split hash, noise-covariance hash, "
            "theta fixture, law, base_rho) is recorded under 'full'"
        ),
    }


def _synthetic_jacobian(theta: np.ndarray, n_obs: int = 256) -> np.ndarray:
    """Deterministic toy (n_obs, 3) Jacobian for synthetic-fixture smoke runs."""
    t = np.arange(n_obs, dtype=np.float64)
    base = np.stack(
        [
            np.sin(2 * np.pi * t / n_obs),
            np.cos(2 * np.pi * t / n_obs),
            np.sin(4 * np.pi * t / n_obs),
        ],
        axis=1,
    )
    scales = np.asarray([1.0, 0.5, 0.2]) * float(theta[0])
    return base * scales


def _synthetic_crlb(theta: np.ndarray, channels: int = 8, samples: int = 32):
    """Toy noise and Jacobian for the synthetic fixture (not the A4 N_actual)."""
    n_obs = channels * samples
    scale = np.repeat(np.sqrt(np.arange(1, channels + 1, dtype=np.float64)), samples)
    noise = np.diag(scale**2)
    jacobians = [_synthetic_jacobian(th, n_obs) for th in theta]
    return noise, jacobians


def _generate_synthetic_role(
    spec: CohortSpec, role: str, ids: np.ndarray, config: dict
) -> Dict[str, np.ndarray]:
    """Deterministic fixture events; same law shape, not the A4 law."""
    seed = int(spec.views_constructor.get("seed", 0))
    channels = int(config["data"]["channels"])
    samples = int(config["data"]["n_samples"])
    role_index = sorted(spec.roles).index(role)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 991, role_index]))
    n = len(ids)
    channel_scale = np.sqrt(np.arange(1, channels + 1, dtype=np.float64))
    raw = (
        rng.normal(size=(n, channels, samples)) * channel_scale[None, :, None]
    ).astype(np.float16)
    theta = np.stack(
        [
            rng.uniform(3.0, 6.0, n),
            rng.uniform(0.30, 0.40, n),
            rng.uniform(0.06, 0.10, n),
        ],
        axis=1,
    ).astype(np.float32)
    # The registered bridge cell has line = 0 and common = 0.02: the line
    # coefficients are undefined, the common ones are declared non-constant.
    nuisance = np.zeros((n, len(NUISANCE_TARGETS)), dtype=np.float32)
    nuisance[:, 2:] = rng.normal(size=(n, 2)).astype(np.float32) * 0.07
    # Registered bridge cell: line = 0 (undefined), common = 0.02 (declared).
    valid = np.asarray([False, False, True, True], dtype=bool)
    return {
        "raw": raw,
        "theta": theta,
        "nuisance": nuisance,
        "nuisance_valid": valid,
        "event_ids": np.asarray(ids, dtype=np.int64),
    }


def _generate_a4_role(
    spec: CohortSpec,
    views,
    role: str,
    ids: np.ndarray,
    config: dict,
    chunk: int = 2048,
) -> Dict[str, np.ndarray]:
    channels = int(config["data"]["channels"])
    samples = int(config["data"]["n_samples"])
    n = len(ids)
    raw = np.empty((n, channels, samples), dtype=np.float16)
    theta = np.empty((n, 3), dtype=np.float32)
    nuisance = np.empty((n, len(NUISANCE_TARGETS)), dtype=np.float32)
    valid = None
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sample = views.sample(ids[start:stop], role=spec.role_names[role])
        raw[start:stop] = sample["raw"].astype(np.float16)
        theta[start:stop] = sample["theta"].astype(np.float32)
        nuisance[start:stop] = sample["nuisance"].astype(np.float32)
        valid = np.asarray(sample["nuisance_valid"], dtype=bool)
    return {
        "raw": raw,
        "theta": theta,
        "nuisance": nuisance,
        "nuisance_valid": valid,
        "event_ids": np.asarray(ids, dtype=np.int64),
    }


def _generate_crlb(
    spec: CohortSpec,
    views,
    theta: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Per-event CRLB variances (3,) for the registered A4 law.

    The Fisher matrix is formed from the exact modified ``N_actual`` (full-rank
    solve, never a pseudoinverse) and the eventwise central-difference Jacobian
    from ``Tier1Views.jacobian`` (v3 §4.2).  ``jepa_noise.crlb`` is used to
    verify the accelerated path on the first event.
    """
    from scipy.linalg import solve_triangular

    noise = np.asarray(views.noise_covariance, dtype=np.float64)
    chol = np.linalg.cholesky(noise)
    out = np.full((len(theta), 3), np.nan, dtype=np.float64)
    invalid = 0
    for index, th in enumerate(theta):
        jacobian = views.jacobian(np.asarray(th, dtype=np.float64))
        whitened = solve_triangular(chol, jacobian, lower=True)
        information = whitened.T @ whitened
        if np.linalg.matrix_rank(information) != 3:
            invalid += 1
            continue
        variances = np.diag(np.linalg.inv(information))
        if index == 0:
            reference = np.diag(
                crlb_matrix(jacobian, noise)
            )
            if not np.allclose(variances, reference, rtol=1e-6, atol=1e-12):
                raise RuntimeError(
                    "accelerated CRLB path disagrees with jepa_noise.crlb: "
                    f"{variances} vs {reference}"
                )
        out[index] = variances
    if invalid:
        raise RuntimeError(
            f"CRLB Fisher rank ineligible for {invalid}/{len(theta)} events"
        )
    return out


def ensure_cohort(
    spec: CohortSpec,
    data_root: Path,
    max_events: Optional[int],
    fixture: str,
    config: dict,
) -> CohortData:
    """Load or generate the cohort event cache under ``data_root``."""
    fixture = "a4" if fixture == "a4" else "synthetic"
    key = _cohort_key(spec, max_events, fixture)
    directory = Path(data_root) / "bridge_v1" / key
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest["files"].values():
            path = directory / entry["file"]
            if not path.exists() or sha256_file(path) != entry["sha256"]:
                raise RuntimeError(
                    f"cohort cache {directory} failed verification for "
                    f"{entry['file']}; remove it to regenerate"
                )
        return _load_cohort(spec, key, directory, manifest)

    directory.mkdir(parents=True, exist_ok=True)
    views = None
    if fixture == "a4":
        from .views import Tier1Views

        if "delay_mode" in spec.views_constructor:
            from .delay_views import DelayedTier1Views

            views = DelayedTier1Views(**spec.views_constructor)
        else:
            views = Tier1Views(**spec.views_constructor)
        views_manifest = views.manifest()
        views_manifest["fixture"] = "a4"
        views_manifest["amplitude_gains"] = [
            float(value) for value in np.asarray(views.gains)
        ]
        views_manifest["amplitude_definition"] = (
            "max over the 256 samples of views.waveform(theta) per channel"
        )
    else:
        views_manifest = _registered_views_manifest(spec)
        views_manifest["fixture"] = "synthetic"
        views_manifest["full"] = None
        views_manifest["amplitude_gains"] = [1.0] * 8
        views_manifest["amplitude_definition"] = (
            "fixture: amplitude * unit channel gain (not the A4 waveform)"
        )
        views_manifest["note"] = (
            "synthetic fixture for smokes/tests only; not the registered A4 law"
        )
    roles: Dict[str, RoleData] = {}
    file_entries: Dict[str, dict] = {}
    for role in spec.roles:
        start, stop = spec.roles[role]
        if max_events is None:
            ids = np.arange(start, stop + 1, dtype=np.int64)
        else:
            count = min(stop - start + 1, int(max_events))
            ids = np.arange(start, start + count, dtype=np.int64)
        if fixture == "a4":
            generated = _generate_a4_role(spec, views, role, ids, config)
        else:
            generated = _generate_synthetic_role(spec, role, ids, config)
        arrays = dict(generated)
        if role in ("probe_training", "probe_test", "probe_validation"):
            # Per-channel pulse amplitude of the noiseless waveform; identical
            # for every arm and seed, so it is cached with the cohort.
            if fixture == "a4":
                arrays["amplitude"] = _amplitude_targets_from_views(
                    views, generated["theta"]
                )
            else:
                arrays["amplitude"] = _amplitude_targets(
                    generated["theta"], np.ones(8, dtype=np.float64)
                )
        if spec.crlb_role == role:
            if fixture == "a4":
                arrays["crlb_variances"] = _generate_crlb(
                    spec, views, generated["theta"], config
                )
            else:
                noise, jacobians = _synthetic_crlb(generated["theta"])
                arrays["crlb_variances"] = _crlb_variances_from(
                    noise, jacobians, verify=True
                )
        raw_path = directory / f"raw_{role}.npy"
        targets_path = directory / f"targets_{role}.npz"
        _atomic_save_numpy(raw_path, arrays["raw"])
        target_arrays = {
            "theta": arrays["theta"],
            "nuisance": arrays["nuisance"],
            "nuisance_valid": arrays["nuisance_valid"],
            "event_ids": arrays["event_ids"],
        }
        if "crlb_variances" in arrays:
            target_arrays["crlb_variances"] = arrays["crlb_variances"]
        if "amplitude" in arrays:
            target_arrays["amplitude"] = arrays["amplitude"].astype(np.float32)
        _atomic_savez(targets_path, **target_arrays)
        for path in (raw_path, targets_path):
            file_entries[f"{role}/{path.name}"] = {
                "file": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        roles[role] = RoleData(
            name=role,
            namespace=spec.role_names[role],
            ids=ids,
            raw=None,  # loaded below from the cache
            theta=arrays["theta"],
            nuisance=arrays["nuisance"],
            nuisance_valid=arrays["nuisance_valid"],
            event_ids=arrays["event_ids"],
            crlb_variances=arrays.get("crlb_variances"),
            amplitude_targets=arrays.get("amplitude"),
        )
    manifest = {
        "schema": "jepa_bridge_cohort_v2",
        "cohort": spec.name,
        "key": key,
        "fixture": fixture,
        "created_at_utc": utcnow(),
        "host": host_name(),
        "views_constructor": dict(spec.views_constructor),
        "views_manifest": views_manifest,
        "roles": {
            role: {
                "ids": [int(spec.roles[role][0]), int(spec.roles[role][1])],
                "actual_ids": [int(roles[role].ids[0]), int(roles[role].ids[-1])],
                "n_events": int(len(roles[role].ids)),
                "namespace": spec.role_names[role],
                "raw_dtype": _DATA_DTYPE,
                "crlb": spec.crlb_role == role,
                "amplitude_targets": role in (
                    "probe_training", "probe_test", "probe_validation"
                ),
            }
            for role in spec.roles
        },
        "max_events": max_events,
        "truncated": max_events is not None,
        "files": file_entries,
        "bytes": int(sum(entry["bytes"] for entry in file_entries.values())),
        "note": (
            "cached once on this host; every arm and seed reads these bytes "
            "(v4 §6.1 data decision)"
        ),
    }
    atomic_write_json(manifest_path, manifest)
    write_sha256sums(directory)
    _forget_roles(roles)
    return _load_cohort(spec, key, directory, manifest)


def _forget_roles(roles: Dict[str, RoleData]) -> None:
    for role in roles.values():
        role.raw = None


def _load_cohort(
    spec: CohortSpec, key: str, directory: Path, manifest: dict
) -> CohortData:
    roles: Dict[str, RoleData] = {}
    for role, meta in manifest["roles"].items():
        raw = np.load(directory / f"raw_{role}.npy", mmap_mode="r")
        with np.load(directory / f"targets_{role}.npz") as data:
            crlb = data["crlb_variances"] if "crlb_variances" in data else None
            amplitude = data["amplitude"] if "amplitude" in data else None
            roles[role] = RoleData(
                name=role,
                namespace=meta["namespace"],
                ids=np.arange(meta["actual_ids"][0], meta["actual_ids"][1] + 1),
                raw=raw,
                theta=np.asarray(data["theta"], dtype=np.float32),
                nuisance=np.asarray(data["nuisance"], dtype=np.float32),
                nuisance_valid=np.asarray(data["nuisance_valid"], dtype=bool),
                event_ids=np.asarray(data["event_ids"], dtype=np.int64),
                crlb_variances=None if crlb is None else np.asarray(crlb),
                amplitude_targets=(
                    None if amplitude is None else np.asarray(amplitude)
                ),
            )
    return CohortData(spec=spec, key=key, directory=directory, roles=roles,
                      manifest=manifest)


def cohort_provenance(cohort: CohortData) -> dict:
    """The id ranges and Tier1Views manifest recorded in every run."""
    manifest = cohort.manifest
    views = manifest["views_manifest"]
    constructor = cohort.spec.views_constructor

    def pick(key, default=None):
        value = views.get(key)
        if value is None:
            value = constructor.get(key, default)
        return value

    return {
        "name": cohort.spec.name,
        "data_root_key": cohort.key,
        "fixture": manifest["fixture"],
        "truncated": manifest["truncated"],
        "max_events": manifest["max_events"],
        "roles": {
            role: {
                "ids": meta["ids"],
                "actual_ids": meta["actual_ids"],
                "n_events": meta["n_events"],
                "namespace": meta["namespace"],
            }
            for role, meta in manifest["roles"].items()
        },
        "views": {
            "law": pick("law"),
            "base_rho": pick("base_rho"),
            "theta_fixture": pick("theta_fixture", "v4-a4-e1b"),
            "split": pick("split"),
            "split_hash": pick("split_hash"),
            "noise_covariance_sha256": pick("noise_covariance_sha256"),
            "signal_mode": pick("signal_mode"),
            "seed": pick("seed"),
            "cell": pick("cell"),
            "model_seed": pick("model_seed"),
            "common_strength": pick("common_strength"),
            "line_multiplier": pick("line_multiplier"),
            "amplitude_gains": views.get("amplitude_gains"),
            "amplitude_definition": views.get("amplitude_definition"),
        },
        "noise_covariance_sha256_note": (
            "informational only: its last bits are BLAS-dependent "
            "(config platform_stability note)"
        ),
    }


# ---------------------------------------------------------------------------
# frozen data products
# ---------------------------------------------------------------------------
def prepare_products(
    config: dict, fixture: str, data_root: Path
) -> Dict[str, object]:
    """Frozen fixed scaling, overlay bank and target marginal of N_actual."""
    if fixture == "a4":
        return data_products.a4_bridge_products(
            config=config,
            verify=True,
            artifact_root=str(Path(data_root) / "products"),
        )
    return data_products.synthetic_fixture(config, seed=0, n_calibration=32)


def products_provenance(products: dict) -> dict:
    metadata = products.get("metadata") or {}
    return {
        "source": metadata.get("verified_against_config", None),
        "fixed_scaling": (metadata.get("fixed_scaling") or {}).get(
            "sha256_informational"
        ),
        "target_marginal": (metadata.get("target_noise_covariance") or {}).get(
            "sha256_informational"
        ),
        "overlay_noise": (metadata.get("overlay_noise") or {}).get(
            "sha256_informational"
        ),
        "computed_on": metadata.get("computed_on"),
        "note": (
            "fixed scaling, overlay bank and N_actual marginal are frozen "
            "data products shared by every arm, seed and cohort; the sha256 "
            "values are informational (last-bit BLAS differences)"
        ),
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    run_id: str
    arm: str
    seed: int
    cohort: str
    steps: int
    batch: int
    beta: Optional[float] = None
    beta_source: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "arm": self.arm,
            "seed": int(self.seed),
            "cohort": self.cohort,
            "steps": int(self.steps),
            "batch": int(self.batch),
            "beta": self.beta,
            "beta_source": self.beta_source,
        }


class EventBatcher:
    """Deterministic step-indexed batch stream over a cached event pool.

    Seed -> epoch permutation -> batch; a step can be resumed without replaying
    earlier steps, and every arm with the same seed sees identical batches.
    """

    def __init__(self, raw: np.ndarray, batch: int, seed: int, key: str):
        if raw.ndim != 3:
            raise ValueError("event pool must be (N, C, T)")
        self.raw = raw
        self.batch = int(batch)
        self.seed = int(seed)
        self.n = int(len(raw))
        if self.n < 1:
            raise ValueError("empty event pool")
        self._epoch: Optional[int] = None
        self._perm: Optional[np.ndarray] = None
        self._key = int(sha256_bytes(key.encode())[:8], 16)

    def _permutation(self, epoch: int) -> np.ndarray:
        if self._epoch != epoch:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(epoch), self._key])
            )
            self._perm = rng.permutation(self.n)
            self._epoch = epoch
        return self._perm

    def indices(self, step: int) -> np.ndarray:
        start = int(step) * self.batch
        epoch, offset = divmod(start, self.n)
        indices = self._permutation(epoch)
        picked = indices[offset : offset + self.batch]
        if len(picked) < self.batch:
            following = self._permutation(epoch + 1)
            needed = self.batch - len(picked)
            picked = np.concatenate([picked, following[:needed]])
            # ``_permutation(epoch + 1)`` is cached; the next call for the
            # following epoch reuses it.
            self._epoch, self._perm = epoch + 1, following
        return picked

    def batch_tensor(self, step: int, device: torch.device) -> torch.Tensor:
        indices = self.indices(step)
        values = np.asarray(self.raw[indices], dtype=np.float32)
        return torch.as_tensor(values, device=device)


def _checkpoint_payload(
    model,
    optimizers,
    run: RunSpec,
    step: int,
    elapsed: float,
    config_sha: str,
    data_key: str,
) -> dict:
    payload = {
        "schema": SCHEMA_CHECKPOINT,
        "config_sha256": config_sha,
        "data_key": data_key,
        "run": run.as_dict(),
        "step": int(step),
        "steps_scheduled": int(run.steps),
        "elapsed_seconds": float(elapsed),
        "model": {name: tensor.detach().cpu() for name, tensor in
                  model.state_dict().items()},
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "overlay_generator_state": None,
    }
    generator = getattr(model, "_overlay_generator", None)
    if generator is not None:
        payload["overlay_generator_state"] = generator.get_state()
    return payload


def _save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validate_checkpoint(payload: dict, run: RunSpec, config_sha: str, key: str):
    if payload.get("schema") != SCHEMA_CHECKPOINT:
        raise RuntimeError("not a bridge checkpoint")
    if int(payload["steps_scheduled"]) != int(run.steps):
        raise RuntimeError(
            "checkpoint scheduled steps "
            f"{payload['steps_scheduled']} != requested {run.steps}"
        )
    previous = payload.get("run", {})
    for field_name in ("arm", "seed", "cohort", "batch"):
        if previous.get(field_name) != run.as_dict()[field_name]:
            raise RuntimeError(
                f"checkpoint {field_name} {previous.get(field_name)!r} does not "
                f"match the requested run {run.as_dict()[field_name]!r}"
            )
    if previous.get("beta") != run.beta:
        raise RuntimeError("checkpoint beta does not match the requested beta")
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("checkpoint config sha256 differs from the config")
    if payload.get("data_key") != key:
        raise RuntimeError("checkpoint data key differs from the cohort cache")


def train_model(
    *,
    model,
    optimizers,
    batcher: EventBatcher,
    run: RunSpec,
    checkpoint_path: Path,
    config_sha: str,
    data_key: str,
    device: torch.device,
    resume: Optional[Path] = None,
    stop_after: Optional[int] = None,
    checkpoint_minutes: float = 30.0,
    log_path: Optional[Path] = None,
    log_every: Optional[int] = None,
) -> dict:
    """Constant-lr loop; returns ``{step, interrupted, elapsed_seconds}``."""
    start_step = 0
    elapsed_before = 0.0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        _validate_checkpoint(payload, run, config_sha, data_key)
        model.load_state_dict(payload["model"], strict=True)
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        generator = getattr(model, "_overlay_generator", None)
        if generator is not None and payload.get("overlay_generator_state") is not None:
            generator.set_state(payload["overlay_generator_state"])
        start_step = int(payload["step"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        if start_step > run.steps:
            raise RuntimeError(
                f"checkpoint step {start_step} is beyond the scheduled {run.steps}"
            )

    stop_requested = {"signal": None}

    def _handler(signum, _frame):
        stop_requested["signal"] = int(signum)

    previous_handler = None
    try:
        previous_handler = signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):  # pragma: no cover - non-main thread
        previous_handler = None
    if log_every is None:
        log_every = max(1, run.steps // 100)
    mid_step = max(1, run.steps // 2)
    next_time_checkpoint = (
        time.perf_counter() + checkpoint_minutes * 60.0
        if checkpoint_minutes and checkpoint_minutes > 0
        else math.inf
    )
    step = start_step
    interrupted = False
    loop_start = time.perf_counter()
    log_lines: List[str] = []
    try:
        while step < run.steps:
            if stop_requested["signal"] is not None:
                interrupted = True
                break
            if stop_after is not None and step >= int(stop_after):
                interrupted = True
                break
            batch = batcher.batch_tensor(step, device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1}")
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            step += 1
            elapsed = elapsed_before + (time.perf_counter() - loop_start)
            if step % log_every == 0 or step == run.steps:
                entry = {"step": step, "elapsed_seconds": round(elapsed, 3)}
                entry.update(
                    {k: float(v) for k, v in out["metrics"].items()
                     if isinstance(v, (int, float))}
                )
                log_lines.append(json.dumps(entry, sort_keys=True))
            if (
                step == run.steps
                or step == mid_step
                or time.perf_counter() >= next_time_checkpoint
            ):
                _save_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        model, optimizers, run, step, elapsed, config_sha, data_key
                    ),
                )
                if time.perf_counter() >= next_time_checkpoint:
                    next_time_checkpoint = (
                        time.perf_counter() + checkpoint_minutes * 60.0
                    )
    finally:
        if previous_handler is not None:
            try:
                signal.signal(signal.SIGTERM, previous_handler)
            except (ValueError, OSError):  # pragma: no cover
                pass
        if log_path is not None and log_lines:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write("\n".join(log_lines) + "\n")
    elapsed = elapsed_before + (time.perf_counter() - loop_start)
    if not interrupted:
        if step != run.steps:  # pragma: no cover - defensive
            raise RuntimeError(f"training stopped at {step}, expected {run.steps}")
        _save_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                model, optimizers, run, step, elapsed, config_sha, data_key
            ),
        )
    elif step > start_step or not checkpoint_path.exists():
        _save_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                model, optimizers, run, step, elapsed, config_sha, data_key
            ),
        )
    return {
        "step": int(step),
        "steps_scheduled": int(run.steps),
        "interrupted": bool(interrupted),
        "interrupted_by": (
            "signal" if stop_requested["signal"] is not None
            else ("stop_after" if interrupted else None)
        ),
        "signal": stop_requested["signal"],
        "elapsed_seconds": float(elapsed),
        "resumed_from_step": int(start_step),
    }


# ---------------------------------------------------------------------------
# embeddings and probes
# ---------------------------------------------------------------------------
def encode_events(
    model,
    raw: np.ndarray,
    device: torch.device,
    *,
    chunk: int = 256,
) -> dict:
    """Unmasked read-out embeddings, float16, plus small probe reductions.

    The read-out point is the pre-``final_proj_norm`` token grid ``(B, N_patch,
    C, d)`` (v4 §3.5.7) and the post-time-block representation ``(B, C,
    N_patch, d)``.  Physics/nuisance probes use the mean over tokens; the
    amplitude probe uses the per-channel mean over patches.
    """
    encoder = model.encoder
    channels = int(encoder.n_channels)
    n_patch = int(encoder.n_patch)
    d_model = int(encoder.config.d_model)
    n = int(len(raw))
    grid_out = np.empty((n, n_patch, channels, d_model), dtype=np.float16)
    post_out = np.empty((n, channels, n_patch, d_model), dtype=np.float16)
    mean_tokens = np.empty((n, d_model), dtype=np.float32)
    post_mean = np.empty((n, channels, d_model), dtype=np.float32)
    grid_mean = np.empty((n, channels, d_model), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            batch = torch.as_tensor(
                np.asarray(raw[start:stop], dtype=np.float32), device=device
            )
            encoded = encoder.encode_unmasked(batch)
            grid = encoded["grid"]
            post = encoded["post_time"]
            grid_out[start:stop] = grid.detach().to(torch.float16).cpu().numpy()
            post_out[start:stop] = post.detach().to(torch.float16).cpu().numpy()
            mean_tokens[start:stop] = (
                grid.detach().to(torch.float32).mean(dim=(1, 2)).cpu().numpy()
            )
            grid_mean[start:stop] = (
                grid.detach().to(torch.float32).mean(dim=1).cpu().numpy()
            )
            post_mean[start:stop] = (
                post.detach().to(torch.float32).mean(dim=2).cpu().numpy()
            )
    model.train()
    return {
        "token_grid": grid_out,
        "post_time": post_out,
        "mean_tokens": mean_tokens,
        "post_mean": post_mean,
        "grid_mean": grid_mean,
    }


def contiguous_groups(n: int, folds: int = 5) -> np.ndarray:
    """Contiguous whole-group folds on probe-train (v3 §1.3)."""
    if n < folds:
        raise ValueError(
            f"need at least {folds} probe-train events for {folds} grouped folds"
        )
    sizes = np.full(folds, n // folds, dtype=np.int64)
    sizes[: n % folds] += 1
    return np.repeat(np.arange(folds), sizes)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two seed embedding matrices (v4 §3.5.8)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires paired rows")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(x.T @ y) ** 2)
    denominator = float(
        np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    )
    if denominator <= 0:
        raise ValueError("CKA undefined for a constant embedding")
    return numerator / denominator


def fit_ridge_probes(
    features_train: np.ndarray,
    targets_train: np.ndarray,
    features_test: np.ndarray,
    targets_test: np.ndarray,
    *,
    folds: int = 5,
) -> dict:
    """Five grouped folds, inside-fold standardisation, ordered alpha choice."""
    groups = contiguous_groups(len(features_train), folds=folds)
    probes = grouped_ridge(
        np.asarray(features_train, dtype=np.float64),
        np.asarray(targets_train, dtype=np.float64),
        groups,
        fit_intercept=True,
    )
    rows = evaluate_probes(
        probes,
        np.asarray(features_test, dtype=np.float64),
        np.asarray(targets_test, dtype=np.float64),
    )
    compact = []
    for probe, row in zip(probes, rows):
        compact.append(
            {
                "status": row["status"],
                "reason": row["reason"],
                "alpha": probe["alpha"],
                "validation_mse": (
                    None
                    if probe["validation_mse"] is None
                    else {str(k): float(v) for k, v in probe["validation_mse"].items()}
                ),
                "test_mse": row["mse"],
                "test_r2": row["r2"],
            }
        )
    return {"probes": probes, "rows": compact, "groups": groups}


def _amplitude_targets(theta: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Per-channel pulse amplitude of the noiseless waveform (max over time)."""
    theta = np.asarray(theta, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64).reshape(1, -1)
    return (theta[:, 0:1] * gains).astype(np.float32)


def _amplitude_targets_from_views(views, theta: np.ndarray) -> np.ndarray:
    out = np.empty((len(theta), 8), dtype=np.float32)
    for index, physical in enumerate(theta):
        waveform = views.waveform(np.asarray(physical, dtype=np.float64))
        out[index] = np.max(waveform, axis=1)
    return out


def _crlb_variances_from(
    noise: np.ndarray, jacobians: Sequence[np.ndarray], verify: bool = True
) -> np.ndarray:
    """Accelerated CRLB diagonal, verified against ``jepa_noise.crlb`` once."""
    from scipy.linalg import solve_triangular

    noise = np.asarray(noise, dtype=np.float64)
    chol = np.linalg.cholesky(noise)
    out = np.full((len(jacobians), 3), np.nan, dtype=np.float64)
    for index, jacobian in enumerate(jacobians):
        whitened = solve_triangular(chol, np.asarray(jacobian, dtype=np.float64),
                                    lower=True)
        information = whitened.T @ whitened
        if np.linalg.matrix_rank(information) != 3:
            continue
        variances = np.diag(np.linalg.inv(information))
        if verify and index == 0:
            reference = np.diag(crlb_matrix(jacobian, noise))
            if not np.allclose(variances, reference, rtol=1e-6, atol=1e-12):
                raise RuntimeError(
                    "accelerated CRLB path disagrees with jepa_noise.crlb"
                )
        out[index] = variances
    return out


def crlb_variances(noise: np.ndarray, jacobians: Sequence[np.ndarray]) -> np.ndarray:
    """Public helper used by the tests and the synthetic fixture."""
    return _crlb_variances_from(noise, jacobians, verify=True)


def fit_probe_suite(
    *,
    train: dict,
    test: dict,
    theta_train: np.ndarray,
    theta_test: np.ndarray,
    amplitude_train: np.ndarray,
    amplitude_test: np.ndarray,
    nuisance_train: np.ndarray,
    nuisance_test: np.ndarray,
    nuisance_valid: np.ndarray,
    crlb_test: Optional[np.ndarray],
) -> dict:
    """Physics, nuisance and amplitude probes with the v3 §1.3 protocol."""
    physics = fit_ridge_probes(
        train["mean_tokens"], theta_train, test["mean_tokens"], theta_test
    )
    physics.pop("probes", None)
    physics.pop("groups", None)
    physics_rows = []
    crlb_means = None
    if crlb_test is not None:
        crlb_means = np.nanmean(np.asarray(crlb_test, dtype=np.float64), axis=0)
    for index, row in enumerate(physics["rows"]):
        entry = dict(row)
        entry["target"] = PHYSICS_TARGETS[index]
        if crlb_means is None or not np.isfinite(crlb_means[index]):
            entry["crlb_mean"] = None
            entry["physics_mse_crlb"] = None
            entry["physics_mse_crlb_reason"] = "crlb_not_requested_for_cohort"
        elif row["status"] != "evaluated":
            entry["crlb_mean"] = float(crlb_means[index])
            entry["physics_mse_crlb"] = None
            entry["physics_mse_crlb_reason"] = row["reason"]
        else:
            entry["crlb_mean"] = float(crlb_means[index])
            entry["physics_mse_crlb"] = (
                None
                if entry["crlb_mean"] <= 0
                else float(row["test_mse"] / entry["crlb_mean"])
            )
        physics_rows.append(entry)
    ratios = [
        row["physics_mse_crlb"]
        for row in physics_rows
        if row["physics_mse_crlb"] is not None
    ]
    physics["targets"] = list(PHYSICS_TARGETS)
    physics["rows"] = physics_rows
    physics["mean_physics_mse_over_crlb"] = (
        float(np.mean(ratios)) if ratios else None
    )

    nuisance = fit_ridge_probes(
        train["mean_tokens"], nuisance_train, test["mean_tokens"], nuisance_test
    )
    nuisance.pop("probes", None)
    nuisance.pop("groups", None)
    declared = [
        index for index, valid in enumerate(np.asarray(nuisance_valid))
        if bool(valid)
    ]
    nuisance_rows = []
    undefined = {}
    for index, row in enumerate(nuisance["rows"]):
        name = NUISANCE_TARGETS[index]
        if index not in declared:
            entry = {
                "target": name,
                "status": "undefined",
                "reason": (
                    "line_target_undefined_at_line_zero"
                    if name in LINE_TARGETS
                    else "constant_by_declaration"
                ),
                "test_r2": None,
                "test_mse": None,
                "alpha": None,
            }
        else:
            entry = dict(row)
            entry["target"] = name
        if entry["status"] == "undefined":
            undefined[name] = entry["reason"]
        nuisance_rows.append(entry)
    defined_scores = [
        row["test_r2"] for row in nuisance_rows
        if row["target"] in {NUISANCE_TARGETS[i] for i in declared}
        and row["status"] == "evaluated"
        and row["test_r2"] is not None
        and np.isfinite(row["test_r2"])
    ]
    nuisance["targets"] = list(NUISANCE_TARGETS)
    nuisance["rows"] = nuisance_rows
    nuisance["declared"] = [NUISANCE_TARGETS[i] for i in declared]
    nuisance["undefined"] = undefined
    nuisance["defined_mean_r2"] = (
        float(np.mean(defined_scores)) if defined_scores else None
    )
    nuisance["defined_target_set_complete"] = len(defined_scores) == len(declared)

    amplitude = {}
    for location, train_features, test_features in (
        ("post_time", train["post_mean"], test["post_mean"]),
        ("tokens", train["grid_mean"], test["grid_mean"]),
    ):
        # One probe per channel on that channel's own representation (the
        # target is a per-channel amplitude; a shared 8*d feature block would
        # be an under-determined cross-channel probe for no benefit).
        per_channel = []
        for channel in range(int(train_features.shape[1])):
            fitted = fit_ridge_probes(
                train_features[:, channel, :],
                amplitude_train[:, channel : channel + 1],
                test_features[:, channel, :],
                amplitude_test[:, channel : channel + 1],
            )
            row = dict(fitted["rows"][0])
            row["channel"] = int(channel)
            per_channel.append(row)
        scores = [
            row["test_r2"] for row in per_channel
            if row["status"] == "evaluated" and row["test_r2"] is not None
            and np.isfinite(row["test_r2"])
        ]
        mean_r2 = float(np.mean(scores)) if scores else None
        amplitude[location] = {
            "rows": per_channel,
            "n_evaluated": len(scores),
            "mean_r2": mean_r2,
            "criterion": "mean over the 8 per-channel R2 values >= 0.9",
            "criterion_met": bool(mean_r2 is not None and mean_r2 >= 0.9),
        }
    amplitude["target"] = "per-channel pulse amplitude (max of the noiseless waveform)"
    return {
        "protocol": {
            "alpha_grid": list(ALPHAS),
            "folds": 5,
            "fold_grouping": "contiguous groups on probe-train only",
            "standardisation": "inside each fold; final scaler on probe-train only",
            "alpha_choice": "lowest validation MSE; exact ties to the larger alpha",
            "intercept": True,
            "feature_reduction": (
                "mean over all N_patch x C tokens for physics and nuisance "
                "(both are global targets); per-channel mean over patches for "
                "the amplitude probe"
            ),
            "unmasked_embeddings": True,
        },
        "physics": physics,
        "nuisance": nuisance,
        "amplitude": amplitude,
    }


# ---------------------------------------------------------------------------
# seed floor, beta rule and verdicts
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _sd(values: Sequence[float], ddof: int = 1) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return None
    return float(np.std(finite, ddof=ddof))


def seed_floor(
    sd_a: Optional[float],
    sd_b: Optional[float],
    all_arm_sds: Sequence[Optional[float]],
) -> float:
    """v4 §3.5.8: largest of SD_a, SD_b and the cell's median arm SD."""
    finite = [float(v) for v in all_arm_sds if v is not None and np.isfinite(v)]
    if not finite:
        raise ValueError("no finite SDs for the seed floor")
    candidates = [float(np.median(finite))]
    if sd_a is not None and np.isfinite(sd_a):
        candidates.append(float(sd_a))
    if sd_b is not None and np.isfinite(sd_b):
        candidates.append(float(sd_b))
    return float(max(candidates))


def select_beta(table: Sequence[dict], rtol: float = 1e-8) -> dict:
    """v3 §1.3 beta rule: minimise the mean normalised validation MSE.

    ``table`` entries are ``{"beta": float, "score": float, ...}``.  Ties
    within ``rtol`` relative favour the smaller coefficient.
    """
    if not table:
        raise ValueError("empty beta table")
    for row in table:
        if not np.isfinite(float(row["score"])):
            raise ValueError("non-finite beta score")
    best = min(float(row["score"]) for row in table)
    tied = [
        row for row in table
        if abs(float(row["score"]) - best) <= rtol * abs(best)
        or float(row["score"]) == best
    ]
    selected = min(tied, key=lambda row: float(row["beta"]))
    return {
        "selected_beta": float(selected["beta"]),
        "best_score": float(best),
        "tied_betas": [float(row["beta"]) for row in sorted(
            tied, key=lambda row: float(row["beta"])
        )],
        "scores": {
            str(float(row["beta"])): float(row["score"]) for row in table
        },
        "rule": (
            "minimise the mean over physics parameters of validation MSE "
            "divided by the development probe-training variance; ties within "
            "1e-8 relative favour the smaller beta (v3 §1.3)"
        ),
    }


def _physics_mse_crlb_from_payload(payload: dict) -> Dict[str, float]:
    """Per-parameter physics MSE/CRLB from a probes.json or a metrics JSON.

    Accepted shapes: ``probes.physics.rows`` (the committed probes.json),
    ``physics`` (the run's ``_metrics_summary``), or either of those nested
    under ``metrics``.  A missing or non-positive parameter is an error: the
    B1-ext rule has no meaning without the plain reference for every target.
    """
    if not isinstance(payload, dict):
        raise ValueError("plain reference is not a JSON object")
    probes = payload.get("probes")
    physics = None
    if isinstance(probes, dict) and isinstance(probes.get("physics"), dict):
        rows = probes["physics"].get("rows")
        if isinstance(rows, list):
            values = {}
            for row in rows:
                if not isinstance(row, dict) or row.get("target") is None:
                    continue
                values[str(row["target"])] = row.get("physics_mse_crlb")
            if values:
                return _validate_physics_values(values)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return _physics_mse_crlb_from_payload(metrics)
    physics = payload.get("physics")
    if isinstance(physics, dict):
        values = {}
        for parameter in PHYSICS_TARGETS:
            entry = physics.get(parameter)
            if isinstance(entry, dict):
                values[parameter] = entry.get("physics_mse_crlb")
        if values:
            return _validate_physics_values(values)
    raise ValueError(
        "plain reference has no per-parameter physics MSE/CRLB table "
        f"({PHYSICS_TARGETS}); expected a run directory's probes.json or a "
        "metrics JSON"
    )


def _validate_physics_values(values) -> Dict[str, float]:
    missing = [parameter for parameter in PHYSICS_TARGETS
               if values.get(parameter) is None]
    if missing:
        raise ValueError(
            "plain reference lacks physics MSE/CRLB for " + ", ".join(missing)
        )
    out = {}
    for parameter in PHYSICS_TARGETS:
        value = float(values[parameter])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"plain reference physics MSE/CRLB for {parameter} is "
                f"{values[parameter]!r}; a positive finite value is required"
            )
        out[parameter] = value
    return out


def read_plain_physics_reference(path) -> Tuple[Dict[str, float], dict]:
    """Read the plain ``sigreg_jepa`` seed-0 reference read-only.

    ``path`` may be a run directory (its ``probes.json`` is read) or a metrics
    JSON file.  The file actually read and its sha256 are recorded so the
    B1-ext decision can be audited.  When the payload carries run metadata it
    must name ``sigreg_jepa`` seed 0 (the committed plain bridge run); a
    metrics JSON without metadata is accepted and recorded as such.
    """
    candidate = Path(path)
    if not candidate.exists():
        raise RuntimeError(f"plain reference {candidate} not found")
    file_path = candidate / "probes.json" if candidate.is_dir() else candidate
    if not file_path.is_file():
        raise RuntimeError(
            f"plain reference {candidate} has no readable {file_path.name}"
        )
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        values = _physics_mse_crlb_from_payload(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(
            f"plain reference {file_path} is unusable: {error}"
        ) from error
    run = payload.get("run") if isinstance(payload, dict) else None
    run = run if isinstance(run, dict) else {}
    arm = run.get("arm")
    seed = run.get("seed")
    if arm is not None and arm != "sigreg_jepa":
        raise RuntimeError(
            f"plain reference {file_path} is arm {arm!r}, not sigreg_jepa"
        )
    if seed is not None and int(seed) != 0:
        raise RuntimeError(
            f"plain reference {file_path} is seed {seed}, not the committed "
            "sigreg_jepa seed-0 bridge run"
        )
    provenance = {
        "path": str(candidate),
        "file": str(file_path),
        "sha256": sha256_file(file_path),
        "run_id": run.get("run_id"),
        "arm": arm,
        "seed": seed,
        "cohort": run.get("cohort"),
        "role": "plain reference (sigreg_jepa seed 0) for B1-ext",
    }
    return values, provenance


def _physics_vectors_close(
    left: Dict[str, float], right: Dict[str, float], rtol: float
) -> bool:
    return all(
        abs(float(left[par]) - float(right[par]))
        <= rtol * max(abs(float(left[par])), abs(float(right[par])))
        for par in PHYSICS_TARGETS
    )


def _physics_vectors_identical(
    left: Dict[str, float], right: Dict[str, float]
) -> bool:
    return all(float(left[par]) == float(right[par]) for par in PHYSICS_TARGETS)


def select_beta_lucid_physics(
    table: Sequence[dict],
    plain_reference: Dict[str, float],
    *,
    tolerance: float = B1_EXT_PHYSICS_TOLERANCE,
    tie_rtol: float = 1e-12,
) -> dict:
    """A9 B1-ext LUCiD selection rule on the bridge development cohort.

    ``table`` entries carry ``beta`` and ``physics_mse_crlb`` (per parameter).
    A beta qualifies when every per-parameter candidate/reference ratio is
    ``<= tolerance``.  The largest qualifying beta is selected; if none
    qualifies the status is ``none_qualified`` and no beta is returned.  Ties
    (numerically equal physics vectors among qualified betas) follow
    :data:`B1_EXT_TIE_RULE`: smaller beta unless the physics is bit-identical,
    in which case the larger beta.
    """
    if not table:
        raise ValueError("empty beta table")
    plain = _validate_physics_values(plain_reference)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive finite physics tolerance required")
    betas = [float(row["beta"]) for row in table]
    if len(set(betas)) != len(betas):
        raise ValueError("duplicate beta in table")
    entries = []
    for row in table:
        physics = _validate_physics_values(row.get("physics_mse_crlb") or {})
        ratios = {par: physics[par]/plain[par] for par in PHYSICS_TARGETS}
        qualifies = all(ratio <= tolerance for ratio in ratios.values())
        entries.append({
            "beta": float(row["beta"]),
            "physics_mse_crlb": physics,
            "ratio_to_plain": ratios,
            "max_ratio": float(max(ratios.values())),
            "qualifies": bool(qualifies),
        })
    qualified = [entry for entry in entries if entry["qualifies"]]
    per_beta = {
        str(entry["beta"]): {
            "physics_mse_crlb": entry["physics_mse_crlb"],
            "ratio_to_plain": entry["ratio_to_plain"],
            "max_ratio": entry["max_ratio"],
            "qualifies": entry["qualifies"],
        }
        for entry in sorted(entries, key=lambda item: item["beta"])
    }
    result = {
        "status": "none_qualified",
        "selected_beta": None,
        "tolerance": float(tolerance),
        "plain_reference_physics_mse_crlb": plain,
        "considered_betas": sorted(betas),
        "qualified_betas": sorted(entry["beta"] for entry in qualified),
        "per_beta": per_beta,
        "tie_rule": B1_EXT_TIE_RULE,
        "rule": (
            "A9 B1-ext / LUCiD rule: largest beta whose physics MSE/CRLB is "
            "<= 1.1 x the plain (sigreg_jepa seed 0) reference, per parameter "
            "(amplitude, t0, tau_t)"
        ),
    }
    if not qualified:
        closest = min(entries, key=lambda item: item["max_ratio"])
        result["reason"] = "no_beta_within_physics_tolerance"
        result["closest_beta"] = closest["beta"]
        result["closest_max_ratio"] = closest["max_ratio"]
        return result
    ordered = sorted(qualified, key=lambda item: item["beta"], reverse=True)
    top = ordered[0]
    ties = [
        entry for entry in ordered[1:]
        if _physics_vectors_close(entry["physics_mse_crlb"],
                                  top["physics_mse_crlb"], tie_rtol)
    ]
    tie_kind = None
    selected = top
    if ties:
        group = [top, *ties]
        if all(_physics_vectors_identical(entry["physics_mse_crlb"],
                                          top["physics_mse_crlb"])
               for entry in ties):
            selected = max(group, key=lambda item: item["beta"])
            tie_kind = "identical_physics_larger_beta"
        else:
            selected = min(group, key=lambda item: item["beta"])
            tie_kind = "numerical_tie_smaller_beta"
    result.update({
        "status": "selected",
        "selected_beta": float(selected["beta"]),
        "selected": {
            "beta": float(selected["beta"]),
            "physics_mse_crlb": selected["physics_mse_crlb"],
            "ratio_to_plain": selected["ratio_to_plain"],
            "max_ratio": selected["max_ratio"],
        },
        "tie_kind": tie_kind,
        "tied_betas": sorted(entry["beta"] for entry in ties),
    })
    return result


def b1_verdict(
    plain: Dict[int, float],
    anchored: Dict[int, float],
    all_arm_sds: Sequence[Optional[float]],
    *,
    sd_plain: Optional[float] = None,
    sd_anchored: Optional[float] = None,
    metric: str = "nuisance_r2",
) -> dict:
    """B1 bridge half: plain > anchored in every seed, beyond the seed floor."""
    seeds = sorted(set(plain) & set(anchored))
    per_seed = {str(seed): float(plain[seed]) > float(anchored[seed])
                for seed in seeds}
    values_plain = [plain[seed] for seed in seeds]
    values_anchored = [anchored[seed] for seed in seeds]
    delta = _mean(values_plain) - _mean(values_anchored) if seeds else None
    floor = None
    if seeds:
        candidate_sds = [
            _sd(values_plain) if sd_plain is None else sd_plain,
            _sd(values_anchored) if sd_anchored is None else sd_anchored,
            *[value for value in all_arm_sds],
        ]
        if any(
            value is not None and np.isfinite(value)
            for value in candidate_sds
        ):
            floor = seed_floor(
                _sd(values_plain) if sd_plain is None else sd_plain,
                _sd(values_anchored) if sd_anchored is None else sd_anchored,
                all_arm_sds,
            )
    verdict = {
        "id": "B1",
        "metric": metric,
        "seeds": [int(seed) for seed in seeds],
        "plain_mean": _mean(values_plain),
        "anchored_mean": _mean(values_anchored),
        "delta_mean": delta,
        "seed_floor": floor,
        "exceeds_floor": bool(delta is not None and floor is not None
                              and delta > floor),
        "exceeds_two_floors": bool(delta is not None and floor is not None
                                   and delta > 2.0 * floor),
        "per_seed_plain_greater": per_seed,
        "all_seeds_plain_greater": bool(per_seed and all(per_seed.values())),
        "paired_differences": {
            str(seed): float(plain[seed]) - float(anchored[seed])
            for seed in seeds
        },
        "paired_difference_mean": (
            _mean([plain[seed] - anchored[seed] for seed in seeds])
            if seeds else None
        ),
        "paired_difference_sd": (
            _sd([plain[seed] - anchored[seed] for seed in seeds])
            if seeds else None
        ),
        "rule": (
            "v4 §3.5.9 B1: plain nuisance R2 > anchored in every seed; "
            "v4 §3.5.8 requires |delta mean| > 2 x the seed floor "
            "(the conservative reading of 'exceeds the seed-floor rule')"
        ),
    }
    if not seeds or delta is None:
        verdict["status"] = "not_evaluable"
        verdict["reason"] = "missing_paired_seeds"
    elif floor is None:
        verdict["status"] = "not_evaluable"
        verdict["reason"] = "missing_seed_floor"
    elif verdict["all_seeds_plain_greater"] and verdict["exceeds_two_floors"]:
        verdict["status"] = "pass"
        verdict["reason"] = None
    else:
        verdict["status"] = "fail"
        verdict["reason"] = (
            "not_every_seed" if not verdict["all_seeds_plain_greater"]
            else "delta_within_two_seed_floors"
        )
    return verdict


def _as_seed_map(values) -> Dict[int, float]:
    """Accept a seed-keyed dict or a seed-ordered sequence of values."""
    if isinstance(values, dict):
        return {
            int(seed): float(value) for seed, value in values.items()
            if value is not None and np.isfinite(value)
        }
    return {
        int(index): float(value)
        for index, value in enumerate(values)
        if value is not None and np.isfinite(value)
    }


def r7_bridge_verdict(
    plain: Dict[int, float],
    anchored: Dict[int, float],
    plain_physics: Dict[str, object],
    anchored_physics: Dict[str, object],
    *,
    tolerance: float = 1.1,
    all_arm_sds: Optional[Sequence[Optional[float]]] = None,
) -> dict:
    """R7 bridge half: nuisance down in every paired seed and physics kept."""
    seeds = sorted(set(plain) & set(anchored))
    per_seed = {str(seed): float(anchored[seed]) < float(plain[seed])
                for seed in seeds}
    physics = {}
    physics_ok = True
    physics_missing = False
    for parameter in PHYSICS_TARGETS:
        plain_map = _as_seed_map(plain_physics.get(parameter, {}))
        anchored_map = _as_seed_map(anchored_physics.get(parameter, {}))
        plain_mean = _mean(list(plain_map.values())) if plain_map else None
        anchored_mean = (
            _mean(list(anchored_map.values())) if anchored_map else None
        )
        if plain_mean is None or anchored_mean is None or plain_mean <= 0:
            physics[parameter] = {
                "plain_seed_mean": plain_mean,
                "anchored_seed_mean": anchored_mean,
                "ratio": None,
                "within_tolerance": None,
                "per_seed_ratio": {},
            }
            physics_ok = False
            physics_missing = True
            continue
        ratio = anchored_mean / plain_mean
        within = bool(ratio <= tolerance)
        physics_ok = physics_ok and within
        paired = sorted(set(plain_map) & set(anchored_map))
        physics[parameter] = {
            "plain_seed_mean": plain_mean,
            "anchored_seed_mean": anchored_mean,
            "ratio": float(ratio),
            "within_tolerance": within,
            "per_seed_ratio": {
                str(seed): (
                    None if plain_map[seed] <= 0
                    else float(anchored_map[seed] / plain_map[seed])
                )
                for seed in paired
            },
            "paired_ratio_mean": (
                _mean([
                    anchored_map[seed] / plain_map[seed]
                    for seed in paired if plain_map[seed] > 0
                ]) if paired else None
            ),
        }
    plain_mean = _mean([plain[seed] for seed in seeds])
    anchored_mean = _mean([anchored[seed] for seed in seeds])
    verdict = {
        "id": "R7",
        "half": "bridge",
        "metric": "nuisance_r2 (down) and physics_MSE_CRLB (kept)",
        "seeds": [int(seed) for seed in seeds],
        "plain_mean": plain_mean,
        "anchored_mean": anchored_mean,
        "nuisance_delta": (
            None if plain_mean is None or anchored_mean is None
            else plain_mean - anchored_mean
        ),
        "per_seed_anchored_lower": per_seed,
        "all_seeds_anchored_lower": bool(per_seed and all(per_seed.values())),
        "paired_differences": {
            str(seed): float(plain[seed]) - float(anchored[seed])
            for seed in seeds
        },
        "paired_difference_mean": (
            _mean([plain[seed] - anchored[seed] for seed in seeds])
            if seeds else None
        ),
        "paired_difference_sd": (
            _sd([plain[seed] - anchored[seed] for seed in seeds])
            if seeds else None
        ),
        "physics_tolerance": tolerance,
        "physics": physics,
        "physics_all_within_tolerance": bool(physics_ok),
        "rule": (
            "v4 §4 R7 (bridge half): anchored nuisance R2 < plain in every "
            "paired seed and anchored seed-mean physics_MSE_CRLB per parameter "
            "<= 1.1 x plain"
        ),
    }
    if all_arm_sds is not None and any(
        sd is not None and np.isfinite(sd) for sd in all_arm_sds
    ):
        verdict["seed_floor"] = seed_floor(
            _sd([plain[seed] for seed in seeds]),
            _sd([anchored[seed] for seed in seeds]),
            all_arm_sds,
        )
    if not seeds:
        verdict["status"] = "not_evaluable"
        verdict["reason"] = "missing_paired_seeds"
    elif physics_missing:
        verdict["status"] = "not_evaluable"
        verdict["reason"] = "missing_physics_target_or_crlb"
    elif verdict["all_seeds_anchored_lower"] and verdict["physics_all_within_tolerance"]:
        verdict["status"] = "pass"
        verdict["reason"] = None
    else:
        verdict["status"] = "fail"
        verdict["reason"] = (
            "not_every_seed" if not verdict["all_seeds_anchored_lower"]
            else "physics_tolerance_exceeded"
        )
    return verdict


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def _run_status(
    run: RunSpec,
    outcome: str,
    *,
    started_at: str,
    started_epoch: float,
    steps_completed: int,
    exit_status: int = 0,
    reason: Optional[str] = None,
    host: Optional[str] = None,
    device: Optional[str] = None,
    extension: Optional[str] = None,
) -> dict:
    finished_epoch = time.time()
    status = {
        "run_id": run.run_id,
        "arm": run.arm,
        "seed": int(run.seed),
        "cohort": run.cohort,
        "host": host or host_name(),
        "device": device,
        "outcome": outcome,
        "reason": reason,
        "exit_status": int(exit_status),
        "steps_completed": int(steps_completed),
        "steps_scheduled": int(run.steps),
        "started_at_utc": started_at,
        "finished_at_utc": utcnow(),
        "wall_seconds": round(finished_epoch - started_epoch, 2),
    }
    if extension is not None:
        status["extension"] = extension
        status["registered"] = False
    if toy_delay_extension():
        status.update(registered=False, toy_delay=toy_delay_extension())
    return status


def run_one(
    run: RunSpec,
    *,
    config: dict,
    products: dict,
    cohort: CohortData,
    out_root: Path,
    config_sha: str,
    code_sha: str,
    status_text: str,
    device: torch.device,
    fixture: str,
    evaluate: bool,
    resume: Optional[Path],
    stop_after: Optional[int],
    checkpoint_minutes: float,
    extension: Optional[str] = None,
) -> dict:
    """Train, freeze and evaluate one run; never writes outside the roots."""
    import random

    run_root = Path(out_root)
    run_dir = run_root / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = utcnow()
    started_epoch = time.time()
    status_path = run_dir / "run_status.json"
    atomic_write_json(
        status_path,
        _run_status(
            run, "running", started_at=started_at, started_epoch=started_epoch,
            steps_completed=0, host=host_name(), device=str(device),
            extension=extension,
        ),
    )
    row = run.as_dict()
    row.update(
        {
            "schema": SCHEMA_RUN,
            "config_sha256": config_sha,
            "code_sha": code_sha,
            "git_status": status_text,
            "host": host_name(),
            "device": str(device),
            "started_at_utc": started_at,
            "finished_at_utc": None,
            "wall_seconds": None,
            "steps_completed": 0,
            "fixture": fixture,
            "cohort_provenance": cohort_provenance(cohort),
            "products": products_provenance(products),
            "registered": False,
        }
    )
    if extension is not None:
        row["extension"] = extension
    if toy_delay_extension():
        row["toy_delay"] = toy_delay_extension()
    try:
        random.seed(run.seed)
        np.random.seed(run.seed % (2**32))
        torch.manual_seed(run.seed)
        model = build_model(
            config,
            run.arm,
            channel_mean=products["channel_mean"],
            channel_std=products["channel_std"],
            target_noise_covariance=products.get(
                "target_marginal", products["target_noise_covariance"]
            ),
            noise_bank=products["noise_bank"],
            beta=run.beta,
            seed=run.seed,
        ).to(device)
        optimizers = build_optimizers(model)
        # The cohort (not the arm) keys the batch stream, so every arm at a
        # given seed sees exactly the same events in the same order.
        batcher = EventBatcher(
            cohort.roles["subject_training"].raw,
            run.batch,
            run.seed,
            f"{run.cohort}/batches",
        )
        checkpoint_path = run_dir / "checkpoint.pt"
        training = train_model(
            model=model,
            optimizers=optimizers,
            batcher=batcher,
            run=run,
            checkpoint_path=checkpoint_path,
            config_sha=config_sha,
            data_key=cohort.key,
            device=device,
            resume=resume,
            stop_after=stop_after,
            checkpoint_minutes=checkpoint_minutes,
            log_path=run_dir / "train_log.jsonl",
        )
        row["steps_completed"] = int(training["step"])
        row["resumed_from_step"] = int(training["resumed_from_step"])
        if training["interrupted"]:
            row["outcome"] = "interrupted"
            row["interrupted_by"] = training["interrupted_by"]
            row["finished_at_utc"] = utcnow()
            row["wall_seconds"] = round(time.time() - started_epoch, 2)
            atomic_write_json(
                status_path,
                _run_status(
                    run, "interrupted", started_at=started_at,
                    started_epoch=started_epoch,
                    steps_completed=training["step"], exit_status=0,
                    reason=training["interrupted_by"], device=str(device),
                    extension=extension,
                ),
            )
            write_sha256sums(run_dir)
            _append_manifest(run_root, row)
            return {"manifest": row, "metrics": None, "training": training,
                    "run_dir": run_dir}
        if training["step"] != run.steps:  # pragma: no cover - defensive
            raise RuntimeError(
                f"run {run.run_id} stopped at {training['step']}, not the final "
                f"scheduled step {run.steps}"
            )

        metrics = None
        if evaluate:
            metrics = evaluate_run(
                model=model,
                run=run,
                cohort=cohort,
                config=config,
                run_dir=run_dir,
                device=device,
                products=products,
            )
        row["registered"] = bool(
            extension is None
            and not toy_delay_extension()
            and fixture == "a4"
            and not cohort.manifest["truncated"]
            and run.steps == int(config["training"]["steps_primary"])
            and run.batch == int(config["training"].get("batch", REGISTERED_BATCH))
        )
        row["outcome"] = "complete"
        row["finished_at_utc"] = utcnow()
        row["wall_seconds"] = round(time.time() - started_epoch, 2)
        row["checkpoint_sha256"] = sha256_file(checkpoint_path)
        if metrics is not None:
            row["metrics"] = _metrics_summary(metrics)
            row["embedding_files"] = metrics.get("embedding_files", {})
        atomic_write_json(
            status_path,
            _run_status(
                run, "complete", started_at=started_at,
                started_epoch=started_epoch, steps_completed=training["step"],
                exit_status=0, device=str(device), extension=extension,
            ),
        )
        write_sha256sums(run_dir)
        _append_manifest(run_root, row)
        return {
            "manifest": row,
            "metrics": metrics,
            "training": training,
            "run_dir": run_dir,
            "embedding_path": (
                None if metrics is None else metrics.get("embedding_path")
            ),
        }
    except Exception as error:
        row["outcome"] = "failed"
        row["failure"] = f"{type(error).__name__}: {error}"
        row["finished_at_utc"] = utcnow()
        row["wall_seconds"] = round(time.time() - started_epoch, 2)
        atomic_write_json(
            status_path,
            _run_status(
                run, "failed", started_at=started_at,
                started_epoch=started_epoch, steps_completed=row["steps_completed"],
                exit_status=1, reason=row["failure"], device=str(device),
                extension=extension,
            ),
        )
        _append_manifest(run_root, row)
        return {"manifest": row, "metrics": None, "training": None,
                "run_dir": run_dir, "error": row["failure"]}


def _append_manifest(root: Path, row: dict) -> None:
    path = Path(root) / "manifest.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _metrics_summary(metrics: dict) -> dict:
    probes = metrics["probes"]
    floor = metrics["random_init_floor"]
    return {
        "physics": {
            row["target"]: {
                "test_mse": row["test_mse"],
                "test_r2": row["test_r2"],
                "physics_mse_crlb": row["physics_mse_crlb"],
                "status": row["status"],
            }
            for row in probes["physics"]["rows"]
        },
        "nuisance": {
            "defined_mean_r2": probes["nuisance"]["defined_mean_r2"],
            "defined_target_set_complete": probes["nuisance"][
                "defined_target_set_complete"
            ],
            "rows": {
                row["target"]: {
                    "test_r2": row["test_r2"],
                    "status": row["status"],
                    "reason": row["reason"],
                }
                for row in probes["nuisance"]["rows"]
            },
        },
        "amplitude": {
            location: {
                "mean_r2": probes["amplitude"][location]["mean_r2"],
                "criterion_met": probes["amplitude"][location]["criterion_met"],
                "n_evaluated": probes["amplitude"][location]["n_evaluated"],
            }
            for location in ("post_time", "tokens")
        },
        "random_init_floor": {
            "physics_mse_crlb": {
                row["target"]: row["physics_mse_crlb"]
                for row in floor["physics"]["rows"]
            },
            "nuisance_defined_mean_r2": floor["nuisance"][
                "defined_mean_r2"
            ],
            "amplitude": {
                location: floor["amplitude"][location][
                    "mean_r2"
                ]
                for location in ("post_time", "tokens")
            },
        },
    }


def evaluate_run(
    *,
    model,
    run: RunSpec,
    cohort: CohortData,
    config: dict,
    run_dir: Path,
    device: torch.device,
    products: dict,
) -> dict:
    """Export float16 probe-set embeddings and fit the frozen read-out probes."""
    train_role = cohort.roles[cohort.spec.probe_train_role]
    test_role = cohort.roles[cohort.spec.probe_test_role]
    amplitude_train = train_role.amplitude_targets
    amplitude_test = test_role.amplitude_targets
    if amplitude_train is None or amplitude_test is None:
        raise RuntimeError(
            "cohort cache lacks the per-channel amplitude targets; remove the "
            "cache directory and regenerate it"
        )
    gains = np.asarray(
        cohort.manifest["views_manifest"].get("amplitude_gains")
        or [1.0] * 8,
        dtype=np.float64,
    )
    crlb_test = test_role.crlb_variances

    trained_train = encode_events(model, train_role.raw, device)
    trained_test = encode_events(model, test_role.raw, device)
    embedding_path = run_dir / "embeddings.npz"
    _atomic_savez(
        embedding_path,
        token_grid_train=trained_train["token_grid"],
        token_grid_test=trained_test["token_grid"],
        post_time_train=trained_train["post_time"],
        post_time_test=trained_test["post_time"],
        mean_tokens_train=trained_train["mean_tokens"],
        mean_tokens_test=trained_test["mean_tokens"],
        event_ids_train=np.asarray(train_role.event_ids, dtype=np.int64),
        event_ids_test=np.asarray(test_role.event_ids, dtype=np.int64),
    )
    params = dict(
        theta_train=train_role.theta,
        theta_test=test_role.theta,
        amplitude_train=amplitude_train,
        amplitude_test=amplitude_test,
        nuisance_train=train_role.nuisance,
        nuisance_test=test_role.nuisance,
        nuisance_valid=train_role.nuisance_valid,
        crlb_test=crlb_test,
    )
    probes = fit_probe_suite(train=trained_train, test=trained_test, **params)
    del trained_train, trained_test

    floor_seed = int(config["training"]["seeds"][run.arm][0])
    floor_model = build_model(
        config,
        run.arm,
        channel_mean=products["channel_mean"],
        channel_std=products["channel_std"],
        target_noise_covariance=products.get(
            "target_marginal", products["target_noise_covariance"]
        ),
        noise_bank=products["noise_bank"],
        beta=run.beta,
        seed=floor_seed,
    ).to(device)
    floor_train = encode_events(floor_model, train_role.raw, device)
    floor_test = encode_events(floor_model, test_role.raw, device)
    floor = fit_probe_suite(
        train=floor_train, test=floor_test, **params
    )
    del floor_train, floor_test

    payload = {
        "schema": "jepa_bridge_probes_v1",
        "run": run.as_dict(),
        "cohort": run.cohort,
        "probe_roles": {
            "train": cohort.spec.probe_train_role,
            "test": cohort.spec.probe_test_role,
            "n_train": int(len(train_role.ids)),
            "n_test": int(len(test_role.ids)),
        },
        "amplitude_targets": {
            "definition": (
                "max over the 256 samples of the noiseless per-channel waveform "
                "(views.waveform(theta) for A4; amplitude * gain for the fixture)"
            ),
            "gains": [float(g) for g in gains],
        },
        "probes": probes,
        "random_init_floor": floor,
        "random_init_floor_seed": floor_seed,
        "embedded_files": {"embeddings.npz": sha256_file(embedding_path)},
    }
    atomic_write_json(run_dir / "probes.json", payload)
    files = {
        "embeddings.npz": {
            "sha256": sha256_file(embedding_path),
            "bytes": embedding_path.stat().st_size,
        },
        "probes.json": {
            "sha256": sha256_file(run_dir / "probes.json"),
            "bytes": (run_dir / "probes.json").stat().st_size,
        },
    }
    return {
        "probes": payload["probes"],
        "random_init_floor": floor,
        "embedding_files": files,
        "embedding_path": embedding_path,
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _defined_nuisance_scores(nuisance: dict) -> dict:
    """Score a probe's nuisance rows over the cell's defined target set.

    At line = 0 the cell's defined nuisance targets are ``common_0`` and
    ``common_1`` (v4 §3.2.8); ``line_I``/``line_Q`` are recorded as undefined
    and never replaced by zero or silently averaged in.  The per-run probe
    payload records ``nuisance.declared`` and ``defined_target_set_complete``;
    this helper re-derives the score from the rows so a damaged or wrongly
    declared set is detected rather than trusted.
    """
    rows = {
        str(row.get("target")): row
        for row in nuisance.get("rows", [])
        if isinstance(row, dict)
    }
    scores: Dict[str, float] = {}
    missing: List[str] = []
    for target in DEFINED_NUISANCE_TARGETS:
        row = rows.get(target)
        value = None if row is None else row.get("test_r2")
        if (
            row is None
            or row.get("status") != "evaluated"
            or value is None
            or not np.isfinite(float(value))
        ):
            missing.append(target)
        else:
            scores[target] = float(value)
    return {
        "scores": scores,
        "mean": float(np.mean(list(scores.values()))) if scores else None,
        "missing": missing,
        "complete": not missing,
        "declared": list(nuisance.get("declared", [])),
        "recorded_complete": bool(
            nuisance.get("defined_target_set_complete", False)
        ),
    }


def _arm_tables(results: Sequence[dict]) -> dict:
    arms: Dict[str, dict] = {}
    for result in results:
        metrics = result.get("metrics")
        if not metrics:
            continue
        row = result["manifest"]
        arm = row["arm"]
        seed = int(row["seed"])
        entry = arms.setdefault(arm, {"seeds": {}})
        probes = metrics["probes"]
        nuisance = _defined_nuisance_scores(probes["nuisance"])
        nuisance_r2 = nuisance["mean"]
        physics = {
            par: probes["physics"]["rows"][index]["physics_mse_crlb"]
            for index, par in enumerate(PHYSICS_TARGETS)
        }
        amplitude = {
            location: probes["amplitude"][location]["mean_r2"]
            for location in ("post_time", "tokens")
        }
        floor = metrics["random_init_floor"]
        entry["seeds"][str(seed)] = {
            "run_id": row["run_id"],
            "nuisance_r2": nuisance_r2,
            "nuisance_targets": list(DEFINED_NUISANCE_TARGETS),
            "nuisance_target_r2": nuisance["scores"],
            "nuisance_missing_targets": nuisance["missing"],
            "nuisance_declared": nuisance["declared"],
            "nuisance_target_set_complete": nuisance["complete"],
            "random_init_floor": {
                "nuisance_r2": floor["nuisance"]["defined_mean_r2"],
                "physics_mse_crlb": {
                    floor_row["target"]: floor_row["physics_mse_crlb"]
                    for floor_row in floor["physics"]["rows"]
                },
                "amplitude": {
                    location: floor["amplitude"][location]["mean_r2"]
                    for location in ("post_time", "tokens")
                },
            },
            "physics_mse_crlb": physics,
            "amplitude_mean_r2": amplitude,
            "embedding_path": (
                None if result.get("embedding_path") is None
                else str(result["embedding_path"])
            ),
            "registered": row.get("registered", False),
        }
    for arm, entry in arms.items():
        seeds = entry["seeds"]
        entry["nuisance_r2"] = {
            "mean": _mean([v["nuisance_r2"] for v in seeds.values()]),
            "sd": _sd([v["nuisance_r2"] for v in seeds.values()]),
            "values": {k: v["nuisance_r2"] for k, v in seeds.items()},
        }
        entry["physics_mse_crlb"] = {
            par: {
                "mean": _mean([v["physics_mse_crlb"][par] for v in seeds.values()]),
                "sd": _sd([v["physics_mse_crlb"][par] for v in seeds.values()]),
                "values": {
                    k: v["physics_mse_crlb"][par] for k, v in seeds.items()
                },
            }
            for par in PHYSICS_TARGETS
        }
        entry["amplitude_mean_r2"] = {
            location: {
                "mean": _mean([
                    v["amplitude_mean_r2"][location] for v in seeds.values()
                ]),
                "values": {
                    k: v["amplitude_mean_r2"][location] for k, v in seeds.items()
                },
            }
            for location in ("post_time", "tokens")
        }
        entry["n_seeds"] = len(seeds)
        entry["nuisance_target_set_complete"] = all(
            value.get("nuisance_target_set_complete", False)
            for value in seeds.values()
        )
        entry["nuisance_missing_targets"] = sorted({
            target
            for value in seeds.values()
            for target in value.get("nuisance_missing_targets", [])
        })
        entry["nuisance_declared"] = sorted({
            target
            for value in seeds.values()
            for target in value.get("nuisance_declared", [])
        })
    return arms


def _seed_cka(arms: dict) -> dict:
    """Seed CKA from each run's ``embeddings.npz``, if present.

    Embeddings are streamed: at most one ``npz`` is opened at a time (a CKA
    pair needs two reduced probe-test matrices, not two full archives), and
    only ``mean_tokens_test`` is retained.  A missing or unreadable archive
    makes CKA alone ``not_evaluable``; it never blocks the rest of the
    aggregate.
    """
    cka: Dict[str, dict] = {}
    for arm, entry in arms.items():
        seeds = sorted(entry["seeds"], key=int)
        rows: Dict[str, dict] = {}
        missing = [
            int(seed) for seed in seeds
            if not entry["seeds"][seed].get("embedding_path")
            or not Path(entry["seeds"][seed]["embedding_path"]).is_file()
        ]
        if missing:
            cka[arm] = {
                "pairs": rows,
                "mean": None,
                "status": "not_evaluable",
                "reason": "embeddings_missing",
                "missing_seeds": missing,
                "note": (
                    "linear CKA between seeds of the same arm on probe-test "
                    "embeddings; embeddings.npz is not present in every run, "
                    "so CKA alone is not_evaluable"
                ),
            }
            continue
        stream_error = None
        try:
            for index, first in enumerate(seeds):
                for second in seeds[index + 1:]:
                    key = f"{first}-{second}"
                    with np.load(
                        entry["seeds"][first]["embedding_path"]
                    ) as data:
                        first_features = np.asarray(
                            data["mean_tokens_test"], dtype=np.float64
                        )
                    with np.load(
                        entry["seeds"][second]["embedding_path"]
                    ) as data:
                        second_features = np.asarray(
                            data["mean_tokens_test"], dtype=np.float64
                        )
                    try:
                        rows[key] = {
                            "value": linear_cka(
                                first_features, second_features
                            ),
                            "status": "computed",
                        }
                    except ValueError as pair_error:
                        rows[key] = {"value": None, "status": str(pair_error)}
                    del first_features, second_features
        except (OSError, ValueError, KeyError) as error:
            stream_error = error
        values = [row["value"] for row in rows.values() if row["value"] is not None]
        readable = stream_error is None
        cka[arm] = {
            "pairs": rows,
            "mean": float(np.mean(values)) if values else None,
            "status": (
                "computed" if readable and values
                else "not_evaluable"
            ),
            "reason": (
                None if readable
                else (
                    "embeddings_unreadable:"
                    f"{type(stream_error).__name__}:{stream_error}"
                )
            ),
            "note": "linear CKA between seeds of the same arm on probe-test embeddings",
        }
    return cka


def _registered(results: Sequence[dict]) -> bool:
    evaluated = [result for result in results if result.get("metrics")]
    return bool(evaluated) and all(
        result["manifest"].get("registered", False) for result in evaluated
    )


def _label_follow_up(aggregate: dict, extension: str) -> dict:
    """Label an extension aggregate as a follow-up that replaces no verdict.

    Every verdict keeps its computed status (for transparent reporting beside
    the registered result) but gains ``extension``, ``registered: false`` and
    ``verdict_role: follow_up``; B1 is labelled ``B1-ext`` so it can never be
    mistaken for the registered B1.
    """
    aggregate["extension"] = extension
    aggregate["registered_run"] = False
    aggregate["verdict_role"] = "follow_up"
    aggregate["replaces_registered_verdict"] = False
    aggregate["budget_gpu_hours_cap"] = B1_EXT_BUDGET_GPU_HOURS
    for key, verdict in aggregate.get("verdicts", {}).items():
        verdict["extension"] = extension
        verdict["registered"] = False
        verdict["verdict_role"] = "follow_up"
        if key == "B1":
            verdict["label"] = f"{extension}"
            verdict["criterion"] = (
                "reported beside the registered B1 under the same criterion; "
                "this labelled follow-up does not replace the registered B1")
        verdict["follow_up_note"] = (
            "labelled follow-up beside the registered result; it is never a "
            "registered verdict")
    return aggregate


def _extension_fields(extension: Optional[str]) -> dict:
    """Optional run-status fields; absent for every registered run."""
    if extension is None:
        return {}
    return {"extension": extension, "registered_run": False}


def build_aggregate(
    results: Sequence[dict],
    *,
    config: dict,
    config_sha: str,
    code_sha: str,
    status_text: str,
    cohorts: Dict[str, dict],
    fixture: str,
) -> dict:
    delay = toy_delay_extension()
    if delay and any(result["manifest"].get("toy_delay") != delay for result in results):
        raise RuntimeError("delayed aggregate refuses missing or mixed delay provenance")
    arms = _arm_tables(results)
    arm_sds = {
        arm: entry["nuisance_r2"]["sd"] for arm, entry in arms.items()
    }
    all_sds = list(arm_sds.values())
    scores = {arm: entry["nuisance_r2"]["values"] for arm, entry in arms.items()}
    plain = {int(k): v for k, v in scores.get("sigreg_jepa", {}).items()}
    anchored = {
        int(k): v for k, v in scores.get("sigreg_jepa_anchored", {}).items()
    }
    verdicts = {}
    verdicts["B1"] = b1_verdict(
        plain,
        anchored,
        all_sds,
        sd_plain=arm_sds.get("sigreg_jepa"),
        sd_anchored=arm_sds.get("sigreg_jepa_anchored"),
    )
    plain_physics = {
        par: dict(arms.get("sigreg_jepa", {}).get(
            "physics_mse_crlb", {}).get(par, {}).get("values", {}))
        for par in PHYSICS_TARGETS
    }
    anchored_physics = {
        par: dict(arms.get("sigreg_jepa_anchored", {}).get(
            "physics_mse_crlb", {}).get(par, {}).get("values", {}))
        for par in PHYSICS_TARGETS
    }
    verdicts["R7"] = r7_bridge_verdict(
        plain, anchored, plain_physics, anchored_physics, all_arm_sds=all_sds
    )
    required = ("sigreg_jepa", "sigreg_jepa_anchored")
    missing_arms = [arm for arm in required if arm not in arms]
    missing_targets: List[str] = []
    for arm in required:
        if arm not in arms:
            continue
        for target in arms[arm].get("nuisance_missing_targets", []):
            if target not in missing_targets:
                missing_targets.append(target)
    if missing_arms:
        aggregate_incomplete_reason = (
            "missing_required_arm:" + ",".join(missing_arms)
        )
    elif missing_targets:
        aggregate_incomplete_reason = (
            "missing_required_nuisance_target:" + ",".join(missing_targets)
        )
    else:
        aggregate_incomplete_reason = None
    if aggregate_incomplete_reason:
        for key in ("B1", "R7"):
            verdicts[key]["status"] = "not_evaluable"
            verdicts[key]["reason"] = aggregate_incomplete_reason
            verdicts[key]["nuisance_missing_targets"] = list(missing_targets)
    mae = arms.get("mae_chi2", {})
    aggregate = {
        "schema": SCHEMA_AGGREGATE,
        "generated_at_utc": utcnow(),
        "config_sha256": config_sha,
        "code_sha": code_sha,
        "git_status": status_text,
        "host": host_name(),
        "fixture": fixture,
        "registered_run": _registered(results),
        "registered_run_note": (
            "a registered bridge run is A4 data, full ID ranges, 10000 steps "
            "and batch 128 (v3 §6 WP4; v4 §3.5.1 keeps the bridge as that "
            "Tier-1 E2); anything else is a smoke"
        ),
        "config_batch_conflict": {
            "config_training_batch": config["training"].get("batch"),
            "registered_batch": REGISTERED_BATCH,
            "resolution": (
                "the bridge uses the frozen config's batch 128 = v3 §6 WP4's "
                "registered Tier-1 E2 batch (v4 §3.5.1); v4 §6.1's batch 32 is "
                "the LUCiD encoder's registered batch (§3.5.5).  The config is "
                "not edited; a run at another batch is provisional."
            ),
        },
        "cohorts": cohorts,
        "runs": [
            {
                "run_id": result["manifest"]["run_id"],
                "arm": result["manifest"]["arm"],
                "seed": result["manifest"]["seed"],
                "outcome": result["manifest"]["outcome"],
                "wall_seconds": result["manifest"].get("wall_seconds"),
                "steps_completed": result["manifest"].get("steps_completed"),
            }
            for result in results
        ],
        "arms": arms,
        "mae_chi2_beside_r7": {
            "nuisance_r2_mean": mae.get("nuisance_r2", {}).get("mean"),
            "physics_mse_crlb_mean": {
                par: mae.get("physics_mse_crlb", {}).get(par, {}).get("mean")
                for par in PHYSICS_TARGETS
            },
            "note": "reported beside R7 without a pass criterion (v4 §3.4 A6)",
        },
        "seed_floor_definition": (
            "largest of SD_a, SD_b and the median SD over all arms in the cell "
            "(v4 §3.5.8; SD uses ddof=1)"
        ),
        "nuisance_defined_targets": list(DEFINED_NUISANCE_TARGETS),
        "nuisance_defined_targets_note": (
            "B1/R7 use the cell's defined nuisance target set at line = 0 "
            "({common_0, common_1}); line_I/line_Q are undefined there and "
            "are never averaged in (v4 §3.2.8)"
        ),
        "seed_cka": _seed_cka(arms),
        "verdicts": verdicts,
        "files": {"manifest_rows": len(results)},
    }
    if delay:
        aggregate.update(registered=False, registered_run=False, toy_delay=delay,
                         extension="toy_delay_20261025", verdict_role="descriptive_extension",
                         replaces_registered_verdict=False, budget_gpu_hours_cap=8)
        for verdict in aggregate["verdicts"].values():
            verdict.update(registered=False, verdict_role="descriptive_extension",
                           replaces_registered_verdict=False)
    return aggregate


def predictions_fragment(
    aggregate: dict, *, out_dir: Path, mode: str = "grid"
) -> str:
    b1 = aggregate["verdicts"]["B1"]
    r7 = aggregate["verdicts"]["R7"]
    registered = aggregate["registered_run"]
    extension = aggregate.get("extension")

    def fmt(value, digits=4):
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}g}"

    lines = [
        "# PREDICTIONS_UPDATE — WP4a Tier-1 bridge",
        "",
        f"Generated {aggregate['generated_at_utc']} by "
        f"`jepa_noise.bridge_run {mode}`; config sha256 "
        f"`{aggregate['config_sha256']}`; code `{aggregate['code_sha']}`.",
        "",
        "Registration: runbook v4 §3.5.9 B1 and the bridge half of R7. "
        "`PREDICTION_STATUS.json` is deliberately **not** edited here; the "
        "Lead merges this fragment.",
        "",
        *([] if extension is None else [
            f"**Labelled follow-up:** `{extension}` (registered: false). These "
            "verdicts are reported beside the registered result and never "
            "replace it.",
            "",
        ]),
        f"**Registered run:** `{registered}`.  A registered bridge run is A4 "
        "data, full cohort ID ranges, 10000 scheduled steps and batch 128 "
        "(v4 §3.5.7, §6.1).  Smoke/truncated runs are marked "
        "`provisional` below.",
        "",
        "Run outcomes: "
        + ", ".join(
            f"`{aggregate['runs'][index]['run_id']}` = "
            f"{aggregate['runs'][index]['outcome']}"
            for index in range(len(aggregate["runs"]))
        )
        + ".",
        "",
        "| ID | Comparison | Value | Criterion | Status |",
        "|---|---|---|---|---|",
    ]
    status_b1 = (
        f"follow_up:{b1['status']}" if extension is not None
        else ("provisional" if not registered else b1["status"]))
    lines.append(
        "| B1 | `sigreg_jepa` nuisance R² > `sigreg_jepa_anchored` | "
        f"plain {fmt(b1['plain_mean'])}, anchored {fmt(b1['anchored_mean'])}, "
        f"delta {fmt(b1['delta_mean'])}, seed floor {fmt(b1['seed_floor'])} "
        f"(2x = {fmt(None if b1['seed_floor'] is None else 2 * b1['seed_floor'])}) "
        f"| every seed, |delta| > 2 x seed floor | "
        f"**{status_b1}** ({b1['status']}; {b1['reason'] or 'criterion met'}) |"
    )
    status_r7 = (
        f"follow_up:{r7['status']}" if extension is not None
        else ("provisional" if not registered else r7["status"]))
    physics = ", ".join(
        f"{par} {fmt(row['ratio'], 3)}x"
        for par, row in r7["physics"].items()
    )
    lines.append(
        "| R7 | bridge half: anchored nuisance R² < plain and physics kept | "
        f"nuisance delta {fmt(r7['nuisance_delta'])}; ratios {physics} "
        f"| every paired seed, physics ratio <= 1.1 | "
        f"**{status_r7}** ({r7['status']}; {r7['reason'] or 'criterion met'}) |"
    )
    lines += [
        "",
        "Per-parameter physics MSE/CRLB (seed mean, anchored / plain):",
        "",
        "| parameter | anchored | plain | ratio |",
        "|---|---|---|---|",
    ]
    for par, row in r7["physics"].items():
        lines.append(
            f"| {par} | {fmt(row['anchored_seed_mean'])} | "
            f"{fmt(row['plain_seed_mean'])} | {fmt(row['ratio'], 3)} |"
        )
    lines += [
        "",
        "Arm seed means (nuisance R² over the defined targets):",
        "",
        "| arm | seeds | nuisance R² mean | SD | random-init floor |",
        "|---|---|---|---|---|",
    ]
    for arm, entry in aggregate["arms"].items():
        first_seed = next(iter(entry["seeds"].values()), {})
        floor = first_seed.get("random_init_floor") or {}
        lines.append(
            f"| `{arm}` | {entry['n_seeds']} | "
            f"{fmt(entry['nuisance_r2']['mean'])} | "
            f"{fmt(entry['nuisance_r2']['sd'])} | "
            f"{fmt(floor.get('nuisance_r2'))} |"
        )
    lines += [
        "",
        "Random-init floor per probe (same architecture, init seed = the arm's "
        "first registered seed; one model per arm, identical across its seeds):",
        "",
        "| arm | nuisance R² | physics MSE/CRLB (amp / t0 / tau_t) | "
        "amplitude R² (post-time / tokens) |",
        "|---|---|---|---|",
    ]
    for arm, entry in aggregate["arms"].items():
        first_seed = next(iter(entry["seeds"].values()), {})
        floor = first_seed.get("random_init_floor") or {}
        physics_floor = floor.get("physics_mse_crlb") or {}
        amplitude_floor = floor.get("amplitude") or {}
        ratios = " / ".join(
            fmt(physics_floor.get(par)) for par in PHYSICS_TARGETS
        )
        lines.append(
            f"| `{arm}` | {fmt(floor.get('nuisance_r2'))} | {ratios} | "
            f"{fmt(amplitude_floor.get('post_time'))} / "
            f"{fmt(amplitude_floor.get('tokens'))} |"
        )
    lines += [
        "",
        "```json",
        json.dumps(
            _jsonable(
                {
                    "B1": b1,
                    "R7_bridge": r7,
                    "registered_run": registered,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    text = "\n".join(lines) + "\n"
    atomic_write_text(Path(out_dir) / "PREDICTIONS_UPDATE.md", text)
    return text


# ---------------------------------------------------------------------------
# beta development-cohort selection
# ---------------------------------------------------------------------------
def write_beta_decision(
    path: Path,
    *,
    selected_beta: float,
    table: Sequence[dict],
    config: dict,
    config_sha: str,
    code_sha: str,
    cohort_prov: dict,
    seed: int,
    steps: int,
    fixture: str,
    registered: bool,
    extension: Optional[str] = None,
    beta_grid: Optional[Sequence[float]] = None,
    tolerance: Optional[float] = None,
    plain_reference: Optional[dict] = None,
    selection: Optional[dict] = None,
) -> dict:
    """Write a beta decision; the extension form is opt-in and never registered.

    The registered form (``extension=None``) is unchanged.  The B1-ext form
    records ``extension``, the plain reference path+sha256, the physics
    tolerance, the full table and ``registered: false``; it refuses a missing
    beta (``none_qualified`` is reported, never frozen) and any beta outside
    the extension grid.
    """
    extension = None if extension is None else str(extension)
    grid = [float(value) for value in (
        beta_grid if beta_grid is not None
        else config["objective"]["anchor"]["beta_grid"])]
    if selected_beta is None:
        raise ValueError(
            "no beta is frozen when no beta qualifies (none_qualified)")
    if float(selected_beta) not in grid:
        raise ValueError(
            f"selected beta {selected_beta} is not in the frozen grid {grid}"
        )
    if extension is not None:
        if extension != B1_EXTENSION:
            raise ValueError(f"unknown extension {extension!r}")
        if registered:
            raise ValueError(
                f"{B1_EXTENSION} is a labelled follow-up and cannot be registered")
        if tolerance is None or not np.isfinite(float(tolerance)):
            raise ValueError(f"{B1_EXTENSION} requires a finite physics tolerance")
        if not isinstance(plain_reference, dict):
            raise ValueError(f"{B1_EXTENSION} requires the plain reference provenance")
        for key in ("path", "file", "sha256"):
            if not plain_reference.get(key):
                raise ValueError(
                    f"{B1_EXTENSION} plain reference lacks {key!r}")
        if not isinstance(selection, dict):
            raise ValueError(f"{B1_EXTENSION} requires the selection record")
        payload = {
            "schema": SCHEMA_BETA,
            "extension": extension,
            "created_at_utc": utcnow(),
            "selected_beta": float(selected_beta),
            "beta_grid": grid,
            "tolerance": float(tolerance),
            "budget_gpu_hours_cap": B1_EXT_BUDGET_GPU_HOURS,
            "plain_reference": dict(plain_reference),
            "rule": B1_EXT_TIE_RULE,
            "selection": selection,
            "cohort": cohort_prov,
            "model_seed": int(seed),
            "steps": int(steps),
            "fixture": fixture,
            "registered": False,
            "config_sha256": config_sha,
            "code_sha": code_sha,
            "table": [
                {
                    "beta": float(row["beta"]),
                    "score": float(row["score"]),
                    "physics_mse": {
                        par: float(row["physics_mse"][par])
                        for par in PHYSICS_TARGETS
                    },
                    "physics_mse_crlb": {
                        par: float(row["physics_mse_crlb"][par])
                        for par in PHYSICS_TARGETS
                    },
                    "ratio_to_plain": {
                        par: float(
                            row.get("ratio_to_plain", {}).get(par)
                            or selection["per_beta"][str(float(row["beta"]))][
                                "ratio_to_plain"][par])
                        for par in PHYSICS_TARGETS
                    },
                    "qualifies": bool(
                        row.get("qualifies", selection["per_beta"][
                            str(float(row["beta"]))]["qualifies"])),
                }
                for row in table
            ],
            "status": (
                f"selected; {B1_EXTENSION} follow-up beta (labelled, "
                "registered: false; does not replace B1)"
            ),
        }
        atomic_write_json(path, payload)
        return payload
    payload = {
        "schema": SCHEMA_BETA,
        "created_at_utc": utcnow(),
        "selected_beta": float(selected_beta),
        "beta_grid": grid,
        "rule": (
            "minimise the mean over physics parameters of validation MSE "
            "divided by the development probe-training variance; ties within "
            "1e-8 relative favour the smaller coefficient (v3 §1.3, v4 §3.5.7)"
        ),
        "cohort": cohort_prov,
        "model_seed": int(seed),
        "steps": int(steps),
        "fixture": fixture,
        "registered": bool(registered),
        "config_sha256": config_sha,
        "code_sha": code_sha,
        "table": [
            {
                "beta": float(row["beta"]),
                "score": float(row["score"]),
                "physics_mse": {
                    par: float(row["physics_mse"][par]) for par in PHYSICS_TARGETS
                },
                "probe_train_variance": {
                    par: float(row["probe_train_variance"][par])
                    for par in PHYSICS_TARGETS
                },
            }
            for row in table
        ],
        "status": (
            "selected; sigreg_jepa_anchored on the bridge grid may start"
            if registered
            else "provisional smoke/beta-dev run: not a registered selection"
        ),
    }
    atomic_write_json(path, payload)
    return payload


def load_beta_decision(
    path: Path, config: dict, config_sha: str, *,
    expect_extension: Optional[str] = None,
) -> dict:
    """Load a beta decision, refusing any cross-use of extension/registered.

    A normal (non-extension) grid calls this with the default
    ``expect_extension=None`` and therefore refuses a B1-ext decision file.  An
    extension grid passes ``expect_extension="B1-ext"`` and refuses a
    registered decision in turn.
    """
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"beta decision {path} not found: sigreg_jepa_anchored may not run "
            "until beta is selected on the independent development cohort "
            "(v4 §3.5.7 'beta', config development_cohort.status)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA_BETA:
        raise RuntimeError(f"{path} is not a bridge beta decision")
    extension = payload.get("extension")
    if extension is not None:
        if extension != B1_EXTENSION:
            raise RuntimeError(
                f"beta decision {path} declares unknown extension {extension!r}")
        if expect_extension != extension:
            raise RuntimeError(
                f"beta decision {path} is a {extension} labelled follow-up, "
                "not a registered beta decision; it is accepted only by an "
                f"explicit `--extension {extension}` invocation")
        grid = [float(value) for value in B1_EXT_BETA_GRID]
        beta = payload.get("selected_beta")
        if beta is None:
            raise RuntimeError(
                f"{extension} decision {path} records no selected beta")
        if float(beta) not in grid:
            raise RuntimeError(
                f"beta decision {path} records {beta!r}, not a value of the "
                f"{extension} grid {grid}")
        if payload.get("registered") is not False:
            raise RuntimeError(
                f"{extension} decision {path} must record registered: false")
        if payload.get("tolerance") is None:
            raise RuntimeError(
                f"{extension} decision {path} records no physics tolerance")
        if not (payload.get("plain_reference") or {}).get("sha256"):
            raise RuntimeError(
                f"{extension} decision {path} records no plain-reference sha256")
        if payload.get("config_sha256") != config_sha:
            raise RuntimeError(
                f"beta decision {path} was made under config "
                f"{payload.get('config_sha256')!r}, not {config_sha!r}")
        if payload.get("cohort", {}).get("name") != "development":
            raise RuntimeError(
                f"beta decision {path} is not from the development cohort")
        return payload
    if expect_extension is not None:
        raise RuntimeError(
            f"beta decision {path} is a registered decision, not an "
            f"{expect_extension} extension decision")
    grid = [float(v) for v in config["objective"]["anchor"]["beta_grid"]]
    beta = payload.get("selected_beta")
    if beta is None or float(beta) not in grid:
        raise RuntimeError(
            f"beta decision {path} records {beta!r}, not a value of the frozen "
            f"grid {grid}"
        )
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError(
            f"beta decision {path} was made under config "
            f"{payload.get('config_sha256')!r}, not {config_sha!r}"
        )
    if payload.get("cohort", {}).get("name") != "development":
        raise RuntimeError(
            f"beta decision {path} is not from the development cohort"
        )
    return payload


def run_beta_dev(args, config: dict, config_sha: str, code_sha: str,
                 status_text: str) -> int:
    extension = getattr(args, "extension", None)
    selection_rule = getattr(args, "selection_rule", None)
    plain_reference = getattr(args, "plain_reference", None)
    if extension not in (None, B1_EXTENSION):
        raise SystemExit(f"unknown --extension {extension!r}")
    if extension is None:
        if selection_rule is not None or plain_reference is not None:
            raise SystemExit(
                "--selection-rule/--plain-reference apply only to the opt-in "
                f"--extension {B1_EXTENSION} beta-dev"
            )
    else:
        if selection_rule != "lucid_physics_tolerance":
            raise SystemExit(
                f"{B1_EXTENSION} requires --selection-rule "
                "lucid_physics_tolerance"
            )
        if not plain_reference:
            raise SystemExit(
                f"{B1_EXTENSION} requires --plain-reference <run_dir or metrics JSON>"
            )
    if args.fixture != "a4":
        raise SystemExit(
            "beta-dev requires the registered A4 law (--fixture a4); the "
            "synthetic fixture cannot select a reported beta"
        )
    if args.no_evaluate:
        raise SystemExit(
            "beta-dev needs probe fitting on the development cohort; "
            "--no-evaluate is a training-only smoke switch"
        )
    if args.resume:
        raise SystemExit(
            "beta-dev runs each beta from scratch under the same run id; "
            "--resume is for grid runs"
        )
    specs = build_cohort_specs(config)
    cohort = ensure_cohort(
        specs["development"], Path(args.data_root), args.max_events,
        args.fixture, config,
    )
    products = prepare_products(config, args.fixture, Path(args.data_root))
    device = configure_runtime(args.device)
    registered_grid = [float(v) for v in
                       config["objective"]["anchor"]["beta_grid"]]
    extension_grid = [float(value) for value in B1_EXT_BETA_GRID]
    if args.beta_grid:
        grid = [float(value) for value in args.beta_grid.split(",")]
        if extension is not None:
            if sorted(set(grid)) != sorted(extension_grid) or len(grid) != len(
                    extension_grid):
                raise SystemExit(
                    f"--beta-grid {grid} differs from the {B1_EXTENSION} grid "
                    f"{extension_grid}; the follow-up grid is fixed"
                )
            grid = sorted(grid)
        elif grid != registered_grid:
            raise SystemExit(
                f"--beta-grid {grid} differs from the frozen grid "
                f"{registered_grid}; the grid is not tunable"
            )
    elif extension is not None:
        raise SystemExit(
            f"{B1_EXTENSION} requires --beta-grid 10,100,1000")
    else:
        grid = registered_grid
    plain_values = None
    plain_prov = None
    if extension is not None:
        plain_values, plain_prov = read_plain_physics_reference(plain_reference)
    steps = int(args.steps or config["objective"]["anchor"][
        "development_cohort"]["steps"])
    seed = int(config["objective"]["anchor"]["development_cohort"][
        "views_constructor"]["model_seed"])
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    beta_started_at = utcnow()
    atomic_write_json(
        out_root / "run_status.json",
        {
            "run_id": out_root.name,
            "mode": "beta-dev",
            "host": host_name(),
            "device": str(device),
            "outcome": "running",
            "started_at_utc": beta_started_at,
            "betas": grid,
            "model_seed": seed,
            "steps": steps,
            **_extension_fields(extension),
        },
    )
    probe_train_theta = np.asarray(
        cohort.roles["probe_training"].theta, dtype=np.float64
    )
    train_variances = {
        par: float(np.var(probe_train_theta[:, index]))
        for index, par in enumerate(PHYSICS_TARGETS)
    }
    results = []
    table = []
    for beta in grid:
        run = RunSpec(
            run_id=f"wp4a_dev_anchored_s{seed}_b{beta:g}",
            arm="sigreg_jepa_anchored",
            seed=seed,
            cohort="development",
            steps=steps,
            batch=int(args.batch or REGISTERED_BATCH),
            beta=float(beta),
        )
        result = run_one(
            run,
            config=config,
            products=products,
            cohort=cohort,
            out_root=out_root,
            config_sha=config_sha,
            code_sha=code_sha,
            status_text=status_text,
            device=device,
            fixture=args.fixture,
            evaluate=not args.no_evaluate,
            resume=None,
            stop_after=args.stop_after,
            checkpoint_minutes=float(args.checkpoint_minutes),
            extension=extension,
        )
        results.append(result)
        metrics = result.get("metrics")
        if metrics is None:
            if result["manifest"]["outcome"] == "interrupted":
                continue
            raise SystemExit(
                f"beta-dev run {run.run_id} failed: "
                f"{result['manifest'].get('failure')}"
            )
        probes = metrics["probes"]
        physics_mse = {
            par: probes["physics"]["rows"][index]["test_mse"]
            for index, par in enumerate(PHYSICS_TARGETS)
        }
        if any(value is None for value in physics_mse.values()):
            raise SystemExit(
                f"beta-dev run {run.run_id} has no validation MSE for a "
                "physics parameter"
            )
        physics_crlb = {
            par: probes["physics"]["rows"][index].get("physics_mse_crlb")
            for index, par in enumerate(PHYSICS_TARGETS)
        }
        if extension is not None and any(
                value is None for value in physics_crlb.values()):
            raise SystemExit(
                f"beta-dev run {run.run_id} has no physics MSE/CRLB for a "
                "physics parameter; the LUCiD rule cannot be applied"
            )
        score = float(np.mean([
            physics_mse[par] / train_variances[par] for par in PHYSICS_TARGETS
        ]))
        table.append(
            {
                "beta": float(beta),
                "score": score,
                "physics_mse": physics_mse,
                "physics_mse_crlb": {
                    par: (None if value is None else float(value))
                    for par, value in physics_crlb.items()
                },
                "probe_train_variance": dict(train_variances),
            }
        )
    if len(table) != len(grid):
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "reason": "one or more beta-dev runs did not complete",
                    "table": table,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    if extension is not None:
        selection = select_beta_lucid_physics(
            table, plain_values, tolerance=B1_EXT_PHYSICS_TOLERANCE)
        selection["plain_reference"] = plain_prov
        # Carry the qualification markers into the full table (the decision
        # records the complete table, including betas that do not qualify).
        by_beta = selection["per_beta"]
        for row in table:
            entry = by_beta[str(float(row["beta"]))]
            row["ratio_to_plain"] = entry["ratio_to_plain"]
            row["qualifies"] = entry["qualifies"]
        if selection["status"] == "none_qualified":
            # No beta is frozen: report the follow-up as not evaluable rather
            # than writing a decision that silently picks a beta.
            atomic_write_json(
                out_root / "aggregate.json",
                {
                    "schema": "jepa_bridge_beta_dev_v1",
                    "extension": extension,
                    "registered_run": False,
                    "generated_at_utc": utcnow(),
                    "config_sha256": config_sha,
                    "code_sha": code_sha,
                    "git_status": status_text,
                    "status": "none_qualified",
                    "selection": selection,
                    "tolerance": B1_EXT_PHYSICS_TOLERANCE,
                    "plain_reference": plain_prov,
                    "table": table,
                    "decision_path": None,
                    "runs": [
                        {
                            "run_id": result["manifest"]["run_id"],
                            "outcome": result["manifest"]["outcome"],
                            "beta": result["manifest"]["beta"],
                            "wall_seconds": result["manifest"].get("wall_seconds"),
                        }
                        for result in results
                    ],
                },
            )
            atomic_write_json(
                out_root / "run_status.json",
                {
                    "run_id": out_root.name,
                    "mode": "beta-dev",
                    "host": host_name(),
                    "device": str(device),
                    "outcome": "none_qualified",
                    "exit_status": 0,
                    "started_at_utc": beta_started_at,
                    "finished_at_utc": utcnow(),
                    "runs_complete": len(table),
                    "runs_requested": len(grid),
                    "selected_beta": None,
                    "extension": extension,
                    "registered_run": False,
                },
            )
            print(
                json.dumps(
                    {
                        "status": "none_qualified",
                        "selected_beta": None,
                        "extension": extension,
                        "beta_decision": None,
                        "registered_run": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
    else:
        selection = select_beta(table)
    cohort_prov = cohort_provenance(cohort)
    registered = bool(
        extension is None
        and args.fixture == "a4"
        and not cohort.manifest["truncated"]
        and steps == int(config["objective"]["anchor"]["development_cohort"]["steps"])
        and int(args.batch or REGISTERED_BATCH) == REGISTERED_BATCH
    )
    decision_path = Path(args.beta_decision_out or (out_root / "beta_decision.json"))
    decision = write_beta_decision(
        decision_path,
        selected_beta=selection["selected_beta"],
        table=table,
        config=config,
        config_sha=config_sha,
        code_sha=code_sha,
        cohort_prov=cohort_prov,
        seed=seed,
        steps=steps,
        fixture=args.fixture,
        registered=registered,
        extension=extension,
        beta_grid=(grid if extension is not None else None),
        tolerance=(B1_EXT_PHYSICS_TOLERANCE if extension is not None else None),
        plain_reference=(plain_prov if extension is not None else None),
        selection=(selection if extension is not None else None),
    )
    atomic_write_json(
        out_root / "aggregate.json",
        {
            "schema": "jepa_bridge_beta_dev_v1",
            "generated_at_utc": utcnow(),
            "config_sha256": config_sha,
            "code_sha": code_sha,
            "git_status": status_text,
            "extension": extension,
            "registered_run": registered,
            "tolerance": (
                B1_EXT_PHYSICS_TOLERANCE if extension is not None else None),
            "plain_reference": plain_prov,
            "budget_gpu_hours_cap": (
                B1_EXT_BUDGET_GPU_HOURS if extension is not None else None),
            "selection": selection,
            "table": table,
            "decision_path": str(decision_path),
            "decision_sha256": sha256_file(decision_path),
            "runs": [
                {
                    "run_id": result["manifest"]["run_id"],
                    "outcome": result["manifest"]["outcome"],
                    "beta": result["manifest"]["beta"],
                    "wall_seconds": result["manifest"].get("wall_seconds"),
                }
                for result in results
            ],
        },
    )
    atomic_write_json(
        out_root / "run_status.json",
        {
            "run_id": out_root.name,
            "mode": "beta-dev",
            "host": host_name(),
            "device": str(device),
            "outcome": "complete",
            "exit_status": 0,
            "started_at_utc": beta_started_at,
            "finished_at_utc": utcnow(),
            "runs_complete": len(table),
            "runs_requested": len(grid),
            "selected_beta": decision["selected_beta"],
            "beta_decision_sha256": sha256_file(decision_path),
            **_extension_fields(extension),
        },
    )
    printed = {
        "status": "selected",
        "selected_beta": decision["selected_beta"],
        "extension": extension,
        "beta_decision": str(decision_path),
        "registered_run": registered,
    }
    if extension is None:
        printed["tied_betas"] = selection["tied_betas"]
        printed["scores"] = selection["scores"]
    else:
        printed["tied_betas"] = selection.get("tied_betas")
        printed["qualified_betas"] = selection["qualified_betas"]
        printed["tie_kind"] = selection.get("tie_kind")
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def load_manifest_rows(out_root: Path) -> Dict[str, dict]:
    """Latest manifest row per run_id (resumed runs append rows)."""
    path = Path(out_root) / "manifest.jsonl"
    rows: Dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["run_id"])] = row
    return rows


def collect_results(out_root: Path) -> List[dict]:
    """Rebuild completed run results from disk (for multi-job grids/resumes)."""
    out_root = Path(out_root)
    results = []
    for run_id, row in sorted(load_manifest_rows(out_root).items()):
        if row.get("outcome") != "complete":
            continue
        run_dir = out_root / "runs" / run_id
        probes_path = run_dir / "probes.json"
        embedding_path = run_dir / "embeddings.npz"
        if not probes_path.exists() or not embedding_path.exists():
            continue
        payload = json.loads(probes_path.read_text(encoding="utf-8"))
        results.append(
            {
                "manifest": row,
                "metrics": {
                    "probes": payload["probes"],
                    "random_init_floor": payload["random_init_floor"],
                    "embedding_files": payload.get("embedded_files", {}),
                },
                "embedding_path": embedding_path,
            }
        )
    return results


def _completed_run_ids_on_disk(out_root: Path) -> set:
    """Run ids whose own run_status says complete and whose probes.json exists.

    Guards the "grid mode re-executed a completed run" incident: a run dir can
    exist without a manifest row in this out root (combined/symlinked roots), so
    the manifest-only check is not enough.  ``--force`` remains the explicit
    override.
    """
    completed: set = set()
    runs_root = Path(out_root) / "runs"
    if not runs_root.is_dir():
        return completed
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        probes = run_dir / "probes.json"
        status_path = run_dir / "run_status.json"
        if not probes.exists() or not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(status.get("outcome")) == "complete":
            completed.add(run_dir.name)
    return completed


def run_grid(args, config: dict, config_sha: str, code_sha: str,
             status_text: str) -> int:
    extension = getattr(args, "extension", None)
    if extension not in (None, B1_EXTENSION):
        raise SystemExit(f"unknown --extension {extension!r}")
    arms = args.arms.split(",") if args.arms else list(ALL_ARMS)
    for arm in arms:
        if arm not in ALL_ARMS:
            raise SystemExit(f"unknown arm {arm!r}; registered: {ALL_ARMS}")
    if extension is not None and "sigreg_jepa_anchored" not in arms:
        raise SystemExit(
            f"--extension {extension} runs the anchored arm at the selected "
            "beta; include sigreg_jepa_anchored in --arms")
    seeds_by_arm = config["training"]["seeds"]
    if args.seeds:
        seeds = [int(value) for value in args.seeds.split(",")]
    else:
        seeds = None
    beta = None
    beta_source = None
    beta_decision = None
    if "sigreg_jepa_anchored" in arms:
        decision = load_beta_decision(
            Path(args.beta_decision), config, config_sha,
            expect_extension=extension,
        )
        beta_decision = decision
        beta = float(decision["selected_beta"])
        beta_source = {
            "decision_path": str(args.beta_decision),
            "decision_sha256": sha256_file(args.beta_decision),
            "decision_registered": bool(decision.get("registered", False)),
        }
        if extension is not None:
            beta_source.update({
                "extension": extension,
                "plain_reference_sha256": (
                    decision.get("plain_reference") or {}).get("sha256"),
                "tolerance": decision.get("tolerance"),
            })
        print(
            f"bridge_run: anchored arm uses selected beta={beta} from "
            f"{args.beta_decision}",
            file=sys.stderr,
        )
    specs = build_cohort_specs(config)
    cohort = ensure_cohort(
        specs["evaluation"], Path(args.data_root), args.max_events,
        args.fixture, config,
    )
    products = prepare_products(config, args.fixture, Path(args.data_root))
    device = configure_runtime(args.device)
    steps = int(args.steps or REGISTERED_STEPS)
    batch = int(args.batch or REGISTERED_BATCH)
    requested = sum(
        1 for arm in arms for _ in (seeds if seeds is not None
                                    else seeds_by_arm[arm])
    )
    if args.resume and requested != 1:
        raise SystemExit(
            "--resume applies to exactly one run; select one --arms/--seeds pair "
            "or resume each run in its own invocation"
        )
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        out_root / "run_status.json",
        {
            "run_id": out_root.name,
            "mode": "grid",
            "host": host_name(),
            "device": str(device),
            "outcome": "running",
            "started_at_utc": utcnow(),
            "arms": list(arms),
            "seeds": seeds if seeds is not None else "config",
            "steps": steps,
            "batch": batch,
            "fixture": args.fixture,
            **_extension_fields(extension),
        },
    )
    existing = {
        run_id for run_id, row in load_manifest_rows(out_root).items()
        if row.get("outcome") == "complete"
        and (out_root / "runs" / run_id / "probes.json").exists()
    }
    existing |= _completed_run_ids_on_disk(out_root)
    results = []
    requested_ids = []
    for arm in arms:
        for seed in (seeds if seeds is not None else seeds_by_arm[arm]):
            run = RunSpec(
                run_id=f"wp4a_eval_{arm}_s{int(seed)}",
                arm=arm,
                seed=int(seed),
                cohort="evaluation",
                steps=steps,
                batch=batch,
                beta=beta if arm == "sigreg_jepa_anchored" else None,
                beta_source=beta_source if arm == "sigreg_jepa_anchored" else None,
            )
            requested_ids.append(run.run_id)
            if run.run_id in existing and not args.force:
                print(
                    json.dumps(
                        {"run_id": run.run_id, "outcome": "skipped_complete"},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            result = run_one(
                run,
                config=config,
                products=products,
                cohort=cohort,
                out_root=out_root,
                config_sha=config_sha,
                code_sha=code_sha,
                status_text=status_text,
                device=device,
                fixture=args.fixture,
                evaluate=not args.no_evaluate,
                resume=Path(args.resume) if args.resume else None,
                stop_after=args.stop_after,
                checkpoint_minutes=float(args.checkpoint_minutes),
                extension=extension,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "outcome": result["manifest"]["outcome"],
                        "steps_completed": result["manifest"].get("steps_completed"),
                        "wall_seconds": result["manifest"].get("wall_seconds"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if result["manifest"]["outcome"] == "interrupted" and args.resume_mode:
                break
    interrupted = [r for r in results
                   if r["manifest"]["outcome"] == "interrupted"]
    failed = [r for r in results if r["manifest"]["outcome"] == "failed"]
    # Rebuild from disk so a grid split over several guard-gated jobs (or
    # resumed after a timeout) still aggregates every completed run.
    disk_results = collect_results(out_root)
    complete_ids = {result["manifest"]["run_id"] for result in disk_results}
    aggregate = build_aggregate(
        disk_results,
        config=config,
        config_sha=config_sha,
        code_sha=code_sha,
        status_text=status_text,
        cohorts={"evaluation": cohort_provenance(cohort)},
        fixture=args.fixture,
    )
    if extension is not None:
        _label_follow_up(aggregate, extension)
        aggregate["beta_decision"] = {
            "decision_path": str(args.beta_decision),
            "decision_sha256": sha256_file(args.beta_decision),
            "selected_beta": beta,
            "plain_reference": (beta_decision or {}).get("plain_reference"),
            "tolerance": (beta_decision or {}).get("tolerance"),
        }
    rows_by_id = load_manifest_rows(out_root)
    aggregate["runs"] = [
        rows_by_id[run_id] for run_id in sorted(rows_by_id)
        if rows_by_id[run_id].get("outcome") in ("complete", "failed", "interrupted")
    ]
    aggregate["requested_runs"] = requested_ids
    aggregate["completed_runs"] = sorted(complete_ids)
    aggregate["pending_runs"] = [
        run_id for run_id in requested_ids if run_id not in complete_ids
    ]
    aggregate["interrupted_runs"] = [r["manifest"]["run_id"] for r in interrupted]
    aggregate["failed_runs"] = [r["manifest"]["run_id"] for r in failed]
    aggregate["data_footprint"] = _footprint(out_root, Path(args.data_root))
    atomic_write_json(out_root / "aggregate.json", aggregate)
    predictions_fragment(aggregate, out_dir=out_root)
    atomic_write_json(
        out_root / "run_status.json",
        {
            "run_id": out_root.name,
            "host": host_name(),
            "device": str(device),
            "outcome": (
                "complete"
                if not failed and not interrupted
                and all(run_id in complete_ids for run_id in requested_ids)
                else ("failed" if failed else "incomplete")
            ),
            "exit_status": 0 if not failed else 1,
            "runs_requested": len(requested_ids),
            "runs_complete": len(complete_ids),
            "runs_pending": len(
                [run_id for run_id in requested_ids if run_id not in complete_ids]
            ),
            "runs_interrupted": len(interrupted),
            "runs_failed": len(failed),
            **_extension_fields(extension),
            "finished_at_utc": utcnow(),
            "aggregate_sha256": sha256_file(out_root / "aggregate.json"),
        },
    )
    print(
        json.dumps(
            {
                "aggregate": str(out_root / "aggregate.json"),
                "predictions": str(out_root / "PREDICTIONS_UPDATE.md"),
                "verdicts": {
                    key: value["status"]
                    for key, value in aggregate["verdicts"].items()
                },
                "registered_run": aggregate["registered_run"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


def _aggregate_load_manifest(root: Path) -> Dict[str, dict]:
    """Read a source out root's job-level manifest (latest row per run)."""
    path = Path(root) / "manifest.jsonl"
    if not path.is_file():
        raise RuntimeError(
            f"aggregate: no manifest.jsonl under --runs-root {root}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"aggregate: cannot read {path}: {error}")
    rows: Dict[str, dict] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"aggregate: {path}:{number} is not valid JSON ({error}); "
                "refusing damaged manifest evidence"
            )
        if not isinstance(row, dict) or "run_id" not in row:
            raise RuntimeError(
                f"aggregate: {path}:{number} has no run_id; refusing damaged "
                "manifest evidence"
            )
        rows[str(row["run_id"])] = row
    return rows


def _aggregate_discover_runs(root: Path) -> List[Path]:
    """Discover ``runs/<run_id>`` directories (reads only, never writes)."""
    runs_dir = Path(root) / "runs"
    if not runs_dir.is_dir():
        raise RuntimeError(f"aggregate: {root} has no runs/ directory")
    found: List[Path] = []
    with os.scandir(runs_dir) as entries:
        for entry in entries:
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            run_dir = Path(entry.path)
            signals = (
                "probes.json", "run_status.json", "train_log.jsonl",
                "checkpoint.pt", "embeddings.npz",
            )
            if not any((run_dir / name).exists() for name in signals):
                continue
            if not (run_dir / "probes.json").is_file():
                raise RuntimeError(
                    f"aggregate: {run_dir} has no probes.json; refusing to "
                    "aggregate or repair an incomplete run directory"
                )
            found.append(run_dir)
    return sorted(found, key=lambda item: item.name)


def _aggregate_read_json(path: Path, label: str) -> Tuple[dict, str, int]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"aggregate: cannot read {label} {path}: {error}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"aggregate: {label} {path} is not valid JSON ({error}); refusing"
        )
    return payload, sha256_bytes(raw), len(raw)


def _aggregate_probes(run_dir: Path) -> dict:
    """Read and validate a run's probes.json (the only per-run input needed)."""
    probes_path = run_dir / "probes.json"
    payload, probes_sha, probes_bytes = _aggregate_read_json(
        probes_path, "probes.json"
    )
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_PROBES:
        schema = payload.get("schema") if isinstance(payload, dict) else None
        raise RuntimeError(
            f"aggregate: {probes_path} is not a bridge probes payload "
            f"(schema {schema!r}); refusing"
        )
    run = payload.get("run")
    probes = payload.get("probes")
    floor = payload.get("random_init_floor")
    if (
        not isinstance(run, dict)
        or not isinstance(probes, dict)
        or not isinstance(floor, dict)
    ):
        raise RuntimeError(
            f"aggregate: {probes_path} lacks run/probes/random_init_floor; "
            "refusing"
        )
    for key in ("run_id", "arm", "seed", "cohort", "steps"):
        if key not in run:
            raise RuntimeError(
                f"aggregate: {probes_path} run record lacks {key!r}; refusing"
            )
    if str(run["run_id"]) != run_dir.name:
        raise RuntimeError(
            f"aggregate: {probes_path} says run_id {run['run_id']!r}, not "
            f"{run_dir.name!r}; refusing"
        )
    if run["arm"] not in ALL_ARMS:
        raise RuntimeError(
            f"aggregate: {probes_path} records unknown arm {run['arm']!r}; "
            f"registered arms are {ALL_ARMS}"
        )
    for key in ("physics", "nuisance", "amplitude"):
        if not isinstance(probes.get(key), dict):
            raise RuntimeError(
                f"aggregate: {probes_path} lacks probes.{key}; refusing"
            )
    return {
        "payload": payload,
        "run": run,
        "probes_sha256": probes_sha,
        "probes_bytes": probes_bytes,
    }


def _aggregate_run_status(run_dir: Path) -> Tuple[Optional[dict], str]:
    path = run_dir / "run_status.json"
    if not path.is_file():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable"
    if not isinstance(payload, dict):
        return None, "unreadable"
    return payload, str(payload.get("outcome"))


def _aggregate_footprint(
    roots: Sequence[Path], out_root: Path
) -> dict:
    """Grid-shaped footprint; source roots are walked read-only."""
    def summarise(root: Path) -> dict:
        files = 0
        size = 0
        for dirpath, _dirnames, filenames in os.walk(
            root, followlinks=False
        ):
            for name in filenames:
                path = Path(dirpath) / name
                try:
                    size += path.stat().st_size
                except OSError:
                    continue
                files += 1
        return {"files": files, "bytes": size}

    per_root = {str(root): summarise(root) for root in roots}
    total_files = sum(row["files"] for row in per_root.values())
    total_bytes = sum(row["bytes"] for row in per_root.values())
    return {
        "out": summarise(Path(out_root)),
        "data_root": {"files": total_files, "bytes": total_bytes},
        "source_roots": per_root,
        "note": (
            "read-only footprint over the source out roots; no run directory "
            "was written and no cohort data was read or written"
        ),
    }


def run_aggregate(
    args,
    config: dict,
    config_sha: str,
    code_sha: str,
    status_text: str,
) -> int:
    """Aggregate completed runs spread over several job out roots, read-only.

    The mode discovers ``runs/<run_id>/probes.json`` directly (never through a
    training/evaluation path), requires the full registered arm x seed set
    unless ``--allow-partial``, and writes only the four aggregate-level files.
    No code path below calls training, encoding, probe fitting or cohort
    generation; every source path is opened for reading only.
    """
    started_at = utcnow()
    extension = getattr(args, "extension", None)
    if extension not in (None, B1_EXTENSION):
        raise RuntimeError(f"unknown --extension {extension!r}")
    roots = [Path(root) for root in args.runs_root]
    out_root = Path(args.out)
    out_resolved = out_root.resolve()
    for root in roots:
        root_resolved = Path(root).resolve()
        if out_resolved == root_resolved or root_resolved in out_resolved.parents:
            raise RuntimeError(
                f"aggregate: --out {out_root} is inside --runs-root {root}; "
                "the aggregate output must be outside every source root so no "
                "run directory can be written"
            )

    declarations = list(getattr(args, "complete_run", None) or [])
    results: List[dict] = []
    sources: List[dict] = []
    anomalies: List[dict] = []
    declared_complete: List[str] = []
    recorded: Dict[str, dict] = {}
    for root in roots:
        root_rows = _aggregate_load_manifest(root)
        for run_dir in _aggregate_discover_runs(root):
            run_id = run_dir.name
            if run_id in recorded:
                raise RuntimeError(
                    f"aggregate: duplicate run_id {run_id!r} discovered at "
                    f"{recorded[run_id]['run_dir']} and {run_dir}; refusing "
                    "ambiguous evidence"
                )
            probes = _aggregate_probes(run_dir)
            run = probes["run"]
            arm = str(run["arm"])
            seed = int(run["seed"])
            row = root_rows.get(run_id)
            status, status_outcome = _aggregate_run_status(run_dir)
            manifest_outcome = (
                None if row is None else str(row.get("outcome"))
            )
            damaged: List[str] = []
            if row is None:
                damaged.append("job-level manifest row missing")
            else:
                if row.get("outcome") != "complete":
                    damaged.append(
                        f"job-level manifest outcome {manifest_outcome!r}"
                    )
                if row.get("arm") != arm:
                    damaged.append(
                        f"manifest arm {row.get('arm')!r} != probes arm {arm!r}"
                    )
                try:
                    row_seed = int(row.get("seed"))
                except (TypeError, ValueError):
                    row_seed = None
                if row_seed != seed:
                    damaged.append(
                        f"manifest seed {row.get('seed')!r} != probes seed {seed}"
                    )
                stored = (
                    (row.get("embedding_files") or {}).get("probes.json", {})
                    or {}
                ).get("sha256")
                if stored and stored != probes["probes_sha256"]:
                    damaged.append(
                        "probes.json sha256 does not match the job manifest"
                    )
            if status_outcome != "complete":
                damaged.append(
                    f"run_status.json outcome {status_outcome!r}"
                )
            is_declared = run_id in declarations
            if damaged and not is_declared:
                raise RuntimeError(
                    f"aggregate: {run_dir}: probes.json parses but the "
                    f"completeness evidence is damaged ({'; '.join(damaged)}). "
                    f"If the job-level manifest records the run complete, pass "
                    f"--complete-run {run_id} to declare this damaged-status "
                    "run complete (recorded in the aggregate provenance)."
                )
            if is_declared:
                declared_complete.append(run_id)
                anomalies.append(
                    {
                        "run_id": run_id,
                        "arm": arm,
                        "seed": seed,
                        "source_run_dir": str(run_dir),
                        "damaged_evidence": damaged,
                        "resolution": "--complete-run declaration",
                    }
                )

            manifest_row = dict(row or {})
            manifest_row.update(
                {
                    "run_id": run_id,
                    "arm": arm,
                    "seed": seed,
                    "cohort": run.get("cohort"),
                    "outcome": "complete",
                    "steps": run.get("steps"),
                    "batch": run.get("batch"),
                    "beta": run.get("beta"),
                    "beta_source": run.get("beta_source"),
                    "wall_seconds": (row or {}).get("wall_seconds"),
                    "steps_completed": (row or {}).get(
                        "steps_completed", run.get("steps")
                    ),
                    "registered": bool((row or {}).get("registered", False)),
                    "extension": (row or {}).get("extension"),
                    "declared_complete": bool(is_declared),
                    "probes_sha256": probes["probes_sha256"],
                    "source_run_dir": str(run_dir),
                    "source_runs_root": str(root),
                    "run_status_outcome": status_outcome,
                    "manifest_outcome": manifest_outcome,
                }
            )
            embedding_path = run_dir / "embeddings.npz"
            results.append(
                {
                    "manifest": manifest_row,
                    "metrics": {
                        "probes": probes["payload"]["probes"],
                        "random_init_floor": probes["payload"][
                            "random_init_floor"
                        ],
                        "embedding_files": (row or {}).get(
                            "embedding_files", {}
                        ),
                    },
                    "embedding_path": (
                        embedding_path if embedding_path.is_file() else None
                    ),
                }
            )
            recorded[run_id] = {"run_dir": run_dir, "result": results[-1]}
            sources.append(
                {
                    "run_id": run_id,
                    "arm": arm,
                    "seed": seed,
                    "source_runs_root": str(root),
                    "source_run_dir": str(run_dir),
                    "probes_path": str(run_dir / "probes.json"),
                    "probes_sha256": probes["probes_sha256"],
                    "probes_bytes": probes["probes_bytes"],
                    "embedding_present": embedding_path.is_file(),
                    "run_status_outcome": status_outcome,
                    "manifest_outcome": manifest_outcome,
                    "extension": (row or {}).get("extension"),
                    "declared_complete": bool(is_declared),
                }
            )
    unknown_declarations = sorted(
        set(declarations) - set(recorded)
    )
    if unknown_declarations:
        raise RuntimeError(
            "aggregate: --complete-run names no discovered run: "
            + ", ".join(unknown_declarations)
        )
    if not results:
        raise RuntimeError(
            "aggregate: no completed runs discovered under the given "
            "--runs-root values"
        )

    extension_arms = ("sigreg_jepa", "sigreg_jepa_anchored")
    if extension is None:
        expected_arms = (("sigreg_jepa", "sigreg_jepa_anchored", "mae_chi2")
                         if toy_delay_extension() else ALL_ARMS)
        expected = [
            (arm, seed) for arm in expected_arms for seed in REGISTERED_SEEDS
        ]
    else:
        # B1-ext is reported beside the registered result: it needs the plain
        # arm and the anchored arm at the three registered seeds.  The other
        # registered arms are irrelevant to this follow-up and are not required.
        expected = [
            (arm, seed) for arm in extension_arms for seed in REGISTERED_SEEDS
        ]
    present = {
        (result["manifest"]["arm"], int(result["manifest"]["seed"]))
        for result in results
    }
    if len(present) != len(results):
        raise RuntimeError(
            "aggregate: two source runs claim the same arm/seed; refusing "
            "ambiguous evidence"
        )
    missing = [
        (arm, seed) for arm, seed in expected if (arm, seed) not in present
    ]
    if extension is not None and missing:
        listed = ", ".join(f"{arm}/s{seed}" for arm, seed in missing)
        raise RuntimeError(
            f"aggregate --extension {extension} requires the plain and "
            "anchored arms at seeds 0,1,2 (aggregate the extension out root "
            "together with the committed plain sigreg_jepa out root); missing: "
            f"{listed}"
        )
    if extension is None and missing and not getattr(args, "allow_partial", False):
        listed = ", ".join(f"{arm}/s{seed}" for arm, seed in missing)
        raise RuntimeError(
            "aggregate requires the full registered set "
            "(sigreg_jepa, sigreg_jepa_anchored, mae_chi2, "
            "sigreg_jepa_parallel x seeds 0,1,2); missing: "
            f"{listed} (use --allow-partial for a debugging aggregate)"
        )
    if getattr(args, "partial_verdicts", False) and not getattr(
        args, "allow_partial", False
    ):
        raise RuntimeError(
            "aggregate: --partial-verdicts requires --allow-partial"
        )

    anchored = [
        result for result in results
        if result["manifest"]["arm"] == "sigreg_jepa_anchored"
    ]
    beta_provenance = None
    if args.beta_decision:
        decision = load_beta_decision(
            Path(args.beta_decision), config, config_sha,
            expect_extension=extension,
        )
        beta_provenance = {
            "decision_path": str(args.beta_decision),
            "decision_sha256": sha256_file(args.beta_decision),
            "selected_beta": float(decision["selected_beta"]),
            "decision_registered": bool(
                decision.get("registered", False)
            ),
            "extension": decision.get("extension"),
            "plain_reference": decision.get("plain_reference"),
            "tolerance": decision.get("tolerance"),
        }
        for result in anchored:
            recorded_beta = result["manifest"].get("beta")
            if recorded_beta is None:
                continue
            if float(recorded_beta) != float(decision["selected_beta"]):
                raise RuntimeError(
                    f"aggregate: {result['manifest']['run_id']} ran with "
                    f"beta {recorded_beta}, but the decision selects "
                    f"{decision['selected_beta']}; refusing"
                )
    elif anchored:
        raise RuntimeError(
            "aggregate with sigreg_jepa_anchored requires --beta-decision "
            "from `bridge_run beta-dev` (v4 §3.5.7)"
        )

    cohorts: Dict[str, dict] = {}
    for result in results:
        provenance = result["manifest"].get("cohort_provenance")
        cohort_name = result["manifest"].get("cohort")
        if provenance and cohort_name:
            cohorts.setdefault(str(cohort_name), provenance)
    fixtures = sorted({
        str(result["manifest"].get("fixture"))
        for result in results
        if result["manifest"].get("fixture")
    })
    fixture = args.fixture or (
        fixtures[0] if len(fixtures) == 1 else "mixed"
    )
    aggregate = build_aggregate(
        results,
        config=config,
        config_sha=config_sha,
        code_sha=code_sha,
        status_text=status_text,
        cohorts=cohorts,
        fixture=fixture,
    )
    if extension is not None:
        _label_follow_up(aggregate, extension)
    requested = [
        f"wp4a_eval_{arm}_s{seed}" for arm, seed in expected
    ]
    completed_ids = sorted(
        result["manifest"]["run_id"] for result in results
    )
    pending_ids = [
        f"wp4a_eval_{arm}_s{seed}" for arm, seed in missing
    ]
    source_config_shas = sorted({
        str(result["manifest"].get("config_sha256"))
        for result in results
        if result["manifest"].get("config_sha256")
    })
    source_code_shas = sorted({
        str(result["manifest"].get("code_sha"))
        for result in results
        if result["manifest"].get("code_sha")
    })
    aggregate["mode"] = "aggregate"
    aggregate["aggregation_mode"] = (
        "aggregate-only (read-only; no run directory written)"
    )
    aggregate["aggregation_sources"] = sources
    aggregate["aggregate_provenance"] = {
        "runs_roots": [str(root) for root in roots],
        "source_run_dirs": [row["source_run_dir"] for row in sources],
        "probes_sha256": {
            row["run_id"]: row["probes_sha256"] for row in sources
        },
        "declared_complete_runs": sorted(declared_complete),
        "declarations": {
            entry["run_id"]: {
                "damaged_evidence": entry["damaged_evidence"],
                "resolution": entry["resolution"],
            }
            for entry in anomalies
        },
        "data_root": (
            None if getattr(args, "data_root", None) is None
            else str(args.data_root)
        ),
        "data_root_used": False,
        "data_root_note": (
            "--data-root is accepted for command-line compatibility only; "
            "aggregate never reads or writes cohort data"
        ),
        "source_config_sha256": source_config_shas,
        "source_code_sha": source_code_shas,
        "current_config_sha256": config_sha,
        "current_code_sha": code_sha,
        **(dict() if extension is None else {"extension": extension}),
        "note": (
            "discovered runs/<run_id>/probes.json read-only; no training, "
            "evaluation, encoding, checkpoint or run-directory write"
        ),
    }
    aggregate["declared_complete_runs"] = sorted(declared_complete)
    aggregate["status_anomalies"] = anomalies
    aggregate["requested_runs"] = requested
    aggregate["completed_runs"] = completed_ids
    aggregate["pending_runs"] = pending_ids
    # Grid-shaped placeholders: aggregate mode never trains, so there are no
    # failed/interrupted runs; a damaged status is either refused or declared
    # complete above and recorded in status_anomalies.
    aggregate["interrupted_runs"] = []
    aggregate["failed_runs"] = []
    aggregate["missing_registered_runs"] = [
        {"arm": arm, "seed": seed,
         "run_id": f"wp4a_eval_{arm}_s{seed}"}
        for arm, seed in missing
    ]
    aggregate["partial_registered_set"] = bool(missing)
    aggregate["beta_decision"] = beta_provenance
    if missing:
        for key in ("B1", "R7"):
            verdict = aggregate["verdicts"][key]
            verdict["partial_registered_set"] = True
            verdict["missing_registered_runs"] = list(pending_ids)
            if not getattr(args, "partial_verdicts", False):
                if not (
                    str(verdict.get("reason") or "").startswith(
                        ("missing_required_arm:",
                         "missing_required_nuisance_target:")
                    )
                ):
                    verdict["reason"] = "partial_registered_set"
                verdict["status"] = "not_evaluable"
    aggregate["data_footprint"] = _aggregate_footprint(roots, out_root)
    aggregate["files"] = {
        "manifest_rows": len(results),
        "source_runs": len(sources),
    }

    out_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_root / "aggregate.json"
    predictions_path = out_root / "PREDICTIONS_UPDATE.md"
    manifest_path = out_root / "manifest.jsonl"
    atomic_write_json(aggregate_path, aggregate)
    predictions_fragment(aggregate, out_dir=out_root, mode="aggregate")
    manifest_rows = [
        {
            "schema": SCHEMA_AGGREGATE_MANIFEST,
            "kind": "source_run",
            "run_id": row["run_id"],
            "arm": row["arm"],
            "seed": row["seed"],
            "source_runs_root": row["source_runs_root"],
            "source_run_dir": row["source_run_dir"],
            "probes_path": row["probes_path"],
            "probes_sha256": row["probes_sha256"],
            "probes_bytes": row["probes_bytes"],
            "embedding_present": row["embedding_present"],
            "run_status_outcome": row["run_status_outcome"],
            "manifest_outcome": row["manifest_outcome"],
            "declared_complete": row["declared_complete"],
            "extension": row.get("extension"),
            "declaration_source": (
                "--complete-run" if row["declared_complete"] else None
            ),
        }
        for row in sources
    ]
    atomic_write_text(
        manifest_path,
        "\n".join(
            json.dumps(_jsonable(row), sort_keys=True)
            for row in manifest_rows
        ) + "\n",
    )
    verdict_status = {
        key: value["status"]
        for key, value in aggregate["verdicts"].items()
    }
    status_payload = {
        "run_id": out_root.name,
        "mode": "aggregate",
        "host": host_name(),
        "device": None,
        "outcome": "complete" if not missing else "incomplete",
        "exit_status": 0,
        "started_at_utc": started_at,
        "finished_at_utc": utcnow(),
        "source_runs_roots": [str(root) for root in roots],
        "runs_aggregated": len(results),
        "runs_requested": len(requested),
        "runs_complete": len(completed_ids),
        "runs_pending": len(pending_ids),
        "declared_complete_runs": sorted(declared_complete),
        "partial_registered_set": bool(missing),
        "registered_run": aggregate["registered_run"],
        **_extension_fields(extension),
        "verdicts": verdict_status,
        "aggregate_sha256": sha256_file(aggregate_path),
        "predictions_sha256": sha256_file(predictions_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(out_root / "run_status.json", status_payload)
    print(json.dumps({
        "aggregate": str(aggregate_path),
        "predictions": str(predictions_path),
        "verdicts": verdict_status,
        "completed_runs": len(completed_ids),
        "pending_runs": pending_ids,
        "declared_complete_runs": sorted(declared_complete),
        "registered_run": aggregate["registered_run"],
    }, indent=2, sort_keys=True))
    return 0


def _footprint(out_root: Path, data_root: Path) -> dict:
    def summarise(root: Path) -> dict:
        files = 0
        size = 0
        for path in root.rglob("*"):
            if path.is_file():
                files += 1
                size += path.stat().st_size
        return {"files": files, "bytes": size}

    return {
        "out": summarise(Path(out_root)),
        "data_root": summarise(Path(data_root)),
        "note": (
            "the /ceph entry quota is at its floor: one embeddings .npz per "
            "run, one checkpoint per run, one event shard + targets file per "
            "cohort role, and one SHA256SUMS per directory"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True, help="output root (the only write root besides --data-root)")
    parser.add_argument("--data-root", required=True, help="host data root (cohort cache + frozen products)")
    parser.add_argument("--config", default=None, help="path to configs/bridge_v1.json")
    parser.add_argument("--fixture", choices=("a4", "synthetic"), default="a4",
                        help="a4 = registered law; synthetic = CPU smoke/test fixture only")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:N")
    parser.add_argument("--steps", type=int, default=None, help="override the scheduled step count (smoke)")
    parser.add_argument("--batch", type=int, default=None, help=f"batch size (default {REGISTERED_BATCH}, v3 §6 WP4)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="truncate every registered cohort range from its start (smoke only)")
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0,
                        help="wall-clock checkpoint interval; 0 disables the timer (v4 §6.1)")
    parser.add_argument("--no-evaluate", action="store_true", help="train and checkpoint only")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="stop after this many steps and keep the checkpoint (interruption test)")
    parser.add_argument("--resume", default=None, help="resume from this checkpoint")
    parser.add_argument("--resume-mode", action="store_true",
                        help="in grid mode, stop the remaining runs on an interruption")
    parser.add_argument("--force", action="store_true",
                        help="re-run grid runs that already completed under --out")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jepa_noise.bridge_run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = _common_parser()
    sub = parser.add_subparsers(dest="mode", required=True)
    grid = sub.add_parser("grid", parents=[common],
                          help="the 4-arm x 3-seed bridge grid on the evaluation cohort")
    grid.add_argument("--arms", default=None,
                      help=f"comma-separated subset of {ALL_ARMS} (default: all)")
    grid.add_argument("--seeds", default=None, help="comma-separated seed override")
    grid.add_argument("--beta-decision", default=None,
                      help="beta_decision.json from beta-dev (required for the anchored arm)")
    grid.add_argument("--extension", choices=(B1_EXTENSION,), default=None,
                      help="opt-in A9 labelled follow-up; B1-ext accepts only a "
                           "B1-ext beta decision and marks every run non-registered")
    aggregate = sub.add_parser(
        "aggregate",
        help="read-only aggregation over completed runs across out roots",
    )
    aggregate.add_argument("--out", required=True,
                           help="aggregate output root (the only write root)")
    aggregate.add_argument("--runs-root", nargs="+", required=True,
                           help="out roots containing runs/<run_id>/")
    aggregate.add_argument("--config", default=None,
                           help="path to configs/bridge_v1.json")
    aggregate.add_argument("--beta-decision", default=None,
                           help="beta_decision.json from beta-dev (required when "
                                "an anchored run is aggregated)")
    aggregate.add_argument("--extension", choices=(B1_EXTENSION,), default=None,
                           help="opt-in A9 labelled follow-up aggregate; requires "
                                "the plain and anchored arms at seeds 0,1,2 and a "
                                "B1-ext beta decision; verdicts are follow_up")
    aggregate.add_argument("--data-root", default=None,
                           help="accepted for compatibility only; aggregate never "
                                "reads or writes cohort data")
    aggregate.add_argument("--fixture", choices=("a4", "synthetic"),
                           default=None,
                           help="recorded fixture label (derived from the source "
                                "manifests when omitted)")
    aggregate.add_argument("--allow-partial", action="store_true",
                           help="debugging aggregate over an incomplete arm x seed "
                                "set; every verdict becomes not_evaluable")
    aggregate.add_argument("--partial-verdicts", action="store_true",
                           help="with --allow-partial, still evaluate B1/R7 from "
                                "the runs present (tagged partial_registered_set)")
    aggregate.add_argument("--complete-run", action="extend", nargs="+",
                           default=[], metavar="RUN_ID",
                           help="declare a run complete despite a damaged "
                                "run_status.json; recorded in the aggregate "
                                "manifest/provenance")
    beta = sub.add_parser("beta-dev", parents=[common],
                          help="development-cohort beta selection (v3 §1.3)")
    beta.add_argument("--beta-grid", default=None,
                      help="registered: must equal the frozen grid {0.1,1,10}; "
                           f"{B1_EXTENSION}: must equal 10,100,1000")
    beta.add_argument("--beta-decision-out", default=None,
                      help="where to write beta_decision.json (default <out>/beta_decision.json)")
    beta.add_argument("--extension", choices=(B1_EXTENSION,), default=None,
                      help="opt-in A9 B1-ext beta development with the LUCiD "
                           "physics-tolerance selection rule")
    beta.add_argument("--selection-rule", default=None,
                      help=f"{B1_EXTENSION} only: lucid_physics_tolerance")
    beta.add_argument("--plain-reference", default=None,
                      help=f"{B1_EXTENSION} only: the committed sigreg_jepa seed-0 "
                           "bridge run directory or its metrics/probes JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config_path = Path(args.config) if args.config else Path(CONFIG_PATH)
    config_sha = sha256_file(config_path)
    root = Path(__file__).resolve().parents[2]
    code_sha = git_head(root)
    status_text = git_status(root)
    try:
        if args.mode == "aggregate":
            # Dispatch before the grid/beta-dev data-root handling: aggregate
            # accepts --data-root for compatibility but must never touch it.
            return run_aggregate(args, config, config_sha, code_sha, status_text)
        if args.fixture == "a4":
            Path(args.data_root).mkdir(parents=True, exist_ok=True)
        if args.mode == "beta-dev":
            return run_beta_dev(args, config, config_sha, code_sha, status_text)
        if args.mode == "grid":
            if args.beta_decision is None and (
                args.arms is None or "sigreg_jepa_anchored" in args.arms.split(",")
            ):
                parser.error(
                    "grid with sigreg_jepa_anchored requires --beta-decision from "
                    "`bridge_run beta-dev`; the anchored arm must not run before "
                    "beta is selected (v4 §3.5.7)"
                )
            return run_grid(args, config, config_sha, code_sha, status_text)
    except RuntimeError as error:
        print(f"bridge_run: refused/failed: {error}", file=sys.stderr)
        return 2
    parser.error(f"unknown mode {args.mode!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
