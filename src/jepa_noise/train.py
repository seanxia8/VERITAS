"""Tiny CPU helper for the WP4a bridge smoke, tests and config freeze -- NOT WP5.

This module provides (a) the real A4 bridge data products from the frozen
``views.py`` API (fixed per-channel scaling, anchor overlay bank, target
marginal of ``N_actual``, and their hashes), (b) a deterministic synthetic
fixture used only for masking/architecture unit tests, and (c) a minimal step
loop for the CPU smoke.  It implements no cohort handling, no checkpoints, no
schedules and no evaluation; those belong to WP5 and are out of scope here.

Platform note: bit-exact hashing of the N_actual marginal was platform-brittle
(the first x86 Titan smoke failed on a Mac-vs-x86 BLAS last-bit difference,
2026-10-17); verification is now tolerance-based against a committed
little-endian float64 reference, and exact per-host sha256 values are
informational only.

The registered bridge cell is A4 at g = 0.02, nominal SNR, channel split:
``Tier1Views(signal_mode='nonlinear', split='channel', line_multiplier=0.,
common_strength=0.02, amplitude_scale=1.0, law='v4-A4-waveform-v1',
base_rho=0.0, seed=0, cell='bridge', model_seed=0)`` (v3 WP5 defines the
Tier-1 E2 signal as nonlinear EventFamily waveforms; the linearised fixture is
E1b-only).

Usage::

    PYTHONPATH=src python -m jepa_noise.train --freeze-config   # writes A4 products
    PYTHONPATH=src python -m jepa_noise.train --smoke --a4 --batch 2 --steps 1
"""

from __future__ import annotations

import os

# Runbook v3 §3.1: no implicit compilation for CPU checks; must precede the
# reconstruction_model import chain (see backbone.py).
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .jepa import (
    ALL_ARMS,
    CONFIG_PATH,
    BridgeModel,
    build_model,
    build_optimizers,
    load_config,
)
from .backbone import TokenGridEncoder

__all__ = [
    "BRIDGE_VIEWS",
    "CALIBRATION_ROLE",
    "CALIBRATION_EVENT_COUNT",
    "OVERLAY_ROLE",
    "OVERLAY_EVENT_COUNT",
    "a4_bridge_views",
    "a4_bridge_products",
    "freeze_bridge_config",
    "REFERENCE_PATH",
    "MATRIX_RELATIVE_TOLERANCE",
    "SCALING_RELATIVE_TOLERANCE",
    "load_reference_marginal",
    "reference_file_sha256",
    "relative_frobenius_error",
    "verify_bridge_matrix",
    "verify_bridge_products",
    "write_reference_marginal",
    "materialize_or_load_marginal",
    "target_marginal_from_full",
    "synthetic_calibration_noise",
    "synthetic_noise_bank",
    "synthetic_target_covariance",
    "synthetic_fixture",
    "synthetic_batch",
    "make_batches",
    "train_steps",
    "smoke_arms",
]


def _seeded_generator(seed: Optional[int]) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(0 if seed is None else int(seed))
    return generator


# ---------------------------------------------------------------------------
# Registered A4 bridge data products (views.py API frozen by WP2-v4, 10693eb)
# ---------------------------------------------------------------------------
#: The registered Tier-1 bridge cell: A4 at g = 0.02, nominal SNR, channel split.
BRIDGE_VIEWS = {
    # v3 WP5 defines the Tier-1 E2 signal as nonlinear EventFamily waveforms;
    # A4 changes the base noise, epsilon_dc and (E1b only) the linearised theta
    # SD, so the bridge keeps the nonlinear signal.  The linearised fixture is
    # E1b-only (E1).  Scaling and N_actual are noise-only and unaffected.
    "signal_mode": "nonlinear",
    "split": "channel",
    "line_multiplier": 0.0,
    "common_strength": 0.02,
    "amplitude_scale": 1.0,
    "law": "v4-A4-waveform-v1",
    "base_rho": 0.0,
    "seed": 0,
    "cell": "bridge",
    "model_seed": 0,
}
#: Fixed per-channel scaling is fitted once on these disjoint calibration-noise
#: records (v3 WP4: "fit on disjoint calibration-noise records, held fixed").
CALIBRATION_ROLE = "noise_cal"
CALIBRATION_EVENT_COUNT = 4096
#: The anchor overlay draws from an independent noise namespace (v3 WP5).
OVERLAY_ROLE = "noise_overlay"
#: Full-rank overlay bank for the 2048-dimensional N_actual: 1024 independent
#: noise_overlay draws (8 MB float32), not a 64-record sample.
OVERLAY_EVENT_COUNT = 1024


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(array, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return hashlib.sha256(array.tobytes()).hexdigest()


def a4_bridge_views(**overrides):
    """Construct the registered bridge cell from the frozen ``views.py`` API."""
    from .views import Tier1Views

    params = dict(BRIDGE_VIEWS)
    params.update(overrides)
    return Tier1Views(**params)


REFERENCE_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "bridge_v1_n_actual_reference.npz"
)
#: Number of upper-triangle values of the symmetric 1024x1024 target marginal.
REFERENCE_UPPER_COUNT = 1024 * 1025 // 2
#: Authoritative tolerances: the matrix is a mathematical object, and its last
#: bits differ between BLAS implementations.  Bit-exact hashing was
#: platform-brittle and made the frozen sha a platform gate (it failed the
#: first x86 Titan smoke, 2026-10-17); verification is now tolerance-based and
#: the exact host sha256 is informational only.
MATRIX_RELATIVE_TOLERANCE = 1e-8
SCALING_RELATIVE_TOLERANCE = 1e-6


def _target_index() -> np.ndarray:
    from .masks import raw_split

    return np.asarray(raw_split("channel").target, dtype=np.int64)


def target_marginal_from_full(covariance) -> np.ndarray:
    """Target marginal of the full ``(C*T, C*T)`` ``N_actual`` covariance."""
    index = _target_index()
    return np.asarray(covariance, dtype=np.float64)[np.ix_(index, index)]


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_file_sha256(path=None) -> str:
    """sha256 of the committed platform-stable reference file."""
    return _file_sha256(REFERENCE_PATH if path is None else path)


def load_reference_marginal(path=None) -> np.ndarray:
    """Load the platform-stable 1024x1024 reference target marginal.

    The committed file stores the little-endian float64 upper triangle (npz);
    it is data, not a hash gate.  Loading is byte-identical on every host.
    """
    path = REFERENCE_PATH if path is None else Path(path)
    with np.load(path) as data:
        upper = np.asarray(data["upper"], dtype=np.float64)
    if upper.shape != (REFERENCE_UPPER_COUNT,):
        raise RuntimeError(
            f"reference marginal has {upper.shape[0]} upper values, expected "
            f"{REFERENCE_UPPER_COUNT}"
        )
    rows, columns = np.triu_indices(1024)
    matrix = np.zeros((1024, 1024), dtype=np.float64)
    matrix[rows, columns] = upper
    matrix[columns, rows] = upper
    return matrix


def relative_frobenius_error(computed, reference) -> float:
    """``||computed - reference||_F / ||reference||_F`` (both float64)."""
    computed = np.asarray(computed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if computed.shape != reference.shape:
        raise RuntimeError(
            f"matrix shape {computed.shape} differs from reference {reference.shape}"
        )
    denominator = float(np.linalg.norm(reference))
    if not denominator:
        raise RuntimeError("reference matrix has zero Frobenius norm")
    return float(np.linalg.norm(computed - reference) / denominator)


def verify_bridge_matrix(
    computed,
    reference,
    rtol: float = MATRIX_RELATIVE_TOLERANCE,
    label: str = "N_actual target marginal",
) -> float:
    """Tolerance verification of a computed matrix against the frozen reference.

    The frozen sha256 is informational only: the last bits of an
    eigh/Cholesky-based covariance differ between ARM and x86 BLAS.  A
    last-bit (1e-12 relative) perturbation passes; a 1e-3 relative
    perturbation fails.  Raises on failure.
    """
    error = relative_frobenius_error(computed, reference)
    if not np.isfinite(error) or error > float(rtol):
        raise RuntimeError(
            f"{label}: relative Frobenius difference {error:.3e} exceeds the "
            f"frozen tolerance {float(rtol):.1e}. The exact sha256 is "
            "informational only; see the platform-stability note in the config."
        )
    return error


def write_reference_marginal(marginal, path=None, warn: bool = True) -> dict:
    """Regenerate the committed reference matrix from the current platform.

    Called by ``--freeze-config``.  The last bits of the matrix are
    platform-dependent, so a loud warning is printed: the file is a
    tolerance-verification reference, not a bit-exact gate.
    """
    path = REFERENCE_PATH if path is None else Path(path)
    upper = np.asarray(marginal, dtype=np.float64)[np.triu_indices(1024)].astype(
        "<f8"
    )
    if warn:
        print(
            "WARNING: regenerating " + str(path) + " from the current platform. "
            "The last bits of a covariance are BLAS-dependent; this reference is "
            "used with a relative-Frobenius tolerance, and the exact sha256 is "
            "informational.",
            file=sys.stderr,
            flush=True,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, upper=upper)
    return {
        "file": path.name,
        "file_sha256": _file_sha256(path),
        "format": (
            "npz of the little-endian float64 (<f8) upper triangle of the "
            "symmetric 1024x1024 N_actual target marginal (platform-stable "
            "literal bytes; not a hash gate)"
        ),
        "n_upper": int(upper.size),
        "relative_frobenius_tolerance": MATRIX_RELATIVE_TOLERANCE,
        "authoritative": (
            "the relative-Frobenius tolerance check against this file is "
            "authoritative; the per-host exact sha256 is informational"
        ),
    }


def materialize_or_load_marginal(
    marginal,
    root,
    reference_sha: str,
    code_sha: Optional[str] = None,
    computed_on: Optional[str] = None,
) -> tuple:
    """Load the host-cached N_actual marginal, or write it once, then load.

    Every run on a host uses identical bytes.  The cache directory is named by
    the platform-stable reference file sha256, so hosts share the key, and the
    cached file's own sha256 is verified before use (a tampered/truncated file
    raises instead of silently training on it).
    """
    import platform

    directory = Path(root) / reference_sha
    path = directory / "N_actual_target_marginal.npy"
    if path.exists():
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"cached bridge marginal {path} has no manifest.json; refusing to "
                "trust it"
            )
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = _file_sha256(path)
        if record.get("file_sha256") != actual:
            raise RuntimeError(
                f"cached bridge marginal {path} sha256 {actual} does not match "
                f"its manifest {record.get('file_sha256')}"
            )
        return np.load(path), f"cache:{path}"
    array = np.ascontiguousarray(np.asarray(marginal, dtype="<f8"))
    directory.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    if code_sha is None:
        code_sha = _code_sha()
    if computed_on is None:
        computed_on = platform.node()
    record = {
        "reference_file_sha256": reference_sha,
        "file_sha256": _file_sha256(path),
        "shape": list(array.shape),
        "dtype": "<f8",
        "computed_on": computed_on,
        "code_sha": code_sha,
        "relative_frobenius_tolerance": MATRIX_RELATIVE_TOLERANCE,
        "views_constructor": dict(BRIDGE_VIEWS),
        "note": (
            "materialised once on this host from the pinned generator; later "
            "runs load these bytes after verifying file_sha256"
        ),
    }
    (directory / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return array, f"computed+written:{path}"


def _code_sha() -> Optional[str]:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        return os.environ.get("JEPA_CODE_SHA")


def _relative_vector_error(computed, reference, scale) -> float:
    computed = np.asarray(computed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    denominator = float(np.linalg.norm(scale))
    if not denominator:
        raise RuntimeError("reference scale has zero norm")
    return float(np.linalg.norm(computed - reference) / denominator)


def verify_bridge_products(config: dict, products: dict) -> None:
    """Verify the products against the frozen config (tolerance, not bytes).

    - committed reference file sha256 vs the config's recorded file sha256
      (both are committed bytes, so this is exact and platform-stable);
    - computed N_actual marginal vs the reference matrix, relative Frobenius
      tolerance ``MATRIX_RELATIVE_TOLERANCE``;
    - host-fitted per-channel scaling vs the frozen config arrays, relative
      tolerance ``SCALING_RELATIVE_TOLERANCE`` (float32 fits differ in their
      last bits between BLAS implementations).
    """
    target = config["data"]["target_noise_covariance"]
    reference_block = target.get("reference") or {}
    expected_file_sha = reference_block.get("file_sha256")
    if expected_file_sha is not None:
        actual_file_sha = reference_file_sha256()
        if actual_file_sha != expected_file_sha:
            raise RuntimeError(
                "committed N_actual reference file sha256 "
                f"{actual_file_sha} does not match the config {expected_file_sha}"
            )
    reference = load_reference_marginal()
    computed_marginal = target_marginal_from_full(
        products["target_noise_covariance"]
    )
    verify_bridge_matrix(
        computed_marginal,
        reference,
        rtol=float(
            reference_block.get(
                "relative_frobenius_tolerance", MATRIX_RELATIVE_TOLERANCE
            )
        ),
    )
    frozen_scaling = config["data"]["fixed_scaling"].get("values_mean_std")
    if frozen_scaling:
        computed = products["metadata"]["fixed_scaling"]
        reference_std = np.asarray(frozen_scaling["std"], dtype=np.float64)
        mean_error = _relative_vector_error(
            computed["mean"], frozen_scaling["mean"], reference_std
        )
        std_error = _relative_vector_error(
            computed["std"], frozen_scaling["std"], reference_std
        )
        if mean_error > SCALING_RELATIVE_TOLERANCE:
            raise RuntimeError(
                f"fixed per-channel mean relative difference {mean_error:.3e} "
                f"exceeds tolerance {SCALING_RELATIVE_TOLERANCE:.1e}"
            )
        if std_error > SCALING_RELATIVE_TOLERANCE:
            raise RuntimeError(
                f"fixed per-channel std relative difference {std_error:.3e} "
                f"exceeds tolerance {SCALING_RELATIVE_TOLERANCE:.1e}"
            )


def a4_bridge_products(
    n_calibration: int = CALIBRATION_EVENT_COUNT,
    n_overlay: int = OVERLAY_EVENT_COUNT,
    progress: bool = False,
    config: Optional[dict] = None,
    verify: bool = True,
    artifact_root=None,
) -> dict:
    """Real A4 bridge products from the registered generator.

    Returns ``channel_mean``/``channel_std`` (the frozen config values when
    ``verify``; the host fit is checked against them within tolerance),
    ``noise_bank`` (independent ``noise_overlay`` draws), the full raw
    ``N_actual`` covariance, the exact ``target_marginal`` (1024x1024) used by
    ``mae_chi2``, and ``metadata`` with the hashes recorded in the reviewed
    config.

    Matrix and scaling verification is tolerance-based (last-bit BLAS
    differences); the exact sha256 values are informational.  With
    ``artifact_root`` (or ``JEPA_BRIDGE_DATA_ROOT``) the marginal is
    materialised once under ``<root>/<reference-file-sha256>/`` and later runs
    load those bytes after verifying the file sha256.
    """
    views = a4_bridge_views()
    chunks = []
    for start in range(0, int(n_calibration), 256):
        stop = min(start + 256, int(n_calibration))
        chunks.append(
            np.stack(
                [
                    views.noise_record(event_id, role=CALIBRATION_ROLE)[0]
                    for event_id in range(start, stop)
                ]
            )
        )
        if progress:
            print(
                f"calibration noise {stop}/{n_calibration}",
                file=sys.stderr,
                flush=True,
            )
    records = np.concatenate(chunks, axis=0)
    channel_mean, channel_std = TokenGridEncoder.fit_scaling(
        torch.as_tensor(records, dtype=torch.float32)
    )
    bank = np.stack(
        [
            views.noise_record(event_id, role=OVERLAY_ROLE)[0]
            for event_id in range(int(n_overlay))
        ]
    )
    split = views.split
    covariance = np.asarray(views.noise_covariance)
    target_index = np.asarray(split.target, dtype=np.int64)
    marginal = covariance[np.ix_(target_index, target_index)]
    manifest = views.manifest()
    mean = [float(value) for value in channel_mean.tolist()]
    std = [float(value) for value in channel_std.tolist()]
    reference_sha = (
        reference_file_sha256() if REFERENCE_PATH.exists() else "uninitialised"
    )
    metadata = {
        "generator": {
            "views_constructor": dict(BRIDGE_VIEWS),
            "law": views.law,
            "base_rho": float(views.base_rho),
            "theta_fixture": views.theta_fixture,
            "theta_sd": [float(value) for value in np.asarray(views.theta_sd)],
            "epsilon_dc": float(views.epsilon_dc),
            "v_white": float(views.v_white),
            "common_strength": float(views.common_strength),
            "line_multiplier": float(views.line_multiplier),
            "split": views.split_name,
            "split_hash": manifest["split_hash"],
            "noise_covariance_sha256_informational": manifest[
                "noise_covariance_sha256"
            ],
            "manifest_sha256_informational": _canonical_sha256(manifest),
            "platform_note": (
                "manifest/noise-covariance sha256 values carry BLAS-dependent "
                "last bits and are informational only"
            ),
        },
        "fixed_scaling": {
            "mean": mean,
            "std": std,
            "computed_mean": mean,
            "computed_std": std,
            "sha256_informational": _array_sha256(
                np.stack(
                    [
                        np.asarray(mean, dtype=np.float32),
                        np.asarray(std, dtype=np.float32),
                    ]
                ),
                "<f4",
            ),
            "role": CALIBRATION_ROLE,
            "event_ids": [0, int(n_calibration) - 1],
            "n_records": int(n_calibration),
            "seed": BRIDGE_VIEWS["seed"],
            "cell": BRIDGE_VIEWS["cell"],
            "relative_tolerance": SCALING_RELATIVE_TOLERANCE,
        },
        "target_noise_covariance": {
            "sha256_informational": _array_sha256(marginal, "<f8"),
            "n_target": int(marginal.shape[0]),
            "target_raw_index_sha256": _array_sha256(target_index, "<i8"),
            "reference_file_sha256": reference_sha,
            "definition": (
                "target marginal of N_actual = A4 base + epsilon_dc*I + line + "
                "common (v3 §4.1), at masks.raw_split('channel').target"
            ),
            "oracle_noise_information": True,
        },
        "overlay_noise": {
            "role": OVERLAY_ROLE,
            "event_ids": [0, int(n_overlay) - 1],
            "n_records": int(n_overlay),
            "sha256_informational": _array_sha256(bank, "<f8"),
        },
        "computed_on": platform.node(),
        "code_sha": _code_sha(),
        "reference_file_sha256": reference_sha,
        "matrix_relative_tolerance": MATRIX_RELATIVE_TOLERANCE,
    }
    products = {
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "noise_bank": torch.as_tensor(bank, dtype=torch.float32),
        "target_noise_covariance": covariance,
        "target_marginal": marginal,
        "metadata": metadata,
    }
    if verify:
        frozen = load_config() if config is None else config
        verify_bridge_products(frozen, products)
        frozen_scaling = frozen["data"]["fixed_scaling"]["values_mean_std"]
        # The frozen config values are authoritative on every host; a host-side
        # refit is accepted within tolerance but never replaces them.
        products["channel_mean"] = torch.as_tensor(
            frozen_scaling["mean"], dtype=torch.float32
        )
        products["channel_std"] = torch.as_tensor(
            frozen_scaling["std"], dtype=torch.float32
        )
        metadata["fixed_scaling"][
            "authoritative_source"
        ] = "frozen config values (host fit verified within tolerance)"
        metadata["verified_against_config"] = True
    else:
        metadata["verified_against_config"] = False
    if REFERENCE_PATH.exists():
        reference = load_reference_marginal()
        metadata["target_noise_covariance"][
            "relative_frobenius_error_to_reference"
        ] = relative_frobenius_error(marginal, reference)
    else:
        metadata["target_noise_covariance"][
            "relative_frobenius_error_to_reference"
        ] = None
    if artifact_root is None:
        artifact_root = os.environ.get("JEPA_BRIDGE_DATA_ROOT")
    if artifact_root:
        stored, source = materialize_or_load_marginal(
            marginal,
            artifact_root,
            reference_sha,
            code_sha=metadata["code_sha"],
            computed_on=metadata["computed_on"],
        )
        products["target_marginal"] = stored
        metadata["target_noise_covariance"]["marginal_source"] = source
        metadata["target_noise_covariance"]["artifact_path"] = source.split(
            ":", 1
        )[-1]
        metadata["target_noise_covariance"]["artifact_file_sha256"] = _file_sha256(
            metadata["target_noise_covariance"]["artifact_path"]
        )
    else:
        metadata["target_noise_covariance"]["marginal_source"] = (
            "computed (no host data root given)"
        )
    return products


def freeze_bridge_config(
    config_path=None, progress: bool = False
) -> tuple:
    """Write the A4 data products and the platform-stable reference matrix.

    Data-derived values only; no hyper-parameter is selected and no outcome is
    used.  The committed reference matrix is regenerated from the current
    platform with an explicit warning: its last bits are BLAS-dependent, so it
    is a tolerance-verification reference, never a bit-exact gate.
    """
    path = CONFIG_PATH if config_path is None else config_path
    config = load_config(path)
    products = a4_bridge_products(progress=progress, verify=False)
    metadata = products["metadata"]
    data = config["data"]
    scaling = metadata["fixed_scaling"]
    data["fixed_scaling"].update(
        {
            "values_mean_std": {"mean": scaling["mean"], "std": scaling["std"]},
            "sha256_informational": scaling["sha256_informational"],
            "role": scaling["role"],
            "event_ids": scaling["event_ids"],
            "n_records": scaling["n_records"],
            "seed": scaling["seed"],
            "cell": scaling["cell"],
            "relative_tolerance": SCALING_RELATIVE_TOLERANCE,
            "status": (
                "frozen from the registered A4 generator; these values are "
                "authoritative and loaded by every host. A host-side refit is "
                "verified within relative tolerance 1e-6 (float32 last bits are "
                "BLAS-dependent)."
            ),
        }
    )
    reference_meta = write_reference_marginal(
        target_marginal_from_full(products["target_noise_covariance"])
    )
    target = metadata["target_noise_covariance"]
    data["target_noise_covariance"].update(
        {
            "sha256_informational": target["sha256_informational"],
            "sha256_informational_note": (
                "exact sha256 of the marginal computed on the freezing host; "
                "informational only. Verification uses the tolerance check "
                "against reference.file."
            ),
            "n_target": target["n_target"],
            "target_raw_index_sha256": target["target_raw_index_sha256"],
            "generator_manifest_sha256_informational": metadata["generator"][
                "manifest_sha256_informational"
            ],
            "generator_noise_covariance_sha256_informational": metadata[
                "generator"
            ]["noise_covariance_sha256_informational"],
            "reference": reference_meta,
            "platform_stability": (
                "the matrix is a mathematical object: its last bits differ "
                "between ARM and x86 BLAS. The first Titan smoke failed on a "
                "bit-exact gate; verification is now relative-Frobenius "
                "tolerance 1e-8 against the committed reference, and the exact "
                "host sha256 is informational."
            ),
            "status": (
                "frozen; reference matrix committed as "
                "configs/bridge_v1_n_actual_reference.npz; hosts materialise it "
                "once under the data root and later runs load identical bytes"
            ),
        }
    )
    data["generator"] = {
        **metadata["generator"],
        "calibration_noise": {
            "role": CALIBRATION_ROLE,
            "event_ids": metadata["fixed_scaling"]["event_ids"],
            "n_records": metadata["fixed_scaling"]["n_records"],
        },
        "overlay_noise": metadata["overlay_noise"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return config, products


def synthetic_calibration_noise(
    config: dict, n_records: int = 64, seed: int = 0
) -> Tensor:
    """Calibration-noise-like records ``(n, C, T)`` with per-channel scale."""
    generator = _seeded_generator(seed)
    channels = int(config["data"]["channels"])
    samples = int(config["data"]["n_samples"])
    scale = torch.arange(1, channels + 1, dtype=torch.float32) ** 0.5
    noise = torch.randn(n_records, channels, samples, generator=generator)
    return noise * scale.view(1, -1, 1)


def synthetic_noise_bank(config: dict, n_records: int = 16, seed: int = 0) -> Tensor:
    """Independent overlay-noise records (a disjoint seed from the events)."""
    return synthetic_calibration_noise(config, n_records=n_records, seed=seed + 7919)


def synthetic_target_covariance(config: dict, seed: int = 0) -> Tensor:
    """A synthetic SPD full ``(C*T, C*T)`` N_actual for the mae_chi2 smoke.

    Diagonal private variance plus a rank-4 shared component.  The real bridge
    uses the registered A4 ``N_actual`` target marginal; this is a fixture.
    """
    generator = _seeded_generator(seed)
    channels = int(config["data"]["channels"])
    samples = int(config["data"]["n_samples"])
    n = channels * samples
    factor = torch.randn(n, 4, generator=generator) / (n**0.25)
    covariance = factor @ factor.transpose(0, 1)
    diagonal = 0.5 + 0.5 * torch.rand(n, generator=generator)
    covariance = covariance + torch.diag(diagonal)
    return 0.5 * (covariance + covariance.transpose(0, 1))


def synthetic_fixture(config: dict, seed: int = 0, n_calibration: int = 64) -> dict:
    """All caller-supplied data products for one CPU smoke/test run."""
    records = synthetic_calibration_noise(config, n_records=n_calibration, seed=seed)
    channel_mean, channel_std = TokenGridEncoder.fit_scaling(records)
    return {
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "noise_bank": synthetic_noise_bank(config, seed=seed),
        "target_noise_covariance": synthetic_target_covariance(config, seed=seed),
    }


def synthetic_batch(config: dict, batch: int = 2, seed: int = 0) -> Tensor:
    """Random raw batch ``(B, C, T)``; identical for every arm in one smoke."""
    generator = _seeded_generator(seed)
    channels = int(config["data"]["channels"])
    samples = int(config["data"]["n_samples"])
    return torch.randn(batch, channels, samples, generator=generator)


def make_batches(config: dict, batch: int = 2, count: int = 10, seed: int = 0) -> List[Tensor]:
    generator = _seeded_generator(seed)
    channels = int(config["data"]["channels"])
    samples = int(config["data"]["n_samples"])
    return [
        torch.randn(batch, channels, samples, generator=generator)
        for _ in range(int(count))
    ]


def train_steps(
    model: BridgeModel,
    optimizers: Sequence,
    batches: Sequence[Tensor],
    n_steps: int,
) -> List[float]:
    """Minimal full-batch step loop; returns the per-step loss values."""
    losses: List[float] = []
    for step in range(int(n_steps)):
        batch = batches[step % len(batches)]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        out["loss"].backward()
        for optimizer in optimizers:
            optimizer.step()
        losses.append(float(out["loss"].detach()))
    return losses


def _beta_placeholder(config: dict) -> float:
    """Middle of the registered beta grid, used only for smoke/unit tests.

    The registered beta is selected on the independent Tier-1 development
    cohort (v3 §1.3); this value is not a selection and is labelled as such.
    """
    grid = config["objective"]["anchor"]["beta_grid"]
    return float(grid[len(grid) // 2])


def smoke_arms(
    config: Optional[dict] = None,
    arms: Sequence[str] = ALL_ARMS,
    batch: int = 2,
    steps: int = 1,
    seed: int = 0,
    fixture: Optional[dict] = None,
    device=None,
    limit_batches: Optional[int] = None,
) -> Dict[str, dict]:
    """One forward+backward+optimizer step per arm on tiny CPU inputs.

    ``fixture`` defaults to the synthetic fixture; pass
    ``a4_bridge_products()`` to run the smoke on the registered A4 data
    products (real fixed scaling and real target marginal of ``N_actual``).
    """
    config = load_config() if config is None else config
    fixture = synthetic_fixture(config, seed=seed) if fixture is None else fixture
    device = torch.device("cpu") if device is None else torch.device(device)
    n_batches = max(1, int(steps))
    if limit_batches is not None:
        n_batches = max(1, min(n_batches, int(limit_batches)))
    batches = [
        batch_tensor.to(device)
        for batch_tensor in make_batches(
            config, batch=batch, count=n_batches, seed=seed + 11
        )
    ]
    results: Dict[str, dict] = {}
    for arm in arms:
        beta = _beta_placeholder(config) if arm == "sigreg_jepa_anchored" else None
        start = time.perf_counter()
        model = build_model(
            config,
            arm,
            channel_mean=fixture["channel_mean"],
            channel_std=fixture["channel_std"],
            target_noise_covariance=fixture.get(
                "target_marginal", fixture["target_noise_covariance"]
            ),
            noise_bank=fixture["noise_bank"],
            beta=beta,
            seed=seed,
        )
        if device.type != "cpu":
            model.to(device)
        optimizers = build_optimizers(model)
        losses = train_steps(model, optimizers, batches, steps)
        elapsed = time.perf_counter() - start
        results[arm] = {
            "seconds": elapsed,
            "losses": losses,
            "metrics": dict(model(batches[-1])["metrics"]),
            "n_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "beta_placeholder": beta,
            "device": str(device),
            "n_batches": len(batches),
        }
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to bridge_v1.json")
    parser.add_argument("--smoke", action="store_true", help="run the CPU smoke")
    parser.add_argument(
        "--a4",
        action="store_true",
        help="use the real registered A4 data products instead of the synthetic fixture",
    )
    parser.add_argument(
        "--freeze-config",
        action="store_true",
        help="fit the A4 data products and write them into the config",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="e.g. cpu or cuda:0; the run manifest records the device",
    )
    parser.add_argument(
        "--limit-batches", type=int, default=None, help="cap the synthetic batches"
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help=("host data root for the frozen N_actual artifact, e.g. "
              "/ceph/dwong/jepa/data/bridge; recorded with path+hash"),
    )
    args = parser.parse_args(argv)
    if args.freeze_config:
        config, products = freeze_bridge_config(args.config, progress=True)
        print(
            json.dumps(
                {
                    "config_path": str(args.config or CONFIG_PATH),
                    "metadata": products["metadata"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.smoke:
        parser.error(
            "this helper implements --smoke and --freeze-config only "
            "(WP5 owns the trainer)"
        )
    config = load_config(args.config)
    fixture = (
        a4_bridge_products(progress=True, artifact_root=args.data_root)
        if args.a4
        else None
    )
    results = smoke_arms(
        config,
        arms=ALL_ARMS,
        batch=args.batch,
        steps=args.steps,
        seed=args.seed,
        fixture=fixture,
        device=args.device,
        limit_batches=args.limit_batches,
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
