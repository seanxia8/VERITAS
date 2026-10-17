"""Tiny CPU helper for the WP4a bridge smoke, tests and config freeze -- NOT WP5.

This module provides (a) the real A4 bridge data products from the frozen
``views.py`` API (fixed per-channel scaling, anchor overlay bank, target
marginal of ``N_actual``, and their hashes), (b) a deterministic synthetic
fixture used only for masking/architecture unit tests, and (c) a minimal step
loop for the CPU smoke.  It implements no cohort handling, no checkpoints, no
schedules and no evaluation; those belong to WP5 and are out of scope here.

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


def write_bridge_artifact(products: dict, root) -> str:
    """Write the 1024x1024 target marginal once under ``root/<sha256>/``.

    Returns the artifact directory.  The file is checked against the frozen
    sha256 if it already exists (never silently overwritten).  ``root`` is the
    host data root, e.g. ``/ceph/dwong/jepa/data/bridge``.
    """
    from .masks import raw_split

    metadata = products["metadata"]
    target = metadata["target_noise_covariance"]
    frozen_sha = target["sha256"]
    index = raw_split("channel").target
    covariance = np.asarray(products["target_noise_covariance"], dtype=np.float64)
    marginal = covariance[np.ix_(index, index)]
    directory = Path(root) / frozen_sha
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "N_actual_target_marginal.npy"
    if path.exists():
        existing = np.load(path)
        if not np.array_equal(existing, marginal):
            raise RuntimeError(
                f"existing bridge artifact {path} does not match the frozen "
                "N_actual target marginal"
            )
    else:
        np.save(path, marginal)
    file_sha = _array_sha256(marginal, "<f8")
    record = {
        "target_marginal_sha256": frozen_sha,
        "file_sha256": file_sha,
        "shape": list(marginal.shape),
        "dtype": "<f8",
        "generator_manifest_sha256": metadata["generator"]["manifest_sha256"],
        "views_constructor": metadata["generator"]["views_constructor"],
        "path": str(path),
    }
    (directory / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["target_noise_covariance"]["artifact_path"] = str(path)
    metadata["target_noise_covariance"]["artifact_file_sha256"] = file_sha
    return str(directory)


def verify_bridge_products(config: dict, products: dict) -> None:
    """Raise unless the products reproduce the frozen config hashes exactly."""
    frozen_target = config["data"]["target_noise_covariance"].get("sha256")
    computed_target = products["metadata"]["target_noise_covariance"]["sha256"]
    if frozen_target is not None and frozen_target != computed_target:
        raise RuntimeError(
            "computed N_actual target marginal sha256 "
            f"{computed_target} does not match the frozen config sha256 "
            f"{frozen_target}"
        )
    frozen_scaling = config["data"]["fixed_scaling"].get("values_mean_std")
    if frozen_scaling:
        expected_mean = torch.as_tensor(frozen_scaling["mean"], dtype=torch.float32)
        expected_std = torch.as_tensor(frozen_scaling["std"], dtype=torch.float32)
        if not torch.equal(expected_mean, products["channel_mean"]):
            raise RuntimeError(
                "computed fixed per-channel mean does not match the frozen config"
            )
        if not torch.equal(expected_std, products["channel_std"]):
            raise RuntimeError(
                "computed fixed per-channel std does not match the frozen config"
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

    Returns ``channel_mean``/``channel_std`` (fixed scaling fitted once on
    ``n_calibration`` disjoint calibration-noise records), ``noise_bank``
    (independent ``noise_overlay`` draws for the anchor), the full raw
    ``N_actual`` covariance (the ``mae_chi2`` model slices the target marginal),
    and a ``metadata`` dict with the sha256 hashes written into the reviewed
    config.  With ``verify`` (default) the computed products are checked against
    the frozen config hashes and a mismatch raises.  With ``artifact_root`` (or
    the ``JEPA_BRIDGE_DATA_ROOT`` environment variable) the target marginal is
    written once under ``<root>/<sha256>/`` and the path+hash are recorded.
    """
    from .masks import raw_split

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
            print(f"calibration noise {stop}/{n_calibration}", file=sys.stderr, flush=True)
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
    split = raw_split(views.split_name)
    covariance = np.asarray(views.noise_covariance)
    target_index = np.asarray(split.target, dtype=np.int64)
    marginal = covariance[np.ix_(target_index, target_index)]
    manifest = views.manifest()
    mean = [float(value) for value in channel_mean.tolist()]
    std = [float(value) for value in channel_std.tolist()]
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
            "noise_covariance_sha256": manifest["noise_covariance_sha256"],
            "manifest_sha256": _canonical_sha256(manifest),
        },
        "fixed_scaling": {
            "mean": mean,
            "std": std,
            "sha256": _array_sha256(
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
        },
        "target_noise_covariance": {
            "sha256": _array_sha256(marginal, "<f8"),
            "n_target": int(marginal.shape[0]),
            "target_raw_index_sha256": _array_sha256(target_index, "<i8"),
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
            "sha256": _array_sha256(bank, "<f8"),
        },
    }
    products = {
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "noise_bank": torch.as_tensor(bank, dtype=torch.float32),
        "target_noise_covariance": covariance,
        "metadata": metadata,
    }
    if verify:
        frozen = load_config() if config is None else config
        verify_bridge_products(frozen, products)
        metadata["verified_against_config"] = True
    else:
        metadata["verified_against_config"] = False
    if artifact_root is None:
        artifact_root = os.environ.get("JEPA_BRIDGE_DATA_ROOT")
    if artifact_root:
        write_bridge_artifact(products, artifact_root)
    return products


def freeze_bridge_config(
    config_path=None, progress: bool = False
) -> tuple:
    """Write the two A4 data products into the reviewed config.

    This freezes data-derived values only; it selects no hyper-parameter and
    uses no outcome.  The full ``N_actual`` covariance is not stored in the
    repository (lean-repository rule); its sha256 and deterministic recipe are
    recorded, and ``a4_bridge_products`` recomputes it.
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
            "sha256": scaling["sha256"],
            "role": scaling["role"],
            "event_ids": scaling["event_ids"],
            "n_records": scaling["n_records"],
            "seed": scaling["seed"],
            "cell": scaling["cell"],
            "status": (
                "frozen from the registered A4 generator (v4-A4-waveform-v1, "
                "base_rho 0, g 0.02, channel split); never refitted by the "
                "bridge, robustness shifts reuse these values"
            ),
        }
    )
    target = metadata["target_noise_covariance"]
    data["target_noise_covariance"].update(
        {
            "sha256": target["sha256"],
            "n_target": target["n_target"],
            "target_raw_index_sha256": target["target_raw_index_sha256"],
            "generator_manifest_sha256": metadata["generator"]["manifest_sha256"],
            "generator_noise_covariance_sha256": metadata["generator"][
                "noise_covariance_sha256"
            ],
            "status": (
                "frozen; matrix not stored (lean-repository rule): recomputed "
                "deterministically by jepa_noise.train.a4_bridge_products()"
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
        json.dump(config, handle, indent=2)
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
            target_noise_covariance=fixture["target_noise_covariance"],
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
