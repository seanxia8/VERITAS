"""Stage 5 training entry point for the Round 2 repair campaign."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import torch

from . import ext_r2 as r2
from . import lucid_bombs as lb
from . import lucid_run as lr
from . import r2_generator as gen
from . import r2_geometry as rg
from . import r2_store as rs
from . import data_roots as dr
from .ext_r2_repair_probes import EXTENSION, RETURNS

ROOT = RETURNS / EXTENSION
SEED = 9000
SNAPSHOTS = (0, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
MATRIX = {
    "wcte": ("sigreg_jepa", "target_posfree", "mae_mse", "sigreg_jepa_all_posfree",
             "jepa_repaired", "mae_repaired"),
    "sk": ("sigreg_jepa", "target_posfree", "jepa_repaired", "mae_repaired"),
    "juno": ("jepa_repaired", "mae_repaired"),
    "hk": ("jepa_repaired", "mae_repaired"),
}
WCTE_SHARDS = r2.ROUND1_LUCID_ROOT / "dev_train__F1__0.05__s20261017__n100000x1000"
#: r2-lucid: LUCiD data are the default (data_roots); the training matrix adds SK mae_mse (L7).
LUCID_EXTENSION = "r2_lucid_2026-11-21"
LUCID_MATRIX = {
    "wcte": ("mae_mse", "mae_repaired", "jepa_repaired", "sigreg_jepa"),
    "sk": ("mae_mse", "mae_repaired", "jepa_repaired", "sigreg_jepa"),
    "juno": ("mae_repaired", "jepa_repaired"),
    "hk": ("mae_repaired", "jepa_repaired"),
}


class WcteTrainingShards:
    """Expose immutable Round 1 sparse data through the Round 2 training loader."""

    truth = {}

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.rows = self.manifest["runs"]
        if (self.manifest["cohort"], self.manifest["law"], self.manifest["setting"],
                self.manifest["n_events"]) != ("dev_train", "F1", 0.05, 100_000):
            raise ValueError(f"unexpected WCTE training manifest at {self.directory}")

    def run_rows(self):
        return self.rows

    def run_store(self, index):
        return lb.read_sparse_hits(self.directory / "runs" / self.rows[index]["run_dir"] / "hits.npz")


def run_id(detector, arm, seed=SEED):
    return f"{detector}__{arm}__s{seed}"


def ts_suffix(args):
    """Run-name suffix of the r2-lucid time-scale sweep (empty for the unchanged 50 ns recipe)."""
    ts = getattr(args, "time_loss_scale_ns", None)
    if ts is None or float(ts) == 50.0:
        return ""
    if args.arm != "mae_repaired":
        raise ValueError("--time-loss-scale-ns applies to mae_repaired only")
    return f"__ts{float(ts):g}".replace(".", "p")


#: r2-tier1 (runbook 2026-11-22 T1.2-T1.4): mask and charge-loss weight of mae_repaired.
TIER1_EXTENSION = "r2_tier1_2026-11-22"
MASKS = ("spatial_patch", "random", "within")
DEFAULT_MASK = "spatial_patch"


def mask_suffix(args):
    """``__m<mask>`` for a non-default mae_repaired mask (empty at ``spatial_patch``)."""
    mask = getattr(args, "mask", None)
    if mask is None or mask == DEFAULT_MASK:
        return ""
    if args.arm != "mae_repaired":
        raise ValueError("--mask applies to mae_repaired only")
    if mask not in MASKS:
        raise ValueError(f"unknown mask {mask!r}")
    return f"__m{mask}"


def patch_size_of(args):
    """r2-mask: the non-default ``--patch-size`` of ``spatial_patch`` (``None`` = unchanged recipe).

    A value equal to the detector's default size is the default recipe (no suffix, no recipe
    field, same hash).  Only ``mae_repaired`` with ``--mask spatial_patch`` accepts the flag.
    """
    size = getattr(args, "patch_size", None)
    if size is None:
        return None
    if args.arm != "mae_repaired":
        raise ValueError("--patch-size applies to mae_repaired only")
    if (getattr(args, "mask", None) or DEFAULT_MASK) != "spatial_patch":
        raise ValueError("--patch-size applies to --mask spatial_patch only")
    from .lucid_views import default_patch_size
    n_pmt = dr.geometry(args.detector, getattr(args, "data_kind", None)).n_pmt
    if not 1 <= int(size) <= n_pmt // 2:
        raise ValueError(f"--patch-size must lie in [1, {n_pmt // 2}]")
    return None if int(size) == default_patch_size(n_pmt) else int(size)


def ps_suffix(args):
    """``__ps<size>`` for a non-default spatial-patch size (r2-mask; empty at the default)."""
    size = patch_size_of(args)
    return "" if size is None else f"__ps{size}"


def cw_suffix(args):
    """``__cw<weight>`` for a non-default mae_repaired charge-loss weight (empty at 1)."""
    cw = getattr(args, "charge_loss_weight", None)
    if cw is None or float(cw) == 1.0:
        return ""
    if args.arm != "mae_repaired":
        raise ValueError("--charge-loss-weight applies to mae_repaired only")
    return f"__cw{float(cw):g}".replace(".", "p")


def tr_suffix(args):
    """``__trmedian`` for the conditional T5.1 time-reference cell (empty at ``trigger``)."""
    ref = getattr(args, "time_reference", None)
    if ref is None or ref == "trigger":
        return ""
    if args.arm != "mae_repaired":
        raise ValueError("--time-reference applies to mae_repaired only")
    return f"__tr{ref}"


def run_suffix(args):
    """Run-name suffixes in the registered order ``__m<mask>``, ``__ps<size>`` (r2-mask),
    ``__ts<unit>``, ``__cw<weight>`` (plus ``__tr<reference>`` for the conditional T5.1 cell)."""
    return (mask_suffix(args) + ps_suffix(args) + ts_suffix(args) + cw_suffix(args)
            + tr_suffix(args))


def refuse_lucid_returns(path):
    """New trainings never write under the read-only r2-lucid returns (r2-tier1 T1.4)."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to((RETURNS / LUCID_EXTENSION).resolve()):
        raise PermissionError(f"{resolved}: the {LUCID_EXTENSION} returns are read-only")
    return resolved


def geometry_and_scaling(detector, kind=None):
    geometry = dr.geometry(detector, kind)
    rule = gen.ROUND1_TIME_RULE if detector == "wcte" else gen.time_rule_for(geometry)
    return geometry, rs.scaling_for(geometry, rule.time_scale_ns)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(lr._safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def sha_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def train(args):
    kind = dr.check(getattr(args, "data_kind", None))
    args.data_kind = kind
    args.seed = int(getattr(args, "seed", SEED) or SEED)
    default_root = ROOT if kind == dr.ANALYTIC_TOY else RETURNS / TIER1_EXTENSION / "train"
    backend_root = refuse_lucid_returns(os.environ.get("R2_REPAIR_OUTPUT_ROOT", default_root))
    matrix = MATRIX if kind == dr.ANALYTIC_TOY else LUCID_MATRIX
    if args.arm not in matrix[args.detector] or r2.BATCH % args.micro_batch:
        raise ValueError("detector/arm or effective batch is outside the training matrix")
    if args.device == "cpu" and not args.smoke:
        raise ValueError("Stage 5 training requires a GPU")
    tag = run_id(args.detector, args.arm, args.seed) + run_suffix(args)
    if args.smoke:
        tag += f"__mb{args.micro_batch}__w{args.window or 'default'}"
    run_dir = backend_root / ("smokes" if args.smoke else "runs") / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return train_locked(args, run_dir)


def train_locked(args, run_dir):
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == "cuda":
        if args.memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
        torch.cuda.reset_peak_memory_stats(0)
    p = prepare(args)
    arm, loader, optimizers, scheduler = p["arm"], p["loader"], p["optimizers"], p["scheduler"]
    recipe, config_sha, identity, steps = p["recipe"], p["config_sha"], p["identity"], p["steps"]
    snapshots, source, cfg = p["snapshots"], p["source"], p["cfg"]
    return _run(args, run_dir, started, arm, loader, optimizers, scheduler, recipe, config_sha,
                identity, steps, snapshots, source, cfg)


def prepare(args):
    """Build arm, loader, optimisers and the recipe (and its hash) without training."""
    seed = args.seed
    geometry, scaling = geometry_and_scaling(args.detector, args.data_kind)
    encoder = r2.encoder_config_for(args.arm, geometry)
    if args.window is not None:
        if args.detector not in ("sk", "hk") or args.arm != "jepa_repaired":
            raise ValueError("a window override is only specified for repaired SK/HK JEPA")
        encoder = replace(encoder, window=args.window, shift=args.window // 2)
    cfg = r2.run_config()
    h5 = Path(args.data) if args.data else dr.pack(args.detector, args.data_kind)
    if args.detector == "wcte":
        cohort = WcteTrainingShards(args.data or dr.wcte_train_shards(args.data_kind))
        data_identity = rs.sha256_file(cohort.directory / "manifest.json")
        mask_seed = cohort.manifest["dataset_seed"]
        source = str(cohort.directory)
    else:
        cohort = rs.open_cohort(h5, "dev_train", truth_fields=())
        data_identity = sha_json({"manifests": cohort.manifests,
                                  "state_sha256": cohort.state_sha256})
        mask_seed = lb.DEFAULT_SEED
        source = str(h5)
    time_loss_scale = float(getattr(args, "time_loss_scale_ns", None) or 50.0)
    charge_weight = float(getattr(args, "charge_loss_weight", None) or 1.0)
    run_suffix(args)  # validates the mae_repaired-only flags before anything is built
    arm = r2.build_r2_arm(args.arm, geometry, seed, scaling=scaling, run_cfg=cfg,
                          encoder_config=encoder, time_loss_scale_ns=time_loss_scale,
                          charge_loss_weight=charge_weight).to(args.device)
    optimizers, optimiser, scheduler = lr.build_optimizers(arm, cfg)
    shuffle = args.arm in ("jepa_repaired", "mae_repaired")
    # r2-tier1: the mask is the only thing --mask changes (encoder, loader shuffle unchanged)
    split = (getattr(args, "mask", None) or DEFAULT_MASK) if args.arm == "mae_repaired" else "within"
    patch_size = patch_size_of(args)  # r2-mask: None keeps the unchanged default masker
    loader_extra = {} if patch_size is None else {"patch_size": patch_size}
    loader = rs.R2TrainingLoader(cohort, geometry, seed=seed, scaling=scaling, batch=r2.BATCH,
                                 mask_seed=mask_seed, split=split, n_min=int(cfg["views"]["n_min"]),
                                 shuffle_events_within_batch=shuffle,
                                 time_reference=getattr(args, "time_reference", None) or "trigger",
                                 **loader_extra)
    snapshots = ((0, args.smoke) if args.smoke else
                 SNAPSHOTS if args.detector in ("wcte", "sk") else (0, 500, 2000, 10000))
    steps = args.smoke or r2.STEPS
    identity = {"run_id": run_id(args.detector, args.arm, seed) + run_suffix(args),
                "cell_id": f"r2_{args.detector}_within",
                "arm": args.arm, "seed": seed, "beta": None, "steps_scheduled": steps,
                "gradient_clip": optimiser.get("gradient_clip")}
    training = dict(cfg["training"], micro_batch=args.micro_batch,
                    gradient_accumulation=r2.BATCH // args.micro_batch)
    recipe = {"identity": identity, "encoder": asdict(encoder), "scaling": asdict(scaling),
              "training": training, "geometry_sha256": geometry.digest, "view_mode": split,
              "data_identity_sha256": data_identity, "mask_seed": int(mask_seed),
              "micro_batch": args.micro_batch, "shuffle": shuffle,
              "full_batch_sigreg": args.arm == "jepa_repaired", "snapshot_steps": snapshots,
              "data_kind": args.data_kind}
    if time_loss_scale != 50.0:  # sweep runs only; the unchanged recipe keeps its hash
        recipe["time_loss_scale_ns"] = time_loss_scale
    if split != DEFAULT_MASK and args.arm == "mae_repaired":  # r2-tier1 mask cells only
        recipe["mask"] = split
    if patch_size is not None:  # r2-mask patch-size cells only; the default keeps its hash
        recipe["patch_size"] = patch_size
    if charge_weight != 1.0:  # r2-tier1 P5 cell only; defaults keep the recipe hash
        recipe["charge_loss_weight"] = charge_weight
    if (getattr(args, "time_reference", None) or "trigger") != "trigger":  # T5.1 cell only
        recipe["time_reference"] = args.time_reference
    config_sha = sha_json(recipe)
    return {"arm": arm, "loader": loader, "optimizers": optimizers, "scheduler": scheduler,
            "recipe": recipe, "config_sha": config_sha, "identity": identity, "steps": steps,
            "snapshots": snapshots, "source": source, "cfg": cfg}


def _run(args, run_dir, started, arm, loader, optimizers, scheduler, recipe, config_sha,
         identity, steps, snapshots, source, cfg):
    checkpoint = run_dir / "checkpoint.pt"
    record = {"extension": EXTENSION if args.data_kind == dr.ANALYTIC_TOY else TIER1_EXTENSION,
              "recipe": recipe, "config_sha256": config_sha,
              "source": source, "code_sha": os.environ.get("JEPA_CODE_SHA") or lb.current_code_sha(),
              "host": socket.gethostname(), "device": args.device,
              "gpu_model": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "registered": False}
    if checkpoint.exists() and args.smoke:
        raise FileExistsError(checkpoint)
    write_json(run_dir / "recipe.json", record)
    result = lr.train_loop(arm=arm, loader=loader, steps=steps, optimizers=optimizers,
                           checkpoint_path=checkpoint, identity=identity, config_sha=config_sha,
                           checkpoint_minutes=cfg["training"]["checkpoint_minutes"],
                           checkpoint_every=cfg["training"]["checkpoint_every_steps"],
                           resume=checkpoint if checkpoint.exists() else None,
                           log_path=run_dir / "train_log.jsonl", log_every=1,
                           accum_steps=r2.BATCH // args.micro_batch, scheduler=scheduler,
                           snapshot_steps=snapshots, full_batch_sigreg=recipe["full_batch_sigreg"])
    record.update(result, wall_seconds=time.monotonic() - started,
                  peak_memory_gib=(torch.cuda.max_memory_reserved(0) / 2**30
                                   if args.device == "cuda" else None),
                  complete=result["step"] == steps and not result["interrupted"])
    write_json(run_dir / "run_record.json", record)
    print(json.dumps({key: record[key] for key in ("complete", "step", "elapsed_seconds",
                                                     "wall_seconds", "peak_memory_gib")}), flush=True)
    return 0 if record["complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detector", choices=MATRIX)
    parser.add_argument("arm", choices=sorted(set().union(*MATRIX.values(), *LUCID_MATRIX.values())))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-kind", choices=dr.DATA_KINDS, default=dr.DEFAULT_DATA_KIND)
    parser.add_argument("--time-loss-scale-ns", type=float, default=None,
                        help="mae_repaired time-loss unit (default 50 ns, the unchanged recipe)")
    parser.add_argument("--mask", choices=MASKS, default=None,
                        help="mae_repaired mask (default spatial_patch, the unchanged recipe)")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="r2-mask: spatial_patch size in PMTs (default: the unchanged size)")
    parser.add_argument("--charge-loss-weight", type=float, default=None,
                        help="mae_repaired log-charge loss weight (default 1, the unchanged recipe)")
    parser.add_argument("--time-reference", choices=("trigger", "median"), default=None,
                        help="mae_repaired token-time reference (default trigger, the unchanged recipe)")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--micro-batch", type=int, choices=(1, 2, 4, 8, 16, 32), required=True)
    parser.add_argument("--window", type=int, choices=(128, 256))
    parser.add_argument("--smoke", type=int, choices=(1, 2, 20, 50))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--memory-fraction", type=float)
    args = parser.parse_args()
    if args.memory_fraction is not None and not 0 < args.memory_fraction <= 0.9:
        parser.error("--memory-fraction must be in (0, 0.9]")
    if args.charge_loss_weight is not None and not (args.charge_loss_weight > 0
                                                    and args.charge_loss_weight < float("inf")):
        parser.error("--charge-loss-weight must be positive and finite")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
