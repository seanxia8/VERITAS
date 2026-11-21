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
    default_root = ROOT if kind == dr.ANALYTIC_TOY else RETURNS / LUCID_EXTENSION / "train"
    backend_root = Path(os.environ.get("R2_REPAIR_OUTPUT_ROOT", default_root)).resolve()
    matrix = MATRIX if kind == dr.ANALYTIC_TOY else LUCID_MATRIX
    if args.arm not in matrix[args.detector] or r2.BATCH % args.micro_batch:
        raise ValueError("detector/arm or effective batch is outside the training matrix")
    if args.device == "cpu" and not args.smoke:
        raise ValueError("Stage 5 training requires a GPU")
    tag = run_id(args.detector, args.arm, args.seed)
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
    arm = r2.build_r2_arm(args.arm, geometry, seed, scaling=scaling, run_cfg=cfg,
                          encoder_config=encoder).to(args.device)
    optimizers, optimiser, scheduler = lr.build_optimizers(arm, cfg)
    shuffle = args.arm in ("jepa_repaired", "mae_repaired")
    split = "spatial_patch" if args.arm == "mae_repaired" else "within"
    loader = rs.R2TrainingLoader(cohort, geometry, seed=seed, scaling=scaling, batch=r2.BATCH,
                                 mask_seed=mask_seed, split=split, n_min=int(cfg["views"]["n_min"]),
                                 shuffle_events_within_batch=shuffle)
    snapshots = ((0, args.smoke) if args.smoke else
                 SNAPSHOTS if args.detector in ("wcte", "sk") else (0, 500, 2000, 10000))
    steps = args.smoke or r2.STEPS
    identity = {"run_id": run_id(args.detector, args.arm, seed), "cell_id": f"r2_{args.detector}_within",
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
    config_sha = sha_json(recipe)
    checkpoint = run_dir / "checkpoint.pt"
    record = {"extension": EXTENSION if args.data_kind == dr.ANALYTIC_TOY else LUCID_EXTENSION,
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
    parser.add_argument("--data", type=Path)
    parser.add_argument("--micro-batch", type=int, choices=(1, 2, 4, 8, 16, 32), required=True)
    parser.add_argument("--window", type=int, choices=(128, 256))
    parser.add_argument("--smoke", type=int, choices=(1, 2, 20, 50))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--memory-fraction", type=float)
    args = parser.parse_args()
    if args.memory_fraction is not None and not 0 < args.memory_fraction <= 0.9:
        parser.error("--memory-fraction must be in (0, 0.9]")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
