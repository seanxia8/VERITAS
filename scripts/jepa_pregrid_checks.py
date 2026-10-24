# SPDX-License-Identifier: MIT
"""Pre-grid diagnostics for LUCiD SIGReg-JEPA (extension, registered: false).

Author request, 24 Oct 2026 (item 4 of that message): prepare the checks to run
before the LUCiD grid resumes.  Nothing here edits a registered file, config or
verdict; every output is written under ``--out`` and labelled ``registered: false``.

Modes
-----
inspect   (no training; Titan is fine)
    For each checkpoint given as ``label=arm:path`` (and for random-init encoders
    given as ``label=arm:init:<seed>``):
      * registered re-evaluation (full unmasked input) -> must reproduce the stored
        photon vertex resolution (acceptance: |diff| <= 1e-3 m);
      * "half-pass" evaluation: every token is computed the way it was during
        training (context tokens from the context-only pass, target tokens from the
        target-only pass), then the same registered probes run on it;
      * event-level statistics on the photon probe-test events: effective rank and
        participation ratio of per-event mean-token embeddings (full and half-pass),
        a per-PMT ln(q) probe on hit-PMT tokens (amplitude preservation), and the
        SIGReg statistic of the per-event embeddings.
train     (L40S; one run per call)
    Trains one extension variant on the development cell with the registered
    run_one() machinery and evaluates it (registered probes + half-pass probes):
      across        sigreg_jepa, split 'across' (is 'within' too local a task?)
      strong_drift  sigreg_jepa, module-gain spread s = 0.30 (is 5 % too weak to absorb?)
      stopgrad      sigreg_jepa with labels detached in the prediction loss only
      lambda        sigreg_jepa with SIGReg coefficient --lam (default 0.2, 10x)
      ijepa         ijepa_ema on the development cell (EMA target encoder)
reselect-beta
    Recomputes the A10 step-3 beta selections from a frozen step-3 decision file
    with a different plain reference (the fixed-label sigreg_jepa resolution).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

EXTENSION = "pregrid_checks_2026-10-24"
VARIANTS = ("across", "strong_drift", "stopgrad", "lambda", "ijepa")
STRONG_DRIFT_S = 0.30


# --------------------------------------------------------------------------- encoders
class HalfPassEncoder(torch.nn.Module):
    """Encodes each token in the pass it saw during training (within split only).

    ``encode(x)`` with no mask returns context-position outputs from the
    context-only pass and target-position outputs from the target-only pass, so the
    encoder never sees an input layout it was not trained on.  An explicit
    ``hidden_mask`` is passed through unchanged (the 'propagated' probe form).
    """

    def __init__(self, base, target_mask: np.ndarray):
        super().__init__()
        self.base = base
        self.register_buffer("target", torch.as_tensor(np.asarray(target_mask, bool)))

    def encode(self, features, hidden_mask=None):
        if hidden_mask is not None:
            return self.base.encode(features, hidden_mask=hidden_mask)
        t = self.target.to(features.device).unsqueeze(0).expand(features.shape[0], -1)
        z_ctx = self.base.encode(features, hidden_mask=t)
        z_tgt = self.base.encode(features, hidden_mask=~t)
        return torch.where(t.unsqueeze(-1), z_tgt, z_ctx)


def within_target_mask(geometry) -> np.ndarray:
    """The registered fixed 'within' target set: odd 1-based pmt_id_in_mpmt."""
    return (np.asarray(geometry.pmt_id_in_mpmt) % 2) == 1


# --------------------------------------------------------------------------- arm patches
def patch_stopgrad(arm):
    """Detach labels in the prediction loss only; SIGReg still sees live labels."""
    def _forward_sigreg(batch):
        context, prediction, labels, _, full = arm._encode_views(batch)
        prediction_loss = F.mse_loss(prediction, labels.detach())
        regulariser = arm.sigreg(arm._context_pooled(context, batch.context_index), labels)
        loss = prediction_loss + arm.coefficient_lambda * regulariser["value"]
        metrics = {"loss": float(loss.detach()), "prediction": float(prediction_loss.detach()),
                   "sigreg": float(regulariser["value"].detach()),
                   "sigreg_context": float(regulariser["context"].detach()),
                   "sigreg_target": float(regulariser["target"].detach()),
                   "label_stopgrad": True}
        return {"loss": loss, "metrics": metrics, "prediction": prediction,
                "context_tokens": context, "target_tokens": labels, "full_readout": full}
    arm._forward_sigreg = _forward_sigreg
    return arm


def patch_lambda(arm, lam: float):
    arm.coefficient_lambda = float(lam)
    return arm


# --------------------------------------------------------------------------- statistics
def effective_rank(x: np.ndarray) -> dict:
    x = np.asarray(x, float) - np.asarray(x, float).mean(0)
    w = np.clip(np.linalg.eigvalsh(x.T @ x / max(len(x) - 1, 1)), 0, None)
    if w.sum() <= 0:
        return {"effective_rank": 0.0, "participation_ratio": 0.0}
    p = w / w.sum(); p = p[p > 0]
    return {"effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "participation_ratio": float(w.sum() ** 2 / (w ** 2).sum())}


def epps_pulley(x: np.ndarray, n_slices: int = 256, seed: int = 0) -> float:
    """Numpy replica of the pinned SIGReg statistic (17 nodes on [0, 3], times N)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(x.shape[1], n_slices)); dirs /= np.linalg.norm(dirs, axis=0)
    proj = x @ dirs
    t = np.linspace(0, 3, 17); dt = 3 / 16
    w = np.full(17, 2 * dt); w[[0, -1]] = dt; phi = np.exp(-.5 * t ** 2); w = w * phi
    xt = proj[..., None] * t
    err = (np.cos(xt).mean(0) - phi) ** 2 + np.sin(xt).mean(0) ** 2
    return float((err @ w * len(x)).mean())


def ridge_r2(xtr, ytr, xte, yte, alphas=(1e-4, 1e-2, 1, 100)) -> dict:
    """Standardised ridge, alpha chosen on the last 20% of train, scored on test."""
    xtr, xte = np.asarray(xtr, float), np.asarray(xte, float)
    ytr, yte = np.asarray(ytr, float).ravel(), np.asarray(yte, float).ravel()
    mu, sd = xtr.mean(0), xtr.std(0); sd[sd == 0] = 1
    ztr, zte = (xtr - mu) / sd, (xte - mu) / sd
    cut = int(0.8 * len(ztr))

    def fit(z, y, a):
        g = z.T @ z + a * np.eye(z.shape[1])
        return np.linalg.solve(g, z.T @ (y - y.mean())), y.mean()
    best = None
    for a in alphas:
        coef, off = fit(ztr[:cut], ytr[:cut], a)
        err = float(((ztr[cut:] @ coef + off - ytr[cut:]) ** 2).mean())
        if best is None or err < best[1]:
            best = (a, err)
    coef, off = fit(ztr, ytr, best[0])
    pred = zte @ coef + off
    sst = float(((yte - yte.mean()) ** 2).sum())
    return {"r2": float(1 - ((yte - pred) ** 2).sum() / sst) if sst > 0 else None,
            "alpha": best[0], "n_train": int(len(ytr)), "n_test": int(len(yte))}


def event_level(model, dataset, geometry, *, split, mask_seed, device, domain="photon",
                max_events=4000, tokens_per_event=32, seed=0):
    """Per-event mean-token embeddings and a hit-PMT token sample with ln(q)."""
    scaling = pe.FeatureScaling.from_geometry(geometry)
    rng = np.random.default_rng(seed)
    emb, tok_x, tok_y, n = [], [], [], 0
    model.eval()
    for row in dataset.runs:
        if n >= max_events:
            break
        batch = lr._probe_batch(dataset.directory / "runs" / row["run_dir"], geometry, split=split,
                                mask_seed=mask_seed, occupancy="natural", scaling=scaling,
                                domain=domain)
        if batch.features.shape[0] == 0:
            continue
        with torch.no_grad():
            tokens = model.encode(lr._to_device(batch, device).features).cpu().numpy()
        emb.append(tokens.mean(1)); n += tokens.shape[0]
        hit = np.asarray(batch.hit_flags, bool); lq = np.asarray(batch.log_charge, float)
        for e in range(tokens.shape[0]):
            idx = np.flatnonzero(hit[e])
            if idx.size:
                pick = rng.choice(idx, size=min(tokens_per_event, idx.size), replace=False)
                tok_x.append(tokens[e, pick]); tok_y.append(lq[e, pick])
    emb = np.concatenate(emb)[:max_events]
    return emb, np.concatenate(tok_x), np.concatenate(tok_y)


# --------------------------------------------------------------------------- setup
def setup(args):
    config = lr.load_run_config(args.config)
    gen = lb.load_generator_config(args.generator_config or (ROOT / config["data"]["generator_config"]))
    arms_config = la.load_config(args.arms_config)
    geometry = lb.load_wcte_geometry(args.geometry)
    device = lr.configure_runtime(args.device)
    return config, gen, arms_config, geometry, device


def variant_cell(config, variant):
    dev = lr.dev_cell(config)
    if variant == "across":
        return dataclasses.replace(dev, cell_id="ext_dev_s005_across", split="across")
    if variant == "strong_drift":
        return dataclasses.replace(dev, cell_id="ext_dev_s030_within", setting=STRONG_DRIFT_S,
                                   s_arm=STRONG_DRIFT_S)
    return dataclasses.replace(dev, cell_id=f"ext_dev_s005_within_{variant}")


def datasets_for(cell, config, gen, geometry, data_root, code_sha=None):
    seed = int(config["data"]["dataset_seed"])
    common = dict(data_root=data_root, geometry=geometry, generator_config=gen, seed=seed,
                  code_sha=code_sha, log=print)
    train = lr.ensure_dataset(cohort_spec=config["data"]["cohorts"][cell.train_cohort],
                              cohort_name=cell.train_cohort, law="F1", setting=cell.setting, **common)
    probes = {role: lr.ensure_dataset(cohort_spec=config["data"]["cohorts"][name], cohort_name=name,
                                      law="F1", setting=cell.setting, **common)
              for role, name in (("train", "dev_probe_train"), ("test", "dev_probe_test"))}
    return train, probes


def build_loaded(arm_name, cell, geometry, arms_config, config, gen, data_root, source, seed):
    dark = None
    if arm_name in la.ANCHOR_ARMS:
        d = lr.ensure_dark_bank(data_root=data_root, geometry=geometry, generator_config=gen,
                                seed=int(config["data"]["dataset_seed"]),
                                records=int(config["data"]["dark_noise"]["records"]))
        dark = la.dark_bank_from_store(d / "hits.npz", geometry)
    torch.manual_seed(int(seed)); np.random.seed(int(seed) % 2 ** 32)
    arm = lr.build_arm(arm_name, geometry, arms_config, cell, seed, dark_bank=dark)
    if source != "init":
        payload = torch.load(source, map_location="cpu", weights_only=False)
        arm.load_state_dict(payload["model"])
    return arm


def photon_resolution(payload):
    dom = (payload.get("domain_results") or {}).get("photon") or {}
    return (dom.get("physics") or payload.get("physics") or {}).get("resolution")


def evaluate_forms(arm, arm_name, cell, geometry, probes, config, device, run_dir):
    """Registered probes on the full input and on the half-pass encoder."""
    mask_seed = int(config["views"]["mask_seed"])
    out = {}
    base = getattr(arm, "jepa", arm).to(device)
    forms = {"full_input": base}
    if cell.split == "within":
        forms["half_pass"] = HalfPassEncoder(base, within_target_mask(geometry)).to(device)
    for name, model in forms.items():
        d = Path(run_dir) / name; d.mkdir(parents=True, exist_ok=True)
        payload = lr.evaluate_arm_run(model, arm_name=arm_name, cell=cell, geometry=geometry,
                                      probe_datasets=probes, config=config, device=device,
                                      mask_seed=mask_seed, save_embeddings=False, run_dir=d)
        res = payload["domain_results"]["photon"]["results"] if "domain_results" in payload else payload["results"]
        out[name] = {"vertex_resolution_m": photon_resolution(payload),
                     "module_gain_run_mean_r2": (res.get("module_gain::run_mean") or {}).get("r2_mean"),
                     "log_yield_r2": (res.get("log_yield::per_event") or {}).get("r2_mean")}
        tr = event_level(model, probes["train"], geometry, split=cell.split, mask_seed=mask_seed, device=device)
        te = event_level(model, probes["test"], geometry, split=cell.split, mask_seed=mask_seed, device=device)
        out[name].update(event_embedding=effective_rank(te[0]),
                         sigreg_statistic_events=epps_pulley(te[0]),
                         token_log_charge_r2=ridge_r2(tr[1], tr[2], te[1], te[2]))
    return out


# --------------------------------------------------------------------------- modes
def mode_inspect(args):
    config, gen, arms_config, geometry, device = setup(args)
    cell = variant_cell(config, args.cell_variant) if args.cell_variant else lr.dev_cell(config)
    _, probes = datasets_for(cell, config, gen, geometry, Path(args.data_root))
    report = {"registered": False, "extension": EXTENSION, "mode": "inspect",
              "cell": cell.cell_id, "items": {}}
    for item in args.item:
        label, spec = item.split("=", 1)
        arm_name, rest = spec.split(":", 1)
        if rest.startswith("init:"):
            source, seed = "init", int(rest.split(":", 1)[1])
        else:
            source, seed = rest, 0
        arm = build_loaded(arm_name, cell, geometry, arms_config, config, gen, Path(args.data_root),
                           source, seed)
        entry = evaluate_forms(arm, arm_name, cell, geometry, probes, config, device,
                               Path(args.out) / label)
        if args.expect:
            for exp in args.expect:
                el, value = exp.split("=")
                if el == label:
                    got = entry["full_input"]["vertex_resolution_m"]
                    entry["reproduction"] = {"expected_m": float(value), "got_m": got,
                                             "pass": got is not None and abs(got - float(value)) <= 1e-3}
        report["items"][label] = {"arm": arm_name, "source": source, "seed": seed, **entry}
        print(label, json.dumps(entry["full_input"]), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "inspect.json").write_text(json.dumps(report, indent=1, default=float))


def mode_train(args):
    config, gen, arms_config, geometry, device = setup(args)
    cell = variant_cell(config, args.variant)
    arm_name = "ijepa_ema" if args.variant == "ijepa" else "sigreg_jepa"
    data_root, out_root = Path(args.data_root), Path(args.out)
    train_ds, probes = datasets_for(cell, config, gen, geometry, data_root)
    original = lr.build_arm

    def build(arm, geometry, arms_config, cell, seed, beta=None, dark_bank=None, encoder_config=None):
        made = original(arm, geometry, arms_config, cell, seed, beta=beta, dark_bank=dark_bank,
                        encoder_config=encoder_config)
        if args.variant == "stopgrad":
            patch_stopgrad(made)
        if args.variant == "lambda":
            patch_lambda(made, args.lam)
        return made
    lr.build_arm = build
    try:
        result = lr.run_one(arm_name=arm_name, cell=cell, seed=int(args.seed), beta=None, config=config,
                            arms_config=arms_config, geometry=geometry, loader_dataset=train_ds,
                            probe_datasets=probes, dark_bank=None, out_root=out_root, data_root=data_root,
                            config_sha=lr.sha256_file(args.config or lr.CONFIG_PATH),
                            code_sha=lr.git_head(ROOT), status_text=lr.git_status(ROOT), device=device,
                            steps=args.steps, evaluate=True, probe_max_events=args.probe_max_events,
                            physics_method="ridge", save_embeddings=False)
    finally:
        lr.build_arm = original
    run_dir = Path(result["run_dir"])
    final = run_dir / "final.pt"
    arm = build_loaded(arm_name, cell, geometry, arms_config, config, gen, data_root, str(final), args.seed)
    if args.variant == "stopgrad":
        patch_stopgrad(arm)
    forms = evaluate_forms(arm, arm_name, cell, geometry, probes, config, device, run_dir / "ext_eval")
    log = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text().splitlines() if line.strip()]
    summary = {"registered": False, "extension": EXTENSION, "variant": args.variant, "arm": arm_name,
               "cell": cell.cell_id, "split": cell.split, "spread": cell.setting, "seed": int(args.seed),
               "lambda": (args.lam if args.variant == "lambda" else None),
               "final_train_metrics": log[-1] if log else None, "forms": forms,
               "wall_seconds": result["manifest"].get("wall_seconds")}
    (run_dir / "ext_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary["forms"], default=float), flush=True)


def mode_reselect(args):
    decision = json.loads(Path(args.decision).read_text())
    plain = float(args.plain_resolution)
    out = {"registered": False, "extension": EXTENSION, "source_decision": str(args.decision),
           "plain_resolution": plain, "plain_source": args.plain_source,
           "previous_plain_resolution": decision.get("plain_resolution"), "selections": {}}
    for arm, sel in decision["selections"].items():
        out["selections"][arm] = lr.select_beta(sel["table"], plain)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v["status"], v["selected"]) for k, v in out["selections"].items()}))


def parser():
    p = argparse.ArgumentParser(prog="jepa_pregrid_checks")
    common = argparse.ArgumentParser(add_help=False)
    for flag in ("--out", "--data-root", "--geometry"):
        common.add_argument(flag, required=True)
    common.add_argument("--config", default=None)
    common.add_argument("--generator-config", default=None)
    common.add_argument("--arms-config", default=None)
    common.add_argument("--device", default="auto")
    sub = p.add_subparsers(dest="mode", required=True)
    i = sub.add_parser("inspect", parents=[common])
    i.add_argument("--item", action="append", required=True,
                   help="label=arm:/path/final.pt  or  label=arm:init:<seed>")
    i.add_argument("--expect", action="append", default=[], help="label=<stored vertex resolution m>")
    i.add_argument("--cell-variant", choices=VARIANTS, default=None)
    t = sub.add_parser("train", parents=[common])
    t.add_argument("--variant", choices=VARIANTS, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--lam", type=float, default=0.2)
    t.add_argument("--steps", type=int, default=None)
    t.add_argument("--probe-max-events", type=int, default=None)
    r = sub.add_parser("reselect-beta")
    r.add_argument("--decision", required=True); r.add_argument("--plain-resolution", required=True)
    r.add_argument("--plain-source", required=True); r.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    {"inspect": mode_inspect, "train": mode_train, "reselect-beta": mode_reselect}[args.mode](args)


if __name__ == "__main__":
    main()
