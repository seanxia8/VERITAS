#!/usr/bin/env python3
"""r2-tier1 T1.7: evaluation entry point (runbook 2026-11-22 §T1.7, §T2.3, §T3.2, Appendix A).

One run (a snapshot step of a trained run, or the legacy init) -> one JSON per test cohort, keyed
by run id and step, with every metric of §T3.2, per-event arrays and bootstrap resamples saved.

**This is the only code that reads sealed packs** (runbook §T2.3). Every sealed read appends a line
(UTC, commit, run id, purpose) to the access log before the pack is opened; a sealed read is refused
unless ``docs/round2/READOUT_SPEC_v1.md`` exists (the readout is frozen first, P5).

Probes are fitted on ``dev_probe_train``; ``dev_probe_test`` is the validation split (model and
early-stopping selection); ``sealed_test`` is applied only.

Usage (from the repository root; also ``python -m scripts.tier1_eval`` under the guard)::

    tier1_eval.py --run-dir <returns>/train/runs/<run_id> --step 10000 --cohorts validation sealed \
        --purpose "E-cell score" [--probes ridge mlp attentive] [--device cuda]
    tier1_eval.py --legacy-init --detector sk --seed 9000 --cohorts validation
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

TIER1_ROOT = Path("/ceph/dwong/jepa/returns/r2_tier1_2026-11-22")
OUT_ROOT = TIER1_ROOT / "eval"
ACCESS_LOG = TIER1_ROOT / "SEALED_ACCESS_LOG.md"
READOUT_SPEC = REPO / "docs/round2/READOUT_SPEC_v1.md"
LUCID_RETURNS = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21")
TRAIN_COHORT = "dev_probe_train"
COHORTS = {"validation": "dev_probe_test", "sealed": "sealed_test"}
PROBES = ("ridge", "mlp", "attentive")
PROBE_SEEDS = (0, 1, 2)
N_BOOT = 2000
BOOT_SEED = 20261122
ATTENTIVE_TOKENS = 512
VERTEX_RIDGE_FORMS = ("all_mean", "spatial_moment_latent", "raw_spatial_moment", "spatial_moment_full",
                      "spatial_moment_latent_allhits")  # last: r2-sparse exploratory P = all hits
MLP_FORMS = ("spatial_moment_full", "spatial_moment_latent")
PRIMARY = "spatial_moment_latent"


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sealed_pack(detector, kind="lucid_v1"):
    from jepa_noise import data_roots as dr
    return dr.sealed_pack(detector, kind)


def cohort_pack(detector, cohort, kind="lucid_v1"):
    """r2-sparse T1.6: the pack of the model's own data kind (lucid_v1 or lucid_v1_w2)."""
    from jepa_noise import data_roots as dr
    return sealed_pack(detector, kind) if cohort == "sealed" else dr.pack(detector, kind)


def code_sha():
    from jepa_noise import lucid_bombs as lb
    return os.environ.get("JEPA_CODE_SHA") or lb.current_code_sha(REPO)


def log_sealed_access(run_key, purpose, log_path=ACCESS_LOG):
    """Append one line to the sealed-access log (created with a header), under an exclusive lock."""
    if not purpose or not purpose.strip():
        raise ValueError("a sealed read needs --purpose")
    if not READOUT_SPEC.is_file():
        raise PermissionError(f"sealed read refused: {READOUT_SPEC} does not exist (freeze the readout first)")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = (f"| {utc()} | {code_sha()[:12]} | {run_key} | {purpose.strip().replace('|', '/')} | "
            f"{socket.gethostname()} |\n")
    with log_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        if not handle.read(1):
            handle.write("# Sealed access log (r2-tier1)\n\nEvery read of a sealed pack, written by "
                         "`scripts/tier1_eval.py` before the pack is opened.\n\n"
                         "| UTC | Commit | Run id @ step | Purpose | Host |\n|---|---|---|---|---|\n")
        handle.seek(0, os.SEEK_END)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return line


# ----------------------------------------------------------------------------- model
def load_model(args):
    """-> (arm, meta) for a snapshot of a trained run, or the legacy (window-64) init."""
    import torch
    from jepa_noise import ext_r2 as r2
    from jepa_noise import r2_store as rs
    from jepa_noise.ext_r2_repair_train import geometry_and_scaling
    from jepa_noise.point_encoder import EncoderConfig
    if args.legacy_init:
        detector, seed = args.detector, int(args.seed)
        geometry, scaling = geometry_and_scaling(detector, "lucid_v1")
        arm = r2.build_r2_arm(r2.INIT_ARM, geometry, seed, scaling=scaling)
        meta = {"run_id": f"{detector}__init__s{seed}", "step": 0, "detector": detector,
                "arm": "init", "seed": seed, "checkpoint": None, "checkpoint_sha256": None,
                "construction": "ext_r2.build_r2_arm('init', geometry, seed) -- EncoderConfig() "
                                "defaults (window 64), as ext_r2_repair_probes.make_model('init')",
                "encoder": None}
        return arm, geometry, scaling, meta
    directory = Path(args.run_dir)
    record = json.loads((directory / "run_record.json").read_text())
    recipe = record["recipe"]
    run_id = recipe["identity"]["run_id"]
    if run_id != directory.name:
        raise ValueError(f"run id {run_id!r} differs from its directory {directory.name!r}")
    detector, arm_name, seed = run_id.split("__")[0], recipe["identity"]["arm"], int(recipe["identity"]["seed"])
    data_kind = recipe.get("data_kind", "lucid_v1")
    geometry, scaling = geometry_and_scaling(detector, data_kind)
    encoder = EncoderConfig(**recipe["encoder"])
    arm = r2.build_r2_arm(arm_name, geometry, seed, scaling=scaling, encoder_config=encoder,
                          sparse_flags=recipe.get("r2_sparse_flags"))
    step = int(args.step)
    path = directory / f"snapshot_step_{step:05d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["step"], payload["identity"], payload["config_sha256"]) != (
            step, recipe["identity"], record["config_sha256"]):
        raise ValueError(f"snapshot identity mismatch at {path}")
    arm.load_state_dict(payload["model"])
    meta = {"run_id": run_id, "step": step, "detector": detector, "arm": arm_name, "seed": seed,
            "checkpoint": str(path), "checkpoint_sha256": rs.sha256_file(path),
            "config_sha256": record["config_sha256"], "encoder": recipe["encoder"],
            "construction": "ext_r2.build_r2_arm(arm, geometry, seed, encoder_config=recipe encoder) "
                            "+ load_state_dict(snapshot)",
            "recipe_extra": {k: recipe[k] for k in ("time_loss_scale_ns", "mask", "charge_loss_weight",
                                                     "time_reference", "view_mode", "r2_sparse_flags",
                                                     "window_ns") if k in recipe},
            "time_reference": recipe.get("time_reference", "trigger"),
            "data_kind": data_kind, "sparse": arm_name in ("sparse_mae", "sparse_time")}
    return arm, geometry, scaling, meta


# ----------------------------------------------------------------------------- streaming
def attentive_pick(event_id, hit_positions):
    from jepa_noise import lucid_bombs as lb
    if hit_positions.size <= ATTENTIVE_TOKENS:
        return hit_positions
    rng = lb.keyed_rng(0, "r2-tier1/attentive", int(event_id))
    return np.sort(rng.choice(hit_positions, ATTENTIVE_TOKENS, replace=False))


def stream_cohort(arm, geometry, scaling, cohort, *, device, chunk, photons_only, attentive, layout,
                  time_reference="trigger", sparse=False):
    """Encode every event of a cohort (photon bombs, and mu/e unless ``photons_only``)."""
    import torch
    from jepa_noise import ext_readout as er
    from jepa_noise import lucid_arms as la
    from jepa_noise import lucid_bombs as lb
    from jepa_noise import r2_store as rs
    from jepa_noise.lucid_views import ViewMasker
    from jepa_noise.r2_probe_repair import moment_features
    from jepa_noise.sparse_readout import all_token_mean, moment_latent_all_hits
    masker = ViewMasker(geometry, "within", seed=0, n_min=0)
    cols = ("event_ids", "run_ids", "domain", "all_mean", "spatial_moment_latent", "spatial_moment_full",
            "raw_spatial_moment", "vertex", "log_yield", "direction", "deposited_MeV", "contained",
            "spatial_moment_latent_allhits")
    rows = {c: [] for c in cols}
    att_z, att_r, att_n = [], [], []
    state = None
    for i, run in enumerate(cohort.run_ids):
        truth = cohort.run_truth(i)
        domain = np.asarray(truth["domain"]).astype(str)
        keep = (domain == lb.DOMAIN_PHOTON) if photons_only else np.isin(
            domain, (lb.DOMAIN_PHOTON,) + tuple(lb.PARTICLE_DOMAINS))
        selected = np.flatnonzero(keep)
        if not selected.size:
            continue
        store = cohort.run_store(i)
        for first in range(0, selected.size, chunk):
            idx = selected[first:first + chunk]
            ids = truth["event_ids"][idx]
            batch = rs.tokenise_sparse(store, geometry, event_ids=ids, scaling=scaling, masker=masker)
            if not np.array_equal(batch.event_ids, ids):
                raise RuntimeError(f"stream lost alignment in {run}")
            with torch.no_grad():  # encoder input in the run's own time reference (T5.1 cell)
                x_in = rs.apply_time_reference(batch.features, time_reference)
                z = arm.encode_tokens(x_in.to(device)).float().cpu().numpy()
            raw = batch.features.numpy()
            latent, full = moment_features(z, raw)
            _, raw_full = moment_features(raw[..., 6:9], raw)
            dom = domain[idx]
            photon = dom == lb.DOMAIN_PHOTON
            if layout and photon.any():
                if state is None:
                    state = er.layout_state(geometry.n_pmt, z.shape[-1])
                er.layout_add(state, z[photon])
            if attentive:
                for e in np.flatnonzero(photon):
                    hits = np.flatnonzero(raw[e, :, 8] > 0.5)
                    pick = attentive_pick(ids[e], hits if hits.size else np.arange(1))
                    zz = np.zeros((ATTENTIVE_TOKENS, z.shape[-1]), np.float16)
                    rr = np.zeros((ATTENTIVE_TOKENS, 3), np.float32)
                    zz[:pick.size] = z[e, pick]
                    rr[:pick.size] = raw[e, pick, 0:3]
                    att_z.append(zz), att_r.append(rr), att_n.append(pick.size)
            rows["event_ids"].append(ids)
            rows["run_ids"].append(np.repeat(str(run), ids.size))
            rows["domain"].append(dom)
            # r2-sparse: sparse encoders return hit tokens scattered onto the PMT grid (zeros at unhit
            # PMTs; sparse_readout adapter), so all-token pooling uses their hit tokens
            rows["all_mean"].append(all_token_mean(z, raw, sparse).astype(np.float32))
            rows["spatial_moment_latent_allhits"].append(moment_latent_all_hits(z, raw).astype(np.float32))
            rows["spatial_moment_latent"].append(latent.astype(np.float32))
            rows["spatial_moment_full"].append(full.astype(np.float32))
            rows["raw_spatial_moment"].append(raw_full.astype(np.float32))
            rows["vertex"].append(np.asarray(truth["vertex_m"], float)[idx])
            rows["log_yield"].append(np.log(np.asarray(truth["yield_photons"], float)[idx]))
            for name, field, default in (("direction", "direction_m", np.nan),
                                         ("deposited_MeV", "deposited_MeV", np.nan),
                                         ("contained", "contained", False)):
                if field in truth:
                    rows[name].append(np.asarray(truth[field])[idx])
                else:
                    shape = (idx.size, 3) if name == "direction" else (idx.size,)
                    rows[name].append(np.full(shape, default))
    out = {c: np.concatenate(v) for c, v in rows.items()}
    out["direction"] = out["direction"].astype(float)
    out["deposited_MeV"] = out["deposited_MeV"].astype(float)
    out["contained"] = out["contained"].astype(bool)
    if attentive:
        out["att_z"], out["att_r"], out["att_n"] = np.stack(att_z), np.stack(att_r), np.asarray(att_n)
    out["layout"] = er.layout_fractions(state) if state is not None else None
    return out


# ----------------------------------------------------------------------------- probes
def ridge_fit(x, y, groups, target_names):
    from jepa_noise import lucid_arms as la
    ds = la.ProbeDataset(kind="tier1", form="tier1", x=np.asarray(x, float), y=np.asarray(y, float),
                         run_ids=np.asarray(groups, dtype=object), target_names=tuple(target_names),
                         meta={})
    return la.fit_probe(ds)


def ridge_predict(fitted, x, n_targets):
    from jepa_noise import lucid_arms as la
    ds = la.ProbeDataset(kind="tier1", form="tier1", x=np.asarray(x, float),
                         y=np.zeros((len(x), n_targets)), run_ids=np.empty(len(x), dtype=object),
                         target_names=tuple(f"t{i}" for i in range(n_targets)), meta={})
    return np.asarray(la.probe_predictions(fitted, ds), float).reshape(len(x), n_targets)


def boot_indices(n, cohort_name):
    """Common resample indices for every run on the same cohort (paired bootstrap)."""
    seed = BOOT_SEED + int(hashlib.sha256(cohort_name.encode()).hexdigest()[:8], 16) % 10_000
    return np.random.default_rng(seed).integers(0, n, size=(N_BOOT, n))


def vertex_summary(errors, radius):
    return {"n": int(errors.size), "mean_m": float(errors.mean()), "median_m": float(np.median(errors)),
            "p68_m": float(np.quantile(errors, 0.68)), "mean_over_r_fid": float(errors.mean() / radius)}


def angles_deg(pred, truth):
    norm = np.linalg.norm(pred, axis=1, keepdims=True)
    unit = np.divide(pred, norm, out=np.zeros_like(pred), where=norm > 0)
    return np.degrees(np.arccos(np.clip((unit * truth).sum(1), -1.0, 1.0)))


def energy_summary(rel):
    q16, q84 = np.quantile(rel, [0.16, 0.84])
    return {"n": int(rel.size), "half_width_68": float(0.5 * (q84 - q16)),
            "rms": float(np.sqrt(np.mean(rel ** 2))), "median_bias": float(np.median(rel))}


def misid(p_mu, is_mu, efficiency=0.99):
    """mu mis-ID at 99 % e efficiency and e mis-ID at 99 % mu efficiency (score = p(mu))."""
    e_scores, mu_scores = p_mu[~is_mu], p_mu[is_mu]
    thr_e = np.quantile(e_scores, efficiency)            # 99 % of e below -> called e
    thr_mu = np.quantile(mu_scores, 1.0 - efficiency)    # 99 % of mu above -> called mu
    return float(np.mean(mu_scores <= thr_e)), float(np.mean(e_scores >= thr_mu))


def fit_attentive(train, val, seed, device, epochs=200, patience=25):
    """Appendix A.4 attentive probe; early stopping on the validation split (mean error, m)."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, scale = train["y"].mean(0), train["y"].std(0)
    scale[scale < 1e-9] = 1.0

    class Probe(nn.Module):
        def __init__(self, d=128, heads=8):
            super().__init__()
            self.w = nn.Linear(d, 64, bias=False)
            self.q = nn.Parameter(torch.randn(heads, 64) * 0.02)
            self.v = nn.Linear(d, 32, bias=False)
            self.norm = nn.LayerNorm(heads * 35)
            self.out = nn.Linear(heads * 35, 3)

        def forward(self, z, r, valid):
            s = torch.einsum("hk,bnk->bhn", self.q, self.w(z)) / 8.0
            s = s.masked_fill(~valid[:, None, :], float("-inf"))
            a = torch.softmax(s, dim=-1)
            pooled = torch.cat([torch.einsum("bhn,bnk->bhk", a, self.v(z)),
                                torch.einsum("bhn,bnk->bhk", a, r)], dim=-1)
            return self.out(self.norm(pooled.flatten(1)))

    def tensors(d):
        valid = np.arange(ATTENTIVE_TOKENS)[None, :] < d["n"][:, None]
        return (torch.as_tensor(d["z"], dtype=torch.float32, device=device),
                torch.as_tensor(d["r"], dtype=torch.float32, device=device),
                torch.as_tensor(valid, device=device))

    tz, tr, tv = tensors(train)
    ty = torch.as_tensor((train["y"] - mean) / scale, dtype=torch.float32, device=device)
    vz, vr, vv = tensors(val)
    model = Probe(d=tz.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    sc = torch.as_tensor(scale, dtype=torch.float32, device=device)
    mu = torch.as_tensor(mean, dtype=torch.float32, device=device)
    vy = torch.as_tensor(val["y"], dtype=torch.float32, device=device)

    def predict(z, r, v):
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(z), 512):
                outs.append(model(z[s:s + 512], r[s:s + 512], v[s:s + 512]) * sc + mu)
        return torch.cat(outs)

    best, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(tz))
        for s in range(0, len(order), 256):
            rows = torch.as_tensor(order[s:s + 256], device=device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(tz[rows], tr[rows], tv[rows]), ty[rows])
            loss.backward()
            opt.step()
        sched.step()
        score = float(torch.linalg.vector_norm(predict(vz, vr, vv) - vy, dim=-1).mean())
        if score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return (lambda d: predict(*tensors(d)).cpu().numpy()), {"best_epoch": best_epoch,
                                                            "best_validation_m": best}


def evaluate(args):
    import torch
    from jepa_noise import ext_followup as ef
    from jepa_noise import ext_r2_repair_probes as probes
    from jepa_noise import lucid_bombs as lb
    from jepa_noise import r2_store as rs
    from jepa_noise.r2_probe_repair import fit_mlp
    started = time.monotonic()
    device = torch.device(args.device)
    if device.type == "cuda" and args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    arm, geometry, scaling, meta = load_model(args)
    arm = arm.to(device).eval()
    run_key = f"{meta['run_id']}@{meta['step']:05d}"
    detector = meta["detector"]
    radius = probes.R_FID[detector]
    out_dir = Path(args.out_root) / meta["run_id"] / f"step_{meta['step']:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(args.cohorts))
    todo = [c for c in requested if args.force or not (out_dir / f"{c}.json").is_file()]
    if not todo:
        print(run_key, "already evaluated", flush=True)
        return 0
    attentive = "attentive" in args.probes
    streams_needed = ["validation"] + [c for c in todo if c == "sealed"]
    if "sealed" in todo:
        log_sealed_access(run_key, args.purpose, args.access_log)
    kind = meta.get("data_kind", "lucid_v1")
    main_pack = args.validation_pack or cohort_pack(detector, "validation", kind)
    packs = {"train": main_pack}
    packs.update({c: (args.sealed_pack or cohort_pack(detector, c, kind)) if c == "sealed" else main_pack
                  for c in streams_needed})
    from jepa_noise import data_roots as dr
    for p in set(packs.values()):  # r2-sparse T1.1: a lucid_v1 / lucid_v1_w2 mix-up raises
        dr.check_pack_kind(p, kind)
    names = {"train": TRAIN_COHORT, **{c: COHORTS[c] for c in streams_needed}}
    data = {}
    for which in ["train"] + streams_needed:
        cohort = rs.open_cohort(packs[which], names[which])
        data[which] = stream_cohort(arm, geometry, scaling, cohort, device=device, chunk=args.chunk,
                                    photons_only=args.photons_only, attentive=attentive,
                                    layout=which != "train",
                                    time_reference=meta.get("time_reference", "trigger"),
                                    sparse=bool(meta.get("sparse")))
        print(run_key, which, len(data[which]["event_ids"]), "events streamed",
              f"{time.monotonic() - started:.0f}s", flush=True)
    tr = data["train"]
    if any(np.intersect1d(tr["run_ids"], data[c]["run_ids"]).size for c in streams_needed):
        raise ValueError("probe-train runs overlap a test cohort")
    ph_tr = tr["domain"] == lb.DOMAIN_PHOTON
    tests = {c: data[c] for c in streams_needed}
    ph = {c: d["domain"] == lb.DOMAIN_PHOTON for c, d in tests.items()}
    res = {c: {"vertex": {}, "direction": {}, "energy": {}, "pid": {}} for c in tests}
    arrays = {c: {"event_ids": d["event_ids"], "run_ids": d["run_ids"].astype(str), "domain": d["domain"]}
              for c, d in tests.items()}
    # --- vertex: Ridge
    for form in VERTEX_RIDGE_FORMS:
        fitted = ridge_fit(tr[form][ph_tr], tr["vertex"][ph_tr], tr["run_ids"][ph_tr], ("x", "y", "z"))
        alphas = [f.get("alpha") for f in fitted["fits"]]
        for c, d in tests.items():
            err = np.linalg.norm(ridge_predict(fitted, d[form][ph[c]], 3) - d["vertex"][ph[c]], axis=1)
            res[c]["vertex"][f"ridge_{form}"] = dict(vertex_summary(err, radius), alpha=alphas)
            arrays[c][f"err_ridge_{form}"] = err
    # --- vertex: MLP (committed fit_mlp, 3 probe seeds)
    if "mlp" in args.probes:
        for form in MLP_FORMS:
            stack = np.concatenate([d[form][ph[c]] for c, d in tests.items()])
            for seed in PROBE_SEEDS:
                pred, info = fit_mlp(tr[form][ph_tr], tr["vertex"][ph_tr], tr["run_ids"][ph_tr], stack,
                                     seed=seed, device=device)
                start = 0
                for c, d in tests.items():
                    n = int(ph[c].sum())
                    err = np.linalg.norm(pred[start:start + n] - d["vertex"][ph[c]], axis=1)
                    start += n
                    res[c]["vertex"][f"mlp_{form}_seed{seed}"] = dict(vertex_summary(err, radius), **info)
                    arrays[c][f"err_mlp_{form}_seed{seed}"] = err
            for c in tests:
                errs = np.stack([arrays[c][f"err_mlp_{form}_seed{s}"] for s in PROBE_SEEDS])
                res[c]["vertex"][f"mlp_{form}_mean3"] = {
                    "mean_m": float(np.mean([e.mean() for e in errs])),
                    "seed_sd_m": float(np.std([e.mean() for e in errs], ddof=1))}
                arrays[c][f"err_mlp_{form}_mean3"] = errs.mean(0)
    # --- vertex: attentive probe (Appendix A.4), early stopping on the validation split
    if attentive:
        def pack(d, mask):
            sub = np.flatnonzero(mask)
            order = np.cumsum(mask) - 1  # position of photon rows inside the photon-only arrays
            k = order[sub]
            return {"z": d["att_z"][k], "r": d["att_r"][k], "n": d["att_n"][k], "y": d["vertex"][sub]}
        train_pack = pack(tr, ph_tr)
        val_pack = pack(tests["validation"], ph["validation"])
        errs_by_cohort = {c: [] for c in tests}
        for seed in PROBE_SEEDS:
            predict, info = fit_attentive(train_pack, val_pack, seed, device)
            for c, d in tests.items():
                p = val_pack if c == "validation" else pack(d, ph[c])
                err = np.linalg.norm(predict(p) - p["y"], axis=1)
                res[c]["vertex"][f"attentive_seed{seed}"] = dict(vertex_summary(err, radius), **info)
                arrays[c][f"err_attentive_seed{seed}"] = err
                errs_by_cohort[c].append(err)
        for c in tests:
            means = [e.mean() for e in errs_by_cohort[c]]
            res[c]["vertex"]["attentive_mean3"] = {"mean_m": float(np.mean(means)),
                                                   "seed_sd_m": float(np.std(means, ddof=1))}
            arrays[c]["err_attentive_mean3"] = np.mean(errs_by_cohort[c], axis=0)
            if c == "validation":
                res[c]["vertex"]["attentive_note"] = "early-stopped on this split (optimistic)"
    # --- particle tasks
    if not args.photons_only:
        part_tr = np.isin(tr["domain"], lb.PARTICLE_DOMAINS)
        for dom in lb.PARTICLE_DOMAINS:
            m_tr = tr["domain"] == dom
            fitted = ridge_fit(tr[PRIMARY][m_tr], tr["direction"][m_tr], tr["run_ids"][m_tr], ("dx", "dy", "dz"))
            for c, d in tests.items():
                m = d["domain"] == dom
                ang = angles_deg(ridge_predict(fitted, d[PRIMARY][m], 3), d["direction"][m])
                res[c]["direction"][f"{dom}_ridge_spatial"] = {
                    "n": int(ang.size), "median_deg": float(np.median(ang)),
                    "q68_deg": float(np.quantile(ang, 0.68))}
                arrays[c][f"ang_{dom}_ridge_spatial"] = ang
                # committed MLP direction probe on all-mean features (continuity, ext_r2.py:1117)
                dd = ef._direction_tasks(tr["all_mean"][m_tr], tr["direction"][m_tr], tr["run_ids"][m_tr],
                                         d["all_mean"][m], d["direction"][m], d["run_ids"][m])
                res[c]["direction"][f"{dom}_committed_mlp_all_mean"] = dd["mlp"]
            c_tr = m_tr & tr["contained"] & (tr["deposited_MeV"] > 0)
            for c, d in tests.items():
                m = (d["domain"] == dom) & d["contained"] & (d["deposited_MeV"] > 0)
                res[c]["energy"][f"{dom}_n_contained"] = int(m.sum())
            if c_tr.sum() < 50:
                for c in tests:
                    res[c]["energy"][f"{dom}_status"] = f"not evaluable: {int(c_tr.sum())} contained train events"
                continue
            fitted = ridge_fit(tr[PRIMARY][c_tr], np.log(tr["deposited_MeV"][c_tr])[:, None],
                               tr["run_ids"][c_tr], ("lnE",))
            for c, d in tests.items():
                m = (d["domain"] == dom) & d["contained"] & (d["deposited_MeV"] > 0)
                if m.sum() < 50:
                    res[c]["energy"][f"{dom}_status"] = f"not evaluable: {int(m.sum())} contained test events"
                    continue
                ln_pred = ridge_predict(fitted, d[PRIMARY][m], 1)[:, 0]
                e_true = d["deposited_MeV"][m]
                rel = np.exp(ln_pred) / e_true - 1.0
                lnt = np.log(e_true)
                r2_ln = 1.0 - np.sum((ln_pred - lnt) ** 2) / np.sum((lnt - lnt.mean()) ** 2)
                r2_e = 1.0 - np.sum((np.exp(ln_pred) - e_true) ** 2) / np.sum((e_true - e_true.mean()) ** 2)
                res[c]["energy"][dom] = dict(energy_summary(rel), r2_lnE=float(r2_ln), r2_E=float(r2_e))
                arrays[c][f"rel_energy_{dom}"] = rel
                arrays[c][f"energy_event_ids_{dom}"] = d["event_ids"][m]
        y_tr = (tr["domain"][part_tr] == lb.DOMAIN_MUON).astype(int)
        for c, d in tests.items():
            m = np.isin(d["domain"], lb.PARTICLE_DOMAINS)
            y = (d["domain"][m] == lb.DOMAIN_MUON).astype(int)
            out = ef._logistic_auc(tr[PRIMARY][part_tr], y_tr, tr["run_ids"][part_tr], d[PRIMARY][m], y)
            p = np.asarray(out.pop("probability"), float)
            mu_mis, e_mis = misid(p, y.astype(bool))
            res[c]["pid"] = {"auc": out.get("auc"), "C": out.get("C"), "n": int(m.sum()),
                             "mu_misid_at_99_e_eff": mu_mis, "e_misid_at_99_mu_eff": e_mis}
            arrays[c]["pid_p_mu"] = p
            arrays[c]["pid_is_mu"] = y
    # --- collapse, bootstrap, write
    for c, d in tests.items():
        if c not in todo:  # e.g. validation streamed only for early stopping; keep the existing file
            continue
        res[c]["collapse"] = {k: (d["layout"] or {}).get(k) for k in ("f_event", "f_layout")}
        boots = {}
        idx = boot_indices(int(ph[c].sum()), COHORTS[c] + "/photon")
        for key in [k for k in arrays[c] if k.startswith("err_")]:
            boots[key] = arrays[c][key][idx].mean(axis=1)
        for dom in lb.PARTICLE_DOMAINS:
            a = arrays[c].get(f"ang_{dom}_ridge_spatial")
            if a is not None:
                bi = boot_indices(a.size, f"{COHORTS[c]}/{dom}/direction")
                boots[f"ang_{dom}_ridge_spatial_median"] = np.median(a[bi], axis=1)
            r = arrays[c].get(f"rel_energy_{dom}")
            if r is not None:
                bi = boot_indices(r.size, f"{COHORTS[c]}/{dom}/energy")
                q = np.quantile(r[bi], [0.16, 0.84], axis=1)
                boots[f"energy_{dom}_half_width_68"] = 0.5 * (q[1] - q[0])
        np.savez_compressed(out_dir / f"{c}.npz", **arrays[c], **{f"boot_{k}": v for k, v in boots.items()})
        row = {"key": run_key, **meta, "cohort": COHORTS[c], "cohort_split": c,
               "pack": str(packs[c]), "pack_sha256": rs.sha256_file(packs[c]) if args.hash_packs else None,
               "probe_train_cohort": TRAIN_COHORT, "probe_train_pack": str(packs["train"]),
               "n_probe_train_photon": int(ph_tr.sum()), "n_test_photon": int(ph[c].sum()),
               "r_fid_m": radius, "probes": list(args.probes), "photons_only": bool(args.photons_only),
               "metrics": res[c], "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "arrays": f"{c}.npz",
                                                "keys": sorted(boots)},
               "code_sha": code_sha(), "host": socket.gethostname(), "device": str(device),
               "readout_spec": str(READOUT_SPEC) if READOUT_SPEC.is_file() else None,
               "wall_seconds": time.monotonic() - started, "written_utc": utc()}
        tmp = out_dir / f".{c}.json.tmp"
        tmp.write_text(json.dumps(row, indent=1, sort_keys=True, default=float) + "\n")
        tmp.replace(out_dir / f"{c}.json")
        print(run_key, c, "spatial Ridge", f"{res[c]['vertex'][f'ridge_{PRIMARY}']['mean_m']:.4f} m", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path)
    src.add_argument("--legacy-init", action="store_true")
    ap.add_argument("--step", type=int, nargs="+", default=[10000],
                    help="one or more snapshot steps, evaluated one after another")
    ap.add_argument("--detector", choices=("wcte", "sk", "juno", "hk"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cohorts", nargs="+", choices=tuple(COHORTS), default=["validation"])
    ap.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES))
    ap.add_argument("--photons-only", action="store_true", help="vertex tasks only (budget axis)")
    ap.add_argument("--purpose", default="")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--access-log", type=Path, default=ACCESS_LOG)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--memory-fraction", type=float, default=None)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--hash-packs", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-only", action="store_true", help="append the sealed-access line and exit")
    ap.add_argument("--sealed-pack", type=Path, default=None, help="batch: staged copy of the sealed pack")
    ap.add_argument("--validation-pack", type=Path, default=None, help="batch: staged copy of the main pack")
    args = ap.parse_args(argv)
    if args.legacy_init and (args.detector is None or args.seed is None):
        ap.error("--legacy-init needs --detector and --seed")
    if "ridge" not in args.probes:
        ap.error("the Ridge readouts are always computed; include 'ridge'")
    out = Path(args.out_root).resolve()
    if out.is_relative_to(LUCID_RETURNS) or out.is_relative_to(LUCID_RETURNS.resolve()):
        ap.error(f"{out}: the r2-lucid returns are read-only")
    steps = [0] if args.legacy_init else list(dict.fromkeys(int(x) for x in args.step))
    if len(steps) > 1:
        rc = 0
        for st in steps:
            sub = argparse.Namespace(**vars(args))
            sub.step = st
            rc |= _single(sub, ap)
        return rc
    args.step = steps[0]
    return _single(args, ap)


def _single(args, ap):
    if args.legacy_init:
        args.step = 0
    if args.log_only:  # batch route: the canonical log line is written here, before the job reads the pack
        if "sealed" not in args.cohorts:
            ap.error("--log-only is for sealed reads")
        if args.legacy_init:
            key = f"{args.detector}__init__s{args.seed}@00000"
        else:
            rid = json.loads((Path(args.run_dir) / "run_record.json").read_text())["recipe"]["identity"]["run_id"]
            key = f"{rid}@{int(args.step):05d}"
        print(log_sealed_access(key, args.purpose + " [batch job; read happens in the job]", args.access_log), end="")
        return 0
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
