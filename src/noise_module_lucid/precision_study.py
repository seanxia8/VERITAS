"""kappa_m precision study (author decision, 18 Oct 2026): SD of K under repeated reference runs.

The same physical placements are read out R times with independent keyed readout seeds (the
photon transport and placement are fixed). Frozen checkpoints predict every realisation. For
disjoint pairs of realisations (2j, 2j+1), K_vertex = mean vertex error(2j+1) / mean(2j) and
K_energy likewise for the relative energy error, and K = max(K_vertex, K_energy). Their SD
over pairs and model seeds is the null spread of K; kappa_m = max(1.10, 1 + 2 SD_K).

Pure NumPy apart from :func:`predict_checkpoints`, which needs torch (CPU is enough).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

KAPPA_FLOOR = 1.10


def placement_errors(pred, truth):
    """Per-placement vertex error (m) and relative energy error; energy columns are log(E/1000 MeV)."""
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    vertex = np.linalg.norm(p[..., 3:6] - t[..., 3:6], axis=-1)
    energy = np.abs(np.exp(p[..., 6] - t[..., 6]) - 1)
    return vertex, energy


def null_k(errors):
    """K for disjoint realisation pairs; ``errors`` has shape (R, n_placements)."""
    e = np.asarray(errors, float)
    if e.ndim != 2 or e.shape[0] < 2:
        raise ValueError("(R >= 2, n) errors required")
    means = e.mean(axis=1)
    pairs = len(means) // 2
    return means[1:2*pairs:2] / means[0:2*pairs:2]


def kappa_from_sd(sd):
    return float(max(KAPPA_FLOOR, 1 + 2*sd))


def summarise(vertex, energy):
    """``vertex``/``energy`` have shape (seeds, R, n); returns the null spread of K and kappa_m."""
    kv = np.concatenate([null_k(v) for v in vertex])
    ke = np.concatenate([null_k(e) for e in energy])
    k = np.maximum(kv, ke)
    out = {name: dict(n_pairs=int(len(a)), mean=float(a.mean()), sd=float(a.std(ddof=1)),
                      min=float(a.min()), max=float(a.max()))
           for name, a in (("K_vertex", kv), ("K_energy", ke), ("K", k))}
    out["kappa_m"] = kappa_from_sd(out["K"]["sd"])
    out["kappa_m_if_vertex_only"] = kappa_from_sd(out["K_vertex"]["sd"])
    return out


def predict_checkpoints(x, checkpoints, batch=64):
    import torch
    from .multienergy_baseline import build_subject
    preds = []
    for path in checkpoints:
        model = build_subject()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            flat = torch.from_numpy(x.reshape(-1, *x.shape[-2:]))
            p = torch.cat([model(flat[i:i+batch]) for i in range(0, len(flat), batch)]).numpy()
        preds.append(p.reshape(*x.shape[:-2], p.shape[-1]))
    return np.stack(preds)


def run(data, checkpoints, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    a = np.load(data)
    x = a["x"]                                      # (N, R, n_sensors, 6)
    if x.ndim != 4:
        raise ValueError("precision data must hold R readout re-draws per placement")
    x = np.ascontiguousarray(np.swapaxes(x, 0, 1))  # (R, N, n_sensors, 6)
    preds = predict_checkpoints(x, checkpoints)     # (seeds, R, N, 7)
    vertex, energy = placement_errors(preds, a["y"][None, None])
    summary = summarise(vertex, energy)
    summary.update(n_placements=int(x.shape[1]), n_physical_events=int(len(set(a["physical_event_id"].tolist()))),
                   n_realisations=int(x.shape[0]), roles=sorted(set(a["role"].tolist())),
                   checkpoints={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in checkpoints},
                   dataset_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
                   energy_endpoint="D-E deposited energy (pending_D-E; diagnostic)", rule="kappa_m = max(1.10, 1 + 2 SD_K), author 18 Oct 2026",
                   per_seed_mean_vertex_error_m=vertex.mean(axis=(1, 2)).tolist(), evidence_stage="development", non_citable=True)
    (out / "precision.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "errors.npz", vertex=vertex, energy=energy)
    print(json.dumps({k: summary[k] for k in ("K_vertex", "K_energy", "K", "kappa_m", "n_placements", "n_realisations")}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    run(**vars(p.parse_args()))
