#!/usr/bin/env python3
"""r2-mask M1 (runbook 2026-11-23 §1.2, §M1): shortcut budget per mask configuration, no network.

Events come from ``dev_probe_train`` of the LUCiD pack; runs are split alternately into fitting
(even run index) and scoring (odd) runs, so fitting and scoring events are disjoint. The hidden
sets are drawn with the training masker (``lucid_views.ViewMasker``, the training mask seed,
``n_min = 0`` so that every configuration sees the same events).

Usage (repository root)::

    mask_shortcut_budget.py --detector sk --out /work/dwong/r2_mask/m1/sk.json
    mask_shortcut_budget.py --detector juno --configs parity random patch_default --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402
from jepa_noise import shortcut_budget as sb  # noqa: E402
from jepa_noise.lucid_views import ViewMasker, default_patch_size  # noqa: E402

#: §1.1 sizes; the default size of each detector is reused as ``patch_default``.
SIZES = {"wcte": (8, 32, 128, 256), "sk": (43, 173, 692)}
#: photon bombs: all 3,400 of dev_probe_train (1,700 + 1,700); §M1.1 asks 2,000 + 2,000, which the
#: cohort cannot supply (DECISIONS 2026-11-23). Muons and electrons: 1,000 + 1,000 as registered.
DOMAIN_COUNTS = {"photon": (1700, 1700), "muon": (1000, 1000), "electron": (1000, 1000)}


def configs_for(detector, names=None):
    n_pmt = dr.geometry(detector, dr.LUCID_V1).n_pmt
    default = default_patch_size(n_pmt)
    if names is None:
        names = ["parity", "random"] + [f"patch{s}" for s in SIZES[detector]]
    out = []
    for name in names:
        if name == "parity":
            out.append((name, "within", None))
        elif name == "random":
            out.append((name, "random", None))
        elif name == "patch_default":
            out.append((f"patch{default}", "spatial_patch", None))
        elif name.startswith("patch"):
            s = int(name[5:])
            out.append((f"patch{s}", "spatial_patch", None if s == default else s))
        else:
            raise ValueError(name)
    return out


def mask_seed(detector):
    if detector == "wcte":
        return int(json.loads((Path(dr.wcte_train_shards(dr.LUCID_V1)) / "manifest.json")
                              .read_text())["dataset_seed"])
    return int(lb.DEFAULT_SEED)


def load_events(detector, counts):
    """{domain: {"fit": (Targets, event_ids), "score": (...)}} from dev_probe_train."""
    path = dr.wcte_probe_pack(dr.LUCID_V1) if detector == "wcte" else dr.pack(detector, dr.LUCID_V1)
    cohort = rs.open_cohort(path, "dev_probe_train", truth_fields=("domain",))
    n_pmt = dr.geometry(detector, dr.LUCID_V1).n_pmt
    want = {(d, part): counts[d][j] for d in counts for j, part in enumerate(("fit", "score"))}
    got = {key: ([], []) for key in want}
    seen = {d: 0 for d in counts}  # runs containing domain d so far: alternate fit / score
    for i in range(len(cohort.run_ids)):
        truth = cohort.run_truth(i)
        dom = truth["domain"].astype(str)
        store = None
        for d in counts:
            if not (dom == d).any():
                continue
            part = "fit" if seen[d] % 2 == 0 else "score"
            seen[d] += 1
            have = sum(len(x) for x in got[(d, part)][1])
            need = want[(d, part)] - have
            if need <= 0:
                continue
            idx = np.flatnonzero(dom == d)[:need]
            if store is None:
                store = cohort.run_store(i)
                if not np.array_equal(np.asarray(store.event_ids), np.asarray(truth["event_ids"])):
                    raise RuntimeError(f"store/truth event order differs in run {cohort.run_ids[i]}")
            if idx.size:
                got[(d, part)][0].append(sb.targets_from_store(store, idx, n_pmt))
                got[(d, part)][1].append(np.asarray(store.event_ids)[idx])
    out = {}
    for (d, part), (ts, ids) in got.items():
        t = sb.Targets(np.concatenate([x.hit for x in ts]), np.concatenate([x.charge for x in ts]),
                       np.concatenate([x.time for x in ts]))
        out.setdefault(d, {})[part] = (t, np.concatenate(ids).astype(np.int64))
        if t.n_events < want[(d, part)]:
            raise RuntimeError(f"{detector} {d} {part}: {t.n_events} < {want[(d, part)]} events")
    return out, str(path)


def hidden_sets(geometry, split, patch_size, seed, event_ids):
    extra = {} if patch_size is None else {"patch_size": patch_size}
    m = ViewMasker(geometry, split, seed=seed, n_min=0, **extra)
    hidden = np.zeros((len(event_ids), geometry.n_pmt), bool)
    zero = np.zeros(geometry.n_pmt, bool)
    for row, e in enumerate(event_ids):
        hidden[row, m.mask(int(e), zero).hidden] = True
    return hidden


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=("wcte", "sk", "juno", "hk"))
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--domains", nargs="+", default=list(DOMAIN_COUNTS))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    geometry = dr.geometry(a.detector, dr.LUCID_V1)
    positions = np.asarray(geometry.positions_m, np.float64)
    seed = mask_seed(a.detector)
    counts = {d: DOMAIN_COUNTS[d] for d in a.domains}
    events, pack = load_events(a.detector, counts)
    spacing = sb.median_nn_spacing(positions)
    result = {"detector": a.detector, "pack": pack, "cohort": "dev_probe_train",
              "n_pmt": int(geometry.n_pmt), "geometry_sha256": geometry.digest, "mask_seed": seed,
              "k": a.k, "nn_spacing_m": spacing, "counts": counts, "domains": {},
              "default_patch_size": default_patch_size(geometry.n_pmt),
              "code_sha": lb.current_code_sha(REPO)}
    for d in a.domains:
        (fit, fit_ids), (score, score_ids) = events[d]["fit"], events[d]["score"]
        lc = sb.correlation_length(positions, fit)
        result["domains"][d] = {"correlation": lc, "configs": {}}
        print(f"{a.detector} {d}: l_c = {lc['l_c_m']:.3f} m ({lc['l_c_spacing']:.2f} spacings)",
              flush=True)
        for name, split, ps in configs_for(a.detector, a.configs):
            hf = hidden_sets(geometry, split, ps, seed, fit_ids)
            hs = hidden_sets(geometry, split, ps, seed, score_ids)
            r = sb.shortcut_budget(positions, fit, score, hf, hs, k=a.k, spacing=spacing)
            r.update(split=split, patch_size=ps, n_fit=int(fit.n_events), n_score=int(score.n_events))
            result["domains"][d]["configs"][name] = r
            print(f"  {name:10s} E_S {r['E_S']:.4f} local {r['local']:.4f} U {r['U']:.4f} "
                  f"AvgDist {r['avg_dist_m']:.3f} m ({r['avg_dist_spacing']:.2f}) "
                  f"hidden {r['hidden_fraction']:.3f} never {r['never_hidden_fraction']:.3f} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=1, allow_nan=True))
    result["wall_s"] = time.time() - t0
    a.out.write_text(json.dumps(result, indent=1, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
