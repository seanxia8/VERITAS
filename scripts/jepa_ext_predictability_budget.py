#!/usr/bin/env python3
"""Prepare the unregistered mixed-domain predictability budget for one cell.

This script reads frozen datasets in place; it never creates a base event or
modifies a dataset. Only the compact per-event moments are written to --out.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from jepa_noise import lucid_bombs as lb
from jepa_noise.lucid_views import module_hilbert_order
from jepa_noise.ext_predictability import event_moments, tokenise_realisations


def check_limits(root, deadline):
    """Fail before the next write if the authorised storage or time budget is exhausted."""
    if time.monotonic() >= deadline:
        raise RuntimeError("S1: CPU wall-time allocation exhausted")
    quota = Path('/ceph/dwong')
    free = (int(os.getxattr(quota, 'ceph.quota.max_files'))
            - int(os.getxattr(quota, 'ceph.dir.rfiles'))
            - int(os.getxattr(quota, 'ceph.dir.rsubdirs')))
    if free < 30000:
        raise RuntimeError("S1: free entries below the write-stop floor")
    entries, size = 0, 0
    for base, dirs, files in os.walk(root):
        entries += len(dirs) + len(files)
        for name in files:
            size += (Path(base) / name).stat().st_size
    if entries >= 2998 or size >= 1_900_000_000:
        raise RuntimeError("S1: extension storage safety margin exhausted")
    return size


def process(args):
    if args.spread != {"nominal": 0.05, "strong_drift": 0.30}[args.cell]:
        raise ValueError("cell label and drift spread disagree")
    dataset = Path(args.dataset).resolve(strict=True)
    out = Path(args.out).resolve(strict=True)
    if not out.is_relative_to(Path('/ceph/dwong/jepa/returns/ext_predictability_20261029')):
        raise ValueError('output must be inside the predictability extension root')
    check_limits(out, args.deadline)
    tmp = out / 'tmp'
    budget = out / 'budget'
    tmp.mkdir(exist_ok=True)
    budget.mkdir(exist_ok=True)
    output = {s: budget / f'{args.cell}_{s}.npz' for s in ('A', 'B')}
    if any(path.exists() for path in output.values()):
        raise FileExistsError('refusing to overwrite existing budget files')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['law'] != 'F1' or float(manifest['setting']) != args.spread:
        raise ValueError('dataset law/spread does not match the requested cell')
    geometry = lb.load_wcte_geometry(args.geometry)
    cfg = lb.load_generator_config(args.generator_config)
    if manifest['geometry_sha256'] != geometry.digest or manifest['generator_config_sha256'] != cfg.digest:
        raise ValueError('dataset geometry or generator fingerprint does not match')
    seed = int(manifest['dataset_seed'])
    rows = manifest['runs'][:20]
    if len(rows) != 20 or any(int(row['n_events']) != 50 for row in rows):
        raise ValueError('expected twenty original runs of fifty events')
    results = {s: {key: [] for key in ('mean', 'variance', 'hit_count', 'hit_sum', 'hit_sum_squares')}
               for s in output}
    event_ids, run_ids, domains, vertices, yields, target_hits = [], [], [], [], [], []
    expected_counts = {'photon': 340, 'muon': 340, 'electron': 320}
    counts = dict.fromkeys(expected_counts, 0)
    readout = lb.make_readout(geometry, cfg)
    module_order = module_hilbert_order(geometry)
    first_half = module_order[:len(module_order) // 2]
    contexts = {
        'within': [np.flatnonzero((geometry.module_of_pmt == m) &
                                  (geometry.pmt_id_in_mpmt % 2 == 0))
                   for m in range(geometry.n_modules)],
        'across_fixed': [np.flatnonzero(geometry.module_of_pmt == m) for m in first_half],
    }
    pooled_means = {name: [] for name in contexts}
    pooled_cov_sum = {name: np.zeros((len(groups) * 3, len(groups) * 3), dtype=np.float64)
                      for name, groups in contexts.items()}
    for row in rows:
        run_id = row['run_id']
        state = lb.state_for_run('F1', args.spread, run_id, geometry, seed=seed)
        if state.digest != row['state_sha256']:
            raise ValueError(f'{run_id}: original state digest mismatch')
        run_dir = dataset / 'runs' / row['run_dir']
        saved = lb.read_sparse_hits(run_dir / 'hits.npz')
        with np.load(run_dir / 'events.npz', allow_pickle=False) as data:
            ids = data['event_ids'].copy()
            domain = data['domain'].astype(str)
            vertex = data['vertex_m'].copy()
            yield_value = data['yield_photons'].copy()
        if len(ids) != 50 or not np.array_equal(ids, saved.event_ids):
            raise ValueError(f'{run_id}: cached event IDs differ')
        for i, event_id in enumerate(ids):
            check_limits(out, args.deadline)
            d = str(domain[i])
            if d not in counts:
                raise ValueError(f'unknown domain {d}')
            # Reconstruct the exact keyed latent draws for this existing event;
            # equality with saved truth and readout is checked before resampling.
            draws = lb.draw_event(geometry, cfg, run_id, int(event_id), seed=seed, domain=d)
            if not np.array_equal(draws.vertex_m, vertex[i]) or draws.yield_photons != yield_value[i]:
                raise ValueError(f'{run_id}/{event_id}: reconstructed truth mismatch')
            hit = readout(state, draws)
            span = slice(int(saved.offsets[i]), int(saved.offsets[i + 1]))
            if not (np.array_equal(hit.pmt_id, saved.pmt_id[span]) and
                    np.array_equal(np.asarray(hit.charge_pe, np.float32), saved.charge_pe[span]) and
                    np.array_equal(np.asarray(hit.time_ns, np.float32), saved.time_ns[span])):
                raise ValueError(f'{run_id}/{event_id}: reconstructed readout mismatch')
            for scheme in ('A', 'B'):
                features, target = tokenise_realisations(
                    draws, geometry, cfg, state, scheme, tmp, spread=args.spread, seed=seed,
                    scratch_dir=args.scratch)
                if scheme == 'B':
                    for name, groups in contexts.items():
                        pooled = np.stack([features[:, group, 6:9].mean(axis=1)
                                           for group in groups], axis=1).reshape(len(features), -1)
                        mean = pooled.mean(axis=0, dtype=np.float64)
                        centered = pooled.astype(np.float64) - mean
                        pooled_means[name].append(mean.astype(np.float32))
                        pooled_cov_sum[name] += centered.T @ centered / (len(features) - 1)
                for key, value in event_moments(features, target).items():
                    results[scheme][key].append(value)
            target_hits.append(int(np.count_nonzero(geometry.pmt_id_in_mpmt[hit.pmt_id] % 2 == 1)))
            counts[d] += 1
            event_ids.append(int(event_id)); run_ids.append(run_id); domains.append(d)
            vertices.append(vertex[i]); yields.append(yield_value[i])
        print(f'{args.cell}: completed {run_id}', flush=True)
    if event_ids != list(range(1_210_000, 1_211_000)):
        raise ValueError('budget is not the first 1000 dev probe-test IDs')
    if counts != expected_counts:
        raise ValueError(f'unexpected domain counts: {counts}')
    estimated = sum(sum(v.nbytes for v in values) for result in results.values() for values in result.values())
    if check_limits(out, args.deadline) + estimated > 2_000_000_000:
        raise RuntimeError('S1: budget files would exceed the storage cap')
    for scheme, path in output.items():
        check_limits(out, args.deadline)
        arrays = {key: np.stack(values) for key, values in results[scheme].items()}
        if scheme == 'B':
            for name in contexts:
                arrays[name + '_context_mean'] = np.stack(pooled_means[name])
                arrays[name + '_context_covariance_mean'] = (pooled_cov_sum[name] / len(event_ids))
        arrays.update(event_ids=np.asarray(event_ids, dtype=np.int64),
                      run_ids=np.asarray(run_ids), domains=np.asarray(domains),
                      vertices_m=np.asarray(vertices, dtype=np.float64),
                      yields_photons=np.asarray(yields, dtype=np.float64),
                      target_index=target, original_target_hits=np.asarray(target_hits, dtype=np.int16),
                      registered=np.asarray(False))
        with path.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
        print(path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--generator-config', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cell', required=True, choices=('nominal', 'strong_drift'))
    parser.add_argument('--spread', required=True, type=float, choices=(0.05, 0.30))
    parser.add_argument('--remaining-seconds', required=True, type=int)
    args = parser.parse_args()
    if not 0 < args.remaining_seconds <= 43200:
        raise ValueError("remaining CPU wall-time must be within the 12-hour budget")
    args.deadline = time.monotonic() + args.remaining_seconds
    root = Path(args.out).resolve(strict=True)
    if not root.is_relative_to(Path("/ceph/dwong/jepa/returns/ext_predictability_20261029")):
        raise ValueError("output must be inside the predictability extension root")
    check_limits(root, args.deadline)
    tmp = root / "tmp"
    tmp.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="budget_", dir=tmp) as scratch:
        args.scratch = scratch
        process(args)


if __name__ == '__main__':
    main()
