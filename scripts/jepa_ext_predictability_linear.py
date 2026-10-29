#!/usr/bin/env python3
"""Mixed-domain module-pooled CCA, an unregistered LUCiD extension."""
import argparse
import json
from pathlib import Path

import numpy as np

from jepa_noise import linear, metrics
from jepa_noise import lucid_bombs as lb
from jepa_noise import point_encoder as pe
from jepa_noise.lucid_views import ViewMasker, module_hilbert_order
from scripts.jepa_pregrid_checks import ridge_r2


def pool(features, groups):
    return np.stack([features[:, ids, 6:9].mean(axis=1) for ids in groups], axis=1).reshape(len(features), -1)


def regularise(cov):
    return cov + 1e-3 * np.trace(cov) / len(cov) * np.eye(len(cov))


def grouped_scores(x, y, group):
    folds = []
    for fold in range(5):
        test = group % 5 == fold
        train = ~test
        if train.sum() < 5 or test.sum() < 2:
            return None
        result = ridge_r2(x[train], y[train], x[test], y[test])
        if result['r2'] is None:
            return None
        folds.append(result['r2'])
    return float(np.mean(folds))


def quadratic(vertex, yields):
    v = np.column_stack((vertex, np.log(yields)))
    i, j = np.triu_indices(4)
    return np.column_stack((v, v[:, i] * v[:, j]))


def extract(paths, geometry, cfg, spread, mask_seed):
    order = module_hilbert_order(geometry)
    middle = len(order) // 2
    module = geometry.module_of_pmt
    parity = geometry.pmt_id_in_mpmt % 2
    groups = {
        'within': ([np.flatnonzero((module == m) & (parity == 0)) for m in range(geometry.n_modules)],
                   [np.flatnonzero((module == m) & (parity == 1)) for m in range(geometry.n_modules)]),
        'across_fixed': ([np.flatnonzero(module == m) for m in order[:middle]],
                         [np.flatnonzero(module == m) for m in order[middle:]]),
    }
    views = {name: [[], []] for name in groups}
    vertices, yields, domains, grouping, gains = [], [], [], [], []
    below, run_index = 0, 0
    scaling = pe.FeatureScaling.from_geometry(geometry)
    for dataset in paths:
        root = Path(dataset).resolve(strict=True)
        m = json.loads((root / 'manifest.json').read_text())
        if m['law'] != 'F1' or float(m['setting']) != spread or m['geometry_sha256'] != geometry.digest or m['generator_config_sha256'] != cfg.digest:
            raise ValueError(f'{root}: dataset identity mismatch')
        seed = int(m['dataset_seed'])
        for row in m['runs']:
            source = root / 'runs' / row['run_dir']
            with np.load(source / 'events.npz', allow_pickle=False) as data:
                ids = data['event_ids'].copy()
                vertex, y, domain = data['vertex_m'].copy(), data['yield_photons'].copy(), data['domain'].astype(str)
            state = lb.state_for_run('F1', spread, row['run_id'], geometry, seed=seed)
            if state.digest != row['state_sha256']:
                raise ValueError(f'{row["run_id"]}: state digest mismatch')
            masker = ViewMasker(geometry, 'within', seed=mask_seed, n_min=0)
            batch = pe.tokenise(source, geometry, split='within', seed=mask_seed,
                                event_ids=ids, scaling=scaling, masker=masker)
            if not np.array_equal(batch.event_ids, ids):
                raise ValueError('original event IDs missing or reordered')
            features = batch.features.numpy()
            target_idx = batch.target_index[0].numpy()
            below += int((batch.hit_flags[:, target_idx].sum(axis=1) < lb.N_MIN).sum())
            for name, (context, target) in groups.items():
                views[name][0].append(pool(features, context)); views[name][1].append(pool(features, target))
            vertices.append(vertex); yields.append(y); domains.append(domain)
            grouping.append(np.full(len(ids), run_index, dtype=np.int64))
            gains.append(np.log(state.module_gain)); run_index += 1
    if run_index != 400 or sum(len(x) for x in yields) != 20000:
        raise ValueError('expected 20,000 original events in 400 runs')
    return ({name: (np.concatenate(pair[0]), np.concatenate(pair[1])) for name, pair in views.items()},
            np.concatenate(vertices), np.concatenate(yields), np.concatenate(domains),
            np.concatenate(grouping), np.stack(gains), below)


def process(args):
    out = Path(args.out).resolve()
    if not out.is_relative_to(Path('/ceph/dwong/jepa/returns/ext_predictability_20261029')):
        raise ValueError('output must be inside extension root')
    if out.exists():
        raise FileExistsError(out)
    geometry = lb.load_wcte_geometry(args.geometry)
    cfg = lb.load_generator_config(args.generator_config)
    data = extract((args.probe_train, args.probe_test), geometry, cfg, args.spread, args.mask_seed)
    views, vertex, yields, domain, group, gain, below = data
    phys = quadratic(vertex, yields)
    with np.load(args.scheme_b, allow_pickle=False) as budget:
        if len(budget['event_ids']) != 1000:
            raise ValueError('scheme B must contain 1000 original events')
        scheme_means = {name: budget[name + '_context_mean'].copy() for name in views}
        scheme_noise = {name: budget[name + '_context_covariance_mean'].copy() for name in views}
    result = {'registered': False, 'extension': 'predictability_2026-10-29',
              'spread': args.spread, 'n_events': 20000, 'n_runs': 400,
              'n_below_registered_n_min': below, 'splits': {}}
    for name, (x, y) in views.items():
        x, y = x.astype(np.float64), y.astype(np.float64)
        xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
        cc, tt, ct = regularise(xc.T @ xc / (len(x) - 1)), regularise(yc.T @ yc / (len(y) - 1)), xc.T @ yc / (len(x) - 1)
        fits = {k: linear.cca(cc, tt, ct, k) for k in (1, 4, 10, 30)}
        mean = scheme_means[name].astype(np.float64)
        signal = regularise(np.cov(mean, rowvar=False))
        noise = regularise(scheme_noise[name])
        reference = linear.gevp(signal, noise, 4)
        displacement = float(np.max(metrics.principal_angles(fits[4].context, reference.context, degrees=True)))
        projection = x @ fits[30].context.T
        run_mean = np.stack([projection[group == i].mean(axis=0) for i in range(400)])
        directions = []
        for j in range(30):
            gain_r2 = grouped_scores(gain, run_mean[:, j], np.arange(400))
            phys_r2 = {}
            for d in ('photon', 'muon', 'electron'):
                sel = domain == d
                phys_r2[d] = grouped_scores(phys[sel], projection[sel, j], group[sel])
            directions.append({'direction': j + 1, 'r2_gain': gain_r2, 'r2_phys': phys_r2})
        result['splits'][name] = {'context_dimensions': x.shape[1], 'target_dimensions': y.shape[1],
                                   'cca_correlations': {str(k): fits[k].spectrum[:k].tolist() for k in fits},
                                   'displacement_degrees': displacement, 'directions': directions}
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    p = argparse.ArgumentParser()
    for opt in ('probe-train', 'probe-test', 'scheme-b', 'generator-config', 'geometry', 'out'):
        p.add_argument('--' + opt, required=True)
    p.add_argument('--spread', type=float, required=True, choices=(.05, .30))
    p.add_argument('--mask-seed', type=int, required=True)
    process(p.parse_args())


if __name__ == '__main__':
    main()
