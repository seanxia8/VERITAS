#!/usr/bin/env python3
"""CPU-only timing preflight; emits engineering estimates, never science results."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jepa_noise import linear
from jepa_noise import lucid_arms as la
from jepa_noise import lucid_run as lr
from jepa_noise import point_encoder as pe
from jepa_noise.lucid_views import ViewMasker, module_hilbert_order
from scripts import jepa_ext_predictability_linear as lin
from scripts import jepa_pregrid_checks as pregrid


def first_batch(dataset, geometry, mask_seed, n):
    root = Path(dataset).resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    row = manifest['runs'][0]
    source = root / 'runs' / row['run_dir']
    with np.load(source / 'events.npz', allow_pickle=False) as data:
        ids = data['event_ids'][:n].copy()
    batch = pe.tokenise(source, geometry, split='within', seed=mask_seed, event_ids=ids,
                        scaling=pe.FeatureScaling.from_geometry(geometry),
                        masker=ViewMasker(geometry, 'within', seed=mask_seed, n_min=0))
    if not np.array_equal(batch.event_ids, ids):
        raise ValueError('benchmark original IDs missing')
    return batch


def main():
    p = argparse.ArgumentParser()
    for name in ('config', 'generator-config', 'geometry', 'data-root', 'dataset', 'strong-dataset', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--arms-config', default=None)
    p.add_argument('--device', default='cpu', choices=('cpu',))
    p.add_argument('--item', action='append', required=True,
                   help='label,arm,checkpoint-or-init,seed,nominal-or-strong_drift')
    args = p.parse_args()
    out = Path(args.out).resolve()
    if not out.is_relative_to(Path('/ceph/dwong/jepa/returns/ext_predictability_20261029')) or out.exists():
        raise ValueError('output must be a fresh path inside the extension root')
    expected = {'init9000', 'fix9000', 'fix9001', 'anc100', 'mae9000', 'ijepa9000', 'ijepa9001',
                'stopgrad9000', 'stopgrad9001', 'lambda9000', 'lambda9001',
                'strong_drift9000', 'strong_drift9001'}
    labels = [spec.split(',', 1)[0] for spec in args.item]
    if len(labels) != len(expected) or set(labels) != expected:
        raise ValueError('the 13 distinct requested encoder labels are required')
    config, gen, arms_config, geometry, device = pregrid.setup(args)
    mask_seed = int(config['views']['mask_seed'])
    seed = int(config['data']['dataset_seed'])
    records = int(config['data']['dark_noise']['records'])
    bank = Path(args.data_root) / 'lucid' / f'dark_noise__s{seed}__n{records}'
    nominal_batch = first_batch(args.dataset, geometry, mask_seed, 8)
    strong_batch = first_batch(args.strong_dataset, geometry, mask_seed, 8)
    timings = []
    for spec in args.item:
        label, arm_name, source, seed_text, cell_name = spec.split(',', 4)
        if cell_name not in ('nominal', 'strong_drift'):
            raise ValueError('unknown cell')
        cell = pregrid.variant_cell(config, 'strong_drift') if cell_name == 'strong_drift' else lr.dev_cell(config)
        if arm_name in la.ANCHOR_ARMS and not (bank / 'manifest.json').is_file():
            raise FileNotFoundError('frozen dark bank missing; no generation allowed')
        arm, _ = pregrid.build_loaded(arm_name, cell, geometry, arms_config, config, gen,
                                      Path(args.data_root), source, int(seed_text))
        model = getattr(arm, 'jepa', arm).to(device).eval()
        batch = strong_batch if cell_name == 'strong_drift' else nominal_batch
        with torch.no_grad():
            model.encode(batch.features[:1])
            start = time.monotonic()
            model.encode(batch.features)
            elapsed = time.monotonic() - start
        timings.append({'label': label, 'events': len(batch.event_ids), 'elapsed_seconds': elapsed,
                        'projected_1000_seconds': 1000 * elapsed / len(batch.event_ids)})
        print(json.dumps(timings[-1]), flush=True)
        del model, arm
    if len(timings) != 13:
        raise ValueError('exactly 13 encoder checkpoints/initializations are required')
    rng = np.random.default_rng(4321)
    # Pregrid ridge has four alphas; benchmark a 10k/10k synthetic fit without storing rows.
    xtr = rng.standard_normal((10000, 128)); xte = rng.standard_normal((10000, 128))
    ytr = rng.standard_normal(10000); yte = rng.standard_normal(10000)
    start = time.monotonic(); pregrid.ridge_r2(xtr, ytr, xte, yte)
    ridge_seconds = time.monotonic() - start
    del xtr, xte, ytr, yte
    # For each of 13 encoders and 9 feature-target pairs, pooled + domain rows
    # total 2 x 1000 x 1050 tokens. Treat all tokens as hit for a safe workload estimate.
    ridge_projection = ridge_seconds * (13 * 9 * 2 * 1000 * 1050) / 20000
    module = geometry.module_of_pmt; parity = geometry.pmt_id_in_mpmt % 2
    order = module_hilbert_order(geometry); half = len(order) // 2
    groups = [([np.flatnonzero((module == m) & (parity == 0)) for m in range(geometry.n_modules)],
               [np.flatnonzero((module == m) & (parity == 1)) for m in range(geometry.n_modules)]),
              ([np.flatnonzero(module == m) for m in order[:half]],
               [np.flatnonzero(module == m) for m in order[half:]])]
    start = time.monotonic()
    f = nominal_batch.features.numpy()
    for context, target in groups:
        lin.pool(f, context); lin.pool(f, target)
    pool_seconds = time.monotonic() - start
    # 40k total events in both cells; timings cover token extraction and module pooling.
    extraction_start = time.monotonic(); first_batch(args.dataset, geometry, mask_seed, 8)
    tokenise_seconds = time.monotonic() - extraction_start
    extraction_projection = (pool_seconds + tokenise_seconds) * 40000 / 8
    # CCA/GEVP and run-grouped ridge costs at realistic dimensions, synthetic values.
    z = rng.standard_normal((1200, 315)); z2 = rng.standard_normal((1200, 315))
    cc, tt, ct = lin.regularise(np.cov(z, rowvar=False)), lin.regularise(np.cov(z2, rowvar=False)), (z-z.mean(0)).T @ (z2-z2.mean(0)) / (len(z)-1)
    start = time.monotonic()
    for k in (1, 4, 10, 30): linear.cca(cc, tt, ct, k)
    linear.gevp(cc, tt, 4)
    cca_seconds = time.monotonic() - start
    gx = rng.standard_normal((400, 105)); gy = rng.standard_normal(400); gg = np.arange(400)
    px = rng.standard_normal((6800, 14)); py = rng.standard_normal(6800); pg = np.repeat(np.arange(400), 17)
    start = time.monotonic(); lin.grouped_scores(gx, gy, gg)
    for _ in range(3): lin.grouped_scores(px, py, pg)
    grouped_seconds = time.monotonic() - start
    # Four cell/split combinations, 30 directions. Full-dimension CCA time is used for all.
    cca_projection = 4 * (cca_seconds + 30 * grouped_seconds) + extraction_projection
    report = {'registered': False, 'kind': 'engineering_cost_preflight', 'encoders': timings,
              'forward_projection_seconds': sum(x['projected_1000_seconds'] for x in timings),
              'ridge_sample_seconds': ridge_seconds, 'ridge_projection_seconds': ridge_projection,
              'pool_8_seconds': pool_seconds, 'tokenise_8_seconds': tokenise_seconds,
              'cca_solver_seconds': cca_seconds, 'grouped_one_direction_seconds': grouped_seconds,
              'linear_projection_seconds': cca_projection,
              'note': 'Workload estimates only, not upper bounds; no scientific result or raw realisations saved.'}
    with out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
