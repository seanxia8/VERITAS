#!/usr/bin/env python3
"""Full-input token probes for the unregistered mixed-domain extension."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from jepa_noise import lucid_arms as la
from jepa_noise import lucid_run as lr
from jepa_noise import point_encoder as pe
from jepa_noise.lucid_views import ViewMasker
from scripts import jepa_pregrid_checks as pregrid


def process(args):
    out = Path(args.out).resolve()
    if not out.is_relative_to(Path('/ceph/dwong/jepa/returns/ext_predictability_20261029')):
        raise ValueError('output must be inside the extension root')
    if out.exists():
        raise FileExistsError('refusing to overwrite an existing probe result')
    config, gen, arms_config, geometry, device = pregrid.setup(args)
    cell = pregrid.variant_cell(config, 'strong_drift') if args.cell == 'strong_drift' else lr.dev_cell(config)
    dataset = Path(args.dataset).resolve(strict=True)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['law'] != 'F1' or float(manifest['setting']) != float(cell.setting):
        raise ValueError('dataset cell mismatch')
    with np.load(args.scheme_a, allow_pickle=False) as data:
        a = {k: data[k].copy() for k in ('event_ids', 'domains', 'target_index', 'mean')}
    with np.load(args.scheme_b, allow_pickle=False) as data:
        b = {k: data[k].copy() for k in ('event_ids', 'domains', 'target_index', 'mean')}
    for key in ('event_ids', 'domains', 'target_index'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'budget alignment mismatch: {key}')
    if len(a['event_ids']) != 1000:
        raise ValueError('expected 1000 original events')
    if args.arm in la.ANCHOR_ARMS:
        seed = int(config['data']['dataset_seed'])
        records = int(config['data']['dark_noise']['records'])
        bank = Path(args.data_root) / 'lucid' / f'dark_noise__s{seed}__n{records}'
        if not (bank / 'manifest.json').is_file():
            raise FileNotFoundError('frozen dark bank is missing; will not generate it')
    model, beta = pregrid.build_loaded(args.arm, cell, geometry, arms_config, config, gen,
                                      Path(args.data_root), args.checkpoint, args.seed)
    model = getattr(model, 'jepa', model).to(device).eval()
    target = a['target_index']
    encoded, observed, hit_flags, domains = [], [], [], []
    cursor = 0
    scaling = pe.FeatureScaling.from_geometry(geometry)
    for row in manifest['runs'][:20]:
        source = dataset / 'runs' / row['run_dir']
        ids = a['event_ids'][cursor:cursor + 50]
        batches, seen, hits, raw = [], [], [], []
        for start in range(0, 50, args.chunk_events):
            selected = ids[start:start + args.chunk_events]
            batch = pe.tokenise(source, geometry, split='within', seed=int(config['views']['mask_seed']),
                                scaling=scaling, event_ids=selected,
                                masker=ViewMasker(geometry, 'within', seed=int(config['views']['mask_seed']), n_min=0))
            if not np.array_equal(batch.event_ids, selected) or not np.array_equal(batch.target_index[0].numpy(), target):
                raise ValueError(f'{row["run_id"]}: original event or target excluded/misaligned')
            with torch.no_grad():
                z = model.encode(lr._to_device(batch, device).features).cpu().numpy()
            batches.append(z[:, target]); seen.append(batch.event_ids)
            hits.append(batch.hit_flags[:, target])
            raw.append(batch.features.cpu().numpy()[:, target, 6:9])
        encoded.append(np.concatenate(batches)); hit_flags.append(np.concatenate(hits))
        observed.append(np.concatenate(raw)); domains.append(a['domains'][cursor:cursor + 50])
        cursor += 50
        print(f'{args.label}: encoded {row["run_id"]}', flush=True)
    z = np.concatenate(encoded); obs = np.concatenate(observed)
    hit = np.concatenate(hit_flags); domain_labels = np.concatenate(domains)
    split = np.arange(1000) < 500
    targets = {'observed': obs, 'predictable': a['mean'], 'physics': b['mean']}
    results = {}
    for domain in ('all', 'photon', 'muon', 'electron'):
        d = np.ones(1000, bool) if domain == 'all' else domain_labels == domain
        results[domain] = {}
        for j, feature in enumerate(('log_charge', 'time', 'hit')):
            results[domain][feature] = {}
            for name, truth in targets.items():
                select = np.ones(hit.shape, bool) if feature == 'hit' else hit
                train = select & (d & split)[:, None]
                test = select & (d & ~split)[:, None]
                results[domain][feature][name] = pregrid.ridge_r2(z[train], truth[:, :, j][train],
                                                                 z[test], truth[:, :, j][test])
    digest = None if args.checkpoint == 'init' else hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    result = {'registered': False, 'extension': 'predictability_2026-10-29',
              'label': args.label, 'cell': args.cell, 'checkpoint': args.checkpoint,
              'checkpoint_sha256': digest, 'arm': args.arm, 'beta': beta,
              'seed': args.seed, 'results': results}
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    p = argparse.ArgumentParser()
    for opt in ('dataset', 'scheme-a', 'scheme-b', 'out', 'config', 'generator-config',
                'geometry', 'data-root', 'label', 'arm', 'checkpoint'):
        p.add_argument('--' + opt, required=True)
    p.add_argument('--arms-config', default=None)
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--cell', default='nominal', choices=('nominal', 'strong_drift'))
    p.add_argument('--seed', type=int, default=9000)
    p.add_argument('--chunk-events', type=int, default=8)
    process(p.parse_args())


if __name__ == '__main__':
    main()
