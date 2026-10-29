#!/usr/bin/env python3
"""Summarise the compact, unregistered predictability budget moments."""
import argparse
import json
from pathlib import Path

import numpy as np

from jepa_noise.ext_predictability import hit_moments_predictability, moments_predictability

FEATURES = ('log_charge', 'time', 'hit')
DOMAINS = ('all', 'photon', 'muon', 'electron')


def summarise(a, b):
    for key in ('event_ids', 'run_ids', 'domains', 'yields_photons', 'target_index', 'original_target_hits'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'scheme alignment mismatch for {key}')
    yields = np.asarray(a['yields_photons'], float)
    cuts = np.quantile(yields, [1 / 3, 2 / 3])
    bins = np.searchsorted(cuts, yields, side='right')
    result = {'registered': False, 'extension': 'predictability_2026-10-29',
              'yield_tercile_cutpoints': cuts.tolist(), 'rows': {}}
    for domain in DOMAINS:
        domain_mask = np.ones(len(yields), bool) if domain == 'all' else a['domains'] == domain
        result['rows'][domain] = {}
        for tercile in ('all', 0, 1, 2):
            selected = domain_mask if tercile == 'all' else domain_mask & (bins == tercile)
            label = str(tercile)
            row = {'n_events': int(selected.sum()),
                   'n_below_registered_n_min': int((a['original_target_hits'][selected] < 20).sum()),
                   'features': {}}
            for j, feature in enumerate(FEATURES):
                item = {}
                for hit_only in (False, True):
                    values = []
                    details = []
                    for data in (a, b):
                        if not selected.any():
                            rho = None
                            info = {'pairs': 0, 'observations': 0} if hit_only else None
                        elif hit_only:
                            count = data['hit_count'][selected]
                            info = hit_moments_predictability(count,
                                data['hit_sum'][selected, :, j],
                                data['hit_sum_squares'][selected, :, j])
                            rho = info['rho']
                        else:
                            rho = moments_predictability(data['mean'][selected, :, j],
                                                         data['variance'][selected, :, j])
                            info = None
                        values.append(rho); details.append(info)
                    state = None if None in values else values[0] - values[1]
                    item['hit_only' if hit_only else 'all_tokens'] = {
                        'rho_total': values[0], 'rho_phys': values[1], 'rho_state': state,
                        'scheme_a_counts': details[0], 'scheme_b_counts': details[1]}
                row['features'][feature] = item
            result['rows'][domain][label] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme-a', required=True)
    parser.add_argument('--scheme-b', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    path = Path(args.out).resolve()
    if not path.is_relative_to(Path('/ceph/dwong/jepa/returns/ext_predictability_20261029')):
        raise ValueError('output must be inside the extension root')
    with np.load(args.scheme_a, allow_pickle=False) as a, np.load(args.scheme_b, allow_pickle=False) as b:
        result = summarise(a, b)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
