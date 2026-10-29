#!/usr/bin/env python3
"""Build descriptive tables from completed predictability extension outputs."""
import argparse
import json
from pathlib import Path

FEATURES = ('log_charge', 'time', 'hit')
TARGETS = ('observed', 'predictable', 'physics')


def read(path):
    return json.loads(Path(path).read_text())


def fmt(value):
    return 'undefined' if value is None else f'{value:.4f}'


def status(value):
    return 'unclear' if value is None else ('supported' if value else 'not supported')


def h_a_reading(hit_rho):
    """Evaluate a conjunction even if the ordering term is undefined."""
    charge = hit_rho[0]
    if charge is not None and charge > .5:
        return False
    if any(value is None for value in hit_rho):
        return None
    return charge <= .5 and charge < min(hit_rho[1:])


def run(args):
    root = Path(args.out).resolve(strict=True)
    if not root.is_relative_to(Path('/ceph/dwong/jepa/returns/ext_predictability_20261029')):
        raise ValueError('output must be inside the extension root')
    b = {'nominal': read(args.nominal_budget), 'strong_drift': read(args.strong_budget)}
    l = {'nominal': read(args.nominal_linear), 'strong_drift': read(args.strong_linear)}
    probes = {}
    for spec in args.probe:
        label, path = spec.split('=', 1)
        record = read(path)
        if label in probes or record['label'] != label or not record['registered'] is False:
            raise ValueError(f'invalid probe label {label}')
        probes[label] = record
    expected = {'init9000', 'fix9000', 'fix9001', 'anc100', 'mae9000', 'ijepa9000', 'ijepa9001',
                'stopgrad9000', 'stopgrad9001', 'lambda9000', 'lambda9001',
                'strong_drift9000', 'strong_drift9001'}
    if set(probes) != expected:
        raise ValueError(f'missing/extra probe records: {sorted(expected ^ set(probes))}')
    for label, record in probes.items():
        expected_cell = 'strong_drift' if label.startswith('strong_drift') else 'nominal'
        if record['cell'] != expected_cell:
            raise ValueError(f'{label} has wrong cell')
    md = ['# Predictability budget and encoder probes', '', 'registered: false', '']
    for cell, budget in b.items():
        md += [f'## {cell} — predictability', '',
               '| Domain | Yield tercile | Feature | All total / phys / state | Hit-only total / phys / state |',
               '|---|---|---|---|---|']
        for domain, by_yield in budget['rows'].items():
            for tercile, row in by_yield.items():
                for feature, item in row['features'].items():
                    render = lambda d: ' / '.join(fmt(d[k]) for k in ('rho_total', 'rho_phys', 'rho_state'))
                    md.append(f'| {domain} | {tercile} | {feature} | {render(item["all_tokens"])} | {render(item["hit_only"])} |')
        md += ['', '## ' + cell + ' — token probes', '',
               '| Encoder | Domain | Feature | Observed | Predictable | Physics |', '|---|---|---|---:|---:|---:|']
        for label, record in probes.items():
            if record['cell'] != cell:
                continue
            for domain, features in record['results'].items():
                for feature, scores in features.items():
                    md.append(f'| {label} | {domain} | {feature} | ' + ' | '.join(fmt(scores[t]['r2']) for t in TARGETS) + ' |')
        md.append('')
    linear_md = ['# Linear CCA on LUCiD', '', 'registered: false', '',
                 'Across-fixed uses the first half of module Hilbert order as context.', '']
    for cell, record in l.items():
        for split, row in record['splits'].items():
            linear_md += [f'## {cell} / {split}', '', f'D = {fmt(row["displacement_degrees"])} degrees.', '',
                          '| Direction | Gain R² | Physics R² photon | Physics R² muon | Physics R² electron |',
                          '|---:|---:|---:|---:|---:|']
            for direction in row['directions'][:10]:
                p = direction['r2_phys']
                linear_md.append(f'| {direction["direction"]} | {fmt(direction["r2_gain"])} | {fmt(p["photon"])} | {fmt(p["muon"])} | {fmt(p["electron"])} |')
            linear_md.append('')
    photon = b['nominal']['rows']['photon']
    fixed = [probes[label]['results']['photon'] for label in ('fix9000', 'fix9001')]
    charge = [r['log_charge'] for r in fixed]
    valid = lambda values: all(v is not None for v in values)
    eb = [r['predictable']['r2'] for r in charge]
    obs = [r['observed']['r2'] for r in charge]
    hb = all(x >= .5 for x in eb) and all(x <= .2 for x in obs) if valid(eb + obs) else None
    hbp = all(x < .2 for x in eb) if valid(eb) else None
    rho = [photon['all']['features'][f]['all_tokens']['rho_total'] for f in FEATURES]
    ranks = [[r[f]['predictable']['r2'] for f in FEATURES] for r in fixed]
    hc = None
    if valid(rho + sum(ranks, [])) and len(set(rho)) == 3 and all(len(set(x)) == 3 for x in ranks):
        order = sorted(range(3), key=lambda i: rho[i])
        hc = all(sorted(range(3), key=lambda i: row[i]) == order for row in ranks)
    win = l['nominal']['splits']['within']; across = l['nominal']['splits']['across_fixed']
    wg = [x['r2_gain'] for x in win['directions'][:4]]
    ag = [x['r2_gain'] for x in across['directions'][:4]]
    hd = (any(x >= .5 for x in wg) and all(x < .2 for x in ag) and
          win['displacement_degrees'] - across['displacement_degrees'] >= 20) if valid(wg + ag) else None
    he = l['strong_drift']['splits']['within']['displacement_degrees'] > win['displacement_degrees']
    init = read(args.init_part_c)['items']['init9000']['full_input']['module_gain_run_mean_r2']
    items = read(args.fixed_part_c)['items']
    fg = [items[label]['full_input']['module_gain_run_mean_r2'] for label in ('fix9000', 'fix9001')]
    hf = all(x < init for x in fg) if hd is True else None
    median_hit_rho = [photon['1']['features'][f]['hit_only']['rho_total'] for f in FEATURES]
    readings = {'H-a': h_a_reading(median_hit_rho), 'H-b': hb, 'H-b-prime': hbp, 'H-c': hc, 'H-d': hd, 'H-e': he, 'H-f': hf}
    report = {'registered': False, 'readings': {key: status(value) for key, value in readings.items()},
              'h_a_median_hit_rho': median_hit_rho,
              'h_a_reason': 'hit-only hit-flag variance is zero; the ordering is undefined, but a failed charge threshold is decisive',
              'h_b_predictable': eb, 'h_b_observed': obs, 'h_c_rho': rho, 'h_c_predictable': ranks,
              'h_d_within_gain': wg, 'h_d_across_gain': ag,
              'h_d_displacement_difference': win['displacement_degrees'] - across['displacement_degrees'],
              'h_e_nominal_d': win['displacement_degrees'],
              'h_e_strong_d': l['strong_drift']['splits']['within']['displacement_degrees'],
              'h_f_fixed_part_c': fg, 'h_f_init_part_c': init}
    for filename, content in (('budget_probes.md', '\n'.join(md) + '\n'),
                              ('linear_cca.md', '\n'.join(linear_md) + '\n'),
                              ('readings.json', json.dumps(report, indent=2, allow_nan=False) + '\n')):
        path = root / filename
        with path.open('x') as stream:
            stream.write(content)


def main():
    p = argparse.ArgumentParser()
    for opt in ('nominal-budget', 'strong-budget', 'nominal-linear', 'strong-linear', 'init-part-c', 'fixed-part-c', 'out'):
        p.add_argument('--' + opt, required=True)
    p.add_argument('--probe', action='append', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
