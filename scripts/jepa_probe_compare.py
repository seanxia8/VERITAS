#!/usr/bin/env python3
"""Compare two `jepa_probe_env.py` outputs — the 10-batch equivalence test of runbook v4 §6.1.

The host probe runs on the GPU in the existing read-only environment; the local probe runs on
the Mac CPU in the accepted local environment (`/private/tmp/jepa-venv-cpu311`). Both use the same
code SHA, seed, model config and fixed input fixture, so the per-batch reductions must agree to
float32 tolerance (relative differences of order 1e-5, larger only for near-zero sums).

Usage:
    python scripts/jepa_probe_compare.py --host <probe.json> --local <probe.json> [--json OUT]
Exit 0 if every compared quantity agrees within the tolerance, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json

TOL_REL = 1e-4     # float32 GPU vs CPU accumulation differences; 1e-4 is 10x the expected scale
TOL_ABS = 1e-3     # absolute floor for near-zero sums


def compare(a: dict, b: dict) -> tuple[bool, list[dict]]:
    rows: list[dict] = []
    fa = {r['batch']: r for r in a.get('fixture', {}).get('per_batch', [])}
    fb = {r['batch']: r for r in b.get('fixture', {}).get('per_batch', [])}
    ok = True
    for batch in sorted(set(fa) & set(fb)):
        for key in ('x_sum', 'x_abs_sum', 'spatial_sum', 'spatial_abs_sum', 'energy_sum'):
            va, vb = fa[batch][key], fb[batch][key]
            diff = abs(va - vb)
            scale = max(abs(va), abs(vb), 1.0)
            rel = diff / scale
            good = diff <= TOL_ABS or rel <= TOL_REL
            ok = ok and good
            rows.append({'batch': batch, 'quantity': key, 'host': va, 'local': vb,
                         'abs_diff': diff, 'rel_diff': rel, 'ok': good})
        for key in ('spatial_first4', 'energy_first2'):
            for i, (va, vb) in enumerate(zip(fa[batch][key], fb[batch][key])):
                diff = abs(va - vb)
                scale = max(abs(va), abs(vb), 1.0)
                rel = diff / scale
                good = diff <= TOL_ABS or rel <= TOL_REL
                ok = ok and good
                rows.append({'batch': batch, 'quantity': f'{key}[{i}]', 'host': va,
                             'local': vb, 'abs_diff': diff, 'rel_diff': rel, 'ok': good})
    config_keys = ('batches', 'batch_size', 'channels', 'samples', 'seed')
    same_fixture = all(a.get('fixture', {}).get(k) == b.get('fixture', {}).get(k)
                       for k in config_keys)
    same_model = a.get('model') == b.get('model')
    same_head = a.get('repo_head') == b.get('repo_head')
    ok = ok and same_fixture and same_model and same_head
    return ok, rows, {'same_fixture_config': same_fixture, 'same_model_config': same_model,
                      'same_repo_head': same_head,
                      'n_batches_compared': len(set(fa) & set(fb))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', required=True)
    ap.add_argument('--local', required=True)
    ap.add_argument('--json')
    args = ap.parse_args()
    a = json.load(open(args.host))
    b = json.load(open(args.local))
    ok, rows, summary = compare(a, b)
    out = {'host_probe': args.host, 'local_probe': args.local,
           'host_run_id': a.get('run_id'), 'local_run_id': b.get('run_id'),
           'tolerance_rel': TOL_REL, 'tolerance_abs': TOL_ABS,
           'pass': ok, **summary,
           'worst_rel_diff': max((r['rel_diff'] for r in rows), default=None),
           'rows': rows}
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1, sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
