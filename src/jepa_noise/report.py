"""Deterministic status-table renderer; empirical aggregation is not implemented."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def render(rows: list[dict]) -> str:
    lines = ['# JEPA v3 prediction status', '',
             'Development output; no prospective grid evidence yet. Analytic fixtures and the disclosed toy are reported separately.', '',
             '| Prediction | Campaign | Status | Reason | Evaluation seeds | Numeric evidence |',
             '|---|---|---|---|---|---|']
    seen = set()
    for row in rows:
        if row['id'] in seen or row['status'] not in {'pass', 'fail', 'not_evaluable', 'not_run'}:
            raise ValueError('invalid_prediction_record')
        seen.add(row['id'])
        if row['status'] in {'pass', 'fail'} and not row.get('evidence'):
            raise ValueError('empirical_outcome_requires_evidence')
        cells = [row['id'], row['campaign'], row['status'], row['reason'],
                 ', '.join(map(str, row['seeds'])) or 'none', row.get('evidence') or 'none']
        if any('|' in str(x) or '\n' in str(x) for x in cells):
            raise ValueError('invalid_table_cell')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', choices=['all', 'E1', 'E2', 'E3', 'E4'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    directory = args.root / 'docs' / 'jepa'
    rows = json.loads((directory / 'PREDICTION_STATUS.json').read_text())['predictions']
    if args.campaign != 'all':
        rows = [r for r in rows if r['campaign'] == args.campaign]
    output = directory / ('PREDICTIONS.md' if args.campaign == 'all' else f'PREDICTIONS_{args.campaign}.md')
    output.write_text(render(rows), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
