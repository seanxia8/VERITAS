#!/usr/bin/env python
"""Regression comparator for the latent_monitor development smoke.

Runbook `docs/RUNBOOK_REFERENCE_UPGRADES_2026-10-09.md` §2.3: every key present in
the baseline `smoke_report.json` must have an identical value in the new one, with
the only permitted exception the Tier-1 kappa_m status/source strings introduced by
WP3. New keys are allowed.

Usage
-----
    python scripts/compare_smoke.py <baseline_dir_or_json> <new_dir_or_json> \
        [--allow-regex REGEX]... [--tol FLOAT]

Exit status is 0 when there is no *hard* difference (a difference not matched by an
`--allow-regex`), 1 otherwise. A small float tolerance can be set with `--tol`;
differences within it are still printed, but classified as float-noise rather than
a hard change.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


def _resolve(path: str) -> str:
    if os.path.isdir(path):
        return os.path.join(path, "smoke_report.json")
    return path


def flatten(obj, prefix: str = ""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, prefix + "/" + str(k)))
    elif isinstance(obj, list):
        out[prefix + "/__len__"] = len(obj)
        for i, v in enumerate(obj):
            out.update(flatten(v, prefix + f"/{i}"))
    else:
        out[prefix] = obj
    return out


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument(
        "--allow-regex",
        action="append",
        default=[],
        help="paths matching this regex are allowed to differ (repeatable)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="absolute float tolerance; within it a diff is float-noise, not a change",
    )
    args = ap.parse_args(argv)

    b_path = _resolve(args.baseline)
    n_path = _resolve(args.new)
    with open(b_path) as fh:
        base = flatten(json.load(fh))
    with open(n_path) as fh:
        new = flatten(json.load(fh))

    allowed = [re.compile(p) for p in args.allow_regex]

    def is_allowed(path):
        return any(r.search(path) for r in allowed)

    missing = []
    changed = []
    float_noise = []
    new_keys = []

    for path, bval in base.items():
        if path not in new:
            missing.append(path)
            continue
        nval = new[path]
        if bval == nval:
            continue
        if _is_num(bval) and _is_num(nval):
            try:
                delta = abs(float(bval) - float(nval))
            except (TypeError, ValueError):
                delta = None
            if args.tol > 0.0 and delta is not None and delta <= args.tol:
                float_noise.append((path, bval, nval, delta))
                continue
        if is_allowed(path):
            continue
        changed.append((path, bval, nval))

    for path in new:
        if path not in base:
            new_keys.append(path)

    hard = len(missing) + len(changed)

    print(f"baseline: {b_path}")
    print(f"new:      {n_path}")
    print(f"baseline keys: {len(base)}   new keys: {len(new)}   new keys added: {len(new_keys)}")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for p in missing:
            print(f"  - {p}")
    if changed:
        print(f"\nCHANGED ({len(changed)}):")
        for p, b, n in changed:
            print(f"  ! {p}: {b!r} -> {n!r}")
    if float_noise:
        print(f"\nfloat-noise within tol={args.tol} ({len(float_noise)}):")
        for p, b, n, d in float_noise:
            print(f"  ~ {p}: {b!r} -> {n!r} (|d|={d:.3e})")
    if new_keys:
        print(f"\nnew keys ({len(new_keys)}), permitted:")
        for p in new_keys:
            print(f"  + {p}")

    print()
    if hard == 0:
        print("SMOKE REGRESSION: identical on all baseline keys"
              + (f" ({len(float_noise)} float-noise, within tol)" if float_noise else ""))
        return 0
    print(f"SMOKE REGRESSION: FAILED — {hard} hard difference(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
