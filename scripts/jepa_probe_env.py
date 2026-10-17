#!/usr/bin/env python3
"""Thin CLI wrapper for `jepa_noise.probe_env` (the Phase-0 probe / equivalence fixture).

The implementation is the module `src/jepa_noise/probe_env.py`, because GPU jobs start through
`scripts/jepa_run.sh`, which launches `python -m <module>` inside the guard-gated launcher.

On the Mac (comparison side):
    PYTHONPATH=src python scripts/jepa_probe_env.py \
        --run-id probe_local --out results/jepa/<run_id>/local --device cpu

On a host (never by hand: use jepa_run.sh with JEPA_JOB_MODULE=jepa_noise.probe_env and
JEPA_JOB_KIND=probe).
"""
from jepa_noise.probe_env import main

if __name__ == '__main__':
    raise SystemExit(main())
