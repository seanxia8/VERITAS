#!/usr/bin/env python3
"""Phase-0 environment probe and the mandatory 10-batch equivalence fixture (module).

Runbook v4 §6.1 (existing GPU torch environment, read-only): "record its path and versions;
run a 10-batch equivalence test against the local environment".  This module is the two-sided
fixture for that test and the "one <= 10-minute GPU probe job per host" of Phase 0 step 4:
imports, one forward pass, an environment freeze and timings.  It is a module so that
``scripts/jepa_run.sh`` (which launches ``python -m <module>``) can start it as a guard-gated job.

It is deliberately independent of `jepa_noise` (which pulls the noise packages): it uses the
frozen `reconstruction_model` compact transformer that WP4a subclasses, with a fixed seed, so
the same numbers can be reproduced on the Mac (CPU) and compared with the host (GPU).

Usage (host, inside the guard-gated launcher):
    PYTHONPATH=src python -m jepa_noise.probe_env \
        --run-id probe_deepthought_<sha> --out /ceph/dwong/jepa/returns/<run_id> --device cuda:0

Usage (Mac, comparison side):
    PYTHONPATH=src python -m jepa_noise.probe_env \
        --run-id probe_local --out results/jepa/<run_id>/local --device cpu

Writes <out>/probe.json and <out>/freeze.txt.  Exit 0 = measurements complete; exit 2 = a
required import or device check failed (the failure is still written to probe.json).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata

import numpy as np
import torch


def _freeze() -> list[str]:
    try:
        return sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                      if d.metadata and d.metadata.get('Name'))
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"freeze-unavailable: {type(exc).__name__}: {exc}"]


def _git_head(repo: str) -> str | None:
    try:
        return subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def build_model(seed: int = 0):
    """The v3 WP4 compact configuration, weights deterministically initialised from `seed`."""
    from reconstruction_model.model import Transformer, TransformerConfig
    cfg = TransformerConfig(d_model=128, d_ff=256, patch_len=32, patch_stride=32, n_head=4,
                            n_time_layers=1, n_channel_layers=1)
    torch.manual_seed(seed)
    model = Transformer(cfg)
    model.init_weights()
    return model, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batches', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--channels', type=int, default=8)
    ap.add_argument('--samples', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--timed-steps', type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    result: dict = {
        'run_id': args.run_id,
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'argv': sys.argv,
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'device_requested': args.device,
        'repo_head': _git_head(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'cwd': os.getcwd(),
        'PYTHONPATH': os.environ.get('PYTHONPATH'),
        'status': 'started',
    }

    # ---- environment / device facts -------------------------------------------------------
    try:
        result['torch'] = {
            'version': torch.__version__,
            'version_cuda': torch.version.cuda,
            'cuda_available': bool(torch.cuda.is_available()),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'arch_list': list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
            'cudnn_version': torch.backends.cudnn.version(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        }
        if torch.cuda.is_available():
            idx = int(str(args.device).split(':')[1]) if ':' in str(args.device) else 0
            result['torch']['device_name'] = torch.cuda.get_device_name(idx)
            result['torch']['device_capability'] = list(torch.cuda.get_device_capability(idx))
            result['torch']['memory_total_mib'] = torch.cuda.get_device_properties(idx).total_memory / 2**20
    except Exception as exc:
        result['torch_error'] = f'{type(exc).__name__}: {exc}'
        result['status'] = 'failed_import'
        _write(args.out, result)
        return 2

    # The runbook requires TF32 disabled everywhere for training; verify, then set explicitly.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result['torch']['tf32_matmul_after'] = torch.backends.cuda.matmul.allow_tf32
    result['torch']['tf32_cudnn_after'] = torch.backends.cudnn.allow_tf32

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    result['device_used'] = str(device)

    # ---- model + fixed 10-batch fixture ---------------------------------------------------
    try:
        model, cfg = build_model(args.seed)
        model.to(device).eval()
        result['model'] = {'class': type(model).__name__, 'config': cfg.__dict__,
                           'n_params': sum(p.numel() for p in model.parameters())}
        # The repository's muP init zeroes every residual output projection and the two heads, so
        # the pooled head outputs are identically zero at initialisation. The equivalence fixture
        # and the timing loss therefore use the pre-final_proj_norm token grid (the WP4a read-out),
        # captured with a forward hook on final_proj_norm; that tensor is the first non-degenerate
        # representation in the network.
        captured: dict = {}

        def _capture(_module, hook_args, _output):
            captured['grid'] = hook_args[0]

        model.final_proj_norm.register_forward_hook(_capture)
        rng = np.random.default_rng(args.seed)
        x_np = rng.standard_normal(
            (args.batches, args.batch_size, args.channels, args.samples)).astype(np.float32)
        fixture = []
        with torch.no_grad():
            for b in range(args.batches):
                x = torch.from_numpy(x_np[b]).to(device)
                model(x)
                grid = captured['grid'].double()
                fixture.append({
                    'batch': b,
                    'x_sum': float(x_np[b].astype(np.float64).sum()),
                    'x_abs_sum': float(np.abs(x_np[b].astype(np.float64)).sum()),
                    'grid_sum': float(grid.sum().item()),
                    'grid_abs_sum': float(grid.abs().sum().item()),
                    'grid_first8': [float(v) for v in grid.flatten()[:8]],
                    'grid_shape': list(grid.shape),
                })
        result['fixture'] = {'batches': args.batches, 'batch_size': args.batch_size,
                             'channels': args.channels, 'samples': args.samples,
                             'seed': args.seed, 'readout': 'pre_final_proj_norm_token_grid',
                             'per_batch': fixture}
    except Exception as exc:
        result['forward_error'] = f'{type(exc).__name__}: {exc}'
        result['status'] = 'failed_forward'
        _write(args.out, result)
        return 2

    # ---- timings (forward + backward, 3 warm-ups then `timed-steps`) ----------------------
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        times = []
        for step in range(3 + args.timed_steps):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            x = torch.from_numpy(x_np[step % args.batches]).to(device)
            model(x)
            loss = captured['grid'].square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if step >= 3:
                times.append(dt)
        result['timing'] = {
            'seconds_per_step': times,
            'mean_seconds_per_step': float(np.mean(times)),
            'events_per_second': float(args.batch_size / np.mean(times)),
            'peak_memory_mib': (torch.cuda.max_memory_allocated(device) / 2**20
                                if device.type == 'cuda' else None),
            'note': 'one forward+backward at the probe batch size; not a training benchmark',
        }
    except Exception as exc:
        result['timing_error'] = f'{type(exc).__name__}: {exc}'

    # ---- 10-batch determinism on this device (repeat batch 0 twice) ----------------------
    try:
        with torch.no_grad():
            x = torch.from_numpy(x_np[0]).to(device)
            model(x)
            a = captured['grid'].double().clone()
            model(x)
            b = captured['grid'].double().clone()
        result['determinism_same_device_max_abs_diff'] = float((a - b).abs().max().item())
    except Exception as exc:
        result['determinism_error'] = f'{type(exc).__name__}: {exc}'

    result['status'] = 'ok'
    _write(args.out, result)
    return 0


def _write(out: str, result: dict) -> None:
    with open(os.path.join(out, 'probe.json'), 'w') as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    with open(os.path.join(out, 'freeze.txt'), 'w') as fh:
        for line in _freeze():
            fh.write(line + '\n')
    print(json.dumps({k: result.get(k) for k in
                      ('run_id', 'status', 'host', 'device_used', 'torch', 'timing')},
                     indent=1, sort_keys=True))


if __name__ == '__main__':
    raise SystemExit(main())
