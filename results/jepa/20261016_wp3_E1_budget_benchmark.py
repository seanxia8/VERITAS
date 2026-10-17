"""WP3 E1 campaign budget benchmark — measurement only.

Purpose: obtain MEASURED single-thread wall times for the representative
operations of the registered E1 design, so that results/jepa/
20261016_wp3_E1_BUDGET_PROJECTION.json rests on measurement rather than
invention. This script runs no campaign, no sample grid, no 100000-step
trajectory and no GPU work, and it writes no file: it prints one JSON object
of measurements to stdout. A3 is UNAPPROVED; nothing here adopts A3 defaults,
it only times representative sizes.

Required environment (runbook §3.1):
  PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  RECONSTRUCTION_DISABLE_TORCH_COMPILE=1
  /private/tmp/jepa-venv-cpu311/bin/python
"""
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

OUT = {"script": os.path.abspath(__file__), "block_seconds": {}, "benchmarks": {}}


def timed(fn, *args, **kwargs):
    t0 = time.monotonic()
    out = fn(*args, **kwargs)
    return out, time.monotonic() - t0


def block(name, fn, payload):
    """Run one benchmark block; a failure is reported, never silently dropped."""
    t0 = time.monotonic()
    try:
        fn(payload)
        payload["status"] = "ok"
    except Exception as exc:  # pragma: no cover - diagnostic path
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    payload["seconds"] = time.monotonic() - t0
    OUT["block_seconds"][name] = payload["seconds"]
    print(f"[block] {name}: {payload['seconds']:.3f}s {payload['status']}",
          file=sys.stderr, flush=True)


# ---------------------------------------------------------------- environment
OUT["env"] = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    "RECONSTRUCTION_DISABLE_TORCH_COMPILE": os.environ.get(
        "RECONSTRUCTION_DISABLE_TORCH_COMPILE"),
    "PYTHONPATH": os.environ.get("PYTHONPATH"),
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

from jepa_noise import crlb as CRLB  # noqa: E402
from jepa_noise import dynamics as DYN  # noqa: E402
from jepa_noise import linear as LIN  # noqa: E402
from jepa_noise import probes as PROBES  # noqa: E402
from jepa_noise.masks import (  # noqa: E402
    namespace_rng, oracle_preprocess, retained_oracle_transform,
)
from jepa_noise.views import ExactModel, Tier1Views  # noqa: E402

K = 3

# ------------------------------------------------------- A: E1a exact model
A = {}


def bench_e1a(p):
    from jepa_noise.views import ExactModel as EM

    gen = {}
    last = None
    for n in (1000, 10000, 100000):
        model = EM(seed=0, theta_scale=.4, eta_scale=3., share="target")
        sample, secs = timed(model.sample, n, 1.0)
        gen[str(n)] = {
            "dataset_seconds": secs,
            "per_event_seconds": secs / n,
            "n_events": n,
            "context_shape": list(np.shape(sample["context"])),
            "target_shape": list(np.shape(sample["target"])),
            "calibration_noise_shape": list(
                np.shape(sample["calibration_noise_context"])),
            "note": "ExactModel.sample draws n calibration-noise records in "
                    "the same call; A3's n_cal=n_subject clause therefore adds "
                    "no separate generation call for E1a",
        }
        last = sample
    p["generation"] = gen

    cov, secs = timed(LIN.sample_covariances, last["context"],
                      last["target"], ridge_relative=1e-6)
    p["sample_covariance_ridge_n100000_d40_seconds"] = secs
    p["sample_covariance_ridge_returns"] = [list(np.shape(x)) for x in cov]

    model = EM(seed=0, theta_scale=.4, eta_scale=3., share="target")
    pop, secs = timed(model.population, 1.0)
    p["population_call_seconds"] = secs
    scc, stt, sct = pop["scc"], pop["stt"], pop["sct"]
    ncc, ntt, nct = pop["ncc"], pop["ntt"], pop["nct"]
    ac, at = pop["ac"], pop["at"]

    idx = {}
    _, idx["cca"] = timed(LIN.cca, scc, stt, sct, K)
    _, idx["hard_isotropy"] = timed(LIN.cca, scc, stt, sct, K)
    _, idx["gevp_sample_style"] = timed(LIN.gevp, scc, ncc, K)
    for beta in (.1, 1., 10.):
        _, idx[f"finite_beta_{beta:g}"] = timed(
            LIN.finite_beta, scc, stt, sct, ncc, K, beta)
    _, idx["pca"] = timed(LIN.pca, scc, K)
    _, idx["whitened_pca"] = timed(LIN.pca, scc, K, ncc)
    _, idx["oracle"] = timed(LIN.oracle, scc, stt, sct, ncc, ntt, nct,
                             ac, at, K)
    _, idx["fisher"] = timed(CRLB.fisher, ac, ncc)
    _, idx["crlb"] = timed(CRLB.crlb, ac, ncc)
    _, idx["of_solve"] = timed(np.linalg.solve, ncc, ac)
    p["solvers_d40_seconds"] = idx


block("A_e1a_exact_model", bench_e1a, A)
OUT["benchmarks"]["A_e1a_exact_model"] = A

# ------------------------------------------------ B: E1b linearized waveform
B = {}


def bench_e1b(p):
    p["splits"] = {}
    for split in ("time", "channel", "interleaved", "block"):
        entry = {}
        tv, secs = timed(Tier1Views, split=split, line_multiplier=1.5,
                         common_strength=.45, seed=0, cell="wp3_bench",
                         model_seed=0)
        entry["construct_seconds"] = secs
        pop = tv.population()
        dc = pop["ncc"].shape[0]
        dt = pop["ntt"].shape[0]
        entry["context_dim"], entry["target_dim"] = dc, dt
        entry["unused_coords"] = int(len(pop["unused_indices"]))
        scc, stt, sct = pop["scc"], pop["stt"], pop["sct"]
        ncc, ntt, nct = pop["ncc"], pop["ntt"], pop["nct"]
        sol = {}
        _, sol["cca"] = timed(LIN.cca, scc, stt, sct, K)
        _, sol["gevp"] = timed(LIN.gevp, scc, ncc, K)
        for beta in (.1, 1., 10.):
            _, sol[f"finite_beta_{beta:g}"] = timed(
                LIN.finite_beta, scc, stt, sct, ncc, K, beta)
        _, sol["pca"] = timed(LIN.pca, scc, K)
        _, sol["whitened_pca"] = timed(LIN.pca, scc, K, ncc)
        _, sol["oracle"] = timed(LIN.oracle, scc, stt, sct, ncc, ntt, nct,
                                 pop["ac"], pop["at"], K)
        _, sol["fisher"] = timed(CRLB.fisher, pop["ac"], ncc)
        _, sol["crlb"] = timed(CRLB.crlb, pop["ac"], ncc)
        _, sol["of_solve"] = timed(np.linalg.solve, ncc, pop["ac"])
        entry["solvers_seconds"] = sol
        tr, secs = timed(retained_oracle_transform, tv.noise_covariance,
                         tv.split)
        entry["retained_oracle_transform_seconds"] = secs
        entry["retained_dim"] = int(tr[0].shape[0])
        raw = np.random.default_rng(11).normal(size=(2000, 8, 256))
        _, secs = timed(oracle_preprocess, raw, tv.split, tr[0], tr[1])
        entry["oracle_preprocess_per_event_seconds"] = secs / 2000
        entry["oracle_preprocess_2000_events_seconds"] = secs
        ids = np.arange(2000)
        _, secs = timed(tv.sample, ids, "E1b_subject_bench")
        entry["subject_sample_per_event_seconds"] = secs / 2000
        entry["subject_sample_2000_events_seconds"] = secs

        def noise_loop():
            # noise_record takes one scalar event id (per-event API), so the
            # registered 100000-record cohorts are inherently a Python loop.
            for event_id in ids:
                tv.noise_record(int(event_id), "E1b_noise_cal_bench")

        _, secs = timed(noise_loop)
        entry["noise_record_per_event_seconds"] = secs / 2000
        entry["noise_record_2000_events_seconds"] = secs

        # sample covariance accumulation at the dimensions this split needs.
        rng = np.random.default_rng(12)
        rawc = rng.normal(size=(10000, dc))
        rawt = rng.normal(size=(10000, dt))
        _, secs = timed(LIN.sample_covariances, rawc, rawt, ridge_relative=1e-6)
        entry["sample_covariance_10000_events_seconds"] = secs
        entry["sample_covariance_10000_events_dims"] = [dc, dt]
        retained_dim = int(tr[0].shape[0])
        rawr = rng.normal(size=(10000, retained_dim))
        _, secs = timed(LIN.sample_covariances, rawr, rawr, ridge_relative=1e-6)
        entry["oracle_path_covariance_10000_events_seconds"] = secs
        entry["oracle_path_covariance_dims"] = [retained_dim, retained_dim]
        p["splits"][split] = entry

    # Surrogate for the not-yet-implemented Criterion-M scoring step only.
    rng = namespace_rng(2718, "M/qr", "wp3_bench")
    q, _ = np.linalg.qr(rng.normal(size=(768, 8)))
    rec_e = np.random.default_rng(13).normal(size=(4096, 768)) @ q
    rec_n = np.random.default_rng(14).normal(size=(4096, 768)) @ q

    def m_surrogate():
        ce = np.cov(rec_e, rowvar=False, bias=True)
        cn = np.cov(rec_n, rowvar=False, bias=True)
        return np.linalg.svd(ce - cn, compute_uv=False)

    _, secs = timed(m_surrogate)
    p["criterion_M_projection_surrogate_seconds"] = secs
    p["criterion_M_note"] = ("Criterion-M scoring is NOT implemented as a "
                             "package function; this is a surrogate timing for "
                             "the 8-dim projected covariance/eigensolve step "
                             "only. The dominant M cost is raw record "
                             "generation, measured above.")


block("B_e1b_waveform", bench_e1b, B)
OUT["benchmarks"]["B_e1b_waveform"] = B

# ------------------------------------------------------------- C: E1c A2 GD
C = {}


def bench_e1c(p):
    rng = np.random.default_rng(7)
    context = rng.normal(size=(100000, 40))
    target = rng.normal(size=(100000, 40))
    moments, secs = timed(DYN.empirical_moments, context, target)
    p["moment_build_n100000_d40_seconds"] = secs
    ac = rng.normal(size=(40, 3))
    bc = rng.normal(size=(40, 1))
    layers, predictor = DYN.initialize(0, 3)
    steps = 2000
    out, secs = timed(DYN.run, moments, layers, predictor, ac, bc,
                      np.full(3, .4), 3., budget_seconds=600.,
                      engineering_steps=steps)
    p["steps"] = steps
    p["run_seconds"] = secs
    p["per_step_seconds"] = secs / steps
    p["status"] = out["status"]
    p["reason"] = out["reason"]
    p["note"] = ("engineering_steps is explicitly incomplete; scaled to the "
                 "registered 100000-step horizon. Numbers here are a timing "
                 "fixture, not an R5 outcome.")
    p["moment_source"] = moments.source
    p["moment_n"] = int(moments.n)


block("C_e1c_dynamics", bench_e1c, C)
OUT["benchmarks"]["C_e1c_dynamics"] = C

# ------------------------------------------------------ D: grouped ridge probes
DD = {}


def bench_probes(p):
    rng = np.random.default_rng(21)
    x = rng.normal(size=(10000, K))
    y = rng.normal(size=(10000, 4))
    groups = np.repeat(np.arange(5), 2000)
    res, secs = timed(PROBES.grouped_ridge, x, y, groups, fit_intercept=True)
    p["grouped_ridge_4outputs_n10000_dim3_seconds"] = secs
    p["per_output_seconds"] = secs / 4
    p["outputs"] = 4
    p["note"] = ("A3's planning count is 26 individual fits per valid output "
                 "(5 alphas x 5 folds + 1 final refit), matching one "
                 "grouped_ridge output.")
    _, secs = timed(PROBES.fit_ridge, x, y[:, 0], alpha=1.0, fit_intercept=True)
    p["single_fit_seconds"] = secs
    _, secs = timed(PROBES.evaluate, res, x, y)
    p["evaluate_4outputs_seconds"] = secs
    for label, dim in (("e1a_context_d40", 40), ("e1b_time_context_d768", 768)):
        feats = rng.normal(size=(20000, dim))
        w = rng.normal(size=(K, dim))
        _, secs = timed(lambda: feats @ w.T)
        p[f"embedding_extract_per_20000_events_{label}_seconds"] = secs
    statuses = [r["status"] for r in res]
    p["status_example"] = statuses


block("D_probes", bench_probes, DD)
OUT["benchmarks"]["D_probes"] = DD

# -------------------------------------------- E: transfer/hashing throughput
E = {}


def bench_io(p):
    payload = np.random.default_rng(31).standard_normal(8 * 1024 * 1024)
    blob = payload.tobytes()
    p["artifact_bytes"] = len(blob)
    t0 = time.monotonic()
    sha = hashlib.sha256(blob).hexdigest()
    secs = time.monotonic() - t0
    p["sha256_seconds"] = secs
    p["sha256_MB_per_s"] = len(blob) / 1e6 / secs
    p["sha256_prefix"] = sha[:16]
    t0 = time.monotonic()
    path = "/private/tmp/wp3_bench_io.bin"
    with open(path, "wb") as handle:
        handle.write(blob)
    p["write_seconds"] = time.monotonic() - t0
    t0 = time.monotonic()
    with open(path, "rb") as handle:
        handle.read()
    p["read_seconds"] = time.monotonic() - t0
    os.remove(path)
    p["note"] = ("Local page-cache throughput only; not a network transfer "
                 "measurement. Transfer/rsync is estimated separately.")


block("E_io_hashing", bench_io, E)
OUT["benchmarks"]["E_io_hashing"] = E

OUT["total_seconds"] = sum(OUT["block_seconds"].values())
OUT["max_block_seconds"] = max(OUT["block_seconds"].values())
OUT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
print(json.dumps(OUT, indent=2, sort_keys=True))
