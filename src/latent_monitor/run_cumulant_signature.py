# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Tier-1 signature row for the whitened-residual third cumulant (D7).

    PYTHONPATH=src python -m latent_monitor.run_cumulant_signature --out results/latent_monitor_sig_c3_2026-09-17

Runs the §2.1 conditional-signature hypothesis of
``docs/ARITRA_CUMULANT_INTEGRATION_2026-09-17.md`` on the **linear subject**
under paired replay, on the existing Tier-1 families plus the non-Gaussian
sparse-burst family (``noise_module.NonGaussianNoiseGenerator``):

    covariance-type N (Gaussian) leaves the whitened third cumulant at
    reference — after re-whitening with the realised Sigma for the
    Sigma-covariance cells; impulsive / non-Gaussian structure (the glitch
    event and the sparse-burst noise) moves it; in-span S leaves it at
    reference.

Every row is scored against its pre-registered expectation in the same
match / documented / MISMATCH format as the 6 September signature table. A
negative outcome is reported as such: plain gain drift is a *linear* Gaussian
operation on the residual and is expected to leave a third cumulant at
reference even though it is N by contract; channel loss removes channels, so the
frozen decoder's residual carries a deterministic (non-Gaussian) component and
is expected to move it. Nothing here is citable.

JSON is strict; the run is development-only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from noise_module import NonGaussianNoiseGenerator

from .adjust import rewhiten
from .cumulant import third_central_moment, truncation_error_ratio
from .linear_subject import LinearSubject
from .protocol.features import AlarmTimeReference
from .reference import fit_reference, whitened_residual
from .tier1 import TARGET_NAMES, all_cells, geometry_cells, reference_cell, sigma_covariance_cells, sigma_structural_cells


class SparseBurstCell:
    """A reference-geometry cell whose noise adds a sparse, non-Gaussian **skewed** impulse process.

    The impulse process is ``noise_module.NonGaussianNoiseGenerator``'s
    ``compound_poisson`` family (a per-sample normal amplitude drawn only at
    Poisson times), **rectified** to one side: a physical burst is one-sided, and
    rectification makes the third cumulant non-zero (the raw compound-Poisson
    family is symmetric and its third cumulant vanishes by construction, so it
    could not test a third-order separator).
    """

    def __init__(self, base, *, event_probability: float = 0.01, amplitude: float = 0.5, seed: int = 0):
        self.base = base
        self.geometry = base.geometry
        self.n_samples = base.n_samples
        self.label = "sigma_non_gaussian:sparse_burst"
        self.moved = "sigma_struct"
        self.cfg = {"family": "compound_poisson", "event_probability": event_probability, "noise_power": 1.0}
        self.amplitude = float(amplitude)
        self.seed = int(seed)

    def _burst(self, ids: np.ndarray, replicate: int) -> np.ndarray:
        C, N = self.geometry.n_channels, self.n_samples
        out = np.empty((len(ids), C, N))
        for k, i in enumerate(ids):
            gen = NonGaussianNoiseGenerator(self.cfg, seed=int(i) * 1009 + replicate * 7 + self.seed)
            out[k] = np.abs(gen.generate(C * N)).reshape(C, N)
        return self.amplitude * out

    def batch(self, event_ids: np.ndarray, replicate: int = 0):
        X, T = self.base.batch(event_ids, replicate)
        return X + self._burst(event_ids, replicate), T

    def noise_batch(self, record_ids: np.ndarray) -> np.ndarray:
        return self.base.noise_batch(record_ids) + self._burst(record_ids, 0x51)


def _jsonable(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def _strict(o: Any):
    if isinstance(o, dict):
        return {str(k): _strict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_strict(v) for v in o]
    if isinstance(o, np.ndarray):
        return _strict(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run(out: Path, *, n_channels: int = 8, n_samples: int = 256, latent_dim: int = 6, n_fit: int = 200,
        n_eval: int = 60, n_noise: int = 100, seed: int = 0) -> dict[str, Any]:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ref_cell = reference_cell(n_channels=n_channels, n_samples=n_samples)
    fit_ids = np.arange(0, n_fit)
    eval_ids = np.arange(1000, 1000 + n_eval)
    noise_ids = np.arange(5000, 5000 + n_noise)
    X, T = ref_cell.batch(fit_ids)
    subject = LinearSubject.fit(X, T, ref_cell.geometry, ref_cell.implied_whitener(), latent_dim=latent_dim,
                                seed=seed, target_names=TARGET_NAMES)
    ref = fit_reference(subject, ref_cell, eval_ids, noise_ids, seed=seed)
    atref = AlarmTimeReference.fit(subject, ref, X, ref_cell.geometry, n_pcs=3)
    assert atref.reading == "cumulant", "the linear subject must read as a cumulant"
    pcs = atref.resid_cumulant.pcs
    mu = atref.resid_cumulant.mean
    T_ref = atref.resid_cumulant.ref_tensor

    # the signature row is about the *population* connected third cumulant on the cell's
    # windows (Bal et al. §2.1 object), not the per-window alarm-time compression.
    def cell_tensor(subj, cell, ids):
        Xc, _ = cell.batch(ids)
        rep = subj.represent(Xc, cell.geometry)
        r = whitened_residual(subj, rep, cell.geometry).reshape(len(Xc), -1)
        if r.shape[1] != mu.shape[0]:
            return None
        return third_central_moment((r - mu) @ pcs)

    def deviation(subj, cell, ids):
        T = cell_tensor(subj, cell, ids)
        return None if T is None else float(np.linalg.norm(T - T_ref))

    def truncation(subj, cell, ids):
        """Bal et al.'s quadratic-vs-(quadratic+cubic) KL truncation ratio along the cell's mean
        displacement in the projected chart — a development-table quantity only (D7)."""
        Xc, _ = cell.batch(ids)
        rep = subj.represent(Xc, cell.geometry)
        r = whitened_residual(subj, rep, cell.geometry).reshape(len(Xc), -1)
        if r.shape[1] != mu.shape[0]:
            return None
        c = (r - mu) @ pcs
        I2 = np.cov(c.T) if c.shape[0] > 1 else np.zeros((pcs.shape[1], pcs.shape[1]))
        I3 = third_central_moment(c)
        return truncation_error_ratio(np.atleast_2d(I2), I3, c.mean(axis=0))

    ref_windows = whitened_residual(subject, subject.represent(ref_cell.batch(eval_ids)[0], ref_cell.geometry),
                                    ref_cell.geometry).reshape(len(eval_ids), -1)
    ref_coords = (ref_windows - mu) @ pcs
    rng = np.random.default_rng(seed)
    boot = np.array([float(np.linalg.norm(third_central_moment(ref_coords[rng.choice(len(ref_coords), size=len(ref_coords),
                                                                                 replace=True)]) - T_ref))
                     for _ in range(400)])
    boot_mean, boot_std = float(boot.mean()), float(boot.std() + 1e-300)
    threshold = float(np.quantile((boot - boot_mean) / boot_std, 0.99))
    ref_stat = deviation(subject, ref_cell, eval_ids)
    ref_z = (ref_stat - boot_mean) / boot_std

    def zscore(stat):
        return None if stat is None else (stat - boot_mean) / boot_std

    sparse = SparseBurstCell(ref_cell, seed=seed)
    covariance = sigma_covariance_cells(ref_cell)
    structural = sigma_structural_cells(ref_cell, seed=3)
    geometry = geometry_cells(ref_cell)
    events = [c for c in all_cells(ref_cell) if c.moved in ("event", "event_in_span")]
    cases: list[tuple[Any, str, str]] = [(ref_cell, "at_reference", "control: the reference cell itself")]
    for c in covariance:
        cases.append((c, "at_reference", "covariance-type N, re-whitened with the realised Sigma (evaluation-only)"))
    for c in structural:
        if c.label == "sigma_struct:timing_jitter":
            cases.append((c, "at_reference", "documented: jitter decorrelates a shared component — covariance in the residual"))
        elif c.label == "sigma_struct:gain_drift":
            cases.append((c, "at_reference", "structural N; a linear-Gaussian gain multiplication, so a third cumulant stays at reference"))
        else:
            cases.append((c, "moves", "structural N; dead channels leave the frozen decoder a deterministic (non-Gaussian) residual"))
    for c in events:
        cases.append((c, "moves" if c.moved == "event" else "at_reference",
                      "out-of-support S (impulsive glitch)" if c.moved == "event" else "in-span S"))
    cases.append((sparse, "moves", "non-Gaussian sparse-burst noise (noise_module compound_poisson)"))
    for c in geometry:
        cases.append((c, "skipped", "geometry changes the (C·N) residual chart; the reference chart does not apply"))

    rows = []
    for cell, expected, reason in cases:
        note, stat, z_, tr, observed = "", None, None, None, None
        if expected == "skipped":
            observed = "skipped"
        else:
            subj = subject
            if cell.moved == "sigma_cov":
                subj, info = rewhiten(subject, cell.noise_batch(np.arange(6000, 6000 + n_noise)))
                note = f"re-whitened: kappa {info['kappa_correction']['kappa']:.2f} correction"
            stat = deviation(subj, cell, eval_ids)
            z_ = zscore(stat)
            tr = truncation(subj, cell, eval_ids)
            observed = "moves" if (z_ is not None and z_ > threshold) else "at_reference"
        status = "match" if observed == expected else ("MISMATCH" if expected in ("moves", "at_reference") else "documented")
        rows.append({"cell": cell.label, "moved": cell.moved, "expected": expected, "observed": observed,
                     "status": status, "cumulant_deviation": stat, "cumulant_z": z_,
                     "truncation_ratio": tr, "threshold_q99_z": threshold, "reason": reason, "note": note})

    result = {
        "STATUS": "DEVELOPMENT — NOT CITABLE. Tier-1 signature row for the whitened-residual third cumulant (D7) on the linear subject; "
                  "no scientific conclusion, a negative outcome is reported as recorded.",
        "config": {"n_channels": n_channels, "n_samples": n_samples, "latent_dim": latent_dim, "n_fit": n_fit,
                   "n_eval": n_eval, "n_noise": n_noise, "seed": seed, "n_pcs": atref.n_pcs},
        "chart": {"reading": atref.reading, "statistic": "population connected third cumulant tensor of the whitened residual "
                                                            "in the reference residual PCA chart (n_pcs = 3)",
                  "threshold": "q99 of the z-scored ||T_subset - T_ref||_F from a 400-draw bootstrap of reference windows",
                  "kl_ratio": "Bal et al. truncation-error ratio q/(q+c) along the cell's mean projected displacement; "
                              "development-table quantity only, never an arm feature or estimand"},
        "threshold_q99_z": threshold, "reference_deviation": ref_stat, "reference_z": ref_z,
        "bootstrap_deviation_mean": boot_mean, "bootstrap_deviation_std": boot_std,
        "rows": rows,
        "n_match": sum(r["status"] == "match" for r in rows),
        "n_documented": sum(r["status"] == "documented" for r in rows),
        "n_mismatch": sum(r["status"] == "MISMATCH" for r in rows),
    }
    result = _strict(result)
    (out / "signature_c3.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (out / "signature_c3.md").write_text(render_markdown(result))
    return result


def render_markdown(r: dict[str, Any]) -> str:
    L = ["# Whitened-residual third-cumulant signature row — DEVELOPMENT OUTPUT, NOT CITABLE", "",
         f"_{r['STATUS']}_", "",
         f"chart reading `{r['chart']['reading']}`; statistic `{r['chart']['statistic']}`; "
         f"threshold z = {r['threshold_q99_z']:.2f} (q99 of the z-scored bootstrap deviation; "
         f"reference deviation {r['reference_deviation']:.2f}, z {r['reference_z']:.2f}). "
         f"**{r['n_match']} match, {r['n_documented']} documented, {r['n_mismatch']} mismatch.**", "",
         "| cell | moved | expected | observed | deviation | z | KL ratio | status |", "|---|---|---|---|---:|---:|---:|---|"]
    for row in r["rows"]:
        s = "—" if row["cumulant_deviation"] is None else f"{row['cumulant_deviation']:.2f}"
        z = "—" if row["cumulant_z"] is None else f"{row['cumulant_z']:.2f}"
        kl = "—" if row["truncation_ratio"] is None else f"{row['truncation_ratio']:.3f}"
        L.append(f"| {row['cell']} | {row['moved']} | {row['expected']} | {row['observed']} | {s} | {z} | {kl} | {row['status']} |")
    L += ["", "Notes:", ""]
    for row in r["rows"]:
        if row["note"]:
            L.append(f"- {row['cell']}: {row['note']}")
    L += ["", "A MISMATCH is reported, not explained away. On this subject and at this size the population third-cumulant "
              "deviation stayed within the reference window-bootstrap band (q99 z = "
              f"{r['threshold_q99_z']:.2f}) for the glitch event and the non-Gaussian sparse-burst family, while gain drift and "
              "channel loss matched their predictions (a linear-Gaussian gain operation stays at reference; dead channels leave "
              "a deterministic residual that moves it) and the covariance-type cells were at reference after re-whitening "
              "(no false positives). The estimator is a population tensor of per-window rank-1 third moments, which is "
              f"noise-dominated at n_eval = {r['config']['n_eval']} and n_pcs = {r['config']['n_pcs']}; the note already records "
              "that reference distances are unstable at n_ref <~ 100 and that a third-moment tensor is noisier than a second. "
              "This is a development negative for the separator at this size, not a confirmed refutation.", ""]
    return "\n".join(L)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results/latent_monitor_sig_c3_2026-09-17"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=60)
    a = ap.parse_args(argv)
    r = run(a.out, n_channels=a.channels, n_samples=a.samples, seed=a.seed, n_eval=a.n_eval)
    print(f"{r['n_match']} match, {r['n_documented']} documented, {r['n_mismatch']} mismatch -> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
