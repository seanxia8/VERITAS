"""E1 (WP3) campaign runner: E1a, E1b, E1c.

Registered grids come only from ``src/jepa_noise/configs/e1_grid_v4.json``.
Per (cell, seed) rows are checkpointed as write-once JSON files under a fresh
attempt directory; a re-run never overwrites a completed row.  Every row file
carries a pass/fail/not_evaluable/not_run status with a reason code, and one
manifest row is appended per executed row.

Engineering choices used here are logged one line each in
``docs/jepa/WP3_E1_RUNNER.md`` (owned by this work package).
"""
from __future__ import annotations

# Single-threaded BLAS/OMP is part of the registered environment contract
# (v3 runbook Sec.3.1, v4 Sec.6).  This must happen before numpy is imported in
# the parent and (via re-import) in every spawned worker.
import os

for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import platform
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "src" / "jepa_noise" / "configs" / "e1_grid_v4.json"
CAMPAIGNS = ("E1a", "E1b", "E1c")
CAMPAIGN_DIR = {
    "E1a": "20261017_wp3_e1a",
    "E1b": "20261017_wp3_e1b",
    "E1c": "20261017_wp3_e1c",
}
STATUSES = ("pass", "fail", "not_evaluable", "not_run")


class CampaignBlocked(RuntimeError):
    """A registered campaign cannot run with the installed source tree."""


class BudgetStop(RuntimeError):
    """Projected or measured cost would exceed the E1 campaign cap."""


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover - git is always available here
        return "unknown"


def git_status(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def src_tree_status(root: Path) -> str:
    """`git status --porcelain -- src scripts` (author rule for registered runs)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "src", "scripts"],
            capture_output=True, text=True,
        )
    except Exception as error:  # pragma: no cover - git missing
        return f"git-status-unavailable: {error}"
    return result.stdout.strip()


def host_name() -> str:
    return platform.node() or "unknown-host"


def fmt_num(value: float) -> str:
    return "%g" % float(value)


def _jsonable(value):
    """Recursively convert numpy values and reject non-finite floats."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite float in result record")
        return number
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"unsupported result type {type(value)!r}")


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact deterministic JSON keeps each results directory inside the
    # v4 Sec.7 size limit while preserving every stored value exactly.
    text = json.dumps(_jsonable(payload), separators=(",", ":"),
                      sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_digest(array) -> str:
    return sha256_bytes(np.ascontiguousarray(array, dtype="<f8").tobytes())


def source_hashes(root: Path, config_path: Path) -> dict:
    paths = sorted((root / "src" / "jepa_noise").rglob("*.py"))
    paths += [
        config_path,
        root / "docs" / "jepa" / "RUNBOOK_JEPA_NOISE_v4_2026-10-16.md",
        root / "docs" / "jepa" / "RUNBOOK_JEPA_NOISE_2026-10-16.md",
    ]
    hashes = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            key = str(path.relative_to(root))
        except ValueError:  # an explicit --config outside the repository root
            key = str(path)
        hashes[key] = sha256_file(path)
    return hashes


def available_memory_gb() -> float:
    """Best-effort available-RAM estimate; recorded, never silently invented."""
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        total = 0
    if sys.platform == "darwin":
        try:
            output = subprocess.check_output(["vm_stat"], text=True)
            page, free, inactive, purgeable = 4096, 0, 0, 0
            for line in output.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split("bytes")[0].strip())
                elif line.startswith("Pages free:"):
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages purgeable:"):
                    purgeable = int(line.split(":")[1].strip().rstrip("."))
            available = (free + inactive + purgeable) * page
        except Exception:  # pragma: no cover - fallback only
            available = total / 4
    else:  # pragma: no cover - linux fallback
        try:
            rows = {}
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    rows[key] = value
            available = int(rows["MemAvailable"].strip().split()[0]) * 1024
        except Exception:
            available = total / 4
    return round(float(available) / 1e9, 2)


def physical_cores() -> int:
    try:
        return int(subprocess.check_output(
            ["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
    except Exception:
        return os.cpu_count() or 1


def choose_workers(config: dict, requested: int | None) -> dict:
    """Worker count from v4 Sec.6, recorded with the measured inputs."""
    cores = physical_cores()
    available = available_memory_gb()
    measured_formula = min(
        cores - 2, int(available // config["hosts"]["gb_per_worker"])
    )
    configured_cap = int(config["hosts"]["max_workers_by_config"])
    if requested is not None:
        workers = int(requested)
    else:
        # v4 Sec.6 registered rule, measured at launch: the recorded worker
        # count is min(physical cores - 2, floor(free GB / 4)), hard-capped by
        # the config and never below one; every attempt records the inputs.
        workers = min(configured_cap, max(1, measured_formula))
    if workers < 1:
        raise ValueError("worker count must be positive")
    return {
        "workers": workers,
        "physical_cores": cores,
        "available_gb_at_start": available,
        "gb_per_worker": config["hosts"]["gb_per_worker"],
        "measured_formula": measured_formula,
        "configured_cap": configured_cap,
        "note": ("workers=min(physical-2, floor(available/4)); minimum 1; "
                 "the measured value is recorded in the attempt manifest"),
    }


# --------------------------------------------------------------------------
# configuration and registered grids
# --------------------------------------------------------------------------
def load_config(path: Path) -> tuple[dict, str]:
    config = read_json(path)
    if config.get("schema") != "jepa-e1-grid-v4":
        raise ValueError("unexpected E1 grid config schema")
    return config, sha256_file(path)


def e1a_cell_label(cell: dict) -> str:
    return (f"th{fmt_num(cell['theta_scale'])}_eta{fmt_num(cell['eta_scale'])}"
            f"_g{fmt_num(cell['gamma_load'])}")


def enumerate_rows(config: dict, campaign: str, scope: str = "full") -> list[dict]:
    """Registered rows in a deterministic scientific-settings order."""
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown campaign {campaign!r}")
    spec = config[campaign]
    rows: list[dict] = []
    if campaign == "E1a":
        n_values = spec["smoke"]["n"] if scope == "smoke" else spec["n"]
        seeds = spec["smoke"]["seeds"] if scope == "smoke" else spec["seeds"]
        for theta in spec["theta_scale"]:
            for eta in spec["eta_scale"]:
                for gamma in spec["gamma_load"]:
                    for n in n_values:
                        for seed in seeds:
                            cell = {"theta_scale": theta, "eta_scale": eta,
                                    "gamma_load": gamma, "n": n}
                            rows.append({
                                "campaign": "E1a",
                                "row_id": f"{e1a_cell_label(cell)}_n{n}_s{seed}",
                                "cell": cell,
                                "seed": int(seed),
                            })
    elif campaign == "E1b":
        if scope == "smoke":
            raise ValueError("E1b has no reduced smoke scope in the registered design")
        for cell in spec["cells"]:
            for seed in spec["seeds"]:
                rows.append({
                    "campaign": "E1b",
                    "row_id": f"{cell['cell_id']}_s{seed}",
                    "cell": {"cell_id": cell["cell_id"],
                             "common_strength": cell["common_strength"],
                             "line_multiplier": cell["line_multiplier"]},
                    "seed": int(seed),
                })
    else:
        if scope == "smoke":
            raise ValueError("E1c has no reduced smoke scope in the registered design")
        for gamma in spec["gamma_load"]:
            for seed in spec["seeds"]:
                rows.append({
                    "campaign": "E1c",
                    "row_id": f"g{fmt_num(gamma)}_s{seed}",
                    "cell": {"gamma_load": gamma},
                    "seed": int(seed),
                })
    identifiers = [row["row_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate row identifiers in registered grid")
    return rows


def expected_row_count(config: dict, campaign: str, scope: str) -> int:
    if campaign == "E1a":
        key = "expected_smoke_rows" if scope == "smoke" else "expected_rows"
        return int(config["E1a"][key])
    return int(config[campaign]["expected_rows"])


# --------------------------------------------------------------------------
# attempt directories, discovery and manifests
# --------------------------------------------------------------------------
def campaign_root(root: Path, campaign: str) -> Path:
    return root / "results" / "jepa" / CAMPAIGN_DIR[campaign]


def list_attempts(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir()
                  if path.is_dir() and path.name.startswith("attempt_"))


def create_attempt(directory: Path, campaign: str) -> Path:
    sequence = len(list_attempts(directory)) + 1
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = directory / f"attempt_{sequence:03d}_{stamp}_{campaign}"
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "rows").mkdir()
    (attempt / "logs" / "rows").mkdir(parents=True)
    return attempt


def discover_rows(directory: Path) -> dict[str, dict]:
    """Latest recorded row per identifier across every attempt (write-once files)."""
    discovered: dict[str, dict] = {}
    for attempt in list_attempts(directory):
        for path in sorted((attempt / "rows").glob("*.json")):
            try:
                record = read_json(path)
            except Exception:
                continue
            identifier = record.get("row_id")
            if not identifier:
                continue
            record["_record_path"] = str(path)
            record["_attempt"] = attempt.name
            discovered[identifier] = record
    return discovered


def append_manifest(attempt: Path, payload: dict) -> None:
    with (attempt / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def make_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def row_logger(attempt: Path, row_id: str):
    path = attempt / "logs" / "rows" / f"{row_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utcnow()} {message}\n")

    return log


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        value /= 1024.0 ** 2
    else:  # pragma: no cover - linux reports KiB
        value /= 1024.0
    return round(value, 1)


# --------------------------------------------------------------------------
# budget guard (v3 Sec.3.2, v4 Sec.6; a G2 decision belongs to the Lead)
# --------------------------------------------------------------------------
def serial_hours_estimate(config: dict, campaign: str) -> float:
    return float(config["budget"].get("serial_hours_estimate", {}).get(campaign, 0.0))


def project_campaign(config: dict, campaign: str, row_count: int, workers: int,
                     row_seconds: float | None = None) -> dict:
    budget = config["budget"]
    if row_seconds is None:
        row_seconds = float(budget["row_seconds_estimate"][campaign])
    worker_hours = row_count * float(row_seconds) / 3600.0
    wall_hours = worker_hours / max(1, workers)
    serial = serial_hours_estimate(config, campaign)
    charged = float(budget["already_charged_cpu_hours"])
    cumulative = charged + wall_hours + serial
    return {
        "row_count": int(row_count),
        "row_seconds_estimate": float(row_seconds),
        "workers": int(workers),
        "worker_hours": round(worker_hours, 6),
        "wall_hours_after_workers": round(wall_hours, 6),
        "serial_hours_estimate": serial,
        "already_charged_hours": charged,
        "cumulative_hours_estimate": round(cumulative, 6),
        "cap_hours": float(budget["cpu_cap_hours"]),
        "fits_cap": cumulative <= float(budget["cpu_cap_hours"]),
        "projection_reference_hours_at_1_worker": float(
            budget["projection_hours_at_1_worker"]),
    }


def allowed_worker_seconds(config: dict, campaign: str, workers: int) -> float:
    budget = config["budget"]
    allowed_wall = (float(budget["cpu_cap_hours"])
                    - float(budget["already_charged_cpu_hours"])
                    - serial_hours_estimate(config, campaign))
    return max(0.0, allowed_wall) * 3600.0 * workers


# --------------------------------------------------------------------------
# shared numerical machinery (E1a exact model, E1b linearised waveform)
# --------------------------------------------------------------------------
def ridge_covariance(covariance, relative: float = 1e-6):
    covariance = np.asarray(covariance, dtype=float)
    return (covariance
            + np.eye(len(covariance)) * relative * np.trace(covariance) / len(covariance))


def reference_rows(population):
    """R_c row functionals: span(N_cc^-1 A_c), recomputed from the supplied blocks."""
    return np.linalg.solve(population["ncc"], population["ac"]).T


def sample_oracle(scc, stt, sct, population, k: int):
    """Sample conditional oracle from the *exact retained* joint noise transform."""
    from jepa_noise import linear

    ncc, ntt, nct = population["ncc"], population["ntt"], population["nct"]
    joint_noise = np.block([[ncc, nct], [nct.T, ntt]])
    transform = np.linalg.solve(np.linalg.cholesky(joint_noise),
                                np.eye(len(joint_noise)))
    joint_sample = np.block([[scc, sct], [sct.T, stt]])
    whitened = transform @ joint_sample @ transform.T
    dc = len(ncc)
    result = linear.cca(whitened[:dc, :dc], whitened[dc:, dc:],
                        whitened[:dc, dc:], k)
    return result.context @ transform[:dc, :dc], result


def population_oracle(population, k: int) -> dict:
    from jepa_noise import linear

    return linear.oracle(
        population["scc"], population["stt"], population["sct"],
        population["ncc"], population["ntt"], population["nct"],
        population["ac"], population["at"], k,
    )


def population_diagnostics(population, k: int, theta_sd=None,
                           displacement_info=None) -> dict:
    """Population ranks, canonical margins, oracle eligibility and A4 D.

    ``views.displacement`` is the committed A4.5 implementation; E1b passes its
    output here instead of duplicating the population formula.
    """
    from jepa_noise import linear, metrics

    population = dict(population)
    if theta_sd is not None:
        population["sigma_theta"] = np.diag(np.asarray(theta_sd, dtype=float) ** 2)
    reference = reference_rows(population)
    out = {"gamma_n": float(population["gamma_n"]),
           "reference_sha256": array_digest(reference)}
    cca_fit = linear.cca(population["scc"], population["stt"],
                         population["sct"], k)
    cca_angles = metrics.principal_angles(cca_fit.context, reference, degrees=True)
    out["cca_angles_deg"] = [float(x) for x in cca_angles]
    out["cca_max_angle_deg"] = float(cca_angles.max())
    spectrum_payload = _spectrum_payload(cca_fit.spectrum, k)
    if "spectrum" in spectrum_payload:
        out["cca_spectrum"] = spectrum_payload["spectrum"]
    else:
        out["cca_spectrum_top"] = spectrum_payload["spectrum_top"]
        out["cca_spectrum_length"] = spectrum_payload["spectrum_length"]
        out["cca_spectrum_kth"] = spectrum_payload["spectrum_kth"]
        out["cca_spectrum_gap"] = spectrum_payload["spectrum_gap"]
    gevp_fit = linear.gevp(population["scc"], population["ncc"], k)
    gevp_angles = metrics.principal_angles(gevp_fit.context, reference, degrees=True)
    out["gevp_angles_deg"] = [float(x) for x in gevp_angles]
    out["gevp_max_angle_deg"] = float(gevp_angles.max())
    beta10 = linear.finite_beta(population["scc"], population["stt"],
                                population["sct"], population["ncc"], k, 10.0)
    out["finite_beta_10_max_angle_deg"] = float(
        metrics.principal_angles(beta10.context, reference, degrees=True).max())
    oracle = population_oracle(population, k)
    out["oracle"] = {
        "status": oracle["status"],
        "context_ratio": float(oracle["context_ratio"]),
        "residual_target_ratio": float(oracle["residual_target_ratio"]),
        "kth_correlation": float(oracle["kth_correlation"]),
        "correlation_gap": float(oracle["correlation_gap"]),
        "recovery_max_angle_rad": (
            float(metrics.principal_angles(oracle["context"], reference).max())
            if oracle["status"] == "eligible" else None),
    }
    if displacement_info is not None:
        out["displacement_deg"] = float(displacement_info["angle_deg"])
        out["displacement_angles_deg"] = [float(x)
                                          for x in displacement_info["angles_deg"]]
        out["displacement_canonical_correlations"] = [
            float(x) for x in displacement_info["canonical_correlations"]]
    return out


def physics_crlb(population):
    from jepa_noise import crlb

    try:
        fisher = crlb.fisher(population["ac"], population["ncc"])
    except (ValueError, np.linalg.LinAlgError) as error:
        return {"status": "not_evaluable", "reason": str(error),
                "fisher": None, "crlb": None, "crlb_diagonal": None}
    covariance = np.linalg.inv(fisher)
    return {"status": "evaluated", "fisher": fisher, "crlb": covariance,
            "crlb_diagonal": np.diag(covariance)}


def of_estimator(population, sample_context, theta_true, crlb_diagonal):
    """Context OF/GLS physics baseline (A3 row 10); no fitted probe, no target."""
    from jepa_noise import metrics

    selection = np.linalg.solve(population["ncc"], population["ac"])
    covariance = np.linalg.inv(population["ac"].T @ selection)
    theta_hat = (np.asarray(sample_context, dtype=float) @ selection @ covariance)
    errors = theta_hat - np.asarray(theta_true, dtype=float)
    mse = np.mean(errors ** 2, axis=0)
    r2 = metrics.nuisance_r2(np.asarray(theta_true, dtype=float), theta_hat)["per_target"]
    return {
        "status": "evaluated",
        "mse": [float(x) for x in mse],
        "mse_over_crlb": [float(mse[i] / crlb_diagonal[i]) for i in range(len(mse))],
        "r2": [float(x) for x in r2],
        "fisher": population["ac"].T @ selection,
        "crlb_diagonal": [float(x) for x in crlb_diagonal],
    }


def fit_probe_block(feature_train, feature_test, target_train, target_test, groups,
                    physics_count: int, nuisance_names, crlb_diagonal,
                    fit_intercept: bool = True) -> dict:
    """Five-fold grouped ridge probes on frozen embeddings (A3 Sec.2, runbook 1.3)."""
    from jepa_noise import probes

    target_train = np.asarray(target_train, dtype=float)
    target_test = np.asarray(target_test, dtype=float)
    if not len(target_train) or not len(target_test):
        raise ValueError("probe cohorts must be nonempty")
    fitted = probes.grouped_ridge(feature_train, target_train, groups,
                                  fit_intercept=bool(fit_intercept))
    evaluated = probes.evaluate(fitted, feature_test, target_test)
    physics, nuisance = [], []
    for index, (fit_row, eval_row) in enumerate(zip(fitted, evaluated)):
        entry = {"status": eval_row["status"], "reason": eval_row["reason"],
                 "alpha": fit_row["alpha"], "mse": eval_row["mse"],
                 "r2": eval_row["r2"]}
        if index < physics_count:
            entry["mse_over_crlb"] = (
                float(eval_row["mse"] / crlb_diagonal[index])
                if eval_row["status"] == "evaluated" and crlb_diagonal is not None
                else None)
            physics.append(entry)
        else:
            entry["target"] = nuisance_names[index - physics_count]
            nuisance.append(entry)
    valid_nuisance = [entry["r2"] for entry in nuisance
                      if entry["status"] == "evaluated"]
    return {
        "physics": physics,
        "nuisance": nuisance,
        "nuisance_valid_count": len(valid_nuisance),
        "nuisance_mean_r2": (float(np.mean(valid_nuisance))
                             if valid_nuisance else None),
        "nuisance_status": ("evaluated" if valid_nuisance else "not_evaluable"),
    }


SPECTRUM_FULL_LIMIT = 64


def _spectrum_payload(spectrum, k: int) -> dict:
    """Full spectrum for small problems; top-k + margins for E1b's 1024-dim views.

    E1a spectra have d=40 and are stored in full.  E1b canonical spectra have up
    to 1024 entries; only the physically used top-k values and the registered
    margins are retained, with the length, so a row file stays reviewable and
    inside the v4 Sec.7 size limit.  The full spectrum is reproducible from the
    stored code SHA/config and the recorded event namespaces.
    """
    values = [float(x) for x in spectrum]
    if len(values) <= SPECTRUM_FULL_LIMIT:
        return {"spectrum": values}
    return {
        "spectrum_top": values[:8],
        "spectrum_length": len(values),
        "spectrum_kth": values[k - 1] if k <= len(values) else None,
        "spectrum_gap": ((values[k - 1] - values[k]) if k < len(values) else None),
        "spectrum_max": values[0] if values else None,
        "spectrum_min": values[-1] if values else None,
    }


def _angle_summary(row_functionals, reference, k: int) -> dict:
    from jepa_noise import metrics

    angles = metrics.principal_angles(row_functionals, reference, degrees=True)
    return {"angles_deg": [float(x) for x in angles],
            "max_angle_deg": float(angles.max())}


def _solver_attempt(function, *args, **kwargs) -> tuple[object | None, str | None]:
    """Isolate one registered solver; a numerical failure is recorded, not fixed."""
    try:
        return function(*args, **kwargs), None
    except Exception as error:  # noqa: BLE001 - recorded per registered solver
        return None, f"{type(error).__name__}: {error}"


def solve_registered_methods(scc_u, stt_u, sct_u, scc_r, tt_r, ct_r, ncc_cal,
                             population, config, k: int) -> dict:
    """Compute the registered representation rows from supplied covariances.

    Each solver is attempted in isolation so a fixed-tolerance numerical
    failure in one registered row is recorded as that row's failure instead of
    erasing the other registered quantities for the dataset.
    """
    from jepa_noise import linear

    methods: dict[str, dict] = {}
    fits: dict[str, tuple] = {}
    fits["r1_cca"] = _solver_attempt(linear.cca, scc_r, tt_r, ct_r, k)
    fits["r3_context_pca"] = _solver_attempt(linear.pca, scc_u, k)
    fits["r4_whitened_pca"] = _solver_attempt(linear.pca, scc_u, k,
                                              population["ncc"])
    fits["r5_gevp"] = _solver_attempt(linear.gevp, scc_u, ncc_cal, k)
    for beta, identifier in ((0.1, "r6_finite_beta_0.1"),
                             (1.0, "r7_finite_beta_1"),
                             (10.0, "r8_finite_beta_10")):
        fits[identifier] = _solver_attempt(
            linear.finite_beta, scc_u, stt_u, sct_u, ncc_cal, k, beta)
    metadata = {
        "r1_cca": {"covariance": "subject_sample_ridge"},
        "r2_hard_isotropy": {
            "covariance": "subject_sample_ridge",
            "equivalent_to": "r1_cca",
            "note": ("not an independent estimator: the reviewed solver maps "
                     "beta=0 hard-isotropy prediction and CCA to the same "
                     "solution; counts as one estimator only")},
        "r3_context_pca": {"covariance": "subject_sample_unregularized"},
        "r4_whitened_pca": {"covariance": "subject_sample_unregularized",
                            "noise": "exact_population_context"},
        "r5_gevp": {"covariance": "subject_sample_unregularized",
                    "noise": "independent_sample_calibration"},
        "r6_finite_beta_0.1": {"covariance": "subject_sample_unregularized",
                               "noise": "independent_sample_calibration",
                               "beta": 0.1},
        "r7_finite_beta_1": {"covariance": "subject_sample_unregularized",
                             "noise": "independent_sample_calibration",
                             "beta": 1.0},
        "r8_finite_beta_10": {"covariance": "subject_sample_unregularized",
                              "noise": "independent_sample_calibration",
                              "beta": 10.0},
    }
    for identifier, (fit, error) in fits.items():
        entry = dict(metadata[identifier])
        entry["fit"] = fit
        if error is not None:
            entry["error"] = error
        methods[identifier] = entry
    methods["r2_hard_isotropy"] = dict(metadata["r2_hard_isotropy"])
    methods["r2_hard_isotropy"]["fit"] = fits["r1_cca"][0]
    if fits["r1_cca"][1] is not None:
        methods["r2_hard_isotropy"]["error"] = fits["r1_cca"][1]
    try:
        oracle_population = population_oracle(population, k)
    except Exception as error:  # noqa: BLE001 - recorded as oracle failure
        methods["r9_oracle"] = {"population_status": "oracle_exception",
                                "fit": None,
                                "error": f"{type(error).__name__}: {error}"}
    else:
        entry = {
            "population_status": oracle_population["status"],
            "context_ratio": float(oracle_population["context_ratio"]),
            "residual_target_ratio": float(oracle_population["residual_target_ratio"]),
            "kth_correlation": float(oracle_population["kth_correlation"]),
            "correlation_gap": float(oracle_population["correlation_gap"]),
            "fit": None,
        }
        if oracle_population["status"] == "eligible":
            oracle_result, oracle_error = _solver_attempt(
                sample_oracle, scc_u, stt_u, sct_u, population, k)
            if oracle_result is not None:
                context, fit = oracle_result
                entry.update(fit=fit, context_raw=context)
            else:
                entry["error"] = oracle_error
        methods["r9_oracle"] = entry
    methods["r10_of"] = {"covariance": "exact_population_context"}
    return methods


def method_record(method_id: str, entry: dict, reference, k: int) -> dict:
    record = {"method": method_id}
    fit = entry.get("fit")
    if entry.get("error") is not None:
        record.update(status="fail",
                      reason=f"solver_exception: {entry['error']}",
                      spectrum=None, angles=None)
        for key in ("population_status", "context_ratio", "residual_target_ratio",
                    "kth_correlation", "correlation_gap", "equivalent_to",
                    "beta", "covariance", "noise", "note"):
            if key in entry:
                record[key] = entry[key]
        return record
    if fit is None:
        record.update(status="not_evaluable",
                      reason=entry.get("population_status", "fit_unavailable"),
                      spectrum=None, angles=None)
        for key in ("population_status", "context_ratio", "residual_target_ratio",
                    "kth_correlation", "correlation_gap", "equivalent_to",
                    "beta", "covariance", "noise", "note"):
            if key in entry:
                record[key] = entry[key]
        return record
    record["status"] = "evaluated"
    record.update(_spectrum_payload(fit.spectrum, k))
    row_functionals = entry.get("context_raw")
    if row_functionals is None:
        row_functionals = fit.context
    record["angles"] = _angle_summary(row_functionals, reference, k)
    diagnostics = fit.diagnostics or {}
    for key in ("objective", "reduced_objective", "eigen_residual",
                "isotropy_residual", "target_cross_rank", "boundary_gap"):
        if key in diagnostics and isinstance(diagnostics[key], (int, float, str)):
            record[key] = diagnostics[key]
    if isinstance(diagnostics.get("status"), str):
        record["solver_status"] = diagnostics["status"]
    for key in ("equivalent_to", "beta", "covariance", "noise", "note"):
        if key in entry:
            record[key] = entry[key]
    return record


# --------------------------------------------------------------------------
# E1a executor
# --------------------------------------------------------------------------
def run_e1a_row(config: dict, row: dict, log) -> dict:
    from jepa_noise import linear
    from jepa_noise.masks import namespace_rng
    from jepa_noise.views import ExactModel

    spec = config["E1a"]
    cell = row["cell"]
    seed = row["seed"]
    d, k = int(spec["d"]), int(spec["k"])
    theta_scale, eta_scale = float(cell["theta_scale"]), float(cell["eta_scale"])
    gamma, n = float(cell["gamma_load"]), int(cell["n"])
    cell_label = e1a_cell_label(cell)
    probe = spec["probe"]
    probe_n = int(probe["train_events"])
    groups = np.arange(probe_n) // int(probe["fold_size"])

    model = ExactModel(d=d, k=k, seed=seed, theta_scale=theta_scale,
                       eta_scale=eta_scale, share=spec["share"])
    # A3 Sec.1: loading geometry is fixed by the evaluation seed; the subject
    # draw namespace includes n.  Replacing only the sample RNG keeps the
    # reviewed ExactModel draw law while giving explicit independent roles.
    model.rng = namespace_rng(seed, "E1a_subject", cell=f"{cell_label}/n{n}")
    sample = model.sample(n, gamma)
    population = sample["population"]
    context, target = sample["context"], sample["target"]
    reference = reference_rows(population)

    calibration_rng = namespace_rng(seed, "E1a_noise_cal", cell=f"{cell_label}/n{n}")
    calibration = (calibration_rng.normal(size=(n, 1)) * eta_scale @ model.bc.T
                   + calibration_rng.normal(size=(n, d)) @ model.chol.T)
    centred = calibration - calibration.mean(0)
    ncc_cal = centred.T @ centred / (n - 1)

    scc_u, stt_u, sct_u = linear.sample_covariances(context, target,
                                                    ridge_relative=0.0)
    scc_r, tt_r, ct_r = linear.sample_covariances(context, target,
                                                  ridge_relative=1e-6)

    model.rng = namespace_rng(seed, "E1a_probe_train", cell=cell_label)
    train = model.sample(probe_n, gamma)
    model.rng = namespace_rng(seed, "E1a_probe_test", cell=cell_label)
    test = model.sample(int(probe["test_events"]), gamma)
    targets_train = np.c_[train["theta"], train["eta"]]
    targets_test = np.c_[test["theta"], test["eta"]]
    nuisance_names = ["eta"]

    crlb_info = physics_crlb(population)
    crlb_diagonal = crlb_info["crlb_diagonal"]
    methods = solve_registered_methods(
        scc_u, stt_u, sct_u, scc_r, tt_r, ct_r, ncc_cal, population, spec, k)
    method_records = {}
    for method_id, entry in methods.items():
        if method_id == "r10_of":
            record = {"method": method_id,
                      "covariance": "exact_population_context"}
            if crlb_info["status"] != "evaluated":
                record.update(status="not_evaluable", reason="fisher_rank_ineligible")
            else:
                record.update(status="evaluated",
                              of=of_estimator(population, test["context"],
                                              test["theta"], crlb_diagonal))
            method_records[method_id] = record
            continue
        fit = entry.get("fit")
        record = method_record(method_id, entry, reference, k)
        if fit is None:
            record["probes"] = {"status": "not_evaluable",
                                "reason": "representation_unavailable"}
        else:
            try:
                context_map = entry.get("context_raw")
                if context_map is None:
                    context_map = fit.context
                feature_train = train["context"] @ np.asarray(context_map).T
                feature_test = test["context"] @ np.asarray(context_map).T
                record["probes"] = fit_probe_block(
                    feature_train, feature_test, targets_train, targets_test,
                    groups, 3, nuisance_names, crlb_diagonal,
                    fit_intercept=bool(probe["fit_intercept"]))
            except Exception as error:  # noqa: BLE001 - recorded per method
                record["probes"] = {
                    "status": "fail",
                    "reason": f"probe_exception:{type(error).__name__}: {error}"}
        method_records[method_id] = record

    try:
        diagnostics = population_diagnostics(population, k)
    except Exception as error:  # noqa: BLE001 - recorded, never repaired
        diagnostics = {"status": "fail",
                       "reason": f"population_diagnostic_exception:"
                                 f"{type(error).__name__}: {error}"}
    diagnostics["crlb"] = {"status": crlb_info["status"],
                           "diagonal": (None if crlb_diagonal is None else
                                        [float(x) for x in crlb_diagonal])}

    r3 = None
    if n == int(spec["angle_n"]):
        try:
            transform_spec = spec.get("r3_transform",
                                      {"namespace": "E1a_R3_transform",
                                       "offset": 5.0, "scale": 1.0})
            transform_rng = namespace_rng(seed, transform_spec["namespace"],
                                          cell=cell_label)
            offset = float(transform_spec.get("offset", 5.0))
            scale = float(transform_spec.get("scale", 1.0))
            t_context = (scale * transform_rng.normal(size=(d, d))
                         + offset * np.eye(d))
            t_target = (scale * transform_rng.normal(size=(d, d))
                        + offset * np.eye(d))
            base = linear.cca(scc_u, stt_u, sct_u, k)
            transformed = linear.cca(t_context @ scc_u @ t_context.T,
                                     t_target @ stt_u @ t_target.T,
                                     t_context @ sct_u @ t_target.T, k)
            r3 = {"status": "evaluated",
                  "max_spectrum_difference": float(
                      np.max(np.abs(base.spectrum - transformed.spectrum))),
                  "base_spectrum": [float(x) for x in base.spectrum],
                  "transformed_spectrum": [float(x)
                                           for x in transformed.spectrum]}
        except Exception as error:  # noqa: BLE001 - diagnostic failure recorded
            r3 = {"status": "fail",
                  "max_spectrum_difference": None,
                  "reason": f"r3_exception:{type(error).__name__}: {error}",
                  "base_spectrum": None, "transformed_spectrum": None}

    ineligible = [f"{method_id}:not_evaluable"
                  for method_id, record in method_records.items()
                  if record["status"] != "evaluated"]
    oracle_diag = (diagnostics.get("oracle") or {})
    if oracle_diag.get("status") != "eligible":
        ineligible.append("population_oracle:not_evaluable")
    if crlb_info["status"] != "evaluated":
        ineligible.append("crlb:not_evaluable")
    status = "not_evaluable" if ineligible else "pass"
    return {
        "campaign": "E1a",
        "row_id": row["row_id"],
        "cell": cell,
        "seed": seed,
        "status": status,
        "reason": ";".join(ineligible) if ineligible else None,
        "population": diagnostics,
        "sample_calibration": {
            "n_cal": n,
            "ncc_sha256": array_digest(ncc_cal),
            "ridge_relative": 0.0,
            "covariance_role": "E1a_noise_cal (n_cal=n_subject, divisor n-1)"},
        "sample_covariances": {
            "scc_ridge_sha256": array_digest(scc_r),
            "scc_unregularized_sha256": array_digest(scc_u),
            "stt_unregularized_sha256": array_digest(stt_u),
            "sct_unregularized_sha256": array_digest(sct_u)},
        "probe": {
            "train_events": probe_n,
            "test_events": int(probe["test_events"]),
            "groups": int(probe["folds"]),
            "fold_size": int(probe["fold_size"]),
            "fit_intercept": bool(probe["fit_intercept"]),
            "targets": "theta(3), eta(1)"},
        "methods": method_records,
        "r3_per_view_invariance": r3,
        "namespace_roles": {
            "subject": "E1a_subject",
            "probe_train": "E1a_probe_train",
            "probe_test": "E1a_probe_test",
            "noise_cal": "E1a_noise_cal"},
    }


# --------------------------------------------------------------------------
# E1b law gate, population preflight and executor (A4 only; never the v3 law)
# --------------------------------------------------------------------------
def supported_laws() -> list[str]:
    import jepa_noise.views as views_module

    laws = getattr(views_module, "SUPPORTED_LAWS", None)
    if laws is None:
        law = getattr(views_module, "LAW", None)
        return [] if law is None else [str(law)]
    return [str(item) for item in laws]


def a4_available(config: dict) -> tuple[bool, str]:
    law = config["E1b"]["law"]
    try:
        laws = supported_laws()
    except Exception as error:  # pragma: no cover - import failure path
        return False, f"cannot import jepa_noise.views: {error}"
    if law not in laws:
        return False, (
            f"installed src/jepa_noise/views.py advertises laws {laws!r}, not "
            f"{law!r}; WP2-v4 A4 has not landed. E1b needs base_rho=0, E1b "
            f"theta SDs {config['E1b']['theta_sd']} and the 13-cell grid.")
    return True, "ok"


def make_tier1(config: dict, split: str, common_strength: float,
               line_multiplier: float, seed: int, cell_label: str):
    import jepa_noise.views as views_module

    law = config["E1b"]["law"]
    available, detail = a4_available(config)
    if not available:
        raise CampaignBlocked(
            f"configured law {law!r} is not supported by the installed "
            f"src/jepa_noise/views.py: {detail}")
    kwargs = dict(signal_mode="linearized", split=split,
                  line_multiplier=float(line_multiplier),
                  common_strength=float(common_strength),
                  seed=int(seed), cell=str(cell_label))
    try:
        view = views_module.Tier1Views(**kwargs, law=law)
    except TypeError as error:
        raise CampaignBlocked(
            f"Tier1Views does not accept law={law!r} ({error}); the A4 API is "
            f"not installed, so E1b must not fall back to the v3 law") from error
    manifest = view.manifest()
    if manifest.get("law") != law:
        raise CampaignBlocked(
            f"Tier1Views manifest law {manifest.get('law')!r} != configured "
            f"{law!r}; refusing to run E1b under an unregistered noise law")
    if not np.allclose(np.asarray(view.theta_sd, dtype=float),
                       np.asarray(config["E1b"]["theta_sd"], dtype=float),
                       rtol=0.0, atol=1e-7):
        raise CampaignBlocked(
            "Tier1Views theta SDs do not match A4 "
            f"{config['E1b']['theta_sd']}: got {np.asarray(view.theta_sd).tolist()}")
    if hasattr(view, "base_rho") and abs(float(view.base_rho)) > 0.0:
        raise CampaignBlocked(
            f"A4 requires base_rho=0 (channel-private base); got {view.base_rho}")
    return view


def e1b_population_preflight(config: dict, attempt: Path, log) -> dict:
    """Population ranks/margins/D before any sample grid (WP3 line 219, A4 Sec.5)."""
    from jepa_noise.masks import raw_split

    spec = config["E1b"]
    k = int(spec["k"])
    record = {"cells": {}, "displacement_checks": {}, "status": "running",
              "manifest_clean_cell_channel": None}
    for cell in spec["cells"]:
        cell_record = {}
        for split in spec["splits"]:
            view = make_tier1(config, split, cell["common_strength"],
                              cell["line_multiplier"], 0,
                              f"E1b_population/{cell['cell_id']}")
            if cell["cell_id"] == "g0_line0" and split == "channel":
                record["manifest_clean_cell_channel"] = view.manifest()
            population = dict(view.population())
            population["sigma_theta"] = np.diag(
                np.asarray(spec["theta_sd"], dtype=float) ** 2)
            diagnostics = population_diagnostics(
                population, k, spec["theta_sd"],
                displacement_info=view.displacement(k))
            diagnostics["context_dim"] = int(len(raw_split(split).context))
            diagnostics["target_dim"] = int(len(raw_split(split).target))
            diagnostics["unused_dim"] = int(len(raw_split(split).unused))
            cell_record[split] = diagnostics
        record["cells"][cell["cell_id"]] = cell_record
        expected = spec["expected_displacement_deg"].get("channel", {}).get(
            cell["cell_id"])
        if expected is not None:
            observed = cell_record["channel"]["displacement_deg"]
            delta = abs(observed - expected)
            record["displacement_checks"][cell["cell_id"]] = {
                "expected_deg": expected, "observed_deg": observed,
                "difference_deg": delta,
                "pass": delta <= float(spec["displacement_mismatch_tolerance_deg"])}
        log(f"population preflight {cell['cell_id']} "
            f"channel D={cell_record['channel'].get('displacement_deg'):.3f} deg")
    mismatches = [cell_id for cell_id, check in record["displacement_checks"].items()
                  if not check["pass"]]
    record["mismatches"] = mismatches
    record["status"] = "pass" if not mismatches else "population_mismatch_gt_1deg"
    atomic_write_json(attempt / "population_diagnostics.json", record)
    return record


def _accumulate_context(accumulator, context_indices, values):
    block = values[:, context_indices]
    accumulator["sum"] += block.sum(axis=0)
    accumulator["xx"] += block.T @ block


def _accumulate_joint(accumulator, context_indices, target_indices, values):
    context = values[:, context_indices]
    target = values[:, target_indices]
    accumulator["sum_c"] += context.sum(axis=0)
    accumulator["sum_t"] += target.sum(axis=0)
    accumulator["xx_c"] += context.T @ context
    accumulator["xx_t"] += target.T @ target
    accumulator["ct"] += context.T @ target


def _finalize_context(accumulator, count: int):
    count = int(count)
    xx = np.asarray(accumulator["xx"], dtype=float)
    total = np.asarray(accumulator["sum"], dtype=float)
    return (xx - np.outer(total, total) / count) / (count - 1)


def _finalize_joint(accumulator, count: int):
    count = int(count)
    sum_c = np.asarray(accumulator["sum_c"], dtype=float)
    sum_t = np.asarray(accumulator["sum_t"], dtype=float)
    xx_c = np.asarray(accumulator["xx_c"], dtype=float)
    xx_t = np.asarray(accumulator["xx_t"], dtype=float)
    cross = np.asarray(accumulator["ct"], dtype=float)
    scc = (xx_c - np.outer(sum_c, sum_c) / count) / (count - 1)
    stt = (xx_t - np.outer(sum_t, sum_t) / count) / (count - 1)
    sct = (cross - np.outer(sum_c, sum_t) / count) / (count - 1)
    return scc, stt, sct


def _stream_e1b_covariances(config: dict, generator, indices, n: int, chunk: int, log):
    """Stream subject and calibration data in <=chunk-event float64 blocks."""
    splits = list(config["E1b"]["splits"])
    joint = {split: {"sum_c": np.zeros(len(indices[split].context)),
                     "sum_t": np.zeros(len(indices[split].target)),
                     "xx_c": np.zeros((len(indices[split].context),) * 2),
                     "xx_t": np.zeros((len(indices[split].target),) * 2),
                     "ct": np.zeros((len(indices[split].context),
                                     len(indices[split].target)))}
             for split in splits}
    calibration = {split: {"sum": np.zeros(len(indices[split].context)),
                           "xx": np.zeros((len(indices[split].context),) * 2)}
                   for split in splits}
    started = time.monotonic()
    for start in range(0, n, chunk):
        event_ids = np.arange(start, min(start + chunk, n), dtype=np.int64)
        raw = generator.sample(event_ids, role="E1b_subject")["raw"]
        values = raw.reshape(len(event_ids), 2048)
        noise = np.empty((len(event_ids), 8, 256))
        for offset, event_id in enumerate(event_ids):
            noise[offset] = generator.noise_record(
                event_id, role="E1b_noise_cal")[0]
        noise_values = noise.reshape(len(event_ids), 2048)
        for split in splits:
            _accumulate_joint(joint[split], indices[split].context,
                              indices[split].target, values)
            _accumulate_context(calibration[split], indices[split].context,
                                noise_values)
        del raw, values, noise, noise_values
    log(f"streamed {n} subject events and {n} calibration noise records "
        f"in {time.monotonic() - started:.1f}s")
    subject = {}
    calibration_cov = {}
    for split in splits:
        scc, stt, sct = _finalize_joint(joint[split], n)
        subject[split] = {"scc": scc, "stt": stt, "sct": sct}
        calibration_cov[split] = _finalize_context(calibration[split], n)
    return subject, calibration_cov


def run_e1b_row(config: dict, row: dict, log) -> dict:
    from jepa_noise import linear
    from jepa_noise.masks import raw_split

    spec = config["E1b"]
    cell = row["cell"]
    seed = row["seed"]
    k = int(spec["k"])
    n = int(spec["n"])
    chunk = int(spec["chunk_events"])
    splits = list(spec["splits"])
    primary = spec["primary_split"]
    common_strength = float(cell["common_strength"])
    line_multiplier = float(cell["line_multiplier"])
    cell_label = f"E1b/{cell['cell_id']}_s{seed}"
    indices = {split: raw_split(split) for split in splits}
    probe = spec["probe"]

    generator = make_tier1(config, primary, common_strength, line_multiplier,
                           seed, cell_label)
    populations = {}
    population_records = {}
    split_manifests = {}
    for split in splits:
        view = make_tier1(config, split, common_strength, line_multiplier,
                          seed, cell_label)
        split_manifests[split] = view.manifest()
        population = dict(view.population())
        population["sigma_theta"] = np.diag(
            np.asarray(spec["theta_sd"], dtype=float) ** 2)
        populations[split] = population
        population_records[split] = population_diagnostics(
            population, k, spec["theta_sd"],
            displacement_info=view.displacement(k))
    log(f"population diagnostics done for {len(splits)} splits")

    subject, calibration_cov = _stream_e1b_covariances(
        config, generator, indices, n, chunk, log)

    train = generator.sample(np.arange(int(probe["train_events"]), dtype=np.int64),
                             role="E1b_probe_train")
    test = generator.sample(np.arange(int(probe["test_events"]), dtype=np.int64),
                            role="E1b_probe_test")
    nuisance_valid = np.asarray(train["nuisance_valid"], dtype=bool)
    nuisance_names = [name for name, valid in zip(train["nuisance_names"],
                                                  nuisance_valid) if valid]
    nuisance_index = np.flatnonzero(nuisance_valid)
    targets_train = np.c_[train["theta"], train["nuisance"][:, nuisance_index]]
    targets_test = np.c_[test["theta"], test["nuisance"][:, nuisance_index]]
    groups = np.arange(int(probe["train_events"])) // int(probe["fold_size"])
    probe_flat_train = train["raw"].reshape(len(train["raw"]), 2048)
    probe_flat_test = test["raw"].reshape(len(test["raw"]), 2048)

    split_records = {}
    for split in splits:
        population = populations[split]
        reference = reference_rows(population)
        scc_u, stt_u, sct_u = (subject[split]["scc"], subject[split]["stt"],
                               subject[split]["sct"])
        scc_r = ridge_covariance(scc_u)
        tt_r = ridge_covariance(stt_u)
        ncc_cal = calibration_cov[split]
        crlb_info = physics_crlb(population)
        crlb_diagonal = crlb_info["crlb_diagonal"]
        methods = solve_registered_methods(
            scc_u, stt_u, sct_u, scc_r, tt_r, sct_u, ncc_cal, population,
            spec, k)
        method_records = {}
        for method_id, entry in methods.items():
            if method_id == "r10_of":
                record = {"method": method_id,
                          "covariance": "exact_population_context"}
                if crlb_info["status"] != "evaluated":
                    record.update(status="not_evaluable",
                                  reason="fisher_rank_ineligible")
                else:
                    record.update(status="evaluated",
                                  of=of_estimator(
                                      population,
                                      probe_flat_test[:, indices[split].context],
                                      test["theta"], crlb_diagonal))
                method_records[method_id] = record
                continue
            fit = entry.get("fit")
            record = method_record(method_id, entry, reference, k)
            if fit is None:
                record["probes"] = {"status": "not_evaluable",
                                    "reason": "representation_unavailable"}
            else:
                try:
                    context_map = entry.get("context_raw")
                    if context_map is None:
                        context_map = fit.context
                    features_train = (
                        probe_flat_train[:, indices[split].context]
                        @ np.asarray(context_map).T)
                    features_test = (
                        probe_flat_test[:, indices[split].context]
                        @ np.asarray(context_map).T)
                    record["probes"] = fit_probe_block(
                        features_train, features_test, targets_train,
                        targets_test, groups, 3, nuisance_names, crlb_diagonal,
                        fit_intercept=bool(probe["fit_intercept"]))
                except Exception as error:  # noqa: BLE001 - recorded per method
                    record["probes"] = {
                        "status": "fail",
                        "reason": f"probe_exception:{type(error).__name__}: {error}"}
            method_records[method_id] = record
        split_records[split] = {
            "role": ("primary" if split == primary else
                     ("alternative" if split == spec["alternative_split"]
                      else "diagnostic")),
            "n_events": n,
            "n_calibration": n,
            "context_dim": int(len(indices[split].context)),
            "target_dim": int(len(indices[split].target)),
            "unused_dim": int(len(indices[split].unused)),
            "determinant_flags": {
                "context_dim_at_least_k": bool(len(indices[split].context) >= k),
                "target_rank_diagnostic": population_records[split]["oracle"]
                ["residual_target_ratio"]},
            "population": population_records[split],
            "sample_calibration": {
                "ncc_sha256": array_digest(ncc_cal),
                "ridge_relative": 0.0,
                "divisor": "n_cal-1",
                "role": "E1b_noise_cal"},
            "sample_covariances": {
                "scc_sha256": array_digest(scc_u),
                "stt_sha256": array_digest(stt_u),
                "sct_sha256": array_digest(sct_u),
                "scc_ridge_sha256": array_digest(scc_r)},
            "crlb": {"status": crlb_info["status"],
                     "diagonal": [float(x) for x in crlb_diagonal]},
            "methods": method_records,
        }
    primary_record = split_records[primary]
    ineligible = []
    if primary_record["population"]["oracle"]["status"] != "eligible":
        ineligible.append("primary_population_oracle:not_evaluable")
    if primary_record["crlb"]["status"] != "evaluated":
        ineligible.append("primary_crlb:not_evaluable")
    status = "not_evaluable" if ineligible else "pass"
    return {
        "campaign": "E1b",
        "row_id": row["row_id"],
        "cell": cell,
        "seed": seed,
        "status": status,
        "reason": ";".join(ineligible) if ineligible else None,
        "law": spec["law"],
        "base_rho": spec["base_rho"],
        "theta_sd": spec["theta_sd"],
        "generator_manifest": generator.manifest(),
        "view_manifests": split_manifests,
        "n": n,
        "chunk_events": chunk,
        "streaming": "subject and calibration sufficient statistics in float64; "
                     "raw n=100000 arrays never held or written",
        "probe": {
            "train_events": int(probe["train_events"]),
            "test_events": int(probe["test_events"]),
            "groups": int(probe["folds"]),
            "fold_size": int(probe["fold_size"]),
            "fit_intercept": bool(probe["fit_intercept"]),
            "physics_targets": "3 delta-theta",
            "nuisance_targets": nuisance_names,
            "nuisance_undefined": [name for name, valid in
                                   zip(train["nuisance_names"], nuisance_valid)
                                   if not valid]},
        "splits": split_records,
    }


def run_m_diagnostic(config: dict, row: dict, log, attempt: Path) -> dict:
    """Criterion M development diagnostic (A3.3 as amended by v4: g=.02, line=0)."""
    from jepa_noise.masks import raw_split

    setting = config["E1b"]["M_diagnostic"]
    common = float(setting["common_strength"])
    line = float(setting["line_multiplier"])
    seed = int(setting["development_seed"])
    k = int(setting["k"])
    candidates = list(setting["candidate_order"])
    generator = make_tier1(config, candidates[0], common, line, seed,
                           "E1b_M_development")
    events = generator.sample(np.arange(int(setting["events"]), dtype=np.int64),
                              role="E1b_M_event")
    flat_events = events["raw"].reshape(len(events["raw"]), 2048)
    noise = np.empty((int(setting["noise_records"]), 8, 256))
    for offset in range(int(setting["noise_records"])):
        noise[offset] = generator.noise_record(
            offset, role="E1b_M_noise")[0]
    flat_noise = noise.reshape(len(noise), 2048)
    projection_dir = attempt / "M_diagnostic"
    projection_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    details = {}
    projection_hashes = {}
    for split in candidates:
        indices = raw_split(split)
        dc = len(indices.context)
        dt = len(indices.target)
        dimension = min(int(setting["projection_dim"]), dc, dt)
        rng = np.random.default_rng(
            np.random.SeedSequence(_seed_entropy(setting["projection_seed"], split)))
        qc, _ = np.linalg.qr(rng.normal(size=(dc, dimension)))
        qt, _ = np.linalg.qr(rng.normal(size=(dt, dimension)))
        event_context = flat_events[:, indices.context] @ qc
        event_target = flat_events[:, indices.target] @ qt
        noise_context = flat_noise[:, indices.context] @ qc
        noise_target = flat_noise[:, indices.target] @ qt
        scc = np.cov(event_context, rowvar=False, ddof=1)
        stt = np.cov(event_target, rowvar=False, ddof=1)
        event_joint = np.cov(event_context, event_target, rowvar=False, ddof=1)
        sct = event_joint[:dimension, dimension:]
        ncc = np.cov(noise_context, rowvar=False, ddof=1)
        ntt = np.cov(noise_target, rowvar=False, ddof=1)
        noise_joint = np.cov(noise_context, noise_target, rowvar=False, ddof=1)
        nct = noise_joint[:dimension, dimension:]
        def eps(matrix):
            return 1e-3 * np.trace(matrix) / len(matrix) * np.eye(len(matrix))
        h_signal = (_invsqrt(scc + eps(scc))
                    @ (sct - nct)
                    @ _invsqrt(stt + eps(stt)))
        h_noise = (_invsqrt(ncc + eps(ncc))
                   @ nct
                   @ _invsqrt(ntt + eps(ntt)))
        signal_singular = np.linalg.svd(h_signal, compute_uv=False)
        noise_singular = np.linalg.svd(h_noise, compute_uv=False)
        score = float(signal_singular[k - 1] - noise_singular[0])
        scores[split] = score
        projection_hashes[split] = {
            "context": array_digest(qc), "target": array_digest(qt)}
        np.save(projection_dir / f"projection_{split}_context.npy", qc)
        np.save(projection_dir / f"projection_{split}_target.npy", qt)
        details[split] = {
            "score": score,
            "signal_singular_values": [float(x) for x in signal_singular],
            "noise_singular_values": [float(x) for x in noise_singular],
            "kth_signal_singular_value": float(signal_singular[k - 1]),
            "top_noise_singular_value": float(noise_singular[0]),
            "projection_sha256": projection_hashes[split],
            "projection_dimension": dimension,
        }
        log(f"M candidate {split}: score={score:.6f}")
    ordered = sorted(candidates, key=lambda split: (-scores[split],
                                                    candidates.index(split)))
    selected = ordered[0]
    tied = [split for split in candidates
            if abs(scores[split] - scores[selected]) <= float(setting["tie_tolerance"])]
    selected = min(tied, key=candidates.index)
    return {
        "campaign": "E1b",
        "row_id": row["row_id"],
        "kind": "M_diagnostic",
        "status": "pass",
        "reason": None,
        "setting": {"common_strength": common, "line_multiplier": line,
                    "development_seed": seed, "law": config["E1b"]["law"],
                    "k": k, "events": int(setting["events"]),
                    "noise_records": int(setting["noise_records"]),
                    "candidate_order": candidates,
                    "projection_seed": int(setting["projection_seed"]),
                    "projection_dim": int(setting["projection_dim"]),
                    "regularization": setting["regularization"],
                    "annotation_only": True},
        "scores": scores,
        "details": details,
        "selected_split": selected,
        "score_order": ordered,
        "projection_sha256": projection_hashes,
        "projection_directory": str(projection_dir),
    }


def _seed_entropy(seed: int, label: str) -> list[int]:
    digest = hashlib.sha256(json.dumps([int(seed), str(label)]).encode()).digest()
    return np.frombuffer(digest, dtype="<u4").tolist()


def _invsqrt(matrix):
    from jepa_noise.linear import invsqrt

    return invsqrt(matrix)


# --------------------------------------------------------------------------
# E1c executor (dynamics.py)
# --------------------------------------------------------------------------
def run_e1c_row(config: dict, row: dict, log) -> dict:
    from jepa_noise import dynamics

    spec = config["E1c"]
    gamma = float(row["cell"]["gamma_load"])
    seed = int(row["seed"])
    row_budget = float(spec["row_budget_seconds"]) / len(spec["depths"])
    moments, ac, bc, theta_sd_array, eta_sd, provenance = (
        dynamics.prepare_campaign_moments(seed, gamma))
    provenance = dict(provenance)
    for key in ("rng_initial_state", "rng_final_state"):
        if key in provenance:
            provenance[key + "_sha256"] = sha256_bytes(
                json.dumps(provenance.pop(key), sort_keys=True,
                           default=str).encode())
    depth_records = {}
    for depth in spec["depths"]:
        layers, predictor = dynamics.initialize(seed, int(depth), d=int(spec["d"]))
        result = dynamics.run(
            moments, layers, predictor, ac, bc,
            np.asarray(theta_sd_array, dtype=float), float(eta_sd),
            budget_seconds=row_budget)
        stride = int(spec["curve_stride"])
        absolute = np.asarray(result["absolute"], dtype=np.float64)
        normalized = np.asarray(result["normalized_diagnostic"], dtype=np.float64)
        loss = np.asarray(result["loss"], dtype=np.float64)
        indices = np.arange(0, len(absolute), stride)
        if indices[-1] != len(absolute) - 1:
            indices = np.r_[indices, len(absolute) - 1]
        depth_records[str(depth)] = {
            "status": result["status"],
            "reason": result["reason"],
            "completed_steps": int(result["completed_steps"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "R5": result["R5"],
            "learning_rate": float(result["learning_rate"]),
            "loss_reduction": result["loss_reduction"],
            "engineering_only": bool(result["engineering_only"]),
            "curve_index": [int(x) for x in indices],
            "absolute_contribution": [absolute[index].tolist() for index in indices],
            "absolute_contribution_sha256": array_digest(absolute),
            "normalized_diagnostic": [normalized[index].tolist() for index in indices],
            "loss": [float(loss[index]) for index in indices],
            "encoder_norm": [float(result["encoder_norm"][index]) for index in indices],
        }
        log(f"depth {depth}: {result['status']} ({result['reason']}) "
            f"{result['completed_steps']} steps in "
            f"{result['elapsed_seconds']:.2f}s")
    statuses = [entry["status"] for entry in depth_records.values()]
    if any(status == "failed" for status in statuses):
        status, reason = "fail", "numerical_instability"
    elif any(status == "stopped" for status in statuses):
        status, reason = "not_evaluable", "resource_cap"
    elif any(status != "complete" for status in statuses):
        status, reason = "not_evaluable", "incomplete_horizon"
    elif any(entry["R5"]["status"] != "eligible" for entry in depth_records.values()):
        status, reason = "not_evaluable", "mode_learning_ineligible"
    else:
        status, reason = "pass", None
    return {
        "campaign": "E1c",
        "row_id": row["row_id"],
        "cell": row["cell"],
        "seed": seed,
        "status": status,
        "reason": reason,
        "gamma_load": gamma,
        "provenance": provenance,
        "steps": int(spec["steps"]),
        "depths": depth_records,
    }


# --------------------------------------------------------------------------
# row dispatch
# --------------------------------------------------------------------------
def execute_row(task: dict) -> dict:
    """Worker: execute one row, write its write-once record, return the manifest row."""
    root = Path(task["root"])
    attempt = Path(task["attempt_dir"])
    config = task["config"]
    row = task["row"]
    campaign = row["campaign"]
    log = row_logger(attempt, row["row_id"])
    started = time.monotonic()
    try:
        if row.get("kind") == "M_diagnostic":
            record = run_m_diagnostic(config, row, log, attempt)
        elif campaign == "E1a":
            record = run_e1a_row(config, row, log)
        elif campaign == "E1b":
            record = run_e1b_row(config, row, log)
        elif campaign == "E1c":
            record = run_e1c_row(config, row, log)
        else:  # pragma: no cover - guarded by CLI
            raise ValueError(f"unknown campaign {campaign!r}")
    except CampaignBlocked as error:
        record = {"campaign": campaign, "row_id": row["row_id"],
                  "cell": row.get("cell"), "seed": row.get("seed"),
                  "status": "fail", "reason": f"campaign_blocked: {error}"}
    except Exception as error:  # noqa: BLE001 - every failure is recorded
        record = {"campaign": campaign, "row_id": row["row_id"],
                  "cell": row.get("cell"), "seed": row.get("seed"),
                  "status": "fail",
                  "reason": f"exception:{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(limit=8)}
    wall = time.monotonic() - started
    record["wall_seconds"] = round(wall, 3)
    record["peak_rss_mb"] = peak_rss_mb()
    record["host"] = host_name()
    record["worker_pid"] = os.getpid()
    record["attempt"] = attempt.name
    record["code_sha"] = task["code_sha"]
    record["config_sha256"] = task["config_sha256"]
    if record.get("status") not in STATUSES:  # pragma: no cover
        raise ValueError(f"invalid row status {record.get('status')!r}")
    atomic_write_json(attempt / "rows" / f"{row['row_id']}.json", record)
    log(f"row {row['row_id']} {record['status']} in {wall:.1f}s")
    return {
        "run_id": task["run_id"],
        "code_sha": task["code_sha"],
        "config_sha256": task["config_sha256"],
        "campaign": campaign,
        "attempt": attempt.name,
        "row_id": row["row_id"],
        "kind": row.get("kind", "grid"),
        "cell": json.dumps(row.get("cell"), sort_keys=True,
                           separators=(",", ":")),
        "seed": row.get("seed"),
        "host": record["host"],
        "worker_pid": record["worker_pid"],
        "workers": task["workers"],
        "wall_seconds": record["wall_seconds"],
        "peak_rss_mb": record["peak_rss_mb"],
        "status": record["status"],
        "reason": record.get("reason"),
        "end_utc": utcnow(),
    }


# --------------------------------------------------------------------------
# campaign driver
# --------------------------------------------------------------------------
def _write_stop(attempt: Path, reason: str, detail: dict, code: int) -> int:
    atomic_write_json(attempt / "STOP.json", {
        "status": "stopped", "reason": reason, "detail": detail,
        "exit_code": code, "written_utc": utcnow()})
    return code


def apply_rerun_rows(rows: list, completed: set, rerun: list) -> tuple[list, set]:
    """Restrict a campaign to explicitly re-run row ids (author-provenance tool)."""
    if not rerun:
        return rows, completed
    known = {row["row_id"] for row in rows}
    unknown = [identifier for identifier in rerun if identifier not in known]
    if unknown:
        raise CampaignBlocked(
            "unknown --rerun-rows identifier(s): " + ", ".join(unknown)
        )
    wanted = set(rerun)
    return [row for row in rows if row["row_id"] in wanted], (
        set(completed) - wanted
    )


def run_campaign(args) -> int:
    root = Path(args.root).resolve()
    config_path = Path(args.config).resolve()
    config, config_sha = load_config(config_path)
    campaign = args.campaign
    scope = "smoke" if args.smoke else "full"
    # Author rule (2026-10-17): a registered run refuses to start on a dirty
    # src/scripts tree and records the clean SHA.  Smokes and explicit
    # --allow-dirty development runs are marked non-registered.
    src_status = src_tree_status(root)
    allow_dirty = bool(getattr(args, "allow_dirty", False))
    registered = scope == "full" and not allow_dirty
    if registered and src_status:
        print(
            "STOP: a registered run refuses to start: `git status --porcelain -- "
            "src scripts` is not empty.\n" + src_status +
            "\nCommit the source tree first (or pass --allow-dirty for a "
            "non-registered development run).",
            file=sys.stderr,
        )
        return 2
    rows = enumerate_rows(config, campaign, scope)
    directory = campaign_root(root, campaign)
    directory.mkdir(parents=True, exist_ok=True)
    discovered = discover_rows(directory)
    if args.rerun_all:
        completed = set()
    elif args.retry_failed:
        completed = {identifier for identifier, record in discovered.items()
                     if record.get("status") == "pass"}
    else:
        # A stopped attempt writes a ``not_run`` placeholder for every row it did
        # not launch (status contract).  Such a placeholder is NOT a completed
        # row: only rows recorded with a real status count as completed.
        completed = {
            identifier for identifier, record in discovered.items()
            if record.get("status") != "not_run"
        }
    rerun = [value.strip() for value in (getattr(args, "rerun_rows", None) or "").split(",")
             if value.strip()]
    try:
        rows, completed = apply_rerun_rows(rows, completed, rerun)
    except CampaignBlocked as error:
        print(f"STOP: {error}", file=sys.stderr)
        return 2
    pending = [row for row in rows if row["row_id"] not in completed]
    skipped = [row["row_id"] for row in rows if row["row_id"] in completed]
    if args.limit_rows is not None:
        pending = pending[: int(args.limit_rows)]

    # Campaign-level preflight before any attempt directory is created.
    if campaign == "E1b":
        available, detail = a4_available(config)
        if not available:
            print(f"STOP: E1b campaign blocked.\n{detail}\n"
                  f"Refusing to run E1b under an unregistered noise law.",
                  file=sys.stderr)
            attempt = create_attempt(directory, campaign)
            _write_stop_raw(attempt, "campaign_blocked_law_not_supported", detail, 3)
            return 3

    worker_choice = choose_workers(config, args.workers)
    workers = int(worker_choice["workers"])
    attempt = create_attempt(directory, campaign)
    logger = make_logger(attempt / "logs" / "run.log",
                         f"jepa_noise.run.{campaign}.{attempt.name}")
    log = logger.info
    code_sha = git_head(root)
    status_start = git_status(root)
    hashes_start = source_hashes(root, config_path)
    run_id = f"E1-{campaign}-{attempt.name}"
    log(f"run {run_id}: campaign={campaign} scope={scope} workers={workers} "
        f"config_sha256={config_sha} code_sha={code_sha}")
    log(f"worker selection: {worker_choice}")
    log(f"registered rows={len(rows)} expected={expected_row_count(config, campaign, scope)} "
        f"completed_rows_found={len(skipped)} pending={len(pending)}")

    row_seconds = (args.row_seconds_estimate
                   if args.row_seconds_estimate is not None else None)
    projection = project_campaign(config, campaign, len(pending), workers,
                                  row_seconds=row_seconds)
    attempt_record = {
        "run_id": run_id, "campaign": campaign, "scope": scope,
        "config_path": str(config_path), "config_sha256": config_sha,
        "code_sha_start": code_sha, "git_status_start": status_start,
        "src_scripts_status": src_status,
        "registered_run": registered,
        "registered_sha": code_sha if (registered and not src_status) else None,
        "source_sha256_start": hashes_start,
        "host": host_name(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "numpy": np.__version__,
        "environment": {key: os.environ.get(key) for key in
                        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "worker_selection": worker_choice, "workers": workers,
        "worker_rule_deviation": (
            None if workers <= worker_choice["measured_formula"] else
            (f"W={workers} used while the v4 Sec.6 formula gave "
             f"{worker_choice['measured_formula']}; measured per-worker peak RSS "
             f"and minimum host free RAM recorded below; fallback to W=1 if free "
             f"RAM < 3x peak/worker")),
        "resource_guard": {
            "stop_worker_peak_rss_mb": 2560.0,
            "stop_host_free_gb": 3.0,
            "stop_action": "stop launching new rows, record not_run, relaunch at W=1 from completed rows",
            "measured_peak_worker_rss_mb": None,
            "min_host_free_gb_observed": None,
        },
        "registered_rows": len(rows), "expected_rows": expected_row_count(
            config, campaign, scope),
        "skipped_completed_rows": len(skipped),
        "pending_rows": len(pending), "budget_projection": projection,
        "argv": sys.argv[1:], "started_utc": utcnow(), "status": "running",
    }
    atomic_write_json(attempt / "attempt.json", attempt_record)
    if skipped:
        reused = []
        for row in rows:
            if row["row_id"] not in completed:
                continue
            source = discovered.get(row["row_id"], {})
            append_manifest(attempt, {
                "run_id": run_id, "code_sha": code_sha,
                "config_sha256": config_sha, "campaign": campaign,
                "attempt": attempt.name, "row_id": row["row_id"],
                "kind": "grid", "cell": json.dumps(row.get("cell"),
                                                  sort_keys=True,
                                                  separators=(",", ":")),
                "seed": row.get("seed"), "host": host_name(),
                "worker_pid": None, "workers": workers, "wall_seconds": None,
                "peak_rss_mb": None, "status": "skipped_completed_row",
                "reason": None,
                "source_attempt": source.get("_attempt"),
                "source_row_file": source.get("_record_path"),
                "source_status": source.get("status"),
                "end_utc": utcnow()})
            reused.append({"row_id": row["row_id"],
                           "source_attempt": source.get("_attempt"),
                           "source_row_file": source.get("_record_path"),
                           "source_status": source.get("status")})
        atomic_write_json(attempt / "reused_rows.json", {
            "count": len(reused), "rows": reused})
        attempt_record["reused_row_count"] = len(reused)
        attempt_record["reused_rows_file"] = "reused_rows.json"
        atomic_write_json(attempt / "attempt.json", attempt_record)
        log(f"reused {len(reused)} write-once rows from earlier attempts; "
            f"manifest entries point at the originating row files")
    log(f"budget projection: {projection}")
    if not projection["fits_cap"] and not args.ignore_budget_projection:
        detail = {"projection": projection, "gate": "G2 territory (Lead owns)"}
        log("STOP: projected cumulative cost exceeds the 24 h E1 cap; "
            "no row was launched")
        attempt_record.update(status="stopped",
                              reason="budget_projection_exceeds_cap",
                              finished_utc=utcnow(), budget_stop=detail)
        atomic_write_json(attempt / "attempt.json", attempt_record)
        return _write_stop(attempt, "budget_projection_exceeds_cap", detail, 2)

    if args.dry_run:
        log("dry run: no row executed")
        attempt_record.update(status="dry_run", finished_utc=utcnow())
        atomic_write_json(attempt / "attempt.json", attempt_record)
        return 0

    # Rows completed in any earlier attempt of this campaign are discovered by
    # row_id and are not recomputed; the new attempt's manifest records a
    # skipped_completed_row entry that points at the originating write-once row
    # file (see reused_rows.json).  --rerun-all is the explicit exception.
    # E1b: determine the M diagnostic checkpoint before any expensive work.
    m_row = None
    if campaign == "E1b":
        m_setting = config["E1b"]["M_diagnostic"]
        m_row = {
            "campaign": "E1b", "kind": "M_diagnostic", "row_id": "M_development",
            "seed": int(m_setting["development_seed"]),
            "cell": {"common_strength": m_setting["common_strength"],
                     "line_multiplier": m_setting["line_multiplier"]},
        }
        previous = discovered.get("M_development")
        if previous is not None and args.rerun_all:
            log("M diagnostic forced by --rerun-all in the new attempt")
            previous = None
        if previous is not None:
            if previous.get("status") == "pass":
                log("M diagnostic already recorded; skipping (write-once row)")
                m_row = None
            else:
                log(f"STOP: recorded M diagnostic status is "
                    f"{previous.get('status')!r} ({previous.get('reason')}); "
                    f"a scientific failure is recorded, not repaired")
                attempt_record.update(status="stopped",
                                      reason="M_diagnostic_previously_failed",
                                      finished_utc=utcnow())
                atomic_write_json(attempt / "attempt.json", attempt_record)
                return _write_stop(attempt, "M_diagnostic_previously_failed", {}, 4)

    if not pending and m_row is None:
        log("nothing to execute: every registered row is already recorded")
        attempt_record.update(status="complete", reason="nothing_to_do",
                              finished_utc=utcnow(), counts={})
        atomic_write_json(attempt / "attempt.json", attempt_record)
        return 0

    # E1b population preflight (ranks/margins/D before sample grids).
    if campaign == "E1b" and pending:
        try:
            population = e1b_population_preflight(config, attempt, log)
        except CampaignBlocked as error:
            log(f"STOP: {error}")
            attempt_record.update(status="stopped",
                                  reason=f"campaign_blocked: {error}",
                                  finished_utc=utcnow())
            atomic_write_json(attempt / "attempt.json", attempt_record)
            return _write_stop(attempt, f"campaign_blocked: {error}", {}, 3)
        attempt_record["a4_manifest"] = population.get(
            "manifest_clean_cell_channel")
        attempt_record["population_preflight_record"] = (
            "population_diagnostics.json")
        atomic_write_json(attempt / "attempt.json", attempt_record)
        if population["status"] != "pass":
            log("STOP: A4 population displacement differs from Appendix A by "
                "more than "
                f"{config['E1b']['displacement_mismatch_tolerance_deg']} deg "
                f"for cells {population['mismatches']}; bug check, no sample grid")
            attempt_record.update(status="stopped",
                                  reason="population_displacement_mismatch_gt_1deg",
                                  finished_utc=utcnow(),
                                  population_preflight=population)
            atomic_write_json(attempt / "attempt.json", attempt_record)
            return _write_stop(attempt,
                               "population_displacement_mismatch_gt_1deg",
                               {"mismatches": population["mismatches"]}, 4)

    tasks = []
    for row in pending:
        tasks.append({
            "root": str(root), "config": config,
            "config_path": str(config_path), "config_sha256": config_sha,
            "attempt_dir": str(attempt), "run_id": run_id,
            "code_sha": code_sha, "workers": workers, "row": row,
        })
    m_task = None
    if m_row is not None:
        m_task = {
            "root": str(root), "config": config,
            "config_path": str(config_path), "config_sha256": config_sha,
            "attempt_dir": str(attempt), "run_id": run_id,
            "code_sha": code_sha, "workers": workers, "row": m_row,
        }

    counts = {"pass": 0, "fail": 0, "not_evaluable": 0, "not_run": 0}
    consumed_worker_seconds = 0.0
    average_row_seconds = projection["row_seconds_estimate"]
    allowed_worker = allowed_worker_seconds(config, campaign, workers)
    budget_stopped = False
    memory_stopped = False
    stopped_reason = None
    m_outcome = None
    exit_code = 0
    total_rows = len(tasks)
    peak_worker_rss_mb = 0.0
    min_host_free_gb = float(worker_choice["available_gb_at_start"])

    def _check_resources(result):
        """v4 Sec.6 per-worker memory guard; records and stops, never fixes."""
        nonlocal peak_worker_rss_mb, min_host_free_gb, memory_stopped
        nonlocal stopped_reason, exit_code
        if result.get("peak_rss_mb") is not None:
            peak_worker_rss_mb = max(peak_worker_rss_mb,
                                     float(result["peak_rss_mb"]))
        free_gb = available_memory_gb()
        min_host_free_gb = min(min_host_free_gb, free_gb)
        guard = (config.get("hosts") or {}).get(
            "resource_guard", {"stop_worker_peak_rss_mb": 2560.0,
                               "stop_host_free_gb": 3.0})
        if peak_worker_rss_mb > float(guard["stop_worker_peak_rss_mb"]):
            memory_stopped = True
            stopped_reason = "resource_guard_worker_rss"
            exit_code = 6
            log(f"STOP: worker peak RSS {peak_worker_rss_mb:.0f} MB exceeds "
                f"{float(guard['stop_worker_peak_rss_mb']):.0f} MB; preserving rows")
        elif free_gb < float(guard["stop_host_free_gb"]):
            memory_stopped = True
            stopped_reason = "resource_guard_host_free_ram"
            exit_code = 6
            log(f"STOP: host free RAM {free_gb:.1f} GB below "
                f"{float(guard['stop_host_free_gb']):.1f} GB; preserving rows")

    def _record_row(result):
        nonlocal consumed_worker_seconds
        append_manifest(attempt, result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        consumed_worker_seconds += float(result["wall_seconds"] or 0.0)
        log(f"[{sum(counts.values())}/{total_rows + (1 if m_task else 0)}] "
            f"{result['row_id']} {result['status']} "
            f"{result['wall_seconds']}s avg="
            f"{consumed_worker_seconds / max(1, sum(counts.values())):.1f}s "
            f"peak_rss={peak_worker_rss_mb:.0f}MB free={min_host_free_gb:.1f}GB")
        _check_resources(result)

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            if m_task is not None:
                m_result = pool.submit(execute_row, m_task).result()
                _record_row(m_result)
                m_record = read_json(attempt / "rows" / "M_development.json")
                m_outcome = m_record
                if m_record["status"] != "pass":
                    log("STOP: M diagnostic did not pass; A3.3 requires the "
                        "four candidate scores and a frozen selection")
                    budget_stopped = True
                    stopped_reason = "M_diagnostic_not_pass"
                    exit_code = 4
            if not budget_stopped and tasks:
                futures = {}
                next_index = 0
                window = max(1, workers * 2)
                while next_index < len(tasks) and len(futures) < window:
                    futures[pool.submit(execute_row, tasks[next_index])] = tasks[next_index]
                    next_index += 1
                while futures:
                    completed_future = next(as_completed(list(futures)))
                    task = futures.pop(completed_future)
                    try:
                        result = completed_future.result()
                    except Exception as error:  # noqa: BLE001
                        result = {
                            "run_id": run_id, "code_sha": code_sha,
                            "config_sha256": config_sha, "campaign": campaign,
                            "attempt": attempt.name,
                            "row_id": task["row"]["row_id"],
                            "cell": json.dumps(task["row"].get("cell"),
                                               sort_keys=True),
                            "seed": task["row"].get("seed"),
                            "host": host_name(), "worker_pid": None,
                            "workers": workers, "wall_seconds": None,
                            "peak_rss_mb": None, "status": "fail",
                            "reason": f"worker_exception:{type(error).__name__}: {error}",
                            "end_utc": utcnow()}
                        atomic_write_json(
                            attempt / "rows" / f"{task['row']['row_id']}.json",
                            {"campaign": campaign,
                             "row_id": task["row"]["row_id"],
                             "cell": task["row"].get("cell"),
                             "seed": task["row"].get("seed"),
                             "status": "fail", "reason": result["reason"]})
                    _record_row(result)
                    if memory_stopped:
                        budget_stopped = True
                    if not budget_stopped:
                        remaining = len(tasks) - next_index
                        projected_workers = (consumed_worker_seconds
                                             + average_row_seconds * remaining)
                        if projected_workers > allowed_worker:
                            budget_stopped = True
                            stopped_reason = "budget_stop_measured_cost"
                            exit_code = 2
                            log("STOP: measured cost would exceed the remaining "
                                "E1 allocation; no further rows will be launched")
                        else:
                            while (next_index < len(tasks)
                                   and len(futures) < window):
                                futures[pool.submit(
                                    execute_row,
                                    tasks[next_index])] = tasks[next_index]
                                next_index += 1
                if budget_stopped:
                    for task in tasks[next_index:]:
                        record = {
                            "campaign": campaign,
                            "row_id": task["row"]["row_id"],
                            "cell": task["row"].get("cell"),
                            "seed": task["row"].get("seed"),
                            "status": "not_run",
                            "reason": ("resource_guard_stop_before_launch"
                                       if memory_stopped
                                       else "budget_stop_before_launch")}
                        atomic_write_json(
                            attempt / "rows" / f"{task['row']['row_id']}.json",
                            record)
                        append_manifest(attempt, {
                            "run_id": run_id, "code_sha": code_sha,
                            "config_sha256": config_sha, "campaign": campaign,
                            "attempt": attempt.name, "row_id": record["row_id"],
                            "cell": json.dumps(record["cell"], sort_keys=True),
                            "seed": record["seed"], "host": host_name(),
                            "worker_pid": None, "workers": workers,
                            "wall_seconds": None, "peak_rss_mb": None,
                            "status": "not_run", "reason": record["reason"],
                            "end_utc": utcnow()})
                        counts["not_run"] += 1
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        log("interrupted by user; completed rows are preserved")
        exit_code = 130
    except Exception as error:  # noqa: BLE001
        log(f"STOP: infrastructure failure {type(error).__name__}: {error}")
        exit_code = 5

    hashes_end = source_hashes(root, config_path)
    changed = [name for name in hashes_start
               if hashes_end.get(name) != hashes_start[name]]
    if changed:
        log(f"WARNING: scientific source changed during the attempt: {changed}")
    attempt_record["resource_guard"]["measured_peak_worker_rss_mb"] = round(
        peak_worker_rss_mb, 1)
    attempt_record["resource_guard"]["min_host_free_gb_observed"] = round(
        min_host_free_gb, 2)
    attempt_record["resource_guard"]["stopped"] = bool(memory_stopped)
    attempt_record.update(
        finished_utc=utcnow(),
        status=("complete" if exit_code == 0 else
                ("stopped" if exit_code in (2, 3, 4, 6) else "failed")),
        reason=stopped_reason,
        exit_code=exit_code,
        counts=counts,
        consumed_worker_seconds=round(consumed_worker_seconds, 3),
        consumed_worker_hours=round(consumed_worker_seconds / 3600.0, 6),
        average_row_seconds=(round(consumed_worker_seconds / max(1, sum(counts.values())), 3)
                             if consumed_worker_seconds else None),
        allowed_worker_seconds=round(allowed_worker, 1),
        source_sha256_end=hashes_end,
        source_changed_during_run=changed,
        m_diagnostic=m_outcome,
    )
    atomic_write_json(attempt / "attempt.json", attempt_record)
    manifest_rows = []
    manifest_path = attempt / "manifest.jsonl"
    if manifest_path.exists():
        manifest_rows = manifest_path.read_text(encoding="utf-8").splitlines()
    sums = {}
    for path in sorted((attempt / "rows").glob("*.json")):
        sums[str(path.relative_to(attempt))] = sha256_file(path)
    sums["manifest.jsonl"] = sha256_file(manifest_path) if manifest_path.exists() else "none"
    sums["attempt.json"] = sha256_file(attempt / "attempt.json")
    (attempt / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8")
    log(f"attempt finished: status={attempt_record['status']} counts={counts} "
        f"consumed_worker_hours={attempt_record['consumed_worker_hours']}")
    return exit_code


def _write_stop_raw(attempt: Path, reason: str, detail, code: int) -> int:
    atomic_write_json(attempt / "STOP.json", {
        "status": "stopped", "reason": reason, "detail": detail,
        "exit_code": code, "written_utc": utcnow()})
    return code


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E1 (WP3) campaign runner: E1a, E1b, E1c")
    parser.add_argument("campaign", choices=[*CAMPAIGNS, "all"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--workers", type=int, default=None,
                        help="override the recorded worker count")
    parser.add_argument("--smoke", action="store_true",
                        help="E1a only: n in {1e3,1e4}, seeds {0}")
    parser.add_argument("--retry-failed", action="store_true",
                        help="re-execute previously failed/not_evaluable rows "
                             "in a new attempt directory")
    parser.add_argument("--rerun-all", action="store_true",
                        help="re-execute every registered row in a new attempt "
                             "directory after an explicit code fix; earlier "
                             "attempts are preserved and superseded")
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="engineering bound: execute at most N pending rows")
    parser.add_argument("--row-seconds-estimate", type=float, default=None,
                        help="budget preflight override for one row's wall time")
    parser.add_argument("--ignore-budget-projection", action="store_true",
                        help="do not stop if the preflight projection exceeds "
                             "the cap (recorded; the Lead owns G2)")
    parser.add_argument("--rerun-rows", default=None,
                        help="comma-separated row ids to re-run in a new attempt "
                             "(provenance repair; earlier records are preserved)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="development-only override: allow a full-scope run on a dirty "
                             "src/scripts tree; the attempt is marked non-registered")
    parser.add_argument("--dry-run", action="store_true",
                        help="record the attempt, projection and rows; execute none")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.campaign == "all":
        if args.smoke:
            parser.error("--smoke only applies to E1a")
        for campaign in CAMPAIGNS:
            exit_code = run_campaign(argparse.Namespace(
                **{**vars(args), "campaign": campaign}))
            if exit_code != 0:
                return exit_code
        return 0
    return run_campaign(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
