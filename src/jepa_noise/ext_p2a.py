# SPDX-License-Identifier: MIT
"""Paper 3 P2 addendum: fair refit ceiling, P2-f labels, F1/F2 fold, hygiene.

Runbook: ``docs/jepa/RUNBOOK_2026-11-11_P2_ADDENDUM.md`` (sha256
``f97ed8101e5d0383a85ada70d974df4bb54bc70e63865d48fe77cb879cf59e0a``),
extension ``paper3_p2_addendum_2026-11-11``, CPU only.

A1  R3u: the readout refit on ``[z_c, u_c]`` with the constant design column
    dropped, plus R3h (``[z_c, h_e/hbar]``), the raw-std refit on
    ``[raw_c, u_c]`` and the C0 hit-count control on N;
A2  P2-f labels recomputed with the rungs R0 < R1 < R2 < R3 < R3u;
A3  the ceiling-normalised repair share eta = rho_R2 / rho_R3u (post hoc,
    exploratory) with the fixed interpretation rule;
A4  the F1/F2 fold into ``<P>`` and the exact equivalence check against the
    committed P2 payloads;
A6  descriptive X3-P shift shares (why invariance is not achieved).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import ext_followup as ef
from . import ext_p2 as p2x
from . import ext_repair as er
from . import lucid_bombs as lb
from . import lucid_run as lr

EXTENSION = "paper3_p2_addendum_2026-11-11"
MODES = ("dry-run", "equivalence", "p2a", "shift", "report")

ROOT = Path(__file__).resolve().parents[2]
P2_EXT_ROOT = p2x.EXT_ROOT
RESULTS_DIR = "results/jepa/20261111_p2a"
P2_RESULTS_DIR = p2x.RESULTS_DIR          # results/jepa/20261109_p2
RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-11_P2_ADDENDUM.md"
RUNBOOK_SHA256 = "f97ed8101e5d0383a85ada70d974df4bb54bc70e63865d48fe77cb879cf59e0a"
ACCEPTANCE_REL = "docs/jepa/ACCEPTANCE_2026-11-11_P2_ADDENDUM.json"
REVIEW_REL = "docs/jepa/REVIEW_2026-11-11_P2_ADDENDUM.md"
DECISIONS_REL = "docs/jepa/DECISIONS.md"
STATE_REL = "docs/jepa/STATE.md"
ACCEPTANCE_KEYS = ("runbook_sha256", "impl_commit", "review_path", "review_verdict",
                   "lead", "date")
P_FILES = tuple(p2x.P_FILES) + (
    "src/jepa_noise/ext_p2a.py",
    "scripts/jepa_ext_p2a.py",
    "scripts/jepa_cpu_launch.py",
    RUNBOOK_REL,
)
P_DIRS = p2x.P_DIRS

SEED = 20261111
PERM_NAMESPACE = "paper3/p2a/perm"
PERMUTATIONS = 10_000
DRIFT_COST_MIN_M = p2x.DRIFT_COST_MIN_M          # 3 mm
DESTROYED_M = p2x.REFIT_DESTROYED_M              # 3 mm
P2F_SHARE = p2x.P2F_SHARE                        # 0.8
ETA_MIN_RHO_R3U = 0.05
CEILING_TOLERANCE_M = 0.005                      # 5 mm beyond which R2 > R3u is a defect
CEILING_CI_M = 0.005
CONDITIONS = p2x.CONDITIONS


# =================================================================== gate A4
def _resolve_root(root=None) -> Path:
    return Path(root).resolve() if root is not None else ROOT


def p_paths() -> list:
    return list(P_FILES) + list(P_DIRS)


def in_p(relative: str) -> bool:
    relative = str(relative)
    return relative in P_FILES or any(relative.startswith(prefix) for prefix in P_DIRS)


def _git(root, *args, timeout=60):
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                             text=True, timeout=timeout)
        return out.returncode, out.stdout, out.stderr
    except Exception as exc:  # pragma: no cover
        return 1, "", str(exc)


def read_json(path, default=None):
    return ef.read_json(path, default)


def write_json(path, payload, *, overwrite=True) -> Path:
    return er.write_json(path, payload, overwrite=overwrite)


def acceptance_record(root=None) -> dict | None:
    return read_json(_resolve_root(root) / ACCEPTANCE_REL, None)


def import_closure(root=None) -> dict:
    root = _resolve_root(root)
    outside, checked = [], 0
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename or not isinstance(filename, (str, os.PathLike)):
            continue
        try:
            path = Path(filename)
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
        except OSError:
            continue
        if not path.is_file() or root not in path.parents:
            continue
        checked += 1
        relative = path.relative_to(root).as_posix()
        if not in_p(relative):
            outside.append({"module": name, "path": relative})
    forced = [row for row in outside
              if str(row["path"]).startswith(ef.FORCED_IMPORT_EXCEPTIONS)]
    violations = [row for row in outside
                  if not str(row["path"]).startswith(ef.FORCED_IMPORT_EXCEPTIONS)]
    return {"registered": False, "extension": EXTENSION,
            "check": "import closure (P2 addendum)", "root": str(root),
            "checked_modules": int(checked),
            "outside_allowlist": sorted(outside, key=lambda row: row["path"]),
            "forced_import_exceptions": sorted(forced, key=lambda row: row["path"]),
            "violations": sorted(violations, key=lambda row: row["path"]),
            "pass": not violations}


def _gate_checks(mode, *, root) -> dict:
    root = _resolve_root(root)
    checks, problems = {}, []
    runbook_path = root / RUNBOOK_REL
    computed = lr.sha256_file(runbook_path) if runbook_path.is_file() else None
    record = acceptance_record(root)
    pending = record is None
    expected = (record or {}).get("runbook_sha256")
    a_pass = bool(computed is not None and expected is not None
                  and computed == expected and computed == RUNBOOK_SHA256)
    checks["a"] = {"pass": a_pass, "runbook_sha256": computed,
                   "expected_sha256": RUNBOOK_SHA256, "record_sha256": expected,
                   "pending_acceptance": bool(pending)}
    if not a_pass and not pending:
        problems.append("(a) the committed addendum runbook sha256 does not match")
    if record is None:
        checks["b"] = {"pass": False, "status": "gate_pending_acceptance"}
    elif not isinstance(record, dict):
        checks["b"] = {"pass": False, "status": "invalid"}
        problems.append("(b) the acceptance record is not valid JSON")
    else:
        missing = [key for key in ACCEPTANCE_KEYS if key not in record]
        review = root / str(record.get("review_path") or REVIEW_REL)
        b_pass = bool(not missing and record.get("review_verdict") == "PASS"
                      and record.get("review_path") == REVIEW_REL and review.is_file())
        checks["b"] = {"pass": b_pass, "status": "PASS" if b_pass else "denied",
                       "record": record, "missing_keys": missing,
                       "review_exists": bool(review.is_file())}
        if not b_pass:
            problems.append("(b) the acceptance record is incomplete or not PASS")
    impl_commit = (record or {}).get("impl_commit") if record else None
    if impl_commit is None:
        code, head, _ = _git(root, "rev-parse", "HEAD")
        impl_commit = head.strip() if code == 0 else None
    paths = p_paths()
    if impl_commit:
        code, _, err = _git(root, "diff", "--quiet", str(impl_commit), "--", *paths)
        diff_clean = code == 0
        diff_error = None if diff_clean else (err.strip() or f"git diff exit {code}")
    else:
        diff_clean, diff_error = False, "no impl_commit"
    code, untracked_out, _ = _git(root, "ls-files", "--others", "--exclude-standard",
                                  "--", *paths)
    untracked = [line for line in untracked_out.splitlines() if line.strip()]
    c_pass = bool(diff_clean and not untracked)
    checks["c"] = {"pass": c_pass, "impl_commit": impl_commit,
                   "diff_clean": bool(diff_clean), "diff_error": diff_error,
                   "untracked": untracked}
    if not c_pass and not pending:
        problems.append(f"(c) the <P> tree differs from {impl_commit}")
    status = ("gate_pending_acceptance" if pending
              else ("granted" if not problems else "denied"))
    return {"authorised": bool(not problems and not pending), "status": status,
            "mode": mode, "extension": EXTENSION, "checks": checks,
            "problems": problems, "acceptance": record,
            "runbook_sha256": computed}


def assert_authorised(mode, *, root=None, dry_run=None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    dry = bool(dry_run) if dry_run is not None else (mode == "dry-run")
    result = _gate_checks(mode, root=root)
    if dry:
        return result
    if not result["authorised"]:
        for problem in result["problems"]:
            print(f"[p2a] authorisation gate: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return result


def assert_closure(mode, root) -> dict:
    closure = import_closure(root)
    if not closure["pass"]:
        print(f"[p2a] {mode}: import-closure STOP: {closure['violations']}",
              file=sys.stderr)
        raise SystemExit(2)
    return closure


# =============================================================== A1 design A1
def _dead_by_run(rows_n, rows_c, condition, geometry) -> dict | None:
    if condition.kind != "dead":
        return None
    runs = set(np.asarray(rows_n["run_ids"]).astype(str)) | set(
        np.asarray(rows_c["run_ids"]).astype(str))
    return {str(run): condition.dead_fraction_by_module(str(run), geometry)
            for run in runs}


def fair_design(rows, condition, dead_by_run, hbar, geometry) -> np.ndarray:
    """``u_c``: ``dead_m h_e/hbar`` (K/P) or ``h_e/hbar`` with the constant dropped (T)."""
    design = p2x.r2_design(rows, condition, dead_by_run, hbar, geometry)
    if condition.kind != "dead" and design.ndim == 2 and design.shape[1] > 1:
        # the T/R1 design is [h_e/hbar, 1]; drop the constant column (A1).
        # K/P designs are dead_m h_e/hbar (105 columns) and have no constant.
        design = design[:, :1]
    return np.asarray(design, float)


def hit_design(rows, hbar) -> np.ndarray:
    """R3h's single column ``h_e / hbar``."""
    return (np.asarray(rows["n_hit"], float) / float(hbar))[:, None]


def augmented(rows, design) -> dict:
    events = np.concatenate([np.asarray(rows["events"], float),
                             np.asarray(design, float)], axis=1)
    return dict(rows, events=events)


def probe_errors(train_rows, test_rows):
    fitted = er.fit_vertex_probe(train_rows)
    metrics = er.vertex_metrics(train_rows, test_rows, fitted=fitted)
    return float(metrics["mean_error_m"]), np.asarray(metrics["errors"], float)


def frozen_errors(train_rows, test_rows, fitted) -> np.ndarray:
    return np.asarray(er.vertex_metrics(train_rows, test_rows,
                                        fitted=fitted)["errors"], float)


def ceiling_gap_ci(err_r2, err_r3u, run_ids) -> dict:
    """CI of ``v_R2 - v_R3u`` (the sign S4 is defined on)."""
    reading = p2x._difference_reading(np.asarray(err_r3u, float),
                                     np.asarray(err_r2, float), run_ids)
    return {"difference_m": float(reading["difference"]),
            "ci95": [float(reading["ci95"][0]), float(reading["ci95"][1])],
            "excludes_zero": bool(reading["excludes_zero"])}


def _aligned_rows(source, target, what) -> dict | None:
    """Key-align ``source`` rows to ``target`` (run_id, event_id) order (M6)."""
    if source is None:
        return None
    target_keys = list(zip(np.asarray(target["run_ids"]).astype(str).tolist(),
                           np.asarray(target["event_ids"], np.int64).tolist()))
    source_keys = list(zip(np.asarray(source["run_ids"]).astype(str).tolist(),
                           np.asarray(source["event_ids"], np.int64).tolist()))
    source_index = {key: position for position, key in enumerate(source_keys)}
    order = []
    for key in target_keys:
        position = source_index.get(key)
        if position is None:
            raise ValueError(f"A1: {what} rows do not align with the embedding rows")
        order.append(position)
    out = {}
    for key, value in source.items():
        array = np.asarray(value)
        out[key] = array[order] if array.ndim >= 1 and array.shape[0] == len(order) else array
    return out


def _raw_probe_errors(train_rows, test_rows):
    fitted = p2x._raw_vertex_probe(train_rows)
    metrics = p2x._raw_vertex_errors(train_rows, test_rows, fitted)
    return float(np.mean(metrics)), np.asarray(metrics, float)


def _diff(left, right, run_ids) -> dict:
    return p2x._difference_reading(np.asarray(left, float), np.asarray(right, float),
                                   run_ids)


def ceiling_cell(data, *, frozen_vertex, p2_cell) -> dict:
    """A1 readings for one model x condition cell (``data`` like ``cell_reading``)."""
    condition = data["condition"]
    n_train, n_test = data["n_train"], data["n_test"]
    c_train, c_test = data["c_train"], data["c_test"]
    hbar, geometry = float(data["hbar"]), data["geometry"]
    if c_test is None or c_test["events"].shape[0] == 0:
        return {"condition": condition.name, "status": "no events"}
    dead = _dead_by_run(c_train, c_test, condition, geometry)
    u_train = fair_design(c_train, condition, dead, hbar, geometry)
    u_test = fair_design(c_test, condition, dead, hbar, geometry)
    # R2's repair is P2's own design: for T/R1 it keeps the constant column
    u2_train = np.asarray(p2x.r2_design(c_train, condition, dead, hbar, geometry),
                          float)
    u2_test = np.asarray(p2x.r2_design(c_test, condition, dead, hbar, geometry),
                         float)
    h_train = hit_design(c_train, hbar)
    h_test = hit_design(c_test, hbar)
    # frozen P_N errors on the paired events
    n_errors_full = frozen_errors(n_train, n_test, frozen_vertex)
    index_n, index_c, run_ids = p2x._align_subset(n_test, c_test)
    n_errors = n_errors_full[index_n]
    c_errors = frozen_errors(n_train, c_test, frozen_vertex)[index_c]
    v_n, v_c = float(np.mean(n_errors)), float(np.mean(c_errors))
    delta = v_c - v_n
    # R3u and R3h
    v_r3u, err_r3u = probe_errors(augmented(c_train, u_train),
                                  augmented(c_test, u_test))
    v_r3h, _ = probe_errors(augmented(c_train, h_train),
                            augmented(c_test, h_test))
    # R2 refit (for the fair ceiling check) — same repair as P2's R2
    index_nt, index_ct, runs_t = p2x._align_subset(n_train, c_train)
    dz_train = (np.asarray(c_train["events"], float)
                - np.asarray(n_train["events"], float)[index_nt])
    index_ne, index_ce, runs_e = p2x._align_subset(n_test, c_test)
    dz_test = (np.asarray(c_test["events"], float)
               - np.asarray(n_test["events"], float)[index_ne])
    fit = er.equivariance_fit(
        u2_train, dz_train, u2_test, dz_test,
        run_ids_train=runs_t, run_ids_test=runs_e,
        kind=("occupancy" if condition.kind == "dead" else "global"))
    a_raw = er.raw_map(fit)
    repaired = dict(c_test)
    repaired["events"] = (np.asarray(c_test["events"], float)
                          - u2_test @ np.asarray(a_raw, float).T)
    err_r2 = frozen_errors(n_train, repaired, frozen_vertex)[index_ce]
    v_r2 = float(np.mean(err_r2))
    ceiling_difference = ceiling_gap_ci(err_r2, err_r3u, runs_e)
    ceiling_gap = float(v_r2 - v_r3u)
    # raw-std refit on [raw_c, u_c], the encoder-independent bound
    raw = {"status": "not available"}
    if data.get("raw_c_train") is not None and data.get("raw_c_test") is not None \
            and data.get("raw_n_train") is not None and data.get("raw_n_test") is not None:
        raw_train = augmented(_aligned_rows(data["raw_c_train"], c_train, "raw_c_train"),
                              u_train)
        raw_test = augmented(_aligned_rows(data["raw_c_test"], c_test, "raw_c_test"),
                             u_test)
        raw_n_fitted = p2x._raw_vertex_probe(data["raw_n_train"])
        raw_n_errors = p2x._raw_vertex_errors(data["raw_n_train"],
                                              data["raw_n_test"], raw_n_fitted)
        raw_c_errors = p2x._raw_vertex_errors(data["raw_n_train"],
                                              data["raw_c_test"], raw_n_fitted)
        idx_raw_n, idx_raw_c, runs_raw = p2x._align_subset(data["raw_n_test"],
                                                           data["raw_c_test"])
        raw_c_aligned = np.asarray(raw_c_errors, float)[idx_raw_c]
        v_raw_n = float(np.mean(np.asarray(raw_n_errors, float)[idx_raw_n]))
        v_raw_c = float(np.mean(raw_c_aligned))
        v_raw_r3u, raw_r3u_errors = _raw_probe_errors(raw_train, raw_test)
        raw_difference = p2x._difference_reading(
            np.asarray(raw_n_errors, float)[idx_raw_n], raw_r3u_errors, runs_raw)
        raw = {"status": "evaluated", "v_N_m": v_raw_n, "v_c_m": v_raw_c,
               "v_R3u_m": float(v_raw_r3u),
               "R_c_u_m": float(v_raw_r3u - v_raw_n),
               "R_c_u_ci_m": raw_difference["ci95"],
               "delta_c_m": float(v_raw_c - v_raw_n),
               "rho_R3u": (None if abs(v_raw_c - v_raw_n) <= 0 else
                           float((v_raw_c - v_raw_r3u) / (v_raw_c - v_raw_n)))}
    log_yield = {"status": "not available"}
    if all(row is not None and "log_yield" in row
           for row in (c_train, c_test, n_train, n_test)):
        augmented_train_y = augmented(c_train, u_train)
        augmented_test_y = augmented(c_test, u_test)
        v_r3u_y = float(er.log_yield_metrics(augmented_train_y,
                                             augmented_test_y)["resolution"])
        frozen_y = er.fit_log_yield_probe(n_train)
        v_n_y = float(er.log_yield_metrics(n_train, n_test,
                                           fitted=frozen_y)["resolution"])
        v_c_y = float(er.log_yield_metrics(n_train, c_test,
                                           fitted=frozen_y)["resolution"])
        delta_y = v_c_y - v_n_y
        log_yield = {"status": "evaluated", "v_N": v_n_y, "v_c": v_c_y,
                     "v_R3u": v_r3u_y, "R_c_u": float(v_r3u_y - v_n_y),
                     "rho_R3u": (None if abs(delta_y) <= 0 else float(
                         (v_c_y - v_r3u_y) / delta_y))}
    out = {
        "condition": condition.name,
        "n_events": int(n_errors.size),
        "v_N_m": v_n, "v_c_m": v_c, "delta_c_m": float(delta),
        "v_R3u_m": v_r3u, "R_c_u_m": float(v_r3u - v_n),
        "R_c_u_ci_m": _diff(n_errors, err_r3u, run_ids)["ci95"],
        "R_c_u_excludes_zero": bool(_diff(n_errors, err_r3u, run_ids)["excludes_zero"]),
        "information_destroying_u": bool(v_r3u - v_n >= DESTROYED_M
                                         and _diff(n_errors, err_r3u,
                                                   run_ids)["excludes_zero"]),
        "rho_R3u": (None if abs(delta) <= 0 else float((v_c - v_r3u) / delta)),
        "v_R3h_m": v_r3h,
        "R_c_h_m": float(v_r3h - v_n),
        "rho_R3h": (None if abs(delta) <= 0 else float((v_c - v_r3h) / delta)),
        "ceiling": {
            "v_R2_recomputed_m": v_r2,
            "v_R2_committed_m": ((p2_cell.get("latent") or {}).get("v_R2_m")
                                 if p2_cell else None),
            "v_R2_minus_v_R3u_m": ceiling_gap,
            "ci_m": ceiling_difference["ci95"],
            "excludes_zero": bool(ceiling_difference["excludes_zero"]),
            "violation": bool(ceiling_gap < -CEILING_TOLERANCE_M
                              and ceiling_difference["ci95"][1] < 0.0),
            "check_abs_lt_1e-9": (None if not p2_cell
                                  else bool(abs(v_r2 - float(
                                      (p2_cell.get("latent") or {}).get("v_R2_m")
                                      or np.nan)) <= 1e-9))},
        "refit": ((p2_cell or {}).get("refit")),
        "raw": raw,
        "log_yield": log_yield,
    }
    return out


def _raw_errors_for(train_rows, test_rows) -> np.ndarray:
    fitted = p2x._raw_vertex_probe(train_rows)
    return np.asarray(p2x._raw_vertex_errors(train_rows, test_rows, fitted), float)


# ================================================================ inputs §2
CORE_MODELS = tuple(p2x.CORE_MODELS)


def load_inputs(root=None, *, config, gen, geometry, data_root=None, log=print) -> dict:
    """The sha-verified P2 caches, transform info, raw-std rows and gain table."""
    root = _resolve_root(root)
    data_root = data_root or p2x.DATA_ROOT_DEFAULT
    probes005 = p2x.open_probe_sets(config, gen, geometry, data_root,
                                    setting=p2x.S005, log=log)
    drift = p2x.build_drift_conditions()
    infos = {}
    for name in list(CONDITIONS) + [p2x.R1_NAME]:
        for role in ("train", "test"):
            infos[(name, role)] = p2x.load_transform_info(P2_EXT_ROOT, role, name)
    raw = p2x.build_raw_features(probes005, drift, geometry, gen, log=log)
    gain = p2x.gain_r2_eq_table(root)
    hbar = None
    n0 = p2x.cache_rows(p2x.n_embedding_cache(CORE_MODELS[0]),
                        er.embedding_cache_key("train", "N"))
    if n0 is not None and np.asarray(n0["n_hit"]).size:
        hbar = float(np.mean(np.asarray(n0["n_hit"], float)))
    if hbar is None:
        raise RuntimeError("A1: the N-train hit counts are unavailable")
    return {"root": root, "data_root": data_root, "probes005": probes005,
            "drift": drift, "infos": infos, "raw": raw, "gain": gain,
            "hbar": hbar, "geometry": geometry, "config": config, "gen": gen}


def committed_p2(root=None) -> dict:
    payload = read_json(_resolve_root(root) / P2_RESULTS_DIR / "p2.json", None)
    if payload is None:
        raise RuntimeError("A1 needs the committed results/jepa/20261109_p2/p2.json")
    return payload


def committed_x3p(root=None) -> dict:
    payload = read_json(_resolve_root(root) / P2_RESULTS_DIR / "x3p.json", None)
    if payload is None:
        raise RuntimeError("A6 needs the committed results/jepa/20261109_p2/x3p.json")
    return payload


def _mask_to_exact(rows, info) -> np.ndarray:
    return p2x._mask_rows(rows, p2x._trig_lookup(info))


def model_cells(inputs, label, *, exact=False) -> dict:
    """Per-condition ``data`` dicts (the same conventions as P2's cell_reading)."""
    cache = p2x.embedding_cache(label)
    n_cache = p2x.n_embedding_cache(label)
    n_train = p2x.cache_rows(n_cache, er.embedding_cache_key("train", "N"))
    n_test = p2x.cache_rows(n_cache, er.embedding_cache_key("test", "N"))
    if n_train is None or n_test is None:
        raise RuntimeError(f"{label}: N embeddings missing")
    frozen = er.fit_vertex_probe(n_train)
    out = {}
    for name in CONDITIONS:
        c_train = p2x.cache_rows(cache, er.embedding_cache_key("train", name))
        c_test = p2x.cache_rows(cache, er.embedding_cache_key("test", name))
        if c_train is None or c_test is None:
            out[name] = {"status": "unavailable",
                         "reason": "condition cache missing"}
            continue
        data = {"condition": inputs["drift"][name], "n_train": n_train,
                "n_test": n_test, "c_train": c_train, "c_test": c_test,
                "hbar": inputs["hbar"], "geometry": inputs["geometry"]}
        raw_train, raw_test = (inputs["raw"].get((name, "train")),
                               inputs["raw"].get((name, "test")))
        if exact:
            data["c_train"] = p2x._subset_rows(
                c_train, _mask_to_exact(c_train, inputs["infos"][(name, "train")]))
            data["c_test"] = p2x._subset_rows(
                c_test, _mask_to_exact(c_test, inputs["infos"][(name, "test")]))
            raw_train = (None if raw_train is None else p2x._subset_rows(
                raw_train, _mask_to_exact(raw_train, inputs["infos"][(name, "train")])))
            raw_test = (None if raw_test is None else p2x._subset_rows(
                raw_test, _mask_to_exact(raw_test, inputs["infos"][(name, "test")])))
        data["raw_n_train"] = inputs["raw"].get(("N", "train"))
        data["raw_n_test"] = inputs["raw"].get(("N", "test"))
        data["raw_c_train"] = raw_train
        data["raw_c_test"] = raw_test
        out[name] = data
    return {"frozen": frozen, "cells": out, "n_train": n_train, "n_test": n_test}


# ================================================================ A2 labels
def p2f_u(p2_cell, ceiling) -> dict:
    """P2-f recomputed with the rungs R0 < R1 < R2 < R3 < R3u (A2)."""
    if ceiling.get("status") == "no events":
        return {"status": "not available"}
    delta = p2_cell.get("delta_c_m")
    if delta is None or abs(float(delta)) <= 0:
        return {"status": "no drift cost"}
    v_c = float(p2_cell["v_c_m"])
    r3 = None
    if (p2_cell.get("refit") or {}).get("v_R3_m") is not None:
        r3 = (v_c - float(p2_cell["refit"]["v_R3_m"])) / float(delta)
    r3u = ceiling.get("rho_R3u")
    rungs = [("R0", 0.0)]
    r1 = (p2_cell.get("r1") or {}).get("rho_R1")
    if r1 is not None:
        rungs.append(("R1", float(r1)))
    r2 = (p2_cell.get("latent") or {}).get("rho_R2")
    if r2 is not None:
        rungs.append(("R2", float(r2)))
    if r3 is not None:
        rungs.append(("R3", float(r3)))
    if r3u is not None:
        rungs.append(("R3u", float(r3u)))
    caps = [value for name, value in rungs if name in ("R3", "R3u")
            and value is not None]
    target = (None if not caps else P2F_SHARE * max(caps))
    cheapest = None
    for rung, value in rungs:
        if target is not None and value >= target:
            cheapest = rung
            break
    beyond = bool(ceiling.get("information_destroying_u"))
    note = None
    if cheapest is not None and target is not None:
        note = (f"cheapest rung {cheapest} reaches {P2F_SHARE:.0%} of the refit "
                f"ceiling")
        if beyond and ceiling.get("R_c_u_m") is not None:
            note += (f"; R_c^u = {1000.0 * float(ceiling['R_c_u_m']):.3f} mm remains "
                     "beyond refit (R4/R5 territory)")
    return {"status": "evaluated", "rho_R0": 0.0, "rho_R1": r1, "rho_R2": r2,
            "rho_R3": r3, "rho_R3u": r3u, "target": target,
            "cheapest_rung": cheapest, "beyond_refit": beyond, "note": note}


# ============================================================== A3 eta (post hoc)
def spearman(x, y):
    return p2x.spearman(x, y)


def permutation_spearman_p2a(x, y, *, level, permutations=PERMUTATIONS) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = spearman(x, y)
    if observed is None:
        return {"rho_s": None, "p": None, "n": int(x.size),
                "permutations": int(permutations), "status": "not evaluable"}
    exceed = 0
    for index in range(int(permutations)):
        rng = lb.keyed_rng(SEED, PERM_NAMESPACE, str(level), str(index))
        value = spearman(x, y[rng.permutation(y.size)])
        if value is not None and value >= observed:
            exceed += 1
    return {"rho_s": float(observed),
            "p": float((exceed + 1.0) / (int(permutations) + 1.0)),
            "n": int(x.size), "permutations": int(permutations),
            "exceed": int(exceed), "one_sided": True, "seed": SEED,
            "namespace": PERM_NAMESPACE}


def eta_readings(p2_payload, ceilings, gain) -> dict:
    """A3: eta = rho_R2 / rho_R3u per model per K level, with its fixed rule."""
    out = {"levels": {}, "secondary_r2eq_K15": None,
           "sensitivity_no_fix": {}}
    for level in ("K5", "K15", "K30"):
        xs, ys, used, excluded = [], [], [], []
        for label in CORE_MODELS:
            cell = ((p2_payload.get("models") or {}).get(label) or {}).get(
                "conditions", {}).get(level) or {}
            rho2 = (cell.get("latent") or {}).get("rho_R2")
            r3u = (ceilings.get(label) or {}).get(level, {}).get("rho_R3u")
            if rho2 is None or r3u is None:
                excluded.append({"model": label, "reason": "missing rho"})
                continue
            if float(r3u) <= ETA_MIN_RHO_R3U:
                excluded.append({"model": label, "reason": "rho_R3u <= 0.05"})
                continue
            if gain.get(label) is None:
                excluded.append({"model": label, "reason": "missing gain R2_eq"})
                continue
            xs.append(float(gain[label])); ys.append(float(rho2) / float(r3u))
            used.append(label)
        stats = (permutation_spearman_p2a(xs, ys, level=level) if len(xs) >= 3
                 else {"rho_s": None, "p": None, "n": len(xs),
                       "status": "fewer than 3 usable models"})
        out["levels"][level] = dict(stats, models=used, excluded=excluded)
    # the existing secondary Spearman(gain, R2_eq,K15), now with a permutation p
    xs, ys = [], []
    for label in CORE_MODELS:
        cell = ((p2_payload.get("models") or {}).get(label) or {}).get(
            "conditions", {}).get("K15") or {}
        r2eq = (cell.get("latent") or {}).get("r2_eq")
        if r2eq is not None and gain.get(label) is not None:
            xs.append(float(gain[label])); ys.append(float(r2eq))
    out["secondary_r2eq_K15"] = (permutation_spearman_p2a(
        xs, ys, level="K15-r2eq") if len(xs) >= 3 else {"rho_s": None, "n": len(xs)})
    # M16: the primary Spearman without fix9000/fix9001, at every K level (n = 13)
    out["sensitivity_no_fix"] = {}
    for level in ("K5", "K15", "K30"):
        x, y, used = p2x._eligible_cells(p2_payload.get("models") or {}, level)
        keep = [position for position, label in enumerate(used)
                if label not in ("fix9000", "fix9001")]
        xs = [float(x[position]) for position in keep]
        ys = [float(y[position]) for position in keep]
        block = (permutation_spearman_p2a(xs, ys, level=f"no-fix-{level}")
                 if len(xs) >= 3 else {"rho_s": None, "n": len(xs)})
        block["models"] = [used[position] for position in keep]
        out["sensitivity_no_fix"][level] = block
    per_level = [out["levels"][level] for level in ("K5", "K15", "K30")]
    support = sum(1 for block in per_level
                  if block.get("rho_s") is not None and block.get("p") is not None
                  and block["rho_s"] >= 0.5 and block["p"] <= 0.05)
    out["rule"] = {
        "criterion": "eta Spearman >= 0.5 and p <= 0.05 in at least two of the "
                     "three K levels",
        "levels_supporting": int(support),
        "verdict": ("supported" if support >= 2 else "not supported"),
        "label": "exploratory post hoc; does not replace the P2-e primary (mixed)"}
    return out


# ================================================================ A6 shift share
def shift_share(w, shift, basis) -> float:
    """Fraction of ||W shift||^2 removed by the orthonormal basis ``basis``."""
    w_shift = np.asarray(w, float) @ np.asarray(shift, float)
    projected = np.asarray(basis, float).T @ w_shift
    total = float(np.sum(w_shift ** 2))
    return 0.0 if total <= 0 else float(np.sum(projected ** 2) / total)


def shift_readings(root=None, *, config, gen, geometry, data_root=None,
                   log=print) -> dict:
    """A6: how much of the N->D latent shift each rank-r projection removes."""
    root = _resolve_root(root)
    gate = assert_authorised("shift", root=root)          # B3: A6 is gated too
    started = time.time()
    data_root = data_root or p2x.DATA_ROOT_DEFAULT
    probes005 = p2x.open_probe_sets(config, gen, geometry, data_root,
                                    setting=p2x.S005, log=log)
    probes030 = p2x.open_probe_sets(config, gen, geometry, data_root,
                                    setting=p2x.S030, log=log)
    delta = er.run_log_gain_vectors(probes030["train"], geometry)
    delta.update(er.run_log_gain_vectors(probes030["test"], geometry))
    occupancy = {role: er.occupancy_counts(
        probes005[role], er.build_condition("N", probes005[role], geometry), geometry)
        for role in ("train", "test")}
    payload = {"registered": False, "extension": EXTENSION, "check": "A6 X3-P shift",
               "ranks": list(p2x.X3P_RANKS), "controls": p2x.X3P_CONTROLS,
               "gate": {"status": gate["status"]}, "models": {},
               "wall_seconds": None}
    for label in p2x.X3P_MODELS:
        cache = p2x.n_embedding_cache(label)
        rows = {"train_N": p2x.cache_rows(cache, er.embedding_cache_key("train", "N")),
                "test_N": p2x.cache_rows(cache, er.embedding_cache_key("test", "N")),
                "test_D": p2x.cache_rows(cache, er.embedding_cache_key("test", "D"))}
        if any(rows[key] is None for key in rows):
            payload["models"][label] = {"status": "unavailable",
                                        "reason": "N/D caches missing"}
            continue
        x2b = p2x._x2b_for(label, cache, geometry, delta, occupancy, log=log)
        if x2b.get("A_hat") is None:
            payload["models"][label] = {"status": "unavailable",
                                        "reason": "X2-b fit unavailable"}
            continue
        events = np.asarray(rows["train_N"]["events"], float)
        whitening = er.whitening_matrix(events, np.asarray(rows["train_N"]["run_ids"]))
        w = whitening["W"]
        paired = ef.pair_events(rows["test_N"], rows["test_D"])
        z_n = np.asarray(rows["test_N"]["events"], float)[paired["index_left"]]
        z_d = np.asarray(rows["test_D"]["events"], float)[paired["index_right"]]
        shift = (z_d - z_n).T                      # (d, n_events)
        w_shift = w @ shift
        total = float(np.sum(w_shift ** 2))
        singular = np.linalg.svd(w @ np.asarray(x2b["A_hat"], float),
                                 compute_uv=False)
        u_matrix = np.linalg.svd(w @ np.asarray(x2b["A_hat"], float))[0]
        frob = float(np.sum(singular ** 2))
        entry = {"status": "evaluated", "n_events": int(paired["n_events"]),
                 "x2b_r2_eq": x2b.get("r2_eq"),
                 "shift_norm_sq": total, "gain_singular_values": [float(s)
                                                                  for s in singular[:12]],
                 "ranks": {}}
        for rank in p2x.X3P_RANKS:
            u_r = u_matrix[:, :int(rank)]
            controls = []
            for index in range(int(p2x.X3P_CONTROLS)):
                rng = lb.keyed_rng(p2x.SEED, f"paper3/p2/x3p/{label}/{rank}/{index}")
                basis = er.random_control_basis(rng, w.shape[0], int(rank))
                controls.append(shift_share(w, shift, basis))
            entry["ranks"][str(rank)] = {
                "share_removed": (None if total <= 0 else shift_share(w, shift, u_r)),
                "share_removed_control_mean": (None if not controls else float(
                    np.mean(controls))),
                "gain_frob_share_top_r": (None if frob <= 0 else float(
                    np.sum(singular[:int(rank)] ** 2) / frob)),
                "n_controls": int(len(controls))}
        payload["models"][label] = entry
    payload["complete"] = all(entry.get("status") == "evaluated"
                              for entry in payload["models"].values())
    payload["pass"] = bool(payload["complete"])
    payload["wall_seconds"] = round(time.time() - started, 3)
    return payload


# ================================================================ A1 driver
def c0_reading(inputs, cells) -> dict:
    """C0: on N, does the embedding carry the hit count (refit [z_N, h/hbar])?"""
    n_train, n_test = cells["n_train"], cells["n_test"]
    frozen = cells["frozen"]
    u_train = hit_design(n_train, inputs["hbar"])
    u_test = hit_design(n_test, inputs["hbar"])
    v_aug, err_aug = probe_errors(augmented(n_train, u_train),
                                  augmented(n_test, u_test))
    err_frozen = frozen_errors(n_train, n_test, frozen)
    run_ids = np.asarray(n_test["run_ids"], dtype=object)
    difference = _diff(err_frozen, err_aug, run_ids)
    v_frozen = float(np.mean(err_frozen))
    return {"v_N_m": v_frozen, "v_N_aug_m": float(v_aug),
            "difference_m": float(v_aug - v_frozen),
            "ci_m": difference["ci95"],
            "excludes_zero": bool(difference["excludes_zero"])}


def run_p2a(root=None, *, config, gen, geometry, data_root=None, log=print) -> dict:
    """A1 + A2 + A3 readings (CPU, from the P2 caches)."""
    root = _resolve_root(root)
    gate = assert_authorised("p2a", root=root)
    started = time.time()
    p2 = committed_p2(root)
    inputs = load_inputs(root, config=config, gen=gen, geometry=geometry,
                         data_root=data_root, log=log)
    # M13: the gain table comes from the committed payload, not the live /ceph file
    gain = {}
    for label in CORE_MODELS:
        value = (((p2.get("models") or {}).get(label) or {}).get("gain_r2_eq"))
        if value is None:
            value = inputs["gain"].get(label)
        gain[label] = value
    primary = (p2.get("P2-e") or {}).get("primary") or {}
    payload = {"registered": False, "extension": EXTENSION,
               "check": "P2 addendum A1/A2/A3", "runbook": RUNBOOK_REL,
               "runbook_sha256": RUNBOOK_SHA256,
               "gate": {"status": gate["status"]}, "models": {},
               "ceiling_check": {}, "A2": {}, "A3": None,
               "gain_r2_eq": gain, "P2-e_primary": (p2.get("P2-e") or {}).get("class"),
               "P2-e_primary_numbers": {
                   level: {"rho_s": (primary.get(level) or {}).get("rho_s"),
                           "p": (primary.get(level) or {}).get("p"),
                           "n": (primary.get(level) or {}).get("n")}
                   for level in ("K5", "K15", "K30")},
               "unavailable": [], "stops": [], "wall_seconds": None}
    ceilings = {}
    for label in CORE_MODELS:
        full_cells = model_cells(inputs, label, exact=False)
        exact_cells = model_cells(inputs, label, exact=True)
        entry = {"C0": c0_reading(inputs, full_cells), "cells": {}}
        for name in CONDITIONS:
            data = full_cells["cells"].get(name) or {}
            p2_cell = (((p2.get("models") or {}).get(label) or {}).get(
                "conditions", {}) or {}).get(name) or {}
            if data.get("status") in ("unavailable", "no events"):
                entry["cells"][name] = data
                payload["unavailable"].append({"model": label, "condition": name,
                                               "status": data.get("status")})
                continue
            full = ceiling_cell(data, frozen_vertex=full_cells["frozen"],
                                p2_cell=p2_cell)
            exact_data = exact_cells["cells"].get(name) or {}
            if exact_data.get("status") in ("unavailable", "no events"):
                exact = exact_data
            else:
                exact = ceiling_cell(exact_data,
                                     frozen_vertex=exact_cells["frozen"],
                                     p2_cell=p2_cell.get("exact_subset_reading") or {})
            full["exact_subset_reading"] = exact
            full["P2-f_original"] = p2_cell.get("P2-f")
            refit = p2_cell.get("refit") or {}
            latent = p2_cell.get("latent") or {}
            delta = p2_cell.get("delta_c_m")
            rho_r3 = (None if not refit.get("v_R3_m") or not delta
                      else float((float(p2_cell["v_c_m"]) - float(refit["v_R3_m"]))
                                 / float(delta)))
            full["P2-committed"] = {
                "rho_R2": latent.get("rho_R2"), "rho_R3": rho_r3,
                "r2_eq": latent.get("r2_eq"),
                "R_c_m": refit.get("R_c_m"),
                "v_R3_m": refit.get("v_R3_m"),
                "information_destroying": refit.get("information_destroying")}
            full["P2-f_u"] = p2f_u(p2_cell, full)
            entry["cells"][name] = full
            ceilings.setdefault(label, {})[name] = full
        payload["models"][label] = entry
        log(f"[p2a] A1 {label}")
    payload["A2"] = _p2f_summary(payload["models"])
    payload["A3"] = eta_readings(p2, ceilings, gain)
    payload["ceiling_check"] = _ceiling_summary(payload["models"])
    check = payload["ceiling_check"]
    expected_cells = len(CORE_MODELS) * len(CONDITIONS) * 2 - len(CORE_MODELS)
    payload["expected_cells_min"] = int(expected_cells)
    payload["complete"] = bool(not payload["unavailable"]
                               and check["n_cells_checked"] >= expected_cells
                               and all((entry["cells"].get(name) or {}).get(
                                   "status") not in ("no events", "unavailable")
                                   for entry in payload["models"].values()
                                   for name in CONDITIONS))
    if check["n_violation_gt_5mm_ci"]:
        payload["stops"].append(f"S4: {check['n_violation_gt_5mm_ci']} ceiling cells "
                                "have v_R2 below v_R3u by > 5 mm with a CI excluding 0")
    if check["recomputed_mismatch"]:
        payload["stops"].append(f"A1: {len(check['recomputed_mismatch'])} cells' "
                                "recomputed v_R2 differs from the committed p2.json "
                                "by more than 1e-9")
    if not payload["complete"]:
        payload["stops"].append("A1/A2: incomplete (missing or unavailable cells)")
    payload["pass"] = bool(not payload["stops"])
    payload["wall_seconds"] = round(time.time() - started, 3)
    return payload

def _p2f_summary(models) -> dict:
    out = {}
    for name in CONDITIONS:
        rungs, beyond, n = {}, 0, 0
        for entry in models.values():
            cell = entry["cells"].get(name) or {}
            block = cell.get("P2-f_u") or {}
            if block.get("status") != "evaluated":
                continue
            n += 1
            rungs[str(block.get("cheapest_rung"))] = rungs.get(
                str(block.get("cheapest_rung")), 0) + 1
            beyond += int(bool(block.get("beyond_refit")))
        out[name] = {"n_cells": n, "cheapest_rung_counts": rungs,
                     "beyond_refit": int(beyond)}
    return out


def _ceiling_summary(models) -> dict:
    """The A1 ceiling check over the full cells and the exact subsets (S4)."""
    n_cells = n_below = 0
    max_abs = 0.0
    mismatches, violations = [], []
    for label, entry in models.items():
        for name, cell in entry["cells"].items():
            scopes = (("full", cell),
                      ("exact", (cell or {}).get("exact_subset_reading") or {}))
            for scope, block in scopes:
                ceiling = (block or {}).get("ceiling") or {}
                if not ceiling:
                    continue
                n_cells += 1
                gap = ceiling.get("v_R2_minus_v_R3u_m")
                if gap is not None and float(gap) < 0:
                    n_below += 1
                if ceiling.get("violation"):
                    violations.append({"model": label, "condition": name,
                                       "scope": scope,
                                       "gap_m": gap, "ci_m": ceiling.get("ci_m")})
                recomputed = ceiling.get("v_R2_recomputed_m")
                committed = ceiling.get("v_R2_committed_m")
                if recomputed is not None and committed is not None:
                    difference = abs(float(recomputed) - float(committed))
                    max_abs = max(max_abs, difference)
                    if difference > 1e-9:
                        mismatches.append({"model": label, "condition": name,
                                           "scope": scope,
                                           "difference": difference,
                                           "recomputed_m": float(recomputed),
                                           "committed_m": float(committed)})
    return {"n_cells_checked": int(n_cells), "n_R2_below_R3u": int(n_below),
            "n_violation_gt_5mm_ci": int(len(violations)),
            "violations": violations,
            "max_abs_recomputed_minus_committed_m": float(max_abs),
            "recomputed_mismatch": mismatches,
            "scopes": "full cell and exact subset",
            "rule": "v_R2 must not be below v_R3u by more than 5 mm with the gap "
                    "CI excluding 0; a larger gap is S4"}


# =============================================================== A4 equivalence
def compare_numbers(left, right, *, path="", tolerance=1e-9, out=None) -> dict:
    """Recursively compare numeric leaves; returns (max diff, count, mismatches)."""
    if out is None:
        out = {"max_abs_difference": 0.0, "n_compared": 0, "mismatches": []}
    if isinstance(left, dict) and isinstance(right, dict):
        for key in left:
            if key in right:
                compare_numbers(left[key], right[key], path=f"{path}/{key}",
                                tolerance=tolerance, out=out)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        for position, (a, b) in enumerate(zip(left, right)):
            compare_numbers(a, b, path=f"{path}/{position}", tolerance=tolerance,
                            out=out)
    elif isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            out["mismatches"].append({"path": path, "left": left, "right": right})
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        out["n_compared"] += 1
        out["max_abs_difference"] = max(out["max_abs_difference"], difference)
        if difference > tolerance:
            out["mismatches"].append({"path": path, "left": float(left),
                                      "right": float(right),
                                      "difference": difference})
    return out


def _cell_equivalence(fresh, committed, path) -> dict:
    """Non-vacuous A4 cell comparison (M8)."""
    result = compare_numbers(fresh, committed, path=path)
    fresh_status = fresh.get("status") if isinstance(fresh, dict) else None
    committed_status = committed.get("status") if isinstance(committed, dict) else None
    statuses_match = (fresh_status is not None or committed_status is not None
                      and fresh_status == committed_status)
    if fresh_status is not None or committed_status is not None:
        if fresh_status == committed_status:
            result["status_match"] = str(fresh_status)
        else:
            result["mismatches"].append({
                "path": path, "reason": "fresh/committed status differ",
                "fresh": fresh_status, "committed": committed_status})
    if (result["n_compared"] == 0 and not result["mismatches"]
            and not (statuses_match and fresh_status in ("no events", "unavailable"))):
        result["mismatches"].append({"path": path,
                                     "reason": "no numeric leaves compared (vacuous)"})
    if fresh_status == "unavailable":
        result["mismatches"].append({"path": path,
                                     "reason": "fresh cell is 'unavailable'"})
    return result


def run_equivalence(root=None, *, config, gen, geometry, data_root=None,
                    log=print) -> dict:
    """A4: the folded code must reproduce the committed X3-P energies and exact cells."""
    root = _resolve_root(root)
    gate = assert_authorised("equivalence", root=root)
    started = time.time()
    p2 = committed_p2(root)
    x3 = committed_x3p(root)
    inputs = load_inputs(root, config=config, gen=gen, geometry=geometry,
                         data_root=data_root, log=log)
    payload = {"registered": False, "extension": EXTENSION,
               "check": "A4 fold equivalence", "gate": {"status": gate["status"]},
               "exact_subset": {}, "x3p_energy": {}, "wall_seconds": None,
               "tolerance": 1e-9}
    # (a) exact-subset readings with the folded _cell_for_model
    exact_rows = {}
    for label in CORE_MODELS:
        cache = p2x.embedding_cache(label)
        n_cache = p2x.n_embedding_cache(label)
        cells = p2x._cell_for_model(label, cache, n_cache, inputs["drift"],
                                    inputs["infos"], inputs["raw"], inputs["hbar"],
                                    inputs["geometry"], exact=True,
                                    log=lambda *a, **k: None)
        for name in CONDITIONS:
            committed = ((((p2.get("models") or {}).get(label) or {}).get(
                "conditions", {}) or {}).get(name) or {}).get(
                "exact_subset_reading") or {}
            compare = _cell_equivalence(cells.get(name), committed,
                                        f"{label}/{name}")
            exact_rows[f"{label}__{name}"] = compare
        log(f"[p2a] equivalence exact {label}")
    payload["exact_subset"] = {
        "cells": len(exact_rows),
        "max_abs_difference": max((row["max_abs_difference"]
                                   for row in exact_rows.values()), default=0.0),
        "n_compared": sum(row["n_compared"] for row in exact_rows.values()),
        "n_cells_with_mismatch": sum(1 for row in exact_rows.values()
                                     if row["mismatches"]),
        "mismatches": [{"cell": key, **row} for key, row in exact_rows.items()
                       if row["mismatches"]][:20]}
    # (b) the X3-P energy blocks with the folded _x3p_blocks
    scratch = Path("/work/dwong/p2a_equiv_ext")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "x0").mkdir(exist_ok=True)
    for name in ("plan.json",):
        source = P2_EXT_ROOT / "x0" / name
        if source.is_file():
            (scratch / "x0" / name).write_bytes(source.read_bytes())
    if (P2_EXT_ROOT / "x2b").is_dir():
        (scratch / "x2b").mkdir(exist_ok=True)
        for source in (P2_EXT_ROOT / "x2b").glob("*.json"):
            (scratch / "x2b" / source.name).write_bytes(source.read_bytes())
    original_root = p2x.EXT_ROOT
    original_gate = p2x.assert_authorised
    original_jobs = p2x.JOBS_RECORD
    original_closure = p2x.assert_closure
    p2x.EXT_ROOT = scratch
    p2x.JOBS_RECORD = scratch / "jobs.jsonl"
    # scoped: the inner x3p must run under the addendum's own acceptance/closure
    p2x.assert_authorised = lambda *a, **k: {"status": "equivalence"}
    p2x.assert_closure = lambda *a, **k: {"pass": True,
                                          "status": "equivalence"}
    try:
        fresh = p2x.x3p(root, config=config, gen=gen, arms_config=None,
                        geometry=geometry, data_root=data_root, device="cpu",
                        labels=list(er.MAE_MODELS), log=log)
    finally:
        p2x.EXT_ROOT = original_root
        p2x.JOBS_RECORD = original_jobs
        p2x.assert_authorised = original_gate
        p2x.assert_closure = original_closure
    comparisons = {}
    for label in er.MAE_MODELS:
        comparisons[label] = _cell_equivalence(
            (fresh.get("models") or {}).get(label),
            (x3.get("models") or {}).get(label), label)
    payload["x3p_energy"] = {
        "models": len(comparisons),
        "max_abs_difference": max((row["max_abs_difference"]
                                   for row in comparisons.values()), default=0.0),
        "n_compared": sum(row["n_compared"] for row in comparisons.values()),
        "mismatches": [{"model": key, **row} for key, row in comparisons.items()
                       if row["mismatches"]][:20]}
    exact_min = 1000
    x3p_min = 100
    payload["exact_subset"]["n_compared_min"] = exact_min
    payload["x3p_energy"]["n_compared_min"] = x3p_min
    payload["pass"] = bool(
        not payload["exact_subset"]["n_cells_with_mismatch"]
        and payload["exact_subset"]["n_compared"] >= exact_min
        and not payload["x3p_energy"]["mismatches"]
        and payload["x3p_energy"]["n_compared"] >= x3p_min)
    payload["wall_seconds"] = round(time.time() - started, 3)
    return payload


# ================================================================ report §6
def _fmt(value, digits=6):
    return er._fmt(value, digits)


def _means(models, name, key, sub=None):
    values = []
    for entry in models.values():
        cell = entry["cells"].get(name) or {}
        value = (cell.get(sub) or {}).get(key) if sub else cell.get(key)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _mean(values):
    return float(np.mean(values)) if values else None


def p2a_summary(p2a, p2, equivalence=None, shift=None, extra=None) -> dict:
    models = p2a["models"]
    per_condition = {}
    for name in CONDITIONS:
        deltas, refit, fair, rho2, rho3, rho3u = [], [], [], [], [], []
        destroying_r3 = destroying_r3u = 0
        for entry in models.values():
            cell = entry["cells"].get(name) or {}
            if not cell or cell.get("status") in ("no events", "unavailable"):
                continue
            committed = cell.get("P2-committed") or {}
            deltas.append(float(cell["delta_c_m"]))
            if committed.get("R_c_m") is not None:
                refit.append(float(committed["R_c_m"]))
            fair.append(float(cell["R_c_u_m"]))
            if committed.get("rho_R2") is not None:
                rho2.append(float(committed["rho_R2"]))
            if committed.get("rho_R3") is not None:
                rho3.append(float(committed["rho_R3"]))
            if cell.get("rho_R3u") is not None:
                rho3u.append(float(cell["rho_R3u"]))
            destroying_r3 += int(bool(committed.get("information_destroying")))
            destroying_r3u += int(bool(cell.get("information_destroying_u")))
        per_condition[name] = {
            "n_cells": len(deltas), "delta_c_m": _mean(deltas),
            "R_c_m": _mean(refit), "R_c_u_m": _mean(fair),
            "n_information_destroying_R3": int(destroying_r3),
            "n_information_destroying_R3u": int(destroying_r3u),
            "rho_R2_mean": _mean(rho2), "rho_R3_mean": _mean(rho3),
            "rho_R3u_mean": _mean(rho3u)}
    per_model = {}
    for label, entry in models.items():
        row = {"C0": entry.get("C0"), "model": label,
               "gain": (p2a.get("gain_r2_eq") or {}).get(label), "levels": {}}
        for level in ("K5", "K15", "K30"):
            cell = entry["cells"].get(level) or {}
            committed = cell.get("P2-committed") or {}
            rho2 = committed.get("rho_R2")
            r3u = cell.get("rho_R3u")
            row["levels"][level] = {
                "rho_R2": rho2, "rho_R3": committed.get("rho_R3"),
                "rho_R3u": r3u,
                "eta": (None if rho2 is None or r3u is None or abs(float(r3u)) <= 0
                        else float(rho2) / float(r3u))}
        per_model[label] = row
    return {"registered": False, "extension": EXTENSION,
            "runbook_sha256": RUNBOOK_SHA256,
            "primary": p2a.get("P2-e_primary"),
            "primary_numbers": p2a.get("P2-e_primary_numbers"),
            "per_condition": per_condition, "per_model": per_model,
            "ceiling_check": p2a.get("ceiling_check"),
            "A2": p2a.get("A2"), "A3": p2a.get("A3"),
            "equivalence": equivalence, "A6": shift, "extra": extra or {}}


def _readme(summary) -> str:
    lines = ["# Paper 3 P2 addendum — fair refit ceiling, P2-f labels, F1/F2 fold, hygiene",
             "",
             f"Runbook `{RUNBOOK_REL}` (sha256 `{RUNBOOK_SHA256}`); extension "
             f"`{EXTENSION}`; CPU only.",
             "", "## 1. Per condition (means over the 15 models)", "",
             "| condition | Delta_c | R_c | R_c^u | destroying R3 | destroying R3u | "
             "rho_R2 | rho_R3 | rho_R3u |", "|---|---|---|---|---|---|---|---|---|"]
    for name, row in summary["per_condition"].items():
        n = row.get("n_cells") or 15
        lines.append(f"| {name} | {_fmt(row['delta_c_m'])} | {_fmt(row['R_c_m'])} | "
                     f"{_fmt(row['R_c_u_m'])} | "
                     f"{row['n_information_destroying_R3']}/{n} | "
                     f"{row['n_information_destroying_R3u']}/{n} | "
                     f"{_fmt(row['rho_R2_mean'])} | {_fmt(row['rho_R3_mean'])} | "
                     f"{_fmt(row['rho_R3u_mean'])} |")
    lines += ["", "## 2. Per model at K5/K15/K30", "",
              "| model | rho_R2 K5 | rho_R3 K5 | rho_R3u K5 | eta K5 | "
              "rho_R2 K15 | rho_R3 K15 | rho_R3u K15 | eta K15 | "
              "rho_R2 K30 | rho_R3 K30 | rho_R3u K30 | eta K30 | "
              "C0 diff | C0 CI |", "|" + "---|" * 15]
    for label, row in summary["per_model"].items():
        c0 = row.get("C0") or {}
        cells = [row["levels"][level] for level in ("K5", "K15", "K30")]
        rendered = " | ".join(
            f"{_fmt(cell['rho_R2'])} | {_fmt(cell['rho_R3'])} | "
            f"{_fmt(cell['rho_R3u'])} | {_fmt(cell['eta'])}" for cell in cells)
        lines.append(f"| {label} | {rendered} | {_fmt(c0.get('difference_m'))} | "
                     f"{_fmt(c0.get('ci_m'))} |")
    ceiling = summary.get("ceiling_check") or {}
    lines += ["", "## 3. Ceiling check (A1)", "",
              f"- cells checked ({ceiling.get('scopes')}): {ceiling.get('n_cells_checked')}; "
              f"v_R2 below v_R3u in {ceiling.get('n_R2_below_R3u')}; beyond 5 mm "
              f"with a CI excluding 0: {ceiling.get('n_violation_gt_5mm_ci')}; max "
              f"|v_R2_recomputed - v_R2_committed| = "
              f"{_fmt(ceiling.get('max_abs_recomputed_minus_committed_m'))} m; "
              f"recomputed mismatches > 1e-9: {len(ceiling.get('recomputed_mismatch') or [])}",
              "", "## 4. P2-f_u cheapest rung (A2)", ""]
    for name, row in (summary.get("A2") or {}).items():
        lines.append(f"- {name}: {row['cheapest_rung_counts']}, beyond_refit "
                     f"{row['beyond_refit']}/{row['n_cells']}")
    a3 = summary.get("A3") or {}
    primary = summary.get("primary") or {}
    numbers = summary.get("primary_numbers") or {}
    primary_bits = " / ".join(
        f"{level}: rho_s={_fmt((numbers.get(level) or {}).get('rho_s'))}, "
        f"p={_fmt((numbers.get(level) or {}).get('p'))}, "
        f"n={(numbers.get(level) or {}).get('n')}" for level in ("K5", "K15", "K30"))
    lines += ["", "## 5. P2-e (primary unchanged) and A3 (post hoc / exploratory)", "",
              f"- primary P2-e verdict: `{primary.get('verdict')}` (unchanged); "
              f"{primary_bits}"]
    for level, block in (a3.get("levels") or {}).items():
        lines.append(f"- eta {level} (exploratory): rho_s={_fmt(block.get('rho_s'))}, "
                     f"p={_fmt(block.get('p'))}, n={block.get('n')}")
    secondary = a3.get("secondary_r2eq_K15") or {}
    lines.append(f"- secondary Spearman(gain R2_eq, R2_eq_K15) (exploratory, "
                 f"permutation p): rho_s={_fmt(secondary.get('rho_s'))}, "
                 f"p={_fmt(secondary.get('p'))}, n={secondary.get('n')}")
    for level, block in (a3.get("sensitivity_no_fix") or {}).items():
        lines.append(f"- sensitivity primary without fix9000/fix9001 at {level}: "
                     f"rho_s={_fmt(block.get('rho_s'))}, p={_fmt(block.get('p'))}, "
                     f"n={block.get('n')}")
    lines.append(f"- A3 rule: {(a3.get('rule') or {}).get('verdict')}")
    shift = summary.get("A6") or {}
    lines += ["", "## 6. A6 X3-P shift shares", "",
              "| model | r | share removed | control mean | top-r gain Frobenius |",
              "|---|---|---|---|---|"]
    for label, entry in (shift.get("models") or {}).items():
        if entry.get("status") != "evaluated":
            continue
        for rank, row in (entry.get("ranks") or {}).items():
            lines.append(f"| {label} | {rank} | {_fmt(row.get('share_removed'))} | "
                         f"{_fmt(row.get('share_removed_control_mean'))} | "
                         f"{_fmt(row.get('gain_frob_share_top_r'))} |")
    extra = summary.get("extra") or {}
    lines += ["", "## 7. Resources, tests, hygiene", ""]
    for key, value in extra.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## 8. Record corrections (A7)", "",
              "- P2-c is a transform check, not a finding: R1 maps T1.0w to T1.0 by "
              "construction (|v_R1 - v_T1.0| ~ 1e-6 m), so the recovery is the "
              "transform working, not an empirical result.",
              "- The fix9000 T0.5 cell is flagged information-destroying only because "
              "its refit (3.18 mm) is worse than no repair (2.94 mm); it is recorded "
              "as refit noise.", ""]
    return "\n".join(lines)


def render_figures(summary, out_dir, *, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures, skipped = [], []

    def ceiling_plot(axes):
        p2a_cells = summary.get("_cells") or []
        xs = [cell["v_R3u_m"] for cell in p2a_cells
              if cell.get("v_R2_m") is not None]
        ys = [cell["v_R2_m"] for cell in p2a_cells
              if cell.get("v_R2_m") is not None]
        if xs:
            axes.scatter(xs, ys, s=12, label="cell")
            low = min(min(xs), min(ys)); high = max(max(xs), max(ys))
            axes.plot([low, high], [low, high], color="grey", linewidth=0.8,
                      label="v_R2 = v_R3u")
        axes.set_xlabel("v_R3u (m)")
        axes.set_ylabel("v_R2 (m)")
        axes.legend(fontsize=7)

    def eta_plot(axes):
        for level, color in (("K5", "tab:blue"), ("K15", "tab:orange"),
                             ("K30", "tab:green")):
            xs, ys = [], []
            for row in summary.get("per_model", {}).values():
                cell = (row.get("levels") or {}).get(level) or {}
                if cell.get("eta") is not None and row.get("gain") is not None:
                    xs.append(row["gain"]); ys.append(cell["eta"])
            if xs:
                axes.scatter(xs, ys, s=14, color=color, label=level)
        axes.set_xlabel("gain R2_eq")
        axes.set_ylabel("eta = rho_R2 / rho_R3u")
        axes.legend(fontsize=7)

    for name, draw in (("p2a_ceiling.png", ceiling_plot),
                       ("p2a_eta_scatter.png", eta_plot)):
        path = out_dir / name
        if er._plot(path, draw):
            figures.append(str(path))
        else:
            skipped.append(name)
    return {"figures": figures, "skipped": skipped}


def _execution_payload(root, p2a, equivalence, shift, extra, stops) -> dict:
    extra = extra or {}
    return {
        "registered": False, "extension": EXTENSION,
        "runbook": RUNBOOK_REL, "runbook_sha256": RUNBOOK_SHA256,
        "impl_commit": extra.get("impl_commit"),
        "acceptance": ACCEPTANCE_REL,
        "stages": extra.get("stages"),
        "tests": extra.get("tests"),
        "resources": extra.get("resources"),
        "hygiene": extra.get("hygiene"),
        "a1_pass": bool(p2a.get("pass")),
        "ceiling_check": p2a.get("ceiling_check"),
        "unavailable": p2a.get("unavailable"),
        "a4_equivalence_pass": bool(equivalence.get("pass")),
        "a6_shift_pass": bool(shift.get("pass")),
        "stops": list(stops),
        "complete": bool(p2a.get("complete") and equivalence.get("pass")
                         and shift.get("pass")),
        "results_dir": RESULTS_DIR,
        "note": "CPU-only addendum; no GPU job, no training, no data generation; "
                "the P2 primary verdict is unchanged (mixed)."}


def assemble_report(root=None, *, p2a, equivalence, shift, extra=None,
                    log=print) -> dict:
    root = _resolve_root(root)
    gate = assert_authorised("report", root=root)
    out = root / RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    p2 = committed_p2(root)
    stops = []
    ceiling = p2a.get("ceiling_check") or {}
    if not p2a.get("pass"):
        stops.append("A1/A2/A3: " + "; ".join(p2a.get("stops")
                                              or ["A1/A2/A3 failed"]))
    if not p2a.get("complete"):
        stops.append("A1/A2: the readings are incomplete")
    if ceiling.get("n_violation_gt_5mm_ci"):
        stops.append(f"S4: {ceiling['n_violation_gt_5mm_ci']} ceiling cells")
    if ceiling.get("recomputed_mismatch"):
        stops.append("A1: recomputed v_R2 differs from the committed p2.json")
    if not equivalence.get("pass"):
        stops.append("A4: the fold equivalence failed (S3)")
    if not shift.get("pass"):
        stops.append("A6: the shift readings are incomplete")
    payload = {"registered": False, "extension": EXTENSION,
               "check": "P2 addendum report §6", "results_dir": str(out),
               "gate": {"status": gate["status"]}, "stops": stops,
               "written": False, "complete": False, "pass": False,
               "figures": None, "wall_seconds": None}
    execution = _execution_payload(root, p2a, equivalence, shift, extra, stops)
    if stops:
        # S4/S3: write nothing to READINGS, only the execution record of the stop;
        # remove any stale output of an earlier report run so nothing is misread
        for name in ("READINGS_SUMMARY.md", "p2a.json", "equivalence.json",
                     "x3p_shift.json", "SUMMARY.json", "p2a_ceiling.png",
                     "p2a_eta_scatter.png", "report.json"):
            stale = out / name
            if stale.is_file():
                stale.unlink()
        write_json(out / "EXECUTION.json", execution)
        log("[p2a] report stopped before writing READINGS: " + "; ".join(stops))
        return payload
    summary = p2a_summary(p2a, p2, equivalence=equivalence, shift=shift,
                          extra=extra)
    cells = []
    for label, entry in p2a["models"].items():
        for name, cell in entry["cells"].items():
            v_r2 = (cell.get("ceiling") or {}).get("v_R2_recomputed_m")
            if v_r2 is None:
                continue
            cells.append({"model": label, "condition": name,
                          "v_R3u_m": cell.get("v_R3u_m"), "v_R2_m": v_r2,
                          "rho_R3u": cell.get("rho_R3u")})
    summary["_cells"] = cells
    write_json(out / "p2a.json", p2a)
    write_json(out / "equivalence.json", equivalence)
    write_json(out / "x3p_shift.json", shift)
    write_json(out / "SUMMARY.json", {key: value for key, value in summary.items()
                                      if key != "_cells"})
    (out / "READINGS_SUMMARY.md").write_text(_readme(summary))
    figures = render_figures(summary, out, log=log)
    execution["complete"] = True
    write_json(out / "EXECUTION.json", execution)
    payload["written"] = True
    payload["complete"] = True
    payload["pass"] = True
    payload["figures"] = figures
    write_json(out / "report.json", payload)
    return payload
