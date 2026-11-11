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
    if design.ndim == 2 and design.shape[1] == 2 and np.allclose(design[:, 1], 1.0):
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
    index_nt, index_ct, _ = p2x._align_subset(n_train, c_train)
    dz_train = (np.asarray(c_train["events"], float)
                - np.asarray(n_train["events"], float)[index_nt])
    index_ne, index_ce, runs_e = p2x._align_subset(n_test, c_test)
    dz_test = (np.asarray(c_test["events"], float)
               - np.asarray(n_test["events"], float)[index_ne])
    fit = er.equivariance_fit(
        u_train, dz_train, u_test, dz_test,
        run_ids_train=np.asarray(c_train["run_ids"]),
        run_ids_test=np.asarray(c_test["run_ids"]),
        kind=("occupancy" if condition.kind == "dead" else "global"))
    a_raw = er.raw_map(fit)
    repaired = dict(c_test)
    repaired["events"] = (np.asarray(c_test["events"], float)
                          - u_test @ np.asarray(a_raw, float).T)
    err_r2 = frozen_errors(n_train, repaired, frozen_vertex)[index_ce]
    v_r2 = float(np.mean(err_r2))
    ceiling_difference = _diff(err_r2, err_r3u, runs_e)
    ceiling_gap = float(v_r2 - v_r3u)
    # raw-std refit on [raw_c, u_c], the encoder-independent bound
    raw = {"status": "not available"}
    if data.get("raw_c_train") is not None and data.get("raw_c_test") is not None \
            and data.get("raw_n_train") is not None and data.get("raw_n_test") is not None:
        raw_train = augmented(data["raw_c_train"], u_train)
        raw_test = augmented(data["raw_c_test"], u_test)
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
        raw_difference = _diff(raw_c_aligned, raw_r3u_errors, runs_raw)
        raw = {"status": "evaluated", "v_N_m": v_raw_n, "v_c_m": v_raw_c,
               "v_R3u_m": float(v_raw_r3u),
               "R_c_u_m": float(v_raw_r3u - v_raw_n),
               "R_c_u_ci_m": raw_difference["ci95"],
               "delta_c_m": float(v_raw_c - v_raw_n),
               "rho_R3u": (None if abs(v_raw_c - v_raw_n) <= 0 else
                           float((v_raw_c - v_raw_r3u) / (v_raw_c - v_raw_n)))}
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
        "raw": raw,
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
    payload = read_json(_resolve_root(root) / RESULTS_DIR / "p2.json", None)
    if payload is None:
        raise RuntimeError("A1 needs the committed results/jepa/20261109_p2/p2.json")
    return payload


def committed_x3p(root=None) -> dict:
    payload = read_json(_resolve_root(root) / RESULTS_DIR / "x3p.json", None)
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
    return {"status": "evaluated", "rho_R0": 0.0, "rho_R1": r1, "rho_R2": r2,
            "rho_R3": r3, "rho_R3u": r3u, "target": target,
            "cheapest_rung": cheapest, "beyond_refit": beyond,
            "note": ("beyond refit (R4/R5 territory)" if beyond else None)}


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
    # sensitivity: the primary Spearman without fix9000/fix9001 (n = 13)
    x, y, used = p2x._eligible_cells(p2_payload.get("models") or {}, "K5")
    keep = [position for position, label in enumerate(used)
            if label not in ("fix9000", "fix9001")]
    xs = [float(x[position]) for position in keep]
    ys = [float(y[position]) for position in keep]
    out["sensitivity_no_fix"] = (
        permutation_spearman_p2a(xs, ys, level="no-fix-K5") if len(xs) >= 3
        else {"rho_s": None, "n": len(xs)})
    out["sensitivity_no_fix"]["models"] = [used[position] for position in keep]
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
               "models": {}, "wall_seconds": None}
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
    payload = {"registered": False, "extension": EXTENSION,
               "check": "P2 addendum A1/A2/A3", "runbook": RUNBOOK_REL,
               "runbook_sha256": RUNBOOK_SHA256,
               "gate": {"status": gate["status"]}, "models": {},
               "ceiling_check": {}, "A2": {}, "A3": None, "wall_seconds": None}
    ceilings = {}
    for label in CORE_MODELS:
        full_cells = model_cells(inputs, label, exact=False)
        exact_cells = model_cells(inputs, label, exact=True)
        entry = {"C0": c0_reading(inputs, full_cells), "cells": {}}
        for name in CONDITIONS:
            p2_cell = (((p2.get("models") or {}).get(label) or {}).get(
                "conditions", {}) or {}).get(name) or {}
            full = ceiling_cell(full_cells["cells"][name],
                                frozen_vertex=full_cells["frozen"], p2_cell=p2_cell)
            committed_exact = p2_cell.get("exact_subset_reading") or {}
            exact = ceiling_cell(exact_cells["cells"][name],
                                 frozen_vertex=exact_cells["frozen"],
                                 p2_cell=committed_exact)
            full["exact_subset_reading"] = exact
            full["P2-f_u"] = p2f_u(p2_cell, full)
            entry["cells"][name] = full
            ceilings.setdefault(label, {})[name] = full
        payload["models"][label] = entry
        log(f"[p2a] A1 {label}")
    payload["gain_r2_eq"] = inputs["gain"]
    payload["A2"] = _p2f_summary(payload["models"])
    payload["A3"] = eta_readings(p2, ceilings, inputs["gain"])
    payload["ceiling_check"] = _ceiling_summary(payload["models"])
    payload["pass"] = bool(not payload["ceiling_check"].get("violations"))
    payload["complete"] = all(
        entry["cells"].get(name, {}).get("status") != "no events"
        for entry in payload["models"].values() for name in CONDITIONS)
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
    n_cells = n_any = n_violation = 0
    max_gap = None
    max_abs_recomputed = 0.0
    recomputed_mismatch = []
    for label, entry in models.items():
        for name, cell in entry["cells"].items():
            ceiling = cell.get("ceiling") or {}
            if not ceiling:
                continue
            n_cells += 1
            gap = ceiling.get("v_R2_minus_v_R3u_m")
            if gap is not None:
                if gap < 0:
                    n_any += 1
                max_gap = gap if max_gap is None else min(max_gap, gap)
            if ceiling.get("violation"):
                n_violation += 1
            absolute = ceiling.get("check_abs_lt_1e-9")
            if absolute is False:
                recomputed_mismatch.append({"model": label, "condition": name,
                                            "gap": ceiling.get(
                                                "v_R2_recomputed_m")})
    return {"n_cells": int(n_cells), "n_R2_below_R3u": int(n_any),
            "n_violation_gt_5mm_ci": int(n_violation),
            "min_gap_m": max_gap,
            "max_abs_recomputed_minus_committed_m": max_abs_recomputed,
            "recomputed_mismatch": recomputed_mismatch,
            "violations": recomputed_mismatch if False else [],
            "rule": "v_R2 must not be below v_R3u by more than 5 mm with a CI "
                    "excluding 0; a larger gap is S4"}


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
            compare = compare_numbers(cells.get(name), committed,
                                      path=f"{label}/{name}")
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
    p2x.EXT_ROOT = scratch
    p2x.assert_authorised = lambda *a, **k: {"status": "equivalence"}
    try:
        fresh = p2x.x3p(root, config=config, gen=gen, arms_config=None,
                        geometry=geometry, data_root=data_root, device="cpu",
                        labels=list(er.MAE_MODELS), log=log)
    finally:
        p2x.EXT_ROOT = original_root
        p2x.assert_authorised = original_gate
    comparisons = {}
    for label in er.MAE_MODELS:
        comparisons[label] = compare_numbers(
            (fresh.get("models") or {}).get(label),
            (x3.get("models") or {}).get(label), path=label)
    payload["x3p_energy"] = {
        "models": len(comparisons),
        "max_abs_difference": max((row["max_abs_difference"]
                                   for row in comparisons.values()), default=0.0),
        "n_compared": sum(row["n_compared"] for row in comparisons.values()),
        "mismatches": [{"model": key, **row} for key, row in comparisons.items()
                       if row["mismatches"]][:20]}
    payload["pass"] = bool(not payload["exact_subset"]["n_cells_with_mismatch"]
                           and not payload["x3p_energy"]["mismatches"])
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


def p2a_summary(p2a, p2, equivalence=None, shift=None, extra=None) -> dict:
    models = p2a["models"]
    per_condition = {}
    for name in CONDITIONS:
        per_condition[name] = {
            "delta_c_m": _means(models, name, "delta_c_m"),
            "R_c_m": _means(models, name, "R_c_m", sub="refit"),
            "R_c_u_m": _means(models, name, "R_c_u_m"),
            "n_information_destroying_R3": int(sum(
                1 for entry in models.values()
                if ((entry["cells"].get(name) or {}).get("refit") or {}).get(
                    "information_destroying"))),
            "n_information_destroying_R3u": int(sum(
                1 for entry in models.values()
                if (entry["cells"].get(name) or {}).get(
                    "information_destroying_u"))),
            "rho_R2_mean": _means(models, name, "rho_R2", sub="latent"),
            "rho_R3_mean": _means(models, name, "R_c_m", sub="refit"),
            "rho_R3u_mean": _means(models, name, "rho_R3u"),
        }
        if per_condition[name]["R_c_m"] is not None and \
                per_condition[name]["delta_c_m"]:
            per_condition[name]["rho_R3_mean"] = float(
                (per_condition[name]["R_c_m"]
                 / per_condition[name]["delta_c_m"]) if per_condition[name][
                    "delta_c_m"] else None)
        else:
            per_condition[name]["rho_R3_mean"] = None
    gain_table = p2a.get("gain_r2_eq") or (p2a.get("extra") or {}).get("gain") or {}
    per_model = {}
    for label, entry in models.items():
        row = {"C0": entry.get("C0"), "gain": gain_table.get(label),
               "model": label}
        for level in ("K5", "K15", "K30"):
            cell = entry["cells"].get(level) or {}
            rho2 = (cell.get("latent") or {}).get("rho_R2")
            r3u = cell.get("rho_R3u")
            row[level] = {"rho_R2": rho2, "rho_R3u": r3u,
                          "eta": (None if rho2 is None or not r3u
                                  or abs(r3u) <= 0 else float(rho2) / float(r3u))}
        per_model[label] = row
    return {"registered": False, "extension": EXTENSION,
            "runbook_sha256": RUNBOOK_SHA256,
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
        lines.append(f"| {name} | {_fmt(row['delta_c_m'])} | {_fmt(row['R_c_m'])} | "
                     f"{_fmt(row['R_c_u_m'])} | {row['n_information_destroying_R3']}/15 | "
                     f"{row['n_information_destroying_R3u']}/15 | "
                     f"{_fmt(row['rho_R2_mean'])} | {_fmt(row['rho_R3_mean'])} | "
                     f"{_fmt(row['rho_R3u_mean'])} |")
    lines += ["", "## 2. Per model at K5/K15/K30", "",
              "| model | rho_R2 K5 | rho_R3u K5 | eta K5 | rho_R2 K15 | rho_R3u K15 | "
              "eta K15 | rho_R2 K30 | rho_R3u K30 | eta K30 | C0 diff | C0 CI |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, row in summary["per_model"].items():
        c0 = row.get("C0") or {}
        cells = [row[level] for level in ("K5", "K15", "K30")]
        lines.append(f"| {label} | " + " | ".join(
            f"{_fmt(cell['rho_R2'])} | {_fmt(cell['rho_R3u'])} | {_fmt(cell['eta'])}"
            for cell in cells) + f" | {_fmt(c0.get('difference_m'))} | "
            f"{_fmt(c0.get('ci_m'))} |")
    ceiling = summary.get("ceiling_check") or {}
    lines += ["", "## 3. Ceiling check (A1)", "",
              f"- cells: {ceiling.get('n_cells')}; v_R2 below v_R3u in "
              f"{ceiling.get('n_R2_below_R3u')}; beyond 5 mm with a CI excluding 0: "
              f"{ceiling.get('n_violation_gt_5mm_ci')}; min gap "
              f"{_fmt(ceiling.get('min_gap_m'))} m",
              "", "## 4. P2-f_u cheapest rung (A2)", ""]
    for name, row in (summary.get("A2") or {}).items():
        lines.append(f"- {name}: {row['cheapest_rung_counts']}, beyond_refit "
                     f"{row['beyond_refit']}/{row['n_cells']}")
    a3 = summary.get("A3") or {}
    lines += ["", "## 5. P2-e (primary unchanged) and A3 secondary", "",
              "- primary P2-e verdict: `mixed` (unchanged); "
              "rho_s K5/K15/K30 = 0.282/0.218/0.286 (p 0.156/0.208/0.147, n=15)"]
    for level, block in (a3.get("levels") or {}).items():
        lines.append(f"- eta {level}: rho_s={_fmt(block.get('rho_s'))}, "
                     f"p={_fmt(block.get('p'))}, n={block.get('n')}")
    secondary = a3.get("secondary_r2eq_K15") or {}
    lines.append("- secondary Spearman(gain R2_eq, R2_eq_K15): "
                 f"rho_s={_fmt(secondary.get('rho_s'))}, "
                 f"p={_fmt(secondary.get('p'))}, n={secondary.get('n')}")
    sensitivity = a3.get("sensitivity_no_fix") or {}
    lines.append("- sensitivity primary without fix9000/fix9001: "
                 f"rho_s={_fmt(sensitivity.get('rho_s'))}, "
                 f"p={_fmt(sensitivity.get('p'))}, n={sensitivity.get('n')}")
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
    lines.append("")
    return "\n".join(lines)


def render_figures(summary, out_dir, *, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures, skipped = [], []

    def ceiling_plot(axes):
        xs, ys = [], []
        for entry in summary.get("per_model", {}).values():
            pass
        for name, row in summary["per_condition"].items():
            if row.get("R_c_u_m") is not None and row.get("R_c_m") is not None:
                xs.append(row["R_c_u_m"]); ys.append(row["R_c_m"])
        for entry in (summary.get("cells") or {}).values():
            pass
        p2a_cells = summary.get("_cells") or []
        if p2a_cells:
            xs = [cell["v_R3u_m"] for cell in p2a_cells]
            ys = [cell["v_R2_m"] for cell in p2a_cells]
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
                cell = row.get(level) or {}
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


def assemble_report(root=None, *, p2a, equivalence, shift, extra=None,
                    log=print) -> dict:
    root = _resolve_root(root)
    gate = assert_authorised("report", root=root)
    out = root / RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    p2 = committed_p2(root)
    summary = p2a_summary(p2a, p2, equivalence=equivalence, shift=shift,
                          extra=extra)
    cells = []
    for label, entry in p2a["models"].items():
        for name, cell in entry["cells"].items():
            if cell.get("v_R2_m") is None:
                continue
            cells.append({"model": label, "condition": name,
                          "v_R3u_m": cell.get("v_R3u_m"),
                          "v_R2_m": (cell.get("ceiling") or {}).get(
                              "v_R2_recomputed_m"),
                          "rho_R3u": cell.get("rho_R3u")})
    summary["_cells"] = cells
    write_json(out / "p2a.json", p2a)
    write_json(out / "equivalence.json", equivalence)
    write_json(out / "x3p_shift.json", shift)
    write_json(out / "SUMMARY.json", {key: value for key, value in summary.items()
                                      if key != "_cells"})
    (out / "READINGS_SUMMARY.md").write_text(_readme(summary))
    figures = render_figures(summary, out, log=log)
    payload = {"registered": False, "extension": EXTENSION,
               "check": "P2 addendum report §6", "results_dir": str(out),
               "figures": figures, "gate": {"status": gate["status"]},
               "complete": bool(p2a.get("complete")),
               "pass": bool(p2a.get("pass") and equivalence.get("pass")
                            and shift.get("pass")),
               "wall_seconds": None}
    write_json(out / "report.json", payload)
    return payload
