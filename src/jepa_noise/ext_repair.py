# SPDX-License-Identifier: MIT
"""Paper 3 repair-pilot extension (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md``, Part A
(design review v3 PASS).  Extension name ``paper3_repair_pilot_2026-11-06``.

Part A trains nothing and generates no data: X0-a/B are preflight and a
reproduction gate, X1 measures what a charge calibration recovers, X2 measures
equivariance / latent repair / identification, X3 prices invariance (a Fisher
accounting and invariant-variant controls).  The library drives the CLI
``scripts/jepa_ext_repair.py``; every data access is read-only and every probe
set is opened only through ``ext_followup.open_probe_sets`` (pinned manifest
hash first).

Sections
--------
* gate (§9.2 (a)-(c)) and the executed-import closure re-asserted at every mode;
* pinned data, run states and the §3.1 input conditions (``hit_transform_context``);
* the §3.2 embedding cache, probes, metrics and the §3.3 classification rule;
* X0-a preflight and X0-b reproduction/identity check (§4);
* X1 calibration readings (§5);
* X2 equivariance, latent repair and identification (§6);
* X3-a Fisher accounting and P0 (§7.1);
* X3-b invariant variants and LEACE (§7.2);
* report assembly and the cost/cap projections (§8).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import ext_followup as ef
from . import lucid_arms as la
from . import lucid_bombs as lb
from . import lucid_run as lr
from . import point_encoder as pe
from . import probes as pr

__all__ = [
    "EXTENSION", "MODES", "EXT_ROOT", "RESULTS_DIR", "RUNBOOK_REL", "ACCEPTANCE_REL",
    "DECISIONS_REL", "P_FILES", "P_DIRS", "FORCED_IMPORT_EXCEPTIONS", "PINNED_DATA",
    "PROBE_KEYS_S005", "PROBE_KEYS_S030", "CORE_MODELS", "OPTIONAL_MODELS",
    "MAE_MODELS", "X1_PARTICLE_MODELS", "X0B_TARGETS", "NOISY_SIGMAS", "FISHER_SIGMAS",
    "section_zero", "p_paths", "in_p", "acceptance_record", "import_closure",
    "assert_authorised", "manifest_path", "data_status", "require_pinned",
    "open_probe_sets", "run_spec", "run_state", "state_pair", "module_of_pmt",
    "CONDITION_NAMES", "Condition", "condition_name", "build_condition",
    "condition_factor_arrays", "hit_transform_context", "invariant_token_context",
    "install_invariant_token_wrapper", "remove_invariant_token_wrapper",
    "EmbeddingCache", "stream_condition_embeddings", "stream_condition_particles",
    "vertex_metrics", "log_yield_metrics", "energy_metrics", "fit_vertex_probe",
    "fit_log_yield_probe", "fit_energy_probe", "cluster_bootstrap",
    "paired_cluster_bootstrap", "drift_cost", "recovered_fraction", "classify_seeds",
    "part_a_inventory", "preflight", "gpu_projection", "cpu_projection",
    "s1_projection", "entries_preflight", "reproduction", "identity_check",
    "x1", "x1_readings", "occupancy_counts", "run_log_gain_vectors",
    "equivariance_fit", "x2_decomposition",
    "source_estimate", "raw_estimate", "latent_estimate",
    "latent_cv", "whitening_matrix", "identifiability_spectrum", "x2", "fisher_inputs", "crlb_report",
    "fisher_accounting", "p0_check", "raw_photon_features", "raw_std_features",
    "raw_inv_features", "invariant_features", "leace_fit", "leace_erase",
    "leace_cross_covariance", "permutation_maxima", "reduction_rank",
    "random_control_basis", "x3", "x3_readings", "mode_setup", "read_json", "write_json",
    "assemble_report",
]

# ---------------------------------------------------------------- extension
EXTENSION = "paper3_repair_pilot_2026-11-06"
MODES = ("dry-run", "preflight", "reproduce", "x1", "x2", "x3-fisher", "x3", "report")

ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_repair_20261106")
RESULTS_DIR = "results/jepa/20261106_repair"
JOBS_DIR = Path("/ceph/dwong/jepa/jobs")
GUARD_JSON = JOBS_DIR / "guard_deepthought2.json"

RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md"
ACCEPTANCE_REL = "docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json"
ACCEPTANCE_KEYS = ("runbook_sha256", "impl_commit", "review_path", "review_verdict",
                   "lead", "date")
DECISIONS_REL = "docs/jepa/DECISIONS.md"

#: §9.2 (c): the reviewed path set ``<P>``.  ``git diff --quiet <impl_commit> -- <P>``
#: and ``git ls-files --others --exclude-standard -- <P>`` must both be empty.
P_FILES = (
    "src/jepa_noise/__init__.py", "src/jepa_noise/ext_repair.py",
    "src/jepa_noise/ext_followup.py", "src/jepa_noise/ext_readout.py",
    "src/jepa_noise/ext_h12.py", "src/jepa_noise/ext_dejitter.py",
    "src/jepa_noise/ext_arch.py", "src/jepa_noise/lucid_run.py",
    "src/jepa_noise/lucid_arms.py", "src/jepa_noise/lucid_bombs.py",
    "src/jepa_noise/lucid_views.py", "src/jepa_noise/point_encoder.py",
    "src/jepa_noise/bridge_run.py", "src/jepa_noise/train.py",
    "src/jepa_noise/crlb.py", "src/jepa_noise/jepa.py",
    "src/jepa_noise/backbone.py", "src/jepa_noise/linear.py",
    "src/jepa_noise/probes.py", "src/jepa_noise/metrics.py",
    "src/jepa_noise/sigreg.py",
    "scripts/__init__.py", "scripts/jepa_ext_repair.py",
    "scripts/jepa_ext_followup.py", "scripts/jepa_pregrid_checks.py",
    "scripts/jepa_ext_dejitter.py", "scripts/jepa_ext_readout.py",
    RUNBOOK_REL,
)
P_DIRS = ("src/noise_module_lucid/", "src/jepa_noise/configs/")
#: Modules outside ``<P>`` that a ``<P>`` module forces into the executed import
#: closure and that this implementation can never edit (same set as
#: ``ext_followup.FORCED_IMPORT_EXCEPTIONS``).
FORCED_IMPORT_EXCEPTIONS = ef.FORCED_IMPORT_EXCEPTIONS

# --------------------------------------------------------------- provenance
CONFIG_REL = ef.CONFIG_REL
GEN_REL = ef.GEN_REL
ARMS_REL = ef.ARMS_REL
GEOMETRY_DEFAULT = ef.GEOMETRY_DEFAULT
DATA_ROOT_DEFAULT = ef.DATA_ROOT_DEFAULT
S005, S030 = ef.S005, ef.S030

#: §2.1 pinned probe data (sha256 of each set's ``manifest.json``).
PINNED_DATA = {
    "dev_probe_train__F1__0.05__s20261017":
        "093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3",
    "dev_probe_test__F1__0.05__s20261017":
        "c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a",
    "dev_probe_train__F1__0.3__s20261017__n10000x50":
        "6df84887b3469723da3ddb0646c292ab655b263d6063f5fd713432a53514bb46",
    "dev_probe_test__F1__0.3__s20261017__n10000x50":
        "c68d83d1ef8cb202f3937f7cd48e3f74bd94f6d44ae203591053fde0605d1be2",
}
for _key, _sha in PINNED_DATA.items():  # ef pins the same four manifests
    if ef.PINNED_DATA.get(_key) != _sha:  # pragma: no cover - guard against upstream edits
        raise RuntimeError(f"pinned manifest for {_key} differs between the two runbooks")
PROBE_KEYS_S005 = {"train": "dev_probe_train__F1__0.05__s20261017",
                   "test": "dev_probe_test__F1__0.05__s20261017"}
PROBE_KEYS_S030 = {"train": "dev_probe_train__F1__0.3__s20261017__n10000x50",
                   "test": "dev_probe_test__F1__0.3__s20261017__n10000x50"}

# ------------------------------------------------------------ model lists §2.3
CORE_MODELS = (
    "mae_within_trigger9000", "mae_within_trigger9001",
    "mae_across_trigger9000", "mae_across_trigger9001",
    "t1_s030_within_trigger9000", "t1_s030_within_trigger9001",
    "t1_s030_across_trigger9000", "t1_s030_across_trigger9001",
    "init9000", "fix9000", "fix9001", "frozen_target9000", "frozen_target9001",
)
OPTIONAL_MODELS = ("ijepa9000", "ijepa9001", "static_target9000", "static_target9001",
                   "mae_within_median9000", "mae_within_median9001")
MAE_MODELS = ("mae_within_trigger9000", "mae_within_trigger9001",
              "mae_across_trigger9000", "mae_across_trigger9001")
X1_PARTICLE_MODELS = MAE_MODELS + ("init9000", "fix9000")
#: §2.3 pinned checkpoint sha256 (the row's path still comes from the inventory).
CORE_SHA256 = {
    "mae_within_trigger9000":
        "a2b5384f68266caf08a2563e3dc5d7d2a037ab90da0bfa5e42f659f64bef1c49",
    "mae_within_trigger9001":
        "210845c4d40f94b1f518de2dbf10c4ab8a3b0d7237a4c97a1f5bf7eba86539f1",
    "mae_across_trigger9000":
        "30eae1fec959eb22fec5417d4d089dcfacd6d1042b5ebf17a71d85d50aa85a64",
    "mae_across_trigger9001":
        "77d36d84b033a9b973afdfefab9b5f0505abf30e5887d8df7c130f52b090986f",
    "t1_s030_within_trigger9000":
        "683f3ef9474ed0be3febcdc005bd4bbb0ed8ecd793b6df580b655e7f25462dbb",
    "t1_s030_within_trigger9001":
        "415b38031182fe4d5dd5f444ff12febe84cdd6a2e9ef9ce3774ab5a0e7bd8611",
    "t1_s030_across_trigger9000":
        "accc9016b9b2270777033f76fc15636ecaca17da6c4531e0853816f78c30a475",
    "t1_s030_across_trigger9001":
        "a8752fde554f7715bc96a67c3f5b717131e313155ae84a6497216710b6d193f8",
    "init9000": None,
    "fix9000": "2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714",
    "fix9001": "95cb5e3f4f158a719098a4b3dc573eebb8c71645c22290f6483223ccc04c484e",
    "frozen_target9000":
        "8c6341c75d1463d0dcf0582a980c4b6cc700c9d755a2904305e7bcb36be8138f",
    "frozen_target9001":
        "54122fa57a0018828754c61f6675ae5a240fc785abfcaab62f394397f375ffb5",
}

#: §2.3 pinned checkpoint paths, relative to ``/ceph/dwong/jepa/returns/``.
RETURNS_ROOT = Path("/ceph/dwong/jepa/returns")
CORE_PATHS = {
    "mae_within_trigger9000":
        "ext_mae_dejitter_20261031/train/runs/ext_dev_s005_within_trigger__mae_mse__s9000/final.pt",
    "mae_within_trigger9001":
        "ext_mae_dejitter_20261031/train/runs/ext_dev_s005_within_trigger__mae_mse__s9001/final.pt",
    "mae_across_trigger9000":
        "ext_mae_dejitter_20261031/train/runs/ext_dev_s005_across__mae_mse__s9000/final.pt",
    "mae_across_trigger9001":
        "ext_mae_dejitter_20261031/train/runs/ext_dev_s005_across__mae_mse__s9001/final.pt",
    "t1_s030_within_trigger9000":
        "ext_followup_20261103/train/runs/ext_dev_s030_within_trigger__mae_mse__s9000/final.pt",
    "t1_s030_within_trigger9001":
        "ext_followup_20261103/train/runs/ext_dev_s030_within_trigger__mae_mse__s9001/final.pt",
    "t1_s030_across_trigger9000":
        "ext_followup_20261103/train/runs/ext_dev_s030_across_trigger__mae_mse__s9000/final.pt",
    "t1_s030_across_trigger9001":
        "ext_followup_20261103/train/runs/ext_dev_s030_across_trigger__mae_mse__s9001/final.pt",
    "fix9000": "ext_readout_20261101/ckpt/fix9000.pt",
    "fix9001": "ext_readout_20261101/ckpt/fix9001.pt",
    "frozen_target9000":
        "ext_h12_20261101/train/frozen_target/runs/dev_s005_within__sigreg_jepa__s9000/final.pt",
    "frozen_target9001":
        "ext_h12_20261101/train/frozen_target/runs/dev_s005_within__sigreg_jepa__s9001/final.pt",
}
#: §2.3 optional rows: paths and hashes are the committed follow-up inventory rows.
OPTIONAL_INVENTORY_REL = "results/jepa/20261103_followup/inventory.json"

#: §4 X0-b reproduction targets (m), columns ``(N refit = frozen, D refit, D frozen)``.
X0B_TARGETS = {
    "mae_within_trigger9000": (0.104916, 0.110177, 0.114908),
    "mae_within_trigger9001": (0.116542, 0.123645, 0.133124),
    "mae_across_trigger9000": (0.120958, 0.127027, 0.129525),
    "mae_across_trigger9001": (0.119626, 0.126149, 0.136818),
    "t1_s030_within_trigger9000": (0.105388, 0.110750, 0.116299),
    "t1_s030_within_trigger9001": (0.100512, 0.105751, 0.110363),
    "t1_s030_across_trigger9000": (0.125605, 0.130727, 0.134039),
    "t1_s030_across_trigger9001": (0.119031, 0.124470, 0.130892),
    "init9000": (0.160072, 0.170301, None),
}
X0B_TOLERANCE_M = 1e-3
IDENTITY_EMBEDDING_TOLERANCE = 1e-6
#: §4 SWtw-vs-N charge content check (float32 relative).
SWTW_CHARGE_RTOL = 2e-7

#: §3.1 the four NOISY σ values; §7.1 the five calibration-prior σ values.
NOISY_SIGMAS = (0.01, 0.02, 0.05, 0.10)
FISHER_SIGMAS = (0.003, 0.01, 0.03, 0.1, 0.3)
NOISY_NAMESPACE = "paper3/x1/noisy"
FISHER_NAMESPACE = "paper3/x3/fisher"
LEACE_PERM_NAMESPACE = "paper3/x3/leace_perm"
LEACE_RAND_NAMESPACE = "paper3/x3/leace_rand"
LEACE_PERMUTATIONS = 200
LEACE_RANDOM_CONTROLS = 20
#: §7.2 X3-3 robustness thresholds: |Δv| <= 2 mm and <= 5 % relative energy change.
_X3_ROBUST_VERTEX_M = 0.002
_X3_ROBUST_ENERGY_REL = 0.05
#: §7.2 X3-1 fixed price class thresholds on rho_E = res_inv(N) / res_std(N).
_X3_PRICE_HIGH = 1.20
_X3_PRICE_NONE = 1.05
LEACE_RTOL = 1e-8
LEACE_SANITY_R2 = 0.02
E2_PLANTED_RANK = 5

#: §3.2 bootstrap (the same keyed bootstrap as A-2).
A2_BOOTSTRAP_RESAMPLES = ef.A2_BOOTSTRAP_RESAMPLES
A2_BOOTSTRAP_SEED = ef.A2_BOOTSTRAP_SEED
A3_MIN_CONTAINED = ef.A3_MIN_CONTAINED
DRIFT_COST_MIN_M = 0.003
RECOVERED_HIGH, RECOVERED_LOW = 0.7, 0.3

#: §8 caps and the combined front-matter S1 rule.
GPU_CAP_H = 4.0
CPU_CAP_H = 6.0
PART_A_GPU_CAP_H = 4.0
PART_B_GPU_CAP_H = 1.5
COMBINED_GPU_CAP_H = PART_A_GPU_CAP_H + PART_B_GPU_CAP_H
S1_CEILING_H = 160.0
PROJECTION_FACTOR = 1.3
PHOTON_SET_WALL_S = 13.0
PARTICLE_SET_WALL_S = 88.0
ENTRIES_CLONE = 3059
ENTRIES_SUITE = 1825
ENTRIES_EXT = 2400
ENTRIES_FLOOR = 30000
#: §8.1 CPU projection (wall hours on Deepthought); descriptive bookkeeping.
CPU_PROJECTION = {
    "focused_tests": 0.30, "full_test_suite_once": 0.50, "x0a_preflight": 0.25,
    "x1_fits_and_bootstrap": 1.10, "x1_embedding_cache_io": 0.30,
    "x2_fits_and_identification": 1.20, "x3a_fisher": 0.80,
    "x3b_leace_and_permutations": 1.00, "report": 0.25,
}
#: §8.1 GPU set counts, by stage (per-set wall × 1.3 projection factor).
X1_PHOTON_SETS_PER_MODEL = 22  # 11 condition names x both roles (§5)
X1_PARTICLE_SETS = {model: (6 if model in MAE_MODELS else 5)
                    for model in X1_PARTICLE_MODELS}
X0B_SETS_PER_MODEL = 2
X2_PHOTON_SETS_PER_MODEL = 5
X3B_ENCODER_SETS = 6


# ===================================================================== gate
def section_zero(text: str) -> str:
    """The runbook §0 block of the part under test (same helper as ext_followup)."""
    return ef.section_zero(text)


def _git(root, *args, timeout=30):
    return ef._git(root, *args, timeout=timeout)


def _resolve_root(root=None) -> Path:
    return Path(root).resolve() if root is not None else ROOT


def p_paths(root=None) -> list:
    """``<P>`` (files and directories) as absolute paths for gate (c)."""
    root = _resolve_root(root)
    return [root / name for name in (*P_FILES, *P_DIRS)]


def in_p(relative: str) -> bool:
    """True when a repository-relative path is inside the reviewed set ``<P>``."""
    relative = str(relative).replace(os.sep, "/")
    if relative in P_FILES:
        return True
    return any(relative.startswith(directory) for directory in P_DIRS)


def acceptance_record(root=None) -> dict | None:
    """The lead's acceptance record (§9.2 (b)), or ``None`` before acceptance."""
    path = _resolve_root(root) / ACCEPTANCE_REL
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return {"invalid": f"{path}: not valid JSON"}
    return payload


def import_closure(root=None) -> dict:
    """§9.2 executed-import closure: every repository module in ``sys.modules`` in ``<P>``."""
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
        except OSError:  # pragma: no cover - exotic module path
            continue
        if not path.is_file() or root not in path.parents:
            continue
        checked += 1
        relative = path.relative_to(root).as_posix()
        if not in_p(relative):
            outside.append({"module": name, "path": relative})
    forced = [row for row in outside
              if str(row["path"]).startswith(FORCED_IMPORT_EXCEPTIONS)]
    violations = [row for row in outside
                  if not str(row["path"]).startswith(FORCED_IMPORT_EXCEPTIONS)]
    return {"checked_modules": int(checked),
            "outside_allowlist": sorted(outside, key=lambda row: row["path"]),
            "forced_import_exceptions": sorted(forced, key=lambda row: row["path"]),
            "violations": sorted(violations, key=lambda row: row["path"]),
            "pass": not violations}


def _gate_checks(mode, *, root) -> dict:
    root = _resolve_root(root)
    checks = {}
    problems = []
    runbook_path = root / RUNBOOK_REL
    computed_runbook = lr.sha256_file(runbook_path) if runbook_path.is_file() else None
    record = acceptance_record(root)
    pending = record is None

    # (a) committed runbook sha256 equals the acceptance record's value
    expected_runbook = (record or {}).get("runbook_sha256")
    a_pass = bool(computed_runbook is not None
                  and expected_runbook is not None
                  and computed_runbook == expected_runbook)
    checks["a"] = {"pass": a_pass, "runbook": str(runbook_path),
                   "runbook_sha256": computed_runbook,
                   "expected_sha256": expected_runbook,
                   "pending_acceptance": bool(pending)}
    if not a_pass and not pending:
        problems.append("(a) the committed runbook sha256 does not match the acceptance record")

    # (b) acceptance record schema and review verdict
    if record is None:
        checks["b"] = {"pass": False, "status": "gate_pending_acceptance",
                       "reason": f"{ACCEPTANCE_REL} does not exist yet"}
    elif "invalid" in record:
        checks["b"] = {"pass": False, "status": "invalid", "reason": record["invalid"]}
        problems.append("(b) the acceptance record is not valid JSON")
    else:
        missing = [key for key in ACCEPTANCE_KEYS if key not in record]
        b_pass = not missing and record.get("review_verdict") == "PASS"
        checks["b"] = {"pass": b_pass, "status": "PASS" if b_pass else "denied",
                       "record": record, "missing_keys": missing,
                       "review_verdict": record.get("review_verdict")}
        if not b_pass:
            problems.append("(b) the acceptance record is incomplete or its review_verdict "
                            "is not PASS")

    # (c) working tree against impl_commit over <P>, and no untracked <P> file.
    # Pre-acceptance HEAD stands in for impl_commit so the dry run works (§9.2).
    impl_commit = (record or {}).get("impl_commit") if record else None
    if impl_commit is None:
        code, head, _ = _git(root, "rev-parse", "HEAD")
        impl_commit = head.strip() if code == 0 else None
    paths = list(P_FILES) + list(P_DIRS)
    if impl_commit:
        code, _, err = _git(root, "diff", "--quiet", str(impl_commit), "--", *paths)
        diff_clean = code == 0
        diff_error = None if diff_clean else (err.strip() or f"git diff exit {code}")
    else:
        diff_clean, diff_error = False, "no impl_commit and no git HEAD"
    code, untracked_out, _ = _git(root, "ls-files", "--others", "--exclude-standard",
                                  "--", *paths)
    untracked = [line for line in untracked_out.splitlines() if line.strip()]
    c_pass = bool(diff_clean and not untracked)
    checks["c"] = {"pass": c_pass, "impl_commit": impl_commit,
                   "impl_commit_source": ("acceptance_record" if record
                                          else "HEAD (pre-acceptance)"),
                   "diff_clean": bool(diff_clean), "diff_error": diff_error,
                   "untracked": untracked, "paths": paths}
    if not c_pass and not pending:
        problems.append(f"(c) the <P> tree differs from {impl_commit} "
                        f"(untracked: {untracked[:3]}"
                        f"{'...' if len(untracked) > 3 else ''})")

    authorised = bool(not problems and not pending)
    status = ("gate_pending_acceptance" if pending else
              ("granted" if authorised else "denied"))
    return {"authorised": bool(authorised), "status": status, "mode": mode,
            "extension": EXTENSION, "checks": checks, "problems": problems,
            "acceptance": record, "runbook_sha256": computed_runbook}


def assert_authorised(mode, *, root=None, dry_run=None, log=None) -> dict:
    """Hard gate (§9.2 (a)-(c)).  Every failure raises ``SystemExit(2)``.

    Pre-acceptance there is no record and no hard-coded runbook sha256: the gate
    returns ``gate_pending_acceptance`` in dry-run mode and ``SystemExit(2)`` for
    every other mode (the runbook has no pre-acceptance execution path).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    dry = bool(dry_run) if dry_run is not None else (mode == "dry-run")
    result = _gate_checks(mode, root=root)
    if dry:
        return result
    if not result["authorised"]:
        for problem in result["problems"]:
            print(f"[repair] authorisation gate: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return result


def _assert_closure(mode, root) -> dict:
    """§9.2/§8.4: re-assert the executed-import closure at the end of every mode."""
    closure = import_closure(root)
    if not closure["pass"]:
        print(f"[repair] {mode}: import-closure STOP: "
              f"{len(closure['violations'])} module(s) outside <P>; outputs quarantined",
              file=sys.stderr)
        raise SystemExit(2)
    return closure


# ==================================================================== data
def manifest_path(data_root, key) -> Path:
    return ef.manifest_path(data_root, key)


def data_status(data_root, *, pins=None, keys=None) -> dict:
    """Read-only sha256 check of the four pinned probe manifests (§2.1)."""
    return ef.data_status(data_root, pins=dict(PINNED_DATA) if pins is None else pins,
                          keys=keys)


def require_pinned(data_root, keys=None, *, pins=None, context="") -> dict:
    """Refuse to continue unless every named pinned manifest matches (§2.1)."""
    return ef.require_pinned(data_root, keys=keys,
                             pins=dict(PINNED_DATA) if pins is None else pins,
                             context=context)


def open_probe_sets(config, gen, geometry, data_root, *, setting=S005, log=print) -> dict:
    """Open the two probe sets through ``ext_followup.open_probe_sets`` (hash first)."""
    probes = ef.open_probe_sets(config, gen, geometry, data_root, setting=setting, log=log)
    keys = PROBE_KEYS_S005 if abs(float(setting) - S005) < 1e-12 else PROBE_KEYS_S030
    for role, dataset in probes.items():
        if dataset.key != keys[role]:
            raise RuntimeError(f"[repair] {role} probe set opened {dataset.key!r}, "
                               f"expected the pinned {keys[role]!r}")
    return probes


def module_of_pmt(geometry) -> np.ndarray:
    """Dense module index per PMT (``DetectorGeometry.module_index``; §2.2 ``mod(p)``)."""
    index = getattr(geometry, "module_of_pmt", None)
    return np.asarray(geometry.module_index if index is None else index, np.int64)


def row_split(row) -> str:
    """The §3.2 view split of an inventory row (``ext_followup`` uses the same rule)."""
    return str((row or {}).get("split", "within"))


def run_spec(row, dataset) -> lb.RunSpec:
    """The RunSpec of one manifest row, exactly as ``lucid_run._run_truth`` builds it."""
    return lb.RunSpec(row["run_id"], dataset.manifest["cohort"], dataset.law,
                      float(row["setting"]), int(row["event_id_start"]),
                      int(row["n_events"]), int(dataset.seed))


def run_state(row, dataset, geometry, setting=None):
    """``RunSpec.state(geometry)`` for a row; ``setting`` replaces the run's setting."""
    spec = run_spec(row, dataset)
    if setting is not None:
        spec = dataclasses.replace(spec, setting=float(setting))
    return spec.state(geometry)


def state_pair(row, dataset, geometry):
    """``(s0, s1)`` states of one run id at s = 0.05 and s = 0.30 (§2.2).

    The run ids and the keyed per-PMT factor do not depend on the setting, so
    building both states from the same row gives ``G0/G1 = m0/m1`` exactly.
    """
    return (run_state(row, dataset, geometry, S005), run_state(row, dataset, geometry, S030))


# =============================================================== conditions §3.1
#: Base condition names plus the parameterised NOISY and EST families.
CONDITION_NAMES = ("N", "D", "S", "SWtw", "SWunit", "SWmod", "SWunitN", "HW",
                   "NOISY", "EST")
_SOURCE = {"N": S005, "D": S030, "S": S005, "SWtw": S030, "SWunit": S030,
           "SWmod": S030, "SWunitN": S005, "HW": S030, "NOISY": S030, "EST": S030}
_FILE = {"N": "hits.npz", "D": "hits.npz", "S": "hits.npz", "SWtw": "hits.npz",
         "SWunit": "hits.npz", "SWmod": "hits.npz", "SWunitN": "hits.npz",
         "HW": "twin_hits.npz", "NOISY": "hits.npz", "EST": "hits.npz"}


def condition_name(base, *, sigma=None, stream=None) -> str:
    """Canonical name of one condition: ``NOISY(σ)``/``EST(k)`` spell the parameter."""
    base = str(base)
    if base in ("N", "D", "S", "SWtw", "SWunit", "SWmod", "SWunitN", "HW"):
        return base
    if base.upper().startswith("NOISY"):
        sigma = float(sigma) if sigma is not None else float(
            base[base.find("(") + 1:base.find(")")] if "(" in base else 0.01)
        return f"NOISY({sigma:g})"
    if base.upper().startswith("EST"):
        stream = stream or (base[base.find("(") + 1:base.find(")")] if "(" in base else "raw")
        return f"EST({stream})"
    raise ValueError(f"unknown condition {base!r}; expected {CONDITION_NAMES}")


def _condition_base(name):
    """``(base, sigma, stream)`` parsed from a canonical condition name."""
    name = str(name)
    if "(" not in name:
        base = name.upper() if name.upper() in ("N", "D", "S", "SWTW", "SWUNIT",
                                                "SWMOD", "SWUNITN", "HW") else name
        base = {"SWTW": "SWtw", "SWUNIT": "SWunit", "SWMOD": "SWmod",
                "SWUNITN": "SWunitN"}.get(base, base)
        if base not in _SOURCE:
            raise ValueError(f"unknown condition {name!r}")
        return base, None, None
    base, inner = name.split("(", 1)
    inner = inner.rstrip(")")
    base = base.upper() if base.upper() in ("NOISY", "EST") else base
    if base == "NOISY":
        return "NOISY", float(inner), None
    if base == "EST":
        if inner not in ("source", "raw", "latent"):
            raise ValueError(f"unknown EST stream {inner!r}")
        return "EST", None, inner
    raise ValueError(f"unknown condition {name!r}")


def _noisy_xi(run_id, geometry) -> np.ndarray:
    """One ξ ~ N(0, 1) per run and module, shared by the four σ (Appendix A)."""
    return lb.keyed_rng(20261106, NOISY_NAMESPACE, str(run_id)).standard_normal(
        int(geometry.n_modules))


def condition_factor_arrays(base, row, dataset, geometry, *, sigma=None,
                            estimates=None) -> np.ndarray:
    """The per-PMT factor f_r(p) of one condition on one run (§3.1, Appendix A)."""
    mod = module_of_pmt(geometry)
    run_id = str(row["run_id"])
    base = str(base)
    if base in ("N", "D"):
        return np.ones(int(geometry.n_pmt))
    s0, s1 = state_pair(row, dataset, geometry)
    if base == "S":
        return s1.gains / s0.gains
    if base == "SWtw":
        return s0.gains / s1.gains
    if base == "SWunit":
        return 1.0 / s1.gains
    if base == "SWmod":
        return 1.0 / np.asarray(s1.module_gain, float)[mod]
    if base == "SWunitN":
        return 1.0 / s0.gains
    if base == "HW":
        return np.ones(int(geometry.n_pmt))
    if base == "NOISY":
        if sigma is None:
            raise ValueError("NOISY needs a sigma")
        xi = _noisy_xi(run_id, geometry)
        return 1.0 / (np.asarray(s1.gains, float)
                      * np.exp(float(sigma) * xi[mod]))
    if base == "EST":
        if not estimates or run_id not in estimates:
            raise ValueError(f"EST: no gain estimate for run {run_id!r}")
        delta = np.nan_to_num(np.asarray(estimates[run_id], float), nan=0.0, posinf=0.0,
                              neginf=0.0)
        if delta.shape != (int(geometry.n_modules),):
            raise ValueError(f"EST: run {run_id!r} estimate must have "
                             f"{int(geometry.n_modules)} modules")
        return 1.0 / np.exp(delta[mod])
    raise ValueError(f"unknown condition base {base!r}")


@dataclasses.dataclass
class Condition:
    """One §3.1 input condition applied to one probe set."""

    name: str
    base: str
    setting: float
    filename: str
    directory: Path
    run_id_by_dir: dict
    factors: dict
    n_pmt: int
    sigma: float | None = None
    stream: str | None = None

    def factor(self, run_id) -> np.ndarray:
        return np.asarray(self.factors[str(run_id)], float)

    def as_dict(self) -> dict:
        return {"name": self.name, "base": self.base, "setting": float(self.setting),
                "file": self.filename, "directory": str(self.directory),
                "n_runs": int(len(self.factors)), "n_pmt": int(self.n_pmt),
                "sigma": self.sigma, "stream": self.stream}


def build_condition(name, dataset, geometry, *, estimates=None) -> Condition:
    """Build one condition over every run of ``dataset`` (§3.1)."""
    base, sigma, stream = _condition_base(name)
    expected = _SOURCE[base]
    if abs(float(dataset.setting) - float(expected)) > 1e-12:
        raise ValueError(f"condition {name}: source set s={dataset.setting} "
                         f"but {base} needs s={expected}")
    factors, by_dir = {}, {}
    for row in dataset.runs:
        run_id = str(row["run_id"])
        factors[run_id] = condition_factor_arrays(base, row, dataset, geometry,
                                                  sigma=sigma, estimates=estimates)
        by_dir[str(row["run_dir"])] = run_id
    return Condition(condition_name(base, sigma=sigma, stream=stream), base,
                     float(dataset.setting), _FILE[base], Path(dataset.directory),
                     by_dir, factors, int(geometry.n_pmt), sigma=sigma, stream=stream)


def _transformed_store(condition: Condition, path, reader):
    """The §2.5 wrapper body: read the condition's store and scale its charges."""
    requested = Path(path)
    if requested.is_dir():
        requested = requested / "hits.npz"
    if requested.name != "hits.npz":
        raise RuntimeError(f"hit_transform_context: refusing non-hit path {requested}")
    run_dir = requested.parent.resolve()
    directory = Path(condition.directory).resolve()
    if run_dir.parent != directory / "runs":
        raise RuntimeError(f"hit_transform_context: {run_dir} is outside the "
                           f"{condition.name} set {directory}")
    run_id = condition.run_id_by_dir.get(run_dir.name)
    if run_id is None:
        raise RuntimeError(f"hit_transform_context: unknown run dir {run_dir.name!r} "
                           f"for condition {condition.name}")
    store = reader(run_dir / condition.filename)
    factor = condition.factor(run_id)
    if np.all(factor == 1.0):
        return store
    charge = (np.asarray(store.charge_pe, np.float32)
              * np.asarray(factor, np.float32)[np.asarray(store.pmt_id, np.int64)])
    return lb.SparseHitStore(store.event_ids, store.offsets, store.pmt_id,
                             charge.astype(np.float32), store.time_ns,
                             store.readout_label)


@contextlib.contextmanager
def hit_transform_context(condition: Condition):
    """§2.5: replace both hit-read bindings; restore both in a ``finally``."""
    if not isinstance(condition, Condition):
        raise TypeError("hit_transform_context needs a Condition")
    original_pc = pe.read_sparse_hits
    original_lb = lb.read_sparse_hits

    def wrapper(path):
        return _transformed_store(condition, path, original_lb)

    pe.read_sparse_hits = wrapper
    lb.read_sparse_hits = wrapper
    try:
        yield condition
    finally:
        pe.read_sparse_hits = original_pc
        lb.read_sparse_hits = original_lb


# ------------------------------------------------- §7.2 invariant-token wrapper
def invariant_features(features, hit_flags, module_index) -> np.ndarray:
    """Subtract each hit token's module mean log charge in its event (§7.2).

    ``features`` is ``(B, N_pmt, F)`` (numpy or torch); the log-charge column is
    shifted for hit tokens only and unhit tokens are unchanged.
    """
    if hasattr(features, "detach"):
        array = features.detach().cpu().numpy().astype(np.float64)
        restore = True
        original_dtype = None
    else:
        source = np.asarray(features)
        original_dtype = source.dtype
        array = source.astype(np.float64, copy=True)
        restore = False
    hit = np.asarray(hit_flags, bool)
    module_index = np.asarray(module_index, np.int64)
    log_charge = array[..., pe.LOG_CHARGE_INDEX]
    shifted = log_charge.copy()
    n_modules = int(module_index.max()) + 1 if module_index.size else 0
    for event in range(array.shape[0]):
        selected = hit[event]
        if not selected.any():
            continue
        modules = module_index[selected]
        values = log_charge[event][selected]
        sums = np.bincount(modules, weights=values, minlength=n_modules)
        counts = np.bincount(modules, minlength=n_modules)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        shifted[event][selected] = values - means[modules]
    array[..., pe.LOG_CHARGE_INDEX] = shifted
    if restore:
        import torch
        return torch.as_tensor(array, dtype=features.dtype, device=features.device)
    return array.astype(original_dtype, copy=False)


def install_invariant_token_wrapper(module, geometry):
    """Replace ``module.tokenise`` with the invariantly shifted log charge (§7.2)."""
    original = module.tokenise
    module_index = module_of_pmt(geometry)

    def wrapped(*args, **kwargs):
        batch = original(*args, **kwargs)
        return dataclasses.replace(
            batch, features=invariant_features(batch.features, batch.hit_flags,
                                               module_index))

    module.tokenise = wrapped
    return original


def remove_invariant_token_wrapper(module, original):
    module.tokenise = original


@contextlib.contextmanager
def invariant_token_context(geometry):
    """The X3-b invariant-token wrapper, restored in a ``finally`` (§7.2).

    Allowed only for the ``init9000`` X3 passes: no trained checkpoint is read.
    """
    original = install_invariant_token_wrapper(pe, geometry)
    try:
        yield
    finally:
        remove_invariant_token_wrapper(pe, original)


# ================================================================= helpers
def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _jsonable(value):
    """Recursive JSON form: numpy arrays/scalars, non-finite floats, paths, tuples."""
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload, *, overwrite=True) -> Path:
    """Atomic JSON write; ``overwrite=False`` refuses an existing file (§9.3)."""
    path = Path(path)
    if not overwrite and path.exists():
        raise RuntimeError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=1, default=float) + "\n")
    os.replace(tmp, path)
    return path


def _truncate(dataset, *, max_runs=None, max_events=None):
    """A dataset view restricted to the first runs / events (dry-run sampling only)."""
    if max_runs is None and max_events is None:
        return dataset
    runs = dataset.runs if max_runs is None else tuple(dataset.runs[:int(max_runs)])
    return dataclasses.replace(dataset, runs=runs)


def hardware_and_software() -> dict:
    return ef.hardware_and_software()


def _strip_internal(value):
    """Drop per-event internal error arrays from a payload before it is written."""
    if isinstance(value, dict):
        return {key: _strip_internal(item) for key, item in value.items()
                if key not in ("__errors__", "errors")}
    if isinstance(value, (list, tuple)):
        return [_strip_internal(item) for item in value]
    return value


# =========================================================== embeddings §3.2
class EmbeddingCache:
    """Write-once ``.npz`` embedding cache under ``<ext>`` with a sha256 sidecar.

    §3.2: every embedding set is written once and reused by later items; nothing
    is recomputed for a second item.  A read whose sha256 does not match the
    sidecar raises rather than silently reusing a different array.
    """

    ARRAY_NAMES = ("events", "run_ids", "event_ids", "vertex", "log_yield", "n_hit",
                   "particle", "contained", "deposited_MeV", "direction", "domain")
    #: R-1 reads the cached N/D probe-test embeddings from this array name.
    ALIASES = {"embeddings": "events"}

    def __init__(self, root):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = read_json(self.manifest_path, {}) or {}

    def path(self, key) -> Path:
        return self.root / f"{str(key)}.npz"

    def _save_manifest(self):
        write_json(self.manifest_path, self.manifest)

    def has(self, key) -> bool:
        row = self.manifest.get(str(key)) or {}
        return bool(row.get("sha256")) and self.path(key).is_file()

    def fetch(self, key, producer, *, log=print) -> dict:
        """Return the cached arrays, or produce, write and register them once."""
        key = str(key)
        row = self.manifest.get(key)
        path = self.path(key)
        if row is not None and path.is_file():
            digest = lr.sha256_file(path)
            if digest != row.get("sha256"):
                raise RuntimeError(f"embedding cache {path}: sha256 {digest} does not "
                                   f"match the sidecar {row.get('sha256')}")
            with np.load(path, allow_pickle=False) as data:
                out = {name: np.asarray(data[name]) for name in data.files}
            for alias, name in self.ALIASES.items():
                if alias in out and name not in out:
                    out[name] = out[alias]
            out["cache_status"] = "reused"
            out["cache_sha256"] = digest
            return out
        out = dict(producer())
        if not out.get("events", np.zeros((0, 0))).shape[0]:
            raise RuntimeError(f"embedding cache {key}: producer returned no events")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: np.asarray(out[name])
                   for name in self.ARRAY_NAMES if name in out}
        for text_name in ("run_ids", "particle", "domain"):
            if text_name in payload:
                payload[text_name] = np.asarray(payload[text_name]).astype(str)
        payload["embeddings"] = np.asarray(out["events"])
        np.savez_compressed(path, **payload)
        digest = lr.sha256_file(path)
        self.path(key).with_suffix(".npz.sha256").write_text(digest + "\n")
        self.manifest[key] = {"sha256": digest, "path": str(path),
                              "n_events": int(np.asarray(out["events"]).shape[0]),
                              "bytes": int(path.stat().st_size)}
        self._save_manifest()
        log(f"[repair] cached embeddings {key} -> {path} ({digest[:12]}...)")
        out["cache_status"] = "written"
        out["cache_sha256"] = digest
        return out


def stream_condition_embeddings(model, dataset, geometry, condition, *, split, config,
                                device="cpu", cache=None, cache_key=None,
                                max_events=None, max_runs=None, log=print) -> dict:
    """Photon embeddings of one probe set under one condition (§3.2)."""
    chunk = int((config.get("evaluation") or {}).get("probe_chunk_events", 64))
    mask_seed = int(config["views"]["mask_seed"])

    def producer():
        with hit_transform_context(condition):
            return ef.stream_event_rows(
                model, _truncate(dataset, max_runs=max_runs), geometry, split=split,
                mask_seed=mask_seed, device=device, chunk_events=chunk,
                max_events=max_events, log=log)

    if cache is None:
        rows = dict(producer())
        rows["cache_status"] = "uncached"
        return rows
    key = cache_key or embedding_cache_key(split, condition.name)
    rows = cache.fetch(key, producer, log=log)
    return {name: rows[name] for name in rows if name in EmbeddingCache.ARRAY_NAMES} | {
        k: rows[k] for k in ("cache_status", "cache_sha256") if k in rows}


def stream_condition_particles(model, dataset, truth, geometry, condition, *, device="cpu",
                               cache=None, cache_key=None, max_events=None,
                               max_runs=None, log=print) -> dict:
    """Particle (muon/electron) embeddings of one probe set under one condition (§3.2)."""
    names = ("events", "run_ids", "event_ids", "particle", "direction", "deposited_MeV",
             "contained", "n_hit", "vertex")

    def producer():
        with hit_transform_context(condition):
            return ef.stream_particle_rows(
                model, _truncate(dataset, max_runs=max_runs), truth, geometry,
                chunk_events=8, device=device, log=log)

    if cache is None:
        rows = dict(producer())
        rows["cache_status"] = "uncached"
        return rows
    key = cache_key or f"particle__{condition.name}"
    rows = cache.fetch(key, producer, log=log)
    return {name: rows[name] for name in rows if name in names} | {
        k: rows[k] for k in ("cache_status", "cache_sha256") if k in rows}


# ============================================================== probes §3.2
def fit_vertex_probe(rows, *, kind="vertex") -> dict:
    """Five-fold run-grouped ridge vertex probe on one condition's probe-train."""
    train = ef._probe_dataset(rows, kind, target_names=("x", "y", "z"))
    return la.fit_probe(train)


def _r2_from_predictions(prediction, truth) -> dict:
    prediction = np.asarray(prediction, float)
    truth = np.asarray(truth, float)
    keep = np.isfinite(prediction).all(1) & np.isfinite(truth).all(1)
    if not keep.any():
        return {"r2": None, "n": 0}
    sse = float(((prediction[keep] - truth[keep]) ** 2).sum())
    sst = float(((truth[keep] - truth[keep].mean(0)) ** 2).sum())
    return {"r2": None if sst <= 0 else float(1.0 - sse / sst),
            "n": int(keep.sum())}


def vertex_metrics(train_rows, test_rows, *, fitted=None) -> dict:
    """Vertex mean and median 3-D error on probe-test with the frozen P_N (§3.2)."""
    test = ef._probe_dataset(test_rows, "vertex", target_names=("x", "y", "z"))
    fitted = fit_vertex_probe(train_rows) if fitted is None else fitted
    prediction = la.probe_predictions(fitted, test)
    vertex = lr._vertex_error(prediction, test.y)
    errors = ef._vertex_errors(test_rows, prediction)
    return {"fitted": fitted, "prediction": prediction, "errors": errors,
            "mean_error_m": vertex.get("resolution"),
            "median_error_m": vertex.get("median"),
            "status": vertex.get("status"), "r2": _r2_from_predictions(prediction, test.y),
            "n_events": int(errors.size), "run_ids": np.asarray(test_rows["run_ids"]),
            "event_ids": np.asarray(test_rows["event_ids"])}


def fit_log_yield_probe(rows) -> dict:
    dataset = ef._probe_dataset(rows, "log_yield", y=rows["log_yield"],
                                target_names=("log_yield",))
    return la.fit_probe(dataset)


def log_yield_metrics(train_rows, test_rows, *, fitted=None) -> dict:
    """Log-yield RMS resolution and R² on probe-test with the frozen P_N (§3.2)."""
    test = ef._probe_dataset(test_rows, "log_yield", y=test_rows["log_yield"],
                             target_names=("log_yield",))
    fitted = fit_log_yield_probe(train_rows) if fitted is None else fitted
    prediction = la.probe_predictions(fitted, test)
    errors = np.asarray(prediction, float).ravel() - np.asarray(test_rows["log_yield"],
                                                                float).ravel()
    keep = np.isfinite(errors)
    resolution = None if not keep.any() else float(np.sqrt(np.mean(errors[keep] ** 2)))
    return {"fitted": fitted, "prediction": prediction, "errors": errors,
            "resolution": resolution, "r2": _r2_from_predictions(prediction, test.y),
            "n_events": int(keep.sum()),
            "run_ids": np.asarray(test_rows["run_ids"]),
            "event_ids": np.asarray(test_rows["event_ids"])}


def _electron_mask(rows) -> np.ndarray:
    particle = np.asarray(rows.get("particle", np.zeros(0, str))).astype(str)
    contained = np.asarray(rows.get("contained", np.zeros(len(particle), bool)), bool)
    return (particle == "electron") & contained & np.isfinite(
        np.asarray(rows.get("deposited_MeV", np.full(len(particle), np.nan)), float))


def fit_energy_probe(rows) -> dict:
    """Frozen energy probe: fit on one condition's contained electrons (§3.2)."""
    mask = _electron_mask(rows)
    dataset = la.ProbeDataset(
        kind="deposited_MeV", form="full_input",
        x=np.asarray(rows["events"], float)[mask],
        y=np.asarray(rows["deposited_MeV"], float)[mask][:, None],
        run_ids=np.asarray(rows["run_ids"], dtype=object)[mask],
        target_names=("deposited_MeV",), meta={})
    return {"fitted": la.fit_probe(dataset), "n_train": int(mask.sum())}


def energy_metrics(train_rows, test_rows, *, fitted=None) -> dict:
    """Electron-energy relative resolution and R² on probe-test (§3.2).

    The probe is fit on the condition's own contained-electron probe-train; the
    ``A3_MIN_CONTAINED`` = 500 test rule applies.
    """
    mask = _electron_mask(test_rows)
    n_contained = int(mask.sum())
    if n_contained < A3_MIN_CONTAINED:
        return {"status": "not_evaluable",
                "reason": f"{n_contained} contained electrons < {A3_MIN_CONTAINED}",
                "n_contained_test": n_contained,
                "n_contained_train": int(_electron_mask(train_rows).sum())}
    fitted = fit_energy_probe(train_rows) if fitted is None else fitted
    test = la.ProbeDataset(
        kind="deposited_MeV", form="full_input",
        x=np.asarray(test_rows["events"], float)[mask],
        y=np.asarray(test_rows["deposited_MeV"], float)[mask][:, None],
        run_ids=np.asarray(test_rows["run_ids"], dtype=object)[mask],
        target_names=("deposited_MeV",), meta={})
    prediction = la.probe_predictions(fitted["fitted"], test)
    truth = np.asarray(test_rows["deposited_MeV"], float)[mask]
    relative = (np.asarray(prediction, float).ravel() - truth) / truth
    keep = np.isfinite(relative)
    resolution = None if not keep.any() else float(np.sqrt(np.mean(relative[keep] ** 2)))
    run_ids = np.asarray(test_rows["run_ids"], dtype=object)[mask]
    return {"status": "evaluated", "resolution": resolution,
            "ci95": cluster_bootstrap(relative, run_ids, statistic=rms)["ci95"],
            "r2": _r2_from_predictions(prediction, test.y),
            "n_contained_test": n_contained,
            "n_contained_train": int(fitted["n_train"]),
            "run_ids": run_ids,
            "event_ids": np.asarray(test_rows["event_ids"])[mask],
            "errors": relative}


def cluster_bootstrap(values, run_ids, *, statistic=np.mean,
                      resamples=A2_BOOTSTRAP_RESAMPLES, seed=A2_BOOTSTRAP_SEED) -> dict:
    """Run-cluster bootstrap CI of one statistic (§3.2; resolutions use the same draw)."""
    values = np.asarray(values, float)
    run_ids = np.asarray(run_ids, dtype=object).astype(str)
    runs = np.unique(run_ids)
    if not runs.size or values.shape[0] != run_ids.shape[0]:
        raise ValueError("cluster_bootstrap: misaligned inputs")
    if not np.isfinite(values).any():
        return {"n_runs": int(len(runs)), "n_events": 0, "statistic": None,
                "ci95": None, "resamples": int(resamples), "seed": int(seed)}
    order = np.argsort(run_ids, kind="stable")
    sorted_runs = run_ids[order]
    boundaries = np.append(np.searchsorted(sorted_runs, runs), len(order))
    groups = [values[order][boundaries[i]:boundaries[i + 1]]
              for i in range(len(runs))]
    observed = float(statistic(np.concatenate(groups))) if groups else None
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), float)
    for index in range(int(resamples)):
        pick = rng.integers(0, len(runs), len(runs))
        draws[index] = float(statistic(np.concatenate([groups[j] for j in pick])))
    return {"n_runs": int(len(runs)), "n_events": int(values.size),
            "statistic": observed, "ci95": [float(np.quantile(draws, 0.025)),
                                            float(np.quantile(draws, 0.975))],
            "resamples": int(resamples), "seed": int(seed)}


def paired_cluster_bootstrap(left, right, run_ids, *, statistic=np.mean,
                             resamples=A2_BOOTSTRAP_RESAMPLES,
                             seed=A2_BOOTSTRAP_SEED) -> dict:
    """Run-cluster bootstrap of ``statistic(right) - statistic(left)``, pairs kept (§3.2)."""
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    run_ids = np.asarray(run_ids, dtype=object).astype(str)
    if left.shape != right.shape or left.shape[0] != run_ids.shape[0]:
        raise ValueError("paired_cluster_bootstrap: misaligned inputs")
    runs = np.unique(run_ids)
    order = np.argsort(run_ids, kind="stable")
    sorted_runs = run_ids[order]
    boundaries = np.append(np.searchsorted(sorted_runs, runs), len(order))
    groups_left = [left[order][boundaries[i]:boundaries[i + 1]] for i in range(len(runs))]
    groups_right = [right[order][boundaries[i]:boundaries[i + 1]] for i in range(len(runs))]
    observed = float(statistic(np.concatenate(groups_right))
                     - statistic(np.concatenate(groups_left)))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), float)
    for index in range(int(resamples)):
        pick = rng.integers(0, len(runs), len(runs))
        draws[index] = float(statistic(np.concatenate([groups_right[j] for j in pick]))
                             - statistic(np.concatenate([groups_left[j] for j in pick])))
    return {"n_runs": int(len(runs)), "n_events": int(left.size),
            "mean_error_left": float(left.mean()), "mean_error_right": float(right.mean()),
            "difference": observed,
            "ci95": [float(np.quantile(draws, 0.025)),
                     float(np.quantile(draws, 0.975))],
            "resamples": int(resamples), "seed": int(seed),
            "definition": "right minus left, run-cluster resampling, pairs kept together"}


def rms(values) -> float:
    values = np.asarray(values, float)
    keep = np.isfinite(values)
    return float(np.sqrt(np.mean(values[keep] ** 2))) if keep.any() else float("nan")


def drift_cost(vertex_n: float, vertex_d: float) -> dict:
    """Δ_D = v_D − v_N (§3.2)."""
    return {"v_N_m": float(vertex_n), "v_D_m": float(vertex_d),
            "delta_D_m": float(vertex_d - vertex_n)}


def recovered_fraction(vertex_n, vertex_c, vertex_d, *, delta_ci=None) -> dict:
    """ρ_c = (v_D − v_c)/(v_D − v_N), or ``no drift cost to recover`` (§3.2)."""
    delta = float(vertex_d) - float(vertex_n)
    eligible = bool(abs(delta) >= DRIFT_COST_MIN_M
                    and (delta_ci is None
                         or (float(delta_ci[0]) > 0.0 or float(delta_ci[1]) < 0.0)))
    if not eligible:
        return {"eligible": False, "reading": "no drift cost to recover",
                "delta_D_m": delta, "rho": None}
    denominator = float(vertex_d) - float(vertex_n)
    return {"eligible": True, "delta_D_m": delta,
            "rho": float((float(vertex_d) - float(vertex_c)) / denominator)}


# ==================================================== classification §3.3
def classify_seeds(entries, classify) -> dict:
    """Apply the §3.3 seeded-reading rule to one reading.

    ``entries`` is a list of ``(seed, value_or_None)``; ``classify(value)`` returns
    a class label (or ``None`` when the value is not eligible).  Rules:
    both seeds in one class -> that class; different -> ``seeds disagree``; one
    eligible and the other not -> ``not classified``; a single seed ->
    ``single seed (not classified)``; unseeded -> classified directly.
    """
    entries = list(entries)
    if not entries:
        return {"class": "not classified", "per_seed": []}
    labelled = [(int(seed), classify(value) if value is not None else None)
                for seed, value in entries]
    classes = {label for _seed, label in labelled if label is not None}
    if len(entries) == 1:
        label = labelled[0][1]
        return {"class": ("single seed (not classified)" if label is not None
                          else "not classified"),
                "single_seed_class": label,
                "per_seed": [{"seed": labelled[0][0], "class": labelled[0][1]}]}
    eligible = [label for _seed, label in labelled if label is not None]
    if len(eligible) != len(labelled):
        return {"class": "not classified", "per_seed":
                [{"seed": seed, "class": label} for seed, label in labelled]}
    if len(classes) == 1:
        return {"class": classes.pop(), "per_seed":
                [{"seed": seed, "class": label} for seed, label in labelled]}
    return {"class": "seeds disagree", "per_seed":
            [{"seed": seed, "class": label} for seed, label in labelled]}


# ================================================================ inventory
def _family(label):
    text = str(label)
    if len(text) > 4 and text[-4:].isdigit():
        return text[:-4], int(text[-4:])
    return text, None


def part_a_inventory(root=None, *, hash_files=True, include_optional=True, log=print) -> dict:
    """§2.3 inventory: the 13 core rows (plus optional rows) accepted on a sha match.

    Rows come from ``ext_followup.build_inventory`` (read-only) and
    ``ext_followup.t1_inventory_rows``; the re-hash is restricted to these labels,
    which §2.3 allows.  ``init9000`` is a seed-9000 rebuild and carries no file.
    """
    root = _resolve_root(root)
    base = ef.build_inventory(root, hash_files=False, log=lambda *a, **k: None)
    t1_rows = ef.t1_inventory_rows(ef.EXT_ROOT)
    optional_meta = (read_json(root / OPTIONAL_INVENTORY_REL, {}) or {}).get("rows") or {}
    wanted = list(CORE_MODELS) + (list(OPTIONAL_MODELS) if include_optional else [])
    payload = {"registered": False, "extension": EXTENSION,
               "check": "§2.3 checkpoint inventory (Part A core list)",
               "root": str(root), "hash_files": bool(hash_files),
               "rows": {}, "unavailable": [], "optional": list(OPTIONAL_MODELS)}
    for label in wanted:
        source_row = base["rows"].get(label) or t1_rows.get(label)
        if source_row is None:
            payload["unavailable"].append({"label": label, "reason": "no inventory row"})
            continue
        row = dict(source_row)
        expected = CORE_SHA256.get(label) or row.get("sha256")
        source = row.get("source")
        if source is None:
            relative = CORE_PATHS.get(label) or (optional_meta.get(label) or {}).get(
                "source") or (optional_meta.get(label) or {}).get("ckpt")
            if relative:
                candidate = Path(str(relative))
                source = candidate if candidate.is_absolute() else RETURNS_ROOT / candidate
        if expected is None:
            expected = (optional_meta.get(label) or {}).get("sha256") or (
                optional_meta.get(label) or {}).get("ckpt_sha256")
        if label == "init9000":
            row.update(status="available", source=None, sha256=None)
            payload["rows"][label] = row
            continue
        if not source or not Path(source).is_file():
            row.update(status="unavailable", reason="checkpoint file is absent")
            payload["unavailable"].append({"label": label, "reason": row["reason"]})
            payload["rows"][label] = row
            continue
        if not hash_files:
            row.update(status="unverified (dry-run)", sha256=expected, source=source)
            payload["rows"][label] = row
            continue
        actual = lr.sha256_file(source)
        if expected and actual != expected:
            row.update(status="unavailable", actual_sha256=actual,
                       reason=f"sha256 {actual} != pinned {expected}")
            payload["unavailable"].append({"label": label, "reason": row["reason"]})
            payload["rows"][label] = row
            continue
        row.update(status="available", sha256=expected, actual_sha256=actual,
                   source=source)
        payload["rows"][label] = row
    payload["counts"] = {
        "labels": len(payload["rows"]),
        "available": sum(1 for row in payload["rows"].values()
                         if row["status"] in ("available", "unverified (dry-run)")),
        "unavailable": len(payload["unavailable"]),
        "core": len(CORE_MODELS), "optional": len(OPTIONAL_MODELS)}
    return payload


def model_rows(inventory, *, include_optional=True) -> list:
    """The inventory labels to evaluate, core first, optional only when available."""
    out = list(CORE_MODELS)
    if include_optional:
        out += [label for label in OPTIONAL_MODELS
                if (inventory["rows"].get(label) or {}).get("status")
                in ("available", "unverified (dry-run)")]
    return out


def embedding_cache(ext_root, label) -> EmbeddingCache:
    """The §3.2/X1 cache root of one model: ``<ext>/x1/<label>/``.

    R-1 reads the cached N and D probe-test embeddings from here; the file names
    are ``<split>__<condition>.npz``, so ``probe_test__N``/``probe_test__D``
    contain ``__N``/``__D`` as the cross-part contract requires.
    """
    return EmbeddingCache(Path(ext_root) / "x1" / str(label))


def embedding_cache_key(role, name) -> str:
    """Cache file key: ``<role>__<condition>`` (the split comes first on purpose)."""
    cleaned = str(name).replace("(", "_").replace(")", "").replace(".", "")
    return f"{str(role)}__{cleaned}"


# ============================================================ projections §8
def gpu_projection(models=None, *, include_optional=True) -> dict:
    """§8.1 GPU projection: Σ sets × per-set wall × 1.3 against the 4.0 h cap."""
    models = list(CORE_MODELS if models is None else models)
    n_models = len(models)
    photon_sets = (n_models * (X1_PHOTON_SETS_PER_MODEL + X2_PHOTON_SETS_PER_MODEL)
                   + X0B_SETS_PER_MODEL * len(X0B_TARGETS) + X3B_ENCODER_SETS)
    particle_sets = int(sum(X1_PARTICLE_SETS.values()))
    seconds = photon_sets * PHOTON_SET_WALL_S + particle_sets * PARTICLE_SET_WALL_S
    hours = seconds / 3600.0 * PROJECTION_FACTOR
    optional_hours = None
    if include_optional:
        optional_photon = len(OPTIONAL_MODELS) * (X1_PHOTON_SETS_PER_MODEL
                                                  + X2_PHOTON_SETS_PER_MODEL)
        optional_seconds = optional_photon * PHOTON_SET_WALL_S
        optional_hours = optional_seconds / 3600.0 * PROJECTION_FACTOR
    out = {"n_models": n_models, "photon_sets": int(photon_sets),
           "particle_sets": int(particle_sets),
           "photon_set_wall_s": PHOTON_SET_WALL_S,
           "particle_set_wall_s": PARTICLE_SET_WALL_S,
           "factor": PROJECTION_FACTOR, "projected_l40s_hours": round(hours, 4),
           "cap_hours": float(PART_A_GPU_CAP_H), "pass": bool(hours <= PART_A_GPU_CAP_H)}
    if optional_hours is not None:
        out["optional_models"] = list(OPTIONAL_MODELS)
        out["optional_projected_hours"] = round(optional_hours, 4)
        out["with_optional_projected_hours"] = round(hours + optional_hours, 4)
        out["with_optional_pass"] = bool(hours + optional_hours <= PART_A_GPU_CAP_H)
        out["rule"] = ("optional models run only if the with-optional projection fits the "
                       "cap; otherwise they are dropped (§8.1)")
    return out


def cpu_projection(*, cap_hours=CPU_CAP_H) -> dict:
    hours = float(sum(CPU_PROJECTION.values()))
    return {"tasks": dict(CPU_PROJECTION), "projected_hours": round(hours, 3),
            "cap_hours": float(cap_hours), "pass": bool(hours <= float(cap_hours))}


def s1_projection(root=None, *, part_b_cap_h=PART_B_GPU_CAP_H) -> dict:
    """Front-matter §F3 combined S1 rule: L40S ledger + both remaining caps <= 160."""
    ledger = read_json(_resolve_root(root) / "docs/jepa/RESOURCE_LEDGER.json", {}) or {}
    entry = ((ledger.get("v4_caps") or {}).get("E2_lucid_l40s_deepthought2") or {})
    charged = entry.get("charged_gpu_hours")
    if charged is None:
        return {"available": False, "pass": None,
                "rule": "L40S ledger + Part A 4.0 + Part B 1.5 <= 160"}
    total = float(charged) + float(part_b_cap_h) + float(PART_A_GPU_CAP_H)
    return {"available": True, "charged_gpu_hours": float(charged),
            "part_a_cap_hours": float(PART_A_GPU_CAP_H),
            "part_b_cap_hours": float(part_b_cap_h),
            "sum_hours": round(total, 4), "ceiling_hours": S1_CEILING_H,
            "pass": bool(total <= S1_CEILING_H),
            "rule": "L1 ledger + Part A cap + Part B cap <= 160 (front matter §F3)"}


def entries_snapshot(guard_json=GUARD_JSON) -> dict:
    return ef.entries_snapshot(guard_json)


def entries_projection(free_before, *, clone=ENTRIES_CLONE, suite=ENTRIES_SUITE,
                       extension=ENTRIES_EXT, floor=ENTRIES_FLOOR) -> dict:
    return ef.entries_projection(free_before, clone=clone, suite=suite,
                                 extension=extension, floor=floor)


def entries_preflight(*, guard_json=GUARD_JSON, log=print) -> dict:
    snapshot = entries_snapshot(guard_json)
    out = {"snapshot": snapshot, "pass": False}
    if not snapshot.get("available"):
        out["error"] = snapshot.get("error", "no free_entries in the guard status")
        return out
    out["projection"] = entries_projection(snapshot["free_entries"])
    out["gpu_slot_available"] = bool(snapshot.get("guard_level") != "HOLD")
    out["pass"] = bool(out["projection"]["pass"])
    return out


# ================================================================ X0-a §4
def _run_manifest(dataset, row):
    path = dataset.directory / "runs" / row["run_dir"] / "manifest.json"
    payload = read_json(path, None)
    return payload if isinstance(payload, dict) else {}


def check_run_states(dataset, geometry, *, limit=None, log=lambda *a, **k: None) -> dict:
    """§2.2/§4: recompute each run's ``state_sha256`` and check the twin and events."""
    out = {"key": dataset.key, "n_runs": len(dataset.runs), "n_checked": 0,
           "mismatches": [], "twin_missing": [], "pass": True}
    for index, row in enumerate(dataset.runs):
        if limit is not None and index >= int(limit):
            break
        out["n_checked"] += 1
        run_dir = dataset.directory / "runs" / row["run_dir"]
        manifest = _run_manifest(dataset, row)
        expected = manifest.get("state_sha256") or row.get("state_sha256")
        actual = run_spec(row, dataset).state_sha256(geometry)
        run_id = str(row["run_id"])
        if actual != expected:
            out["mismatches"].append({"run_id": run_id, "field": "state_sha256",
                                      "expected": expected, "actual": actual})
        manifest_twin = manifest.get("twin_present", row.get("twin_present"))
        twin = run_dir / "twin_hits.npz"
        hits = run_dir / "hits.npz"
        if manifest_twin is not True or not twin.is_file():
            out["twin_missing"].append({"run_id": run_id, "twin_present": manifest_twin})
        elif hits.is_file():
            left = lb.read_sparse_hits(hits)
            right = lb.read_sparse_hits(twin)
            if not np.array_equal(np.asarray(left.event_ids), np.asarray(right.event_ids)):
                out["mismatches"].append({"run_id": run_id,
                                          "field": "twin event_ids differ"})
        if not (run_dir / "events.npz").is_file():
            out["mismatches"].append({"run_id": run_id, "field": "events.npz missing"})
        else:
            with np.load(run_dir / "events.npz", allow_pickle=False) as data:
                for field in ("expected_pe", "domain"):
                    if field not in data.files:
                        out["mismatches"].append(
                            {"run_id": run_id, "field": f"events.npz lacks {field}"})
    out["pass"] = bool(not out["mismatches"] and not out["twin_missing"])
    return out


def _pair_run_rows(dataset_a, dataset_b):
    """Zip two probe sets on run id; a differing run grid is a content problem."""
    rows_a = {str(row["run_id"]): row for row in dataset_a.runs}
    rows_b = {str(row["run_id"]): row for row in dataset_b.runs}
    if set(rows_a) != set(rows_b):
        raise ValueError("the two probe sets do not share their run-id grid")
    return [(rows_a[run_id], rows_b[run_id]) for run_id in sorted(rows_a)]


def check_twin_content(sets005, sets030, *, limit=None) -> dict:
    """§4: the twin_hits arrays of a run are equal in the s = 0.05 and s = 0.30 sets."""
    out = {"n_runs": 0, "n_events": 0, "mismatches": [], "pass": True}
    for role in ("train", "test"):
        if role not in sets005 or role not in sets030:
            continue
        pairs = _pair_run_rows(sets005[role], sets030[role])
        for index, (row_a, row_b) in enumerate(pairs):
            if limit is not None and index >= int(limit):
                break
            path_a = (sets005[role].directory / "runs" / row_a["run_dir"]
                      / "twin_hits.npz")
            path_b = (sets030[role].directory / "runs" / row_b["run_dir"]
                      / "twin_hits.npz")
            if not (path_a.is_file() and path_b.is_file()):
                out["mismatches"].append({"role": role, "run_id": row_a["run_id"],
                                          "field": "twin_hits.npz missing"})
                continue
            left, right = lb.read_sparse_hits(path_a), lb.read_sparse_hits(path_b)
            out["n_runs"] += 1
            out["n_events"] += int(np.asarray(left.event_ids).size)
            for field in ("event_ids", "offsets", "pmt_id", "charge_pe", "time_ns"):
                if not np.array_equal(np.asarray(getattr(left, field)),
                                      np.asarray(getattr(right, field))):
                    out["mismatches"].append({"role": role, "run_id": row_a["run_id"],
                                              "field": field})
    out["pass"] = not out["mismatches"]
    return out


def check_swtw_content(sets005, sets030, geometry, *, limit=None,
                       rtol=SWTW_CHARGE_RTOL) -> dict:
    """§4: SWtw charges equal the s = 0.05 charges on the hits common to both."""
    out = {"n_events": 0, "n_common_hits": 0, "max_relative_difference": 0.0,
           "mismatches": [], "pass": True, "rtol": float(rtol)}
    for role in ("train", "test"):
        if role not in sets005 or role not in sets030:
            continue
        pairs = _pair_run_rows(sets005[role], sets030[role])
        for index, (row_a, row_b) in enumerate(pairs):
            if limit is not None and index >= int(limit):
                break
            store_n = lb.read_sparse_hits(sets005[role].directory / "runs"
                                          / row_a["run_dir"] / "hits.npz")
            store_d = lb.read_sparse_hits(sets030[role].directory / "runs"
                                          / row_b["run_dir"] / "hits.npz")
            index_n = {int(event): position
                       for position, event in enumerate(np.asarray(store_n.event_ids))}
            index_d = {int(event): position
                       for position, event in enumerate(np.asarray(store_d.event_ids))}
            common_events = sorted(set(index_n) & set(index_d))
            if not common_events:
                continue
            s0, s1 = state_pair(row_b, sets030[role], geometry)
            factor = s0.gains / s1.gains
            out["n_events"] += len(common_events)
            for event in common_events:
                span_n = slice(int(store_n.offsets[index_n[event]]),
                               int(store_n.offsets[index_n[event] + 1]))
                span_d = slice(int(store_d.offsets[index_d[event]]),
                               int(store_d.offsets[index_d[event] + 1]))
                pmt_n = np.asarray(store_n.pmt_id[span_n], np.int64)
                pmt_d = np.asarray(store_d.pmt_id[span_d], np.int64)
                common, index_common_n, index_common_d = np.intersect1d(
                    pmt_n, pmt_d, return_indices=True)
                if not common.size:
                    continue
                charge_n = np.asarray(store_n.charge_pe[span_n], np.float64)[index_common_n]
                charge_d = np.asarray(store_d.charge_pe[span_d], np.float64)[index_common_d]
                transformed = charge_d * factor[common]
                denominator = np.maximum(np.abs(charge_n), np.finfo(np.float32).tiny)
                relative = np.abs(transformed - charge_n) / denominator
                out["n_common_hits"] += int(common.size)
                if relative.size:
                    worst = float(relative.max())
                    out["max_relative_difference"] = max(
                        out["max_relative_difference"], worst)
                    if worst > float(rtol):
                        out["mismatches"].append({
                            "role": role, "run_id": row_a["run_id"],
                            "event_id": int(event), "max_relative_difference": worst})
    out["pass"] = not out["mismatches"]
    return out


def mu_reconstruction(dataset, geometry, gen, *, dataset_seed, limit_runs=None,
                      rtol=1e-6) -> dict:
    """§2.2 derived truth: Σ_p μ_p(e) equals the stored ``expected_pe[e]``.

    Uses the keyed QE stream of ``draw_event`` and no Poisson draw, no
    ``draw_event`` and no readout: this is an analytic recomputation, not data
    generation (§2.2/§2.6).
    """
    dataset_seed = int(dataset_seed)
    fixed = lb.fixed_charge_gain(geometry)
    state = lb.nominal_state(geometry)
    out = {"key": dataset.key, "n_events": 0, "max_relative_difference": 0.0,
           "mismatches": [], "pass": True, "rtol": float(rtol)}
    for index, row in enumerate(dataset.runs):
        if limit_runs is not None and index >= int(limit_runs):
            break
        run_dir = dataset.directory / "runs" / row["run_dir"]
        store = lb.read_sparse_hits(run_dir / "hits.npz")
        with np.load(run_dir / "events.npz", allow_pickle=False) as data:
            event_ids = np.asarray(data["event_ids"], np.int64)
            vertex = np.asarray(data["vertex_m"], float)
            yields = np.asarray(data["yield_photons"], float)
            expected = np.asarray(data["expected_pe"], np.float64)
            domains = (np.asarray(data["domain"]).astype(str)
                       if "domain" in data.files else np.full(event_ids.size, "photon"))
        position = {int(event): place for place, event in enumerate(event_ids)}
        for event_id in np.asarray(store.event_ids, np.int64):
            place = position.get(int(event_id))
            if place is None or str(domains[place]) != lb.DOMAIN_PHOTON:
                continue
            run_id = str(row["run_id"])
            qe = lb.sample_qe(
                geometry,
                lb.keyed_rng(dataset_seed, lb.NS_DRAWS, run_id, int(event_id),
                             "wavelength"), gen)
            mu = (lb.expected_charge(geometry, vertex[place], yields[place], qe,
                                     state.attenuation_length_m, gen) * fixed)
            total = float(np.sum(mu))
            out["n_events"] += 1
            stored = float(expected[place])
            denominator = max(abs(stored), 1e-30)
            relative = abs(total - stored) / denominator
            out["max_relative_difference"] = max(out["max_relative_difference"], relative)
            if relative > float(rtol):
                out["mismatches"].append({"run_id": run_id, "event_id": int(event_id),
                                          "recomputed": total, "stored": stored,
                                          "relative_difference": relative})
    out["pass"] = not out["mismatches"]
    return out


def preflight(root=None, *, data_root=DATA_ROOT_DEFAULT, config, gen, geometry,
              inventory=None, device="cpu", limit=None, log=print) -> dict:
    """X0-a preflight (CPU, read-only): ``plan.json`` (§4)."""
    root = _resolve_root(root)
    gate = assert_authorised("preflight", root=root, log=log)
    started = time.time()
    inv = inventory if inventory is not None else part_a_inventory(root, log=log)
    payload = {"registered": False, "extension": EXTENSION, "check": "X0-a preflight §4",
               "runbook": RUNBOOK_REL, "runbook_sha256": (
                   lr.sha256_file(root / RUNBOOK_REL)
                   if (root / RUNBOOK_REL).is_file() else None),
               "code_sha": lr.git_head(root), "git_status": lr.git_status(root),
               "data_root": str(data_root), "gate": {"status": gate["status"]},
               "stops": [], "wall_seconds": None}
    payload["data"] = data_status(data_root)
    if not payload["data"]["pass"]:
        payload["stops"].append("pinned data manifest missing or mismatched (§2.1)")
    try:
        probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
        probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    except SystemExit:
        raise
    except Exception as problem:  # pragma: no cover - host data problem
        payload["stops"].append(f"probe sets unavailable: {type(problem).__name__}: {problem}")
        probes005 = probes030 = None
    payload["runs"] = {}
    if probes005 is not None:
        for key, dataset in (("s005_train", probes005["train"]),
                             ("s005_test", probes005["test"]),
                             ("s030_train", probes030["train"]),
                             ("s030_test", probes030["test"])):
            payload["runs"][key] = check_run_states(dataset, geometry, limit=limit, log=log)
        payload["twin_content"] = check_twin_content(probes005, probes030, limit=limit)
        payload["swtw_content"] = check_swtw_content(probes005, probes030, geometry,
                                                     limit=limit)
        payload["mu_reconstruction"] = mu_reconstruction(
            probes030["test"], geometry, gen,
            dataset_seed=int(config["data"]["dataset_seed"]), limit_runs=limit)
        # §4: a content mismatch stops only the readings that rely on it
        blocked = {"twin_mismatch": [], "swtw_mismatch": [], "mu_p_unavailable": []}
        if not payload["twin_content"]["pass"]:
            blocked["twin_mismatch"] = ["HW condition (X1, X1-d, X1-g secondary, X3-b)"]
        if not payload["swtw_content"]["pass"]:
            blocked["swtw_mismatch"] = ["X1-b", "X1-g", "X1-d's v_HW − v_N", "X2-a",
                                        "X2-b", "X2-c"]
        if not payload["mu_reconstruction"]["pass"]:
            blocked["mu_p_unavailable"] = ["X2-d source stream"]
        roles = {}
        for name, dataset in (("s005_train", probes005["train"]),
                              ("s005_test", probes005["test"]),
                              ("s030_train", probes030["train"]),
                              ("s030_test", probes030["test"])):
            roles[name] = "available" if all(
                (dataset.directory / "runs" / row["run_dir"]
                 / "twin_hits.npz").is_file() for row in dataset.runs) else "unavailable"
        payload["hw_condition"] = roles
        payload["readings_stopped"] = blocked
        payload["content_pass"] = bool(payload["twin_content"]["pass"]
                                       and payload["swtw_content"]["pass"]
                                       and payload["mu_reconstruction"]["pass"])
    payload["inventory"] = inv
    if inv["unavailable"]:
        payload["stops"].append("checkpoint inventory: "
                                f"{len(inv['unavailable'])} unavailable row(s)")
    payload["hardware_and_software"] = hardware_and_software()
    payload["gpu_projection"] = gpu_projection()
    payload["gpu_projection_core"] = gpu_projection(list(CORE_MODELS))
    payload["cpu_projection"] = cpu_projection()
    payload["entries"] = entries_preflight(log=log)
    payload["s1"] = s1_projection(root)
    for name in ("entries", "s1", "gpu_projection", "cpu_projection"):
        row = payload[name]
        if row.get("pass") is False:
            payload["stops"].append(f"{name} projection failed (§8)")
    if payload["gpu_projection"].get("pass") is False:
        if payload["gpu_projection_core"].get("pass"):
            payload["optional_dropped"] = True
            payload["stops"].append(
                "optional models dropped: the with-optional GPU projection "
                "exceeds the cap (§8.1)")
        else:
            payload["stops"].append("core GPU projection exceeds 4.0 L0 h (STOP S1)")
    payload["pass"] = bool(not payload["stops"])
    payload["closure"] = _assert_closure("preflight", root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    return payload


# ================================================================ X0-b §4
def check_reproduction(label, got, expected, *, tolerance=X0B_TOLERANCE_M) -> dict:
    """One X0-b value against its committed target (§4)."""
    if expected is None:
        return {"label": label, "expected": None, "got": got, "pass": True,
                "skipped": True}
    difference = abs(float(got) - float(expected))
    return {"label": label, "expected": float(expected), "got": float(got),
            "difference_m": difference, "tolerance_m": float(tolerance),
            "pass": bool(difference <= float(tolerance))}


def _frozen_refit_vertex(model, probes, condition, split, config, device, *,
                         cache=None, log=print) -> dict:
    """N (frozen = refit) and D (refit and frozen) vertex rows of one model.

    The caller passes the single-setting probe dict
    ``{"train", "test", "geometry"}``; the §2.5 condition is built per role on the
    dataset it is about to stream (the same convention as ``identity_check``), so
    the train run directories are inside their own condition's set.
    """
    geometry = probes["geometry"]
    rows = {}
    for role, dataset in (("train", probes["train"]), ("test", probes["test"])):
        role_condition = build_condition(condition["name"], dataset, geometry)
        rows[role] = stream_condition_embeddings(
            model, dataset, geometry, role_condition, split=split,
            config=config, device=device, cache=cache,
            cache_key=embedding_cache_key(role, condition["name"]), log=log)
    fitted = fit_vertex_probe(rows["train"])
    prediction = la.probe_predictions(
        fitted, ef._probe_dataset(rows["test"], "vertex", target_names=("x", "y", "z")))
    vertex = lr._vertex_error(prediction, np.asarray(rows["test"]["vertex"], float))
    return {"rows": rows, "fitted": fitted, "prediction": prediction,
            "errors": ef._vertex_errors(rows["test"], prediction),
            "vertex": vertex, "mean_error_m": vertex.get("resolution")}


def reproduction(root=None, *, config, gen, arms_config, geometry,
                 inventory=None, data_root=DATA_ROOT_DEFAULT, device="cpu",
                 ext_root=EXT_ROOT, limit=None, log=print) -> dict:
    """X0-b reproduction gate (GPU, one job, one process, one GPU; §4)."""
    root = _resolve_root(root)
    gate = assert_authorised("reproduce", root=root, log=log)
    inv = inventory if inventory is not None else part_a_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    identity = build_condition("N", probes005["test"], geometry)
    condition_n = {"name": "N", "train": "s005", "test": "s005", "condition": identity}
    drift = build_condition("D", probes030["test"], geometry)
    condition_d = {"name": "D", "train": "s030", "test": "s030", "condition": drift}
    sets = {"s005": dict(probes005, geometry=geometry),
            "s030": dict(probes030, geometry=geometry)}
    payload = {"registered": False, "extension": EXTENSION,
               "check": "X0-b reproduction gate §4", "runbook": RUNBOOK_REL,
               "tolerance_m": X0B_TOLERANCE_M, "models": {},
               "gate": {"status": gate["status"]}, "pass": True, "wall_seconds": None}
    for label in X0B_TARGETS:
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            payload["models"][label] = {"status": "unavailable",
                                        "reason": "inventory row unavailable"}
            payload["pass"] = False
            continue
        cache = embedding_cache(ext_root, label)
        with ef.item_eval_context(row):
            arm = ef.build_item_arm(row, config=config, gen=gen, arms_config=arms_config,
                                    geometry=geometry, data_root=data_root,
                                    device=device, log=log)
        model = getattr(arm, "jepa", arm)
        split = row_split(row)
        n = _frozen_refit_vertex(model, dict(sets["s005"], geometry=geometry), condition_n,
                                 split, config, device, cache=cache, log=log)
        d = _frozen_refit_vertex(model, dict(sets["s030"], geometry=geometry), condition_d,
                                 split, config, device, cache=cache, log=log)
        # D frozen: the N probe applied unchanged to the D test embeddings
        frozen_d = la.probe_predictions(
            n["fitted"],
            ef._probe_dataset(d["rows"]["test"], "vertex", target_names=("x", "y", "z")))
        v_d_frozen = lr._vertex_error(frozen_d, np.asarray(d["rows"]["test"]["vertex"],
                                                           float))
        targets = X0B_TARGETS[label]
        comparisons = {
            "N_refit_frozen": check_reproduction(label, n["mean_error_m"], targets[0]),
            "D_refit": check_reproduction(label, d["mean_error_m"], targets[1]),
            "D_frozen": check_reproduction(label, v_d_frozen.get("resolution"),
                                           targets[2]),
        }
        entry = {"status": "evaluated",
                 "N_refit_frozen_m": n["mean_error_m"],
                 "D_refit_m": d["mean_error_m"],
                 "D_frozen_m": v_d_frozen.get("resolution"),
                 "comparisons": comparisons,
                 "pass": all(row["pass"] for row in comparisons.values())}
        payload["models"][label] = entry
        payload["pass"] = bool(payload["pass"] and entry["pass"])
    payload["identity"] = identity_check(root=root, config=config, gen=gen,
                                         arms_config=arms_config, geometry=geometry,
                                         data_root=data_root, inventory=inv, device=device,
                                         label="mae_within_trigger9000",
                                         split=row_split(inv["rows"].get(
                                             "mae_within_trigger9000") or {}),
                                         log=log)
    payload["pass"] = bool(payload["pass"] and payload["identity"]["pass"])
    payload["closure"] = _assert_closure("reproduce", root)
    payload["wall_seconds"] = None
    return payload


def identity_check(*, root=None, config, gen, arms_config, geometry,
                   data_root=DATA_ROOT_DEFAULT, inventory=None, device="cpu",
                   label="mae_within_trigger9000", limit=None, probes005=None,
                   probes030=None, model=None, split=None, log=print) -> dict:
    """§4 identity check: factor 1 through §2.5 is bit-identical, embeddings <= 1e-6.

    The conditions are built per role (N on the s = 0.05 sets, D on the s = 0.30
    sets), so both the train and the test run directories are inside the condition's
    own set; the view split comes from the model's inventory row.
    """
    root = _resolve_root(root)
    inv = inventory if inventory is not None else part_a_inventory(root, log=log)
    if probes005 is None:
        probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005,
                                    log=log)
    if probes030 is None:
        probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030,
                                    log=log)
    row = inv["rows"].get(label) or {}
    if model is None:
        if not row:
            return {"pass": False, "reason": f"{label} absent from the inventory"}
        with ef.item_eval_context(row):
            arm = ef.build_item_arm(row, config=config, gen=gen,
                                    arms_config=arms_config, geometry=geometry,
                                    data_root=data_root, device=device, log=log)
        model = getattr(arm, "jepa", arm)
    split = row_split(row) if split is None else str(split)
    mask_seed = int(config["views"]["mask_seed"])
    scaling = pe.FeatureScaling.from_geometry(geometry)
    out = {"label": label, "sets": {}, "pass": True, "split": split,
           "tolerance": IDENTITY_EMBEDDING_TOLERANCE, "features_bit_identical": True}
    import torch
    for name, probes, condition_name in (("s005", probes005, "N"),
                                         ("s030", probes030, "D")):
        entry = {"n_runs": 0, "max_embedding_difference": 0.0,
                 "features_bit_identical": True, "pass": True}
        for role in ("train", "test"):
            if role not in probes:
                continue
            dataset = _truncate(probes[role], max_runs=limit)
            condition = build_condition(condition_name, dataset, geometry)
            for run_row in dataset.runs:
                source = dataset.directory / "runs" / run_row["run_dir"]
                plain = lr._probe_batch(source, geometry, split=split,
                                        mask_seed=mask_seed, occupancy="natural",
                                        scaling=scaling, limit=8)
                with hit_transform_context(condition):
                    wrapped = lr._probe_batch(source, geometry, split=split,
                                              mask_seed=mask_seed, occupancy="natural",
                                              scaling=scaling, limit=8)
                plain_features = np.asarray(plain.features.detach().cpu())
                wrapped_features = np.asarray(wrapped.features.detach().cpu())
                if not np.array_equal(plain_features, wrapped_features):
                    entry["features_bit_identical"] = False
                    entry["pass"] = False
                encode_device = (device if isinstance(device, torch.device)
                                 else torch.device(device))
                with torch.no_grad():
                    left = model.encode(plain.features.to(encode_device))
                    right = model.encode(wrapped.features.to(encode_device))
                difference = float(np.max(np.abs(np.asarray(left.detach().cpu())
                                                  - np.asarray(right.detach().cpu())))) \
                    if left.numel() else 0.0
                entry["max_embedding_difference"] = max(
                    entry["max_embedding_difference"], difference)
                if difference > IDENTITY_EMBEDDING_TOLERANCE:
                    entry["pass"] = False
                entry["n_runs"] += 1
        out["sets"][name] = entry
        out["features_bit_identical"] = bool(out["features_bit_identical"]
                                             and entry["features_bit_identical"])
        out["pass"] = bool(out["pass"] and entry["pass"])
    return out


# ================================================================== X1 §5
PHOTON_CONDITIONS_X1 = ("N", "D", "SWtw", "SWunit", "SWmod", "SWunitN", "HW")
PHOTON_REFIT_VERTEX = ("D", "SWtw", "SWunit", "HW")
LOG_YIELD_FROZEN = ("N", "D", "SWtw", "SWunit", "HW")
LOG_YIELD_REFIT = ("D", "SWunit", "HW")
ENERGY_CONDITIONS = ("N", "D", "SWunit", "HW")
PARTICLE_CONDITIONS = ("N", "D", "SWunit", "HW")


def condition_names_x1(estimates=None) -> list:
    """Every X1 photon-set condition, in a fixed order."""
    names = list(PHOTON_CONDITIONS_X1)
    names += [f"NOISY({sigma:g})" for sigma in NOISY_SIGMAS]
    if estimates:
        names += [f"EST({stream})" for stream in ("source", "raw", "latent")]
    return names


def build_conditions(datasets, geometry, names, *, estimates=None) -> dict:
    """Build every named condition on the matching probe split of one role."""
    out = {}
    for name in names:
        base, _sigma, _stream = _condition_base(name)
        source = datasets["s005"] if _SOURCE[base] == S005 else datasets["s030"]
        out[name] = build_condition(name, source, geometry, estimates=estimates)
    return out


def bootstrap_difference(left_rows, left_errors, right_rows, right_errors) -> dict:
    """Run-cluster bootstrap of ``right - left`` for mean error and RMS (§3.2)."""
    paired = ef.pair_events(left_rows, right_rows)
    left = np.asarray(left_errors, float)[paired["index_left"]]
    right = np.asarray(right_errors, float)[paired["index_right"]]
    mean = ef.paired_run_bootstrap(left, right, paired["run_ids"])
    spread = paired_cluster_bootstrap(left, right, paired["run_ids"], statistic=rms)
    return {"mean": mean, "rms": spread, "n_events": int(paired["n_events"]),
            "n_runs": int(paired["n_runs"])}


def vertex_row(rows, *, fitted) -> dict:
    """Frozen-probe vertex metrics on one condition's test rows."""
    test = ef._probe_dataset(rows, "vertex", target_names=("x", "y", "z"))
    prediction = la.probe_predictions(fitted, test)
    vertex = lr._vertex_error(prediction, test.y)
    errors = ef._vertex_errors(rows, prediction)
    return {"mean_error_m": vertex.get("resolution"),
            "median_error_m": vertex.get("median"), "status": vertex.get("status"),
            "n_events": int(errors.size), "run_ids": np.asarray(rows["run_ids"]),
            "event_ids": np.asarray(rows["event_ids"]), "__errors__": errors}


def log_yield_row(rows, *, fitted) -> dict:
    """Frozen-probe log-yield resolution and R² on one condition's test rows."""
    test = ef._probe_dataset(rows, "log_yield", y=rows["log_yield"],
                             target_names=("log_yield",))
    prediction = la.probe_predictions(fitted, test)
    errors = np.asarray(prediction, float).ravel() - np.asarray(rows["log_yield"],
                                                                float).ravel()
    keep = np.isfinite(errors)
    run_ids = np.asarray(rows["run_ids"], dtype=object)[keep]
    event_ids = np.asarray(rows["event_ids"], np.int64)[keep]
    return {"resolution": (float(np.sqrt(np.mean(errors[keep] ** 2)))
                           if keep.any() else None),
            "ci95": (cluster_bootstrap(errors[keep], run_ids, statistic=rms)["ci95"]
                     if keep.any() else None),
            "r2": _r2_from_predictions(prediction, test.y),
            "n_events": int(keep.sum()), "errors": errors[keep],
            "run_ids": run_ids, "event_ids": event_ids}


def _mean_difference(comparison) -> float | None:
    return ((comparison or {}).get("mean") or {}).get("mean_difference_m")


def _ci95(comparison):
    return ((comparison or {}).get("mean") or {}).get("ci95_m")


def x1(root=None, *, config, gen, arms_config, geometry, inventory=None,
       data_root=DATA_ROOT_DEFAULT, device="cpu", ext_root=EXT_ROOT, labels=None,
       include_optional=True, limit=None, log=print) -> dict:
    """X1: what does calibration recover?  §5 frozen-model readouts."""
    root = _resolve_root(root)
    gate = assert_authorised("x1", root=root, log=log)
    inv = inventory if inventory is not None else part_a_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    particle_truth = {role: ef.load_particle_truth(data_root, PROBE_KEYS_S005[role])
                      for role in ("train", "test")}
    hw_available = _twins_present(probes030["train"]) and \
        _twins_present(probes030["test"])
    names = [name for name in condition_names_x1()
             if name != "HW" or hw_available]
    conditions = {role: build_conditions(
        {"s005": probes005[role], "s030": probes030[role]}, geometry, names)
        for role in ("train", "test")}
    payload = {"registered": False, "extension": EXTENSION, "check": "X1 §5 calibration",
               "runbook": RUNBOOK_REL, "conditions": names, "models": {},
               "unavailable": [], "hw_unavailable": not hw_available,
               "gate": {"status": gate["status"]}, "wall_seconds": None}
    selected = model_rows(inv, include_optional=include_optional) if labels is None \
        else list(labels)
    for label in selected:
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            payload["unavailable"].append({"label": label, "reason": "inventory unavailable"})
            continue
        cache = embedding_cache(ext_root, label)
        try:
            with ef.item_eval_context(row):
                arm = ef.build_item_arm(row, config=config, gen=gen,
                                        arms_config=arms_config, geometry=geometry,
                                        data_root=data_root, device=device, log=log)
            model = getattr(arm, "jepa", arm)
            split = row_split(row)
            photon = {role: {} for role in ("train", "test")}
            for role in ("train", "test"):
                for name in names:
                    base = _condition_base(name)[0]
                    dataset = (probes005[role] if _SOURCE[base] == S005
                               else probes030[role])
                    photon[role][name] = stream_condition_embeddings(
                        model, dataset, geometry, conditions[role][name],
                        split=split, config=config, device=device, cache=cache,
                        cache_key=embedding_cache_key(role, name),
                        max_events=limit, log=log)
            particle = None
            if label in X1_PARTICLE_MODELS:
                particle_names = tuple(name for name in PARTICLE_CONDITIONS
                                       if name != "HW" or hw_available)
                particle = _x1_particle(model, particle_truth, geometry, probes005,
                                        probes030, conditions, label=label,
                                        names=particle_names, cache=cache,
                                        device=device, limit=limit, log=log)
            model_payload = _x1_model(row, photon, particle, config=config)
        except SystemExit:
            raise
        except Exception as problem:  # pragma: no cover - a missing checkpoint/GPU
            payload["unavailable"].append({
                "label": label, "reason": f"{type(problem).__name__}: {problem}"})
            continue
        payload["models"][label] = model_payload
    payload["readings"] = x1_readings(payload)
    payload = _strip_internal(payload)
    payload["complete"] = not payload["unavailable"]
    payload["closure"] = _assert_closure("x1", root)
    return payload


def _twins_present(dataset) -> bool:
    """True when every run of a set has its nominal twin on disk (§4 HW condition)."""
    return all((Path(dataset.directory) / "runs" / row["run_dir"]
                / "twin_hits.npz").is_file() for row in dataset.runs)


def _x1_particle(model, particle_truth, geometry, probes005, probes030, conditions,
                 *, label=None, names=None, cache=None, device="cpu", limit=None,
                 log=print) -> dict:
    """The §5 particle sets of the six listed models.

    The four MAE models also carry the D probe-train set, which the X3-b particle
    eraser needs; every particle embedding is cached for that item.
    """
    out = {"train": {}, "test": {}}
    particle_names = tuple(PARTICLE_CONDITIONS if names is None else names)
    plan = {"train": (("N", "D") if label in MAE_MODELS else ("N",)),
            "test": particle_names}
    for role in ("train", "test"):
        for name in plan[role]:
            base = _condition_base(name)[0]
            dataset = probes005[role] if _SOURCE[base] == S005 else probes030[role]
            out[role][name] = stream_condition_particles(
                model, dataset, particle_truth[role], geometry, conditions[role][name],
                device=device, cache=cache,
                cache_key=f"particle__{role}__{name}", max_events=limit, log=log)
    return out


def attach_x1_canonical(out) -> dict:
    """Attach the R-1 canonical X1 keys to one per-model record (Part A outputs)."""
    comparisons = out.get("comparisons") or {}
    delta = _mean_difference(comparisons.get("D"))
    out["delta_d_m"] = delta
    out["delta_d_eligible"] = eligibility(delta, _ci95(comparisons.get("D")))
    v_n, v_d = _frozen_mean(out, "N"), _frozen_mean(out, "D")
    v_swtw, v_swunit = _frozen_mean(out, "SWtw"), _frozen_mean(out, "SWunit")
    denominator = (None if v_n is None or v_d is None or abs(float(v_d) - float(v_n)) <= 0
                   else float(v_d) - float(v_n))
    out["rho_swtw"] = (None if denominator is None or v_swtw is None
                       else float((float(v_d) - float(v_swtw)) / denominator))
    out["rho_swunit"] = (None if denominator is None or v_swunit is None
                         else float((float(v_d) - float(v_swunit)) / denominator))
    out["share_cs"] = _x1g(out).get("share_cs")
    return out


def attach_x2_canonical(out) -> dict:
    """Attach the R-1 canonical X2 keys to one per-model record (Part A outputs)."""
    occupancy = (out.get("X2-b") or {}).get("occupancy") or {}
    out["r2_eq"] = occupancy.get("r2_cs")
    out["class"] = occupancy.get("class")
    out["rho_lr"] = (out.get("X2-c") or {}).get("rho_LR")
    return out


def _x1_model(row, photon, particle, *, config) -> dict:
    """One model's §5 readout block, photon and particle."""
    label = str(row["label"])
    family, seed = _family(label)
    out = {"label": label, "family": family, "seed": seed, "arm": row.get("arm"),
           "split": row.get("split"), "checkpoint_sha256": row.get("sha256"),
           "data_setting": float(row.get("data_setting", S005)),
           "conditions": {}, "comparisons": {}, "particle": {}}
    frozen_vertex = fit_vertex_probe(photon["train"]["N"])
    frozen_yield = fit_log_yield_probe(photon["train"]["N"])
    for name in list(photon["test"]):
        entry = {"vertex": vertex_row(photon["test"][name], fitted=frozen_vertex),
                 "log_yield": log_yield_row(photon["test"][name], fitted=frozen_yield)}
        if name in PHOTON_REFIT_VERTEX:
            entry["refit_vertex"] = vertex_row(
                photon["test"][name], fitted=fit_vertex_probe(photon["train"][name]))
        else:
            entry["refit_vertex"] = None
        if name in LOG_YIELD_REFIT:
            entry["refit_log_yield"] = log_yield_row(
                photon["test"][name], fitted=fit_log_yield_probe(photon["train"][name]))
        else:
            entry["refit_log_yield"] = None
        out["conditions"][name] = entry
    frozen = {name: out["conditions"][name]["vertex"]["__errors__"]
              for name in out["conditions"]}
    n_rows = photon["test"]["N"]
    for name in ("D", "SWtw", "SWunit", "SWmod", "SWunitN", "HW"):
        if name in out["conditions"]:
            out["comparisons"][name] = bootstrap_difference(
                n_rows, frozen["N"], photon["test"][name], frozen[name])
    if out["conditions"]["D"]["refit_vertex"] is not None:
        out["comparisons"]["D_refit"] = bootstrap_difference(
            n_rows, frozen["N"], photon["test"]["D"],
            out["conditions"]["D"]["refit_vertex"]["__errors__"])
    for name in ("SWtw", "SWunit"):
        if out["conditions"][name]["refit_vertex"] is not None:
            out["comparisons"][f"{name}_refit"] = bootstrap_difference(
                photon["test"]["D"], out["conditions"]["D"]["refit_vertex"]["__errors__"],
                photon["test"][name],
                out["conditions"][name]["refit_vertex"]["__errors__"])
    for name in ("SWunit",):
        out["comparisons"]["SWunit_vs_D_refit"] = bootstrap_difference(
            photon["test"]["D"], out["conditions"]["D"]["refit_vertex"]["__errors__"],
            photon["test"][name], frozen[name])
        out["comparisons"]["SWunit_refit_vs_D_refit"] = bootstrap_difference(
            photon["test"]["D"], out["conditions"]["D"]["refit_vertex"]["__errors__"],
            photon["test"][name],
            out["conditions"][name]["refit_vertex"]["__errors__"])
    if "HW" in out["conditions"] and "SWunit" in out["conditions"]:
        out["comparisons"]["SWunit_vs_HW"] = bootstrap_difference(
            photon["test"]["HW"], frozen["HW"], photon["test"]["SWunit"],
            frozen["SWunit"])
    # §5 X1-g secondary (descriptive): the unit-gain reference split
    hw_refit = (out["conditions"]["HW"]["refit_vertex"]
                if "HW" in out["conditions"] else None)
    d_refit = (out["conditions"]["D"]["refit_vertex"]
               if "D" in out["conditions"] else None)
    swunit_refit = (out["conditions"]["SWunit"]["refit_vertex"]
                    if "SWunit" in out["conditions"] else None)
    if hw_refit is not None and d_refit is not None:
        out["comparisons"]["D_refit_vs_HW_refit"] = bootstrap_difference(
            photon["test"]["D"], d_refit["__errors__"], photon["test"]["HW"],
            hw_refit["__errors__"])
    if hw_refit is not None and swunit_refit is not None:
        out["comparisons"]["SWunit_refit_vs_HW_refit"] = bootstrap_difference(
            photon["test"]["SWunit"], swunit_refit["__errors__"],
            photon["test"]["HW"], hw_refit["__errors__"])
    if hw_refit is not None:
        out["comparisons"]["N_vs_HW_refit"] = bootstrap_difference(
            n_rows, frozen["N"], photon["test"]["HW"], hw_refit["__errors__"])
    attach_x1_canonical(out)
    if particle:
        frozen_energy = fit_energy_probe(particle["train"]["N"])
        for name in list(particle["test"]):
            rows = particle["test"][name]
            out["particle"][name] = {
                "energy": energy_metrics(particle["train"]["N"], rows,
                                         fitted=frozen_energy),
                "n_events": int(np.asarray(rows["event_ids"]).size)}
    return out


# ------------------------------------------------------------ X1 readings §5
def sigma_half(rho_noisy, rho_swunit, sigmas=NOISY_SIGMAS) -> dict:
    """§5 X1-e: the first grid σ with ρ_NOISY ≤ ½ ρ_SWunit, log-σ interpolated."""
    if rho_swunit is None or rho_swunit <= 0.1:
        return {"status": "undefined", "reason": "ρ_SWunit ≤ 0.1"}
    target = 0.5 * float(rho_swunit)
    values = [(float(sigma), rho_noisy(sigma)) for sigma in sigmas]
    if any(value is None for _sigma, value in values):
        return {"status": "undefined", "reason": "a grid ρ_NOISY is not available"}
    if values[0][1] <= target:
        return {"status": "evaluated", "value": "≤ 0.01", "sigma_half": None,
                "target": target}
    for (sigma_lo, value_lo), (sigma_hi, value_hi) in zip(values, values[1:]):
        if value_hi <= target:
            fraction = (value_lo - target) / (value_lo - value_hi)
            log_sigma = math.log(sigma_lo) + fraction * (math.log(sigma_hi)
                                                         - math.log(sigma_lo))
            return {"status": "evaluated", "value": float(math.exp(log_sigma)),
                    "sigma_half": float(math.exp(log_sigma)), "target": target,
                    "bracket": [sigma_lo, sigma_hi]}
    return {"status": "evaluated", "value": "> 0.10", "sigma_half": None,
            "target": target}


def eligibility(delta_m, ci95_m) -> bool:
    """§3.2 eligibility: |Δ| ≥ 3 mm and a bootstrap CI that excludes 0."""
    return bool(delta_m is not None and abs(float(delta_m)) >= DRIFT_COST_MIN_M
                and ci95_m is not None
                and (float(ci95_m[0]) > 0.0 or float(ci95_m[1]) < 0.0))


def _frozen_mean(model, name):
    entry = ((model.get("conditions") or {}).get(name) or {}).get("vertex") or {}
    return entry.get("mean_error_m")


def _refit_mean(model, name):
    entry = ((model.get("conditions") or {}).get(name) or {}).get("refit_vertex") or {}
    return entry.get("mean_error_m")


def _x1b_class(rho):
    if rho is None:
        return None
    if rho >= RECOVERED_HIGH:
        return "charge-scale dominated"
    if rho <= RECOVERED_LOW:
        return "non-invertible dominated"
    return "mixed"


def _x1g_class(reading) -> str | None:
    if not reading.get("eligible") or reading.get("share_cs") is None:
        return None
    share = reading["share_cs"]
    if share >= RECOVERED_HIGH:
        return "residual charge-scale dominated"
    if share <= RECOVERED_LOW:
        return "residual non-invertible dominated"
    return "mixed"


def x1_readings(payload) -> dict:
    """The fixed §5 readings X1-a … X1-f, per family and seed (§3.3)."""
    families = {}
    for label, model in (payload.get("models") or {}).items():
        family, seed = _family(label)
        families.setdefault(family, {})[seed] = model

    out = {"registered": False, "extension": EXTENSION,
           "check": "X1 readings §5", "families": {}}
    for family, seeds in sorted(families.items()):
        block = {"seeds": {}, "classes": {}}
        for seed, model in sorted(seeds.items()):
            comparisons = model.get("comparisons") or {}
            drift = comparisons.get("D") or {}
            delta = _mean_difference(drift)
            delta_ci = _ci95(drift)
            row = {"seed": seed}
            row["X1-a"] = {"delta_D_m": delta, "ci95_m": delta_ci,
                           "eligible": eligibility(delta, delta_ci)}
            v_n, v_d = _frozen_mean(model, "N"), _frozen_mean(model, "D")
            v_swtw = _frozen_mean(model, "SWtw")
            rho_swtw = None
            if v_n is not None and v_d is not None and v_swtw is not None \
                    and abs(float(v_d) - float(v_n)) > 0.0:
                rho_swtw = float((float(v_d) - float(v_swtw))
                                 / (float(v_d) - float(v_n)))
            row["X1-b"] = {"rho": rho_swtw, "eligible": row["X1-a"]["eligible"],
                           "class": (_x1b_class(rho_swtw)
                                     if row["X1-a"]["eligible"] else None)}
            row["X1-g"] = _x1g(model)
            row["X1-c"] = _x1c(comparisons)
            row["X1-d"] = _x1d(comparisons)
            row["X1-e"] = _x1e(model, rho_swtw)
            row["X1-f"] = _x1f(model)
            block["seeds"][seed] = row
        block["classes"]["X1-a"] = classify_seeds(
            [(seed, row["X1-a"]["delta_D_m"]) for seed, row in block["seeds"].items()],
            lambda value: ("drift cost to recover" if value is not None
                           and abs(value) >= DRIFT_COST_MIN_M else None))
        block["classes"]["X1-b"] = classify_seeds(
            [(seed, row["X1-b"]["rho"]) for seed, row in block["seeds"].items()],
            lambda value: _x1b_class(value)
            if any(r["X1-b"]["eligible"] for r in block["seeds"].values()) else None)
        block["classes"]["X1-g"] = classify_seeds(
            [(seed, row["X1-g"].get("share_cs")
              if row["X1-g"].get("eligible") else None)
             for seed, row in block["seeds"].items()], lambda value: _x1g_class(
                {"eligible": True, "share_cs": value}))
        block["classes"]["X1-c"] = classify_seeds(
            [(seed, (row["X1-c"].get("SWunit") or {}).get("reading"))
             for seed, row in block["seeds"].items()],
            lambda value: value if value != "not available" else None)
        out["families"][family] = block
    return out


def _x1g(model) -> dict:
    comparisons = model.get("comparisons") or {}
    v_n = _frozen_mean(model, "N")
    v_d_refit = _refit_mean(model, "D")
    v_swtw_refit = _refit_mean(model, "SWtw")
    if v_n is None or v_d_refit is None or v_swtw_refit is None:
        return {"eligible": False, "reading": "not available"}
    residual = float(v_d_refit) - float(v_n)
    r_cs = float(v_d_refit) - float(v_swtw_refit)
    r_ni = float(v_swtw_refit) - float(v_n)
    comparison = comparisons.get("D_refit") or {}
    delta, ci = _mean_difference(comparison), _ci95(comparison)
    eligible = eligibility(delta, ci)
    share = (r_cs / residual) if abs(residual) > 1e-15 else None
    return {"R_m": residual, "R_cs_m": r_cs, "R_ni_m": r_ni, "share_cs": share,
            "ci95_m": ci, "eligible": eligible,
            "reading": (_x1g_class({"eligible": True, "share_cs": share})
                        if eligible else "no residual to attribute")}


def _x1c(comparisons) -> dict:
    out = {}
    for key in ("SWunit_vs_D_refit", "SWunit_refit_vs_D_refit"):
        comparison = comparisons.get(key) or {}
        difference, ci = _mean_difference(comparison), _ci95(comparison)
        short = key.split("_vs_")[0]
        entry_key = "SWunit" if short == "SWunit" else "SWunit_refit"
        if ci is None or difference is None:
            out[entry_key] = {"status": "not available"}
            continue
        if float(ci[1]) < 0.0:
            reading = "calibration better"
        elif float(ci[0]) > 0.0:
            reading = "refit better"
        else:
            reading = "not resolved"
        out[entry_key] = {"difference_m": difference, "ci95_m": ci, "reading": reading}
    return out


def _comparison_row(comparison) -> dict:
    return {"difference_m": _mean_difference(comparison),
            "ci95_m": _ci95(comparison)}


def _x1d(comparisons) -> dict:
    """X1-d ceilings, plus the §5 secondary unit-gain split (descriptive)."""
    return {"primary": {"HW": _comparison_row(comparisons.get("HW")),
                        "SWunitN": _comparison_row(comparisons.get("SWunitN")),
                        "SWunit_vs_HW": _comparison_row(
                            comparisons.get("SWunit_vs_HW"))},
            "secondary": {
                "D_refit_minus_HW_refit": _comparison_row(
                    comparisons.get("D_refit_vs_HW_refit")),
                "SWunit_refit_minus_HW_refit": _comparison_row(
                    comparisons.get("SWunit_refit_vs_HW_refit")),
                "N_minus_HW_refit": _comparison_row(
                    comparisons.get("N_vs_HW_refit"))}}


def _x1e(model, rho_swunit) -> dict:
    comparisons = model.get("comparisons") or {}
    delta = _mean_difference(comparisons.get("D"))
    v_n, v_d = _frozen_mean(model, "N"), _frozen_mean(model, "D")
    rho = {}
    for sigma in NOISY_SIGMAS:
        name = f"NOISY({sigma:g})"
        v_c = _frozen_mean(model, name)
        rho[sigma] = (None if not (v_n is not None and v_d is not None
                                   and v_c is not None and delta) else
                      float((float(v_d) - float(v_c)) / float(delta)))
    return {"rho": {f"{sigma:g}": value for sigma, value in rho.items()},
            "sigma_half": sigma_half(lambda sigma: rho[sigma], rho_swunit)}


def _x1f(model) -> dict:
    out = {}
    for name in ENERGY_CONDITIONS:
        energy = ((model.get("particle") or {}).get(name) or {}).get("energy") or {}
        out[name] = {"resolution": energy.get("resolution"),
                     "r2": (energy.get("r2") or {}).get("r2"),
                     "n_contained_test": energy.get("n_contained_test")}
    return out


# ================================================================== X2 §6
def run_log_gain_vectors(dataset, geometry, run_ids=None) -> dict:
    """δ_r = ln m¹_r − ln m⁰_r (105 values per run; §6 X2-b, Appendix A)."""
    out = {}
    for row in dataset.runs:
        run_id = str(row["run_id"])
        if run_ids is not None and run_id not in run_ids:
            continue
        s0, s1 = state_pair(row, dataset, geometry)
        out[run_id] = (np.log(np.asarray(s1.module_gain, float))
                       - np.log(np.asarray(s0.module_gain, float)))
    return out


def occupancy_counts(dataset, condition: Condition, geometry) -> dict:
    """n_e,m: number of hit PMTs of module m per event, from the condition's store (§6)."""
    mod = module_of_pmt(geometry)
    n_modules = int(geometry.n_modules)
    out = {}
    with hit_transform_context(condition):
        for row in dataset.runs:
            run_id = str(row["run_id"])
            store = lb.read_sparse_hits(Path(dataset.directory) / "runs"
                                        / row["run_dir"] / "hits.npz")
            for position, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
                span = slice(int(store.offsets[position]), int(store.offsets[position + 1]))
                pmt = np.asarray(store.pmt_id[span], np.int64)
                counts = np.bincount(mod[pmt], minlength=n_modules).astype(float)
                out[(run_id, int(event_id))] = counts
    return out


def occupancy_design(rows, delta_by_run, counts, geometry) -> np.ndarray:
    """u_e,m = δ_r,m · n_e,m for each event row of one condition (§6 X2-b)."""
    n_modules = int(geometry.n_modules)
    design = np.zeros((len(rows["event_ids"]), n_modules))
    run_ids = np.asarray(rows["run_ids"]).astype(str)
    event_ids = np.asarray(rows["event_ids"], np.int64)
    for index, (run_id, event_id) in enumerate(zip(run_ids, event_ids)):
        key = (str(run_id), int(event_id))
        if key not in counts:
            raise KeyError(f"occupancy design: no hit count for {key}")
        design[index] = delta_by_run[str(run_id)] * counts[key]
    return design


def occupancy_count_design(rows, counts, geometry) -> np.ndarray:
    """n_e,m for each event row (the occupancy design without the gain factor)."""
    n_modules = int(geometry.n_modules)
    design = np.zeros((len(rows["event_ids"]), n_modules))
    for index, key in enumerate(zip(np.asarray(rows["run_ids"]).astype(str),
                                    np.asarray(rows["event_ids"], np.int64))):
        design[index] = counts[(str(key[0]), int(key[1]))]
    return design


def global_design(rows, delta_by_run, geometry) -> np.ndarray:
    """u_e = δ_r for each event row (105 columns; §6 X2-b global model)."""
    design = np.zeros((len(rows["event_ids"]), int(geometry.n_modules)))
    for index, run_id in enumerate(np.asarray(rows["run_ids"]).astype(str)):
        design[index] = delta_by_run[str(run_id)]
    return np.asarray(design, float)


def pair_embedding_rows(reference, other):
    """Align ``other`` rows to ``reference`` on (run_id, event_id); truth must agree."""
    paired = ef.pair_events(reference, other)
    aligned = {name: np.asarray(other[name])[paired["index_right"]]
               for name in ("events", "run_ids", "event_ids", "vertex", "log_yield",
                            "n_hit") if name in other}
    return paired, aligned


def _r2_eq(delta_test, prediction, train_mean) -> dict:
    delta_test = np.asarray(delta_test, float)
    prediction = np.asarray(prediction, float)
    denominator = float(((delta_test - np.asarray(train_mean, float)) ** 2).sum())
    numerator = float(((delta_test - prediction) ** 2).sum())
    return {"r2_eq": (None if denominator <= 0 else 1.0 - numerator / denominator),
            "sse": numerator, "sst": denominator,
            "n_events": int(delta_test.shape[0]),
            "n_dimensions": int(delta_test.shape[1])}


def equivariance_fit(u_train, delta_train, u_test, delta_test, *, run_ids_train,
                     run_ids_test, kind="occupancy") -> dict:
    """Fit Δz ≈ Â u with one run-grouped ridge per embedding dimension (§6 X2-b).

    The fit sees only the probe-train rows; ``run_ids_train`` and
    ``run_ids_test`` are run-disjoint by construction, and the returned
    ``fit_run_ids`` records exactly which runs entered the fit (or its CV).
    """
    u_train = np.asarray(u_train, float)
    delta_train = np.asarray(delta_train, float)
    u_test = np.asarray(u_test, float)
    delta_test = np.asarray(delta_test, float)
    if u_train.shape[0] != delta_train.shape[0] or u_test.shape[0] != delta_test.shape[0]:
        raise ValueError("equivariance_fit: design and target rows disagree")
    if set(np.asarray(run_ids_train).astype(str)) & set(
            np.asarray(run_ids_test).astype(str)):
        raise ValueError("equivariance_fit: probe-test runs must be disjoint")
    train = la.ProbeDataset(kind="delta_z", form="occupancy", x=u_train, y=delta_train,
                            run_ids=np.asarray(run_ids_train, dtype=object),
                            target_names=tuple(f"z{i}" for i in range(delta_train.shape[1])),
                            meta={"kind": kind})
    fitted = la.fit_probe(train)
    test = la.ProbeDataset(kind="delta_z", form="occupancy", x=u_test, y=delta_test,
                           run_ids=np.asarray(run_ids_test, dtype=object),
                           target_names=train.target_names, meta={"kind": kind})
    prediction = la.probe_predictions(fitted, test)
    train_mean = delta_train.mean(axis=0)
    return {"fitted": fitted, "prediction": prediction, "train_mean": train_mean,
            "fit_run_ids": sorted(set(np.asarray(run_ids_train).astype(str))),
            "n_train": int(u_train.shape[0]), "n_test": int(u_test.shape[0]),
            "kind": kind, **_r2_eq(delta_test, prediction, train_mean)}


def equivariance_class(r2_eq) -> str | None:
    if r2_eq is None:
        return None
    if r2_eq >= 0.8:
        return "equivariant"
    if r2_eq >= 0.4:
        return "partly equivariant"
    return "non-equivariant"


def latent_repair_class(rho) -> str | None:
    if rho is None:
        return None
    if rho >= 0.5:
        return "latent repair works"
    if rho <= 0.2:
        return "fails"
    return "partial"


def x2_decomposition(n_test, d_test, s_test, *, frozen_fit, swtw_fit,
                     swtw_test=None) -> dict:
    """X2-a: variance shares and frozen-probe costs of the two drift parts (§6)."""
    _paired_s, aligned_s = pair_embedding_rows(n_test, s_test)
    _paired_d, aligned_d = pair_embedding_rows(n_test, d_test)
    dz_cs = (np.asarray(aligned_s["events"], float)
             - np.asarray(n_test["events"], float))
    dz_ni = (np.asarray(aligned_d["events"], float)
             - np.asarray(aligned_s["events"], float))
    dz_tot = (np.asarray(aligned_d["events"], float)
              - np.asarray(n_test["events"], float))
    denom = float(np.mean(np.sum(dz_tot ** 2, axis=1)))
    out = {"share_cs": (None if denom <= 0 else float(np.mean(np.sum(dz_cs ** 2, axis=1)))
                        / denom),
           "share_ni": (None if denom <= 0 else float(np.mean(np.sum(dz_ni ** 2, axis=1)))
                        / denom),
           "n_events": int(dz_tot.shape[0])}
    v_n = vertex_row(n_test, fitted=frozen_fit)["mean_error_m"]
    v_s = vertex_row(s_test, fitted=frozen_fit)["mean_error_m"]
    v_d = vertex_row(d_test, fitted=frozen_fit)["mean_error_m"]
    v_swtw = (None if swtw_test is None
              else vertex_row(swtw_test, fitted=swtw_fit)["mean_error_m"])
    out.update({"v_S_minus_v_N": None if None in (v_s, v_n) else float(v_s) - float(v_n),
                "v_D_minus_v_S": None if None in (v_d, v_s) else float(v_d) - float(v_s),
                "additivity_check": None if None in (v_d, v_n, v_s, v_swtw) else
                float(v_d) - float(v_n) - ((float(v_s) - float(v_n))
                                           + (float(v_swtw) - float(v_n)))})
    return out


def whitening_matrix(events, run_ids, *, eps_scale=1e-6) -> dict:
    """W = (C + εI)^(−1/2), C the pooled within-run covariance (§6, Appendix C)."""
    events = np.asarray(events, float)
    run_ids = np.asarray(run_ids).astype(str)
    d = int(events.shape[1])
    centred = np.zeros_like(events)
    for run in np.unique(run_ids):
        selected = events[run_ids == run]
        centred[run_ids == run] = selected - selected.mean(axis=0, keepdims=True)
    covariance = centred.T @ centred / max(len(events), 1)
    epsilon = float(eps_scale) * float(np.trace(covariance)) / float(d)
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    inverse_sqrt = (vectors * (1.0 / np.sqrt(values + epsilon))) @ vectors.T
    return {"W": inverse_sqrt, "W_inv": (vectors * np.sqrt(values + epsilon)) @ vectors.T,
            "covariance": covariance, "epsilon": epsilon}


def latent_estimate(zbar_delta, occupancy, A_hat, W, *, lam=0.0, centre=None,
                    scale=None) -> dict:
    """Generalised ridge for one run's module gains from the latent stream (§6 X2-d).

    The design of run r is ``W Â diag(ū_r)`` and the target is ``W(z̄ᴰ_r − z̄ᴺ)``;
    the solve estimates that run's ``δ_r``.  The columns are standardised on the
    training folds when ``centre``/``scale`` are given (§6, Appendix C).
    """
    zbar_delta = np.asarray(zbar_delta, float).ravel()
    occupancy = np.asarray(occupancy, float).ravel()
    A_hat = np.asarray(A_hat, float)
    W = np.asarray(W, float)
    design = (W @ A_hat) * occupancy[None, :]
    if centre is not None and scale is not None:
        design = (design - np.asarray(centre, float)) / np.asarray(scale, float)
    target = W @ zbar_delta
    gram = design.T @ design
    rhs = design.T @ target
    if float(lam) == 0.0:
        delta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    else:
        delta = np.linalg.solve(gram + float(lam) * np.eye(gram.shape[0]), rhs)
    residual = target - design @ delta
    return {"delta": delta, "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
            "lam": float(lam), "design": design, "target": target}


def _standardisation(designs, *, eps=0.0) -> tuple:
    """Column centre/scale pooled from the training folds' designs (§6, Appendix C)."""
    designs = [np.asarray(design, float) for design in designs if np.asarray(design).size]
    if not designs:
        return None, None
    flat = np.concatenate(designs, axis=0)
    centre = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale <= float(eps), 1.0, scale)
    return centre, scale


def latent_cv(*, features_n, features_d, run_ids, occupancy_by_run, truth,
              seed=0, folds=5) -> dict:
    """λ by five-fold run-grouped CV on probe-train, refit inside every fold (§6).

    Simulation truth of the probe-train runs selects λ; no probe-test truth or
    probe-test run enters this routine.  Each validation run is estimated on its
    own (one δ per run), with W, Â, z̄ᴺ and the standardisation refit per fold.
    """
    features_n = np.asarray(features_n, float)
    features_d = np.asarray(features_d, float)
    run_ids = np.asarray(run_ids).astype(str)
    occurrence = occupancy_by_run["events"]
    runs = list(occupancy_by_run["runs"])
    run_index = {run: index for index, run in enumerate(runs)}
    fold_of = _run_grouped_folds(np.asarray(runs), k=int(folds), seed=int(seed))
    scores = {float(alpha): [] for alpha in pr.ALPHAS}
    for fold in range(int(folds)):
        train_runs = [runs[index] for index in range(len(runs))
                      if fold_of[index] != fold]
        validation_runs = [runs[index] for index in range(len(runs))
                           if fold_of[index] == fold]
        if len(train_runs) < 5 or not validation_runs:
            # five run-grouped folds need at least five runs (§6/Appendix C)
            continue
        train_mask = np.isin(run_ids, train_runs)
        whitening = whitening_matrix(features_n[train_mask], run_ids[train_mask])
        A_hat = _fit_equivariance_occupancy(
            features_n[train_mask], features_d[train_mask], occurrence[train_mask],
            run_ids[train_mask])
        zbar_n = features_n[train_mask].mean(axis=0)
        train_designs = [(whitening["W"] @ A_hat) * occupancy_by_run["occupancy"][
            run_index[run]][None, :] for run in train_runs]
        centre, scale = _standardisation(train_designs)
        for run in validation_runs:
            index = run_index[run]
            events = run_ids == run
            zbar = features_d[events].mean(axis=0) - zbar_n
            for alpha in pr.ALPHAS:
                fitted = latent_estimate(
                    zbar, occupancy_by_run["occupancy"][index], A_hat,
                    whitening["W"], lam=float(alpha), centre=centre, scale=scale)
                error = np.asarray(fitted["delta"], float) - np.asarray(truth[run], float)
                scores[float(alpha)].append(float(np.mean(error ** 2)))
    best = None
    for alpha, values in scores.items():
        if values:
            mean = float(np.mean(values))
            if best is None or mean < best[1]:
                best = (alpha, mean)
    return {"lambda_grid": list(pr.ALPHAS),
            "cv_mse": {f"{alpha:g}": (float(np.mean(values)) if values else None)
                       for alpha, values in scores.items()},
            "best_lambda": None if best is None else float(best[0]),
            "n_folds": int(folds), "seed": int(seed)}


def latent_stream_estimates(*, features_n_train, features_d_train, run_ids_train,
                            occupancy_events_train, occupancy_mean_train,
                            features_n_reference, features_d_test, run_ids_test,
                            occupancy_events_test, occupancy_mean_test,
                            truth_train, seed=0, folds=5) -> dict:
    """X2-d latent stream: choose λ on probe-train, then estimate each test run."""
    training = {"events": np.asarray(occupancy_events_train, float),
                "runs": [str(run) for run in
                         np.unique(np.asarray(run_ids_train).astype(str))],
                "occupancy": np.asarray(occupancy_mean_train, float)}
    cv = latent_cv(features_n=np.asarray(features_n_train, float),
                   features_d=np.asarray(features_d_train, float),
                   run_ids=np.asarray(run_ids_train).astype(str),
                   occupancy_by_run=training, truth=truth_train, seed=seed,
                   folds=folds)
    whitening = whitening_matrix(np.asarray(features_n_train, float),
                                 np.asarray(run_ids_train).astype(str))
    A_hat = _fit_equivariance_occupancy(
        np.asarray(features_n_train, float), np.asarray(features_d_train, float),
        np.asarray(occupancy_events_train, float),
        np.asarray(run_ids_train).astype(str))
    zbar_n = np.asarray(features_n_reference, float).mean(axis=0)
    train_runs = [str(run) for run in
                  np.unique(np.asarray(run_ids_train).astype(str))]
    designs = [(whitening["W"] @ A_hat) * np.asarray(occupancy_mean_train, float)[index][
        None, :] for index in range(len(train_runs))]
    centre, scale = _standardisation(designs)
    run_ids_test = np.asarray(run_ids_test).astype(str)
    estimates = {}
    for index, run in enumerate([str(value) for value in np.unique(run_ids_test)]):
        events = run_ids_test == run
        zbar = np.asarray(features_d_test, float)[events].mean(axis=0) - zbar_n
        fitted = latent_estimate(zbar, np.asarray(occupancy_mean_test, float)[index],
                                 A_hat, whitening["W"],
                                 lam=float(cv["best_lambda"] or 0.0),
                                 centre=centre, scale=scale)
        estimates[run] = np.asarray(fitted["delta"], float)
    return {"estimates": estimates, "cv": cv, "A_hat": A_hat,
            "W": whitening["W"], "zbar_n": zbar_n, "centre": centre,
            "scale": scale, "stream": "latent"}


def _run_grouped_folds(run_ids, k=5, seed=0) -> np.ndarray:
    runs = np.unique(np.asarray(run_ids).astype(str))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(runs))
    fold_of = {run: index % int(k) for index, run in enumerate(runs[order])}
    return np.asarray([fold_of[str(run)] for run in run_ids], np.int64)


def _fit_equivariance_occupancy(features_n, features_d, occupancy, run_ids) -> np.ndarray:
    """Â from probe-train only: Δz_cs ≈ Â u with u the occupancy design (§6)."""
    occupancy = np.asarray(occupancy, float)
    _, aligned = pair_embedding_rows(
        {"run_ids": np.asarray(run_ids, dtype=object),
         "event_ids": np.arange(len(run_ids)), "vertex": np.zeros((len(run_ids), 3))},
        {"run_ids": np.asarray(run_ids, dtype=object),
         "event_ids": np.arange(len(run_ids)), "vertex": np.zeros((len(run_ids), 3)),
         "events": np.asarray(features_d, float)})
    delta = np.asarray(features_d, float) - np.asarray(features_n, float)
    train = la.ProbeDataset(kind="delta_z", form="occupancy", x=occupancy, y=delta,
                            run_ids=np.asarray(run_ids, dtype=object),
                            target_names=tuple(f"z{i}" for i in range(delta.shape[1])),
                            meta={})
    fitted = la.fit_probe(train)
    coefficients = np.stack([probe["fit"].coefficients if probe["status"] == "fit"
                             else np.zeros(occupancy.shape[1])
                             for probe in fitted["fits"]], axis=1)
    return coefficients.T


def _pearson(x, y) -> float | None:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return None
    x, y = x[keep] - x[keep].mean(), y[keep] - y[keep].mean()
    denominator = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return None if denominator <= 0 else float((x @ y) / denominator)


def source_estimate(dataset, geometry, gen, *, dataset_seed, runs=None) -> dict:
    """X2-d source stream: δ̂ = ln(Σ q / Σ μ) per run and module (§6, Appendix C)."""
    dataset_seed = int(dataset_seed)
    fixed = lb.fixed_charge_gain(geometry)
    state = lb.nominal_state(geometry)
    mod = module_of_pmt(geometry)
    n_modules = int(geometry.n_modules)
    out = {"estimates": {}, "coverage": {}, "stream": "source"}
    for row in dataset.runs:
        run_id = str(row["run_id"])
        if runs is not None and run_id not in runs:
            continue
        run_dir = dataset.directory / "runs" / row["run_dir"]
        store = lb.read_sparse_hits(run_dir / "hits.npz")
        with np.load(run_dir / "events.npz", allow_pickle=False) as data:
            event_ids = np.asarray(data["event_ids"], np.int64)
            vertex = np.asarray(data["vertex_m"], float)
            yields = np.asarray(data["yield_photons"], float)
            domains = (np.asarray(data["domain"]).astype(str)
                       if "domain" in data.files else np.full(event_ids.size, "photon"))
        position = {int(event): place for place, event in enumerate(event_ids)}
        q_sum = np.zeros(n_modules)
        mu_sum = np.zeros(n_modules)
        for store_position, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
            place = position.get(int(event_id))
            if place is None or str(domains[place]) != lb.DOMAIN_PHOTON:
                continue
            qe = lb.sample_qe(geometry, lb.keyed_rng(
                dataset_seed, lb.NS_DRAWS, run_id, int(event_id), "wavelength"), gen)
            mu = (lb.expected_charge(geometry, vertex[place], yields[place], qe,
                                     state.attenuation_length_m, gen) * fixed)
            span = slice(int(store.offsets[store_position]),
                         int(store.offsets[store_position + 1]))
            pmt = np.asarray(store.pmt_id[span], np.int64)
            charge = np.asarray(store.charge_pe[span], float)
            keep = (charge > 0) & (mu[pmt] >= 3.0)
            if not keep.any():
                continue
            q_sum += np.bincount(mod[pmt[keep]], weights=charge[keep],
                                 minlength=n_modules)
            mu_sum += np.bincount(mod[pmt[keep]], weights=mu[pmt[keep]],
                                  minlength=n_modules)
        estimate = np.full(n_modules, np.nan)
        positive = mu_sum > 0
        estimate[positive] = np.log(q_sum[positive] / mu_sum[positive])
        out["estimates"][run_id] = estimate
        out["coverage"][run_id] = float(positive.mean())
    return out


def raw_estimate(dataset, reference_dataset, geometry, *, runs=None) -> dict:
    """X2-d raw stream: run-mean module hit-mean log charge minus the N mean (§6)."""
    mod = module_of_pmt(geometry)
    n_modules = int(geometry.n_modules)

    def event_module_means(source):
        store = lb.read_sparse_hits(source)
        rows = []
        for position in range(int(np.asarray(store.event_ids).size)):
            span = slice(int(store.offsets[position]), int(store.offsets[position + 1]))
            pmt = np.asarray(store.pmt_id[span], np.int64)
            charge = np.asarray(store.charge_pe[span], float)
            if pmt.size == 0:
                continue
            by_module = np.full(n_modules, np.nan)
            sums = np.bincount(mod[pmt], weights=np.log(np.maximum(charge, 1e-30)),
                               minlength=n_modules)
            counts = np.bincount(mod[pmt], minlength=n_modules)
            positive = counts > 0
            by_module[positive] = sums[positive] / counts[positive]
            rows.append(by_module)
        return np.asarray(rows) if rows else np.zeros((0, n_modules))

    reference = []
    for row in reference_dataset.runs:
        reference.append(event_module_means(Path(reference_dataset.directory) / "runs"
                                            / row["run_dir"] / "hits.npz"))
    reference_mean = np.nanmean(np.concatenate(reference), axis=0) if reference else \
        np.full(n_modules, np.nan)
    out = {"estimates": {}, "coverage": {}, "stream": "raw",
           "reference_mean": reference_mean}
    for row in dataset.runs:
        run_id = str(row["run_id"])
        if runs is not None and run_id not in runs:
            continue
        per_event = event_module_means(Path(dataset.directory) / "runs"
                                       / row["run_dir"] / "hits.npz")
        mean = np.nanmean(per_event, axis=0) if len(per_event) else \
            np.full(n_modules, np.nan)
        out["estimates"][run_id] = mean - reference_mean
        out["coverage"][run_id] = float(np.isfinite(out["estimates"][run_id]).mean())
    return out


def identification_report(estimates: dict, truth: dict) -> dict:
    """Per-run RMS error, Pearson r over (run, module) and coverage (§6 X2-d)."""
    common = sorted(set(estimates) & set(truth))
    if not common:
        return {"status": "not available", "n_runs": 0}
    errors = np.stack([np.asarray(estimates[run], float) - np.asarray(truth[run], float)
                       for run in common])
    keep = np.isfinite(errors)
    per_run_rms = np.asarray([float(np.sqrt(np.mean(errors[index][keep[index]] ** 2)))
                              if keep[index].any() else np.nan
                              for index in range(len(common))])
    predicted = np.concatenate([np.asarray(estimates[run], float) for run in common])
    observed = np.concatenate([np.asarray(truth[run], float) for run in common])
    return {"status": "evaluated", "n_runs": int(len(common)),
            "median_run_rms": (float(np.nanmedian(per_run_rms))
                               if np.isfinite(per_run_rms).any() else None),
            "pearson_r": _pearson(predicted, observed),
            "coverage": float(np.mean([np.isfinite(np.asarray(estimates[run], float)
                                                   ).mean() for run in common])),
            "per_run_rms": {run: (None if not np.isfinite(per_run_rms[index])
                                  else float(per_run_rms[index]))
                            for index, run in enumerate(common)}}


def _stack_coefficients(fit) -> np.ndarray:
    """The fitted map Â (``n_dim x n_modules``) of one ``equivariance_fit`` result."""
    fits = (fit or {}).get("fitted", {}).get("fits") or []
    if not fits:
        return None
    width = int(np.asarray(fits[0]["fit"].scale).size) if fits[0].get("status") == "fit" \
        else int((fit.get("train_mean") or np.zeros(0)).size)
    columns = []
    for probe in fits:
        if probe.get("status") == "fit":
            columns.append(np.asarray(probe["fit"].coefficients, float))
        else:
            columns.append(np.zeros(width))
    return np.stack(columns, axis=1).T


def _x2c(rows, delta, occupancy, geometry, frozen, A_hat) -> dict:
    """X2-c: z̃ = z_D − Â u(δ_r) with the occupancy model, then P_N (§6)."""
    if A_hat is None:
        return {"status": "not available", "rho_LR": None}
    test_u = occupancy_design(rows["test"]["D"], delta, occupancy, geometry)
    repaired = dict(rows["test"]["D"])
    repaired["events"] = (np.asarray(rows["test"]["D"]["events"], float)
                          - np.asarray(test_u, float) @ np.asarray(A_hat, float).T)
    prediction = la.probe_predictions(
        frozen, ef._probe_dataset(repaired, "vertex", target_names=("x", "y", "z")))
    repaired_errors = ef._vertex_errors(repaired, prediction)
    v_n = vertex_row(rows["test"]["N"], fitted=frozen)["mean_error_m"]
    v_d = vertex_row(rows["test"]["D"], fitted=frozen)["mean_error_m"]
    v_lr = float(repaired_errors.mean())
    rho = None if v_d is None or v_n is None or abs(float(v_d) - float(v_n)) <= 0 \
        else float((float(v_d) - v_lr) / (float(v_d) - float(v_n)))
    return {"v_LR_m": v_lr, "v_D_m": v_d, "v_N_m": v_n, "rho_LR": rho,
            "errors": repaired_errors, "class": latent_repair_class(rho),
            "A_hat": np.asarray(A_hat, float).tolist()}


def identifiability_spectrum(A_hat, W, occupancy_mean, *, n_events_per_run=17) -> dict:
    """X2-f: singular values of W Â at mean occupancy and the implied per-module CRLB."""
    A_hat = np.asarray(A_hat, float)
    W = np.asarray(W, float)
    occupancy_mean = np.asarray(occupancy_mean, float)
    design = W @ A_hat @ np.diag(occupancy_mean)
    singular = np.linalg.svd(design, compute_uv=False)
    gram = design.T @ design
    information = gram * float(n_events_per_run)
    condition = np.linalg.cond(information) if information.size else None
    try:
        crlb = np.diag(np.linalg.pinv(information, rcond=1e-12))
    except np.linalg.LinAlgError:  # pragma: no cover
        crlb = np.full(information.shape[0], np.inf)
    return {"singular_values": singular.tolist(),
            "max_singular_value": float(singular[0]) if singular.size else None,
            "condition_number": None if condition is None else float(condition),
            "n_events_per_run": int(n_events_per_run),
            "crlb_per_module": crlb.tolist(),
            "median_crlb": float(np.median(crlb)) if crlb.size else None}


def x2(root=None, *, config, gen, arms_config, geometry, inventory=None,
       data_root=DATA_ROOT_DEFAULT, device="cpu", ext_root=EXT_ROOT, labels=None,
       include_optional=False, limit=None, log=print) -> dict:
    """X2: equivariance, latent repair and identification (§6)."""
    root = _resolve_root(root)
    gate = assert_authorised("x2", root=root, log=log)
    inv = inventory if inventory is not None else part_a_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    names = ("N", "D", "S")
    conditions = {role: build_conditions(
        {"s005": probes005[role], "s030": probes030[role]}, geometry, names)
        for role in ("train", "test")}
    delta = run_log_gain_vectors(probes030["train"], geometry)
    delta.update(run_log_gain_vectors(probes030["test"], geometry))
    # §6 X2-b uses the N hits for n_e,m; X2-c and the latent stream use the D hits
    occupancy = {"n": {role: occupancy_counts(probes005[role], conditions[role]["N"],
                                              geometry)
                       for role in ("train", "test")},
                 "d": {role: occupancy_counts(probes030[role], conditions[role]["D"],
                                              geometry)
                       for role in ("train", "test")}}
    payload = {"registered": False, "extension": EXTENSION, "check": "X2 §6",
               "runbook": RUNBOOK_REL, "models": {}, "unavailable": [],
               "gate": {"status": gate["status"]}, "wall_seconds": None}
    selected = (list(CORE_MODELS) if labels is None else list(labels))
    for label in selected:
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            payload["unavailable"].append({"label": label, "reason": "inventory unavailable"})
            continue
        cache = embedding_cache(ext_root, label)
        try:
            with ef.item_eval_context(row):
                arm = ef.build_item_arm(row, config=config, gen=gen,
                                        arms_config=arms_config, geometry=geometry,
                                        data_root=data_root, device=device, log=log)
            model = getattr(arm, "jepa", arm)
            split = row_split(row)
            rows = {role: {} for role in ("train", "test")}
            for role in ("train", "test"):
                for name in names:
                    base = _condition_base(name)[0]
                    dataset = (probes005[role] if _SOURCE[base] == S005
                               else probes030[role])
                    rows[role][name] = stream_condition_embeddings(
                        model, dataset, geometry, conditions[role][name],
                        split=split, config=config, device=device, cache=cache,
                        cache_key=embedding_cache_key(role, name),
                        max_events=limit, log=log)
            payload["models"][label] = _x2_model(
                row, model, rows, delta, occupancy, conditions, probes005, probes030,
                geometry, config=config, gen=gen, data_root=data_root,
                ext_root=ext_root, device=device, cache=cache, split=split, log=log)
        except SystemExit:
            raise
        except Exception as problem:  # pragma: no cover
            payload["unavailable"].append({
                "label": label, "reason": f"{type(problem).__name__}: {problem}"})
            continue
    payload["readings"] = x2_readings(payload)
    payload = _strip_internal(payload)
    payload["complete"] = not payload["unavailable"]
    payload["closure"] = _assert_closure("x2", root)
    return payload


def _run_mean_vectors(run_ids, per_event):
    """Per-run mean of a per-event array, aligned with the sorted unique run ids."""
    run_ids = np.asarray(run_ids).astype(str)
    per_event = np.asarray(per_event, float)
    runs = [str(run) for run in np.unique(run_ids)]
    return runs, np.stack([per_event[run_ids == run].mean(axis=0) for run in runs])


def _x2_model(row, model, rows, delta, occupancy, conditions, probes005, probes030,
              geometry, *, config, gen, data_root, ext_root, device, cache,
              split=None, log=print) -> dict:
    """One model's X2-a … X2-f block."""
    label = str(row["label"])
    family, seed = _family(label)
    split = row_split(row) if split is None else str(split)
    out = {"label": label, "family": family, "seed": seed, "split": split,
           "X2-a": None, "X2-b": {}, "X2-c": None, "X2-d": {}, "X2-e": {}, "X2-f": None}
    frozen = fit_vertex_probe(rows["train"]["N"])
    # X2-a decomposition (the additivity check uses the SWtw rows, not the S rows)
    swtw_key = embedding_cache_key("test", "SWtw")
    swtw_test = (cache.fetch(swtw_key, lambda: {})
                 if cache is not None and cache.has(swtw_key) else None)
    out["X2-a"] = x2_decomposition(rows["test"]["N"], rows["test"]["D"],
                                   rows["test"]["S"], frozen_fit=frozen,
                                   swtw_fit=frozen, swtw_test=swtw_test)
    delta_train = {run: delta[run] for run in set(
        np.asarray(rows["train"]["D"]["run_ids"]).astype(str)) if run in delta}
    delta_test = {run: delta[run] for run in set(
        np.asarray(rows["test"]["D"]["run_ids"]).astype(str)) if run in delta}
    # X2-b equivariance on the occupancy and global designs (fit train, score test)
    for kind in ("occupancy", "global"):
        if kind == "occupancy":
            u_train = occupancy_design(rows["train"]["S"], delta_train,
                                       occupancy["n"]["train"], geometry)
            u_test = occupancy_design(rows["test"]["S"], delta_test,
                                      occupancy["n"]["test"], geometry)
        else:
            u_train = global_design(rows["train"]["S"], delta_train, geometry)
            u_test = global_design(rows["test"]["S"], delta_test, geometry)
        aligned_n_train = np.asarray(rows["train"]["N"]["events"], float)
        aligned_s_train = np.asarray(pair_embedding_rows(
            rows["train"]["N"], rows["train"]["S"])[1]["events"], float)
        aligned_d_train = np.asarray(pair_embedding_rows(
            rows["train"]["N"], rows["train"]["D"])[1]["events"], float)
        aligned_n_test = np.asarray(rows["test"]["N"]["events"], float)
        aligned_s_test = np.asarray(pair_embedding_rows(
            rows["test"]["N"], rows["test"]["S"])[1]["events"], float)
        aligned_d_test = np.asarray(pair_embedding_rows(
            rows["test"]["N"], rows["test"]["D"])[1]["events"], float)
        fit_cs = equivariance_fit(
            u_train, aligned_s_train - aligned_n_train, u_test,
            aligned_s_test - aligned_n_test,
            run_ids_train=np.asarray(rows["train"]["N"]["run_ids"]),
            run_ids_test=np.asarray(rows["test"]["N"]["run_ids"]), kind=kind)
        fit_tot = equivariance_fit(
            u_train, aligned_d_train - aligned_n_train, u_test,
            aligned_d_test - aligned_n_test,
            run_ids_train=np.asarray(rows["train"]["N"]["run_ids"]),
            run_ids_test=np.asarray(rows["test"]["N"]["run_ids"]), kind=kind)
        entry = {"r2_cs": fit_cs["r2_eq"], "r2_tot": fit_tot["r2_eq"],
                 "n_train": fit_cs["n_train"], "n_test": fit_cs["n_test"],
                 "A_hat": None, "fit_run_ids": fit_cs["fit_run_ids"]}
        if kind == "occupancy":
            entry["class"] = equivariance_class(fit_cs["r2_eq"])
            entry["A_hat"] = _stack_coefficients(fit_cs)
        out["X2-b"][kind] = entry
    A_hat = out["X2-b"]["occupancy"]["A_hat"]
    # X2-c latent repair with the true gains (n_e,m from the D hits)
    out["X2-c"] = _x2c(rows, delta, occupancy["d"]["test"], geometry, frozen, A_hat)
    # X2-d identification: truth and the three streams
    truth_test, truth_train = _gain_truth(probes030, geometry)
    dataset_seed = int(config["data"]["dataset_seed"])
    source = source_estimate(probes030["test"], geometry, gen,
                             dataset_seed=dataset_seed)
    raw = raw_estimate(probes030["test"], probes005["train"], geometry)
    out["X2-d"]["source"] = identification_report(source["estimates"], truth_test)
    out["X2-d"]["raw"] = identification_report(raw["estimates"], truth_test)
    out["X2-d"]["truth"] = {run: value.tolist() for run, value in truth_test.items()}
    occ_events_train = occupancy_design(rows["train"]["N"], delta_train,
                                        occupancy["d"]["train"], geometry)
    _, occ_mean_train = _run_mean_vectors(rows["train"]["N"]["run_ids"],
                                          occ_events_train)
    occ_events_test = occupancy_design(rows["test"]["N"], delta_test,
                                       occupancy["d"]["test"], geometry)
    _, occ_mean_test = _run_mean_vectors(rows["test"]["N"]["run_ids"],
                                         occ_events_test)
    latent = latent_stream_estimates(
        features_n_train=rows["train"]["N"]["events"],
        features_d_train=aligned_d_train,
        run_ids_train=np.asarray(rows["train"]["N"]["run_ids"]),
        occupancy_events_train=occ_events_train,
        occupancy_mean_train=occ_mean_train,
        features_n_reference=rows["train"]["N"]["events"],
        features_d_test=aligned_d_test,
        run_ids_test=np.asarray(rows["test"]["N"]["run_ids"]),
        occupancy_events_test=occ_events_test,
        occupancy_mean_test=occ_mean_test, truth_train=truth_train)
    out["X2-d"]["latent"] = identification_report(latent["estimates"], truth_test)
    out["X2-d"]["latent_cv"] = latent["cv"]
    # X2-e closing the loop: EST(k) conditions and latent repair with δ̂_latent
    out["X2-e"] = _x2e(model=model, rows=rows, probes030=probes030, geometry=geometry,
                       config=config, gen=gen, device=device, cache=cache,
                       frozen=frozen, delta=delta, occupancy=occupancy["d"],
                       split=split,
                       estimates={"source": source["estimates"],
                                  "raw": raw["estimates"],
                                  "latent": latent["estimates"]},
                       A_hat=A_hat, log=log)
    # X2-f identifiability spectrum at the mean occupancy
    if A_hat is not None:
        whitening = whitening_matrix(np.asarray(rows["train"]["N"]["events"], float),
                                     np.asarray(rows["train"]["N"]["run_ids"]))
        occ_n_train = occupancy_design(rows["train"]["N"], delta_train,
                                       occupancy["n"]["train"], geometry)
        out["X2-f"] = identifiability_spectrum(
            A_hat, whitening["W"], occ_n_train.mean(axis=0))
    return attach_x2_canonical(out)


def _gain_truth(probes030, geometry):
    """Per-run true module log-gain change on the D probe-train and probe-test runs."""
    truth = {}
    for role in ("train", "test"):
        for row in probes030[role].runs:
            s1 = run_state(row, probes030[role], geometry, S030)
            s0 = run_state(row, probes030[role], geometry, S005)
            truth[str(row["run_id"])] = (np.log(np.asarray(s1.module_gain, float))
                                         - np.log(np.asarray(s0.module_gain, float)))
    train = {run: value for run, value in truth.items()
             if run in {str(row["run_id"]) for row in probes030["train"].runs}}
    test = {run: value for run, value in truth.items()
            if run in {str(row["run_id"]) for row in probes030["test"].runs}}
    return test, train


def _x2e(*, model, rows, probes030, geometry, config, gen, device, cache, frozen,
         delta, occupancy, estimates, A_hat, split="within", log=print) -> dict:
    """X2-e: ρ_EST(k) from each stream next to the ceiling ρ_SWmod (§6)."""
    out = {"streams": {}, "ceiling_rho_SWmod": None}
    v_n = vertex_row(rows["test"]["N"], fitted=frozen)["mean_error_m"]
    v_d = vertex_row(rows["test"]["D"], fitted=frozen)["mean_error_m"]
    if v_n is None or v_d is None or abs(float(v_d) - float(v_n)) <= 0:
        return {"status": "not available"}
    swmod = (cache.fetch(embedding_cache_key("test", "SWmod"), lambda: {})
             if cache is not None and cache.has(embedding_cache_key("test", "SWmod"))
             else None)
    if swmod is not None:
        v_swmod = vertex_row(swmod, fitted=frozen)["mean_error_m"]
        out["ceiling_rho_SWmod"] = float((float(v_d) - float(v_swmod))
                                         / (float(v_d) - float(v_n)))
    for stream, estimate in estimates.items():
        try:
            condition = build_condition(f"EST({stream})", probes030["test"], geometry,
                                        estimates=estimate)
            est_rows = stream_condition_embeddings(
                model, probes030["test"], geometry, condition, split=split,
                config=config, device=device, cache=None, log=log)
            v_est = vertex_row(est_rows, fitted=frozen)["mean_error_m"]
            out["streams"][stream] = {
                "rho_EST": float((float(v_d) - float(v_est))
                                 / (float(v_d) - float(v_n)))}
        except Exception as problem:  # pragma: no cover - a stream may be unavailable
            out["streams"][stream] = {"status": "unavailable", "reason": str(problem)}
    if A_hat is not None and estimates.get("latent"):
        repaired_rows = _latent_repair_rows(rows["test"]["D"], occupancy["test"],
                                            geometry, A_hat, estimates["latent"])
        v_lr = vertex_row(repaired_rows, fitted=frozen)["mean_error_m"]
        out["latent_repair_rho_LR"] = float((float(v_d) - float(v_lr))
                                            / (float(v_d) - float(v_n)))
    return out


def _latent_repair_rows(target_rows, counts, geometry, A_hat, estimate) -> dict:
    """z̃ = z_D − Â u(δ̂_latent) with the estimated gains, then P_N (§6 X2-e)."""
    run_ids = np.asarray(target_rows["run_ids"]).astype(str)
    estimated = np.stack([np.asarray(estimate[str(run)], float) for run in run_ids])
    design = occupancy_count_design(target_rows, counts, geometry) * estimated
    repaired = dict(target_rows)
    repaired["events"] = (np.asarray(target_rows["events"], float)
                          - design @ np.asarray(A_hat, float).T)
    return repaired



def x2_readings(payload) -> dict:
    """X2 readings, per family and seed (§3.3); descriptive."""
    families = {}
    for label, model in (payload.get("models") or {}).items():
        family, seed = _family(label)
        families.setdefault(family, {})[seed] = model

    def combine(function):
        entries = [(seed, function(model)) for seed, model in sorted(
            families[family].items())]
        return classify_seeds(entries, lambda value: value)

    out = {"registered": False, "extension": EXTENSION,
           "check": "X2 readings §6", "families": {}}
    for family, seeds in sorted(families.items()):
        block = {"seeds": {seed: {
            "X2-a": model.get("X2-a"), "X2-b": model.get("X2-b"),
            "X2-c": (model.get("X2-c") or {}).get("rho_LR"),
            "X2-d": model.get("X2-d"), "X2-f": model.get("X2-f")}
            for seed, model in sorted(seeds.items())}, "classes": {}}
        block["classes"]["X2-b"] = combine(
            lambda model: equivariance_class(
                ((model.get("X2-b") or {}).get("occupancy") or {}).get("r2_cs")))
        block["classes"]["X2-c"] = combine(
            lambda model: latent_repair_class((model.get("X2-c") or {}).get("rho_LR")))
        out["families"][family] = block
    return out


# ================================================================ X3-a §7.1
THETA_NAMES = ("x", "y", "z", "lnY")
PARAM_INDEX = {name: index for index, name in enumerate(THETA_NAMES)}


def module_design(mod, n_modules) -> np.ndarray:
    """The ``(n_pmt, n_modules)`` module indicator matrix of the δ derivatives."""
    mod = np.asarray(mod, np.int64)
    design = np.zeros((mod.size, int(n_modules)))
    design[np.arange(mod.size), mod] = 1.0
    return design


def fisher_inputs(geometry, gen, *, n_events=1000, seed=20261106,
                  h_m=1e-3) -> list:
    """§7.1 events: fiducial vertex, log-uniform yield, sampled QE, μ and its Jacobian."""
    rng = lb.keyed_rng(int(seed), FISHER_NAMESPACE, "photon")
    state = lb.nominal_state(geometry)
    fixed = lb.fixed_charge_gain(geometry)
    mod = module_of_pmt(geometry)
    n_modules = int(geometry.n_modules)
    out = []
    for _ in range(int(n_events)):
        vertex = lb.fiducial_vertex(geometry, rng, float(gen.fiducial_fraction))
        yield_photons = float(np.exp(rng.uniform(np.log(float(gen.y_lo)),
                                                 np.log(float(gen.y_hi)))))
        qe = lb.sample_qe(geometry, rng, gen)

        def mu_at(point):
            return (lb.expected_charge(geometry, point, yield_photons, qe,
                                       state.attenuation_length_m, gen) * fixed)

        mu = mu_at(vertex)
        dmu = np.zeros((int(geometry.n_pmt), 4))
        for axis in range(3):
            plus = vertex.copy()
            minus = vertex.copy()
            plus[axis] += float(h_m)
            minus[axis] -= float(h_m)
            dmu[:, axis] = (mu_at(plus) - mu_at(minus)) / (2.0 * float(h_m))
        dmu[:, 3] = mu
        out.append({"vertex_m": vertex, "yield_photons": yield_photons, "qe": qe,
                    "mu": mu, "dmu": dmu, "mod": mod, "n_modules": n_modules})
    return out


def variance_term_information(mu, dmu_theta, mod, *, cuts=None,
                              sigma_spe=lb.SIGMA_SPE, v_dark=lb.V_DARK_READOUT,
                              s_p=lb.F3_SPREAD) -> np.ndarray:
    """The C-var Gaussian variance term ``½ ∂V ∂Vᵀ / V²`` per PMT in the cut (§7.1)."""
    mu = np.asarray(mu, float)
    dmu_theta = np.asarray(dmu_theta, float)
    design = module_design(mod, int(np.max(mod)) + 1)
    variance = mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark) \
        + float(s_p) ** 2 * mu ** 2
    derivative_theta = (1.0 + float(sigma_spe) ** 2 + 2.0 * float(s_p) ** 2 * mu)[:, None] \
        * dmu_theta
    derivative_delta = 2.0 * variance[:, None] * design
    joint = np.concatenate([derivative_theta, derivative_delta], axis=1)
    keep = (mu > 0) & (variance > 0)
    if cuts is not None:
        keep = keep & (mu >= float(cuts))
    information = np.zeros((joint.shape[1], joint.shape[1]))
    if keep.any():
        selected = joint[keep]
        information = 0.5 * (selected.T * (1.0 / variance[keep])) @ selected
    return information


def fisher_blocks(mu, dmu_theta, mod, *, model="C-mean", cuts=None,
                  variance_term=True, sigma_spe=lb.SIGMA_SPE,
                  v_dark=lb.V_DARK_READOUT, s_p=lb.F3_SPREAD) -> dict:
    """§7.1/Appendix E information blocks in ``(θ, δ)`` for one event."""
    mu = np.asarray(mu, float)
    dmu_theta = np.asarray(dmu_theta, float)
    mod = np.asarray(mod, np.int64)
    n_modules = int(mod.max()) + 1
    design = module_design(mod, n_modules)
    joint = np.concatenate([dmu_theta, mu[:, None] * design], axis=1)
    variance = mu * (1.0 + float(sigma_spe) ** 2) + float(v_dark) \
        + float(s_p) ** 2 * mu ** 2
    keep = (mu > 0) & (variance > 0)
    information = np.zeros((joint.shape[1], joint.shape[1]))
    if keep.any():
        information = (joint[keep].T * (1.0 / variance[keep])) @ joint[keep]
    if model == "C-var" and variance_term:
        information = information + variance_term_information(
            mu, dmu_theta, mod, cuts=cuts, sigma_spe=sigma_spe, v_dark=v_dark, s_p=s_p)
    if model == "C-mean+occupancy":
        probability = 1.0 - np.exp(-mu)
        keep_occ = (mu > 0) & (probability > 0) & (probability < 1)
        if keep_occ.any():
            derivative = np.exp(-mu[keep_occ])[:, None] * dmu_theta[keep_occ]
            weight = 1.0 / (probability[keep_occ] * (1.0 - probability[keep_occ]))
            information[:4, :4] += (derivative.T * weight) @ derivative
    if model not in ("C-mean", "C-var", "C-mean+occupancy"):
        raise ValueError(f"unknown Fisher model {model!r}")
    return {"theta_theta": information[:4, :4],
            "theta_delta": information[:4, 4:],
            "delta_theta": information[4:, :4],
            "delta_delta": information[4:, 4:],
            "model": model, "cuts": cuts}


def _case_information(blocks, *, case, sigma=None) -> np.ndarray:
    theta_theta = np.asarray(blocks["theta_theta"], float)
    theta_delta = np.asarray(blocks["theta_delta"], float)
    delta_delta = np.asarray(blocks["delta_delta"], float)
    if case == "known":
        return theta_theta
    if case == "inv":
        effective = theta_theta - theta_delta @ np.linalg.pinv(delta_delta, rcond=1e-12) \
            @ theta_delta.T
        return (effective + effective.T) / 2.0
    if case == "cal":
        if sigma is None:
            raise ValueError("the calibrated case needs a prior width sigma")
        regularised = delta_delta + (1.0 / float(sigma) ** 2) * np.eye(delta_delta.shape[0])
        effective = theta_theta - theta_delta @ np.linalg.solve(regularised,
                                                               theta_delta.T)
        return (effective + effective.T) / 2.0
    raise ValueError(f"unknown Fisher case {case!r}")


def crlb_report(blocks, *, sigmas=FISHER_SIGMAS, relative_floor=1e-6) -> dict:
    """CRLBs and ∞ rules for the known / invariant / calibrated cases (§7.1, App. E)."""
    known_matrix = _case_information(blocks, case="known")
    try:
        known_crlb = np.diag(np.linalg.pinv(known_matrix, rcond=1e-12))
    except np.linalg.LinAlgError:  # pragma: no cover
        known_crlb = np.full(4, np.inf)

    def one(matrix):
        information = np.asarray(np.diag(matrix), float)
        threshold = float(relative_floor) * np.asarray(
            np.abs(np.diag(known_matrix)), float)
        infinite = information <= threshold
        crlb = np.full(4, np.inf)
        remaining = [index for index in range(4) if not infinite[index]]
        if remaining:
            block = np.asarray(matrix, float)[np.ix_(remaining, remaining)]
            values = np.diag(np.linalg.pinv(block, rcond=1e-12))
            for position, index in enumerate(remaining):
                crlb[index] = values[position]
        ratio = np.full(4, np.inf)
        for index in range(4):
            if np.isinf(crlb[index]):
                ratio[index] = np.inf
            elif known_crlb[index] is None or not np.isfinite(known_crlb[index]) \
                    or known_crlb[index] <= 0:
                ratio[index] = np.inf
            else:
                ratio[index] = float(crlb[index] / known_crlb[index])
        return {"crlb": crlb.tolist(), "ratio": ratio.tolist(),
                "n_infinite": int(infinite.sum()),
                "infinite": [THETA_NAMES[index] for index in range(4) if infinite[index]]}

    out = {"known": {"crlb": known_crlb.tolist(), "n_infinite":
                     int(np.sum(~np.isfinite(known_crlb)))},
           "inv": one(_case_information(blocks, case="inv")),
           "cal": {f"{sigma:g}": one(_case_information(blocks, case="cal",
                                                       sigma=sigma))
                   for sigma in sigmas}}
    out["inv/known"] = out["inv"]["ratio"]
    return out


def p0_check(blocks, *, relative_floor=1e-6) -> dict:
    """Proposition check P0: δ-free ln Y information is ≤ 1e-6 of its known value."""
    known = float(np.asarray(blocks["theta_theta"], float)[3, 3])
    efficient = float(np.asarray(_case_information(blocks, case="inv"), float)[3, 3])
    ratio = (abs(efficient) / known) if known > 0 else 0.0
    return {"known_information": known, "inv_information": efficient,
            "ratio": float(ratio), "pass": bool(ratio <= relative_floor)}


def sigma90(sigmas, ratios, *, target=0.90) -> dict:
    """Largest prior width recovering ≥ 90 % of the known-gain information (§7.1)."""
    bound = 1.0 / float(target)
    pairs = [(float(sigma), ratio) for sigma, ratio in zip(sigmas, ratios)]
    finite = [(sigma, ratio) for sigma, ratio in pairs
              if ratio is not None and np.isfinite(ratio)]
    if not finite:
        return {"value": "not reached", "sigma90": None}
    if finite[0][1] > bound:
        return {"value": "not reached at 0.003", "sigma90": None}
    best = None
    for (sigma_lo, ratio_lo), (sigma_hi, ratio_hi) in zip(finite, finite[1:]):
        if ratio_lo <= bound < ratio_hi:
            fraction = (math.log(bound) - math.log(ratio_lo)) \
                / (math.log(ratio_hi) - math.log(ratio_lo))
            log_sigma = math.log(sigma_lo) + fraction * (math.log(sigma_hi)
                                                         - math.log(sigma_lo))
            return {"value": float(math.exp(log_sigma)),
                    "sigma90": float(math.exp(log_sigma))}
        if ratio_hi <= bound:
            best = sigma_hi
    return {"value": float(best) if best is not None else "≥ 0.3",
            "sigma90": best}


def fisher_accounting(geometry, gen, *, n_events=1000, seed=20261106,
                      sigmas=FISHER_SIGMAS, limit=None, root=None, log=print) -> dict:
    """X3-a: Fisher accounting on the WCTE geometry (CPU, §7.1)."""
    started = time.time()
    gate = assert_authorised("x3-fisher", root=root, log=log)
    inputs = fisher_inputs(geometry, gen, n_events=n_events, seed=seed)
    models = {"C-mean": {"cuts": None, "variance_term": False},
              "C-mean+occupancy": {"cuts": None, "variance_term": False},
              "C-var_all": {"cuts": None, "variance_term": True},
              "C-var_ge1": {"cuts": 1.0, "variance_term": True},
              "C-var_ge3": {"cuts": 3.0, "variance_term": True}}
    payload = {"registered": False, "extension": EXTENSION,
               "check": "X3-a Fisher accounting §7.1", "runbook": RUNBOOK_REL,
               "gate": {"status": gate["status"]},
               "theta": list(THETA_NAMES), "sigmas": list(sigmas),
               "n_events": int(len(inputs)), "models": {}, "p0": {},
               "wall_seconds": None}
    for model, options in models.items():
        crlbs = {name: [] for name in ("known", "inv")}
        crlbs.update({f"{sigma:g}": [] for sigma in sigmas})
        n_infinite = {"inv": [0, 0, 0, 0],
                      **{f"{sigma:g}": [0, 0, 0, 0] for sigma in sigmas}}
        p0_failures = 0
        for index, event in enumerate(inputs):
            if limit is not None and index >= int(limit):
                break
            if model == "C-mean+occupancy":
                blocks = fisher_blocks(event["mu"], event["dmu"], event["mod"],
                                       model="C-mean+occupancy")
            elif model == "C-mean":
                blocks = fisher_blocks(event["mu"], event["dmu"], event["mod"],
                                       model="C-mean")
            else:
                blocks = fisher_blocks(event["mu"], event["dmu"], event["mod"],
                                       model="C-var", cuts=options["cuts"],
                                       variance_term=options["variance_term"])
            report = crlb_report(blocks, sigmas=sigmas)
            crlbs["known"].append(report["known"]["crlb"])
            crlbs["inv"].append(report["inv"]["crlb"])
            for name in report["inv"]["infinite"]:
                n_infinite["inv"][THETA_NAMES.index(name)] += 1
            for sigma in sigmas:
                key = f"{sigma:g}"
                crlbs[key].append(report["cal"][key]["crlb"])
                for name in report["cal"][key]["infinite"]:
                    n_infinite[key][THETA_NAMES.index(name)] += 1
            if model == "C-mean":
                check = p0_check(blocks)
                payload["p0"].setdefault("ratios", []).append(check["ratio"])
                if not check["pass"]:
                    p0_failures += 1
        block = {"n_events": int(len(crlbs["inv"])),
                 "n_infinite": {"per_case_total": n_infinite,
                                "per_case_parameter": {
                                    case: dict(zip(THETA_NAMES, counts))
                                    for case, counts in n_infinite.items()}},
                 "median_crlb": {}, "median_ratio": {}}
        for name, values in crlbs.items():
            array = np.asarray(values, float)
            with np.errstate(invalid="ignore"):
                block["median_crlb"][name] = np.median(array, axis=0).tolist()
        known_median = np.asarray(block["median_crlb"]["known"], float)
        for name in ("inv", *_sigmas_keys(sigmas)):
            median = np.asarray(block["median_crlb"][name], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = median / known_median
            block["median_ratio"][name] = np.where(np.isinf(median), np.inf,
                                                   ratio).tolist()
        block["sigma90"] = {
            name: sigma90(sigmas, [block["median_ratio"][f"{sigma:g}"][index]
                                   for sigma in sigmas])
            for index, name in enumerate(THETA_NAMES)}
        payload["models"][model] = block
    ratios = np.asarray(payload["p0"].get("ratios", []), float)
    payload["p0"] = {"n_events": int(ratios.size),
                     "max_ratio": (float(ratios.max()) if ratios.size else None),
                     "relative_floor": 1e-6,
                     "pass": bool(ratios.size and ratios.max() <= 1e-6),
                     "rule": "C-mean efficient ln Y charge information / known-gain "
                             "ln Y information <= 1e-6 on every event (§7.1 P0)"}
    payload["closure"] = _assert_closure("x3-fisher", _resolve_root(root))
    payload["wall_seconds"] = round(time.time() - started, 3)
    return payload


def _sigmas_keys(sigmas) -> list:
    return [f"{sigma:g}" for sigma in sigmas]


# ================================================================ X3-b §7.2
def _event_truth(dataset) -> dict:
    """Per-run event truth (vertex, yield, particle, contained, deposited) from events.npz."""
    out = {}
    for row in dataset.runs:
        run_dir = dataset.directory / "runs" / row["run_dir"]
        with np.load(run_dir / "events.npz", allow_pickle=False) as data:
            fields = {name: np.asarray(data[name]) for name in data.files}
        out[str(row["run_id"])] = fields
    return out


def _features_from_stores(dataset, geometry, *, weighted=True, domains=None) -> dict:
    """Raw features from the hit stores (std = charge-weighted, inv = hit-count-weighted)."""
    rows = {"features": [], "run_ids": [], "event_ids": [], "vertex": [],
            "log_yield": [], "particle": [], "contained": [], "deposited_MeV": [],
            "direction": [], "n_hit": [], "domain": []}
    positions = np.asarray(geometry.positions_m, float)
    truth = _event_truth(dataset)
    for row in dataset.runs:
        run_id = str(row["run_id"])
        store = lb.read_sparse_hits(dataset.directory / "runs" / row["run_dir"]
                                    / "hits.npz")
        run_truth = truth.get(run_id, {})
        index = {int(event): position for position, event in
                 enumerate(np.asarray(run_truth.get("event_ids", store.event_ids),
                                      np.int64))}
        for position, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
            domain = (str(np.asarray(run_truth["domain"])[index[int(event_id)]])
                      if "domain" in run_truth and int(event_id) in index else "photon")
            if domains is not None and domain not in domains:
                continue
            span = slice(int(store.offsets[position]), int(store.offsets[position + 1]))
            pmt = np.asarray(store.pmt_id[span], np.int64)
            charge = np.asarray(store.charge_pe[span], float)
            time = np.asarray(store.time_ns[span], float) / 100.0
            total = float(charge.sum())
            if weighted:
                weight = charge / total if total > 0 else np.zeros_like(charge)
            else:
                weight = (np.ones_like(charge) / charge.size if charge.size
                          else np.zeros_like(charge))
            means = [float((weight * positions[pmt, axis]).sum()) for axis in range(3)] \
                if pmt.size else [0.0, 0.0, 0.0]
            stds = [float(np.sqrt(max((weight * (positions[pmt, axis] - means[axis]) ** 2)
                                      .sum(), 0.0))) for axis in range(3)] \
                if pmt.size else [0.0, 0.0, 0.0]
            t_mean = float((weight * time).sum()) if pmt.size else 0.0
            t_std = float(np.sqrt(max((weight * (time - t_mean) ** 2).sum(), 0.0))) \
                if pmt.size else 0.0
            if weighted:
                features = [math.log(max(total, 1e-6)), float(pmt.size)] + means + stds \
                    + [t_mean, t_std]
            else:
                features = [float(pmt.size)] + means + stds + [t_mean, t_std]
            place = index.get(int(event_id))
            rows["features"].append(features)
            rows["run_ids"].append(run_id)
            rows["event_ids"].append(int(event_id))
            rows["vertex"].append(
                np.asarray(run_truth.get("vertex_m", np.full(3, np.nan)))[place]
                if place is not None else np.full(3, np.nan))
            rows["log_yield"].append(
                float(np.log(max(float(np.asarray(run_truth.get("yield_photons",
                                                                np.nan))[place]),
                                 1e-30))) if place is not None else np.nan)
            particle = (str(np.asarray(run_truth.get("particle", "none"))[place])
                        if place is not None else "none")
            rows["particle"].append(particle)
            rows["contained"].append(
                bool(np.asarray(run_truth.get("contained", False))[place])
                if place is not None else False)
            rows["deposited_MeV"].append(
                float(np.asarray(run_truth.get("deposited_MeV", np.nan))[place])
                if place is not None else np.nan)
            rows["direction"].append(
                np.asarray(run_truth.get("direction_m", np.full(3, np.nan)))[place]
                if place is not None else np.full(3, np.nan))
            rows["n_hit"].append(int(pmt.size))
            rows["domain"].append(domain)
    out = {"features": np.asarray(rows["features"], float),
           "run_ids": np.asarray(rows["run_ids"], dtype=object),
           "event_ids": np.asarray(rows["event_ids"], np.int64),
           "vertex": np.asarray(rows["vertex"], float),
           "log_yield": np.asarray(rows["log_yield"], float),
           "particle": np.asarray(rows["particle"]).astype(str),
           "contained": np.asarray(rows["contained"], bool),
           "deposited_MeV": np.asarray(rows["deposited_MeV"], float),
           "direction": np.asarray(rows["direction"], float),
           "n_hit": np.asarray(rows["n_hit"], np.int64),
           "domain": np.asarray(rows["domain"]).astype(str)}
    out["feature_names"] = (("ln_sum_q", "n_hits", "weighted_mean_x", "weighted_mean_y",
                             "weighted_mean_z", "weighted_sd_x", "weighted_sd_y",
                             "weighted_sd_z", "weighted_mean_t", "weighted_sd_t")
                            if weighted else
                            ("n_hits", "mean_x", "mean_y", "mean_z", "sd_x", "sd_y",
                             "sd_z", "mean_t", "sd_t"))
    return out


def raw_photon_features(dataset, geometry, *, condition=None) -> dict:
    """The §7.2 photon version of the raw-std feature formula."""
    if condition is not None:
        with hit_transform_context(condition):
            return _features_from_stores(dataset, geometry, weighted=True,
                                         domains=("photon",))
    return _features_from_stores(dataset, geometry, weighted=True, domains=("photon",))


_UNIFIED_FIELDS = ("features", "run_ids", "event_ids", "vertex", "log_yield",
                   "particle", "contained", "deposited_MeV", "direction", "n_hit",
                   "domain")


def merge_standard_rows(particle_rows, photon_rows) -> dict:
    """Merge the particle and photon raw-std tables into one event table."""
    particle = {name: np.asarray(particle_rows[name]) for name in _UNIFIED_FIELDS
                if name in particle_rows and name != "domain"}
    particle["domain"] = np.asarray(
        [("muon" if value == "muon" else "electron" if value == "electron" else "none")
         for value in np.asarray(particle_rows.get("particle", [])).astype(str)])
    merged = {}
    for name in _UNIFIED_FIELDS:
        left = particle.get(name)
        right = np.asarray(photon_rows[name]) if name in photon_rows else None
        if left is None and right is None:
            continue
        if left is None:
            merged[name] = right
        elif right is None:
            merged[name] = left
        else:
            merged[name] = np.concatenate([left, right])
    names = photon_rows.get("feature_names")
    if names is not None:
        merged["feature_names"] = names
    return merged


def raw_std_features(dataset, geometry, truth=None, *, condition=None) -> dict:
    """raw-std: the §7.2 standard raw features (no encoder).

    For muon/electron events the particle rows are ``ext_followup.raw_event_features``
    itself, so they reproduce it bit-identically; the photon rows use the same
    formula written for photon events.  When ``condition`` is given every hit read
    goes through the §2.5 transform, so ``condition=SWunit-D`` is the calibrated
    raw variant.
    """
    def produce():
        return {"particle": ef.raw_event_features(dataset, geometry, truth=truth),
                "photon": _features_from_stores(dataset, geometry, weighted=True,
                                                domains=("photon",))}

    if condition is not None:
        with hit_transform_context(condition):
            produced = produce()
    else:
        produced = produce()
    produced["merged"] = merge_standard_rows(produced["particle"], produced["photon"])
    return produced


def raw_inv_features(dataset, geometry, *, condition=None) -> dict:
    """raw-inv: ``[NHits, hit-count mean/SD of x, y, z, t]`` (no charge; §7.2)."""
    if condition is not None:
        with hit_transform_context(condition):
            return _features_from_stores(dataset, geometry, weighted=False)
    return _features_from_stores(dataset, geometry, weighted=False)


def feature_probe_metrics(train_rows, test_rows, *, kind="vertex", fitted=None) -> dict:
    """Fit a probe on a feature table's N probe-train and score one test table."""
    x_train = np.asarray(train_rows["features"], float)
    x_test = np.asarray(test_rows["features"], float)
    if kind == "vertex":
        y_train = np.asarray(train_rows["vertex"], float)
        y_test = np.asarray(test_rows["vertex"], float)
        mask_train = np.isfinite(y_train).all(1) & np.isfinite(x_train).all(1)
        mask_test = np.isfinite(y_test).all(1) & np.isfinite(x_test).all(1)
    elif kind == "log_yield":
        y_train = np.asarray(train_rows["log_yield"], float)[:, None]
        y_test = np.asarray(test_rows["log_yield"], float)[:, None]
        mask_train = np.isfinite(y_train[:, 0]) & (np.asarray(train_rows["domain"])
                                                   == "photon")
        mask_test = np.isfinite(y_test[:, 0]) & (np.asarray(test_rows["domain"])
                                                 == "photon")
    elif kind == "energy":
        y_train = np.asarray(train_rows["deposited_MeV"], float)[:, None]
        y_test = np.asarray(test_rows["deposited_MeV"], float)[:, None]
        mask_train = _electron_mask(train_rows) & np.isfinite(x_train).all(1)
        mask_test = _electron_mask(test_rows) & np.isfinite(x_test).all(1)
    else:
        raise ValueError(f"unknown feature probe kind {kind!r}")
    if mask_test.sum() < 2 or mask_train.sum() < 2:
        return {"status": "not_evaluable", "reason": "too few rows"}
    if kind == "energy" and int(mask_test.sum()) < A3_MIN_CONTAINED:
        return {"status": "not_evaluable",
                "reason": f"{int(mask_test.sum())} contained electrons < "
                          f"{A3_MIN_CONTAINED}"}
    train = la.ProbeDataset(kind=kind, form="raw", x=x_train[mask_train],
                            y=np.asarray(y_train, float)[mask_train],
                            run_ids=np.asarray(train_rows["run_ids"], dtype=object)[
                                mask_train],
                            target_names=("target",), meta={})
    test = la.ProbeDataset(kind=kind, form="raw", x=x_test[mask_test],
                           y=np.asarray(y_test, float)[mask_test],
                           run_ids=np.asarray(test_rows["run_ids"], dtype=object)[
                               mask_test],
                           target_names=("target",), meta={})
    fitted = la.fit_probe(train) if fitted is None else fitted
    prediction = la.probe_predictions(fitted, test)
    out = {"status": "evaluated", "fitted": fitted, "n_test": int(mask_test.sum()),
           "run_ids": np.asarray(test_rows["run_ids"], dtype=object)[mask_test],
           "event_ids": np.asarray(test_rows["event_ids"])[mask_test]}
    if kind == "vertex":
        vertex = lr._vertex_error(prediction, test.y)
        out.update({"mean_error_m": vertex.get("resolution"),
                    "median_error_m": vertex.get("median"),
                    "__errors__": np.linalg.norm(
                        np.asarray(prediction, float) - np.asarray(test.y, float), axis=1)})
    elif kind == "log_yield":
        errors = np.asarray(prediction, float).ravel() - np.asarray(test.y,
                                                                    float).ravel()
        out.update({"resolution": rms(errors), "__errors__":
                    np.abs(errors),
                    "r2": _r2_from_predictions(prediction, test.y)})
    else:
        relative = ((np.asarray(prediction, float).ravel()
                     - np.asarray(test.y, float).ravel())
                    / np.asarray(test.y, float).ravel())
        out.update({"resolution": rms(relative), "__errors__": np.abs(relative),
                    "r2": _r2_from_predictions(prediction, test.y)})
    return out


# ------------------------------------------------------------------- LEACE
def whitening_from_covariance(covariance, *, eps_scale=1e-6) -> tuple:
    covariance = np.asarray(covariance, float)
    d = covariance.shape[0]
    epsilon = float(eps_scale) * float(np.trace(covariance)) / float(d)
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
    W = (vectors * (1.0 / np.sqrt(values + epsilon))) @ vectors.T
    W_inv = (vectors * np.sqrt(values + epsilon)) @ vectors.T
    return W, W_inv, epsilon


def leace_fit(x, z, *, rank=None, eps_scale=1e-6) -> dict:
    """LEACE eraser parameters (Appendix D); ``rank=None`` is the full-rank rule."""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    if x.ndim != 2:
        raise ValueError("leace_fit: x must be a matrix")
    z = z.reshape(z.shape[0], -1)
    n = x.shape[0]
    mu_x = x.mean(axis=0)
    mu_z = z.mean(axis=0)
    xc = x - mu_x
    zc = z - mu_z
    covariance = xc.T @ xc / n
    cross = xc.T @ zc / n
    W, W_inv, epsilon = whitening_from_covariance(covariance, eps_scale=eps_scale)
    matrix = W @ cross
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    if rank is None:
        keep = S > (1e-8 * S[0] if S.size and S[0] > 0 else 0.0)
        U_r = U[:, keep]
        r = int(keep.sum())
    else:
        r = int(rank)
        U_r = U[:, :r]
    return {"mu_x": mu_x, "mu_z": mu_z, "W": W, "W_inv": W_inv, "epsilon": epsilon,
            "singular_values": S, "U_r": U_r, "rank": int(r),
            "cross_covariance": cross, "covariance": covariance}


def leace_erase(eraser, x, *, U_r=None) -> np.ndarray:
    """r(x) = x − W⁻¹ P W (x − μ_X), with P = U_r U_rᵀ (§7.2, Appendix D)."""
    x = np.asarray(x, float)
    basis = eraser["U_r"] if U_r is None else np.asarray(U_r, float)
    projection = basis @ basis.T
    centred = x - np.asarray(eraser["mu_x"], float)
    inner = np.asarray(eraser["W"], float) @ centred.T
    return x - (np.asarray(eraser["W_inv"], float) @ (projection @ inner)).T


def leace_cross_covariance(eraser, x, z, *, U_r=None) -> dict:
    """Cov(r(X), Z) and its norm relative to Cov(X, Z) (Appendix D check)."""
    erased = leace_erase(eraser, x, U_r=U_r)
    z = np.asarray(z, float).reshape(np.asarray(x).shape[0], -1)
    erased_c = erased - erased.mean(axis=0)
    z_c = z - z.mean(axis=0)
    residual = erased_c.T @ z_c / len(x)
    original = (np.asarray(x, float) - np.asarray(x, float).mean(axis=0)).T @ z_c / len(x)
    denominator = float(np.linalg.norm(original))
    return {"residual_norm": float(np.linalg.norm(residual)),
            "original_norm": denominator,
            "relative": (0.0 if denominator <= 0
                         else float(np.linalg.norm(residual) / denominator))}


def random_control_basis(rng, d, r) -> np.ndarray:
    """A Haar-random orthonormal ``d × r`` basis (the same-rank random control)."""
    matrix = np.asarray(rng.standard_normal((int(d), int(r))), float)
    basis, _ = np.linalg.qr(matrix)
    return basis


def permute_concepts_by_run(z, run_ids, rng) -> np.ndarray:
    """Permute run ids, moving each run's N and D concept vectors together (§7.2)."""
    z = np.asarray(z, float)
    run_ids = np.asarray(run_ids).astype(str)
    runs = np.unique(run_ids)
    order = rng.permutation(len(runs))
    permuted = np.array(z, copy=True)
    for position, run in enumerate(runs):
        source = runs[order[position]]
        permuted[run_ids == run] = z[run_ids == source]
    return permuted


def permutation_maxima(x, z, run_ids, *, n_permutations=LEACE_PERMUTATIONS,
                       rng=None) -> dict:
    """The null of the largest singular value of W Σ_XZ under run-id permutation."""
    x = np.asarray(x, float)
    z = np.asarray(z, float).reshape(x.shape[0], -1)
    run_ids = np.asarray(run_ids).astype(str)
    W, _W_inv, _eps = whitening_from_covariance(x.T @ x / len(x)
                                                - np.outer(x.mean(0), x.mean(0)))
    runs = np.unique(run_ids)
    rng = rng if rng is not None else np.random.default_rng(0)
    maxima = np.empty(int(n_permutations), float)
    for index in range(int(n_permutations)):
        permuted = permute_concepts_by_run(z, run_ids, rng)
        cross = (x - x.mean(0)).T @ (permuted - permuted.mean(0)) / len(x)
        maxima[index] = float(np.linalg.svd(W @ cross, compute_uv=False)[0])
    return {"maxima": maxima, "q95": float(np.quantile(maxima, 0.95)),
            "n_permutations": int(n_permutations), "n_runs": int(len(runs))}


def reduction_rank(x, z, run_ids, *, key, label, domain, n_permutations=LEACE_PERMUTATIONS,
                   eps_scale=1e-6) -> dict:
    """E2r rank rule: r = #{j : s_j > q95(null s₁)} (§7.2, Appendix D)."""
    x = np.asarray(x, float)
    z = np.asarray(z, float).reshape(x.shape[0], -1)
    eraser = leace_fit(x, z, eps_scale=eps_scale)
    cross = eraser["cross_covariance"]
    singular = np.linalg.svd(eraser["W"] @ cross, compute_uv=False)
    rng = lb.keyed_rng(20261106, LEACE_PERM_NAMESPACE, f"{label}/{domain}")
    null = permutation_maxima(x, z, run_ids, n_permutations=n_permutations, rng=rng)
    rank = int(np.sum(singular > null["q95"]))
    return {"rank": rank, "singular_values": singular.tolist(),
            "q95_null": null["q95"], "n_permutations": int(n_permutations),
            "eraser": eraser}


# ------------------------------------------------------ X3-b variants §7.2
def e1_concepts(dataset, geometry) -> dict:
    """E1: the twin-charge-weighted effective log gain of each event (§7.2)."""
    out = {}
    for row in dataset.runs:
        run_id = str(row["run_id"])
        state = run_state(row, dataset, geometry)
        log_gain = np.log(np.asarray(state.gains, float))
        twin = lb.read_sparse_hits(dataset.directory / "runs" / row["run_dir"]
                                   / "twin_hits.npz")
        for position, event_id in enumerate(np.asarray(twin.event_ids, np.int64)):
            span = slice(int(twin.offsets[position]), int(twin.offsets[position + 1]))
            pmt = np.asarray(twin.pmt_id[span], np.int64)
            weight = np.asarray(twin.charge_pe[span], float)
            total = float(weight.sum())
            out[(run_id, int(event_id))] = (float((weight * log_gain[pmt]).sum() / total)
                                            if total > 0 else 0.0)
    return out


def e2_concepts(dataset, geometry) -> dict:
    """E2r: the run's module log-gain vector (105) (§7.2; the permutation unit)."""
    out = {}
    for row in dataset.runs:
        state = run_state(row, dataset, geometry)
        out[str(row["run_id"])] = np.log(np.asarray(state.module_gain, float))
    return out


def _concept_matrix(rows, concepts, *, kind) -> np.ndarray:
    event_ids = np.asarray(rows["event_ids"], np.int64)
    run_ids = np.asarray(rows["run_ids"]).astype(str)
    if kind == "E1":
        return np.asarray([concepts[(run, int(event))]
                           for run, event in zip(run_ids, event_ids)], float)[:, None]
    return np.stack([concepts[run] for run in run_ids])


def erase_rows(rows, eraser, *, U_r=None) -> dict:
    out = dict(rows)
    out["events"] = leace_erase(eraser, np.asarray(rows["events"], float), U_r=U_r)
    return out


def _embedding_variant(standard, erased=None) -> dict:
    """Vertex / log-yield / energy metrics of one variant's embedding rows."""
    def metrics(rows_train, rows_n, rows_d):
        out = {}
        if "vertex" in rows_train and "vertex" in rows_n:
            vertex_fit = fit_vertex_probe(rows_train)
            out["vertex"] = {"N": vertex_row(rows_n, fitted=vertex_fit),
                             "D": vertex_row(rows_d, fitted=vertex_fit)}
        if "log_yield" in rows_train and "log_yield" in rows_n:
            yield_fit = fit_log_yield_probe(rows_train)
            out["log_yield"] = {"N": log_yield_row(rows_n, fitted=yield_fit),
                                "D": log_yield_row(rows_d, fitted=yield_fit)}
        if "particle" in rows_n and np.asarray(rows_n["particle"]).size:
            energy_fit = fit_energy_probe(rows_train)
            out["energy"] = {"N": energy_metrics(rows_train, rows_n, fitted=energy_fit),
                             "D": energy_metrics(rows_train, rows_d,
                                                 fitted=energy_fit)}
        return out

    source = erased if erased is not None else standard
    return metrics(source["train"], source["test"], source["drift"])


def _feature_variant(name, standard_rows, invariant_rows, calibrated_rows=None) -> dict:
    """Raw-feature family variant block: probe on own N train, metrics N and D."""
    def block(train, test_n, test_d):
        return {"vertex": {"N": feature_probe_metrics(train, test_n, kind="vertex"),
                           "D": feature_probe_metrics(train, test_d, kind="vertex")},
                "log_yield": {"N": feature_probe_metrics(train, test_n,
                                                         kind="log_yield"),
                              "D": feature_probe_metrics(train, test_d,
                                                         kind="log_yield")},
                "energy": {"N": feature_probe_metrics(train, test_n, kind="energy"),
                           "D": feature_probe_metrics(train, test_d, kind="energy")}}

    out = {"standard": block(standard_rows["train"], standard_rows["test"],
                             standard_rows["drift"]),
           "invariant": block(invariant_rows["train"], invariant_rows["test"],
                              invariant_rows["drift"]), "name": name}
    if calibrated_rows is not None:
        # the calibrated raw variant is evaluated at D only; keep the same
        # {"metric": {"D": block}} shape _compare_at_d expects (review B10.3)
        train = standard_rows["train"]
        out["calibrated"] = {
            "vertex": {"D": feature_probe_metrics(train, calibrated_rows,
                                                  kind="vertex")},
            "log_yield": {"D": feature_probe_metrics(train, calibrated_rows,
                                                     kind="log_yield")},
            "energy": {"D": feature_probe_metrics(train, calibrated_rows,
                                                  kind="energy")}}
    return out


def _random_control_bases(label, domain, concept, d, r, n=LEACE_RANDOM_CONTROLS):
    """§7.2 random controls: one keyed stream per control index ``i``."""
    bases = []
    for index in range(int(n)):
        rng = lb.keyed_rng(20261106, LEACE_RAND_NAMESPACE,
                           f"{label}/{domain}/{concept}/{index}")
        bases.append(random_control_basis(rng, int(d), int(r)))
    return bases


def _leace_variant(*, label, domain, rows, concepts, rank=None,
                   permutations=LEACE_PERMUTATIONS,
                   random_controls=LEACE_RANDOM_CONTROLS) -> dict:
    """Fit a LEACE eraser on pooled N and D probe-train embeddings and score it.

    The §7.2 erasure sanity is enforced: an unerased concept R² <= 0.02 makes the
    variant ``not_applicable``; an erased R² > 0.02 makes it
    ``unavailable (erasure incomplete)``.
    """
    train_n, train_d = rows["train"]["N"], rows["train"]["D"]
    test_n, test_d = rows["test"]["N"], rows["test"]["D"]
    x = np.concatenate([np.asarray(train_n["events"], float),
                        np.asarray(train_d["events"], float)])
    run_ids = np.concatenate([np.asarray(train_n["run_ids"]).astype(str),
                              np.asarray(train_d["run_ids"]).astype(str)])
    z = np.concatenate([np.asarray(concepts["train"]["N"], float),
                        np.asarray(concepts["train"]["D"], float)]).reshape(x.shape[0], -1)
    eraser = leace_fit(x, z, rank=rank)
    sanity = leace_sanity(eraser, x, z, run_ids,
                          np.asarray(test_n["events"], float),
                          np.asarray(concepts["test"]["N"], float),
                          np.asarray(test_n["run_ids"]).astype(str))
    concept = "E1" if eraser["rank"] == 1 else "E2r"
    if not sanity["applicable"]:
        return {"status": "not_applicable", "sanity": sanity, "concept": concept,
                "rank": int(eraser["rank"]),
                "reason": "unerased concept R² <= 0.02: concept not linearly present"}
    if not sanity["complete"]:
        return {"status": "unavailable (erasure incomplete)", "sanity": sanity,
                "concept": concept, "rank": int(eraser["rank"]),
                "reason": "erased concept R² > 0.02: erasure incomplete"}
    erased = {"train": erase_rows(train_n, eraser),
              "test": erase_rows(test_n, eraser),
              "drift": erase_rows(test_d, eraser)}
    standard = {"train": train_n, "test": test_n, "drift": test_d}
    out = {"status": "evaluated", "standard": _embedding_variant(standard),
           "variant": _embedding_variant(erased), "sanity": sanity,
           "concept": concept, "rank": int(eraser["rank"]), "controls": []}
    bases = _random_control_bases(label, domain, concept, eraser["W"].shape[0],
                                  eraser["U_r"].shape[1], random_controls)
    for basis in bases:
        erased_random = {"train": erase_rows(train_n, eraser, U_r=basis),
                         "test": erase_rows(test_n, eraser, U_r=basis),
                         "drift": erase_rows(test_d, eraser, U_r=basis)}
        out["controls"].append(_embedding_variant(erased_random))
    return out


def leace_sanity(eraser, x_train, z_train, run_ids_train, x_test, z_test,
                 run_ids_test) -> dict:
    """§7.2 erasure sanity: concept R² unerased and erased on probe-test."""
    x_train = np.asarray(x_train, float)
    z_train = np.asarray(z_train, float).reshape(x_train.shape[0], -1)
    x_test = np.asarray(x_test, float)
    z_test = np.asarray(z_test, float).reshape(x_test.shape[0], -1)
    run_ids_train = np.asarray(run_ids_train).astype(str)
    run_ids_test = np.asarray(run_ids_test).astype(str)
    train = la.ProbeDataset(kind="concept", form="raw", x=x_train, y=z_train,
                            run_ids=run_ids_train,
                            target_names=tuple(f"c{i}" for i in range(z_train.shape[1])),
                            meta={})
    test = la.ProbeDataset(kind="concept", form="raw", x=x_test, y=z_test,
                           run_ids=run_ids_test,
                           target_names=train.target_names, meta={})
    unerased = la.probe_r2(train, test)
    train_e = dataclasses.replace(train, x=leace_erase(eraser, x_train))
    test_e = dataclasses.replace(test, x=leace_erase(eraser, x_test))
    erased = la.probe_r2(train_e, test_e)
    applicable = bool(unerased.get("status") == "evaluated"
                      and (unerased.get("r2_mean") or 0.0) > LEACE_SANITY_R2)
    return {"applicable": applicable, "unerased_r2": unerased.get("r2_mean"),
            "erased_r2": erased.get("r2_mean"),
            "status": ("evaluated" if applicable else "not_applicable"),
            "complete": (not applicable
                         or (erased.get("r2_mean") is not None
                             and erased["r2_mean"] <= LEACE_SANITY_R2))}


def _variant_metric_rows(rows) -> dict:
    return {"features": np.asarray(rows["features"], float),
            "vertex": np.asarray(rows["vertex"], float),
            "log_yield": np.asarray(rows["log_yield"], float),
            "particle": np.asarray(rows["particle"]).astype(str),
            "contained": np.asarray(rows["contained"], bool),
            "deposited_MeV": np.asarray(rows["deposited_MeV"], float),
            "run_ids": np.asarray(rows["run_ids"], dtype=object),
            "event_ids": np.asarray(rows["event_ids"], np.int64),
            "domain": np.asarray(rows["domain"]).astype(str)}


def x3(root=None, *, config, gen, arms_config, geometry, inventory=None,
       data_root=DATA_ROOT_DEFAULT, device="cpu", ext_root=EXT_ROOT,
       fisher_payload=None, limit=None, log=print) -> dict:
    """X3-b: invariant variants against retain-and-calibrate (§7.2)."""
    root = _resolve_root(root)
    gate = assert_authorised("x3", root=root, log=log)
    inv = inventory if inventory is not None else part_a_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    truth = {role: ef.load_particle_truth(data_root, PROBE_KEYS_S005[role])
             for role in ("train", "test")}
    swunit = build_condition("SWunit", probes030["test"], geometry)
    payload = {"registered": False, "extension": EXTENSION, "check": "X3-b §7.2",
               "runbook": RUNBOOK_REL, "families": {}, "unavailable": [],
               "gate": {"status": gate["status"]}, "wall_checks": {},
               "wall_seconds": None}
    # raw features (no encoder); the drift (s = 0.30) probe-test set reuses the
    # s = 0.05 probe-test particle truth, the same convention as _x3_init9000
    # (the probe sets share their events and run ids across settings; §2.2)
    standard = {}
    for role, dataset, truth_role in (("train", probes005["train"], "train"),
                                      ("test", probes005["test"], "test"),
                                      ("drift", probes030["test"], "test")):
        standard[role] = _variant_metric_rows(
            raw_std_features(dataset, geometry, truth=truth[truth_role])["merged"])
    invariant = {"train": _variant_metric_rows(raw_inv_features(probes005["train"],
                                                                geometry)),
                 "test": _variant_metric_rows(raw_inv_features(probes005["test"],
                                                              geometry)),
                 "drift": _variant_metric_rows(raw_inv_features(probes030["test"],
                                                               geometry))}
    calibrated = _variant_metric_rows(raw_std_features(
        probes030["test"], geometry, truth=truth["test"],
        condition=swunit)["merged"])
    payload["families"]["raw"] = _feature_variant("raw", standard, invariant,
                                                  calibrated_rows=calibrated)
    # init9000 invariant tokens (photon and particle)
    payload["families"]["init9000"] = _x3_init9000(
        inv, probes005, probes030, geometry, config, arms_config, gen, data_root,
        device, truth, ext_root, limit=limit, log=log)
    # trained encoders: LEACE on the X1 embeddings
    payload["families"][
        "leace"] = _x3_leace(inv, probes005, probes030, geometry, ext_root, limit=limit,
                             log=log)
    payload["readings"] = x3_readings(payload, fisher=fisher_payload)
    payload = _strip_internal(payload)
    payload["complete"] = bool(payload["readings"]["variants"])
    payload["closure"] = _assert_closure("x3", root)
    return payload


def _x3_init9000(inv, probes005, probes030, geometry, config, arms_config, gen,
                 data_root, device, truth, ext_root, *, limit=None, log=print) -> dict:
    """X3-b init9000: invariant tokens for photon and particle events (§7.2).

    Photon N (train, test) and D (test), particle N (train, test) and D (test); the
    calibrated variant is the standard frozen probe on the X1 SWunit D inputs.
    """
    row = inv["rows"].get("init9000")
    if row is None:
        return {"status": "unavailable", "reason": "init9000 absent"}
    cache = embedding_cache(ext_root, "init9000")
    split = row_split(row)
    with ef.item_eval_context(row):
        arm = ef.build_item_arm(row, config=config, gen=gen, arms_config=arms_config,
                                geometry=geometry, data_root=data_root, device=device,
                                log=log)
    model = getattr(arm, "jepa", arm)
    datasets = {"train": probes005["train"], "test": probes005["test"],
                "drift": probes030["test"]}
    condition_names = {"train": "N", "test": "N", "drift": "D"}
    standard = {"train": {}, "test": {}, "drift": {}}
    invariant = {"train": {}, "test": {}, "drift": {}}
    for role, dataset in datasets.items():
        condition = build_condition(condition_names[role], dataset, geometry)
        standard[role] = stream_condition_embeddings(
            model, dataset, geometry, condition, split=split, config=config,
            device=device, cache=None, max_events=limit, log=log)
        with invariant_token_context(geometry):
            invariant[role] = stream_condition_embeddings(
                model, dataset, geometry, condition, split=split, config=config,
                device=device, cache=None, max_events=limit, log=log)
    standard_variant = _embedding_variant(standard)
    invariant_variant = _embedding_variant(invariant)
    # particle N (train, test) and D (test): the electron-energy reading
    particle = {}
    for name in ("standard", "invariant"):
        particle[name] = {"train": {}, "test": {}, "drift": {}}
    for role, dataset in datasets.items():
        condition = build_condition(condition_names[role], dataset, geometry)
        truth_key = "test" if role == "drift" else role
        particle["standard"][role] = stream_condition_particles(
            model, dataset, truth[truth_key], geometry, condition, device=device,
            max_events=limit, log=log)
        with invariant_token_context(geometry):
            particle["invariant"][role] = stream_condition_particles(
                model, dataset, truth[truth_key], geometry, condition, device=device,
                max_events=limit, log=log)
    standard_particle = _embedding_variant(particle["standard"])
    invariant_particle = _embedding_variant(particle["invariant"])
    if standard_particle.get("energy"):
        standard_variant["energy"] = standard_particle["energy"]
    if invariant_particle.get("energy"):
        invariant_variant["energy"] = invariant_particle["energy"]
    out = {"status": "evaluated", "split": split, "standard": standard_variant,
           "invariant": invariant_variant, "particle": particle["invariant"]}
    # calibrated: standard frozen probe on the X1 SWunit D inputs
    calibrated = {}
    try:
        n_train = (cache.fetch(embedding_cache_key("train", "N"), lambda: {})
                   if cache.has(embedding_cache_key("train", "N")) else None)
        swunit = (cache.fetch(embedding_cache_key("test", "SWunit"), lambda: {})
                  if cache.has(embedding_cache_key("test", "SWunit")) else None)
        if n_train is not None and swunit is not None:
            calibrated["vertex"] = {"D": vertex_row(
                swunit, fitted=fit_vertex_probe(n_train))}
            calibrated["log_yield"] = {"D": log_yield_row(
                swunit, fitted=fit_log_yield_probe(n_train))}
        n_particle = (cache.fetch("particle__train__N", lambda: {})
                      if cache.has("particle__train__N") else None)
        sw_particle = (cache.fetch("particle__test__SWunit", lambda: {})
                       if cache.has("particle__test__SWunit") else None)
        if n_particle is not None and sw_particle is not None:
            calibrated["energy"] = {"D": energy_metrics(
                n_particle, sw_particle, fitted=fit_energy_probe(n_particle))}
    except Exception as problem:  # pragma: no cover - a missing/shifted cache
        calibrated = {"status": "unavailable", "reason": str(problem)}
    out["calibrated"] = calibrated
    return out


def _x3_leace(inv, probes005, probes030, geometry, ext_root, *, limit=None, log=print):
    """LEACE variants of the X1-cached embeddings (photon; particle for the MAEs)."""
    planned = ("mae_within_trigger9000", "mae_within_trigger9001",
               "mae_across_trigger9000", "mae_across_trigger9001",
               "t1_s030_within_trigger9000", "t1_s030_within_trigger9001",
               "t1_s030_across_trigger9000", "t1_s030_across_trigger9001",
               "frozen_target9000", "frozen_target9001")
    out = {"status": "evaluated", "models": {}, "unavailable": []}
    e1 = {role: {setting: e1_concepts(probes005[role] if setting == "N"
                                      else probes030[role], geometry)
                 for setting in ("N", "D")} for role in ("train", "test")}
    e2 = {role: {setting: e2_concepts(probes005[role] if setting == "N"
                                      else probes030[role], geometry)
                 for setting in ("N", "D")} for role in ("train", "test")}
    keys = {(role, setting): embedding_cache_key(role, setting)
            for role in ("train", "test") for setting in ("N", "D")}
    for label in planned:
        cache = embedding_cache(ext_root, label)
        if not all(cache.has(key) for key in keys.values()):
            out["unavailable"].append({"label": label,
                                       "reason": "X1 embeddings are not cached"})
            continue
        rows = {role: {setting: cache.fetch(keys[(role, setting)], lambda: {})
                       for setting in ("N", "D")} for role in ("train", "test")}
        for concept, rank in (("E1", 1), ("E2r", None)):
            source = e1 if concept == "E1" else e2
            concepts = {role: {setting: _concept_matrix(rows[role][setting],
                                                       source[role][setting],
                                                       kind=concept)
                               for setting in ("N", "D")}
                        for role in ("train", "test")}
            if concept == "E2r" and rank is None:
                x_train = np.concatenate([rows["train"]["N"]["events"],
                                          rows["train"]["D"]["events"]])
                z_train = np.concatenate([concepts["train"]["N"],
                                          concepts["train"]["D"]])
                run_ids = np.concatenate([
                    np.asarray(rows["train"]["N"]["run_ids"]).astype(str),
                    np.asarray(rows["train"]["D"]["run_ids"]).astype(str)])
                ranked = reduction_rank(x_train, z_train, run_ids, key=label,
                                        label=label, domain="photon")
                rank = ranked["rank"]
            if concept == "E2r" and rank == 0:
                out["models"][f"{label}/{concept}"] = {
                    "status": "not_applicable", "rank": 0,
                    "reason": "r = 0: the reduced-rank concept is not applicable"}
                continue
            try:
                variant = _leace_variant(label=label, domain="photon", rows=rows,
                                         concepts=concepts, rank=rank)
            except Exception as problem:  # pragma: no cover
                out["models"][f"{label}/{concept}"] = {
                    "status": "unavailable", "reason": str(problem)}
                continue
            if variant.get("status") != "evaluated":
                out["models"][f"{label}/{concept}"] = variant
                continue
            swunit_key = embedding_cache_key("test", "SWunit")
            n_key = embedding_cache_key("train", "N")
            if cache.has(swunit_key) and cache.has(n_key):
                swuni = cache.fetch(swunit_key, lambda: {})
                n_train = cache.fetch(n_key, lambda: {})
                variant["calibrated"] = {
                    "vertex": {"D": vertex_row(
                        swuni, fitted=fit_vertex_probe(n_train))},
                    "log_yield": {"D": log_yield_row(
                        swuni, fitted=fit_log_yield_probe(n_train))}}
            if label in MAE_MODELS:
                _attach_particle_energy(variant, label, geometry, e1, e2, cache,
                                        concept=concept, log=log)
            out["models"][f"{label}/{concept}"] = variant
    return out


def _particle_rows(cache, role, setting):
    key = f"particle__{role}__{setting}"
    return cache.fetch(key, lambda: {}) if cache.has(key) else None


def _particle_concepts(rows, e1, e2, role, setting, concept):
    source = e1 if concept == "E1" else e2
    return _concept_matrix(rows, source[role][setting], kind=concept)


def _attach_particle_energy(variant, label, geometry, e1, e2, cache, *, concept,
                            log=print) -> dict:
    """Add the §7.2 electron-energy LEACE reading of the four MAE models.

    The eraser is fit on the pooled N and D particle probe-train embeddings, with
    the same concept family as the photon variant; only the energy metric is kept.
    """
    particle = {role: {setting: _particle_rows(cache, role, setting)
                       for setting in ("N", "D")}
                for role in ("train", "test")}
    if any(particle[role][setting] is None for role in ("train", "test")
           for setting in ("N", "D")):
        variant["particle_energy"] = {"status": "unavailable",
                                      "reason": "particle embeddings are not cached"}
        return variant
    rows = {"train": {"N": particle["train"]["N"], "D": particle["train"]["D"]},
            "test": {"N": particle["test"]["N"], "D": particle["test"]["D"]}}
    concepts = {role: {setting: _particle_concepts(rows[role][setting], e1, e2,
                                                   role, setting, concept)
                       for setting in ("N", "D")}
                for role in ("train", "test")}
    rank = None
    if concept == "E2r":
        x_train = np.concatenate([rows["train"]["N"]["events"],
                                  rows["train"]["D"]["events"]])
        z_train = np.concatenate([concepts["train"]["N"], concepts["train"]["D"]])
        run_ids = np.concatenate([
            np.asarray(rows["train"]["N"]["run_ids"]).astype(str),
            np.asarray(rows["train"]["D"]["run_ids"]).astype(str)])
        rank = reduction_rank(x_train, z_train, run_ids, key=label,
                              label=label, domain="particle")["rank"]
        if rank == 0:
            variant["particle_energy"] = {"status": "not_applicable", "rank": 0}
            return variant
    particle_variant = _leace_variant(label=label, domain="particle", rows=rows,
                                      concepts=concepts, rank=rank)
    variant["particle_energy"] = particle_variant
    if particle_variant["variant"].get("energy"):
        variant["variant"]["energy"] = particle_variant["variant"]["energy"]
        variant["standard"]["energy"] = particle_variant["standard"].get("energy")
        for index, control in enumerate(particle_variant.get("controls") or []):
            if index < len(variant.get("controls") or []) and control.get("energy"):
                variant["controls"][index]["energy"] = control["energy"]
    return variant


def _metric_rows(variant, metric, environment) -> dict | None:
    """One variant's per-event metric block (resolution, errors, run/event ids)."""
    block = ((variant or {}).get(metric) or {}).get(environment)
    if not block or block.get("status") == "not_evaluable":
        return None
    errors = block.get("__errors__", block.get("errors"))
    errors = None if errors is None else np.asarray(errors, float).ravel()
    run_ids = np.asarray(block.get("run_ids", []), dtype=object).astype(str)
    event_ids = np.asarray(block.get("event_ids", []), np.int64)
    if metric == "vertex":
        resolution = block.get("mean_error_m")
    else:
        resolution = block.get("resolution")
    if resolution is None:
        return None
    return {"resolution": float(resolution), "errors": errors,
            "run_ids": run_ids, "event_ids": event_ids}


def _align_metric_pair(left, right):
    """Align two metric blocks on ``(run_id, event_id)``; raise if keys differ."""
    if left is None or right is None or left.get("errors") is None \
            or right.get("errors") is None:
        return None
    if left["errors"].size != left["run_ids"].size \
            or right["errors"].size != right["run_ids"].size:
        return None
    keys_left = [(run, int(event)) for run, event in zip(left["run_ids"],
                                                         left["event_ids"])]
    keys_right = {key: index for index, key in enumerate(
        zip(right["run_ids"], right["event_ids"]))}
    if len(keys_left) != len(set(keys_left)) or set(keys_left) != set(keys_right):
        raise ValueError("metric blocks do not share their (run_id, event_id) keys")
    order = np.asarray([keys_right[key] for key in keys_left], np.int64)
    return (left["errors"], right["errors"][order], left["run_ids"])


def _price_class(rho, ci95) -> str:
    """§7.2 X3-1 fixed price class from rho_E and its run-cluster CI."""
    if rho is None:
        return "not available"
    excludes_zero = bool(ci95 is not None
                         and (float(ci95[0]) > 0.0 or float(ci95[1]) < 0.0))
    if rho >= _X3_PRICE_HIGH:
        if ci95 is None:
            return "not available (CI missing)"
        return "price present" if excludes_zero else "unclear"
    if rho <= _X3_PRICE_NONE:
        return "no price"
    return "unclear"


def _ratio_and_ci(left, right) -> dict:
    """res_right / res_left and the paired run-cluster CI of the RMS difference."""
    if left is None or right is None:
        return {"rho": None, "ci95": None, "bootstrap": None}
    rho = (float(right["resolution"]) / float(left["resolution"])
           if left["resolution"] else None)
    out = {"rho": rho, "ci95": None, "bootstrap": None}
    aligned = _align_metric_pair(left, right)
    if aligned is not None:
        left_errors, right_errors, run_ids = aligned
        bootstrap = paired_cluster_bootstrap(left_errors, right_errors, run_ids,
                                             statistic=rms)
        out["bootstrap"] = bootstrap
        out["ci95"] = bootstrap["ci95"]
    return out


def x3_readings(payload, *, fisher=None) -> dict:
    """X3-1 … X3-5 readings (§7.2)."""
    out = {"registered": False, "extension": EXTENSION,
           "check": "X3-b readings §7.2", "variants": {}, "X3-5": None}
    families = payload.get("families") or {}
    raw = families.get("raw") or {}
    if raw:
        out["variants"]["raw"] = _variant_reading(raw, raw.get("standard"),
                                                  raw.get("invariant"))
    init = families.get("init9000") or {}
    if init.get("status") == "evaluated":
        out["variants"]["init9000"] = _variant_reading(
            init, init.get("standard"), init.get("invariant"))
    leace = families.get("leace") or {}
    for key, entry in (leace.get("models") or {}).items():
        if entry.get("status") != "evaluated" or entry.get("standard") is None:
            continue
        out["variants"][key] = _variant_reading(entry, entry.get("standard"),
                                                entry.get("variant"), leace=entry)
    if fisher:
        out["X3-5"] = _x3_five(fisher, out["variants"])
    return out


def _variant_reading(entry, standard, invariant, *, leace=None) -> dict:
    """X3-1 … X3-4 for one variant pair (``standard`` may be a raw/embedding block)."""
    standard = standard or {}
    invariant = invariant or {}
    result = {"X3-1": {}}
    for metric in ("log_yield", "energy"):
        left = _metric_rows(standard, metric, "N")
        right = _metric_rows(invariant, metric, "N")
        ratio = _ratio_and_ci(left, right)
        ratio["rho_E"] = ratio.get("rho")
        ci = ratio.get("ci95")
        ratio["ci_excludes_zero"] = bool(ci is not None
                                         and (float(ci[0]) > 0.0
                                              or float(ci[1]) < 0.0))
        ratio["class"] = _price_class(ratio["rho_E"], ci)
        ratio["gain_specific"] = None
        result["X3-1"][metric] = ratio
    v_std = _metric_rows(standard, "vertex", "N")
    v_inv = _metric_rows(invariant, "vertex", "N")
    result["X3-2"] = {"difference_m": (None if not (v_std and v_inv)
                                       else float(v_inv["resolution"])
                                       - float(v_std["resolution"]))}
    result["X3-3"] = _x3_robustness(standard, invariant)
    calibrated = (entry or {}).get("calibrated")
    result["X3-4"] = _x3_dominance(calibrated, invariant)
    if leace and leace.get("controls"):
        gain = _gain_specific(result["X3-1"], standard, invariant,
                              leace["controls"])
        result["X3-1-gain_specific"] = gain
        # carry the per-control ratios and the gain-specific flag in X3-1 itself
        for metric in ("log_yield", "energy"):
            entry = result["X3-1"].get(metric) or {}
            block = gain.get(metric) or {}
            entry["rho_controls"] = block.get("rho_controls")
            entry["rho_control_q95"] = block.get("rho_control_q95")
            entry["gain_specific"] = block.get("gain_specific")
            entry["gain_specific_conditions"] = {
                "control_q95": block.get("condition_q95"),
                "ci_excludes_zero": block.get("condition_ci_excludes_zero")}
    result["X3-1"]["classes"] = {
        metric: (result["X3-1"].get(metric) or {}).get("class")
        for metric in ("log_yield", "energy")}
    return result


def _compare_at_d(calibrated, invariant, metric) -> dict:
    """Calibrated minus invariant at D for one metric, with a run-cluster CI."""
    left = _metric_rows(calibrated, metric, "D")
    right = _metric_rows(invariant, metric, "D")
    if left is None or right is None:
        return {"status": "not available", "calibration_better": None,
                "invariance_better": None}
    difference = float(right["resolution"]) - float(left["resolution"])
    out = {"difference": difference, "ci95": None,
           "calibration_better": None, "invariance_better": None}
    aligned = _align_metric_pair(left, right)
    if aligned is not None:
        left_errors, right_errors, run_ids = aligned
        bootstrap = paired_cluster_bootstrap(left_errors, right_errors, run_ids,
                                             statistic=rms)
        out["ci95"] = bootstrap["ci95"]
        # difference = res_invariant - res_calibrated: positive means calibration
        # gives the lower (better) resolution
        out["calibration_better"] = bool(bootstrap["ci95"][0] > 0)
        out["invariance_better"] = bool(bootstrap["ci95"][1] < 0)
    return out


def _x3_robustness(standard, invariant) -> dict:
    """X3-3: robustness at D against the same variant's N reading (§7.2)."""
    out = {}
    for name, variant in (("standard", standard), ("invariant", invariant)):
        v_n = _metric_rows(variant, "vertex", "N")
        v_d = _metric_rows(variant, "vertex", "D")
        y_n = _metric_rows(variant, "log_yield", "N")
        y_d = _metric_rows(variant, "log_yield", "D")
        e_n = _metric_rows(variant, "energy", "N")
        e_d = _metric_rows(variant, "energy", "D")
        delta_v = (None if not (v_n and v_d) else float(v_d["resolution"])
                   - float(v_n["resolution"]))
        energy_change = (None if not (e_n and e_d) else
                         (float(e_d["resolution"]) - float(e_n["resolution"]))
                         / float(e_n["resolution"]) if e_n["resolution"] else None)
        out[name] = {"delta_vertex_m": delta_v,
                     "delta_log_yield": (None if not (y_n and y_d) else
                                         float(y_d["resolution"])
                                         - float(y_n["resolution"])),
                     "relative_energy_change": energy_change,
                     "robust": None if (delta_v is None or energy_change is None) else
                     bool(abs(delta_v) <= _X3_ROBUST_VERTEX_M
                          and abs(energy_change) <= _X3_ROBUST_ENERGY_REL)}
    return out


def _x3_dominance(calibrated, invariant) -> dict:
    """X3-4: calibrated against invariant at D, then the combined verdict (§7.2)."""
    if not calibrated:
        return {"status": "not available", "verdict": "not available"}
    out = {metric: _compare_at_d(calibrated, invariant, metric)
           for metric in ("vertex", "energy", "log_yield")}
    vertex, energy = out["vertex"], out["energy"]
    evaluated = vertex.get("status") != "not available" \
        and energy.get("status") != "not available"
    calibration = bool(vertex.get("calibration_better")
                       and energy.get("calibration_better"))
    invariance = bool(vertex.get("invariance_better")
                      and energy.get("invariance_better"))
    out["verdict"] = ("calibration dominates" if calibration else
                      "invariance dominates" if invariance else
                      "mixed" if evaluated else "not available")
    out["status"] = "evaluated" if evaluated else "not available"
    return out


def _gain_specific(x3_1, standard, invariant, controls) -> dict:
    """§7.2 gain-specific flag: both the control-q95 and the control-mean CI conditions."""
    out = {}
    for metric in ("log_yield", "energy"):
        entry = x3_1.get(metric) or {}
        reference = _metric_rows(standard, metric, "N")
        leace = _metric_rows(invariant, metric, "N")
        if entry.get("rho_E") is None or reference is None or leace is None:
            out[metric] = {"status": "not available", "gain_specific": None}
            continue
        standard_resolution = float(reference["resolution"])
        ratios, control_errors = [], []
        aligned_leace = None
        for control in controls:
            rows = _metric_rows(control, metric, "N")
            if rows is None or standard_resolution <= 0:
                continue
            ratios.append(float(rows["resolution"]) / standard_resolution)
            aligned = _align_metric_pair(leace, rows)
            if aligned is not None:
                aligned_leace = (aligned[0], aligned[2])
                control_errors.append(aligned[1])
        q95 = float(np.quantile(ratios, 0.95)) if ratios else None
        condition_q95 = bool(q95 is not None and entry["rho_E"] > q95)
        if not control_errors or aligned_leace is None:
            out[metric] = {"status": "not available (CI missing)",
                           "n_controls": len(ratios), "rho_control_q95": q95,
                           "rho_controls": ratios, "ci95": None,
                           "condition_q95": condition_q95,
                           "condition_ci_excludes_zero": None,
                           "gain_specific": None}
            continue
        mean_control = np.mean(np.stack(control_errors, axis=0), axis=0)
        bootstrap = paired_cluster_bootstrap(aligned_leace[0], mean_control,
                                             aligned_leace[1], statistic=rms)
        ci95 = bootstrap["ci95"]
        condition_ci = bool(bootstrap["ci95"][0] > 0.0
                            or bootstrap["ci95"][1] < 0.0)
        out[metric] = {"status": "evaluated", "n_controls": len(ratios),
                       "rho_control_q95": q95, "rho_controls": ratios,
                       "ci95": ci95, "condition_q95": condition_q95,
                       "condition_ci_excludes_zero": condition_ci,
                       "gain_specific": bool(condition_q95 and condition_ci)}
    return out


def _x3_five(fisher, variants) -> dict:
    """X3-5: empirical ρ_E(log-yield) next to √(CRLB_inv/CRLB_known) for ln Y."""
    rows = {}
    models = (fisher.get("models") or {})
    for name in ("C-mean", "C-mean+occupancy", "C-var_all", "C-var_ge1", "C-var_ge3"):
        model = models.get(name) or {}
        known = (model.get("median_crlb") or {}).get("known")
        inv = (model.get("median_crlb") or {}).get("inv")
        ratio = None
        if known and inv and known[3] > 0:
            ratio = (float("inf") if np.isinf(inv[3]) else
                     float(math.sqrt(inv[3] / known[3])))
        rows[name] = {"theory_ratio_lnY": ratio}
    empirical = None
    for key, variant in (variants or {}).items():
        entry = (variant.get("X3-1") or {}).get("log_yield") or {}
        if entry.get("rho_E") is not None:
            empirical = entry["rho_E"]
            break
    return {"empirical_rho_E_log_yield": empirical, "theory_rows": rows,
            "rule": "C-mean alone gives ∞ by P0 and is listed as the charge-only "
                    "limit, not as a comparison point"}


# ================================================================ setup paths
def _mode_setup_objects(mode, *, root, config, gen, arms_config, geometry, data_root,
                        ext_root, inventory=None, probes=None, particle_truth=None,
                        build_model=False, limit=None, log=print) -> dict:
    """The shared setup path of one CLI mode (no output is computed here)."""
    root = _resolve_root(root)
    setup = {"mode": mode, "root": str(root), "data_root": str(data_root),
             "ext_root": str(ext_root), "objects": {
                 "geometry": str(getattr(geometry, "name", geometry)),
                 "geometry_digest": getattr(geometry, "digest", None),
                 "n_pmt": int(geometry.n_pmt),
                 "n_modules": int(geometry.n_modules)},
             "plan": {}}
    setup["data"] = data_status(data_root)
    setup["inventory"] = inventory if inventory is not None else part_a_inventory(
        root, hash_files=False, log=lambda *a, **k: None)
    setup["inventory_counts"] = setup["inventory"]["counts"]
    if probes is None and mode != "report":
        probes = {"s005": open_probe_sets(config, gen, geometry, data_root,
                                          setting=S005, log=lambda *a, **k: None),
                  "s030": open_probe_sets(config, gen, geometry, data_root,
                                          setting=S030, log=lambda *a, **k: None)}
    setup["probes"] = {} if probes is None else {
        setting: {role: dataset.key for role, dataset in value.items()}
        for setting, value in probes.items()}
    if probes is not None:
        if mode in ("x1", "dry-run"):
            names = condition_names_x1()
            setup["conditions"] = {
                role: {name: condition.as_dict()
                       for name, condition in build_conditions(
                           {"s005": probes["s005"][role],
                            "s030": probes["s030"][role]}, geometry, names).items()}
                for role in ("train", "test")}
            setup["plan"]["photon_sets_per_model"] = X1_PHOTON_SETS_PER_MODEL
            setup["plan"]["particle_models"] = list(X1_PARTICLE_MODELS)
        if mode in ("x2", "dry-run"):
            conditions = {role: build_conditions(
                {"s005": probes["s005"][role], "s030": probes["s030"][role]},
                geometry, ("N", "D", "S")) for role in ("train", "test")}
            delta = run_log_gain_vectors(probes["s030"]["train"], geometry)
            counts = occupancy_counts(probes["s030"]["test"],
                                      conditions["test"]["D"], geometry)
            setup["plan"]["x2"] = {"n_conditions": 3, "n_train_runs": len(delta),
                                   "n_test_events": len(counts)}
            if mode == "dry-run":
                # exercise the equivariance fit on the first few events (CPU only)
                runs = sorted({key[0] for key in counts})
                if len(runs) >= 5:
                    train_keys = [key for key in counts if key[0] == runs[0]]
                    test_keys = [key for key in counts if key[0] == runs[1]]
                    design = global_design(
                        {"run_ids": np.asarray([key[0] for key in
                                                train_keys + test_keys], dtype=object),
                         "event_ids": np.asarray([key[1] for key in
                                                  train_keys + test_keys], np.int64)},
                        {run: delta[run] for run in runs}, geometry)
                    target = np.zeros((len(train_keys) + len(test_keys),
                                       int(geometry.n_modules)))
                    fit = equivariance_fit(
                        design[:len(train_keys)], target[:len(train_keys)],
                        design[len(train_keys):], target[len(train_keys):],
                        run_ids_train=np.asarray([key[0] for key in train_keys]),
                        run_ids_test=np.asarray([key[0] for key in test_keys]),
                        kind="global")
                    setup["plan"]["x2"]["probe_fit_r2"] = fit["r2_eq"]
                else:
                    setup["plan"]["x2"]["probe_fit_r2"] = None
                    setup["plan"]["x2"]["probe_fit_note"] = (
                        "sample fit needs >= 5 run-grouped folds")
        if mode in ("x3-fisher", "dry-run"):
            fisher = fisher_inputs(geometry, gen, n_events=2)
            blocks = fisher_blocks(fisher[0]["mu"], fisher[0]["dmu"], fisher[0]["mod"],
                                   model="C-mean")
            report = crlb_report(blocks)
            setup["plan"]["x3_fisher"] = {
                "n_events_planned": 1000, "sample_p0": p0_check(blocks)["pass"],
                "sample_known_lnY_crlb": report["known"]["crlb"][3],
                "sample_sigma90_lnY": report["cal"][
                    f"{FISHER_SIGMAS[-1]:g}"] is not None}
        if mode in ("x3", "dry-run") and mode == "dry-run":
            rng = np.random.default_rng(20261106)
            x = rng.standard_normal((40, 16))
            z = rng.standard_normal((40, 1)) @ np.ones((1, 16))
            eraser = leace_fit(x, z, rank=1)
            residual = leace_cross_covariance(eraser, x, z)
            setup["plan"]["x3"] = {"n_families": 3,
                                   "leace_sample_relative": residual["relative"],
                                   "leace_idempotent": bool(np.allclose(
                                       leace_erase(eraser, leace_erase(eraser, x)),
                                       leace_erase(eraser, x), atol=1e-12))}
        if mode in ("reproduce", "preflight", "dry-run"):
            setup["plan"]["x0b_models"] = list(X0B_TARGETS)
            setup["plan"]["projection"] = gpu_projection()
    if mode in ("x1", "x3", "dry-run"):
        setup["particle_truth"] = (
            particle_truth if particle_truth is not None else
            {role: ef.load_particle_truth(data_root, PROBE_KEYS_S005[role])
             for role in ("train", "test")})
    if build_model:
        row = (setup["inventory"]["rows"].get("init9000")
               if setup["inventory"] is not None else None)
        if row is not None:
            with ef.item_eval_context(row):
                arm = ef.build_item_arm(row, config=config, gen=gen,
                                        arms_config=arms_config, geometry=geometry,
                                        data_root=data_root, device="cpu", log=log)
            setup["plan"]["init_arm"] = {"arm": row.get("arm"),
                                         "d_model": int(getattr(arm, "d_model", 0) or 0)}
        setup["plan"]["entries"] = entries_preflight(log=log)
    setup["plan"]["projection"] = setup["plan"].get("projection") or gpu_projection()
    setup["plan"]["cpu_projection"] = cpu_projection()
    setup["plan"]["s1"] = s1_projection(root)
    return setup


def mode_setup(mode, *, root=None, config, gen, arms_config, geometry,
               data_root=DATA_ROOT_DEFAULT, ext_root=EXT_ROOT, dry=None,
               **kwargs) -> dict:
    """The gated setup path of one CLI mode."""
    root = _resolve_root(root)
    gate = assert_authorised(mode, root=root, dry_run=dry)
    setup = _mode_setup_objects(mode, root=root, config=config, gen=gen,
                                arms_config=arms_config, geometry=geometry,
                                data_root=data_root, ext_root=ext_root, **kwargs)
    setup["gate"] = {"status": gate["status"], "authorised": gate["authorised"]}
    return setup


# ================================================================== report
def _read_stage(ext_root, relative):
    return read_json(Path(ext_root) / relative, None)


def large_arrays(ext_root, *, min_bytes=5 * 1024 * 1024) -> dict:
    """LARGE_ARRAYS.json: sha256 of every cached embedding file > 5 MB (§9.6)."""
    root = Path(ext_root)
    out = {"threshold_bytes": int(min_bytes), "files": {}}
    if not root.is_dir():
        return out
    for manifest in sorted(root.glob("*/manifest.json")) + sorted(
            root.glob("*/*/manifest.json")):
        payload = read_json(manifest, {}) or {}
        for key, row in payload.items():
            path = Path(row.get("path", ""))
            if not path.is_file() or path.stat().st_size <= int(min_bytes):
                continue
            digest = lr.sha256_file(path)
            out["files"][str(path)] = {"sha256": digest,
                                       "bytes": int(path.stat().st_size),
                                       "label": manifest.parent.name, "key": key}
    out["n_files"] = len(out["files"])
    return out


def _figure_names() -> list:
    return ["drift_ladder.png", "x1g_residual_split.png", "rho_by_repair.png",
            "r2_eq.png", "identification_error.png", "fisher_crlb_ratio.png",
            "price_of_invariance.png"]


def _plot(figure_path, draw):
    """Draw one §9.6 figure; a failure is recorded, never raised."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(figsize=(7, 4))
        draw(axes)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=120)
        plt.close(figure)
        return True
    except Exception:  # pragma: no cover - matplotlib is optional
        try:
            plt.close("all")
        except Exception:
            pass
        return False


def render_figures(readings, out_dir, *, x1_payload=None, x2_payload=None,
                   x3_payload=None, fisher_payload=None, log=print) -> dict:
    """Write the seven §9.6 figures from the readings; skipped without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception as problem:  # pragma: no cover - matplotlib is optional
        return {"status": "unavailable", "reason": str(problem), "figures": []}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x1 = (readings.get("x1") or {}).get("families") or {}
    x3 = (readings.get("x3") or {}).get("variants") or {}
    written, skipped = [], []

    def family_rows():
        rows = []
        for family, block in sorted(x1.items()):
            for seed, row in sorted((block.get("seeds") or {}).items()):
                rows.append((family, seed, row))
        return rows

    def draw_drift_ladder(axes):
        values = {}
        for model in ((x1_payload or {}).get("models") or {}).values():
            conditions = model.get("conditions") or {}
            for condition in ("N", "D", "SWtw", "SWunit", "HW"):
                entry = (conditions.get(condition) or {}).get("vertex") or {}
                if entry.get("mean_error_m") is not None:
                    values.setdefault(condition, []).append(float(entry["mean_error_m"]))
        labels = [name for name in ("N", "D", "SWtw", "SWunit", "HW") if values.get(name)]
        axes.boxplot([values[name] for name in labels], labels=labels)
        axes.set_ylabel("vertex mean error (m)")

    def draw_rho(axes):
        rows = family_rows()
        labels = [f"{family}/{seed}" for family, seed, _row in rows]
        width = 0.38
        index = np.arange(len(rows))
        for offset, key in (("-w", "rho_swtw"), ("+w", "rho_swunit")):
            values = [((row.get("X1-b") or {}).get("rho") if key == "rho_swtw"
                       else None) for _f, _s, row in rows]
            axes.bar(index + (width if offset == "+w" else 0), values, width=width,
                     label=key)
        axes.set_xticks(index + width / 2)
        axes.set_xticklabels(labels, rotation=90, fontsize=6)
        axes.set_ylabel("recovered fraction")
        axes.legend()

    def draw_r2(axes):
        names, values = [], []
        for model in ((x2_payload or {}).get("models") or {}).values():
            names.append(f"{model.get('family')}{model.get('seed')}")
            values.append(((model.get("X2-b") or {}).get("occupancy") or {}).get("r2_cs"))
            axes.set_ylabel("R²_eq")
        axes.bar(np.arange(len(names)), [value or 0.0 for value in values])
        axes.set_xticks(np.arange(len(names)))
        axes.set_xticklabels(names, rotation=90, fontsize=6)

    def draw_identification(axes):
        names, errors = [], {}
        for model in ((x2_payload or {}).get("models") or {}).values():
            name = f"{model.get('family')}{model.get('seed')}"
            names.append(name)
            for stream, block in (model.get("X2-d") or {}).items():
                if isinstance(block, dict) and block.get("median_run_rms") is not None:
                    errors.setdefault(stream, []).append(float(block["median_run_rms"]))
        index = np.arange(len(names))
        for offset, (stream, values) in enumerate(sorted(errors.items())):
            axes.bar(index + 0.8 * offset / max(len(errors), 1), values, width=0.8)
        axes.set_ylabel("median run RMS error")

    def draw_fisher(axes):
        models = (fisher_payload or {}).get("models") or {}
        for name, block in sorted(models.items()):
            ratios = (block.get("median_crlb") or {})
            known = np.asarray(ratios.get("known") or [], float)
            inv = np.asarray(ratios.get("inv") or [], float)
            if known.size == 4 and inv.size == 4:
                with np.errstate(divide="ignore", invalid="ignore"):
                    axes.plot(np.arange(4), inv / known, marker="o", label=name)
        axes.set_xticks(np.arange(4))
        axes.set_xticklabels(THETA_NAMES)
        axes.set_ylabel("median CRLB ratio (inv/known)")
        axes.legend(fontsize=6)

    def draw_price(axes):
        names, values = [], []
        for name, variant in sorted(x3.items()):
            entry = (variant.get("X3-1") or {}).get("log_yield") or {}
            if entry.get("rho_E") is not None:
                names.append(name)
                values.append(float(entry["rho_E"]))
        axes.bar(np.arange(len(names)), values)
        axes.axhline(1.0, color="k", linewidth=0.8)
        axes.set_xticks(np.arange(len(names)))
        axes.set_xticklabels(names, rotation=90, fontsize=6)
        axes.set_ylabel("ρ_E (log-yield)")

    plan = {
        "drift_ladder.png": draw_drift_ladder,
        "x1g_residual_split.png": lambda axes: _plot_residual_split(axes, family_rows),
        "rho_by_repair.png": draw_rho,
        "r2_eq.png": draw_r2,
        "identification_error.png": draw_identification,
        "fisher_crlb_ratio.png": draw_fisher,
        "price_of_invariance.png": draw_price,
    }
    for _name, draw in plan.items():
        path = out_dir / _name
        if _plot(path, draw):
            written.append(str(path))
        else:
            skipped.append(_name)
    return {"status": "evaluated", "figures": written, "n_figures": len(written),
            "skipped": skipped}


def _plot_residual_split(axes, family_rows):
    rows = family_rows()
    names = [f"{family}/{seed}" for family, seed, _row in rows]
    r_cs = [((row.get("X1-g") or {}).get("R_cs_m") or 0.0) for _f, _s, row in rows]
    r_ni = [((row.get("X1-g") or {}).get("R_ni_m") or 0.0) for _f, _s, row in rows]
    index = np.arange(len(rows))
    axes.bar(index, r_cs, label="R_cs (charge-scale)")
    axes.bar(index, r_ni, bottom=r_cs, label="R_ni (non-invertible)")
    axes.set_xticks(index)
    axes.set_xticklabels(names, rotation=90, fontsize=6)
    axes.set_ylabel("residual (m)")
    axes.legend()


def assemble_report(*, root=None, ext_root=EXT_ROOT, results_dir=None, log=print) -> dict:
    """Merge the stage outputs into ``results/jepa/20261106_repair/`` (§9.6)."""
    root = _resolve_root(root)
    gate = assert_authorised("report", root=root, log=log)
    ext_root = Path(ext_root)
    target = Path(results_dir) if results_dir is not None else root / RESULTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    plan = _read_stage(ext_root, "x0/plan.json")
    inventory = _read_stage(ext_root, "x0/inventory.json") or (
        plan.get("inventory") if plan else None)
    reproduction = _read_stage(ext_root, "x0/reproduction.json")
    x1 = _read_stage(ext_root, "x1/x1.json")
    x2 = _read_stage(ext_root, "x2/x2.json")
    x3_fisher = _read_stage(ext_root, "x3/x3_fisher.json")
    x3 = _read_stage(ext_root, "x3/x3.json")
    x1_complete = bool((x1 or {}).get("complete"))
    x2_complete = bool((x2 or {}).get("complete"))
    x3_complete = bool((x3 or {}).get("complete"))
    readings = {"registered": False, "extension": EXTENSION,
                "check": "repair-pilot readings", "reproduction":
                (reproduction or {}).get("pass"),
                "x1": {"complete": x1_complete, "completed": x1_complete,
                       **((x1 or {}).get("readings") or {})},
                "x2": {"complete": x2_complete, "completed": x2_complete,
                       **((x2 or {}).get("readings") or {})},
                "x3": {"complete": x3_complete, "completed": x3_complete,
                       **((x3 or {}).get("readings") or {})},
                "plan": {"pass": (plan or {}).get("pass")},
                "x3_fisher_p0": ((x3_fisher or {}).get("p0"))}
    large = large_arrays(ext_root)
    figures = render_figures(readings, target, x1_payload=x1, x2_payload=x2,
                             x3_payload=x3, fisher_payload=x3_fisher, log=log)
    write_json(target / "plan.json", plan) if plan else None
    write_json(target / "inventory.json", inventory) if inventory else None
    write_json(target / "reproduction.json", reproduction) if reproduction else None
    write_json(target / "x1.json", x1) if x1 else None
    write_json(target / "x2.json", x2) if x2 else None
    write_json(target / "x3_fisher.json", x3_fisher) if x3_fisher else None
    write_json(target / "x3.json", x3) if x3 else None
    write_json(target / "readings.json", readings)
    write_json(target / "LARGE_ARRAYS.json", large)
    readme = _report_readme(readings, reproduction or {}, figures)
    (target / "README.md").write_text(readme)
    payload = {"registered": False, "extension": EXTENSION, "check": "report §9.6",
               "path": str(target), "readings": readings, "figures": figures,
               "large_arrays": {"n_files": large["n_files"]},
               "reproduction_pass": (reproduction or {}).get("pass"),
               "complete": bool(plan is not None and reproduction is not None
                                and x1_complete and x2_complete and x3_complete
                                and x3_fisher is not None),
               "completeness": {"x1": x1_complete, "x2": x2_complete,
                                "x3": x3_complete,
                                "reproduction": reproduction is not None,
                                "fisher": x3_fisher is not None,
                                "plan": plan is not None},
               "too_big": False, "gate": {"status": gate["status"]}}
    payload["closure"] = _assert_closure("report", root)
    return payload


def _report_readme(readings, reproduction, figures) -> str:
    lines = ["# Paper 3 repair pilot (Part A) — readings", "",
             f"Extension `{EXTENSION}`; runbook `{RUNBOOK_REL}`; `registered: false`.",
             "", f"X0-b reproduction pass: `{reproduction.get('pass')}`.",
             "Figures: " + ", ".join(f"`{name}`" for name in _figure_names()) + ".",
             "", "## X1 — what does calibration recover?", ""]
    x1 = (readings.get("x1") or {}).get("families") or {}
    lines += ["| family | seed | Δ_D (m) | ρ_SWtw | X1-b class | share_cs | X1-g class |",
              "|---|---|---|---|---|---|---|"]
    for family, block in sorted(x1.items()):
        for seed, row in sorted((block.get("seeds") or {}).items()):
            a = row.get("X1-a") or {}
            b = row.get("X1-b") or {}
            g = row.get("X1-g") or {}
            lines.append(f"| {family} | {seed} | {_fmt(a.get('delta_D_m'))} | "
                         f"{_fmt(b.get('rho'))} | {b.get('class')} | "
                         f"{_fmt(g.get('share_cs'))} | {g.get('reading')} |")
    lines += ["", "## X2 — equivariance, latent repair, identification", ""]
    x2 = (readings.get("x2") or {}).get("families") or {}
    lines += ["| family | X2-b class | ρ_LR | X2-c class |", "|---|---|---|---|"]
    for family, block in sorted(x2.items()):
        lines.append(f"| {family} | "
                     f"{(block.get('classes') or {}).get('X2-b', {}).get('class')} | "
                     f"{block.get('seeds')} | "
                     f"{(block.get('classes') or {}).get('X2-c', {}).get('class')} |")
    lines += ["", "## X3 — the price of invariance", ""]
    x3 = (readings.get("x3") or {}).get("variants") or {}
    lines += ["| variant | ρ_E (log-yield) | class (log-yield) | ρ_E (energy) | "
              "class (energy) | gain-specific | Δv (m) |",
              "|---|---|---|---|---|---|---|"]
    for name, variant in sorted(x3.items()):
        price = variant.get("X3-1") or {}
        yield_entry = price.get("log_yield") or {}
        energy_entry = price.get("energy") or {}
        gain = (yield_entry.get("gain_specific")
                if yield_entry.get("gain_specific") is not None
                else energy_entry.get("gain_specific"))
        lines.append(f"| {name} | {_fmt(yield_entry.get('rho_E'))} | "
                     f"{yield_entry.get('class')} | "
                     f"{_fmt(energy_entry.get('rho_E'))} | "
                     f"{energy_entry.get('class')} | {_fmt(gain)} | "
                     f"{_fmt((variant.get('X3-2') or {}).get('difference_m'))} |")
    fisher = (readings.get("x3") or {}).get("X3-5") or {}
    if fisher:
        lines += ["", "## X3-5 — theory beside experiment (ln Y)", "",
                  f"Empirical ρ_E (log-yield): `{_fmt(fisher.get('empirical_rho_E_log_yield'))}`"]
        for name, row in sorted((fisher.get("theory_rows") or {}).items()):
            lines.append(f"- {name}: √(CRLB_inv/CRLB_known) = "
                         f"`{_fmt(row.get('theory_ratio_lnY'))}`")
    lines += ["", "## Budget and stops", "",
              "See `plan.json` for the GPU/CPU projections, the combined S1 rule and the "
              "free-entry preflight. Per-readings tables: `readings.json`."]
    return "\n".join(lines) + "\n"


def _fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "∞"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)
