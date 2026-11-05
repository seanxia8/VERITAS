# SPDX-License-Identifier: MIT
"""MAE/JEPA follow-up extension (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md`` (DRAFT v7,
3 Nov 2026; sha256 ``063e9087...``).  Extension name
``mae_jepa_followup_2026-11-03``.

Stage A is five frozen-model readouts (A-1, A-1b, A-2, A-3, A-4) with **no
training**; T-1 is the author-authorised MAE strong-drift training control
(four runs + one 20-step smoke, §4).  This module is the library the CLI
``scripts/jepa_ext_followup.py`` drives; it never edits or monkeypatches a
registered module.  Every data access is read-only: the pinned manifests are
hash-checked *before* any ``lucid_run.ensure_dataset`` call, the s = 0.30 probe
sets are opened only through their ``registered=False`` keys, and the two
s = 0.05 loaders that hard-code ``registered=True`` are never used for them.

Sections
--------
* gate (§6.2 (a)-(e)) and the executed-import closure of the dry run;
* pinned data, read-only manifest checks and the read-only dataset openers;
* §2.1 checkpoint inventory and the §3 A0 plan/projection/pairing check;
* §3 A1 reproduction gate;
* the shared frozen-encoder streaming core and per-item evaluation;
* §3 A-1, A-1b, A-2, A-3, A-4 readout bodies;
* §4 T-1 cells, the exact ``mode_train`` call sequence and the host manifest
  check;
* the report/readings assembly.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import ext_arch as ea
from . import ext_dejitter as ed
from . import ext_h12 as eh
from . import ext_readout as er
from . import lucid_arms as la
from . import lucid_bombs as lb
from . import lucid_run as lr
from . import lucid_views as lv
from . import point_encoder as pe

__all__ = [
    "EXTENSION", "MODES", "READOUT_ITEMS", "STAGE_A_ORDER", "EXT_ROOT", "SMOKE_ROOT",
    "TRAIN_ROOT", "STAGE_A_ROOT", "RESULTS_DIR", "RUNBOOK_REL", "RUNBOOK_SHA256",
    "ACCEPTANCE_REL", "DECISIONS_REL", "AUTHOR_LINE", "STANDING_RULE", "PINNED_DATA",
    "T1_VARIANTS", "T1_SEEDS", "T1_STEPS", "T1_SMOKE_STEPS", "T1_SMOKE",
    "T1_ALLOWED", "T1_ALLOWED_DIFFERENCES", "section_zero", "assert_authorised",
    "import_closure", "data_status", "require_pinned", "open_probe_sets",
    "build_inventory", "pairing_check", "stage_a_projection", "build_plan",
    "reproduction", "evaluate_item_row", "readout_a1", "readout_a1b", "readout_a2",
    "readout_a3", "readout_a4", "t1_cell", "t1_run_kwargs", "train_t1",
    "t1_host_manifest_check", "assemble_report", "pair_events", "context_time_shift",
    "apply_time_shift", "membership_enumeration",
]

# ---------------------------------------------------------------- extension
EXTENSION = "mae_jepa_followup_2026-11-03"
MODES = ("dry-run", "inventory", "reproduce", "readout", "train-t1", "report")
READOUT_ITEMS = ("A-1", "A-1b", "A-2", "A-3", "A-4")
#: §6.5 fixed execution order (T-1 may start once both A1 s = 0.30 rows pass).
STAGE_A_ORDER = ("A0", "A1", "A-1", "A-1b", "A-2", "A-4", "A-3")

ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_followup_20261103")
SMOKE_ROOT = EXT_ROOT / "smoke"
TRAIN_ROOT = EXT_ROOT / "train"
STAGE_A_ROOT = EXT_ROOT / "stageA"
RESULTS_DIR = "results/jepa/20261103_followup"
JOBS_DIR = Path("/ceph/dwong/jepa/jobs")
GUARD_JSON = JOBS_DIR / "guard_deepthought2.json"
READOUT_EXT = Path("/ceph/dwong/jepa/returns/ext_readout_20261101")
DEJITTER_EXT = Path("/ceph/dwong/jepa/returns/ext_dejitter_20261030")
MAE_DEJITTER_EXT = Path("/ceph/dwong/jepa/returns/ext_mae_dejitter_20261031")
H12_EXT = Path("/ceph/dwong/jepa/returns/ext_h12_20261101")
STATIC_EXT = Path("/ceph/dwong/jepa/returns/ext_static_20261102")
SCALE_EXT = Path("/ceph/dwong/jepa/returns/ext_scale_20261102")
ARCH_EXT = Path("/ceph/dwong/jepa/returns/ext_arch_20261102")
VIEW_EXT = Path("/ceph/dwong/jepa/returns/ext_view_20261103")

RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md"
RUNBOOK_SHA256 = "063e908774ffe29b3d59ef40ab54ef5d02ff462b6aa1e49eef57c61c49e7af24"
ACCEPTANCE_REL = "docs/jepa/ACCEPTANCE_2026-11-03_MAE_JEPA_FOLLOWUP.json"
ACCEPTANCE_KEYS = ("runbook_sha256", "impl_commit", "review_path", "review_verdict",
                   "lead", "date")
DECISIONS_REL = "docs/jepa/DECISIONS.md"
AUTHOR_LINE = "我同意T-1训练"            # runbook §0 line 1, verbatim
STANDING_RULE = "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

#: §6.2 (d): the reviewed path set.  ``git diff --quiet <impl_commit> -- <P>`` and
#: ``git ls-files --others --exclude-standard -- <P>`` must both be empty.
P_FILES = (
    "src/jepa_noise/__init__.py", "src/jepa_noise/ext_followup.py",
    "src/jepa_noise/ext_readout.py", "src/jepa_noise/ext_h12.py",
    "src/jepa_noise/ext_dejitter.py", "src/jepa_noise/ext_arch.py",
    "src/jepa_noise/lucid_run.py", "src/jepa_noise/lucid_arms.py",
    "src/jepa_noise/lucid_bombs.py", "src/jepa_noise/lucid_views.py",
    "src/jepa_noise/point_encoder.py", "src/jepa_noise/bridge_run.py",
    "src/jepa_noise/train.py", "src/jepa_noise/crlb.py", "src/jepa_noise/jepa.py",
    "src/jepa_noise/backbone.py", "src/jepa_noise/linear.py",
    "src/jepa_noise/probes.py", "src/jepa_noise/sigreg.py",
    "scripts/__init__.py", "scripts/jepa_ext_followup.py",
    "scripts/jepa_pregrid_checks.py", "scripts/jepa_ext_dejitter.py",
    "scripts/jepa_ext_readout.py", RUNBOOK_REL,
)
P_DIRS = ("src/noise_module_lucid/", "src/jepa_noise/configs/")
#: Modules outside <P> that a <P> module *forces* into the executed import
#: closure and that this implementation can never edit:
#: * ``lucid_run`` imports ``bridge_run`` (in <P>), which imports
#:   ``train``/``jepa``/``backbone`` (in <P>), which import the repository's
#:   no-touch ``reconstruction_model`` package;
#: * ``src/noise_module_lucid/`` (in <P>) imports the underlying
#:   ``noise_module`` package at module level (``dataset.py``,
#:   ``interventions.py`` and the lazy readout dependencies).
#: Runbook §6.2 (d) lists the importers but not these tracked dependencies;
#: the dry run records them separately as forced, read-only imports and does
#: not count them as violations of the implemented path set.
FORCED_IMPORT_EXCEPTIONS = ("src/reconstruction_model/", "src/noise_module/")

# --------------------------------------------------------------- provenance
CONFIG_REL = "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GEN_REL = "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
ARMS_REL = "src/jepa_noise/configs/lucid_arms_v0.json"
CONFIG_SHA256 = "e6730ac14ddf475b857a0f8b07407f8b54e487e2e7af827fd00bcecc5268e034"
GEN_SHA256 = "90ac1601c40171eeb7c669a08df1fe6e6b5b332fee2a3cca689ae057a51dfe88"
GEOMETRY_DEFAULT = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT_DEFAULT = "/ceph/dwong/jepa/data/ext_pregrid_dev_20261025"

#: §2 pinned data (sha256 of each dataset's ``manifest.json``).
PINNED_DATA = {
    "dev_probe_train__F1__0.05__s20261017":
        "093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3",
    "dev_probe_test__F1__0.05__s20261017":
        "c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a",
    "dev_train__F1__0.05__s20261017__n100000x1000":
        "b1b3f4f401bcee48c49c347b67d7f5cbd4410213f08ef8b9d2848edc9f2cd13a",
    "dev_probe_train__F1__0.3__s20261017__n10000x50":
        "6df84887b3469723da3ddb0646c292ab655b263d6063f5fd713432a53514bb46",
    "dev_probe_test__F1__0.3__s20261017__n10000x50":
        "c68d83d1ef8cb202f3937f7cd48e3f74bd94f6d44ae203591053fde0605d1be2",
    "dev_train__F1__0.3__s20261017__n100000x1000":
        "37e727cc6fe402396c9091209a47e6dcfd3be6ed6f756bd38fb75ef10f88c75d",
    "dev_train_300k__F1__0.05__s20261017__n300000x1000":
        "9a9dbe81ba63cea1c2ccad27e3d5cce7df9cb6e99faf0a9b340c6cec18c23a4a",
    "dark_noise__s20261017__n512":
        "fff2f4b0d4a51ddef8229442933f7474c31d1c0953796ebf434ea5be664b016b",
}
PROBE_KEYS_S005 = {"train": "dev_probe_train__F1__0.05__s20261017",
                   "test": "dev_probe_test__F1__0.05__s20261017"}
PROBE_KEYS_S030 = {"train": "dev_probe_train__F1__0.3__s20261017__n10000x50",
                   "test": "dev_probe_test__F1__0.3__s20261017__n10000x50"}
TRAIN_KEY_S030 = "dev_train__F1__0.3__s20261017__n100000x1000"
TRAIN_KEY_S005 = "dev_train__F1__0.05__s20261017__n100000x1000"
TRAIN_KEY_300K = "dev_train_300k__F1__0.05__s20261017__n300000x1000"
DARK_BANK_KEY = "dark_noise__s20261017__n512"
S005, S030 = 0.05, 0.30

#: §3 A1 committed reproduction rows (tolerance 1e-3 m / 1e-3 absolute R²).
A1_S005 = {
    "init9000": 0.160072,
    "fix9000": 0.219258,
    "mae9000": 0.104949,
    "sigreg_within_median9000": 0.222653,
    "mae_within_trigger9000": 0.104916,
}
A1_S030 = {
    "init9000": {"vertex_resolution_m": 0.170301, "module_gain_r2": 0.841585},
    "strong_drift9000": {"vertex_resolution_m": 0.233209, "module_gain_r2": 0.097482},
}
A1_TOLERANCE = 1e-3
#: §3 A-2 module-gain floors (one per environment; both probe splits).
MODULE_GAIN_FLOOR = {S005: 0.200498, S030: 0.841585}
#: §3 A-1b reproduction target: fix9000 C-group ridge R² (committed 0.1043).
A1B_REPRODUCTION_R2 = 0.10429278172645076
A1B_REPRODUCTION_TOLERANCE = 1e-3
A1B_LABEL_CACHE = H12_EXT / "e0e1_rows.npz"
A1B_LABEL_FIX_SHA256 = "2d912a2deef7eb74137405062c0bb55d82ad3d1cf97d4b37e02ba381088ef714"
A1B_LABEL_EVENTS = 2000
#: §3 A-4 reading threshold (m) and the pinned context-only conventions.
A4_VERTEX_FLOOR_M = 0.01
A4_TOKEN_MEDIAN = "median"
A4_TOKEN_TRIGGER = "trigger"

#: §3 A-2 bootstrap.
A2_BOOTSTRAP_RESAMPLES = 1000
A2_BOOTSTRAP_SEED = 20261103
#: §3 A-3 the 500-contained-event rule.
A3_MIN_CONTAINED = 500
A3_MIN_EVENTS = 10

#: §5 caps.
STAGE_A_CAP_H = 5.5
T1_CAP_H = 3.5
CPU_CAP_H = 8.0
SECONDS_PER_ITEM = 126.0
PROJECTION_FACTOR = 1.3
ENTRIES_CLONE = 3059
ENTRIES_SUITE = 1825
ENTRIES_EXT = 1000
ENTRIES_FLOOR = 30000
S1_VIEW_CAP_H = 8.0
S1_PROJECTION_H = 9.0
S1_CEILING_H = 160.0

#: §4 T-1 launch table.
T1_ARM = "mae_mse"
T1_VARIANTS = ("s030_within_trigger", "s030_across_trigger")
T1_SEEDS = (9000, 9001)
T1_STEPS = 10000
T1_SMOKE_STEPS = 20
T1_SMOKE = (T1_ARM, "s030_within_trigger", 9000, T1_SMOKE_STEPS)
T1_CELL_IDS = {"s030_within_trigger": "ext_dev_s030_within_trigger",
               "s030_across_trigger": "ext_dev_s030_across_trigger"}
T1_ALLOWED = frozenset(
    {(T1_ARM, variant, seed, T1_STEPS) for variant in T1_VARIANTS for seed in T1_SEEDS}
    | {T1_SMOKE})
#: §4 host check: the rows may differ only in this set.
T1_ALLOWED_DIFFERENCES = frozenset({
    "run_id", "cell_id", "setting", "s_arm", "loader_dataset",
    "loader_dataset_manifest_sha256", "probe_datasets", "probe_dataset_sha256",
    "code_sha", "git_status", "clean_tree", "host", "device", "started_at_utc",
    "finished_at_utc", "wall_seconds", "final_checkpoint_sha256", "probes", "physics",
    "probe_files",
})
T1_SMOKE_EXTRA_DIFFERENCES = frozenset({"steps", "steps_completed", "final_step"})
T1_REQUIRED_LOADER_SHA = PINNED_DATA[TRAIN_KEY_S030]
T1_REQUIRED_PROBE_SHAS = {PINNED_DATA[PROBE_KEYS_S030["train"]],
                          PINNED_DATA[PROBE_KEYS_S030["test"]]}
T1_REFERENCE_MANIFEST = MAE_DEJITTER_EXT / "train" / "manifest.jsonl"

#: §2.1 evaluation configuration of each family.
_ARCH_VARIANTS = {"window1289000": "window128", "window1289001": "window128",
                  "window329000": "window32", "window329001": "window32",
                  "pred49000": "pred4", "pred49001": "pred4"}


# ===================================================================== gate
def section_zero(text: str) -> str:
    """The runbook §0 block (same helper as ``ext_view._section_zero``)."""
    start = text.find("## 0.")
    if start < 0:
        start = text.find("## 0 ")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 3)
    return text[start: end if end >= 0 else None]


def _git(root, *args, timeout=30):
    """Run git in ``root``; returns ``(returncode, stdout, stderr)``."""
    try:
        finished = subprocess.run(["git", *args], cwd=str(root), text=True,
                                  capture_output=True, timeout=timeout)
        return (finished.returncode, finished.stdout, finished.stderr)
    except Exception as problem:  # pragma: no cover - git missing
        return (128, "", f"{type(problem).__name__}: {problem}")


def _resolve_root(root=None) -> Path:
    return Path(root).resolve() if root is not None else ROOT


def p_paths(root=None) -> list:
    """``<P>`` (files **and** directories) as absolute paths for gate (d)."""
    root = _resolve_root(root)
    return [root / name for name in (*P_FILES, *P_DIRS)]


def in_p(relative: str) -> bool:
    """True when a repository-relative path is inside the reviewed set ``<P>``."""
    relative = str(relative).replace(os.sep, "/")
    if relative in P_FILES:
        return True
    return any(relative.startswith(directory) for directory in P_DIRS)


def acceptance_record(root=None) -> dict | None:
    """The lead's acceptance record (§6.2 (c)), or ``None`` before acceptance."""
    path = _resolve_root(root) / ACCEPTANCE_REL
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except ValueError:
        return {"invalid": f"{path}: not valid JSON"}
    return payload


def import_closure(root=None) -> dict:
    """§6.2 (d) executed-import closure: every repo file in ``sys.modules`` in ``<P>``.

    A forced, read-only dependency that a ``<P>`` module imports
    (``reconstruction_model`` via ``bridge_run``/``jepa``/``backbone``) is
    recorded separately; it is not an implemented path.
    """
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
              if row["path"].startswith(FORCED_IMPORT_EXCEPTIONS)]
    violations = [row for row in outside
                  if not row["path"].startswith(FORCED_IMPORT_EXCEPTIONS)]
    return {"checked_modules": int(checked),
            "outside_allowlist": sorted(outside, key=lambda row: row["path"]),
            "forced_import_exceptions": sorted(forced, key=lambda row: row["path"]),
            "violations": sorted(violations, key=lambda row: row["path"]),
            "pass": not violations}


def _gate_checks(mode, *, root, arm, variant, seed, steps, out_root, dry_run,
                 ext_root=None) -> dict:
    root = _resolve_root(root)
    checks = {}
    problems = []
    runbook_path = root / RUNBOOK_REL
    computed_runbook = lr.sha256_file(runbook_path) if runbook_path.is_file() else None
    record = acceptance_record(root)
    pending = record is None

    # (a) committed runbook sha256 equals the acceptance-record value
    expected_runbook = (record or {}).get("runbook_sha256") if record else RUNBOOK_SHA256
    a_pass = bool(computed_runbook is not None and computed_runbook == expected_runbook)
    if record is None and computed_runbook != RUNBOOK_SHA256:
        a_pass = False
    checks["a"] = {"pass": a_pass, "runbook": str(runbook_path),
                   "runbook_sha256": computed_runbook,
                   "expected_sha256": expected_runbook, "pending_acceptance": pending}
    if not a_pass and not pending:
        problems.append("(a) committed runbook sha256 does not match the acceptance record")

    # (b) train-t1: exact §0 line, read through section_zero, and DECISIONS.md
    decisions_path = root / DECISIONS_REL
    decisions = decisions_path.read_text() if decisions_path.is_file() else ""
    section = section_zero(runbook_path.read_text()) if runbook_path.is_file() else ""
    b_book = AUTHOR_LINE in section
    b_decisions = AUTHOR_LINE in decisions
    checks["b"] = {"applicable": mode == "train-t1", "author_line": AUTHOR_LINE,
                   "in_section_zero": bool(b_book), "in_decisions": bool(b_decisions),
                   "pass": bool(b_book and b_decisions)}
    if mode == "train-t1" and not (b_book and b_decisions):
        problems.append("(b) the T-1 author line is missing from runbook §0 or DECISIONS.md")

    # (c) acceptance record schema and review verdict
    if record is None:
        c_row = {"pass": False, "status": "gate_pending_acceptance",
                 "reason": f"{ACCEPTANCE_REL} does not exist yet"}
    elif "invalid" in record:
        c_row = {"pass": False, "status": "invalid", "reason": record["invalid"]}
    else:
        missing = [key for key in ACCEPTANCE_KEYS if key not in record]
        c_pass = not missing and record.get("review_verdict") == "PASS"
        c_row = {"pass": c_pass, "status": "PASS" if c_pass else "denied",
                 "record": record, "missing_keys": missing,
                 "review_verdict": record.get("review_verdict")}
        if not c_pass:
            problems.append("(c) the acceptance record is incomplete or its review_verdict "
                            "is not PASS")
    checks["c"] = c_row

    # (d) working tree against impl_commit over <P>, and no untracked <P> file
    impl_commit = (record or {}).get("impl_commit") if record else None
    if impl_commit is None:
        code, head, _ = _git(root, "rev-parse", "HEAD")
        impl_commit = head.strip() if code == 0 else None
    paths = [name for name in (*P_FILES, *P_DIRS)]
    if impl_commit:
        code, _, err = _git(root, "diff", "--quiet", str(impl_commit), "--", *paths)
        diff_clean = code == 0
        diff_error = None if diff_clean else (err.strip() or f"git diff exit {code}")
    else:
        diff_clean, diff_error = False, "no impl_commit and no git HEAD"
    code, untracked_out, _ = _git(root, "ls-files", "--others", "--exclude-standard",
                                  "--", *paths)
    untracked = [line for line in untracked_out.splitlines() if line.strip()]
    d_pass = bool(diff_clean and not untracked)
    checks["d"] = {"pass": d_pass, "impl_commit": impl_commit,
                   "impl_commit_source": ("acceptance_record" if record
                                          else "HEAD (pre-acceptance)"),
                   "diff_clean": bool(diff_clean), "diff_error": diff_error,
                   "untracked": untracked,
                   "paths": list(P_FILES) + list(P_DIRS)}
    if not d_pass:
        problems.append(f"(d) the <P> tree differs from {impl_commit} "
                        f"(untracked: {untracked[:3]}{'...' if len(untracked) > 3 else ''})")

    # (e) train-t1: only the pinned (arm, variant, seed, steps) launches
    if mode == "train-t1":
        launched = (str(arm), str(variant), int(seed) if seed is not None else None,
                    int(steps) if steps is not None else None)
        pair_ok = launched in T1_ALLOWED
        smoke = launched == T1_SMOKE
        # N3: the out root must be exactly <ext>/{smoke,train} on the resolved tree,
        # not merely a directory with the right basename.
        ext_base = (Path(ext_root) if ext_root is not None
                    else (Path(out_root).parent if out_root is not None else EXT_ROOT))
        expected_root = Path(ext_base) / ("smoke" if smoke else "train")
        root_ok = (out_root is None
                   or Path(out_root).resolve() == expected_root.resolve())
        run_id = lr.run_id_for(T1_CELL_IDS.get(str(variant), ""), T1_ARM, int(seed or 0))
        existing = (Path(out_root) / "runs" / run_id).exists() if out_root is not None else False
        e_pass = bool(pair_ok and root_ok and not existing)
        checks["e"] = {"pass": e_pass, "launched": list(launched),
                       "allowed": sorted(list(row) for row in T1_ALLOWED),
                       "smoke": bool(smoke), "out_root": str(out_root) if out_root else None,
                       "ext_root": str(ext_base),
                       "expected_out_root": str(expected_root),
                       "run_id": run_id, "run_dir_exists": bool(existing)}
        if not e_pass:
            problems.append(f"(e) {launched} is not an authorised T-1 launch, its out root "
                            f"is not exactly {expected_root}, or its run dir exists")
    else:
        checks["e"] = {"applicable": False}

    # (d) import closure: only asserted by the dry run (§6.2)
    if mode == "dry-run":
        closure = import_closure(root)
        checks["closure"] = closure
        if not closure["pass"]:
            problems.append(f"(d) executed-import closure has {len(closure['violations'])} "
                            f"module(s) outside <P>")

    authorised = bool(not problems and not pending)
    status = ("gate_pending_acceptance" if pending else
              ("granted" if authorised else "denied"))
    return {"authorised": bool(authorised), "status": status, "mode": mode,
            "extension": EXTENSION, "checks": checks, "problems": problems,
            "acceptance": record, "runbook_sha256": computed_runbook}


def assert_authorised(mode, *, root=None, arm=None, variant=None, seed=None, steps=None,
                      out_root=None, ext_root=None, dry_run=None, log=None) -> dict:
    """Hard gate (§6.2 (a)-(e)).  Any failure raises ``SystemExit(2)``.

    ``dry_run=True`` (or ``mode == "dry-run"``) records the checks and returns
    them: a missing acceptance record is then the normal pre-acceptance state
    ``gate_pending_acceptance`` rather than a failure, and the executed-import
    closure is asserted as part of the same report.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    dry = bool(dry_run) if dry_run is not None else (mode == "dry-run")
    result = _gate_checks(mode, root=root, arm=arm, variant=variant, seed=seed,
                          steps=steps, out_root=out_root, dry_run=dry,
                          ext_root=ext_root)
    if dry:
        return result
    if not result["authorised"]:
        for problem in result["problems"]:
            print(f"[followup] authorisation gate: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return result


# ==================================================================== data
def manifest_path(data_root, key) -> Path:
    return Path(data_root) / "lucid" / str(key) / "manifest.json"


def data_status(data_root, *, pins=None, keys=None) -> dict:
    """Read-only sha256 check of the pinned dataset manifests (§2)."""
    pins = dict(PINNED_DATA if pins is None else pins)
    selected = list(keys) if keys is not None else list(pins)
    out = {"registered": False, "extension": EXTENSION, "check": "§2 pinned data manifests",
           "data_root": str(data_root), "sets": {}, "pass": True,
           "regeneration": "forbidden (no data generation in this extension)"}
    for key in selected:
        expected = pins[key]
        path = manifest_path(data_root, key)
        row = {"key": key, "manifest": str(path), "exists": path.is_file(),
               "expected_sha256": expected}
        if path.is_file():
            row["manifest_sha256"] = lr.sha256_file(path)
            row["pass"] = row["manifest_sha256"] == expected
        else:
            row["manifest_sha256"] = None
            row["pass"] = False
        out["pass"] = bool(out["pass"] and row["pass"])
        out["sets"][key] = row
    return out


def require_pinned(data_root, keys=None, *, pins=None, context="") -> dict:
    """Refuse to continue unless every named pinned manifest matches (§2)."""
    status = data_status(data_root, pins=pins, keys=keys)
    bad = [key for key, row in status["sets"].items() if not row["pass"]]
    if bad:
        raise SystemExit(f"[followup] {context or 'pinned data'}: missing or mismatched "
                         f"manifest(s) {bad}; no regeneration is allowed (§2)")
    return status


def _cohort(config, name):
    return config["data"]["cohorts"][name]


def dataset_key_for(config, name, setting, *, registered=None) -> str:
    cohort = _cohort(config, name)
    if registered is None:
        registered = not (abs(float(setting) - S030) < 1e-12)
    return lr.dataset_key(name, "F1", float(setting), int(config["data"]["dataset_seed"]),
                          None, int(cohort["n_events"]), int(cohort["run_length"]),
                          bool(registered))


def open_probe_sets(config, gen, geometry, data_root, *, setting=S005, log=print) -> dict:
    """Open the two probe sets after a read-only manifest-hash check (§2).

    s = 0.05 sets use their registered keys; s = 0.30 sets use the
    ``registered=False`` keys and are never passed through a loader that
    hard-codes ``registered=True``.
    """
    keys = PROBE_KEYS_S005 if abs(float(setting) - S005) < 1e-12 else PROBE_KEYS_S030
    require_pinned(data_root, keys=list(keys.values()), context=f"probe sets s={setting}")
    registered = abs(float(setting) - S030) >= 1e-12
    probes = {}
    seed = int(config["data"]["dataset_seed"])
    for role, name in (("train", "dev_probe_train"), ("test", "dev_probe_test")):
        cohort = _cohort(config, name)
        expected = keys[role]
        dataset = lr.ensure_dataset(
            data_root=Path(data_root), geometry=geometry, generator_config=gen,
            cohort_spec=cohort, cohort_name=name, law="F1", setting=float(setting),
            settings=None, setting_kind=None, registered=registered, seed=seed,
            code_sha=None, verify_hashes=False, log=log)
        if dataset.key != expected:
            raise RuntimeError(f"[followup] {role} probe set opened {dataset.key!r}, "
                               f"expected the pinned {expected!r}")
        probes[role] = dataset
    return probes


def open_train_set(config, gen, geometry, data_root, *, setting=S030, log=print) -> dict:
    """Open one training cohort after a read-only manifest-hash check (§2/§4)."""
    name = "dev_train"
    key = dataset_key_for(config, name, setting, registered=False)
    require_pinned(data_root, keys=[key], context=f"train cohort s={setting}")
    dataset = lr.ensure_dataset(
        data_root=Path(data_root), geometry=geometry, generator_config=gen,
        cohort_spec=_cohort(config, name), cohort_name=name, law="F1",
        setting=float(setting), settings=None, setting_kind=None, registered=False,
        seed=int(config["data"]["dataset_seed"]), code_sha=None, verify_hashes=False,
        log=log)
    if dataset.key != key:
        raise RuntimeError(f"[followup] train cohort opened {dataset.key!r}, expected {key!r}")
    return dataset


def require_dark_bank(config, gen, geometry, data_root, *, log=print):
    """Read-only hash check of the pinned dark bank **before** ``ensure_dark_bank``.

    ``pregrid.build_loaded`` reaches ``lr.ensure_dark_bank`` for the anchor arms;
    the §2 rule requires the pinned dark-bank manifest sha256 to be checked first.
    """
    require_pinned(data_root, keys=[DARK_BANK_KEY], context="dark bank (anc100)")
    directory = lr.ensure_dark_bank(
        data_root=Path(data_root), geometry=geometry, generator_config=gen,
        seed=int(config["data"]["dataset_seed"]),
        records=int(config["data"]["dark_noise"]["records"]), code_sha=None)
    return la.dark_bank_from_store(Path(directory) / "hits.npz", geometry)


# ================================================================ helpers
def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, payload, *, overwrite=True) -> Path:
    """Atomic JSON write; ``overwrite=False`` refuses an existing file."""
    path = Path(path)
    if not overwrite and path.exists():
        raise RuntimeError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(lr._safe(payload), indent=1, default=float) + "\n")
    os.replace(tmp, path)
    return path


def refuse_existing_run_dir(out_root, run_id) -> Path:
    """§2/§5: never launch into an existing run directory (``run_one`` auto-resumes)."""
    path = Path(out_root) / "runs" / str(run_id)
    if path.exists():
        raise RuntimeError(f"refusing existing run directory {path} "
                           f"(no auto-resume in this extension, §5)")
    return path


def prepare_run_dir(out_root, run_id, sidecar=None, *, steps=None, device=None) -> Path:
    """Refuse an existing run dir and write the per-run sidecar (§5)."""
    path = refuse_existing_run_dir(out_root, run_id)
    path.mkdir(parents=True, exist_ok=False)
    payload = {"registered": False, "extension": EXTENSION, "run_id": str(run_id),
               "out_root": str(out_root), "steps": steps, "device": str(device)}
    payload.update(sidecar or {})
    (path / "variant.json").write_text(json.dumps(payload, indent=1) + "\n")
    return path


def entries_snapshot(guard_json=GUARD_JSON) -> dict:
    path = Path(guard_json)
    if not path.is_file():
        return {"available": False, "guard_json": str(path),
                "error": "guard status file is missing"}
    try:
        payload = json.loads(path.read_text())
    except ValueError as problem:
        return {"available": False, "guard_json": str(path), "error": str(problem)}
    caps = payload.get("caps") or {}
    free = caps.get("free_entries")
    return {"available": free is not None, "guard_json": str(path),
            "free_entries": None if free is None else int(free),
            "guard_level": payload.get("level"),
            "guard_reasons": payload.get("reasons"),
            "guard_timestamp": payload.get("timestamp")}


def entries_projection(free_before, *, clone=ENTRIES_CLONE, suite=ENTRIES_SUITE,
                       extension=ENTRIES_EXT, floor=ENTRIES_FLOOR) -> dict:
    projected = int(free_before) - (int(clone) + int(suite) + int(extension))
    return {"free_before": int(free_before), "clone": int(clone), "suite": int(suite),
            "extension": int(extension), "projected_after": int(projected),
            "floor": int(floor),
            "pass": bool(projected >= int(floor) and int(free_before) >= int(floor))}


def entries_preflight(*, guard_json=GUARD_JSON, log=print) -> dict:
    """§5 entries preflight; a failure is a STOP (no GPU work)."""
    snapshot = entries_snapshot(guard_json)
    out = {"snapshot": snapshot, "pass": False}
    if not snapshot.get("available"):
        out["error"] = snapshot.get("error", "no free_entries in the guard status")
        return out
    out["projection"] = entries_projection(snapshot["free_entries"])
    # §5 entries preflight is about free *entries*; the guard HOLD flag is about a
    # free GPU slot (the expected view-design jobs hold it) and is recorded, not a
    # STOP: the launch waits for a slot (§5 host rules).
    out["gpu_slot_available"] = bool(snapshot.get("guard_level") != "HOLD")
    out["pass"] = bool(out["projection"]["pass"])
    return out


def hardware_and_software() -> dict:
    software = {"python": sys.version.split()[0], "platform": platform.platform(),
                "host": lr.host_name(), "numpy": np.__version__}
    try:
        import torch
        software["torch"] = torch.__version__
        software["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch is a dependency
        software["torch"] = None
    hardware = {"cpu_count": os.cpu_count(), "machine": platform.machine()}
    return {"hardware": hardware, "software": software}


# ============================================================== inventory
#: §2.1 label families whose path+sha come from the committed readout items.json.
_MATRIX_LABELS = ("old9000", "old9001", "fix9000", "fix9001", "mae9000",
                  "across9000", "across9001")
#: §2.1 SIGReg variants, from the committed predictability probe JSONs.
_PRED_LABELS = {
    "ijepa9000": "ijepa9000.json", "ijepa9001": "ijepa9001.json",
    "lambda9000": "lambda9000.json", "lambda9001": "lambda9001.json",
    "stopgrad9000": "stopgrad9000.json", "stopgrad9001": "stopgrad9001.json",
    "strong_drift9000": "strong_drift9000.json", "strong_drift9001": "strong_drift9001.json",
}
#: §2.1 SIGReg de-jitter retrains (large_arrays keys are relative to DEJITTER_EXT).
_DEJITTER_LABELS = {
    "ext_dev_s005_within_trigger__sigreg_jepa": "sigreg_within_trigger",
    "ext_dev_s005_within_median__sigreg_jepa": "sigreg_within_median",
    "ext_dev_s005_across_median__sigreg_jepa": "sigreg_across_median",
}
#: §2.1 MAE de-jitter checkpoints (relative to MAE_DEJITTER_EXT).
_MAE_DEJITTER_LABELS = {
    "ext_dev_s005_within_trigger__mae_mse": "mae_within_trigger",
    "ext_dev_s005_within_median__mae_mse": "mae_within_median",
    "ext_dev_s005_across__mae_mse": "mae_across_trigger",
    "ext_dev_s005_across_median__mae_mse": "mae_across_median",
}
_SEED_SUFFIX = lambda seed: f"s{int(seed)}"          # noqa: E731
_MEDIAN_VARIANTS = ("within_median", "across_median")
_ACROSS_VARIANTS = ("across", "across_trigger", "across_median")
_SKIP_WALK = {"tmp", "logs", "part1", "token_time", "validation", "ckpt",
              "smoke_archive", "train_archive", "eval", "results_h12",
              "results_static", "results_scale", "results_arch", "results_readout"}


def _row(label, *, family, arm, split, seed, beta, source, sha256, data_setting=S005,
         status="available", data_key=None, **extra) -> dict:
    if data_key is None:
        data_key = (TRAIN_KEY_S030 if abs(float(data_setting) - S030) < 1e-12
                    else TRAIN_KEY_S005)
    row = {"label": label, "family": family, "arm": arm, "cell": None, "split": split,
           "seed": int(seed), "beta": beta, "encoder": None, "data_key": str(data_key),
           "source": None if source is None else str(source), "sha256": sha256,
           "data_setting": float(data_setting), "status": status,
           "eval": {"zero_geometry": False, "median_wrapper": False,
                    "encoder_variant": None, "anchor": False,
                    "time_convention": A4_TOKEN_TRIGGER}}
    row.update(extra)
    if row["data_setting"] != S005:
        row["cell"] = f"ext_dev_s{int(round(row['data_setting'] * 100)):03d}_{split}"
    else:
        row["cell"] = "dev_s005_within" if split == "within" else f"ext_dev_s005_{split}"
    return row


def _verify(source, expected, *, hash_files=True) -> tuple:
    """``(status, actual_sha)`` for one checkpoint path."""
    if not source:
        return "unavailable", None
    path = Path(source)
    if not path.is_file():
        return "unavailable", None
    if not hash_files:
        return "unverified (dry-run)", None
    actual = lr.sha256_file(path)
    if expected and actual != expected:
        return "unavailable", actual
    return "available", actual


def find_final_pt(root, wanted, *, max_files=500, max_depth=8) -> dict:
    """Search ``root`` for ``final.pt`` files whose sha256 is in ``wanted``.

    ``wanted`` maps sha256 -> label; returns ``{label: path}``.  Read-only.
    """
    root = Path(root)
    found = {}
    if not root.is_dir() or not wanted:
        return found
    remaining = dict(wanted)
    seen = 0
    for base, dirs, files in os.walk(root):
        directory = Path(base)
        depth = len(directory.relative_to(root).parts)
        dirs[:] = sorted(name for name in dirs
                         if name not in _SKIP_WALK and depth < max_depth)
        if "final.pt" not in files:
            continue
        path = directory / "final.pt"
        seen += 1
        if seen > max_files:
            break
        try:
            digest = lr.sha256_file(path)
        except OSError:  # pragma: no cover - racing cleanup
            continue
        if digest in remaining:
            found[remaining.pop(digest)] = str(path)
            if not remaining:
                break
    return found


def _readout_items_fallback(root) -> dict:
    """``scripts.jepa_ext_readout.ITEMS`` (path+sha), read without running the CLI."""
    try:
        from scripts import jepa_ext_readout as readout_cli
    except Exception:  # pragma: no cover - scripts package is present
        return {}
    return {label: dict(spec) for label, spec in getattr(readout_cli, "ITEMS", {}).items()}


def build_inventory(root=None, *, hash_files=True, view_root=None, log=print) -> dict:
    """§2.1 checkpoint inventory: one row per label, accepted only on a sha match."""
    root = _resolve_root(root)
    payload = {"registered": False, "extension": EXTENSION, "check": "§2.1 checkpoint inventory",
               "root": str(root), "rows": {}, "pending": [], "unavailable": [],
               "hash_files": bool(hash_files)}
    rows = payload["rows"]

    def register(row):
        rows[row["label"]] = row
        if row["status"] == "unavailable":
            payload["unavailable"].append(row["label"])
        elif row["status"].startswith("pending"):
            payload["pending"].append(row["label"])

    # reference: rebuilt init9000 and the D1 strong-drift init floor (no file)
    register(_row("init9000", family="matrix", arm="sigreg_jepa", split="within", seed=9000,
                  beta=None, source=None, sha256=None, init=True))
    register(_row("strong_drift_init", family="init_floor", arm="sigreg_jepa", split="within",
                  seed=9000, beta=None, source=None, sha256=None, data_setting=S030))

    # §2.1 reference + SIGReg plain + MAE (Part C): readout items.json + ITEMS
    items = (read_json(READOUT_EXT / "items.json", {}) or {}).get("items") or {}
    fallback = _readout_items_fallback(root)
    for label in _MATRIX_LABELS:
        spec = items.get(label) or {}
        old = fallback.get(label) or {}
        source = spec.get("ckpt") or old.get("source")
        expected = spec.get("ckpt_sha256") or old.get("sha256")
        status, actual = _verify(source, expected, hash_files=hash_files)
        arm = spec.get("arm") or old.get("arm") or ("mae_mse" if label.startswith("mae")
                                                    else "sigreg_jepa")
        split = spec.get("split") or old.get("split") or "within"
        seed = int(spec.get("seed") or old.get("seed") or 9000)
        register(_row(label, family="matrix", arm=arm, split=split, seed=seed, beta=None,
                      source=source if status == "available" else None,
                      sha256=expected, status=status, actual_sha256=actual))

    # §2.1 SIGReg variants (predictability probe JSONs) and anc100
    for label, filename in _PRED_LABELS.items():
        probe = read_json(root / "results/jepa/20261029_predictability/probes" / filename,
                          {}) or {}
        source = probe.get("checkpoint")
        expected = probe.get("checkpoint_sha256")
        status, actual = _verify(source, expected, hash_files=hash_files)
        strong = label.startswith("strong_drift")
        register(_row(label, family="predictability", arm=probe.get("arm") or "sigreg_jepa",
                      split="within", seed=int(probe.get("seed") or label[-4:]),
                      beta=probe.get("beta"), source=source if status == "available" else None,
                      sha256=expected, status=status, actual_sha256=actual,
                      data_setting=S030 if strong else S005,
                      eval={"time_convention": A4_TOKEN_TRIGGER}))
    anc = read_json(root / "results/jepa/20261029_predictability/probes/anc100.json", {}) or {}
    anc_status, anc_actual = _verify(anc.get("checkpoint"), anc.get("checkpoint_sha256"),
                                     hash_files=hash_files)
    register(_row("anc100", family="predictability", arm="sigreg_jepa_anchored",
                  split="within", seed=int(anc.get("seed") or 9103),
                  beta=float(anc.get("beta") or 100.0),
                  source=anc.get("checkpoint") if anc_status == "available" else None,
                  sha256=anc.get("checkpoint_sha256"), status=anc_status,
                  actual_sha256=anc_actual,
                  eval={"anchor": True, "time_convention": A4_TOKEN_TRIGGER}))

    # §2.1 SIGReg de-jitter retrains
    dejitter = read_json(root / "results/jepa/20261030_dejitter/large_arrays.json", {}) or {}
    for key, meta in dejitter.items():
        if not (key.startswith("train/runs/") and key.endswith("/final.pt")):
            continue
        run_id = key.split("/")[2]
        prefix = run_id.rsplit("__", 1)[0]
        seed = int(run_id.rsplit("__", 1)[1].lstrip("s"))
        label_base = next((value for name, value in _DEJITTER_LABELS.items()
                           if prefix == name), None)
        if label_base is None:
            continue
        variant = prefix.replace("ext_dev_s005_", "").split("__")[0]
        source = DEJITTER_EXT / key
        status, actual = _verify(source, meta.get("sha256"), hash_files=hash_files)
        register(_row(f"{label_base}{seed}", family="dejitter_sigreg", arm="sigreg_jepa",
                      split="across" if "across" in variant else "within", seed=seed,
                      beta=None, source=source if status == "available" else None,
                      sha256=meta.get("sha256"), status=status, actual_sha256=actual,
                      eval={"median_wrapper": "median" in variant,
                            "time_convention": (A4_TOKEN_MEDIAN if "median" in variant
                                                else A4_TOKEN_TRIGGER)}))

    # §2.1 MAE de-jitter checkpoints
    mae = read_json(root / "results/jepa/20261031_mae_dejitter/large_arrays.json", {}) or {}
    for key, meta in mae.items():
        if not (key.startswith("train/runs/") and key.endswith("/final.pt")):
            continue
        run_id = key.split("/")[2]
        prefix, seed_text = run_id.rsplit("__", 1)
        seed = int(seed_text.lstrip("s"))
        label_base = _MAE_DEJITTER_LABELS.get(prefix)
        if label_base is None:
            continue
        variant = prefix.replace("ext_dev_s005_", "")
        source = MAE_DEJITTER_EXT / key
        status, actual = _verify(source, meta.get("sha256"), hash_files=hash_files)
        register(_row(f"{label_base}{seed}", family="dejitter_mae", arm="mae_mse",
                      split="across" if variant.startswith("across") else "within",
                      seed=seed, beta=None, source=source if status == "available" else None,
                      sha256=meta.get("sha256"), status=status, actual_sha256=actual,
                      eval={"median_wrapper": "median" in variant,
                            "time_convention": (A4_TOKEN_MEDIAN if "median" in variant
                                                else A4_TOKEN_TRIGGER)}))

    # §2.1 target controls: E2 geomblind, E3 frozen_target (h12) and static_target
    h12_arrays = read_json(H12_EXT / "large_arrays.json", {}) or {}
    for key, meta in (h12_arrays.get("checkpoints") or {}).items():
        if not key.endswith("/final.pt"):
            continue
        run_id = key.split("/")[-2]
        if "__" not in run_id:
            continue
        label_base = ("geomblind" if key.startswith("train/geomblind/")
                      else "frozen_target" if key.startswith("train/frozen_target/") else None)
        if label_base is None:
            continue
        seed = int(run_id.rsplit("__", 1)[1].lstrip("s"))
        source = H12_EXT / key
        status, actual = _verify(source, meta.get("sha256"), hash_files=hash_files)
        register(_row(f"{label_base}{seed}", family="h12", arm="sigreg_jepa", split="within",
                      seed=seed, beta=None, source=source if status == "available" else None,
                      sha256=meta.get("sha256"), status=status, actual_sha256=actual,
                      eval={"zero_geometry": label_base == "geomblind"}))
    static_arrays = read_json(STATIC_EXT / "large_arrays.json", {}) or {}
    for key, meta in (static_arrays.get("checkpoints") or {}).items():
        if not (key.startswith("train/") and key.endswith("/final.pt")):
            continue
        run_id = key.split("/")[-2]
        if "__" not in run_id:
            continue
        seed = int(run_id.rsplit("__", 1)[1].lstrip("s"))
        source = STATIC_EXT / key
        status, actual = _verify(source, meta.get("sha256"), hash_files=hash_files)
        register(_row(f"static_target{seed}", family="static", arm="sigreg_jepa",
                      split="within", seed=seed, beta=None,
                      source=source if status == "available" else None,
                      sha256=meta.get("sha256"), status=status, actual_sha256=actual))

    # §2.1 scale / arch: promoted runs.json values (sha only), paths found by search
    sha_only = {}
    for label, path, shas in _sha_only_specs(root):
        wanted = {}
        for value, sha in shas.items():
            wanted[sha] = value
        found = find_final_pt(path, wanted, max_files=400) if hash_files else {}
        for value, sha in shas.items():
            source = found.get(value)
            status = ("available" if source else
                      ("unavailable" if hash_files else "unverified (dry-run)"))
            sha_only[value] = (status, source, sha)
    for label in ("scale300k9000", "scale300k9001", "window1289000", "window1289001",
                  "window329000", "window329001", "pred49000", "pred49001"):
        status, source, sha = sha_only.get(label, ("unavailable", None, None))
        variant = _ARCH_VARIANTS.get(label)
        register(_row(label, family="arch" if variant else "scale", arm="sigreg_jepa",
                      split="within", seed=int(label[-4:]), beta=None,
                      source=source if status == "available" else None, sha256=sha,
                      status=status,
                      data_key=(None if variant else TRAIN_KEY_300K),
                      eval={"encoder_variant": variant}))

    # §2.1 view design: only if the design's committed runs.json exists.  The
    # view writer keys its rows ``<split>__<seed>`` (scripts/jepa_ext_view.py);
    # the inventory label is ``view_<split><seed>``.
    view_runs = root / "results/jepa/20261103_view/runs.json"
    view_search = Path(view_root) if view_root is not None else VIEW_EXT
    view_committed = {}
    if view_runs.is_file():
        view_payload = read_json(view_runs, {}) or {}
        for key, row in (view_payload.get("runs") or {}).items():
            try:
                split, seed_text = str(key).split("__")
                seed = int(seed_text)
            except ValueError:
                continue
            label = f"view_{split}{seed}"
            view_committed[label] = {
                "sha256": row.get("checkpoint_sha256") or row.get("sha256"),
                "vertex_m": ((row.get("full_input") or {}).get("vertex_m")),
                "split": split}
    view_found = {}
    wanted = {meta["sha256"]: label for label, meta in view_committed.items()
              if meta["sha256"]}
    if wanted and hash_files:
        view_found = find_final_pt(view_search, wanted, max_files=200)
    for split in ("within", "across", "module_parity", "random_half"):
        for seed in (9000, 9001):
            label = f"view_{split}{seed}"
            if not view_runs.is_file():
                register(_row(label, family="view", arm="sigreg_jepa", split="within",
                              seed=seed, beta=None, source=None, sha256=None,
                              status="pending (view design)"))
                continue
            meta = view_committed.get(label) or {}
            source = view_found.get(label)
            status = ("available" if source else
                      ("unavailable" if hash_files else "unverified (dry-run)"))
            register(_row(label, family="view", arm="sigreg_jepa", split="within",
                          seed=seed, beta=None, source=source,
                          sha256=meta.get("sha256"), status=status,
                          committed_vertex_m=meta.get("vertex_m")))

    payload["counts"] = {"labels": len(rows),
                         "available": sum(1 for row in rows.values()
                                          if row["status"] == "available"),
                         "unavailable": len(payload["unavailable"]),
                         "pending": len(payload["pending"])}
    return payload


def _sha_only_specs(root):
    """``(label, search root, {label: sha})`` for the sha-only families (§2.1)."""
    specs = []
    scale = read_json(root / "results/jepa/20261102_scale/runs.json", {}) or {}
    arch = read_json(root / "results/jepa/20261102_arch/runs.json", {}) or {}
    scale_shas = {f"scale300k{int(seed)}": row.get("checkpoint_sha256")
                  for seed, row in (scale.get("runs") or {}).items()
                  if row.get("checkpoint_sha256")}
    arch_shas = {}
    for key, row in (arch.get("runs") or {}).items():
        variant, seed = key.split("__")
        label = {"window128": "window128", "window32": "window32",
                 "pred4": "pred4"}.get(variant)
        if label and row.get("checkpoint_sha256"):
            arch_shas[f"{label}{int(seed)}"] = row["checkpoint_sha256"]
    specs.append(("scale/arch", SCALE_EXT, scale_shas))
    specs.append(("scale/arch", ARCH_EXT, arch_shas))
    return specs

# ==================================================================== A0
def readout_label_list(item, rows, *, include_t1=True) -> list:
    """The labels a ``readout --item`` body must evaluate.

    The fixed item table plus, when ``--include-t1`` is given, the completed T-1
    checkpoints (``t1_*``) that were merged into the inventory (B4).  The T-1
    readouts are booked to the T-1 cap (§5).
    """
    if item == "A-1":
        base = list(A1_COMPUTE_LABELS) + list(A1_MLP_FILL_LABELS)
    elif item == "A-2":
        base = list(A2_LABELS)
    else:
        base = []
    extra = (sorted(str(label) for label in rows if str(label).startswith("t1_"))
             if include_t1 else [])
    seen = set(base)
    return base + [label for label in extra if label not in seen]


A1_S005_LABELS = tuple(A1_S005)
A1_S030_LABELS = tuple(A1_S030)
A1_COMPUTE_LABELS = (
    "ijepa9000", "ijepa9001", "lambda9000", "lambda9001", "stopgrad9000", "stopgrad9001",
    "strong_drift9000", "strong_drift9001", "anc100",
    "sigreg_within_trigger9000", "sigreg_within_trigger9001",
    "sigreg_within_median9000", "sigreg_within_median9001",
    "sigreg_across_median9000", "sigreg_across_median9001",
    "mae_within_trigger9000", "mae_within_trigger9001",
    "mae_within_median9000", "mae_within_median9001",
    "mae_across_trigger9000", "mae_across_trigger9001",
    "mae_across_median9000", "mae_across_median9001",
)
A1_MLP_FILL_LABELS = (
    "geomblind9000", "geomblind9001", "frozen_target9000", "frozen_target9001",
    "static_target9000", "static_target9001", "scale300k9000", "scale300k9001",
    "window1289000", "window1289001", "window329000", "window329001",
    "pred49000", "pred49001", "across9000", "across9001",
)
A1B_LABELS = (
    "fix9000", "fix9001", "scale300k9000", "scale300k9001", "window1289000",
    "window1289001", "window329000", "window329001", "pred49000", "pred49001",
    "across9000", "across9001", "init9000",
)
A2_LABELS = ("mae_within_trigger9000", "mae_within_trigger9001",
             "mae_across_trigger9000", "mae_across_trigger9001")
A3_LABELS = ("init9000", "fix9000", "fix9001", "frozen_target9000", "frozen_target9001",
             "ijepa9000", "ijepa9001", "mae_within_trigger9000", "mae_within_trigger9001",
             "mae_within_median9000", "mae_within_median9001")
A4_LABELS = ("mae_within_trigger9000", "mae_within_trigger9001",
             "mae_within_median9000", "mae_within_median9001",
             "fix9000", "fix9001", "frozen_target9000", "frozen_target9001")
#: §3 A0 pass factors.  A-4 is counted **conservatively** at 3.0 per checkpoint
#: (1 full-input reproduction pass + 2 masked reference passes: all-token and
#: context-only); the runbook's "0.5 for an A-4 masked-only reference" wording
#: disagrees with the implemented work (spec defect S3), so the projection states
#: what it counts instead of implying the 0.5 sentence.
STAGE_A_PASS_FACTORS = {"A1": 1.0, "A-1": 1.0, "A-1b": 0.6, "A-2": 2.0, "A-3": 2.0,
                        "A-4": 3.0}
CPU_PROJECTION = {
    "focused tests and the full suite": 1.5,
    "A0 preflight (hashes, inventory, pairing)": 0.5,
    "A-1b dev_train marginals": 1.5,
    "A-1b label passes and probes": 0.5,
    "A-3 standalone fits": 1.5,
    "A-2 bootstrap": 0.5,
    "readings and report": 1.0,
}
T1_PROJECTED_GPU_H = 2.7


def stage_a_items() -> list:
    """The §3 A0 item table: every GPU item of Stage A with its pass factor."""
    items = []
    for label in A1_S005_LABELS:
        items.append({"task": "A1", "label": label, "pass_factor": 1.0,
                      "environment": "s005"})
    for label in A1_S030_LABELS:
        items.append({"task": "A1", "label": label, "pass_factor": 1.0,
                      "environment": "s030"})
    for label in A1_COMPUTE_LABELS:
        items.append({"task": "A-1", "label": label, "pass_factor": 1.0})
    for label in A1_MLP_FILL_LABELS:
        items.append({"task": "A-1", "label": label, "pass_factor": 1.0,
                      "detail": "missing MLP token/vertex only"})
    for label in A1B_LABELS:
        items.append({"task": "A-1b", "label": label, "pass_factor": 0.6,
                      "detail": "target-only forward pass"})
    for label in A2_LABELS:
        items.append({"task": "A-2", "label": label, "pass_factor": 2.0,
                      "detail": "two environments"})
    for label in A3_LABELS:
        items.append({"task": "A-3", "label": label, "pass_factor": 2.0,
                      "domains": ["muon", "electron"]})
    for label in A4_LABELS:
        items.append({"task": "A-4", "label": label, "pass_factor": 3.0,
                      "detail": "1 full-input reproduction + 2 masked reference passes "
                                "(conservative, not the runbook's 0.5 wording)"})
    return items


def stage_a_projection(items=None, *, seconds_per_item=SECONDS_PER_ITEM,
                       factor=PROJECTION_FACTOR, cap_hours=STAGE_A_CAP_H) -> dict:
    """§3 A0 projection: Σ items × 126 s × pass factor × 1.3 (STOP S1 above the cap)."""
    items = stage_a_items() if items is None else list(items)
    equivalents = float(sum(float(item["pass_factor"]) for item in items))
    seconds = equivalents * float(seconds_per_item) * float(factor)
    hours = seconds / 3600.0
    per_task = {}
    for item in items:
        bucket = per_task.setdefault(item["task"], {"n_items": 0, "item_equivalents": 0.0})
        bucket["n_items"] += 1
        bucket["item_equivalents"] = round(bucket["item_equivalents"]
                                           + float(item["pass_factor"]), 3)
    return {"n_items": len(items), "item_equivalents": round(equivalents, 3),
            "seconds_per_item": float(seconds_per_item), "factor": float(factor),
            "projected_seconds": round(seconds, 1), "projected_l40s_hours": round(hours, 4),
            "cap_hours": float(cap_hours), "pass": bool(hours <= float(cap_hours)),
            "per_task": per_task, "pass_factors": dict(STAGE_A_PASS_FACTORS),
            "rule": "sum(items x 126 s x pass factors x 1.3); pass factors "
                    "{A1:1, A-1:1, A-1b:0.6, A-2:2, A-3:2, A-4:3} — A-4 is counted "
                    "conservatively at 3.0 per checkpoint (1 full-input reproduction + "
                    "2 masked reference passes), not the runbook §3 '0.5 masked-only "
                    "reference' wording (spec defect S3); STOP S1 if > 5.5 L40S GPU-h"}


def cpu_projection(*, cap_hours=CPU_CAP_H) -> dict:
    hours = float(sum(CPU_PROJECTION.values()))
    return {"tasks": dict(CPU_PROJECTION), "projected_hours": round(hours, 3),
            "cap_hours": float(cap_hours), "pass": bool(hours <= float(cap_hours)),
            "note": "CPU wall-hours (Deepthought), separate from the generation cap"}


def _s1_projection(root) -> dict:
    """§5 S1 launch rule: L40S ledger + view cap 8.0 + 9.0 ≤ 160."""
    ledger = read_json(Path(root) / "docs/jepa/RESOURCE_LEDGER.json", {}) or {}
    entry = (ledger.get("v4_caps") or {}).get("E2_lucid_l40s_deepthought2") or {}
    charged = entry.get("charged_gpu_hours")
    if charged is None:
        return {"available": False, "rule": "L40S + 8.0 + 9.0 <= 160", "pass": None}
    total = float(charged) + S1_VIEW_CAP_H + S1_PROJECTION_H
    return {"available": True, "charged_gpu_hours": float(charged),
            "view_cap_hours": S1_VIEW_CAP_H, "projection_hours": S1_PROJECTION_H,
            "sum_hours": round(total, 4), "ceiling_hours": S1_CEILING_H,
            "pass": bool(total <= S1_CEILING_H),
            "rule": "L40S ledger + view-design cap (8.0) + 9.0 <= 160"}


def pairing_check(data_root, *, limit=None, log=print) -> dict:
    """§2: join the s = 0.05 and s = 0.30 sets on ``(run_id, event_id)`` and compare truth.

    Truth and photon counts are identical; only charge, threshold crossing, walk and
    trigger differ.  Read-only (``events.npz`` only; no dataset is opened).
    """
    out = {"registered": False, "extension": EXTENSION,
           "check": "§2 (run_id, event_id) truth pairing, s=0.05 vs s=0.30",
           "sets": {}, "pass": True, "limit": limit}
    pairs = (("probe_train", PROBE_KEYS_S005["train"], PROBE_KEYS_S030["train"]),
             ("probe_test", PROBE_KEYS_S005["test"], PROBE_KEYS_S030["test"]),
             ("dev_train", TRAIN_KEY_S005, TRAIN_KEY_S030))
    truth_fields = ("vertex_m", "yield_photons", "particle", "domain", "contained",
                    "deposited_MeV", "direction_m", "energy_MeV")
    for role, key05, key30 in pairs:
        manifest05 = read_json(manifest_path(data_root, key05), {}) or {}
        manifest30 = read_json(manifest_path(data_root, key30), {}) or {}
        rows05 = manifest05.get("runs") or []
        rows30 = manifest30.get("runs") or []
        entry = {"key_s005": key05, "key_s030": key30, "n_runs_s005": len(rows05),
                 "n_runs_s030": len(rows30), "n_compared": 0, "mismatches": [],
                 "pass": True}
        if len(rows05) != len(rows30):
            entry["mismatches"].append({"reason": "run count differs"})
        for index, (row05, row30) in enumerate(zip(rows05, rows30)):
            if limit is not None and index >= int(limit):
                break
            entry["n_compared"] += 1
            run_id05, run_id30 = row05.get("run_id"), row30.get("run_id")
            if run_id05 != run_id30:
                entry["mismatches"].append({"reason": "run_id differs", "index": index,
                                            "s005": run_id05, "s030": run_id30})
                continue
            path05 = manifest_path(data_root, key05).parent / "runs" / row05["run_dir"]
            path30 = manifest_path(data_root, key30).parent / "runs" / row30["run_dir"]
            got05 = _read_truth(path05 / "events.npz")
            got30 = _read_truth(path30 / "events.npz")
            if got05 is None or got30 is None:
                entry["mismatches"].append({"reason": "events.npz unreadable",
                                            "run_id": run_id05})
                continue
            joined = _join_truth(got05, got30, truth_fields)
            if joined["problems"]:
                entry["mismatches"].append({"run_id": run_id05, **joined["problems"]})
        entry["pass"] = not entry["mismatches"]
        out["sets"][role] = entry
        out["pass"] = bool(out["pass"] and entry["pass"])
    return out


def _read_truth(events_path) -> dict | None:
    path = Path(events_path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files
                    if name in ("event_ids", "vertex_m", "yield_photons", "particle",
                                "domain", "contained", "deposited_MeV", "direction_m",
                                "energy_MeV")}
    except (OSError, ValueError):
        return None


#: Continuous truth columns are compared up to float representation (the s = 0.30
#: probe sets were generated separately and store the same draws with ~1e-10
#: relative rounding); identity/enum columns are compared exactly.
_TRUTH_CONTINUOUS = ("vertex_m", "yield_photons", "deposited_MeV", "direction_m",
                     "energy_MeV", "track_length_m")
_TRUTH_RTOL = 1e-9
_TRUTH_ATOL = 1e-12


def _join_truth(left, right, fields) -> dict:
    """Compare two runs' truth arrays on the common ``event_ids`` (exact join)."""
    ids_left = np.asarray(left.get("event_ids", []), np.int64)
    ids_right = np.asarray(right.get("event_ids", []), np.int64)
    problems = {"columns": [], "events": [], "non_bitwise": [], "max_abs_diff": {}}
    if not np.array_equal(np.sort(ids_left), np.sort(ids_right)):
        problems["events"].append("event_id sets differ")
        return {"problems": problems}
    order = np.argsort(ids_right)
    index = np.searchsorted(ids_right[order], ids_left)
    for name in fields:
        if name not in left or name not in right:
            continue
        a = np.asarray(left[name])[np.argsort(ids_left)]
        b = np.asarray(right[name])[order][index]
        if name in _TRUTH_CONTINUOUS or a.dtype.kind == "f":
            fa, fb = a.astype(float), b.astype(float)
            finite = np.isfinite(fa) & np.isfinite(fb)
            if finite.any():
                problems["max_abs_diff"][name] = float(
                    np.max(np.abs(fa[finite] - fb[finite])))
            same = np.allclose(fa, fb, rtol=_TRUTH_RTOL, atol=_TRUTH_ATOL,
                               equal_nan=True)
            if not np.array_equal(np.nan_to_num(fa, nan=-1.0e30),
                                  np.nan_to_num(fb, nan=-1.0e30)) and same:
                problems["non_bitwise"].append(name)
        else:
            same = np.array_equal(np.asarray(a).astype(str),
                                  np.asarray(b).astype(str))
        if not same:
            problems["columns"].append(name)
    return {"problems": problems if (problems["columns"] or problems["events"]) else {}}


def domain_key_assertion(data_root, keys=None, *, limit=None, log=print) -> dict:
    """§3 A0: every ``dev_train`` run has ``events.npz`` with a ``domain`` key.

    A non-mixed training run would silently give empty muon/electron marginals for
    A-1b, so this is a hard A0 assertion (read-only ``events.npz`` headers only).
    """
    keys = (TRAIN_KEY_S005, TRAIN_KEY_300K) if keys is None else tuple(keys)
    out = {"registered": False, "extension": EXTENSION,
           "check": "§3 A0 dev_train events.npz domain-key assertion",
           "sets": {}, "pass": True}
    for key in keys:
        manifest = read_json(manifest_path(data_root, key), {}) or {}
        directory = manifest_path(data_root, key).parent
        rows = manifest.get("runs") or []
        missing, checked = [], 0
        for row in rows:
            if limit is not None and checked >= int(limit):
                break
            checked += 1
            path = directory / "runs" / row["run_dir"] / "events.npz"
            try:
                with np.load(path, allow_pickle=False) as data:
                    has_domain = "domain" in data.files
            except (OSError, ValueError):
                missing.append(f"{row['run_dir']}: unreadable")
                continue
            if not has_domain:
                missing.append(row["run_dir"])
        entry = {"key": key, "n_runs": len(rows), "n_checked": checked,
                 "n_missing": len(missing), "missing": missing[:10],
                 "pass": not missing}
        out["sets"][key] = entry
        out["pass"] = bool(out["pass"] and entry["pass"])
        log(f"[A0] domain-key assertion {key}: {checked} checked, "
            f"{len(missing)} missing")
    return out


def build_plan(root=None, data_root=DATA_ROOT_DEFAULT, *, inventory=None,
               pairing=True, pairing_limit=None, log=print) -> dict:
    """§3 A0 ``plan.json``: provenance, hashes, inventory, pairing, projection, entries."""
    root = _resolve_root(root)
    assert_authorised("inventory", root=root)
    started = time.time()
    inv = inventory if inventory is not None else build_inventory(root, log=log)
    config_path = root / CONFIG_REL
    generator_path = root / GEN_REL
    arms_path = root / ARMS_REL
    hashes = {}
    for name, path, expected in (("config", config_path, CONFIG_SHA256),
                                 ("generator_config", generator_path, GEN_SHA256)):
        row = {"path": str(path), "exists": path.is_file(), "expected_sha256": expected}
        row["sha256"] = lr.sha256_file(path) if path.is_file() else None
        row["pass"] = bool(row["sha256"] is not None and row["sha256"] == expected)
        hashes[name] = row
    hashes["arms_config"] = {"path": str(arms_path),
                             "sha256": lr.sha256_file(arms_path)
                             if arms_path.is_file() else None}
    runbook = root / RUNBOOK_REL
    runbook_sha = lr.sha256_file(runbook) if runbook.is_file() else None
    projection = stage_a_projection()
    cpu = cpu_projection()
    entries = entries_preflight(log=log)
    data = data_status(data_root)
    pairing_payload = (pairing_check(data_root, limit=pairing_limit, log=log)
                       if pairing else {"skipped": True, "pass": None})
    domain_keys = domain_key_assertion(data_root, log=log)
    s1 = _s1_projection(root)
    wall = round(time.time() - started, 3)
    contract = {"config_hashes": hashes,
                "runbook": {"path": RUNBOOK_REL, "sha256": runbook_sha,
                            "expected_sha256": RUNBOOK_SHA256,
                            "pass": bool(runbook_sha == RUNBOOK_SHA256)},
                "data": data, "pairing": pairing_payload,
                "domain_keys": domain_keys,
                "inventory": inv.get("counts"),
                "stage_a": projection, "cpu": cpu, "entries": entries, "s1": s1}
    stopped = []
    if not all(row["pass"] for row in hashes.values() if "pass" in row):
        stopped.append("config hash mismatch")
    if not contract["runbook"]["pass"]:
        stopped.append("runbook sha256 mismatch")
    if not data["pass"]:
        stopped.append("pinned data missing or mismatched (affected items only)")
    if not domain_keys["pass"]:
        stopped.append("a dev_train run has no domain key in events.npz (§3 A0)")
    if not projection["pass"]:
        stopped.append("S1: the Stage A projection exceeds the 5.5 h cap")
    if entries["pass"] is False:
        stopped.append("entries preflight failed")
    if s1["pass"] is False:
        stopped.append("S1 launch rule would be exceeded")
    payload = {"registered": False, "extension": EXTENSION, "mode": "A0",
               "check": "§3 A0 preflight (read-only)", "runbook": RUNBOOK_REL,
               "data_root": str(data_root),
               "code": {"git_head": lr.git_head(root), "git_status": lr.git_status(root),
                        "host": lr.host_name()},
               "hardware_software": hardware_and_software(),
               **contract, "stop": stopped, "pass": bool(not stopped),
               "wall_seconds": wall}
    if inventory is not None:
        payload["inventory_path"] = None
    return payload


# ==================================================================== A1
def _full_input_values(metrics: dict) -> dict:
    full = (metrics or {}).get("full_input") or {}
    layout = full.get("layout") or {}
    return {"vertex_resolution_m": full.get("vertex_resolution_m"),
            "token_log_charge_r2": (full.get("token_log_charge_r2") or {}).get("r2"),
            "module_gain_r2": (full.get("module_gain_r2") or {}).get("value"),
            "effective_rank": full.get("effective_rank"),
            "participation_ratio": full.get("participation_ratio"),
            "f_layout": layout.get("f_layout"), "f_event": layout.get("f_event"),
            "layout": layout or None}


def reproduction(*, root=None, inventory=None, config, gen, arms_config, geometry,
                 data_root=DATA_ROOT_DEFAULT, device="cpu", log=print) -> dict:
    """§3 A1 reproduction gate (s = 0.05 rows stop Stage A; s = 0.30 rows stop T-1)."""
    root = _resolve_root(root)
    gate = assert_authorised("reproduce", root=root)
    inv = inventory if inventory is not None else build_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    rows_s005, rows_s030 = {}, {}
    s005_pass = True
    for label, target in A1_S005.items():
        row = inv["rows"].get(label)
        if (row is None or row.get("status") != "available"
                or (not row.get("source") and not row.get("init"))):
            rows_s005[label] = {"expected_m": float(target), "got_m": None,
                                "abs_diff_m": None, "pass": False,
                                "status": "unavailable"}
            s005_pass = False
            continue
        metrics = evaluate_item_row(row, config=config, gen=gen, arms_config=arms_config,
                                    geometry=geometry, probes=probes005,
                                    data_root=data_root, device=device, with_mlp=False,
                                    log=log)
        got = _full_input_values(metrics)["vertex_resolution_m"]
        diff = None if got is None else float(abs(float(got) - float(target)))
        passed = bool(diff is not None and diff <= A1_TOLERANCE)
        s005_pass = bool(s005_pass and passed)
        rows_s005[label] = {"expected_m": float(target), "got_m": got,
                            "abs_diff_m": diff, "pass": passed, "status": "recomputed"}
    # s = 0.30 rows: the random-init floor and strong_drift9000, vertex and gain R²
    init_row = dict(inv["rows"].get("init9000") or {})
    init_row["data_setting"] = S030
    strong_row = inv["rows"].get("strong_drift9000") or {}
    s030_pass = True
    for label, row, targets in (("init9000", init_row, A1_S030["init9000"]),
                                ("strong_drift9000", strong_row,
                                 A1_S030["strong_drift9000"])):
        entry = {"vertex": {"expected_m": float(targets["vertex_resolution_m"]),
                            "got_m": None, "abs_diff_m": None, "pass": False},
                 "module_gain_r2": {"expected": float(targets["module_gain_r2"]),
                                    "got": None, "abs_diff": None, "pass": False}}
        if not row or (label != "init9000" and row.get("status") != "available"):
            entry["status"] = "unavailable"
            s030_pass = False
            rows_s030[label] = entry
            continue
        metrics = evaluate_item_row(row, config=config, gen=gen, arms_config=arms_config,
                                    geometry=geometry, probes=probes030,
                                    data_root=data_root, device=device, with_mlp=False,
                                    log=log)
        values = _full_input_values(metrics)
        got_vertex = values["vertex_resolution_m"]
        got_gain = values["module_gain_r2"]
        if got_vertex is not None:
            diff = float(abs(float(got_vertex) - float(targets["vertex_resolution_m"])))
            entry["vertex"].update({"got_m": got_vertex, "abs_diff_m": diff,
                                    "pass": bool(diff <= A1_TOLERANCE)})
        if got_gain is not None:
            diff = float(abs(float(got_gain) - float(targets["module_gain_r2"])))
            entry["module_gain_r2"].update({"got": got_gain, "abs_diff": diff,
                                            "pass": bool(diff <= A1_TOLERANCE)})
        entry["status"] = "recomputed"
        ok = bool(entry["vertex"]["pass"] and entry["module_gain_r2"]["pass"])
        s030_pass = bool(s030_pass and ok)
        rows_s030[label] = entry
    return {"registered": False, "extension": EXTENSION, "check": "§3 A1 reproduction gate",
            "runbook": RUNBOOK_REL, "tolerance": A1_TOLERANCE, "s005": rows_s005,
            "s030": rows_s030, "s005_pass": bool(s005_pass),
            "s030_pass": bool(s030_pass),
            "reproduction_pass": bool(s005_pass and s030_pass),
            "gate": {"authorised": gate["authorised"], "status": gate["status"]},
            "stops": ([s for s, ok, what in (("A1-s005", s005_pass, "all Stage A"),
                                             ("A1-s030", s030_pass,
                                              "A-2, strong-drift A-1 rows and all of T-1"))
                       if not ok for s in [what]]),
            "wall_seconds": None}


# ==================================================== shared evaluation core
def item_cell(row, config):
    """The evaluation cell of one inventory row (§2.1: its own configuration)."""
    setting = float(row.get("data_setting", S005))
    split = str(row.get("split", "within"))
    base = lr.dev_cell(config)
    if abs(setting - S005) < 1e-12 and split == "within":
        return base
    cell_id = (f"ext_dev_s005_{split}" if abs(setting - S005) < 1e-12
               else f"ext_dev_s{int(round(setting * 100)):03d}_{split}")
    return dataclasses.replace(base, cell_id=cell_id, setting=float(setting),
                               s_arm=float(setting), split=split)


@contextlib.contextmanager
def median_time_context():
    """The committed median-time token transform, installed and restored (§2.1)."""
    original = ed.install_median_time_wrapper(pe)
    try:
        yield
    finally:
        ed.remove_median_time_wrapper(pe, original)


@contextlib.contextmanager
def item_eval_context(row):
    """Per-item evaluation configuration (§2.1): arch wrapper + median wrapper."""
    config = row.get("eval") or {}
    with contextlib.ExitStack() as stack:
        if config.get("encoder_variant"):
            stack.enter_context(ea.build_arm_patch(config["encoder_variant"]))
        if config.get("median_wrapper"):
            stack.enter_context(median_time_context())
        yield


def build_item_arm(row, *, config, gen, arms_config, geometry, data_root, device="cpu",
                   log=print):
    """Build one inventory row's arm under its own evaluation configuration."""
    if row.get("status") not in ("available", "unverified (dry-run)"):
        raise RuntimeError(f"{row.get('label')}: item is {row.get('status')!r}")
    cell = item_cell(row, config)
    evaluation = row.get("eval") or {}
    if evaluation.get("anchor"):
        if not row.get("source"):
            raise RuntimeError(f"{row['label']}: anchor item has no checkpoint")
        require_pinned(data_root, keys=[DARK_BANK_KEY], context=f"{row['label']} dark bank")
        from scripts import jepa_pregrid_checks as pregrid
        arm, _beta = pregrid.build_loaded(
            row["arm"], cell, geometry, arms_config, config, gen, Path(data_root),
            row["source"], int(row["seed"]))
        log(f"[followup] built anchor arm {row['label']} (dark-bank hash checked first)")
        arm.to(device)
        arm._ext_item_cell = {"cell_id": cell.cell_id, "setting": float(cell.setting),
                              "s_arm": float(cell.s_arm), "s_p": float(cell.s_p),
                              "split": cell.split}
        return arm
    item = {"label": row["label"], "arm": row["arm"], "split": row.get("split", "within"),
            "seed": int(row["seed"]), "beta": row.get("beta"),
            "source": row.get("source")}
    arm = er.item_arm(item, config, arms_config, geometry, device=device)
    # N2: ``er.item_arm`` rebuilds the arm with the hard-coded s = 0.05 dev cell;
    # install this item's §2.1 cell on the built arm and record it.
    return apply_item_cell(arm, cell, arms_config, row)


def apply_item_cell(arm, cell, arms_config, row) -> dict:
    """Install the row's §2.1 cell on an arm built by ``er.item_arm``.

    Only ``ArmSettings`` records the cell's ``(s_arm, s_p)``; rebuild it for this
    row's cell and keep the arm's weights.  Numerically inert for every Stage A /
    T-1 arm (only ``mae_chi2`` reads ``settings.s``), but the s = 0.30 cell is
    then actually used and recorded on the row/readout, as §2.1 requires.
    """
    if arms_config is not None:
        settings = la.ArmSettings.from_config(arms_config, cell=(cell.s_arm, cell.s_p),
                                              beta=row.get("beta"))
        arm.settings = settings
    arm._ext_item_cell = {"cell_id": cell.cell_id, "setting": float(cell.setting),
                          "s_arm": float(cell.s_arm), "s_p": float(cell.s_p),
                          "split": cell.split}
    return arm


def stream_full_input_metrics(model, probes, geometry, *, split, config, device,
                              with_mlp=False, max_events=None, zero_geometry=False,
                              log=print) -> dict:
    """§3 readout protocol on the cached probe sets for one frozen encoder."""
    mask_seed = int(config["views"]["mask_seed"])
    chunk = int((config.get("evaluation") or {}).get("probe_chunk_events", 64))
    streamed = er.stream_model_forms(
        model, probes, ["full_input"], geometry, split=split, mask_seed=mask_seed,
        device=device, chunk_events=chunk, max_events=max_events,
        domain=er.PRIMARY_DOMAIN, zero_geometry=bool(zero_geometry))
    train = streamed["train"]["full_input"]
    test = streamed["test"]["full_input"]
    metrics = er.form_metrics(train, test, module_gain=streamed["module_gain"])
    out = {"full_input": metrics,
           "n_events": {"train": train.n_events, "test": test.n_events}}
    if with_mlp:
        token = er.mlp_probe(train.tokens, train.token_y, test.tokens, test.token_y,
                             seed=er.SAMPLE_SEED)
        vertex = er.mlp_probe(train.events, train.vertex, test.events, test.vertex,
                              seed=er.SAMPLE_SEED, metric="resolution")
        out["mlp_token"] = {"status": token.get("status"), "r2_mean": token.get("r2_mean")}
        out["mlp_vertex"] = {"status": vertex.get("status"),
                             "resolution": vertex.get("resolution"),
                             "resolution_median": vertex.get("resolution_median"),
                             "r2_mean": vertex.get("r2_mean")}
    return out


def evaluate_item_row(row, *, config, gen, arms_config, geometry, probes, data_root,
                      device="cpu", with_mlp=False, max_events=None, log=print) -> dict:
    """One inventory row through the exact ``20261101_readout`` protocol."""
    evaluate = row.get("eval") or {}
    with item_eval_context(row):
        arm = build_item_arm(row, config=config, gen=gen, arms_config=arms_config,
                             geometry=geometry, data_root=data_root, device=device, log=log)
        payload = stream_full_input_metrics(
            getattr(arm, "jepa", arm), probes, geometry, split=row.get("split", "within"),
            config=config, device=device, with_mlp=with_mlp, max_events=max_events,
            zero_geometry=bool(evaluate.get("zero_geometry")), log=log)
    payload.update(label=row.get("label"), family=row.get("family"),
                   checkpoint_sha256=row.get("sha256"),
                   data_setting=float(row.get("data_setting", S005)),
                   split=row.get("split", "within"), seed=int(row.get("seed", 9000)),
                   item_status=row.get("status"),
                   evaluation_cell=getattr(arm, "_ext_item_cell", None),
                   evaluation_config=dict(evaluate))
    return payload


# --------------------------------------------------- committed-value reuse §3
#: §3 A1 pins for the D2 strong-drift rows (`results/jepa/20261029_d2_partial/METRICS.md`).
STRONG_DRIFT_VERTEX = {9000: 0.233209, 9001: 0.217213}
#: Acceptance condition C4 (6 Nov 2026): the D2 rows for ijepa/lambda/stopgrad are
#: s = 0.05 evaluations of the ``ext_dev_s005_within_*`` checkpoints on the same
#: protocol as the Q1/Q2 matrix (D2's across rows equal matrix across9000/9001 and its
#: fix references equal fix9000/9001), so they are these items' committed full-input
#: vertices.  Pinned here; a METRICS.md row that disagrees is a hard error.
D2_S005_VERTEX = {
    "ijepa9000": 0.209150, "ijepa9001": 0.194992,
    "lambda9000": 0.220055, "lambda9001": 0.199603,
    "stopgrad9000": 0.219583, "stopgrad9001": 0.189377,
}
D2_METRICS_REL = "results/jepa/20261029_d2_partial/METRICS.md"
_SIGREG_DEJITTER_VARIANTS = ("within_trigger", "within_median", "across_median")
_MAE_DEJITTER_VARIANTS = ("within_trigger", "within_median", "across_trigger",
                          "across_median")
#: S1: these items have no committed full-input vertex at their evaluation setting.
NO_COMMITTED_VERTEX_REASON = ("no committed full-input vertex at this item's evaluation "
                              "setting (runbook §3 spec gap S1; recorded by the Lead in "
                              "DECISIONS)")


def strong_drift_committed_vertices(root=None) -> dict:
    """D2 strong-drift s = 0.30 vertices, parsed from METRICS.md and pinned.

    The two values are pinned here (with the file cited) and a parsed table that
    disagrees with the pins is a hard error, so a changed METRICS.md cannot pass.
    """
    root = _resolve_root(root)
    path = root / D2_METRICS_REL
    values = dict(STRONG_DRIFT_VERTEX)
    if path.is_file():
        parsed = {}
        for line in path.read_text().splitlines():
            if "strong_drift" not in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4 or cells[0] != "strong_drift":
                continue
            try:
                seed, vertex = int(cells[1]), float(cells[2])
            except ValueError:
                continue
            parsed[seed] = vertex
        for seed, vertex in parsed.items():
            if seed in values and abs(vertex - values[seed]) > 5e-7:
                raise RuntimeError(f"{path}: strong_drift seed {seed} vertex {vertex!r} "
                                   f"differs from the pinned {values[seed]!r} (§3 A1)")
        values.update({seed: vertex for seed, vertex in parsed.items()
                       if seed in (9000, 9001)})
    return values


def d2_s005_committed_vertices(root=None) -> dict:
    """C4: the D2 s = 0.05 vertices of ijepa/lambda/stopgrad, parsed and pinned."""
    root = _resolve_root(root)
    path = root / D2_METRICS_REL
    values = dict(D2_S005_VERTEX)
    if path.is_file():
        for line in path.read_text().splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 4 or cells[0] not in ("ijepa", "lambda", "stopgrad"):
                continue
            try:
                label, vertex = f"{cells[0]}{int(cells[1])}", float(cells[2])
            except ValueError:
                continue
            if label in values and abs(vertex - values[label]) > 5e-7:
                raise RuntimeError(f"{path}: {label} vertex {vertex!r} differs from the "
                                   f"pinned {values[label]!r} (acceptance C4)")
    return values


def _dejitter_committed_vertices(path, reading, prefix, variants) -> dict:
    """Committed full-input vertices from a P-5 / M-f readings block (<variant>/s<seed>)."""
    payload = read_json(path, {}) or {}
    out = {}
    for key, row in ((payload.get(reading) or {}).get("runs") or {}).items():
        try:
            variant, seed_text = str(key).split("/")
            seed = int(seed_text.lstrip("s"))
        except ValueError:
            continue
        if variant not in variants:
            continue
        vertex = (row or {}).get("vertex_resolution_m")
        if vertex is not None:
            out[f"{prefix}{variant}{seed}"] = float(vertex)
    return out


def _h12_label(label) -> str:
    """``geomblind_9000`` (h12 runs.json) -> ``geomblind9000`` (inventory label)."""
    label = str(label)
    for prefix in ("geomblind", "frozen_target"):
        if label.startswith(prefix + "_"):
            return prefix + label.rsplit("_", 1)[1]
    return label


def committed_vertex_map(root=None) -> dict:
    """Committed full-input photon vertex per label, for the §3 reproduction rule."""
    root = _resolve_root(root)
    values = {}
    matrix = read_json(root / "results/jepa/20261101_readout/matrix.json", {}) or {}
    for label, item in (matrix.get("items") or {}).items():
        full = ((item.get("forms") or {}).get("full_input") or {})
        if full.get("vertex_resolution_m") is not None:
            values[label] = float(full["vertex_resolution_m"])
    nonlinear = read_json(root / "results/jepa/20261101_readout/nonlinear.json", {}) or {}
    for label, item in (nonlinear.get("items") or {}).items():
        full = ((item.get("forms") or {}).get("full_input") or {})
        ridge = full.get("vertex_ridge") or {}
        if ridge.get("resolution") is not None and label not in values:
            values[label] = float(ridge["resolution"])
    h12 = read_json(root / "results/jepa/20261101_h12/runs.json", {}) or {}
    for label, run in (h12.get("runs") or {}).items():
        full = run.get("full_input") or {}
        if full.get("vertex_resolution_m") is not None:
            values[_h12_label(label)] = float(full["vertex_resolution_m"])
    static = read_json(root / "results/jepa/20261102_static/runs.json", {}) or {}
    for seed, run in (static.get("runs") or {}).items():
        full = run.get("full_input") or {}
        if full.get("vertex_resolution_m") is not None:
            values[f"static_target{int(seed)}"] = float(full["vertex_resolution_m"])
    arch = read_json(root / "results/jepa/20261102_arch/runs.json", {}) or {}
    for key, run in (arch.get("runs") or {}).items():
        variant, seed = key.split("__")
        values[f"{variant}{int(seed)}"] = float((run.get("full_input") or {}).get(
            "vertex_m", float("nan")))
    scale = read_json(root / "results/jepa/20261102_scale/runs.json", {}) or {}
    for seed, run in (scale.get("runs") or {}).items():
        values[f"scale300k{int(seed)}"] = float((run.get("full_input") or {}).get(
            "vertex_m", float("nan")))
    # the six SIGReg de-jitter retrains (P-5) and eight MAE de-jitter checkpoints (M-f)
    values.update(_dejitter_committed_vertices(
        root / "results/jepa/20261030_dejitter/readings.json", "P-5", "sigreg_",
        _SIGREG_DEJITTER_VARIANTS))
    values.update(_dejitter_committed_vertices(
        root / "results/jepa/20261031_mae_dejitter/readings.json", "M-f", "mae_",
        _MAE_DEJITTER_VARIANTS))
    # the two strong-drift s = 0.30 rows (D2 METRICS.md, pinned above)
    for seed, vertex in strong_drift_committed_vertices(root).items():
        values[f"strong_drift{seed}"] = float(vertex)
    # C4: the six D2 s = 0.05 rows (ijepa/lambda/stopgrad), pinned above
    values.update(d2_s005_committed_vertices(root))
    return {label: value for label, value in values.items() if np.isfinite(value)}


def check_item_reproduction(label, got_vertex, committed, *, tolerance=A1_TOLERANCE) -> dict:
    """§3 A1: a recomputed item must reproduce its own committed full-input vertex.

    Labels with no committed full-input vertex at their evaluation setting (the
    runbook spec gap S1, narrowed by acceptance C4: anc100 and the T-1 checkpoints) are
    reported explicitly as ``not_applicable`` — never as a silent pass — and a
    label with a committed value is marked ``unavailable`` on a mismatch.
    """
    target = committed.get(label)
    if target is None:
        return {"checked": False, "pass": None, "status": "not_applicable",
                "reason": NO_COMMITTED_VERTEX_REASON, "expected_m": None,
                "got_m": got_vertex, "abs_diff_m": None}
    if got_vertex is None:
        return {"checked": True, "pass": False, "status": "unavailable",
                "reason": "the item produced no recomputed full-input vertex",
                "expected_m": float(target), "got_m": None, "abs_diff_m": None}
    diff = float(abs(float(got_vertex) - float(target)))
    passed = bool(diff <= float(tolerance))
    return {"checked": True, "pass": passed,
            "status": "pass" if passed else "unavailable",
            "expected_m": float(target), "got_m": float(got_vertex),
            "abs_diff_m": diff}


def committed_readout_values(root=None) -> dict:
    """Committed ``f_layout``/``f_event``/vertex/token-R²/effective-rank values (§3 A-1)."""
    root = _resolve_root(root)
    values = {}

    def put(label, full, layout=None):
        entry = {"vertex_resolution_m": full.get("vertex_resolution_m"),
                 "token_log_charge_r2": (full.get("token_log_charge_r2") or {}).get("r2")
                 if isinstance(full.get("token_log_charge_r2"), dict)
                 else full.get("token_log_charge_r2"),
                 "effective_rank": full.get("effective_rank"),
                 "module_gain_r2": (full.get("module_gain_r2") or {}).get("value")
                 if isinstance(full.get("module_gain_r2"), dict)
                 else full.get("module_gain_r2"),
                 "f_layout": (layout or {}).get("f_layout", full.get("f_layout")),
                 "f_event": (layout or {}).get("f_event", full.get("f_event"))}
        values[label] = entry

    matrix = read_json(root / "results/jepa/20261101_readout/matrix.json", {}) or {}
    for label, item in (matrix.get("items") or {}).items():
        full = (item.get("forms") or {}).get("full_input") or {}
        put(label, {**full, "layout": full.get("layout")})
    nonlinear = read_json(root / "results/jepa/20261101_readout/nonlinear.json", {}) or {}
    for label, item in (nonlinear.get("items") or {}).items():
        full = (item.get("forms") or {}).get("full_input") or {}
        entry = values.setdefault(label, {})
        entry["mlp_token_r2"] = (full.get("token_log_charge_mlp") or {}).get("r2_mean")
        entry["mlp_vertex_resolution_m"] = (full.get("vertex_mlp") or {}).get("resolution")
    h12 = read_json(root / "results/jepa/20261101_h12/runs.json", {}) or {}
    for label, run in (h12.get("runs") or {}).items():
        full = run.get("full_input") or {}
        put(_h12_label(label), full, full.get("layout"))
    static = read_json(root / "results/jepa/20261102_static/runs.json", {}) or {}
    for seed, run in (static.get("runs") or {}).items():
        put(f"static_target{int(seed)}", run.get("full_input") or {},
            run.get("full_input") or {})
    arch = read_json(root / "results/jepa/20261102_arch/runs.json", {}) or {}
    for key, run in (arch.get("runs") or {}).items():
        variant, seed = key.split("__")
        full = run.get("full_input") or {}
        put(f"{variant}{int(seed)}",
            {"vertex_resolution_m": full.get("vertex_m"),
             "token_log_charge_r2": full.get("token_charge_r2"),
             "effective_rank": run.get("eff_rank")},
            {"f_layout": run.get("f_layout"), "f_event": run.get("f_event")})
    scale = read_json(root / "results/jepa/20261102_scale/runs.json", {}) or {}
    for seed, run in (scale.get("runs") or {}).items():
        full = run.get("full_input") or {}
        put(f"scale300k{int(seed)}",
            {"vertex_resolution_m": full.get("vertex_m"),
             "token_log_charge_r2": full.get("token_charge_r2"),
             "effective_rank": run.get("eff_rank")},
            {"f_layout": run.get("f_layout"), "f_event": run.get("f_event")})
    mae = read_json(root / "results/jepa/20261031_mae_dejitter/readings.json", {}) or {}
    for key, run in ((mae.get("M-f") or {}).get("runs") or {}).items():
        variant, seed_text = key.split("/")
        label = f"mae_{variant}{int(seed_text.lstrip('s'))}"
        full = {"vertex_resolution_m": run.get("vertex_resolution_m"),
                "token_log_charge_r2": run.get("token_log_charge_r2"),
                "effective_rank": (run.get("event_embedding") or {}).get("effective_rank"),
                "module_gain_r2": run.get("module_gain_run_mean_r2")}
        put(label, full)
    view_runs = root / "results/jepa/20261103_view/runs.json"
    if view_runs.is_file():
        view = read_json(view_runs, {}) or {}
        for key, run in (view.get("runs") or {}).items():
            split, seed = key.split("__")
            full = run.get("full_input") or {}
            put(f"view_{split}{int(seed)}", full)
    return values


def _seed_pair_groups():
    """(family prefix, seed) label groups for the fixed §3 A-1 classes."""
    prefixes = (
        "old", "fix", "across",
        "ijepa", "lambda", "stopgrad", "strong_drift",
        "sigreg_within_trigger", "sigreg_within_median", "sigreg_across_median",
        "mae_within_trigger", "mae_within_median", "mae_across_trigger",
        "mae_across_median",
        "geomblind", "frozen_target", "static_target",
        "scale300k", "window128", "window32", "pred4",
        "view_within", "view_across", "view_module_parity", "view_random_half",
    )
    return {prefix: (f"{prefix}9000", f"{prefix}9001") for prefix in prefixes}


def _classify_event(f_event_by_seed: dict) -> dict:
    """§3 A-1 fixed classes: event-dependent / event-invariant / intermediate."""
    seeds = sorted(int(seed) for seed in f_event_by_seed)
    values = [f_event_by_seed[seed] for seed in seeds]
    if len(seeds) < 2 or any(value is None for value in values):
        return {"class": "single-seed (not classified)",
                "f_event": {str(seed): f_event_by_seed[seed] for seed in seeds},
                "seeds": seeds}
    if all(float(value) >= 0.01 for value in values):
        verdict = "event-dependent"
    elif all(float(value) <= 1e-3 for value in values):
        verdict = "event-invariant"
    else:
        verdict = "intermediate"
    return {"class": verdict,
            "f_event": {str(seed): f_event_by_seed[seed] for seed in seeds},
            "seeds": seeds}


def _spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3:
        return None

    def ranks(values):
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), float)
        ranks[order] = np.arange(1, len(values) + 1, dtype=float)
        return ranks

    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 0 else None


def _a1_readings(rows: dict) -> dict:
    """The §3 A-1 J-a/J-b/J-c readings over the merged row table."""
    classes = {}
    for label, row in rows.items():
        if row.get("status") != "available":
            continue
        f_event = {int(seed): value for seed, value
                   in (row.get("f_event") or {}).items()}
        if not f_event:
            continue
        classes[label] = {"f_event": f_event,
                          "f_layout": row.get("f_layout"),
                          "vertex_m": row.get("vertex_resolution_m"),
                          "family": row.get("family")}
    groups = {}
    for prefix, (label9000, label9001) in _seed_pair_groups().items():
        present = {seed: classes[label]["f_event"].get(seed)
                   for seed, label in ((9000, label9000), (9001, label9001))
                   if label in classes}
        if not present:
            continue
        groups[prefix] = _classify_event(present)
    singles = {label: {"class": "single-seed (not classified)",
                       "f_event": classes[label]["f_event"]}
               for label in ("init9000", "mae9000", "anc100", "strong_drift_init")
               if label in classes}
    j_a_sigreg = {prefix: groups[prefix]["class"] for prefix in
                  ("ijepa", "lambda", "stopgrad", "strong_drift") if prefix in groups}
    if "anc100" in singles:
        j_a_sigreg["anc100"] = singles["anc100"]["class"]
    j_b_labels = ["init9000", "frozen_target9000", "frozen_target9001",
                  "static_target9000", "static_target9001"] + [
        f"mae_{variant}{seed}" for variant in ("within_trigger", "within_median",
                                              "across_trigger", "across_median")
        for seed in (9000, 9001)]
    j_b = {label: {key: classes[label].get(key) for key in ("f_event", "f_layout",
                                                           "vertex_m")}
           for label in j_b_labels if label in classes}
    # J-c population: s = 0.05, trigger convention, full-geometry trained checkpoints
    members, xs, ys = [], [], []
    for label in A1_COMPUTE_LABELS + A1_MLP_FILL_LABELS + (
            "init9000", "old9000", "old9001", "fix9000", "fix9001", "mae9000",
            "across9000", "across9001"):
        row = rows.get(label)
        if row is None or row.get("status") != "available":
            continue
        # the fixed J-c population excludes strong_drift, median-time, E2 and anc100
        if label == "anc100" or label in ("geomblind9000", "geomblind9001"):
            continue
        if label.startswith("strong_drift"):
            continue
        if (row.get("eval") or {}).get("median_wrapper"):
            continue
        if float(row.get("data_setting") or S005) != S005:
            continue
        values = [value for value in (row.get("f_event") or {}).values()
                  if value is not None]
        if not values or row.get("vertex_resolution_m") is None:
            continue
        members.append(label)
        xs.append(float(np.mean(values)))
        ys.append(float(row["vertex_resolution_m"]))
    j_c = {"members": members, "n": len(members),
           "spearman_f_event_vs_vertex": _spearman(xs, ys), "p_value": None,
           "population": "s = 0.05, trigger convention, full-geometry trained "
                         "checkpoints; strong_drift, median-time, E2 and anc100 excluded"}
    return {"groups": groups, "singles": singles, "classes": {
        label: {"class": data.get("class", "single-seed (not classified)"),
                "f_event": data.get("f_event")} for label, data in
        list(groups.items()) + list(singles.items())},
        "J-a": j_a_sigreg, "J-b": j_b, "J-c": j_c,
        "thresholds": {"event_dependent": "f_event >= 0.01 in both seeds",
                       "event_invariant": "f_event <= 1e-3 in both seeds",
                       "otherwise": "intermediate"}}


def readout_a1(*, root=None, inventory=None, config, gen, arms_config, geometry,
               data_root=DATA_ROOT_DEFAULT, device="cpu", max_events=None, labels=None,
               log=print) -> dict:
    """§3 A-1 event dependence: reuse the committed rows, compute the missing ones."""
    root = _resolve_root(root)
    gate = assert_authorised("readout", root=root)
    inv = inventory if inventory is not None else build_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    committed = committed_readout_values(root)
    vertex_targets = committed_vertex_map(root)
    rows, unavailable = {}, []
    for label, values in committed.items():
        row = inv["rows"].get(label) or {}
        seed = int(row.get("seed", 9000))
        f_event = values.get("f_event")
        rows[label] = {"label": label, "status": "available", "reused": True,
                       "vertex_resolution_m": values.get("vertex_resolution_m"),
                       "token_log_charge_r2": values.get("token_log_charge_r2"),
                       "effective_rank": values.get("effective_rank"),
                       "module_gain_r2": values.get("module_gain_r2"),
                       "f_layout": values.get("f_layout"),
                       "f_event": ({seed: f_event} if f_event is not None else {}),
                       "mlp_token_r2": values.get("mlp_token_r2"),
                       "mlp_vertex_resolution_m": values.get("mlp_vertex_resolution_m"),
                       "family": row.get("family"),
                       "data_setting": row.get("data_setting"),
                       "eval": row.get("eval")}
    for label in (list(labels) if labels is not None
                  else list(A1_COMPUTE_LABELS) + list(A1_MLP_FILL_LABELS)):
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            unavailable.append({"label": label, "reason": "inventory row unavailable"})
            continue
        probes = probes030 if float(row.get("data_setting", S005)) != S005 else probes005
        try:
            metrics = evaluate_item_row(row, config=config, gen=gen,
                                        arms_config=arms_config, geometry=geometry,
                                        probes=probes, data_root=data_root,
                                        device=device, with_mlp=True,
                                        max_events=max_events, log=log)
        except SystemExit:
            raise
        except Exception as problem:  # a recomputed item that cannot run is unavailable
            unavailable.append({"label": label,
                                "reason": f"{type(problem).__name__}: {problem}"})
            continue
        values = _full_input_values(metrics)
        reproduction_check = check_item_reproduction(label, values["vertex_resolution_m"],
                                                     vertex_targets)
        if reproduction_check["checked"] and not reproduction_check["pass"]:
            unavailable.append({"label": label, "reason": "own reproduction mismatch",
                                "reproduction": reproduction_check})
            continue
        seed = int(row["seed"])
        merged = rows.setdefault(label, {"label": label, "reused": False})
        merged.update(status="available", family=row.get("family"),
                      data_setting=float(row.get("data_setting", S005)),
                      eval=row.get("eval"),
                      vertex_resolution_m=values["vertex_resolution_m"],
                      token_log_charge_r2=values["token_log_charge_r2"],
                      effective_rank=values["effective_rank"],
                      module_gain_r2=values["module_gain_r2"],
                      f_layout=values["f_layout"],
                      f_event={seed: values["f_event"]},
                      mlp_token_r2=(metrics.get("mlp_token") or {}).get("r2_mean"),
                      mlp_vertex_resolution_m=(metrics.get("mlp_vertex") or {}).get("resolution"),
                      reproduction=reproduction_check)
    readings = _a1_readings(rows)
    reproduction = {"rule": "§3 A1: each recomputed item must reproduce its own committed "
                            "full-input vertex within 1e-3 m; a mismatch marks the row "
                            "unavailable",
                    "checked": {}, "not_applicable": []}
    for label, row in rows.items():
        record = row.get("reproduction")
        if not record:
            continue
        if record.get("status") == "not_applicable":
            reproduction["not_applicable"].append(label)
        else:
            reproduction["checked"][label] = record
    reproduction["reason_not_applicable"] = NO_COMMITTED_VERTEX_REASON
    return {"registered": False, "extension": EXTENSION,
            "check": "§3 A-1 event dependence", "runbook": RUNBOOK_REL,
            "cell": "photon probe-test, 20261101_readout protocol",
            "rows": rows, "unavailable": unavailable, "readings": readings,
            "reproduction": reproduction,
            "n_computed_expected": len(A1_COMPUTE_LABELS) + len(A1_MLP_FILL_LABELS),
            "gate": {"authorised": gate["authorised"], "status": gate["status"]}}

# =============================================================== A-1b §3
M_GROUPS = ("all", "photon", "muon", "electron")
M_STATS = ("hit_rate", "lnq_mean", "lnq_sd", "time_mean", "time_sd")
M_FEATURE_NAMES = tuple(f"{group}__{stat}" for group in M_GROUPS for stat in M_STATS)
A1B_READING_ORDER = ("input-statistics code", "fixed-feature code (geometry or probe-test raw)",
                     "not determined by fixed per-PMT features", "intermediate")


def read_dev_marginals(config, geometry, data_root, *, key=None, expected_sha256=None,
                       time_scale_ns=100.0, max_runs=None, log=print) -> dict:
    """§3 A-1b group M: read-only per-PMT marginals of the **training distribution**.

    Opens no ``ensure_dataset`` path: after the read-only manifest-hash check it reads
    ``hits.npz``/``events.npz`` directly (§3 A-1b forbids the h12 collector here).
    """
    key = TRAIN_KEY_S005 if key is None else str(key)
    expected = PINNED_DATA[key] if expected_sha256 is None else str(expected_sha256)
    require_pinned(data_root, keys=[key], pins={key: expected},
                   context=f"dev_train marginals ({key})")
    manifest = read_json(manifest_path(data_root, key), {}) or {}
    directory = manifest_path(data_root, key).parent
    n_pmt = int(geometry.n_pmt)
    counts = {group: np.zeros(n_pmt, np.float64) for group in M_GROUPS}
    sum_q = {group: np.zeros(n_pmt, np.float64) for group in M_GROUPS}
    sum_q2 = {group: np.zeros(n_pmt, np.float64) for group in M_GROUPS}
    sum_t = {group: np.zeros(n_pmt, np.float64) for group in M_GROUPS}
    sum_t2 = {group: np.zeros(n_pmt, np.float64) for group in M_GROUPS}
    n_events = {group: 0 for group in M_GROUPS}
    runs = manifest.get("runs") or []
    if max_runs is not None:
        runs = runs[:int(max_runs)]
    floor = float(pe.FeatureScaling.from_geometry(geometry).charge_floor_pe)
    domains_present = set()
    for row in runs:
        run_dir = directory / "runs" / row["run_dir"]
        truth = _read_truth(run_dir / "events.npz") or {}
        store = lb.read_sparse_hits(run_dir / "hits.npz")
        labels = np.asarray(truth.get("domain", np.full(len(store.event_ids), "photon"))).astype(str)
        by_event = {}
        for index, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
            span = slice(int(store.offsets[index]), int(store.offsets[index + 1]))
            pmt = np.asarray(store.pmt_id[span], np.int64)
            if pmt.size:
                charge = np.asarray(store.charge_pe[span], np.float64)
                log_q = np.log(np.maximum(charge, floor))
                time = np.asarray(store.time_ns[span], np.float64) / float(time_scale_ns)
                by_event[int(event_id)] = (pmt, log_q, time)
        for event_id, (pmt, log_q, time) in by_event.items():
            label = labels[np.searchsorted(np.asarray(store.event_ids, np.int64), event_id)] \
                if labels.size == len(store.event_ids) else "photon"
            group_names = ("all", label) if label in M_GROUPS[1:] else ("all",)
            domains_present.add(label)
            for group in group_names:
                counts[group][pmt] += 1.0
                sum_q[group][pmt] += log_q
                sum_q2[group][pmt] += log_q * log_q
                sum_t[group][pmt] += time
                sum_t2[group][pmt] += time * time
        for group in M_GROUPS:
            n_events[group] += int(len(by_event))
    groups = {}
    for group in M_GROUPS:
        count = counts[group]
        safe = np.maximum(count, 1.0)
        mean_q = sum_q[group] / safe
        var_q = np.maximum(sum_q2[group] / safe - mean_q * mean_q, 0.0)
        mean_t = sum_t[group] / safe
        var_t = np.maximum(sum_t2[group] / safe - mean_t * mean_t, 0.0)
        groups[group] = {"hit_rate": count / max(n_events[group], 1),
                         "lnq_mean": np.where(count > 0, mean_q, np.nan),
                         "lnq_sd": np.where(count > 1,
                                            np.sqrt(var_q * count / np.maximum(count - 1, 1)),
                                            np.nan),
                         "time_mean": np.where(count > 0, mean_t, np.nan),
                         "time_sd": np.where(count > 1,
                                             np.sqrt(var_t * count / np.maximum(count - 1, 1)),
                                             np.nan),
                         "count": count}
    return {"registered": False, "extension": EXTENSION,
            "check": "§3 A-1b training-distribution per-PMT marginals",
            "key": key, "manifest_sha256": expected,
            "n_pmt": n_pmt, "n_events": n_events,
            "domains_present": sorted(domains_present),
            "feature_names": list(M_FEATURE_NAMES), "groups": groups,
            "registered_false": bool(manifest.get("registered") is False)}


def marginal_matrix(marginals) -> np.ndarray:
    """The 20-feature M matrix ``(n_pmt, 20)`` (NaN -> 0 for PMTs with no hits)."""
    columns = []
    for group in M_GROUPS:
        block = marginals["groups"][group]
        for stat in M_STATS:
            columns.append(np.nan_to_num(np.asarray(block[stat], float), nan=0.0))
    return np.stack(columns, axis=1)


def collect_probe_test_features(config, geometry, dataset, *, max_label_events=A1B_LABEL_EVENTS,
                                chunk_events=8, device="cpu", log=print) -> dict:
    """§3 A-1b group C: per-PMT fixed features over the probe-test events (read-only).

    Tokenisation only, no encoder, no ``ext_h12.collect_per_pmt_features`` (that
    helper reaches ``load_probes``/``ensure_dataset``).
    """
    import torch
    device = device if isinstance(device, torch.device) else torch.device(device)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    n_pmt = int(geometry.n_pmt)
    n_features = int(pe.N_FEATURES)
    hit_count = np.zeros(n_pmt, np.float64)
    charge_sum = np.zeros(n_pmt, np.float64)
    raw_sum = np.zeros((n_pmt, n_features), np.float64)
    n_events = 0
    for row in dataset.runs:
        if n_events >= int(max_label_events):
            break
        source = dataset.directory / "runs" / row["run_dir"]
        ids = lr._run_event_ids(source)
        allowed = lr._domain_event_ids(source, er.PRIMARY_DOMAIN)
        if allowed is not None:
            ids = np.intersect1d(ids, allowed)
        if ids.size == 0:
            continue
        ids = ids[: int(max_label_events) - n_events]
        for start in range(0, int(ids.size), int(chunk_events)):
            batch = lr._probe_batch(source, geometry, split="within",
                                    mask_seed=int(config["views"]["mask_seed"]),
                                    occupancy="natural", scaling=scaling,
                                    event_ids=ids[start:start + int(chunk_events)])
            if batch.event_ids.size == 0:
                continue
            hit = np.asarray(batch.hit_flags, bool)
            log_charge = np.asarray(batch.log_charge, float)
            raw = batch.features.detach().to(torch.float32).cpu().numpy()
            event_index, pmt_index = np.nonzero(hit)
            np.add.at(hit_count, pmt_index, 1.0)
            np.add.at(charge_sum, pmt_index, log_charge[event_index, pmt_index])
            np.add.at(raw_sum, pmt_index, raw[event_index, pmt_index])
            n_events += int(batch.event_ids.size)
    return {"hit_count": hit_count, "charge_sum": charge_sum, "raw_sum": raw_sum,
            "n_events": int(n_events),
            "positions": pe.geometry_features(geometry, scaling)[:, :3].astype(np.float64),
            "raw_feature_names": list(pe.FEATURES)}


def _pmt_probe_split(rows, seed=0, fraction=0.8):
    present = np.flatnonzero(np.bincount(np.asarray(rows["pmts"], np.int64),
                                         minlength=int(rows["n_pmt"])) > 0)
    rng = np.random.default_rng(int(seed))
    order = present[rng.permutation(len(present))]
    cut = int(float(fraction) * len(order))
    is_train = np.isin(present, order[:cut])
    return present, order, cut, is_train


def _probe_one_group(x, m_labels, present, order, cut, is_train, methods, seed=0) -> dict:
    entry = {}
    validation = max(1, int(round(0.1 * cut)))
    fit_mask = np.isin(present, order[:cut - validation])
    val_mask = np.isin(present, order[cut - validation:])
    test_mask = ~is_train
    if "linear" in methods:
        entry["linear"] = eh._record(eh._multioutput_ridge(
            x[is_train], m_labels[is_train], x[test_mask], m_labels[test_mask]))
    if "mlp" in methods:
        prediction, epochs_run = eh._mlp_fit(x[fit_mask], m_labels[fit_mask],
                                             x[val_mask], m_labels[val_mask],
                                             x[test_mask], seed=int(seed))
        record = eh._r2_mean(m_labels[test_mask], prediction)
        record.update(epochs_run=epochs_run, n_train=int(fit_mask.sum()),
                      n_val=int(val_mask.sum()), n_test=int(test_mask.sum()))
        entry["mlp"] = eh._record(record, n_train=record["n_train"],
                                  n_test=record["n_test"])
        entry["mlp"]["n_val"] = record["n_val"]
    if "knn" in methods:
        prediction = eh._knn_predict(x[is_train], m_labels[is_train], x[test_mask],
                                     k=eh.PROBE_K)
        record = eh._r2_mean(m_labels[test_mask], prediction)
        entry["knn5"] = eh._record(record, n_train=int(is_train.sum()),
                                   n_test=int(test_mask.sum()))
        entry["knn5"]["k"] = eh.PROBE_K
    return entry


def a1b_probes(rows, geometry, marginals, c_features, *, methods=eh.CODE_ORIGIN_METHODS,
               seed=0, requested=("M", "C", "All", "singles"), log=print) -> dict:
    """§3 A-1b probes: groups M, C, All and each single M statistic (ridge/MLP/kNN-5).

    ``requested`` restricts the groups (the committed 0.1043 reproduction needs only
    ``C``, so it does not read the 100k-event dev_train marginals — N11).
    """
    methods = tuple(methods)
    requested = set(requested)
    present, order, cut, is_train = _pmt_probe_split(rows, seed=seed)
    _, m_labels, n_present = eh._group_rows_by_pmt(np.asarray(rows["positions"], float),
                                                   np.asarray(rows["y"], float),
                                                   np.asarray(rows["pmts"], np.int64),
                                                   int(rows["n_pmt"]))
    c_groups = eh.code_origin_feature_groups(c_features, geometry, present)
    c_all = np.asarray(c_groups["all"], float)
    m_matrix = None
    if {"M", "All", "singles"} & requested:
        if marginals is None:
            raise ValueError("a1b_probes needs marginals for the M/All groups")
        m_matrix = marginal_matrix(marginals)[present]
    blocks = {}
    if "M" in requested:
        blocks["M"] = m_matrix
    if "C" in requested:
        blocks["C"] = c_all
    if "All" in requested:
        blocks["All"] = np.concatenate([m_matrix, c_all], axis=1)
    if "singles" in requested:
        for index, name in enumerate(M_FEATURE_NAMES):
            blocks[f"M::{name}"] = m_matrix[:, index:index + 1]
    results = {}
    for name, values in blocks.items():
        results[name] = _probe_one_group(np.asarray(values, float), m_labels, present,
                                         order, cut, is_train, methods, seed=seed)
        log("[A-1b] " + str(name) + ": " + ", ".join(
            f"{method}={cell.get('r2')}" for method, cell in results[name].items()))
    return {"split": {"by": "PMT", "seed": int(seed), "n_pmt": int(n_present),
                      "n_pmt_train": int(is_train.sum()),
                      "n_pmt_test": int((~is_train).sum()),
                      "methods": list(methods)},
            "groups": results, "n_features": {name: int(values.shape[1])
                                              for name, values in blocks.items()}}


def a1b_class(m_mlp, all_mlp) -> str:
    """§3 A-1b readings, checked in the listed order (first match applies)."""
    if m_mlp is None or all_mlp is None:
        return "unavailable"
    if float(m_mlp) >= 0.7:
        return "input-statistics code"
    if float(all_mlp) >= 0.7:
        return "fixed-feature code (geometry or probe-test raw)"
    if float(all_mlp) <= 0.5:
        return "not determined by fixed per-PMT features"
    return "intermediate"


def _item_label_rows(row, *, config, gen, arms_config, geometry, test_dataset, data_root,
                     device, tokens_per_event=32, log=print) -> dict:
    with item_eval_context(row):
        arm = build_item_arm(row, config=config, gen=gen, arms_config=arms_config,
                             geometry=geometry, data_root=data_root, device=device, log=log)
        rows = collect_label_rows(getattr(arm, "jepa", arm), test_dataset, config, geometry,
                                  split=row.get("split", "within"),
                                  tokens_per_event=tokens_per_event, device=device, log=log)
    return rows


def collect_label_rows(model, dataset, config, geometry, *, split="within",
                       max_label_events=A1B_LABEL_EVENTS, chunk_events=8,
                       tokens_per_event=32, device="cpu", log=print) -> dict:
    """§3 A-1b general label-row collector (read-only re-implementation).

    The committed E0/E1 core: ``lr._probe_batch(split, mask_seed = views.mask_seed)``,
    ``arm.jepa.encode(features, hidden_mask=~batch.target)``, 32 tokens from
    ``hidden & hit`` per event (sampling seed ``er.SAMPLE_SEED``), over the photon
    probe-test events.  It never touches ``ext_h12._collect_label_rows`` (which is
    hard-wired to fix9000 and ``load_probes``).
    """
    import torch
    device = device if isinstance(device, torch.device) else torch.device(device)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    positions = pe.geometry_features(geometry, scaling)[:, :3].astype(np.float64)
    mask_seed = int(config["views"]["mask_seed"])
    rng = np.random.default_rng(er.SAMPLE_SEED)
    x_rows, y_rows, pmt_rows, event_rows = [], [], [], []
    n_events = 0
    event_offset = 0
    layout = er.layout_state(geometry.n_pmt, 0)
    for row in dataset.runs:
        if n_events >= int(max_label_events):
            break
        source = dataset.directory / "runs" / row["run_dir"]
        ids = lr._run_event_ids(source)
        allowed = lr._domain_event_ids(source, er.PRIMARY_DOMAIN)
        if allowed is not None:
            ids = np.intersect1d(ids, allowed)
        if ids.size == 0:
            continue
        ids = ids[: int(max_label_events) - n_events]
        for start in range(0, int(ids.size), int(chunk_events)):
            batch = lr._probe_batch(source, geometry, split=split, mask_seed=mask_seed,
                                    occupancy="natural", scaling=scaling,
                                    event_ids=ids[start:start + int(chunk_events)])
            if batch.event_ids.size == 0:
                continue
            batch = lr._to_device(batch, device)
            with torch.no_grad():
                tokens = model.encode(batch.features)
                labels = model.encode(batch.features, hidden_mask=~batch.target)
            token_np = tokens.detach().to(torch.float32).cpu().numpy()
            if layout["dim"] == 0:
                layout = er.layout_state(geometry.n_pmt, token_np.shape[-1])
            er.layout_add(layout, token_np)
            label_np = labels.detach().to(torch.float32).cpu().numpy()
            hidden = np.asarray(batch.hidden.detach().cpu(), bool)
            hit = np.asarray(batch.hit_flags, bool)
            candidate = hidden & hit
            for event_index in range(candidate.shape[0]):
                positions_hit = np.flatnonzero(candidate[event_index])
                if positions_hit.size == 0:
                    continue
                take = min(int(tokens_per_event), int(positions_hit.size))
                pick = (rng.choice(positions_hit, size=take, replace=False)
                        if positions_hit.size > take else positions_hit)
                for pmt in pick:
                    x_rows.append(positions[pmt])
                    y_rows.append(label_np[event_index, pmt])
                    pmt_rows.append(int(pmt))
                    event_rows.append(int(event_offset))
            event_offset += int(batch.event_ids.size)
            n_events += int(batch.event_ids.size)
            if n_events % 200 == 0:
                log(f"[A-1b] {n_events} probe events, {len(y_rows)} label rows")
    if not y_rows:
        raise RuntimeError("A-1b label collector produced no rows")
    y = np.asarray(y_rows, np.float64)
    return {"x": np.asarray(x_rows, np.float64), "y": y,
            "pmts": np.asarray(pmt_rows, np.int64),
            "events": np.asarray(event_rows, np.int64), "positions": positions,
            "n_pmt": int(geometry.n_pmt), "n_events": int(n_events), "layout": layout,
            "label_dim": int(y.shape[1]),
            "sampling": {"max_label_events": int(max_label_events),
                         "tokens_per_event": int(tokens_per_event),
                         "seed": er.SAMPLE_SEED}}


def verify_label_collector(rows, reference) -> dict:
    """§3 A-1b: the collector must reproduce the committed fix9000 rows first."""
    if reference is None:
        return {"checked": False, "pass": None,
                "reason": "committed label cache is absent"}
    checks = {}
    for name in ("y", "pmts", "events"):
        if name not in reference:
            checks[name] = {"checked": False}
            continue
        want = np.asarray(reference[name])
        got = np.asarray(rows[name])
        checks[name] = {"checked": True, "shapes": [list(got.shape), list(want.shape)],
                        "equal": bool(got.shape == want.shape
                                      and np.array_equal(got, want))}
    if checks["y"].get("equal") and checks["pmts"].get("equal") and checks["events"].get("equal"):
        return {"checked": True, "pass": True, "checks": checks}
    return {"checked": True, "pass": False, "checks": checks,
            "reason": "the general label-row collector does not reproduce the committed "
                      "fix9000 rows; A-1b is unavailable (§3 A-1b)"}


def readout_a1b(*, root=None, inventory=None, config, gen, arms_config, geometry,
                data_root=DATA_ROOT_DEFAULT, device="cpu", log=print) -> dict:
    """§3 A-1b origin of the per-PMT code (no training)."""
    root = _resolve_root(root)
    gate = assert_authorised("readout", root=root)
    inv = inventory if inventory is not None else build_inventory(root, log=log)
    probes = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    test_dataset = probes["test"]
    payload = {"registered": False, "extension": EXTENSION,
               "check": "§3 A-1b origin of the per-PMT code", "runbook": RUNBOOK_REL,
               "items": A1B_LABELS, "rows": {}, "unavailable": [],
               "reproduction": {}, "readings": {},
               "gate": {"authorised": gate["authorised"], "status": gate["status"]}}
    # group C: one read-only probe-test pass, shared by every item
    c_features = collect_probe_test_features(config, geometry, test_dataset, log=log)
    payload["C"] = {"n_events": c_features["n_events"],
                    "n_features": int(3 + 1 + 1 + 1 + int(geometry.n_modules)
                                      + int(pe.N_FEATURES))}
    # (1) the collector must reproduce the committed fix9000 rows before anything else
    fix_row = inv["rows"].get("fix9000")
    fix_sha = A1B_LABEL_FIX_SHA256
    cache = eh._read_label_cache(A1B_LABEL_CACHE, fix_sha=fix_sha,
                                 max_label_events=A1B_LABEL_EVENTS)
    fix_rows = None
    if fix_row is not None and fix_row.get("status") in ("available", "unverified (dry-run)"):
        try:
            fix_rows = _item_label_rows(fix_row, config=config, gen=gen,
                                        arms_config=arms_config, geometry=geometry,
                                        test_dataset=test_dataset, data_root=data_root,
                                        device=device, log=log)
        except Exception as problem:
            payload["unavailable"].append({"label": "fix9000",
                                           "reason": f"{type(problem).__name__}: {problem}"})
    reproduction = verify_label_collector(fix_rows, cache)
    payload["reproduction"]["fix9000_rows"] = reproduction
    if fix_rows is None or not reproduction.get("pass"):
        payload["readings"] = {}
        payload["unavailable"].append({"label": "A-1b", "reason":
                                       "label collector reproduction did not pass"})
        payload["complete"] = False
        return payload
    # (2) the C-group reproduction target: fix9000 ridge R² 0.1043 within 1e-3
    fix_probes = a1b_probes(fix_rows, geometry, None, c_features, methods=("linear",),
                            seed=er.SAMPLE_SEED, requested=("C",), log=log)
    c_r2 = ((fix_probes["groups"].get("C") or {}).get("linear") or {}).get("r2")
    c_pass = bool(c_r2 is not None
                  and abs(float(c_r2) - A1B_REPRODUCTION_R2) <= A1B_REPRODUCTION_TOLERANCE)
    payload["reproduction"]["c_group"] = {"expected_r2": A1B_REPRODUCTION_R2,
                                          "got_r2": c_r2,
                                          "abs_diff": (None if c_r2 is None
                                                       else float(abs(float(c_r2)
                                                                      - A1B_REPRODUCTION_R2))),
                                          "tolerance": A1B_REPRODUCTION_TOLERANCE,
                                          "pass": c_pass}
    if not c_pass:
        payload["unavailable"].append({"label": "A-1b", "reason":
                                       "C-group ridge R² did not reproduce 0.1043"})
        payload["complete"] = False
        return payload
    # (3) every item
    marginal_cache = {}
    for label in A1B_LABELS:
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            payload["unavailable"].append({"label": label, "reason": "inventory unavailable"})
            continue
        try:
            rows = fix_rows if label == "fix9000" else _item_label_rows(
                row, config=config, gen=gen, arms_config=arms_config, geometry=geometry,
                test_dataset=test_dataset, data_root=data_root, device=device, log=log)
            marginal_key = TRAIN_KEY_300K if label.startswith("scale300k") else TRAIN_KEY_S005
            if marginal_key not in marginal_cache:
                marginal_cache[marginal_key] = read_dev_marginals(
                    config, geometry, data_root, key=marginal_key, log=log)
            probes_payload = a1b_probes(rows, geometry, marginal_cache[marginal_key],
                                        c_features, seed=er.SAMPLE_SEED, log=log)
        except SystemExit:
            raise
        except Exception as problem:
            payload["unavailable"].append({"label": label,
                                           "reason": f"{type(problem).__name__}: {problem}"})
            continue
        entry = {"arm": row.get("arm"), "split": row.get("split"), "seed": int(row["seed"]),
                 "n_pmt": probes_payload["split"]["n_pmt"],
                 "marginal_key": marginal_key,
                 "mlp_r2": {group: (probes_payload["groups"].get(group) or {}).get(
                     "mlp", {}).get("r2") for group in ("M", "C", "All")},
                 "linear_r2": {group: (probes_payload["groups"].get(group) or {}).get(
                     "linear", {}).get("r2") for group in ("M", "C", "All")},
                 "singles_mlp_r2": {name: (probes_payload["groups"].get(f"M::{name}") or {})
                                    .get("mlp", {}).get("r2")
                                    for name in M_FEATURE_NAMES},
                 "probes": probes_payload["groups"]}
        payload["rows"][label] = entry
    payload["readings"] = _a1b_readings(payload["rows"], inv)
    payload["complete"] = not payload["unavailable"]
    return payload


def _a1b_readings(rows, inventory) -> dict:
    """§3 A-1b classes (both seeds must agree) beside the init9000/fix9000 references."""
    items = {}
    for label, entry in rows.items():
        if label == "init9000":
            continue
        items[label] = {"mlp_r2": entry["mlp_r2"],
                        "class": a1b_class(entry["mlp_r2"].get("M"),
                                           entry["mlp_r2"].get("All"))}
    pairs = {}
    for prefix in ("fix", "scale300k", "window128", "window32", "pred4", "across"):
        labels = (f"{prefix}9000", f"{prefix}9001")
        if all(label in items for label in labels):
            classes = [items[label]["class"] for label in labels]
            if classes[0] == classes[1]:
                pairs[prefix] = {"class": classes[0], "seeds": list(classes),
                                 "mlp_r2": {label: items[label]["mlp_r2"] for label in labels}}
            else:
                pairs[prefix] = {"class": "seed disagreement", "seeds": list(classes),
                                 "mlp_r2": {label: items[label]["mlp_r2"] for label in labels}}
    init = rows.get("init9000") or {}
    committed = readout_a1b_committed(inventory, root=(inventory or {}).get("root")
                                      if isinstance(inventory, dict) else None)
    return {"rules": {"order": list(A1B_READING_ORDER),
                      "input-statistics code": "M MLP R² >= 0.7",
                      "fixed-feature code": "M < 0.7 and All >= 0.7",
                      "not determined by fixed per-PMT features": "All <= 0.5",
                      "otherwise": "intermediate",
                      "both_seeds_agree": True},
            "items": items, "classes": pairs,
            "init9000_reference": {"mlp_r2": init.get("mlp_r2")},
            "committed_fix9000": committed}


def readout_a1b_committed(inventory=None, root=None) -> dict:
    """§3 A-1b committed fix9000 code-origin values (All MLP 0.434; C ridge 0.1043)."""
    root = _resolve_root(root if root is not None else
                         (inventory or {}).get("root") if isinstance(inventory, dict)
                         else None)
    payload = read_json(root / "results/jepa/20261101_h12/e0e1_code_origin.json", {}) or {}
    fix = ((payload.get("probes_by_label") or {}).get("fix9000") or {})
    out = {"source": "results/jepa/20261101_h12/e0e1_code_origin.json",
           "n_features_all": ((payload.get("features") or {}).get("n_features_all"))}
    for group in ("position", "hit_rate", "charge", "parity", "module", "raw_token", "all"):
        cell = fix.get(group) or {}
        out[group] = {method: (cell.get(method) or {}).get("r2")
                      for method in ("linear", "mlp", "knn5")}
    return out


# ================================================================ A-2 §3
def _run_stats_role(run_ids):
    return np.asarray(run_ids, dtype=object)


def stream_event_rows(model, dataset, geometry, *, split, mask_seed, device="cpu",
                      chunk_events=8, max_events=None, domain=er.PRIMARY_DOMAIN,
                      log=print) -> dict:
    """One streaming pass over a probe set: per-event rows + module-gain accumulator.

    Like ``er.stream_model_forms`` but records ``event_ids`` so the A-2 pairing can
    join the two environments on ``(run_id, event_id)`` without a second pass.
    """
    import torch
    device = device if isinstance(device, torch.device) else torch.device(device)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    module_index = np.asarray(geometry.module_index, np.int64)
    acc = la.ProbeAccumulator("module_gain", "run_mean")
    rows = {"events": [], "run_ids": [], "event_ids": [], "vertex": [],
            "log_yield": [], "n_hit": []}
    for row in dataset.runs:
        source = dataset.directory / "runs" / row["run_dir"]
        truth = lr._run_truth(row, dataset, geometry)
        ids = lr._run_event_ids(source)
        if domain not in (None, "all"):
            allowed = lr._domain_event_ids(source, domain)
            if allowed is not None:
                ids = np.intersect1d(ids, allowed)
        if ids.size == 0:
            continue
        if max_events is not None:
            ids = ids[:int(max_events)]
        partial = None
        for start in range(0, int(ids.size), int(chunk_events)):
            batch = lr._probe_batch(source, geometry, split=split, mask_seed=mask_seed,
                                    occupancy="natural", scaling=scaling,
                                    event_ids=ids[start:start + int(chunk_events)])
            if batch.event_ids.size == 0:
                continue
            batch = lr._to_device(batch, device)
            selected = la._select_truth(truth, np.asarray(batch.event_ids))
            with torch.no_grad():
                tokens = model.encode(batch.features)
            token_np = tokens.detach().to(torch.float32).cpu().numpy()
            hit = np.asarray(batch.hit_flags, bool)
            hidden = np.asarray(batch.hidden.detach().cpu(), bool)
            data = la.RunProbeData(run_id=str(row["run_id"]),
                                   event_ids=np.asarray(batch.event_ids),
                                   tokens=token_np, context_tokens=None, raw=None,
                                   hit=hit, hidden=hidden, module_index=module_index,
                                   truth=selected)
            part = la.probe_run_mean_partial(data, "module_gain")
            partial = part if partial is None else la._merge_run_mean_partial(partial, part)
            rows["events"].append(la._mean_tokens(token_np, None).astype(np.float32))
            rows["run_ids"].append(np.full(len(batch.event_ids), row["run_id"], dtype=object))
            rows["event_ids"].append(np.asarray(batch.event_ids, np.int64))
            rows["vertex"].append(np.asarray(selected["vertex"], float))
            rows["log_yield"].append(np.asarray(selected["log_yield"], float))
            rows["n_hit"].append(hit.sum(axis=1).astype(np.int64))
        if partial is not None:
            features, targets, _key, names = la.finalise_run_mean(partial)
            if features.shape[0]:
                acc.target_names = names
                acc.add(row["run_id"], features, targets)
    out = {"module_gain": acc}
    for name, values in rows.items():
        out[name] = np.concatenate(values) if values else np.zeros(0)
    out["events"] = np.asarray(out["events"], np.float32)
    out["run_ids"] = _run_stats_role(out["run_ids"])
    return out


def _probe_dataset(rows, kind, y=None, form="full_input", target_names=None):
    y = rows["vertex"] if y is None else y
    if np.asarray(y).ndim == 1:
        y = np.asarray(y, float)[:, None]
    return la.ProbeDataset(kind=kind, form=form, x=np.asarray(rows["events"], np.float64),
                           y=np.asarray(y, np.float64), run_ids=rows["run_ids"],
                           target_names=tuple(target_names or ()), meta={})


def _vertex_errors(rows, prediction) -> np.ndarray:
    return np.linalg.norm(np.asarray(prediction, float) - np.asarray(rows["vertex"], float),
                          axis=1)


def pair_events(left, right) -> dict:
    """Join two per-event row sets on ``(run_id, event_id)`` and verify the truth.

    A deliberately mismatched key set (or unequal truth) raises ``ValueError``: the
    A-2 pairing must never silently compare different events.
    """
    keys_left = [(str(run), int(event)) for run, event in
                 zip(left["run_ids"], left["event_ids"])]
    keys_right = [(str(run), int(event)) for run, event in
                  zip(right["run_ids"], right["event_ids"])]
    if len(keys_left) != len(set(keys_left)) or len(keys_right) != len(set(keys_right)):
        raise ValueError("pair_events: duplicate (run_id, event_id) keys")
    index_right = {key: index for index, key in enumerate(keys_right)}
    if set(keys_left) != set(keys_right):
        missing = sorted(set(keys_left) - set(keys_right))[:5]
        extra = sorted(set(keys_right) - set(keys_left))[:5]
        raise ValueError(f"pair_events: key sets differ (left-only {missing}, "
                         f"right-only {extra})")
    order_left = np.arange(len(keys_left))
    order_right = np.array([index_right[key] for key in keys_left], np.int64)
    vertex_left = np.asarray(left["vertex"], float)[order_left]
    vertex_right = np.asarray(right["vertex"], float)[order_right]
    if not np.array_equal(vertex_left, vertex_right):
        raise ValueError("pair_events: the paired events carry different truth")
    return {"n_events": int(len(keys_left)),
            "n_runs": int(len(np.unique([key[0] for key in keys_left]))),
            "index_left": order_left, "index_right": order_right,
            "run_ids": np.asarray([key[0] for key in keys_left], dtype=object),
            "event_ids": np.asarray([key[1] for key in keys_left], np.int64)}


def paired_run_bootstrap(errors_left, errors_right, run_ids, *,
                         resamples=A2_BOOTSTRAP_RESAMPLES, seed=A2_BOOTSTRAP_SEED) -> dict:
    """Run-cluster bootstrap of the paired mean vertex-error difference (§3 A-2)."""
    errors_left = np.asarray(errors_left, float)
    errors_right = np.asarray(errors_right, float)
    run_ids = np.asarray(run_ids, dtype=object)
    runs = np.unique(run_ids.astype(str))
    if not runs.size or errors_left.shape != errors_right.shape \
            or errors_left.shape[0] != run_ids.shape[0]:
        raise ValueError("paired_run_bootstrap: misaligned inputs")
    order = np.argsort(run_ids.astype(str), kind="stable")
    sorted_runs = run_ids.astype(str)[order]
    boundaries = np.searchsorted(sorted_runs, runs)
    sums_left = np.add.reduceat(errors_left[order], boundaries)
    sums_right = np.add.reduceat(errors_right[order], boundaries)
    counts = np.diff(np.append(boundaries, len(order))).astype(float)
    observed = float((sums_right.sum() / counts.sum())
                     - (sums_left.sum() / counts.sum()))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), float)
    for index in range(int(resamples)):
        pick = rng.integers(0, len(runs), len(runs))
        count = counts[pick].sum()
        draws[index] = float(sums_right[pick].sum() / count
                             - sums_left[pick].sum() / count)
    return {"n_runs": int(len(runs)), "n_events": int(len(errors_left)),
            "resamples": int(resamples), "seed": int(seed),
            "mean_error_left_m": float(errors_left.mean()),
            "mean_error_right_m": float(errors_right.mean()),
            "mean_difference_m": observed,
            "ci95_m": [float(np.quantile(draws, 0.025)),
                       float(np.quantile(draws, 0.975))],
            "definition": "right minus left, run-cluster resampling, probe fit held fixed"}


def _a2_environment(row, probes, *, config, gen, arms_config, geometry, data_root, device,
                    log=print) -> dict:
    with item_eval_context(row):
        arm = build_item_arm(row, config=config, gen=gen, arms_config=arms_config,
                             geometry=geometry, data_root=data_root, device=device, log=log)
        mask_seed = int(config["views"]["mask_seed"])
        chunk = int((config.get("evaluation") or {}).get("probe_chunk_events", 64))
        train = stream_event_rows(getattr(arm, "jepa", arm), probes["train"], geometry,
                                  split=row.get("split", "within"), mask_seed=mask_seed,
                                  device=device, chunk_events=chunk, log=log)
        test = stream_event_rows(getattr(arm, "jepa", arm), probes["test"], geometry,
                                 split=row.get("split", "within"), mask_seed=mask_seed,
                                 device=device, chunk_events=chunk, log=log)
    train_ds = _probe_dataset(train, "vertex", target_names=("x", "y", "z"))
    test_ds = _probe_dataset(test, "vertex", target_names=("x", "y", "z"))
    fitted = la.fit_probe(train_ds)
    prediction = la.probe_predictions(fitted, test_ds)
    vertex = lr._vertex_error(prediction, test_ds.y)
    yield_ds_train = _probe_dataset(train, "log_yield", y=train["log_yield"],
                                    target_names=("log_yield",))
    yield_ds_test = _probe_dataset(test, "log_yield", y=test["log_yield"],
                                   target_names=("log_yield",))
    log_yield = la.probe_r2(yield_ds_train, yield_ds_test)
    gain = la.probe_r2_streamed(train["module_gain"], test["module_gain"])
    return {"train": train, "test": test, "train_ds": train_ds, "test_ds": test_ds,
            "fitted": fitted, "prediction": prediction,
            "errors": _vertex_errors(test, prediction),
            "vertex": vertex, "log_yield": log_yield, "module_gain": gain}


def readout_a2(*, root=None, inventory=None, config, gen, arms_config, geometry,
               data_root=DATA_ROOT_DEFAULT, device="cpu", labels=None, log=print) -> dict:
    """§3 A-2: frozen MAE under strong drift, paired events, run-cluster bootstrap."""
    root = _resolve_root(root)
    gate = assert_authorised("readout", root=root)
    inv = inventory if inventory is not None else build_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    payload = {"registered": False, "extension": EXTENSION,
               "check": "§3 A-2 frozen MAE under strong drift", "runbook": RUNBOOK_REL,
               "floors": {str(key): float(value) for key, value in MODULE_GAIN_FLOOR.items()},
               "models": {}, "unavailable": [],
               "gate": {"authorised": gate["authorised"], "status": gate["status"]}}
    label_list = readout_label_list("A-2", inv["rows"]) if labels is None else list(labels)
    env_cache = {}
    for label in label_list:
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            payload["unavailable"].append({"label": label, "reason": "inventory unavailable"})
            continue
        envs = {}
        try:
            for name, probes, setting in (("s005", probes005, S005),
                                          ("s030", probes030, S030)):
                envs[name] = _a2_environment(row, probes, config=config, gen=gen,
                                             arms_config=arms_config, geometry=geometry,
                                             data_root=data_root, device=device, log=log)
        except SystemExit:
            raise
        except Exception as problem:
            payload["unavailable"].append({"label": label,
                                           "reason": f"{type(problem).__name__}: {problem}"})
            continue
        try:
            paired = pair_events(envs["s005"]["test"], envs["s030"]["test"])
        except ValueError as problem:
            payload["unavailable"].append({"label": label,
                                           "reason": f"pairing failed: {problem}"})
            continue
        # freeze the s = 0.05 probe and apply it to both environments
        frozen = la.fit_probe(envs["s005"]["train_ds"])
        frozen_vertex, frozen_errors = {}, {}
        for name in ("s005", "s030"):
            prediction = la.probe_predictions(frozen, envs[name]["test_ds"])
            frozen_errors[name] = _vertex_errors(envs[name]["test"], prediction)
            frozen_vertex[name] = float(frozen_errors[name].mean())
        bootstrap = paired_run_bootstrap(frozen_errors["s005"][paired["index_left"]],
                                         frozen_errors["s030"][paired["index_right"]],
                                         paired["run_ids"])
        model = {"arm": row.get("arm"), "split": row.get("split"),
                 "seed": int(row["seed"]), "checkpoint_sha256": row.get("sha256"),
                 "environments": {}}
        for name, probes, setting in (("s005", probes005, S005), ("s030", probes030, S030)):
            env = envs[name]
            gain = env["module_gain"].get("r2_mean")
            model["environments"][name] = {
                "data_setting": float(setting),
                "n_events": {"train": int(env["train"]["events"].shape[0]),
                             "test": int(env["test"]["events"].shape[0])},
                "module_gain_r2": gain,
                "module_gain_minus_floor": (None if gain is None
                                            else float(gain - MODULE_GAIN_FLOOR[setting])),
                "vertex_refit_m": env["vertex"].get("resolution"),
                "vertex_refit_median_m": env["vertex"].get("median"),
                "vertex_frozen_m": frozen_vertex[name],
                "log_yield_r2": env["log_yield"].get("r2_mean"),
                "probe_refit": "probe fit on this environment's own probe-train",
                "probe_frozen": "probe fit frozen from s = 0.05"}
        model["paired"] = bootstrap
        payload["models"][label] = model
        # keep the s = 0.30 environment rows of the s = 0.05-trained MAE and of the
        # T-1 checkpoints so G-2 can pair them event by event
        seed = int(row["seed"])
        g2_key = (f"mae_across_trigger{seed}" if str(row.get("split")) == "across"
                  else f"mae_within_trigger{seed}")
        if str(label) == g2_key or str(label).startswith("t1_"):
            env_cache[label] = envs
    payload["t1_g2"] = _t1_g2(env_cache)
    payload["complete"] = not payload["unavailable"]
    return payload


def _t1_g2(env_cache) -> dict:
    """§4 G-2: s = 0.30-trained MAE minus the s = 0.05-trained MAE, paired events.

    Both models are evaluated on the s = 0.30 probe-test set; the per-event full-input
    vertex errors are paired on ``(run_id, event_id)`` (truth identical) and the A-2
    run-cluster bootstrap is applied.  Descriptive, no threshold.
    """
    out = {}
    for seed in T1_SEEDS:
        for split, base_variant, t1_variant in (
                ("within", "mae_within_trigger", "t1_s030_within_trigger"),
                ("across", "mae_across_trigger", "t1_s030_across_trigger")):
            base_label = f"{base_variant}{seed}"
            t1_label = f"{t1_variant}{seed}"
            if base_label not in env_cache or t1_label not in env_cache:
                continue
            base_env = env_cache[base_label]["s030"]
            t1_env = env_cache[t1_label]["s030"]
            try:
                paired = pair_events(base_env["test"], t1_env["test"])
            except ValueError as problem:
                out[f"{split}__{seed}"] = {"status": "pairing_failed",
                                           "reason": str(problem)}
                continue
            bootstrap = paired_run_bootstrap(
                base_env["errors"][paired["index_left"]],
                t1_env["errors"][paired["index_right"]], paired["run_ids"])
            out[f"{split}__{seed}"] = {
                "split": split, "seed": int(seed),
                "s005_trained_model": base_label, "s030_trained_model": t1_label,
                "difference_m": bootstrap["mean_difference_m"],
                "ci95_m": bootstrap["ci95_m"], "n_runs": bootstrap["n_runs"],
                "n_events": bootstrap["n_events"],
                "definition": "s = 0.30-trained minus s = 0.05-trained, paired events, "
                              "run-cluster bootstrap (A-2 protocol)"}
    return out


# ================================================================ A-3 §3
PARTICLE_DOMAINS = ("muon", "electron")


def load_particle_truth(data_root, key) -> dict:
    """§3 A-3 read-only truth loader: ``events.npz`` particle fields + NHits, by event."""
    manifest = read_json(manifest_path(data_root, key), {}) or {}
    directory = manifest_path(data_root, key).parent
    out = {"key": key, "runs": {}}
    for row in (manifest.get("runs") or []):
        run_dir = directory / "runs" / row["run_dir"]
        truth = _read_truth(run_dir / "events.npz") or {}
        store = lb.read_sparse_hits(run_dir / "hits.npz")
        ids = np.asarray(truth.get("event_ids", store.event_ids), np.int64)
        n_hit = np.diff(np.asarray(store.offsets, np.int64))
        position = {int(event): index for index, event in enumerate(store.event_ids)}
        aligned = np.asarray([n_hit[position[int(event)]] if int(event) in position else 0
                              for event in ids], np.int64)
        out["runs"][str(row["run_id"])] = {
            "event_ids": ids,
            "particle": np.asarray(truth.get("particle", np.array([]))).astype(str),
            "direction_m": np.asarray(truth.get("direction_m", np.zeros((len(ids), 3)))),
            "energy_MeV": np.asarray(truth.get("energy_MeV", np.full(len(ids), np.nan))),
            "deposited_MeV": np.asarray(truth.get("deposited_MeV",
                                                   np.full(len(ids), np.nan))),
            "contained": np.asarray(truth.get("contained", np.zeros(len(ids), bool))),
            "domain": np.asarray(truth.get("domain", np.array([]))).astype(str),
            "vertex": np.asarray(truth.get("vertex_m", np.zeros((len(ids), 3)))),
            "n_hit": aligned,
        }
    return out


def stream_particle_rows(model, dataset, truth, geometry, *, chunk_events=8,
                         device="cpu", log=print) -> dict:
    """Stream full-input pooled embeddings for the muon/electron events of one set."""
    import torch
    device = device if isinstance(device, torch.device) else torch.device(device)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    rows = {"events": [], "run_ids": [], "event_ids": [], "particle": [],
            "direction": [], "deposited_MeV": [], "contained": [], "n_hit": [],
            "vertex": []}
    for row in dataset.runs:
        run_id = str(row["run_id"])
        run_truth = (truth.get("runs") or {}).get(run_id)
        if run_truth is None:
            continue
        particle = np.asarray(run_truth["particle"]).astype(str)
        keep = np.isin(particle, PARTICLE_DOMAINS)
        ids = np.asarray(run_truth["event_ids"], np.int64)[keep]
        if ids.size == 0:
            continue
        source = dataset.directory / "runs" / row["run_dir"]
        index = {int(event): position for position, event in
                 enumerate(np.asarray(run_truth["event_ids"], np.int64))}
        for start in range(0, int(ids.size), int(chunk_events)):
            batch = lr._probe_batch(source, geometry, split="within", mask_seed=0,
                                    occupancy="natural", scaling=scaling,
                                    event_ids=ids[start:start + int(chunk_events)])
            if batch.event_ids.size == 0:
                continue
            batch = lr._to_device(batch, device)
            with torch.no_grad():
                tokens = model.encode(batch.features)
            token_np = tokens.detach().to(torch.float32).cpu().numpy()
            rows["events"].append(la._mean_tokens(token_np, None).astype(np.float32))
            rows["run_ids"].append(np.full(len(batch.event_ids), run_id, dtype=object))
            rows["event_ids"].append(np.asarray(batch.event_ids, np.int64))
            positions = np.asarray([index[int(event)] for event in batch.event_ids])
            for name in ("particle", "direction", "deposited_MeV", "contained", "vertex"):
                key = "direction_m" if name == "direction" else (
                    "vertex" if name == "vertex" else name)
                rows[name].append(np.asarray(run_truth[key])[positions])
            rows["n_hit"].append(np.asarray(run_truth["n_hit"])[positions])
    out = {}
    for name, values in rows.items():
        if name == "events":
            out[name] = np.asarray(np.concatenate(values), np.float32) if values \
                else np.zeros((0, 0), np.float32)
        elif name in ("run_ids", "particle"):
            out[name] = np.concatenate(values).astype(str) if values else np.zeros(0, str)
        elif name == "contained":
            out[name] = np.concatenate(values).astype(bool) if values else np.zeros(0, bool)
        elif name == "n_hit":
            out[name] = np.concatenate(values).astype(np.int64) if values \
                else np.zeros(0, np.int64)
        else:
            out[name] = (np.concatenate(values).astype(float) if values
                         else np.zeros((0, 3) if name in ("direction", "vertex") else 0))
    return out


def raw_event_features(dataset, geometry, *, hand=None, truth=None) -> dict:
    """§3 A-3 raw-feature baseline: ``[ln Σq, NHits, Σq-weighted mean/SD of x,y,z,t]``."""
    rows = {"features": [], "run_ids": [], "event_ids": [], "particle": [],
            "contained": [], "deposited_MeV": [], "direction": [], "vertex": [],
            "n_hit": []}
    truth = truth or {}
    positions = np.asarray(geometry.positions_m, float)
    for row in dataset.runs:
        run_id = str(row["run_id"])
        run_truth = (truth.get("runs") or {}).get(run_id) or {}
        particle = np.asarray(run_truth.get("particle", np.array([]))).astype(str)
        keep = np.isin(particle, PARTICLE_DOMAINS)
        store = lb.read_sparse_hits(dataset.directory / "runs" / row["run_dir"] / "hits.npz")
        index = {int(event): position for position, event in
                 enumerate(np.asarray(run_truth.get("event_ids", store.event_ids),
                                      np.int64))}
        features = {}
        for position, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
            if not keep[position]:
                continue
            span = slice(int(store.offsets[position]), int(store.offsets[position + 1]))
            pmt = np.asarray(store.pmt_id[span], np.int64)
            charge = np.asarray(store.charge_pe[span], float)
            time = np.asarray(store.time_ns[span], float) / 100.0
            total = float(charge.sum())
            weight = charge / total if total > 0 else np.zeros_like(charge)
            means = [float((weight * positions[pmt, axis]).sum()) for axis in range(3)] \
                if pmt.size else [0.0, 0.0, 0.0]
            stds = [float(np.sqrt(max((weight * (positions[pmt, axis] - means[axis]) ** 2)
                                      .sum(), 0.0))) for axis in range(3)] \
                if pmt.size else [0.0, 0.0, 0.0]
            t_mean = float((weight * time).sum()) if pmt.size else 0.0
            t_std = float(np.sqrt(max((weight * (time - t_mean) ** 2).sum(), 0.0))) \
                if pmt.size else 0.0
            features[int(event_id)] = [math.log(max(total, 1e-6)), float(pmt.size)] + \
                means + stds + [t_mean, t_std]
        ids = np.asarray([event for event in features], np.int64)
        if ids.size == 0:
            continue
        rows["features"].append(np.asarray([features[int(event)] for event in ids], float))
        rows["run_ids"].append(np.full(len(ids), run_id, dtype=object))
        rows["event_ids"].append(ids)
        positions_index = np.asarray([index[int(event)] for event in ids])
        rows["particle"].append(np.asarray(run_truth["particle"])[positions_index])
        rows["contained"].append(np.asarray(run_truth["contained"])[positions_index]
                                 .astype(bool))
        rows["deposited_MeV"].append(np.asarray(run_truth["deposited_MeV"])[positions_index])
        rows["direction"].append(np.asarray(run_truth["direction_m"])[positions_index])
        rows["vertex"].append(np.asarray(run_truth["vertex"])[positions_index])
        rows["n_hit"].append(np.asarray(run_truth["n_hit"])[positions_index])
    out = {"features": np.concatenate(rows["features"]) if rows["features"]
           else np.zeros((0, 10))}
    for name in ("run_ids", "particle"):
        out[name] = np.concatenate(rows[name]).astype(str) if rows[name] \
            else np.zeros(0, str)
    for name, dtype in (("event_ids", np.int64), ("contained", bool), ("n_hit", np.int64)):
        out[name] = np.concatenate(rows[name]).astype(dtype) if rows[name] \
            else np.zeros(0, dtype)
    out["deposited_MeV"] = np.concatenate(rows["deposited_MeV"]) if rows["deposited_MeV"] \
        else np.zeros(0)
    out["direction"] = np.concatenate(rows["direction"]) if rows["direction"] \
        else np.zeros((0, 3))
    out["vertex"] = np.concatenate(rows["vertex"]) if rows["vertex"] \
        else np.zeros((0, 3))
    out["feature_names"] = ("ln_sum_q", "n_hits", "weighted_mean_x", "weighted_mean_y",
                            "weighted_mean_z", "weighted_sd_x", "weighted_sd_y",
                            "weighted_sd_z", "weighted_mean_t", "weighted_sd_t")
    return out


def _mlp_direction(x_train, y_train, x_test, *, seed=0, hidden=256, epochs=200,
                   patience=10, batch=4096) -> np.ndarray:
    """2 x 256 MLP onto the unit direction vector (fixed committed architecture/seed)."""
    import torch
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, float)
    centre, scale = x_train.mean(0), x_train.std(0)
    scale[scale == 0] = 1.0
    y_centre = y_train.mean(0, keepdims=True)
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, y_train.shape[1]))
    xt = torch.as_tensor((x_train - centre) / scale, dtype=torch.float32)
    yt = torch.as_tensor(y_train - y_centre, dtype=torch.float32)
    cut = max(1, int(0.9 * len(xt)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(int(seed) + 1)
    best, state, bad = math.inf, None, 0
    for _epoch in range(int(epochs)):
        model.train()
        order = torch.randperm(cut, generator=generator)
        for start in range(0, cut, int(batch)):
            index = order[start:start + int(batch)]
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(xt[index]), yt[index]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(torch.nn.functional.mse_loss(model(xt[cut:]), yt[cut:]))
        if validation < best:
            best, state, bad = validation, {key: value.clone()
                                            for key, value in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor((np.asarray(x_test, float) - centre) / scale,
                                           dtype=torch.float32)).numpy() + y_centre
    return prediction


def angular_errors(prediction, direction) -> dict:
    """Angular error (rad and deg) of a predicted direction; median and 68% quantile."""
    prediction = np.asarray(prediction, float)
    direction = np.asarray(direction, float)
    norm = np.linalg.norm(prediction, axis=1)
    unit = np.divide(prediction, norm[:, None], out=np.zeros_like(prediction),
                     where=norm[:, None] > 0)
    dot = np.clip((unit * direction).sum(axis=1), -1.0, 1.0)
    angles = np.arccos(dot)
    return {"median_deg": float(np.degrees(np.median(angles))),
            "q68_deg": float(np.degrees(np.quantile(angles, 0.68))),
            "n": int(len(angles))}


def _run_grouped_folds(run_ids, k=5, seed=0) -> np.ndarray:
    runs = np.unique(np.asarray(run_ids).astype(str))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(runs))
    fold_of = {run: index % int(k) for index, run in enumerate(runs[order])}
    return np.asarray([fold_of[str(run)] for run in run_ids], np.int64)


def _logistic_auc(x_train, y_train, runs_train, x_test, y_test) -> dict:
    """L2 logistic on standardised features; C by run-grouped 5-fold CV on train."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    x_train = np.asarray(x_train, float)
    x_test = np.asarray(x_test, float)
    centre, scale = x_train.mean(0), x_train.std(0)
    scale[scale == 0] = 1.0
    z_train = (x_train - centre) / scale
    z_test = (x_test - centre) / scale
    folds = _run_grouped_folds(runs_train, k=5, seed=0)
    best = None
    for C in (0.01, 0.1, 1.0, 10.0):
        scores = []
        for fold in range(5):
            train = folds != fold
            if (len(np.unique(np.asarray(y_train)[train])) < 2
                    or len(np.unique(np.asarray(y_train)[~train])) < 2):
                continue
            model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            model.fit(z_train[train], np.asarray(y_train)[train])
            scores.append(roc_auc_score(np.asarray(y_train)[~train],
                                        model.predict_proba(z_train[~train])[:, 1]))
        if scores and (best is None or np.mean(scores) > best[1]):
            best = (C, float(np.mean(scores)))
    if len(np.unique(np.asarray(y_train))) < 2:
        return {"status": "not_evaluable", "reason": "one class in the training rows",
                "auc": None, "balanced_accuracy": None, "C": None, "cv_auc": None,
                "n_train": int(len(z_train)), "n_test": int(len(z_test)),
                "probability": np.full(len(z_test), np.nan)}
    C = 1.0 if best is None else best[0]
    model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    model.fit(z_train, np.asarray(y_train))
    probability = model.predict_proba(z_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {"C": C, "cv_auc": None if best is None else best[1],
            "auc": (float(roc_auc_score(y_test, probability))
                    if len(np.unique(np.asarray(y_test))) > 1 else None),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "n_train": int(len(z_train)), "n_test": int(len(z_test)),
            "probability": probability}


def _nhits_quintile_auc(probability, y_test, n_hit) -> dict:
    from sklearn.metrics import roc_auc_score
    n_hit = np.asarray(n_hit, float)
    edges = np.quantile(n_hit, [0.2, 0.4, 0.6, 0.8])
    out = {}
    for name, low, high in (("Q1", -np.inf, edges[0]), ("Q2", edges[0], edges[1]),
                            ("Q3", edges[1], edges[2]), ("Q4", edges[2], edges[3]),
                            ("Q5", edges[3], np.inf)):
        mask = (n_hit > low) & (n_hit <= high) if name != "Q1" else (n_hit <= high)
        if not mask.any():
            out[name] = {"n": 0, "auc": None}
            continue
        labels = np.asarray(y_test)[mask]
        out[name] = {"n": int(mask.sum()),
                     "auc": (float(roc_auc_score(labels, probability[mask]))
                             if len(np.unique(labels)) > 1 else None)}
    return {"edges": [float(value) for value in edges], "quintiles": out}


def _direction_error_row(rows, prediction) -> dict:
    return angular_errors(prediction, np.asarray(rows["direction"], float))


def _direction_tasks(x_train, direction_train, runs_train, x_test, direction_test,
                     runs_test, *, seed=0) -> dict:
    train_ds = la.ProbeDataset(kind="direction", form="full_input",
                               x=np.asarray(x_train, float),
                               y=np.asarray(direction_train, float),
                               run_ids=np.asarray(runs_train, dtype=object),
                               target_names=("dx", "dy", "dz"), meta={})
    test_ds = la.ProbeDataset(kind="direction", form="full_input",
                              x=np.asarray(x_test, float),
                              y=np.asarray(direction_test, float),
                              run_ids=np.asarray(runs_test, dtype=object),
                              target_names=("dx", "dy", "dz"), meta={})
    fitted = la.fit_probe(train_ds)
    ridge = la.probe_predictions(fitted, test_ds)
    mlp = _mlp_direction(np.asarray(x_train, float), np.asarray(direction_train, float),
                         np.asarray(x_test, float), seed=seed)
    return {"ridge": angular_errors(ridge, direction_test),
            "mlp": angular_errors(mlp, direction_test),
            "n_train": int(len(train_ds.x)), "n_test": int(len(test_ds.x))}


def _energy_task(x_train, deposited_train, runs_train, contained_train, x_test,
                 deposited_test, runs_test, contained_test) -> dict:
    n_contained_train = int(np.sum(contained_train))
    n_contained_test = int(np.sum(contained_test))
    if n_contained_test < A3_MIN_CONTAINED:
        return {"status": "not_evaluable",
                "reason": f"{n_contained_test} contained events < {A3_MIN_CONTAINED}",
                "n_contained_train": n_contained_train,
                "n_contained_test": n_contained_test}
    train_ds = la.ProbeDataset(kind="deposited_MeV", form="full_input",
                               x=np.asarray(x_train, float)[contained_train],
                               y=np.asarray(deposited_train, float)[contained_train][:, None],
                               run_ids=np.asarray(runs_train, dtype=object)[contained_train],
                               target_names=("deposited_MeV",), meta={})
    test_ds = la.ProbeDataset(kind="deposited_MeV", form="full_input",
                              x=np.asarray(x_test, float)[contained_test],
                              y=np.asarray(deposited_test, float)[contained_test][:, None],
                              run_ids=np.asarray(runs_test, dtype=object)[contained_test],
                              target_names=("deposited_MeV",), meta={})
    out = la.probe_r2(train_ds, test_ds)
    out["n_contained_train"] = n_contained_train
    out["n_contained_test"] = n_contained_test
    return out


def _a3_model_rows(train_rows, test_rows, raw_train, raw_test) -> dict:
    """The §3 A-3 tasks for one model embedding, with the raw-feature baseline."""
    tasks = {"domains": {}}
    for domain in PARTICLE_DOMAINS:
        train_mask = np.asarray(train_rows["particle"]).astype(str) == domain
        test_mask = np.asarray(test_rows["particle"]).astype(str) == domain
        entry = {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}
        entry["direction"] = _direction_tasks(
            np.asarray(train_rows["events"], float)[train_mask],
            np.asarray(train_rows["direction"], float)[train_mask],
            np.asarray(train_rows["run_ids"])[train_mask],
            np.asarray(test_rows["events"], float)[test_mask],
            np.asarray(test_rows["direction"], float)[test_mask],
            np.asarray(test_rows["run_ids"])[test_mask])
        for name, train_source, test_source in (
                ("model", train_rows, test_rows), ("raw", raw_train, raw_test)):
            entry[f"energy_{name}"] = _energy_task(
                train_source["events"] if name == "model" else train_source["features"],
                train_source["deposited_MeV"], train_source["run_ids"],
                train_mask & np.asarray(train_source["contained"], bool),
                test_source["events"] if name == "model" else test_source["features"],
                test_source["deposited_MeV"], test_source["run_ids"],
                test_mask & np.asarray(test_source["contained"], bool))
        tasks["domains"][domain] = entry
    y_train = (np.asarray(train_rows["particle"]).astype(str) == "muon").astype(int)
    y_test = (np.asarray(test_rows["particle"]).astype(str) == "muon").astype(int)
    classification = {}
    for name, x_train, x_test, runs_train, runs_test, n_hit in (
            ("model", train_rows["events"], test_rows["events"],
             train_rows["run_ids"], test_rows["run_ids"], test_rows["n_hit"]),
            ("raw", raw_train["features"], raw_test["features"],
             raw_train["run_ids"], raw_test["run_ids"], raw_test["n_hit"])):
        scored = _logistic_auc(np.asarray(x_train, float), y_train, runs_train,
                               np.asarray(x_test, float), y_test)
        probability = scored.pop("probability")
        scored["nhits_quintiles"] = _nhits_quintile_auc(probability, y_test, n_hit)
        classification[name] = scored
    tasks["mue_classification"] = classification
    return tasks


def _a3_rows(payload) -> dict:
    return {name: np.asarray(payload[name]) for name in
            ("events", "run_ids", "event_ids", "particle", "direction", "deposited_MeV",
             "contained", "vertex", "n_hit")}


def readout_a3(*, root=None, inventory=None, config, gen, arms_config, geometry,
               data_root=DATA_ROOT_DEFAULT, device="cpu", log=print) -> dict:
    """§3 A-3 particle-task probes (muon/electron) on the frozen checkpoints."""
    root = _resolve_root(root)
    gate = assert_authorised("readout", root=root)
    inv = inventory if inventory is not None else build_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    truth = {"train": load_particle_truth(data_root, PROBE_KEYS_S005["train"]),
             "test": load_particle_truth(data_root, PROBE_KEYS_S005["test"])}
    raw = {role: raw_event_features(probes005[role], geometry, truth=truth[role])
           for role in ("train", "test")}
    raw_rows = {}
    for role in ("train", "test"):
        raw_rows[role] = {"events": raw[role]["features"],
                          "run_ids": raw[role]["run_ids"],
                          "particle": raw[role]["particle"],
                          "direction": raw[role]["direction"],
                          "deposited_MeV": raw[role]["deposited_MeV"],
                          "contained": raw[role]["contained"],
                          "n_hit": raw[role]["n_hit"]}
    payload = {"registered": False, "extension": EXTENSION,
               "check": "§3 A-3 particle-task probes", "runbook": RUNBOOK_REL,
               "labels": "simplified A10 Cherenkov generator; simulator-internal; "
                         "A-3 does not evaluate the registered R13",
               "raw_baseline": {"n_train": int(raw["train"]["features"].shape[0]),
                                "n_test": int(raw["test"]["features"].shape[0]),
                                "features": list(raw["train"]["feature_names"]),
                                "tasks": _a3_model_rows(raw_rows["train"],
                                                        raw_rows["test"],
                                                        raw["train"], raw["test"])},
               "models": {}, "unavailable": [],
               "gate": {"authorised": gate["authorised"], "status": gate["status"]}}
    committed = committed_vertex_map(root)
    for label in A3_LABELS:
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            payload["unavailable"].append({"label": label, "reason": "inventory unavailable"})
            continue
        try:
            with item_eval_context(row):
                arm = build_item_arm(row, config=config, gen=gen, arms_config=arms_config,
                                     geometry=geometry, data_root=data_root,
                                     device=device, log=log)
                model = getattr(arm, "jepa", arm)
                reproduction = _a3_photon_reproduction(row, model, probes005, geometry,
                                                       config, device, committed, label)
                if reproduction.get("checked") and not reproduction.get("pass"):
                    payload["unavailable"].append({"label": label,
                                                   "reason": "own reproduction mismatch",
                                                   "reproduction": reproduction})
                    continue
                particles = {role: stream_particle_rows(
                    model, probes005[role], truth[role], geometry, device=device, log=log)
                    for role in ("train", "test")}
            tasks = _a3_model_rows(_a3_rows(particles["train"]),
                                   _a3_rows(particles["test"]),
                                   raw["train"], raw["test"])
            tasks["contained_counts"] = {
                domain: {"train": int(((particles["train"]["particle"] == domain)
                                       & particles["train"]["contained"]).sum()),
                         "test": int(((particles["test"]["particle"] == domain)
                                      & particles["test"]["contained"]).sum())}
                for domain in PARTICLE_DOMAINS}
            tasks["photon_reproduction"] = reproduction
            payload["models"][label] = {"arm": row.get("arm"), "split": row.get("split"),
                                        "seed": int(row["seed"]), "tasks": tasks}
        except SystemExit:
            raise
        except Exception as problem:
            payload["unavailable"].append({"label": label,
                                           "reason": f"{type(problem).__name__}: {problem}"})
    payload["reproduction"] = _reproduction_summary(
        (model.get("tasks") or {}).get("photon_reproduction")
        for model in payload["models"].values())
    payload["complete"] = not payload["unavailable"]
    return payload


def _a3_photon_reproduction(row, model, probes, geometry, config, device, committed,
                            label) -> dict:
    metrics = stream_full_input_metrics(model, probes, geometry, split=row.get("split",
                                                                              "within"),
                                        config=config, device=device, with_mlp=False,
                                        zero_geometry=False)
    got = _full_input_values(metrics)["vertex_resolution_m"]
    return check_item_reproduction(label, got, committed)


# ================================================================ A-4 §3
def _time_matrix(features) -> np.ndarray:
    values = features.detach().cpu().numpy() if hasattr(features, "detach") \
        else np.asarray(features)
    return np.asarray(values, float)[..., pe.TIME_INDEX]


def context_time_shift(features, hidden, hit, convention, *, visible=None,
                       time_scale_ns=100.0) -> dict:
    """§3 A-4 context-only time reference: per-event shift in ns.

    ``visible`` defaults to ``hit & ~hidden`` (the context-only reference); pass
    ``hit`` for the all-token reference recomputed from the tokens.  Median models:
    the median raw time of the visible hits (0 if none).  Trigger models: the NHits
    trigger time of the visible hits (``None`` -> shift 0 and the event counted in
    ``n_ctx_no_trigger``).
    """
    times_feature = _time_matrix(features) * float(time_scale_ns)
    hit = np.asarray(hit, bool)
    hidden = np.asarray(hidden, bool)
    visible = (hit & ~hidden) if visible is None else (np.asarray(visible, bool) & hit)
    n_events = hit.shape[0]
    shifts = np.zeros(n_events, float)
    n_no_hit = n_no_trigger = 0
    nhits = None
    if convention == A4_TOKEN_TRIGGER:
        rule = lb.pmt_time_rule()
        nhits = rule[0]
    for event in range(n_events):
        selected = times_feature[event][visible[event]]
        if selected.size == 0:
            n_no_hit += 1
            if convention == A4_TOKEN_TRIGGER:
                n_no_trigger += 1        # no visible digit at all -> T_ctx is None
            continue
        if convention == A4_TOKEN_MEDIAN:
            shifts[event] = float(np.median(selected))
        elif convention == A4_TOKEN_TRIGGER:
            trigger = nhits(selected)
            if trigger is None:
                n_no_trigger += 1
            else:
                shifts[event] = float(trigger)
        else:
            raise ValueError(f"unknown time convention {convention!r}")
    return {"shift_ns": shifts, "n_ctx_no_hit": int(n_no_hit),
            "n_ctx_no_trigger": int(n_no_trigger), "convention": convention,
            "time_scale_ns": float(time_scale_ns)}


def apply_time_shift(features, hit, shift_ns, *, time_scale_ns=100.0):
    """Reduce every hit token's time feature by ``shift_ns / time_scale_ns`` (§3 A-4)."""
    import torch
    values = features.detach().clone() if hasattr(features, "detach") \
        else np.array(features, copy=True)
    times = _time_matrix(features)
    shifted = times - (np.asarray(shift_ns, float)[:, None] / float(time_scale_ns))
    mask = np.asarray(hit, bool)
    if hasattr(values, "detach"):
        tensor = torch.as_tensor(shifted, dtype=features.dtype, device=features.device)
        values[..., pe.TIME_INDEX] = torch.where(
            torch.as_tensor(mask, device=features.device),
            tensor, values[..., pe.TIME_INDEX])
    else:
        values[..., pe.TIME_INDEX] = np.where(mask, shifted,
                                              values[..., pe.TIME_INDEX])
    return values


def masked_reference_readout(model, probes, geometry, *, split, config, device,
                             convention, reference="context", forms=("context_only",
                                                                     "propagated"),
                             chunk_events=8, max_events=None, return_streams=False,
                             log=print) -> dict:
    """§3 A-4 readout of the masked pass under the all-token or context-only reference."""
    import torch
    scaling = pe.FeatureScaling.from_geometry(geometry)
    mask_seed = int(config["views"]["mask_seed"])
    out = {}
    for role in ("train", "test"):
        dataset = probes[role]
        rows = {form: {"events": [], "run_ids": [], "vertex": [], "log_yield": [],
                       "tokens": [], "token_y": []} for form in forms}
        rng = np.random.default_rng(er.SAMPLE_SEED)
        n_no_trigger = n_no_hit = 0
        for row in dataset.runs:
            source = dataset.directory / "runs" / row["run_dir"]
            truth = lr._run_truth(row, dataset, geometry)
            ids = lr._run_event_ids(source)
            allowed = lr._domain_event_ids(source, er.PRIMARY_DOMAIN)
            if allowed is not None:
                ids = np.intersect1d(ids, allowed)
            if ids.size == 0:
                continue
            if max_events is not None:
                ids = ids[:int(max_events)]
            for start in range(0, int(ids.size), int(chunk_events)):
                batch = lr._probe_batch(source, geometry, split=split,
                                        mask_seed=mask_seed, occupancy="natural",
                                        scaling=scaling,
                                        event_ids=ids[start:start + int(chunk_events)])
                if batch.event_ids.size == 0:
                    continue
                batch = lr._to_device(batch, device)
                selected = la._select_truth(truth, np.asarray(batch.event_ids))
                hit = np.asarray(batch.hit_flags, bool)
                hidden = np.asarray(batch.hidden.detach().cpu(), bool)
                visible = hit if reference == "all" else (hit & ~hidden)
                info = context_time_shift(batch.features, hidden, hit, convention,
                                          visible=visible)
                n_no_trigger += int(info["n_ctx_no_trigger"])
                n_no_hit += int(info["n_ctx_no_hit"])
                shifted = apply_time_shift(batch.features, hit, info["shift_ns"])
                with torch.no_grad():
                    tokens = model.encode(shifted, hidden_mask=batch.hidden)
                token_np = tokens.detach().to(torch.float32).cpu().numpy()
                log_charge = np.asarray(batch.log_charge, float)
                for form in forms:
                    mask = hidden if form == "propagated" else ~hidden
                    event = la._mean_tokens(token_np, mask)
                    bucket = rows[form]
                    bucket["events"].append(np.asarray(event, np.float32))
                    bucket["run_ids"].append(np.full(len(batch.event_ids),
                                                     row["run_id"], dtype=object))
                    bucket["vertex"].append(np.asarray(selected["vertex"], float))
                    bucket["log_yield"].append(np.asarray(selected["log_yield"], float))
                    candidate = hit & mask
                    for event_index in range(candidate.shape[0]):
                        positions = np.flatnonzero(candidate[event_index])
                        if positions.size == 0:
                            continue
                        take = min(er.TOKENS_PER_EVENT, int(positions.size))
                        pick = (rng.choice(positions, size=take, replace=False)
                                if positions.size > take else positions)
                        bucket["tokens"].append(token_np[event_index, pick])
                        bucket["token_y"].append(log_charge[event_index, pick])
        streams = {}
        for form in forms:
            bucket = rows[form]
            streams[form] = er.FormStream(
                form=form,
                events=(np.concatenate(bucket["events"]) if bucket["events"]
                        else np.zeros((0, 0), np.float32)),
                run_ids=(np.concatenate(bucket["run_ids"]) if bucket["run_ids"]
                         else np.zeros(0, dtype=object)),
                vertex=(np.concatenate(bucket["vertex"]) if bucket["vertex"]
                        else np.zeros((0, 3))),
                log_yield=(np.concatenate(bucket["log_yield"]) if bucket["log_yield"]
                           else np.zeros(0)),
                tokens=(np.concatenate(bucket["tokens"]) if bucket["tokens"]
                        else np.zeros((0, 0), np.float32)),
                token_y=(np.concatenate(bucket["token_y"]) if bucket["token_y"]
                         else np.zeros(0)))
        out[role] = {"streams": streams, "n_ctx_no_trigger": int(n_no_trigger),
                     "n_ctx_no_hit": int(n_no_hit)}
    metrics = {}
    for form in forms:
        train = out["train"]["streams"][form]
        test = out["test"]["streams"][form]
        metrics[form] = er.form_metrics(train, test)
    payload = {"metrics": metrics,
               "reference": reference, "convention": convention,
               "counts": {"train": {"n_ctx_no_trigger": out["train"]["n_ctx_no_trigger"],
                                    "n_ctx_no_hit": out["train"]["n_ctx_no_hit"]},
                          "test": {"n_ctx_no_trigger": out["test"]["n_ctx_no_trigger"],
                                   "n_ctx_no_hit": out["test"]["n_ctx_no_hit"]}}}
    if return_streams:
        payload["streams"] = {role: dict(out[role]["streams"])
                              for role in ("train", "test")}
    return payload


def readout_a4(*, root=None, inventory=None, config, gen, arms_config, geometry,
               data_root=DATA_ROOT_DEFAULT, device="cpu", log=print) -> dict:
    """§3 A-4 time-reference leakage: context-only vs all-token reference, per seed."""
    root = _resolve_root(root)
    gate = assert_authorised("readout", root=root)
    inv = inventory if inventory is not None else build_inventory(root, log=log)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    committed = committed_vertex_map(root)
    payload = {"registered": False, "extension": EXTENSION,
               "check": "§3 A-4 time-reference leakage", "runbook": RUNBOOK_REL,
               "threshold_m": A4_VERTEX_FLOOR_M, "models": {}, "unavailable": [],
               "residual_leak": "the prompt-window cut decides which context hits exist; "
                                "removing it needs re-digitisation (out of scope)",
               "gate": {"authorised": gate["authorised"], "status": gate["status"]}}
    for label in A4_LABELS:
        row = inv["rows"].get(label)
        if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
            payload["unavailable"].append({"label": label, "reason": "inventory unavailable"})
            continue
        convention = ((row.get("eval") or {}).get("time_convention")
                      or A4_TOKEN_TRIGGER)
        try:
            # the committed evaluation configuration (median wrapper included) is used
            # only to reproduce the item's own committed full-input vertex; the A-4
            # reference readout below consumes the *raw* token times.
            with item_eval_context(row):
                arm = build_item_arm(row, config=config, gen=gen,
                                     arms_config=arms_config, geometry=geometry,
                                     data_root=data_root, device=device, log=log)
                model = getattr(arm, "jepa", arm)
                reproduction = _a4_reproduction(row, model, probes005, geometry, config,
                                                device, committed, label)
            if reproduction.get("checked") and not reproduction.get("pass"):
                payload["unavailable"].append({"label": label,
                                               "reason": "own reproduction mismatch",
                                               "reproduction": reproduction})
                continue
            all_token = masked_reference_readout(
                model, probes005, geometry, split=row.get("split", "within"),
                config=config, device=device, convention=convention,
                reference="all", log=log)
            context = masked_reference_readout(
                model, probes005, geometry, split=row.get("split", "within"),
                config=config, device=device, convention=convention,
                reference="context", log=log)
            entry = {"arm": row.get("arm"), "split": row.get("split"),
                     "seed": int(row["seed"]), "convention": convention,
                     "checkpoint_sha256": row.get("sha256"),
                     "all_token": all_token, "context_only": context,
                     "reproduction": reproduction}
            for form in ("context_only", "propagated"):
                entry[f"delta_{form}_m"] = _delta_vertex(
                    all_token["metrics"].get(form) or {},
                    context["metrics"].get(form) or {})
            payload["models"][label] = entry
        except SystemExit:
            raise
        except Exception as problem:
            payload["unavailable"].append({"label": label,
                                           "reason": f"{type(problem).__name__}: {problem}"})
    payload["readings"] = a4_readings(payload["models"])
    payload["reproduction"] = _reproduction_summary(
        model.get("reproduction") for model in payload["models"].values())
    payload["complete"] = not payload["unavailable"]
    return payload


def _delta_vertex(all_token, context_only) -> dict:
    a = all_token.get("vertex_resolution_m")
    b = context_only.get("vertex_resolution_m")
    if a is None or b is None:
        return {"available": False, "delta_m": None}
    all_r2 = (all_token.get("token_log_charge_r2") or {}).get("r2")
    context_r2 = (context_only.get("token_log_charge_r2") or {}).get("r2")
    return {"available": True, "delta_m": float(b - a), "all_token_m": float(a),
            "context_only_m": float(b),
            "context_only_minus_all_token_token_r2":
                (None if all_r2 is None or context_r2 is None
                 else float(context_r2 - all_r2))}


def _a4_reproduction(row, model, probes, geometry, config, device, committed, label):
    metrics = stream_full_input_metrics(model, probes, geometry, split=row.get("split",
                                                                              "within"),
                                        config=config, device=device, with_mlp=False,
                                        zero_geometry=False)
    got = _full_input_values(metrics)["vertex_resolution_m"]
    return check_item_reproduction(label, got, committed)


def _reproduction_summary(records) -> dict:
    """Group the per-item reproduction records into checked / not_applicable."""
    out = {"rule": "§3 A1: each recomputed item must reproduce its own committed "
                   "full-input vertex within 1e-3 m; a mismatch marks the row unavailable",
           "checked": {}, "not_applicable": []}
    for record in records:
        if not record:
            continue
        label = record.get("label")
        if record.get("status") == "not_applicable":
            out["not_applicable"].append(label)
        else:
            out["checked"][label or "?"] = record
    return out


def a4_readings(models) -> dict:
    """§3 A-4 readings: material / immaterial / unclear, per model (seed pair)."""
    groups = _seed_pair_groups()
    by_model = {}
    for prefix in ("mae_within_trigger", "mae_within_median", "fix", "frozen_target"):
        pair = groups.get(prefix, (None, None))
        if not all(label in models for label in pair):
            continue
        rows = [models[label] for label in pair]
        all_vertex = [(row["all_token"]["metrics"].get("context_only") or {})
                      .get("vertex_resolution_m") for row in rows]
        context_vertex = [(row["context_only"]["metrics"].get("context_only") or {})
                          .get("vertex_resolution_m") for row in rows]
        if any(value is None for value in all_vertex + context_vertex):
            by_model[prefix] = {"reading": "unavailable"}
            continue
        deltas = [float(context - all_token) for context, all_token in
                  zip(context_vertex, all_vertex)]
        spread = abs(float(all_vertex[0]) - float(all_vertex[1]))
        threshold = max(A4_VERTEX_FLOOR_M, spread)
        material = all(abs(value) > threshold for value in deltas)
        immaterial = all(abs(value) <= A4_VERTEX_FLOOR_M for value in deltas)
        reading = "material" if material else ("immaterial" if immaterial else "unclear")
        by_model[prefix] = {
            "reading": reading,
            "delta_m": {str(seed): value for seed, value in zip((9000, 9001), deltas)},
            "all_token_vertex_m": {str(seed): value for seed, value in
                                   zip((9000, 9001), all_vertex)},
            "context_only_vertex_m": {str(seed): value for seed, value in
                                      zip((9000, 9001), context_vertex)},
            "seed_spread_all_token_m": spread, "threshold_m": threshold,
            "token_charge_r2_delta": {
                str(seed): (models[label]["delta_context_only_m"]
                            .get("context_only_minus_all_token_token_r2"))
                for seed, label in zip((9000, 9001), pair)},
            "rule": "Δ = context-only minus all-token (context_only form, per seed); "
                    "material: |Δvertex| > max(0.01 m, |v9000 - v9001| of the all-token "
                    "readout) in both seeds; immaterial: |Δvertex| <= 0.01 m in both "
                    "seeds; else unclear"}
    return {"models": by_model,
            "note": "half_pass is descriptive only (its target sub-pass sees the target "
                    "tokens) and is not part of this reading"}

# =================================================================== T-1 §4
def t1_cell(config, variant):
    """§4 new cells: the dev cell with ``setting``/``s_arm`` = 0.30 and a T-1 cell_id."""
    from scripts import jepa_pregrid_checks as pregrid
    variant = str(variant)
    if variant not in T1_VARIANTS:
        raise ValueError(f"unknown T-1 variant {variant!r}; expected {T1_VARIANTS}")
    if variant == "s030_within_trigger":
        base = lr.dev_cell(config)
    else:
        base = pregrid.variant_cell(config, "across")
    return dataclasses.replace(base, cell_id=T1_CELL_IDS[variant], setting=S030, s_arm=S030)


def t1_cells(config) -> dict:
    return {variant: t1_cell(config, variant) for variant in T1_VARIANTS}


def t1_run_kwargs(*, cell, seed, config, arms_config, geometry, loader_dataset,
                  probe_datasets, out_root, data_root, config_sha, code_sha, status_text,
                  device, steps) -> dict:
    """The exact ``lr.run_one`` keyword arguments of ``jepa_ext_dejitter.mode_train``.

    It differs from ``mode_train`` only in ``cell``, ``out_root``, ``loader_dataset``
    and ``probe_datasets`` (§4 required check 1); ``mode_train`` is neither edited
    nor monkeypatched.
    """
    return {"arm_name": T1_ARM, "cell": cell, "seed": int(seed), "beta": None,
            "config": config, "arms_config": arms_config, "geometry": geometry,
            "loader_dataset": loader_dataset, "probe_datasets": probe_datasets,
            "dark_bank": None, "out_root": Path(out_root), "data_root": Path(data_root),
            "config_sha": config_sha, "code_sha": code_sha, "status_text": status_text,
            "device": device, "steps": int(steps), "evaluate": True,
            "probe_max_events": None, "physics_method": "ridge", "save_embeddings": False}


REQUIRED_A1_S030_ROWS = ("init9000", "strong_drift9000")


def require_a1_s030_gate(ext_root=EXT_ROOT, *, path=None) -> dict:
    """§5: T-1 launches only after both A1 s = 0.30 rows pass.

    Reads ``<ext>/stageA/A1/reproduction.json`` (written by ``reproduce``) and
    refuses the launch with ``SystemExit(2)`` when it is missing or
    ``s030_pass`` is not true.  The required rows are named in the error.
    """
    path = Path(path) if path is not None else (
        Path(ext_root) / "stageA" / "A1" / "reproduction.json")
    payload = read_json(path, None)
    if payload is None:
        print(f"[followup] T-1 launch refused: the A1 s = 0.30 gate record {path} is "
              f"missing; every T-1 launch requires both A1 s = 0.30 rows "
              f"({', '.join(REQUIRED_A1_S030_ROWS)}) to pass (§5: 'T-1 launches only "
              f"after both A1 s = 0.30 rows pass')", file=sys.stderr)
        raise SystemExit(2)
    s030 = payload.get("s030") or {}
    failed = []
    for label in REQUIRED_A1_S030_ROWS:
        row = s030.get(label) or {}
        vertex = (row.get("vertex") or {}).get("pass")
        gain = (row.get("module_gain_r2") or {}).get("pass")
        if not (vertex and gain):
            failed.append(label)
    if payload.get("s030_pass") is not True or failed:
        print(f"[followup] T-1 launch refused: {path} does not show both A1 s = 0.30 "
              f"rows passing ({', '.join(REQUIRED_A1_S030_ROWS)}); s030_pass="
              f"{payload.get('s030_pass')!r}, failed rows={failed}", file=sys.stderr)
        raise SystemExit(2)
    return {"path": str(path), "s030_pass": True,
            "s005_pass": payload.get("s005_pass"),
            "rows": sorted(s030), "required_rows": list(REQUIRED_A1_S030_ROWS)}


def t1_prepare_launch(*, root=None, config, data_root, variant, seed, steps, out_root,
                      gen=None, geometry=None, ext_root=None, log=print) -> dict:
    """§4 steps 1-3 + §5 dependency: gate, A1 s = 0.30, run-dir refusal, hash check."""
    from scripts import jepa_pregrid_checks as pregrid
    ext_base = Path(ext_root) if ext_root is not None else Path(out_root).parent
    gate = assert_authorised("train-t1", root=root, arm=T1_ARM, variant=variant,
                             seed=int(seed), steps=int(steps), out_root=out_root,
                             ext_root=ext_base)
    # §5: T-1 launches only after both A1 s = 0.30 rows pass (B3)
    a1_gate = require_a1_s030_gate(ext_base)
    cell = t1_cell(config, variant)
    run_id = lr.run_id_for(cell.cell_id, T1_ARM, int(seed))
    refuse_existing_run_dir(out_root, run_id)
    require_pinned(data_root, keys=[TRAIN_KEY_S030, PROBE_KEYS_S030["train"],
                                    PROBE_KEYS_S030["test"]],
                   context=f"T-1 {variant} s{int(seed)}")
    train_ds, probes = pregrid.datasets_for(cell, config, gen, geometry,
                                            Path(data_root), code_sha=None)
    if train_ds.key != TRAIN_KEY_S030:
        raise RuntimeError(f"T-1 loader dataset {train_ds.key!r} != pinned {TRAIN_KEY_S030!r}")
    expected_probes = set(PROBE_KEYS_S030.values())
    if {probe.key for probe in probes.values()} != expected_probes:
        raise RuntimeError(f"T-1 probe sets {sorted(probe.key for probe in probes.values())} "
                           f"!= pinned {sorted(expected_probes)}")
    return {"gate": gate, "a1_gate": a1_gate, "cell": cell, "run_id": run_id,
            "train_ds": train_ds, "probes": probes}


def train_t1(*, root=None, config, gen, arms_config, geometry, data_root, config_path,
             out_root, variant, seed, steps, device="cpu", code_sha=None,
             status_text=None, host_check=True, ext_root=None, log=print) -> dict:
    """§4: one authorised T-1 run (or the single 20-step smoke), no other training."""
    from scripts import jepa_pregrid_checks as pregrid
    root = _resolve_root(root)
    steps = int(steps)
    ext_base = Path(ext_root) if ext_root is not None else Path(out_root).parent
    prepared = t1_prepare_launch(root=root, config=config, data_root=data_root,
                                 variant=variant, seed=int(seed), steps=steps,
                                 out_root=out_root, gen=gen, geometry=geometry,
                                 ext_root=ext_base, log=log)
    cell = prepared["cell"]
    if code_sha is None:
        code_sha = lr.git_head(root)
    if status_text is None:
        status_text = lr.git_status(root)
    kwargs = t1_run_kwargs(cell=cell, seed=int(seed), config=config,
                           arms_config=arms_config, geometry=geometry,
                           loader_dataset=prepared["train_ds"],
                           probe_datasets=prepared["probes"], out_root=out_root,
                           data_root=data_root,
                           config_sha=lr.sha256_file(Path(config_path)),
                           code_sha=code_sha, status_text=status_text, device=device,
                           steps=steps)
    started = time.time()
    result = lr.run_one(**kwargs)
    run_dir = Path(result["run_dir"])
    final = run_dir / "final.pt"
    if not final.is_file():
        raise RuntimeError(f"T-1 run {prepared['run_id']} produced no final.pt")
    arm, _beta = pregrid.build_loaded(T1_ARM, cell, geometry, arms_config, config, gen,
                                      Path(data_root), str(final), int(seed))
    forms = pregrid.evaluate_forms(arm, T1_ARM, cell, geometry, prepared["probes"],
                                   config, device, run_dir / "ext_eval",
                                   probe_max_events=None)
    del arm
    log_rows = ([json.loads(line) for line in
                 (run_dir / "train_log.jsonl").read_text().splitlines() if line.strip()]
                if (run_dir / "train_log.jsonl").is_file() else [])
    summary = {"registered": False, "extension": EXTENSION, "variant": str(variant),
               "transform": "none", "arm": T1_ARM, "beta": None, "cell": cell.cell_id,
               "split": cell.split, "spread": cell.setting, "seed": int(seed),
               "final_train_metrics": log_rows[-1] if log_rows else None,
               "forms": forms, "wall_seconds": result["manifest"].get("wall_seconds"),
               "checkpoint": str(final), "checkpoint_sha256": lr.sha256_file(final)}
    (run_dir / "ext_summary.json").write_text(
        json.dumps(lr._safe(summary), indent=1, default=float) + "\n")
    host = (t1_host_manifest_check(ext_root=ext_base, variant=variant,
                                   seed=int(seed), smoke=(steps == T1_SMOKE_STEPS),
                                   log=log) if host_check else None)
    cost = t1_cost(ext_base)          # N4: measured T-1 GPU hours vs the 3.5 h cap
    return {"run_id": prepared["run_id"], "run_dir": str(run_dir), "summary": summary,
            "host_check": host, "gate": prepared["gate"],
            "a1_gate": prepared["a1_gate"], "cost": cost,
            "wall_seconds": round(time.time() - started, 3)}


def _manifest_row(path, run_id):
    path = Path(path)
    if not path.is_file():
        return None
    found = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if str(row.get("run_id")) == str(run_id):
            found = row
    return found


def t1_host_manifest_check(*, ext_root=EXT_ROOT, variant, seed, smoke=False,
                           reference_manifest=T1_REFERENCE_MANIFEST, log=print) -> dict:
    """§4 required check 2: compare the new ``manifest.jsonl`` row with the s = 0.05 row.

    The rows may differ only in ``T1_ALLOWED_DIFFERENCES`` (plus
    ``T1_SMOKE_EXTRA_DIFFERENCES`` for the smoke); the required values are checked
    explicitly.  Any other difference is S9.
    """
    variant = str(variant)
    if variant not in T1_VARIANTS:
        raise ValueError(f"unknown T-1 variant {variant!r}")
    run_id = lr.run_id_for(T1_CELL_IDS[variant], T1_ARM, int(seed))
    root = Path(ext_root) / ("smoke" if smoke else "train")
    new_row = _manifest_row(root / "manifest.jsonl", run_id)
    reference_variant = {"s030_within_trigger": "within_trigger",
                         "s030_across_trigger": "across"}[variant]
    reference_run_id = (f"ext_dev_s005_{reference_variant}__{T1_ARM}__"
                        f"s{int(seed):04d}")
    reference_row = _manifest_row(reference_manifest, reference_run_id)
    allowed = set(T1_ALLOWED_DIFFERENCES)
    if smoke:
        allowed |= set(T1_SMOKE_EXTRA_DIFFERENCES)
    problems = []
    if new_row is None:
        problems.append(f"no manifest row {run_id} under {root}")
    if reference_row is None:
        problems.append(f"no s = 0.05 reference row {reference_run_id} in "
                        f"{reference_manifest}")
    differences, unexpected = [], []
    if new_row is not None and reference_row is not None:
        keys = sorted(set(new_row) | set(reference_row))
        differences = [key for key in keys
                       if new_row.get(key) != reference_row.get(key)]
        unexpected = [key for key in differences if key not in allowed]
    required = {}
    if new_row is not None:
        required["outcome_complete"] = new_row.get("outcome") == "complete"
        required["resumed_from_step"] = new_row.get("resumed_from_step") in (0, None)
        required["loader_dataset_manifest_sha256"] = (
            new_row.get("loader_dataset_manifest_sha256") == T1_REQUIRED_LOADER_SHA)
        probe_shas = new_row.get("probe_dataset_sha256") or {}
        required["probe_dataset_sha256"] = (set(probe_shas.values())
                                            == T1_REQUIRED_PROBE_SHAS)
        if reference_row is not None:
            required["reference_outcome_complete"] = (
                reference_row.get("outcome") == "complete")
    if unexpected:
        problems.append(f"unexpected manifest differences (S9): {unexpected}")
    failed = [name for name, ok in required.items() if not ok]
    if failed:
        problems.append(f"required values failed: {failed}")
    payload = {"registered": False, "extension": EXTENSION,
               "check": "§4 T-1 host manifest comparison", "variant": variant,
               "seed": int(seed), "smoke": bool(smoke), "run_id": run_id,
               "reference_run_id": reference_run_id,
               "manifest": str(root / "manifest.jsonl"),
               "reference_manifest": str(reference_manifest),
               "new_row_found": new_row is not None,
               "reference_row_found": reference_row is not None,
               "differences": differences, "unexpected_differences": unexpected,
               "allowed_differences": sorted(allowed),
               "required": required, "problems": problems, "pass": not problems}
    log(f"[T-1] host check {run_id}: {'PASS' if payload['pass'] else 'FAIL'} "
        f"{problems}")
    return payload


def t1_manifest_rows(ext_root=EXT_ROOT) -> list:
    """All completed/attempted rows in ``<ext>/{train,smoke}/manifest.jsonl``."""
    rows = []
    for root_name in ("train", "smoke"):
        path = Path(ext_root) / root_name / "manifest.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row["_root"] = root_name
            rows.append(row)
    return rows


def t1_cost(ext_root=EXT_ROOT, *, cap_hours=T1_CAP_H,
            projected_hours=T1_PROJECTED_GPU_H) -> dict:
    """N4: measured T-1 GPU hours (sum of completed-run ``wall_seconds``) vs the cap."""
    rows = t1_manifest_rows(ext_root)
    completed = [row for row in rows if row.get("outcome") == "complete"]
    wall_seconds = sum(float(row.get("wall_seconds") or 0.0) for row in completed)
    measured = wall_seconds / 3600.0
    return {"cap_hours": float(cap_hours),
            "projected_hours": float(projected_hours),
            "n_manifest_rows": len(rows), "n_completed_rows": len(completed),
            "measured_wall_seconds": round(wall_seconds, 3),
            "measured_gpu_hours": round(measured, 4),
            "cap_pass": bool(measured <= float(cap_hours)),
            "projection_pass": bool(float(projected_hours) <= float(cap_hours)),
            "rule": "GPU device-hours = sum of completed-run wall_seconds (runbook §4); "
                    "cap 3.5 L40S GPU-h"}


def t1_inventory_rows(ext_root=EXT_ROOT) -> dict:
    """Inventory rows for completed T-1 runs (for the post-run A-1/A-2 readouts)."""
    rows = {}
    for smoke, root_name in ((False, "train"), (True, "smoke")):
        manifest = Path(ext_root) / root_name / "manifest.jsonl"
        if not manifest.is_file():
            continue
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("arm")) != T1_ARM or row.get("outcome") != "complete":
                continue
            cell_id = str(row.get("cell_id", ""))
            variant = {"ext_dev_s030_within_trigger": "s030_within_trigger",
                       "ext_dev_s030_across_trigger": "s030_across_trigger"}.get(cell_id)
            if variant is None:
                continue
            run_id = str(row.get("run_id"))
            source = Path(ext_root) / root_name / "runs" / run_id / "final.pt"
            # Acceptance check 6 Nov 2026: the 20-step smoke has the same variant and
            # seed as the seed-9000 within run, so a shared label let the smoke row
            # (read second) replace the real 10 000-step run in the --include-t1
            # readouts.  The smoke keeps its own "t1smoke_" label, which
            # readout_label_list (prefix "t1_") never evaluates.
            prefix = "t1smoke_" if smoke else "t1_"
            label = f"{prefix}{variant}{int(row.get('seed'))}"
            item = _row(label, family="t1", arm=T1_ARM,
                        split="across" if "across" in variant else "within",
                        seed=int(row.get("seed")), beta=None, source=source,
                        sha256=row.get("final_checkpoint_sha256"), data_setting=S030,
                        status="available" if source.is_file() else "unavailable")
            item["smoke"] = bool(smoke)
            rows[label] = item
    return rows


def t1_readings(*, a2_payload=None, a1_payload=None, ext_root=EXT_ROOT) -> dict:
    """§4 readings G-1 (gain R² minus the 0.841585 floor) and G-2 (paired vertex change)."""
    out = {"registered": False, "extension": EXTENSION,
           "check": "§4 T-1 readings (two-sided, descriptive)",
           "G-1": {"floor": MODULE_GAIN_FLOOR[S030], "rule":
                   "retained: gain R² - floor >= -0.10 in both seeds; suppressed: "
                   "<= -0.30 in both seeds; otherwise intermediate", "models": {}},
           "G-2": {"rule": "full-input vertex of the s = 0.30-trained MAE minus that of "
                           "the s = 0.05-trained MAE, paired events, A-2 bootstrap; "
                           "descriptive, no threshold", "models": {}},
           "A-1": {}}
    a2_models = (a2_payload or {}).get("models") or {}
    for label, model in a2_models.items():
        if not str(label).startswith("t1_"):
            continue
        s030 = (model.get("environments") or {}).get("s030") or {}
        gain = s030.get("module_gain_r2")
        minus = s030.get("module_gain_minus_floor")
        if gain is None:
            out["G-1"]["models"][label] = {"available": False}
            continue
        if minus is not None and float(minus) >= -0.10:
            reading = "retained"
        elif minus is not None and float(minus) <= -0.30:
            reading = "suppressed"
        else:
            reading = "intermediate"
        out["G-1"]["models"][label] = {"available": True, "gain_r2": gain,
                                       "gain_minus_floor": minus, "reading": reading}
    for key, block in ((a2_payload or {}).get("t1_g2") or {}).items():
        out["G-2"]["models"][key] = dict(block, available=block.get("status") is None)
    a1_rows = (a1_payload or {}).get("rows") or {}
    out["A-1"] = {str(label): row for label, row in a1_rows.items()
                  if str(label).startswith("t1_")}
    return out


# ================================================================ report §6.5
def _readout_path(ext_root, item) -> Path:
    return Path(ext_root) / "stageA" / item / f"readout_{item}.json"


def assemble_report(*, root=None, ext_root=EXT_ROOT, results_dir=None,
                    results_root=None, log=print) -> dict:
    """§6.5 report: merge the Stage A readouts and the T-1 evidence into the results dir."""
    root = _resolve_root(root)
    gate = assert_authorised("report", root=root)
    ext_root = Path(ext_root)
    stage_a = ext_root / "stageA"
    out = (Path(results_dir) if results_dir is not None else
           (Path(results_root) if results_root is not None else root / RESULTS_DIR))
    plan = read_json(stage_a / "A0" / "plan.json", {}) or {}
    inventory = read_json(stage_a / "A0" / "inventory.json", {}) or {}
    reproduction_payload = read_json(stage_a / "A1" / "reproduction.json", {}) or {}
    readouts = {item: read_json(_readout_path(ext_root, item), {}) or {}
                for item in READOUT_ITEMS}
    t1_rows = t1_inventory_rows(ext_root)
    t1_a2 = readouts.get("A-2") or {}
    t1_a1 = readouts.get("A-1") or {}
    t1_cost_payload = t1_cost(ext_root)
    t1_reading = t1_readings(a2_payload=t1_a2, a1_payload=t1_a1, ext_root=ext_root)
    t1_reading["cost"] = t1_cost_payload
    readings = {"registered": False, "extension": EXTENSION,
                "A-1": (readouts.get("A-1") or {}).get("readings"),
                "A-1b": (readouts.get("A-1b") or {}).get("readings"),
                "A-2": (readouts.get("A-2") or {}).get("models"),
                "A-3": (readouts.get("A-3") or {}).get("models"),
                "A-4": (readouts.get("A-4") or {}).get("readings"),
                "T-1": t1_reading}
    runs = {"registered": False, "extension": EXTENSION, "stageA": {},
            "t1": t1_rows, "t1_cost": t1_cost_payload}
    for item, payload in readouts.items():
        runs["stageA"][item] = {
            "available": bool(payload), "complete": payload.get("complete"),
            "unavailable": payload.get("unavailable")}
        if item == "A-1":
            runs["stageA"][item]["rows"] = payload.get("rows")
        elif item == "A-1b":
            runs["stageA"][item]["rows"] = payload.get("rows")
        elif item == "A-2":
            runs["stageA"][item]["models"] = payload.get("models")
        elif item == "A-3":
            runs["stageA"][item]["models"] = payload.get("models")
        elif item == "A-4":
            runs["stageA"][item]["models"] = payload.get("models")
    output = {"registered": False, "extension": EXTENSION,
              "runbook": RUNBOOK_REL, "runbook_sha256": RUNBOOK_SHA256,
              "code_sha": lr.git_head(root), "gate": {"status": gate["status"]},
              "plan": plan, "inventory": inventory.get("counts"),
              "reproduction": reproduction_payload, "readouts": readouts}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "runs.json", runs)
    write_json(out / "readings.json", readings)
    write_json(out / "reproduction.json", reproduction_payload)
    write_json(out / "plan.json", plan)
    write_json(out / "inventory.json", inventory)
    write_json(out / "output.json", output)
    readme = _report_readme(plan, inventory, reproduction_payload, readouts, readings,
                            t1_rows)
    (out / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(out.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 1_000_000}
    print(json.dumps({"out": str(out), "readouts": sorted(readouts),
                      "reproduction_pass": reproduction_payload.get(
                          "reproduction_pass"),
                      "t1_runs": sorted(t1_rows), "sizes": sizes,
                      "too_big": too_big}, default=float), flush=True)
    return {"out": str(out), "sizes": sizes, "too_big": too_big,
            "reproduction_pass": reproduction_payload.get("reproduction_pass"),
            "gate": gate}


def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _report_readme(plan, inventory, reproduction, readouts, readings, t1_rows) -> str:
    lines = ["# MAE/JEPA follow-up — Stage A readouts and T-1 control",
             "",
             "`registered: false`. Extension `mae_jepa_followup_2026-11-03`; runbook "
             f"`{RUNBOOK_REL}` (sha256 `{RUNBOOK_SHA256}`).",
             "",
             f"- code SHA: `{plan.get('code', {}).get('git_head')}`",
             f"- Stage A projection: "
             f"{_fmt((plan.get('stage_a') or {}).get('projected_l40s_hours'), 3)} L40S h "
             f"(cap 5.5); entries pass: "
             f"{(plan.get('entries') or {}).get('pass')}",
             f"- inventory: {_fmt((inventory.get('counts') or {}).get('labels'))} labels, "
             f"{_fmt((inventory.get('counts') or {}).get('unavailable'))} unavailable, "
             f"{_fmt((inventory.get('counts') or {}).get('pending'))} pending",
             f"- §3 A1 reproduction: "
             f"{'PASS' if reproduction.get('reproduction_pass') else 'FAIL'}",
             "", "## Stage A readouts", "",
             "| item | complete | unavailable |", "|---|---|---|"]
    for item in READOUT_ITEMS:
        payload = readouts.get(item) or {}
        lines.append(f"| {item} | {payload.get('complete')} | "
                     f"`{payload.get('unavailable')}` |")
    t1_cost_payload = (readings.get("T-1") or {}).get("cost") or {}
    lines += ["", "## T-1 runs", "",
              "| label | split | seed | checkpoint sha256 |", "|---|---|---:|---|"]
    for label, row in t1_rows.items():
        lines.append(f"| {label} | {row.get('split')} | {row.get('seed')} | "
                     f"`{row.get('sha256')}` |")
    lines += ["", "## T-1 GPU cost (N4)", "",
              f"- cap: {_fmt(t1_cost_payload.get('cap_hours'), 3)} L40S GPU-h",
              f"- projected: {_fmt(t1_cost_payload.get('projected_hours'), 3)} h",
              f"- measured (sum of completed-run wall_seconds): "
              f"{_fmt(t1_cost_payload.get('measured_gpu_hours'), 4)} h",
              f"- pass: {t1_cost_payload.get('cap_pass')}"]
    lines += ["", "## Readings (§3/§4, fixed in advance)", "",
              "- A-1: `readings.json` -> `A-1` (J-a variant classes, J-b MAE checkpoints "
              "beside init/E3/static, J-c Spearman over the fixed population).",
              "- A-1b: `readings.json` -> `A-1b` (M/C/All MLP R² and the ordered classes; "
              "both seeds must agree).",
              "- A-2: `readings.json` -> `A-2` (gain R² minus the environment floor, "
              "refit and frozen vertices, log-yield R², paired run-cluster bootstrap).",
              "- A-3: `readings.json` -> `A-3` (μ/e AUC and balanced accuracy overall and "
              "in NHits quintiles, direction ridge/MLP angular errors, contained-energy "
              "probes under the 500 rule, raw-feature and random-init baselines).",
              "- A-4: `readings.json` -> `A-4` (material/immaterial/unclear per model, "
              "Δ token charge R² descriptive).",
              "- T-1: `readings.json` -> `T-1` (G-1 retained/suppressed/intermediate; "
              "G-2 paired vertex difference, descriptive).",
              "", "## Outputs", "",
              "- `plan.json`, `inventory.json`, `reproduction.json`, `runs.json`, "
              "`readings.json`, `README.md` (§6.5 deliverables).", ""]
    return "\n".join(lines) + "\n"
