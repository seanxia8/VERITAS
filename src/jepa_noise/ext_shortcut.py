# SPDX-License-Identifier: MIT
"""Shortcut-collapse follow-up extension (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md`` (7 Nov 2026,
Part B).  Extension name: ``shortcut_followup_2026-11-07``.

Part B has three pieces:

* **R-1** (CPU, report only): after the Part A repair pilot has produced its X1/X2
  readings, compare how repairable the generative (MAE) encoders are against the
  joint-embedding (JEPA) encoders.  R-1 reads the pilot's
  ``results/jepa/20261106_repair/`` outputs read-only and is ``not_run`` until
  they exist.  It never writes into the pilot's ``<ext>`` or results directory.
* **R-2** (GPU, training authorised): the variant ``target_posfree`` -- the
  fixed-label SIGReg-JEPA with **one** change: the label pass receives a clone of
  ``batch.features`` with columns 0-5 (scaled position and direction) set to
  zero.  The context pass, the full pass, the predictor and its position
  queries, the SIGReg terms, the loss weights, the optimiser, the schedule, the
  data, the cell and the seeds are unchanged; the label is **not** detached and
  the encoder stays shared and trainable.
* **R-3** (desk, done at design time): citation only (the CLI ``report`` mode
  writes the four references into the README).

The variant is installed by rebinding ``arm._encode_views`` through a
process-level ``lucid_run.build_arm`` wrapper
(:func:`build_arm_patch` / :func:`attach_posfree_target`), as in
``ext_static.attach_static_target`` / ``ext_static.build_arm_patch``; the
structure of ``prepare_run_dir``, ``run_one_training``, ``evaluate_run_dirs``,
``summarize_run`` follows ``ext_static`` and is renamed here.

Gate, import closure and provenance
-----------------------------------
:func:`assert_authorised` is the Part B §9.2 gate (``SystemExit(2)`` on any
failure).  Every mode re-checks, at its end, that every repository module in
``sys.modules`` (by ``__file__``) lies inside ``<P>`` or the forced-import
exceptions (:func:`import_closure`).  The dry run asserts the same closure.

Importing ``ext_followup`` (in ``<P>``; needed for
``FORCED_IMPORT_EXCEPTIONS`` and its pure ``_classify_event``) also imports
``src/jepa_noise/ext_arch.py`` and ``src/jepa_noise/ext_dejitter.py``, which are
**not** in Part B's ``<P>`` list.  They are read-only forced transitive imports:
the dry run records them separately in
``part_b_forced_transitive_imports`` (the same precedent as
``ext_followup.FORCED_IMPORT_EXCEPTIONS`` for ``reconstruction_model`` and
``noise_module``) and does not count them as violations.  No other module
outside ``<P>`` is imported.

R-1 input contract (read-only)
------------------------------
R-1 expects the pilot's report-mode outputs in
``results/jepa/20261106_repair/``: ``readings.json`` with ``x1`` and ``x2``
sections, ``x1.json`` and ``x2.json`` with one block per model label (either
``payload["models"][label]`` / ``payload["readings"][label]`` /
``payload[label]`` or a label-keyed block found recursively).  The fields it
reads, by candidate key (the first present wins), are listed in
:data:`R1_FIELDS`.  For the response size it reads the pilot's cached N and D
photon probe-test embeddings from ``<pilot ext>/x1/<label>/``; a cache file is
accepted only if its ``.sha256`` sidecar (or the pilot's ``LARGE_ARRAYS.json``)
matches, and a mismatch is refused.  The cache file name must contain the
``N`` or ``D`` condition token (``...__N.npz`` / ``...__D.npz``) and the
``.npz`` must hold the embeddings under one of :data:`CACHE_EMBEDDING_KEYS` and
the pairing keys under :data:`CACHE_EVENT_KEYS` / :data:`CACHE_RUN_KEYS`.

No data generation of any kind happens anywhere in this module.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import ext_followup as ef
from . import ext_h12 as eh
from . import ext_readout as er
from . import lucid_arms as la
from . import lucid_bombs as lb
from . import lucid_run as lr
from . import point_encoder as pe

__all__ = [
    "EXTENSION", "VARIANT", "ARM_NAME", "CELL_ID", "MODES", "EXT_ROOT",
    "RESULTS_DIR", "PILOT_RESULTS_DIR", "PILOT_EXT_ROOT", "RUNBOOK_REL",
    "RUNBOOK_SHA256", "ACCEPTANCE_REL", "ACCEPTANCE_KEYS", "AUTHOR_LINE",
    "STANDING_RULE", "P_FILES", "P_DIRS", "FORCED_IMPORT_EXCEPTIONS",
    "PART_B_FORCED_TRANSITIVE_IMPORTS", "TRAIN_SEEDS", "TRAIN_STEPS",
    "SMOKE_SEED", "SMOKE_STEPS", "AUTHOR_LINE", "section_zero", "in_p",
    "acceptance_record", "import_closure", "assert_authorised",
    "assert_launch_allowed", "posfree_features", "attach_posfree_target",
    "build_arm_patch", "manipulation_check", "manipulation_run",
    "open_datasets", "preflight", "prepare_run_dir", "reproduce", "run_id_for",
    "run_dir_for", "run_one_training", "evaluate_run_dirs", "label_readout",
    "label_fractions", "event_class", "recovery_rule", "r2_readings",
    "family_of", "response_size", "x2b_class", "r1_class", "r1_reading",
    "r1_payload", "s1_check", "entries_preflight", "cost_projection",
    "assert_gpu_budget", "assert_cpu_budget", "record_job", "job_hours",
    "assemble_report", "large_arrays",
]

# ---------------------------------------------------------------- extension
EXTENSION = "shortcut_followup_2026-11-07"
VARIANT = "target_posfree"
ARM_NAME = "sigreg_jepa"
CELL_ID = "dev_s005_within"
MODES = ("dry-run", "preflight", "manipulation", "reproduce", "smoke", "train",
         "evaluate", "r1", "report")
#: §5.3 / §9.1: the only authorised training launches.
TRAIN_SEEDS = (9000, 9001)
TRAIN_STEPS = 10000
SMOKE_SEED = 9000
SMOKE_STEPS = 20

ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_shortcut_20261107")
RESULTS_DIR = "results/jepa/20261107_shortcut"
#: R-1 reads the Part A repair pilot read-only (§3); never writes there.
PILOT_RESULTS_DIR = "results/jepa/20261106_repair"
PILOT_EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_repair_20261106")
READOUT_EXT = Path("/ceph/dwong/jepa/returns/ext_readout_20261101")
#: TMPDIR root (§8.3); nothing here writes outside <ext>, results and TMPDIR.
TMPDIR_ROOT = Path("/ceph/dwong/jepa/tmp/jepa_rb_20261107")
JOBS_DIR = Path("/ceph/dwong/jepa/jobs")
GUARD_JSON = JOBS_DIR / "guard_deepthought2.json"

RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md"
#: sha256 of the committed runbook (front matter §F5: the runbook is committed
#: unchanged before any outcome-generating step; the acceptance record repeats
#: this value and gate (a) compares against it).
RUNBOOK_SHA256 = "45e5ff98f8b852569828a172dccfaebce1bf20b48cdb1a78e732f181423d119b"
ACCEPTANCE_REL = "docs/jepa/ACCEPTANCE_2026-11-07_REPAIR_AND_SHORTCUT.json"
ACCEPTANCE_KEYS = ("runbook_sha256", "impl_commit", "review_path", "review_verdict",
                   "lead", "date")
DECISIONS_REL = "docs/jepa/DECISIONS.md"
#: Part B §0 line 1, verbatim; the date is recorded on the runbook (line 1170).
AUTHOR_LINE = "一二三我都批，写成一个runbook"
STANDING_RULE = "别大事小事都问我，自己找解决方案，按最好的解决方案解决"

#: §9.2 <P>: the reviewed path set.  ``git diff --quiet <impl_commit> -- <P>``
#: and ``git ls-files --others --exclude-standard -- <P>`` must both be empty.
P_FILES = (
    "src/jepa_noise/__init__.py", "src/jepa_noise/ext_shortcut.py",
    "src/jepa_noise/ext_h12.py", "src/jepa_noise/ext_readout.py",
    "src/jepa_noise/ext_followup.py", "src/jepa_noise/lucid_run.py",
    "src/jepa_noise/lucid_arms.py", "src/jepa_noise/lucid_bombs.py",
    "src/jepa_noise/lucid_views.py", "src/jepa_noise/point_encoder.py",
    "src/jepa_noise/bridge_run.py", "src/jepa_noise/train.py",
    "src/jepa_noise/crlb.py", "src/jepa_noise/jepa.py",
    "src/jepa_noise/backbone.py", "src/jepa_noise/linear.py",
    "src/jepa_noise/probes.py", "src/jepa_noise/metrics.py",
    "src/jepa_noise/sigreg.py",
    "scripts/__init__.py", "scripts/jepa_ext_shortcut.py",
    RUNBOOK_REL,
)
P_DIRS = ("src/noise_module_lucid/", "src/jepa_noise/configs/")
#: Modules outside <P> that a <P> module *forces* into the executed import
#: closure and that this implementation can never edit (ext_followup's own
#: precedent for ``reconstruction_model`` and ``noise_module``).
FORCED_IMPORT_EXCEPTIONS = tuple(ef.FORCED_IMPORT_EXCEPTIONS)
#: Part B addition: ``ext_followup`` (in <P>) imports ``ext_arch`` and
#: ``ext_dejitter`` at module level; neither is in Part B's §9.2 <P> list.
#: Read-only forced transitive imports, recorded separately and not violations.
PART_B_FORCED_TRANSITIVE_IMPORTS = ("src/jepa_noise/ext_arch.py",
                                    "src/jepa_noise/ext_dejitter.py")

# ----------------------------------------------------------- variant / §5.1
#: ``point_encoder.FEATURES``: 0-2 scaled position, 3-5 direction.
POSITION_COLUMNS = slice(0, 6)
#: 6 ln q, 7 time_100ns, 8 hit -- the event-dependent columns (unchanged).
EVENT_COLUMNS = slice(6, 9)

#: §4 floors (``ext_static.INIT_FULL_VERTEX_M`` / ``INIT_HALF_VERTEX_M``).
INIT_FULL_VERTEX_M = 0.160072
INIT_HALF_VERTEX_M = 0.174821
CHARGE_R2_MIN = 0.5
#: §4 event-dependence class (the A-1 rule, ``ext_followup._classify_event``).
EVENT_DEPENDENT_MIN = 0.01
EVENT_INVARIANT_MAX = 1e-3
#: §5.2: the descriptive label fractions are reported over 64 events.
LABEL_EVENT_TARGET = 64
MANIP_LABEL_DELTA_MAX = 1e-6
MANIP_CONTEXT_DELTA_MIN = 1e-3
#: §4 descriptive reference values (full-input vertex in m; never thresholds).
REFERENCE_VALUES = {
    "fix_full_vertex_m": [0.219259, 0.204281],
    "frozen_target_full_vertex_m": [0.138, 0.131],
    "static_target_full_vertex_m": [0.160374, 0.161148],
    "mae9000_full_vertex_m": 0.104949,
    "init_full_vertex_m": INIT_FULL_VERTEX_M,
    "init_half_vertex_m": INIT_HALF_VERTEX_M,
}

# --------------------------------------------------------------- caps / §8
#: §8.1 / front matter §F3.
GPU_CAP_H = 1.5
PART_A_GPU_CAP_H = 4.0
S1_CEILING_H = 160.0
CPU_CAP_H = 2.0
RUN_WALL_CAP_H = 1.0
#: §8.1 cost projection (h): two runs about 0.45-0.6 h each (0.5 used),
#: smoke 0.06 h, evaluation 0.14 h, reproduction evaluation 0.14 h.
COST_PER_RUN_H = 0.5
COST_SMOKE_H = 0.06
COST_EVALUATION_H = 0.14
COST_REPRODUCTION_H = 0.14
COST_BY_JOB = {"reproduce": COST_REPRODUCTION_H, "smoke": COST_SMOKE_H,
               "train": COST_PER_RUN_H, "evaluate": COST_EVALUATION_H,
               "preflight": 0.02, "manipulation": 0.20, "r1": 0.02,
               "report": 0.02, "dry-run": 0.20}
#: §F3 entries rule: >= 30 000 free after the projected writes of both parts.
ENTRIES_CLONE = 3059
ENTRIES_SUITE = 1825
ENTRIES_PART_A = 1000
ENTRIES_PART_B = 1000
ENTRIES_FLOOR = 30000

#: The default config/geometry paths (§2.3: the ``jepa_ext_static.py`` default).
CONFIG_REL = "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GENERATOR_REL = "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
CONFIG_PATH = ROOT / CONFIG_REL
GENERATOR_PATH = ROOT / GENERATOR_REL
GEOMETRY_DEFAULT = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT_DEFAULT = eh.DATA_ROOT_DEFAULT


# --------------------------------------------------------------------- io
def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, payload, *, overwrite=True) -> Path:
    """Atomic JSON write (non-finite floats become ``None`` via ``lr._safe``)."""
    path = Path(path)
    if not overwrite and path.exists():
        raise RuntimeError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(lr._safe(payload), indent=1, default=float) + "\n")
    os.replace(tmp, path)
    return path


def _resolve_root(root=None) -> Path:
    return Path(root).resolve() if root is not None else ROOT


# ===================================================================== gate
def section_zero(text: str) -> str:
    """The first runbook ``## 0.`` block (the helper of the sibling extensions)."""
    start = text.find("## 0.")
    if start < 0:
        start = text.find("## 0 ")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 3)
    return text[start: end if end >= 0 else None]


def author_section(text: str, line: str) -> str:
    """The ``## 0.`` block that contains ``line`` verbatim (Part B's §0).

    The combined runbook holds two ``## 0.`` sections (Part A's and Part B's), so
    :func:`section_zero` alone would test the wrong one; the §9.2 (a) rule is that
    the §0 author line is present verbatim, and both parts put their own line in
    their own §0 block (Part B's block is the one with this runbook's line).  The
    line is also quoted in front matter §F0, so every occurrence is considered and
    the one inside a ``## 0.`` block is used.
    """
    position = text.find(line)
    while position >= 0:
        starts = [start for start in (text.rfind("## 0.", 0, position),
                                      text.rfind("## 0 ", 0, position)) if start >= 0]
        for start in starts:
            end = text.find("\n## ", start + 3)
            end = len(text) if end < 0 else end
            if start < position < end:
                return text[start:end]
        position = text.find(line, position + 1)
    return ""


def _git(root, *args, timeout=60):
    """Run git in ``root``; returns ``(returncode, stdout, stderr)``."""
    try:
        finished = subprocess.run(["git", *args], cwd=str(root), text=True,
                                  capture_output=True, timeout=timeout)
        return (finished.returncode, finished.stdout, finished.stderr)
    except Exception as problem:  # pragma: no cover - git missing
        return (128, "", f"{type(problem).__name__}: {problem}")


def p_paths(root=None) -> list:
    """``<P>`` (files **and** directories) as absolute paths for gate (c)."""
    root = _resolve_root(root)
    return [root / name for name in (*P_FILES, *P_DIRS)]


def in_p(relative: str) -> bool:
    """True when a repository-relative path is inside the reviewed set ``<P>``."""
    relative = str(relative).replace(os.sep, "/")
    if relative in P_FILES:
        return True
    return any(relative.startswith(directory) for directory in P_DIRS)


def forced_import(relative: str) -> bool:
    """True when a path outside ``<P>`` is a recorded forced read-only import."""
    relative = str(relative).replace(os.sep, "/")
    return bool(relative.startswith(FORCED_IMPORT_EXCEPTIONS)
                or relative in PART_B_FORCED_TRANSITIVE_IMPORTS)


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
    """§9.2 executed-import closure: every repo module in ``sys.modules`` in ``<P>``.

    Forced read-only dependencies are recorded separately and are not violations:
    ``FORCED_IMPORT_EXCEPTIONS`` (``reconstruction_model`` via
    ``bridge_run``/``jepa``/``backbone`` and the underlying ``noise_module``) and
    the Part B addition ``PART_B_FORCED_TRANSITIVE_IMPORTS`` (``ext_arch`` /
    ``ext_dejitter``, imported by ``ext_followup``).
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
    allowed = [row for row in outside if forced_import(row["path"])]
    forced_exceptions = [row for row in allowed
                         if row["path"].startswith(FORCED_IMPORT_EXCEPTIONS)]
    forced_part_b = [row for row in allowed
                     if row["path"] in PART_B_FORCED_TRANSITIVE_IMPORTS]
    violations = [row for row in outside if row not in allowed]
    return {"checked_modules": int(checked),
            "outside_allowlist": sorted(outside, key=lambda row: row["path"]),
            "forced_import_exceptions": sorted(forced_exceptions,
                                               key=lambda row: row["path"]),
            "part_b_forced_transitive_imports": sorted(
                forced_part_b, key=lambda row: row["path"]),
            "violations": sorted(violations, key=lambda row: row["path"]),
            "pass": not violations}


def _gate_checks(mode, *, root) -> dict:
    """The §9.2 (a)-(c) checks; a missing record is ``gate_pending_acceptance``."""
    root = _resolve_root(root)
    checks, problems = {}, []
    runbook_path = root / RUNBOOK_REL
    computed_runbook = lr.sha256_file(runbook_path) if runbook_path.is_file() else None
    record = acceptance_record(root)
    pending = record is None

    # (a) committed runbook sha256 equals the acceptance-record value, and the §0
    # author line is present verbatim in the committed runbook.
    expected_runbook = ((record or {}).get("runbook_sha256") if record
                        else RUNBOOK_SHA256)
    text = runbook_path.read_text() if runbook_path.is_file() else ""
    block = author_section(text, AUTHOR_LINE)
    sha_pass = bool(computed_runbook is not None
                    and computed_runbook == expected_runbook)
    line_pass = bool(block and AUTHOR_LINE in block)
    a_pass = bool(sha_pass and line_pass)
    checks["a"] = {"pass": a_pass, "runbook": str(runbook_path),
                   "runbook_sha256": computed_runbook,
                   "expected_sha256": expected_runbook,
                   "author_line": AUTHOR_LINE, "author_line_in_section_zero": line_pass,
                   "pending_acceptance": bool(pending)}
    if not a_pass and not pending:
        problems.append("(a) the committed runbook sha256 and/or the §0 author line "
                        "does not match the acceptance record")

    # (b) acceptance record schema and review verdict.
    if record is None:
        checks["b"] = {"pass": False, "status": "gate_pending_acceptance",
                       "reason": f"{ACCEPTANCE_REL} does not exist yet"}
    elif "invalid" in record:
        checks["b"] = {"pass": False, "status": "invalid",
                       "reason": record["invalid"]}
    else:
        missing = [key for key in ACCEPTANCE_KEYS if key not in record]
        b_pass = bool(not missing and record.get("review_verdict") == "PASS")
        checks["b"] = {"pass": b_pass, "status": "PASS" if b_pass else "denied",
                       "record": record, "missing_keys": missing,
                       "review_verdict": record.get("review_verdict")}
        if not b_pass:
            problems.append("(b) the acceptance record is incomplete or its "
                            "review_verdict is not PASS")

    # (c) working tree against impl_commit over <P>, and no untracked <P> file.
    # Pre-acceptance HEAD stands in for impl_commit so the dry run works.
    impl_commit = (record or {}).get("impl_commit") if record else None
    source = "acceptance_record"
    if not impl_commit:
        code, head, _ = _git(root, "rev-parse", "HEAD")
        impl_commit = head.strip() if code == 0 else None
        source = "HEAD (pre-acceptance)"
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
    c_pass = bool(diff_clean and not untracked)
    checks["c"] = {"pass": c_pass, "impl_commit": impl_commit,
                   "impl_commit_source": source, "diff_clean": bool(diff_clean),
                   "diff_error": diff_error, "untracked": untracked,
                   "paths": list(P_FILES) + list(P_DIRS)}
    if not c_pass and not pending:
        problems.append(f"(c) the <P> tree differs from {impl_commit} "
                        f"(untracked: {untracked[:3]}"
                        f"{'...' if len(untracked) > 3 else ''})")

    # The dry run asserts the executed-import closure in the same report.
    if mode == "dry-run":
        closure = import_closure(root)
        checks["closure"] = closure
        if not closure["pass"]:
            problems.append(f"(c) executed-import closure has "
                            f"{len(closure['violations'])} module(s) outside <P>")

    authorised = bool(not problems and not pending)
    status = ("gate_pending_acceptance" if pending else
              ("granted" if authorised else "denied"))
    return {"authorised": bool(authorised), "status": status, "mode": mode,
            "extension": EXTENSION, "checks": checks, "problems": problems,
            "acceptance": record, "runbook_sha256": computed_runbook}


def assert_authorised(mode, *, root=None, dry_run=None, log=None) -> dict:
    """Hard gate (§9.2 (a)-(c)).  Any failure raises ``SystemExit(2)``.

    ``dry_run=True`` (or ``mode == "dry-run"``) records the checks and returns
    them: a missing acceptance record is then the normal pre-acceptance state
    ``gate_pending_acceptance`` rather than a failure, and the executed-import
    closure is asserted as part of the same report.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    dry = bool(dry_run) if dry_run is not None else (mode == "dry-run")
    result = _gate_checks(mode, root=root)
    if dry:
        return result
    if not result["authorised"]:
        for problem in result["problems"]:
            print(f"[shortcut] authorisation gate: {problem}", file=sys.stderr)
        if result["status"] == "gate_pending_acceptance":
            print(f"[shortcut] authorisation gate: {ACCEPTANCE_REL} does not exist "
                  f"yet (the lead writes it after review)", file=sys.stderr)
        raise SystemExit(2)
    return result


def assert_launch_allowed(mode, *, seed=None, steps=None, out_root=None) -> dict:
    """The §5.3/§9.1 training guard: only the authorised launches; no resume.

    ``train`` accepts only seeds 9000/9001 and 10 000 steps; ``smoke`` only 20
    steps (and seed 9000); an existing run root or run directory is refused so a
    run can never be resumed.  Any failure raises ``SystemExit(2)``.
    """
    detail = {"mode": mode, "seed": None if seed is None else int(seed),
              "steps": None if steps is None else int(steps),
              "out_root": None if out_root is None else str(out_root)}
    problems = []
    if mode == "train":
        if int(seed if seed is not None else -1) not in TRAIN_SEEDS:
            problems.append(f"train seed {seed!r} is not one of {TRAIN_SEEDS}")
        if int(steps if steps is not None else -1) != TRAIN_STEPS:
            problems.append(f"train steps {steps!r} is not {TRAIN_STEPS}")
    elif mode == "smoke":
        if int(seed if seed is not None else -1) != SMOKE_SEED:
            problems.append(f"smoke seed {seed!r} is not {SMOKE_SEED}")
        if int(steps if steps is not None else -1) != SMOKE_STEPS:
            problems.append(f"smoke steps {steps!r} is not {SMOKE_STEPS}")
    elif mode in ("reproduce", "evaluate", "preflight", "manipulation", "r1", "report"):
        pass
    else:
        problems.append(f"unknown mode {mode!r}")
    run_id = run_dir = None
    if out_root is not None and mode in ("train", "smoke") and not problems:
        run_id = run_id_for(int(seed))
        run_dir = run_dir_for(out_root, int(seed))
        if run_dir.exists():
            problems.append(f"{run_dir} already exists; refusing to launch (run_one "
                            f"would auto-resume it).  Smoke and runs are archived and "
                            f"never resumed (§5.3).")
    detail.update(run_id=run_id, run_dir=None if run_dir is None else str(run_dir),
                  pass_=not problems, problems=problems)
    if problems:
        for problem in problems:
            print(f"[shortcut] training guard: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return detail


# ================================================================ variant §5.1
def posfree_features(features: torch.Tensor) -> torch.Tensor:
    """A clone of ``features`` with the position/direction columns 0-5 zeroed.

    Never mutates its input.  This is the *only* difference of the
    ``target_posfree`` variant: the label pass is fed this clone; the context and
    full passes keep ``batch.features``.
    """
    if features.size(-1) < EVENT_COLUMNS.stop:
        raise ValueError("target_posfree needs at least nine feature columns")
    clone = features.clone()
    clone[..., POSITION_COLUMNS] = 0.0
    return clone


def _encode_views(arm, batch, *, need_full=True, zero_positions=True):
    """The ``lucid_arms.LucidArm._encode_views`` body with the label-pass change.

    Identical to ``lucid_arms.py:405-420`` except that the label pass receives
    ``posfree_features(batch.features)``.  The label is **not** detached: the
    encoder is trained through it, exactly as in the fixed-label arm.  With
    ``zero_positions=False`` this is the original body bit for bit (used by the
    §9.3 variant test).
    """
    hidden_index = arm._hidden_index(batch)
    context = arm.jepa.encode(batch.features, hidden_mask=batch.hidden)
    prediction = arm.jepa.predict(context, batch.context_index, hidden_index)
    if not need_full:
        return context, prediction, None, hidden_index, None
    full = arm.jepa.encode(batch.features)
    label_features = posfree_features(batch.features) if zero_positions else batch.features
    label_full = arm.jepa.encode(label_features, hidden_mask=~batch.target)
    gather = hidden_index.unsqueeze(-1).expand(-1, -1, arm.d_model)
    return context, prediction, label_full.gather(1, gather), hidden_index, full


def attach_posfree_target(arm, *, zero_positions=True):
    """Rebind ``arm._encode_views`` to the position-free label pass (§5.1)."""
    def bound(batch, *, need_full=True):
        return _encode_views(arm, batch, need_full=need_full,
                             zero_positions=zero_positions)

    object.__setattr__(arm, "_encode_views", bound)
    return arm


@contextlib.contextmanager
def build_arm_patch(log=print, *, only_arm=ARM_NAME):
    """Process-level wrapper on ``lucid_run.build_arm`` installing the variant.

    Applies only to ``sigreg_jepa`` (``only_arm``); every other arm is built
    untouched.  ``lucid_run.build_arm`` is restored in a ``finally``, also after
    an exception.  Yields counters ``{"installed": n, "skipped": n}``.
    """
    original = lr.build_arm
    state = {"installed": 0, "skipped": 0}

    def build(arm_name, geometry, arms_config, cell, seed, beta=None, dark_bank=None,
              encoder_config=None):
        made = original(arm_name, geometry, arms_config, cell, seed, beta=beta,
                        dark_bank=dark_bank, encoder_config=encoder_config)
        if str(arm_name) == str(only_arm):
            attach_posfree_target(made)
            state["installed"] += 1
            log(f"[shortcut] build_arm patch: variant={VARIANT} "
                f"run={lr.run_id_for(cell.cell_id, arm_name, seed)}")
        else:
            state["skipped"] += 1
        return made

    lr.build_arm = build
    try:
        yield state
    finally:
        lr.build_arm = original


# ======================================================== manipulation §5.2
def _replace_batch_features(batch, features):
    return dataclasses.replace(batch, features=features)


def _input_projection(arm):
    """The encoder's input projection (``PointEncoder.input_proj``)."""
    for candidate in (lambda: arm.jepa.encoder.input_proj,
                      lambda: arm.jepa.input_proj):
        try:
            return candidate()
        except AttributeError:
            continue
    raise AttributeError("no input projection found on the arm's encoder")


def _encode_calls(arm, batch):
    """Run the rebound ``_encode_views`` and record the three encode inputs."""
    recorded = []
    original = arm.jepa.encode

    def recording(inputs, hidden_mask=None):
        recorded.append({"features": inputs.detach().clone(),
                         "mask": None if hidden_mask is None
                         else hidden_mask.detach().clone()})
        return original(inputs, hidden_mask=hidden_mask)

    arm.jepa.encode = recording
    try:
        out = arm._encode_views(batch)
    finally:
        arm.jepa.encode = original
    return out, recorded


def label_readout(arm, batches, *, device=None) -> np.ndarray:
    """The label-pass full read-out ``(E, N_pmt, d)`` over ``batches`` (no grad).

    ``arm.jepa.encode(posfree_features(features), hidden_mask=~batch.target)`` --
    the label pass of §5.1 -- keeps the target positions visible and replaces the
    context positions by the learned mask token, the same convention as the
    full-input read-out.  Used for the descriptive label ``f_layout``/``f_event``
    of §5.2 and §5.3.
    """
    tokens = []
    with torch.no_grad():
        for batch in batches:
            moved = batch if device is None else lr._to_device(batch, device)
            features = posfree_features(moved.features)
            label_full = arm.jepa.encode(features, hidden_mask=~moved.target)
            tokens.append(label_full.detach().to(torch.float32).cpu().numpy())
    if not tokens:
        raise ValueError("label_readout needs at least one batch")
    return np.concatenate(tokens, axis=0)


def label_fractions(arm, batches, *, device=None) -> dict:
    """Descriptive label ``f_layout``/``f_event`` over the given batches (§5.2)."""
    tokens = label_readout(arm, batches, device=device)
    fractions = er.layout_and_event_fractions(tokens.mean(axis=1), tokens)
    fractions["n_events"] = int(tokens.shape[0])
    fractions["n_tokens"] = int(tokens.shape[1])
    fractions["form"] = "label pass (posfree features, hidden_mask=~target)"
    return fractions


def manipulation_check(arm, batch, *, extra_batches=(), device=None,
                       label_event_target: int = LABEL_EVENT_TARGET) -> dict:
    """The §5.2 manipulation check on one tokenised batch (seed 9000 init).

    Checks, in order: the label-pass input has columns 0-5 exactly zero and
    columns 6-8 equal to the original; the original ``batch.features`` is not
    mutated; the context and full passes receive the original; identity
    blindness (a PMT permutation of columns 0-5 leaves the label tensor unchanged
    to ``<= 1e-6`` but changes the context read-out by ``> 1e-3``); gradients reach
    the encoder through the label pass (non-zero gradient on
    ``input_proj.weight`` from the label MSE alone).  Descriptively reports the
    label ``f_layout``/``f_event`` over ``label_event_target`` events at init.
    """
    arm.eval()
    features = batch.features
    original = features.detach().clone()
    out, recorded = _encode_calls(arm, batch)
    context, prediction, labels, hidden_index, full = out
    if len(recorded) != 3:
        raise RuntimeError(f"§5.2: expected three encode passes, got {len(recorded)}")
    context_features, full_features, label_features = (
        recorded[0]["features"], recorded[1]["features"], recorded[2]["features"])
    zero = torch.zeros_like(label_features[..., POSITION_COLUMNS])
    checks = {
        "label_position_columns_zero": bool(torch.equal(
            label_features[..., POSITION_COLUMNS].detach(), zero)),
        "label_event_columns_equal": bool(torch.equal(
            label_features[..., EVENT_COLUMNS].detach(),
            features[..., EVENT_COLUMNS].detach())),
        "original_features_unchanged": bool(torch.equal(features.detach(), original)),
        "context_receives_original": bool(torch.equal(context_features, original)),
        "full_pass_receives_original": bool(torch.equal(full_features, original)),
    }
    masks = {
        "context_mask_is_hidden": bool(torch.equal(
            recorded[0]["mask"], batch.hidden.detach())),
        "full_mask_is_none": bool(recorded[1]["mask"] is None),
        "label_mask_is_not_target": bool(torch.equal(
            recorded[2]["mask"], (~batch.target).detach())),
    }

    # identity blindness: permute columns 0-5 across PMTs
    n_pmt = int(features.size(1))
    rng = np.random.default_rng(20261107)
    permutation = torch.as_tensor(rng.permutation(n_pmt), device=features.device)
    permuted = features.detach().clone()
    permuted[..., POSITION_COLUMNS] = permuted[..., POSITION_COLUMNS].index_select(
        1, permutation)
    permuted_batch = _replace_batch_features(batch, permuted)
    permuted_out, _ = _encode_calls(arm, permuted_batch)
    label_delta = float((labels.detach() - permuted_out[2].detach()).abs().max())
    context_delta = float((context.detach() - permuted_out[0].detach()).abs().max())
    blindness = {
        "permutation": permutation.detach().cpu().tolist()[:8],
        "label_max_abs_delta": label_delta,
        "context_max_abs_delta": context_delta,
        "label_unchanged": bool(label_delta <= MANIP_LABEL_DELTA_MAX),
        "context_changed": bool(context_delta > MANIP_CONTEXT_DELTA_MIN),
        "label_delta_max": MANIP_LABEL_DELTA_MAX,
        "context_delta_min": MANIP_CONTEXT_DELTA_MIN,
    }

    # gradients reach the encoder through the label pass: the label MSE alone
    arm.zero_grad(set_to_none=True)
    with torch.enable_grad():
        label_grad_out = _encode_views(arm, batch)
        label_mse = F.mse_loss(label_grad_out[2], torch.zeros_like(label_grad_out[2]))
        label_mse.backward()
    grad = _input_projection(arm).weight.grad
    grad_absmax = 0.0 if grad is None else float(grad.detach().abs().max())
    label_grad = {"input_proj_grad_absmax": grad_absmax,
                  "label_requires_grad": bool(label_grad_out[2].requires_grad),
                  "label_mse": float(label_mse.detach()),
                  "non_zero": bool(grad_absmax > 0.0)}
    arm.zero_grad(set_to_none=True)

    batches = [batch, *extra_batches]
    fractions = label_fractions(arm, batches, device=device)
    fractions["n_events_target"] = int(label_event_target)
    fractions["n_events_short"] = bool(
        int(fractions["n_events"]) < int(label_event_target))

    passed = bool(all(checks.values()) and all(masks.values())
                  and blindness["label_unchanged"]
                  and blindness["context_changed"] and label_grad["non_zero"])
    reasons = []
    if not all(checks.values()):
        reasons.append("label/context/feature check failed")
    if not all(masks.values()):
        reasons.append("encode mask plumbing does not match the original body")
    if not blindness["label_unchanged"]:
        reasons.append(f"label changed under the position permutation "
                       f"({blindness['label_max_abs_delta']:.3e} > "
                       f"{MANIP_LABEL_DELTA_MAX:g})")
    if not blindness["context_changed"]:
        reasons.append(f"context read-out did not change under the position "
                       f"permutation ({blindness['context_max_abs_delta']:.3e} <= "
                       f"{MANIP_CONTEXT_DELTA_MIN:g})")
    if not label_grad["non_zero"]:
        reasons.append("no gradient on input_proj.weight from the label MSE")
    return {"registered": False, "extension": EXTENSION, "variant": VARIANT,
            "check": "§5.2 manipulation check (implementation guard)",
            "checks": checks, "masks": masks, "identity_blindness": blindness,
            "label_gradient": label_grad, "label_fractions": fractions,
            "pass": passed, "reasons": reasons,
            "note": "columns 0-5 are zeroed in the label pass only; the label is "
                    "not detached and the encoder stays shared and trainable"}


def label_batches(train_ds, geometry, config, *, cell=None, device=None,
                  seed=SMOKE_SEED, max_steps=8) -> list:
    """The training-cell batches for the label fractions (§5.2: 64 events).

    Deterministic: ``TrainingLoader(seed=9000).batch_at(0..)`` always selects the
    same events, so §5.2 and §5.3 report the fractions over the same events.
    """
    cell = cell if cell is not None else lr.dev_cell(config)
    loader = lr.TrainingLoader(train_ds, geometry, split=cell.split, seed=int(seed),
                               occupancy=cell.occupancy,
                               n_min=int(config["views"]["n_min"]),
                               batch=int(config["training"]["batch"]))
    batches, total, step = [], 0, 0
    while total < LABEL_EVENT_TARGET and step < int(max_steps):
        batch, _ = loader.batch_at(step)
        if device is not None:
            batch = lr._to_device(batch, device)
        batches.append(batch)
        total += int(batch.features.size(0))
        step += 1
    return batches


def manipulation_run(config, gen, arms_config, geometry, data_root, *, root=None,
                     dry_run=None, device="cpu", log=print) -> dict:
    """§5.2 on the training cell with the untrained seed-9000 arm.

    The §9.2 gate is called here (the library entry point, not only the CLI).
    """
    root = _resolve_root(root)
    assert_authorised("manipulation", root=root, dry_run=dry_run)
    device = device if isinstance(device, torch.device) else torch.device(device)
    cell = lr.dev_cell(config)
    preflight_record = eh.preflight_data(Path(data_root))
    if not preflight_record.get("pass"):
        raise SystemExit(f"[shortcut] §2.3 data preflight failed; no data generation "
                         f"is allowed: {preflight_record}")
    train_ds, _probes = open_datasets(cell, config, gen, geometry, data_root)
    torch.manual_seed(SMOKE_SEED)
    np.random.seed(SMOKE_SEED % 2 ** 32)
    arm = lr.build_arm(ARM_NAME, geometry, arms_config, cell, SMOKE_SEED).to(device)
    attach_posfree_target(arm)
    arm.eval()
    batches = label_batches(train_ds, geometry, config, cell=cell, device=device)
    check = manipulation_check(arm, batches[0], extra_batches=batches[1:],
                               device=device)
    check.update({"cell_id": cell.cell_id, "arm": ARM_NAME, "seed": SMOKE_SEED,
                  "batch": {"step": 0, "n_events": int(batches[0].features.size(0)),
                            "batches": len(batches),
                            "label": "TrainingLoader.batch_at(0..), seed 9000"}})
    log(f"[shortcut] manipulation check: pass={check['pass']} "
        f"label f_layout={check['label_fractions'].get('f_layout')} "
        f"f_event={check['label_fractions'].get('f_event')} "
        f"context delta={check['identity_blindness']['context_max_abs_delta']:.3e}")
    return check


# ==================================================================== data §2.3
def open_datasets(cell, config, gen, geometry, data_root):
    """The pinned ``dev_train`` + probe sets via ``ext_h12.datasets_for`` only.

    This is the *only* dataset path of the extension: the pinned manifests are
    hash-checked first (:func:`preflight`), then ``ext_h12.datasets_for`` /
    ``ext_h12.load_probes`` open the read-only sets.  It never calls
    ``ensure_dark_bank`` and never generates data.
    """
    return eh.datasets_for(cell, config, gen, geometry, Path(data_root), code_sha=None)


def preflight(root=None, *, data_root=DATA_ROOT_DEFAULT, ext_root=EXT_ROOT,
              readout_items=None, dry_run=None, log=print) -> dict:
    """§2.3/§8 preflight: data hashes, S1, caps, entries, readout items."""
    root = _resolve_root(root)
    assert_authorised("preflight", root=root, dry_run=dry_run)
    payload = {"registered": False, "extension": EXTENSION, "variant": VARIANT,
               "mode": "preflight", "root": str(root),
               "runbook": RUNBOOK_REL, "runbook_sha256": lr.sha256_file(
                   root / RUNBOOK_REL) if (root / RUNBOOK_REL).is_file() else None,
               "code_sha": lr.git_head(root), "git_status": lr.git_status(root),
               "data_root": str(data_root), "ext_root": str(ext_root)}
    payload["data"] = eh.preflight_data(Path(data_root))
    payload["s1"] = s1_check(root)
    payload["caps"] = cost_projection()
    payload["entries"] = entries_preflight()
    items = Path(readout_items) if readout_items else (READOUT_EXT / "items.json")
    payload["readout_items"] = {"path": str(items), "exists": items.is_file()}
    payload["pass"] = bool(payload["data"].get("pass") and payload["s1"]["pass"]
                           and payload["caps"]["pass"] and payload["entries"]["pass"])
    if not payload["pass"]:
        log(f"[shortcut] preflight failed: data={payload['data'].get('pass')} "
            f"s1={payload['s1']['pass']} caps={payload['caps']['pass']} "
            f"entries={payload['entries']['pass']}")
    return payload


# -------------------------------------------------------------- run dirs
def run_id_for(seed) -> str:
    return lr.run_id_for(CELL_ID, ARM_NAME, int(seed))


def run_dir_for(out_root, seed) -> Path:
    return Path(out_root) / "runs" / run_id_for(seed)


def prepare_run_dir(out_root, seed, *, steps=None, device=None) -> tuple:
    """Refuse an existing run directory and write the per-run variant sidecar."""
    run_id, run_dir = run_id_for(seed), run_dir_for(out_root, seed)
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; refusing to launch (run_one "
                           f"would auto-resume it).  Smoke and runs are archived and "
                           f"never resumed (§5.3).")
    run_dir.mkdir(parents=True, exist_ok=False)
    sidecar = {"registered": False, "extension": EXTENSION, "variant": VARIANT,
               "seed": int(seed), "arm": ARM_NAME, "cell_id": CELL_ID,
               "steps": steps, "device": str(device), "out_root": str(out_root),
               "run_id": run_id, "runbook": RUNBOOK_REL}
    (run_dir / "variant.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return run_id, run_dir


def assert_fresh_resume(status: dict, run_id: str = "") -> int:
    """The run must not have resumed (``resumed_from_step`` 0 / null)."""
    if "resumed_from_step" not in status:
        raise RuntimeError(f"{run_id}: run status has no resumed_from_step")
    resumed = status["resumed_from_step"]
    resumed = 0 if resumed is None else int(resumed)
    if int(resumed) != 0:
        raise RuntimeError(f"{run_id}: resumed_from_step={resumed} != 0; a resumed run is "
                           f"not authorised (§5.3: fresh start, never resumed)")
    return int(resumed)


def summarize_run(run_dir, *, seed, wall_seconds=None, steps=None,
                  resumed_from_step=0, forms=None) -> dict:
    """Finite ``ext_summary.json`` payload for the smoke or an approved run."""
    run_dir = Path(run_dir)
    check = eh.train_log_check(run_dir)
    checkpoint = run_dir / "final.pt"
    status = (json.loads((run_dir / "run_status.json").read_text())
              if (run_dir / "run_status.json").is_file() else {})
    return {"registered": False, "extension": EXTENSION, "variant": VARIANT,
            "seed": int(seed), "run_id": run_dir.name, "arm": ARM_NAME,
            "cell_id": CELL_ID, "steps_scheduled": steps,
            "steps_completed": status.get("steps_completed"),
            "resumed_from_step": int(resumed_from_step),
            "status": status.get("outcome"),
            "final_train_metrics": check.get("final_metrics"),
            "non_finite_keys": check.get("non_finite_keys"),
            "metrics_finite": check.get("finite"),
            "final_checkpoint_sha256": (lr.sha256_file(checkpoint)
                                        if checkpoint.is_file() else None),
            "wall_seconds": wall_seconds, "forms": forms}


def run_one_training(*, mode, seed, steps, out_root, config, gen, arms_config, geometry,
                     device, data_root, config_path, code_sha, status_text,
                     probe_max_events=None, root=None, ext_root=EXT_ROOT,
                     log=print) -> dict:
    """One authorised run (or the 20-step smoke) of the variant (§5.3).

    The §9.2 gate is called here (the library entry point), the launch guard pins
    the seed/step pair and refuses an existing run directory, and the variant is
    installed through the ``lucid_run.build_arm`` wrapper for the whole
    ``run_one`` call (restored afterwards).
    """
    root = _resolve_root(root)
    assert_authorised(mode, root=root)
    assert_launch_allowed(mode, seed=seed, steps=steps, out_root=out_root)
    assert_gpu_budget(Path(ext_root), COST_BY_JOB.get(mode), root=root)
    preflight_record = eh.preflight_data(Path(data_root))
    if not preflight_record.get("pass"):
        raise SystemExit(f"[shortcut] §2.3 data preflight failed (STOP for R-2): "
                         f"{preflight_record}")
    cell = lr.dev_cell(config)
    train_ds, probes = open_datasets(cell, config, gen, geometry, data_root)
    scheduled = int(steps)
    run_id, run_dir = prepare_run_dir(out_root, seed, steps=scheduled, device=device)
    started = time.time()
    with build_arm_patch(log=log):
        result = lr.run_one(
            arm_name=ARM_NAME, cell=cell, seed=int(seed), beta=None, config=config,
            arms_config=arms_config, geometry=geometry, loader_dataset=train_ds,
            probe_datasets=probes, dark_bank=None, out_root=Path(out_root),
            data_root=Path(data_root), config_sha=lr.sha256_file(config_path),
            code_sha=code_sha, status_text=status_text, device=device,
            steps=scheduled, micro_batch=None, resume=None, stop_after=None,
            checkpoint_minutes=None, evaluate=False, allow_pending=False,
            step_callback=None, probe_max_events=probe_max_events,
            save_embeddings=False, physics_method="ridge", tf32=False, log=log)
    status = json.loads((run_dir / "run_status.json").read_text())
    resumed = assert_fresh_resume(status, run_id)
    summary = summarize_run(run_dir, seed=seed,
                            wall_seconds=round(time.time() - started, 3),
                            steps=scheduled, resumed_from_step=resumed)
    summary["smoke"] = bool(mode == "smoke")
    summary["wall_cap_h"] = RUN_WALL_CAP_H
    summary["wall_cap_exceeded"] = bool(
        summary["wall_seconds"] is not None
        and float(summary["wall_seconds"]) > RUN_WALL_CAP_H * 3600.0)
    (run_dir / "ext_summary.json").write_text(
        json.dumps(summary, indent=1, default=float) + "\n")
    return {"result": result, "run_dir": run_dir, "summary": summary,
            "run_id": run_id, "seed": int(seed)}


# ============================================================= reproduction §5.3
def reproduce(config, arms_config, geometry, probes, readout_items, *, device,
              root=None, dry_run=None, tolerance=None) -> dict:
    """The §5.3 reproduction gate (``ext_h12.REPRODUCTION``), gated here.

    The §9.2 gate is called from this library entry point, not only from the CLI;
    the body is ``ext_h12.reproduction`` unchanged (``fix9000`` 0.219258 and
    ``init9000`` 0.160072 full-input vertex within 1e-3 m).
    """
    assert_authorised("reproduce", root=_resolve_root(root), dry_run=dry_run)
    kwargs = {} if tolerance is None else {"tol": tolerance}
    payload = eh.reproduction(config, arms_config, geometry, probes, readout_items,
                              device=device, **kwargs)
    payload["mode"] = "reproduce"
    return payload


# ============================================================= evaluation §5.3
def evaluate_run_dirs(run_dirs, probes, geometry, config, arms_config, *, device,
                      batches=None, max_events=None, root=None, dry_run=None) -> dict:
    """The readout protocol (full input + half pass) for every ``final.pt``.

    For each final checkpoint it also reports the label-pass ``f_event`` over the
    same events as §5.2 (``batches``, the 64-event training-cell selection).
    """
    assert_authorised("evaluate", root=_resolve_root(root), dry_run=dry_run)
    runs = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        sidecar_path = run_dir / "variant.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        seed = int(sidecar.get("seed", 0))
        source = run_dir / "final.pt"
        if not source.is_file():
            raise RuntimeError(f"{run_dir}: final.pt is missing")
        item = {"label": f"target_posfree_{seed}", "arm": ARM_NAME, "split": "within",
                "seed": seed, "beta": None, "source": str(source)}
        metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                  device=device, max_events=max_events)
        entry = {"variant": VARIANT, "seed": seed, "run_id": run_dir.name,
                 "run_dir": str(run_dir), "checkpoint_sha256": lr.sha256_file(source),
                 "full_input": metrics["forms"]["full_input"],
                 "half_pass": metrics["forms"]["half_pass"],
                 "ext_summary": (json.loads((run_dir / "ext_summary.json").read_text())
                                 if (run_dir / "ext_summary.json").is_file() else None)}
        if batches:
            arm = er.item_arm(item, config, arms_config, geometry, device=device)
            attach_posfree_target(arm)
            arm.eval()
            entry["label_fractions"] = label_fractions(arm, batches, device=device)
        runs[str(seed)] = entry
    return runs


# ================================================================= readings §4
def event_class(f_event_by_seed: dict) -> dict:
    """§4 event-dependence class on full-input ``f_event`` (the A-1 rule)."""
    return ef._classify_event(f_event_by_seed)


def recovery_rule(runs: dict, *, full_floor: float = INIT_FULL_VERTEX_M,
                  half_floor: float = INIT_HALF_VERTEX_M,
                  charge_min: float = CHARGE_R2_MIN) -> dict:
    """The §4 recovery rule (the ``static_target_r`` / E3 rule, unchanged).

    * recovered: in both seeds full-input vertex < 0.160072 m, full-input token
      ln q R2 >= 0.5, half-pass vertex < 0.174821 m and half-pass token R2 >= 0.5;
    * not recovered: in both seeds full-input vertex >= 0.160072 m or full-input
      token R2 < 0.5;
    * otherwise unclear.
    """
    rows = {}
    for seed, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        full_v = full.get("vertex_resolution_m")
        full_r2 = (full.get("token_log_charge_r2") or {}).get("r2")
        half_v = half.get("vertex_resolution_m")
        half_r2 = (half.get("token_log_charge_r2") or {}).get("r2")
        recovers = bool(full_v is not None and float(full_v) < float(full_floor)
                        and full_r2 is not None and float(full_r2) >= float(charge_min)
                        and half_v is not None and float(half_v) < float(half_floor)
                        and half_r2 is not None and float(half_r2) >= float(charge_min))
        fails = bool((full_v is not None and float(full_v) >= float(full_floor))
                     or full_r2 is None or float(full_r2) < float(charge_min))
        rows[str(seed)] = {"full_input_vertex_m": full_v,
                           "full_input_token_r2": full_r2,
                           "half_pass_vertex_m": half_v,
                           "half_pass_token_r2": half_r2,
                           "recovers": recovers, "fails": fails}
    values = list(rows.values())
    if values and all(row["recovers"] for row in values):
        verdict = "recovered"
    elif values and all(row["fails"] for row in values):
        verdict = "not recovered"
    else:
        verdict = "unclear"
    return {"registered": False, "extension": EXTENSION, "id": "R2-b",
            "question": "does the position-free trainable target recover the "
                        "event-invariant collapse?",
            "criteria": "recovered: both seeds full vertex < %.6f, full token R2 >= "
                        "%.1f, half vertex < %.6f, half token R2 >= %.1f; not "
                        "recovered: both seeds full vertex >= %.6f or full token "
                        "R2 < %.1f"
                        % (full_floor, charge_min, half_floor, charge_min, full_floor,
                           charge_min),
            "verdict": verdict, "seeds": rows,
            "floors": {"full_input_m": float(full_floor),
                       "half_pass_m": float(half_floor),
                       "charge_r2_min": float(charge_min)}}


def _layout_field(form: dict, name: str):
    layout = form.get("layout") or {}
    return layout.get(name)


def _module_gain(form: dict):
    gain = form.get("module_gain_r2")
    return gain.get("value") if isinstance(gain, dict) else gain


def r2_readings(runs: dict, *, references=None) -> dict:
    """R2-a, R2-b and R2-c from the evaluation runs (§5.4)."""
    f_event = {}
    for seed, run in runs.items():
        value = _layout_field(run["full_input"], "f_event")
        if value is None and run.get("label_fractions"):
            value = run["label_fractions"].get("f_event")
        f_event[int(seed)] = value
    r2_a = event_class(f_event)
    r2_a["id"] = "R2-a"
    r2_a["question"] = "the §4 class of target_posfree on full-input f_event"
    r2_b = recovery_rule(runs)
    rows = {}
    for seed, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        label = run.get("label_fractions") or {}
        rows[str(seed)] = {
            "full_input_vertex_m": full.get("vertex_resolution_m"),
            "full_input_token_r2": (full.get("token_log_charge_r2") or {}).get("r2"),
            "half_pass_vertex_m": half.get("vertex_resolution_m"),
            "half_pass_token_r2": (half.get("token_log_charge_r2") or {}).get("r2"),
            "effective_rank": full.get("effective_rank"),
            "module_gain_r2": _module_gain(full),
            "f_layout": _layout_field(full, "f_layout"),
            "label_pass_f_event": label.get("f_event"),
            "label_pass_f_layout": label.get("f_layout"),
            "label_pass_n_events": label.get("n_events"),
        }
    r2_c = {"id": "R2-c", "descriptive": True, "seeds": rows,
            "references": dict(REFERENCE_VALUES if references is None else references),
            "note": "descriptive; the reference values are never thresholds"}
    return {"r2_a": r2_a, "r2_b": r2_b, "r2_c": r2_c}


# ==================================================================== R-1 §3
#: The fixed §3 family table (the collapsed joint-embedding models are reported
#: separately and are never pooled with the working ones).
R1_FAMILIES = {
    "generative": ("mae_within_trigger9000", "mae_within_trigger9001",
                   "mae_across_trigger9000", "mae_across_trigger9001",
                   "t1_s030_within_trigger9000", "t1_s030_within_trigger9001",
                   "t1_s030_across_trigger9000", "t1_s030_across_trigger9001"),
    "joint_embedding_working": ("frozen_target9000", "frozen_target9001"),
    "joint_embedding_collapsed": ("fix9000", "fix9001"),
    "untrained_reference": ("init9000",),
}
#: Optional rows, reported when the pilot ran them; never pooled into R1.
R1_OPTIONAL = {
    "generative": ("mae_within_median9000", "mae_within_median9001"),
    "joint_embedding_working": ("ijepa9000", "ijepa9001"),
    "joint_embedding_collapsed": ("static_target9000", "static_target9001"),
}
R1_LABELS = {label: family for family, labels in R1_FAMILIES.items()
             for label in labels}
R1_ALL_LABELS = {label: family for family, labels in
                 list(R1_FAMILIES.items()) + list(R1_OPTIONAL.items())
                 for label in labels}
#: Candidate field keys (first present wins) in the pilot's per-model blocks.
R1_FIELDS = {
    "response_size": ("response_size", "S", "s", "x2_a.response_size"),
    "r2_eq": ("r2_eq", "R2_eq", "r2_eq_occupancy", "occupancy_r2_eq",
              "x2_b.r2_eq", "X2-b.r2_eq"),
    "x2_class": ("x2_class", "class", "x2_b.class", "X2-b.class"),
    "rho_lr": ("rho_lr", "rho_LR", "x2_c.rho_lr", "X2-c.rho_lr"),
    "rho_swtw": ("rho_swtw", "rho_SWtw", "x1_b.rho_swtw", "X1-b.rho_swtw"),
    "rho_swunit": ("rho_swunit", "rho_SWunit", "x1_c.rho_swunit"),
    "share_cs": ("share_cs", "x1_g.share_cs", "X1-g.share_cs"),
    "delta_d": ("delta_d_m", "delta_D_m", "delta_d", "x1_a.delta_d_m",
                "X1-a.delta_d_m"),
    "delta_d_eligible": ("delta_d_eligible", "eligible", "x1_a.eligible",
                         "X1-a.eligible"),
    "f_layout": ("f_layout",),
    "f_event": ("f_event",),
}
#: Candidate npz keys for a cached embedding set and its pairing keys.
CACHE_EMBEDDING_KEYS = ("z", "embeddings", "embedding", "tokens", "event_means",
                        "z_mean", "features")
CACHE_EVENT_KEYS = ("event_ids", "event_id", "events")
CACHE_RUN_KEYS = ("run_ids", "run_id", "runs")
LARGE_ARRAY_MIN_BYTES = 5 * 1024 * 1024


def _seed_of_label(label):
    """The seed suffix of a fixed model label (``frozen_target9001`` -> ``9001``)."""
    text = str(label)
    for seed in ("9000", "9001"):
        if text.endswith(seed):
            return seed
    return None


def _model_block(block, label):
    """Descend a per-model block to its seed-specific dict when it has one."""
    seed = _seed_of_label(label)
    if seed and isinstance(block, dict):
        for key in (seed, "seeds", "by_seed", "per_seed"):
            nested = block.get(key)
            if key == seed and isinstance(nested, dict):
                return nested
            if key != seed and isinstance(nested, dict) and seed in nested:
                return nested[seed]
        value = block.get("seed")
        if str(value) == seed:
            return block
    return block


def family_of(label: str) -> str:
    """The fixed §3 family of a model label; raises on an unknown label.

    R-1 must never pool an unlisted model (an unknown label would silently change
    the comparison after the pilot outcomes are seen).
    """
    try:
        return R1_ALL_LABELS[str(label)]
    except KeyError:
        raise ValueError(f"unknown R-1 model label {label!r}; the §3 family table is "
                         f"fixed and rejects labels outside "
                         f"{sorted(R1_ALL_LABELS)}") from None


def family_table(labels):
    """The §3 family table for ``labels``; an unknown label is refused."""
    return {str(label): family_of(label) for label in labels}


def _get(block, *paths):
    """First present dotted-path value in ``block`` (a ``{"value": x}`` unwraps)."""
    if not isinstance(block, dict):
        return None
    for path in paths:
        current = block
        ok = True
        for part in str(path).split("."):
            if not isinstance(current, dict) or part not in current:
                ok = False
                break
            current = current[part]
        if ok and current is not None:
            if isinstance(current, dict) and "value" in current:
                return current["value"]
            return current
    return None


def find_model_block(payload, label):
    """Locate the per-model block of ``label`` in a pilot payload (tolerant).

    Accepts ``payload[label]``, a label entry under the common containers
    (``models``/``readings``/``rows``/``per_model``/``model_rows``/``items``), or
    a label-keyed dict found recursively.  Returns ``None`` when absent.
    """
    label = str(label)
    if isinstance(payload, dict):
        if label in payload and isinstance(payload[label], dict):
            return payload[label]
        for key in ("models", "readings", "rows", "per_model", "model_rows", "items",
                    "labels", "results"):
            container = payload.get(key)
            if isinstance(container, dict) and label in container \
                    and isinstance(container[label], dict):
                return container[label]
        for value in payload.values():
            found = find_model_block(value, label)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_model_block(item, label)
            if found is not None:
                return found
    return None


def _pair_keys(keys):
    """Normalise per-event pairing keys to a list of tuples of strings."""
    rows = []
    for key in keys:
        if isinstance(key, (tuple, list, np.ndarray)):
            rows.append(tuple(str(part) for part in key))
        else:
            rows.append((str(key),))
    return rows


def response_size(n_embeddings, d_embeddings, *, n_keys=None, d_keys=None) -> dict:
    """§3 response size ``S = E||z_D - z_N||^2 / E||z_N - mean z_N||^2``.

    Paired on ``(run_id, event_id)``.  ``n_keys``/``d_keys`` are the per-event
    pairing keys; when absent the rows are paired positionally.  The denominator
    is the N-row variance by definition, so it does not depend on the D set.
    """
    n = np.asarray(n_embeddings, dtype=np.float64)
    d = np.asarray(d_embeddings, dtype=np.float64)
    if n.ndim != 2 or d.ndim != 2:
        raise ValueError("response_size needs (N_events, dim) embedding arrays")
    if n.shape[1] != d.shape[1]:
        raise ValueError(f"response_size: embedding dims differ ({n.shape[1]} vs "
                         f"{d.shape[1]})")
    if n_keys is None or d_keys is None:
        if n.shape[0] != d.shape[0]:
            raise ValueError("response_size: unpaired rows without keys")
        n_index = d_index = np.arange(n.shape[0])
    else:
        n_pairs = _pair_keys(n_keys)
        d_pairs = _pair_keys(d_keys)
        if len(n_pairs) != n.shape[0] or len(d_pairs) != d.shape[0]:
            raise ValueError("response_size: pairing keys do not match the rows")
        index = {pair: position for position, pair in enumerate(d_pairs)}
        common = [(position, index[pair]) for position, pair in enumerate(n_pairs)
                  if pair in index]
        if not common:
            raise ValueError("response_size: no paired (run_id, event_id) rows")
        n_index = np.asarray([row[0] for row in common])
        d_index = np.asarray([row[1] for row in common])
    if n_index.size < 2:
        raise ValueError("response_size: fewer than two paired events")
    difference = n[n_index] - d[d_index]
    numerator = float(np.mean(np.sum(difference * difference, axis=1)))
    centered = n[n_index] - n[n_index].mean(axis=0, keepdims=True)
    denominator = float(np.mean(np.sum(centered * centered, axis=1)))
    ratio = None if denominator == 0 else numerator / denominator
    return {"response_size": ratio, "numerator": numerator, "denominator": denominator,
            "n_paired": int(n_index.size), "dim": int(n.shape[1]),
            "definition": "E||z_D - z_N||^2 / E||z_N - mean(z_N)||^2, paired on "
                          "(run_id, event_id)"}


def x2b_class(r2_eq) -> str:
    """§3 X2-b class (occupancy model, on Δz_cs)."""
    if r2_eq is None:
        return "not classified"
    value = float(r2_eq)
    if value >= 0.8:
        return "equivariant"
    if value >= 0.4:
        return "partly equivariant"
    return "non-equivariant"


def r1_class(row: dict) -> dict:
    """§3 four-way taxonomy for one R-1 table row."""
    response = row.get("response_size")
    eligible = row.get("delta_d_eligible")
    if response is not None and float(response) < 0.01 and not eligible:
        klass = "invariant"
    else:
        klass = row.get("x2_class") or x2b_class(row.get("r2_eq"))
    repairs = [row.get(key) for key in ("rho_lr", "rho_swtw", "rho_swunit")]
    repairs = [float(value) for value in repairs if value is not None]
    information_losing = bool(eligible and repairs and all(value <= 0.2 for value in repairs))
    return {"class": klass, "information_losing_candidate": information_losing,
            "criteria": "invariant: S < 0.01 and Δ_D not eligible; otherwise the X2-b "
                        "class; flagged information-losing (candidate) when Δ_D is "
                        "eligible and every repair ρ <= 0.2"}


def _median(values):
    values = [float(value) for value in values if value is not None
              and math.isfinite(float(value))]
    return float(np.median(values)) if values else None


def r1_reading(rows: dict) -> dict:
    """The §3 R-1 reading (fixed before the pilot outcomes are seen).

    Compare the median over the generative models with each ``frozen_target`` seed,
    for R²_eq and for ρ_LR: supports when the generative median exceeds both seeds
    on both quantities, contradicts when it is below both seeds on both, otherwise
    mixed.
    """
    generative = [row for label, row in rows.items()
                  if R1_ALL_LABELS.get(str(label)) == "generative"
                  and row.get("status") == "available"]
    frozen = {seed: rows.get(f"frozen_target{seed}") for seed in (9000, 9001)}
    quantities = ("r2_eq", "rho_lr")
    medians = {name: _median([row.get(name) for row in generative])
               for name in quantities}
    seeds = {name: {str(seed): (None if frozen[seed] is None
                                else frozen[seed].get(name))
                    for seed in frozen}
             for name in quantities}
    comparisons = []
    for name in quantities:
        median = medians[name]
        pair = [seeds[name]["9000"], seeds[name]["9001"]]
        if median is None or any(value is None for value in pair):
            comparisons.append(None)
        elif float(median) > max(float(value) for value in pair):
            comparisons.append("above")
        elif float(median) < min(float(value) for value in pair):
            comparisons.append("below")
        else:
            comparisons.append("mixed")
    if not generative or any(value is None for value in comparisons):
        verdict = "mixed"
        note = "incomplete (a required value is missing); the reading is descriptive"
    elif all(value == "above" for value in comparisons):
        verdict = "supports"
        note = "the generative median exceeds both frozen_target seeds on both quantities"
    elif all(value == "below" for value in comparisons):
        verdict = "contradicts"
        note = "the generative median is below both frozen_target seeds on both quantities"
    else:
        verdict = "mixed"
        note = "mixed: the two quantities do not agree"
    return {"registered": False, "extension": EXTENSION, "id": "R-1 reading",
            "question": "are generative (MAE) encoders more repairable than the "
                        "working joint-embedding encoder?",
            "caveat": "with 8 against 2 models this is a description, not a test",
            "generative_median": dict(medians, n_models=len(generative)),
            "frozen_target": seeds, "verdict": verdict, "note": note,
            "comparisons": comparisons,
            "criteria": "supports: generative median > both frozen_target seeds on "
                        "R²_eq and ρ_LR; contradicts: below both seeds on both; "
                        "otherwise mixed"}

def cache_condition(name: str):
    """The condition token (``N``/``D``) of a cached embedding file name."""
    text = str(Path(str(name)).name)
    for condition in ("N", "D"):
        for token in (f"__{condition}.npz", f"__{condition}__", f"_{condition}.npz"):
            if token in text:
                return condition
    return None


def _sidecar_sha(path: Path, *, large_arrays=None, ext_root=None) -> tuple:
    """The expected sha256 of a cache file and its provenance (no verification)."""
    sidecar_text = path.with_name(path.name + ".sha256")
    sidecar_json = path.with_name(path.name + ".json")
    if sidecar_text.is_file():
        return sidecar_text.read_text().split()[0].strip(), f"sidecar {sidecar_text.name}"
    if sidecar_json.is_file():
        payload = read_json(sidecar_json, {}) or {}
        value = payload.get("sha256") or payload.get("manifest_sha256")
        if value:
            return str(value), f"sidecar {sidecar_json.name}"
    if large_arrays:
        candidates = []
        files = large_arrays.get("files") if isinstance(large_arrays, dict) else None
        if isinstance(files, dict):
            candidates = list(files.items())
        elif isinstance(files, list):
            candidates = [(row.get("path"), row.get("sha256")) for row in files
                          if isinstance(row, dict)]
        if ext_root is not None:
            try:
                relative = path.resolve().relative_to(Path(ext_root).resolve()).as_posix()
            except ValueError:
                relative = path.as_posix()
        else:
            relative = path.as_posix()
        for key, value in candidates:
            key_text = str(key)
            if key_text == relative or key_text.endswith("/" + relative) or \
                    Path(key_text).name == path.name:
                return str(value), f"LARGE_ARRAYS {key_text}"
    return None, None


def verify_cache_file(path, expected_sha256=None, *, large_arrays=None,
                      ext_root=None) -> dict:
    """Accept a cached embedding file only if its sha256 matches; refuse a mismatch."""
    path = Path(path)
    computed = lr.sha256_file(path) if path.is_file() else None
    expected = expected_sha256
    source = "caller"
    if expected is None:
        expected, source = _sidecar_sha(path, large_arrays=large_arrays,
                                        ext_root=ext_root)
    if computed is None:
        raise RuntimeError(f"R-1 cache {path} is missing")
    if expected is None:
        raise RuntimeError(f"R-1 cache {path} has no sha256 sidecar or LARGE_ARRAYS "
                           f"entry; the pilot's cache is refused")
    matched = str(computed) == str(expected)
    if not matched:
        raise RuntimeError(f"R-1 cache {path}: sha256 mismatch ({computed} != "
                           f"{expected} from {source}); refused (the pilot's caches are "
                           f"read-only and must be accepted only by hash)")
    return {"path": str(path), "sha256": computed, "expected_sha256": str(expected),
            "source": source, "pass": True}


def load_cached_embedding(path, *, expected_sha256=None, large_arrays=None,
                          ext_root=None) -> dict:
    """Load one verified cached embedding ``.npz`` (embeddings + pairing keys)."""
    verification = verify_cache_file(path, expected_sha256, large_arrays=large_arrays,
                                     ext_root=ext_root)
    with np.load(Path(path), allow_pickle=True) as payload:
        keys = list(payload.keys())
        embeddings = None
        for key in CACHE_EMBEDDING_KEYS:
            if key in payload:
                embeddings = np.asarray(payload[key], dtype=np.float64)
                break
        if embeddings is None:
            candidates = [key for key in keys
                          if np.asarray(payload[key]).ndim == 2]
            if not candidates:
                raise RuntimeError(f"R-1 cache {path}: no 2-D embedding array found "
                                   f"(keys {keys})")
            embeddings = np.asarray(payload[candidates[0]], dtype=np.float64)
        events = next((np.asarray(payload[key]) for key in CACHE_EVENT_KEYS
                       if key in payload), None)
        runs = next((np.asarray(payload[key]) for key in CACHE_RUN_KEYS
                     if key in payload), None)
    if embeddings.ndim != 2:
        raise RuntimeError(f"R-1 cache {path}: embeddings are not 2-D")
    pairing = None
    if events is not None and runs is not None:
        if len(events) != embeddings.shape[0] or len(runs) != embeddings.shape[0]:
            raise RuntimeError(f"R-1 cache {path}: pairing keys do not match the "
                               f"embeddings")
        pairing = [(str(run), str(event)) for run, event in zip(runs, events)]
    return {"path": str(path), "verification": verification,
            "embeddings": embeddings, "pairing": pairing,
            "n_events": int(embeddings.shape[0])}


def cache_files(ext_root, label, *, large_arrays=None) -> dict:
    """The pilot's cached N and D photon probe-test embedding files for a model.

    Read-only: the pilot's cache directory is inspected, never written.
    """
    directory = Path(ext_root) / "x1" / str(label)
    out = {"directory": str(directory), "exists": directory.is_dir(),
           "conditions": {}}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.npz")):
        condition = cache_condition(path.name)
        if condition is None:
            continue
        expected, source = _sidecar_sha(path, large_arrays=large_arrays,
                                        ext_root=ext_root)
        out["conditions"].setdefault(condition, []).append(
            {"path": str(path), "expected_sha256": expected, "sha_source": source})
    return out


def pilot_status(root=None, *, results_dir=None) -> dict:
    """Whether the Part A repair pilot outputs exist with X1/X2 completed (§3)."""
    root = _resolve_root(root)
    directory = Path(results_dir) if results_dir is not None else (root / PILOT_RESULTS_DIR)
    files = {name: (directory / name) for name in
             ("readings.json", "x1.json", "x2.json")}
    status = {"results_dir": str(directory),
              "files": {name: {"path": str(path), "exists": path.is_file()}
                        for name, path in files.items()},
              "readings": None, "complete": False, "reason": None}
    if not all(row["exists"] for row in status["files"].values()):
        missing = [name for name, row in status["files"].items() if not row["exists"]]
        status["reason"] = f"pilot outputs missing ({', '.join(missing)})"
        return status
    readings = read_json(files["readings.json"])
    if readings is None:
        status["reason"] = "pilot readings.json is not valid JSON"
        return status
    status["readings"] = readings
    sections = []
    for name in ("x1", "x2"):
        section = readings.get(name) if isinstance(readings, dict) else None
        present = section is not None
        failed = False
        if isinstance(section, dict):
            failed = (section.get("status") in ("failed", "not_run", "incomplete")
                      or section.get("complete") is False)
        sections.append(bool(present and not failed))
    status["complete"] = all(sections)
    if not status["complete"]:
        status["reason"] = ("the pilot's report mode has not completed X1 and X2 "
                            "(readings.json lacks a completed x1/x2 section)")
    status["x1"] = read_json(files["x1.json"], {}) or {}
    status["x2"] = read_json(files["x2.json"], {}) or {}
    return status


def r1_payload(root=None, *, results_dir=None, ext_root=PILOT_EXT_ROOT,
               dry_run=None, log=print) -> dict:
    """The R-1 report payload; ``not_run`` until the pilot outputs exist (§3)."""
    root = _resolve_root(root)
    assert_authorised("r1", root=root, dry_run=dry_run)
    status = pilot_status(root, results_dir=results_dir)
    base = {"registered": False, "extension": EXTENSION, "id": "R-1",
            "results_dir": status["results_dir"],
            "pilot_ext_root": str(ext_root),
            "read_only": "the pilot's <ext> and results directories are never written"}
    if not status["complete"]:
        log(f"[shortcut] R-1 not_run: {status['reason']}")
        return dict(base, status="not_run", reason=status["reason"],
                    files=status["files"], reading=None, table={})
    x1, x2 = status["x1"], status["x2"]
    large_arrays = read_json(Path(results_dir or (root / PILOT_RESULTS_DIR))
                             / "LARGE_ARRAYS.json", {}) or {}
    labels = sorted(R1_ALL_LABELS)
    table, unknown = {}, []
    for label in labels:
        block = find_model_block(x1, label) or find_model_block(x2, label)
        row = {"family": family_of(label),
               "status": "available" if block is not None else "unavailable"}
        if block is not None:
            block = _model_block(block, label)
            for name, candidates in R1_FIELDS.items():
                row[name] = _get(block, *candidates)
        row.update(r1_class(row))
        table[label] = row
    for payload in (x1, x2):
        for candidate in _label_like_keys(payload):
            if candidate not in R1_ALL_LABELS:
                unknown.append(str(candidate))
    response = None
    for label in labels:
        row = table[label]
        if row.get("response_size") is not None:
            continue
        cache = cache_files(ext_root, label, large_arrays=large_arrays)
        conditions = cache.get("conditions") or {}
        if "N" not in conditions or "D" not in conditions:
            row["response_size_status"] = "cache missing (not_run)"
            continue
        try:
            n_row = _first_verified(conditions["N"], large_arrays, ext_root)
            d_row = _first_verified(conditions["D"], large_arrays, ext_root)
            n_cache = load_cached_embedding(n_row["path"], large_arrays=large_arrays,
                                            ext_root=ext_root)
            d_cache = load_cached_embedding(d_row["path"], large_arrays=large_arrays,
                                            ext_root=ext_root)
            if n_cache["pairing"] is not None and d_cache["pairing"] is not None:
                size = response_size(n_cache["embeddings"], d_cache["embeddings"],
                                     n_keys=n_cache["pairing"],
                                     d_keys=d_cache["pairing"])
            else:
                size = response_size(n_cache["embeddings"], d_cache["embeddings"])
            row["response_size"] = size["response_size"]
            row["response_size_detail"] = size
            row.update(r1_class(row))
            response = response or size["response_size"]
        except (RuntimeError, ValueError) as problem:
            row["response_size_status"] = f"refused: {problem}"
    reading = r1_reading(table)
    return dict(base, status="complete", files=status["files"], reading=reading,
                table=table, unknown_labels=sorted(set(unknown)),
                n_models=len(table),
                response_size_available=response is not None)


def _label_like_keys(payload, depth=0):
    """Label-like keys of a pilot payload (for the unknown-label check)."""
    found = []
    if depth > 4:
        return found
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict) and any(suffix in str(key) for suffix in
                                               ("9000", "9001")):
                found.append(str(key))
            else:
                found.extend(_label_like_keys(value, depth + 1))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_label_like_keys(item, depth + 1))
    return found


def _first_verified(rows, large_arrays, ext_root):
    """The first cache row whose sha verifies; a mismatch is refused."""
    problems = []
    for row in rows:
        try:
            verify_cache_file(row["path"], row.get("expected_sha256"),
                              large_arrays=large_arrays, ext_root=ext_root)
            return row
        except RuntimeError as problem:
            problems.append(str(problem))
    raise RuntimeError("; ".join(problems) or "no cache row")


# ============================================================ caps / §8.1
def s1_check(root=None) -> dict:
    """Front matter §F3 / Part B §8.1 S1 launch rule."""
    root = _resolve_root(root)
    ledger = read_json(root / "docs/jepa/RESOURCE_LEDGER.json", {}) or {}
    entry = (ledger.get("v4_caps") or {}).get("E2_lucid_l40s_deepthought2") or {}
    charged = entry.get("charged_gpu_hours")
    total = (float(charged) + PART_A_GPU_CAP_H + GPU_CAP_H
             if charged is not None else None)
    return {"registered": False, "extension": EXTENSION, "id": "S1",
            "ledger": "docs/jepa/RESOURCE_LEDGER.json",
            "ledger_job": "E2_lucid_l40s_deepthought2",
            "ledger_charged_gpu_hours": charged,
            "part_a_cap_hours": PART_A_GPU_CAP_H,
            "part_b_cap_hours": GPU_CAP_H,
            "sum_hours": None if total is None else round(total, 4),
            "ceiling_hours": S1_CEILING_H,
            "pass": bool(total is not None and total <= S1_CEILING_H),
            "rule": "L40S ledger + Part A cap (4.0) + Part B cap (1.5) <= 160 "
                    "(front matter §F3)"}


def s1_check_or_stop(root=None) -> dict:
    """S1 is a STOP: no GPU job may launch when the combined rule is crossed."""
    result = s1_check(root)
    if not result["pass"]:
        print(f"[shortcut] S1 STOP: L40S ledger {result['ledger_charged_gpu_hours']} "
              f"+ {result['part_a_cap_hours']} + {result['part_b_cap_hours']} = "
              f"{result['sum_hours']} > {result['ceiling_hours']} h", file=sys.stderr)
        raise SystemExit(2)
    return result


def cost_projection() -> dict:
    """§8.1 cost projection against the 1.5 GPU-h cap; STOP before crossing it."""
    items = {"runs": 2 * COST_PER_RUN_H, "smoke": COST_SMOKE_H,
             "evaluation": COST_EVALUATION_H, "reproduction": COST_REPRODUCTION_H}
    total = float(sum(items.values()))
    return {"registered": False, "extension": EXTENSION, "id": "§8.1 cost projection",
            "items_hours": items, "projected_total_hours": round(total, 4),
            "cap_hours": GPU_CAP_H, "pass": bool(total <= GPU_CAP_H),
            "rule": "reproduction, smoke, two runs and evaluation together <= 1.5 h"}


def entries_snapshot(guard_json=GUARD_JSON) -> dict:
    path = Path(guard_json)
    payload = read_json(path)
    if payload is None:
        return {"available": False, "guard_json": str(path),
                "error": "guard status file is missing or invalid"}
    caps = payload.get("caps") or {}
    free = caps.get("free_entries")
    return {"available": free is not None, "guard_json": str(path),
            "free_entries": None if free is None else int(free),
            "guard_level": payload.get("level"),
            "guard_timestamp": payload.get("timestamp")}


def entries_preflight(guard_json=GUARD_JSON) -> dict:
    """§F3 entries: >= 30 000 free after the projected writes of both parts."""
    snapshot = entries_snapshot(guard_json)
    out = {"registered": False, "extension": EXTENSION, "snapshot": snapshot,
           "projection": None, "pass": False}
    if not snapshot.get("available"):
        out["error"] = snapshot.get("error", "no free_entries in the guard status")
        return out
    projected = (int(snapshot["free_entries"])
                 - (ENTRIES_CLONE + ENTRIES_SUITE + ENTRIES_PART_A + ENTRIES_PART_B))
    out["projection"] = {"free_before": int(snapshot["free_entries"]),
                         "clone": ENTRIES_CLONE, "suite": ENTRIES_SUITE,
                         "part_a_writes": ENTRIES_PART_A,
                         "part_b_writes": ENTRIES_PART_B,
                         "projected_after": projected, "floor": ENTRIES_FLOOR,
                         "pass": bool(projected >= ENTRIES_FLOOR
                                      and int(snapshot["free_entries"]) >= ENTRIES_FLOOR)}
    out["gpu_slot_available"] = bool(snapshot.get("guard_level") != "HOLD")
    out["pass"] = bool(out["projection"]["pass"])
    return out


def job_hours(ext_root=EXT_ROOT) -> dict:
    """Recorded per-job hours from ``<ext>/jobs.jsonl`` (§9.6 README)."""
    path = Path(ext_root) / "jobs.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = read_json_text(line)
            if row is not None:
                rows.append(row)
    gpu = sum(float(row.get("wall_seconds") or 0.0) for row in rows
              if row.get("kind") == "gpu")
    cpu = sum(float(row.get("wall_seconds") or 0.0) for row in rows
              if row.get("kind") == "cpu")
    return {"rows": rows, "gpu_hours": round(gpu / 3600.0, 4),
            "cpu_hours": round(cpu / 3600.0, 4),
            "gpu_cap_hours": GPU_CAP_H, "cpu_cap_hours": CPU_CAP_H}


def read_json_text(text):
    try:
        return json.loads(text)
    except ValueError:
        return None


def record_job(ext_root, *, mode, kind, wall_seconds, extra=None) -> dict:
    """Append one job record to ``<ext>/jobs.jsonl`` (per-job hours, §9.6)."""
    path = Path(ext_root) / "jobs.jsonl"
    row = {"registered": False, "extension": EXTENSION, "mode": str(mode),
           "kind": str(kind), "wall_seconds": None if wall_seconds is None
           else round(float(wall_seconds), 3), "recorded_utc": lr.utcnow(),
           "host": lr.host_name()}
    row.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(lr._safe(row), default=float) + "\n")
    return row


def assert_gpu_budget(ext_root, estimate_h, *, root=None) -> dict:
    """Refuse a GPU job that would cross the 1.5 L40S-h cap (§8.1 STOP)."""
    spent = job_hours(ext_root)["gpu_hours"]
    estimate = 0.0 if estimate_h is None else float(estimate_h)
    total = round(spent + estimate, 4)
    if total > GPU_CAP_H:
        print(f"[shortcut] cap STOP: R-2 GPU {spent:.4f} h booked + {estimate:.2f} h "
              f"projected = {total:.4f} h > {GPU_CAP_H} h; the remaining items are "
              f"not_run (cap) (§8.1)", file=sys.stderr)
        raise SystemExit(2)
    return {"spent_gpu_hours": spent, "estimate_hours": estimate,
            "projected_total_hours": total, "cap_hours": GPU_CAP_H, "pass": True}


def assert_cpu_budget(ext_root, estimate_h) -> dict:
    """Refuse a CPU job that would cross the 2 wall-h cap (§8.1 STOP)."""
    spent = job_hours(ext_root)["cpu_hours"]
    estimate = 0.0 if estimate_h is None else float(estimate_h)
    total = round(spent + estimate, 4)
    if total > CPU_CAP_H:
        print(f"[shortcut] cap STOP: CPU {spent:.4f} h booked + {estimate:.2f} h "
              f"projected = {total:.4f} h > {CPU_CAP_H} h (§8.1)", file=sys.stderr)
        raise SystemExit(2)
    return {"spent_cpu_hours": spent, "estimate_hours": estimate,
            "projected_total_hours": total, "cap_hours": CPU_CAP_H, "pass": True}


# ================================================================ report §9.6
def large_arrays(ext_root=EXT_ROOT, *, min_bytes=LARGE_ARRAY_MIN_BYTES) -> dict:
    """``LARGE_ARRAYS.json``: sha256 of every file > 5 MB under ``<ext>`` (§9.6)."""
    ext_root = Path(ext_root)
    files = []
    if ext_root.is_dir():
        for path in sorted(ext_root.rglob("*")):
            if not path.is_file() or path.name == "LARGE_ARRAYS.json":
                continue
            size = path.stat().st_size
            if size <= min_bytes:
                continue
            files.append({"path": path.as_posix(),
                          "relative": path.relative_to(ext_root).as_posix(),
                          "size_bytes": int(size),
                          "sha256": lr.sha256_file(path)})
    return {"registered": False, "extension": EXTENSION,
            "min_bytes": int(min_bytes), "files": files,
            "note": "read-only sha256 inventory of the extension's large files on /ceph"}


def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


R3_CITATIONS = (
    ("Baade et al., *Generative Representation Learning for Scalable Latent "
     "Tokenization*, arXiv 2610.20533",
     "Discriminative and LeJEPA objectives can be satisfied by a feature subset "
     "(feature suppression); on watermarked data LeJEPA and DINOv3 collapse to the "
     "shared watermark while a generative target avoids this -- external evidence "
     "that SIGReg prevents dimensional collapse but not shortcut collapse, and "
     "support for the R-1 expectation."),
    ("Chen, Luo and Li, *Intriguing Properties of Contrastive Losses*, NeurIPS 2021",
     "Introduces feature suppression in contrastive learning, the older name for the "
     "phenomenon reported here."),
    ("*When Graph-JEPA Learns the Wrong Thing*, arXiv 2608.20516",
     "A Graph-JEPA with a healthy linear probe and effective rank carries zero "
     "instance information; the degenerate solution is a global optimum of the "
     "coupled predictor / EMA-target objective.  \"Effective rank misses the "
     "collapse\" is already published and is not claimed as new here."),
    ("Yang et al., *Denoising Vision Transformers*, ECCV 2024",
     "ViT features split into an input-dependent term and an input-independent, "
     "position-related term -- the precedent for the f_layout / f_event "
     "decomposition."),
)


def _readme(*, runbook_sha256, code_sha, plan, manipulation, reproduction, runs,
            readings, r1, hours) -> str:
    lines = [
        "# Shortcut-collapse follow-up (Part B) — 7 Nov 2026", "",
        "`registered: false`. Extension `shortcut_followup_2026-11-07`, variant "
        "`target_posfree` (`arm_name` stays `sigreg_jepa`).",
        f"Runbook: `{RUNBOOK_REL}` (sha256 `{runbook_sha256}`).",
        "Author-approved training: exactly two 10 000-step runs (seeds 9000/9001) and "
        "one 20-step smoke; no other training, no data generation.",
        "", f"- code SHA: `{code_sha}`",
        f"- gate: `{(plan or {}).get('gate', {}).get('status', 'n/a')}`",
        f"- §5.2 manipulation check: "
        f"{'PASS' if (manipulation or {}).get('pass') else 'FAIL'}",
        f"- §5.3 reproduction: "
        f"{'PASS' if (reproduction or {}).get('reproduction_pass') else 'not_run/FAIL'}",
        "", "## R-2 variant", "",
        "`target_posfree`: identical to the fixed-label SIGReg-JEPA except that the "
        "label pass receives a clone of `batch.features` with columns 0-5 (scaled "
        "position and direction) set to zero.  Context pass, full pass, predictor and "
        "position queries, SIGReg terms, loss weights, optimiser, schedule, data, cell "
        "and seeds unchanged; the label is not detached and the encoder stays shared "
        "and trainable.", "",
        "## §5.2 manipulation check", ""]
    if manipulation:
        lines += ["| check | value |", "|---|---|"]
        for key, value in (manipulation.get("checks") or {}).items():
            lines.append(f"| {key} | {_fmt(value)} |")
        blind = manipulation.get("identity_blindness") or {}
        grad = manipulation.get("label_gradient") or {}
        frac = manipulation.get("label_fractions") or {}
        lines += [
            f"| label max abs Δ under a position permutation | "
            f"{_fmt(blind.get('label_max_abs_delta'))} (max "
            f"{_fmt(blind.get('label_delta_max'), 1)}) |",
            f"| context max abs Δ under a position permutation | "
            f"{_fmt(blind.get('context_max_abs_delta'))} (min "
            f"{_fmt(blind.get('context_delta_min'), 1)}) |",
            f"| input_proj.weight grad (label MSE alone) | "
            f"{_fmt(grad.get('input_proj_grad_absmax'))} |",
            f"| label f_layout (init) | {_fmt(frac.get('f_layout'))} |",
            f"| label f_event (init) | {_fmt(frac.get('f_event'))} |",
            f"| label fraction events | {_fmt(frac.get('n_events'))} "
            f"(target 64) |",
            f"| pass | {_fmt(manipulation.get('pass'))} |", ""]
    else:
        lines += ["not_run", ""]
    lines += ["## R2-a event dependence (primary)", ""]
    r2a = (readings or {}).get("r2_a") or {}
    lines += [f"- class: **{r2a.get('class', 'n/a')}**",
              f"- criteria: {r2a.get('criteria', '')}", ""]
    lines += ["## R2-b recovery", ""]
    r2b = (readings or {}).get("r2_b") or {}
    lines += [f"- verdict: **{r2b.get('verdict', 'n/a')}**",
              f"- criteria: {r2b.get('criteria', '')}", "",
              "## R2-c descriptive (per seed)", "",
              "| seed | full vertex m | full token R² | half vertex m | half token R² | "
              "eff. rank | module-gain R² | f_layout | label f_event |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, row in ((readings or {}).get("r2_c", {}).get("seeds") or {}).items():
        lines.append(
            f"| {seed} | {_fmt(row.get('full_input_vertex_m'))} | "
            f"{_fmt(row.get('full_input_token_r2'))} | "
            f"{_fmt(row.get('half_pass_vertex_m'))} | "
            f"{_fmt(row.get('half_pass_token_r2'))} | "
            f"{_fmt(row.get('effective_rank'))} | "
            f"{_fmt(row.get('module_gain_r2'))} | {_fmt(row.get('f_layout'))} | "
            f"{_fmt(row.get('label_pass_f_event'))} |")
    lines += ["", "Reference values (descriptive; never thresholds): "
              + "; ".join(f"{key} = {_fmt(value)}"
                          for key, value in REFERENCE_VALUES.items()), ""]
    if runs:
        lines += ["| seed | run_id | steps | resumed_from_step | checkpoint sha256 | "
                  "wall h |", "|---:|---|---:|---:|---|---:|"]
        for seed, run in runs.items():
            summary = run.get("ext_summary") or {}
            wall = summary.get("wall_seconds")
            lines.append(
                f"| {seed} | {run.get('run_id')} | {_fmt(summary.get('steps_completed'))} "
                f"| {_fmt(summary.get('resumed_from_step'))} | "
                f"`{_fmt(run.get('checkpoint_sha256'), 12)}` | "
                f"{_fmt(None if wall is None else float(wall) / 3600.0)} |")
        lines.append("")
    lines += ["## R-1 — are generative encoders more repairable?", ""]
    if not r1 or r1.get("status") != "complete":
        lines += [f"`not_run` ({(r1 or {}).get('reason', 'pilot outputs missing')}).",
                  "This is not a STOP; fill it in later with `report --r1-only`.", ""]
    else:
        table = r1.get("table") or {}
        lines += ["| model | family | S | R²_eq | R²_eq class | ρ_LR | ρ_SWtw | "
                  "ρ_SWunit | share_cs | Δ_D (m) | eligible | class |",
                  "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|"]
        for label in sorted(table):
            row = table[label]
            lines.append(
                f"| {label} | {row.get('family')} | {_fmt(row.get('response_size'))} | "
                f"{_fmt(row.get('r2_eq'))} | {row.get('x2_class') or ''} | "
                f"{_fmt(row.get('rho_lr'))} | {_fmt(row.get('rho_swtw'))} | "
                f"{_fmt(row.get('rho_swunit'))} | {_fmt(row.get('share_cs'))} | "
                f"{_fmt(row.get('delta_d'))} | {_fmt(row.get('delta_d_eligible'))} | "
                f"{row.get('class')} |")
        reading = r1.get("reading") or {}
        lines += ["", f"R1 reading: **{reading.get('verdict', 'n/a')}** — "
                  f"{reading.get('note', '')}",
                  f"Generative median: R²_eq = "
                  f"{_fmt((reading.get('generative_median') or {}).get('r2_eq'))}, "
                  f"ρ_LR = {_fmt((reading.get('generative_median') or {}).get('rho_lr'))}; "
                  f"frozen_target: R²_eq/ρ_LR = "
                  f"{_fmt((reading.get('frozen_target') or {}).get('r2_eq'))} / "
                  f"{_fmt((reading.get('frozen_target') or {}).get('rho_lr'))}.",
                  "With 8 generative against 2 frozen_target models this is a "
                  "description, not a test.", ""]
    lines += ["## R-3 positioning (citations only; §7)", ""]
    for citation, consequence in R3_CITATIONS:
        lines.append(f"- {citation}: {consequence}")
    lines += ["", "## Per-job hours", ""]
    if hours and hours.get("rows"):
        lines += ["| mode | kind | wall h |", "|---|---|---:|"]
        for row in hours["rows"]:
            lines.append(f"| {row.get('mode')} | {row.get('kind')} | "
                         f"{_fmt(float(row.get('wall_seconds') or 0.0) / 3600.0)} |")
        lines += ["", f"GPU {hours.get('gpu_hours')} h of {GPU_CAP_H} h; CPU "
                  f"{hours.get('cpu_hours')} h of {CPU_CAP_H} h.", ""]
    else:
        lines += ["not recorded yet", ""]
    lines += ["## Outputs", "",
              "- `plan.json` — gate, preflight, S1, caps, entries.",
              "- `manipulation.json` — §5.2 check.",
              "- `reproduction.json` — §5.3 reproduction gate.",
              "- `runs.json` — per-run readouts and the R2 readings.",
              "- `readings.json` — R2-a/R2-b/R2-c and R1 or `not_run`.",
              "- `LARGE_ARRAYS.json` — sha256 of every file > 5 MB under `<ext>`.",
              "- `README.md` — this file.", ""]
    return "\n".join(lines) + "\n"


def assemble_report(*, root=None, data_dir=None, results_dir=None, ext_root=EXT_ROOT,
                    r1_only=False, dry_run=None, log=print) -> dict:
    """Assemble ``results/jepa/20261107_shortcut/`` (§9.6).

    ``r1_only`` fills only the R-1 block (later, when the pilot outputs exist) and
    leaves the R-2 outputs untouched.
    """
    root = _resolve_root(root)
    assert_authorised("report", root=root, dry_run=dry_run)
    ext_root = Path(ext_root)
    results = (Path(results_dir) if results_dir is not None
               else root / RESULTS_DIR)
    read_dir = Path(data_dir) if data_dir is not None else ext_root
    results.mkdir(parents=True, exist_ok=True)
    plan = read_json(read_dir / "plan.json", {}) or {}
    manipulation = read_json(read_dir / "manipulation.json", {}) or {}
    reproduction = read_json(read_dir / "reproduction.json", {}) or {}
    evaluation = read_json(read_dir / "eval" / "eval.json", {}) or {}
    runs = evaluation.get("runs") or {}
    r1 = r1_payload(root, ext_root=PILOT_EXT_ROOT, log=log)
    readings_path = results / "readings.json"
    existing = read_json(readings_path, {}) or {}
    if r1_only:
        payload = existing if existing.get("r2_a") else {}
        payload.setdefault("registered", False)
        payload.setdefault("extension", EXTENSION)
        payload["r1"] = r1
        write_json(readings_path, payload)
        readme = _readme(runbook_sha256=lr.sha256_file(root / RUNBOOK_REL)
                         if (root / RUNBOOK_REL).is_file() else None,
                         code_sha=lr.git_head(root), plan=plan,
                         manipulation=manipulation, reproduction=reproduction,
                         runs=(read_json(results / "runs.json", {}) or {}).get("runs"),
                         readings=payload, r1=r1, hours=job_hours(ext_root))
        (results / "README.md").write_text(readme)
        log(f"[shortcut] report --r1-only: R-1 {r1.get('status')} -> {readings_path}")
        return {"results_dir": str(results), "r1_only": True,
                "r1_status": r1.get("status"), "readings": str(readings_path)}
    readings = {"registered": False, "extension": EXTENSION, "variant": VARIANT}
    readings.update(r2_readings(runs))
    readings["r1"] = r1
    runs_table = {str(seed): {"run_id": run.get("run_id"),
                              "run_dir": run.get("run_dir"),
                              "checkpoint_sha256": run.get("checkpoint_sha256"),
                              "ext_summary": run.get("ext_summary"),
                              "label_fractions": run.get("label_fractions"),
                              "full_input": run.get("full_input"),
                              "half_pass": run.get("half_pass")}
                  for seed, run in runs.items()}
    write_json(results / "plan.json", plan)
    write_json(results / "manipulation.json", manipulation)
    write_json(results / "reproduction.json", reproduction)
    write_json(results / "runs.json", {"registered": False, "extension": EXTENSION,
                                       "variant": VARIANT,
                                       "manipulation": manipulation.get("pass"),
                                       "runs": runs_table,
                                       "readings": {key: readings[key]
                                                    for key in ("r2_a", "r2_b")}})
    write_json(readings_path, readings)
    large = large_arrays(ext_root)
    write_json(results / "LARGE_ARRAYS.json", large)
    hours = job_hours(ext_root)
    readme = _readme(runbook_sha256=lr.sha256_file(root / RUNBOOK_REL)
                     if (root / RUNBOOK_REL).is_file() else None,
                     code_sha=lr.git_head(root), plan=plan, manipulation=manipulation,
                     reproduction=reproduction, runs=runs_table, readings=readings,
                     r1=r1, hours=hours)
    (results / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(results.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 5_000_000}
    log(f"[shortcut] report: {results} (R2-a {readings['r2_a'].get('class')}, "
        f"R2-b {readings['r2_b'].get('verdict')}, R1 {r1.get('status')})")
    return {"results_dir": str(results), "readings": readings,
            "r1_status": r1.get("status"), "sizes": sizes, "too_big": too_big,
            "large_arrays": len(large["files"])}
