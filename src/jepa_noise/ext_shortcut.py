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

Addendum (7 Nov 2026, v1; ``RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md``, sha256
``abe0904a…0621a``)
------------------------------------------------------------------------------
The addendum closes the 7 Nov gaps without training or data generation:

* **D5** -- the R-2 evaluation that the closed Part B cap refused: one job for the
  two existing run directories (full input + half pass via
  ``ext_h12.evaluate_run``, plus the label-pass ``f_event`` over the §5.2 64
  events).  ``r2_readings`` reports ``not_run`` when the run directories carry no
  evaluation, never the 7 Nov "single-seed (not classified)" label.
* **D1/R-1** -- the R-1 rerun re-reads the pilot's updated
  ``x1.json``/``x2.json``/``readings.json`` read-only and re-emits R-1 with the
  corrected ``rho_LR`` (the §3 rule, families and table unchanged).
* **D2 (Part B side)** -- :func:`statistic_audit` checks every comparison for a
  mean point estimate beside an RMS interval; the vertex point here is the mean
  3-D error and no comparison pairs a point with a bootstrap interval.
* **budget/gate** -- the addendum modes book into the shared
  :data:`ADDENDUM_JOBS` ledger under the 0.35 L40S-h / 3 wall-h CPU caps (the
  closed Part A 4.0 h and Part B 1.5 h caps are never consulted), and the gate
  additionally requires the committed addendum's sha256, its §0 author line and
  the record's ``addendum_path``/``addendum_sha256`` (a record missing them is
  DENIED, not pending).
* **A2 (review round 7)** -- the corrected R-1 hand-off: ``addendum-r1`` writes
  its payload both to ``<ext>/r1_addendum.json`` and to the stable results path
  ``results/jepa/20261107_shortcut/r1_addendum.json`` with the reading under
  top-level ``reading``/``r1``/``R-1``; ``addendum-report`` preserves the 7 Nov
  ``readings.json`` write-once under ``<ext>/addendum/preserved_7nov/`` (sha256
  sidecar) before rewriting in place, and reads the superseded table's old value
  from that preserved copy so report ordering cannot lose it.
* **A4 (review round 7)** -- the addendum-only entries preflight: free entries
  from the live CephFS quota xattrs (``ceph.quota.max_files`` -
  ``ceph.dir.rfiles`` - ``ceph.dir.rsubdirs``) or the guard snapshot, projected
  writes = the addendum's own few-dozen writes only (200; no clone/suite/Part
  A/B terms or caps), pass iff ``free >= 30_000`` and ``free - projected >=
  30_000``.  Every addendum mode runs it before work and records it; it is the
  entries-preflight STOP basis (the combined-preflight block stays as history).

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
sections (each marked ``completed`` or ``complete``), and ``x1.json`` /
``x2.json`` with one block per model label (either ``payload["models"][label]``
/ ``payload["readings"][label]`` / ``payload[label]`` or a label-keyed block
found recursively).  Part A writes the X1 fields (``delta_d_m``,
``delta_d_eligible``, ``rho_swtw``, ``rho_swunit``, ``share_cs``) into
``x1.json`` and the X2 fields (``r2_eq``, ``class``, ``rho_lr``) into
``x2.json``; R-1 therefore reads **both** payloads for every model, taking each
field from the payload that carries it and falling back to the other payload
when it is absent.  The fields it reads, by candidate key (the first present
wins), are listed in :data:`R1_FIELDS`.

For the response size it reads the pilot's cached N and D **photon** probe-test
embeddings from ``<pilot ext>/x1/<label>/`` (particle/muon/electron caches are
excluded, and photon ``test__N``/``test__D`` names are preferred).  A cache file
is accepted only if its ``.sha256`` sidecar (or the pilot's
``LARGE_ARRAYS.json``) matches, and a mismatch is refused.  The cache file name
must contain the ``N`` or ``D`` condition token (``...__N.npz`` /
``...__D.npz``) and the ``.npz`` must hold the embeddings under one of
:data:`CACHE_EMBEDDING_KEYS` and the pairing keys under :data:`CACHE_EVENT_KEYS`
/ :data:`CACHE_RUN_KEYS`.

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
from . import lucid_run as lr

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
    # addendum (7 Nov 2026, v1): the shared gate, budget and report helpers
    "ADDENDUM_REL", "ADDENDUM_SHA256", "ADDENDUM_AUTHOR_LINE",
    "ADDENDUM_GPU_CAP_H", "ADDENDUM_CPU_CAP_H", "ADDENDUM_JOBS",
    "ADDENDUM_MODES", "ADDENDUM_COST_BY_ITEM", "ADDENDUM_COST_BY_JOB",
    "ADDENDUM_RESULTS_FILE", "ADDENDUM_DATE", "addendum_jobs",
    "addendum_hours", "assert_addendum_budget", "record_addendum_job",
    "addendum_s1_check", "addendum_s1_check_or_stop", "statistic_audit",
    "COMPARISON_STATISTICS", "r2_evaluation_record", "assemble_addendum_report",
    # addendum A2/A4 (review round 7): R-1 hand-off, 7 Nov preservation, entries
    "R1_RESULTS_FILE", "PRESERVED_7NOV_DIR", "PRESERVED_7NOV_FILE",
    "ADDENDUM_ENTRIES_ROOT", "ADDENDUM_ENTRIES_FLOOR",
    "ADDENDUM_ENTRIES_PROJECTED", "ADDENDUM_ENTRY_XATTRS",
    "results_r1_payload", "write_results_r1", "results_r1_path",
    "preserved_7nov_path", "preserve_7nov_readings", "seven_nov_readings",
    "addendum_entries_snapshot", "addendum_entries_preflight",
    "assert_addendum_entries_preflight",
]

# ---------------------------------------------------------------- extension
EXTENSION = "shortcut_followup_2026-11-07"
VARIANT = "target_posfree"
ARM_NAME = "sigreg_jepa"
CELL_ID = "dev_s005_within"
MODES = ("dry-run", "preflight", "manipulation", "reproduce", "smoke", "train",
         "evaluate", "r1", "report")
#: Addendum §2/§6/§8: the addendum-only modes (run under the addendum 0.35 h GPU /
#: 3 wall-h CPU caps, never the closed 7 Nov caps).
ADDENDUM_MODES = ("addendum-evaluate", "addendum-r1", "addendum-report")
ALL_MODES = MODES + ADDENDUM_MODES
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
#: --- addendum (7 Nov 2026, v1; sha256 abe0904a…0621a) ----------------------
#: Addendum §7: the re-issued acceptance record has the six 7 Nov keys plus
#: ``addendum_path`` and ``addendum_sha256``; a record missing either is DENIED,
#: never ``gate_pending_acceptance`` (that state is only for an absent record).
ADDENDUM_REL = "docs/jepa/RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md"
ADDENDUM_SHA256 = "abe0904af0ed0b5a6586cee459c507ae0fa2a7c60c65ffb58192c71b7180621a"
ADDENDUM_AUTHOR_LINE = "写进weekly report，然后我选A，你写一个小runbook"
ADDENDUM_DATE = "2026-11-07"
ADDENDUM_GPU_CAP_H = 0.35
ADDENDUM_CPU_CAP_H = 3.0
#: The shared addendum ledger (one JSON line per CPU/GPU job; O_APPEND).
ADDENDUM_JOBS = Path("/ceph/dwong/jepa/returns/addendum_20261107_jobs.jsonl")
#: Addendum §6 projections (L40S h); the R-2 evaluation is Part B's only GPU item.
ADDENDUM_COST_BY_ITEM = {"r2_evaluation": 0.14, "est_latent": 0.06,
                         "init9000_tokens": 0.07}
ADDENDUM_COST_BY_JOB = {"addendum-evaluate": ADDENDUM_COST_BY_ITEM["r2_evaluation"],
                        "addendum-r1": 0.02, "addendum-report": 0.02}
ADDENDUM_RESULTS_FILE = "ADDENDUM_2026-11-07.json"
#: Addendum A2 (review round 7): the corrected R-1 hand-off.
#: ``addendum-r1`` writes its payload both to ``<ext>/r1_addendum.json`` and to
#: the stable results path ``results/jepa/20261107_shortcut/r1_addendum.json``,
#: with the reading at top-level ``reading`` (also ``r1``/``R-1`` for Part A's
#: superseded-table picker).  The 7 Nov ``readings.json`` is preserved
#: write-once (with a sha256 sidecar) before any in-place rewrite, and the old
#: value is always read from that preserved copy when it exists.
R1_RESULTS_FILE = "r1_addendum.json"
PRESERVED_7NOV_DIR = "addendum/preserved_7nov"
PRESERVED_7NOV_FILE = "readings.json"
#: Addendum A4 (review round 7): the addendum-only entries preflight.  The
#: runbook §F3 clone/suite/Part A/B terms do not apply to the addendum's own
#: few-dozen writes; the combined-preflight block is history only and is not the
#: addendum STOP basis.
ADDENDUM_ENTRIES_ROOT = Path("/ceph/dwong")
ADDENDUM_ENTRIES_FLOOR = 30000
ADDENDUM_ENTRIES_PROJECTED = 200
ADDENDUM_ENTRY_XATTRS = (("quota_max_files", "ceph.quota.max_files"),
                         ("quota_rfiles", "ceph.dir.rfiles"),
                         ("quota_rsubdirs", "ceph.dir.rsubdirs"))
ACCEPTANCE_KEYS = ("runbook_sha256", "impl_commit", "review_path", "review_verdict",
                   "lead", "date", "addendum_path", "addendum_sha256")
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
    #: Addendum §7: the committed addendum must be unchanged as well.
    ADDENDUM_REL,
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
    # (a, addendum §7) the committed addendum: its sha256 is the committed value
    # (and the acceptance record's), its §0 block carries the §0 author line, and
    # the record names ``addendum_path`` and ``addendum_sha256``.
    addendum_file = root / ADDENDUM_REL
    computed_addendum = (lr.sha256_file(addendum_file) if addendum_file.is_file()
                         else None)
    expected_addendum = ((record or {}).get("addendum_sha256") if record
                         else ADDENDUM_SHA256)
    addendum_text = addendum_file.read_text() if addendum_file.is_file() else ""
    addendum_block = author_section(addendum_text, ADDENDUM_AUTHOR_LINE)
    addendum_sha_pass = bool(computed_addendum is not None
                             and computed_addendum == ADDENDUM_SHA256
                             and computed_addendum == expected_addendum)
    addendum_line_pass = bool(addendum_block
                              and ADDENDUM_AUTHOR_LINE in addendum_block)
    addendum_path_pass = bool(record is None
                              or str(record.get("addendum_path")) == ADDENDUM_REL)
    a_pass = bool(sha_pass and line_pass and addendum_sha_pass
                  and addendum_line_pass and addendum_path_pass)
    checks["a"] = {"pass": a_pass, "runbook": str(runbook_path),
                   "runbook_sha256": computed_runbook,
                   "expected_sha256": expected_runbook,
                   "author_line": AUTHOR_LINE, "author_line_in_section_zero": line_pass,
                   "addendum": str(addendum_file),
                   "addendum_sha256": computed_addendum,
                   "addendum_sha256_committed": ADDENDUM_SHA256,
                   "expected_addendum_sha256": expected_addendum,
                   "addendum_sha256_pass": addendum_sha_pass,
                   "addendum_author_line": ADDENDUM_AUTHOR_LINE,
                   "addendum_author_line_in_section_zero": addendum_line_pass,
                   "addendum_path_in_record": (record or {}).get("addendum_path"),
                   "addendum_path_match": addendum_path_pass,
                   "pending_acceptance": bool(pending)}
    if not a_pass and not pending:
        problems.append("(a) the committed runbook sha256 and/or the §0 author line "
                        "does not match the acceptance record; or the committed "
                        "addendum (sha256, §0 author line, "
                        "addendum_path/addendum_sha256 in the record) does not")

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
    if mode not in ALL_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {ALL_MODES}")
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


# =============================================================== addendum §6
#: The shared addendum budget (Part A and Part B book into one ledger; addendum
#: §6).  The CLOSED 7 Nov Part A (4.0 h) and Part B (1.5 h) caps are never
#: consulted by an addendum mode.
def addendum_jobs(path=None) -> list:
    """The rows of the shared addendum ledger (one JSON line per job)."""
    path = Path(path) if path is not None else ADDENDUM_JOBS
    recorded = []
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = read_json_text(line)
            if isinstance(row, dict):
                recorded.append(row)
    return recorded


def addendum_hours(path=None) -> dict:
    """Recorded addendum hours by kind (addendum §6: GPU 0.35 h, CPU 3.0 h)."""
    path = Path(path) if path is not None else ADDENDUM_JOBS
    rows = addendum_jobs(path)
    gpu = sum(float(row.get("hours") or 0.0) for row in rows
              if str(row.get("kind")) == "gpu")
    cpu = sum(float(row.get("hours") or 0.0) for row in rows
              if str(row.get("kind")) == "cpu")
    return {"path": str(path), "rows": rows, "n_rows": len(rows),
            "gpu_hours": round(gpu, 4), "cpu_hours": round(cpu, 4),
            "gpu_cap_hours": ADDENDUM_GPU_CAP_H,
            "cpu_cap_hours": ADDENDUM_CPU_CAP_H,
            "parts": sorted({str(row.get("part")) for row in rows
                             if row.get("part") is not None})}


def assert_addendum_budget(kind, projected_hours, part="B", *, path=None) -> dict:
    """Refuse an addendum job that would cross 0.35 GPU-h or 3.0 CPU wall-h.

    The projection is the addendum §6 item projection (Part B R-2 evaluation
    0.14 h; Part A EST(latent) 0.06 h; init9000 gain-blind tokens <= 0.07 h).
    A refusal is ``SystemExit(2)`` and the item is recorded as ``not_run
    (cap)`` (addendum §6).  The closed 7 Nov caps are not read.
    """
    kind = str(kind).lower()
    if kind not in ("gpu", "cpu"):
        raise ValueError(f"kind must be 'gpu' or 'cpu', got {kind!r}")
    hours = addendum_hours(path)
    spent = float(hours[f"{kind}_hours"])
    cap = ADDENDUM_GPU_CAP_H if kind == "gpu" else ADDENDUM_CPU_CAP_H
    estimate = 0.0 if projected_hours is None else float(projected_hours)
    total = round(spent + estimate, 4)
    if total > cap:
        print(f"[shortcut] addendum cap STOP: {kind.upper()} {spent:.4f} h booked + "
              f"{estimate:.4f} h projected = {total:.4f} h > {cap} h (addendum §6, "
              f"part {part}); the item is not_run (cap)", file=sys.stderr)
        raise SystemExit(2)
    return {"kind": kind, "part": str(part), "spent_hours": spent,
            "estimate_hours": estimate, "projected_total_hours": total,
            "cap_hours": cap, "pass": True, "ledger": str(hours["path"])}


def record_addendum_job(part, *, mode, kind, wall_seconds, path=None) -> dict:
    """Append one addendum job row to the shared ledger (O_APPEND).

    The row has exactly the addendum §6 fields ``{utc, part, mode, kind,
    wall_seconds, hours}``; one JSON line is appended in a single ``write`` to a
    file opened with ``O_APPEND`` so concurrent Part A/Part B bookings do not
    interleave.
    """
    path = Path(path) if path is not None else ADDENDUM_JOBS
    seconds = None if wall_seconds is None else round(float(wall_seconds), 3)
    row = {"utc": lr.utcnow(), "part": str(part), "mode": str(mode),
           "kind": str(kind), "wall_seconds": seconds,
           "hours": None if seconds is None else round(seconds / 3600.0, 6)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:  # O_APPEND: atomic line
        handle.write(json.dumps(lr._safe(row), default=float) + "\n")
    return row


def addendum_s1_check(root=None) -> dict:
    """Addendum §6 S1: L40S ledger + the addendum GPU cap (0.35) <= 160."""
    root = _resolve_root(root)
    ledger = read_json(root / "docs/jepa/RESOURCE_LEDGER.json", {}) or {}
    entry = (ledger.get("v4_caps") or {}).get("E2_lucid_l40s_deepthought2") or {}
    charged = entry.get("charged_gpu_hours")
    total = (float(charged) + ADDENDUM_GPU_CAP_H if charged is not None else None)
    return {"registered": False, "extension": EXTENSION, "id": "S1 (addendum)",
            "ledger": "docs/jepa/RESOURCE_LEDGER.json",
            "ledger_job": "E2_lucid_l40s_deepthought2",
            "ledger_charged_gpu_hours": charged,
            "addendum_cap_hours": ADDENDUM_GPU_CAP_H,
            "sum_hours": None if total is None else round(total, 4),
            "ceiling_hours": S1_CEILING_H,
            "pass": bool(total is not None and total <= S1_CEILING_H),
            "rule": "L40S ledger + addendum GPU cap (0.35) <= 160 (addendum §6; the "
                    "closed Part A and Part B caps are not consulted)"}


def addendum_s1_check_or_stop(root=None) -> dict:
    """Addendum S1 is a STOP: no addendum GPU job may launch when it is crossed."""
    result = addendum_s1_check(root)
    if not result["pass"]:
        print(f"[shortcut] addendum S1 STOP: L40S ledger "
              f"{result['ledger_charged_gpu_hours']} + "
              f"{result['addendum_cap_hours']} = {result['sum_hours']} > "
              f"{result['ceiling_hours']} h (addendum §6)", file=sys.stderr)
        raise SystemExit(2)
    return result


# ===================================== addendum A2 (review round 7): R-1 hand-off
#: The results-dir R-1 hand-off (Part A's superseded-table picker reads the
#: stable file and a top-level ``reading`` / ``r1`` / ``R-1`` key).
def results_r1_path(root=None, *, results_dir=None) -> Path:
    """The stable results-dir R-1 hand-off path (``.../r1_addendum.json``)."""
    directory = (Path(results_dir) if results_dir is not None
                 else _resolve_root(root) / RESULTS_DIR)
    return directory / R1_RESULTS_FILE


def results_r1_payload(r1, *, results_dir=None) -> dict:
    """The stable results-dir R-1 payload with the reading under the fixed keys.

    Part A's superseded table reads a top-level ``r1``, ``R-1`` or ``reading``;
    all three are written so the D1-corrected value is found regardless of which
    picker key it tries.  The full R-1 payload (status, table, reading) is kept.
    """
    payload = dict(r1) if isinstance(r1, dict) else {}
    reading = payload.get("reading")
    payload.setdefault("registered", False)
    payload.setdefault("extension", EXTENSION)
    payload["reading"] = reading
    payload["r1"] = reading
    payload["R-1"] = reading
    payload["stable_path"] = str(results_r1_path(results_dir=results_dir))
    payload["addendum"] = {"path": ADDENDUM_REL, "sha256": ADDENDUM_SHA256,
                           "date": ADDENDUM_DATE,
                           "hand_off": "stable results-dir R-1 for the Part A "
                                       "superseded table (addendum A2)"}
    return payload


def write_results_r1(root=None, payload=None, *, results_dir=None) -> dict:
    """Write the stable results-dir R-1 hand-off (addendum A2, review round 7)."""
    target = results_r1_path(root, results_dir=results_dir)
    full = results_r1_payload(payload, results_dir=target.parent)
    write_json(target, full)
    return {"path": str(target), "sha256": lr.sha256_file(target),
            "status": full.get("status"),
            "reading": full.get("reading")}


#: Addendum A2: the write-once preserved 7 Nov readings.
def preserved_7nov_path(ext_root=EXT_ROOT) -> Path:
    """``<ext>/addendum/preserved_7nov/readings.json`` (write-once)."""
    return Path(ext_root) / PRESERVED_7NOV_DIR / PRESERVED_7NOV_FILE


def preserve_7nov_readings(results_dir=None, *, ext_root=EXT_ROOT) -> dict:
    """Preserve the 7 Nov ``readings.json`` write-once before it is rewritten.

    The superseded table's old value (in particular the 7 Nov R-1 ``rho_LR``)
    must not depend on the order of the Part A/Part B reports: the first
    ``addendum-report`` copies the current results ``readings.json`` to
    ``<ext>/addendum/preserved_7nov/readings.json`` and writes a sha256 sidecar;
    every later run keeps that copy exactly as it is.
    """
    ext_root = Path(ext_root)
    results = (Path(results_dir) if results_dir is not None
               else _resolve_root(None) / RESULTS_DIR)
    source = results / "readings.json"
    target = preserved_7nov_path(ext_root)
    sidecar = target.with_name(target.name + ".sha256")
    if target.is_file():
        computed = lr.sha256_file(target)
        expected = (sidecar.read_text().split()[0].strip()
                    if sidecar.is_file() else None)
        if expected is None:  # sidecar lost: repair it, never the preserved file
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(f"{computed}  {target.name}\n")
            expected = computed
        if str(expected) != str(computed):
            raise RuntimeError(
                f"preserved 7 Nov readings {target}: sha256 mismatch ({computed} != "
                f"{expected}); the preserved copy is write-once and corrupt")
        return {"path": str(target), "sha256": computed, "written": False,
                "write_once": True, "source": str(source),
                "sidecar": str(sidecar)}
    if not source.is_file():
        return {"path": str(target), "sha256": None, "written": False,
                "write_once": True, "source": str(source),
                "reason": "no results readings.json exists yet"}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    computed = lr.sha256_file(target)
    sidecar.write_text(f"{computed}  {target.name}\n")
    return {"path": str(target), "sha256": computed, "written": True,
            "write_once": True, "source": str(source),
            "source_sha256": computed, "sidecar": str(sidecar)}


def seven_nov_readings(results_dir=None, *, ext_root=EXT_ROOT) -> dict:
    """The 7 Nov readings; the preserved write-once copy wins when it exists."""
    target = preserved_7nov_path(ext_root)
    preserved = read_json(target, None)
    if isinstance(preserved, dict) and preserved:
        return {"readings": preserved, "source": str(target), "preserved": True}
    results = (Path(results_dir) if results_dir is not None
               else _resolve_root(None) / RESULTS_DIR)
    current = results / "readings.json"
    return {"readings": read_json(current, {}) or {}, "source": str(current),
            "preserved": False}


# ============================ addendum A4 (review round 7): entries preflight
def _entry_xattr_reading(root=None) -> dict:
    """Live CephFS quota xattrs of the quota root (guard's own rule)."""
    path = Path(root) if root is not None else ADDENDUM_ENTRIES_ROOT
    values = {}
    try:
        for key, attribute in ADDENDUM_ENTRY_XATTRS:
            values[key] = os.getxattr(str(path), attribute)
    except OSError as problem:
        return {"available": False, "path": str(path),
                "error": f"{type(problem).__name__}: {problem}"}
    parsed = {}
    for key, raw in values.items():
        try:
            parsed[key] = int(bytes(raw).decode().strip())
        except (TypeError, ValueError, UnicodeDecodeError):
            return {"available": False, "path": str(path),
                    "error": f"{key} is not an integer: {raw!r}"}
    maximum = parsed["quota_max_files"]
    used = max(0, parsed["quota_rfiles"]) + max(0, parsed["quota_rsubdirs"])
    free = maximum - used
    return {"available": True, "basis": "live CephFS quota xattrs",
            "path": str(path), "quota_max_files": maximum,
            "quota_rfiles": parsed["quota_rfiles"],
            "quota_rsubdirs": parsed["quota_rsubdirs"],
            "free_entries": max(0, free), "free_entries_clamped": free < 0,
            "rule": "free_entries = ceph.quota.max_files - ceph.dir.rfiles - "
                    "ceph.dir.rsubdirs (the guard's own rule)"}


def addendum_entries_snapshot(*, root=None, guard_json=GUARD_JSON) -> dict:
    """Free entries: live Ceph quota xattrs first, the guard snapshot second."""
    live = _entry_xattr_reading(root)
    if live.get("available"):
        return {"available": True, "basis": live["basis"],
                "free_entries": int(live["free_entries"]),
                "path": live.get("path"), "live": live,
                "guard_snapshot": None}
    snapshot = entries_snapshot(guard_json)
    return {"available": bool(snapshot.get("available")),
            "basis": ("guard snapshot (live Ceph quota xattrs unavailable: "
                      f"{live.get('error')})"),
            "free_entries": snapshot.get("free_entries"),
            "path": snapshot.get("guard_json"), "live": live,
            "guard_snapshot": snapshot,
            "level": snapshot.get("guard_level")}


def addendum_entries_preflight(*, root=None, guard_json=GUARD_JSON,
                               free_entries=None, projected=None) -> dict:
    """Addendum A4: the addendum's own entries preflight (revised basis).

    The projection is the addendum's own writes only (a small constant; no
    clone/suite/Part A/B terms and no Part A/B caps).  The rule is
    ``free >= 30_000`` **and** ``free - projected >= 30_000``; a failure is the
    addendum §9 entries-preflight STOP.  This replaces the combined-preflight
    block as the addendum STOP basis (that block stays as history only).
    """
    projected = (ADDENDUM_ENTRIES_PROJECTED if projected is None
                 else int(projected))
    floor = ADDENDUM_ENTRIES_FLOOR
    if free_entries is not None:
        snapshot = {"available": True, "basis": "caller-provided free entries",
                    "free_entries": int(free_entries), "path": None,
                    "live": None, "guard_snapshot": None}
    else:
        snapshot = addendum_entries_snapshot(root=root, guard_json=guard_json)
    out = {"registered": False, "extension": EXTENSION,
           "id": "entries-preflight (addendum)", "snapshot": snapshot,
           "floor": floor, "projected_writes": projected,
           "projected_writes_basis": "the addendum's own writes only (no "
                                     "clone/suite/Part A/B terms; a small "
                                     "constant)",
           "rule": "free entries >= 30000 AND free - projected addendum writes "
                   "(200) >= 30000",
           "part_caps_consulted": False}
    if not snapshot.get("available") or snapshot.get("free_entries") is None:
        out["pass"] = False
        out["error"] = ("free entries unavailable (no live Ceph quota xattrs and "
                        "no guard snapshot free_entries)")
        return out
    free = int(snapshot["free_entries"])
    after = free - projected
    out.update(free_before=free, projected_after=after)
    out["pass"] = bool(free >= floor and after >= floor)
    return out


def assert_addendum_entries_preflight(**kwargs) -> dict:
    """The addendum entries-preflight STOP (addendum §9; SystemExit(2) on failure)."""
    result = addendum_entries_preflight(**kwargs)
    if not result["pass"]:
        print(f"[shortcut] addendum entries-preflight STOP: free "
              f"{result.get('free_before')} - projected "
              f"{result.get('projected_writes')} = {result.get('projected_after')}"
              f" vs floor {result.get('floor')} ({result.get('basis')}); the "
              f"addendum item is not_run (entries)", file=sys.stderr)
        raise SystemExit(2)
    return result


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
                      batches=None, max_events=None, root=None, dry_run=None,
                      mode="evaluate") -> dict:
    """The readout protocol (full input + half pass) for every ``final.pt``.

    For each final checkpoint it also reports the label-pass ``f_event`` over the
    same events as §5.2 (``batches``, the 64-event training-cell selection).

    ``mode`` is the gating mode (``evaluate``, or ``addendum-evaluate`` for the
    addendum §6 R-2 evaluation); both use the same readout protocol.
    """
    assert_authorised(mode, root=_resolve_root(root), dry_run=dry_run)
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


def _run_has_evaluation(run) -> bool:
    """True when a run entry carries the §5.3 readout (full input and half pass)."""
    if not isinstance(run, dict):
        return False
    full, half = run.get("full_input"), run.get("half_pass")
    return bool(isinstance(full, dict) and full and isinstance(half, dict) and half)


def r2_not_run_readings(*, reason, references=None) -> dict:
    """The ``not_run`` R2-a/R2-b/R2-c payload (addendum §6: never "single-seed").

    Until the single evaluation job has produced a readout for at least one run
    directory there is no reading at all; the 7 Nov label "single-seed (not
    classified)" described a run without an evaluation and is forbidden here.
    """
    base_r2_b = recovery_rule({})
    base_r2_b.update({"verdict": "not_run", "status": "not_run", "seeds": {}})
    return {
        "status": "not_run",
        "reason": str(reason),
        "r2_a": {"registered": False, "extension": EXTENSION, "id": "R2-a",
                 "question": "the §4 class of target_posfree on full-input f_event",
                 "class": "not_run", "status": "not_run", "f_event": {}, "seeds": [],
                 "criteria": "the §4 A-1 class needs full-input f_event in both "
                             "seeds (>= 0.01 both: event-dependent; <= 1e-3 both: "
                             "event-invariant; otherwise intermediate)"},
        "r2_b": base_r2_b,
        "r2_c": {"id": "R2-c", "descriptive": True, "status": "not_run",
                 "seeds": {},
                 "references": dict(REFERENCE_VALUES if references is None
                                    else references),
                 "note": "descriptive; the reference values are never thresholds"},
    }


def r2_readings(runs: dict, *, references=None) -> dict:
    """R2-a, R2-b and R2-c from the evaluation runs (§5.4).

    Addendum §6 label fix: when the run directories carry no evaluation (the
    ``evaluate``/``addendum-evaluate`` job has not run) the readings are
    ``not_run``; they are never labelled "single-seed (not classified)".
    """
    runs = {seed: run for seed, run in dict(runs or {}).items()
            if _run_has_evaluation(run)}
    if not runs:
        return r2_not_run_readings(
            reason="the run directories carry no evaluation (the R-2 evaluation "
                   "job has not run); R2-a/R2-b/R2-c are not_run",
            references=references)
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
    r2_c = {"id": "R2-c", "descriptive": True, "status": "complete", "seeds": rows,
            "references": dict(REFERENCE_VALUES if references is None else references),
            "note": "descriptive; the reference values are never thresholds"}
    return {"status": "complete", "r2_a": r2_a, "r2_b": r2_b, "r2_c": r2_c}


# ===================================== D2 statistic audit (Part B side, addendum)
#: The statistic of every point estimate that ``ext_shortcut`` reports or
#: compares.  ``vertex_resolution_m`` is the **mean** 3-D error
#: (``ext_readout.vertex_resolution`` -> ``lucid_run._vertex_error`` ->
#: ``errors.mean()``); addendum D2's defect was a mean point beside an RMS
#: bootstrap interval, so a vertex interval may only be a mean-error interval.
COMPARISON_STATISTICS = {
    "vertex_resolution_m": "mean",
    "resolution": "mean",
    "vertex": "mean",
    "median": "median",
    "failure_rate": "fraction",
    "token_log_charge_r2": "r2",
    "r2": "r2",
    "module_gain_r2": "r2",
    "f_layout": "fraction",
    "f_event": "fraction",
    "effective_rank": "rank",
    "participation_ratio": "rank",
}
#: Interval names that carry a root-mean-square statistic (the addendum D2 defect).
RMS_STATISTIC_ALIASES = ("rms", "rmse", "root_mean_square", "root-mean-square",
                         "root mean square")
_INTERVAL_TOKENS = ("ci95", "ci_", "_ci", "interval", "bootstrap", "confidence")
_INTERVAL_SUFFIXES = ("_ci95", "_ci", "_bootstrap_interval", "_bootstrap",
                      "_confidence_interval", "_interval")


def _declared_point_statistic(metric) -> str | None:
    """The declared statistic of a point estimate, or ``None`` when unmapped."""
    text = str(metric).lower()
    if text in COMPARISON_STATISTICS:
        return COMPARISON_STATISTICS[text]
    head = text.rsplit(".", 1)[0]
    if head in COMPARISON_STATISTICS:
        return COMPARISON_STATISTICS[head]
    leaf = text.rsplit(".", 1)[-1]
    if leaf in COMPARISON_STATISTICS:
        return COMPARISON_STATISTICS[leaf]
    for name in sorted(COMPARISON_STATISTICS, key=len, reverse=True):
        if text.startswith(name + ".") or text.endswith("." + name):
            return COMPARISON_STATISTICS[name]
    return None


def _interval_statistic(interval) -> str | None:
    """The statistic an interval block declares (``statistic``/``estimator``/...)."""
    if not isinstance(interval, dict):
        return None
    for key in ("statistic", "estimator", "method", "interval_statistic"):
        value = interval.get(key)
        if value is not None:
            return str(value).strip().lower()
    return None


def _interval_metric(key, interval) -> str:
    """The point metric an interval belongs to (explicit field or key suffix)."""
    if isinstance(interval, dict):
        for name in ("metric", "point_metric", "for_metric"):
            value = interval.get(name)
            if value:
                return str(value)
    text = str(key).lower()
    for suffix in _INTERVAL_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _interval_blocks(block) -> list:
    """``(key, interval_dict)`` pairs in ``block`` whose name marks an interval."""
    found = []
    if not isinstance(block, dict):
        return found
    for key, value in block.items():
        if not isinstance(value, dict):
            continue
        lowered = str(key).lower()
        if any(token in lowered for token in _INTERVAL_TOKENS):
            found.append((str(key), value))
    return found


def _audit_interval(seed, form_name, interval_key, base_metric, interval) -> dict:
    """Compare one interval's statistic with the statistic of its point metric."""
    point_statistic = _declared_point_statistic(base_metric)
    interval_statistic = _interval_statistic(interval)
    record = {"seed": str(seed), "form": form_name, "interval": interval_key,
              "metric": str(base_metric), "point_statistic": point_statistic,
              "interval_statistic": interval_statistic, "match": None,
              "reason": None, "pass": False}
    if point_statistic is None or interval_statistic is None:
        record["reason"] = ("unmapped statistic: the interval's statistic cannot be "
                            "checked against its point estimate")
        return record
    same = point_statistic == interval_statistic or (
        point_statistic in RMS_STATISTIC_ALIASES
        and interval_statistic in RMS_STATISTIC_ALIASES)
    record["match"] = bool(same)
    record["pass"] = bool(same)
    if not same:
        record["reason"] = (f"point statistic {point_statistic!r} but interval "
                            f"statistic {interval_statistic!r} (addendum D2)")
    return record


def statistic_audit(runs=None, *, evaluations=None) -> dict:
    """Addendum D2 audit of every ``ext_shortcut`` comparison, Part B side.

    The rule (addendum §3): a point estimate and any interval reported beside it
    must use the same statistic, and the point must be computed on the same
    aligned pairs as the interval.  In this module:

    * ``evaluate_run_dirs`` reports the **mean** vertex error
      (``er.vertex_resolution``, ``errors.mean()``), the token ln q R² and the
      effective rank from the same streamed event pairs; it reports no bootstrap
      interval, so there is no mean-point/RMS-interval pair (the X3-4 defect is
      Part A's);
    * R2-a is a §4 threshold decision on full-input ``f_event`` (a fraction) and
      R2-b a set of threshold decisions on the vertex mean and token R² points;
      neither pairs a point with an interval.

    Any interval that a future payload does carry is checked here: its declared
    statistic must equal the statistic of the point it belongs to.  A mismatch is
    a finding and ``pass`` is False.
    """
    source = evaluations if evaluations is not None else runs
    source = dict(source or {})
    points, pairs, findings = [], [], []
    for seed, run in sorted(source.items(), key=lambda row: str(row[0])):
        if not isinstance(run, dict):
            continue
        for form_name in ("full_input", "half_pass"):
            form = run.get(form_name)
            if not isinstance(form, dict) or not form:
                continue
            for metric_key, value in form.items():
                statistic = _declared_point_statistic(metric_key)
                if statistic is not None:
                    points.append({"seed": str(seed), "form": form_name,
                                   "metric": str(metric_key),
                                   "statistic": statistic, "interval_paired": False})
                if not isinstance(value, dict):
                    continue
                # the point leaf of the metric block (``vertex.resolution``,
                # ``token_log_charge_r2.r2``, ``module_gain_r2.value``, ...)
                leaf = None
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, dict):
                        continue
                    if _declared_point_statistic(f"{metric_key}.{inner_key}") is not None:
                        leaf = str(inner_key)
                        break
                for inner_key, inner_value in value.items():
                    if not isinstance(inner_value, dict):
                        continue
                    lowered = str(inner_key).lower()
                    if not any(token in lowered for token in _INTERVAL_TOKENS):
                        continue
                    base = (inner_value.get("metric") if isinstance(inner_value, dict)
                            else None)
                    if not base:
                        base = f"{metric_key}.{leaf or 'value'}"
                    record = _audit_interval(
                        seed, form_name, f"{metric_key}.{inner_key}", base,
                        inner_value)
                    pairs.append(record)
                    if not record["pass"]:
                        findings.append(record)
            for interval_key, interval in _interval_blocks(form):
                base = _interval_metric(interval_key, interval)
                record = _audit_interval(seed, form_name, interval_key, base, interval)
                pairs.append(record)
                if not record["pass"]:
                    findings.append(record)
    decisions = [
        {"id": "R2-a", "comparison": "the §4 class of target_posfree on full-input "
                                     "f_event",
         "point_statistic": "fraction", "interval_paired": False,
         "rule": "three-valued threshold on both seeds' point f_event"},
        {"id": "R2-b", "comparison": "the §4 recovery rule",
         "point_statistic": "mean (vertex) / r2 (token ln q)",
         "interval_paired": False,
         "rule": "thresholds on the point estimates; no interval is compared"},
        {"id": "R2-c", "comparison": "the per-seed descriptive table",
         "point_statistic": "mean vertex / r2 / rank / fraction; no interval",
         "interval_paired": False, "rule": "descriptive points only"},
    ]
    return {"registered": False, "extension": EXTENSION, "id": "D2 (Part B)",
            "scope": "ext_shortcut",
            "vertex_point_statistic": "mean",
            "vertex_point_source": "ext_readout.vertex_resolution -> "
                                   "lucid_run._vertex_error: errors.mean()",
            "point_estimates": points, "interval_pairs": pairs,
            "findings": findings, "decisions": decisions,
            "pass": bool(not findings),
            "rule": "a point estimate may only be paired with an interval computed "
                    "from the same statistic on the same aligned pairs; the vertex "
                    "point is the mean error, so a vertex interval must be a mean "
                    "interval (never RMS) -- addendum D2",
            "n_evaluated_runs": len([run for run in source.values()
                                     if _run_has_evaluation(run)])}


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
#: §3 "the eight generative models": the fixed core list, **never** the optional
#: ``mae_within_median`` rows (N11).
R1_GENERATIVE_CORE = tuple(R1_FAMILIES["generative"])
R1_ALL_LABELS = {label: family for family, labels in
                 list(R1_FAMILIES.items()) + list(R1_OPTIONAL.items())
                 for label in labels}
#: Which pilot payload a field belongs to (Part A writes the X1 fields into
#: ``x1.json`` and the X2 fields into ``x2.json``); each field is taken from its
#: own payload first and falls back to the other payload when absent (B7).
R1_FIELD_PAYLOAD = {
    "delta_d": "x1", "delta_d_eligible": "x1", "rho_swtw": "x1",
    "rho_swunit": "x1", "share_cs": "x1",
    "r2_eq": "x2", "x2_class": "x2", "rho_lr": "x2",
    "f_layout": "x1", "f_event": "x1", "response_size": "x1",
}
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


def _field_from_payloads(blocks, candidates, preferred=None):
    """First present field value across the per-payload model blocks (B7).

    ``blocks`` maps ``"x1"``/``"x2"`` to the model block of that payload (only the
    payloads that carry the label are present).  Part A writes the X1 fields into
    ``x1.json`` and the X2 fields into ``x2.json``; a field is therefore read from
    its own payload first (``preferred``) and falls back to the other payload
    only when it is absent there.
    """
    ordered = []
    if preferred in blocks and blocks[preferred] is not None:
        ordered.append(blocks[preferred])
    for name in ("x1", "x2"):
        if name != preferred and blocks.get(name) is not None:
            ordered.append(blocks[name])
    for block in ordered:
        value = _get(block, *candidates)
        if value is not None:
            return value
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
    # §3: the median over the eight generative models, never the optional
    # ``mae_within_median`` rows (N11).
    generative = [rows.get(label) for label in R1_GENERATIVE_CORE
                  if (rows.get(label) or {}).get("status") == "available"]
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
            "generative_median": dict(medians, n_models=len(generative),
                                      n_core_models=len(R1_GENERATIVE_CORE),
                                      labels=list(R1_GENERATIVE_CORE)),
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


#: §3 response size uses the **photon** probe-test cache only: the pilot also
#: caches particle/muon/electron embeddings (``particle__test__N.npz``) in the
#: same directory, and a name-only condition match would pick those first (B8).
PARTICLE_CACHE_TOKENS = ("particle", "muon", "electron")
PHOTON_CACHE_PREFIXES = ("test__", "probe_test__", "probe__test__", "photon")


def is_particle_cache(name) -> bool:
    """True when a cache file name denotes a particle/muon/electron embedding."""
    text = Path(str(name)).name.lower()
    return any(token in text for token in PARTICLE_CACHE_TOKENS)


def _photon_cache_rank(name) -> int:
    """Sort key: photon probe-test names first, then other photon-like names."""
    text = Path(str(name)).name.lower()
    return 0 if any(text.startswith(prefix) or prefix in text
                    for prefix in PHOTON_CACHE_PREFIXES) else 1


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
           "conditions": {}, "excluded_particle": [],
           "selection_rule": "photon probe-test caches only (particle/muon/electron "
                             "excluded; test__N/test__D preferred)"}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.npz")):
        if is_particle_cache(path.name):
            out["excluded_particle"].append(path.name)
            continue
        condition = cache_condition(path.name)
        if condition is None:
            continue
        expected, source = _sidecar_sha(path, large_arrays=large_arrays,
                                        ext_root=ext_root)
        out["conditions"].setdefault(condition, []).append(
            {"path": str(path), "expected_sha256": expected, "sha_source": source,
             "photon": True})
    for rows in out["conditions"].values():
        rows.sort(key=lambda row: (_photon_cache_rank(row["path"]),
                                   Path(row["path"]).name))
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
            # Part A writes ``completed``; accept ``complete`` too (cross-contract).
            finished = section.get("completed", section.get("complete"))
            failed = (section.get("status") in ("failed", "not_run", "incomplete")
                      or finished is False)
        sections.append(bool(present and not failed))
    status["complete"] = all(sections)
    if not status["complete"]:
        status["reason"] = ("the pilot's report mode has not completed X1 and X2 "
                            "(readings.json lacks a completed x1/x2 section)")
    status["x1"] = read_json(files["x1.json"], {}) or {}
    status["x2"] = read_json(files["x2.json"], {}) or {}
    return status


def r1_payload(root=None, *, results_dir=None, ext_root=PILOT_EXT_ROOT,
               dry_run=None, mode="r1", log=print) -> dict:
    """The R-1 report payload; ``not_run`` until the pilot outputs exist (§3).

    ``mode`` is the gating mode (``r1``, or ``addendum-r1`` for the addendum §2
    R-1 rerun that re-reads the pilot's updated ``x1.json``/``x2.json``/
    ``readings.json`` read-only and re-emits R-1 with the D1-corrected
    ``rho_LR``; the §3 rule, families and table are unchanged).
    """
    root = _resolve_root(root)
    assert_authorised(mode, root=root, dry_run=dry_run)
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
        blocks, sources = {}, []
        for name, payload in (("x1", x1), ("x2", x2)):
            block = find_model_block(payload, label)
            if block is not None:
                blocks[name] = _model_block(block, label)
                sources.append(name)
        row = {"family": family_of(label),
               "status": "available" if blocks else "unavailable",
               "payloads": sources}
        for name, candidates in R1_FIELDS.items():
            row[name] = _field_from_payloads(
                blocks, candidates, R1_FIELD_PAYLOAD.get(name))
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


# ==================================== addendum §2/§6/§7: D5, R-1 rerun, report
def r2_evaluation_record(runs, *, run_dirs=(), evaluation=None, budget=None,
                         audit=None) -> dict:
    """The addendum ``ADDENDUM_2026-11-07.json`` record of the D5 R-2 evaluation.

    One evaluation job for the two run directories: full input and half pass via
    ``ext_h12.evaluate_run`` plus the label-pass ``f_event`` over the §5.2 events
    (Part B §5.3), read by the unchanged §5.4 readings.
    """
    runs = dict(runs or {})
    return {
        "registered": False, "extension": EXTENSION, "id": "D5",
        "item": "R-2 evaluation (evaluation only; no training)",
        "protocol": "one job for the two run directories, exactly Part B §5.3: "
                    "ext_h12.evaluate_run full input and half pass, plus the "
                    "label-pass f_event over the §5.2 64 events",
        "run_dirs": [str(path) for path in run_dirs],
        "seeds": sorted(str(seed) for seed in runs),
        "n_runs": len(runs),
        "runs": {str(seed): {"run_id": run.get("run_id"),
                             "run_dir": run.get("run_dir"),
                             "checkpoint_sha256": run.get("checkpoint_sha256"),
                             "full_input": run.get("full_input"),
                             "half_pass": run.get("half_pass"),
                             "label_fractions": run.get("label_fractions")}
                 for seed, run in runs.items()},
        "readings": r2_readings(runs),
        "statistic_audit": (audit if audit is not None
                            else statistic_audit(runs)),
        "budget": budget,
        "evaluation_source": None if evaluation is None else str(evaluation),
        "status": "complete" if runs else "not_run",
    }


def _r1_numeric(payload, quantity, seed=None):
    """A numeric R-1 reading field from the reading payload (or ``None``)."""
    reading = (payload or {}).get("reading")
    if not isinstance(reading, dict):
        return None
    if quantity in ("r2_eq", "rho_lr"):
        if seed is None:
            return (reading.get("generative_median") or {}).get(quantity)
        values = ((reading.get("frozen_target") or {}).get(quantity) or {})
        return values.get(str(seed))
    return reading.get(quantity)


def addendum_superseded(old_readings, old_outcome, readings, r1) -> list:
    """The "Superseded 7 Nov values" table for Part B (addendum §5 D4/§10).

    Lists each replaced number: R-1's ``rho_LR`` (invalid under D1, replaced by the
    D1-corrected rerun), and R2-a/R2-b when they change from the empty 7 Nov
    readings (the R-2 evaluation was refused by the closed Part B cap); R2-c's
    per-seed rows are likewise replaced from empty.  The 7 Nov values are read
    from the committed results files before they are updated in place.
    """
    old_readings = old_readings if isinstance(old_readings, dict) else {}
    old_outcome = old_outcome if isinstance(old_outcome, dict) else {}
    old_r1 = old_readings.get("r1")
    if not isinstance(old_r1, dict):
        old_r1 = {"reading": old_outcome.get("r1_reading") or {}}
    rows = []

    def _row(quantity, old, new, defect, evidence, changed=None):
        rows.append({"quantity": str(quantity),
                     "superseded_7_nov": old,
                     "new_value": new,
                     "changed": bool(changed) if changed is not None else
                     (str(old) != str(new)),
                     "defect": defect, "evidence": evidence})

    # R-1 rho_LR: the generative median and each frozen_target seed (D1).
    _row("R-1 generative-median ρ_LR", _r1_numeric(old_r1, "rho_lr"),
         _r1_numeric(r1, "rho_lr"), "D1",
         f"{r1.get('results_dir', PILOT_RESULTS_DIR)}/x2.json (corrected ρ_LR, "
         f"read-only)")
    for seed in (9000, 9001):
        _row(f"R-1 frozen_target{seed} ρ_LR",
             _r1_numeric(old_r1, "rho_lr", seed),
             _r1_numeric(r1, "rho_lr", seed), "D1",
             f"{r1.get('results_dir', PILOT_RESULTS_DIR)}/x2.json (corrected "
             f"ρ_LR, read-only)")
    new_r2a = (readings.get("r2_a") or {}).get("class")
    old_r2a = (old_readings.get("r2_a") or {}).get("class")
    if new_r2a and new_r2a != old_r2a:
        _row("R2-a class", old_r2a, new_r2a, "D5",
             "the addendum R-2 evaluation (the 7 Nov reading had no evaluation)")
    new_r2b = (readings.get("r2_b") or {}).get("verdict")
    old_r2b = (old_readings.get("r2_b") or {}).get("verdict")
    if new_r2b and new_r2b != old_r2b:
        _row("R2-b verdict", old_r2b, new_r2b, "D5",
             "the addendum R-2 evaluation (the 7 Nov reading had no evaluation)")
    old_r2c = (old_readings.get("r2_c") or {}).get("seeds") or {}
    new_r2c = (readings.get("r2_c") or {}).get("seeds") or {}
    if new_r2c and not old_r2c:
        _row("R2-c per-seed rows", "empty", sorted(new_r2c), "D5",
             "the addendum R-2 evaluation")
    return rows


def _addendum_readme_section(*, record, superseded, r1, readings, hours,
                             completeness, audit, gate=None, preservation=None,
                             results_r1=None) -> str:
    """The README's addendum section (D4/§10: completeness, superseded, figures)."""
    lines = ["", "## Addendum (7 Nov 2026, v1) — D5 R-2 evaluation and corrected R-1",
             "",
             f"Addendum: `{ADDENDUM_REL}` (sha256 `{ADDENDUM_SHA256}`), committed "
             f"unchanged; §0 author line: > \"{ADDENDUM_AUTHOR_LINE}\".",
             "This section updates the 7 Nov record in place (addendum §5 D4 / §10): "
             "the R2 readings were empty in the 7 Nov record because the closed Part B "
             "cap refused the evaluation, and the R-1 `ρ_LR` values were invalid "
             "through the D1 drift-map scale defect.", "",
             "### D5 R-2 evaluation (evaluation only; no training)", "",
             f"- protocol: {record.get('protocol', '')}",
             f"- status: **{record.get('status')}**",
             f"- evaluation source: `{record.get('evaluation_source')}`",
             "- run directories: "
             + (", ".join(f"`{path}`" for path in record.get("run_dirs") or [])
                or "n/a"), "",
             "| seed | run dir | full vertex m | half vertex m | full token R² | "
             "half token R² | eff. rank | f_layout | label f_event |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, run in sorted((record.get("runs") or {}).items()):
        full = run.get("full_input") or {}
        half = run.get("half_pass") or {}
        label = run.get("label_fractions") or {}
        lines.append(
            f"| {seed} | `{run.get('run_dir')}` | "
            f"{_fmt(full.get('vertex_resolution_m'))} | "
            f"{_fmt(half.get('vertex_resolution_m'))} | "
            f"{_fmt((full.get('token_log_charge_r2') or {}).get('r2'))} | "
            f"{_fmt((half.get('token_log_charge_r2') or {}).get('r2'))} | "
            f"{_fmt(full.get('effective_rank'))} | "
            f"{_fmt(_layout_field(full, 'f_layout'))} | "
            f"{_fmt(label.get('f_event'))} |")
    r2a = readings.get("r2_a") or {}
    r2b = readings.get("r2_b") or {}
    lines += ["",
              f"- R2-a event dependence (primary): **{r2a.get('class', 'n/a')}**",
              f"- R2-b recovery: **{r2b.get('verdict', 'n/a')}**",
              f"- R2-c: {len((readings.get('r2_c') or {}).get('seeds') or {})} seed "
              f"row(s), descriptive; reference values are never thresholds.", ""]
    lines += ["### Readings completeness (Part B)", "",
              "| reading | status | value |", "|---|---|---|"]
    for name, row in (completeness or {}).items():
        value = row.get("value")
        if value is None:
            value = row.get("verdict") or row.get("seeds") or ""
        lines.append(f"| {name} | {row.get('status')} | {_fmt(value)} |")
    lines += ["", "`complete` is true only when R2-a/R2-b/R2-c and R-1 are all "
              "present (addendum §5 D4 readings completeness for Part B).", ""]
    lines += ["### R-1 hand-off (addendum A2, review round 7)", ""]
    if results_r1:
        lines += [f"- stable results-dir hand-off: `{results_r1.get('path')}` "
                  f"(sha256 `{_fmt(results_r1.get('sha256'), 16)}`), with the "
                  f"reading at top-level `reading`/`r1`/`R-1` for Part A's "
                  f"superseded-table picker.",
                  f"- `ADDENDUM_2026-11-07.json` carries the rerun under "
                  f"`items.R1_rerun` (status `{r1.get('status')}`)."]
    if preservation:
        lines += [f"- 7 Nov readings preserved write-once: "
                  f"`{preservation.get('path')}` (sha256 "
                  f"`{_fmt(preservation.get('sha256'), 16)}`, "
                  f"written now: {_fmt(bool(preservation.get('written')))}) "
                  f"before the in-place rewrite; the superseded old values are "
                  f"read from that copy."]
    lines += ["", "### Superseded 7 Nov values (Part B)", ""]
    if superseded:
        lines += ["| quantity | superseded 7 Nov value | new value | defect | evidence |",
                  "|---|---|---|---|---|"]
        for row in superseded:
            lines.append(f"| {row['quantity']} | {_fmt(row['superseded_7_nov'])} | "
                         f"{_fmt(row['new_value'])} | {row['defect']} | "
                         f"{row['evidence']} |")
    else:
        lines += ["No Part B value is replaced: there is no R-1 or R2 value in the "
                  "7 Nov record that the addendum changes."]
    lines += ["", "### D2 statistic audit (Part B side)", "",
              f"- pass: {_fmt(audit.get('pass'))}; vertex point statistic: "
              f"{audit.get('vertex_point_statistic')} "
              f"(`{audit.get('vertex_point_source')}`)",
              "- no comparison in `ext_shortcut` pairs a point estimate with a "
              "differently-defined interval; if a payload carries an interval, its "
              "declared statistic is checked against the point it belongs to.", "",
              "### Figures", "",
              "Part B produces **no figures**.  The addendum's six figures (with the "
              "D4 axis labels and legends, and the three-stream identification figure) "
              "are Part A's, under `results/jepa/20261106_repair/`; this README "
              "therefore has no figure legends to list (addendum §5 D4, Part B).", "",
              "### Addendum budget (shared ledger, addendum §6)", ""]
    ledger = hours or {}
    lines += [f"- GPU {ledger.get('gpu_hours')} h of "
              f"{ADDENDUM_GPU_CAP_H} h; CPU {ledger.get('cpu_hours')} h of "
              f"{ADDENDUM_CPU_CAP_H} h; ledger `{ledger.get('path')}`.",
              "- the closed Part A (4.0 h) and Part B (1.5 h) caps are not consulted "
              "by the addendum modes.", ""]
    return "\n".join(lines) + "\n"


def assemble_addendum_report(*, root=None, data_dir=None, results_dir=None,
                             ext_root=EXT_ROOT, dry_run=None, budget=None,
                             log=print) -> dict:
    """Update ``results/jepa/20261107_shortcut/`` in place (addendum §5/§10).

    Writes ``README.md``, ``readings.json``, ``report.json``, ``runs.json``,
    ``OUTCOME.json`` and ``ADDENDUM_2026-11-07.json`` with the addendum R-2
    readings, the corrected R-1, the Part B superseded table and the R-2
    evaluation record.  The pilot results ``<pilot results>`` and ``<pilot ext>``
    are only read.
    """
    root = _resolve_root(root)
    gate = assert_authorised("addendum-report", root=root, dry_run=dry_run)
    ext_root = Path(ext_root)
    results = (Path(results_dir) if results_dir is not None
               else root / RESULTS_DIR)
    read_dir = Path(data_dir) if data_dir is not None else ext_root
    results.mkdir(parents=True, exist_ok=True)

    # ---- inputs, read-only -------------------------------------------------
    evaluation_path = read_dir / "eval" / "addendum_eval.json"
    if not evaluation_path.is_file():
        evaluation_path = read_dir / "eval" / "eval.json"
    evaluation = read_json(evaluation_path, {}) or {}
    runs = evaluation.get("runs") or {}
    r1_path = read_dir / "r1_addendum.json"
    r1 = read_json(r1_path, None)
    if not isinstance(r1, dict) or not r1:
        r1 = r1_payload(root, mode="addendum-r1", ext_root=PILOT_EXT_ROOT, log=log)
        r1_path = None
    run_dirs = evaluation.get("run_dirs") or [
        run.get("run_dir") for run in runs.values() if run.get("run_dir")]
    plan = read_json(read_dir / "plan.json", {}) or {}
    manipulation = read_json(read_dir / "manipulation.json", {}) or {}
    reproduction = read_json(read_dir / "reproduction.json", {}) or {}

    # A2 (review round 7): the stable results-dir R-1 hand-off for Part A's
    # superseded table; recomputed read-only if addendum-r1 did not write it.
    results_r1 = write_results_r1(root, r1, results_dir=results)

    # A2 (review round 7): preserve the 7 Nov readings write-once BEFORE any
    # in-place rewrite, and read the old values from that preserved copy so the
    # superseded table cannot lose the old R-1 rho_LR to report ordering.
    preservation = preserve_7nov_readings(results, ext_root=ext_root)
    seven_nov = seven_nov_readings(results, ext_root=ext_root)
    old_readings = seven_nov["readings"]
    old_outcome = read_json(results / "OUTCOME.json", {}) or {}
    old_runs = read_json(results / "runs.json", {}) or {}

    # ---- readings ----------------------------------------------------------
    r2 = r2_readings(runs)
    audit = statistic_audit(runs)
    record = r2_evaluation_record(runs, run_dirs=run_dirs,
                                  evaluation=evaluation_path, budget=budget,
                                  audit=audit)
    readings = dict(old_readings) if isinstance(old_readings, dict) else {}
    readings.setdefault("registered", False)
    readings.setdefault("extension", EXTENSION)
    readings.setdefault("variant", VARIANT)
    readings.update(r2)
    readings["r1"] = r1
    superseded = addendum_superseded(old_readings, old_outcome, readings, r1)
    completeness = {
        "R2-a": {"status": r2.get("status"),
                 "value": (readings.get("r2_a") or {}).get("class")},
        "R2-b": {"status": r2.get("status"),
                 "value": (readings.get("r2_b") or {}).get("verdict")},
        "R2-c": {"status": r2.get("status"),
                 "seeds": sorted((readings.get("r2_c") or {}).get("seeds") or {})},
        "R-1": {"status": r1.get("status"),
                "value": (r1.get("reading") or {}).get("verdict")},
    }
    complete = bool(r2.get("status") == "complete" and r1.get("status") == "complete")
    readings["addendum"] = {
        "path": ADDENDUM_REL, "sha256": ADDENDUM_SHA256, "date": ADDENDUM_DATE,
        "author_line": ADDENDUM_AUTHOR_LINE, "status": "updated",
        "evaluation": str(evaluation_path),
        "r2_status": r2.get("status"), "r1_status": r1.get("status"),
        "complete": complete, "completeness": completeness,
        "superseded": superseded, "statistic_audit_pass": audit.get("pass"),
        "budget": budget,
        "seven_nov_source": seven_nov["source"],
        "seven_nov_preserved": seven_nov["preserved"],
        "preserved_7nov": preservation,
        "results_r1": results_r1,
    }

    # ---- runs.json ---------------------------------------------------------
    runs_table = {str(seed): {key: run.get(key) for key in
                              ("run_id", "run_dir", "checkpoint_sha256",
                               "ext_summary", "label_fractions", "full_input",
                               "half_pass")}
                  for seed, run in runs.items()}
    runs_payload = dict(old_runs) if isinstance(old_runs, dict) else {}
    runs_payload.setdefault("registered", False)
    runs_payload.setdefault("extension", EXTENSION)
    runs_payload.setdefault("variant", VARIANT)
    runs_payload["manipulation"] = manipulation.get("pass")
    runs_payload["runs"] = runs_table
    runs_payload["readings"] = {"r2_a": readings["r2_a"], "r2_b": readings["r2_b"]}
    runs_payload["r2_c"] = readings["r2_c"]
    runs_payload["addendum"] = {"path": ADDENDUM_REL, "sha256": ADDENDUM_SHA256,
                                "runs": sorted(runs_table)}

    # ---- OUTCOME.json ------------------------------------------------------
    outcome = dict(old_outcome) if isinstance(old_outcome, dict) else {}
    if not outcome:
        outcome = {"registered": False, "extension": EXTENSION,
                   "runbook": RUNBOOK_REL, "runbook_sha256": RUNBOOK_SHA256,
                   "acceptance_record": ACCEPTANCE_REL}
    outcome["status"] = "addendum_updated"
    outcome["addendum"] = {"path": ADDENDUM_REL, "sha256": ADDENDUM_SHA256,
                           "date": ADDENDUM_DATE,
                           "author_line": ADDENDUM_AUTHOR_LINE,
                           "impl_commit": lr.git_head(root),
                           "results_file": ADDENDUM_RESULTS_FILE}
    items = dict(outcome.get("items") or {})
    items["evaluate (addendum D5)"] = (
        f"{'PASS' if r2.get('status') == 'complete' else 'not_run'} (D5: one "
        f"evaluation job for the two run directories under the addendum "
        f"{ADDENDUM_GPU_CAP_H} h cap; R2-a "
        f"{(readings.get('r2_a') or {}).get('class')!r}, R2-b "
        f"{(readings.get('r2_b') or {}).get('verdict')!r}; "
        f"{record.get('n_runs')} run(s) evaluated)")
    items["R-1 (addendum rerun)"] = (
        f"{r1.get('status')} (D1-corrected ρ_LR re-read read-only from "
        f"{r1.get('results_dir')}; the §3 rule, families and table are unchanged)")
    outcome["items"] = items
    if isinstance(r1.get("reading"), dict):
        reading = dict(r1["reading"])
        outcome["r1_reading"] = {
            "verdict": reading.get("verdict"),
            "meaning": reading.get("note"),
            "generative_median": reading.get("generative_median"),
            "frozen_target": reading.get("frozen_target"),
            "caveat": reading.get("caveat"),
            "supersedes_7_nov": True, "defect": "D1",
        }
    outcome["r2_readings"] = {"R2-a": (readings.get("r2_a") or {}).get("class"),
                              "R2-b": (readings.get("r2_b") or {}).get("verdict"),
                              "R2-c": sorted((readings.get("r2_c") or {})
                                             .get("seeds") or {}),
                              "status": r2.get("status")}
    outcome["superseded"] = superseded
    outcome["readings_completeness"] = completeness
    outcome["preserved_7nov"] = preservation
    outcome["results_r1"] = results_r1
    resources = dict(outcome.get("resources") or {})
    resources["addendum"] = addendum_hours()
    resources["addendum_gpu_cap_hours"] = ADDENDUM_GPU_CAP_H
    resources["addendum_cpu_cap_hours"] = ADDENDUM_CPU_CAP_H
    resources["addendum_note"] = ("booked on the shared addendum ledger "
                                  f"{ADDENDUM_JOBS}; the 7 Nov caps are closed and "
                                  f"are not reused")
    outcome["resources"] = resources

    # ---- ADDENDUM_2026-11-07.json -----------------------------------------
    addendum_payload = {
        "registered": False, "extension": EXTENSION, "part": "B",
        "addendum": {"path": ADDENDUM_REL, "sha256": ADDENDUM_SHA256,
                     "date": ADDENDUM_DATE, "author_line": ADDENDUM_AUTHOR_LINE,
                     "impl_commit": lr.git_head(root),
                     "acceptance_record": ACCEPTANCE_REL,
                     "preserved_7nov_readings": preservation},
        "items": {
            "D5_r2_evaluation": record,
            "R1_rerun": {"status": r1.get("status"),
                         "reading": r1.get("reading"),
                         "r1": r1.get("reading"),
                         "source": (str(r1_path) if r1_path is not None
                                    else "recomputed read-only from the pilot"),
                         "stable_results_path": results_r1["path"],
                         "stable_results_sha256": results_r1["sha256"],
                         "pilot_results_dir": r1.get("results_dir"),
                         "superseded": superseded},
            "D2_statistic_audit": audit,
            "D4_readings_completeness": completeness,
            "completed": complete,
        },
        "figures": "Part B produces no figures; the addendum's six figures are "
                   "Part A's under results/jepa/20261106_repair/",
        "budget": addendum_hours(),
        "outputs": ["README.md", "readings.json", "report.json", "runs.json",
                    "OUTCOME.json", ADDENDUM_RESULTS_FILE],
    }

    # ---- README.md ---------------------------------------------------------
    hours = job_hours(ext_root)
    readme = _readme(runbook_sha256=lr.sha256_file(root / RUNBOOK_REL)
                     if (root / RUNBOOK_REL).is_file() else None,
                     code_sha=lr.git_head(root), plan=plan,
                     manipulation=manipulation, reproduction=reproduction,
                     runs=runs_table, readings=readings, r1=r1, hours=hours)
    readme += _addendum_readme_section(
        record=record, superseded=superseded, r1=r1, readings=readings,
        hours=addendum_hours(), completeness=completeness, audit=audit, gate=gate,
        preservation=preservation, results_r1=results_r1)

    # ---- write in place ----------------------------------------------------
    write_json(results / "readings.json", readings)
    write_json(results / "runs.json", runs_payload)
    write_json(results / ADDENDUM_RESULTS_FILE, addendum_payload)
    write_json(results / "OUTCOME.json", outcome)
    (results / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(results.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 5_000_000}
    report_payload = {"registered": False, "extension": EXTENSION,
                      "mode": "addendum-report", "gate": gate,
                      "addendum": addendum_payload["addendum"],
                      "results_dir": str(results),
                      "evaluation": str(evaluation_path), "r1_source": str(r1_path),
                      "readings": readings, "runs": runs_table,
                      "r1_status": r1.get("status"), "r2_status": r2.get("status"),
                      "complete": complete, "statistic_audit": audit,
                      "superseded": superseded, "budget": budget,
                      "addendum_hours": addendum_hours(),
                      "sizes": sizes, "too_big": too_big}
    (results / "report.json").write_text(
        json.dumps(lr._safe(report_payload), indent=1, default=float) + "\n")
    log(f"[shortcut] addendum report: {results} (R2-a "
        f"{(readings.get('r2_a') or {}).get('class')}, R2-b "
        f"{(readings.get('r2_b') or {}).get('verdict')}, R1 {r1.get('status')}, "
        f"{len(superseded)} superseded value(s))")
    return {"results_dir": str(results), "readings": readings,
            "r1_status": r1.get("status"), "r2_status": r2.get("status"),
            "complete": complete, "superseded": superseded,
            "addendum_file": str(results / ADDENDUM_RESULTS_FILE),
            "statistic_audit_pass": audit.get("pass"), "sizes": sizes,
            "too_big": too_big, "report_path": str(results / "report.json"),
            "preserved_7nov": preservation, "results_r1": results_r1,
            "seven_nov_source": seven_nov["source"]}
