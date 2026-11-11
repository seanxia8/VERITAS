# SPDX-License-Identifier: MIT
"""Paper 3 P2 extension: information-destroying drifts and the price of invariance.

Runbook: ``docs/jepa/RUNBOOK_2026-11-09_P2_INFOLOSS.md`` (v1, 9 Nov 2026),
extension ``paper3_p2_infoloss_2026-11-09``, ``registered: false``.

Stage 1 trains nothing and generates no data.  Every factor of §2.3 is an exact
transform of the stored hits, built only from the generator's own primitives
(``lucid_bombs.leading_edge_time`` and the imported trigger rule
``lucid_bombs.pmt_time_rule()``); nothing is reimplemented.

Sections
--------
* gate (§8.2) and the executed-import closure re-asserted at every mode;
* the §2.3 hit-set transforms (threshold raise, dead channels, R1) and the
  window-edge accounting (§2.3 item 3);
* the §2.2 inventory (the 13 Part A core rows plus the two ``target_posfree``
  models), §3.2-style embedding caches and §3 probes/metrics (all reused from
  ``ext_repair``);
* X0-a preflight (§4) and X0-b reproduction (§4);
* the GPU modes ``reproduce``, ``new-models`` and ``conditions``;
* the CPU readings ``p2`` (§5) and ``x3p`` (§6), and ``report`` (§8.6).

The modes are ``dry-run``, ``preflight``, ``reproduce``, ``new-models``,
``conditions``, ``p2``, ``x3p`` and ``report``.  There is no training or data
generation mode; the only generation in this file is the test-only synthetic
generator of the test module.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import ext_followup as ef
from . import ext_repair as er
from . import lucid_bombs as lb
from . import lucid_run as lr
from . import lucid_arms as la
from . import point_encoder as pe

# =============================================================== identity §8.1
EXTENSION = "paper3_p2_infoloss_2026-11-09"
MODES = ("dry-run", "preflight", "reproduce", "new-models", "conditions", "p2",
         "x3p", "report")

ROOT = Path(__file__).resolve().parents[2]
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_p2_20261109")
RESULTS_DIR = "results/jepa/20261109_p2"
JOBS_DIR = Path("/ceph/dwong/jepa/jobs")
GUARD_JSON = JOBS_DIR / "guard_deepthought2.json"
JOBS_RECORD = EXT_ROOT / "jobs.jsonl"

PILOT_EXT_ROOT = er.EXT_ROOT
SHORTCUT_EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_shortcut_20261107")
CONFIG_REL = ef.CONFIG_REL
GENERATOR_REL = ef.GEN_REL
ARMS_REL = ef.ARMS_REL
GEOMETRY_DEFAULT = ef.GEOMETRY_DEFAULT
DATA_ROOT_DEFAULT = ef.DATA_ROOT_DEFAULT
S005, S030 = ef.S005, ef.S030

RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-09_P2_INFOLOSS.md"
RUNBOOK_SHA256 = "0dbabf1698f0deccaf1076c5a03a59c8bb9196a310cba2b1b0c741af5a56c57f"
ACCEPTANCE_REL = "docs/jepa/ACCEPTANCE_2026-11-09_P2_INFOLOSS.json"
REVIEW_REL = "docs/jepa/REVIEW_2026-11-09_P2_INFOLOSS.md"
DECISIONS_REL = "docs/jepa/DECISIONS.md"
STATE_REL = "docs/jepa/STATE.md"
ACCEPTANCE_KEYS = ("runbook_sha256", "impl_commit", "review_path", "review_verdict",
                   "lead", "date")

#: §8.2: Part A's list plus ``ext_p2``, ``ext_repair``, ``ext_shortcut`` and the
#: P2 CLI (``ext_repair`` is already in the Part A list; it is listed once).
P_FILES = tuple(er.P_FILES) + (
    "src/jepa_noise/ext_p2.py",
    "src/jepa_noise/ext_repair.py",
    "src/jepa_noise/ext_shortcut.py",
    "scripts/jepa_ext_p2.py",
    RUNBOOK_REL,
)
P_DIRS = er.P_DIRS
FORCED_IMPORT_EXCEPTIONS = ef.FORCED_IMPORT_EXCEPTIONS

# ============================================================ models §2.2
CORE_MODELS = tuple(er.CORE_MODELS) + ("target_posfree9000", "target_posfree9001")
NEW_MODELS = ("target_posfree9000", "target_posfree9001")
OPTIONAL_MODELS = tuple(er.OPTIONAL_MODELS)
#: Part B R-2 runs: arm/cell/seed and the recorded checkpoint of each run.
NEW_MODEL_SPECS = {
    "target_posfree9000": {
        "arm": "sigreg_jepa", "split": "within", "seed": 9000, "data_setting": S005,
        "run_id": "dev_s005_within__sigreg_jepa__s9000",
        "sha256": "3bb04c30dd51a90602e632f20f553b635c211a73f3bb048778ef5b2ad45b610c",
        "source": ("/ceph/dwong/jepa/returns/ext_shortcut_20261107/train/runs/"
                   "dev_s005_within__sigreg_jepa__s9000/final.pt"),
        "full_input_vertex_m": 0.12600451326211795,
    },
    "target_posfree9001": {
        "arm": "sigreg_jepa", "split": "within", "seed": 9001, "data_setting": S005,
        "run_id": "dev_s005_within__sigreg_jepa__s9001",
        "sha256": "e36a437d77714985e59009528b6e73ba092e213b7b209c630b970f275ddc71a2",
        "source": ("/ceph/dwong/jepa/returns/ext_shortcut_20261107/train/runs/"
                   "dev_s005_within__sigreg_jepa__s9001/final.pt"),
        "full_input_vertex_m": 0.14054395507374678,
    },
}
X0B_MODELS = ("mae_within_trigger9000", "fix9000", "init9000") + NEW_MODELS
#: §4 X0-b N vertex targets (the three old models from Part A X0-b, ``fix9000``
#: from ``results/jepa/20261107_shortcut/reproduction.json`` = 0.219258; the two
#: new models from ``readings.json`` r2_b/r2_c).
X0B_EXPECTED = {
    "mae_within_trigger9000": 0.104916,
    "fix9000": 0.219258,
    "init9000": 0.160072,
    "target_posfree9000": 0.12600451326211795,
    "target_posfree9001": 0.14054395507374678,
}

# ============================================================ drifts §2.3
SEED = 20261109
MODULE_NAMESPACE = "paper3/p2/dead_modules"
PMT_NAMESPACE = "paper3/p2/dead_pmts"
PERM_NAMESPACE = "paper3/p2/perm"
X3P_NAMESPACE = "paper3/p2/x3p"
#: Condition table of §2.3 (ordered; the §8.5 GPU projection is 7 conditions).
CONDITIONS = ("T0.5", "T1.0", "T1.0w", "K5", "K15", "K30", "P10")
CONDITION_SPECS = {
    "T0.5": {"kind": "threshold", "tau": 0.5, "cal": 0.5},
    "T1.0": {"kind": "threshold", "tau": 1.0, "cal": 1.0},
    "T1.0w": {"kind": "threshold", "tau": 1.0, "cal": 0.25},
    "K5": {"kind": "dead", "fraction": 0.05, "unit": "module"},
    "K15": {"kind": "dead", "fraction": 0.15, "unit": "module"},
    "K30": {"kind": "dead", "fraction": 0.30, "unit": "module"},
    "P10": {"kind": "dead", "fraction": 0.10, "unit": "pmt"},
}
#: R1 (§3, T1.0w only): re-apply the walk correction at c = tau on the T1.0w hits.
R1_NAME = "R1"

#: §5 fixed classification thresholds (never changed after outcomes).
DRIFT_COST_MIN_M = 0.003
REFIT_DESTROYED_M = 0.003
P2C_RECOVERY_TOL_M = 0.002
P2E_RHO_SUPPORT = 0.5
P2E_P_SUPPORT = 0.05
P2E_RHO_REFUTE = 0.1
P2E_PERMUTATIONS = 10_000
P2F_SHARE = 0.8
#: §6 X3-P thresholds (Part A §7.2, unchanged).
X3P_PRICE_HIGH = 1.20
X3P_PRICE_NONE = 1.05
X3P_ROBUST_VERTEX_M = 0.002
X3P_ROBUST_VERTEX_CI_M = 0.003
X3P_ROBUST_ENERGY_REL = 0.05
X3P_RANKS = (1, 3, 10)
X3P_CONTROLS = 20
X3P_RAW_FEATURE_PRICE = 3.56
X3P_MODELS = (tuple(er.MAE_MODELS) + tuple(er.CORE_MODELS[4:8])
              + ("frozen_target9000", "frozen_target9001")
              + ("target_posfree9000", "target_posfree9001"))

# ============================================================ resources §7
GPU_CAP_H = 1.3
CPU_CAP_H = 8.0
S1_CEILING_H = 160.0
PHOTON_SET_WALL_S = 13.0
PROJECTION_FACTOR = 1.3
ENTRIES_FLOOR = 30000
ENTRIES_PROJECTION = 600
LEDGER_CONSISTENCY_H = 1e-3
IDENTITY_TOLERANCE_NS = 1e-9
IDENTITY_EVENTS = 200
IDENTITY_CHARGE = 0.25

#: §8.5 execution order (bookkeeping).
EXECUTION_ORDER = ("preflight", "reproduce", "new-models", "conditions", "p2",
                   "x3p", "report")

# ================================================================= gate §8.2
def _resolve_root(root=None) -> Path:
    return Path(root).resolve() if root is not None else ROOT


def p_paths(root=None) -> list:
    return list(P_FILES) + list(P_DIRS)


def in_p(relative: str) -> bool:
    relative = str(relative)
    if relative in P_FILES:
        return True
    return any(relative.startswith(prefix) for prefix in P_DIRS)


def _git(root, *args, timeout=60):
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                             text=True, timeout=timeout)
        return out.returncode, out.stdout, out.stderr
    except Exception as exc:  # pragma: no cover - host without git
        return 1, "", str(exc)


def section_zero(text: str) -> str:
    return ef.section_zero(text)


def read_json(path, default=None):
    return ef.read_json(path, default)


def write_json(path, payload, *, overwrite=True) -> Path:
    return er.write_json(path, payload, overwrite=overwrite)


def acceptance_record(root=None) -> dict | None:
    path = _resolve_root(root) / ACCEPTANCE_REL
    payload = read_json(path, None)
    if payload is None and path.is_file():
        return {"invalid": f"{path} is not valid JSON"}
    return payload


def import_closure(root=None) -> dict:
    """The executed-import closure over ``<P>`` (§8.2, as Part A §9.2)."""
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
    return {"registered": False, "extension": EXTENSION, "check": "import closure §8.2",
            "root": str(root), "checked_modules": int(checked),
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

    # (a) the committed runbook is the accepted one, at the fixed sha256.
    expected = (record or {}).get("runbook_sha256")
    a_pass = bool(computed is not None and expected is not None
                  and computed == expected and computed == RUNBOOK_SHA256)
    checks["a"] = {"pass": a_pass, "runbook": str(runbook_path),
                   "runbook_sha256": computed, "expected_sha256": RUNBOOK_SHA256,
                   "record_sha256": expected, "pending_acceptance": bool(pending)}
    if not a_pass and not pending:
        problems.append("(a) the committed runbook sha256 does not match the "
                        "acceptance record / the fixed P2 sha256")

    # (b) the acceptance record schema, review path and PASS verdict.
    if record is None:
        checks["b"] = {"pass": False, "status": "gate_pending_acceptance",
                       "reason": f"{ACCEPTANCE_REL} does not exist yet"}
    elif "invalid" in record:
        checks["b"] = {"pass": False, "status": "invalid", "reason": record["invalid"]}
        problems.append("(b) the acceptance record is not valid JSON")
    else:
        missing = [key for key in ACCEPTANCE_KEYS if key not in record]
        review = root / str(record.get("review_path") or REVIEW_REL)
        b_pass = bool(not missing and record.get("review_verdict") == "PASS"
                      and record.get("review_path") == REVIEW_REL and review.is_file())
        checks["b"] = {"pass": b_pass, "status": "PASS" if b_pass else "denied",
                       "record": record, "missing_keys": missing,
                       "review_path": record.get("review_path"),
                       "review_exists": bool(review.is_file()),
                       "review_verdict": record.get("review_verdict")}
        if not b_pass:
            problems.append("(b) the acceptance record is incomplete, its review_path "
                            "is not the P2 review, or review_verdict is not PASS")

    # (c) the working tree over <P> is the implementation commit, no untracked <P>.
    impl_commit = (record or {}).get("impl_commit") if record else None
    if impl_commit is None:
        code, head, _ = _git(root, "rev-parse", "HEAD")
        impl_commit = head.strip() if code == 0 else None
    paths = p_paths(root)
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
                        f"(untracked: {untracked[:3]})")

    authorised = bool(not problems and not pending)
    status = ("gate_pending_acceptance" if pending else
              ("granted" if authorised else "denied"))
    return {"authorised": authorised, "status": status, "mode": mode,
            "extension": EXTENSION, "checks": checks, "problems": problems,
            "acceptance": record, "runbook_sha256": computed}


def assert_authorised(mode, *, root=None, dry_run=None, log=None) -> dict:
    """Hard gate: every failure raises ``SystemExit(2)`` outside dry-run."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    dry = bool(dry_run) if dry_run is not None else (mode == "dry-run")
    result = _gate_checks(mode, root=root)
    if dry:
        return result
    if not result["authorised"]:
        for problem in result["problems"]:
            print(f"[p2] authorisation gate: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return result


def assert_closure(mode, root) -> dict:
    closure = import_closure(root)
    if not closure["pass"]:
        print(f"[p2] {mode}: import-closure STOP: {closure['violations']} "
              f"(outputs quarantined, §8.2)", file=sys.stderr)
        raise SystemExit(2)
    return closure


# =========================================================== data helpers §2.1
def open_probe_sets(config, gen, geometry, data_root, *, setting=S005, log=print) -> dict:
    """Open the pinned probe sets read-only through ``ext_followup`` (Part A §2.1)."""
    return er.open_probe_sets(config, gen, geometry, data_root, setting=setting, log=log)


# ===================================================== pinned-data RAM mirror
def mirror_base(base=None) -> Path:
    """Where the GPU modes stage a verified read-only mirror of the pinned sets.

    ``JEPA_P2_MIRROR`` wins (the operator points it at RAM, e.g. ``/dev/shm``);
    otherwise the job's ``TMPDIR`` (``/work/dwong/...``) is used.  The mirrored
    bytes are copies only: the pinned data under ``/ceph`` is never written and
    every mirrored run's ``SHA256SUMS`` is verified against the pinned manifest.
    """
    if base is None:
        base = (os.environ.get("JEPA_P2_MIRROR") or os.environ.get("TMPDIR")
                or "/work/dwong")
    return Path(base) / "p2_data_mirror"


def _mirror_ok(dataset, root) -> bool:
    marker = root / "mirror.json"
    if not marker.is_file():
        return False
    payload = read_json(marker, {}) or {}
    if payload.get("source_manifest_sha256") != dataset.manifest_sha256():
        return False
    for row in dataset.runs:
        run = root / "runs" / row["run_dir"]
        if not (run / "hits.npz").is_file() or not (run / "events.npz").is_file():
            return False
        try:
            lb.verify_sha256sums(run)
        except Exception:
            return False
    return True


def mirror_root(base=None) -> Path:
    """The mirror root directory: ``base`` verbatim, else ``mirror_base()``."""
    return Path(base) if base is not None else mirror_base()


def mirror_dataset(dataset, *, base=None, stage=False, log=print):
    """A verified read-only copy of one pinned probe set for fast GPU-job reads.

    The copy is staged once by the CPU ``preflight`` mode (``stage=True``) into a
    declared TMPDIR/``JEPA_P2_MIRROR`` tree; the GPU modes only ever *verify* it
    (``stage=False``) and fail closed when it is absent, so no GPU job's booked
    wall includes staging I/O.
    """
    root = mirror_root(base) / dataset.key
    if _mirror_ok(dataset, root):
        return dataclasses.replace(dataset, directory=root)
    if not stage:
        raise RuntimeError(f"the pinned-data mirror for {dataset.key} is missing or "
                           f"unverified at {root}; run the CPU preflight staging step")
    root.mkdir(parents=True, exist_ok=True)
    for row in dataset.runs:
        source = dataset.directory / "runs" / row["run_dir"]
        target = root / "runs" / row["run_dir"]
        target.mkdir(parents=True, exist_ok=True)
        for name in ("hits.npz", "events.npz", "twin_hits.npz", "manifest.json",
                     "SHA256SUMS"):
            if (source / name).is_file():
                shutil.copyfile(source / name, target / name)
        lb.verify_sha256sums(target)
    for name in ("manifest.json", "SHA256SUMS"):
        if (dataset.directory / name).is_file():
            shutil.copyfile(dataset.directory / name, root / name)
    write_json(root / "mirror.json",
               {"source": str(dataset.directory),
                "source_manifest_sha256": dataset.manifest_sha256(),
                "n_runs": len(dataset.runs), "verified": True,
                "note": "read-only copy of the pinned set for fast job reads"})
    log(f"[p2] mirrored {dataset.key} -> {root} ({len(dataset.runs)} runs, verified)")
    return dataclasses.replace(dataset, directory=root)


def mirror_probe_sets(probes, *, base=None, stage=False, log=print) -> dict:
    return {role: mirror_dataset(dataset, base=base, stage=stage, log=log)
            for role, dataset in probes.items()}


# =========================================================== geometry helpers
def module_of_pmt(geometry) -> np.ndarray:
    return er.module_of_pmt(geometry)


def module_names(geometry) -> int:
    return int(geometry.n_modules)


def n_pmt(geometry) -> int:
    return int(geometry.n_pmt)


# ======================================================== transforms §2.3
@dataclasses.dataclass(frozen=True)
class DriftCondition:
    """One §2.3 hit-set transform.

    ``kind`` is ``threshold`` (keep hits that cross the new discriminator, shift
    each kept time by ``leading_edge_time(q, tau) - leading_edge_time(q, cal)``),
    ``dead`` (delete every hit of a dead PMT) or ``r1`` (the §3 T1.0w input
    repair: the T1.0w transform followed by subtracting the same walk residual,
    re-triggering and cutting).  Dead sets are keyed per run from ``keyed_rng``
    and are nested across the three K levels; the permutation is the same for
    every K level of one run because it is drawn once from the run id.
    """

    name: str
    kind: str
    tau: float | None = None
    cal: float | None = None
    fraction: float | None = None
    unit: str | None = None
    namespace: str | None = None

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "tau": self.tau,
                "cal": self.cal, "fraction": self.fraction, "unit": self.unit,
                "namespace": self.namespace}

    def dead_pmts(self, run_id, geometry) -> np.ndarray | None:
        """The dead-PMT mask of this condition on one run (§2.3 K/P)."""
        if self.kind != "dead":
            return None
        if self.unit == "pmt":
            count_total = int(geometry.n_pmt)
        elif self.unit == "module":
            count_total = int(geometry.n_modules)
        else:
            raise ValueError(f"unknown dead unit {self.unit!r}")
        permutation = lb.keyed_rng(SEED, str(self.namespace), str(run_id)).permutation(
            count_total)
        count = int(math.ceil(float(self.fraction) * count_total))
        chosen = np.sort(permutation[:count]).astype(np.int64)
        if self.unit == "module":
            return np.isin(np.asarray(geometry.module_of_pmt, np.int64), chosen)
        return np.isin(np.arange(int(geometry.n_pmt), dtype=np.int64), chosen)

    def dead_fraction_by_module(self, run_id, geometry) -> np.ndarray:
        """The fraction of each module's PMTs that this condition kills."""
        dead = self.dead_pmts(run_id, geometry)
        n_modules = int(geometry.n_modules)
        if dead is None:
            return np.zeros(n_modules)
        module = np.asarray(geometry.module_of_pmt, np.int64)
        killed = np.bincount(module, weights=dead.astype(float), minlength=n_modules)
        per_module = np.bincount(module, minlength=n_modules).astype(float)
        return killed / np.where(per_module > 0, per_module, 1.0)


def build_drift_conditions() -> dict:
    """The seven §2.3 conditions, in the registered order."""
    out = {}
    for name in CONDITIONS:
        spec = CONDITION_SPECS[name]
        if spec["kind"] == "threshold":
            out[name] = DriftCondition(name, "threshold", tau=float(spec["tau"]),
                                       cal=float(spec["cal"]))
        else:
            out[name] = DriftCondition(name, "dead", fraction=float(spec["fraction"]),
                                       unit=str(spec["unit"]),
                                       namespace=(MODULE_NAMESPACE
                                                  if spec["unit"] == "module"
                                                  else PMT_NAMESPACE))
    out[R1_NAME] = DriftCondition(R1_NAME, "r1", tau=1.0, cal=0.25)
    return out


def identity_condition() -> DriftCondition:
    """T(0.25, 0.25): the §4 identity transform of the stored hits."""
    return DriftCondition("IDENTITY", "threshold", tau=IDENTITY_CHARGE,
                          cal=IDENTITY_CHARGE)


def pmt_rule():
    """The imported trigger rule (never reimplemented; §2.3)."""
    return lb.pmt_time_rule()


def threshold_keep_and_delta(charge, tau, cal, cfg):
    """``(keep, delta)`` for one threshold: keep = isfinite(le(q, tau)); delta = le(tau)-le(cal)."""
    mv = float(cfg.mv_per_pe)
    leading = lb.leading_edge_time
    new = leading(np.asarray(charge, float), float(tau) * mv, mv_per_pe=mv)
    reference = leading(np.asarray(charge, float), float(cal) * mv, mv_per_pe=mv)
    keep = np.isfinite(new)
    delta = np.zeros(np.asarray(charge).shape, float)
    finite = keep & np.isfinite(reference)
    delta[finite] = new[finite] - reference[finite]
    return keep, delta


def _retrigger_and_cut(time_ns, rule):
    """Re-trigger on the kept times, then cut with the imported prompt window.

    ``time_ns`` are relative to the **old** trigger.  ``nhits_trigger_time`` is
    shift-equivariant, so re-deriving the trigger on these times returns the new
    trigger **relative to the old one**; the returned times are relative to the
    new trigger.  ``None`` means the event did not trigger and is dropped (§2.3).
    """
    nhits_trigger_time, prompt_window_mask, n_hits, window_ns, _prompt = rule
    time_ns = np.asarray(time_ns, float)
    if time_ns.size == 0:
        return None
    trigger = nhits_trigger_time(time_ns, n=int(n_hits), window_ns=float(window_ns))
    if trigger is None or not np.isfinite(trigger):
        return None
    keep = np.asarray(prompt_window_mask(time_ns, float(trigger)), bool)
    if not keep.any():
        return None
    return {"time_ns": time_ns[keep] - float(trigger), "keep": keep,
            "trig_rel": float(trigger)}


def transform_event(condition, run_id, pmt_id, charge_pe, time_ns, geometry, cfg, rule):
    """Apply one §2.3 transform to one event's stored hits (pure; never mutates).

    Returns ``None`` when the event is dropped (no trigger after the transform),
    else a dict with the surviving ``pmt_id``/``charge_pe``/``time_ns`` and the
    window-edge bookkeeping: ``trig_rel`` is the trigger shift used by the §2.3
    item 3 exact-subset rule (for R1 it is the T1.0w stage's shift), and
    ``window_edge`` is ``trig_rel > 0``.
    """
    pmt = np.asarray(pmt_id, np.int64)
    charge = np.asarray(charge_pe, float)
    time = np.asarray(time_ns, float)
    n_before = int(pmt.size)
    trig_rel, trig_final = 0.0, 0.0
    if condition.kind in ("threshold", "r1"):
        keep, delta = threshold_keep_and_delta(charge, condition.tau, condition.cal, cfg)
        pmt, charge, time = pmt[keep], charge[keep], time[keep] + delta[keep]
    elif condition.kind == "dead":
        dead = condition.dead_pmts(run_id, geometry)
        if dead is None:
            raise ValueError("dead condition without a dead mask")
        alive = ~np.asarray(dead, bool)[pmt]
        pmt, charge, time = pmt[alive], charge[alive], time[alive]
    else:
        raise ValueError(f"unknown transform kind {condition.kind!r}")
    cut = _retrigger_and_cut(time, rule)
    if cut is None:
        return None
    pmt, charge, time = pmt[cut["keep"]], charge[cut["keep"]], cut["time_ns"]
    trig_rel = trig_final = float(cut["trig_rel"])
    if condition.kind == "r1":
        # §3 R1: the T1.0w stage's shift is the one that can lose early hits, so
        # the exact-subset rule of §2.3 item 3 reads that stage.
        trig_rel = trig_final
        _keep2, delta2 = threshold_keep_and_delta(charge, condition.tau,
                                                  condition.cal, cfg)
        time = time - delta2
        cut2 = _retrigger_and_cut(time, rule)
        if cut2 is None:
            return None
        pmt, charge, time = pmt[cut2["keep"]], charge[cut2["keep"]], cut2["time_ns"]
        trig_final = float(cut2["trig_rel"])
    if pmt.size == 0:
        return None
    if pmt.size > 1 and not np.all(np.diff(pmt) > 0):
        raise RuntimeError("transform produced unsorted or duplicate PMT ids")
    if not np.isfinite(charge).all() or np.any(charge < 0) or not np.isfinite(time).all():
        raise RuntimeError("transform produced non-finite charge/time")
    return {"pmt_id": pmt, "charge_pe": charge, "time_ns": time,
            "trig_rel": float(trig_rel), "trig_final": float(trig_final),
            "n_before": n_before, "n_after": int(pmt.size),
            "window_edge": bool(trig_rel > 0.0), "dropped": False}


def transform_store(condition, run_id, store, geometry, cfg, *, rule=None) -> tuple:
    """Transform one run's ``SparseHitStore``; returns ``(new_store, info)``."""
    rule = pmt_rule() if rule is None else rule
    offsets = np.asarray(store.offsets, np.int64)
    pmt_all = np.asarray(store.pmt_id)
    charge_all = np.asarray(store.charge_pe)
    time_all = np.asarray(store.time_ns)
    event_ids = np.asarray(store.event_ids, np.int64)
    kept_pmt, kept_charge, kept_time, kept_ids = [], [], [], []
    out_offsets = [0]
    info = {"event_ids": [], "run_ids": [], "trig_rel": [], "trig_final": [],
            "window_edge": [], "dropped": [], "n_before": [], "n_after": []}
    for index, event_id in enumerate(event_ids):
        span = slice(int(offsets[index]), int(offsets[index + 1]))
        result = transform_event(condition, run_id, pmt_all[span], charge_all[span],
                                 time_all[span], geometry, cfg, rule)
        info["event_ids"].append(int(event_id))
        info["run_ids"].append(str(run_id))
        if result is None:
            info["trig_rel"].append(np.nan)
            info["trig_final"].append(np.nan)
            info["window_edge"].append(False)
            info["dropped"].append(True)
            info["n_before"].append(int(offsets[index + 1] - offsets[index]))
            info["n_after"].append(0)
            continue
        info["trig_rel"].append(result["trig_rel"])
        info["trig_final"].append(result["trig_final"])
        info["window_edge"].append(result["window_edge"])
        info["dropped"].append(False)
        info["n_before"].append(result["n_before"])
        info["n_after"].append(result["n_after"])
        kept_ids.append(int(event_id))
        kept_pmt.append(result["pmt_id"])
        kept_charge.append(result["charge_pe"].astype(np.float32))
        kept_time.append(result["time_ns"].astype(np.float32))
        out_offsets.append(out_offsets[-1] + int(result["pmt_id"].size))
    if kept_ids:
        pmt = np.concatenate(kept_pmt).astype(np.int32)
        charge = np.concatenate(kept_charge).astype(np.float32)
        time = np.concatenate(kept_time).astype(np.float32)
    else:
        pmt = np.empty(0, np.int32); charge = np.empty(0, np.float32)
        time = np.empty(0, np.float32)
    new_store = lb.SparseHitStore(np.asarray(kept_ids, np.int64),
                                  np.asarray(out_offsets, np.int64), pmt, charge, time,
                                  store.readout_label)
    for name in ("event_ids", "trig_rel", "trig_final", "window_edge", "dropped",
                 "n_before", "n_after"):
        info[name] = np.asarray(info[name])
    info["run_ids"] = np.asarray(info["run_ids"]).astype(str)
    return new_store, info


@contextlib.contextmanager
def hit_set_context(dataset, condition, geometry, cfg, *, memo=None):
    """The §2.4 hit-**set** transform context (Part A §2.5 extended).

    Both hit-read bindings are replaced for the duration and restored in a
    ``finally``; every read must be a ``hits.npz`` under this dataset's
    ``runs/`` and the wrapper fails closed otherwise.  Transformed stores are
    memoised per ``(run_dir, condition)`` so one encoding pass reads each run
    once.  The input stores are never mutated.
    """
    if not isinstance(condition, DriftCondition):
        raise TypeError("hit_set_context needs a DriftCondition")
    directory = Path(dataset.directory).resolve()
    by_dir = {str(row["run_dir"]): str(row["run_id"]) for row in dataset.runs}
    original_pc = pe.read_sparse_hits
    original_lb = lb.read_sparse_hits
    memo = {} if memo is None else memo
    rule = pmt_rule()

    def wrapper(path):
        requested = Path(path)
        if requested.is_dir():
            requested = requested / "hits.npz"
        if requested.name != "hits.npz":
            raise RuntimeError(f"hit_set_context: refusing non-hit path {requested}")
        run_dir = requested.parent.resolve()
        if run_dir.parent != directory / "runs":
            raise RuntimeError(f"hit_set_context: {run_dir} is outside the "
                               f"{condition.name} set {directory}")
        run_id = by_dir.get(run_dir.name)
        if run_id is None:
            raise RuntimeError(f"hit_set_context: unknown run dir {run_dir.name!r}")
        key = (str(run_dir.name), condition.name)
        if key not in memo:
            store = original_lb(requested)
            new_store, info = transform_store(condition, run_id, store, geometry, cfg,
                                              rule=rule)
            memo[key] = (new_store, info)
        return memo[key][0]

    pe.read_sparse_hits = wrapper
    lb.read_sparse_hits = wrapper
    try:
        yield memo
    finally:
        pe.read_sparse_hits = original_pc
        lb.read_sparse_hits = original_lb


# ----------------------------------------------------- transforms appendix A
def transform_info(dataset, condition, geometry, cfg) -> dict:
    """Per-event transform bookkeeping of one (condition, dataset role).

    Model-independent, so it is computed once per condition and role and stored
    under ``<ext>/transforms``.  It reports the trigger efficiency (§2.3), the
    ``trig'`` distribution and the window-edge counts (§2.3 item 3).
    """
    rule = pmt_rule()
    events, run_ids, trig, trig_final, edge, dropped, before, after = ([] for _ in range(8))
    for row in dataset.runs:
        run_id = str(row["run_id"])
        store = lb.read_sparse_hits(Path(dataset.directory) / "runs"
                                    / row["run_dir"] / "hits.npz")
        _new, info = transform_store(condition, run_id, store, geometry, cfg, rule=rule)
        events.append(info["event_ids"]); run_ids.append(info["run_ids"])
        trig.append(info["trig_rel"]); trig_final.append(info["trig_final"])
        edge.append(info["window_edge"]); dropped.append(info["dropped"])
        before.append(info["n_before"]); after.append(info["n_after"])
    out = {"event_ids": np.concatenate(events) if events else np.zeros(0, np.int64),
           "run_ids": np.concatenate(run_ids).astype(str) if run_ids else np.zeros(0, str),
           "trig_rel": np.concatenate(trig) if trig else np.zeros(0),
           "trig_final": np.concatenate(trig_final) if trig_final else np.zeros(0),
           "window_edge": np.concatenate(edge) if edge else np.zeros(0, bool),
           "dropped": np.concatenate(dropped) if dropped else np.zeros(0, bool),
           "n_before": np.concatenate(before) if before else np.zeros(0, np.int64),
           "n_after": np.concatenate(after) if after else np.zeros(0, np.int64)}
    out["condition"] = condition.as_dict()
    out["n_events"] = int(out["event_ids"].size)
    out["n_triggered"] = int(out["n_events"] - out["dropped"].sum())
    out["trigger_efficiency"] = (None if out["n_events"] == 0
                                 else float(out["n_triggered"] / out["n_events"]))
    finite = out["trig_rel"][~out["dropped"]]
    out["trig_rel_distribution"] = {
        "n": int(finite.size),
        "min": None if finite.size == 0 else float(finite.min()),
        "max": None if finite.size == 0 else float(finite.max()),
        "n_positive": int(np.count_nonzero(finite > 0.0)),
        "n_zero": int(np.count_nonzero(finite == 0.0)),
        "n_negative": int(np.count_nonzero(finite < 0.0)),
        "quantiles": ({} if finite.size == 0 else
                      {f"q{q:g}": float(np.quantile(finite, q / 100.0))
                       for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}),
    }
    out["n_window_edge"] = int(np.count_nonzero(out["window_edge"]))
    return out


def transform_manifest_path(ext_root) -> Path:
    return Path(ext_root) / "transforms" / "manifest.json"


def write_transform_info(ext_root, split, info) -> dict:
    """Write one condition/role transform bookkeeping file (write-once, sha sidecar)."""
    root = Path(ext_root) / "transforms"
    root.mkdir(parents=True, exist_ok=True)
    key = f"{info['condition']['name']}__{split}"
    path = root / f"{key}.npz"
    manifest = read_json(transform_manifest_path(ext_root), {}) or {}
    if path.is_file() and manifest.get(key, {}).get("sha256") == lr.sha256_file(path):
        return {"key": key, "path": str(path), "status": "reused",
                "sha256": manifest[key]["sha256"]}
    arrays = {name: info[name] for name in
              ("event_ids", "run_ids", "trig_rel", "trig_final", "window_edge",
               "dropped", "n_before", "n_after")}
    np.savez_compressed(path, **arrays)
    digest = lr.sha256_file(path)
    manifest[key] = {"sha256": digest, "path": str(path), "n_events": info["n_events"],
                     "trigger_efficiency": info["trigger_efficiency"],
                     "n_window_edge": info["n_window_edge"]}
    write_json(transform_manifest_path(ext_root), manifest)
    return {"key": key, "path": str(path), "status": "written", "sha256": digest}


def load_transform_info(ext_root, split, condition_name) -> dict:
    root = Path(ext_root) / "transforms"
    key = f"{condition_name}__{split}"
    path = root / f"{key}.npz"
    manifest = read_json(transform_manifest_path(ext_root), {}) or {}
    row = manifest.get(key) or {}
    if not path.is_file() or not row.get("sha256"):
        raise RuntimeError(f"transform bookkeeping missing for {key} in {root}")
    digest = lr.sha256_file(path)
    if digest != row["sha256"]:
        raise RuntimeError(f"transform bookkeeping {path}: sha256 mismatch")
    with np.load(path, allow_pickle=False) as data:
        out = {name: np.asarray(data[name]) for name in data.files}
    out["condition"] = condition_name
    out["split"] = split
    return out


# ========================================================= inventory §2.2
def _new_row(label, spec, *, status, sha256=None, actual_sha256=None) -> dict:
    """One inventory row for a Part B R-2 model (same schema as ``ef._row``)."""
    return {"label": label, "family": "joint_embedding_working", "arm": spec["arm"],
            "cell": "dev_s005_within", "split": spec["split"], "seed": int(spec["seed"]),
            "beta": None, "encoder": None, "data_key": str(ef.TRAIN_KEY_S005),
            "source": str(spec["source"]), "sha256": sha256,
            "actual_sha256": actual_sha256, "data_setting": float(spec["data_setting"]),
            "status": status,
            "eval": {"zero_geometry": False, "median_wrapper": False,
                     "encoder_variant": None, "anchor": False,
                     "time_convention": ef.A4_TOKEN_TRIGGER},
            "run_id": spec["run_id"],
            "full_input_vertex_m": spec["full_input_vertex_m"]}


def new_model_rows(root=None, *, hash_files=True) -> dict:
    """The two §2.2 Part B rows, re-verified against their recorded sha256."""
    rows = {}
    for label, spec in NEW_MODEL_SPECS.items():
        source = Path(spec["source"])
        if not source.is_file():
            rows[label] = _new_row(label, spec, status="unavailable", sha256=spec["sha256"])
            rows[label]["reason"] = "checkpoint file is absent"
            continue
        if not hash_files:
            rows[label] = _new_row(label, spec, status="unverified (dry-run)",
                                   sha256=spec["sha256"])
            continue
        actual = lr.sha256_file(source)
        if actual != spec["sha256"]:
            rows[label] = _new_row(label, spec, status="unavailable",
                                   sha256=spec["sha256"], actual_sha256=actual)
            rows[label]["reason"] = f"sha256 {actual} != pinned {spec['sha256']}"
            continue
        rows[label] = _new_row(label, spec, status="available", sha256=spec["sha256"],
                               actual_sha256=actual)
    return rows


def p2_inventory(root=None, *, hash_files=True, include_optional=False) -> dict:
    """The §2.2 core inventory: 13 Part A rows plus the two new models."""
    root = _resolve_root(root)
    base = er.part_a_inventory(root, hash_files=hash_files,
                               include_optional=include_optional,
                               log=lambda *a, **k: None)
    rows = dict(base["rows"])
    rows.update(new_model_rows(root, hash_files=hash_files))
    unavailable = list(base["unavailable"]) + [
        {"label": label, "reason": rows[label].get("reason", "")}
        for label in NEW_MODELS
        if rows[label].get("status") not in ("available", "unverified (dry-run)")]
    return {"registered": False, "extension": EXTENSION, "check": "§2.2 inventory",
            "root": str(root), "hash_files": bool(hash_files), "rows": rows,
            "unavailable": unavailable,
            "counts": {"labels": len(rows),
                       "available": sum(1 for row in rows.values()
                                        if row.get("status") in
                                        ("available", "unverified (dry-run)")),
                       "unavailable": len(unavailable), "core": len(CORE_MODELS),
                       "new": len(NEW_MODELS), "optional": len(OPTIONAL_MODELS)}}


def model_row(inventory, label) -> dict:
    row = (inventory.get("rows") or {}).get(label)
    if row is None or row.get("status") not in ("available", "unverified (dry-run)"):
        raise RuntimeError(f"{label}: inventory row is not available")
    return row


def split_of(row) -> str:
    return str(row.get("split", "within"))


def build_model(row, *, config, gen, arms_config, geometry, data_root, device="cpu",
                log=print):
    """Build one inventory row's encoder (checkpoint load, eval mode; no optimiser)."""
    with ef.item_eval_context(row):
        arm = ef.build_item_arm(row, config=config, gen=gen, arms_config=arms_config,
                                geometry=geometry, data_root=data_root, device=device,
                                log=log)
    return getattr(arm, "jepa", arm)


# ========================================================= embedding cache §3.2
def embedding_cache(label) -> er.EmbeddingCache:
    return er.EmbeddingCache(Path(EXT_ROOT) / "x1" / str(label))


def cache_rows(cache, key) -> dict | None:
    """Read one sha-verified embedding cache entry, or ``None`` when absent."""
    if not cache.has(key):
        return None
    return cache.fetch(key, lambda: {})


def _cache_available(model_rows, cache, roles=("train", "test")) -> dict:
    out = {}
    for role in roles:
        out[role] = bool(cache.has(er.embedding_cache_key(role, "N")))
    return out


def stream_drift_embeddings(model, dataset, condition, *, split, geometry, cfg, config,
                            device, cache, cache_key, log=print) -> dict:
    """Photon embeddings of one N probe set under one hit-set condition."""
    chunk = int((config.get("evaluation") or {}).get("probe_chunk_events", 64))
    mask_seed = int(config["views"]["mask_seed"])

    def producer():
        with hit_set_context(dataset, condition, geometry, cfg):
            return ef.stream_event_rows(model, dataset, geometry, split=split,
                                        mask_seed=mask_seed, device=device,
                                        chunk_events=chunk, max_events=None, log=log)

    rows = cache.fetch(cache_key, producer, log=log)
    return {name: rows[name] for name in er.EmbeddingCache.ARRAY_NAMES if name in rows} | {
        name: rows[name] for name in ("cache_status", "cache_sha256") if name in rows}


# ============================================================ budget/ledger §7
def job_hours(ledger_path=None) -> dict:
    path = Path(JOBS_RECORD if ledger_path is None else ledger_path)
    total = {"gpu": 0.0, "cpu": 0.0, "jobs": 0, "path": str(path)}
    if not path.is_file():
        return total
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        kind = str(row.get("kind", "")).lower()
        if kind in ("gpu", "cpu"):
            total[kind] += float(row.get("hours") or 0.0)
            total["jobs"] += 1
    return total


def record_job(mode, kind, wall_seconds) -> dict:
    kind = str(kind).lower()
    hours = float(wall_seconds) / 3600.0
    row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "mode": str(mode), "kind": kind, "wall_seconds": float(wall_seconds),
           "hours": hours}
    path = Path(JOBS_RECORD)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def assert_budget(kind, projected_hours) -> dict:
    """Refuse a job that would cross the §7 Stage 1 cap (1.3 GPU / 8 CPU hours)."""
    kind = str(kind).lower()
    cap = GPU_CAP_H if kind == "gpu" else CPU_CAP_H
    used = float(job_hours().get(kind, 0.0))
    projected = float(projected_hours or 0.0)
    if used + projected > cap:
        print(f"[p2] {kind} budget: already {used:.4f} h + projected {projected:.4f} h "
              f"> {cap:.2f} h cap; refusing: not_run (cap)", file=sys.stderr)
        raise SystemExit(2)
    return {"kind": kind, "used_hours": used, "projected_hours": projected,
            "cap_hours": cap, "remaining_hours": cap - used, "pass": True}


def gpu_set_projection(*, models=None) -> dict:
    """§7 GPU projection: 15 models × (7 conditions × 2 + R1 test) + X0-b/new-model sets."""
    models = list(CORE_MODELS if models is None else models)
    conditions_sets = len(models) * (len(CONDITIONS) * 2 + 1)
    reproduce_sets = len(X0B_MODELS) * 2
    new_model_sets = len(NEW_MODELS) * 7
    total_sets = conditions_sets + reproduce_sets + new_model_sets
    hours = total_sets * PHOTON_SET_WALL_S / 3600.0 * PROJECTION_FACTOR
    runbook_sets = len(models) * (len(CONDITIONS) * 2) + len(NEW_MODELS) * 7
    runbook_hours = runbook_sets * PHOTON_SET_WALL_S / 3600.0 * PROJECTION_FACTOR
    mirror_verify_s = 5.0
    mirror_sets = 4
    hours += mirror_sets * mirror_verify_s / 3600.0 * PROJECTION_FACTOR
    optional_sets = len(OPTIONAL_MODELS) * (len(CONDITIONS) * 2 + 1)
    optional_hours = optional_sets * PHOTON_SET_WALL_S / 3600.0 * PROJECTION_FACTOR
    return {"n_models": len(models), "conditions_sets": int(conditions_sets),
            "reproduce_sets": int(reproduce_sets),
            "new_model_sets": int(new_model_sets), "total_sets": int(total_sets),
            "photon_set_wall_s": PHOTON_SET_WALL_S, "factor": PROJECTION_FACTOR,
            "projected_l40s_hours": round(hours, 4), "cap_hours": float(GPU_CAP_H),
            "pass": bool(hours <= GPU_CAP_H),
            "runbook_224_set_projection_hours": round(runbook_hours, 4),
            "mirror_verify_sets": int(mirror_sets),
            "mirror_verify_wall_s": float(mirror_verify_s),
            "note": ("the runbook §7 figure covers the 7 condition sets and the two "
                     "new models' 14 Part A sets; X0-b (10 sets) and the R1 test set "
                     "(15 sets) are added here"),
            "optional_models": list(OPTIONAL_MODELS),
            "optional_sets": int(optional_sets),
            "with_optional_hours": round(hours + optional_hours, 4),
            "with_optional_pass": bool(hours + optional_hours <= GPU_CAP_H),
            "rule": ("optional models run only if the with-optional projection fits "
                     "the 1.3 h cap; they are dropped here (§7)")}


CPU_PROJECTION = {
    "preflight": 0.20, "p2": 2.60, "x3p": 2.20, "report": 0.20, "new-models-x2b": 0.10,
    "dry-run": 0.10,
}


def cpu_projection() -> dict:
    hours = float(sum(CPU_PROJECTION.values()))
    return {"tasks": dict(CPU_PROJECTION), "projected_hours": round(hours, 3),
            "cap_hours": float(CPU_CAP_H), "pass": bool(hours <= CPU_CAP_H)}


def p2_s1_check(root=None, *, ledger=None) -> dict:
    """§7 S1: ledger charged + the 1.3 h GPU cap <= 160, fail closed on mismatch."""
    root = _resolve_root(root)
    payload = (ledger if ledger is not None
               else (read_json(root / "docs/jepa/RESOURCE_LEDGER.json", {}) or {}))
    entry = ((payload.get("v4_caps") or {}).get("E2_lucid_l40s_deepthought2") or {})
    charged = entry.get("charged_gpu_hours")
    booked = er.booked_gpu_hours(payload)
    consistency = er._ledger_consistency(charged, booked)
    total = float(booked["sum_hours"]) + GPU_CAP_H if consistency["consistent"] else None
    return {"registered": False, "extension": EXTENSION, "id": "S1 (P2 stage 1)",
            "ledger": "docs/jepa/RESOURCE_LEDGER.json",
            "ledger_job": "E2_lucid_l40s_deepthought2",
            "ledger_charged_gpu_hours": charged,
            "booked_gpu_hours": booked.get("sum_hours"),
            "ledger_consistent": bool(consistency["consistent"]),
            "p2_gpu_cap_hours": float(GPU_CAP_H),
            "sum_hours": None if total is None else round(total, 4),
            "ceiling_hours": float(S1_CEILING_H),
            "pass": bool(total is not None and total <= S1_CEILING_H),
            "reason": None if consistency["consistent"] else consistency["reason"],
            "rule": "booked L40S sum + the 1.3 h Stage 1 GPU cap <= 160 "
                    "(fail closed on a ledger inconsistency)"}


def p2_s1_check_or_stop(root=None, *, ledger=None) -> dict:
    result = p2_s1_check(root, ledger=ledger)
    if not result["pass"]:
        print(f"[p2] S1 STOP: booked {result['booked_gpu_hours']} + "
              f"{result['p2_gpu_cap_hours']} = {result['sum_hours']} > "
              f"{result['ceiling_hours']} h", file=sys.stderr)
        raise SystemExit(2)
    return result


def entries_preflight(*, guard_json=GUARD_JSON, free_entries=None,
                      projected=ENTRIES_PROJECTION, log=print) -> dict:
    if free_entries is not None:
        snapshot = {"available": True, "free_entries": int(free_entries),
                    "source": "injected"}
    else:
        snapshot = ef.entries_snapshot(Path(guard_json))
    out = {"registered": False, "extension": EXTENSION, "id": "entries (P2)",
           "projection": {"writes": int(projected), "floor": int(ENTRIES_FLOOR)},
           "snapshot": snapshot, "pass": False}
    if not snapshot.get("available"):
        out["error"] = snapshot.get("error", "no free_entries in the guard status")
        return out
    free = int(snapshot["free_entries"])
    after = free - int(projected)
    out.update({"free_entries": free, "projected_after": after,
                "floor": int(ENTRIES_FLOOR),
                "pass": bool(free >= ENTRIES_FLOOR and after >= ENTRIES_FLOOR)})
    return out


def require_stage(relative, *, message) -> dict:
    payload = read_json(Path(EXT_ROOT) / relative, None)
    if payload is None:
        raise RuntimeError(f"{message}: {Path(EXT_ROOT) / relative} is missing")
    return payload


def load_verified_inventory(root=None) -> dict:
    """The X0-a inventory (hashes verified there); later stages reuse it.

    X0-a is the §4 hash gate; re-hashing every checkpoint on the throttled GPU
    host client would multiply the job wall for no extra assurance.  Fails closed
    when the X0-a inventory is absent or unverified.
    """
    payload = read_json(Path(EXT_ROOT) / "x0" / "inventory.json", None)
    if payload is None or payload.get("hash_files") is not True:
        raise RuntimeError("the X0-a verified inventory is missing; run preflight first")
    return payload


def require_mirror_base(root=None) -> str:
    """The mirror root for the GPU stages (fails closed when none is usable).

    An explicit ``JEPA_P2_MIRROR`` (a node-local fast tree, e.g. ``/dev/shm``)
    wins over the base recorded by the X0-a preflight staging step; both are
    recorded in the payloads so the read path is auditable.  The mirror content
    is verified against the pinned source manifest before it is read.
    """
    environment = os.environ.get("JEPA_P2_MIRROR")
    if environment:
        return str(mirror_root())
    plan = require_preflight_plan(root)
    info = plan.get("data_mirror") or {}
    base = info.get("base")
    if not base:
        raise RuntimeError("the X0-a plan carries no data_mirror base; re-run preflight")
    return str(base)


def mirror_audit(root=None) -> dict:
    """The mirror bases in play: the plan-recorded one and the env override."""
    plan = read_json(Path(EXT_ROOT) / "x0" / "plan.json", {}) or {}
    recorded = ((plan.get("data_mirror") or {}).get("base"))
    environment = os.environ.get("JEPA_P2_MIRROR")
    return {"plan_base": recorded,
            "env_base": (None if not environment
                         else str(mirror_root())),
            "used_base": require_mirror_base(root),
            "rule": "JEPA_P2_MIRROR wins when set; content verified against the "
                    "pinned source manifest"}


def gpu_cap_check() -> dict:
    """§7 post-job STOP: booked Stage-1 GPU hours must not exceed the 1.3 h cap."""
    used = float(job_hours().get("gpu", 0.0))
    out = {"booked_gpu_hours": used, "cap_hours": float(GPU_CAP_H),
           "pass": bool(used <= GPU_CAP_H)}
    if not out["pass"]:
        print(f"[p2] S1 STOP: booked GPU hours {used:.4f} > {GPU_CAP_H} h cap",
              file=sys.stderr)
        raise SystemExit(2)
    return out


def require_preflight_plan(root=None) -> dict:
    path = Path(EXT_ROOT) / "x0" / "plan.json"
    payload = read_json(path, None)
    if payload is None:
        raise RuntimeError(f"X0-a plan missing at {path}; run the preflight mode first")
    return payload


# ============================================================ X0-a §4
def identity_check(dataset, geometry, cfg, *, limit=IDENTITY_EVENTS, log=print) -> dict:
    """§4: T(0.25, 0.25) reproduces the stored hits exactly (same hits, same times)."""
    condition = identity_condition()
    rule = pmt_rule()
    samples = mismatches = fallback = 0
    examples = []
    max_time_difference = 0.0
    for row in dataset.runs:
        if samples >= int(limit):
            break
        store = lb.read_sparse_hits(Path(dataset.directory) / "runs"
                                   / row["run_dir"] / "hits.npz")
        offsets = np.asarray(store.offsets, np.int64)
        for index, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
            if samples >= int(limit):
                break
            span = slice(int(offsets[index]), int(offsets[index + 1]))
            pmt = np.asarray(store.pmt_id[span]); charge = np.asarray(store.charge_pe[span])
            time = np.asarray(store.time_ns[span])
            pmt_before, charge_before, time_before = pmt.copy(), charge.copy(), time.copy()
            samples += 1
            # the identity audit rejects any in-place mutation of the input
            result = transform_event(condition, str(row["run_id"]), pmt, charge, time,
                                     geometry, cfg, rule)
            if not (np.array_equal(pmt, pmt_before) and np.array_equal(charge, charge_before)
                    and np.array_equal(time, time_before)):
                mismatches += 1
                examples.append({"run_id": str(row["run_id"]), "event_id": int(event_id),
                                 "reason": "input mutated"})
                continue
            # a stored token outside the prompt window is the no-trigger fallback origin
            if time.size and float(np.max(time)) > 60.5:
                fallback += 1
            if result is None:
                mismatches += 1
                examples.append({"run_id": str(row["run_id"]), "event_id": int(event_id),
                                 "reason": "identity dropped the event"})
                continue
            same = bool(np.array_equal(result["pmt_id"], pmt)
                        and np.array_equal(result["charge_pe"], charge.astype(float)))
            difference = (float(np.max(np.abs(result["time_ns"] - time.astype(float))))
                          if time.size else 0.0)
            max_time_difference = max(max_time_difference, difference)
            if not same or difference > IDENTITY_TOLERANCE_NS or result["trig_rel"] != 0.0:
                mismatches += 1
                if len(examples) < 10:
                    examples.append({"run_id": str(row["run_id"]),
                                     "event_id": int(event_id),
                                     "same_hits": same,
                                     "max_time_difference_ns": difference,
                                     "trigger": result["trig_rel"]})
    return {"check": "identity T(0.25, 0.25) §4", "samples": int(samples),
            "mismatches": int(mismatches), "no_trigger_fallback_events": int(fallback),
            "max_time_difference_ns": float(max_time_difference),
            "tolerance_ns": IDENTITY_TOLERANCE_NS, "examples": examples,
            "pass": bool(samples > 0 and mismatches == 0)}


def condition_determinism(dataset, geometry, cfg, *, limit=IDENTITY_EVENTS) -> dict:
    """A transform applied twice returns the same store (never mutates its input)."""
    out = {}
    for name, condition in build_drift_conditions().items():
        samples = mutations = differences = 0
        for row in dataset.runs:
            if samples >= int(limit):
                break
            store = lb.read_sparse_hits(Path(dataset.directory) / "runs"
                                       / row["run_dir"] / "hits.npz")
            offsets = np.asarray(store.offsets, np.int64)
            for index, event_id in enumerate(np.asarray(store.event_ids, np.int64)):
                if samples >= int(limit):
                    break
                span = slice(int(offsets[index]), int(offsets[index + 1]))
                pmt = np.asarray(store.pmt_id[span]).copy()
                charge = np.asarray(store.charge_pe[span]).copy()
                time = np.asarray(store.time_ns[span]).copy()
                pmt0, charge0, time0 = pmt.copy(), charge.copy(), time.copy()
                first = transform_event(condition, str(row["run_id"]), pmt, charge, time,
                                        geometry, cfg, pmt_rule())
                second = transform_event(condition, str(row["run_id"]), pmt, charge, time,
                                         geometry, cfg, pmt_rule())
                samples += 1
                if not (np.array_equal(pmt, pmt0) and np.array_equal(charge, charge0)
                        and np.array_equal(time, time0)):
                    mutations += 1
                if (first is None) != (second is None):
                    differences += 1
                elif first is not None:
                    same = bool(np.array_equal(first["pmt_id"], second["pmt_id"])
                                and np.array_equal(first["charge_pe"], second["charge_pe"])
                                and np.array_equal(first["time_ns"], second["time_ns"])
                                and first["trig_rel"] == second["trig_rel"])
                    if not same:
                        differences += 1
        out[name] = {"samples": int(samples), "mutations": int(mutations),
                     "differences": int(differences),
                     "pass": bool(samples > 0 and mutations == 0 and differences == 0)}
    return out


def preflight(root=None, *, config, gen, arms_config, geometry,
              data_root=DATA_ROOT_DEFAULT, device="cpu", limit=None, log=print) -> dict:
    """X0-a (CPU, read-only): hashes, self-checks, projection, entries, S1 (: ``plan.json``)."""
    root = _resolve_root(root)
    gate = assert_authorised("preflight", root=root, log=log)
    started = time.time()
    payload = {"registered": False, "extension": EXTENSION, "check": "X0-a preflight §4",
               "runbook": RUNBOOK_REL, "runbook_sha256": (
                   lr.sha256_file(root / RUNBOOK_REL)
                   if (root / RUNBOOK_REL).is_file() else None),
               "code_sha": lr.git_head(root), "git_status": lr.git_status(root),
               "data_root": str(data_root), "gate": {"status": gate["status"]},
               "stops": [], "wall_seconds": None,
               "execution_order": list(EXECUTION_ORDER)}
    payload["data"] = er.data_status(data_root)
    if not payload["data"]["pass"]:
        payload["stops"].append("pinned data manifest missing or mismatched (§2.1)")
    payload["s1"] = p2_s1_check(root)
    payload["entries"] = entries_preflight(log=log)
    payload["gpu_projection"] = gpu_set_projection()
    payload["gpu_projection_core"] = gpu_set_projection(models=CORE_MODELS)
    payload["cpu_projection"] = cpu_projection()
    payload["hardware_and_software"] = er.hardware_and_software()
    inventory = p2_inventory(root, hash_files=True)
    payload["inventory"] = inventory
    if inventory["unavailable"]:
        payload["stops"].append(
            f"checkpoint inventory: {len(inventory['unavailable'])} unavailable row(s)")
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    mirror_started = time.time()
    mirror_sets = {}
    for setting, probes in (("s005", probes005), ("s030", probes030)):
        for role, dataset in probes.items():
            mirrored = mirror_dataset(dataset, stage=True, log=log)
            mirror_sets[f"{setting}__{role}"] = {
                "dataset_key": dataset.key,
                "source": str(dataset.directory),
                "directory": str(mirrored.directory),
                "source_manifest_sha256": dataset.manifest_sha256()}
    payload["data_mirror"] = {
        "base": str(mirror_root()),
        "stage_wall_seconds": round(time.time() - mirror_started, 3),
        "sets": mirror_sets,
        "rule": "staged once by this CPU preflight; the GPU stages verify and read "
                "it only, so no GPU job wall includes staging I/O",
        "verified": True}
    payload["identity"] = identity_check(probes005["test"], geometry, gen,
                                         limit=limit or IDENTITY_EVENTS, log=log)
    payload["determinism"] = condition_determinism(probes005["test"], geometry, gen,
                                                   limit=limit or IDENTITY_EVENTS)
    payload["transforms"] = {}
    for name, condition in build_drift_conditions().items():
        for role, dataset in (("train", probes005["train"]), ("test", probes005["test"])):
            info = transform_info(dataset, condition, geometry, gen)
            written = write_transform_info(EXT_ROOT, role, info)
            payload["transforms"][f"{name}__{role}"] = {
                "written": written, "condition": condition.as_dict(),
                "n_events": info["n_events"], "n_triggered": info["n_triggered"],
                "trigger_efficiency": info["trigger_efficiency"],
                "n_window_edge": info["n_window_edge"],
                "trig_rel_distribution": info["trig_rel_distribution"]}
    payload["trigger_efficiency"] = {
        key: value["trigger_efficiency"] for key, value in payload["transforms"].items()}
    payload["window_edge"] = {
        key: value["n_window_edge"] for key, value in payload["transforms"].items()}
    out_dir = Path(EXT_ROOT) / "x0"
    write_json(out_dir / "inventory.json", inventory)
    for name in ("identity", "determinism"):
        if name == "identity" and not payload[name]["pass"]:
            payload["stops"].append("the identity transform does not reproduce the "
                                    "stored hits (§4, STOP S9)")
        if name == "determinism" and not all(row["pass"]
                                             for row in payload[name].values()):
            payload["stops"].append("a condition transform is not deterministic (§2.3)")
    for name in ("entries", "s1"):
        if payload[name].get("pass") is False:
            payload["stops"].append(f"{name} projection failed (§7)")
    if payload["gpu_projection"].get("pass") is False:
        payload["stops"].append("the core GPU projection exceeds the 1.3 h cap (STOP S1)")
    if payload["cpu_projection"].get("pass") is False:
        payload["stops"].append("the CPU projection exceeds the 8 h cap (STOP S1)")
    payload["pass"] = bool(not payload["stops"])
    payload["closure"] = assert_closure("preflight", root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(out_dir / "plan.json", payload)
    record_job("preflight", "cpu", time.time() - started)
    return payload


# ============================================================ X0-b §4
def reproduce(root=None, *, config, gen, arms_config, geometry,
              data_root=DATA_ROOT_DEFAULT, device="auto", inventory=None,
              limit=None, log=print) -> dict:
    """X0-b reproduction gate (GPU): the X0-b N vertex values and the two new models."""
    root = _resolve_root(root)
    gate = assert_authorised("reproduce", root=root, log=log)
    started = time.time()
    inv = inventory if inventory is not None else load_verified_inventory(root)
    assert_budget("gpu", len(X0B_MODELS) * 2 * PHOTON_SET_WALL_S / 3600.0
                  * PROJECTION_FACTOR)
    require_preflight_plan(root)
    mirror = require_mirror_base(root)
    probes005 = mirror_probe_sets(
        open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log),
        base=mirror, stage=False, log=log)
    payload = {"registered": False, "extension": EXTENSION,
               "check": "X0-b reproduction gate §4", "runbook": RUNBOOK_REL,
               "tolerance_m": er.X0B_TOLERANCE_M, "models": {},
               "gate": {"status": gate["status"]}, "pass": True, "wall_seconds": None}
    for label in X0B_MODELS:
        row = model_row(inv, label)
        cache = embedding_cache(label)
        if label in NEW_MODELS:
            keys = {role: er.embedding_cache_key(role, "N") for role in ("train", "test")}
        else:
            keys = {role: f"x0b_{role}__N" for role in ("train", "test")}
        model = build_model(row, config=config, gen=gen, arms_config=arms_config,
                            geometry=geometry, data_root=data_root, device=device, log=log)
        split = split_of(row)
        rows = {}
        for role in ("train", "test"):
            condition = er.build_condition("N", probes005[role], geometry)
            rows[role] = er.stream_condition_embeddings(
                model, probes005[role], geometry, condition, split=split, config=config,
                device=device, cache=cache, cache_key=keys[role], max_events=limit, log=log)
        fitted = er.fit_vertex_probe(rows["train"])
        metrics = er.vertex_metrics(rows["train"], rows["test"], fitted=fitted)
        got = metrics["mean_error_m"]
        expected = X0B_EXPECTED[label]
        if label in NEW_MODEL_SPECS:
            source = "results/jepa/20261107_shortcut/readings.json r2_b/r2_c"
        elif label == "fix9000":
            source = "results/jepa/20261107_shortcut/reproduction.json item fix9000"
        else:
            source = "Part A X0-b N (refit = frozen)"
        check = er.check_reproduction(label, got, expected)
        entry = {"status": "evaluated", "expected_m": check["expected"],
                 "got_m": check["got"], "difference_m": check["difference_m"],
                 "tolerance_m": check["tolerance_m"], "pass": check["pass"],
                 "expected_source": source, "n_test": metrics["n_events"]}
        payload["models"][label] = entry
        payload["pass"] = bool(payload["pass"] and entry["pass"])
    payload["data_mirror"] = dict(mirror_audit(root), verified=True,
                                    staged_by="preflight")
    payload["closure"] = assert_closure("reproduce", root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(EXT_ROOT) / "x0" / "reproduction.json", payload)
    record_job("reproduce", "gpu", time.time() - started)
    payload["gpu_cap"] = gpu_cap_check()
    return payload


# ============================================================ new models §2.2
def _aligned_pair(reference, other):
    paired, aligned = er.pair_embedding_rows(reference, other)
    return paired, aligned


def _x2b_fit(rows, occupancy, delta, geometry, *, log=print) -> dict:
    """The X2-b occupancy fit of one model (gain R²_eq and the corrected raw map)."""
    u_train = er.occupancy_design(rows["train"]["S"], delta["train"],
                                  occupancy["train"], geometry)
    u_test = er.occupancy_design(rows["test"]["S"], delta["test"],
                                 occupancy["test"], geometry)
    _p, s_train = _aligned_pair(rows["train"]["N"], rows["train"]["S"])
    _p, s_test = _aligned_pair(rows["test"]["N"], rows["test"]["S"])
    n_train = rows["train"]["N"]
    n_test = rows["test"]["N"]
    fit = er.equivariance_fit(u_train, np.asarray(s_train["events"], float)
                              - np.asarray(n_train["events"], float),
                              u_test, np.asarray(s_test["events"], float)
                              - np.asarray(n_test["events"], float),
                              run_ids_train=np.asarray(rows["train"]["N"]["run_ids"]),
                              run_ids_test=np.asarray(rows["test"]["N"]["run_ids"]),
                              kind="occupancy")
    A_hat = er.raw_map(fit)
    identity = None
    if A_hat is not None and u_test.shape[0]:
        prediction = np.asarray(fit["prediction"], float)
        manual = u_test @ np.asarray(A_hat, float).T
        # D1 identity: u Â_rawᵀ = predict(u) − predict(0) for the same fitted probes
        test = la.ProbeDataset(kind="delta_z", form="occupancy", x=u_test,
                               y=np.asarray(s_test["events"], float)
                               - np.asarray(n_test["events"], float),
                               run_ids=np.asarray(rows["test"]["N"]["run_ids"],
                                                  dtype=object),
                               target_names=tuple(f"z{i}" for i in
                                                  range(prediction.shape[1])),
                               meta={"kind": "occupancy"})
        zero_prediction = la.probe_predictions(
            fit["fitted"], dataclasses.replace(test, x=np.zeros_like(u_test)))
        identity = {"max_abs_difference": float(np.max(np.abs(
            manual - (prediction - zero_prediction)))), "tolerance": 1e-8}
        identity["pass"] = bool(identity["max_abs_difference"] <= 1e-8)
    return {"r2_eq": fit["r2_eq"], "r2_eq_class": er.equivariance_class(fit["r2_eq"]),
            "n_train": fit["n_train"], "n_test": fit["n_test"],
            "A_hat": None if A_hat is None else np.asarray(A_hat, float).tolist(),
            "A_hat_identity": identity,
            "fit_run_ids": fit["fit_run_ids"]}


def new_models(root=None, *, config, gen, arms_config, geometry,
               data_root=DATA_ROOT_DEFAULT, device="auto", inventory=None,
               limit=None, log=print) -> dict:
    """§2.2: the two new models' Part A sets and their corrected X2-b fit (GPU)."""
    root = _resolve_root(root)
    gate = assert_authorised("new-models", root=root, log=log)
    started = time.time()
    inv = inventory if inventory is not None else load_verified_inventory(root)
    assert_budget("gpu", len(NEW_MODELS) * 7 * PHOTON_SET_WALL_S / 3600.0
                  * PROJECTION_FACTOR)
    require_preflight_plan(root)
    mirror = require_mirror_base(root)
    probes005 = mirror_probe_sets(
        open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log),
        base=mirror, stage=False, log=log)
    probes030 = mirror_probe_sets(
        open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log),
        base=mirror, stage=False, log=log)
    delta = er.run_log_gain_vectors(probes030["train"], geometry)
    delta.update(er.run_log_gain_vectors(probes030["test"], geometry))
    payload = {"registered": False, "extension": EXTENSION,
               "check": "new models §2.2", "runbook": RUNBOOK_REL,
               "models": {}, "gate": {"status": gate["status"]}, "wall_seconds": None}
    out_dir = Path(EXT_ROOT) / "x2b"
    for label in NEW_MODELS:
        row = model_row(inv, label)
        cache = embedding_cache(label)
        model = build_model(row, config=config, gen=gen, arms_config=arms_config,
                            geometry=geometry, data_root=data_root, device=device, log=log)
        split = split_of(row)
        rows = {role: {} for role in ("train", "test")}
        requests = [("N", probes005, probes005), ("D", probes030, probes030),
                    ("S", probes005, probes005)]
        for name, train_set, test_set in requests:
            for role, dataset in (("train", train_set), ("test", test_set)):
                condition = er.build_condition(name, dataset[role], geometry)
                rows[role][name] = er.stream_condition_embeddings(
                    model, dataset[role], geometry, condition, split=split,
                    config=config, device=device, cache=cache,
                    cache_key=er.embedding_cache_key(role, name), max_events=limit,
                    log=log)
        condition = er.build_condition("SWunit", probes030["test"], geometry)
        rows["test"]["SWunit"] = er.stream_condition_embeddings(
            model, probes030["test"], geometry, condition, split=split, config=config,
            device=device, cache=cache,
            cache_key=er.embedding_cache_key("test", "SWunit"), max_events=limit, log=log)
        occupancy = {role: er.occupancy_counts(
            probes005[role], er.build_condition("N", probes005[role], geometry), geometry)
            for role in ("train", "test")}
        fit = _x2b_fit(rows, occupancy, {"train": delta, "test": delta}, geometry, log=log)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / f"{label}.json", dict(fit, label=label, split=split))
        payload["models"][label] = {
            "status": "evaluated", "split": split, "r2_eq": fit["r2_eq"],
            "r2_eq_class": fit["r2_eq_class"], "n_train": fit["n_train"],
            "n_test": fit["n_test"], "A_hat_shape": (
                None if fit["A_hat"] is None
                else [len(fit["A_hat"]), len(fit["A_hat"][0])]),
            "A_hat_identity": fit["A_hat_identity"],
            "cache": {role: {name: rows[role][name].get("cache_status")
                             for name in rows[role]} for role in ("train", "test")}}
    payload["pass"] = all(entry.get("status") == "evaluated"
                          for entry in payload["models"].values())
    payload["data_mirror"] = dict(mirror_audit(root), verified=True,
                                    staged_by="preflight")
    payload["closure"] = assert_closure("new-models", root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(EXT_ROOT) / "new_models.json", payload)
    record_job("new-models", "gpu", time.time() - started)
    payload["gpu_cap"] = gpu_cap_check()
    return payload


# ============================================================ conditions §2.3
def n_embedding_cache(label) -> er.EmbeddingCache:
    """Where a model's N embeddings live: Part A for the 13, the P2 cache for the new."""
    if label in NEW_MODEL_SPECS:
        return embedding_cache(label)
    return er.EmbeddingCache(PILOT_EXT_ROOT / "x1" / str(label))


def ensure_transform_info(probes, conditions, geometry, cfg, *, log=print) -> dict:
    """Load the transform bookkeeping, computing and writing any missing file."""
    out = {}
    for name, condition in conditions.items():
        for role in ("train", "test"):
            key = f"{name}__{role}"
            try:
                info = load_transform_info(EXT_ROOT, role, name)
                out[key] = {"status": "reused"}
            except RuntimeError:
                info = transform_info(probes[role], condition, geometry, cfg)
                written = write_transform_info(EXT_ROOT, role, info)
                out[key] = {"status": "written", **written}
    return out


def conditions(root=None, *, config, gen, arms_config, geometry,
               data_root=DATA_ROOT_DEFAULT, device="auto", inventory=None,
               limit=None, log=print) -> dict:
    """§2.3/§8.5: encode the 7 condition sets (and the R1 test set) for the 15 models."""
    root = _resolve_root(root)
    gate = assert_authorised("conditions", root=root, log=log)
    started = time.time()
    inv = inventory if inventory is not None else load_verified_inventory(root)
    projection = gpu_set_projection()
    assert_budget("gpu", len(CORE_MODELS) * (len(CONDITIONS) * 2 + 1)
                  * PHOTON_SET_WALL_S / 3600.0 * PROJECTION_FACTOR)
    require_stage("x0/plan.json", message="conditions needs the X0-a plan")
    mirror = require_mirror_base(root)
    probes005 = mirror_probe_sets(
        open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log),
        base=mirror, stage=False, log=log)
    drift = build_drift_conditions()
    payload = {"registered": False, "extension": EXTENSION, "check": "§2.3 conditions",
               "runbook": RUNBOOK_REL, "conditions": list(drift), "models": {},
               "transforms": ensure_transform_info(probes005, drift, geometry, gen,
                                                   log=log),
               "gate": {"status": gate["status"]},
               "projection": projection, "wall_seconds": None}
    for label in CORE_MODELS:
        row = model_row(inv, label)
        cache = embedding_cache(label)
        nc = n_embedding_cache(label)
        n_present = {role: nc.has(er.embedding_cache_key(role, "N"))
                     for role in ("train", "test")}
        if not all(n_present.values()):
            payload["models"][label] = {"status": "unavailable",
                                        "reason": f"N embeddings missing: {n_present}"}
            continue
        model = build_model(row, config=config, gen=gen, arms_config=arms_config,
                            geometry=geometry, data_root=data_root, device=device, log=log)
        split = split_of(row)
        entry = {"status": "evaluated", "split": split, "n_embedding_cache": str(cache.root),
                 "sets": {}}
        for name, condition in drift.items():
            roles = ("test",) if name == R1_NAME else ("train", "test")
            for role in roles:
                rows = stream_drift_embeddings(
                    model, probes005[role], condition, split=split, geometry=geometry,
                    cfg=gen, config=config, device=device, cache=cache,
                    cache_key=er.embedding_cache_key(role, name), log=log)
                entry["sets"][f"{name}__{role}"] = {
                    "cache_status": rows.get("cache_status"),
                    "n_events": int(np.asarray(rows["events"]).shape[0]),
                    "cache_sha256": rows.get("cache_sha256")}
        payload["models"][label] = entry
    payload["complete"] = all(entry.get("status") == "evaluated"
                              for entry in payload["models"].values())
    payload["pass"] = bool(payload["complete"])
    payload["data_mirror"] = dict(mirror_audit(root), verified=True,
                                    staged_by="preflight")
    payload["closure"] = assert_closure("conditions", root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(EXT_ROOT) / "conditions.json", payload)
    record_job("conditions", "gpu", time.time() - started)
    payload["gpu_cap"] = gpu_cap_check()
    return payload


# ============================================================ readings §5
def _align_subset(reference, other) -> tuple:
    """N indices (in ``other`` order) for the events ``other`` shares with ``reference``."""
    keys_ref = [(str(run), int(event)) for run, event in
                zip(reference["run_ids"], reference["event_ids"])]
    index_ref = {key: index for index, key in enumerate(keys_ref)}
    if len(index_ref) != len(keys_ref):
        raise ValueError("_align_subset: duplicate reference keys")
    index_left, index_right, runs = [], [], []
    vertex_ref = np.asarray(reference["vertex"], float)
    vertex_other = np.asarray(other["vertex"], float)
    for position, key in enumerate(zip(np.asarray(other["run_ids"]).astype(str),
                                       np.asarray(other["event_ids"], np.int64))):
        key = (str(key[0]), int(key[1]))
        if key not in index_ref:
            raise ValueError(f"_align_subset: {key} is absent from the reference rows")
        left = index_ref[key]
        if not np.array_equal(vertex_ref[left], vertex_other[position]):
            raise ValueError("_align_subset: the paired events carry different truth")
        index_left.append(left); index_right.append(position); runs.append(key[0])
    return (np.asarray(index_left, np.int64), np.asarray(index_right, np.int64),
            np.asarray(runs, dtype=object))


def _vertex_errors(train_rows, test_rows, fitted) -> np.ndarray:
    return np.asarray(er.vertex_metrics(train_rows, test_rows, fitted=fitted)["errors"],
                      float)


def _yield_errors(train_rows, test_rows, fitted) -> np.ndarray:
    return np.asarray(er.log_yield_metrics(train_rows, test_rows, fitted=fitted)["errors"],
                      float)


def _rms(values) -> float:
    return er.rms(values)


def _mean(values) -> float:
    return float(np.mean(np.asarray(values, float)))


def _paired(left, right, run_ids, *, statistic="mean") -> dict:
    if statistic == "mean":
        return ef.paired_run_bootstrap(np.asarray(left, float), np.asarray(right, float),
                                       np.asarray(run_ids, dtype=object))
    return er.paired_cluster_bootstrap(np.asarray(left, float), np.asarray(right, float),
                                       np.asarray(run_ids, dtype=object), statistic=er.rms)


def _difference_reading(left_errors, right_errors, run_ids, *, statistic="mean") -> dict:
    """``right - left`` with its run-cluster CI (pairs kept; never mixed statistics)."""
    bootstrap = _paired(left_errors, right_errors, run_ids, statistic=statistic)
    difference = float(bootstrap["mean_difference_m"] if statistic == "mean"
                       else bootstrap["difference"])
    ci = (bootstrap["ci95_m"] if statistic == "mean" else bootstrap["ci95"])
    return {"difference": difference, "ci95": [float(ci[0]), float(ci[1])],
            "bootstrap": bootstrap,
            "excludes_zero": bool(ci[0] > 0.0 or ci[1] < 0.0)}


def _trig_lookup(info) -> dict:
    return {(str(run), int(event)): float(trig)
            for run, event, trig in zip(info["run_ids"], info["event_ids"],
                                        info["trig_rel"])}


def _mask_rows(rows, lookup) -> np.ndarray:
    mask = np.zeros(len(np.asarray(rows["event_ids"])), bool)
    for position, (run, event) in enumerate(zip(np.asarray(rows["run_ids"]).astype(str),
                                                np.asarray(rows["event_ids"], np.int64))):
        trig = lookup.get((str(run), int(event)))
        mask[position] = bool(trig is not None and np.isfinite(trig) and trig <= 0.0)
    return mask


def _subset_rows(rows, mask) -> dict:
    mask = np.asarray(mask, bool)
    out = {}
    n_rows = int(mask.size)
    for key, value in rows.items():
        array = np.asarray(value)
        out[key] = array[mask] if (array.ndim >= 1 and array.shape[0] == n_rows) else array
    return out


def _dead_by_run(rows, condition, geometry) -> dict | None:
    if condition.kind != "dead":
        return None
    return {str(run): condition.dead_fraction_by_module(str(run), geometry)
            for run in np.unique(np.asarray(rows["run_ids"]).astype(str))}


def r2_design(rows, condition, dead_by_run, hbar, geometry) -> np.ndarray:
    """The §3 R2 design: dead_m · h_e / h̄ (K/P) or h_e / h̄ + an intercept (T/R1)."""
    n_hit = np.asarray(rows["n_hit"], float)
    if condition.kind == "dead":
        design = np.zeros((n_hit.size, int(geometry.n_modules)))
        for position, run in enumerate(np.asarray(rows["run_ids"]).astype(str)):
            design[position] = dead_by_run[str(run)] * (n_hit[position] / float(hbar))
        return design
    return np.stack([n_hit / float(hbar), np.ones(n_hit.size)], axis=1)


def mahalanobis_scores(train_rows, rows) -> np.ndarray:
    """Mahalanobis distance of each row's embedding to the N-train distribution (§3 R5)."""
    train = np.asarray(train_rows["events"], float)
    mean = train.mean(axis=0)
    centred = train - mean
    covariance = centred.T @ centred / max(train.shape[0], 1)
    d = covariance.shape[0]
    epsilon = 1e-3 * float(np.trace(covariance)) / float(d)
    inverse = np.linalg.pinv(covariance + epsilon * np.eye(d))
    delta = np.asarray(rows["events"], float) - mean
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, inverse, delta), 0.0))


def abstention_reading(train_rows, n_test_rows, c_test_rows, frozen_vertex) -> dict:
    """§5 P2-g: vertex error at 100/80/50 % coverage next to N at the same coverage."""
    n_errors = _vertex_errors(train_rows, n_test_rows, frozen_vertex)
    c_errors = _vertex_errors(train_rows, c_test_rows, frozen_vertex)
    n_scores = mahalanobis_scores(train_rows, n_test_rows)
    c_scores = mahalanobis_scores(train_rows, c_test_rows)
    index_n, index_c, _runs = _align_subset(n_test_rows, c_test_rows)
    n_errors, n_scores = n_errors[index_n], n_scores[index_n]
    out = {}
    for coverage in (1.0, 0.8, 0.5):
        n_keep = max(int(round(coverage * n_errors.size)), 1)
        c_keep = max(int(round(coverage * c_errors.size)), 1)
        n_selected = n_errors[np.argsort(n_scores)[:n_keep]]
        c_selected = c_errors[np.argsort(c_scores)[:c_keep]]
        out[f"{int(coverage * 100)}"] = {
            "coverage": coverage,
            "v_c_m": float(np.mean(c_selected)), "v_N_m": float(np.mean(n_selected)),
            "difference_m": float(np.mean(c_selected) - np.mean(n_selected)),
            "n_c": int(c_keep), "n_N": int(n_keep)}
    return out


def cell_reading(data, *, frozen_vertex, frozen_yield, frozen_raw_vertex=None,
                 frozen_raw_yield=None) -> dict:
    """All §5 readings of one model on one condition (full or exact-subset rows).

    ``data`` carries the N and condition probe rows (train and test), the R1 test
    rows for T1.0w, the condition, the N-train mean hit count and the dead
    fractions per run.  Every comparison uses the frozen P_N probe on the same
    paired event subset (run-cluster paired bootstrap), so dropped events leave
    both members of a pair (§2.3).
    """
    condition = data["condition"]
    n_train, n_test = data["n_train"], data["n_test"]
    c_train, c_test = data["c_train"], data["c_test"]
    if c_test is None or c_test["events"].shape[0] == 0:
        return {"condition": condition.name, "status": "no events"}
    index_n, index_c, run_ids = _align_subset(n_test, c_test)
    n_errors = _vertex_errors(n_train, n_test, frozen_vertex)[index_n]
    c_errors = _vertex_errors(n_train, c_test, frozen_vertex)[index_c]
    n_yield = _yield_errors(n_train, n_test, frozen_yield)[index_n]
    c_yield = _yield_errors(n_train, c_test, frozen_yield)[index_c]
    delta = _difference_reading(n_errors, c_errors, run_ids)
    delta_yield = _difference_reading(n_yield, c_yield, run_ids, statistic="rms")
    v_n, v_c = _mean(n_errors), _mean(c_errors)
    # §3: the drift cost is eligible if it is >= 3 mm with a CI excluding 0.
    eligible = bool(delta["difference"] >= DRIFT_COST_MIN_M and delta["excludes_zero"])
    out = {"condition": condition.name, "n_events": int(n_errors.size),
           "v_N_m": v_n, "v_c_m": v_c,
           "delta_c_m": delta["difference"], "delta_ci_m": delta["ci95"],
           "delta_eligible": eligible,
           "v_N_log_yield": _rms(n_yield), "v_c_log_yield": _rms(c_yield),
           "delta_log_yield": delta_yield["difference"],
           "delta_log_yield_ci": delta_yield["ci95"]}
    # ---- R3 readout refit (§5 P2-b)
    refit_vertex = er.fit_vertex_probe(c_train)
    refit_yield = er.fit_log_yield_probe(c_train)
    r3_errors = _vertex_errors(c_train, c_test, refit_vertex)[index_c]
    refit_difference = _difference_reading(n_errors, r3_errors, run_ids)
    R_c = float(_mean(r3_errors) - v_n)
    out["refit"] = {"v_R3_m": _mean(r3_errors), "R_c_m": R_c,
                    "ci_m": refit_difference["ci95"],
                    "excludes_zero": refit_difference["excludes_zero"],
                    "information_destroying": bool(
                        R_c >= REFIT_DESTROYED_M and refit_difference["excludes_zero"])}
    r3_yield = _yield_errors(c_train, c_test, refit_yield)[index_c]
    out["refit_log_yield"] = {
        "v_R3": _rms(r3_yield),
        "R_c": float(_rms(r3_yield) - _rms(n_yield)),
        "ci": _difference_reading(n_yield, r3_yield, run_ids, statistic="rms")["ci95"]}
    # ---- R2 latent repair from the known detector state (label-free, §3)
    runs_both = set(np.asarray(c_train["run_ids"]).astype(str)) | set(
        np.asarray(c_test["run_ids"]).astype(str))
    dead_by_run = (None if condition.kind != "dead" else
                   {str(run): condition.dead_fraction_by_module(str(run), data["geometry"])
                    for run in runs_both})
    u_train = r2_design(c_train, condition, dead_by_run, data["hbar"], data["geometry"])
    u_test = r2_design(c_test, condition, dead_by_run, data["hbar"], data["geometry"])
    index_nt, index_ct, runs_t = _align_subset(n_train, c_train)
    dz_train = (np.asarray(c_train["events"], float)
                - np.asarray(n_train["events"], float)[index_nt])
    index_ne, index_ce, runs_e = _align_subset(n_test, c_test)
    dz_test = (np.asarray(c_test["events"], float)
               - np.asarray(n_test["events"], float)[index_ne])
    latent = {"status": "not available"}
    if condition.kind == "dead" and dead_by_run is None:
        pass
    try:
        fit = er.equivariance_fit(
            u_train, dz_train, u_test, dz_test,
            run_ids_train=np.asarray(c_train["run_ids"]),
            run_ids_test=np.asarray(c_test["run_ids"]),
            kind=("occupancy" if condition.kind == "dead" else "global"))
        A_raw = er.raw_map(fit)
    except (ValueError, KeyError) as problem:  # pragma: no cover - degenerate cell
        fit, A_raw = None, None
        latent = {"status": "not available", "reason": f"{type(problem).__name__}: {problem}"}
    if A_raw is not None:
        repaired = dict(c_test)
        repaired["events"] = (np.asarray(c_test["events"], float)
                              - u_test @ np.asarray(A_raw, float).T)
        r2_errors = _vertex_errors(n_train, repaired, frozen_vertex)[index_ce]
        r2_difference = _difference_reading(n_errors, r2_errors, runs_e)
        rho = (None if abs(v_c - v_n) <= 0
               else float((v_c - _mean(r2_errors)) / (v_c - v_n)) if eligible else None)
        latent = {"status": "evaluated", "r2_eq": fit["r2_eq"],
                  "r2_eq_class": er.equivariance_class(fit["r2_eq"]),
                  "v_R2_m": _mean(r2_errors), "rho_R2": rho,
                  "class": er.latent_repair_class(rho),
                  "difference_vs_N_m": _mean(r2_errors) - v_n,
                  "ci_vs_N_m": r2_difference["ci95"],
                  "n_train": fit["n_train"], "n_test": fit["n_test"]}
    out["latent"] = latent
    # ---- R1 input repair (§5 P2-c, T1.0w only)
    out["r1"] = {"status": "not applicable"}
    if condition.name == "T1.0w" and data.get("r1_test") is not None:
        r1_test, t10_test = data["r1_test"], data.get("t10_test")
        if r1_test is not None and t10_test is not None:
            index_n1, index_r1, runs_1 = _align_subset(n_test, r1_test)
            index_nt10, index_t10, runs_t10 = _align_subset(n_test, t10_test)
            index_nw, index_w, runs_w = _align_subset(n_test, c_test)
            common = (set(zip(runs_w, np.asarray(c_test["event_ids"])[index_w]))
                      & set(zip(runs_t10, np.asarray(t10_test["event_ids"])[index_t10]))
                      & set(zip(runs_1, np.asarray(r1_test["event_ids"])[index_r1])))
            if common:
                pairs = sorted(common)
                v_w = _errors_on_keys(n_test, c_test, frozen_vertex, pairs)
                v_t10 = _errors_on_keys(n_test, t10_test, frozen_vertex, pairs)
                v_r1 = _errors_on_keys(n_test, r1_test, frozen_vertex, pairs)
                runs_c = np.asarray([key[0] for key in pairs], dtype=object)
                v_n_common = _errors_on_keys(n_test, n_test, frozen_vertex, pairs)
                bootstrap = _paired(v_t10, v_r1, runs_c)
                denominator = _mean(v_w) - _mean(v_n_common)
                walk_share = (None if abs(denominator) <= 0
                              else float((_mean(v_w) - _mean(v_t10)) / denominator))
                rho_r1 = (None if abs(denominator) <= 0
                          else float((_mean(v_w) - _mean(v_r1)) / denominator))
                out["r1"] = {
                    "status": "evaluated", "n_events": int(len(pairs)),
                    "v_T10w_m": _mean(v_w), "v_T10_m": _mean(v_t10),
                    "v_R1_m": _mean(v_r1),
                    "walk_share": walk_share, "rho_R1": rho_r1,
                    "v_R1_minus_v_T10_m": float(_mean(v_r1) - _mean(v_t10)),
                    "ci_vs_T10_m": [float(bootstrap["ci95_m"][0]),
                                    float(bootstrap["ci95_m"][1])],
                    "within_tolerance": bool(
                        abs(float(bootstrap["mean_difference_m"])) <= P2C_RECOVERY_TOL_M
                        and abs(bootstrap["ci95_m"][0]) <= P2C_RECOVERY_TOL_M
                        and abs(bootstrap["ci95_m"][1]) <= P2C_RECOVERY_TOL_M),
                    "class": ("invertible part recovered"
                              if abs(float(bootstrap["mean_difference_m"]))
                              <= P2C_RECOVERY_TOL_M
                              and abs(bootstrap["ci95_m"][0]) <= P2C_RECOVERY_TOL_M
                              and abs(bootstrap["ci95_m"][1]) <= P2C_RECOVERY_TOL_M
                              else "not recovered")}
    # ---- raw-feature reference (§5 P2-b)
    out["raw"] = {"status": "not available"}
    if frozen_raw_vertex is not None and data.get("raw_n_train") is not None:
        raw = {"frozen": {}, "refit": {}}
        n_raw_train = data["raw_n_train"]
        n_raw_test = data.get("raw_n_test")
        c_raw_train = data.get("raw_c_train")
        c_raw_test = data.get("raw_c_test")
        if n_raw_test is not None and c_raw_test is not None:
            index_n, index_c, runs = _align_subset(n_raw_test, c_raw_test)
            frozen_errors = _vertex_errors(n_raw_train, n_raw_test,
                                           frozen_raw_vertex)[index_n]
            c_errors = _vertex_errors(n_raw_train, c_raw_test,
                                      frozen_raw_vertex)[index_c]
            difference = _difference_reading(frozen_errors, c_errors, runs)
            raw["frozen"] = {"v_N_m": _mean(frozen_errors), "v_c_m": _mean(c_errors),
                             "delta_m": difference["difference"],
                             "ci_m": difference["ci95"]}
            if c_raw_train is not None:
                refit = er.fit_vertex_probe(c_raw_train)
                refit_errors = _vertex_errors(c_raw_train, c_raw_test, refit)[index_c]
                refit_difference = _difference_reading(frozen_errors, refit_errors, runs)
                raw["refit"] = {"v_R3_m": _mean(refit_errors),
                                "R_c_m": float(_mean(refit_errors) - _mean(frozen_errors)),
                                "ci_m": refit_difference["ci95"],
                                "information_destroying": bool(
                                    _mean(refit_errors) - _mean(frozen_errors)
                                    >= REFIT_DESTROYED_M
                                    and refit_difference["excludes_zero"])}
        out["raw"] = raw
    # ---- R5 abstention (§5 P2-g, descriptive)
    out["abstention"] = abstention_reading(n_train, n_test, c_test, frozen_vertex)
    return out


def _errors_on_keys(n_test, rows, fitted, keys) -> np.ndarray:
    errors = _vertex_errors(n_test, rows, fitted)
    lookup = {(str(run), int(event)): index
              for index, (run, event) in enumerate(zip(np.asarray(rows["run_ids"]).astype(str),
                                                       np.asarray(rows["event_ids"], np.int64)))}
    return np.asarray([errors[lookup[key]] for key in keys], float)


# ============================================================ raw features §5
def _raw_rows(rows) -> dict:
    """Adapt ``ext_repair.raw_photon_features`` output to the probe row schema."""
    if rows is None:
        return None
    return {"events": np.asarray(rows["features"], float),
            "run_ids": np.asarray(rows["run_ids"]).astype(str),
            "event_ids": np.asarray(rows["event_ids"], np.int64),
            "vertex": np.asarray(rows["vertex"], float),
            "log_yield": np.asarray(rows["log_yield"], float)}


def _raw_vertex_probe(rows) -> dict:
    return la.fit_probe(la.ProbeDataset(
        kind="vertex", form="raw", x=np.asarray(rows["events"], float),
        y=np.asarray(rows["vertex"], float),
        run_ids=np.asarray(rows["run_ids"], dtype=object),
        target_names=("x", "y", "z"), meta={}))


def _raw_vertex_errors(train_rows, test_rows, fitted) -> np.ndarray:
    test = la.ProbeDataset(
        kind="vertex", form="raw", x=np.asarray(test_rows["events"], float),
        y=np.asarray(test_rows["vertex"], float),
        run_ids=np.asarray(test_rows["run_ids"], dtype=object),
        target_names=("x", "y", "z"), meta={})
    prediction = la.probe_predictions(fitted, test)
    return np.linalg.norm(np.asarray(prediction, float) - test.y, axis=1)


def build_raw_features(probes005, drift, geometry, cfg, *, log=print) -> dict:
    """raw-std (§7.2) of N and of every condition, from the transformed stores."""
    out = {("N", "train"): _raw_rows(er.raw_photon_features(probes005["train"], geometry)),
           ("N", "test"): _raw_rows(er.raw_photon_features(probes005["test"], geometry))}
    for name, condition in drift.items():
        for role in ("train", "test"):
            with hit_set_context(probes005[role], condition, geometry, cfg):
                out[(name, role)] = _raw_rows(
                    er.raw_photon_features(probes005[role], geometry))
        log(f"[p2] raw features {name}: train {out[(name, 'train')]['events'].shape[0]} "
            f"test {out[(name, 'test')]['events'].shape[0]}")
    return out


def _restrict_raw(raw, condition_name, lookup_train, lookup_test) -> tuple:
    train = raw.get((condition_name, "train"))
    test = raw.get((condition_name, "test"))
    if train is not None:
        train = _subset_rows(train, _mask_rows(train, _trig_lookup(lookup_train)))
    if test is not None:
        test = _subset_rows(test, _mask_rows(test, _trig_lookup(lookup_test)))
    return train, test


# ====================================================== P2-e statistic §5
def spearman(x, y) -> float | None:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size != y.size or x.size < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_spearman(x, y, *, level, permutations=P2E_PERMUTATIONS) -> dict:
    """Spearman with a one-sided keyed permutation p (``paper3/p2/perm``)."""
    observed = spearman(x, y)
    if observed is None:
        return {"rho_s": None, "p": None, "n": int(np.asarray(x).size),
                "permutations": int(permutations), "status": "not evaluable"}
    y = np.asarray(y, float)
    exceed = 0
    for index in range(int(permutations)):
        rng = lb.keyed_rng(SEED, PERM_NAMESPACE, str(level), str(index))
        permuted = y[rng.permutation(y.size)]
        value = spearman(x, permuted)
        if value is not None and value >= observed:
            exceed += 1
    p_value = (exceed + 1.0) / (int(permutations) + 1.0)
    return {"rho_s": observed, "p": float(p_value), "n": int(np.asarray(x).size),
            "permutations": int(permutations), "exceed": int(exceed),
            "one_sided": True, "seed": SEED, "namespace": PERM_NAMESPACE}


def p2e_class(levels: dict) -> dict:
    """§5 P2-e fixed class rule (supported / refuted / mixed / not evaluable)."""
    ns = [levels[level].get("n") for level in ("K5", "K15", "K30")]
    rho = [levels[level].get("rho_s") for level in ("K5", "K15", "K30")]
    p = [levels[level].get("p") for level in ("K5", "K15", "K30")]
    if any(n is None or int(n) < 3 for n in ns):
        return {"verdict": "not evaluable",
                "reason": "a K level has fewer than 3 eligible models",
                "n": dict(zip(("K5", "K15", "K30"), ns)),
                "rho_s": dict(zip(("K5", "K15", "K30"), rho)),
                "p": dict(zip(("K5", "K15", "K30"), p)),
                "rule": "not evaluable when a K level has fewer than 3 eligible models"}
    supported = sum(1 for r, q in zip(rho, p)
                    if r is not None and q is not None and r >= P2E_RHO_SUPPORT
                    and q <= P2E_P_SUPPORT)
    refuted = all(r is not None and r <= P2E_RHO_REFUTE for r in rho)
    verdict = ("supported" if supported >= 2 else "refuted" if refuted else "mixed")
    return {"verdict": verdict, "n_levels_supporting": int(supported),
            "rho_s": dict(zip(("K5", "K15", "K30"), rho)),
            "p": dict(zip(("K5", "K15", "K30"), p)),
            "rule": "supported: rho_s >= 0.5 and p <= 0.05 in at least two of the "
                    "three K levels; refuted: rho_s <= 0.1 in all three; else mixed"}


def gain_r2_eq_table(root=None) -> dict:
    """The §5 P2-e x-values: Part A X2-b occupancy R²_eq (corrected), per model."""
    root = _resolve_root(root)
    table = {}
    shortcut = read_json(root / "results/jepa/20261107_shortcut/readings.json", {}) or {}
    rows = ((shortcut.get("r1") or {}).get("table") or {})
    for label, row in rows.items():
        value = row.get("r2_eq")
        if value is not None:
            table[str(label)] = float(value)
    summary = read_json(root / "results/jepa/20261106_repair/SUMMARY.json", {}) or {}
    families = (summary.get("x2") or {})
    fallback = {}
    for family, seeds in families.items():
        if not isinstance(seeds, dict):
            continue
        for seed, block in seeds.items():
            if not isinstance(block, dict):
                continue
            value = (((block.get("X2-b") or {}).get("r2_cs")))
            label = f"{family}{seed}"
            if value is not None:
                fallback[label] = float(value)
    for label in er.CORE_MODELS:
        if label not in table and label in fallback:
            table[label] = fallback[label]
    for label in NEW_MODELS:
        payload = read_json(Path(EXT_ROOT) / "x2b" / f"{label}.json", None)
        if payload is not None and payload.get("r2_eq") is not None:
            table[label] = float(payload["r2_eq"])
    return table


# ============================================================ p2 mode §8.5
def _cell_for_model(label, cache, n_cache, drift, infos, raw, hbar, geometry, *, exact,
                    log=print) -> dict:
    n_train = cache_rows(n_cache, er.embedding_cache_key("train", "N"))
    n_test = cache_rows(n_cache, er.embedding_cache_key("test", "N"))
    if n_train is None or n_test is None:
        return {"status": "unavailable", "reason": "N embeddings missing"}
    frozen_vertex = er.fit_vertex_probe(n_train)
    frozen_yield = er.fit_log_yield_probe(n_train)
    n_raw_train, n_raw_test = raw.get(("N", "train")), raw.get(("N", "test"))
    frozen_raw = _raw_vertex_probe(n_raw_train) if n_raw_train is not None else None
    out = {}
    for name in CONDITIONS:
        c_train = cache_rows(cache, er.embedding_cache_key("train", name))
        c_test = cache_rows(cache, er.embedding_cache_key("test", name))
        if c_train is None or c_test is None:
            out[name] = {"status": "unavailable", "reason": "condition cache missing"}
            continue
        data = {"condition": drift[name], "n_train": n_train, "n_test": n_test,
                "c_train": c_train, "c_test": c_test, "hbar": hbar,
                "geometry": geometry}
        if name == "T1.0w":
            data["r1_test"] = cache_rows(cache, er.embedding_cache_key("test", R1_NAME))
            data["t10_test"] = cache_rows(cache, er.embedding_cache_key("test", "T1.0"))
        if name in ("K5", "K15", "K30", "P10"):
            data["c_train"] = c_train; data["c_test"] = c_test
        out[name] = None
        if exact:
            train_lookup = infos[(name, "train")]
            test_lookup = infos[(name, "test")]
            mask_train = _mask_rows(c_train, _trig_lookup(train_lookup))
            mask_test = _mask_rows(c_test, _trig_lookup(test_lookup))
            data["c_train"] = _subset_rows(c_train, mask_train)
            data["c_test"] = _subset_rows(c_test, mask_test)
            if data.get("r1_test") is not None:
                data["r1_test"] = _subset_rows(
                    data["r1_test"], _mask_rows(data["r1_test"],
                                                _trig_lookup(infos[(R1_NAME, "test")])))
            if data.get("t10_test") is not None:
                data["t10_test"] = _subset_rows(
                    data["t10_test"], _mask_rows(data["t10_test"],
                                                 _trig_lookup(infos[("T1.0", "test")])))
        raw_train, raw_test = _restrict_raw(raw, name,
                                            infos[(name, "train")],
                                            infos[(name, "test")]) if exact else (
            raw.get((name, "train")), raw.get((name, "test")))
        data["raw_n_train"] = n_raw_train; data["raw_n_test"] = n_raw_test
        data["raw_c_train"] = raw_train; data["raw_c_test"] = raw_test
        try:
            out[name] = cell_reading(data, frozen_vertex=frozen_vertex,
                                     frozen_yield=frozen_yield,
                                     frozen_raw_vertex=frozen_raw)
        except (ValueError, KeyError) as problem:  # pragma: no cover - degenerate cell
            out[name] = {"status": "unavailable",
                         "reason": f"{type(problem).__name__}: {problem}"}
        out[name]["condition"] = name
        out[name]["exact_subset"] = bool(exact)
    return out


def _p2f(full: dict, condition_name: str) -> dict:
    """§5 P2-f: the cheapest rung reaching >= 80 % of R3's recovered fraction."""
    if full.get("status") == "unavailable":
        return {"status": "not available"}
    delta = full.get("delta_c_m")
    if delta is None or abs(delta) <= 0:
        return {"status": "no drift cost"}
    v_c = full["v_c_m"]
    r3 = full.get("refit") or {}
    rho_r3 = (None if r3.get("v_R3_m") is None
              else (v_c - float(r3["v_R3_m"])) / float(delta))
    rungs = [("R0", 0.0)]
    if condition_name == "T1.0w" and (full.get("r1") or {}).get("rho_R1") is not None:
        rungs.append(("R1", float(full["r1"]["rho_R1"])))
    if (full.get("latent") or {}).get("rho_R2") is not None:
        rungs.append(("R2", float(full["latent"]["rho_R2"])))
    if rho_r3 is not None:
        rungs.append(("R3", float(rho_r3)))
    target = None if rho_r3 is None else P2F_SHARE * rho_r3
    cheapest = None
    for rung, rho in rungs:
        if target is not None and rho >= target:
            cheapest = rung
            break
    flag = (full.get("refit") or {}).get("information_destroying")
    return {"status": "evaluated", "rho_R0": 0.0,
            "rho_R1": (full.get("r1") or {}).get("rho_R1"),
            "rho_R2": (full.get("latent") or {}).get("rho_R2"),
            "rho_R3": rho_r3, "target": target, "cheapest_rung": cheapest,
            "beyond_refit": bool(flag),
            "note": ("beyond refit (R4/R5 territory)" if flag else None)}


def p2(root=None, *, config, gen, arms_config, geometry,
       data_root=DATA_ROOT_DEFAULT, device="cpu", inventory=None, limit=None,
       log=print) -> dict:
    """§5 P2 readings (CPU, from the sha-verified cached embeddings)."""
    root = _resolve_root(root)
    gate = assert_authorised("p2", root=root, log=log)
    started = time.time()
    plan = require_preflight_plan(root)
    require_stage("conditions.json", message="p2 needs the conditions stage")
    del plan
    inv = inventory if inventory is not None else p2_inventory(root, hash_files=True)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    drift = build_drift_conditions()
    infos = {}
    for name in list(CONDITIONS) + [R1_NAME]:
        for role in ("train", "test"):
            infos[(name, role)] = load_transform_info(EXT_ROOT, role, name)
    raw = build_raw_features(probes005, drift, geometry, gen, log=log)
    gain_table = gain_r2_eq_table(root)
    # the N-train mean hit count (h̄) is a property of the N set, not the model
    hbar = None
    n0 = cache_rows(n_embedding_cache(CORE_MODELS[0]), er.embedding_cache_key("train", "N"))
    if n0 is not None and np.asarray(n0["n_hit"]).size:
        hbar = float(np.mean(np.asarray(n0["n_hit"], float)))
    if hbar is None:
        raise RuntimeError("p2: the N-train hit counts are unavailable")
    payload = {"registered": False, "extension": EXTENSION,
               "check": "P2 readings §5", "runbook": RUNBOOK_REL,
               "models": {}, "conditions": list(CONDITIONS), "hbar": hbar,
               "gain_r2_eq_source": "results/jepa/20261107_shortcut/readings.json "
                                    "r1.table (D1 overlay); new models from <ext>/x2b",
               "gate": {"status": gate["status"]}, "wall_seconds": None}
    for label in CORE_MODELS:
        row = model_row(inv, label)
        nc = n_embedding_cache(label)
        cache = embedding_cache(label)
        full = _cell_for_model(label, cache, nc, drift, infos, raw, hbar, geometry,
                               exact=False, log=log)
        exact = _cell_for_model(label, cache, nc, drift, infos, raw, hbar, geometry,
                                exact=True, log=log)
        conditions_out = {}
        for name in CONDITIONS:
            cell = full.get(name) if isinstance(full, dict) else None
            if not isinstance(cell, dict) or cell.get("status") == "unavailable":
                conditions_out[name] = {"status": "unavailable",
                                        "reason": (cell or {}).get("reason")}
                continue
            cell = dict(cell)
            cell["exact_subset_reading"] = (
                exact.get(name) if isinstance(exact, dict) else None)
            cell["P2-f"] = _p2f(cell, name)
            conditions_out[name] = cell
        payload["models"][label] = {
            "label": label, "split": split_of(row),
            "gain_r2_eq": gain_table.get(label),
            "gain_r2_eq_source": ("part_a_r1_table" if label not in NEW_MODELS
                                  else "<ext>/x2b"),
            "conditions": conditions_out}
    payload["P2-e"] = _p2e(payload["models"])
    payload["P2-b"] = _p2b(payload["models"])
    payload["P2-g"] = _p2g(payload["models"])
    missing = [label for label in CORE_MODELS
               if payload["models"].get(label, {}).get("gain_r2_eq") is None]
    missing_conditions = [label for label, entry in payload["models"].items()
                          if not all((entry.get("conditions") or {}).get(name, {}).get(
                              "status") != "unavailable" for name in CONDITIONS)]
    payload["missing_gain_r2_eq"] = missing
    payload["missing_conditions"] = missing_conditions
    payload["complete"] = bool(not missing and not missing_conditions
                               and len(payload["models"]) == len(CORE_MODELS))
    payload["pass"] = bool(payload["complete"])
    payload["closure"] = assert_closure("p2", root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(EXT_ROOT) / "p2.json", payload)
    record_job("p2", "cpu", time.time() - started)
    return payload


def _missing_gain(models) -> list:
    return [label for label, entry in models.items() if entry.get("gain_r2_eq") is None]


def _eligible_cells(models, level) -> tuple:
    xs, ys, used = [], [], []
    for label, entry in models.items():
        x = entry.get("gain_r2_eq")
        cell = (entry.get("conditions") or {}).get(level) or {}
        latent = cell.get("latent") or {}
        y = latent.get("rho_R2")
        if x is None or y is None or not cell.get("delta_eligible"):
            continue
        xs.append(float(x)); ys.append(float(y)); used.append(label)
    return np.asarray(xs), np.asarray(ys), used


def _p2e(models) -> dict:
    out = {"primary": {}, "secondary_T1.0": {},
           "n_models_expected": len(CORE_MODELS),
           "n_models_in_table": len(models),
           "missing_gain_r2_eq": _missing_gain(models)}
    for level in ("K5", "K15", "K30"):
        x, y, used = _eligible_cells(models, level)
        stats = permutation_spearman(x, y, level=level) if x.size >= 3 else {
            "rho_s": None, "p": None, "n": int(x.size),
            "permutations": P2E_PERMUTATIONS, "status": "fewer than 3 eligible models"}
        out["primary"][level] = dict(stats, models=used)
    out["class"] = p2e_class(out["primary"])
    if out["missing_gain_r2_eq"] or len(models) != len(CORE_MODELS):
        out["class"]["verdict"] = "not evaluable"
        out["class"]["reason"] = ("gain R2_eq missing for: "
                                  + ", ".join(out["missing_gain_r2_eq"]))
    x, y, used = _eligible_cells(models, "T1.0")
    out["secondary_T1.0"] = dict(
        (permutation_spearman(x, y, level="T1.0") if x.size >= 3
         else {"rho_s": None, "p": None, "n": int(x.size),
               "permutations": P2E_PERMUTATIONS, "status": "fewer than 3 eligible models"}),
        models=used)
    # descriptive: is equivariance a property of the representation across factors?
    pairs = []
    for label, entry in models.items():
        cell = (entry.get("conditions") or {}).get("K15") or {}
        latent = cell.get("latent") or {}
        if entry.get("gain_r2_eq") is not None and latent.get("r2_eq") is not None:
            pairs.append((float(entry["gain_r2_eq"]), float(latent["r2_eq"]), label))
    out["secondary_equivariance_K15"] = {
        "rho_s": (spearman([p[0] for p in pairs], [p[1] for p in pairs])
                  if len(pairs) >= 3 else None),
        "n": len(pairs), "models": [p[2] for p in pairs],
        "question": "Spearman between gain R2_eq and R2_eq under K15 (§5 secondary)"}
    return out


def _p2b(models) -> dict:
    """§5 P2-b: per condition, the cells whose refit residual destroys information."""
    out = {}
    for name in CONDITIONS:
        destroying, recoverable, n = [], [], 0
        for label, entry in models.items():
            cell = (entry.get("conditions") or {}).get(name) or {}
            refit = cell.get("refit") or {}
            if refit.get("information_destroying") is None:
                continue
            n += 1
            (destroying if refit["information_destroying"] else recoverable).append(label)
        out[name] = {"information_destroying": destroying,
                     "refit_recoverable": recoverable, "n_cells": n,
                     "n_destroying": len(destroying)}
    return out


def _p2g(models) -> dict:
    out = {}
    for name in ("K30", "T1.0"):
        rows = {}
        for label, entry in models.items():
            cell = (entry.get("conditions") or {}).get(name) or {}
            if cell.get("abstention"):
                rows[label] = cell["abstention"]
        out[name] = rows
    return out


# ============================================================ X3-P mode §6
def _metric_block(train_rows, test_rows, kind) -> dict | None:
    """One metric's frozen-probe block on one set (probe fit on ``train_rows``)."""
    if train_rows is None or test_rows is None:
        return None
    if kind == "vertex":
        fitted = er.fit_vertex_probe(train_rows)
        metrics = er.vertex_metrics(train_rows, test_rows, fitted=fitted)
        # ``_metric_rows`` (reused) reads ``mean_error_m`` for vertex and
        # ``resolution`` otherwise; emit both so the Part A X3 helpers work.
        return {"resolution": metrics["mean_error_m"],
                "mean_error_m": metrics["mean_error_m"],
                "errors": metrics["errors"], "run_ids": metrics["run_ids"],
                "event_ids": metrics["event_ids"], "n_events": metrics["n_events"]}
    if kind == "log_yield":
        fitted = er.fit_log_yield_probe(train_rows)
        metrics = er.log_yield_metrics(train_rows, test_rows, fitted=fitted)
        return {"resolution": metrics["resolution"], "errors": metrics["errors"],
                "run_ids": metrics["run_ids"], "event_ids": metrics["event_ids"],
                "n_events": metrics["n_events"]}
    if kind == "energy":
        fitted = er.fit_energy_probe(train_rows)
        metrics = er.energy_metrics(train_rows, test_rows, fitted=fitted)
        if metrics.get("status") != "evaluated":
            return None
        return {"resolution": metrics["resolution"], "errors": metrics["errors"],
                "run_ids": metrics["run_ids"], "event_ids": metrics["event_ids"],
                "n_events": int(np.asarray(metrics["errors"]).size)}
    raise ValueError(f"unknown metric {kind!r}")


def _erase_rows(rows, eraser, *, U_r=None) -> dict | None:
    if rows is None:
        return None
    out = dict(rows)
    out["events"] = er.leace_erase(eraser, np.asarray(rows["events"], float), U_r=U_r)
    return out


def _x3p_blocks(rows, *, kinds=("vertex", "log_yield")) -> dict:
    """Build a variant block {metric: {"N": block, "D": block}} from its rows."""
    block = {}
    for kind in kinds:
        if kind == "energy":
            continue        # F1: energy uses the particle rows below
        block[kind] = {"N": _metric_block(rows.get("train_N"), rows.get("test_N"), kind),
                       "D": _metric_block(rows.get("train_N"), rows.get("test_D"), kind)}
    if "energy" in kinds:
        block["energy"] = {
            "N": _metric_block(rows.get("particle_train_N"), rows.get("particle_test_N"),
                               "energy"),
            "D": _metric_block(rows.get("particle_train_N"), rows.get("particle_test_D"),
                               "energy")}
    return block


def _calibrated_block(rows) -> dict:
    """The calibrated unerased model: the N-train probe applied to SWunit at D."""
    out = {"vertex": {}, "log_yield": {}}
    if rows.get("train_N") is not None and rows.get("test_SWunit") is not None:
        out["vertex"]["D"] = _metric_block(rows["train_N"], rows["test_SWunit"], "vertex")
        out["log_yield"]["D"] = _metric_block(rows["train_N"], rows["test_SWunit"],
                                              "log_yield")
    if rows.get("particle_train_N") is not None and rows.get("particle_test_SWunit") is not None:
        block = _metric_block(rows["particle_train_N"], rows["particle_test_SWunit"],
                              "energy")
        if block is not None:
            out["energy"] = {"D": block}
    return out


def _x2b_for(label, cache, geometry, delta, occupancy, *, log=print) -> dict:
    """The X2-b occupancy fit of one X3-P model (raw-unit Â, corrected map)."""
    if label in NEW_MODEL_SPECS:
        stored = read_json(Path(EXT_ROOT) / "x2b" / f"{label}.json", None)
        if stored is not None and stored.get("A_hat") is not None:
            return {"r2_eq": stored.get("r2_eq"), "A_hat": np.asarray(stored["A_hat"], float),
                    "source": "<ext>/x2b"}
    rows = {role: {} for role in ("train", "test")}
    for role in ("train", "test"):
        rows[role]["N"] = cache_rows(cache, er.embedding_cache_key(role, "N"))
        rows[role]["S"] = cache_rows(cache, er.embedding_cache_key(role, "S"))
    if any(rows[role][name] is None for role in ("train", "test") for name in ("N", "S")):
        return {"r2_eq": None, "A_hat": None, "source": "missing cache"}
    fit = _x2b_fit(rows, occupancy, {"train": delta, "test": delta}, geometry, log=log)
    return {"r2_eq": fit["r2_eq"],
            "A_hat": None if fit["A_hat"] is None else np.asarray(fit["A_hat"], float),
            "source": "refit from cache (raw_map)"}


def x3p(root=None, *, config, gen, arms_config, geometry,
        data_root=DATA_ROOT_DEFAULT, device="cpu", inventory=None,
        labels=None, limit=None, log=print) -> dict:
    """§6 X3-P: the price of invariance by projecting out the gain directions (CPU)."""
    root = _resolve_root(root)
    gate = assert_authorised("x3p", root=root, log=log)
    started = time.time()
    require_preflight_plan(root)
    inv = inventory if inventory is not None else p2_inventory(root, hash_files=True)
    probes005 = open_probe_sets(config, gen, geometry, data_root, setting=S005, log=log)
    probes030 = open_probe_sets(config, gen, geometry, data_root, setting=S030, log=log)
    delta = er.run_log_gain_vectors(probes030["train"], geometry)
    delta.update(er.run_log_gain_vectors(probes030["test"], geometry))
    occupancy = {role: er.occupancy_counts(
        probes005[role], er.build_condition("N", probes005[role], geometry), geometry)
        for role in ("train", "test")}
    payload = {"registered": False, "extension": EXTENSION, "check": "X3-P §6",
               "runbook": RUNBOOK_REL, "ranks": list(X3P_RANKS),
               "controls": X3P_CONTROLS, "models": {},
               "gate": {"status": gate["status"]}, "wall_seconds": None,
               "raw_feature_price": X3P_RAW_FEATURE_PRICE}
    fisher = read_json(root / "results/jepa/20261106_repair/x3_fisher.json", None)
    fisher_rows = None
    if fisher:
        model_rows = {}
        for name, entry in (fisher.get("models") or {}).items():
            if not isinstance(entry, dict):
                continue
            model_rows[str(name)] = {
                "median_crlb": entry.get("median_crlb"),
                "sigma90": entry.get("sigma90"),
                "n_events": entry.get("n_events")}
        fisher_rows = {"source": "results/jepa/20261106_repair/x3_fisher.json",
                       "theta": fisher.get("theta"), "sigmas": fisher.get("sigmas"),
                       "models": model_rows,
                       "underfull_caveat": "descriptive; the Part A Fisher rows "
                                           "are reported beside rho_E, unchanged"}
    payload["X3P-4"] = {"raw_feature_price": X3P_RAW_FEATURE_PRICE,
                        "fisher": fisher_rows,
                        "note": "rho_E beside the Fisher rows and the raw-feature "
                                "3.56 (descriptive)"}
    selected = list(X3P_MODELS if labels is None else labels)
    for label in selected:
        row = model_row(inv, label)
        cache = n_embedding_cache(label)
        rows = {"train_N": cache_rows(cache, er.embedding_cache_key("train", "N")),
                "test_N": cache_rows(cache, er.embedding_cache_key("test", "N")),
                "test_D": cache_rows(cache, er.embedding_cache_key("test", "D")),
                "test_SWunit": cache_rows(cache, er.embedding_cache_key("test", "SWunit"))}
        if label in er.MAE_MODELS:
            rows["particle_train_N"] = cache_rows(cache, "particle__train__N")
            rows["particle_test_N"] = cache_rows(cache, "particle__test__N")
            rows["particle_test_D"] = cache_rows(cache, "particle__test__D")
            rows["particle_test_SWunit"] = cache_rows(cache, "particle__test__SWunit")
        if any(rows[key] is None for key in ("train_N", "test_N", "test_D")):
            payload["models"][label] = {"status": "unavailable",
                                        "reason": "N/D embedding cache missing"}
            continue
        x2b = _x2b_for(label, cache, geometry, delta, occupancy, log=log)
        if x2b["A_hat"] is None:
            payload["models"][label] = {"status": "unavailable",
                                        "reason": "X2-b occupancy fit unavailable"}
            continue
        events = np.asarray(rows["train_N"]["events"], float)
        whitening = er.whitening_matrix(events, np.asarray(rows["train_N"]["run_ids"]))
        W, W_inv, mu = whitening["W"], whitening["W_inv"], events.mean(axis=0)
        _, singular, Vt = np.linalg.svd(W @ np.asarray(x2b["A_hat"], float))
        d = W.shape[0]
        kinds = ["vertex", "log_yield"]
        if (rows.get("particle_train_N") is not None
                and rows.get("particle_test_N") is not None):
            kinds.append("energy")
        standard = _x3p_blocks(rows, kinds=kinds)
        calibrated = _calibrated_block(rows)
        entry = {"status": "evaluated", "split": split_of(row),
                 "x2b_source": x2b["source"], "x2b_r2_eq": x2b["r2_eq"],
                 "gain_singular_values": [float(value) for value in singular[:12]],
                 "mu_N_norm": float(np.linalg.norm(mu)), "embedding_dim": int(d),
                 "price": {}, "robustness": {}, "dominance": {},
                 "gain_specific": {}}
        for r in X3P_RANKS:
            if r > d:
                continue
            U_r = np.linalg.svd(W @ np.asarray(x2b["A_hat"], float))[0][:, :int(r)]
            eraser = {"mu_x": mu, "W": W, "W_inv": W_inv, "U_r": U_r}
            erased = {key: _erase_rows(value, eraser) for key, value in rows.items()}
            variant = _x3p_blocks(erased, kinds=kinds)
            controls = []
            for index in range(X3P_CONTROLS):
                rng = lb.keyed_rng(SEED, f"{X3P_NAMESPACE}/{label}/{r}/{index}")
                basis = er.random_control_basis(rng, int(d), int(r))
                control_block = {"log_yield": {}, "vertex": {}}
                erased_control = {key: _erase_rows(value, eraser, U_r=basis)
                                  for key, value in rows.items()}
                if "energy" in kinds:
                    control_block["energy"] = {
                        "N": _metric_block(erased_control.get("particle_train_N"),
                                           erased_control.get("particle_test_N"), "energy")}
                control_block["log_yield"]["N"] = _metric_block(
                    erased_control.get("train_N"), erased_control.get("test_N"),
                    "log_yield")
                controls.append(control_block)
            reading = er._variant_reading({"calibrated": calibrated}, standard, variant,
                                          leace={"controls": controls, "concept": "P2"})
            entry["price"][str(r)] = reading.get("X3-1")
            entry["robustness"][str(r)] = reading.get("X3-3")
            entry["dominance"][str(r)] = reading.get("X3-4")
            entry["gain_specific"][str(r)] = reading.get("X3-1-gain_specific")
            entry.setdefault("X3P-2", {})[str(r)] = _x3p2(variant)
        payload["models"][label] = entry
    payload["X3P-classes"] = _x3p_classes(payload["models"])
    payload["X3P-4"]["rho_E_beside_fisher"] = {
        label: {str(r): (((entry.get("price") or {}).get(str(r)) or {}).get(
            "log_yield") or {}).get("rho_E")
                for r in X3P_RANKS}
        for label, entry in payload["models"].items()
        if entry.get("status") == "evaluated"}
    payload["complete"] = all(entry.get("status") == "evaluated"
                              for entry in payload["models"].values())
    payload["pass"] = bool(payload["complete"])
    payload["closure"] = assert_closure("x3p", root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(EXT_ROOT) / "x3p.json", payload)
    record_job("x3p", "cpu", time.time() - started)
    return payload


def _x3p2(variant) -> dict:
    """§6 X3P-2: v_erased(D) - v_erased(N) with the erased frozen probe."""
    n_block = (variant.get("vertex") or {}).get("N")
    d_block = (variant.get("vertex") or {}).get("D")
    out = {"delta_vertex_m": None, "ci_m": None, "invariance_achieved": None,
           "energy_relative_change": None}
    if n_block is None or d_block is None:
        return out
    aligned = er._align_metric_pair(n_block, d_block)
    if aligned is not None:
        n_errors, d_errors, run_ids = aligned
        bootstrap = ef.paired_run_bootstrap(n_errors, d_errors, run_ids)
        out["delta_vertex_m"] = float(bootstrap["mean_difference_m"])
        out["ci_m"] = [float(bootstrap["ci95_m"][0]), float(bootstrap["ci95_m"][1])]
        out["invariance_achieved"] = bool(
            abs(out["delta_vertex_m"]) <= X3P_ROBUST_VERTEX_M
            and out["ci_m"][0] >= -X3P_ROBUST_VERTEX_CI_M
            and out["ci_m"][1] <= X3P_ROBUST_VERTEX_CI_M)
    e_n = (variant.get("energy") or {}).get("N")
    e_d = (variant.get("energy") or {}).get("D")
    if e_n is not None and e_d is not None and e_n.get("resolution"):
        out["energy_relative_change"] = float(
            (e_d["resolution"] - e_n["resolution"]) / e_n["resolution"])
    return out


def _x3p_classes(models) -> dict:
    """Per r: the §6 X3-P-1/2/3 classes and the achieved-invariance ranks."""
    out = {}
    for r in X3P_RANKS:
        price_present, no_price, unclear = [], [], []
        invariance, not_invariance = [], []
        dominance = {}
        for label, entry in models.items():
            if entry.get("status") != "evaluated":
                continue
            price = (entry.get("price") or {}).get(str(r)) or {}
            block = price.get("log_yield") or {}
            klass = block.get("class")
            if klass == "price present":
                price_present.append(label)
            elif klass == "no price":
                no_price.append(label)
            elif klass is not None:
                unclear.append(label)
            x3p2 = (entry.get("X3P-2") or {}).get(str(r)) or {}
            if x3p2.get("invariance_achieved") is True:
                invariance.append(label)
            elif x3p2.get("invariance_achieved") is False:
                not_invariance.append(label)
            verdict = ((entry.get("dominance") or {}).get(str(r)) or {}).get("verdict")
            dominance.setdefault(str(verdict), []).append(label)
        out[str(r)] = {"price_present": price_present, "no_price": no_price,
                       "unclear": unclear,
                       "invariance_achieved": invariance,
                       "invariance_not_achieved": not_invariance,
                       "dominance": dominance}
    return out


# ============================================================ report §8.6
def _compact_cells(p2_payload) -> dict:
    out = {}
    for label, entry in (p2_payload.get("models") or {}).items():
        cells = {}
        for name, cell in (entry.get("conditions") or {}).items():
            if cell.get("status") == "unavailable":
                cells[name] = {"status": "unavailable"}
                continue
            refit = cell.get("refit") or {}
            latent = cell.get("latent") or {}
            exact = cell.get("exact_subset_reading") or {}
            cells[name] = {
                "n_events": cell.get("n_events"),
                "v_N_m": cell.get("v_N_m"), "v_c_m": cell.get("v_c_m"),
                "delta_c_m": cell.get("delta_c_m"), "delta_ci_m": cell.get("delta_ci_m"),
                "eligible": cell.get("delta_eligible"),
                "refit": {"v_R3_m": refit.get("v_R3_m"), "R_c_m": refit.get("R_c_m"),
                          "ci_m": refit.get("ci_m"),
                          "information_destroying": refit.get("information_destroying")},
                "latent": {"r2_eq": latent.get("r2_eq"), "rho_R2": latent.get("rho_R2"),
                           "v_R2_m": latent.get("v_R2_m"),
                           "class": latent.get("class")},
                "r1": cell.get("r1"),
                "P2-f": cell.get("P2-f"),
                "exact_subset": ({"n_events": exact.get("n_events"),
                                  "delta_c_m": exact.get("delta_c_m"),
                                  "v_c_m": exact.get("v_c_m"),
                                  "R_c_m": (exact.get("refit") or {}).get("R_c_m"),
                                  "rho_R2": (exact.get("latent") or {}).get("rho_R2")}
                                 if exact and exact.get("status") != "unavailable" else None),
            }
        out[label] = {"gain_r2_eq": entry.get("gain_r2_eq"),
                      "gain_r2_eq_source": entry.get("gain_r2_eq_source"),
                      "conditions": cells}
    return out


def _readme(summary) -> str:
    lines = ["# Paper 3 P2 — information-destroying drifts and the price of invariance",
             "",
             f"Runbook: `{RUNBOOK_REL}` (sha256 `{summary.get('runbook_sha256')}`).",
             f"Extension `{EXTENSION}`; implementation `{summary.get('impl_commit')}`.",
             "", "## X0", ""]
    x0 = summary.get("x0") or {}
    lines += [f"- data manifest pins: {'PASS' if x0.get('data_pass') else 'FAIL'}",
              f"- identity T(0.25, 0.25): {x0.get('identity_samples')} events, "
              f"{x0.get('identity_mismatches')} mismatches, "
              f"max time difference {x0.get('identity_max_time_difference_ns')} ns",
              f"- S1 (ledger + 1.3 h <= 160): {'PASS' if x0.get('s1_pass') else 'FAIL'}",
              f"- entries preflight: {'PASS' if x0.get('entries_pass') else 'FAIL'}",
              f"- GPU projection: {x0.get('gpu_projection_hours')} h of "
              f"{x0.get('gpu_cap_hours')} h",
              ""]
    x0b = summary.get("x0b") or {}
    lines += ["## X0-b reproduction", ""]
    for label, entry in (x0b.get("models") or {}).items():
        lines.append(f"- {label}: got {entry.get('got_m')} m, expected "
                     f"{entry.get('expected_m')} m, "
                     f"{'PASS' if entry.get('pass') else 'FAIL'}")
    lines += ["", "## Trigger efficiency and window-edge counts", ""]
    lines += ["| condition/role | trigger efficiency | trig'>0 | trig'<=0 | trig' median |",
              "|---|---|---|---|---|"]
    for key, entry in (summary.get("transforms") or {}).items():
        distribution = entry.get("trig_rel_distribution") or {}
        lines.append(f"| {key} | {entry.get('trigger_efficiency')} | "
                     f"{distribution.get('n_positive')} | {distribution.get('n_negative')} "
                     f"+ {distribution.get('n_zero')} zero | "
                     f"{(distribution.get('quantiles') or {}).get('q50')} |")
    lines += ["", "## P2-e (primary)", ""]
    p2e = (summary.get("p2") or {}).get("P2-e") or {}
    for level, block in (p2e.get("primary") or {}).items():
        lines.append(f"- {level}: rho_s={block.get('rho_s')}, p={block.get('p')}, "
                     f"n={block.get('n')}")
    lines.append(f"- verdict: **{(p2e.get('class') or {}).get('verdict')}**")
    lines += ["", "## Per-cell readings", ""]
    lines += ["The exact-subset columns are the §2.3 item 3 robustness reading "
              "(events with trig' <= 0) alongside the full-cell reading.",
              "",
              "| model | condition | n | v_N | v_c | delta_c | R_c (refit) | "
              "information-destroying | rho_R2 | r2_eq | exact n | exact delta_c | "
              "exact R_c | exact rho_R2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, entry in (summary.get("cells") or {}).items():
        for name, cell in (entry.get("conditions") or {}).items():
            if cell.get("status") == "unavailable":
                continue
            refit = cell.get("refit") or {}
            latent = cell.get("latent") or {}
            exact = cell.get("exact_subset") or {}
            lines.append(f"| {label} | {name} | {cell.get('n_events')} | "
                         f"{_fmt(cell.get('v_N_m'))} | "
                         f"{_fmt(cell.get('v_c_m'))} | {_fmt(cell.get('delta_c_m'))} | "
                         f"{_fmt(refit.get('R_c_m'))} | "
                         f"{refit.get('information_destroying')} | "
                         f"{_fmt(latent.get('rho_R2'))} | {_fmt(latent.get('r2_eq'))} | "
                         f"{exact.get('n_events') if exact else None} | "
                         f"{_fmt(exact.get('delta_c_m')) if exact else None} | "
                         f"{_fmt(exact.get('R_c_m')) if exact else None} | "
                         f"{_fmt(exact.get('rho_R2')) if exact else None} |")
    lines += ["", "## P2-c (T1.0w invertible part)", ""]
    for label, block in (summary.get("p2c") or {}).items():
        lines.append(f"- {label}: walk_share={_fmt(block.get('walk_share'))}, "
                     f"rho_R1={_fmt(block.get('rho_R1'))}, v_R1-v_T1.0="
                     f"{_fmt(block.get('v_R1_minus_v_T10_m'))} m, "
                     f"{block.get('class')}")
    lines += ["", "## X3-P", ""]
    x3p = summary.get("x3p") or {}
    for r, block in (x3p.get("classes") or {}).items():
        lines.append(f"- r={r}: invariance achieved "
                     f"{len(block.get('invariance_achieved') or [])} model(s), "
                     f"price present {len(block.get('price_present') or [])}, "
                     f"no price {len(block.get('no_price') or [])}")
    lines += ["", "## Resources", ""]
    resources = summary.get("resources") or {}
    lines.append(f"- GPU booked: {resources.get('gpu_hours')} h of "
                 f"{resources.get('gpu_cap')} h; CPU booked {resources.get('cpu_hours')} h "
                 f"of {resources.get('cpu_cap')} h")
    lines.append(f"- entries: free {resources.get('entries_free')}, projected "
                 f"{resources.get('entries_projected')}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value, digits=6):
    return er._fmt(value, digits)


def _plot(figure_path, draw) -> bool:
    return er._plot(figure_path, draw)


def render_figures(p2_payload, x3p_payload, out_dir, *, log=print) -> dict:
    """The four §8.6 figures (never raises; a missing matplotlib is reported)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures, skipped = [], []

    def _cells():
        return _compact_cells(p2_payload)

    def drift_ladder(axes):
        cells = _cells()
        names = [name for name in CONDITIONS]
        for index, name in enumerate(names):
            v_n = [c["conditions"][name]["v_N_m"] for c in cells.values()
                   if c["conditions"].get(name, {}).get("v_N_m") is not None]
            v_c = [c["conditions"][name]["v_c_m"] for c in cells.values()
                   if c["conditions"].get(name, {}).get("v_c_m") is not None]
            v_r2 = [c["conditions"][name]["latent"]["v_R2_m"] for c in cells.values()
                    if (c["conditions"].get(name, {}).get("latent") or {}).get("v_R2_m")
                    is not None]
            v_r3 = [c["conditions"][name]["refit"]["v_R3_m"] for c in cells.values()
                    if (c["conditions"].get(name, {}).get("refit") or {}).get("v_R3_m")
                    is not None]
            for offset, values, label in ((-0.24, v_n, "N"), (-0.08, v_c, "drift"),
                                          (0.08, v_r2, "R2"), (0.24, v_r3, "R3")):
                if values:
                    axes.bar(index + offset, float(np.mean(values)), width=0.14,
                             label=label if index == 0 else None)
        axes.set_xticks(range(len(names)))
        axes.set_xticklabels(names, rotation=45, fontsize=7)
        axes.set_ylabel("mean vertex error (m)")
        axes.set_xlabel("condition")
        axes.legend(fontsize=7)

    def p2e_scatter(axes):
        p2_payload_models = p2_payload.get("models") or {}
        colors = {"K5": "tab:blue", "K15": "tab:orange", "K30": "tab:green"}
        for level, color in colors.items():
            xs, ys = [], []
            for entry in p2_payload_models.values():
                cell = (entry.get("conditions") or {}).get(level) or {}
                latent = cell.get("latent") or {}
                if entry.get("gain_r2_eq") is not None and latent.get("rho_R2") is not None \
                        and cell.get("delta_eligible"):
                    xs.append(float(entry["gain_r2_eq"])); ys.append(float(latent["rho_R2"]))
            if xs:
                axes.scatter(xs, ys, s=18, color=color, label=level)
        axes.set_xlabel("gain R2_eq (X2-b occupancy)")
        axes.set_ylabel("rho_R2 under K drift")
        axes.legend(fontsize=7)

    def abstention(axes):
        for name, marker in (("K30", "o"), ("T1.0", "s")):
            curves = {}
            for label, entry in (p2_payload.get("models") or {}).items():
                cell = (entry.get("conditions") or {}).get(name) or {}
                abst = cell.get("abstention") or {}
                if abst:
                    curves[label] = abst
            xs, ys_c, ys_n = [], [], []
            for coverage in ("100", "80", "50"):
                values_c, values_n = [], []
                for block in curves.values():
                    values_c.append(float(block[coverage]["v_c_m"]))
                    values_n.append(float(block[coverage]["v_N_m"]))
                if values_c:
                    xs.append(float(coverage))
                    ys_c.append(float(np.mean(values_c)))
                    ys_n.append(float(np.mean(values_n)))
            if xs:
                axes.plot(xs, ys_c, marker=marker, label=f"{name} drift")
                axes.plot(xs, ys_n, marker=marker, linestyle="--",
                          label=f"{name} N at coverage")
        axes.set_xlabel("coverage (%)")
        axes.set_ylabel("mean vertex error (m)")
        axes.legend(fontsize=7)

    def x3p_price(axes):
        classes = (x3p_payload.get("X3P-classes") or x3p_payload.get("classes") or {})
        ranks = sorted(int(r) for r in classes)
        mean_price, mean_control = [], []
        for r in ranks:
            models = x3p_payload.get("models") or {}
            prices, controls = [], []
            for entry in models.values():
                if entry.get("status") != "evaluated":
                    continue
                block = ((entry.get("price") or {}).get(str(r)) or {}).get("log_yield") or {}
                if block.get("rho_E") is not None:
                    prices.append(float(block["rho_E"]))
                q95 = block.get("rho_control_q95")
                if q95 is not None:
                    controls.append(float(q95))
            mean_price.append(float(np.mean(prices)) if prices else np.nan)
            mean_control.append(float(np.mean(controls)) if controls else np.nan)
        axes.plot(ranks, mean_price, marker="o", label="rho_E (erased, log-yield)")
        axes.plot(ranks, mean_control, marker="s", linestyle="--",
                  label="control q95")
        axes.axhline(1.0, color="grey", linewidth=0.8, label="no price")
        axes.set_xlabel("rank r")
        axes.set_ylabel("rho_E")
        axes.legend(fontsize=7)

    for name, draw in (("p2_drift_ladder.png", drift_ladder),
                       ("p2e_scatter.png", p2e_scatter),
                       ("p2g_abstention.png", abstention),
                       ("x3p_price.png", x3p_price)):
        path = out_dir / name
        if _plot(path, draw):
            figures.append(str(path))
        else:
            skipped.append(name)
    return {"status": "evaluated" if figures else "unavailable",
            "figures": figures, "skipped": skipped}


def report(root=None, *, log=print) -> dict:
    """§8.6: assemble ``results/jepa/20261109_p2/`` from the stage payloads."""
    root = _resolve_root(root)
    gate = assert_authorised("report", root=root, log=log)
    started = time.time()
    ext = Path(EXT_ROOT)
    results = root / RESULTS_DIR
    results.mkdir(parents=True, exist_ok=True)
    read = lambda name: read_json(ext / name, None)
    plan = read("x0/plan.json")
    reproduction = read("x0/reproduction.json")
    inventory = read("x0/inventory.json")
    p2_payload = read("p2.json")
    x3p_payload = read("x3p.json")
    conditions_payload = read("conditions.json")
    new_models_payload = read("new_models.json")
    missing = [name for name, payload in
               (("x0/plan.json", plan), ("x0/reproduction.json", reproduction),
                ("p2.json", p2_payload), ("x3p.json", x3p_payload))
               if payload is None]
    if missing:
        raise RuntimeError(f"report: stage payloads missing: {missing}")
    stops = []
    if not p2_payload.get("pass", True):
        stops.append("P2 readings are incomplete: " + str(
            p2_payload.get("missing_gain_r2_eq") or p2_payload.get(
                "missing_conditions")))
    if not reproduction.get("pass", True):
        stops.append("X0-b reproduction did not pass")
    if float(job_hours()["gpu"]) > GPU_CAP_H:
        stops.append(f"booked GPU hours {job_hours()['gpu']:.4f} > {GPU_CAP_H} h cap")
    if stops:
        raise RuntimeError("report STOP: " + "; ".join(stops))
    for name, payload in (("plan.json", plan), ("reproduction.json", reproduction),
                          ("inventory.json", inventory), ("p2.json", p2_payload),
                          ("x3p.json", x3p_payload),
                          ("conditions.json", conditions_payload),
                          ("new_models.json", new_models_payload)):
        if payload is not None:
            write_json(results / name, payload)
    summary = {
        "registered": False, "extension": EXTENSION,
        "runbook": RUNBOOK_REL,
        "runbook_sha256": plan.get("runbook_sha256"),
        "impl_commit": plan.get("code_sha"),
        "x0": {"data_pass": (plan.get("data") or {}).get("pass"),
               "identity_samples": (plan.get("identity") or {}).get("samples"),
               "identity_mismatches": (plan.get("identity") or {}).get("mismatches"),
               "identity_max_time_difference_ns": (plan.get("identity") or {}).get(
                   "max_time_difference_ns"),
               "s1_pass": (plan.get("s1") or {}).get("pass"),
               "s1_sum_hours": (plan.get("s1") or {}).get("sum_hours"),
               "entries_pass": (plan.get("entries") or {}).get("pass"),
               "entries_free": (plan.get("entries") or {}).get("free_entries"),
               "entries_projected": (plan.get("entries") or {}).get(
                   "projected_after"),
               "gpu_projection_hours": (plan.get("gpu_projection") or {}).get(
                   "projected_l40s_hours"),
               "gpu_projection_sets": (plan.get("gpu_projection") or {}).get(
                   "total_sets"),
               "gpu_cap_hours": GPU_CAP_H,
               "inventory_counts": plan.get("inventory", {}).get("counts"),
               "identity": plan.get("identity"),
               "determinism": plan.get("determinism")},
        "x0b": reproduction,
        "transforms": plan.get("transforms"),
        "trigger_efficiency": plan.get("trigger_efficiency"),
        "window_edge": plan.get("window_edge"),
        "p2": {"P2-e": p2_payload.get("P2-e"), "P2-b": p2_payload.get("P2-b"),
               "P2-g": p2_payload.get("P2-g")},
        "cells": _compact_cells(p2_payload),
        "p2c": {label: (entry.get("conditions", {}).get("T1.0w", {}) or {}).get("r1")
                for label, entry in (p2_payload.get("models") or {}).items()},
        "x3p": {"classes": x3p_payload.get("X3P-classes"),
                "X3P-4": x3p_payload.get("X3P-4"),
                "per_model": {label: {"status": entry.get("status"),
                                      "x2b_r2_eq": entry.get("x2b_r2_eq"),
                                      "price": {r: ((b or {}).get("log_yield") or {}).get(
                                          "rho_E")
                                                for r, b in (entry.get("price") or {}).items()},
                                      "X3P-2": entry.get("X3P-2"),
                                      "dominance": entry.get("dominance")}
                              for label, entry in (x3p_payload.get("models") or {}).items()}},
        "resources": {"gpu_hours": job_hours()["gpu"], "gpu_cap": GPU_CAP_H,
                      "cpu_hours": job_hours()["cpu"], "cpu_cap": CPU_CAP_H,
                      "entries_free": (plan.get("entries") or {}).get("free_entries"),
                      "entries_projected": (plan.get("entries") or {}).get(
                          "projected_after")},
        "execution_order": list(EXECUTION_ORDER),
    }
    write_json(results / "SUMMARY.json", summary)
    write_json(results / "LARGE_ARRAYS.json", er.large_arrays(EXT_ROOT))
    (results / "READINGS_SUMMARY.md").write_text(_readme(summary))
    figures = render_figures(p2_payload, x3p_payload, results, log=log)
    payload = {"registered": False, "extension": EXTENSION, "check": "report §8.6",
               "results_dir": str(results), "figures": figures,
               "complete": bool(p2_payload.get("complete")),
               "gate": {"status": gate["status"]},
               "closure": assert_closure("report", root),
               "wall_seconds": None}
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(ext / "report.json", payload)
    record_job("report", "cpu", time.time() - started)
    return payload
