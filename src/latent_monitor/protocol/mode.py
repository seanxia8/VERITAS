# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Two modes, never mixed (plan §0.2, §0.12, §8.3 R5).

``dev`` may touch anything and produces nothing citable. ``confirmatory``
refuses to start unless every freeze part it depends on exists under
``protocol/frozen/<part>.json`` with a SHA-256 that matches the file it
hashed, and every harm threshold it needs is ``declared``. Nothing here writes
a freeze on its own initiative: :func:`freeze_part` exists so the mechanism is
testable, and the repository ships **no** frozen part — the draft
pre-registration (``docs/PREREGISTRATION.md``) is unfrozen and says so.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .consequence import HarmThreshold


class ProtocolMode(str, Enum):
    DEV = "dev"
    CONFIRMATORY = "confirmatory"


class ConfirmatoryGateError(RuntimeError):
    """Confirmatory mode requested but a required freeze or threshold is missing."""


@dataclass(frozen=True)
class FreezeRecord:
    part: str
    path: str            # the file that was hashed
    sha256: str
    commit: str

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # pragma: no cover - git absent
        return "unknown"


def freeze_part(part: str, source: Path, frozen_dir: Path, commit: str | None = None) -> FreezeRecord:
    """Write ``frozen_dir/<part>.json``. Deliberately never called by any runner in this repository."""
    rec = FreezeRecord(part=part, path=str(source), sha256=sha256_file(source), commit=commit or git_commit(source.parent))
    frozen_dir.mkdir(parents=True, exist_ok=True)
    (frozen_dir / f"{part}.json").write_text(json.dumps(rec.to_dict(), indent=2))
    return rec


def load_freeze(part: str, frozen_dir: Path) -> FreezeRecord | None:
    p = Path(frozen_dir) / f"{part}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return FreezeRecord(**d)


def require_gates(mode: ProtocolMode | str, *, freezes: Sequence[str] = (), frozen_dir: Path | str = "protocol/frozen",
                  thresholds: Mapping[str, HarmThreshold] | None = None, repo_root: Path | str = ".") -> dict:
    """Check the gates. In confirmatory mode any failure raises; in dev mode failures are returned as warnings."""
    mode = ProtocolMode(mode)
    frozen_dir = Path(repo_root) / frozen_dir
    problems: list[str] = []
    records: dict[str, dict] = {}
    for part in freezes:
        rec = load_freeze(part, frozen_dir)
        if rec is None:
            problems.append(f"freeze part {part!r} is missing ({frozen_dir / (part + '.json')})")
            continue
        src = Path(repo_root) / rec.path if not Path(rec.path).is_absolute() else Path(rec.path)
        if not src.exists():
            problems.append(f"freeze part {part!r} points at a missing file {rec.path}")
        elif sha256_file(src) != rec.sha256:
            problems.append(f"freeze part {part!r} is stale: {rec.path} no longer matches its frozen hash")
        records[part] = rec.to_dict()
    for name, thr in (thresholds or {}).items():
        if mode.value not in thr.usable_in:
            problems.append(f"threshold {name!r} has status {thr.status!r}; not usable in {mode.value} mode")
    if mode is ProtocolMode.CONFIRMATORY and problems:
        raise ConfirmatoryGateError("confirmatory mode refused:\n  - " + "\n  - ".join(problems))
    return {"mode": mode.value, "freezes": records, "warnings": problems, "citable": mode is ProtocolMode.CONFIRMATORY}


@dataclass
class RunDependencies:
    """Everything a confirmatory run must verify before it may write a citable result (I11)."""

    freezes: tuple[str, ...] = ("core", "counts", "bridge")
    environment_lock: str | None = "uv.lock"          # path whose hash is recorded and, when frozen, verified
    data_manifest: str | None = None                  # path to the data manifest that must exist and hash-match
    model_hash: str | None = None                     # declared hash of the frozen subject / checkpoint
    result_dir: str = "results/confirmatory"
    frozen_environment_sha256: str | None = None
    frozen_data_manifest_sha256: str | None = None
    frozen_model_sha256: str | None = None
    frozen_source_tree_sha256: str | None = None      # allows a dirty tree only when the tree hash is itself frozen


def source_tree_hash(repo_root: Path) -> str:
    """SHA-256 over the tracked files' contents (git ls-files); 'unknown' outside a git checkout."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=repo_root, capture_output=True, timeout=30)
        if out.returncode != 0:
            return "unknown"
        h = hashlib.sha256()
        for rel in sorted(p for p in out.stdout.decode().split("\0") if p):
            f = Path(repo_root) / rel
            if f.is_file():
                h.update(rel.encode()); h.update(f.read_bytes())
        return h.hexdigest()
    except Exception:  # pragma: no cover
        return "unknown"


def is_ancestor(commit: str, repo_root: Path) -> bool:
    """True when ``commit`` is HEAD or an ancestor of HEAD (the freeze predates, and survives in, the current history)."""
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=repo_root, capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:  # pragma: no cover
        return False


def require_run_dependencies(mode: ProtocolMode | str, deps: RunDependencies, *, repo_root: Path | str = ".",
                             thresholds: Mapping[str, HarmThreshold] | None = None, out_dir: Path | str | None = None) -> dict:
    """The confirmatory gate for a whole run: freezes, commit/dirty state, environment, data, model, destination.

    Dev mode returns warnings and ``citable=False``; confirmatory mode raises on
    the first failing dependency set. Nothing here writes a freeze.
    """
    mode = ProtocolMode(mode)
    root = Path(repo_root)
    base = require_gates(ProtocolMode.DEV, freezes=deps.freezes, thresholds=thresholds, repo_root=root)   # collect, raise below
    problems = list(base["warnings"])
    commit = git_commit(root)
    record = {"git_commit": commit, "source_tree_sha256": source_tree_hash(root)}
    if commit == "unknown":
        problems.append("not a git checkout: the commit cannot be verified")
    elif commit.endswith("-dirty"):
        if deps.frozen_source_tree_sha256 is None or deps.frozen_source_tree_sha256 != record["source_tree_sha256"]:
            problems.append("working tree is dirty and no matching frozen source-tree hash is declared")
    for part in deps.freezes:
        rec = load_freeze(part, root / "protocol" / "frozen")
        if rec is not None and commit != "unknown":
            base = rec.commit.split("-")[0]
            if base in ("", "unknown"):
                problems.append(f"freeze part {part!r} carries no commit")
            elif not is_ancestor(base, root):
                problems.append(f"freeze part {part!r} was recorded at commit {rec.commit}, which is not an ancestor of HEAD {commit}")
    if deps.environment_lock:
        lock = root / deps.environment_lock
        if not lock.exists():
            problems.append(f"environment lock {deps.environment_lock!r} is missing")
        else:
            record["environment_sha256"] = sha256_file(lock)
            if deps.frozen_environment_sha256 is None:
                problems.append("environment lock hash is not frozen")
            elif deps.frozen_environment_sha256 != record["environment_sha256"]:
                problems.append("environment lock does not match its frozen hash")
    if deps.data_manifest is None:
        problems.append("no data manifest declared")
    else:
        dm = root / deps.data_manifest
        if not dm.exists():
            problems.append(f"data manifest {deps.data_manifest!r} is missing")
        else:
            record["data_manifest_sha256"] = sha256_file(dm)
            if deps.frozen_data_manifest_sha256 != record["data_manifest_sha256"]:
                problems.append("data manifest does not match its frozen hash")
    if deps.model_hash is None:
        problems.append("no model / checkpoint hash declared")
    elif deps.frozen_model_sha256 != deps.model_hash:
        problems.append("model hash does not match its frozen value")
    if out_dir is not None:
        od = Path(out_dir)
        rel = od if od.is_absolute() else root / od
        if deps.result_dir not in str(rel):
            problems.append(f"confirmatory results must be written under {deps.result_dir!r}, not {out_dir}")
    if mode is ProtocolMode.CONFIRMATORY and problems:
        raise ConfirmatoryGateError("confirmatory run refused:\n  - " + "\n  - ".join(problems))
    return {"mode": mode.value, "citable": mode is ProtocolMode.CONFIRMATORY and not problems, "warnings": problems, **record}


def config_hash(config: Mapping) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def provenance(config: Mapping, *, mode: ProtocolMode | str, seeds: Sequence[int], repo_root: Path | str = ".",
               extra: Mapping | None = None) -> dict:
    """Everything a result table must carry (plan §0.6): config hash, commit, seeds, versions, machine, mode."""
    from .. import __version__
    return {
        "mode": ProtocolMode(mode).value,
        "citable": ProtocolMode(mode) is ProtocolMode.CONFIRMATORY,
        "config_hash": config_hash(config),
        "config": dict(config),
        "git_commit": git_commit(Path(repo_root)),
        "seeds": [int(s) for s in seeds],
        "latent_monitor_version": __version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "machine": platform.node() or "unknown",
        "platform": platform.platform(),
        **(dict(extra) if extra else {}),
    }
