#!/home/dwong/anaconda3/bin/python3
"""Resource guard for the JEPA study on shared servers (runbook v4 section 6.2).

Purpose
-------
Keep our JEPA processes inside the runbook limits and never crash or fill the
shared servers.  The guard runs in tmux ``jepa_guard`` on each host every
2 minutes (``--interval 120``), reads the host, writes
``<jobs-dir>/guard_<host>.json`` atomically, appends one JSON line per event to
``<jobs-dir>/guard_<host>.log`` and, when the level is SHED, sends SIGTERM to
our most recently started JEPA job (only if the three conditions of section 6.2
all hold: PID listed in one of our PID files, owner ``dwong``, command line
tagged ``jepa_``).

Design
------
The module is split in two, on purpose:

* **Pure decision core** -- plain dict in, plain dict out, no /proc, no
  nvidia-smi, no torch, no GPU.  This is what the unit tests drive:
  ``evaluate``, ``evaluate_start``, ``assess_gpus``, ``select_shed_target``,
  ``shed_action_due``, ``shed_signal`` (dry-run path), ``run_params_from_status``,
  ``status_is_usable``, ``timeout_seconds``.
* **I/O layer** -- ``collect_reading`` (nvidia-smi, /proc, statvfs/df, du, PID
  files), ``write_status_atomic``, ``append_log_line``, ``verify_pid_matches``,
  ``main``.

Confirmed host facts (lead, 2026-10-17)
---------------------------------------
* Interpreter on both hosts: ``/home/dwong/anaconda3/bin/python3`` (3.12.2).
  The system python on deepthought2 is 3.9 and cannot parse PEP 604 ``X | Y``
  annotations; ``jepa_run.sh`` defaults to the conda path.
* Free file entries are a CephFS directory quota, read from virtual xattrs::

      getfattr -n ceph.quota.max_files --only-values /ceph/dwong   -> max files
      getfattr -n ceph.dir.rfiles    --only-values /ceph/dwong   -> files used
      getfattr -n ceph.dir.rsubdirs  --only-values /ceph/dwong   -> subdirs used
      free_entries = max_files - rfiles - rsubdirs

  Those raw numbers are recorded in the status JSON every cycle.  A
  ``statvfs``/``df -i -P`` fallback with a logged reason is kept for non-Ceph
  paths only; its parsing is unit tested.
* GPU kinds and owners are taken from nvidia-smi, preferring the *observed*
  GPU index over the PID file's declared index.  Unattributed GPU memory
  (``memory.used`` - listed processes) is treated as another user's; > 1 GiB
  is a HOLD and unknown per-process memory is a HOLD.
* CPU/RSS of our jobs are summed over the job's process tree (pid.json PID,
  pgid and /proc parent links) so DataLoader workers count against the caps.
* PID files are filtered by host (the jobs directory is shared over /ceph);
  stale PID files of dead processes are pruned.
* The 2 TB storage limit applies to the JEPA data directory
  ``<jobs-dir>/../data`` (``/ceph/dwong/jepa/data``), measured with
  ``du -sb`` in decimal terabytes and cached for 30 minutes.
* The allocator fraction is the section 6.2 cap; a documented CUDA-context
  tolerance (5%, at least 512 MiB) is applied when deciding to SHED on
  per-process GPU memory, so a job that legitimately fills its cap
  (Titan batch-32 ~9.6 GB) is not shed.

Reading schema (plain JSON-able dict; every field required unless stated)
--------------------------------------------------------------------------
::

    {
      "host": "deepthought",
      "cores": 40,                                  # os.cpu_count()
      "nvidia_smi_available": True,                 # False on a host without it
      "gpus": [                                     # [] when unavailable
        {"index": 0,
         "name": "NVIDIA GeForce GTX TITAN",
         "memory_total_mib": 12288.0,
         "memory_used_mib": 512.0,                  # all users
         "utilization_5m_mean_pct": 3.0,            # mean over last 5 min
         "processes": [                             # nvidia-smi per-process table
            {"pid": 1234, "owner": "dwong", "memory_mib": 9000.0,
             "in_pid_file": True,                   # pid is in one of our PID files
             "utilization_pct": 55.0,               # optional, nvidia-smi pmon
             "cmdline": "python ... jepa_run1 ..."}]}
      ],
      "host_memory_total_mib": 524288.0,            # /proc/meminfo MemTotal
      "host_memory_available_mib": 400000.0,        # /proc/meminfo MemAvailable
      "loadavg_15m": 1.2,                           # /proc/loadavg field 3
      "free_entries": 31944,                        # CephFS quota: max_files-rfiles-rsubdirs
      "entry_source": {                              # raw quota numbers, recorded in status
          "source": "cephfs_xattr", "quota_max_files": 500000,
          "quota_rfiles": 416387, "quota_rsubdirs": 51669},
      "jepa_data_tb": 0.4,                          # decimal TB, /ceph/dwong/jepa/data
      "process_metrics_available": True,            # /proc CPU+RSS readable
      "per_process_gpu_table_available": True,      # nvidia-smi compute-apps readable (required)
      "our_processes": [                            # one entry per our PID file
        {"pid": 1234, "run_id": "run1", "tag": "jepa_run1",
         "owner": "dwong", "cmdline": "python ... jepa_run1 ...",
         "in_pid_file": True,                       # listed in <jobs-dir>/<run_id>/pid.json
         "gpu_index": 0, "gpu_index_declared": 0, "pgid": 1234,
         "tree_pids": [1234, 1235, 1236],
         "gpu_memory_mib": 9000.0,
         "cpu_percent": 350.0, "rss_mib": 4096.0,   # summed over tree_pids
         "started_at_utc": "2026-10-17T10:00:00Z",
         "job_kind": "training",                    # training | evaluation | probe | smoke | other
         "planned_minutes": 120.0,
         "pid_file": "/ceph/dwong/jepa/jobs/run1/pid.json"}
      ]
    }

Missing required input (key absent or ``None``) is never silent: it produces a
HOLD reason ``"missing input: <key> ..."``.  A host without nvidia-smi produces
the reason ``"nvidia-smi unavailable"`` and never crashes.

Levels (runbook v4 section 6.2)
-------------------------------
* OK   -- a new job may start, within the limits (``start`` block in the status
          file says on which GPU and with which duration cap).
* HOLD -- a start condition fails, or another user started heavy work on a GPU
          we use.  No new starts; running jobs continue.
* SHED -- a hard limit is exceeded; the guard SIGTERMs our most recently
          started JEPA job, re-checks after 5 minutes and repeats while SHED
          persists.

Severity order: SHED > HOLD > OK.  The default start candidate used by the
periodic evaluation is "a job of at most 30 minutes", so that on Deepthought a
4th GPU stays usable for smokes/evaluation/probes while 3 trainings run; the
``start.max_planned_minutes`` field carries the 30-minute cap and
``jepa_run.sh`` refuses to start a longer job against it.  Pass
``--candidate-kind/--candidate-minutes`` to evaluate a specific job instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

GUARD_VERSION = "jepa_guard_v4_2026-10-17"

# The only hard-coded host path.  Everything else is derived from it.
JOBS_DIR_DEFAULT = Path("/ceph/dwong/jepa/jobs")

#: owner that may be signalled (runbook section 6.2)
GUARD_OWNER = "dwong"

#: job kinds that count against the "training jobs" limit; anything else is a
#: short/evaluation job.  Unknown kinds count as training (conservative).
TRAINING_KINDS = frozenset({"training"})
NON_TRAINING_KINDS = frozenset(
    {"evaluation", "eval", "probe", "smoke", "profile", "benchmark", "generation", "other"}
)

#: periodic evaluation asks "may a new job of at most 30 min start?"
DEFAULT_START_CANDIDATE_MINUTES = 30.0

#: cmdline tag required for signals (runbook section 6.2)
TAG_RE = re.compile(r"(?:^|\s)(jepa_[A-Za-z0-9][A-Za-z0-9._-]*)")

#: interpreter on both hosts (system python on deepthought2 is 3.9)
PYTHON_INTERPRETER = "/home/dwong/anaconda3/bin/python3"

#: CephFS virtual xattrs for the /ceph/dwong directory quota (confirmed source)
CEPH_QUOTA_MAX_FILES = "ceph.quota.max_files"
CEPH_QUOTA_RFILES = "ceph.dir.rfiles"
CEPH_QUOTA_RSUBDIRS = "ceph.dir.rsubdirs"

#: per-process GPU cap: documented CUDA context tolerance before SHED
GPU_MEMORY_CAP_TOLERANCE_FRACTION = 0.05
GPU_MEMORY_CAP_TOLERANCE_MIN_MIB = 512.0

#: unattributed GPU memory above this is treated as another user's and HOLDs
UNATTRIBUTED_GPU_MEMORY_HOLD_MIB = 1024.0

#: JEPA data scope for the 2 TB limit (relative to the jobs directory)
JEPA_DATA_SUBDIR = "data"

#: R1: the 5-minute utilisation window needs >=3 samples spanning >=4 minutes
UTIL_WINDOW_SAMPLES_MIN = 3
UTIL_WINDOW_SPAN_SECONDS = 240.0
UTIL_WINDOW_SECONDS = 300.0

#: required reading keys -> human description (used verbatim in HOLD reasons)
REQUIRED_READING_KEYS: tuple[tuple[str, str], ...] = (
    ("cores", "host CPU core count"),
    ("nvidia_smi_available", "nvidia-smi availability"),
    ("gpus", "GPU state from nvidia-smi"),
    ("host_memory_total_mib", "/proc/meminfo MemTotal"),
    ("host_memory_available_mib", "/proc/meminfo MemAvailable"),
    ("loadavg_15m", "/proc/loadavg 15-minute load"),
    ("free_entries", "free entry count for /ceph/dwong"),
    ("jepa_data_tb", "JEPA data size in TB"),
    ("our_processes", "our PID-file processes"),
    ("process_metrics_available", "CPU/RSS metrics for our processes"),
    ("utilization_window", "5-minute GPU utilisation sample window"),
)

LEVEL_OK = "OK"
LEVEL_HOLD = "HOLD"
LEVEL_SHED = "SHED"

MIB_PER_GIB = 1024.0
BYTES_PER_TB = 1.0e12


# ---------------------------------------------------------------------------
# Host profiles: the numbers of runbook v4 section 6.2, in one place.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostProfile:
    """Limits for one host (runbook v4 section 6.2)."""

    host: str
    gpu_count_max: int
    training_jobs_max: int
    fourth_gpu_short_job_max_minutes: float | None
    gpu_memory_fraction_max: float
    start_util_5m_max_pct: float
    other_user_memory_mib_max: float | None
    others_plus_ours_fraction_max: float | None
    cpu_cores_abs_max: int
    cpu_core_fraction_max: float | None
    ram_total_max_gb: float
    free_ram_floor_fraction: float = 0.20
    load_start_ratio: float = 0.70
    free_entries_start_min: int = 30_000
    jepa_data_max_tb: float = 2.0
    # SHED thresholds
    free_ram_shed_fraction: float = 0.10
    gpu_memory_shed_fraction: float = 0.95
    load_shed_ratio: float = 0.90
    free_entries_shed_min: int = 10_000
    # "another user starts heavy work on a GPU we use"
    foreign_heavy_memory_mib: float = 2048.0
    foreign_heavy_util_pct: float = 30.0

    def cpu_core_cap(self, cores: int | None) -> int | None:
        """CPU core cap for our processes on this host."""
        if cores is None:
            return None
        if self.cpu_core_fraction_max is None:
            return int(self.cpu_cores_abs_max)
        if cores <= 0:
            return None
        return max(1, min(self.cpu_cores_abs_max, int(math.floor(self.cpu_core_fraction_max * cores))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "gpu_count_max": self.gpu_count_max,
            "training_jobs_max": self.training_jobs_max,
            "fourth_gpu_short_job_max_minutes": self.fourth_gpu_short_job_max_minutes,
            "per_process_gpu_memory_fraction": self.gpu_memory_fraction_max,
            "start_util_5m_max_pct": self.start_util_5m_max_pct,
            "other_user_memory_mib_max": self.other_user_memory_mib_max,
            "others_plus_ours_fraction_max": self.others_plus_ours_fraction_max,
            "cpu_cores_abs_max": self.cpu_cores_abs_max,
            "cpu_core_fraction_max": self.cpu_core_fraction_max,
            "ram_total_max_gb": self.ram_total_max_gb,
            "free_ram_floor_fraction": self.free_ram_floor_fraction,
            "load_start_ratio": self.load_start_ratio,
            "free_entries_start_min": self.free_entries_start_min,
            "jepa_data_max_tb": self.jepa_data_max_tb,
            "free_ram_shed_fraction": self.free_ram_shed_fraction,
            "gpu_memory_shed_fraction": self.gpu_memory_shed_fraction,
            "load_shed_ratio": self.load_shed_ratio,
            "free_entries_shed_min": self.free_entries_shed_min,
            "foreign_heavy_memory_mib": self.foreign_heavy_memory_mib,
            "foreign_heavy_util_pct": self.foreign_heavy_util_pct,
        }


#: runbook v4 section 6.2 table, verbatim numbers.
PROFILES: dict[str, HostProfile] = {
    "deepthought": HostProfile(
        host="deepthought",
        gpu_count_max=4,
        training_jobs_max=3,
        fourth_gpu_short_job_max_minutes=30.0,
        gpu_memory_fraction_max=0.80,
        start_util_5m_max_pct=20.0,
        other_user_memory_mib_max=1024.0,       # no other user's process holds > 1 GB on it
        others_plus_ours_fraction_max=None,
        cpu_cores_abs_max=16,                   # <= 16 of 40
        cpu_core_fraction_max=None,
        ram_total_max_gb=128.0,
    ),
    "deepthought2": HostProfile(
        host="deepthought2",
        gpu_count_max=2,
        training_jobs_max=2,
        fourth_gpu_short_job_max_minutes=None,
        gpu_memory_fraction_max=0.45,
        start_util_5m_max_pct=60.0,
        other_user_memory_mib_max=None,
        others_plus_ours_fraction_max=0.85,     # others' memory + our cap <= 85% of card
        cpu_cores_abs_max=16,                   # <= 16, or 40% of the cores if fewer
        cpu_core_fraction_max=0.40,
        ram_total_max_gb=64.0,
    ),
}


def get_profile(host: str) -> HostProfile:
    try:
        return PROFILES[host]
    except KeyError:
        raise ValueError(f"unknown host {host!r}; choose one of {sorted(PROFILES)}") from None


# ---------------------------------------------------------------------------
# Small numeric / time / text helpers (None-safe; never raise on bad input).
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def utc_iso(epoch: float | None = None) -> str:
    instant = time.time() if epoch is None else float(epoch)
    return datetime.fromtimestamp(instant, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: Any) -> float | None:
    """Parse an ISO-8601 UTC timestamp into epoch seconds; None if unreadable."""
    if not isinstance(text, str) or not text.strip():
        return None
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _job_kind(job: Mapping[str, Any]) -> str:
    kind = job.get("job_kind")
    if not isinstance(kind, str) or not kind.strip():
        return "training"  # conservative: an unknown kind counts as training
    kind = kind.strip().lower()
    return kind


def _is_training_job(job: Mapping[str, Any]) -> bool:
    kind = _job_kind(job)
    return kind in TRAINING_KINDS or kind not in NON_TRAINING_KINDS


def _is_pidfile_listed(job: Mapping[str, Any]) -> bool:
    return job.get("in_pid_file") is True


# ---------------------------------------------------------------------------
# Pure decision core.
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """Result of a pure evaluation.  ``reasons`` are human-readable, stable."""

    level: str
    reasons: list[str] = field(default_factory=list)
    start_gpu: int | None = None
    start_max_planned_minutes: float | None = None
    gpu_blocks: list[dict[str, Any]] = field(default_factory=list)
    caps: dict[str, Any] = field(default_factory=dict)
    our_jobs: list[dict[str, Any]] = field(default_factory=list)
    candidate: dict[str, Any] = field(default_factory=dict)


def _normalize_candidate(candidate: Mapping[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {"kind": "other", "planned_minutes": DEFAULT_START_CANDIDATE_MINUTES}
    kind = candidate.get("kind") or "other"
    if not isinstance(kind, str):
        kind = "other"
    kind = kind.strip().lower()
    planned = _to_float(candidate.get("planned_minutes"))
    return {"kind": kind, "planned_minutes": planned}


def _normalize_gpus(reading: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return all GPUs with normalized process tables and computed fields."""
    our_pids = _our_pids(reading)
    gpus: list[dict[str, Any]] = []
    raw_gpus = reading.get("gpus")
    if not isinstance(raw_gpus, list):
        return gpus
    for position, raw in enumerate(raw_gpus):
        if not isinstance(raw, Mapping):
            continue
        index = _to_int(raw.get("index"))
        if index is None:
            index = position
        total = _to_float(raw.get("memory_total_mib"))
        used = _to_float(raw.get("memory_used_mib"))
        util = _to_float(raw.get("utilization_5m_mean_pct"))
        processes: list[dict[str, Any]] = []
        raw_processes = raw.get("processes")
        if isinstance(raw_processes, list):
            for proc in raw_processes:
                if not isinstance(proc, Mapping):
                    continue
                pid = _to_int(proc.get("pid"))
                owner = proc.get("owner")
                memory = _to_float(proc.get("memory_mib"))
                cmdline = proc.get("cmdline")
                listed = proc.get("in_pid_file") is True or (pid is not None and pid in our_pids)
                processes.append(
                    {
                        "pid": pid,
                        "owner": owner if isinstance(owner, str) else None,
                        "memory_mib": memory,
                        "in_pid_file": bool(listed),
                        "utilization_pct": _to_float(proc.get("utilization_pct")),
                        "cmdline": cmdline if isinstance(cmdline, str) else "",
                    }
                )
        ours = [p for p in processes if p["in_pid_file"] and p["pid"] is not None]
        our_ids = {id(p) for p in ours}
        foreign = [p for p in processes if id(p) not in our_ids]
        ours_mib = _sum_known(p["memory_mib"] for p in ours)
        foreign_mib = _sum_known(p["memory_mib"] for p in foreign)
        foreign_util_known = [p["utilization_pct"] for p in foreign if p["utilization_pct"] is not None]
        unknown_memory_pids = [p["pid"] for p in processes if p["memory_mib"] is None]
        allocated = used if used is not None else (ours_mib + foreign_mib if processes else None)
        fraction = None
        if allocated is not None and total is not None and total > 0:
            fraction = allocated / total
        # F5: memory the per-process table does not attribute is treated as
        # another user's (hidden/foreign context); > 1 GiB is a HOLD.
        unattributed_mib: float | None = None
        if used is not None and not unknown_memory_pids:
            unattributed_mib = max(0.0, used - ours_mib - foreign_mib)
        foreign_total_mib = foreign_mib + (unattributed_mib or 0.0)
        gpus.append(
            {
                "index": index,
                "name": raw.get("name") if isinstance(raw.get("name"), str) else "",
                "memory_total_mib": total,
                "memory_used_mib": used,
                "memory_allocated_mib": allocated,
                "memory_used_fraction": fraction,
                "utilization_5m_mean_pct": util,
                "processes": processes,
                "our_processes": ours,
                "foreign_processes": foreign,
                "our_memory_mib": ours_mib,
                "foreign_memory_mib": foreign_mib,
                "foreign_memory_total_mib": foreign_total_mib,
                "unattributed_memory_mib": unattributed_mib,
                "unknown_memory_pids": unknown_memory_pids,
                "foreign_utilization_pct": (sum(foreign_util_known) if foreign_util_known else None),
            }
        )
    return gpus


def utilization_window(
    util_history: Iterable[tuple[float, Mapping[int, float]]],
    now_epoch: float,
) -> dict[str, Any]:
    """R1: is the 5-minute GPU-utilisation window actually populated?

    nvidia-smi only reports instantaneous utilisation, so the guard accumulates
    samples.  A single sample is not a 5-minute mean: the window is warm only
    with at least ``UTIL_WINDOW_SAMPLES_MIN`` samples spanning at least
    ``UTIL_WINDOW_SPAN_SECONDS``.  Until then the guard reports HOLD.
    """
    window_start = float(now_epoch) - UTIL_WINDOW_SECONDS
    stamps = sorted(
        float(stamp)
        for stamp, _sample in util_history
        if float(stamp) >= window_start
    )
    count = len(stamps)
    span = (stamps[-1] - stamps[0]) if count >= 2 else 0.0
    warm = count >= UTIL_WINDOW_SAMPLES_MIN and span >= UTIL_WINDOW_SPAN_SECONDS
    reason = None
    if not warm:
        reason = (
            f"utilisation window not populated ({count}/{UTIL_WINDOW_SAMPLES_MIN} samples, "
            f"span {span:.0f} s/{UTIL_WINDOW_SPAN_SECONDS:.0f} s; warming up)"
        )
    return {
        "warm": warm,
        "sample_count": count,
        "span_seconds": round(span, 1),
        "window_seconds": UTIL_WINDOW_SECONDS,
        "samples_required": UTIL_WINDOW_SAMPLES_MIN,
        "span_required_seconds": UTIL_WINDOW_SPAN_SECONDS,
        "reason": reason,
    }


def _sum_known(values: Iterable[float | None]) -> float:
    return float(sum(v for v in values if v is not None))


def _our_pids(reading: Mapping[str, Any]) -> set[int]:
    pids: set[int] = set()
    jobs = reading.get("our_processes")
    if isinstance(jobs, list):
        for job in jobs:
            if not isinstance(job, Mapping):
                continue
            pid = _to_int(job.get("pid"))
            if pid is not None and _is_pidfile_listed(job):
                pids.add(pid)
    return pids


def _our_jobs(reading: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalized list of our PID-file processes."""
    jobs: list[dict[str, Any]] = []
    raw_jobs = reading.get("our_processes")
    if not isinstance(raw_jobs, list):
        return jobs
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            continue
        jobs.append(
            {
                "pid": _to_int(raw.get("pid")),
                "run_id": raw.get("run_id"),
                "tag": raw.get("tag"),
                "owner": raw.get("owner"),
                "cmdline": raw.get("cmdline") if isinstance(raw.get("cmdline"), str) else "",
                "in_pid_file": _is_pidfile_listed(raw),
                "gpu_index": _to_int(raw.get("gpu_index")),
                "gpu_index_declared": _to_int(raw.get("gpu_index_declared")),
                "pgid": _to_int(raw.get("pgid")),
                "tree_pids": list(raw.get("tree_pids") or []) if isinstance(raw.get("tree_pids"), list) else [raw.get("pid")],
                "gpu_memory_mib": _to_float(raw.get("gpu_memory_mib")),
                "cpu_percent": _to_float(raw.get("cpu_percent")),
                "rss_mib": _to_float(raw.get("rss_mib")),
                "started_at_utc": raw.get("started_at_utc"),
                "job_kind": _job_kind(raw),
                "planned_minutes": _to_float(raw.get("planned_minutes")),
                "pid_file": raw.get("pid_file"),
            }
        )
    return jobs


def _used_gpu_indices(jobs: list[dict[str, Any]]) -> list[int]:
    used: set[int] = set()
    for job in jobs:
        if job.get("in_pid_file") and job.get("gpu_index") is not None:
            used.add(int(job["gpu_index"]))
    return sorted(used)


def _cpu_cores_used(jobs: list[dict[str, Any]]) -> float:
    total_percent = 0.0
    for job in jobs:
        value = job.get("cpu_percent")
        if value is not None and value > 0:
            total_percent += value
    return total_percent / 100.0


def _ram_used_gib(jobs: list[dict[str, Any]]) -> float:
    total_mib = 0.0
    for job in jobs:
        value = job.get("rss_mib")
        if value is not None and value > 0:
            total_mib += value
    return total_mib / MIB_PER_GIB


def _host_free_ram_fraction(reading: Mapping[str, Any]) -> float | None:
    total = _to_float(reading.get("host_memory_total_mib"))
    available = _to_float(reading.get("host_memory_available_mib"))
    if total is None or available is None or total <= 0:
        return None
    return max(0.0, min(1.0, available / total))


def _gpu_memory_cap_mib(profile: HostProfile, gpu: Mapping[str, Any]) -> float | None:
    total = gpu.get("memory_total_mib")
    if total is None or total <= 0:
        return None
    return profile.gpu_memory_fraction_max * total


def _gpu_memory_shed_cap_mib(profile: HostProfile, gpu: Mapping[str, Any]) -> float | None:
    """Per-process SHED threshold: nominal cap + documented CUDA-context tolerance."""
    cap = _gpu_memory_cap_mib(profile, gpu)
    if cap is None:
        return None
    tolerance = max(GPU_MEMORY_CAP_TOLERANCE_MIN_MIB, GPU_MEMORY_CAP_TOLERANCE_FRACTION * cap)
    return cap + tolerance


def assess_gpus(reading: Mapping[str, Any], profile: HostProfile) -> list[dict[str, Any]]:
    """Per-GPU block: usage, owners, and whether a new job could start on it."""
    gpus = _normalize_gpus(reading)
    used = set(_used_gpu_indices(_our_jobs(reading)))
    for gpu in gpus:
        gpu["used_by_us"] = gpu["index"] in used
        start_ok, blockers = _gpu_start_condition(gpu, profile)
        gpu["start_ok"] = bool(start_ok)
        gpu["start_blockers"] = blockers
        gpu["free"] = not gpu["used_by_us"] and not gpu["processes"]
    return gpus


def _gpu_start_condition(gpu: Mapping[str, Any], profile: HostProfile) -> tuple[bool, list[str]]:
    """Runbook v4 section 6.2 "Start on a GPU only if" rules for one GPU."""
    index = gpu["index"]
    blockers: list[str] = []
    if gpu.get("used_by_us"):
        blockers.append(f"GPU {index} is already used by one of our jobs (one JEPA job per GPU)")
    total = gpu.get("memory_total_mib")
    if total is None or total <= 0:
        blockers.append(f"GPU {index}: card memory total is unknown")
    util = gpu.get("utilization_5m_mean_pct")
    if util is None:
        blockers.append(f"GPU {index}: 5-minute mean utilisation is unknown")
    elif util >= profile.start_util_5m_max_pct:
        blockers.append(
            f"GPU {index}: 5-minute mean utilisation {util:.1f}% >= "
            f"{profile.start_util_5m_max_pct:.0f}% limit"
        )
    foreign = gpu.get("foreign_processes") or []
    for proc in foreign:
        pid = proc.get("pid")
        memory = proc.get("memory_mib")
        if memory is None:
            blockers.append(f"GPU {index}: other user's process PID {pid} memory is unknown")
            continue
        if (
            profile.other_user_memory_mib_max is not None
            and memory > profile.other_user_memory_mib_max
        ):
            blockers.append(
                f"GPU {index}: other user's process PID {pid} holds {memory:.0f} MiB > "
                f"{profile.other_user_memory_mib_max:.0f} MiB limit"
            )
    if profile.other_user_memory_mib_max is not None:
        unattributed = gpu.get("unattributed_memory_mib") or 0.0
        if unattributed > profile.other_user_memory_mib_max:
            blockers.append(
                f"GPU {index}: {unattributed:.0f} MiB not attributed to any process > "
                f"{profile.other_user_memory_mib_max:.0f} MiB (treated as another user's)"
            )
    if profile.others_plus_ours_fraction_max is not None and total is not None and total > 0:
        foreign_mib = gpu.get("foreign_memory_total_mib") or 0.0
        our_cap = profile.gpu_memory_fraction_max * total
        allowed = profile.others_plus_ours_fraction_max * total
        if foreign_mib + our_cap > allowed:
            blockers.append(
                f"GPU {index}: other users' {foreign_mib:.0f} MiB + our cap {our_cap:.0f} MiB > "
                f"{profile.others_plus_ours_fraction_max * 100:.0f}% of {total:.0f} MiB"
            )
    return (not blockers), blockers


def _foreign_heavy_reasons(
    gpus: list[dict[str, Any]], profile: HostProfile
) -> list[str]:
    """HOLD when another user starts heavy work on a GPU we use (section 6.2)."""
    reasons: list[str] = []
    for gpu in gpus:
        if not gpu.get("used_by_us"):
            continue
        index = gpu["index"]
        memory = gpu.get("foreign_memory_mib") or 0.0
        if memory > profile.foreign_heavy_memory_mib:
            reasons.append(
                f"GPU {index}: another user's processes hold {memory:.0f} MiB > "
                f"{profile.foreign_heavy_memory_mib:.0f} MiB on a GPU we use"
            )
            continue
        foreign_util = gpu.get("foreign_utilization_pct")
        if foreign_util is not None and foreign_util > profile.foreign_heavy_util_pct:
            reasons.append(
                f"GPU {index}: another user's processes use {foreign_util:.0f}% > "
                f"{profile.foreign_heavy_util_pct:.0f}% on a GPU we use"
            )
    return reasons


def _unattributed_memory_reasons(gpus: list[dict[str, Any]]) -> list[str]:
    """F5: memory no listed process accounts for is treated as foreign."""
    reasons: list[str] = []
    for gpu in gpus:
        index = gpu["index"]
        if gpu.get("memory_used_mib") is None:
            reasons.append(f"GPU {index}: memory used is unknown; cannot attribute processes")
            continue
        for pid in gpu.get("unknown_memory_pids") or []:
            reasons.append(f"GPU {index}: process PID {pid} memory is unknown")
        unattributed = gpu.get("unattributed_memory_mib")
        if unattributed is not None and unattributed > UNATTRIBUTED_GPU_MEMORY_HOLD_MIB:
            reasons.append(
                f"GPU {index}: {unattributed:.0f} MiB GPU memory is not attributed to any process "
                f"(> {UNATTRIBUTED_GPU_MEMORY_HOLD_MIB:.0f} MiB); treating it as another user's"
            )
    return reasons


def _duplicate_gpu_reasons(jobs: list[dict[str, Any]]) -> list[str]:
    """F3: two of our PID files naming the same GPU is a HOLD."""
    by_gpu: dict[int, list[str]] = {}
    for job in jobs:
        if not job.get("in_pid_file") or job.get("gpu_index") is None:
            continue
        label = str(job.get("run_id") or job.get("pid"))
        by_gpu.setdefault(int(job["gpu_index"]), []).append(label)
    reasons: list[str] = []
    for gpu_index in sorted(by_gpu):
        runs = by_gpu[gpu_index]
        if len(runs) > 1:
            reasons.append(
                f"GPU {gpu_index}: {len(runs)} of our jobs name it "
                f"(duplicate GPU assignment: {', '.join(runs)})"
            )
    return reasons


def _gpu_index_mismatch_reasons(jobs: list[dict[str, Any]]) -> list[str]:
    """F7: the observed nvidia-smi GPU index must win over the declared index."""
    reasons: list[str] = []
    for job in jobs:
        if not job.get("in_pid_file"):
            continue
        declared = job.get("gpu_index_declared")
        observed = job.get("gpu_index")
        if declared is None or observed is None:
            continue
        if int(declared) != int(observed):
            reasons.append(
                f"GPU index mismatch: run {job.get('run_id')!r} declares GPU {declared} "
                f"but nvidia-smi attributes PID {job.get('pid')} to GPU {observed}"
            )
    return reasons


def _missing_input_reasons(reading: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key, description in REQUIRED_READING_KEYS:
        if key not in reading or reading.get(key) is None:
            reasons.append(
                f"missing input: {key} ({description}); start conditions cannot be verified"
            )
    return reasons


def _shed_reasons(
    reading: Mapping[str, Any],
    profile: HostProfile,
    gpus: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[str]:
    """Hard-limit SHED conditions of runbook v4 section 6.2."""
    reasons: list[str] = []
    free_ram = _host_free_ram_fraction(reading)
    if free_ram is not None and free_ram < profile.free_ram_shed_fraction:
        reasons.append(
            f"host free RAM {free_ram * 100:.1f}% < {profile.free_ram_shed_fraction * 100:.0f}%"
        )
    for gpu in gpus:
        fraction = gpu.get("memory_used_fraction")
        if fraction is not None and fraction > profile.gpu_memory_shed_fraction:
            reasons.append(
                f"GPU {gpu['index']} total memory {fraction * 100:.1f}% > "
                f"{profile.gpu_memory_shed_fraction * 100:.0f}%"
            )
    cores = _to_int(reading.get("cores"))
    load = _to_float(reading.get("loadavg_15m"))
    if cores is not None and cores > 0 and load is not None and load > profile.load_shed_ratio * cores:
        reasons.append(
            f"15-minute load {load:.1f} > {profile.load_shed_ratio:.1f} x {cores} cores"
        )
    free_entries = _to_int(reading.get("free_entries"))
    if free_entries is not None and free_entries < profile.free_entries_shed_min:
        reasons.append(
            f"free entries {free_entries} < {profile.free_entries_shed_min} (hard limit)"
        )
    # our processes exceeding their caps
    gpu_by_index = {gpu["index"]: gpu for gpu in gpus}
    for job in jobs:
        if not job.get("in_pid_file"):
            continue
        memory = job.get("gpu_memory_mib")
        gpu_index = job.get("gpu_index")
        if memory is None or gpu_index is None or gpu_index not in gpu_by_index:
            continue
        cap = _gpu_memory_shed_cap_mib(profile, gpu_by_index[gpu_index])
        if cap is not None and memory > cap:
            reasons.append(
                f"our process PID {job.get('pid')} uses {memory:.0f} MiB GPU memory > "
                f"per-process cap {cap:.0f} MiB on GPU {gpu_index}"
            )
    # also check the nvidia-smi process table, in case our job record lacks memory
    flagged_pids = {job.get("pid") for job in jobs if job.get("in_pid_file")}
    for gpu in gpus:
        cap = _gpu_memory_shed_cap_mib(profile, gpu)
        if cap is None:
            continue
        for proc in gpu.get("our_processes") or []:
            if proc.get("pid") in flagged_pids:
                continue  # already checked above
            memory = proc.get("memory_mib")
            if memory is not None and memory > cap:
                reasons.append(
                    f"our process PID {proc.get('pid')} uses {memory:.0f} MiB GPU memory > "
                    f"per-process cap {cap:.0f} MiB on GPU {gpu['index']}"
                )
    cores_cap = profile.cpu_core_cap(cores)
    if cores_cap is not None:
        cpu_used = _cpu_cores_used(jobs)
        if cpu_used > cores_cap:
            reasons.append(
                f"our processes use {cpu_used:.1f} CPU cores > cap {cores_cap} cores"
            )
    ram_used = _ram_used_gib(jobs)
    if ram_used > profile.ram_total_max_gb:
        reasons.append(
            f"our processes use {ram_used:.1f} GiB RAM > cap {profile.ram_total_max_gb:.0f} GiB"
        )
    used_gpus = _used_gpu_indices(jobs)
    if len(used_gpus) > profile.gpu_count_max:
        reasons.append(
            f"our jobs use {len(used_gpus)} GPUs > limit {profile.gpu_count_max}"
        )
    training = [job for job in jobs if _is_training_job(job)]
    if len(training) > profile.training_jobs_max:
        reasons.append(
            f"our jobs include {len(training)} training jobs > limit {profile.training_jobs_max}"
        )
    return reasons


def evaluate_start(
    reading: Mapping[str, Any],
    profile: HostProfile,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[int | None, float | None, list[str]]:
    """Pure start decision.

    Returns ``(gpu_index, max_planned_minutes, blockers)``.  ``gpu_index`` is
    None when no start is allowed; ``max_planned_minutes`` is the duration cap
    for the chosen GPU (30 on Deepthought's 4th GPU, else None).
    """
    cand = _normalize_candidate(candidate)
    blockers: list[str] = []
    cores = _to_int(reading.get("cores"))
    if cores is None or cores <= 0:
        blockers.append("missing input: cores (host CPU core count); start conditions cannot be verified")
        cores = None
    load = _to_float(reading.get("loadavg_15m"))
    if cores is not None:
        if load is not None and load > profile.load_start_ratio * cores:
            blockers.append(
                f"15-minute load {load:.2f} > {profile.load_start_ratio:.2f} x {cores} cores; "
                "start nothing"
            )
        cpu_cap = profile.cpu_core_cap(cores)
    else:
        cpu_cap = None
    free_ram = _host_free_ram_fraction(reading)
    if free_ram is not None and free_ram < profile.free_ram_floor_fraction:
        blockers.append(
            f"host free RAM {free_ram * 100:.1f}% < {profile.free_ram_floor_fraction * 100:.0f}% floor"
        )
    free_entries = _to_int(reading.get("free_entries"))
    if free_entries is not None and free_entries < profile.free_entries_start_min:
        blockers.append(
            f"free entries {free_entries} < {profile.free_entries_start_min}; write no new data"
        )
    data_tb = _to_float(reading.get("jepa_data_tb"))
    if data_tb is not None and data_tb > profile.jepa_data_max_tb:
        blockers.append(
            f"JEPA data {data_tb:.2f} TB > {profile.jepa_data_max_tb:.1f} TB; write no new data"
        )
    jobs = _our_jobs(reading)
    cpu_used = _cpu_cores_used(jobs)
    if cpu_cap is not None and cpu_used > cpu_cap:
        blockers.append(f"our processes already use {cpu_used:.1f} CPU cores > cap {cpu_cap}")
    ram_used = _ram_used_gib(jobs)
    if ram_used > profile.ram_total_max_gb:
        blockers.append(
            f"our processes already use {ram_used:.1f} GiB RAM > cap {profile.ram_total_max_gb:.0f} GiB"
        )
    used_gpus = _used_gpu_indices(jobs)
    training_count = sum(1 for job in jobs if _is_training_job(job) and job.get("in_pid_file"))
    if len(used_gpus) >= profile.gpu_count_max:
        blockers.append(
            f"GPU count: our jobs use {len(used_gpus)} GPUs (limit {profile.gpu_count_max})"
        )
        return None, None, blockers
    gpu_blocks = assess_gpus(reading, profile)
    eligible = [gpu for gpu in gpu_blocks if not gpu.get("used_by_us") and gpu.get("start_ok")]
    if not eligible:
        blockers.append("no GPU satisfies the start condition")
        for gpu in gpu_blocks:
            for reason in gpu.get("start_blockers") or []:
                blockers.append(reason)
        return None, None, blockers
    chosen = min(eligible, key=lambda gpu: gpu["index"])
    max_planned: float | None = None
    refuse_start = False
    if cand["kind"] == "training" and training_count >= profile.training_jobs_max:
        blockers.append(
            f"training jobs: {training_count} running (limit {profile.training_jobs_max}); "
            f"the {training_count + 1}th training job is refused"
        )
        refuse_start = True
    fourth_gpu_rule = (
        profile.fourth_gpu_short_job_max_minutes is not None
        and len(used_gpus) + 1 >= profile.gpu_count_max
    )
    if fourth_gpu_rule:
        max_planned = profile.fourth_gpu_short_job_max_minutes
        planned = cand["planned_minutes"]
        if planned is None or planned > max_planned:
            shown = "unknown" if planned is None else f"{planned:.0f} min"
            blockers.append(
                f"4th GPU: only jobs <= {max_planned:.0f} min may use it; candidate is {shown}"
            )
            refuse_start = True
    if refuse_start:
        return None, None, blockers
    return int(chosen["index"]), max_planned, blockers


def _caps(
    reading: Mapping[str, Any],
    profile: HostProfile,
    gpu_blocks: list[dict[str, Any]] | None = None,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cores = _to_int(reading.get("cores"))
    cpu_cap = profile.cpu_core_cap(cores)
    free_ram = _host_free_ram_fraction(reading)
    jobs = jobs if jobs is not None else _our_jobs(reading)
    gpu_blocks = gpu_blocks if gpu_blocks is not None else assess_gpus(reading, profile)
    per_gpu_caps: dict[str, float | None] = {}
    per_gpu_shed_caps: dict[str, float | None] = {}
    for gpu in gpu_blocks:
        per_gpu_caps[str(gpu["index"])] = _gpu_memory_cap_mib(profile, gpu)
        per_gpu_shed_caps[str(gpu["index"])] = _gpu_memory_shed_cap_mib(profile, gpu)
    jobs_dir = reading.get("reading_notes", {}).get("jobs_dir") if isinstance(reading.get("reading_notes"), Mapping) else None
    data_path = None
    if jobs_dir:
        data_path = str(Path(jobs_dir).parent / JEPA_DATA_SUBDIR)
    return {
        "gpu_count_max": profile.gpu_count_max,
        "training_jobs_max": profile.training_jobs_max,
        "fourth_gpu_short_job_max_minutes": profile.fourth_gpu_short_job_max_minutes,
        "per_process_gpu_memory_fraction": profile.gpu_memory_fraction_max,
        "per_process_gpu_memory_cap_mib": per_gpu_caps,
        "per_process_gpu_memory_shed_cap_mib": per_gpu_shed_caps,
        "gpu_memory_shed_tolerance_fraction": GPU_MEMORY_CAP_TOLERANCE_FRACTION,
        "gpu_memory_shed_tolerance_min_mib": GPU_MEMORY_CAP_TOLERANCE_MIN_MIB,
        "unattributed_gpu_memory_hold_mib": UNATTRIBUTED_GPU_MEMORY_HOLD_MIB,
        "start_util_5m_max_pct": profile.start_util_5m_max_pct,
        "other_user_memory_mib_max": profile.other_user_memory_mib_max,
        "others_plus_ours_fraction_max": profile.others_plus_ours_fraction_max,
        "cpu_cores_max": cpu_cap,
        "cpu_cores_used": round(_cpu_cores_used(jobs), 2),
        "ram_total_max_gb": profile.ram_total_max_gb,
        "ram_used_gib": round(_ram_used_gib(jobs), 2),
        "free_ram_floor_fraction": profile.free_ram_floor_fraction,
        "host_free_ram_fraction": None if free_ram is None else round(free_ram, 4),
        "load_start_ratio": profile.load_start_ratio,
        "load_shed_ratio": profile.load_shed_ratio,
        "cores": cores,
        "loadavg_15m": _to_float(reading.get("loadavg_15m")),
        "free_entries_start_min": profile.free_entries_start_min,
        "free_entries_shed_min": profile.free_entries_shed_min,
        "free_entries": _to_int(reading.get("free_entries")),
        "jepa_data_max_tb": profile.jepa_data_max_tb,
        "jepa_data_tb": _to_float(reading.get("jepa_data_tb")),
        "jepa_data_path": data_path,
        "jepa_data_scope": (data_path + " (decimal TB, du -sb, cached 30 min)") if data_path else None,
        "entry_source": reading.get("entry_source"),
        "gpu_memory_shed_fraction": profile.gpu_memory_shed_fraction,
        "foreign_heavy_memory_mib": profile.foreign_heavy_memory_mib,
        "foreign_heavy_util_pct": profile.foreign_heavy_util_pct,
    }


def evaluate(
    reading: Mapping[str, Any],
    profile: HostProfile,
    candidate: Mapping[str, Any] | None = None,
) -> Decision:
    """Pure level decision for one reading.

    Level is SHED if any hard limit is exceeded, else HOLD if any HOLD reason
    exists (missing input, a start condition fails, or a foreign user started
    heavy work on a GPU we use), else OK with the GPU a new job may use.
    """
    cand = _normalize_candidate(candidate)
    jobs = _our_jobs(reading)
    nvidia_available = reading.get("nvidia_smi_available") is True
    raw_gpus = reading.get("gpus") if nvidia_available else []
    gpus = assess_gpus({**reading, "gpus": raw_gpus}, profile) if isinstance(raw_gpus, list) else []
    caps = _caps(reading, profile, gpus, jobs)

    shed_reasons = _shed_reasons(reading, profile, gpus, jobs)
    if shed_reasons:
        return Decision(
            level=LEVEL_SHED,
            reasons=shed_reasons,
            start_gpu=None,
            start_max_planned_minutes=None,
            gpu_blocks=gpus,
            caps=caps,
            our_jobs=jobs,
            candidate=cand,
        )

    hold_reasons = _missing_input_reasons(reading)
    if reading.get("nvidia_smi_available") is False:
        hold_reasons.append("nvidia-smi unavailable: cannot read GPU state")
        hold_reasons.append("nvidia-smi unavailable: GPU start conditions cannot be verified")
    elif reading.get("per_process_gpu_table_available") is not True:
        hold_reasons.append(
            "missing input: per_process_gpu_table_available (nvidia-smi per-process table "
            "must be True); per-process GPU memory caps cannot be verified"
        )
    elif nvidia_available and not gpus:
        hold_reasons.append(
            "missing input: gpus (nvidia-smi reported no GPUs); start conditions cannot be verified"
        )
    window = reading.get("utilization_window")
    if isinstance(window, Mapping) and not window.get("warm", True):
        hold_reasons.append(
            str(window.get("reason") or "utilisation window not populated; warming up")
        )
    hold_reasons.extend(_foreign_heavy_reasons(gpus, profile))
    hold_reasons.extend(_unattributed_memory_reasons(gpus))
    hold_reasons.extend(_duplicate_gpu_reasons(jobs))
    hold_reasons.extend(_gpu_index_mismatch_reasons(jobs))
    start_gpu, max_planned, start_blockers = evaluate_start(reading, profile, cand)
    hold_reasons.extend(start_blockers)
    level = LEVEL_HOLD if hold_reasons else LEVEL_OK
    return Decision(
        level=level,
        reasons=hold_reasons,
        start_gpu=start_gpu if level == LEVEL_OK else None,
        start_max_planned_minutes=max_planned if level == LEVEL_OK else None,
        gpu_blocks=gpus,
        caps=caps,
        our_jobs=jobs,
        candidate=cand,
    )


def select_shed_target(reading: Mapping[str, Any]) -> dict[str, Any] | None:
    """Most recently started JEPA job that may be signalled, or None.

    All three conditions of section 6.2 must hold: listed in one of our PID
    files, owned by ``dwong``, and tagged ``jepa_`` in the command line.  A job
    with an unreadable start time is not selectable (we never guess order).
    """
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    raw_jobs = reading.get("our_processes")
    if not isinstance(raw_jobs, list):
        return None
    for raw in raw_jobs:
        if not isinstance(raw, Mapping):
            continue
        if not _is_pidfile_listed(raw):
            continue
        if raw.get("owner") != GUARD_OWNER:
            continue
        cmdline = raw.get("cmdline") if isinstance(raw.get("cmdline"), str) else ""
        if not TAG_RE.search(cmdline):
            continue
        tag = raw.get("tag")
        if isinstance(tag, str) and tag and tag not in cmdline:
            continue
        pid = _to_int(raw.get("pid"))
        if pid is None or pid <= 1:
            continue
        started = parse_utc(raw.get("started_at_utc"))
        if started is None:
            continue
        candidates.append((started, pid, dict(raw)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def shed_action_due(
    now_epoch: float, last_shed_epoch: float | None, retry_seconds: float = 300.0
) -> bool:
    """True when a SHED signal may be sent now ("re-check after 5 minutes")."""
    if last_shed_epoch is None:
        return True
    return float(now_epoch) - float(last_shed_epoch) >= float(retry_seconds)


def shed_signal(
    target: Mapping[str, Any],
    *,
    dry_run: bool,
    verifier: Any = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Send SIGTERM to ``target`` (or, in dry-run, only describe it).

    In dry-run nothing is verified on the host and nothing is signalled; the
    returned event says exactly which PID would have received SIGTERM.  In live
    mode ``verifier`` re-checks the three conditions immediately before the
    signal; a mismatch refuses the signal (never touch anything else).
    """
    now = time.time() if now_epoch is None else float(now_epoch)
    event: dict[str, Any] = {
        "timestamp": utc_iso(now),
        "timestamp_epoch": now,
        "event": "shed_signal",
        "dry_run": bool(dry_run),
        "pid": _to_int(target.get("pid")),
        "run_id": target.get("run_id"),
        "tag": target.get("tag"),
        "signal": "SIGTERM",
    }
    if dry_run:
        event["action"] = "would_signal"
        return event
    check = verifier if verifier is not None else verify_pid_matches
    try:
        allowed, detail = check(target)
    except Exception as exc:  # noqa: BLE001 - a verifier bug must never signal
        allowed, detail = False, f"verifier error: {type(exc).__name__}: {exc}"
    if not allowed:
        event["action"] = "refused"
        event["detail"] = detail
        return event
    pid = event["pid"]
    if pid is None:
        event["action"] = "refused"
        event["detail"] = "no PID in target"
        return event
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        event["action"] = "refused"
        event["detail"] = f"PID {pid} no longer exists"
        return event
    except PermissionError as exc:
        event["action"] = "refused"
        event["detail"] = f"permission denied signalling PID {pid}: {exc}"
        return event
    event["action"] = "signalled"
    return event


# ---------------------------------------------------------------------------
# Status file / launcher interface (pure).
# ---------------------------------------------------------------------------


def build_status(
    reading: Mapping[str, Any],
    profile: HostProfile,
    decision: Decision,
    *,
    now_epoch: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """The JSON written to ``guard_<host>.json``."""
    now = time.time() if now_epoch is None else float(now_epoch)
    start = None
    if decision.level == LEVEL_OK and decision.start_gpu is not None:
        start = {
            "gpu": int(decision.start_gpu),
            "max_planned_minutes": decision.start_max_planned_minutes,
        }
    return {
        "timestamp": utc_iso(now),
        "timestamp_epoch": now,
        "host": profile.host,
        "level": decision.level,
        "reasons": list(decision.reasons),
        "gpus": decision.gpu_blocks,
        "our_jobs": decision.our_jobs,
        "caps": decision.caps,
        "start": start,
        "candidate": decision.candidate,
        "utilization_window": dict(reading.get("utilization_window") or {}),
        "storage": {
            "free_entries": _to_int(reading.get("free_entries")),
            "entry_source": reading.get("entry_source"),
            "free_entries_start_min": profile.free_entries_start_min,
            "free_entries_shed_min": profile.free_entries_shed_min,
            "jepa_data_tb": _to_float(reading.get("jepa_data_tb")),
            "jepa_data_max_tb": profile.jepa_data_max_tb,
            "jepa_data_path": decision.caps.get("jepa_data_path"),
            "jepa_data_scope": decision.caps.get("jepa_data_scope"),
        },
        "guard": {
            "version": GUARD_VERSION,
            "dry_run": bool(dry_run),
            "reading_host": reading.get("host"),
            "interpreter": PYTHON_INTERPRETER,
            "pid_files_pruned": list(reading.get("pid_files_pruned") or []),
            "pid_file_errors": list(reading.get("pid_file_errors") or []),
            "process_metrics_available": reading.get("process_metrics_available"),
        },
    }


#: consumed by ``run_params_from_status``; guard_config.json may override these.
DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "default_planned_minutes": 60.0,
    "timeout_factor": 1.5,
    "kill_after_seconds": 300,
}


def load_run_config(path: Path | None = None) -> dict[str, Any]:
    """Run defaults from ``src/jepa_noise/guard_config.json`` (fallback: built-in)."""
    config = dict(DEFAULT_RUN_CONFIG)
    if path is None:
        path = Path(__file__).resolve().parent.parent / "src" / "jepa_noise" / "guard_config.json"
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return config
    run = payload.get("run") if isinstance(payload, dict) else None
    if isinstance(run, dict):
        for key in config:
            if key in run:
                parsed = _to_float(run[key])
                if parsed is not None:
                    config[key] = int(parsed) if key == "kill_after_seconds" else parsed
    return config


def timeout_seconds(planned_minutes: float, factor: float = 1.5) -> int:
    """``timeout`` seconds: factor x planned minutes, rounded up."""
    minutes = max(0.0, float(planned_minutes))
    return int(math.ceil(minutes * float(factor) * 60.0))


def status_age_seconds(status: Mapping[str, Any], now_epoch: float | None = None) -> float | None:
    """Age of a guard status in seconds; None if its timestamp is unreadable."""
    if not isinstance(status, Mapping):
        return None
    stamp = _to_float(status.get("timestamp_epoch"))
    if stamp is None:
        stamp = parse_utc(status.get("timestamp"))
    if stamp is None:
        return None
    now = time.time() if now_epoch is None else float(now_epoch)
    return now - stamp


def status_is_usable(
    status: Mapping[str, Any],
    *,
    now_epoch: float | None = None,
    max_age_seconds: float = 120.0,
) -> tuple[bool, str]:
    """Freshness gate for launchers: status must be younger than 2 minutes."""
    if not isinstance(status, Mapping):
        return False, "guard status is missing or not a JSON object"
    if not status.get("level"):
        return False, "guard status has no level"
    age = status_age_seconds(status, now_epoch)
    if age is None:
        return False, "guard status has no readable timestamp"
    if age >= max_age_seconds:
        return False, f"guard status is stale: {age:.0f} s old >= {max_age_seconds:.0f} s"
    if age < -max_age_seconds:
        return False, f"guard status timestamp is {abs(age):.0f} s in the future"
    if status.get("level") != LEVEL_OK:
        reasons = status.get("reasons") or []
        shown = "; ".join(str(r) for r in reasons[:5]) if isinstance(reasons, list) else ""
        return False, f"guard level is {status.get('level')}, not OK" + (f" ({shown})" if shown else "")
    return True, ""


def run_params_from_status(
    status: Mapping[str, Any],
    *,
    planned_minutes: float | None = None,
    now_epoch: float | None = None,
    max_age_seconds: float = 120.0,
    expected_host: str | None = None,
    run_config: Mapping[str, Any] | None = None,
    job_kind: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """What ``jepa_run.sh`` needs, or ``(None, refusal reason)``.

    Enforces: real (not dry-run) status, level OK, timestamp strictly younger
    than ``max_age_seconds``, host match, a named start GPU, the guard's
    per-start duration cap, the per-process GPU memory fraction, and -- for a
    training job -- that the status does not already show the training-job cap
    reached (F2: a 4th training job is refused even if it is short).
    """
    config = dict(DEFAULT_RUN_CONFIG)
    if run_config:
        config.update(run_config)
    ok, why = status_is_usable(status, now_epoch=now_epoch, max_age_seconds=max_age_seconds)
    if not ok:
        return None, why
    guard_block = status.get("guard")
    if not isinstance(guard_block, Mapping) or guard_block.get("dry_run") is not False:
        return None, "guard status is a dry-run (or has no dry_run flag); refusing to start"
    if expected_host is not None and status.get("host") != expected_host:
        return None, f"guard status is for host {status.get('host')!r}, expected {expected_host!r}"
    caps = status.get("caps")
    caps_block = caps if isinstance(caps, Mapping) else {}
    if job_kind is not None and str(job_kind).strip().lower() == "training":
        training_cap = _to_int(caps_block.get("training_jobs_max"))
        our_jobs = status.get("our_jobs")
        if training_cap is not None and isinstance(our_jobs, list):
            training_running = 0
            for job in our_jobs:
                if not isinstance(job, Mapping) or job.get("in_pid_file") is False:
                    continue
                kind = str(job.get("job_kind") or "training").strip().lower()
                if kind == "training":
                    training_running += 1
            if training_running >= training_cap:
                return None, (
                    f"training jobs: {training_running} running (limit {training_cap}); "
                    "a further training start is refused"
                )
    start = status.get("start")
    if not isinstance(start, Mapping):
        return None, "guard status does not name a start GPU"
    gpu = _to_int(start.get("gpu"))
    if gpu is None or gpu < 0:
        return None, "guard status names no valid start GPU"
    planned = _to_float(planned_minutes)
    if planned is None:
        planned = _to_float(config.get("default_planned_minutes"))
    if planned is None or planned <= 0:
        return None, "planned duration is missing or not positive"
    max_planned = _to_float(start.get("max_planned_minutes"))
    if max_planned is not None and planned > max_planned:
        return None, (
            f"planned {planned:.0f} min > guard's {max_planned:.0f} min limit for this start"
        )
    fraction = _to_float(caps_block.get("per_process_gpu_memory_fraction"))
    if fraction is None:
        return None, "guard status has no per-process GPU memory cap"
    factor = _to_float(config.get("timeout_factor")) or 1.5
    return {
        "gpu_index": gpu,
        "per_process_memory_fraction": fraction,
        "planned_minutes": planned,
        "max_planned_minutes": max_planned,
        "timeout_seconds": timeout_seconds(planned, factor),
        "kill_after_seconds": int(_to_float(config.get("kill_after_seconds")) or 300),
    }, None


# ---------------------------------------------------------------------------
# I/O layer: read the host.
# ---------------------------------------------------------------------------


def _run_command(command: list[str], timeout: float = 30.0) -> str | None:
    """Run a read-only helper; None on any failure (never raises)."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _read_proc_text(pid: int, name: str) -> str | None:
    try:
        return Path(f"/proc/{int(pid)}/{name}").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _read_proc_cmdline(pid: int) -> str | None:
    """Command line of a process, or None when the process does not exist."""
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    if not raw:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _read_proc_owner(pid: int) -> str | None:
    status = _read_proc_text(pid, "status")
    if not status:
        return None
    for line in status.splitlines():
        if line.startswith("Uid:"):
            parts = line.split()
            if len(parts) >= 2:
                uid = _to_int(parts[1])
                if uid is None:
                    return None
                try:
                    import pwd  # noqa: PLC0415 - Unix only; imported lazily

                    return pwd.getpwuid(uid).pw_name
                except (ImportError, KeyError):
                    return str(uid)
    return None


def _read_proc_cpu_rss(
    pid: int,
    state: "CollectorState",
    now_epoch: float,
    started_epoch: float | None = None,
) -> tuple[float | None, float | None]:
    """CPU percent and RSS in MiB.

    CPU percent is measured over the guard's own sampling interval.  On the
    first sample after a guard (re)start there is no interval yet, so the
    lifetime average since the job's PID-file start time is used instead; the
    next cycle switches to interval measurement.
    """
    stat = _read_proc_text(pid, "stat")
    cpu_percent: float | None = None
    if stat:
        # /proc/<pid>/stat: comm may contain spaces, so cut at the last ')'.
        close = stat.rfind(")")
        if close > 0:
            fields = stat[close + 2 :].split()
            if len(fields) > 12:
                utime = _to_int(fields[11])
                stime = _to_int(fields[12])
                if utime is not None and stime is not None:
                    ticks = utime + stime
                    hz = _clock_ticks()
                    previous = state.process_cpu.get(pid)
                    if previous is not None:
                        previous_epoch, previous_ticks = previous
                        elapsed = now_epoch - previous_epoch
                        if elapsed > 0.5:
                            cpu_percent = max(0.0, 100.0 * (ticks - previous_ticks) / hz / elapsed)
                    elif started_epoch is not None and now_epoch > started_epoch:
                        cpu_percent = max(
                            0.0, 100.0 * (ticks / hz) / (now_epoch - started_epoch)
                        )
                    state.process_cpu[pid] = (now_epoch, ticks)
    status = _read_proc_text(pid, "status")
    rss_mib: float | None = None
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = _to_float(parts[1])
                    if kb is not None:
                        rss_mib = kb / 1024.0
                break
    return cpu_percent, rss_mib


def _clock_ticks() -> float:
    try:
        return float(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError):
        return 100.0


def _read_meminfo() -> tuple[float | None, float | None]:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None, None
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            value = _to_float(parts[1])
            if value is not None:
                values[parts[0][:-1]] = value  # kB
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    return (
        None if total is None else total / 1024.0,
        None if available is None else available / 1024.0,
    )


def _read_loadavg() -> float | None:
    try:
        parts = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    except OSError:
        return None
    if len(parts) < 3:
        return None
    return _to_float(parts[2])


def parse_ceph_quota(
    max_files_text: str | None,
    rfiles_text: str | None,
    rsubdirs_text: str | None,
) -> dict[str, Any] | None:
    """Free entries from CephFS directory-quota xattrs (F4, confirmed source).

    ``free_entries = max_files - rfiles - rsubdirs``.  Values are single
    integers (possibly with trailing newline/whitespace).
    """
    max_files = _to_int(str(max_files_text).strip()) if max_files_text is not None else None
    if max_files is None or max_files < 0:
        return None
    rfiles = _to_int(str(rfiles_text).strip()) if rfiles_text is not None else None
    rsubdirs = _to_int(str(rsubdirs_text).strip()) if rsubdirs_text is not None else None
    rfiles = 0 if rfiles is None else max(0, rfiles)
    rsubdirs = 0 if rsubdirs is None else max(0, rsubdirs)
    free = max_files - rfiles - rsubdirs
    return {
        "source": "cephfs_xattr",
        "quota_max_files": max_files,
        "quota_rfiles": rfiles,
        "quota_rsubdirs": rsubdirs,
        "free_entries": max(0, free),
        "free_entries_clamped": free < 0,
    }


def parse_df_i_free(output: str | None) -> int | None:
    """Parse the IFree column of POSIX ``df -i -P`` (6 columns, IFree at index 3)."""
    if not output:
        return None
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        columns = line.split()
        if len(columns) < 4:
            continue
        if columns[0].lower().startswith("filesystem"):
            continue
        value = _to_int(columns[3])
        if value is not None:
            return value
    return None


def _read_free_entries(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Free entry count: CephFS xattr quota first, logged fallback otherwise.

    Returns ``(entry_info, error)`` where ``entry_info`` always records the
    source and, for CephFS, the three raw quota numbers.
    """
    raw = {}
    getfattr_ok = True
    for key, xattr in (
        ("max_files", CEPH_QUOTA_MAX_FILES),
        ("rfiles", CEPH_QUOTA_RFILES),
        ("rsubdirs", CEPH_QUOTA_RSUBDIRS),
    ):
        text = _run_command(
            ["getfattr", "-n", xattr, "--only-values", str(path)], timeout=20.0
        )
        if text is None:
            getfattr_ok = False
            break
        raw[key] = text
    if getfattr_ok:
        info = parse_ceph_quota(raw["max_files"], raw["rfiles"], raw["rsubdirs"])
        if info is not None:
            return info, None
    fallback_reason = "CephFS quota xattrs unavailable; using non-Ceph fallback"
    try:
        stats = os.statvfs(str(path))
        if stats.f_files > 0:
            return (
                {
                    "source": "statvfs",
                    "free_entries": int(stats.f_ffree),
                    "statvfs_files": int(stats.f_files),
                    "fallback_reason": fallback_reason,
                },
                None,
            )
    except (OSError, ValueError):
        pass
    output = _run_command(["df", "-i", "-P", str(path)], timeout=20.0)
    free = parse_df_i_free(output)
    if free is not None:
        return (
            {
                "source": "df_i",
                "free_entries": free,
                "fallback_reason": fallback_reason,
                "df_i_line": output.strip().splitlines()[-1] if output and output.strip() else None,
            },
            None,
        )
    return None, (
        "cannot read free entry count: CephFS quota xattrs, statvfs and df -i all failed"
    )


def _read_dir_size_tb(path: Path, timeout: float = 300.0) -> float | None:
    """Apparent size of ``path`` in decimal terabytes; None when unmeasurable."""
    try:
        if not path.exists():
            return 0.0
    except OSError:
        return None
    output = _run_command(["du", "-sb", str(path)], timeout=timeout)
    if not output:
        return None
    first = output.split(None, 1)[0] if output.split() else None
    size = _to_float(first)
    if size is None:
        return None
    return size / BYTES_PER_TB


def _read_pmon_utilization() -> dict[int, float] | None:
    """Per-process SM utilisation from ``nvidia-smi pmon -c 1`` (optional)."""
    output = _run_command(["nvidia-smi", "pmon", "-c", "1"], timeout=30.0)
    if not output:
        return None
    usage: dict[int, float] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 4:
            continue
        pid = _to_int(columns[1])
        sm = _to_float(columns[3])
        if pid is not None and pid > 0 and sm is not None:
            usage[pid] = sm
    return usage or None


def _read_gpus(
    our_pids: set[int],
    util_history: deque[tuple[float, dict[int, float]]],
    now_epoch: float,
) -> tuple[bool, list[dict[str, Any]], bool]:
    """Read nvidia-smi.

    Returns ``(available, gpus, per_process_available)``.  A host without
    nvidia-smi yields ``(False, [], False)`` instead of raising.
    """
    base = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30.0,
    )
    if base is None:
        return False, [], False
    gpus: list[dict[str, Any]] = []
    uuid_to_index: dict[str, int] = {}
    instantaneous: dict[int, float] = {}
    for line in base.splitlines():
        line = line.strip()
        if not line:
            continue
        columns = [part.strip() for part in line.split(",")]
        if len(columns) < 6:
            continue
        index = _to_int(columns[0])
        if index is None:
            index = len(gpus)
        uuid = columns[2]
        total = _to_float(columns[3])
        used = _to_float(columns[4])
        util = _to_float(columns[5])
        uuid_to_index[uuid] = index
        if util is not None:
            instantaneous[index] = util
        gpus.append(
            {
                "index": index,
                "name": columns[1],
                "uuid": uuid,
                "memory_total_mib": total,
                "memory_used_mib": used,
                "utilization_5m_mean_pct": util,  # replaced below by the 5-minute mean
                "processes": [],
            }
        )
    util_history.append((now_epoch, dict(instantaneous)))
    while util_history and util_history[0][0] < now_epoch - 900.0:
        util_history.popleft()
    window = utilization_window(util_history, now_epoch)
    window_start = now_epoch - UTIL_WINDOW_SECONDS
    samples = [sample for stamp, sample in util_history if stamp >= window_start]
    for gpu in gpus:
        values = [sample[gpu["index"]] for sample in samples if gpu["index"] in sample]
        # R1: an unpopulated window must never masquerade as a 5-minute mean
        gpu["utilization_5m_mean_pct"] = (
            (sum(values) / len(values)) if window["warm"] and values else None
        )
    apps = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=30.0,
    )
    per_process_available = apps is not None
    pmon = _read_pmon_utilization()
    processes: list[dict[str, Any]] = []
    if apps:
        for line in apps.splitlines():
            line = line.strip()
            if not line:
                continue
            columns = [part.strip() for part in line.split(",")]
            if len(columns) < 3:
                continue
            gpu_index = uuid_to_index.get(columns[0])
            pid = _to_int(columns[1])
            memory = _to_float(columns[2])
            if gpu_index is None or pid is None:
                continue
            owner = _read_proc_owner(pid)
            cmdline = _read_proc_cmdline(pid) or ""
            processes.append(
                {
                    "gpu_index": gpu_index,
                    "pid": pid,
                    "owner": owner,
                    "memory_mib": memory,
                    "in_pid_file": pid in our_pids,
                    "utilization_pct": pmon.get(pid) if pmon else None,
                    "cmdline": cmdline,
                }
            )
    for gpu in gpus:
        gpu["processes"] = [proc for proc in processes if proc["gpu_index"] == gpu["index"]]
    return True, gpus, per_process_available


def _pid_alive(pid: int | None) -> bool:
    if pid is None or int(pid) <= 1:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def pidfile_tag_matches_cmdline(tag: Any, cmdline: str | None) -> bool:
    """F10: a PID file is live only if its tag is on the live command line.

    Guards against PID reuse: a stale pid.json whose PID now belongs to an
    unrelated process must be pruned, not counted as one of our GPU jobs.
    """
    if not isinstance(tag, str) or not tag or not cmdline:
        return False
    return tag in cmdline and TAG_RE.search(cmdline) is not None


def pidfile_belongs_to_host(payload: Mapping[str, Any] | None, host: str) -> bool:
    """F10: the jobs directory is shared, so a PID file is ours only on its host."""
    return isinstance(payload, Mapping) and payload.get("host") == host


def load_pidfile_records(
    jobs_dir: Path,
    host: str,
    liveness: Any = None,
    *,
    prune: bool = True,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Read host-filtered PID files and prune stale ones (F10).

    Returns ``(records, pruned_paths, errors)``.  A PID file whose process is
    gone is unlinked when ``prune`` is True (R9: ``--dry-run`` passes False, so
    a dry run never writes); a PID file for the other host is ignored.
    """
    records: list[dict[str, Any]] = []
    pruned: list[str] = []
    errors: list[str] = []
    alive_check = liveness if liveness is not None else _pid_alive
    try:
        pid_files = sorted(Path(jobs_dir).glob("*/pid.json"))
    except OSError:
        return records, pruned, errors
    for pid_file in pid_files:
        try:
            payload = json.loads(pid_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{pid_file}: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{pid_file}: not a JSON object")
            continue
        if not pidfile_belongs_to_host(payload, host):
            continue  # other host's job in the shared jobs dir
        pid = _to_int(payload.get("pid"))
        if pid is None or pid <= 1:
            errors.append(f"{pid_file}: no valid pid")
            continue
        if not alive_check(pid):
            pruned.append(str(pid_file))
            if prune:
                try:
                    pid_file.unlink()
                except OSError:
                    pass
            continue
        records.append({"pid": pid, "payload": payload, "pid_file": str(pid_file)})
    return records, pruned, errors


def build_process_table(entries: Iterable[Mapping[str, Any]]) -> dict[int, dict[str, int | None]]:
    """Build ``{pid: {ppid, pgrp}}`` from parsed /proc entries (pure)."""
    table: dict[int, dict[str, int | None]] = {}
    for entry in entries:
        pid = _to_int(entry.get("pid"))
        if pid is None or pid <= 0:
            continue
        table[pid] = {
            "ppid": _to_int(entry.get("ppid")),
            "pgrp": _to_int(entry.get("pgrp")),
        }
    return table


def process_tree_pids(root_pid: int | None, table: Mapping[int, Mapping[str, Any]]) -> list[int]:
    """``root_pid`` plus all /proc parent-link descendants (pure)."""
    root = _to_int(root_pid)
    if root is None:
        return []
    children: dict[int, list[int]] = {}
    for pid, info in table.items():
        parent = _to_int(info.get("ppid"))
        if parent is not None:
            children.setdefault(parent, []).append(pid)
    out: list[int] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def process_group_pids(pgid: int | None, table: Mapping[int, Mapping[str, Any]]) -> list[int]:
    """PIDs in ``pgid``'s process group (pure); empty for an unknown pgid."""
    group = _to_int(pgid)
    if group is None or group <= 0:
        return []
    return [pid for pid, info in table.items() if _to_int(info.get("pgrp")) == group]


def _read_process_table() -> dict[int, dict[str, int | None]]:
    """Read ``{pid: {ppid, pgrp}}`` for all processes from /proc."""
    entries: list[dict[str, Any]] = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return {}
    for name in names:
        if not name.isdigit():
            continue
        stat = _read_proc_text(int(name), "stat")
        if not stat:
            continue
        close = stat.rfind(")")
        if close <= 0:
            continue
        fields = stat[close + 2 :].split()
        if len(fields) < 3:
            continue
        entries.append({"pid": int(name), "ppid": fields[1], "pgrp": fields[2]})
    return build_process_table(entries)


def _read_our_jobs(
    records: list[dict[str, Any]],
    state: "CollectorState",
    now_epoch: float,
    gpu_processes: list[dict[str, Any]],
    process_table: Mapping[int, Mapping[str, Any]],
    *,
    prune: bool = True,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Our jobs with /proc CPU/RSS summed over the job's process tree (F6).

    Returns ``(jobs, metrics_available, errors)``.  The GPU index is the
    *observed* nvidia-smi index when available (F7); the declared index is
    kept for mismatch detection.
    """
    jobs: list[dict[str, Any]] = []
    errors: list[str] = []
    metrics_available = True
    proc_mounted = os.path.isdir("/proc")
    gpu_by_pid = {proc["pid"]: proc for proc in gpu_processes}
    for record in records:
        pid = record["pid"]
        payload = record["payload"]
        pid_file = record["pid_file"]
        run_id = payload.get("run_id") or Path(pid_file).parent.name
        tag = payload.get("tag") or f"jepa_{run_id}"
        if not proc_mounted:
            metrics_available = False
            continue
        cmdline = _read_proc_cmdline(pid)
        if cmdline is None or not pidfile_tag_matches_cmdline(tag, cmdline):
            # raced with exit, or the PID now belongs to another process
            if prune:
                try:
                    Path(pid_file).unlink()
                except OSError:
                    pass
                errors.append(f"{pid_file}: stale or recycled PID (pruned)")
            else:
                errors.append(f"{pid_file}: stale or recycled PID (dry-run, not pruned)")
            continue
        owner = _read_proc_owner(pid)
        pgid = _to_int(payload.get("pgid"))
        if pgid is None:
            pgid = pid
        tree = set(process_tree_pids(pid, process_table))
        tree.update(process_group_pids(pgid, process_table))
        tree.add(pid)
        cpu_total: float | None = None
        rss_total: float | None = None
        for tree_pid in sorted(tree):
            cpu, rss = _read_proc_cpu_rss(
                tree_pid,
                state,
                now_epoch,
                started_epoch=parse_utc(payload.get("started_at_utc")) if tree_pid == pid else None,
            )
            if cpu is not None:
                cpu_total = (cpu_total or 0.0) + cpu
            if rss is not None:
                rss_total = (rss_total or 0.0) + rss
        if cpu_total is None or rss_total is None:
            metrics_available = False
        gpu_proc = gpu_by_pid.get(pid)
        declared_index = _to_int(payload.get("gpu_index"))
        observed_index = gpu_proc["gpu_index"] if gpu_proc else None
        jobs.append(
            {
                "pid": pid,
                "run_id": run_id,
                "tag": tag,
                "owner": owner,
                "cmdline": cmdline,
                "in_pid_file": True,
                "gpu_index": observed_index if observed_index is not None else declared_index,
                "gpu_index_declared": declared_index,
                "pgid": pgid,
                "tree_pids": sorted(tree),
                "gpu_memory_mib": gpu_proc["memory_mib"] if gpu_proc else None,
                "cpu_percent": cpu_total,
                "rss_mib": rss_total,
                "started_at_utc": payload.get("started_at_utc"),
                "job_kind": _job_kind(payload),
                "planned_minutes": _to_float(payload.get("planned_minutes")),
                "pid_file": str(pid_file),
            }
        )
    our_pids = {job["pid"] for job in jobs}
    for proc in gpu_processes:
        if proc["pid"] in our_pids:
            proc["in_pid_file"] = True
    for stale_pid in [p for p, value in state.process_cpu.items() if now_epoch - value[0] > 900.0]:
        state.process_cpu.pop(stale_pid, None)
    return jobs, metrics_available, errors


@dataclass
class CollectorState:
    """Small in-memory history the guard needs between cycles."""

    util_samples: deque[tuple[float, dict[int, float]]] = field(default_factory=deque)
    data_size: tuple[float, float] | None = None  # (epoch, TB)
    process_cpu: dict[int, tuple[float, float]] = field(default_factory=dict)
    data_size_seconds: float = 1800.0


def collect_reading(
    profile: HostProfile,
    jobs_dir: Path,
    state: CollectorState,
    *,
    now_epoch: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Read everything the decision core needs.  Missing inputs stay None.

    ``dry_run=True`` makes the collection read-only: stale PID files are
    reported but not pruned (R9).
    """
    now = time.time() if now_epoch is None else float(now_epoch)
    jobs_dir = Path(jobs_dir)
    jepa_root = jobs_dir.parent
    quota_path = jepa_root.parent
    data_path = jepa_root / JEPA_DATA_SUBDIR
    prune = not dry_run
    # F10: only this host's PID files count; stale ones are pruned unless dry-run.
    records, pruned_pid_files, pid_errors = load_pidfile_records(
        jobs_dir, profile.host, prune=prune
    )
    our_pids = {record["pid"] for record in records}
    nvidia_available, gpus, per_process_available = _read_gpus(
        our_pids, state.util_samples, now
    )
    gpu_processes = [proc for gpu in gpus for proc in gpu["processes"]]
    process_table = _read_process_table()
    jobs, metrics_available, job_errors = _read_our_jobs(
        records, state, now, gpu_processes, process_table, prune=prune
    )
    pid_errors.extend(job_errors)
    our_pids = {job["pid"] for job in jobs}
    for gpu in gpus:
        for proc in gpu["processes"]:
            proc["in_pid_file"] = proc["pid"] in our_pids
    mem_total, mem_available = _read_meminfo()
    load15 = _read_loadavg()
    cores = os.cpu_count()
    entry_source, entry_error = _read_free_entries(quota_path)
    free_entries = _to_int(entry_source.get("free_entries")) if entry_source else None
    if entry_source is None:
        entry_source = {"source": "unavailable", "error": entry_error}
    data_tb: float | None
    if state.data_size is not None and now - state.data_size[0] <= state.data_size_seconds:
        data_tb = state.data_size[1]
    else:
        data_tb = _read_dir_size_tb(data_path)
        if data_tb is not None:
            state.data_size = (now, data_tb)
    reading: dict[str, Any] = {
        "host": profile.host,
        "collected_at_utc": utc_iso(now),
        "collected_at_epoch": now,
        "cores": cores,
        "nvidia_smi_available": nvidia_available,
        "gpus": gpus,
        "host_memory_total_mib": mem_total,
        "host_memory_available_mib": mem_available,
        "loadavg_15m": load15,
        "utilization_window": utilization_window(state.util_samples, now),
        "free_entries": free_entries,
        "entry_source": entry_source,
        "jepa_data_tb": data_tb,
        "process_metrics_available": metrics_available,
        "per_process_gpu_table_available": per_process_available,
        "pid_files_pruned": pruned_pid_files,
        "pid_file_errors": pid_errors,
        "our_processes": jobs,
        "reading_notes": {
            "jobs_dir": str(jobs_dir),
            "quota_path": str(quota_path),
            "data_path": str(data_path),
            "jepa_data_scope": str(data_path),
            "interpreter": PYTHON_INTERPRETER,
        },
    }
    return reading


def write_status_atomic(path: Path, status: Mapping[str, Any]) -> Path:
    """Write JSON atomically (temp file + fsync + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(status, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def append_log_line(path: Path, event: Mapping[str, Any]) -> str:
    """Append one JSON line; return the line (without newline)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return line


def read_last_signal_epoch(log_path: Path, max_lines: int = 2000) -> float | None:
    """Recover the time of the last live SHED SIGTERM from the event log.

    This keeps the 5-minute re-check rule across guard restarts (and across
    ``--once`` invocations) without a second state file.  Dry-run events do not
    count: they describe a signal that was not sent.
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as stream:
            lines = deque(stream, maxlen=max_lines)
    except OSError:
        return None
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("event") == "shed_signal" and event.get("action") == "signalled":
            epoch = _to_float(event.get("timestamp_epoch"))
            if epoch is not None:
                return epoch
    return None


def pid_listed_in_pid_files(jobs_dir: Path, pid: int, tag: str | None = None) -> bool:
    """True when ``pid`` (and optionally ``tag``) is in one of our PID files."""
    try:
        pid_files = sorted(Path(jobs_dir).glob("*/pid.json"))
    except OSError:
        return False
    for pid_file in pid_files:
        try:
            payload = json.loads(pid_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if _to_int(payload.get("pid")) != int(pid):
            continue
        if tag:
            file_tag = payload.get("tag")
            if not isinstance(file_tag, str) or not file_tag:
                file_tag = f"jepa_{payload.get('run_id') or pid_file.parent.name}"
            if file_tag != tag:
                continue
        return True
    return False


def verify_pid_matches(
    target: Mapping[str, Any], jobs_dir: Path | None = None
) -> tuple[bool, str]:
    """Re-check owner, command line, tag and PID-file listing before a signal."""
    pid = _to_int(target.get("pid"))
    if pid is None or pid <= 1:
        return False, "invalid PID"
    if pid == os.getpid():
        return False, "refusing to signal the guard itself"
    cmdline = _read_proc_cmdline(pid)
    if cmdline is None:
        return False, f"PID {pid} no longer exists"
    owner = _read_proc_owner(pid)
    if owner != GUARD_OWNER:
        return False, f"PID {pid} owner is {owner!r}, not {GUARD_OWNER!r}"
    tag = target.get("tag")
    if isinstance(tag, str) and tag and tag not in cmdline:
        return False, f"PID {pid} command line does not contain tag {tag!r}"
    if not TAG_RE.search(cmdline):
        return False, f"PID {pid} command line is not tagged jepa_"
    if not _is_pidfile_listed(target):
        return False, f"PID {pid} is not listed in one of our PID files"
    if jobs_dir is not None and not pid_listed_in_pid_files(jobs_dir, pid, tag if isinstance(tag, str) else None):
        return False, f"PID {pid} is not in a PID file under {jobs_dir}"
    return True, ""


# ---------------------------------------------------------------------------
# Main loop / CLI.
# ---------------------------------------------------------------------------


def _status_paths(jobs_dir: Path, host: str) -> tuple[Path, Path]:
    return jobs_dir / f"guard_{host}.json", jobs_dir / f"guard_{host}.log"


def run_once(
    profile: HostProfile,
    jobs_dir: Path,
    state: CollectorState,
    *,
    candidate: Mapping[str, Any] | None,
    dry_run: bool,
    shed_state: dict[str, float | int | None],
    now_epoch: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """One evaluation cycle: collect, decide, log/write, act.  Returns events."""
    start_epoch = time.time() if now_epoch is None else float(now_epoch)
    reading = collect_reading(
        profile, jobs_dir, state, now_epoch=start_epoch, dry_run=dry_run
    )
    # F11: the published timestamp is taken after the (possibly slow) collection,
    # so a status can never look fresher than the reading it describes.
    now = time.time() if now_epoch is None else float(now_epoch)
    decision = evaluate(reading, profile, candidate)
    status = build_status(reading, profile, decision, now_epoch=now, dry_run=dry_run)
    status_path, log_path = _status_paths(jobs_dir, profile.host)
    evaluate_event = dict(status)
    evaluate_event["event"] = "evaluate"
    append_log_line(log_path, evaluate_event)
    if not dry_run:
        write_status_atomic(status_path, status)

    signal_event: dict[str, Any] | None = None
    shed_event: dict[str, Any] | None = None
    if decision.level == LEVEL_SHED:
        target = select_shed_target(reading)
        if target is None:
            shed_event = {
                "timestamp": utc_iso(now),
                "timestamp_epoch": now,
                "event": "shed_no_target",
                "level": LEVEL_SHED,
                "detail": "SHED but no PID-file/dwong/jepa_-tagged process can be selected",
            }
            append_log_line(log_path, shed_event)
        elif shed_action_due(now, shed_state.get("last_shed_epoch")):
            verifier = (
                None
                if dry_run
                else (lambda item: verify_pid_matches(item, jobs_dir=jobs_dir))
            )
            signal_event = shed_signal(
                target, dry_run=dry_run, verifier=verifier, now_epoch=now
            )
            append_log_line(log_path, signal_event)
            if signal_event.get("action") in {"signalled", "would_signal"}:
                shed_state["last_shed_epoch"] = now
                shed_state["last_target_pid"] = signal_event.get("pid")
        else:
            last = shed_state.get("last_shed_epoch")
            wait = 300.0 - (now - float(last)) if last is not None else 0.0
            shed_event = {
                "timestamp": utc_iso(now),
                "timestamp_epoch": now,
                "event": "shed_retry_wait",
                "level": LEVEL_SHED,
                "detail": f"SHED persists; next signal allowed in {max(0.0, wait):.0f} s",
            }
            append_log_line(log_path, shed_event)
    else:
        shed_state["last_shed_epoch"] = None
        shed_state["last_target_pid"] = None
    return reading, status, signal_event or shed_event


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="JEPA resource guard (runbook v4 section 6.2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", choices=sorted(PROFILES), help="host profile to use")
    parser.add_argument("--interval", type=float, default=120.0, help="loop period in seconds")
    parser.add_argument("--once", action="store_true", help="one evaluation, then exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="log only: no status file, no signals (required for the 10-minute host dry run)",
    )
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR_DEFAULT, help="jobs directory")
    parser.add_argument(
        "--candidate-kind",
        default=None,
        help="kind of the job a start decision is asked for (training, evaluation, ...)",
    )
    parser.add_argument(
        "--candidate-minutes",
        type=float,
        default=None,
        help="planned duration of the candidate job in minutes",
    )
    # Launcher interface (used by scripts/jepa_run.sh).
    parser.add_argument(
        "--status-for-run",
        type=Path,
        default=None,
        metavar="PATH",
        help="print START_GPU=... / PER_PROCESS_MEMORY_FRACTION=... for a launcher; exit 3 to refuse",
    )
    parser.add_argument("--planned-minutes", type=float, default=None, help="planned job duration")
    parser.add_argument(
        "--job-kind",
        default=None,
        help="kind of the job asking to start (training/evaluation/...); F2 training cap",
    )
    parser.add_argument("--max-age-seconds", type=float, default=120.0, help="guard status freshness")
    args = parser.parse_args(argv)

    if args.status_for_run is not None:
        try:
            status = json.loads(Path(args.status_for_run).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"jepa_guard: refusing start: cannot read guard status {args.status_for_run}: {exc}", file=sys.stderr)
            return 3
        params, refusal = run_params_from_status(
            status,
            planned_minutes=args.planned_minutes,
            max_age_seconds=args.max_age_seconds,
            expected_host=args.host,
            run_config=load_run_config(),
            job_kind=args.job_kind,
        )
        if params is None:
            print(f"jepa_guard: refusing start: {refusal}", file=sys.stderr)
            return 3
        max_planned = params["max_planned_minutes"]
        print(f"START_GPU={params['gpu_index']}")
        print(f"PER_PROCESS_MEMORY_FRACTION={params['per_process_memory_fraction']}")
        print(f"PLANNED_MINUTES={params['planned_minutes']}")
        print(f"TIMEOUT_SECONDS={params['timeout_seconds']}")
        print(f"KILL_AFTER_SECONDS={params['kill_after_seconds']}")
        print(f"MAX_PLANNED_MINUTES={'' if max_planned is None else max_planned}")
        return 0

    if not args.host:
        parser.error("--host is required (deepthought or deepthought2)")
    profile = get_profile(args.host)
    candidate: dict[str, Any] | None = None
    if args.candidate_kind is not None or args.candidate_minutes is not None:
        candidate = {
            "kind": args.candidate_kind or "other",
            "planned_minutes": args.candidate_minutes
            if args.candidate_minutes is not None
            else DEFAULT_START_CANDIDATE_MINUTES,
        }
    state = CollectorState()
    _, log_path = _status_paths(Path(args.jobs_dir), profile.host)
    shed_state: dict[str, float | int | None] = {
        "last_shed_epoch": read_last_signal_epoch(log_path),
        "last_target_pid": None,
    }
    if args.once:
        reading, status, action = run_once(
            profile,
            Path(args.jobs_dir),
            state,
            candidate=candidate,
            dry_run=args.dry_run,
            shed_state=shed_state,
        )
        print(json.dumps(status, indent=2))
        if action is not None:
            print(json.dumps(action, indent=2))
        return 0
    print(
        f"jepa_guard: host={profile.host} jobs_dir={args.jobs_dir} interval={args.interval:.0f}s "
        f"dry_run={args.dry_run}",
        flush=True,
    )
    try:
        while True:
            run_once(
                profile,
                Path(args.jobs_dir),
                state,
                candidate=candidate,
                dry_run=args.dry_run,
                shed_state=shed_state,
            )
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("jepa_guard: interrupted", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
