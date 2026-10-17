"""Unit tests for the JEPA resource guard (runbook v4 section 6.2).

No GPU, no nvidia-smi, no network: every test drives the pure decision core
with a synthetic "reading" dict.  The I/O helpers that are exercised
(atomic write, log append, dry-run signal path) use a temp directory only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("jepa_guard", ROOT / "scripts" / "jepa_guard.py")
guard = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = guard  # dataclasses need the module registered with __future__ annotations
_SPEC.loader.exec_module(guard)

_LAUNCH_SPEC = importlib.util.spec_from_file_location(
    "jepa_launch", ROOT / "scripts" / "jepa_launch.py"
)
launcher = importlib.util.module_from_spec(_LAUNCH_SPEC)
sys.modules[_LAUNCH_SPEC.name] = launcher
_LAUNCH_SPEC.loader.exec_module(launcher)

LAUNCHER_SCRIPT = ROOT / "scripts" / "jepa_launch.py"

DT = "deepthought"
DT2 = "deepthought2"
NOW = 1_760_000_000.0  # fixed UTC epoch for status/freshness tests
TITAN_TOTAL_MIB = 12288.0
L40S_TOTAL_MIB = 49152.0
GPU_COUNT = {DT: 4, DT2: 2}
HOST_TOTAL_MIB = {DT: 512.0 * 1024.0, DT2: 256.0 * 1024.0}


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu(index: int, *, total: float | None = None, used: float = 0.0, util: float = 2.0, processes=()):
    if total is None:
        total = TITAN_TOTAL_MIB
    return {
        "index": index,
        "name": "synthetic-gpu",
        "memory_total_mib": total,
        "memory_used_mib": used,
        "utilization_5m_mean_pct": util,
        "processes": list(processes),
    }


def process(pid: int, *, owner: str = "ssjostrom", memory_mib: float = 100.0,
            in_pid_file: bool = False, utilization_pct=None, cmdline: str = ""):
    return {
        "pid": pid,
        "owner": owner,
        "memory_mib": memory_mib,
        "in_pid_file": in_pid_file,
        "utilization_pct": utilization_pct,
        "cmdline": cmdline,
    }


def our_job(
    run_id: str,
    pid: int,
    gpu_index: int | None,
    *,
    owner: str = "dwong",
    job_kind: str = "training",
    started_epoch: float = NOW - 3600.0,
    cpu_percent: float = 100.0,
    rss_mib: float = 2048.0,
    gpu_memory_mib: float | None = None,
    in_pid_file: bool = True,
    cmdline: str | None = None,
):
    tag = f"jepa_{run_id}"
    return {
        "pid": pid,
        "run_id": run_id,
        "tag": tag,
        "owner": owner,
        "cmdline": f"python -c '...' 0.8 jepa_noise.train {tag} --config x" if cmdline is None else cmdline,
        "in_pid_file": in_pid_file,
        "gpu_index": gpu_index,
        "gpu_memory_mib": gpu_memory_mib,
        "cpu_percent": cpu_percent,
        "rss_mib": rss_mib,
        "started_at_utc": iso(started_epoch),
        "job_kind": job_kind,
        "planned_minutes": 120.0,
        "pid_file": f"/tmp/jepa_jobs/{run_id}/pid.json",
    }


def reading(host: str, **overrides):
    """A clean synthetic reading: everything idle and healthy."""
    base = {
        "host": host,
        "cores": 40,
        "nvidia_smi_available": True,
        "gpus": [
            gpu(i, total=TITAN_TOTAL_MIB if host == DT else L40S_TOTAL_MIB)
            for i in range(GPU_COUNT[host])
        ],
        "host_memory_total_mib": HOST_TOTAL_MIB[host],
        "host_memory_available_mib": 0.78 * HOST_TOTAL_MIB[host],
        "loadavg_15m": 1.0,
        "utilization_window": {
            "warm": True,
            "sample_count": 3,
            "span_seconds": 240.0,
            "window_seconds": 300.0,
            "samples_required": 3,
            "span_required_seconds": 240.0,
            "reason": None,
        },
        "free_entries": 31_944,
        "jepa_data_tb": 0.4,
        "process_metrics_available": True,
        "per_process_gpu_table_available": True,
        "entry_source": {
            "source": "cephfs_xattr",
            "quota_max_files": 500_000,
            "quota_rfiles": 416_387,
            "quota_rsubdirs": 51_669,
            "free_entries": 31_944,
        },
        "our_processes": [],
        "reading_notes": {
            "jobs_dir": "/ceph/dwong/jepa/jobs",
            "quota_path": "/ceph/dwong",
            "data_path": "/ceph/dwong/jepa/data",
        },
    }
    base.update(overrides)
    return base


def evaluate(host: str, *, candidate=None, **overrides):
    r = reading(host, **overrides)
    return guard.evaluate(r, guard.get_profile(host), candidate)


def assert_hold(decision, needle: str):
    assert decision.level == "HOLD", decision.reasons
    assert decision.start_gpu is None
    joined = " | ".join(decision.reasons)
    assert needle in joined, joined


def assert_shed(decision, needle: str):
    assert decision.level == "SHED", decision.reasons
    assert decision.start_gpu is None
    joined = " | ".join(decision.reasons)
    assert needle in joined, joined


# ---------------------------------------------------------------------------
# Profiles: the exact section 6.2 numbers.
# ---------------------------------------------------------------------------


def test_deepthought_profile_numbers():
    p = guard.PROFILES[DT]
    assert p.gpu_count_max == 4
    assert p.training_jobs_max == 3
    assert p.fourth_gpu_short_job_max_minutes == 30.0
    assert p.gpu_memory_fraction_max == pytest.approx(0.80)
    # "<= 80% of the card (about 9.6 GB)" on a 12 GB Titan
    assert p.gpu_memory_fraction_max * TITAN_TOTAL_MIB / 1024.0 == pytest.approx(9.6)
    assert p.start_util_5m_max_pct == 20.0
    assert p.other_user_memory_mib_max == 1024.0
    assert p.others_plus_ours_fraction_max is None
    assert p.cpu_cores_abs_max == 16 and p.cpu_core_fraction_max is None
    assert p.cpu_core_cap(40) == 16
    assert p.ram_total_max_gb == 128.0
    assert p.free_ram_floor_fraction == pytest.approx(0.20)
    assert p.load_start_ratio == pytest.approx(0.70)
    assert p.free_entries_start_min == 30_000
    assert p.jepa_data_max_tb == 2.0
    assert p.free_ram_shed_fraction == pytest.approx(0.10)
    assert p.gpu_memory_shed_fraction == pytest.approx(0.95)
    assert p.load_shed_ratio == pytest.approx(0.90)
    assert p.free_entries_shed_min == 10_000
    assert p.foreign_heavy_memory_mib == 2048.0
    assert p.foreign_heavy_util_pct == 30.0


def test_deepthought2_profile_numbers():
    p = guard.PROFILES[DT2]
    assert p.gpu_count_max == 2
    assert p.training_jobs_max == 2
    assert p.fourth_gpu_short_job_max_minutes is None
    assert p.gpu_memory_fraction_max == pytest.approx(0.45)
    # "<= 45% of the card (about 21 GB)" on a 48 GB L40S
    assert p.gpu_memory_fraction_max * L40S_TOTAL_MIB / 1024.0 == pytest.approx(21.6)
    assert p.start_util_5m_max_pct == 60.0
    assert p.other_user_memory_mib_max is None
    assert p.others_plus_ours_fraction_max == pytest.approx(0.85)
    assert p.cpu_cores_abs_max == 16 and p.cpu_core_fraction_max == pytest.approx(0.40)
    assert p.cpu_core_cap(40) == 16           # 40% of 40 = 16
    assert p.cpu_core_cap(20) == 8            # "or 40% of the cores if fewer"
    assert p.cpu_core_cap(10) == 4
    assert p.ram_total_max_gb == 64.0
    assert p.free_ram_floor_fraction == pytest.approx(0.20)
    assert p.load_start_ratio == pytest.approx(0.70)
    assert p.free_entries_start_min == 30_000
    assert p.jepa_data_max_tb == 2.0
    assert p.free_ram_shed_fraction == pytest.approx(0.10)
    assert p.gpu_memory_shed_fraction == pytest.approx(0.95)
    assert p.load_shed_ratio == pytest.approx(0.90)
    assert p.free_entries_shed_min == 10_000


def test_get_profile_unknown_host_raises():
    with pytest.raises(ValueError):
        guard.get_profile("some-other-host")


# ---------------------------------------------------------------------------
# Clean readings are OK on both hosts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [DT, DT2])
def test_clean_reading_is_ok(host):
    d = evaluate(host)
    assert d.level == "OK", d.reasons
    assert d.reasons == []
    assert d.start_gpu is not None
    assert 0 <= d.start_gpu < GPU_COUNT[host]
    # the status file must expose the per-process cap of the host's card
    total = TITAN_TOTAL_MIB if host == DT else L40S_TOTAL_MIB
    fraction = guard.PROFILES[host].gpu_memory_fraction_max
    assert d.caps["per_process_gpu_memory_fraction"] == fraction
    assert d.caps["per_process_gpu_memory_cap_mib"]["0"] == pytest.approx(fraction * total)


@pytest.mark.parametrize("host", [DT, DT2])
def test_clean_reading_ok_selected_gpu_is_free(host):
    r = reading(host, gpus=[gpu(i, total=TITAN_TOTAL_MIB if host == DT else L40S_TOTAL_MIB,
                                used=0.0, util=1.0) for i in range(GPU_COUNT[host])])
    d = guard.evaluate(r, guard.get_profile(host))
    assert d.level == "OK"
    assert d.start_gpu == 0


# ---------------------------------------------------------------------------
# HOLD: every start condition, one reading each.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [DT, DT2])
def test_hold_on_host_load(host):
    d = evaluate(host, loadavg_15m=0.71 * 40)  # > 0.70 x 40 cores
    assert_hold(d, "15-minute load")


@pytest.mark.parametrize("host", [DT, DT2])
def test_hold_on_low_free_ram(host):
    r = reading(host)
    r["host_memory_available_mib"] = 0.19 * HOST_TOTAL_MIB[host]
    d = guard.evaluate(r, guard.get_profile(host))
    assert_hold(d, "free RAM")


@pytest.mark.parametrize("host", [DT, DT2])
def test_hold_on_low_free_entries(host):
    d = evaluate(host, free_entries=29_999)
    assert_hold(d, "free entries")


@pytest.mark.parametrize("host", [DT, DT2])
def test_hold_on_jepa_data_above_2tb(host):
    d = evaluate(host, jepa_data_tb=2.01)
    assert_hold(d, "JEPA data")


@pytest.mark.parametrize("host", [DT, DT2])
def test_hold_on_missing_input(host):
    d = evaluate(host, loadavg_15m=None)
    assert_hold(d, "missing input: loadavg_15m")


@pytest.mark.parametrize(
    "missing_key",
    [key for key, _ in guard.REQUIRED_READING_KEYS],
)
def test_missing_required_input_is_never_ok(missing_key):
    r = reading(DT)
    r.pop(missing_key, None)
    d = guard.evaluate(r, guard.get_profile(DT))
    assert d.level != "OK"
    assert any("missing input" in reason for reason in d.reasons), d.reasons


def test_nvidia_smi_unavailable_reports_reason_instead_of_crashing():
    r = reading(DT)
    r["nvidia_smi_available"] = False
    r["gpus"] = []
    d = guard.evaluate(r, guard.get_profile(DT))
    assert d.level == "HOLD"
    assert any("nvidia-smi unavailable" in reason for reason in d.reasons), d.reasons


def test_missing_per_process_gpu_table_is_not_ok():
    d = evaluate(DT, per_process_gpu_table_available=False)
    assert_hold(d, "per_process_gpu_table_available")


@pytest.mark.parametrize("host", [DT, DT2])
def test_foreign_heavy_work_on_a_gpu_we_use_holds(host):
    total = TITAN_TOTAL_MIB if host == DT else L40S_TOTAL_MIB
    jobs = [our_job("r1", 1001, 0)]
    gpus = [gpu(i, total=total) for i in range(GPU_COUNT[host])]
    gpus[0] = gpu(
        0,
        total=total,
        used=3000.0,
        processes=[process(1001, owner="dwong", memory_mib=9000.0, in_pid_file=True),
                   process(2002, owner="otheruser", memory_mib=3000.0)],
    )
    d = guard.evaluate(reading(host, gpus=gpus, our_processes=jobs), guard.get_profile(host))
    assert_hold(d, "another user's processes hold")


def test_foreign_heavy_utilisation_on_a_gpu_we_use_holds():
    total = TITAN_TOTAL_MIB
    jobs = [our_job("r1", 1001, 0)]
    gpus = [gpu(i) for i in range(4)]
    gpus[0] = gpu(
        0,
        total=total,
        processes=[
            process(1001, owner="dwong", memory_mib=9000.0, in_pid_file=True, utilization_pct=80.0),
            process(2002, owner="otheruser", memory_mib=500.0, utilization_pct=40.0),
        ],
    )
    d = guard.evaluate(reading(DT, gpus=gpus, our_processes=jobs), guard.get_profile(DT))
    assert_hold(d, "another user's processes use")


def test_foreign_light_utilisation_on_a_gpu_we_use_does_not_hold():
    total = TITAN_TOTAL_MIB
    jobs = [our_job("r1", 1001, 0)]
    gpus = [gpu(i) for i in range(4)]
    gpus[0] = gpu(
        0,
        total=total,
        processes=[
            process(1001, owner="dwong", memory_mib=9000.0, in_pid_file=True, utilization_pct=80.0),
            process(2002, owner="otheruser", memory_mib=500.0, utilization_pct=20.0),
        ],
    )
    d = guard.evaluate(reading(DT, gpus=gpus, our_processes=jobs), guard.get_profile(DT))
    assert d.level == "OK", d.reasons
    assert d.start_gpu == 1


@pytest.mark.parametrize("host", [DT, DT2])
def test_foreign_gpu_owner_reading_holds_when_a_candidate_gpu_is_blocked(host):
    total = TITAN_TOTAL_MIB if host == DT else L40S_TOTAL_MIB
    # foreign processes that block every GPU by the host's own start condition
    foreign_mib = 1500.0 if host == DT else 25000.0
    gpus = [
        gpu(i, total=total, used=foreign_mib, processes=[process(3000 + i, memory_mib=foreign_mib)])
        for i in range(GPU_COUNT[host])
    ]
    d = guard.evaluate(reading(host, gpus=gpus), guard.get_profile(host))
    assert_hold(d, "no GPU satisfies the start condition")


def test_deepthought_one_blocked_gpu_does_not_block_the_host():
    # section 6.2's start condition is per GPU: a >1 GiB foreign process blocks
    # that GPU, but a free second GPU is still a valid start.
    gpus = [gpu(i) for i in range(4)]
    gpus[0] = gpu(0, used=1500.0, processes=[process(3000, memory_mib=1500.0)])
    d = guard.evaluate(reading(DT, gpus=gpus), guard.get_profile(DT))
    assert d.level == "OK", d.reasons
    assert d.start_gpu == 1


def test_deepthought_small_foreign_process_does_not_block_start():
    # "no other user's process holds > 1 GB on it": 512 MiB is allowed.
    gpus = [gpu(i) for i in range(4)]
    gpus[0] = gpu(0, used=512.0, processes=[process(3000, memory_mib=512.0)])
    d = guard.evaluate(reading(DT, gpus=gpus), guard.get_profile(DT))
    assert d.level == "OK", d.reasons
    assert d.start_gpu == 0


def test_deepthought2_foreign_memory_plus_our_cap_must_fit_85_percent():
    total = L40S_TOTAL_MIB
    our_cap = guard.PROFILES[DT2].gpu_memory_fraction_max * total
    allowed_foreign = 0.85 * total - our_cap
    # just below the bound: start allowed
    gpus = [gpu(i, total=total) for i in range(2)]
    gpus[0] = gpu(0, total=total, used=allowed_foreign, processes=[process(3000, memory_mib=allowed_foreign)])
    d = guard.evaluate(reading(DT2, gpus=gpus), guard.get_profile(DT2))
    assert d.level == "OK", d.reasons
    # over the bound on both cards: HOLD
    too_much = allowed_foreign + 1.0
    gpus = [
        gpu(i, total=total, used=too_much, processes=[process(3000 + i, memory_mib=too_much)])
        for i in range(2)
    ]
    d = guard.evaluate(reading(DT2, gpus=gpus), guard.get_profile(DT2))
    assert_hold(d, "other users'")


def test_deepthought2_util_5m_threshold_is_60():
    total = L40S_TOTAL_MIB
    gpus = [gpu(i, total=total, util=59.0) for i in range(2)]
    assert guard.evaluate(reading(DT2, gpus=gpus), guard.get_profile(DT2)).level == "OK"
    gpus = [gpu(i, total=total, util=60.0) for i in range(2)]
    assert_hold(guard.evaluate(reading(DT2, gpus=gpus), guard.get_profile(DT2)), "utilisation")


def test_deepthought_util_5m_threshold_is_20():
    gpus = [gpu(i, util=19.0) for i in range(4)]
    assert guard.evaluate(reading(DT, gpus=gpus), guard.get_profile(DT)).level == "OK"
    gpus = [gpu(i, util=20.0) for i in range(4)]
    assert_hold(guard.evaluate(reading(DT, gpus=gpus), guard.get_profile(DT)), "utilisation")


@pytest.mark.parametrize("host", [DT, DT2])
def test_hold_when_all_gpus_used(host):
    # deepthought: 3 training + 1 evaluation on the 4th GPU (not >3 training)
    kinds = ["training"] * GPU_COUNT[host]
    if host == DT:
        kinds[-1] = "evaluation"
    jobs = [our_job(f"r{i}", 1000 + i, i, job_kind=kinds[i]) for i in range(GPU_COUNT[host])]
    d = guard.evaluate(reading(host, our_processes=jobs), guard.get_profile(host))
    assert_hold(d, "GPU count")


def test_deepthought2_two_gpu_limit():
    jobs = [our_job("r0", 1000, 0), our_job("r1", 1001, 1)]
    d = guard.evaluate(reading(DT2, our_processes=jobs), guard.get_profile(DT2))
    assert_hold(d, "GPU count")


# ---------------------------------------------------------------------------
# SHED: every hard limit, one reading each.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", [DT, DT2])
def test_shed_on_free_ram_below_10_percent(host):
    r = reading(host)
    r["host_memory_available_mib"] = 0.09 * HOST_TOTAL_MIB[host]
    d = guard.evaluate(r, guard.get_profile(host))
    assert_shed(d, "host free RAM")


@pytest.mark.parametrize("host", [DT, DT2])
def test_shed_on_gpu_memory_above_95_percent(host):
    total = TITAN_TOTAL_MIB if host == DT else L40S_TOTAL_MIB
    gpus = [gpu(i, total=total) for i in range(GPU_COUNT[host])]
    gpus[0] = gpu(0, total=total, used=0.96 * total)
    d = guard.evaluate(reading(host, gpus=gpus), guard.get_profile(host))
    assert_shed(d, "total memory")


@pytest.mark.parametrize("host", [DT, DT2])
def test_shed_on_load_above_90_percent_of_cores(host):
    d = evaluate(host, loadavg_15m=0.91 * 40)
    assert_shed(d, "15-minute load")


@pytest.mark.parametrize("host", [DT, DT2])
def test_shed_on_free_entries_below_10000(host):
    d = evaluate(host, free_entries=9_999)
    assert_shed(d, "free entries")


@pytest.mark.parametrize("host", [DT, DT2])
def test_shed_when_our_process_exceeds_the_tolerant_gpu_memory_cap(host):
    total = TITAN_TOTAL_MIB if host == DT else L40S_TOTAL_MIB
    profile = guard.PROFILES[host]
    cap = profile.gpu_memory_fraction_max * total
    tolerance = max(guard.GPU_MEMORY_CAP_TOLERANCE_MIN_MIB, guard.GPU_MEMORY_CAP_TOLERANCE_FRACTION * cap)
    # at the nominal cap (Titan batch-32 ~9.6 GB) and inside the context
    # tolerance the job must NOT be shed
    job = our_job("r1", 1001, 0, gpu_memory_mib=cap + 100.0)
    assert guard.evaluate(reading(host, our_processes=[job]), guard.get_profile(host)).level != "SHED"
    # beyond cap + tolerance it is shed
    job = our_job("r1", 1001, 0, gpu_memory_mib=cap + tolerance + 50.0)
    d = guard.evaluate(reading(host, our_processes=[job]), guard.get_profile(host))
    assert_shed(d, "per-process cap")


@pytest.mark.parametrize("host", [DT, DT2])
def test_shed_when_our_processes_exceed_cpu_cap(host):
    job = our_job("r1", 1001, 0, cpu_percent=1700.0)  # 17 cores > 16
    d = guard.evaluate(reading(host, our_processes=[job]), guard.get_profile(host))
    assert_shed(d, "CPU cores")


@pytest.mark.parametrize("host", [DT, DT2])
def test_shed_when_our_processes_exceed_ram_cap(host):
    ram_gib = guard.PROFILES[host].ram_total_max_gb + 1.0
    job = our_job("r1", 1001, 0, rss_mib=ram_gib * 1024.0)
    d = guard.evaluate(reading(host, our_processes=[job]), guard.get_profile(host))
    assert_shed(d, "RAM")


def test_shed_takes_precedence_over_hold_and_missing_inputs():
    r = reading(DT, free_entries=None)
    r["host_memory_available_mib"] = 0.05 * HOST_TOTAL_MIB[DT]
    d = guard.evaluate(r, guard.get_profile(DT))
    assert d.level == "SHED"
    assert any("host free RAM" in reason for reason in d.reasons)


def test_limits_are_strict_where_the_runbook_says_below_or_above():
    # free RAM exactly 20% is not below the floor; exactly 10% is not below SHED
    r = reading(DT)
    r["host_memory_available_mib"] = 0.20 * HOST_TOTAL_MIB[DT]
    assert guard.evaluate(r, guard.get_profile(DT)).level == "OK"
    r["host_memory_available_mib"] = 0.10 * HOST_TOTAL_MIB[DT]
    d = guard.evaluate(r, guard.get_profile(DT))
    assert d.level == "HOLD"  # below the 20% start floor, but not below the 10% SHED floor


# ---------------------------------------------------------------------------
# The 4-training-job / 4th-GPU rule (deepthought).
# ---------------------------------------------------------------------------


def _three_training_jobs_reading():
    jobs = [
        our_job(f"train{i}", 2000 + i, i, job_kind="training", started_epoch=NOW - 7200 + 600 * i)
        for i in range(3)
    ]
    return reading(DT, our_processes=jobs)


def test_deepthought_refuses_a_fourth_training_job():
    r = _three_training_jobs_reading()
    gpu, cap, blockers = guard.evaluate_start(
        r, guard.get_profile(DT), {"kind": "training", "planned_minutes": 20.0}
    )
    assert gpu is None and cap is None
    assert any("training jobs" in reason for reason in blockers), blockers
    # and the level decision is HOLD for that candidate
    d = guard.evaluate(r, guard.get_profile(DT), {"kind": "training", "planned_minutes": 20.0})
    assert d.level == "HOLD"


def test_deepthought_allows_the_fourth_gpu_only_for_jobs_up_to_30_minutes():
    r = _three_training_jobs_reading()
    profile = guard.get_profile(DT)
    # a 20-minute evaluation job may use the 4th GPU, with an explicit 30-min cap
    gpu, cap, blockers = guard.evaluate_start(
        r, profile, {"kind": "evaluation", "planned_minutes": 20.0}
    )
    assert (gpu, cap, blockers) == (3, 30.0, [])
    d = guard.evaluate(r, profile, {"kind": "evaluation", "planned_minutes": 20.0})
    assert d.level == "OK"
    assert d.start_gpu == 3
    assert d.start_max_planned_minutes == 30.0
    # a 45-minute job may not, even if it is not training
    gpu, cap, blockers = guard.evaluate_start(
        r, profile, {"kind": "evaluation", "planned_minutes": 45.0}
    )
    assert gpu is None and cap is None
    assert any("4th GPU" in reason for reason in blockers), blockers
    # an unknown duration may not either
    gpu, cap, blockers = guard.evaluate_start(r, profile, {"kind": "evaluation"})
    assert gpu is None
    # but with only two training jobs a long third job is fine on the 3rd GPU
    two = reading(DT, our_processes=[our_job("train0", 2000, 0), our_job("train1", 2001, 1)])
    gpu, cap, blockers = guard.evaluate_start(
        two, profile, {"kind": "training", "planned_minutes": 300.0}
    )
    assert (gpu, cap, blockers) == (2, None, [])


# ---------------------------------------------------------------------------
# SHED target selection (section 6.2: PID file + dwong + jepa_ tag).
# ---------------------------------------------------------------------------


def test_select_shed_target_our_pidfile_dwong_tagged():
    job = our_job("r1", 1001, 0)
    target = guard.select_shed_target(reading(DT, our_processes=[job]))
    assert target is not None
    assert target["pid"] == 1001
    assert target["run_id"] == "r1"


def test_select_shed_target_rejects_wrong_owner():
    job = our_job("r1", 1001, 0, owner="otheruser")
    assert guard.select_shed_target(reading(DT, our_processes=[job])) is None


def test_select_shed_target_rejects_not_in_pid_file():
    job = our_job("r1", 1001, 0, in_pid_file=False)
    assert guard.select_shed_target(reading(DT, our_processes=[job])) is None


def test_select_shed_target_rejects_missing_jepa_tag():
    job = our_job("r1", 1001, 0, cmdline="python -m jepa_noise.train --config x")
    assert guard.select_shed_target(reading(DT, our_processes=[job])) is None


def test_select_shed_target_rejects_unreadable_start_time():
    job = our_job("r1", 1001, 0)
    job["started_at_utc"] = "not a timestamp"
    assert guard.select_shed_target(reading(DT, our_processes=[job])) is None


def test_select_shed_target_picks_most_recently_started():
    older = our_job("old", 1001, 0, started_epoch=NOW - 7200)
    middle = our_job("mid", 1002, 1, started_epoch=NOW - 3600)
    newest = our_job("new", 1003, 2, started_epoch=NOW - 600)
    target = guard.select_shed_target(reading(DT, our_processes=[middle, newest, older]))
    assert target is not None and target["run_id"] == "new"
    assert target["pid"] == 1003


# ---------------------------------------------------------------------------
# SHED action: dry run vs live, 5-minute retry.
# ---------------------------------------------------------------------------


def test_pid_listed_in_pid_files_requires_pid_and_tag(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "pid.json").write_text(json.dumps({"pid": 4242, "tag": "jepa_run1", "run_id": "run1"}))
    assert guard.pid_listed_in_pid_files(tmp_path, 4242) is True
    assert guard.pid_listed_in_pid_files(tmp_path, 4242, "jepa_run1") is True
    assert guard.pid_listed_in_pid_files(tmp_path, 4242, "jepa_other") is False
    assert guard.pid_listed_in_pid_files(tmp_path, 9999) is False
    assert guard.pid_listed_in_pid_files(tmp_path / "missing", 4242) is False


def test_dry_run_reports_would_signal_and_never_calls_kill(monkeypatch):
    def _must_not_kill(*args, **kwargs):
        raise AssertionError("os.kill must not be called in dry-run")

    monkeypatch.setattr(guard.os, "kill", _must_not_kill)
    target = our_job("r1", 1001, 0)
    event = guard.shed_signal(target, dry_run=True, now_epoch=NOW)
    assert event["action"] == "would_signal"
    assert event["pid"] == 1001
    assert event["tag"] == "jepa_r1"
    assert event["signal"] == "SIGTERM"
    assert event["dry_run"] is True


def test_live_signal_refuses_when_verification_fails(monkeypatch):
    def _must_not_kill(*args, **kwargs):
        raise AssertionError("os.kill must not be called when verification fails")

    monkeypatch.setattr(guard.os, "kill", _must_not_kill)
    target = our_job("r1", 1001, 0)
    event = guard.shed_signal(
        target, dry_run=False, verifier=lambda target: (False, "owner changed"), now_epoch=NOW
    )
    assert event["action"] == "refused"
    assert "owner changed" in event["detail"]


def test_live_signal_sends_sigterm_after_verification(monkeypatch):
    calls = []

    def _fake_kill(pid, sig):
        calls.append((pid, sig))

    monkeypatch.setattr(guard.os, "kill", _fake_kill)
    target = our_job("r1", 1001, 0)
    event = guard.shed_signal(
        target, dry_run=False, verifier=lambda target: (True, ""), now_epoch=NOW
    )
    assert event["action"] == "signalled"
    assert calls == [(1001, guard.signal.SIGTERM)]


def test_shed_retry_is_five_minutes():
    assert guard.shed_action_due(NOW, None) is True
    assert guard.shed_action_due(NOW, NOW - 100.0) is False
    assert guard.shed_action_due(NOW, NOW - 299.0) is False
    assert guard.shed_action_due(NOW, NOW - 300.0) is True


# ---------------------------------------------------------------------------
# Status file / launcher gate / atomic I/O.
# ---------------------------------------------------------------------------


def _ok_status(host: str = DT, *, age: float = 10.0, level: str = "OK", start=None) -> dict:
    status = {
        "timestamp": iso(NOW - age),
        "timestamp_epoch": NOW - age,
        "host": host,
        "level": level,
        "reasons": [] if level == "OK" else ["synthetic reason"],
        "gpus": [],
        "our_jobs": [],
        "caps": {
            "per_process_gpu_memory_fraction": guard.PROFILES[host].gpu_memory_fraction_max,
            "training_jobs_max": guard.PROFILES[host].training_jobs_max,
        },
        "start": start if start is not None else {"gpu": 3, "max_planned_minutes": None},
        "candidate": {"kind": "other", "planned_minutes": 30.0},
        "guard": {"version": guard.GUARD_VERSION, "dry_run": False, "reading_host": host},
    }
    return status


def test_status_freshness_accepts_recent_status():
    ok, why = guard.status_is_usable(_ok_status(age=10.0), now_epoch=NOW)
    assert ok, why


def test_status_freshness_rejects_stale_status():
    ok, why = guard.status_is_usable(_ok_status(age=121.0), now_epoch=NOW)
    assert not ok and "stale" in why


def test_status_freshness_rejects_missing_timestamp():
    status = _ok_status()
    status.pop("timestamp")
    status.pop("timestamp_epoch")
    ok, why = guard.status_is_usable(status, now_epoch=NOW)
    assert not ok and "timestamp" in why


def test_status_freshness_rejects_future_timestamp():
    ok, why = guard.status_is_usable(_ok_status(age=-1000.0), now_epoch=NOW)
    assert not ok and "future" in why


def test_status_freshness_rejects_hold():
    ok, why = guard.status_is_usable(_ok_status(level="HOLD"), now_epoch=NOW)
    assert not ok and "HOLD" in why


def test_launcher_params_from_ok_status():
    params, why = guard.run_params_from_status(
        _ok_status(), planned_minutes=20.0, now_epoch=NOW, expected_host=DT
    )
    assert why is None and params is not None
    assert params["gpu_index"] == 3
    assert params["per_process_memory_fraction"] == pytest.approx(0.80)
    assert params["timeout_seconds"] == 1800  # 1.5 x 20 min
    assert params["planned_minutes"] == 20.0


def test_launcher_uses_config_default_duration():
    params, why = guard.run_params_from_status(_ok_status(), now_epoch=NOW)
    assert why is None and params is not None
    assert params["planned_minutes"] == pytest.approx(60.0)
    assert params["timeout_seconds"] == 5400  # 1.5 x 60 min


def test_launcher_refuses_stale_status():
    params, why = guard.run_params_from_status(
        _ok_status(age=300.0), planned_minutes=20.0, now_epoch=NOW
    )
    assert params is None and "stale" in why


def test_launcher_refuses_hold_status():
    params, why = guard.run_params_from_status(
        _ok_status(level="HOLD"), planned_minutes=20.0, now_epoch=NOW
    )
    assert params is None and "HOLD" in why


def test_launcher_refuses_when_guard_caps_the_duration():
    status = _ok_status(start={"gpu": 3, "max_planned_minutes": 30.0})
    params, why = guard.run_params_from_status(status, planned_minutes=60.0, now_epoch=NOW)
    assert params is None and "limit" in why
    params, why = guard.run_params_from_status(status, planned_minutes=20.0, now_epoch=NOW)
    assert why is None and params["gpu_index"] == 3


def test_launcher_refuses_wrong_host():
    params, why = guard.run_params_from_status(
        _ok_status(host=DT2), planned_minutes=20.0, now_epoch=NOW, expected_host=DT
    )
    assert params is None and "host" in why


def test_launcher_refuses_status_without_start_gpu():
    status = _ok_status()
    status["start"] = None
    params, why = guard.run_params_from_status(status, planned_minutes=20.0, now_epoch=NOW)
    assert params is None and "start GPU" in why


def test_timeout_is_1_5_times_planned_minutes():
    assert guard.timeout_seconds(20.0) == 1800
    assert guard.timeout_seconds(1.0) == 90
    assert guard.timeout_seconds(60.0) == 5400
    assert guard.timeout_seconds(0.5) == 45


def test_run_config_defaults_and_override(tmp_path):
    config = guard.load_run_config()
    assert config["default_planned_minutes"] == pytest.approx(60.0)
    assert config["timeout_factor"] == pytest.approx(1.5)
    custom = tmp_path / "guard_config.json"
    custom.write_text(json.dumps({"run": {"default_planned_minutes": 12, "timeout_factor": 2.0}}))
    config = guard.load_run_config(custom)
    assert config["default_planned_minutes"] == pytest.approx(12.0)
    assert config["timeout_factor"] == pytest.approx(2.0)


def test_build_status_has_required_fields_and_caps():
    host = DT
    r = reading(host)
    decision = guard.evaluate(r, guard.get_profile(host))
    status = guard.build_status(r, guard.get_profile(host), decision, now_epoch=NOW)
    assert status["host"] == host
    assert status["level"] == "OK"
    assert status["timestamp"] == iso(NOW)
    assert isinstance(status["timestamp_epoch"], float)
    assert isinstance(status["reasons"], list)
    assert isinstance(status["gpus"], list) and len(status["gpus"]) == 4
    assert status["our_jobs"] == []
    assert status["caps"]["gpu_count_max"] == 4
    assert status["caps"]["training_jobs_max"] == 3
    assert status["caps"]["cpu_cores_max"] == 16
    assert status["caps"]["ram_total_max_gb"] == pytest.approx(128.0)
    assert status["caps"]["free_entries_start_min"] == 30_000
    assert status["caps"]["per_process_gpu_memory_fraction"] == pytest.approx(0.80)
    assert status["caps"]["per_process_gpu_memory_cap_mib"]["0"] == pytest.approx(0.80 * TITAN_TOTAL_MIB)
    assert status["start"]["gpu"] == decision.start_gpu
    assert status["guard"]["version"] == guard.GUARD_VERSION


def test_build_status_has_no_start_when_hold():
    r = reading(DT, loadavg_15m=30.0)
    decision = guard.evaluate(r, guard.get_profile(DT))
    status = guard.build_status(r, guard.get_profile(DT), decision, now_epoch=NOW)
    assert status["level"] == "HOLD"
    assert status["start"] is None


def test_status_file_written_atomically_is_parseable(tmp_path):
    host = DT
    r = reading(host)
    decision = guard.evaluate(r, guard.get_profile(host))
    status = guard.build_status(r, guard.get_profile(host), decision, now_epoch=NOW)
    path = tmp_path / f"guard_{host}.json"
    guard.write_status_atomic(path, status)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["level"] == "OK"
    assert parsed["host"] == host
    assert parsed["caps"]["per_process_gpu_memory_fraction"] == pytest.approx(0.80)
    # no temporary files may be left behind
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != path.name]
    assert leftovers == []
    # overwrite atomically with a new level
    status["level"] = "HOLD"
    status["reasons"] = ["second write"]
    status["start"] = None
    guard.write_status_atomic(path, status)
    assert json.loads(path.read_text(encoding="utf-8"))["level"] == "HOLD"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_last_signal_epoch_recovers_only_live_signals(tmp_path):
    log_path = tmp_path / "guard_deepthought.log"
    guard.append_log_line(log_path, {"event": "evaluate", "timestamp_epoch": NOW - 100.0})
    guard.append_log_line(
        log_path, {"event": "shed_signal", "action": "would_signal", "timestamp_epoch": NOW - 90.0}
    )
    assert guard.read_last_signal_epoch(log_path) is None  # dry-run is not a signal
    guard.append_log_line(
        log_path, {"event": "shed_signal", "action": "signalled", "timestamp_epoch": NOW - 10.0}
    )
    assert guard.read_last_signal_epoch(log_path) == pytest.approx(NOW - 10.0)
    assert guard.read_last_signal_epoch(tmp_path / "missing.log") is None


def test_log_line_is_valid_json_and_appends(tmp_path):
    log_path = tmp_path / "guard_deepthought.log"
    event = {"event": "evaluate", "level": "OK", "timestamp_epoch": NOW}
    line = guard.append_log_line(log_path, event)
    assert json.loads(line) == event
    guard.append_log_line(log_path, {"event": "shed_signal", "action": "would_signal", "pid": 7})
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "evaluate"
    assert json.loads(lines[1])["action"] == "would_signal"


def test_guard_module_has_no_heavy_imports():
    # The guard runs in a tmux loop on hosts; it must stay stdlib-only.
    source = (ROOT / "scripts" / "jepa_guard.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "import numpy", "import pandas", "import cupy"):
        assert forbidden not in source


def test_parse_utc_round_trip():
    assert guard.parse_utc(iso(NOW)) == pytest.approx(NOW, abs=1.0)
    assert guard.parse_utc("2026-10-17T10:00:00Z") is not None
    assert guard.parse_utc("garbage") is None
    assert guard.parse_utc(None) is None


# ===========================================================================
# F4: CephFS directory-quota entry source (confirmed by the lead).
# ===========================================================================


def test_parse_ceph_quota_confirmed_numbers():
    info = guard.parse_ceph_quota("500000\n", "416387\n", "51669\n")
    assert info is not None
    assert info["source"] == "cephfs_xattr"
    assert info["quota_max_files"] == 500_000
    assert info["quota_rfiles"] == 416_387
    assert info["quota_rsubdirs"] == 51_669
    assert info["free_entries"] == 31_944
    assert info["free_entries_clamped"] is False


def test_parse_ceph_quota_rejects_garbage_and_clamps_negative():
    assert guard.parse_ceph_quota(None, "1", "1") is None
    assert guard.parse_ceph_quota("abc", "1", "1") is None
    info = guard.parse_ceph_quota("10", "9", "9")
    assert info["free_entries"] == 0 and info["free_entries_clamped"] is True


def test_parse_df_i_free_uses_posix_columns():
    # GNU/POSIX df -i -P: Filesystem Inodes IUsed IFree IUse% Mounted on
    output = (
        "Filesystem Inodes IUsed IFree IUse% Mounted on\n"
        "/dev/sda1 100000 50000 50000 50% /ceph\n"
    )
    assert guard.parse_df_i_free(output) == 50_000
    assert guard.parse_df_i_free("") is None
    assert guard.parse_df_i_free(None) is None
    assert guard.parse_df_i_free("Filesystem Inodes IUsed IFree IUse% Mounted\nfs 1 2 - 3\n") is None


def test_read_free_entries_uses_ceph_xattrs_first(monkeypatch):
    values = {
        guard.CEPH_QUOTA_MAX_FILES: "500000\n",
        guard.CEPH_QUOTA_RFILES: "416387\n",
        guard.CEPH_QUOTA_RSUBDIRS: "51669\n",
    }

    def fake_run(command, timeout=30.0):
        assert command[0] == "getfattr"
        return values[command[2]]

    monkeypatch.setattr(guard, "_run_command", fake_run)
    info, error = guard._read_free_entries(Path("/ceph/dwong"))
    assert error is None
    assert info["source"] == "cephfs_xattr"
    assert info["free_entries"] == 31_944
    assert info["quota_max_files"] == 500_000


def test_read_free_entries_falls_back_with_logged_reason(monkeypatch):
    def fake_run(command, timeout=30.0):
        if command[0] == "getfattr":
            return None
        if command[0] == "df":
            return "Filesystem Inodes IUsed IFree IUse% Mounted on\nfs 9 8 7 1% /tmp\n"
        return None

    monkeypatch.setattr(guard, "_run_command", fake_run)
    monkeypatch.setattr(guard.os, "statvfs", lambda path: types.SimpleNamespace(f_files=0, f_ffree=0))
    info, error = guard._read_free_entries(Path("/tmp"))
    assert error is None
    assert info["source"] == "df_i"
    assert info["free_entries"] == 7
    assert "fallback" in info["fallback_reason"]


def test_read_free_entries_reports_failure_when_all_sources_fail(monkeypatch):
    monkeypatch.setattr(guard, "_run_command", lambda command, timeout=30.0: None)

    def broken_statvfs(path):
        raise OSError("no such path")

    monkeypatch.setattr(guard.os, "statvfs", broken_statvfs)
    info, error = guard._read_free_entries(Path("/nope"))
    assert info is None and error and "cannot read free entry count" in error


def test_status_records_raw_quota_numbers_each_cycle():
    r = reading(DT)
    decision = guard.evaluate(r, guard.get_profile(DT))
    status = guard.build_status(r, guard.get_profile(DT), decision, now_epoch=NOW)
    assert status["storage"]["free_entries"] == 31_944
    assert status["storage"]["entry_source"]["quota_max_files"] == 500_000
    assert status["storage"]["entry_source"]["quota_rfiles"] == 416_387
    assert status["storage"]["entry_source"]["quota_rsubdirs"] == 51_669
    assert status["caps"]["entry_source"]["source"] == "cephfs_xattr"


# ===========================================================================
# F5: unattributed GPU memory and unknown per-process memory.
# ===========================================================================


def test_unattributed_gpu_memory_above_1gib_holds():
    r = reading(DT)
    r["gpus"][0] = gpu(0, used=1500.0)  # no listed processes
    d = guard.evaluate(r, guard.get_profile(DT))
    assert_hold(d, "not attributed to any process")


def test_unattributed_gpu_memory_below_1gib_does_not_hold():
    r = reading(DT)
    r["gpus"][0] = gpu(0, used=800.0)  # under 1 GiB, treated as foreign memory only
    d = guard.evaluate(r, guard.get_profile(DT))
    assert d.level == "OK", d.reasons
    assert d.start_gpu == 0


def test_unknown_per_process_gpu_memory_holds():
    r = reading(DT)
    r["gpus"][0] = gpu(0, used=1000.0, processes=[process(2002, memory_mib=None)])
    d = guard.evaluate(r, guard.get_profile(DT))
    assert_hold(d, "memory is unknown")


def test_unknown_gpu_memory_used_holds():
    r = reading(DT)
    gpus = [gpu(i) for i in range(4)]
    gpus[0] = gpu(0)
    gpus[0]["memory_used_mib"] = None
    r["gpus"] = gpus
    d = guard.evaluate(r, guard.get_profile(DT))
    assert_hold(d, "memory used is unknown")


# ===========================================================================
# F3: duplicate GPU assignment (HOLD) and atomic claim.
# ===========================================================================


def test_duplicate_gpu_assignment_holds():
    jobs = [our_job("r1", 1001, 0), our_job("r2", 1002, 0)]
    d = guard.evaluate(reading(DT, our_processes=jobs), guard.get_profile(DT))
    assert_hold(d, "duplicate GPU assignment")


def test_claim_is_exclusive_and_stale_claims_are_reclaimed(tmp_path):
    jobs_dir = tmp_path / "jobs"
    path, why = launcher.acquire_claim(jobs_dir, DT, 0, "a", launcher_pid=os.getpid())
    assert path is not None and why is None and path.exists()
    path_b, why_b = launcher.acquire_claim(jobs_dir, DT, 0, "b", launcher_pid=os.getpid())
    assert path_b is None and "already claimed" in why_b
    # independent GPU claim works
    path_gpu1, why_gpu1 = launcher.acquire_claim(jobs_dir, DT, 1, "c", launcher_pid=os.getpid())
    assert path_gpu1 is not None and why_gpu1 is None
    # dead PIDs but a fresh claim are held by the grace period
    payload = json.loads(path.read_text())
    payload["launcher_pid"] = 2_000_000
    payload["job_pid"] = 2_000_001
    path.write_text(json.dumps(payload))
    path_fresh, why_fresh = launcher.acquire_claim(jobs_dir, DT, 0, "c", launcher_pid=os.getpid())
    assert path_fresh is None and "already claimed" in why_fresh
    # after the grace period with both PIDs dead it is reclaimed
    old = time.time() - launcher.CLAIM_GRACE_SECONDS - 10.0
    os.utime(path, (old, old))
    path_c, why_c = launcher.acquire_claim(jobs_dir, DT, 0, "c", launcher_pid=os.getpid())
    assert path_c is not None and why_c is None
    assert launcher.release_claim(path_c) is True
    assert launcher.release_claim(path_gpu1) is True


def test_claim_is_stale_logic():
    dead = {"launcher_pid": 2_000_000, "job_pid": 2_000_001}
    live = {"launcher_pid": os.getpid(), "job_pid": None}
    # a young claim is never stale, not even an unreadable one
    assert launcher.claim_is_stale(None, age_seconds=0.0) is False
    assert launcher.claim_is_stale(dead, age_seconds=0.0) is False
    # a live claim is never stale
    assert launcher.claim_is_stale(live, age_seconds=3600.0) is False
    # after the grace period: unreadable => stale; both PIDs dead => stale
    assert launcher.claim_is_stale(None, age_seconds=launcher.CLAIM_GRACE_SECONDS + 1.0) is True
    assert launcher.claim_is_stale(dead, age_seconds=launcher.CLAIM_GRACE_SECONDS + 1.0) is True
    # callers that do not pass an age are conservative: a claim is never
    # stolen on missing age information
    assert launcher.claim_is_stale(None) is False
    assert launcher.claim_is_stale(live) is False
    assert launcher.claim_is_stale(dead) is False


def test_claim_release_does_not_remove_another_launchers_claim(tmp_path):
    jobs_dir = tmp_path / "jobs"
    path, _ = launcher.acquire_claim(jobs_dir, DT, 0, "a", launcher_pid=os.getpid())
    assert path is not None
    assert launcher.release_claim(path, launcher_pid=12345) is False
    assert path.exists()
    assert launcher.release_claim(path, launcher_pid=os.getpid()) is True


# ===========================================================================
# F6: CPU/RSS over the job process tree.
# ===========================================================================


def test_process_tree_and_group_pids():
    table = guard.build_process_table(
        [
            {"pid": 100, "ppid": 1, "pgrp": 100},
            {"pid": 101, "ppid": 100, "pgrp": 100},
            {"pid": 102, "ppid": 101, "pgrp": 100},
            {"pid": 103, "ppid": 1, "pgrp": 100},  # same group, not a descendant
            {"pid": 200, "ppid": 1, "pgrp": 200},
        ]
    )
    assert set(guard.process_tree_pids(100, table)) == {100, 101, 102}
    assert set(guard.process_group_pids(100, table)) == {100, 101, 102, 103}
    assert 200 not in guard.process_tree_pids(100, table)
    assert guard.process_group_pids(None, table) == []
    assert guard.process_tree_pids(None, table) == []


# ===========================================================================
# F7: observed GPU index wins; declared/observed mismatch HOLDs.
# ===========================================================================


def test_gpu_index_mismatch_holds():
    job = our_job("r1", 1001, 0)
    job["gpu_index_declared"] = 1
    d = guard.evaluate(reading(DT, our_processes=[job]), guard.get_profile(DT))
    assert_hold(d, "GPU index mismatch")


def test_matching_observed_and_declared_gpu_index_is_fine():
    job = our_job("r1", 1001, 0)
    job["gpu_index_declared"] = 0
    d = guard.evaluate(reading(DT, our_processes=[job]), guard.get_profile(DT))
    assert d.level == "OK", d.reasons


# ===========================================================================
# F8: CUDA-context tolerance in the per-process cap SHED.
# ===========================================================================


def test_shed_cap_includes_documented_cuda_context_tolerance():
    profile = guard.get_profile(DT)
    gpus = guard.assess_gpus(reading(DT), profile)
    caps = guard.evaluate(reading(DT), profile).caps
    nominal = caps["per_process_gpu_memory_cap_mib"]["0"]
    shed_cap = caps["per_process_gpu_memory_shed_cap_mib"]["0"]
    assert nominal == pytest.approx(0.80 * TITAN_TOTAL_MIB)
    assert shed_cap == pytest.approx(
        nominal + max(guard.GPU_MEMORY_CAP_TOLERANCE_MIN_MIB, guard.GPU_MEMORY_CAP_TOLERANCE_FRACTION * nominal)
    )
    assert caps["gpu_memory_shed_tolerance_fraction"] == guard.GPU_MEMORY_CAP_TOLERANCE_FRACTION
    assert gpus  # sanity: assessment available


# ===========================================================================
# F9: the per-process GPU table is required to be present.
# ===========================================================================


def test_per_process_gpu_table_false_is_not_ok():
    d = evaluate(DT, per_process_gpu_table_available=False)
    assert d.level != "OK"
    assert any("per_process_gpu_table_available" in reason for reason in d.reasons)


def test_per_process_gpu_table_missing_is_not_ok():
    r = reading(DT)
    r.pop("per_process_gpu_table_available")
    d = guard.evaluate(r, guard.get_profile(DT))
    assert d.level != "OK"
    assert any("per_process_gpu_table_available" in reason for reason in d.reasons)


# ===========================================================================
# F10: PID files filtered by host; stale PID files pruned.
# ===========================================================================


def _write_pidfile(jobs_dir: Path, run_id: str, payload: dict) -> Path:
    run_dir = jobs_dir / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / "pid.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_pidfile_records_filters_host_and_prunes_stale(tmp_path):
    jobs_dir = tmp_path / "jobs"
    good = {"pid": 111111, "host": DT, "run_id": "good", "tag": "jepa_good"}
    other_host = {"pid": 222222, "host": DT2, "run_id": "other", "tag": "jepa_other"}
    stale = {"pid": 333333, "host": DT, "run_id": "stale", "tag": "jepa_stale"}
    good_path = _write_pidfile(jobs_dir, "good", good)
    other_path = _write_pidfile(jobs_dir, "other", other_host)
    stale_path = _write_pidfile(jobs_dir, "stale", stale)
    records, pruned, errors = guard.load_pidfile_records(
        jobs_dir, DT, liveness=lambda pid: pid == 111111
    )
    assert [record["pid"] for record in records] == [111111]
    assert records[0]["pid_file"] == str(good_path)
    assert str(stale_path) in pruned and not stale_path.exists()
    assert other_path.exists()  # other host's file is ignored, not pruned
    assert errors == []


def test_pidfile_tag_matches_cmdline_guards_recycled_pids():
    assert guard.pidfile_tag_matches_cmdline(
        "jepa_r1", "python -c '...' jepa_noise.train jepa_r1 --config x"
    ) is True
    assert guard.pidfile_tag_matches_cmdline("jepa_r1", "python -c 'other job'") is False
    assert guard.pidfile_tag_matches_cmdline("jepa_r1", None) is False
    assert guard.pidfile_tag_matches_cmdline(None, "jepa_r1") is False


def test_pidfile_belongs_to_host():
    assert guard.pidfile_belongs_to_host({"host": DT}, DT) is True
    assert guard.pidfile_belongs_to_host({"host": DT2}, DT) is False
    assert guard.pidfile_belongs_to_host({}, DT) is False
    assert guard.pidfile_belongs_to_host(None, DT) is False


# ===========================================================================
# F11: >= staleness bound, real-status requirement.
# ===========================================================================


def test_status_exactly_two_minutes_old_is_stale():
    ok, why = guard.status_is_usable(_ok_status(age=120.0), now_epoch=NOW, max_age_seconds=120.0)
    assert not ok and "stale" in why
    ok, why = guard.status_is_usable(_ok_status(age=119.0), now_epoch=NOW, max_age_seconds=120.0)
    assert ok, why


def test_launcher_refuses_dry_run_status():
    status = _ok_status()
    status["guard"]["dry_run"] = True
    params, why = guard.run_params_from_status(status, planned_minutes=20.0, now_epoch=NOW)
    assert params is None and "dry-run" in why


def test_launcher_refuses_status_without_guard_block():
    status = _ok_status()
    status.pop("guard")
    params, why = guard.run_params_from_status(status, planned_minutes=20.0, now_epoch=NOW)
    assert params is None and "dry_run flag" in why


# ===========================================================================
# F2: training-job cap checked for the actual job kind.
# ===========================================================================


def _status_with_three_trainings() -> dict:
    status = _ok_status(start={"gpu": 3, "max_planned_minutes": 30.0})
    status["caps"]["training_jobs_max"] = 3
    status["our_jobs"] = [
        {
            "run_id": f"train{i}",
            "pid": 2000 + i,
            "gpu_index": i,
            "job_kind": "training",
            "in_pid_file": True,
        }
        for i in range(3)
    ]
    return status


def test_launcher_refuses_a_training_job_at_the_training_cap():
    status = _status_with_three_trainings()
    params, why = guard.run_params_from_status(
        status, planned_minutes=25.0, now_epoch=NOW, job_kind="training"
    )
    assert params is None and "training jobs" in why
    # 3 trainings + a 25-min evaluation on the 4th GPU is allowed
    params, why = guard.run_params_from_status(
        status, planned_minutes=25.0, now_epoch=NOW, job_kind="evaluation"
    )
    assert why is None and params is not None and params["gpu_index"] == 3
    # a non-training job still cannot exceed the guard's per-start duration cap
    status_long = _ok_status(start={"gpu": 3, "max_planned_minutes": 30.0})
    status_long["caps"]["training_jobs_max"] = 3
    status_long["our_jobs"] = status["our_jobs"]
    params, why = guard.run_params_from_status(
        status_long, planned_minutes=45.0, now_epoch=NOW, job_kind="evaluation"
    )
    assert params is None and "limit" in why


# ===========================================================================
# F12: config keys are consumed; 2 TB scope is the JEPA data directory.
# ===========================================================================


def test_guard_config_keys_are_all_consumed():
    payload = json.loads((ROOT / "src" / "jepa_noise" / "guard_config.json").read_text())
    assert set(payload["run"]) == set(guard.DEFAULT_RUN_CONFIG)


def test_two_tb_limit_scope_is_the_jepa_data_directory():
    r = reading(DT)
    decision = guard.evaluate(r, guard.get_profile(DT))
    status = guard.build_status(r, guard.get_profile(DT), decision, now_epoch=NOW)
    assert status["caps"]["jepa_data_max_tb"] == 2.0
    assert status["caps"]["jepa_data_path"] == "/ceph/dwong/jepa/data"
    assert "decimal TB" in status["caps"]["jepa_data_scope"]
    assert status["storage"]["jepa_data_path"] == "/ceph/dwong/jepa/data"


# ===========================================================================
# F1: jepa_launch.py PID contract (integration-style, fake job, no GPU/torch).
# ===========================================================================


def _write_fake_torch(package_root: Path) -> None:
    torch_pkg = package_root / "torch"
    torch_pkg.mkdir(parents=True)
    (torch_pkg / "__init__.py").write_text("from . import cuda\n")
    (torch_pkg / "cuda.py").write_text(
        "import os\n\n"
        "def set_per_process_memory_fraction(fraction):\n"
        "    with open(os.environ['FAKE_TORCH_MARK'], 'w', encoding='utf-8') as fh:\n"
        "        fh.write(repr(float(fraction)))\n"
    )


def _write_fake_job(package_root: Path) -> None:
    (package_root / "fake_job.py").write_text(
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "with open(os.environ['FAKE_JOB_PID_FILE'], 'w', encoding='utf-8') as fh:\n"
        "    fh.write(str(os.getpid()))\n"
        "with open(os.environ['FAKE_JOB_CHILD_FILE'], 'w', encoding='utf-8') as fh:\n"
        "    fh.write(str(child.pid))\n"
        "with open(os.environ['FAKE_JOB_ARGV_FILE'], 'w', encoding='utf-8') as fh:\n"
        "    fh.write(repr(sys.argv))\n"
        "time.sleep(120)\n"
    )


def _fake_launch_setup(tmp_path: Path) -> tuple[dict, Path]:
    package_root = tmp_path / "fake"
    package_root.mkdir()
    fake_torch_root = package_root / "torch_pkg"
    fake_torch_root.mkdir()
    _write_fake_torch(fake_torch_root)
    _write_fake_job(package_root)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake_torch_root), str(package_root), str(ROOT / "src")]
    )
    env["FAKE_TORCH_MARK"] = str(tmp_path / "torch_mark")
    env["FAKE_JOB_PID_FILE"] = str(tmp_path / "job_pid")
    env["FAKE_JOB_CHILD_FILE"] = str(tmp_path / "job_child")
    env["FAKE_JOB_ARGV_FILE"] = str(tmp_path / "job_argv")
    return env, package_root


def _launcher_command(
    tmp_path: Path,
    package_root: Path,
    *,
    run_id: str,
    gpu: int,
    timeout_seconds: int,
    job_args: list[str] | None = None,
) -> list[str]:
    run_dir = tmp_path / run_id
    return [
        sys.executable,
        str(LAUNCHER_SCRIPT),
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--tag",
        f"jepa_{run_id}",
        "--host",
        DT,
        "--gpu-index",
        str(gpu),
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--job-kind",
        "evaluation",
        "--planned-minutes",
        "1",
        "--timeout-seconds",
        str(timeout_seconds),
        "--kill-after-seconds",
        "3",
        "--memory-fraction",
        "0.45",
        "--module",
        "fake_job",
        "--python",
        sys.executable,
        "--workdir",
        str(package_root),
        "--log-file",
        str(run_dir / "run.log"),
        "--",
        *(job_args or []),
    ]


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _pid_gone(pid: int, timeout: float = 10.0) -> bool:
    def gone():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return True
        return False

    return _wait_until(gone, timeout)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_launcher_records_the_real_job_pid_and_it_is_never_foreign(tmp_path):
    env, package_root = _fake_launch_setup(tmp_path)
    command = _launcher_command(
        tmp_path,
        package_root,
        run_id="int1",
        gpu=0,
        timeout_seconds=60,
        job_args=["--config", "sample.yaml"],
    )
    launcher_process = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        run_dir = tmp_path / "int1"
        assert _wait_until(lambda: (run_dir / "pid.json").exists()), "pid.json not written"
        assert _wait_until(lambda: (tmp_path / "job_pid").exists()), "fake job did not start"
        assert _wait_until(lambda: (tmp_path / "torch_mark").exists()), "bootstrap did not run"
        payload = json.loads((run_dir / "pid.json").read_text())
        job_pid = int((tmp_path / "job_pid").read_text())
        assert payload["pid"] == job_pid, "recorded PID must be the real job process"
        assert payload["pid"] != launcher_process.pid, "recorded PID must not be the launcher"
        assert payload["pgid"] == job_pid
        assert payload["tag"] == "jepa_int1"
        assert float((tmp_path / "torch_mark").read_text()) == pytest.approx(0.45)
        assert "--config" in (tmp_path / "job_argv").read_text()
        # The GPU holder (that PID) must be classified as ours, never foreign.
        r = reading(DT)
        r["our_processes"] = [
            our_job("int1", job_pid, 0, job_kind="evaluation", cpu_percent=50.0, rss_mib=100.0)
        ]
        r["gpus"][0] = gpu(
            0,
            used=9000.0,
            processes=[process(job_pid, owner="dwong", memory_mib=9000.0, in_pid_file=True)],
        )
        d = guard.evaluate(r, guard.get_profile(DT))
        assert d.level == "OK", d.reasons
        assert guard.assess_gpus(r, guard.get_profile(DT))[0]["foreign_memory_mib"] == 0
        child_pid = int((tmp_path / "job_child").read_text())
        _stop_process(launcher_process)
        assert _pid_gone(job_pid)
        assert _pid_gone(child_pid)
        assert (run_dir / "run_status.json").exists()
        assert not (run_dir / "pid.json").exists()
        status = json.loads((run_dir / "run_status.json").read_text())
        assert status["outcome"] in {"signalled", "exited"}
    finally:
        _stop_process(launcher_process)


def test_launcher_enforces_the_timeout_and_writes_status(tmp_path):
    env, package_root = _fake_launch_setup(tmp_path)
    command = _launcher_command(tmp_path, package_root, run_id="t1", gpu=0, timeout_seconds=2)
    launcher_process = subprocess.Popen(
        command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        returncode = launcher_process.wait(timeout=30)
    finally:
        _stop_process(launcher_process)
    assert returncode == 124
    run_dir = tmp_path / "t1"
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["timed_out"] is True
    assert status["outcome"] == "timeout"
    assert not (run_dir / "pid.json").exists()


def test_launcher_claim_blocks_a_second_start_on_the_same_gpu(tmp_path):
    env, package_root = _fake_launch_setup(tmp_path)
    first_command = _launcher_command(
        tmp_path, package_root, run_id="first", gpu=0, timeout_seconds=60
    )
    first = subprocess.Popen(
        first_command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    try:
        assert _wait_until(lambda: (tmp_path / "first" / "pid.json").exists())
        second_command = _launcher_command(
            tmp_path, package_root, run_id="second", gpu=0, timeout_seconds=60
        )
        second = subprocess.run(
            second_command, env=env, capture_output=True, text=True, timeout=20
        )
        assert second.returncode == 3
        assert "already claimed" in second.stderr
        # a different GPU is not blocked by the first claim
        third_command = _launcher_command(
            tmp_path, package_root, run_id="third", gpu=1, timeout_seconds=60
        )
        third = subprocess.Popen(
            third_command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        try:
            assert _wait_until(lambda: (tmp_path / "third" / "pid.json").exists())
        finally:
            _stop_process(third)
    finally:
        _stop_process(first)


# ===========================================================================
# F3 delta: atomic claim publication, unreadable-claim grace, and the race.
# ===========================================================================


def test_unreadable_claim_is_held_while_fresh(tmp_path):
    jobs_dir = tmp_path / "jobs"
    claim = launcher.claim_path(jobs_dir, DT, 0)
    claim.parent.mkdir(parents=True)
    claim.write_text("")  # empty in-flight claim
    path, why = launcher.acquire_claim(jobs_dir, DT, 0, "b", launcher_pid=os.getpid())
    assert path is None and "already claimed" in why
    assert claim.exists()
    # still held for a second launcher
    path2, why2 = launcher.acquire_claim(jobs_dir, DT, 0, "c", launcher_pid=os.getpid())
    assert path2 is None and "already claimed" in why2


def test_empty_object_claim_is_held_while_fresh(tmp_path):
    jobs_dir = tmp_path / "jobs"
    claim = launcher.claim_path(jobs_dir, DT, 1)
    claim.parent.mkdir(parents=True)
    claim.write_text("{}")
    path, why = launcher.acquire_claim(jobs_dir, DT, 1, "b", launcher_pid=os.getpid())
    assert path is None and "already claimed" in why


def test_unreadable_and_empty_claims_are_reclaimed_after_grace(tmp_path):
    jobs_dir = tmp_path / "jobs"
    old = time.time() - launcher.CLAIM_GRACE_SECONDS - 10.0
    for index, content in enumerate(["", "{}"]):
        claim = launcher.claim_path(jobs_dir, DT, index)
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.write_text(content)
        os.utime(claim, (old, old))
        path, why = launcher.acquire_claim(
            jobs_dir, DT, index, f"r{index}", launcher_pid=os.getpid()
        )
        assert path is not None and why is None
        # the reclaimed claim now has full content
        assert json.loads(path.read_text())["run_id"] == f"r{index}"
        launcher.release_claim(path)


def test_claim_race_never_yields_two_successes(tmp_path):
    import concurrent.futures

    jobs_dir = tmp_path / "jobs"
    for iteration in range(50):
        gpu = iteration  # a fresh (host, GPU) key every iteration

        def attempt(run_id: str):
            return launcher.acquire_claim(
                jobs_dir, DT, gpu, run_id, launcher_pid=os.getpid()
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, [f"a{iteration}", f"b{iteration}"]))
        successes = [path for path, why in results if path is not None]
        assert len(successes) == 1, (iteration, results)
        launcher.release_claim(successes[0])


# ===========================================================================
# R1: 5-minute utilisation warm-up.
# ===========================================================================


def test_utilization_window_requires_three_samples_over_four_minutes():
    now = NOW
    empty = guard.utilization_window([], now)
    assert empty["warm"] is False and empty["sample_count"] == 0
    assert "utilisation window not populated" in empty["reason"]
    one = guard.utilization_window([(now - 10.0, {0: 2.0})], now)
    assert one["warm"] is False and one["sample_count"] == 1
    assert "1/3" in one["reason"]
    two = guard.utilization_window(
        [(now - 250.0, {0: 2.0}), (now - 130.0, {0: 2.0})], now
    )
    assert two["warm"] is False and two["sample_count"] == 2
    # three samples but a too-short span is still warming up
    short_span = guard.utilization_window(
        [(now - 150.0, {0: 2.0}), (now - 100.0, {0: 2.0}), (now - 50.0, {0: 2.0})], now
    )
    assert short_span["warm"] is False and short_span["span_seconds"] == pytest.approx(100.0)
    # three samples over >= 4 minutes: populated
    warm = guard.utilization_window(
        [(now - 250.0, {0: 2.0}), (now - 130.0, {0: 2.0}), (now - 10.0, {0: 2.0})], now
    )
    assert warm["warm"] is True and warm["reason"] is None
    assert warm["span_seconds"] >= 240.0


def test_cold_utilisation_window_holds():
    r = reading(DT)
    r["utilization_window"] = {
        "warm": False,
        "sample_count": 1,
        "span_seconds": 0.0,
        "reason": "utilisation window not populated (1/3 samples, span 0 s/240 s; warming up)",
    }
    d = guard.evaluate(r, guard.get_profile(DT))
    assert_hold(d, "warming up")
    status = guard.build_status(r, guard.get_profile(DT), d, now_epoch=NOW)
    assert status["utilization_window"]["sample_count"] == 1
    assert status["utilization_window"]["warm"] is False


def test_read_gpus_withholds_utilisation_until_window_is_warm():
    def fake_run(command, timeout=30.0):
        joined = " ".join(command)
        if "--query-gpu" in joined:
            return "0, NVIDIA GeForce GTX TITAN, GPU-aaaa, 12288, 0, 2.0\n"
        if "pmon" in command:
            return None
        if "--query-compute-apps" in joined:
            return ""
        return None

    original = guard._run_command
    guard._run_command = fake_run
    try:
        gpus, window = [], None
        # zero samples (current sample appended inside _read_gpus)
        history = guard.deque()
        available, gpus, _pp = guard._read_gpus(set(), history, NOW)
        assert available is True
        assert gpus[0]["utilization_5m_mean_pct"] is None
        # two samples: still warming
        history = guard.deque([(NOW - 120.0, {0: 2.0})])
        available, gpus, _pp = guard._read_gpus(set(), history, NOW)
        window = guard.utilization_window(history, NOW)
        assert window["warm"] is False and gpus[0]["utilization_5m_mean_pct"] is None
        # three samples spanning 4 minutes: warm and averaged
        history = guard.deque([(NOW - 240.0, {0: 2.0}), (NOW - 120.0, {0: 2.0})])
        available, gpus, _pp = guard._read_gpus(set(), history, NOW)
        window = guard.utilization_window(history, NOW)
        assert window["warm"] is True
        assert gpus[0]["utilization_5m_mean_pct"] == pytest.approx(2.0)
    finally:
        guard._run_command = original


# ===========================================================================
# R3: host derivation and mismatch refusal.
# ===========================================================================


def test_detect_host_and_host_mismatch():
    assert launcher.detect_host("deepthought") == "deepthought"
    assert launcher.detect_host("deepthought2.example.iap.kit.edu") == "deepthought2"
    assert launcher.detect_host("author-mac.local") is None
    assert launcher.host_mismatch("deepthought", "deepthought") is None
    assert launcher.host_mismatch("deepthought", None) is None  # local dev machine
    mismatch = launcher.host_mismatch("deepthought2", "deepthought")
    assert mismatch is not None and "host mismatch" in mismatch


# ===========================================================================
# R9: --dry-run must not prune (dry-run writes nothing).
# ===========================================================================


def test_dry_run_does_not_prune_stale_pid_files(tmp_path):
    jobs_dir = tmp_path / "jobs"
    run_dir = jobs_dir / "dead"
    run_dir.mkdir(parents=True)
    pid_file = run_dir / "pid.json"
    pid_file.write_text(
        json.dumps(
            {
                "pid": 2_000_000,
                "host": DT,
                "run_id": "dead",
                "tag": "jepa_dead",
            }
        )
    )
    profile = guard.get_profile(DT)
    state = guard.CollectorState()
    dry = guard.collect_reading(profile, jobs_dir, state, dry_run=True)
    assert pid_file.exists(), "dry-run must never prune"
    assert str(pid_file) in dry["pid_files_pruned"]  # would prune, but did not
    live = guard.collect_reading(profile, jobs_dir, state, dry_run=False)
    assert not pid_file.exists(), "a live collection prunes stale PID files"
    assert str(pid_file) in live["pid_files_pruned"]


def test_launcher_exits_2_on_host_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "detect_host", lambda: "deepthought")
    rc = launcher.main(
        [
            "--run-dir",
            str(tmp_path / "r"),
            "--run-id",
            "x",
            "--tag",
            "jepa_x",
            "--host",
            "deepthought2",
            "--gpu-index",
            "0",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--timeout-seconds",
            "60",
            "--memory-fraction",
            "0.4",
        ]
    )
    assert rc == 2
    assert not (tmp_path / "jobs" / ".locks").exists()
