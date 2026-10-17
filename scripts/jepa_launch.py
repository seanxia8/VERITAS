#!/home/dwong/anaconda3/bin/python3
"""jepa_launch.py -- PID-contract launcher for one JEPA GPU job.

``scripts/jepa_run.sh`` is the guard-gated front door; it execs this launcher
with the guard-approved GPU, memory fraction and timeout.  This process:

* takes an atomic per-(host, GPU) claim under ``<jobs-dir>/.locks/`` so two
  starts cannot both use the same GPU (release on exit; stale claims from dead
  launchers/jobs are reclaimed);
* forks a child, which calls ``os.setsid()`` (new process group/session),
  ``os.nice(10)``, ``chdir`` to the code root, redirects stdout/stderr to
  ``<run-dir>/job.log`` and then ``execv``s the job;
* writes ``<run-dir>/pid.json`` atomically **before the job may run** with the
  child's PID (the process that holds the GPU), the child's pgid, the command
  tag ``jepa_<run_id>``, host, GPU index, job kind, planned duration, timeout,
  module, launcher PID and the full command.  A pipe gates the child until the
  PID file exists;
* enforces the 1.5 x benchmarked timeout itself (SIGTERM to the job's process
  group, then SIGKILL after ``--kill-after-seconds``), forwards SIGTERM/SIGINT/
  SIGHUP received by the launcher to the job's process group, reaps the child
  and writes ``<run-dir>/run_status.json`` atomically with the exit status;
* removes ``pid.json`` and releases the GPU claim on exit.

Why: ``timeout(1)`` + ``nice(1)`` wrappers record the wrapper PID, so the guard
would classify our own GPU process as foreign and the per-process caps could
never fire.  The recorded PID here is the real job process (``os.getpid()`` of
the forked child across its exec).

The interpreter must be the conda Python on both hosts:
``/home/dwong/anaconda3/bin/python3`` (3.12.2); the system Python on
deepthought2 is 3.9 and cannot parse the guard's PEP 604 annotations.

Stdlib only; never imports torch in this process (the child job does).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

LAUNCH_VERSION = "jepa_launch_v4_2026-10-17"

DEFAULT_PYTHON = "/home/dwong/anaconda3/bin/python3"

#: exec'd in the child: set the section 6.2 cap, then run the job module.
#: argv: python -c BOOTSTRAP <fraction> <module> <tag> [job args...]
JOB_BOOTSTRAP = """
import runpy
import sys

import torch

torch.cuda.set_per_process_memory_fraction(float(sys.argv[1]))
module = sys.argv[2]
sys.argv = [module] + sys.argv[4:]
runpy.run_module(module, run_name="__main__")
"""

CLAIM_DIR_NAME = ".locks"

#: a claim younger than this is never stolen, even if it does not parse yet
CLAIM_GRACE_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Small helpers (atomic JSON, time).
# ---------------------------------------------------------------------------


def utc_iso(epoch: float | None = None) -> str:
    instant = time.time() if epoch is None else float(epoch)
    return datetime.fromtimestamp(instant, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, payload: Any) -> Path:
    """Write JSON atomically (temp file + fsync + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=False)
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


def pid_alive(pid: int | None) -> bool:
    """True when ``pid`` exists (PermissionError still means alive)."""
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


# ---------------------------------------------------------------------------
# Atomic per-(host, GPU) claim.  Unit-tested with tmp dirs.
# ---------------------------------------------------------------------------


def claim_path(jobs_dir: Path, host: str, gpu_index: int) -> Path:
    return Path(jobs_dir) / CLAIM_DIR_NAME / f"{host}_gpu{int(gpu_index)}.claim"


def read_claim(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def claim_age_seconds(path: Path, now_epoch: float | None = None) -> float:
    """Age of the claim file from its mtime; 0.0 (fresh) when unreadable."""
    now = time.time() if now_epoch is None else float(now_epoch)
    try:
        return max(0.0, now - Path(path).stat().st_mtime)
    except OSError:
        return 0.0


def claim_is_stale(
    payload: dict[str, Any] | None,
    alive=pid_alive,
    *,
    age_seconds: float = 0.0,
    grace_seconds: float = CLAIM_GRACE_SECONDS,
) -> bool:
    """Whether an existing claim may be removed.

    A claim younger than ``grace_seconds`` is never stolen (it may be a
    launcher between ``link`` and ``fork``, or an unreadable in-flight file).
    After the grace period it is stale only when it parses as a dict and both
    its launcher and job PIDs are dead; an unreadable claim is stale after the
    grace period (nothing can be recovered from it).
    """
    if age_seconds < float(grace_seconds):
        return False
    if not isinstance(payload, dict):
        return True
    return not (alive(payload.get("launcher_pid")) or alive(payload.get("job_pid")))


def _publish_claim_atomically(claim: Path, payload: dict[str, Any]) -> None:
    """Create ``claim`` with full content via ``os.link`` (atomic publication).

    The claim name only ever appears after a complete, fsynced JSON payload
    exists, so a concurrent launcher can never read an empty/in-flight claim.
    Raises ``FileExistsError`` when another launcher owns the claim.
    """
    claim = Path(claim)
    claim.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{claim.name}.", suffix=".tmp", dir=str(claim.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, claim)  # atomic: fails if the claim exists
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def acquire_claim(
    jobs_dir: Path,
    host: str,
    gpu_index: int,
    run_id: str,
    *,
    launcher_pid: int | None = None,
    now_epoch: float | None = None,
) -> tuple[Path | None, str | None]:
    """Take the per-(host, GPU) claim atomically.

    Returns ``(path, None)`` on success or ``(None, reason)`` when the GPU is
    claimed.  A claim is only reclaimed after ``CLAIM_GRACE_SECONDS`` with both
    recorded PIDs dead (see ``claim_is_stale``); an unreadable/incomplete claim
    is treated as held while it is young.
    """
    now = time.time() if now_epoch is None else float(now_epoch)
    claim = claim_path(jobs_dir, host, gpu_index)
    payload = {
        "host": host,
        "gpu_index": int(gpu_index),
        "run_id": run_id,
        "launcher_pid": int(launcher_pid if launcher_pid is not None else os.getpid()),
        "job_pid": None,
        "acquired_at_utc": utc_iso(now),
        "acquired_at_epoch": now,
        "claim_file": str(claim),
    }
    for _attempt in range(3):
        try:
            _publish_claim_atomically(claim, payload)
            return claim, None
        except FileExistsError:
            existing = read_claim(claim)
            age = claim_age_seconds(claim, now)
            if not claim_is_stale(existing, age_seconds=age):
                holder = existing or {}
                detail = (
                    f"run {holder.get('run_id')!r} (launcher pid "
                    f"{holder.get('launcher_pid')}, job pid {holder.get('job_pid')})"
                    if existing
                    else "an unreadable claim younger than the grace period"
                )
                return None, (
                    f"GPU {gpu_index} on {host} is already claimed by {detail}"
                )
            try:
                claim.unlink()
            except FileNotFoundError:
                pass
            continue
    return None, f"could not claim GPU {gpu_index} on {host} after retries"


def update_claim_job_pid(path: Path, job_pid: int, now_epoch: float | None = None) -> None:
    payload = read_claim(path) or {}
    payload["job_pid"] = int(job_pid)
    payload["job_started_at_utc"] = utc_iso(now_epoch)
    atomic_write_json(path, payload)


def release_claim(path: Path, launcher_pid: int | None = None) -> bool:
    """Remove own claim; never remove a claim another launcher now owns."""
    payload = read_claim(path)
    if payload is not None:
        owner = payload.get("launcher_pid")
        if launcher_pid is not None and owner not in (None, int(launcher_pid)):
            return False
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return False
    return True


# ---------------------------------------------------------------------------
# Host derivation (R3): the shared /ceph jobs directory means a mislabelled
# host would make the real host's guard ignore the job.
# ---------------------------------------------------------------------------

KNOWN_HOSTS = ("deepthought", "deepthought2")


def detect_host(hostname: str | None = None) -> str | None:
    """Return ``deepthought``/``deepthought2`` for this machine, else None."""
    name = (hostname if hostname is not None else socket.gethostname()) or ""
    short = name.strip().lower().split(".")[0]
    return short if short in KNOWN_HOSTS else None


def host_mismatch(requested: str | None, detected: str | None) -> str | None:
    """Refusal text when a requested host contradicts this machine.

    On a machine that is not one of the two GPU hosts (e.g. the author's Mac)
    the explicit host is allowed for local dry runs.
    """
    if detected is None or requested is None:
        return None
    if requested != detected:
        return (
            f"host mismatch: requested {requested!r} but this machine is {detected!r}; "
            "pid.json and the guard status/claim paths would be wrong"
        )
    return None


# ---------------------------------------------------------------------------
# Launcher.
# ---------------------------------------------------------------------------


def _manual_exit_code(status: int) -> int:
    if os.WIFEXITED(status):
        return int(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        return 128 + int(os.WTERMSIG(status))
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fork/exec one JEPA job, record its real PID, enforce its timeout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tag", required=True, help="command tag jepa_<run_id>")
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--jobs-dir", type=Path, default=Path("/ceph/dwong/jepa/jobs"))
    parser.add_argument("--job-kind", default="training")
    parser.add_argument("--planned-minutes", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--kill-after-seconds", type=int, default=300)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--module", default="jepa_noise.train")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("job_args", nargs=argparse.REMAINDER, help="job arguments after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    job_args = list(args.job_args)
    if job_args and job_args[0] == "--":
        job_args = job_args[1:]
    # R3: refuse a --host that contradicts this machine (on a GPU host).
    mismatch = host_mismatch(args.host, detect_host())
    if mismatch is not None:
        print(f"jepa_launch: refusing to start {args.run_id}: {mismatch}", file=sys.stderr)
        return 2
    # resolve a bare interpreter name through PATH (execv does not search PATH)
    python_executable = shutil.which(str(args.python)) or str(args.python)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file) if args.log_file is not None else run_dir / "run.log"

    def log(message: str) -> None:
        line = f"{utc_iso()} {message}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            pass

    claim, refusal = acquire_claim(
        Path(args.jobs_dir), args.host, args.gpu_index, args.run_id
    )
    if claim is None:
        print(f"jepa_launch: refusing to start {args.run_id}: {refusal}", file=sys.stderr)
        log(f"jepa_launch: claim refused: {refusal}")
        return 3

    started_epoch = time.time()
    started_monotonic = time.monotonic()
    job_log_path = run_dir / "job.log"
    job_log_fd = os.open(str(job_log_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    read_fd, write_fd = os.pipe()
    launcher_pid = os.getpid()
    child_pid: int | None = None
    forwarded_signal: int | None = None

    def forward(signum: int, _frame) -> None:
        nonlocal forwarded_signal
        forwarded_signal = signum
        if child_pid is not None:
            try:
                os.killpg(child_pid, signum)
            except ProcessLookupError:
                pass

    previous_handlers = {
        signum: signal.signal(signum, forward) for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        child_pid = os.fork()
        if child_pid == 0:  # child: gate -> new session -> nice -> exec job
            try:
                os.close(write_fd)
                gate = os.read(read_fd, 1)
                os.close(read_fd)
                if gate != b"1":
                    os._exit(126)  # parent died before pid.json was committed
                # R6: do not swallow a signal in the pre-exec window; the job
                # must start with default dispositions.
                for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                    signal.signal(signum, signal.SIG_DFL)
                os.setsid()
                os.nice(max(0, int(args.nice)))
                os.chdir(str(args.workdir))
                os.dup2(job_log_fd, 1)
                os.dup2(job_log_fd, 2)
                if job_log_fd > 2:
                    os.close(job_log_fd)
                exec_argv = [
                    python_executable,
                    "-c",
                    JOB_BOOTSTRAP,
                    str(args.memory_fraction),
                    str(args.module),
                    str(args.tag),
                    *job_args,
                ]
                os.execv(python_executable, exec_argv)
            except BaseException:  # pragma: no cover - child never returns
                os._exit(127)
        # parent
        os.close(read_fd)
        os.close(job_log_fd)
        pid_payload = {
            "pid": int(child_pid),
            "pgid": int(child_pid),  # child called setsid(): pgid == pid
            "tag": args.tag,
            "run_id": args.run_id,
            "host": args.host,
            "gpu_index": int(args.gpu_index),
            "job_kind": args.job_kind,
            "planned_minutes": float(args.planned_minutes),
            "timeout_seconds": int(args.timeout_seconds),
            "started_at_utc": utc_iso(started_epoch),
            "started_at_epoch": started_epoch,
            "module": args.module,
            "launcher_pid": launcher_pid,
            "launcher_version": LAUNCH_VERSION,
            "command": [
                python_executable,
                "-c",
                JOB_BOOTSTRAP,
                str(args.memory_fraction),
                str(args.module),
                str(args.tag),
                *job_args,
            ],
        }
        atomic_write_json(run_dir / "pid.json", pid_payload)
        update_claim_job_pid(claim, child_pid, started_epoch)
        os.write(write_fd, b"1")
        os.close(write_fd)
        log(
            f"jepa_launch: started run_id={args.run_id} pid={child_pid} pgid={child_pid} "
            f"gpu={args.gpu_index} host={args.host} timeout={args.timeout_seconds}s"
        )

        deadline = started_monotonic + max(1.0, float(args.timeout_seconds))
        kill_deadline: float | None = None
        outcome = "exited"
        timed_out = False
        while True:
            waited, status = os.waitpid(child_pid, os.WNOHANG)
            if waited == child_pid:
                break
            now = time.monotonic()
            if forwarded_signal is not None and kill_deadline is None:
                kill_deadline = now + max(1.0, float(args.kill_after_seconds))
            if kill_deadline is None and now >= deadline:
                timed_out = True
                outcome = "timeout"
                log(f"jepa_launch: timeout {args.timeout_seconds}s reached; SIGTERM process group {child_pid}")
                try:
                    os.killpg(child_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                kill_deadline = now + max(1.0, float(args.kill_after_seconds))
            elif kill_deadline is not None and now >= kill_deadline:
                log(f"jepa_launch: kill-after reached; SIGKILL process group {child_pid}")
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                kill_deadline = float("inf")
            time.sleep(0.2)
        exit_status = _manual_exit_code(status)
        if not timed_out:
            if forwarded_signal is not None or os.WIFSIGNALED(status):
                outcome = "signalled"
        finished_epoch = time.time()
        status_payload = {
            "run_id": args.run_id,
            "tag": args.tag,
            "host": args.host,
            "gpu_index": int(args.gpu_index),
            "pid": int(child_pid),
            "pgid": int(child_pid),
            "exit_status": int(exit_status),
            "wifsignaled": bool(os.WIFSIGNALED(status)),
            "wtermsig": int(os.WTERMSIG(status)) if os.WIFSIGNALED(status) else None,
            "outcome": outcome,
            "timed_out": bool(timed_out),
            "forwarded_signal": forwarded_signal,
            "planned_minutes": float(args.planned_minutes),
            "timeout_seconds": int(args.timeout_seconds),
            "started_at_utc": utc_iso(started_epoch),
            "finished_at_utc": utc_iso(finished_epoch),
            "wall_seconds": round(finished_epoch - started_epoch, 2),
        }
        atomic_write_json(run_dir / "run_status.json", status_payload)
        try:
            (run_dir / "pid.json").unlink()
        except FileNotFoundError:
            pass
        log(f"jepa_launch: finished run_id={args.run_id} outcome={outcome} exit={exit_status}")
        if timed_out:
            return 124
        return int(exit_status)
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass
        release_claim(claim, launcher_pid)


if __name__ == "__main__":
    raise SystemExit(main())
