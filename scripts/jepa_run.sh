#!/usr/bin/env bash
# jepa_run.sh -- guard-gated front door for one JEPA GPU job.
#
# Runbook v4 sections 6.1 ("Running a job") and 6.2 (resource guard):
#   * refuses to start unless the guard status is OK, younger than 2 minutes,
#     not a dry-run status, and consistent with this job's kind/duration;
#   * uses exactly the one GPU index the guard names in CUDA_VISIBLE_DEVICES;
#   * exports at most 4 BLAS/OMP threads and 4 data-loader workers;
#   * execs scripts/jepa_launch.py, which takes the per-(host, GPU) atomic
#     claim, forks/execs the real job with nice -n 10, writes pid.json with the
#     real job PID (not a timeout wrapper), enforces the 1.5 x benchmarked
#     timeout, forwards SIGTERM and writes the exit status.
#
# Python: use the conda interpreter on both hosts (3.12.2):
#   /home/dwong/anaconda3/bin/python3   (override with JEPA_PYTHON)
# The system python on deepthought2 is 3.9 and cannot run the guard/launcher
# (PEP 604 annotations).
#
# Usage:
#   scripts/jepa_run.sh <run_id> [--duration-minutes M] [--host deepthought|deepthought2]
#                       [--dry-run] [-- <job arguments...>]
#
# Environment overrides:
#   JEPA_PYTHON      GPU python interpreter (default: /home/dwong/anaconda3/bin/python3)
#   JEPA_JOB_MODULE  module launched with -m (default: jepa_noise.train)
#   JEPA_JOB_KIND    training | evaluation | probe | smoke | other (default: training)
#   JEPA_JOBS_DIR    jobs directory (default: /ceph/dwong/jepa/jobs)
#   JEPA_HOST        host profile when hostname inference is not enough
#   JEPA_DURATION_MINUTES  same as --duration-minutes
#
# Exit codes: 0 success, 2 usage error, 3 guard/claim refused the start,
# 124 timeout, otherwise the job's own exit status.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GUARD_SCRIPT="$SCRIPT_DIR/jepa_guard.py"
LAUNCH_SCRIPT="$SCRIPT_DIR/jepa_launch.py"

JOBS_DIR="${JEPA_JOBS_DIR:-/ceph/dwong/jepa/jobs}"
PYTHON_BIN="${JEPA_PYTHON:-/home/dwong/anaconda3/bin/python3}"
JOB_MODULE="${JEPA_JOB_MODULE:-jepa_noise.train}"
JOB_KIND="${JEPA_JOB_KIND:-training}"
HOST="${JEPA_HOST:-}"
DURATION_MINUTES="${JEPA_DURATION_MINUTES:-}"
DRY_RUN=0
JOB_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/jepa_run.sh <run_id> [options] [-- <job arguments...>]

Launches one JEPA GPU job under the section 6.2 resource guard.

Options:
  --duration-minutes M   planned (benchmarked) job duration in minutes;
                         timeout is 1.5 x M. Defaults to guard_config.json.
  --host HOST            deepthought | deepthought2 (default: infer from hostname)
  --dry-run              print the exact environment and command, start nothing
  -h, --help             show this help

Environment overrides: JEPA_PYTHON (default /home/dwong/anaconda3/bin/python3),
JEPA_JOB_MODULE, JEPA_JOB_KIND, JEPA_JOBS_DIR, JEPA_HOST, JEPA_DURATION_MINUTES,
JEPA_BLAS_THREADS.
EOF
}

die() {
    printf 'jepa_run: ERROR: %s\n' "$*" >&2
    exit 2
}

print_export() {
    printf 'export %s=%q\n' "$1" "$2"
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
if [ "$#" -lt 1 ]; then
    usage >&2
    exit 2
fi
case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
esac
RUN_ID="$1"
shift
if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    die "run_id must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ (got '$RUN_ID')"
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --duration-minutes)
            [ "$#" -ge 2 ] || die "--duration-minutes needs a value"
            DURATION_MINUTES="$2"
            shift 2
            ;;
        --duration-minutes=*)
            DURATION_MINUTES="${1#*=}"
            shift
            ;;
        --host)
            [ "$#" -ge 2 ] || die "--host needs a value"
            HOST="$2"
            shift 2
            ;;
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                JOB_ARGS+=("$1")
                shift
            done
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option '$1' (job arguments go after --)"
            ;;
        *)
            die "unexpected argument '$1' (job arguments go after --)"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Host and Python
# ---------------------------------------------------------------------------
if [ -z "$HOST" ]; then
    SHORT_HOST="$(hostname -s 2>/dev/null || hostname)"
    case "$SHORT_HOST" in
        deepthought) HOST=deepthought ;;
        deepthought2) HOST=deepthought2 ;;
        *)
            die "cannot infer host from '$SHORT_HOST'; pass --host deepthought|deepthought2"
            ;;
    esac
fi
case "$HOST" in
    deepthought | deepthought2) : ;;
    *) die "unknown host '$HOST' (expected deepthought or deepthought2)" ;;
esac

# R3: the jobs directory is shared over /ceph; a mislabelled host would make
# pid.json claim the wrong host, so the real host's guard would ignore the job
# and the claim path would be wrong.  Derive the host from this machine.
DETECTED_HOST="$(hostname -s 2>/dev/null || hostname)"
case "$DETECTED_HOST" in
    deepthought) MACHINE_HOST=deepthought ;;
    deepthought2) MACHINE_HOST=deepthought2 ;;
    *) MACHINE_HOST="" ;;
esac
if [ -n "$MACHINE_HOST" ] && [ "$MACHINE_HOST" != "$HOST" ]; then
    die "host mismatch: --host/JEPA_HOST is '$HOST' but this machine is '$MACHINE_HOST'; refusing to write a mislabelled pid.json"
fi
if [ -z "$MACHINE_HOST" ]; then
    printf 'jepa_run: WARNING: hostname %s is not deepthought/deepthought2; trusting --host %s (local dry runs only)\n' \
        "$DETECTED_HOST" "$HOST" >&2
fi

case "$JOB_KIND" in
    training | evaluation | probe | smoke | other) : ;;
    *) die "JEPA_JOB_KIND must be training|evaluation|probe|smoke|other (got '$JOB_KIND')" ;;
esac

command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
    die "python interpreter '$PYTHON_BIN' not found (set JEPA_PYTHON; default is /home/dwong/anaconda3/bin/python3)"
[ -f "$GUARD_SCRIPT" ] || die "guard script not found: $GUARD_SCRIPT"
[ -f "$LAUNCH_SCRIPT" ] || die "launcher script not found: $LAUNCH_SCRIPT"

BLAS_THREADS="${JEPA_BLAS_THREADS:-4}"
DATA_WORKERS=4
if ! [[ "$BLAS_THREADS" =~ ^[0-9]+$ ]]; then
    die "JEPA_BLAS_THREADS must be an integer (got '$BLAS_THREADS')"
fi
if [ "$BLAS_THREADS" -gt 4 ]; then
    BLAS_THREADS=4
fi
if [ "$BLAS_THREADS" -lt 1 ]; then
    BLAS_THREADS=1
fi

GUARD_JSON="$JOBS_DIR/guard_${HOST}.json"
TAG="jepa_${RUN_ID}"
RUN_DIR="$JOBS_DIR/$RUN_ID"
PID_FILE="$RUN_DIR/pid.json"
RUN_LOG="$RUN_DIR/run.log"

# ---------------------------------------------------------------------------
# Guard gate: level OK, status younger than 2 minutes, not a dry-run status,
# host match, a named start GPU, the per-start duration cap (4th GPU <= 30 min)
# and the per-kind training-job limit (F2).  The guard script owns the JSON
# parsing and the freshness rules (unit tested); it prints refusal reasons to
# stderr and exits 3.
# ---------------------------------------------------------------------------
DURATION_ARGS=()
if [ -n "$DURATION_MINUTES" ]; then
    if ! [[ "$DURATION_MINUTES" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        die "--duration-minutes must be a positive number (got '$DURATION_MINUTES')"
    fi
    DURATION_ARGS=(--planned-minutes "$DURATION_MINUTES")
fi

if ! STATUS_OUTPUT="$(
    "$PYTHON_BIN" "$GUARD_SCRIPT" \
        --status-for-run "$GUARD_JSON" \
        --host "$HOST" \
        --job-kind "$JOB_KIND" \
        --max-age-seconds 120 \
        "${DURATION_ARGS[@]+"${DURATION_ARGS[@]}"}"
)"; then
    printf 'jepa_run: refusing to start %s: guard status not usable (see above)\n' "$RUN_ID" >&2
    exit 3
fi

START_GPU=""
PER_PROCESS_MEMORY_FRACTION=""
PLANNED_MINUTES=""
TIMEOUT_SECONDS=""
KILL_AFTER_SECONDS=""
MAX_PLANNED_MINUTES=""
while IFS='=' read -r key value; do
    [ -n "$key" ] || continue
    case "$key" in
        START_GPU) START_GPU="$value" ;;
        PER_PROCESS_MEMORY_FRACTION) PER_PROCESS_MEMORY_FRACTION="$value" ;;
        PLANNED_MINUTES) PLANNED_MINUTES="$value" ;;
        TIMEOUT_SECONDS) TIMEOUT_SECONDS="$value" ;;
        KILL_AFTER_SECONDS) KILL_AFTER_SECONDS="$value" ;;
        MAX_PLANNED_MINUTES) MAX_PLANNED_MINUTES="$value" ;;
        *) die "unexpected output line from guard: '$key'" ;;
    esac
done <<<"$STATUS_OUTPUT"

if ! [[ "$START_GPU" =~ ^[0-9]+$ ]]; then
    die "guard returned an invalid GPU index ('$START_GPU')"
fi
if ! [[ "$PER_PROCESS_MEMORY_FRACTION" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    die "guard returned an invalid memory fraction ('$PER_PROCESS_MEMORY_FRACTION')"
fi
if ! [[ "$PLANNED_MINUTES" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    die "guard returned an invalid planned duration ('$PLANNED_MINUTES')"
fi
if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
    die "guard returned an invalid timeout ('$TIMEOUT_SECONDS')"
fi
if ! [[ "$KILL_AFTER_SECONDS" =~ ^[0-9]+$ ]]; then
    die "guard returned an invalid kill-after ('$KILL_AFTER_SECONDS')"
fi
if [ -n "$MAX_PLANNED_MINUTES" ] && ! [[ "$MAX_PLANNED_MINUTES" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    die "guard returned an invalid duration cap ('$MAX_PLANNED_MINUTES')"
fi

# ---------------------------------------------------------------------------
# Environment (exported to the launcher and therefore to the job process)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="$START_GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
export NUMEXPR_NUM_THREADS="$BLAS_THREADS"
export VECLIB_MAXIMUM_THREADS="$BLAS_THREADS"
export JEPA_NUM_WORKERS="$DATA_WORKERS"
export JEPA_DATALOADER_WORKERS="$DATA_WORKERS"
export JEPA_RUN_ID="$RUN_ID"
export JEPA_TAG="$TAG"
export JEPA_HOST="$HOST"
export JEPA_JOB_KIND="$JOB_KIND"
export JEPA_PLANNED_MINUTES="$PLANNED_MINUTES"
export JEPA_GUARD_STATUS="$GUARD_JSON"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

LAUNCH_CMD=(
    "$PYTHON_BIN" "$LAUNCH_SCRIPT"
    --run-dir "$RUN_DIR"
    --run-id "$RUN_ID"
    --tag "$TAG"
    --host "$HOST"
    --gpu-index "$START_GPU"
    --jobs-dir "$JOBS_DIR"
    --job-kind "$JOB_KIND"
    --planned-minutes "$PLANNED_MINUTES"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --kill-after-seconds "$KILL_AFTER_SECONDS"
    --memory-fraction "$PER_PROCESS_MEMORY_FRACTION"
    --module "$JOB_MODULE"
    --python "$PYTHON_BIN"
    --workdir "$REPO_ROOT"
    --log-file "$RUN_LOG"
    -- "${JOB_ARGS[@]+"${JOB_ARGS[@]}"}"
)

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'jepa_run: DRY RUN for %s -- nothing will be started\n' "$RUN_ID"
    printf 'host=%s\n' "$HOST"
    printf 'jobs_dir=%s\n' "$JOBS_DIR"
    printf 'guard_status=%s\n' "$GUARD_JSON"
    printf 'python=%s\n' "$PYTHON_BIN"
    printf 'run_dir=%s\n' "$RUN_DIR"
    printf 'pid_file=%s\n' "$PID_FILE"
    printf 'tag=%s\n' "$TAG"
    printf 'job_kind=%s\n' "$JOB_KIND"
    printf 'start_gpu=%s\n' "$START_GPU"
    printf 'per_process_memory_fraction=%s\n' "$PER_PROCESS_MEMORY_FRACTION"
    printf 'planned_minutes=%s\n' "$PLANNED_MINUTES"
    printf 'timeout_seconds=%s (1.5 x planned, enforced by jepa_launch.py)\n' "$TIMEOUT_SECONDS"
    printf 'kill_after_seconds=%s\n' "$KILL_AFTER_SECONDS"
    printf 'max_planned_minutes_from_guard=%s\n' "${MAX_PLANNED_MINUTES:-none}"
    printf '\n# environment\n'
    print_export CUDA_VISIBLE_DEVICES "$START_GPU"
    print_export CUDA_DEVICE_ORDER PCI_BUS_ID
    print_export OMP_NUM_THREADS "$BLAS_THREADS"
    print_export MKL_NUM_THREADS "$BLAS_THREADS"
    print_export OPENBLAS_NUM_THREADS "$BLAS_THREADS"
    print_export NUMEXPR_NUM_THREADS "$BLAS_THREADS"
    print_export VECLIB_MAXIMUM_THREADS "$BLAS_THREADS"
    print_export JEPA_NUM_WORKERS "$DATA_WORKERS"
    print_export JEPA_DATALOADER_WORKERS "$DATA_WORKERS"
    print_export JEPA_RUN_ID "$RUN_ID"
    print_export JEPA_TAG "$TAG"
    print_export JEPA_HOST "$HOST"
    print_export JEPA_JOB_KIND "$JOB_KIND"
    print_export JEPA_PLANNED_MINUTES "$PLANNED_MINUTES"
    print_export PYTHONPATH "$PYTHONPATH"
    print_export PYTHONDONTWRITEBYTECODE 1
    printf '\n# launcher command (jepa_launch.py forks/execs the real job and records its PID)\n'
    printf 'cd %q &&' "$REPO_ROOT"
    for word in "${LAUNCH_CMD[@]}"; do
        printf ' %q' "$word"
    done
    printf '\n'
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch (exec: this process becomes jepa_launch.py, which writes pid.json)
# ---------------------------------------------------------------------------
mkdir -p "$RUN_DIR"
: >"$RUN_LOG"
printf '%s jepa_run: run_id=%s host=%s gpu=%s tag=%s kind=%s planned_minutes=%s timeout_seconds=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$RUN_ID" "$HOST" "$START_GPU" "$TAG" "$JOB_KIND" \
    "$PLANNED_MINUTES" "$TIMEOUT_SECONDS" >>"$RUN_LOG"
if [ -n "$MAX_PLANNED_MINUTES" ]; then
    printf '%s jepa_run: guard duration cap for this start: <= %s min\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$MAX_PLANNED_MINUTES" >>"$RUN_LOG"
fi
cd "$REPO_ROOT"
exec "${LAUNCH_CMD[@]}"
