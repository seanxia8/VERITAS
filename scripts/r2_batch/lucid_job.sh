#!/bin/bash
# r2-lucid HTCondor batch job: LUCiD generation (RUNBOOK_2026-11-21_R2_LUCID_REGEN.md §L6).
# Inputs over xrootd from root://ceph-node-j.etp.kit.edu://dwong/r2/ (sha256-checked): the code
# tarball (R2_CODE), the LUCiD wheelhouse, the LUCiD checkout at e651e836 (code, config, data) and
# the pinned round-2 LUCiD config. Outputs leave as returns_out.tar (HTCondor transfer).
# The batch GPU is a dedicated, unshared slot: JAX may use 80% of it (runbook §0.3), never preallocates.
set -u
export USE_USER=dwong
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd "$W"
XR=root://ceph-node-j.etp.kit.edu://dwong/r2
export XRD_NETWORKSTACK=IPv4
log() { echo "$(date -u +%FT%TZ) lucid_job.sh: $*"; }
python3 -m pip install -q --no-index --target xrdpy ./xrootd-*.whl || { log "xrootd wheel install failed"; exit 5; }
cat > xrcp.py <<"PY"
import sys
from XRootD import client
proc = client.CopyProcess()
proc.add_job(sys.argv[1], sys.argv[2], force=True)
st = proc.prepare()
if not st.ok: sys.exit(f"prepare: {st.message}")
st, res = proc.run()
if not st.ok or (res and not res[0]["status"].ok):
    sys.exit(f"copy: {st.message} {res[0]['status'].message if res else ''}")
PY
xr() { local d="$2"; [[ "$d" = /* ]] || d="$W/$d"; PYTHONPATH=$W/xrdpy timeout 1800 python3 $W/xrcp.py "$1" "$d"; }
mkdir -p returns
finish() { log "packing outputs"; tar cf returns_out.tar returns 2>/dev/null || true; }
trap finish EXIT
fetch() {
  mkdir -p "$(dirname "$2")"
  xr "$XR/$1" "$2" 2>&1 | tail -2 || true; [ -s "$2" ] || { log "xrdcp failed: $1"; exit 3; }
  want=$(awk -v f="$1" '$2==f {print $1}' SHA256SUMS); got=$(sha256sum "$2" | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { log "sha256 mismatch for $1 ($got vs $want)"; exit 4; }
}
log "host $(hostname)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
xr "$XR/SHA256SUMS" SHA256SUMS 2>&1 | tail -3; [ -s SHA256SUMS ] || { log "cannot fetch SHA256SUMS"; exit 3; }
fetch "$R2_CODE" code.tar.gz; mkdir -p code && tar xzf code.tar.gz -C code
fetch lucid/wheels_lucid_cp311.tar wheels.tar; tar xf wheels.tar
fetch lucid/LUCiD_e651e836.tar lucidsrc.tar; mkdir -p lucidsrc && tar xf lucidsrc.tar -C lucidsrc
fetch lucid/lucid_config.tar lucidpin.tar; mkdir -p lucidpin && tar xf lucidpin.tar -C lucidpin
for f in ${R2_FILES:-}; do fetch "$f" "staged/$f"; done  # r2-tier1: optional staged inputs
python3 -m pip install -q --no-index --find-links wheels --target pylib -r code/scripts/r2_batch/requirements_lucid_cp311.txt \
  || { log "pip failed"; exit 5; }
PIN=$(dirname "$(find lucidpin -name WCTE_geom_config.json | head -1)")
export PYTHONPATH=$W/code/src:$W/lucidsrc:$W/pylib LUCID_ROOT=$W/lucidsrc JEPA_LUCID_CONFIG=$W/$PIN
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 TMPDIR=$W
export OMP_NUM_THREADS=${RequestCPUs:-4}
for kv in ${R2_EXTRA_ENV:-}; do export "$kv"; done
python3 -c "import jax, numpy, lucid; print('env', jax.__version__, numpy.__version__, jax.devices())"
log "run: lucid_generate.py $*"
timeout -s TERM -k 300 ${R2_TIME_LIMIT_S:-80000} python3 code/scripts/lucid_generate.py "$@" \
  --shards returns/shards --cache returns/cache 2>&1 | grep -v -i "warn"
rc=${PIPESTATUS[0]}; log "exit $rc"; echo $rc > returns/JOB_EXIT_CODE
nvidia-smi --query-gpu=memory.used --format=csv,noheader > returns/gpu_mem_end.txt 2>/dev/null || true
exit 0
