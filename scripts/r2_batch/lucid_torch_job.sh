#!/bin/bash
# r2-lucid HTCondor batch job: training / evaluation on LUCiD data (runbook §L7/§L8), same recipe
# code as round 2. Inputs over xrootd (sha256-checked): code (R2_CODE), the round-2 cp311 wheels,
# the pinned LUCiD config, data files (R2_FILES; *.tar files are unpacked in place). Job state
# (run directories to resume or evaluate) arrives as returns_in.tar; outputs leave as returns_out.tar.
set -u
export USE_USER=dwong
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd "$W"
XR=root://ceph-node-j.etp.kit.edu://dwong/r2
export XRD_NETWORKSTACK=IPv4
log() { echo "$(date -u +%FT%TZ) lucid_torch_job.sh: $*"; }
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
mkdir -p returns; [ -f returns_in.tar ] && tar xf returns_in.tar
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
fetch env/wheels_cp311.tar wheels.tar; tar xf wheels.tar
fetch lucid/lucid_config.tar lucidpin.tar; mkdir -p lucidpin && tar xf lucidpin.tar -C lucidpin
for f in $R2_FILES; do
  fetch "$f" "staged/$f"
  case "$f" in *.tar) tar xf "staged/$f" -C "$(dirname "staged/$f")" && rm "staged/$f";; esac
done
python3 -m pip install -q --no-index --find-links wheels --target pylib -r code/scripts/r2_batch/requirements_cp311.txt || { log "pip failed"; exit 5; }
PIN=$(dirname "$(find lucidpin -name WCTE_geom_config.json | head -1)")
export PYTHONPATH=$W/code/src:$W/code:$W/pylib JEPA_LUCID_CONFIG=$W/$PIN R2_REPAIR_OUTPUT_ROOT=$W/returns
export TMPDIR=$W OMP_NUM_THREADS=${RequestCPUs:-4} MKL_NUM_THREADS=${RequestCPUs:-4} JEPA_DATA_KIND=lucid_v1
for kv in ${R2_EXTRA_ENV:-}; do export "$kv"; done
python3 -c "import torch,numpy,sys;print('env', sys.version.split()[0], torch.__version__, numpy.__version__, torch.cuda.get_device_name(0))"
log "run: ${R2_MODULE} $*"
timeout -s TERM -k 300 ${R2_TIME_LIMIT_S:-80000} python3 -m "${R2_MODULE}" "$@"
rc=$?; log "exit $rc"; echo $rc > returns/JOB_EXIT_CODE
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader > returns/gpu_end.txt 2>/dev/null || true
exit 0
