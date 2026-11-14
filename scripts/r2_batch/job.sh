#!/bin/bash
# Round-2 HTCondor batch job (RUNBOOK_2026-11-13_R2_HTCONDOR_BATCH.md, _STAGING_DWONG.md, _BATCH_REOPEN.md).
# Inputs over xrootd from root://ceph-node-j.etp.kit.edu://dwong/r2/ (sha256-checked against SHA256SUMS);
# job state (ledger copy, smokes, run directory) arrives as returns_in.tar by HTCondor transfer; outputs
# leave as returns_out.tar by HTCondor transfer (xrootd writes are refused from batch nodes).
# Environment from the JDL: R2_FILES (space-separated staged paths), R2_CODE (code tarball path),
# R2_TIME_LIMIT_S (SIGTERM before the walltime), R2_EXTRA_ENV (optional "K=V K=V").
set -u
export USE_USER=dwong
W=${_CONDOR_SCRATCH_DIR:-$PWD}; cd "$W"
XR=root://ceph-node-j.etp.kit.edu://dwong/r2
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_101cuda/x86_64-centos7-gcc8-opt/setup.sh
log() { echo "$(date -u +%FT%TZ) job.sh: $*"; }
finish() { log "packing outputs"; tar cf returns_out.tar returns 2>/dev/null || true; }
trap finish EXIT
fetch() {  # fetch <staged relative path> <local path>
  mkdir -p "$(dirname "$2")"
  ( source $LCG >/dev/null 2>&1; xrdcp -s -f "$XR/$1" "$2" ) || { log "xrdcp failed: $1"; exit 3; }
  want=$(awk -v f="$1" '$2==f {print $1}' SHA256SUMS); got=$(sha256sum "$2" | cut -d' ' -f1)
  [ -n "$want" ] && [ "$want" = "$got" ] || { log "sha256 mismatch for $1 ($got vs $want)"; exit 4; }
}
log "host $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
( source $LCG >/dev/null 2>&1; xrdcp -s -f "$XR/SHA256SUMS" SHA256SUMS ) || { log "cannot fetch SHA256SUMS"; exit 3; }
fetch "$R2_CODE" code.tar.gz; mkdir -p code && tar xzf code.tar.gz -C code
fetch env/wheels_cp311.tar wheels.tar; tar xf wheels.tar
fetch lucid/lucid_config.tar lucid.tar; mkdir -p lucid && tar xf lucid.tar -C lucid
for f in $R2_FILES; do fetch "$f" "staged/$f"; done
python3 -m pip install -q --no-index --find-links wheels --target pylib -r code/scripts/r2_batch/requirements_cp311.txt || { log "pip failed"; exit 5; }
mkdir -p returns; [ -f returns_in.tar ] && tar xf returns_in.tar
export PYTHONPATH=$W/code/src:$W/code:$W/pylib JEPA_LUCID_CONFIG=$W/lucid/config R2_GPU_PROFILE=batch
export TMPDIR=$W OMP_NUM_THREADS=${RequestCPUs:-4} MKL_NUM_THREADS=${RequestCPUs:-4}
export R2_BATCH_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
export R2_BATCH_MEM_LIMIT_GIB=$(python3 -c "import torch;print(round(0.9*torch.cuda.get_device_properties(0).total_memory/2**30,2))")
export R2_SITE="$(grep -E '^CloudSite' ${_CONDOR_MACHINE_AD:-/dev/null} 2>/dev/null | cut -d'"' -f2):$(hostname)"
for kv in ${R2_EXTRA_ENV:-}; do export "$kv"; done
python3 -c "import torch,numpy,scipy,h5py,sys;print('env', sys.version.split()[0], torch.__version__, numpy.__version__, scipy.__version__, h5py.__version__, torch.cuda.get_device_name(0))"
log "run: $*"
timeout -s TERM -k 300 ${R2_TIME_LIMIT_S:-80000} python3 -m jepa_noise.ext_r2 "$@"
rc=$?; log "ext_r2 exit $rc"; echo $rc > returns/JOB_EXIT_CODE
exit 0
