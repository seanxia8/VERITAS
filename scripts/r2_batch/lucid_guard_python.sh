#!/bin/bash
# Guard-compatible interpreter for LUCiD (JAX) jobs launched through scripts/jepa_run.sh.
# jepa_launch execs: <python> -c <torch BOOTSTRAP> <memory_fraction> <module> <tag> [args...].
# The LUCiD env has no torch, so this shim maps the guard memory fraction onto the JAX allocator
# (never preallocate; fraction = min(guard fraction, 0.40), runbook 0.3) and runs the job with
# the guard tag kept in its command line (the guard matches pid.json to the process by the tag).
# Every other invocation (guard status, launcher) is passed straight to the LUCiD interpreter.
PY=/ceph/dwong/envs/lucid_e651e836/bin/python
if [ "$1" = "-c" ] && [ "$#" -ge 5 ]; then
  frac=$3; module=$4; tag=$5; shift 5
  cap=$($PY -c "print(min(float('$frac'), 0.40))")
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=$cap
  echo "lucid_py: guard fraction $frac -> XLA_PYTHON_CLIENT_MEM_FRACTION=$cap module=$module tag=$tag" >&2
  if [ "$module" = "lucid_generate" ]; then
    script=/ceph/dwong/jepa/r2_lucid_repo/scripts/lucid_generate.py
    exec $PY -c "import runpy, sys; sys.argv = ['$script'] + sys.argv[2:]; runpy.run_path('$script', run_name='__main__')" "$tag" "$@"
  fi
  exec $PY -c "import runpy, sys; sys.argv = ['$module'] + sys.argv[2:]; runpy.run_module('$module', run_name='__main__')" "$tag" "$@"
fi
exec $PY "$@"
