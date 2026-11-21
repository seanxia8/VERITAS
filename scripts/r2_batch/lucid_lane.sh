#!/bin/bash
# r2-lucid lane dispatcher on deepthought2: runs the jobs of a queue file one after another, each
# through the guard (scripts/jepa_run.sh). Queue line: run_id|module|kind|planned_minutes|python|args...
# A job whose run_status.json shows exit 0 is skipped; a guard/claim refusal is retried every 60 s;
# a guard HOLD waits. Stops when the queue is exhausted or a STOP file appears next to the queue.
Q=$1
REPO=/ceph/dwong/jepa/r2_lucid_repo
cd $REPO
while IFS='|' read -r rid module kind minutes py args; do
  [ -z "$rid" ] && continue; case "$rid" in \#*) continue;; esac
  [ -f "$Q.STOP" ] && { echo "$(date -u +%FT%TZ) STOP file, exiting"; exit 0; }
  st=/ceph/dwong/jepa/jobs/$rid/run_status.json
  if [ -f $st ] && grep -q '"exit_status": 0' $st; then echo "$(date -u +%FT%TZ) skip $rid (done)"; continue; fi
  for try in $(seq 1 600); do
    [ -f "$Q.STOP" ] && exit 0
    echo "$(date -u +%FT%TZ) start $rid (try $try)"
    JEPA_PYTHON=$py JEPA_JOB_MODULE=$module JEPA_JOB_KIND=$kind bash scripts/jepa_run.sh $rid --duration-minutes $minutes --host deepthought2 -- $args
    rc=$?
    echo "$(date -u +%FT%TZ) $rid rc=$rc"
    if [ $rc -eq 3 ] || { [ $rc -ne 0 ] && grep -q "claim refused\|already claimed" /ceph/dwong/jepa/jobs/$rid/run.log 2>/dev/null && [ ! -f $st ]; }; then sleep 60; continue; fi
    break
  done
done < "$Q"
echo "$(date -u +%FT%TZ) queue $Q finished"
