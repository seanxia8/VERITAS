#!/usr/bin/env bash
set -uo pipefail
ROOT=/work/dwong/jepa_a10
WORK=/work/dwong/jepa_a10_work
JOBS=$ROOT/jobs
JEPA=$ROOT/scripts/jepa_run.sh
CFG=src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json
GEOM=/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json
LOG=$WORK/a10_chain.log
mkdir -p "$WORK/frozen_decisions"
exec >>"$LOG" 2>&1
log(){ echo "$(date -u +%FT%TZ) $*"; }
log "CHAIN START host=$(hostname) step2_running=$(pgrep -f 'jepa_noise.lucid_run.*a10_dev_step2' | tr '\n' ',')"

STEP2_DEC=$WORK/dev_step2/dev/step2_probe_decision.json
while [ ! -f "$STEP2_DEC" ]; do
  if ! pgrep -f 'jepa_noise.lucid_run.*a10_dev_step2' >/dev/null 2>&1; then
    log "CHAIN ABORT: a10_dev_step2 process gone without $STEP2_DEC"; exit 2
  fi
  sleep 60
done
log "STEP2 decision present: $STEP2_DEC"
sleep 5
# wait for the two step2 run_status files to be complete
for i in $(seq 1 30); do
  incomplete=$(find "$WORK/dev_step2/runs" -name run_status.json -exec grep -L '"outcome": "complete"' {} \; 2>/dev/null | wc -l)
  total=$(find "$WORK/dev_step2/runs" -name run_status.json 2>/dev/null | wc -l)
  if [ "$total" -ge 2 ] && [ "$incomplete" -eq 0 ]; then break; fi
  sleep 20
done
PLAIN=$(find "$WORK/dev_step2/runs" -path '*sigreg_jepa*s9000*/probes.json' 2>/dev/null | head -1)
if [ -z "$PLAIN" ]; then
  PLAIN=$(find "$WORK/dev_step2/runs" -path '*sigreg_jepa*/probes.json' 2>/dev/null | head -1)
fi
log "STEP2 plain reference: ${PLAIN:-MISSING}"
[ -n "$PLAIN" ] || { log "CHAIN ABORT: no step2 sigreg_jepa probes.json"; exit 3; }

log "STEP3 launch"
(cd "$ROOT" && env JEPA_JOBS_DIR="$JOBS" JEPA_JOB_MODULE=jepa_noise.lucid_run JEPA_JOB_KIND=training \
  "$JEPA" a10_dev_step3 --host deepthought2 --duration-minutes 720 -- \
  dev --step 3 --out "$WORK/dev_step2" --data-root "$WORK/data_dev" \
  --config "$CFG" --geometry "$GEOM" --plain-reference "$PLAIN")
RC=$?
log "STEP3 exit=$RC"
[ "$RC" -eq 0 ] || { log "CHAIN ABORT: step3 failed"; exit 4; }
STEP3_DEC=$WORK/dev_step2/dev/step3_beta_decision.json
[ -f "$STEP3_DEC" ] || { log "CHAIN ABORT: missing $STEP3_DEC"; exit 5; }

cp "$STEP2_DEC" "$WORK/frozen_decisions/step2_probe_decision.json"
cp "$STEP3_DEC" "$WORK/frozen_decisions/step3_beta_decision.json"
sha256sum "$WORK/frozen_decisions/"*.json > "$WORK/frozen_decisions/SHA256SUMS"
log "DECISIONS FROZEN: $(cat "$WORK/frozen_decisions/SHA256SUMS" | tr '\n' ' ')"
log "STEP2 content: $(tr -d '\n' < "$STEP2_DEC" | head -c 1200)"
log "STEP3 content: $(tr -d '\n' < "$STEP3_DEC" | head -c 1200)"

run_grid() {
  local tag=$1 cells=$2
  log "GRID launch $tag cells=$cells"
  (cd "$ROOT" && env JEPA_JOBS_DIR="$JOBS" JEPA_JOB_MODULE=jepa_noise.lucid_run JEPA_JOB_KIND=training \
    "$JEPA" "$tag" --host deepthought2 --duration-minutes 3000 -- \
    grid --out "$WORK/$tag" --data-root "$WORK/data_$tag" \
    --config "$CFG" --geometry "$GEOM" --cells "$cells" \
    --beta-decision "$STEP3_DEC" --physics-probe-decision "$STEP2_DEC")
  log "GRID exit $tag rc=$?"
}
CELLS0=main_s002_across,main_s002_within,main_s005_across,r11_f1f2f3_within,r11b_law_within
CELLS1=main_s005_within,main_s010_across,main_s010_within,dark_anchor_s010_within,law_t_aging_within,occupancy_s005_within_balanced
run_grid a10_grid_gpu0 "$CELLS0" & P0=$!
sleep 150
run_grid a10_grid_gpu1 "$CELLS1" & P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
log "CHAIN DONE grid0=$R0 grid1=$R1"
