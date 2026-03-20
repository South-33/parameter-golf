#!/usr/bin/env bash
set -euo pipefail

cd /workspace/parameter-golf

RUN_ID="${RUN_ID:-runpod_job}"
RUN_CMD="${RUN_CMD:-bash experiments/runpod_sp4096_smoke.sh}"

mkdir -p logs

driver_log="logs/${RUN_ID}.driver.log"
trainer_log="logs/${RUN_ID}.txt"
status_file="logs/${RUN_ID}.status"
pid_file="logs/${RUN_ID}.pid"
done_file="logs/${RUN_ID}.done"
failed_file="logs/${RUN_ID}.failed"

rm -f "$driver_log" "$status_file" "$pid_file" "$done_file" "$failed_file"

{
  echo "run_id=$RUN_ID"
  echo "started_at=$(date -Is)"
  echo "cwd=$(pwd)"
  echo "run_cmd=$RUN_CMD"
  echo "trainer_log=$trainer_log"
  echo "status=RUNNING"
} | tee -a "$driver_log" > "$status_file"

echo "$$" > "$pid_file"

set +e
bash -lc "$RUN_CMD" >> "$driver_log" 2>&1
cmd_status=$?
set -e

if [[ $cmd_status -eq 0 ]] && [[ -f "$trainer_log" ]] && grep -q "final_int8_zlib_roundtrip_exact" "$trainer_log"; then
  exact_line="$(grep "final_int8_zlib_roundtrip_exact" "$trainer_log" | tail -n 1)"
  {
    echo "finished_at=$(date -Is)"
    echo "status=DONE"
    echo "exact=$exact_line"
  } | tee -a "$driver_log" > "$status_file"
  printf '%s\n' "$exact_line" > "$done_file"
  exit 0
fi

{
  echo "finished_at=$(date -Is)"
  echo "status=FAILED"
  echo "cmd_status=$cmd_status"
  if [[ -f "$trainer_log" ]]; then
    echo "trainer_tail_start"
    tail -n 40 "$trainer_log"
    echo "trainer_tail_end"
  else
    echo "trainer_log_missing=1"
  fi
} | tee -a "$driver_log" > "$status_file"

printf 'cmd_status=%s\n' "$cmd_status" > "$failed_file"
exit "${cmd_status:-1}"
