#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

QUASAR_RUN_ID="${QUASAR_RUN_ID:-${RUN_ID:-}}"
if [[ -z "${QUASAR_RUN_ID:-}" ]]; then
  QUASAR_RUN_ID="$(quasar-incentive current-run --run-id-only)"
fi
export QUASAR_RUN_ID

POLL="${VALIDATION_JOBS_POLL_INTERVAL:-${QUASAR_VALIDATION_JOBS_POLL_INTERVAL:-30}}"
MAX_JOBS="${VALIDATION_JOBS_MAX_JOBS:-${QUASAR_VALIDATION_JOBS_MAX_JOBS:-0}}"
MAX_PASSES="${VALIDATION_JOBS_MAX_PASSES:-${QUASAR_VALIDATION_JOBS_MAX_PASSES:-0}}"
JOB_TTL_SEC="${VALIDATION_JOB_TTL_SEC:-${QUASAR_VALIDATION_JOB_TTL_SEC:-900}}"
GRANT_TTL_SEC="${VALIDATION_GRANT_TTL_SEC:-${QUASAR_GRANT_TTL_SEC:-900}}"
GRANT_MODE="${VALIDATION_GRANT_MODE:-${QUASAR_GRANT_MODE:-presigned}}"
SAMPLE_RATE="${VALIDATION_SAMPLE_RATE:-${QUASAR_VALIDATION_SAMPLE_RATE:-1.0}}"

if [[ -z "${QUASAR_RUN_ID:-}" ]]; then
  echo "error: no active run in bucket; start scripts/run_orchestrator.sh first" >&2
  exit 2
fi

echo "[validation-jobs] run_id=${QUASAR_RUN_ID} netuid=${QUASAR_NETUID:-508}"
echo "[validation-jobs] poll=${POLL} max_passes=${MAX_PASSES} max_jobs=${MAX_JOBS} sample_rate=${SAMPLE_RATE} grant_mode=${GRANT_MODE}"

pass=0
while true; do
  pass=$((pass + 1))
  echo "[validation-jobs] pass=${pass} starting at $(date -u +%FT%TZ)"

  args=(
    --run-id "${QUASAR_RUN_ID}"
    --max-jobs "${MAX_JOBS}"
    --job-ttl-sec "${JOB_TTL_SEC}"
    --grant-ttl-sec "${GRANT_TTL_SEC}"
    --grant-mode "${GRANT_MODE}"
    --sample-rate "${SAMPLE_RATE}"
  )
  if [[ -n "${QUASAR_VALIDATOR_HOTKEY:-}" ]]; then
    args+=(--validator-hotkey "${QUASAR_VALIDATOR_HOTKEY}")
  fi
  if [[ "${QUASAR_ALLOW_VALIDATOR_HEARTBEAT_DISCOVERY:-0}" == "1" || "${QUASAR_ALLOW_VALIDATOR_HEARTBEAT_DISCOVERY:-}" == "true" ]]; then
    args+=(--allow-validator-heartbeat-discovery)
  fi

  quasar-incentive validation-jobs "${args[@]}"
  echo "[validation-jobs] pass=${pass} done"

  if [[ "${MAX_PASSES}" != "0" && "${pass}" -ge "${MAX_PASSES}" ]]; then
    exit 0
  fi
  sleep "${POLL}"
done
