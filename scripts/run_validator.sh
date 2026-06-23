#!/usr/bin/env bash
set -uo pipefail

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

POLL="${VALIDATOR_POLL_INTERVAL:-${QUASAR_VALIDATOR_POLL_INTERVAL:-30}}"
MAX_RECEIPTS="${VALIDATOR_MAX_RECEIPTS:-${QUASAR_VALIDATOR_MAX_RECEIPTS:-0}}"
MAX_PASSES="${VALIDATOR_MAX_PASSES:-${QUASAR_VALIDATOR_MAX_PASSES:-0}}"
WINDOW_ID="${QUASAR_SCORE_WINDOW_ID:-run=${QUASAR_RUN_ID}}"
WORKER_ID="${VALIDATOR_WORKER_ID:-${QUASAR_VALIDATOR_WORKER_ID:-validator0}}"
OWNER_IDENTITY="${QUASAR_OWNER_IDENTITY:-${OWNER_IDENTITY:-}}"
QUASAR_INDEPENDENT_QUASAR_EVAL="${QUASAR_INDEPENDENT_QUASAR_EVAL:-1}"
export QUASAR_INDEPENDENT_QUASAR_EVAL

if [[ -z "${QUASAR_RUN_ID:-}" ]]; then
  echo "error: no active run in bucket; start scripts/run_orchestrator.sh first" >&2
  exit 2
fi
if [[ -z "${OWNER_IDENTITY:-}" ]]; then
  echo "error: set QUASAR_OWNER_IDENTITY" >&2
  exit 2
fi

echo "[validator] run_id=${QUASAR_RUN_ID} netuid=${QUASAR_NETUID:-508}"
echo "[validator] worker_id=${WORKER_ID} poll=${POLL} max_passes=${MAX_PASSES} max_jobs=${MAX_RECEIPTS}"
echo "[validator] set_weights=live"
echo "[validator] independent_quasar_eval=${QUASAR_INDEPENDENT_QUASAR_EVAL}"

args=(
  --run-id "${QUASAR_RUN_ID}"
  --worker-id "${WORKER_ID}"
  --owner-identity "${OWNER_IDENTITY}"
  --max-jobs "${MAX_RECEIPTS}"
  --max-passes "${MAX_PASSES}"
  --poll-interval-sec "${POLL}"
  --window-id "${WINDOW_ID}"
)
if [[ "${QUASAR_INDEPENDENT_QUASAR_EVAL:-0}" == "1" || "${QUASAR_INDEPENDENT_QUASAR_EVAL:-}" == "true" ]]; then
  args+=(--independent-quasar-eval)
fi
exec quasar-incentive validator-run "${args[@]}"
