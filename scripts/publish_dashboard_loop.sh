#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${QUASAR_DASHBOARD_PUBLISH_ENV:-$ROOT_DIR/.secrets/hippius.env}"
INTERVAL="${QUASAR_DASHBOARD_PUBLISH_INTERVAL:-5}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export QUASAR_BUCKET_ENDPOINT_URL="${QUASAR_BUCKET_ENDPOINT_URL:-https://s3.hippius.com}"
export QUASAR_BUCKET_KEY="${QUASAR_BUCKET_KEY:-dashboard.json}"
export QUASAR_BUCKET_REGION="${QUASAR_BUCKET_REGION:-us-east-1}"
export QUASAR_BUCKET_ADDRESSING_STYLE="${QUASAR_BUCKET_ADDRESSING_STYLE:-path}"

if [[ -z "${QUASAR_BUCKET_NAME:-}" ]]; then
  echo "QUASAR_BUCKET_NAME is required" >&2
  exit 2
fi

while true; do
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/publish_dashboard_json.py" || true
  sleep "$INTERVAL"
done
