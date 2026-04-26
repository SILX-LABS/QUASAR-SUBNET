#!/bin/bash
set -euo pipefail

# Optional legacy helper for split-host deployments.
# In the consolidated quasar layout, validator and API share one filesystem,
# so this script only refreshes REVISION unless QUASAR_API_REMOTE is set.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCAL_STATE="${QUASAR_STATE_DIR:-${DISTIL_STATE_DIR:-$REPO_ROOT/state/}}"
REVISION_FILE="${QUASAR_REVISION_FILE:-${DISTIL_REVISION_FILE:-$REPO_ROOT/REVISION}}"
REMOTE="${QUASAR_API_REMOTE:-${DISTIL_API_REMOTE:-}}"
REMOTE_ROOT="${QUASAR_REMOTE_ROOT:-${DISTIL_REMOTE_ROOT:-/opt/quasar/repo}}"
REMOTE_STATE="${QUASAR_REMOTE_STATE:-${DISTIL_REMOTE_STATE:-$REMOTE_ROOT/state/}}"

mkdir -p "${LOCAL_STATE%/}"

git -C "$REPO_ROOT" rev-parse --short HEAD > "$REVISION_FILE" 2>/dev/null || true

if [ -z "$REMOTE" ] || [ "$REMOTE" = "localhost" ] || [ "$REMOTE" = "local" ] || [ "$REMOTE" = "self" ]; then
  echo "quasar API is local; state rsync skipped"
  exit 0
fi

ssh "$REMOTE" "mkdir -p \"$REMOTE_STATE\""

rsync -az --timeout=10 \
  --temp-dir=/tmp \
  --delay-updates \
  --exclude='announcement.json' \
  --include='*.json' \
  --include='api_cache/' \
  --include='api_cache/*.json' \
  --include='pod_logs/' \
  --include='pod_logs/*.log' \
  --include='eval_data/' \
  --include='eval_data/*.json' \
  --include='benchmarks/' \
  --include='benchmarks/*.json' \
  --exclude='*.tmp' \
  --exclude='*' \
  "$LOCAL_STATE" "$REMOTE:$REMOTE_STATE" 2>/dev/null

rsync -az "$REVISION_FILE" "$REMOTE:$REMOTE_ROOT/REVISION" 2>/dev/null
