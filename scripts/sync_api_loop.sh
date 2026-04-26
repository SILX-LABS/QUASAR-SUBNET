#!/bin/bash
set -euo pipefail

# Legacy loop wrapper for optional split-host syncs.
# In the single-host quasar layout, sync_api_state.sh is a quick no-op and this
# loop only matters if benchmark syncing is still enabled.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SYNC_INTERVAL="${QUASAR_SYNC_INTERVAL:-${DISTIL_SYNC_INTERVAL:-15}}"
BENCHMARK_EVERY="${QUASAR_BENCHMARK_SYNC_EVERY:-${DISTIL_BENCHMARK_SYNC_EVERY:-5}}"
ENABLE_BENCHMARK_SYNC="${QUASAR_ENABLE_BENCHMARK_SYNC:-${DISTIL_ENABLE_BENCHMARK_SYNC:-1}}"

COUNT=0
while true; do
    "$SCRIPT_DIR/sync_api_state.sh"
    COUNT=$((COUNT + 1))
    if [ "$ENABLE_BENCHMARK_SYNC" = "1" ] && [ $((COUNT % BENCHMARK_EVERY)) -eq 0 ]; then
        "$SCRIPT_DIR/sync_benchmarks.sh" 2>/dev/null || true
    fi
    sleep "$SYNC_INTERVAL"
done
