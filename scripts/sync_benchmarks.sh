#!/bin/bash
# Sync benchmark results from the eval host into local state/benchmarks/.
# Exits 0 even when the remote is unreachable so systemd does not log-spam.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EVAL_POD="${QUASAR_BENCHMARK_HOST:-${DISTIL_BENCHMARK_HOST:-root@localhost}}"
EVAL_PORT="${QUASAR_BENCHMARK_PORT:-${DISTIL_BENCHMARK_PORT:-6039}}"
REMOTE_DIR="${QUASAR_BENCHMARK_REMOTE_DIR:-${DISTIL_BENCHMARK_REMOTE_DIR:-/root/benchmark_results/}}"
LOCAL_DIR="${QUASAR_BENCHMARK_LOCAL_DIR:-${DISTIL_BENCHMARK_LOCAL_DIR:-$REPO_ROOT/state/benchmarks/}}"

mkdir -p "$LOCAL_DIR"

SYNC_ERR="$(mktemp)"
trap 'rm -f "$SYNC_ERR"' EXIT

SSH_CMD="ssh -p $EVAL_PORT -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no"

# Prefer rsync when the remote has it (incremental, fast). Fall back to
# ssh+tar streaming when rsync is missing — this is the case for freshly
# provisioned Lium eval pods where rsync isn't in the base image. Before
# this fallback the timer logged 'rsync: command not found' every cycle
# and the dashboard never picked up new benchmark_results/uid_*_summary.json
# files until someone manually apt-installed rsync on the pod.
has_remote_rsync=1
if ! $SSH_CMD "$EVAL_POD" "command -v rsync >/dev/null 2>&1" 2>/dev/null; then
    has_remote_rsync=0
fi

if [ "$has_remote_rsync" = "1" ] && rsync -az --timeout=10 \
      -e "$SSH_CMD" \
      --include='uid_*_summary.json' \
      --exclude='*' \
      "$EVAL_POD:$REMOTE_DIR" "$LOCAL_DIR" 2>"$SYNC_ERR"; then
    :
else
    # Fallback: stream the summary files over ssh as a plain tar archive.
    # We limit to ``uid_*_summary.json`` to avoid pulling the full
    # per-prompt prediction logs (each benchmark run can produce 1GB+ of
    # JSONL, and we only need the aggregated scores).
    REMOTE_LIST_CMD="cd \"$REMOTE_DIR\" 2>/dev/null && ls uid_*_summary.json 2>/dev/null | tr '\n' '\0'"
    if files_nul="$($SSH_CMD "$EVAL_POD" "$REMOTE_LIST_CMD" 2>"$SYNC_ERR")"; then
        if [ -n "$files_nul" ]; then
            # xargs -0 to survive any weird filenames; tar c | ssh | tar x
            # over stdin gives us the full-atomic copy semantics rsync had.
            if ! $SSH_CMD "$EVAL_POD" \
                    "cd \"$REMOTE_DIR\" && printf '%s' \"$files_nul\" | xargs -0 tar -cf -" \
                    2>>"$SYNC_ERR" | tar -xf - -C "$LOCAL_DIR" 2>>"$SYNC_ERR"; then
                REASON="$(tr '\n' ' ' < "$SYNC_ERR" | sed 's/  */ /g' | head -c 200)"
                echo "benchmark-sync: tar fallback failed (${REASON:-unknown}); skipping this cycle" >&2
                exit 0
            fi
        fi
    else
        REASON="$(tr '\n' ' ' < "$SYNC_ERR" | sed 's/  */ /g' | head -c 200)"
        echo "benchmark-sync: remote unreachable (${REASON:-unknown}); skipping this cycle" >&2
        exit 0
    fi
fi

NEW=$(find "$LOCAL_DIR" -name 'uid_*_summary.json' -newer "$LOCAL_DIR/.last_sync" 2>/dev/null | wc -l)
touch "$LOCAL_DIR/.last_sync"
if [ "$NEW" -gt 0 ]; then
    echo "Synced $NEW new benchmark summaries"
fi
