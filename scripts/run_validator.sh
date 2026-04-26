#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${QUASAR_ENV_FILE:-${DISTIL_ENV_FILE:-$HOME/.secrets/quasar.env}}"
PYTHON_BIN="${QUASAR_PYTHON:-${DISTIL_PYTHON:-}}"

if [ -z "$PYTHON_BIN" ]; then
    for candidate in "$REPO_ROOT/.venv/bin/python" "/opt/quasar/venv/bin/python" "$(command -v python3)"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi

cd "$REPO_ROOT"

if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || echo '')}"

EVAL_BACKEND="${QUASAR_EVAL_BACKEND:-${DISTIL_EVAL_BACKEND:-lium}}"
STATE_DIR="${QUASAR_STATE_DIR:-${DISTIL_STATE_DIR:-$REPO_ROOT/state}}"
NETWORK="${QUASAR_NETWORK:-${DISTIL_NETWORK:-finney}}"
NETUID="${QUASAR_NETUID:-${DISTIL_NETUID:-24}}"

ARGS=(
  scripts/remote_validator.py
  --network "$NETWORK"
  --netuid "$NETUID"
  --wallet-name "${QUASAR_WALLET_NAME:-${DISTIL_WALLET_NAME:-validator}}"
  --hotkey-name "${QUASAR_HOTKEY_NAME:-${DISTIL_HOTKEY_NAME:-validator}}"
  --wallet-path "${QUASAR_WALLET_PATH:-${DISTIL_WALLET_PATH:-$HOME/.bittensor/wallets}}"
  --state-dir "$STATE_DIR"
  --eval-backend "$EVAL_BACKEND"
  --tempo "${QUASAR_VALIDATOR_TEMPO:-${DISTIL_VALIDATOR_TEMPO:-600}}"
  --use-vllm
)

if [ "$EVAL_BACKEND" = "lium" ]; then
  if [ -z "${LIUM_API_KEY:-}" ]; then
    echo "LIUM_API_KEY is required when QUASAR_EVAL_BACKEND=lium" >&2
    exit 2
  fi
  ARGS+=(--lium-api-key "$LIUM_API_KEY")
  ARGS+=(--lium-pod-name "${QUASAR_LIUM_POD_NAME:-${DISTIL_LIUM_POD_NAME:-quasar-eval}}")
elif [ "$EVAL_BACKEND" = "local" ]; then
  LOCAL_EVAL_DIR="${QUASAR_LOCAL_EVAL_DIR:-${DISTIL_LOCAL_EVAL_DIR:-$STATE_DIR/local_eval_runs}}"
  ARGS+=(--local-eval-dir "$LOCAL_EVAL_DIR")
else
  echo "Unknown eval backend: $EVAL_BACKEND (expected lium or local)" >&2
  exit 2
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
