#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-}"

cd "${ROOT}"

if [[ -z "${PYTHON}" ]]; then
  if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
    python3 -m venv "${ROOT}/.venv"
  fi
  PYTHON="${ROOT}/.venv/bin/python"
fi

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -e ".[miner]"

quasar-incentive miner --help >/dev/null
quasar-miner --help >/dev/null
quasar-incentive-train --help >/dev/null

if [[ "${QUASAR_PREPARE_MODEL_CODE:-1}" != "0" ]]; then
  prepare_args=()
  if [[ "${QUASAR_INCLUDE_MODEL_WEIGHTS:-0}" == "1" ]]; then
    prepare_args+=(--include-weights)
  fi
  "${PYTHON}" scripts/prepare_quasar_model_code.py \
    --model-id "${QUASAR_MODEL_SOURCE_ID:-silx-ai/Quasar-Preview}" \
    --revision "${QUASAR_MODEL_REVISION:-main}" \
    --output-dir "${QUASAR_MODEL_CODE_DIR:-${ROOT}/.quasar-model/Quasar-Preview}" \
    --train-deps-dir "${QUASAR_TRAIN_DEPS_DIR:-${ROOT}/.train-deps}" \
    "${prepare_args[@]}"
fi

printf '%s\n' "Miner training runtime ready."
printf '%s\n' "Recommended runtime env: export PYTHONPATH=${QUASAR_TRAIN_DEPS_DIR:-${ROOT}/.train-deps}:${QUASAR_MODEL_CODE_DIR:-${ROOT}/.quasar-model/Quasar-Preview}:\${PYTHONPATH:-}"
printf '%s\n' "Recommended training model path: export QUASAR_TRAINING_MODEL_ID=${QUASAR_MODEL_CODE_DIR:-${ROOT}/.quasar-model/Quasar-Preview}"
printf '%s\n' "Use: quasar-incentive miner run --worker-id <worker-id> --owner-identity <orchestrator-hotkey>"
