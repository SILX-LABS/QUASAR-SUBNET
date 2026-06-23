#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
PYTHON="${PYTHON:-}"

cd "${ROOT}"

if [[ -f scripts/install_system_deps.sh ]]; then
  bash scripts/install_system_deps.sh
fi

if [[ -z "${PYTHON}" ]]; then
  if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
    python3 -m venv "${ROOT}/.venv"
  fi
  PYTHON="${ROOT}/.venv/bin/python"
fi

if [[ -z "${TMPDIR:-}" && -d /workspace && -w /workspace ]]; then
  mkdir -p /workspace/buildtmp
  export TMPDIR=/workspace/buildtmp
elif [[ -z "${TMPDIR:-}" && -d /workspace/buildtmp && -w /workspace/buildtmp ]]; then
  export TMPDIR=/workspace/buildtmp
fi
PYTHON_BIN_DIR="$("${PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.executable).parent)')"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -e ".[orchestrator]"

quasar-incentive orchestrator --help >/dev/null
quasar-orchestrator --help >/dev/null
quasar-generate-ed25519-hotkey --help >/dev/null

if [[ "${QUASAR_PREPARE_MODEL_CODE:-1}" != "0" ]]; then
  "${PYTHON}" scripts/prepare_quasar_model_code.py \
    --model-id "${QUASAR_MODEL_SOURCE_ID:-silx-ai/Quasar-Preview}" \
    --revision "${QUASAR_MODEL_REVISION:-main}" \
    --output-dir "${QUASAR_MODEL_CODE_DIR:-${ROOT}/.quasar-model/Quasar-Preview}" \
    --train-deps-dir "${QUASAR_TRAIN_DEPS_DIR:-${ROOT}/.train-deps}"
fi

printf '%s\n' "Orchestrator runtime ready."
printf '%s\n' "Recommended runtime env: export PYTHONPATH=${QUASAR_TRAIN_DEPS_DIR:-${ROOT}/.train-deps}:${QUASAR_MODEL_CODE_DIR:-${ROOT}/.quasar-model/Quasar-Preview}:\${PYTHONPATH:-}"
printf '%s\n' "Recommended training model path: export QUASAR_TRAINING_MODEL_ID=${QUASAR_MODEL_CODE_DIR:-${ROOT}/.quasar-model/Quasar-Preview}"
printf '%s\n' "Use: quasar-incentive orchestrator run --run-id <run-id>"
