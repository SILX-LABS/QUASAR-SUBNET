#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python3}"

cd "${ROOT}"

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install -e ".[validator]"

if [[ "${QUASAR_RUN_TESTS:-0}" == "1" ]]; then
  "${PYTHON}" -m pip install -e ".[dev]"
  "${PYTHON}" -m pytest -q tests/test_incentive_protocol.py
fi

cat <<EOF
Validator runtime ready.

Use:
  quasar-validator --help
  quasar-incentive --help
EOF
