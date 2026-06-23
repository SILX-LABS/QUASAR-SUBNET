#!/usr/bin/env bash
set -euo pipefail

if [[ "${QUASAR_INSTALL_SYSTEM_DEPS:-1}" == "0" ]]; then
  exit 0
fi

if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" == "0" ]]; then
  export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
  apt-get update
  apt-get install -y \
    build-essential \
    git \
    libssl-dev \
    pkg-config \
    python3-venv \
    python3.12-venv
  exit 0
fi

missing=()
command -v git >/dev/null 2>&1 || missing+=(git)
command -v pkg-config >/dev/null 2>&1 || missing+=(pkg-config)
if ! python3 - <<'PY' >/dev/null 2>&1
import tempfile
import venv
with tempfile.TemporaryDirectory() as path:
    venv.EnvBuilder(with_pip=True).create(path)
PY
then
  missing+=(python3-venv)
fi

if (( ${#missing[@]} )); then
  printf 'Missing system dependencies: %s\n' "${missing[*]}" >&2
  printf 'Install them first, or rerun as root on an apt-based machine with QUASAR_INSTALL_SYSTEM_DEPS=1.\n' >&2
  exit 2
fi
