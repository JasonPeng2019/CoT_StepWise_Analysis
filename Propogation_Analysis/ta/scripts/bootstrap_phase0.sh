#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LOCKFILE="${ROOT_DIR}/requirements.lock.txt"

cd "${ROOT_DIR}"

python3 -m venv --system-site-packages "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  'accelerate>=1.7,<2' \
  'pyarrow>=16,<18' \
  'safetensors>=0.4,<1' \
  'transformers>=4.51,<5'

python -m pip freeze | sort > "${LOCKFILE}"

echo "Bootstrap complete."
echo "Virtualenv: ${VENV_DIR}"
echo "Lockfile: ${LOCKFILE}"
