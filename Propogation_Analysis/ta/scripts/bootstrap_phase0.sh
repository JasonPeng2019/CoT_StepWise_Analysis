#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
LOCKFILE="${ROOT_DIR}/requirements.lock.txt"

cd "${ROOT_DIR}"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  'accelerate>=1.7,<2' \
  'numpy>=1.26,<2' \
  'pyarrow>=16,<18' \
  'pyyaml>=6,<7' \
  'safetensors>=0.4,<1' \
  'scikit-learn>=1.4,<2' \
  'scipy>=1.11,<2' \
  'torch>=2.7,<3' \
  'transformers>=4.51,<5'

python -m pip freeze | sort > "${LOCKFILE}"

echo "Bootstrap complete."
echo "Virtualenv: ${VENV_DIR}"
echo "Lockfile: ${LOCKFILE}"
