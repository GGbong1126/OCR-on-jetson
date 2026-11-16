#!/usr/bin/env bash
set -euo pipefail
source_root="${AUTOOCR_SOURCE_ROOT:-/mnt/d/docuyyy/autoocr}"
business_root="$HOME/projects/autoocr-business"
if [[ -e "$business_root/.venv" ]]; then
  echo 'Business environment already exists; inspect it before reinstalling.' >&2
  exit 1
fi
mkdir -p "$business_root/runtime"
python3 -m venv "$business_root/.venv"
python="$business_root/.venv/bin/python"
"$python" -m pip install --upgrade pip
"$python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
"$python" -m pip install -r "$source_root/edge_deploy/business/requirements.txt"
"$python" -m pip check
"$python" -m pip freeze > "$business_root/runtime/requirements-resolved.txt"
echo 'Installed. Activate for this WSL device before starting; the existing edge environment is untouched.'
