#!/usr/bin/env bash
set -euo pipefail
export AUTOOCR_SOURCE_ROOT="${AUTOOCR_SOURCE_ROOT:-/mnt/d/docuyyy/autoocr}"
export AUTOOCR_RUNTIME_DIR="${AUTOOCR_RUNTIME_DIR:-$HOME/projects/autoocr-business/runtime}"
venv="${AUTOOCR_BUSINESS_VENV:-$HOME/projects/autoocr-business/.venv}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_AUTOINSTALL=false
export AUTOOCR_USE_GPU=false
if [[ "${AUTOOCR_OTHER_BACKEND:-cpu}" = cuda ]]; then
  export AUTOOCR_USE_GPU=true
fi
export AUTOOCR_HOST=127.0.0.1
export AUTOOCR_PORT_ONLINE=5051
cd "$AUTOOCR_SOURCE_ROOT/edge_deploy/business"
"$venv/bin/python" serve_business.py --check-license
# Initialize once inside one sync worker. Do not add --preload, threads or workers.
exec "$venv/bin/gunicorn" \
  --bind 127.0.0.1:5051 --workers 1 --worker-class sync --threads 1 \
  --timeout 600 --graceful-timeout 60 --backlog 16 \
  'serve_business:create_app()'
