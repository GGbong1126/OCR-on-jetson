#!/bin/bash
# E127 opt-in delivery. No activation, license override or automatic promotion.
set -euo pipefail
root=/home/xchq/work/autoocr
package="$root/overlays/rec-wide-service-e127-r1"
overlay="$root/overlays/rec-wide-e124-r1"
release="$root/releases/jetson-r2"
action="${1:---check}"
if [[ $# -gt 1 || ( "$action" != --check && "$action" != --serve ) ]]; then
  echo 'Use --check or --serve; activation is not provided.' >&2
  exit 64
fi
mode="${AUTOOCR_REC_WIDE_MODE:-on}"
if [[ "$mode" != on && "$mode" != off ]]; then
  echo 'Invalid AUTOOCR_REC_WIDE_MODE; use on or off.' >&2
  exit 78
fi
for name in AUTOOCR_SOURCE_ROOT AUTOOCR_RUNTIME_DIR AUTOOCR_BASE_DIR AUTOOCR_LICENSE_FILE \
 AUTOOCR_LOG_FILE AUTOOCR_TEMP_DIR AUTOOCR_LAYOUT_MODEL AUTOOCR_TABLE_CELL_MODEL \
 AUTOOCR_OCR_MODEL AUTOOCR_NER_MODEL AUTOOCR_IMGPRE_MODEL AUTOOCR_ONNX_LAYOUT_MODEL \
 AUTOOCR_ONNX_CELL_MODEL AUTOOCR_ONNX_OCR_DET_MODEL AUTOOCR_ONNX_OCR_REC_MODEL \
 AUTOOCR_FORCE_PADDLE AUTOOCR_YOLO_CONF AUTOOCR_NER_THRESHOLD AUTOOCR_REC_WIDE_CONFIG \
 AUTOOCR_HOST AUTOOCR_PORT_ONLINE AUTOOCR_USE_GPU GUNICORN_CMD_ARGS; do
  if [[ -v "$name" ]]; then
    echo "Unexpected inherited override: $name" >&2
    exit 78
  fi
done
if [[ "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" != "$package" ]]; then
  echo 'Unexpected service package directory.' >&2
  exit 78
fi
# Fail before importing the business module or starting a listener. Do not read,
# copy, generate, migrate or initialize licensing/counter state in this branch.
if [[ ! -f "$root/runtime/license.dat" ]]; then
  echo 'Existing Jetson license is absent; no models, counters or HTTP started.' >&2
  exit 78
fi
export AUTOOCR_SOURCE_ROOT="$release" AUTOOCR_RUNTIME_DIR="$root/runtime"
export LD_LIBRARY_PATH="$root/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export AUTOOCR_HOST=127.0.0.1 AUTOOCR_PORT_ONLINE=5051 AUTOOCR_USE_GPU=false
if [[ "$mode" == on ]]; then
  export AUTOOCR_REC_WIDE_CONFIG="$overlay/config.json"
else
  unset AUTOOCR_REC_WIDE_CONFIG
fi
unset PYTHONPATH PYTHONHOME
umask 077
cd "$overlay"
python="$root/envs/business-py310/bin/python"
# Exact original validation, never a truthy replacement or borrowed license.
"$python" "$release/edge_deploy/jetson/serve_jetson.py" --check-license || exit 78
"$python" -c 'import serve_rec_wide; serve_rec_wide.selected_config()' || exit 78
"$python" "$release/edge_deploy/business/service_health.py" --check-port || exit 78
if [[ "$action" == --check ]]; then
  echo 'Original license, selected package and loopback port preflight passed; no HTTP started.'
  exit 0
fi
exec "$root/envs/business-py310/bin/gunicorn" --config /dev/null --chdir "$overlay" \
  --bind 127.0.0.1:5051 --workers 1 --worker-class sync --threads 1 --timeout 600 \
  --graceful-timeout 60 --backlog 16 'serve_rec_wide:create_app()'
