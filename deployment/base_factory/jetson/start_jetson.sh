#!/bin/bash
# Audited board only. No system installs, automatic activation, or count resets.
set -euo pipefail
root=/home/xchq/work/autoocr
# Reject inherited alternate model/license locations before setting our roots.
for name in AUTOOCR_BASE_DIR AUTOOCR_LICENSE_FILE AUTOOCR_LOG_FILE AUTOOCR_TEMP_DIR \
 AUTOOCR_LAYOUT_MODEL AUTOOCR_TABLE_CELL_MODEL AUTOOCR_OCR_MODEL AUTOOCR_NER_MODEL \
 AUTOOCR_IMGPRE_MODEL AUTOOCR_ONNX_LAYOUT_MODEL AUTOOCR_ONNX_CELL_MODEL \
 AUTOOCR_ONNX_OCR_DET_MODEL AUTOOCR_ONNX_OCR_REC_MODEL AUTOOCR_FORCE_PADDLE \
 AUTOOCR_YOLO_CONF AUTOOCR_NER_THRESHOLD; do
  if [[ -v "$name" ]]; then
    echo "Unexpected override: $name" >&2
    exit 1
  fi
done
release="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
case "$release" in
  "$root/releases/jetson-r1"|"$root/releases/jetson-r2") ;;
  *) echo "Unexpected release directory" >&2; exit 1 ;;
esac
export AUTOOCR_SOURCE_ROOT="$release"
export AUTOOCR_RUNTIME_DIR="$root/runtime"
export LD_LIBRARY_PATH="$root/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export AUTOOCR_HOST=127.0.0.1 AUTOOCR_PORT_ONLINE=5051 AUTOOCR_USE_GPU=false
unset PYTHONPATH PYTHONHOME
umask 077
cd "$AUTOOCR_SOURCE_ROOT/edge_deploy/jetson"
python="$root/envs/business-py310/bin/python"
"$python" serve_jetson.py --check-license
"$python" ../business/service_health.py --check-port
exec "$root/envs/business-py310/bin/gunicorn" --bind 127.0.0.1:5051 \
    --workers 1 --worker-class sync --threads 1 --timeout 600 \
    --graceful-timeout 60 --backlog 16 'serve_jetson:create_app()'
