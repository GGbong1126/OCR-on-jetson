#!/usr/bin/env bash
# Explicit local release entry; does not install, activate, stop or switch services.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
: "${AUTOOCR_RUNTIME_DIR:?Set the existing machine runtime directory explicitly}"
: "${AUTOOCR_BUSINESS_VENV:?Set the Linux business venv explicitly}"
[[ "$AUTOOCR_RUNTIME_DIR" = /* && "$AUTOOCR_BUSINESS_VENV" = /* ]]
runtime="$(cd -- "$AUTOOCR_RUNTIME_DIR" && pwd -P)"
venv="$(cd -- "$AUTOOCR_BUSINESS_VENV" && pwd -P)"
for directory in "$runtime" "$venv"; do
  if [[ "$directory" = "$root" || "$directory" = "$root/"* ]]; then
    echo 'Runtime and venv must stay outside the release.' >&2
    exit 1
  fi
done
# Refuse model/license path overrides that could silently escape the verified payload/runtime.
for name in AUTOOCR_BASE_DIR AUTOOCR_LICENSE_FILE AUTOOCR_LOG_FILE AUTOOCR_TEMP_DIR \
  AUTOOCR_LAYOUT_MODEL AUTOOCR_TABLE_CELL_MODEL AUTOOCR_OCR_MODEL AUTOOCR_NER_MODEL \
  AUTOOCR_IMGPRE_MODEL AUTOOCR_ONNX_LAYOUT_MODEL AUTOOCR_ONNX_CELL_MODEL \
  AUTOOCR_ONNX_OCR_DET_MODEL AUTOOCR_ONNX_OCR_REC_MODEL AUTOOCR_FORCE_PADDLE \
  AUTOOCR_YOLO_CONF AUTOOCR_NER_THRESHOLD; do
  if [[ -v "$name" ]]; then
    echo "Unexpected override: $name; use a clean deployment environment." >&2
    exit 1
  fi
done
export AUTOOCR_SOURCE_ROOT="$root"
export AUTOOCR_RUNTIME_DIR="$runtime"
export AUTOOCR_BUSINESS_VENV="$venv"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
umask 077
"$venv/bin/python" "$root/edge_deploy/business/release_business.py" verify --root "$root"
"$venv/bin/python" "$root/edge_deploy/business/service_health.py" --check-port
exec /bin/bash "$root/edge_deploy/business/start.sh"
