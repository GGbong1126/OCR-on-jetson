#!/bin/bash
set -euo pipefail
root=/home/xchq/work/autoocr
package="$root/overlays/jetson-dev-e129-r1"
action="${1:---check}"
if [[ $# -gt 1 || ( "$action" != --check && "$action" != --serve ) ]]; then exit 64; fi
if [[ "${AUTOOCR_ENABLE_DEV_NO_AUTH:-}" != 1 ]]; then
  echo 'Explicit AUTOOCR_ENABLE_DEV_NO_AUTH=1 is required for this development entry.' >&2
  exit 78
fi
for name in ${!AUTOOCR_@}; do
  if [[ "$name" != AUTOOCR_ENABLE_DEV_NO_AUTH ]]; then
    echo "Unexpected inherited override: $name" >&2
    exit 78
  fi
done
if [[ -v GUNICORN_CMD_ARGS || "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" != "$package" ]]; then exit 78; fi
export AUTOOCR_SOURCE_ROOT="$root/releases/jetson-r2"
export AUTOOCR_RUNTIME_DIR="$root/runtime/jetson-dev-e129-r1"
export AUTOOCR_REC_WIDE_CONFIG="$root/overlays/rec-wide-e124-r1/config.json"
export AUTOOCR_LOG_FILE=/dev/null AUTOOCR_HOST=127.0.0.1 AUTOOCR_PORT_ONLINE=5051 AUTOOCR_USE_GPU=false
export LD_LIBRARY_PATH="$root/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONPATH PYTHONHOME
umask 077
cd "$package"
python="$root/envs/business-py310/bin/python"
"$python" serve_jetson_dev.py || exit 78
"$python" "$root/releases/jetson-r2/edge_deploy/business/service_health.py" --check-port || exit 78
if [[ "$action" == --check ]]; then exit 0; fi
exec "$root/envs/business-py310/bin/gunicorn" --config /dev/null --chdir "$package" \
  --bind 127.0.0.1:5051 --workers 1 --worker-class sync --threads 1 --timeout 600 \
  --graceful-timeout 60 --backlog 16 'serve_jetson_dev:create_app()'
