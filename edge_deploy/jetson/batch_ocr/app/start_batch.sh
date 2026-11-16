#!/bin/bash
set -euo pipefail
root=/home/xchq/work/autoocr
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export AUTOOCR_ENABLE_DEV_NO_AUTH=1
export AUTOOCR_BATCH_DET="${AUTOOCR_BATCH_DET-1}"
export AUTOOCR_BATCH_JPEG="${AUTOOCR_BATCH_JPEG-1}"
export LD_LIBRARY_PATH="$root/lib/cuda10.2:/usr/local/cuda-10.2/targets/aarch64-linux/lib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME GUNICORN_CMD_ARGS
umask 077
cd "$here"
exec "$root/envs/business-py310/bin/gunicorn" --config /dev/null --chdir "$here" \
 --bind 127.0.0.1:5052 --workers 1 --worker-class gthread --threads 4 --timeout 600 \
 --worker-connections 8 --graceful-timeout 40 --backlog 4 --access-logfile /dev/null 'serve_http:create_app()'
