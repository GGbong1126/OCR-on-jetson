#!/bin/bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=/home/xchq/work/autoocr
slots=$("$root/envs/business-py310/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); n=r["selected_rec_threads"]; assert n in (1,2,4); print(n)' "$here/selection.json")
export AUTOOCR_REC_THREADS=1 AUTOOCR_REC_SLOTS="$slots"
exec bash "$here/app/start_batch.sh"
