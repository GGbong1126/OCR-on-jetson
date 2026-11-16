"""Frozen E168 paths; all outputs remain in the project's new staging directory."""
from pathlib import Path
import sys

B=Path('/home/xchq/work/autoocr')
S=B/'staging/http_rec_trace_e168_r2'
SOURCE=B/'overlays/batch-ocr-threaded-e167-r1'
PY=B/'envs/business-py310/bin/python'
TOOL=B/'staging/multistream_e164_r1/profiler/extracted/usr/local/cuda-10.2'
sys.path.insert(0,str(B/'staging/threaded_e167_r1'))
import controller as e167
