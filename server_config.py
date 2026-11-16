# -*- coding: utf-8 -*-
"""OCR 自动著录服务 — 配置文件（打包专用版本）。

检测 PyInstaller 打包环境，自动调整模型路径。
"""
import os
import sys

# 检测是否在 PyInstaller 打包环境中运行
if getattr(sys, '_MEIPASS', None):
    # 打包环境：优先使用 exe 所在目录（pth 通常放在 exe 旁边）
    _exe_pth = os.path.join(os.path.dirname(sys.executable), "pth")
    _meipass_pth = os.path.join(sys._MEIPASS, "pth")
    if os.path.isdir(_exe_pth):
        _BASE_DIR = os.path.dirname(sys.executable)
    elif os.path.isdir(_meipass_pth):
        _BASE_DIR = sys._MEIPASS
    else:
        _BASE_DIR = os.path.dirname(sys.executable)  # 默认使用 exe 目录
else:
    # 开发环境：模型路径相对于当前工作目录
    _BASE_DIR = os.path.abspath('.')

# Explicit Linux deployment root; unchanged defaults for existing source/EXE users.
_BASE_DIR = os.path.abspath(os.environ.get("AUTOOCR_BASE_DIR", _BASE_DIR))

# ============== 服务配置 ==============
HOST = os.environ.get("AUTOOCR_HOST", "0.0.0.0")
PORT_ONLINE = int(os.environ.get("AUTOOCR_PORT_ONLINE", "5051"))  # 联机版端口

# ============== 模型路径 ==============
# 版式分析模型目录（含 PP-DocLayoutV3 权重 + det_photo/best1.pt）
LAYOUT_MODEL_DIR = os.environ.get("AUTOOCR_LAYOUT_MODEL",
    os.path.join(_BASE_DIR, "pth", "rec", "Det", "Det"))

# 单元格检测模型目录（含 RT-DETR-L 权重）
TABLE_CELL_MODEL_DIR = os.environ.get("AUTOOCR_TABLE_CELL_MODEL",
    os.path.join(_BASE_DIR, "pth", "rec", "Cell", "Cell"))

# OCR 模型目录（可留空使用默认 PP-OCRv5_mobile）
OCR_MODEL_DIR = os.environ.get("AUTOOCR_OCR_MODEL",
    os.path.join(_BASE_DIR, "pth", "ocr", "server"))

# NER 模型目录（transformers token-classification 模型）
NER_MODEL_DIR = os.environ.get("AUTOOCR_NER_MODEL",
    os.path.join(_BASE_DIR, "pth", "ner", "ner_model_cadre_docs_continuous_v3", "ner_model_cadre_docs_continuous_v3"))

# 图像预处理模型目录（含 .pth 文件）
IMGPRE_MODEL_DIR = os.environ.get("AUTOOCR_IMGPRE_MODEL",
    os.path.join(_BASE_DIR, "pth", "imgpre"))

# ONNX 模型路径（优先使用，不存在则回退到 Paddle 模型）
ONNX_LAYOUT_MODEL = os.environ.get("AUTOOCR_ONNX_LAYOUT_MODEL",
    os.path.join(_BASE_DIR, "pth", "onnx_rec", "layout_det.onnx"))
ONNX_CELL_MODEL = os.environ.get("AUTOOCR_ONNX_CELL_MODEL",
    os.path.join(_BASE_DIR, "pth", "onnx_rec", "cell_det.onnx"))
ONNX_OCR_DET_MODEL = os.environ.get("AUTOOCR_ONNX_OCR_DET_MODEL",
    os.path.join(_BASE_DIR, "pth", "onnx_rec", "ocr_det.onnx"))
ONNX_OCR_REC_MODEL = os.environ.get("AUTOOCR_ONNX_OCR_REC_MODEL",
    os.path.join(_BASE_DIR, "pth", "onnx_rec", "ocr_rec.onnx"))

# ============== 运行参数 ==============
# 是否使用 GPU（自动检测：如果 paddle 和 torch 都支持 CUDA 则用 GPU）
USE_GPU = os.environ.get("AUTOOCR_USE_GPU", "auto")  # "auto", "true", "false"

# YOLO 置信度阈值
YOLO_CONF = float(os.environ.get("AUTOOCR_YOLO_CONF", "0.3"))

# 临时文件目录（base64 输入时使用）
TEMP_DIR = os.environ.get("AUTOOCR_TEMP_DIR", "./temp")

# NER 后处理置信度阈值
NER_SCORE_THRESHOLD = float(os.environ.get("AUTOOCR_NER_THRESHOLD", "0.98"))
