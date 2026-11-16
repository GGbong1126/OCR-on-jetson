from importlib import import_module
from .layout_fusion import LayoutFusion, LayoutFusionConfig

__all__ = [
	"PaddlePicoDetector",
	"PaddlePicoDetectorConfig",
	"TableCellPostProcessor",
	"TableCellPostProcessorConfig",
	"LayoutFusion",
	"LayoutFusionConfig",
]

def __getattr__(name):
    """Keep ONNX-only fusion imports independent of optional Paddle installations."""
    modules = {
        "PaddlePicoDetector": ".paddlepico_detector",
        "PaddlePicoDetectorConfig": ".paddlepico_detector",
        "TableCellPostProcessor": ".table_cell_postprocessor",
        "TableCellPostProcessorConfig": ".table_cell_postprocessor",
        "LayoutAnalyzer": ".structure_analyzer",
        "LayoutAnalyzerConfig": ".structure_analyzer",
    }
    if name not in modules:
        raise AttributeError(name)
    try:
        value = getattr(import_module(modules[name], __name__), name)
    except ImportError:
        if name not in ("LayoutAnalyzer", "LayoutAnalyzerConfig"):
            raise
        value = None
    globals()[name] = value
    return value
