"""Restore the original module name referenced by best1.pt, without editing site-packages."""
import importlib
import sys


def ensure_yolo_custom_layers():
    name = 'ultralytics.nn.bifpn'
    try:
        # Preserve the original Windows environment if its custom module exists.
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
    parent = importlib.import_module('ultralytics.nn')
    module = importlib.import_module('layout.yolo_layers.bifpn')
    sys.modules[name] = module
    setattr(parent, 'bifpn', module)
    return module
