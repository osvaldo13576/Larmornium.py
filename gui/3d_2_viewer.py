#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui/3d_2_viewer.py - Recurso de la GUI para visualización 3D.
Expone las funciones de transformación de processing/3d_2_viewer.py.
"""

import os
import sys

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
_PROCESSING_DIR = os.path.join(_PROJECT_ROOT, "processing")
if _PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _PROCESSING_DIR)

import importlib
_proc_mod = importlib.import_module("3d_2_viewer")

is_supported_3d_file = _proc_mod.is_supported_3d_file
load_stl = _proc_mod.load_stl
load_obj = _proc_mod.load_obj
load_glb = _proc_mod.load_glb
get_centered_polydata = _proc_mod.get_centered_polydata
load_3d_file = _proc_mod.load_3d_file
transform_3d_for_viewer = _proc_mod.transform_3d_for_viewer
SUPPORTED_EXTENSIONS = _proc_mod.SUPPORTED_EXTENSIONS

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = transform_3d_for_viewer(sys.argv[1])
        print(f"Archivo 3D cargado: {data['file_name']} ({data['format']}, {data['num_points']} vértices)")
