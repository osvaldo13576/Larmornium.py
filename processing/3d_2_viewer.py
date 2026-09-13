#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3d_2_viewer.py - Recurso de la GUI para transformar y cargar archivos 3D
(GLB, OBJ, STL) a formato visualizable en la GUI de Larmornium.

Carga mallas 3D en formato nativo preservando las dimensiones milimétricas,
geometría, coordenadas y atributos originales sin alteraciones artificiales.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("larmornium.3d_2_viewer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

try:
    import vtk
    from vtkmodules.vtkIOGeometry import vtkSTLReader, vtkOBJReader
    from vtkmodules.vtkIOGeometry import vtkGLTFReader
    from vtkmodules.vtkFiltersGeometry import vtkCompositeDataGeometryFilter
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals, vtkCleanPolyData
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
    VTK_AVAILABLE = True
except Exception as e:
    try:
        import vtk
        VTK_AVAILABLE = True
    except Exception:
        VTK_AVAILABLE = False
        logger.warning("VTK no disponible en el entorno actual.")

SUPPORTED_EXTENSIONS = (".stl", ".obj", ".glb", ".gltf")


def is_supported_3d_file(file_path: str) -> bool:
    """Verifica si la ruta corresponde a un archivo 3D con extensión soportada."""
    if not file_path or not isinstance(file_path, str):
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def load_stl(file_path: str) -> Any:
    """Carga un archivo STL utilizando vtkSTLReader en formato nativo."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo STL no encontrado: {file_path}")
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"No se pudieron leer puntos válidos del archivo STL: {file_path}")
    return polydata


def load_obj(file_path: str) -> Any:
    """Carga un archivo OBJ utilizando vtkOBJReader en formato nativo."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo OBJ no encontrado: {file_path}")
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"No se pudieron leer puntos válidos del archivo OBJ: {file_path}")
    return polydata


def load_glb(file_path: str) -> Any:
    """
    Carga un archivo GLB o glTF utilizando vtkGLTFReader y consolida
    bloques jerárquicos en vtkPolyData en formato nativo.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo GLB/glTF no encontrado: {file_path}")
    reader = vtk.vtkGLTFReader()
    reader.SetFileName(file_path)
    reader.Update()

    output = reader.GetOutput()
    if output is None:
        raise ValueError(f"vtkGLTFReader no generó salida para: {file_path}")

    # Si es un conjunto multbloque, consolidar a PolyData único
    if hasattr(output, "GetNumberOfBlocks") or not hasattr(output, "GetPolys"):
        geom_filter = vtk.vtkCompositeDataGeometryFilter()
        geom_filter.SetInputConnection(reader.GetOutputPort())
        geom_filter.Update()
        polydata = geom_filter.GetOutput()
    else:
        polydata = output

    if polydata is None or polydata.GetNumberOfPoints() == 0:
        raise ValueError(f"No se pudieron leer geometrías del archivo GLB: {file_path}")

    return polydata


def get_centered_polydata(polydata: Any) -> Tuple[Any, Tuple[float, float, float]]:
    """
    Retorna una copia centrada en el origen (0, 0, 0) para permitir que la órbita
    y rotación de cámara en el visor giren con respecto al centro propio del objeto.
    También retorna el centroide original para conservar las mediciones milimétricas nativas.
    """
    if polydata is None:
        return None, (0.0, 0.0, 0.0)

    center = polydata.GetCenter()
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

    transform = vtk.vtkTransform()
    transform.Translate(-cx, -cy, -cz)

    tfilter = vtk.vtkTransformPolyDataFilter()
    tfilter.SetInputData(polydata)
    tfilter.SetTransform(transform)
    tfilter.Update()

    return tfilter.GetOutput(), (cx, cy, cz)


def load_3d_file(file_path: str, center_geometry: bool = False, compute_normals: bool = True) -> Dict[str, Any]:
    """
    Carga un archivo 3D (.stl, .obj, .glb, .gltf) a vtkPolyData en su formato nativo.

    Args:
        file_path: Ruta al archivo 3D.
        center_geometry: Si es True, centra la malla en el origen (0,0,0) para visualización.
        compute_normals: Si es True y la malla carece de normales, calcula normales continuas.

    Returns:
        Diccionario con metadatos y polydata.
    """
    if not VTK_AVAILABLE:
        raise RuntimeError("VTK no se encuentra disponible para cargar archivos 3D.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo 3D no encontrado: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".stl":
        raw_poly = load_stl(file_path)
    elif ext == ".obj":
        raw_poly = load_obj(file_path)
    elif ext in (".glb", ".gltf"):
        raw_poly = load_glb(file_path)
    else:
        raise ValueError(f"Formato no soportado '{ext}'. Extensiones admitidas: {SUPPORTED_EXTENSIONS}")

    # Limpiar vértices duplicados o degenerados en formato nativo
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(raw_poly)
    cleaner.Update()
    polydata = cleaner.GetOutput()

    # Calcular normales si se requiere para sombreado suave
    if compute_normals:
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(polydata)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.Update()
        polydata = normals.GetOutput()

    # Dimensiones y centro nativos originales (en mm)
    orig_bounds = polydata.GetBounds()
    orig_center = polydata.GetCenter()
    dx = float(orig_bounds[1] - orig_bounds[0])
    dy = float(orig_bounds[3] - orig_bounds[2])
    dz = float(orig_bounds[5] - orig_bounds[4])

    display_poly = polydata
    if center_geometry:
        display_poly, _ = get_centered_polydata(polydata)

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)

    result = {
        "file_path": os.path.abspath(file_path),
        "file_name": os.path.basename(file_path),
        "format": ext.lstrip(".").upper(),
        "polydata": display_poly,
        "native_polydata": polydata,
        "num_points": int(polydata.GetNumberOfPoints()),
        "num_cells": int(polydata.GetNumberOfCells()),
        "bounds": tuple(float(b) for b in orig_bounds),
        "center": tuple(float(c) for c in orig_center),
        "dimensions_mm": (dx, dy, dz),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
    }
    return result


def transform_3d_for_viewer(file_path: str, center_geometry: bool = True) -> Dict[str, Any]:
    """
    Función de recurso principal para la GUI de Larmornium: transforma el archivo 3D
    al formato vtkPolyData listo para ser proyectado en el lienzo de visualización.
    """
    return load_3d_file(file_path, center_geometry=center_geometry, compute_normals=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 3d_2_viewer.py <archivo.stl | archivo.obj | archivo.glb>")
        sys.exit(1)

    target_file = sys.argv[1]
    try:
        data = transform_3d_for_viewer(target_file)
        print(f"--- Información de Archivo 3D (Formato Nativo) ---")
        print(f"Archivo: {data['file_name']}")
        print(f"Formato: {data['format']}")
        print(f"Tamaño: {data['file_size_mb']:.2f} MB")
        print(f"Vértices: {data['num_points']}")
        print(f"Caras / Celdas: {data['num_cells']}")
        print(f"Dimensiones (mm): X={data['dimensions_mm'][0]:.2f}, Y={data['dimensions_mm'][1]:.2f}, Z={data['dimensions_mm'][2]:.2f}")
        print(f"Límites nativos: {data['bounds']}")
        print(f"Centro nativo: {data['center']}")
        print(f"Transformación completada con éxito.")
    except Exception as err:
        print(f"Error al procesar archivo 3D: {err}")
        sys.exit(1)
