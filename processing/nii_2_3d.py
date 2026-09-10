#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import struct
import time

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import marching_cubes

logger = logging.getLogger("nii_2_3d")

FORMATOS_VALIDOS = ("stl", "obj", "glb")


def _load_segmentation_volume(volume_input, json_path=None, voxel_spacing=None):
    """Carga un volumen de segmentacion binario desde NIfTI o ndarray.

    Retorna (volume, spacing) donde volume es un ndarray 3D de uint8
    y spacing es una lista [sx, sy, sz] en mm.
    """
    if isinstance(volume_input, str):
        if not os.path.exists(volume_input):
            raise FileNotFoundError(f"No se encontro el archivo: {volume_input}")
        nii = nib.load(volume_input)
        volume = nii.get_fdata()
        if volume.ndim == 4:
            volume = volume[:, :, :, 0]
        volume = (volume > 0).astype(np.uint8)

        if voxel_spacing is not None:
            spacing = [float(v) for v in voxel_spacing[:3]]
        elif json_path and os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            spacing = [float(v) for v in meta.get("voxel_spacing_mm", [1.0, 1.0, 1.0])[:3]]
        else:
            spacing = [float(v) for v in nii.header.get_zooms()[:3]]

    elif isinstance(volume_input, np.ndarray):
        if volume_input.ndim == 4:
            volume_input = volume_input[:, :, :, 0]
        if volume_input.ndim != 3:
            raise ValueError(f"El volumen debe ser 3D, se recibio {volume_input.ndim}D")
        volume = (volume_input > 0).astype(np.uint8)

        if voxel_spacing is not None:
            spacing = [float(v) for v in voxel_spacing[:3]]
        else:
            raise ValueError("Para volumenes crudos (ndarray) se requiere voxel_spacing")
    else:
        raise ValueError(f"Tipo de entrada no soportado: {type(volume_input)}")

    for i in range(3):
        if spacing[i] <= 0:
            spacing[i] = 1.0

    num_voxels = int(np.sum(volume))
    if num_voxels == 0:
        raise ValueError("El volumen de segmentacion esta vacio (no contiene voxeles activos)")

    logger.info("Volumen cargado: forma=%s, spacing=%s mm, voxeles activos=%d",
                list(volume.shape), [round(s, 4) for s in spacing], num_voxels)
    return volume, spacing


def _generate_mesh(volume, spacing, smooth_sigma=0.5):
    """Genera una malla triangular a partir del volumen binario usando marching cubes.

    Retorna (vertices, faces) donde vertices tiene coordenadas en unidades
    de voxel escaladas por spacing.
    """
    if smooth_sigma > 0:
        volume_smooth = ndi.gaussian_filter(volume.astype(np.float32), sigma=smooth_sigma)
    else:
        volume_smooth = volume.astype(np.float32)

    verts, faces, _, _ = marching_cubes(volume_smooth, level=0.5, spacing=spacing)

    logger.info("Malla generada: %d vertices, %d triangulos", len(verts), len(faces))
    return verts, faces


def _decimate_mesh(vertices, faces, quality=1.0):
    """Reduce la cantidad de triangulos de la malla segun el parametro de calidad.

    Parametros:
        vertices: ndarray (N, 3) con coordenadas de vertices
        faces: ndarray (M, 3) con indices de triangulos
        quality: valor entre 0.0 (minima calidad / maxima reduccion) y 1.0 (calidad original)

    Retorna:
        (vertices, faces) decimados
    """
    if quality >= 1.0:
        return vertices, faces

    target_reduction = min(0.98, max(0.0, 1.0 - float(quality)))
    if target_reduction <= 0.0:
        return vertices, faces

    try:
        import vtk
        from vtkmodules.util import numpy_support
    except ImportError:
        logger.warning("VTK no disponible para decimado de malla. Se conserva la calidad original.")
        return vertices, faces

    try:
        num_faces = len(faces)
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        pts.SetData(numpy_support.numpy_to_vtk(vertices.astype(np.float32), deep=True))
        poly.SetPoints(pts)

        ca = vtk.vtkCellArray()
        offsets = np.arange(0, (num_faces + 1) * 3, 3, dtype=np.int64)
        conn = faces.ravel().astype(np.int64)
        ca.SetData(
            numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True),
            numpy_support.numpy_to_vtkIdTypeArray(conn, deep=True)
        )
        poly.SetPolys(ca)

        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(poly)
        decimate.SetTargetReduction(target_reduction)
        decimate.Update()

        out_poly = decimate.GetOutput()
        out_pts = numpy_support.vtk_to_numpy(out_poly.GetPoints().GetData())

        polys = out_poly.GetPolys()
        if hasattr(polys, "GetConnectivityArray") and polys.GetConnectivityArray() is not None:
            conn_out = numpy_support.vtk_to_numpy(polys.GetConnectivityArray())
            out_faces = conn_out.reshape(-1, 3).astype(np.int64)
        else:
            arr = numpy_support.vtk_to_numpy(polys.GetData())
            out_faces = arr.reshape(-1, 4)[:, 1:].astype(np.int64)

        if len(out_faces) > 0 and len(out_pts) > 0:
            logger.info(
                "Malla decimada (calidad %.2f): %d -> %d triangulos (%.1f%%), %d vertices",
                quality, num_faces, len(out_faces),
                (len(out_faces) / num_faces * 100.0) if num_faces > 0 else 0.0,
                len(out_pts)
            )
            return out_pts, out_faces
        return vertices, faces
    except Exception as exc:
        logger.warning("Error durante el decimado de malla: %s. Conservando malla original.", exc)
        return vertices, faces


def _apply_scale(vertices, scale_factor):
    """Aplica un factor de escala uniforme a los vertices de la malla."""
    if scale_factor != 1.0:
        vertices = vertices * scale_factor
        logger.info("Escala aplicada: factor=%.4f", scale_factor)
    return vertices


def export_stl(vertices, faces, output_path, ascii_mode=False):
    """Exporta la malla a formato STL (binario o ASCII).

    Parametros:
        vertices: ndarray (N, 3) con coordenadas de vertices
        faces: ndarray (M, 3) con indices de triangulos
        output_path: ruta del archivo de salida
        ascii_mode: si True, escribe STL en formato ASCII; si False, binario
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if ascii_mode:
        _write_stl_ascii(vertices, faces, output_path)
    else:
        _write_stl_binary(vertices, faces, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("STL exportado: %s (%.2f MB, %d triangulos)",
                output_path, size_mb, len(faces))
    return output_path


def _write_stl_binary(vertices, faces, output_path):
    """Escribe un archivo STL en formato binario."""
    num_triangles = len(faces)
    with open(output_path, "wb") as f:
        header = b"Larmornium nii_2_3d export" + b"\x00" * (80 - 26)
        f.write(header)
        f.write(struct.pack("<I", num_triangles))

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len

            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def _write_stl_ascii(vertices, faces, output_path):
    """Escribe un archivo STL en formato ASCII."""
    with open(output_path, "w", encoding="ascii") as f:
        f.write("solid larmornium\n")
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}\n")
            f.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
            f.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid larmornium\n")


def export_obj(vertices, faces, output_path, write_mtl=False):
    """Exporta la malla a formato Wavefront OBJ.

    Parametros:
        vertices: ndarray (N, 3) con coordenadas de vertices
        faces: ndarray (M, 3) con indices de triangulos
        output_path: ruta del archivo de salida
        write_mtl: si True, genera un archivo .mtl con material basico
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Larmornium nii_2_3d export\n")
        f.write(f"# Vertices: {len(vertices)}, Triangulos: {len(faces)}\n")

        if write_mtl:
            mtl_name = base_name + ".mtl"
            f.write(f"mtllib {mtl_name}\n")
            f.write(f"usemtl material_0\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # OBJ usa indices base 1
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    if write_mtl:
        mtl_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), base_name + ".mtl")
        with open(mtl_path, "w", encoding="utf-8") as f:
            f.write("# Larmornium nii_2_3d material\n")
            f.write("newmtl material_0\n")
            f.write("Ka 0.2 0.2 0.2\n")
            f.write("Kd 0.8 0.8 0.8\n")
            f.write("Ks 0.5 0.5 0.5\n")
            f.write("Ns 50.0\n")
            f.write("d 1.0\n")
        logger.info("MTL exportado: %s", mtl_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("OBJ exportado: %s (%.2f MB, %d vertices, %d triangulos)",
                output_path, size_mb, len(vertices), len(faces))
    return output_path


def export_glb(vertices, faces, output_path, title="modelo"):
    """Exporta la malla a formato glTF binario (.glb).

    Parametros:
        vertices: ndarray (N, 3) con coordenadas de vertices
        faces: ndarray (M, 3) con indices de triangulos
        output_path: ruta del archivo de salida
        title: nombre de la malla dentro del archivo
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Calcular normales por triangulo y asignarlas por vertice (flat shading)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    face_normals = face_normals / norms

    # Expandir a vertices individuales por triangulo (flat shading)
    num_tris = len(faces)
    flat_positions = np.empty((num_tris * 3, 3), dtype=np.float32)
    flat_normals = np.empty((num_tris * 3, 3), dtype=np.float32)
    flat_positions[0::3] = v0
    flat_positions[1::3] = v1
    flat_positions[2::3] = v2
    flat_normals[0::3] = face_normals
    flat_normals[1::3] = face_normals
    flat_normals[2::3] = face_normals

    flat_indices = np.arange(num_tris * 3, dtype=np.uint32)

    # Datos binarios del buffer
    pos_bytes = flat_positions.tobytes()
    norm_bytes = flat_normals.tobytes()
    idx_bytes = flat_indices.tobytes()

    pos_offset = 0
    pos_length = len(pos_bytes)
    norm_offset = pos_length
    norm_length = len(norm_bytes)
    idx_offset = norm_offset + norm_length
    idx_length = len(idx_bytes)
    total_buffer = pos_length + norm_length + idx_length

    # Limites del bounding box para el accessor de posiciones
    pos_min = flat_positions.min(axis=0).tolist()
    pos_max = flat_positions.max(axis=0).tolist()

    num_vertices_flat = num_tris * 3
    num_indices = num_tris * 3

    # Estructura JSON del glTF
    gltf = {
        "asset": {"version": "2.0", "generator": "Larmornium nii_2_3d"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": title}],
        "meshes": [{
            "name": title,
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1},
                "indices": 2,
                "mode": 4
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": num_vertices_flat,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": num_vertices_flat,
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "componentType": 5125,
                "count": num_indices,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_length, "target": 34962},
            {"buffer": 0, "byteOffset": norm_offset, "byteLength": norm_length, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_length, "target": 34963}
        ],
        "buffers": [{"byteLength": total_buffer}]
    }

    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    # Alinear JSON a 4 bytes con espacios
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad
    json_length = len(json_bytes)

    # Alinear buffer binario a 4 bytes con ceros
    bin_data = pos_bytes + norm_bytes + idx_bytes
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_data += b"\x00" * bin_pad
    bin_length = len(bin_data)

    # Tamano total del archivo GLB
    total_length = 12 + 8 + json_length + 8 + bin_length

    with open(output_path, "wb") as f:
        # Header GLB: magic, version, length
        f.write(struct.pack("<I", 0x46546C67))  # glTF magic
        f.write(struct.pack("<I", 2))            # version 2
        f.write(struct.pack("<I", total_length))
        # Chunk 0: JSON
        f.write(struct.pack("<I", json_length))
        f.write(struct.pack("<I", 0x4E4F534A))  # JSON chunk type
        f.write(json_bytes)
        # Chunk 1: BIN
        f.write(struct.pack("<I", bin_length))
        f.write(struct.pack("<I", 0x004E4942))  # BIN chunk type
        f.write(bin_data)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("GLB exportado: %s (%.2f MB, %d triangulos)",
                output_path, size_mb, num_tris)
    return output_path


def segmentation_to_3d(volume_input, output_path, output_format="stl",
                       json_path=None, voxel_spacing=None,
                       scale=1.0, smooth_sigma=0.5, quality=1.0,
                       ascii_stl=False,
                       write_mtl_obj=False,
                       title_glb="modelo"):
    """Convierte un volumen de segmentacion binario a un archivo 3D imprimible.

    Parametros:
        volume_input: ruta a archivo NIfTI (.nii/.nii.gz) o ndarray 3D de 0s y 1s
        output_path: ruta del archivo 3D de salida
        output_format: formato de salida ('stl', 'obj', 'glb')
        json_path: ruta al JSON sidecar del NIfTI (opcional, para spacing)
        voxel_spacing: lista [sx, sy, sz] en mm (requerido para ndarray crudo)
        scale: factor de escala (1.0=original, 2.0=doble, 0.5=mitad)
        smooth_sigma: sigma del suavizado gaussiano previo a marching cubes (0=sin suavizado)
        quality: calidad de la malla de 0.0 (minima/maxima reduccion) a 1.0 (calidad original)
        ascii_stl: para STL, si True escribe en ASCII en vez de binario
        write_mtl_obj: para OBJ, si True genera archivo .mtl acompanante
        title_glb: para GLB, nombre de la malla dentro del archivo

    Retorna:
        dict con ruta de salida, estadisticas de la malla y tiempo de proceso
    """
    t0 = time.time()
    fmt = output_format.lower().strip().lstrip(".")
    if fmt not in FORMATOS_VALIDOS:
        raise ValueError(f"Formato no soportado: '{output_format}'. Usar: {FORMATOS_VALIDOS}")

    if scale <= 0:
        raise ValueError(f"El factor de escala debe ser positivo, se recibio: {scale}")

    if not (0.0 <= quality <= 1.0):
        raise ValueError(f"El parametro quality debe estar entre 0.0 y 1.0, se recibio: {quality}")

    volume, spacing = _load_segmentation_volume(volume_input, json_path, voxel_spacing)
    vertices, faces = _generate_mesh(volume, spacing, smooth_sigma)
    if quality < 1.0:
        vertices, faces = _decimate_mesh(vertices, faces, quality=quality)
    vertices = _apply_scale(vertices, scale)

    if fmt == "stl":
        export_stl(vertices, faces, output_path, ascii_mode=ascii_stl)
    elif fmt == "obj":
        export_obj(vertices, faces, output_path, write_mtl=write_mtl_obj)
    elif fmt == "glb":
        export_glb(vertices, faces, output_path, title=title_glb)

    elapsed = time.time() - t0
    result = {
        "output_path": os.path.abspath(output_path),
        "format": fmt,
        "num_vertices": len(vertices),
        "num_triangles": len(faces),
        "scale_factor": scale,
        "quality": quality,
        "voxel_spacing_mm": spacing,
        "volume_shape": list(volume.shape),
        "processing_time_seconds": round(elapsed, 3),
    }

    logger.info("Conversion completada en %.2f s: %s", elapsed, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="nii_2_3d.py - Conversion de segmentaciones binarias a archivos 3D imprimibles (STL/OBJ/GLB)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos de uso:
  python3 nii_2_3d.py -i segmentacion.nii.gz --json segmentacion.json -f stl -o modelo.stl
  python3 nii_2_3d.py -i segmentacion.nii.gz -f obj --scale 2.0 --write-mtl -o modelo.obj
  python3 nii_2_3d.py -i segmentacion.nii.gz -f glb --glb-title cerebro -o modelo.glb
        """,
    )

    parser.add_argument("-i", "--input", required=True,
                        help="Ruta al volumen NIfTI de segmentacion (.nii o .nii.gz)")
    parser.add_argument("-o", "--output", required=True,
                        help="Ruta del archivo 3D de salida")
    parser.add_argument("-f", "--format", required=True, choices=["stl", "obj", "glb"],
                        help="Formato del archivo 3D de salida")
    parser.add_argument("--json", default=None,
                        help="Ruta al JSON sidecar del NIfTI (para obtener voxel_spacing_mm)")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, default=None,
                        metavar=("SX", "SY", "SZ"),
                        help="Espaciado de voxel en mm (sx sy sz), alternativa al JSON")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Factor de escala del volumen de salida (default: 1.0)")
    parser.add_argument("--smooth-sigma", type=float, default=0.5,
                        help="Sigma del suavizado gaussiano previo (0=sin suavizado, default: 0.5)")
    parser.add_argument("--quality", type=float, default=1.0,
                        help="Calidad de la malla entre 0.0 (minima/maxima reduccion) y 1.0 (calidad original, default: 1.0)")

    stl_group = parser.add_argument_group("Opciones STL")
    stl_group.add_argument("--ascii", action="store_true", default=False,
                           help="Escribir STL en formato ASCII en vez de binario")

    obj_group = parser.add_argument_group("Opciones OBJ")
    obj_group.add_argument("--write-mtl", action="store_true", default=False,
                           help="Generar archivo .mtl con material basico")

    glb_group = parser.add_argument_group("Opciones GLB")
    glb_group.add_argument("--glb-title", default="modelo",
                           help="Nombre de la malla dentro del archivo GLB (default: 'modelo')")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Mostrar informacion detallada en consola")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    result = segmentation_to_3d(
        volume_input=args.input,
        output_path=args.output,
        output_format=args.format,
        json_path=args.json,
        voxel_spacing=args.voxel_spacing,
        scale=args.scale,
        smooth_sigma=args.smooth_sigma,
        quality=args.quality,
        ascii_stl=args.ascii,
        write_mtl_obj=args.write_mtl,
        title_glb=args.glb_title,
    )

    print("\n" + "=" * 60)
    print("  RESULTADO DE CONVERSION A 3D")
    print("=" * 60)
    print(f"  Archivo de salida   : {result['output_path']}")
    print(f"  Formato             : {result['format'].upper()}")
    print(f"  Vertices            : {result['num_vertices']:,}")
    print(f"  Triangulos          : {result['num_triangles']:,}")
    print(f"  Factor de escala    : {result['scale_factor']}")
    print(f"  Calidad de malla    : {result['quality']}")
    print(f"  Espaciado voxel (mm): {result['voxel_spacing_mm']}")
    print(f"  Dimensiones volumen : {result['volume_shape']}")
    print(f"  Tiempo de proceso   : {result['processing_time_seconds']} s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
