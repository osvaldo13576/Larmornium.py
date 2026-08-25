#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calcular_HU_CT.py - Cálculo eficiente de Unidades Hounsfield (HU) para cortes CT

Calcula la matriz 2D de Unidades Hounsfield (HU) para un corte individual de CT
a partir de la ruta de la imagen (obtenida del archivo .json de indexación) y
los parámetros de calibración (obtenidos de la base de datos .db de indexación).

Fórmula:
    HU = PixelArray * RescaleSlope + RescaleIntercept

Uso como módulo:
    from calcular_HU_CT import calcular_hu_ct

    # Con parámetros desacoplados del .db (máxima eficiencia)
    hu_matrix = calcular_hu_ct(
        image_path="PET_CT/fwhm_puntos_rad/1_2_RE_BP_ALL/SE000001/CT000000",
        rescale_slope=1.0,
        rescale_intercept=-1024.0,
        dicom_root="./DICOM"
    )

    # O dejando que extraiga los metadatos directamente del DICOM si no se pasan
    hu_matrix = calcular_hu_ct("ruta/a/CT000000")

Uso desde CLI:
    python3 calcular_HU_CT.py \
        --image-path "PET_CT/fwhm_puntos_rad/1_2_RE_BP_ALL/SE000001/CT000000" \
        --rescale-slope 1.0 \
        --rescale-intercept -1024.0 \
        --dicom-root ./DICOM \
        --stats
"""

import argparse
import logging
import os
import sys

import numpy as np
import pydicom

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("calcular_HU_CT")


def resolver_ruta_imagen(image_path: str, dicom_root: str = None) -> str:
    """
    Resuelve la ruta completa al archivo DICOM manejando rutas absolutas,
    relativas, sin extensión y con diversas extensiones (.dcm, .ima).

    Parameters
    ----------
    image_path : str
        Ruta al archivo DICOM (absoluta o relativa).
    dicom_root : str, optional
        Directorio raíz DICOM para resolver rutas relativas.

    Returns
    -------
    str
        Ruta absoluta validada existente.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra el archivo en ninguna de las rutas posibles.
    """
    candidatos = [image_path]

    if dicom_root:
        candidatos.append(os.path.join(dicom_root, image_path))
        if image_path.startswith("DICOM/") or image_path.startswith("DICOM\\"):
            candidatos.append(os.path.join(dicom_root, image_path[6:]))

    ext_candidatos = []
    for cand in candidatos:
        ext_candidatos.append(cand)
        if not os.path.splitext(cand)[1]:
            ext_candidatos.append(cand + ".dcm")
            ext_candidatos.append(cand + ".DCM")
            ext_candidatos.append(cand + ".ima")
            ext_candidatos.append(cand + ".IMA")

    for ruta in ext_candidatos:
        if os.path.isfile(ruta):
            return os.path.abspath(ruta)

    raise FileNotFoundError(
        f"No se pudo encontrar el archivo DICOM de CT: '{image_path}'. "
        f"Rutas probadas: {ext_candidatos}"
    )


def leer_pixels_ct_raw(image_path: str, dicom_root: str = None):
    """
    Lee los píxeles crudos (sin calibrar) de un archivo DICOM de CT.

    Returns
    -------
    tuple (numpy.ndarray, dict)
        - raw_pixels: Matriz 2D de tipo float64.
        - header_info: Diccionario con metadatos relevantes del encabezado.
    """
    ruta_real = resolver_ruta_imagen(image_path, dicom_root)
    try:
        ds = pydicom.dcmread(ruta_real, force=True)
    except Exception as e:
        raise RuntimeError(f"Error al leer archivo DICOM '{ruta_real}': {e}")

    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"El archivo DICOM '{ruta_real}' no contiene 'pixel_array'.")

    raw_pixels = ds.pixel_array.astype(np.float64)

    def _safe_float(attr, default=None):
        val = getattr(ds, attr, None)
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _safe_int(attr, default=None):
        val = getattr(ds, attr, None)
        if val is None or val == "":
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    header_info = {
        "file_path": ruta_real,
        "rows": getattr(ds, "Rows", raw_pixels.shape[0]),
        "columns": getattr(ds, "Columns", raw_pixels.shape[1]),
        "rescale_slope": _safe_float("RescaleSlope", 1.0),
        "rescale_intercept": _safe_float("RescaleIntercept", -1024.0),
        "pixel_spacing": [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0]) if v is not None and v != ""],
        "slice_thickness": _safe_float("SliceThickness", None),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "instance_number": _safe_int("InstanceNumber", None),
    }

    return raw_pixels, header_info


def calcular_hu_ct(image_path: str,
                   rescale_slope: float = None,
                   rescale_intercept: float = None,
                   dicom_root: str = None,
                   return_metadata: bool = False):
    """
    Calcula la matriz 2D de Unidades Hounsfield (HU) para un corte de CT.

    Parameters
    ----------
    image_path : str
        Ruta al archivo del corte CT (obtenida del JSON de indexación).
    rescale_slope : float, optional
        Pendiente de calibración (obtenida del .db de indexación).
        Si es None, se lee del encabezado DICOM.
    rescale_intercept : float, optional
        Intercepción de calibración (obtenida del .db de indexación).
        Si es None, se lee del encabezado DICOM.
    dicom_root : str, optional
        Ruta al directorio raíz DICOM si image_path es relativa.
    return_metadata : bool, optional
        Si True, retorna también un diccionario con información y estadísticas.

    Returns
    -------
    numpy.ndarray o tuple(numpy.ndarray, dict)
        - hu_matrix: Arreglo 2D NumPy (float64) con los valores calibrados en HU.
        - metadata (si return_metadata=True): Diccionario con detalles de calibración.
    """
    raw_pixels, header_info = leer_pixels_ct_raw(image_path, dicom_root)

    slope = float(rescale_slope) if rescale_slope is not None else header_info["rescale_slope"]
    intercept = float(rescale_intercept) if rescale_intercept is not None else header_info["rescale_intercept"]

    # Cálculo físico de HU: HU = Pixel * Slope + Intercept
    hu_matrix = raw_pixels * slope + intercept

    if return_metadata:
        meta = {
            "file_path": header_info["file_path"],
            "shape": list(hu_matrix.shape),
            "rescale_slope_used": slope,
            "rescale_intercept_used": intercept,
            "min_hu": float(np.min(hu_matrix)),
            "max_hu": float(np.max(hu_matrix)),
            "mean_hu": float(np.mean(hu_matrix)),
            "std_hu": float(np.std(hu_matrix)),
            "pixel_spacing": header_info["pixel_spacing"],
            "slice_thickness": header_info["slice_thickness"],
            "sop_instance_uid": header_info["sop_instance_uid"],
            "instance_number": header_info["instance_number"],
        }
        return hu_matrix, meta

    return hu_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Calcula la matriz 2D de Unidades Hounsfield (HU) para un corte de CT."
    )
    parser.add_argument(
        "-i", "--image-path", required=True,
        help="Ruta al archivo DICOM del corte CT (ej: PET_CT/paciente_00/osteo1/SE000001/CT000000)"
    )
    parser.add_argument(
        "-s", "--rescale-slope", type=float, default=None,
        help="RescaleSlope obtenido del .db (ej: 1.0). Si se omite, se lee del DICOM."
    )
    parser.add_argument(
        "-t", "--rescale-intercept", type=float, default=None,
        help="RescaleIntercept obtenido del .db (ej: -1024.0). Si se omite, se lee del DICOM."
    )
    parser.add_argument(
        "-d", "--dicom-root", default=None,
        help="Directorio raíz DICOM para rutas relativas."
    )
    parser.add_argument(
        "-o", "--output-npy", default=None,
        help="Ruta para guardar la matriz de salida como archivo binario NumPy (.npy)."
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Ruta para guardar la matriz de salida como archivo de texto CSV."
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Imprime estadísticas descriptivas de los valores HU del corte."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Modo detallado de depuración."
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        hu_matrix, meta = calcular_hu_ct(
            image_path=args.image_path,
            rescale_slope=args.rescale_slope,
            rescale_intercept=args.rescale_intercept,
            dicom_root=args.dicom_root,
            return_metadata=True
        )

        logger.info("Corte CT procesado exitosamente: %s", meta["file_path"])
        logger.info("Dimensiones de la matriz: %s", meta["shape"])
        logger.info("Calibración aplicada: Slope = %.4f, Intercept = %.4f",
                    meta["rescale_slope_used"], meta["rescale_intercept_used"])

        if args.stats or args.verbose:
            print("\n" + "=" * 55)
            print("  ESTADÍSTICAS DEL CORTE CT (Unidades Hounsfield)")
            print("=" * 55)
            print(f"  Archivo           : {os.path.basename(meta['file_path'])}")
            print(f"  Dimensiones (fil, col) : {meta['shape'][0]} x {meta['shape'][1]}")
            print(f"  HU Mínimo         : {meta['min_hu']:.1f} HU")
            print(f"  HU Máximo         : {meta['max_hu']:.1f} HU")
            print(f"  HU Promedio       : {meta['mean_hu']:.2f} HU")
            print(f"  HU Desv. Estándar : {meta['std_hu']:.2f} HU")
            print(f"  PixelSpacing (mm) : {meta['pixel_spacing']}")
            print("=" * 55 + "\n")

        if args.output_npy:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)
            np.save(args.output_npy, hu_matrix)
            logger.info("Matriz HU guardada en formato NumPy: %s", args.output_npy)

        if args.output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
            np.savetxt(args.output_csv, hu_matrix, delimiter=",", fmt="%.2f")
            logger.info("Matriz HU guardada en formato CSV: %s", args.output_csv)

    except Exception as e:
        logger.error("Error al calcular HU: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
