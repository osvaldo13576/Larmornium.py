#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
from datetime import datetime

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger("gen_volume_CT")

IGNORE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("._",)
NON_DICOM_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".fig", ".tif",
    ".tiff", ".mat", ".xlsx", ".csv", ".txt", ".nii", ".gz",
}


def _safe_float(ds, attr, default=None):
    val = getattr(ds, attr, None)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(ds, attr, default=None):
    val = getattr(ds, attr, None)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_str(ds, attr, default=None):
    val = getattr(ds, attr, None)
    if val is None or val == "":
        return default
    return str(val)


def _read_dicom_slice(filepath):
    basename = os.path.basename(filepath)
    if basename in IGNORE_FILES:
        return None
    if any(basename.startswith(p) for p in IGNORE_PREFIXES):
        return None
    _, ext = os.path.splitext(basename)
    if ext.lower() in NON_DICOM_EXTENSIONS:
        return None

    try:
        ds = pydicom.dcmread(filepath, force=True)
    except (InvalidDicomError, Exception):
        return None

    # Verificar que tiene datos de píxel
    if not hasattr(ds, "pixel_array"):
        return None

    # Filtrar solo modalidades compatibles con CT (CT, SC, OT con descripción CT)
    modality = str(getattr(ds, "Modality", "")).upper()
    if modality not in ("CT", "SC", "OT", "") and not basename.startswith("CT"):
        return None

    # Asignar valores por defecto seguros si faltan metadatos geométricos
    if not hasattr(ds, "PixelSpacing") or ds.PixelSpacing is None:
        ds.PixelSpacing = [1.0, 1.0]

    if not hasattr(ds, "ImageOrientationPatient") or ds.ImageOrientationPatient is None:
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    if not hasattr(ds, "ImagePositionPatient") or ds.ImagePositionPatient is None:
        # Intentar derivar de SliceLocation o InstanceNumber
        loc = getattr(ds, "SliceLocation", None)
        if loc is not None:
            ds.ImagePositionPatient = [0.0, 0.0, float(loc)]
        else:
            inst = getattr(ds, "InstanceNumber", 0)
            thick = _safe_float(ds, "SliceThickness", 1.0)
            ds.ImagePositionPatient = [0.0, 0.0, float(inst) * thick]

    return ds


def _build_affine(first_ds, slice_spacing):
    # ImageOrientationPatient: [row_x, row_y, row_z, col_x, col_y, col_z]
    raw_iop = getattr(first_ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    iop = [float(v) for v in raw_iop] if raw_iop else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    row_cosine = np.array(iop[0:3])  # dirección de las columnas de la imagen
    col_cosine = np.array(iop[3:6])  # dirección de las filas de la imagen

    # Dirección del eje Z (producto cruz)
    slice_cosine = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(slice_cosine)
    if norm > 1e-6:
        slice_cosine = slice_cosine / norm
    else:
        slice_cosine = np.array([0.0, 0.0, 1.0])

    # PixelSpacing: [row_spacing, col_spacing] en mm
    raw_ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    ps = [float(v) for v in raw_ps] if raw_ps else [1.0, 1.0]
    row_spacing = float(ps[0])  # Espaciado entre filas (dirección Y)
    col_spacing = float(ps[1])  # Espaciado entre columnas (dirección X)

    # ImagePositionPatient: origen del primer corte [x, y, z]
    raw_ipp = getattr(first_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
    ipp = [float(v) for v in raw_ipp] if raw_ipp else [0.0, 0.0, 0.0]

    # Construir la matriz afín en el sistema estándar NIfTI (RAS+):
    affine = np.eye(4)
    
    # Eje i (columnas de la imagen)
    affine[0, 0] = -row_cosine[0] * col_spacing
    affine[1, 0] = -row_cosine[1] * col_spacing
    affine[2, 0] =  row_cosine[2] * col_spacing

    # Eje j (filas de la imagen)
    affine[0, 1] = -col_cosine[0] * row_spacing
    affine[1, 1] = -col_cosine[1] * row_spacing
    affine[2, 1] =  col_cosine[2] * row_spacing

    # Eje k (cortes / dirección Z)
    affine[0, 2] = -slice_cosine[0] * slice_spacing
    affine[1, 2] = -slice_cosine[1] * slice_spacing
    affine[2, 2] =  slice_cosine[2] * slice_spacing

    # Origen (traslación)
    affine[0, 3] = -ipp[0]
    affine[1, 3] = -ipp[1]
    affine[2, 3] =  ipp[2]

    return affine


def generate_ct_volume(ct_directory, dicom_root, output_path=None, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        if not logger.handlers and not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if os.path.isabs(ct_directory):
        abs_ct_dir = ct_directory
    else:
        abs_ct_dir = os.path.join(dicom_root, ct_directory)

    if not os.path.isdir(abs_ct_dir):
        logger.error("Directorio CT no encontrado: %s", abs_ct_dir)
        raise FileNotFoundError(f"Directorio CT no encontrado: {abs_ct_dir}")

    logger.info("Leyendo cortes DICOM CT de: %s", abs_ct_dir)

    # Leer todos los archivos DICOM del directorio
    slices = []
    for fname in sorted(os.listdir(abs_ct_dir)):
        fpath = os.path.join(abs_ct_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ds = _read_dicom_slice(fpath)
        if ds is not None:
            slices.append(ds)

    if not slices:
        logger.error("No se encontraron archivos DICOM CT en: %s", abs_ct_dir)
        raise RuntimeError(f"No se encontraron archivos DICOM CT en: {abs_ct_dir}")

    logger.info("Cortes CT leídos: %d", len(slices))

    # Ordenar cortes por posición Z o número de instancia
    def _slice_sort_key(ds):
        if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
            try:
                return (0, float(ds.ImagePositionPatient[2]))
            except (ValueError, IndexError):
                pass
        if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
            try:
                return (1, float(ds.SliceLocation))
            except ValueError:
                pass
        return (2, _safe_int(ds, "InstanceNumber", 0))

    slices.sort(key=_slice_sort_key)

    # Extraer metadata del primer corte
    first_ds = slices[0]
    last_ds = slices[-1]

    # Extraer dimensiones de la matriz de píxeles
    sample_arr = first_ds.pixel_array
    if sample_arr.ndim == 3 and sample_arr.shape[-1] == 3:
        rows, cols = sample_arr.shape[0], sample_arr.shape[1]
    elif sample_arr.ndim == 3 and sample_arr.shape[0] == 3:
        rows, cols = sample_arr.shape[1], sample_arr.shape[2]
    else:
        rows, cols = sample_arr.shape[0], sample_arr.shape[1]

    num_slices = len(slices)

    raw_ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing_row = float(raw_ps[0]) if raw_ps else 1.0
    pixel_spacing_col = float(raw_ps[1]) if raw_ps else 1.0

    # Calcular espaciado entre cortes a partir de posiciones reales
    if num_slices > 1:
        z_positions = []
        for ds in slices:
            if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
                try:
                    z_positions.append(float(ds.ImagePositionPatient[2]))
                except (ValueError, IndexError):
                    pass
        if len(z_positions) == num_slices:
            z_diffs = np.diff(z_positions)
            slice_spacing = float(np.median(np.abs(z_diffs))) if len(z_diffs) > 0 else 1.0
            if slice_spacing < 1e-4:
                slice_spacing = _safe_float(first_ds, "SliceThickness", 1.0)
        else:
            slice_spacing = _safe_float(first_ds, "SliceThickness", 1.0)
    else:
        slice_spacing = _safe_float(first_ds, "SliceThickness", 1.0)

    if slice_spacing <= 0:
        slice_spacing = 1.0

    logger.info("Dimensiones: %d × %d × %d", cols, rows, num_slices)
    logger.info("Espaciado vóxel (mm): [%.4f, %.4f, %.4f]",
                pixel_spacing_col, pixel_spacing_row, slice_spacing)

    # Construir el volumen 3D
    volume = np.zeros((cols, rows, num_slices), dtype=np.float32)

    for k, ds in enumerate(slices):
        raw_pixel = ds.pixel_array
        if raw_pixel.ndim == 3 and raw_pixel.shape[-1] == 3:
            # Convertir imagen RGB a escala de grises (luminancia)
            pixel_data = (0.299 * raw_pixel[:, :, 0] + 0.587 * raw_pixel[:, :, 1] + 0.114 * raw_pixel[:, :, 2]).astype(np.float32)
        elif raw_pixel.ndim == 3 and raw_pixel.shape[0] == 3:
            pixel_data = (0.299 * raw_pixel[0, :, :] + 0.587 * raw_pixel[1, :, :] + 0.114 * raw_pixel[2, :, :]).astype(np.float32)
        else:
            pixel_data = raw_pixel.astype(np.float32)

        # Calibración a Unidades Hounsfield si tiene slope/intercept
        slope = _safe_float(ds, "RescaleSlope", 1.0)
        intercept = _safe_float(ds, "RescaleIntercept", 0.0)
        pixel_data = pixel_data * slope + intercept

        # Transponer de (filas, columnas) a (columnas, filas) para NIfTI
        volume[:, :, k] = pixel_data.T

    hu_min = float(volume.min())
    hu_max = float(volume.max())
    logger.info("Rango HU: [%.1f, %.1f]", hu_min, hu_max)

    affine = _build_affine(first_ds, slice_spacing)
    logger.debug("Matriz afín:\n%s", affine)

    nifti_img = nib.Nifti1Image(volume, affine)

    header = nifti_img.header
    header.set_xyzt_units("mm")
    header["descrip"] = np.bytes_(
        f"CT volume from {ct_directory}"[:80]
    )

    if output_path is None:
        safe_name = ct_directory.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_dir = os.path.join(os.path.dirname(dicom_root) or ".", "volumes")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{safe_name}.nii.gz")
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    nib.save(nifti_img, output_path)
    logger.info("Volumen NIfTI guardado: %s", output_path)

    origin = [float(v) for v in getattr(first_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]
    iop = [float(v) for v in getattr(first_ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]

    z_first = float(getattr(first_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])[2]) if hasattr(first_ds, "ImagePositionPatient") and first_ds.ImagePositionPatient else 0.0
    z_last = float(getattr(last_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])[2]) if hasattr(last_ds, "ImagePositionPatient") and last_ds.ImagePositionPatient else float(num_slices)

    metadata = {
        "ct_directory": ct_directory,
        "dicom_root": os.path.abspath(dicom_root),
        "num_slices": num_slices,
        "dimensions": [cols, rows, num_slices],
        "voxel_spacing_mm": [pixel_spacing_col, pixel_spacing_row, slice_spacing],
        "origin_mm": origin,
        "orientation_row": iop[0:3] if len(iop) >= 3 else [1.0, 0.0, 0.0],
        "orientation_col": iop[3:6] if len(iop) >= 6 else [0.0, 1.0, 0.0],
        "hu_range": [hu_min, hu_max],
        "output_path": os.path.abspath(output_path),
        "patient_name": _safe_str(first_ds, "PatientName"),
        "patient_id": _safe_str(first_ds, "PatientID"),
        "study_date": _safe_str(first_ds, "StudyDate"),
        "study_description": _safe_str(first_ds, "StudyDescription"),
        "series_description": _safe_str(first_ds, "SeriesDescription"),
        "institution_name": _safe_str(first_ds, "InstitutionName"),
        "manufacturer": _safe_str(first_ds, "Manufacturer"),
        "manufacturer_model": _safe_str(first_ds, "ManufacturerModelName"),
        "kvp": _safe_float(first_ds, "KVP"),
        "slice_thickness_dicom": _safe_float(first_ds, "SliceThickness", slice_spacing),
        "z_range_mm": [z_first, z_last],
        "generation_timestamp": datetime.now().isoformat(),
    }

    return nifti_img, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generador de volúmenes 3D NIfTI a partir de series CT DICOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  python3 gen_volume_CT.py \\
      --ct-dir PET_CT/pacientes/paciente_00/osteo1/SE000001 \\
      --dicom-root ./DICOM

  python3 gen_volume_CT.py \\
      --ct-dir PET_CT/pacientes/paciente_00/osteo1/SE000001 \\
      --dicom-root ./DICOM \\
      --output ./volumes/paciente_00_ct.nii.gz \\
      --verbose
        """
    )
    parser.add_argument(
        "--ct-dir", required=True,
        help="Ruta relativa al directorio de la serie CT dentro del DICOM root "
             "(ej: PET_CT/pacientes/paciente_00/osteo1/SE000001)"
    )
    parser.add_argument(
        "--dicom-root", required=True,
        help="Ruta al directorio raíz DICOM (ej: ./DICOM)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el archivo NIfTI (.nii o .nii.gz). "
             "Si no se especifica, se genera automáticamente en ./volumes/"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir información detallada"
    )

    args = parser.parse_args()

    nifti_img, metadata = generate_ct_volume(
        ct_directory=args.ct_dir,
        dicom_root=args.dicom_root,
        output_path=args.output,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("VOLUMEN CT GENERADO")
    print("=" * 60)
    print(f"  Directorio CT      : {metadata['ct_directory']}")
    print(f"  Dimensiones        : {metadata['dimensions']}")
    print(f"  Espaciado (mm)     : {metadata['voxel_spacing_mm']}")
    print(f"  Rango HU           : {metadata['hu_range']}")
    print(f"  Rango Z (mm)       : {metadata['z_range_mm']}")
    print(f"  Serie              : {metadata['series_description']}")
    print(f"  Paciente           : {metadata['patient_name']}")
    print(f"  Archivo NIfTI      : {metadata['output_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
