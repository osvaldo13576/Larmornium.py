#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import math
import os
from datetime import datetime

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger("gen_volume_PET")

IGNORE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("._",)
NON_DICOM_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".fig", ".tif",
    ".tiff", ".mat", ".xlsx", ".csv", ".txt", ".nii", ".gz",
}

# Vida media estándar de 18F-FDG en segundos (109.77 min x 60)
DEFAULT_HALF_LIFE_F18 = 6586.2


def _safe_float(ds, attr, default=None):
    val = getattr(ds, attr, None)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_str(ds, attr, default=None):
    val = getattr(ds, attr, None)
    if val is None or val == "":
        return default
    return str(val)


def _safe_int(ds, attr, default=0):
    val = getattr(ds, attr, None)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _parse_dicom_time_seconds(time_val):
    if time_val is None:
        return None
    s = str(time_val).strip()
    # Eliminar caracteres no numéricos al final
    if "." in s:
        s_main, _ = s.split(".", 1)
    else:
        s_main = s
    # HHMMSS
    if len(s_main) >= 6:
        try:
            hh = int(s_main[0:2])
            mm = int(s_main[2:4])
            ss = int(s_main[4:6])
            return float(hh * 3600 + mm * 60 + ss)
        except ValueError:
            pass
    # HHMM
    if len(s_main) >= 4:
        try:
            hh = int(s_main[0:2])
            mm = int(s_main[2:4])
            return float(hh * 3600 + mm * 60)
        except ValueError:
            pass
    return None


def _extract_suv_calibration(ds):
    params = {
        "patient_weight_kg": None,
        "radionuclide_total_dose_bq": None,
        "radionuclide_half_life_s": None,
        "radiopharmaceutical_start_time_s": None,
        "series_time_s": None,
        "acquisition_time_s": None,
        "decay_correction": None,
    }

    # Peso del paciente (kg)
    pw = _safe_float(ds, "PatientWeight", None)
    if pw and pw > 0:
        params["patient_weight_kg"] = pw

    # Tiempos
    params["series_time_s"] = _parse_dicom_time_seconds(
        _safe_str(ds, "SeriesTime") or _safe_str(ds, "StudyTime")
    )
    params["acquisition_time_s"] = _parse_dicom_time_seconds(
        _safe_str(ds, "AcquisitionTime")
    )

    # Corrección de decaimiento (ADMIN=ya corregida, START=hay que corregir)
    params["decay_correction"] = _safe_str(ds, "DecayCorrection", "START")

    # RadiopharmaceuticalInformationSequence
    rp_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if rp_seq and len(rp_seq) > 0:
        rp = rp_seq[0]
        dose = _safe_float(rp, "RadionuclideTotalDose", None)
        if dose and dose > 0:
            params["radionuclide_total_dose_bq"] = dose

        hl = _safe_float(rp, "RadionuclideHalfLife", None)
        if hl and hl > 0:
            params["radionuclide_half_life_s"] = hl
        else:
            params["radionuclide_half_life_s"] = DEFAULT_HALF_LIFE_F18

        start_time_str = _safe_str(rp, "RadiopharmaceuticalStartTime", None)
        params["radiopharmaceutical_start_time_s"] = _parse_dicom_time_seconds(start_time_str)

    return params


def _compute_suv_factor(calibration_params):
    p = calibration_params
    w = p.get("patient_weight_kg")
    dose = p.get("radionuclide_total_dose_bq")
    hl = p.get("radionuclide_half_life_s") or DEFAULT_HALF_LIFE_F18
    inj_t = p.get("radiopharmaceutical_start_time_s")
    scan_t = p.get("series_time_s") or p.get("acquisition_time_s")
    decay_corr = p.get("decay_correction", "START")

    if not (w and w > 0 and dose and dose > 0):
        return None

    # Si la imagen ya tiene corrección de decaimiento ADMIN (dosis al momento
    # de la administración), no es necesario aplicar decaimiento adicional.
    if str(decay_corr).upper() == "START" and inj_t and scan_t:
        delta_t = scan_t - inj_t
        if delta_t < 0:
            delta_t += 86400.0  # crossing midnight
        decay_factor = math.exp(-math.log(2) * delta_t / hl)
        dose_decayed = dose * decay_factor
        logger.debug(
            "Decaimiento: delta_t=%.1f s, factor=%.6f, dosis_decay=%.1f Bq",
            delta_t, decay_factor, dose_decayed
        )
    else:
        dose_decayed = dose

    suv_factor = (w * 1000.0) / dose_decayed
    logger.debug("Factor SUVbw: %.8f (peso=%.1f kg, dosis=%.1f Bq)", suv_factor, w, dose_decayed)
    return suv_factor


def _read_dicom_pet_slice(filepath):
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

    if not hasattr(ds, "pixel_array"):
        return None

    modality = str(getattr(ds, "Modality", "")).upper()
    # Aceptar PT (PET), NM (Nuclear Medicine) u OT si el basename contiene PT
    if modality not in ("PT", "NM", "OT", "") and not basename.upper().startswith("PT"):
        return None

    # Asignar metadatos geométricos por defecto si faltan
    if not hasattr(ds, "PixelSpacing") or ds.PixelSpacing is None:
        ds.PixelSpacing = [1.0, 1.0]

    if not hasattr(ds, "ImageOrientationPatient") or ds.ImageOrientationPatient is None:
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    if not hasattr(ds, "ImagePositionPatient") or ds.ImagePositionPatient is None:
        loc = getattr(ds, "SliceLocation", None)
        if loc is not None:
            ds.ImagePositionPatient = [0.0, 0.0, float(loc)]
        else:
            inst = _safe_int(ds, "InstanceNumber", 0)
            thick = _safe_float(ds, "SliceThickness", 1.0) or 1.0
            ds.ImagePositionPatient = [0.0, 0.0, float(inst) * thick]

    return ds


def _build_affine(first_ds, slice_spacing):
    raw_iop = getattr(first_ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    iop = [float(v) for v in raw_iop] if raw_iop else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    row_cosine = np.array(iop[0:3])  # dirección de las columnas de la imagen
    col_cosine = np.array(iop[3:6])  # dirección de las filas de la imagen

    slice_cosine = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(slice_cosine)
    if norm > 1e-6:
        slice_cosine = slice_cosine / norm
    else:
        slice_cosine = np.array([0.0, 0.0, 1.0])

    raw_ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    ps = [float(v) for v in raw_ps] if raw_ps else [1.0, 1.0]
    row_spacing = float(ps[0])
    col_spacing = float(ps[1])

    raw_ipp = getattr(first_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
    ipp = [float(v) for v in raw_ipp] if raw_ipp else [0.0, 0.0, 0.0]

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


def generate_pet_volume(pet_directory, dicom_root, output_path=None,
                        convert_suv=False, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        if not logger.handlers and not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if os.path.isabs(pet_directory):
        abs_pet_dir = pet_directory
    else:
        abs_pet_dir = os.path.join(dicom_root, pet_directory)

    if not os.path.isdir(abs_pet_dir):
        logger.error("Directorio PET no encontrado: %s", abs_pet_dir)
        raise FileNotFoundError(f"Directorio PET no encontrado: {abs_pet_dir}")

    logger.info("Leyendo cortes DICOM PET de: %s", abs_pet_dir)

    # Leer todos los archivos DICOM del directorio
    slices = []
    for fname in sorted(os.listdir(abs_pet_dir)):
        fpath = os.path.join(abs_pet_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ds = _read_dicom_pet_slice(fpath)
        if ds is not None:
            slices.append(ds)

    # Si no se encontraron cortes PET en el directorio dado, buscar inteligentemente en subdirectorios o hermanos
    if not slices:
        # Buscar en subdirectorios (ej: si se pasó el directorio del estudio en lugar de la serie)
        for root, dirs, files in os.walk(abs_pet_dir):
            if root == abs_pet_dir:
                continue
            cand_slices = []
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                if os.path.isfile(fpath):
                    ds = _read_dicom_pet_slice(fpath)
                    if ds is not None:
                        cand_slices.append(ds)
            if cand_slices:
                logger.info("Se encontraron cortes PET en el subdirectorio: %s", root)
                abs_pet_dir = root
                slices = cand_slices
                break

    if not slices:
        # Buscar en directorios hermanos (ej: si se pasó SE000001 de CT en lugar de SE000003 de PET)
        parent_dir = os.path.dirname(abs_pet_dir)
        if os.path.isdir(parent_dir):
            for sibling in sorted(os.listdir(parent_dir)):
                sib_path = os.path.join(parent_dir, sibling)
                if os.path.isdir(sib_path) and sib_path != abs_pet_dir:
                    cand_slices = []
                    for fname in sorted(os.listdir(sib_path)):
                        fpath = os.path.join(sib_path, fname)
                        if os.path.isfile(fpath):
                            ds = _read_dicom_pet_slice(fpath)
                            if ds is not None:
                                cand_slices.append(ds)
                    if cand_slices:
                        logger.info(
                            "Directorio '%s' no contenía cortes PET. Utilizando serie PET hermana detectada: %s (%d cortes)",
                            os.path.basename(abs_pet_dir), sibling, len(cand_slices)
                        )
                        abs_pet_dir = sib_path
                        slices = cand_slices
                        break

    if not slices:
        logger.error("No se encontraron archivos DICOM PET en: %s", abs_pet_dir)
        raise RuntimeError(f"No se encontraron archivos DICOM PET en: {abs_pet_dir}")

    logger.info("Cortes PET leídos: %d", len(slices))

    # Ordenar cortes por posición Z
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

    # Extraer metadata del primer y último corte
    first_ds = slices[0]
    last_ds = slices[-1]

    sample_arr = first_ds.pixel_array
    if sample_arr.ndim == 3 and sample_arr.shape[-1] in (1, 3, 4):
        rows, cols = sample_arr.shape[0], sample_arr.shape[1]
    else:
        rows, cols = sample_arr.shape[0], sample_arr.shape[1]

    num_slices = len(slices)

    raw_ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing_row = float(raw_ps[0]) if raw_ps else 1.0
    pixel_spacing_col = float(raw_ps[1]) if raw_ps else 1.0

    # Calcular espaciado entre cortes
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
                slice_spacing = _safe_float(first_ds, "SliceThickness", 1.0) or 1.0
        else:
            slice_spacing = _safe_float(first_ds, "SliceThickness", 1.0) or 1.0
    else:
        slice_spacing = _safe_float(first_ds, "SliceThickness", 1.0) or 1.0

    if slice_spacing <= 0:
        slice_spacing = 1.0

    logger.info("Dimensiones: %d × %d × %d", cols, rows, num_slices)
    logger.info("Espaciado vóxel (mm): [%.4f, %.4f, %.4f]",
                pixel_spacing_col, pixel_spacing_row, slice_spacing)

    # Extraer parámetros SUV del primer corte (si se pide conversión)
    suv_factor = None
    calibration_params = {}
    if convert_suv:
        calibration_params = _extract_suv_calibration(first_ds)
        suv_factor = _compute_suv_factor(calibration_params)
        if suv_factor is None:
            logger.warning(
                "No se pudieron extraer todos los parámetros SUVbw. "
                "El volumen se guardará en Bq/mL."
            )
        else:
            logger.info("Factor de conversión SUVbw: %.8f", suv_factor)

    # Construir el volumen 3D en Bq/mL
    volume = np.zeros((cols, rows, num_slices), dtype=np.float32)

    for k, ds in enumerate(slices):
        raw_pixel = ds.pixel_array.astype(np.float32)
        slope = _safe_float(ds, "RescaleSlope", 1.0) or 1.0
        intercept = _safe_float(ds, "RescaleIntercept", 0.0) or 0.0
        # Aplicar calibración a Bq/mL
        pixel_data = raw_pixel * slope + intercept
        # Transponer (rows, cols) a (cols, rows) para NIfTI
        volume[:, :, k] = pixel_data.T

    # Convertir a SUVbw si se solicitó y fue posible
    units = "Bq/mL"
    if convert_suv and suv_factor is not None:
        volume = volume * suv_factor
        units = "SUVbw"
        logger.info("Volumen convertido a SUVbw (factor=%.8f)", suv_factor)

    bqml_min = float(volume.min())
    bqml_max = float(volume.max())
    logger.info("Rango %s: [%.2f, %.2f]", units, bqml_min, bqml_max)

    affine = _build_affine(first_ds, slice_spacing)
    logger.debug("Matriz afín:\n%s", affine)

    nifti_img = nib.Nifti1Image(volume, affine)
    header = nifti_img.header
    header.set_xyzt_units("mm")
    unit_str = "SUVbw" if units == "SUVbw" else "Bq/mL"
    header["descrip"] = np.bytes_(
        f"PET volume [{unit_str}] from {pet_directory}"[:80]
    )

    if output_path is None:
        safe_name = pet_directory.replace("/", "_").replace("\\", "_").replace(":", "_")
        if convert_suv and suv_factor is not None:
            safe_name += "_suv"
        output_dir = os.path.join(os.path.dirname(os.path.abspath(dicom_root)), "volumes")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{safe_name}.nii.gz")
    else:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    nib.save(nifti_img, output_path)
    logger.info("Volumen NIfTI guardado: %s", output_path)

    origin = [float(v) for v in getattr(first_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]
    iop = [float(v) for v in getattr(first_ds, "ImageOrientationPatient",
                                     [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])]
    z_first = float(getattr(first_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])[2]) \
        if hasattr(first_ds, "ImagePositionPatient") and first_ds.ImagePositionPatient else 0.0
    z_last = float(getattr(last_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])[2]) \
        if hasattr(last_ds, "ImagePositionPatient") and last_ds.ImagePositionPatient else float(num_slices)

    metadata = {
        "pet_directory": pet_directory,
        "dicom_root": os.path.abspath(dicom_root),
        "num_slices": num_slices,
        "dimensions": [cols, rows, num_slices],
        "voxel_spacing_mm": [pixel_spacing_col, pixel_spacing_row, slice_spacing],
        "origin_mm": origin,
        "orientation_row": iop[0:3] if len(iop) >= 3 else [1.0, 0.0, 0.0],
        "orientation_col": iop[3:6] if len(iop) >= 6 else [0.0, 1.0, 0.0],
        "units": units,
        "value_range": [bqml_min, bqml_max],
        "output_path": os.path.abspath(output_path),
        "suv_factor": suv_factor,
        "suv_calibration": calibration_params,
        "patient_name": _safe_str(first_ds, "PatientName"),
        "patient_id": _safe_str(first_ds, "PatientID"),
        "patient_weight_kg": _safe_float(first_ds, "PatientWeight"),
        "study_date": _safe_str(first_ds, "StudyDate"),
        "study_description": _safe_str(first_ds, "StudyDescription"),
        "series_description": _safe_str(first_ds, "SeriesDescription"),
        "institution_name": _safe_str(first_ds, "InstitutionName"),
        "manufacturer": _safe_str(first_ds, "Manufacturer"),
        "manufacturer_model": _safe_str(first_ds, "ManufacturerModelName"),
        "decay_correction": _safe_str(first_ds, "DecayCorrection"),
        "actual_frame_duration_ms": _safe_float(first_ds, "ActualFrameDuration"),
        "slice_thickness_dicom": _safe_float(first_ds, "SliceThickness", slice_spacing),
        "z_range_mm": [z_first, z_last],
        "generation_timestamp": datetime.now().isoformat(),
    }

    return nifti_img, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generador de volúmenes 3D NIfTI a partir de series PET DICOM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  # Generar volumen en Bq/mL:
  python3 gen_volume_PET.py \\
      --pet-dir PET_CT/fwhm_puntos_rad/3_puntos_11_01_24/SE000003 \\
      --dicom-root ./DICOM

  # Generar volumen en SUVbw:
  python3 gen_volume_PET.py \\
      --pet-dir PET_CT/fwhm_puntos_rad/3_puntos_11_01_24/SE000003 \\
      --dicom-root ./DICOM \\
      --suv \\
      --output ./volumes/fwhm_pet_suv.nii.gz \\
      --verbose
        """
    )
    parser.add_argument(
        "--pet-dir", required=True,
        help="Ruta relativa al directorio de la serie PET dentro del DICOM root "
             "(ej: PET_CT/fwhm_puntos_rad/3_puntos_11_01_24/SE000003)"
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
        "--suv", action="store_true",
        help="Convertir el volumen a SUVbw usando los metadatos DICOM de "
             "RadiopharmaceuticalInformationSequence y PatientWeight"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir información detallada"
    )

    args = parser.parse_args()

    nifti_img, metadata = generate_pet_volume(
        pet_directory=args.pet_dir,
        dicom_root=args.dicom_root,
        output_path=args.output,
        convert_suv=args.suv,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("VOLUMEN PET GENERADO")
    print("=" * 60)
    print(f"  Directorio PET     : {metadata['pet_directory']}")
    print(f"  Dimensiones        : {metadata['dimensions']}")
    print(f"  Espaciado (mm)     : {metadata['voxel_spacing_mm']}")
    print(f"  Unidades           : {metadata['units']}")
    print(f"  Rango valores      : [{metadata['value_range'][0]:.2f}, {metadata['value_range'][1]:.2f}] {metadata['units']}")
    print(f"  Rango Z (mm)       : {metadata['z_range_mm']}")
    print(f"  Serie              : {metadata['series_description']}")
    print(f"  Paciente           : {metadata['patient_name']}")
    print(f"  Archivo NIfTI      : {metadata['output_path']}")
    if metadata.get("suv_factor"):
        print(f"  Factor SUVbw       : {metadata['suv_factor']:.8f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
