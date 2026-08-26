#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_pet_ct.py - Fusión espacial de imágenes CT y PET

Genera imágenes fusionadas de PET/CT a partir de un par de directorios
de imágenes DICOM (CT y PET) usando los metadatos espaciales DICOM:

  - ImagePositionPatient (0020,0032): esquina superior izquierda del primer píxel
  - ImageOrientationPatient (0020,0037): cosenos directores de filas y columnas
  - PixelSpacing (0028,0030): [row_spacing, col_spacing] en mm
  - ReconstructionTargetCenterPatient (0018,9313): centro del FOV reconstruido

Algoritmo de fusión:
  1. Calcular la extensión espacial de ambas imágenes (CT y PET) en
     coordenadas del paciente, usando IPP + PixelSpacing + IOP.
  2. Determinar la región de solapamiento (overlap) entre ambas.
  3. Recortar la imagen CT a la región de solapamiento (define la grilla
     de salida a resolución CT, que es más fina).
  4. Para cada píxel de la grilla de salida, mapear sus coordenadas del
     paciente a coordenadas fraccionarias de píxel en la imagen PET.
  5. Interpolar los valores PET en esas coordenadas (map_coordinates).
  6. Aplicar calibración (RescaleSlope/Intercept) a ambas modalidades.
  7. Combinar CT (escala de grises) + PET (mapa de color) con transparencia.

Uso desde terminal:
    python3 fusion_pet_ct.py \
        --ct-dir DICOM/PET_CT/paciente_00/osteo1/SE000001 \
        --pet-dir DICOM/PET_CT/paciente_00/osteo1/SE000003 \
        --dicom-root DICOM \
        --slice-index 187

Uso como módulo:
    from fusion_pet_ct import fuse_from_directories
    result = fuse_from_directories(ct_dir, pet_dir, dicom_root, slice_index=187)
"""

import argparse
import datetime
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.ndimage import map_coordinates


def parse_dicom_time_seconds(time_val):
    """Convierte una representacion de tiempo DICOM a segundos desde medianoche."""
    if time_val is None or time_val == "":
        return 0.0
    if isinstance(time_val, (datetime.datetime, datetime.time)):
        return time_val.hour * 3600.0 + time_val.minute * 60.0 + time_val.second + time_val.microsecond / 1e6
    if isinstance(time_val, (int, float)):
        if float(time_val) < 86400.0 and float(time_val) >= 0.0 and not str(int(time_val)).endswith("00"):
            return float(time_val)
        time_val = str(int(time_val)).zfill(6)
    t_str = str(time_val).strip()
    if "T" in t_str:
        t_str = t_str.split("T")[-1]
    elif len(t_str) >= 14 and "." not in t_str[:14] and t_str.isdigit():
        t_str = t_str[8:]
    parts = t_str.split(".")
    base = parts[0].zfill(6)
    usec = float("0." + parts[1]) if len(parts) > 1 else 0.0
    try:
        hours = int(base[0:2])
        minutes = int(base[2:4])
        seconds = int(base[4:6])
        return hours * 3600.0 + minutes * 60.0 + seconds + usec
    except (ValueError, IndexError):
        return 0.0


def _pixel_to_patient_2d(row_idx, col_idx, ipp, ps, row_dir, col_dir):
    """
    Convierte indices de pixel (row, col) a coordenadas del paciente (x, y).

    Parametros:
        row_idx, col_idx : array-like (indices de pixel)
        ipp : array (3,) ImagePositionPatient [x, y, z]
        ps : array (2,) PixelSpacing [row_spacing, col_spacing]
        row_dir : array (3,) direccion de fila
        col_dir : array (3,) direccion de columna

    Retorna:
        patient_x, patient_y : arrays con las coordenadas del paciente.
    """
    ps_row, ps_col = float(ps[0]), float(ps[1])
    x = ipp[0] + col_idx * ps_col * row_dir[0] + row_idx * ps_row * col_dir[0]
    y = ipp[1] + col_idx * ps_col * row_dir[1] + row_idx * ps_row * col_dir[1]
    return x, y


def _patient_to_pixel_2d(patient_x, patient_y, ipp, ps, row_dir, col_dir):
    """
    Convierte coordenadas del paciente (x, y) a indices fraccionarios de pixel.

    Retorna:
        row_float, col_float : arrays con indices fraccionarios de pixel.
    """
    ps_row, ps_col = float(ps[0]), float(ps[1])

    M = np.array([
        [row_dir[0] * ps_col, col_dir[0] * ps_row],
        [row_dir[1] * ps_col, col_dir[1] * ps_row],
    ], dtype=np.float64)
    M_inv = np.linalg.inv(M)

    delta_x = np.asarray(patient_x, dtype=np.float64) - ipp[0]
    delta_y = np.asarray(patient_y, dtype=np.float64) - ipp[1]

    if delta_x.ndim == 0:
        delta = np.array([float(delta_x), float(delta_y)])
        cr = M_inv @ delta
        return cr[1], cr[0]

    original_shape = delta_x.shape
    delta = np.stack([delta_x.ravel(), delta_y.ravel()], axis=0)
    cr = M_inv @ delta
    col_float = cr[0].reshape(original_shape)
    row_float = cr[1].reshape(original_shape)
    return row_float, col_float


def compute_spatial_extent(ipp, ps, rows, cols, row_dir, col_dir):
    """
    Calcula la caja envolvente de una imagen DICOM en coordenadas del paciente.

    Retorna:
        dict con claves: x_min, x_max, y_min, y_max (en mm).
    """
    corners_r = np.array([0, 0, rows - 1, rows - 1], dtype=np.float64)
    corners_c = np.array([0, cols - 1, 0, cols - 1], dtype=np.float64)
    cx, cy = _pixel_to_patient_2d(corners_r, corners_c, ipp, ps, row_dir, col_dir)
    return {
        "x_min": float(np.min(cx)),
        "x_max": float(np.max(cx)),
        "y_min": float(np.min(cy)),
        "y_max": float(np.max(cy)),
    }


def extract_spatial_metadata(ds):
    """
    Extrae los metadatos espaciales y de calibracion de un dataset pydicom.

    Retorna un diccionario con metadatos espaciales, calibracion HU y factor SUV.
    """
    ipp = np.array([float(v) for v in ds.ImagePositionPatient], dtype=np.float64) \
        if hasattr(ds, "ImagePositionPatient") else np.array([0.0, 0.0, 0.0])

    ps = np.array([float(v) for v in ds.PixelSpacing], dtype=np.float64) \
        if hasattr(ds, "PixelSpacing") else np.array([1.0, 1.0])

    iop_raw = [float(v) for v in ds.ImageOrientationPatient] \
        if hasattr(ds, "ImageOrientationPatient") else [1, 0, 0, 0, 1, 0]
    iop = np.array(iop_raw, dtype=np.float64)
    row_dir = iop[:3]
    col_dir = iop[3:]

    # ReconstructionTargetCenterPatient (0018,9313) - puede estar ausente en PET
    rtcp = None
    if hasattr(ds, "ReconstructionTargetCenterPatient"):
        rtcp = np.array([float(v) for v in ds.ReconstructionTargetCenterPatient])
    elif (0x0018, 0x9313) in ds:
        try:
            rtcp = np.array([float(v) for v in ds[(0x0018, 0x9313)].value])
        except Exception:
            pass

    rad_dose = None
    half_life = 6586.2
    inj_time = None
    if hasattr(ds, "RadiopharmaceuticalInformationSequence") and len(ds.RadiopharmaceuticalInformationSequence) > 0:
        r_seq = ds.RadiopharmaceuticalInformationSequence[0]
        try:
            if hasattr(r_seq, "RadionuclideTotalDose") and r_seq.RadionuclideTotalDose not in (None, ""):
                rad_dose = float(r_seq.RadionuclideTotalDose)
        except (ValueError, TypeError):
            pass
        try:
            if hasattr(r_seq, "RadionuclideHalfLife") and r_seq.RadionuclideHalfLife not in (None, ""):
                half_life = float(r_seq.RadionuclideHalfLife)
        except (ValueError, TypeError):
            pass
        inj_time = getattr(r_seq, "RadiopharmaceuticalStartTime", None)

    scan_time = getattr(ds, "SeriesTime", getattr(ds, "AcquisitionTime", getattr(ds, "StudyTime", None)))
    weight_kg = None
    if hasattr(ds, "PatientWeight") and ds.PatientWeight not in (None, ""):
        try:
            weight_kg = float(ds.PatientWeight)
        except (ValueError, TypeError):
            pass

    suv_factor = None
    if rad_dose is not None and rad_dose > 0 and weight_kg is not None and weight_kg > 0:
        t_inj_sec = parse_dicom_time_seconds(inj_time)
        t_scan_sec = parse_dicom_time_seconds(scan_time)
        delta_t = t_scan_sec - t_inj_sec
        if delta_t < 0:
            delta_t += 86400.0
        decay_factor = float(np.exp(-np.log(2.0) * delta_t / half_life)) if half_life > 0 else 1.0
        decayed_dose = rad_dose * decay_factor
        dose_per_g = decayed_dose / (weight_kg * 1000.0)
        if dose_per_g > 0:
            suv_factor = 1.0 / dose_per_g

    return {
        "ipp": ipp,
        "pixel_spacing": ps,
        "iop": iop,
        "row_dir": row_dir,
        "col_dir": col_dir,
        "rows": int(ds.Rows),
        "cols": int(ds.Columns),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
        "reconstruction_target_center": rtcp,
        "suv_factor": suv_factor,
        "patient_weight": weight_kg,
        "radionuclide_total_dose": rad_dose,
    }


def fuse_slice(ct_pixels, pet_pixels, ct_meta, pet_meta,
               ct_window=(40, 400), alpha=0.4, pet_vmax_percentile=99.5,
               pet_colormap="hot"):
    """
    Fusiona un par de cortes CT y PET usando los metadatos espaciales.

    Algoritmo:
        1. Calcular la extensión espacial de CT y PET usando IPP + PS + IOP.
        2. Determinar la región de solapamiento (overlap) en coordenadas
           del paciente.
        3. Recortar CT a la región de solapamiento (la resolución CT
           más fina define la grilla de salida).
        4. Para cada píxel de la grilla CT recortada, mapear a coordenadas
           fraccionarias del PET usando la transformación inversa.
        5. Interpolar los valores PET en esas coordenadas (bilineal).
        6. Aplicar calibración a ambas modalidades (HU y SUV/Actividad).
        7. Crear la imagen fusionada RGB.
    """
    ct_ipp = ct_meta["ipp"]
    ct_ps = ct_meta["pixel_spacing"]
    ct_row_dir = ct_meta["row_dir"]
    ct_col_dir = ct_meta["col_dir"]
    ct_rows = ct_meta["rows"]
    ct_cols = ct_meta["cols"]

    pet_ipp = pet_meta["ipp"]
    pet_ps = pet_meta["pixel_spacing"]
    pet_row_dir = pet_meta["row_dir"]
    pet_col_dir = pet_meta["col_dir"]
    pet_rows = pet_meta["rows"]
    pet_cols = pet_meta["cols"]

    # Verificar que las orientaciones sean compatibles
    dot_row = np.abs(np.dot(ct_row_dir, pet_row_dir))
    dot_col = np.abs(np.dot(ct_col_dir, pet_col_dir))
    if dot_row < 0.99 or dot_col < 0.99:
        warnings.warn(
            f"Las orientaciones de CT y PET difieren significativamente: "
            f"dot(row_dirs)={dot_row:.4f}, dot(col_dirs)={dot_col:.4f}. "
            f"La fusión puede ser imprecisa."
        )

    # 1. Extensión espacial de cada imagen
    ct_extent = compute_spatial_extent(ct_ipp, ct_ps, ct_rows, ct_cols,
                                       ct_row_dir, ct_col_dir)
    pet_extent = compute_spatial_extent(pet_ipp, pet_ps, pet_rows, pet_cols,
                                        pet_row_dir, pet_col_dir)

    # 2. Región de solapamiento
    ov_x_min = max(ct_extent["x_min"], pet_extent["x_min"])
    ov_x_max = min(ct_extent["x_max"], pet_extent["x_max"])
    ov_y_min = max(ct_extent["y_min"], pet_extent["y_min"])
    ov_y_max = min(ct_extent["y_max"], pet_extent["y_max"])

    if ov_x_min >= ov_x_max or ov_y_min >= ov_y_max:
        raise ValueError(
            f"No hay solapamiento espacial entre CT y PET.\n"
            f"CT extent: x=[{ct_extent['x_min']:.2f}, {ct_extent['x_max']:.2f}], "
            f"y=[{ct_extent['y_min']:.2f}, {ct_extent['y_max']:.2f}]\n"
            f"PET extent: x=[{pet_extent['x_min']:.2f}, {pet_extent['x_max']:.2f}], "
            f"y=[{pet_extent['y_min']:.2f}, {pet_extent['y_max']:.2f}]"
        )

    overlap = {
        "x_min": ov_x_min, "x_max": ov_x_max,
        "y_min": ov_y_min, "y_max": ov_y_max,
        "width_mm": ov_x_max - ov_x_min,
        "height_mm": ov_y_max - ov_y_min,
    }

    # 3. Recortar CT a la región de solapamiento
    ct_r_min, ct_c_min = _patient_to_pixel_2d(
        ov_x_min, ov_y_min, ct_ipp, ct_ps, ct_row_dir, ct_col_dir
    )
    ct_r_max, ct_c_max = _patient_to_pixel_2d(
        ov_x_max, ov_y_max, ct_ipp, ct_ps, ct_row_dir, ct_col_dir
    )

    # Ordenar y limitar a las dimensiones de la imagen
    ct_row_start = int(max(0, np.ceil(min(ct_r_min, ct_r_max))))
    ct_row_end = int(min(ct_rows, np.floor(max(ct_r_min, ct_r_max)) + 1))
    ct_col_start = int(max(0, np.ceil(min(ct_c_min, ct_c_max))))
    ct_col_end = int(min(ct_cols, np.floor(max(ct_c_min, ct_c_max)) + 1))

    ct_crop = {
        "row_start": ct_row_start, "row_end": ct_row_end,
        "col_start": ct_col_start, "col_end": ct_col_end,
        "rows_cropped": ct_row_end - ct_row_start,
        "cols_cropped": ct_col_end - ct_col_start,
        "rows_removed": ct_rows - (ct_row_end - ct_row_start),
        "cols_removed": ct_cols - (ct_col_end - ct_col_start),
    }

    ct_cropped = ct_pixels[ct_row_start:ct_row_end,
                           ct_col_start:ct_col_end].astype(np.float64)

    out_rows = ct_row_end - ct_row_start
    out_cols = ct_col_end - ct_col_start

    # IPP de la esquina de la imagen de salida
    out_ipp_x, out_ipp_y = _pixel_to_patient_2d(
        ct_row_start, ct_col_start, ct_ipp, ct_ps, ct_row_dir, ct_col_dir
    )

    # 4. Mapear cada píxel de la grilla CT recortada a coordenadas PET
    ct_row_indices = np.arange(ct_row_start, ct_row_end)
    ct_col_indices = np.arange(ct_col_start, ct_col_end)
    ct_cc, ct_rr = np.meshgrid(ct_col_indices, ct_row_indices)

    # Píxel CT a coordenadas del paciente
    patient_x, patient_y = _pixel_to_patient_2d(
        ct_rr, ct_cc, ct_ipp, ct_ps, ct_row_dir, ct_col_dir
    )

    # Coordenadas del paciente a píxel PET (fraccionario)
    pet_rr_float, pet_cc_float = _patient_to_pixel_2d(
        patient_x, patient_y, pet_ipp, pet_ps, pet_row_dir, pet_col_dir
    )

    # 5. Interpolar PET y aplicar calibración
    pet_resampled = map_coordinates(
        pet_pixels.astype(np.float64),
        [pet_rr_float.ravel(), pet_cc_float.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(out_rows, out_cols)

    ct_slope = ct_meta["rescale_slope"]
    ct_intercept = ct_meta["rescale_intercept"]
    ct_hu = ct_cropped * ct_slope + ct_intercept

    pet_slope = pet_meta["rescale_slope"]
    pet_intercept = pet_meta["rescale_intercept"]
    pet_activity = pet_resampled * pet_slope + pet_intercept

    suv_factor = pet_meta.get("suv_factor")
    if suv_factor is not None and suv_factor > 0:
        pet_suv = pet_activity * suv_factor
        pet_units = "SUV"
    else:
        pet_suv = pet_activity
        pet_units = "Bq/mL"

    # 6. Crear imagen fusionada RGB
    wl, ww = ct_window
    vmin_ct = wl - ww / 2.0
    vmax_ct = wl + ww / 2.0
    ct_norm = np.clip((ct_hu - vmin_ct) / (vmax_ct - vmin_ct), 0, 1)

    positive_mask = pet_suv > 0
    if np.any(positive_mask):
        pet_vmax = float(np.percentile(pet_suv[positive_mask],
                                       pet_vmax_percentile))
    else:
        pet_vmax = 1.0
    pet_norm = np.clip(pet_suv / pet_vmax, 0, 1)

    cmap_pet = plt.colormaps[pet_colormap]
    ct_rgb = plt.cm.gray(ct_norm)[:, :, :3]
    pet_rgb = cmap_pet(pet_norm)[:, :, :3]

    pet_weight = alpha * pet_norm[:, :, np.newaxis]
    fusion_rgb = ct_rgb * (1.0 - pet_weight) + pet_rgb * pet_weight
    fusion_rgb = np.clip(fusion_rgb, 0, 1)

    return {
        "fusion_rgb": fusion_rgb,
        "ct_hu": ct_hu.astype(np.float32),
        "pet_activity": pet_activity.astype(np.float32),
        "pet_suv": pet_suv.astype(np.float32),
        "pet_units": pet_units,
        "ct_cropped_raw": ct_cropped,
        "ct_norm": ct_norm,
        "pet_norm": pet_norm,
        "pet_vmax": pet_vmax,
        "overlap": overlap,
        "ct_extent": ct_extent,
        "pet_extent": pet_extent,
        "ct_crop": ct_crop,
        "output_shape": (out_rows, out_cols),
        "output_pixel_spacing": [float(ct_ps[0]), float(ct_ps[1])],
        "output_ipp": [float(out_ipp_x), float(out_ipp_y), float(ct_ipp[2])],
    }


def _list_dicom_files_sorted_by_z(directory):
    """
    Lista archivos DICOM de un directorio y los ordena por posicion Z
    (ImagePositionPatient[2]) descendente (superior a inferior).
    """
    entries = []
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
                entries.append((fpath, z))
        except Exception:
            continue

    entries.sort(key=lambda x: x[1], reverse=True)
    return entries


def _match_ct_pet_slices(ct_abs_dir, pet_abs_dir):
    """
    Empareja los cortes de dos directorios de series DICOM (CT y PET) por
    coordenada Z, ordenados de superior a inferior.
    """
    ct_files = _list_dicom_files_sorted_by_z(ct_abs_dir)
    pet_files = _list_dicom_files_sorted_by_z(pet_abs_dir)

    if len(ct_files) == 0:
        raise ValueError(f"No se encontraron archivos DICOM en {ct_abs_dir}")
    if len(pet_files) == 0:
        raise ValueError(f"No se encontraron archivos DICOM en {pet_abs_dir}")

    if len(pet_files) <= len(ct_files):
        primary_files = pet_files
        secondary_files = ct_files
        primary_is_pet = True
    else:
        primary_files = ct_files
        secondary_files = pet_files
        primary_is_pet = False

    secondary_zs = np.array([z for _, z in secondary_files], dtype=np.float64)

    matched_pairs = []
    for prim_fpath, prim_z in primary_files:
        diffs = np.abs(secondary_zs - prim_z)
        min_idx = int(np.argmin(diffs))
        min_diff = diffs[min_idx]
        if min_diff <= 15.0:
            sec_fpath = secondary_files[min_idx][0]
            if primary_is_pet:
                matched_pairs.append((sec_fpath, prim_fpath, prim_z))
            else:
                matched_pairs.append((prim_fpath, sec_fpath, prim_z))

    if len(matched_pairs) == 0:
        raise ValueError(
            f"No se pudieron emparejar cortes CT y PET por posicion Z.\n"
            f"CT z rango: [{ct_files[-1][1]:.1f}, {ct_files[0][1]:.1f}]\n"
            f"PET z rango: [{pet_files[-1][1]:.1f}, {pet_files[0][1]:.1f}]"
        )

    matched_pairs.sort(key=lambda x: x[2], reverse=True)
    return matched_pairs


def fuse_from_directories(ct_dir, pet_dir, dicom_root=".",
                          slice_index=None, alpha=0.4,
                          ct_window=(40, 400), pet_colormap="hot",
                          pet_vmax_percentile=99.5):
    """
    Fusiona un corte CT y PET a partir de directorios de series DICOM.
    """
    ct_abs_dir = os.path.join(dicom_root, ct_dir)
    pet_abs_dir = os.path.join(dicom_root, pet_dir)

    if not os.path.isdir(ct_abs_dir):
        raise FileNotFoundError(f"Directorio CT no encontrado: {ct_abs_dir}")
    if not os.path.isdir(pet_abs_dir):
        raise FileNotFoundError(f"Directorio PET no encontrado: {pet_abs_dir}")

    matched_pairs = _match_ct_pet_slices(ct_abs_dir, pet_abs_dir)

    if slice_index is None:
        slice_index = len(matched_pairs) // 2

    if slice_index < 0 or slice_index >= len(matched_pairs):
        raise IndexError(
            f"slice_index={slice_index} fuera de rango. "
            f"Cortes emparejados disponibles: {len(matched_pairs)}"
        )

    ct_file_path, pet_file_path, z_pos = matched_pairs[slice_index]

    ds_ct = pydicom.dcmread(ct_file_path, force=True)
    ds_pet = pydicom.dcmread(pet_file_path, force=True)

    ct_pixels = ds_ct.pixel_array
    pet_pixels = ds_pet.pixel_array

    ct_meta = extract_spatial_metadata(ds_ct)
    pet_meta = extract_spatial_metadata(ds_pet)

    result = fuse_slice(
        ct_pixels, pet_pixels, ct_meta, pet_meta,
        ct_window=ct_window, alpha=alpha,
        pet_vmax_percentile=pet_vmax_percentile,
        pet_colormap=pet_colormap,
    )

    result["ct_file_path"] = ct_file_path
    result["pet_file_path"] = pet_file_path
    result["slice_index"] = slice_index
    result["z_position"] = z_pos
    result["total_matched_slices"] = len(matched_pairs)
    result["ct_meta"] = ct_meta
    result["pet_meta"] = pet_meta

    return result


def fuse_volume_from_directories(ct_dir, pet_dir, dicom_root=".", alpha=0.4,
                                 ct_window=(40, 400), pet_colormap="hot",
                                 pet_vmax_percentile=99.5,
                                 progress_callback=None):
    """
    Fusiona todos los cortes emparejados de CT y PET generando un volumen 3D completo.
    Retorna tanto el volumen RGB pre-renderizado como las matrices desacopladas
    ct_volume (en HU) y pet_volume (en SUV/Actividad).
    """
    ct_abs_dir = os.path.join(dicom_root, ct_dir)
    pet_abs_dir = os.path.join(dicom_root, pet_dir)

    if not os.path.isdir(ct_abs_dir):
        raise FileNotFoundError(f"Directorio CT no encontrado: {ct_abs_dir}")
    if not os.path.isdir(pet_abs_dir):
        raise FileNotFoundError(f"Directorio PET no encontrado: {pet_abs_dir}")

    matched_pairs = _match_ct_pet_slices(ct_abs_dir, pet_abs_dir)

    slices_rgb = []
    ct_slices = []
    pet_slices = []
    z_positions = []
    pixel_spacing = None
    slice_thickness = None
    pet_units = "SUV"
    total = len(matched_pairs)

    for idx, (ct_file_path, pet_file_path, z_pos) in enumerate(matched_pairs):
        ds_ct = pydicom.dcmread(ct_file_path, force=True)
        ds_pet = pydicom.dcmread(pet_file_path, force=True)

        ct_meta = extract_spatial_metadata(ds_ct)
        pet_meta = extract_spatial_metadata(ds_pet)

        result = fuse_slice(
            ds_ct.pixel_array, ds_pet.pixel_array, ct_meta, pet_meta,
            ct_window=ct_window, alpha=alpha,
            pet_vmax_percentile=pet_vmax_percentile,
            pet_colormap=pet_colormap,
        )
        slices_rgb.append(result["fusion_rgb"].astype(np.float32))
        ct_slices.append(result["ct_hu"].astype(np.float32))
        pet_slices.append(result["pet_suv"].astype(np.float32))
        z_positions.append(z_pos)
        pet_units = result.get("pet_units", "SUV")

        if pixel_spacing is None:
            pixel_spacing = result["output_pixel_spacing"]
            slice_thickness = getattr(ds_ct, "SliceThickness", None)

        if progress_callback:
            progress_callback(idx + 1, total)

    fusion_volume = np.stack(slices_rgb, axis=0)
    ct_volume = np.stack(ct_slices, axis=0)
    pet_volume = np.stack(pet_slices, axis=0)

    positive_pet = pet_volume[pet_volume > 0]
    pet_vmax = float(np.percentile(positive_pet, pet_vmax_percentile)) if len(positive_pet) > 0 else 1.0

    return {
        "fusion_volume": fusion_volume,
        "ct_volume": ct_volume,
        "pet_volume": pet_volume,
        "pet_vmax": pet_vmax,
        "pet_units": pet_units,
        "z_positions": z_positions,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": float(slice_thickness) if slice_thickness else None,
        "output_shape": tuple(fusion_volume.shape[1:3]),
        "num_slices": int(fusion_volume.shape[0]),
    }


def fuse_and_save_pair(ct_dir, pet_dir, dicom_root, output_nii_path, output_json_path,
                       pair_metadata=None, alpha=0.4, ct_window=(40, 400),
                       pet_colormap="hot", pet_vmax_percentile=99.5,
                       progress_callback=None):
    """
    Genera el volumen fusionado CT+PET y lo guarda como un archivo NIfTI (.nii/.nii.gz)
    multicanal (canal 0 = CT en HU, canal 1 = PET en SUV/actividad) conservando la
    informacion cuantitativa fisica completa, junto con su archivo .json de metadatos.
    """
    import nibabel as nib

    result = fuse_volume_from_directories(
        ct_dir, pet_dir, dicom_root=dicom_root, alpha=alpha,
        ct_window=ct_window, pet_colormap=pet_colormap,
        pet_vmax_percentile=pet_vmax_percentile,
        progress_callback=progress_callback,
    )

    ct_transposed = np.transpose(result["ct_volume"], (2, 1, 0))
    pet_transposed = np.transpose(result["pet_volume"], (2, 1, 0))
    volume_4d = np.stack([ct_transposed, pet_transposed], axis=-1).astype(np.float32)

    row_spacing, col_spacing = result["pixel_spacing"]
    slice_spacing = result["slice_thickness"] or 1.0
    affine = np.diag([col_spacing, row_spacing, slice_spacing, 1.0]).astype(np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(output_nii_path)), exist_ok=True)
    nib.save(nib.Nifti1Image(volume_4d, affine), output_nii_path)

    metadata = dict(pair_metadata or {})
    metadata.update({
        "nii_path": os.path.abspath(output_nii_path),
        "num_slices": result["num_slices"],
        "output_shape": list(result["output_shape"]),
        "pixel_spacing": result["pixel_spacing"],
        "slice_thickness": result["slice_thickness"],
        "z_positions": result["z_positions"],
        "ct_window": list(ct_window),
        "alpha": alpha,
        "pet_colormap": pet_colormap,
        "pet_vmax": result["pet_vmax"],
        "pet_units": result["pet_units"],
        "channels": ["CT_HU", "PET_%s" % result["pet_units"]],
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return output_nii_path, output_json_path


def plot_fusion_result(result, title=None, figsize=(18, 6)):
    """
    Visualiza el resultado de la fusion: CT, PET y fusion lado a lado.
    """
    ct_hu = result["ct_hu"]
    pet_activity = result["pet_activity"]
    fusion = result["fusion_rgb"]
    z = result.get("z_position", None)
    idx = result.get("slice_index", None)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    wl, ww = 40, 400
    axes[0].imshow(ct_hu, cmap="gray", vmin=wl - ww / 2, vmax=wl + ww / 2)
    ct_label = f"CT - Corte #{idx}\nz = {z:.1f} mm" if z is not None else "CT"
    ct_label += f"\nShape: {ct_hu.shape}\nPixelSpacing: {result['output_pixel_spacing']} mm"
    axes[0].set_title(ct_label, fontsize=10)
    axes[0].axis("off")

    im_pet = axes[1].imshow(pet_activity, cmap="hot", vmin=0,
                            vmax=result["pet_vmax"])
    pet_label = f"PET (remuestreado a grilla CT)\nz = {z:.1f} mm" \
        if z is not None else "PET"
    pet_label += f"\nShape: {pet_activity.shape}"
    axes[1].set_title(pet_label, fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im_pet, ax=axes[1], label="Actividad", fraction=0.046, pad=0.04)

    axes[2].imshow(fusion)
    fuse_label = f"Fusión CT + PET\nz = {z:.1f} mm" if z is not None else "Fusión"
    fuse_label += f"\nOverlap: {result['overlap']['width_mm']:.1f} x " \
                  f"{result['overlap']['height_mm']:.1f} mm"
    axes[2].set_title(fuse_label, fontsize=10)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(
        description="Fusión espacial de imágenes CT y PET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
    python3 fusion_pet_ct.py \\
        --ct-dir PET_CT/paciente_00/osteo1/SE000001 \\
        --pet-dir PET_CT/paciente_00/osteo1/SE000003 \\
        --dicom-root ./DICOM \\
        --slice-index 187 \\
        --alpha 0.4
        """,
    )
    parser.add_argument("--ct-dir", required=True,
                        help="Ruta relativa al directorio de la serie CT")
    parser.add_argument("--pet-dir", required=True,
                        help="Ruta relativa al directorio de la serie PET")
    parser.add_argument("--dicom-root", default=".",
                        help="Ruta raíz de DICOM (por defecto: .)")
    parser.add_argument("--slice-index", type=int, default=None,
                        help="Índice del corte (por defecto: medio)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Transparencia del PET (por defecto: 0.4)")
    parser.add_argument("--ct-center", type=float, default=40,
                        help="Centro de la ventana CT en HU (por defecto: 40)")
    parser.add_argument("--ct-width", type=float, default=400,
                        help="Ancho de la ventana CT en HU (por defecto: 400)")
    parser.add_argument("--pet-cmap", default="hot",
                        help="Mapa de color para PET (por defecto: hot)")
    parser.add_argument("--output", default=None,
                        help="Archivo de salida para la imagen (por defecto: mostrar)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mostrar información detallada")

    args = parser.parse_args()

    ct_window = (args.ct_center, args.ct_width)

    if args.verbose:
        print(f"CT dir  : {args.ct_dir}")
        print(f"PET dir : {args.pet_dir}")
        print(f"Root    : {args.dicom_root}")
        print(f"Ventana CT: center={args.ct_center}, width={args.ct_width}")
        print(f"Alpha   : {args.alpha}")
        print()

    result = fuse_from_directories(
        ct_dir=args.ct_dir,
        pet_dir=args.pet_dir,
        dicom_root=args.dicom_root,
        slice_index=args.slice_index,
        alpha=args.alpha,
        ct_window=ct_window,
        pet_colormap=args.pet_cmap,
    )

    if args.verbose:
        print(f"Corte #{result['slice_index']} de {result['total_matched_slices']}")
        print(f"Z = {result['z_position']:.1f} mm")
        print(f"CT:  {result['ct_meta']['rows']} x {result['ct_meta']['cols']} px, "
              f"PS={list(result['ct_meta']['pixel_spacing'])}")
        print(f"PET: {result['pet_meta']['rows']} x {result['pet_meta']['cols']} px, "
              f"PS={list(result['pet_meta']['pixel_spacing'])}")
        print(f"Output shape: {result['output_shape']}")
        ov = result["overlap"]
        print(f"Overlap: {ov['width_mm']:.1f} x {ov['height_mm']:.1f} mm")
        ct_crop = result["ct_crop"]
        print(f"CT crop: rows[{ct_crop['row_start']}:{ct_crop['row_end']}], "
              f"cols[{ct_crop['col_start']}:{ct_crop['col_end']}] "
              f"({ct_crop['rows_removed']} filas, {ct_crop['cols_removed']} cols eliminadas)")
        if result["ct_meta"]["reconstruction_target_center"] is not None:
            rtcp = result["ct_meta"]["reconstruction_target_center"]
            print(f"CT RTCP: ({rtcp[0]:.1f}, {rtcp[1]:.1f}, {rtcp[2]:.1f})")
        print()

    fig, _ = plot_fusion_result(result, title=f"Fusión PET/CT - z={result['z_position']:.1f} mm")

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Imagen guardada en: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
