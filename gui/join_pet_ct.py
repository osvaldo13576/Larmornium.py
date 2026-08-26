#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_pet_ct.py - Construye y gestiona los volúmenes de CT, PET y Fusión CT+PET

Gestiona la generación bajo demanda de volúmenes (.nii.gz, .json) para series
individuales (CT, PET) y pares fusionables detectados en el índice combinado,
almacenando el registro en el archivo larmornium.conf.
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys

import numpy as np
import pydicom

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
_PROCESSING_DIR = os.path.join(_PROJECT_ROOT, "processing")
if _PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _PROCESSING_DIR)

import fusion_pet_ct  # noqa: E402

logger = logging.getLogger("join_pet_ct")

FUSION_VOL_DIRNAME = "fusion_vol"
CT_VOL_DIRNAME = "ct_vol"
PET_VOL_DIRNAME = "pet_vol"

CONFIG_KEY = "fusion_volumes"
CT_CONFIG_KEY = "ct_volumes"
PET_CONFIG_KEY = "pet_volumes"


def is_non_volume_series(series_or_desc, modality=None, num_images=None):
    """
    Determina si una serie corresponde a archivos de estadísticas, topogramas,
    reportes, capturas o imágenes ya fusionadas derivadas que se excluyen de la
    construcción de volúmenes 3D y se visualizan únicamente en 2D.
    """
    if isinstance(series_or_desc, dict):
        desc = str(series_or_desc.get("series_description", "") or "").lower()
        mod = str(series_or_desc.get("modality", modality or "") or "").upper()
        n_imgs = series_or_desc.get("num_images", num_images)
    else:
        desc = str(series_or_desc or "").lower()
        mod = str(modality or "").upper()
        n_imgs = num_images

    # Modalidades no tomográficas / documentales
    if mod in ("SR", "DOC", "KO", "PR", "OT", "SC", "REPORT"):
        return True

    # Palabras clave descriptivas a excluir de reconstrucción volumétrica 3D
    non_volume_keywords = [
        "topogram", "scout", "localizer", "surview", "pilot",
        "statistic", "estadistica", "estadística",
        "dose report", "report", "protocol", "protocolo",
        "key_images", "key images", "captura", "screen save", "screenshot",
        "analysis", "análisis",
        "fusion", "fused", "fusión", "merged",
        "range-",
    ]

    for kw in non_volume_keywords:
        if kw in desc:
            return True

    if n_imgs is not None and int(n_imgs) == 1:
        if any(w in desc for w in ["topo", "local", "scout", "pilot", "view", "ref"]):
            return True

    return False


def _pair_key(pair):
    """Identificador estable de un par fusionable, usado como clave en larmornium.conf."""
    raw = "%s|%s|%s" % (
        pair.get("study_instance_uid", ""),
        pair.get("ct_series_instance_uid", ""),
        pair.get("pet_series_instance_uid", ""),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _series_key(series_instance_uid):
    """Identificador hash para series individuales."""
    return hashlib.sha1(str(series_instance_uid).encode("utf-8")).hexdigest()[:16]


def load_fusion_pairs(db_path):
    """Lee todos los pares fusionables del índice combinado."""
    if not os.path.isfile(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        table_names = {
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "pet_ct_fusion_pairs" not in table_names:
            return []
        rows = conn.execute(
            "SELECT fp.*, s.patient_id, s.patient_name, s.study_description, "
            "s.study_date FROM pet_ct_fusion_pairs fp "
            "LEFT JOIN pet_ct_studies s ON s.study_instance_uid = fp.study_instance_uid"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def load_fusion_pairs_for_study(db_path, study_instance_uid):
    """Lee los pares fusionables correspondientes a un study_instance_uid específico."""
    if not os.path.isfile(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        table_names = {
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "pet_ct_fusion_pairs" not in table_names:
            return []
        rows = conn.execute(
            "SELECT fp.*, s.patient_id, s.patient_name, s.study_description, "
            "s.study_date FROM pet_ct_fusion_pairs fp "
            "LEFT JOIN pet_ct_studies s ON s.study_instance_uid = fp.study_instance_uid "
            "WHERE fp.study_instance_uid = ?",
            (study_instance_uid,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _load_config(config_path):
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_config(config_path, config):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _load_built_pairs(config_path):
    return _load_config(config_path).get(CONFIG_KEY, {})


def _load_built_ct_volumes(config_path):
    return _load_config(config_path).get(CT_CONFIG_KEY, {})


def _load_built_pet_volumes(config_path):
    return _load_config(config_path).get(PET_CONFIG_KEY, {})


def _mark_pair_built(config_path, pair_key, record):
    config = _load_config(config_path)
    built = config.get(CONFIG_KEY, {})
    built[pair_key] = record
    config[CONFIG_KEY] = built
    _save_config(config_path, config)


def _mark_ct_series_built(config_path, series_instance_uid, record):
    config = _load_config(config_path)
    built = config.get(CT_CONFIG_KEY, {})
    built[series_instance_uid] = record
    config[CT_CONFIG_KEY] = built
    _save_config(config_path, config)


def _mark_pet_series_built(config_path, series_instance_uid, record):
    config = _load_config(config_path)
    built = config.get(PET_CONFIG_KEY, {})
    built[series_instance_uid] = record
    config[PET_CONFIG_KEY] = built
    _save_config(config_path, config)


def load_fully_built_study_uids(db_path, config_path):
    """Retorna el conjunto de study_instance_uid que tienen volúmenes fusionados construidos y existentes."""
    built = _load_built_pairs(config_path)
    built_uids = set()
    for key, record in built.items():
        if isinstance(record, dict):
            nii_path = record.get("nii_path", "")
            suid = record.get("study_instance_uid")
            if suid and nii_path and os.path.isfile(nii_path):
                built_uids.add(suid)
    return built_uids


def load_built_pair_keys(config_path):
    """Retorna el conjunto de claves de pares fusionables cuyos volúmenes existen en disco."""
    built = _load_built_pairs(config_path)
    built_keys = set()
    for key, record in built.items():
        if isinstance(record, dict):
            nii_path = record.get("nii_path", "")
            if nii_path and os.path.isfile(nii_path):
                built_keys.add(key)
    return built_keys


def load_built_ct_series_uids(config_path):
    """Retorna el conjunto de series_instance_uid de CT cuyos volúmenes existen en disco."""
    built = _load_built_ct_volumes(config_path)
    return {
        uid for uid, rec in built.items()
        if isinstance(rec, dict) and rec.get("nii_path") and os.path.isfile(rec["nii_path"])
    }


def load_built_pet_series_uids(config_path):
    """Retorna el conjunto de series_instance_uid de PET cuyos volúmenes existen en disco."""
    built = _load_built_pet_volumes(config_path)
    return {
        uid for uid, rec in built.items()
        if isinstance(rec, dict) and rec.get("nii_path") and os.path.isfile(rec["nii_path"])
    }


def is_pair_built(pair, config_path, larmornium_files_dir):
    """Verifica si un par fusionable ya fue generado y sus archivos existen."""
    key = _pair_key(pair)
    built = _load_built_pairs(config_path)
    if key not in built:
        return False
    record = built[key]
    nii_path = record.get("nii_path", "")
    return bool(nii_path and os.path.isfile(nii_path))


def ensure_fusion_volume_for_pair(pair, dicom_root, larmornium_files_dir, config_path,
                                  progress_callback=None):
    """
    Genera el volumen fusionado (.nii.gz y .json) para un par individual si no
    existe aún, y registra el resultado en larmornium.conf.
    """
    fusion_vol_dir = os.path.join(larmornium_files_dir, FUSION_VOL_DIRNAME)
    os.makedirs(fusion_vol_dir, exist_ok=True)

    key = _pair_key(pair)
    built = _load_built_pairs(config_path)
    if key in built and os.path.isfile(built[key].get("nii_path", "")):
        return key, built[key]

    description = "%s + %s" % (
        pair.get("ct_series_description") or "CT",
        pair.get("pet_series_description") or "PET",
    )
    if progress_callback:
        progress_callback("Generando volumen fusionado: %s ..." % description)

    nii_path = os.path.join(fusion_vol_dir, key + ".nii.gz")
    json_path = os.path.join(fusion_vol_dir, key + ".json")

    def _report_slice_progress(slice_index, total_slices, description=description):
        if progress_callback and total_slices:
            progress_callback("  %s: corte %d/%d" % (description, slice_index, total_slices))

    fusion_pet_ct.fuse_and_save_pair(
        ct_dir=pair["ct_directory"],
        pet_dir=pair["pet_directory"],
        dicom_root=dicom_root,
        output_nii_path=nii_path,
        output_json_path=json_path,
        pair_metadata=pair,
        progress_callback=_report_slice_progress,
    )

    record = {
        "study_instance_uid": pair["study_instance_uid"],
        "ct_series_instance_uid": pair["ct_series_instance_uid"],
        "pet_series_instance_uid": pair["pet_series_instance_uid"],
        "nii_path": os.path.abspath(nii_path),
        "json_path": os.path.abspath(json_path),
    }
    _mark_pair_built(config_path, key, record)

    if progress_callback:
        progress_callback("Volumen fusionado guardado: %s" % os.path.basename(nii_path))

    return key, record


def ensure_ct_volume_for_series(series, dicom_root, larmornium_files_dir, config_path,
                                progress_callback=None):
    """
    Genera el volumen 3D (.nii.gz y .json) en unidades HU para una serie CT y lo registra en larmornium.conf.
    """
    if is_non_volume_series(series):
        return None, None

    import nibabel as nib
    series_uid = series.get("series_instance_uid")
    if not series_uid:
        return None, None

    built = _load_built_ct_volumes(config_path)
    if series_uid in built and os.path.isfile(built[series_uid].get("nii_path", "")):
        return series_uid, built[series_uid]

    ct_dir = series.get("series_directory", "")
    ct_abs_dir = os.path.join(dicom_root, ct_dir) if not os.path.isabs(ct_dir) else ct_dir
    if not os.path.isdir(ct_abs_dir):
        raise FileNotFoundError(f"Directorio de serie CT no encontrado: {ct_abs_dir}")

    files_z = fusion_pet_ct._list_dicom_files_sorted_by_z(ct_abs_dir)
    if not files_z:
        raise ValueError(f"No se encontraron imágenes DICOM en: {ct_abs_dir}")

    ct_vol_dir = os.path.join(larmornium_files_dir, CT_VOL_DIRNAME)
    os.makedirs(ct_vol_dir, exist_ok=True)
    key = _series_key(series_uid)
    nii_path = os.path.join(ct_vol_dir, key + ".nii.gz")
    json_path = os.path.join(ct_vol_dir, key + ".json")

    slices = []
    z_positions = []
    pixel_spacing = [1.0, 1.0]
    slice_thickness = 1.0
    total = len(files_z)

    for idx, (fpath, z_pos) in enumerate(files_z):
        ds = pydicom.dcmread(fpath, force=True)
        meta = fusion_pet_ct.extract_spatial_metadata(ds)
        hu = ds.pixel_array.astype(np.float32) * meta["rescale_slope"] + meta["rescale_intercept"]
        slices.append(hu)
        z_positions.append(float(z_pos))
        if idx == 0:
            pixel_spacing = [float(v) for v in meta["pixel_spacing"]]
            slice_thickness = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
        if progress_callback:
            progress_callback("Generando volumen CT: corte %d/%d" % (idx + 1, total))

    ct_volume = np.stack(slices, axis=0)
    if len(z_positions) > 1:
        z_diff = abs(float(z_positions[1]) - float(z_positions[0]))
        if z_diff > 0:
            slice_thickness = z_diff
    ct_transposed = np.transpose(ct_volume, (2, 1, 0))

    affine = np.diag([pixel_spacing[1], pixel_spacing[0], slice_thickness, 1.0]).astype(np.float64)
    nib.save(nib.Nifti1Image(ct_transposed, affine), nii_path)

    meta_dict = {
        "series_instance_uid": series_uid,
        "study_instance_uid": series.get("study_instance_uid", ""),
        "patient_id": series.get("patient_id", ""),
        "patient_name": series.get("patient_name", ""),
        "series_description": series.get("series_description", "CT"),
        "modality": "CT",
        "num_slices": total,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "z_positions": z_positions,
        "nii_path": os.path.abspath(nii_path),
        "json_path": os.path.abspath(json_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    _mark_ct_series_built(config_path, series_uid, meta_dict)
    return series_uid, meta_dict


def ensure_pet_volume_for_series(series, dicom_root, larmornium_files_dir, config_path,
                                 progress_callback=None):
    """
    Genera el volumen 3D (.nii.gz y .json) en unidades SUV para una serie PET y lo registra en larmornium.conf.
    """
    if is_non_volume_series(series):
        return None, None
    import nibabel as nib
    series_uid = series.get("series_instance_uid")
    if not series_uid:
        return None, None

    built = _load_built_pet_volumes(config_path)
    if series_uid in built and os.path.isfile(built[series_uid].get("nii_path", "")):
        return series_uid, built[series_uid]

    pet_dir = series.get("series_directory", "")
    pet_abs_dir = os.path.join(dicom_root, pet_dir) if not os.path.isabs(pet_dir) else pet_dir
    if not os.path.isdir(pet_abs_dir):
        raise FileNotFoundError(f"Directorio de serie PET no encontrado: {pet_abs_dir}")

    files_z = fusion_pet_ct._list_dicom_files_sorted_by_z(pet_abs_dir)
    if not files_z:
        raise ValueError(f"No se encontraron imágenes DICOM en: {pet_abs_dir}")

    pet_vol_dir = os.path.join(larmornium_files_dir, PET_VOL_DIRNAME)
    os.makedirs(pet_vol_dir, exist_ok=True)
    key = _series_key(series_uid)
    nii_path = os.path.join(pet_vol_dir, key + ".nii.gz")
    json_path = os.path.join(pet_vol_dir, key + ".json")

    slices = []
    z_positions = []
    pixel_spacing = [1.0, 1.0]
    slice_thickness = 1.0
    total = len(files_z)

    for idx, (fpath, z_pos) in enumerate(files_z):
        ds = pydicom.dcmread(fpath, force=True)
        meta = fusion_pet_ct.extract_spatial_metadata(ds)
        suv = ds.pixel_array.astype(np.float32) * meta["rescale_slope"] + meta["rescale_intercept"]
        if meta["suv_factor"] > 0:
            suv *= meta["suv_factor"]
        slices.append(suv)
        z_positions.append(float(z_pos))
        if idx == 0:
            pixel_spacing = [float(v) for v in meta["pixel_spacing"]]
            slice_thickness = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
        if progress_callback:
            progress_callback("Generando volumen PET: corte %d/%d" % (idx + 1, total))

    pet_volume = np.stack(slices, axis=0)
    if len(z_positions) > 1:
        z_diff = abs(float(z_positions[1]) - float(z_positions[0]))
        if z_diff > 0:
            slice_thickness = z_diff
    max_suv = float(np.nanmax(pet_volume)) if pet_volume.size > 0 else 1.0
    pet_transposed = np.transpose(pet_volume, (2, 1, 0))

    affine = np.diag([pixel_spacing[1], pixel_spacing[0], slice_thickness, 1.0]).astype(np.float64)
    nib.save(nib.Nifti1Image(pet_transposed, affine), nii_path)

    meta_dict = {
        "series_instance_uid": series_uid,
        "study_instance_uid": series.get("study_instance_uid", ""),
        "patient_id": series.get("patient_id", ""),
        "patient_name": series.get("patient_name", ""),
        "series_description": series.get("series_description", "PET"),
        "modality": "PET",
        "num_slices": total,
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "z_positions": z_positions,
        "max_suv": max_suv,
        "pet_max_suv": max_suv,
        "pet_units": "SUV",
        "nii_path": os.path.abspath(nii_path),
        "json_path": os.path.abspath(json_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    _mark_pet_series_built(config_path, series_uid, meta_dict)
    return series_uid, meta_dict


def ensure_fusion_volumes_for_study(study_instance_uid, dicom_root, db_path,
                                    larmornium_files_dir, config_path,
                                    progress_callback=None):
    """
    Genera los volúmenes fusionados de todos los pares pertenecientes a un estudio específico.
    """
    pairs = load_fusion_pairs_for_study(db_path, study_instance_uid)
    results = []
    for pair in pairs:
        key, record = ensure_fusion_volume_for_pair(
            pair, dicom_root, larmornium_files_dir, config_path, progress_callback
        )
        results.append((key, record, pair))
    return results


def load_fused_volume_data(record_or_key, larmornium_files_dir, dicom_root=None, pair=None):
    """
    Carga los arreglos de volumen CT (HU), PET (SUV/Actividad) y metadatos desde el archivo NIfTI .nii.gz.
    """
    import nibabel as nib

    if isinstance(record_or_key, str):
        key = record_or_key
        fusion_vol_dir = os.path.join(larmornium_files_dir, FUSION_VOL_DIRNAME)
        json_path = os.path.join(fusion_vol_dir, key + ".json")
        nii_path = os.path.join(fusion_vol_dir, key + ".nii.gz")
    else:
        json_path = record_or_key.get("json_path", "")
        nii_path = record_or_key.get("nii_path", "")

    metadata = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    if os.path.isfile(nii_path):
        nii = nib.load(nii_path)
        data = nii.get_fdata().astype(np.float32)

        if data.ndim == 4 and data.shape[-1] == 2:
            ct_volume = np.transpose(data[..., 0], (2, 1, 0))
            pet_volume = np.transpose(data[..., 1], (2, 1, 0))
        elif data.ndim == 4 and data.shape[-1] >= 3:
            rgb_vol = np.transpose(data, (2, 1, 0, 3))
            ct_volume = rgb_vol[..., 0] * 400.0 + 40.0
            pet_volume = rgb_vol[..., 1]
        elif data.ndim == 3:
            ct_volume = np.transpose(data, (2, 1, 0))
            pet_volume = np.zeros_like(ct_volume)
        else:
            raise ValueError("Formato de dimensiones NIfTI no reconocido: %s" % str(data.shape))

        max_suv = metadata.get("max_suv") or metadata.get("pet_max_suv")
        if max_suv is None or float(max_suv) <= 0:
            max_suv = float(np.nanmax(pet_volume)) if pet_volume.size > 0 else 1.0
        else:
            max_suv = float(max_suv)

        pet_vmax = max_suv

        pet_units = str(metadata.get("pet_units", "SUV"))
        z_positions = metadata.get("z_positions", [float(i) for i in range(ct_volume.shape[0])])
        pixel_spacing = metadata.get("pixel_spacing", [1.0, 1.0])
        if len(z_positions) > 1:
            z_diff = abs(float(z_positions[1]) - float(z_positions[0]))
            slice_thickness = z_diff if z_diff > 0 else float(metadata.get("slice_thickness") or 1.0)
        else:
            slice_thickness = float(metadata.get("slice_thickness") or 1.0)

        return {
            "ct_volume": ct_volume,
            "pet_volume": pet_volume,
            "pet_vmax": pet_vmax,
            "max_suv": max_suv,
            "pet_max_suv": max_suv,
            "pet_units": pet_units,
            "z_positions": z_positions,
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
            "metadata": metadata,
            "num_slices": ct_volume.shape[0],
            "output_shape": (ct_volume.shape[1], ct_volume.shape[2]),
        }

    if pair and dicom_root and os.path.isdir(dicom_root):
        _, record = ensure_fusion_volume_for_pair(
            pair, dicom_root, larmornium_files_dir,
            os.path.join(larmornium_files_dir, "larmornium.conf")
        )
        return load_fused_volume_data(record, larmornium_files_dir)

    raise FileNotFoundError("No se encontró el archivo del volumen fusionado: %s" % nii_path)


def load_single_volume_data(record, modality="CT"):
    """
    Carga los arreglos de volumen 3D y metadatos de un archivo NIfTI de serie individual (CT o PET).
    """
    import nibabel as nib
    nii_path = record.get("nii_path", "")
    json_path = record.get("json_path", "")

    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"No se encontró el archivo NIfTI: {nii_path}")

    metadata = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)

    if data.ndim == 3:
        vol = np.transpose(data, (2, 1, 0))
    elif data.ndim == 4:
        vol = np.transpose(data[..., 0], (2, 1, 0))
    else:
        raise ValueError("Dimensiones no soportadas: %s" % str(data.shape))

    pixel_spacing = metadata.get("pixel_spacing", [1.0, 1.0])
    z_positions = metadata.get("z_positions", [float(i) for i in range(vol.shape[0])])
    if len(z_positions) > 1:
        z_diff = abs(float(z_positions[1]) - float(z_positions[0]))
        slice_thickness = z_diff if z_diff > 0 else float(metadata.get("slice_thickness") or 1.0)
    else:
        slice_thickness = float(metadata.get("slice_thickness") or 1.0)
    max_suv = float(metadata.get("max_suv") or (np.nanmax(vol) if vol.size > 0 else 1.0))

    return {
        "volume": vol,
        "modality": modality.upper(),
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "z_positions": z_positions,
        "max_suv": max_suv,
        "metadata": metadata,
        "num_slices": vol.shape[0],
    }


def ensure_fusion_volumes(dicom_root, db_path, larmornium_files_dir, config_path,
                          progress_callback=None):
    """
    Genera los volúmenes fusionados pendientes para todos los pares detectados en el índice.
    """
    pairs = load_fusion_pairs(db_path)
    built = _load_built_pairs(config_path)
    new_count = 0

    for pair in pairs:
        key = _pair_key(pair)
        if key in built and os.path.isfile(built[key].get("nii_path", "")):
            continue

        try:
            ensure_fusion_volume_for_pair(
                pair, dicom_root, larmornium_files_dir, config_path, progress_callback
            )
            new_count += 1
        except Exception as exc:
            logger.warning("Error al fusionar par %s: %s", key, exc)
            if progress_callback:
                progress_callback("Error al fusionar: %s" % exc)

    return new_count
