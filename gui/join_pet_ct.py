#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_pet_ct.py - Construye y gestiona los volumenes fusionados CT+PET

Gestiona la generacion bajo demanda de volumenes fusionados (.nii.gz, .json, .npz)
para pares fusionables detectados en el indice combinado y almacena el registro
en la seccion fusion_volumes del archivo larmornium.conf.
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys

import numpy as np

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
_PROCESSING_DIR = os.path.join(_PROJECT_ROOT, "processing")
if _PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _PROCESSING_DIR)

import fusion_pet_ct  # noqa: E402

logger = logging.getLogger("join_pet_ct")

FUSION_VOL_DIRNAME = "fusion_vol"
CONFIG_KEY = "fusion_volumes"


def _pair_key(pair):
    """Identificador estable de un par fusionable, usado como clave en larmornium.conf."""
    raw = "%s|%s|%s" % (
        pair.get("study_instance_uid", ""),
        pair.get("ct_series_instance_uid", ""),
        pair.get("pet_series_instance_uid", ""),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_fusion_pairs(db_path):
    """Lee todos los pares fusionables del indice combinado."""
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
    """Lee los pares fusionables correspondientes a un study_instance_uid especifico."""
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


def _mark_pair_built(config_path, pair_key, record):
    config = _load_config(config_path)
    built = config.get(CONFIG_KEY, {})
    built[pair_key] = record
    config[CONFIG_KEY] = built
    _save_config(config_path, config)


def load_fully_built_study_uids(db_path, config_path):
    """Retorna el conjunto de study_instance_uid que tienen volumenes fusionados
    construidos y existentes en disco (usado para anteponer [ok] en la GUI)."""
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
    """Retorna el conjunto de claves de pares fusionables cuyos volumenes existen en disco."""
    built = _load_built_pairs(config_path)
    built_keys = set()
    for key, record in built.items():
        if isinstance(record, dict):
            nii_path = record.get("nii_path", "")
            if nii_path and os.path.isfile(nii_path):
                built_keys.add(key)
    return built_keys


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
    existe aun, y registra el resultado en larmornium.conf.
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


def ensure_fusion_volumes_for_study(study_instance_uid, dicom_root, db_path,
                                    larmornium_files_dir, config_path,
                                    progress_callback=None):
    """
    Genera los volumenes fusionados de todos los pares pertenecientes a un estudio especifico.
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

        pet_vmax = metadata.get("pet_vmax")
        if pet_vmax is None or float(pet_vmax) <= 0:
            positive_pet = pet_volume[pet_volume > 0]
            pet_vmax = float(np.percentile(positive_pet, 99.5)) if len(positive_pet) > 0 else 1.0
        else:
            pet_vmax = float(pet_vmax)

        pet_units = str(metadata.get("pet_units", "SUV"))
        z_positions = metadata.get("z_positions", [float(i) for i in range(ct_volume.shape[0])])
        pixel_spacing = metadata.get("pixel_spacing", [1.0, 1.0])
        slice_thickness = float(metadata.get("slice_thickness") or 1.0)

        return {
            "ct_volume": ct_volume,
            "pet_volume": pet_volume,
            "pet_vmax": pet_vmax,
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

    raise FileNotFoundError("No se encontro el archivo del volumen fusionado: %s" % nii_path)


def ensure_fusion_volumes(dicom_root, db_path, larmornium_files_dir, config_path,
                          progress_callback=None):
    """
    Genera los volumenes fusionados pendientes para todos los pares detectados en el indice.
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
