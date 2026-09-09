#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import json
import logging
import os
import sys

import numpy as np

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
_PROCESSING_DIR = os.path.join(_PROJECT_ROOT, "processing")
if _PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _PROCESSING_DIR)

import gen_volume_MRI  # noqa: E402

logger = logging.getLogger("join_mri")

MRI_VOL_DIRNAME = "mri_vol"
MRI_CONFIG_KEY = "mri_volumes"


def _mri_key(study_or_series_uid):
    return hashlib.sha1(str(study_or_series_uid).encode("utf-8")).hexdigest()[:16]


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


def _load_built_mri_volumes(config_path):
    return _load_config(config_path).get(MRI_CONFIG_KEY, {})


def _mark_mri_volume_built(config_path, key, record):
    config = _load_config(config_path)
    built = config.get(MRI_CONFIG_KEY, {})
    built[key] = record
    config[MRI_CONFIG_KEY] = built
    _save_config(config_path, config)


def load_built_mri_series_uids(config_path):
    built = _load_built_mri_volumes(config_path)
    uids = set()
    for k, rec in built.items():
        if isinstance(rec, dict) and rec.get("nii_path") and os.path.isfile(rec["nii_path"]):
            ser_uid = rec.get("series_instance_uid")
            if ser_uid:
                uids.add(ser_uid)
            else:
                uids.add(k)
    return uids


def load_built_mri_study_uids(config_path):
    built = _load_built_mri_volumes(config_path)
    uids = set()
    for k, rec in built.items():
        if isinstance(rec, dict) and rec.get("nii_path") and os.path.isfile(rec["nii_path"]):
            st_uid = rec.get("study_instance_uid")
            if st_uid:
                uids.add(st_uid)
    return uids


def load_built_mri_uids(config_path):
    return load_built_mri_series_uids(config_path) | load_built_mri_study_uids(config_path)


def resolve_mri_directory(target_dict, dicom_root):
    mri_dir = target_dict.get("series_directory") or target_dict.get("study_directory") or ""
    if not mri_dir:
        slice_dirs = target_dict.get("slice_directories")
        if isinstance(slice_dirs, str):
            try:
                dirs = json.loads(slice_dirs)
                if dirs and isinstance(dirs, list):
                    mri_dir = dirs[0]
            except Exception:
                pass
        elif isinstance(slice_dirs, list) and slice_dirs:
            mri_dir = slice_dirs[0]

    # Si es ruta absoluta existente, retornarla
    if mri_dir and os.path.isabs(mri_dir) and (os.path.isdir(mri_dir) or os.path.isfile(mri_dir)):
        return mri_dir

    # Probar combinaciones respecto a dicom_root
    if dicom_root:
        candidates = [
            os.path.join(dicom_root, mri_dir) if mri_dir else "",
            os.path.join(dicom_root, mri_dir[4:]) if mri_dir and mri_dir.startswith("MRI/") else "",
            os.path.join(dicom_root, "MRI", mri_dir) if mri_dir else "",
        ]
        for c in candidates:
            if c and (os.path.isdir(c) or os.path.isfile(c)):
                return c

    return mri_dir or dicom_root


def ensure_mri_volume(target_dict, dicom_root, larmornium_files_dir, config_path,
                      progress_callback=None):
    series_uid = target_dict.get("series_instance_uid", "")
    study_uid = target_dict.get("study_instance_uid", "")
    # Priorizar siempre series_uid
    uid = series_uid or study_uid
    if not uid:
        return None, None

    built = _load_built_mri_volumes(config_path)

    # Si se especificó series_uid, buscar estrictamente por series_uid
    if series_uid and series_uid in built and os.path.isfile(built[series_uid].get("nii_path", "")):
        return series_uid, built[series_uid]
    # Si solo se especificó study_uid
    if not series_uid and study_uid and study_uid in built and os.path.isfile(built[study_uid].get("nii_path", "")):
        return study_uid, built[study_uid]

    mri_path = resolve_mri_directory(target_dict, dicom_root)

    mri_vol_dir = os.path.join(larmornium_files_dir, MRI_VOL_DIRNAME)
    os.makedirs(mri_vol_dir, exist_ok=True)
    key = _mri_key(uid)
    nii_path = os.path.join(mri_vol_dir, key + ".nii.gz")
    json_path = os.path.join(mri_vol_dir, key + ".json")

    desc = target_dict.get("series_description") or target_dict.get("study_description") or "MRI"
    if progress_callback:
        progress_callback("Generando volumen MRI: %s ..." % desc)

    _nifti_img, gen_metadata = gen_volume_MRI.generate_mri_volume(
        mri_directory=mri_path,
        dicom_root=dicom_root,
        output_path=nii_path,
    )

    voxel_sp = gen_metadata.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
    col_sp = float(voxel_sp[0]) if len(voxel_sp) >= 1 else 1.0
    row_sp = float(voxel_sp[1]) if len(voxel_sp) >= 2 else 1.0
    slice_thickness = float(voxel_sp[2]) if len(voxel_sp) >= 3 else 1.0

    dimensions = gen_metadata.get("dimensions", [0, 0, 0])
    num_slices = gen_metadata.get("num_slices") or (dimensions[2] if len(dimensions) >= 3 else 0)
    z_positions = [i * slice_thickness for i in range(num_slices)]

    meta_dict = {
        "study_instance_uid": study_uid,
        "series_instance_uid": series_uid,
        "patient_id": target_dict.get("patient_id") or gen_metadata.get("patient_id", ""),
        "patient_name": target_dict.get("patient_name") or gen_metadata.get("patient_name", ""),
        "study_description": target_dict.get("study_description") or gen_metadata.get("study_description", ""),
        "series_description": target_dict.get("series_description") or gen_metadata.get("series_description", desc),
        "modality": "MRI",
        "num_slices": num_slices,
        "voxel_spacing": [col_sp, row_sp, slice_thickness],
        "pixel_spacing": [row_sp, col_sp],
        "slice_thickness": slice_thickness,
        "z_positions": z_positions,
        "intensity_min": float(gen_metadata.get("intensity_min", 0.0)),
        "intensity_max": float(gen_metadata.get("intensity_max", 1.0)),
        "nii_path": os.path.abspath(nii_path),
        "json_path": os.path.abspath(json_path),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    # Registrar bajo el UID específico (series_uid preferido)
    _mark_mri_volume_built(config_path, uid, meta_dict)

    if progress_callback:
        progress_callback("Volumen MRI generado: %s" % os.path.basename(nii_path))

    return uid, meta_dict


def load_mri_volume_data(record):
    import nibabel as nib
    nii_path = record.get("nii_path", "")
    json_path = record.get("json_path", "")

    if not os.path.isfile(nii_path):
        raise FileNotFoundError(f"No se encontró el archivo NIfTI MRI: {nii_path}")

    metadata = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)

    # NIfTI guardado por gen_volume_MRI tiene forma [cols, rows, slices] (nx, ny, nz)
    if data.ndim == 3:
        vol = np.transpose(data, (2, 1, 0))  # [slices, rows, cols]
    elif data.ndim == 4:
        vol = np.transpose(data[..., 0], (2, 1, 0))
    else:
        raise ValueError("Dimensiones de volumen MRI no soportadas: %s" % str(data.shape))

    voxel_spacing = metadata.get("voxel_spacing") or metadata.get("voxel_spacing_mm") or record.get("voxel_spacing")
    if not voxel_spacing:
        zooms = nii.header.get_zooms()
        voxel_spacing = [float(v) for v in zooms[:3]]

    col_sp = float(voxel_spacing[0]) if len(voxel_spacing) > 0 else 1.0
    row_sp = float(voxel_spacing[1]) if len(voxel_spacing) > 1 else 1.0
    slice_sp = float(voxel_spacing[2]) if len(voxel_spacing) > 2 else 1.0

    num_slices = vol.shape[0]
    z_positions = metadata.get("z_positions") or [i * slice_sp for i in range(num_slices)]

    v_min = float(metadata.get("intensity_min", np.nanmin(vol) if vol.size > 0 else 0.0))
    v_max = float(metadata.get("intensity_max", np.nanmax(vol) if vol.size > 0 else 1.0))

    return {
        "volume": vol,
        "modality": "MRI",
        "voxel_spacing": [col_sp, row_sp, slice_sp],
        "pixel_spacing": [row_sp, col_sp],
        "slice_thickness": slice_sp,
        "z_positions": z_positions,
        "num_slices": num_slices,
        "intensity_min": v_min,
        "intensity_max": v_max,
        "metadata": metadata,
    }
