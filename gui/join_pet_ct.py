#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_pet_ct.py - Construye los volumenes fusionados CT+PET pendientes

Lee los pares fusionables (tabla pet_ct_fusion_pairs) del indice combinado
generado por index_dicom_all.py y, para cada par que aun no tenga un volumen
fusionado construido, genera el volumen (.nii/.nii.gz) y su .json de
metadatos usando processing/fusion_pet_ct.py, guardandolos en
<larmornium_files>/fusion_vol/.

El registro de que pares ya fueron construidos se guarda en la seccion
"fusion_volumes" del archivo de configuracion larmornium.conf, para no
repetir la fusion en cada apertura del indice.
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys

# Permitir ejecutar/importar este modulo de forma independiente asegurando
# que processing/ (fusion_pet_ct.py) sea importable.
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
    """Identificador estable de un par fusionable, usado como nombre de
    archivo y como clave de registro en larmornium.conf."""
    raw = "%s|%s|%s" % (
        pair.get("study_instance_uid", ""),
        pair.get("ct_series_instance_uid", ""),
        pair.get("pet_series_instance_uid", ""),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_fusion_pairs(db_path):
    """Lee todos los pares fusionables del indice combinado, con la
    informacion de paciente/estudio necesaria para fusionarlos."""
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
    """Retorna el conjunto de study_instance_uid cuyos pares fusionables
    ya tienen TODOS sus volumenes .nii construidos (usado por la GUI para
    anteponer "[ok]" al nombre del estudio en el arbol)."""
    pairs = load_fusion_pairs(db_path)
    if not pairs:
        return set()

    built = _load_built_pairs(config_path)
    pairs_by_study = {}
    for pair in pairs:
        pairs_by_study.setdefault(pair["study_instance_uid"], []).append(_pair_key(pair))

    return {
        study_uid for study_uid, keys in pairs_by_study.items()
        if keys and all(key in built for key in keys)
    }


def ensure_fusion_volumes(dicom_root, db_path, larmornium_files_dir, config_path,
                          progress_callback=None):
    """
    Genera los volumenes fusionados (.nii.gz + .json) pendientes para todos
    los pares fusionables detectados en el indice combinado.

    Parametros
    ----------
    progress_callback : callable(str), optional
        Recibe mensajes de progreso legibles para mostrar en la bitacora.

    Retorna
    -------
    int
        Cantidad de volumenes nuevos generados en esta llamada.
    """
    fusion_vol_dir = os.path.join(larmornium_files_dir, FUSION_VOL_DIRNAME)
    os.makedirs(fusion_vol_dir, exist_ok=True)

    pairs = load_fusion_pairs(db_path)
    built = _load_built_pairs(config_path)
    new_count = 0

    for pair in pairs:
        key = _pair_key(pair)
        if key in built:
            continue

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
                progress_callback(
                    "  %s: corte %d/%d" % (description, slice_index, total_slices)
                )

        try:
            fusion_pet_ct.fuse_and_save_pair(
                ct_dir=pair["ct_directory"],
                pet_dir=pair["pet_directory"],
                dicom_root=dicom_root,
                output_nii_path=nii_path,
                output_json_path=json_path,
                pair_metadata=pair,
                progress_callback=_report_slice_progress,
            )
        except Exception as exc:
            logger.warning("No se pudo fusionar %s: %s", description, exc)
            if progress_callback:
                progress_callback("Error al fusionar %s: %s" % (description, exc))
            continue

        record = {
            "study_instance_uid": pair["study_instance_uid"],
            "ct_series_instance_uid": pair["ct_series_instance_uid"],
            "pet_series_instance_uid": pair["pet_series_instance_uid"],
            "nii_path": nii_path,
            "json_path": json_path,
        }
        _mark_pair_built(config_path, key, record)
        built[key] = record
        new_count += 1
        if progress_callback:
            progress_callback("Volumen fusionado guardado: %s" % os.path.basename(nii_path))

    return new_count
