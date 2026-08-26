#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_pet_ct.py — Indexador de estudios DICOM PET/CT
=====================================================

Recorre la carpeta DICOM/PET_CT/, lee cada archivo DICOM (sin extensión)
con pydicom, y genera:
  - pet_ct_index.db  : Base de datos SQLite con metadata de los estudios
  - pet_ct_tree.json : Árbol de directorios en formato JSON

Uso (a traves de larmornium.py):
    python3 larmornium.py index-pet-ct --dicom-dir ./DICOM
    python3 larmornium.py index-pet-ct --dicom-dir ./DICOM --output-dir ./output --verbose

Los archivos PET/CT provienen de un solo hospital (UNAM, equipo Siemens
Biograph64_Vision 600) y ninguno tiene extensión de archivo.
"""

import collections
import json
import logging
import os
import sqlite3
from datetime import datetime

import pydicom
from pydicom.errors import InvalidDicomError

# Logging
logger = logging.getLogger("index_pet_ct")

# Constantes — Archivos / carpetas a ignorar
IGNORE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("._",)
# Extensiones de archivos que NO son DICOM (resultados guardados, etc.)
NON_DICOM_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".fig", ".tif",
                        ".tiff", ".mat", ".xlsx", ".csv", ".txt"}

# Esquema SQLite
SCHEMA_SQL = """
-- Tabla de estudios (nivel paciente / estudio)
CREATE TABLE IF NOT EXISTS studies (
    study_instance_uid       TEXT PRIMARY KEY,
    study_date               TEXT,
    study_time               TEXT,
    study_description        TEXT,
    accession_number         TEXT,
    patient_name             TEXT,
    patient_id               TEXT,
    patient_birth_date       TEXT,
    patient_sex              TEXT,
    patient_age              TEXT,
    patient_weight           REAL,
    patient_size             REAL,
    institution_name         TEXT,
    institution_address      TEXT,
    manufacturer             TEXT,
    manufacturer_model_name  TEXT,
    station_name             TEXT,
    device_serial_number     TEXT,
    software_versions        TEXT,
    protocol_name            TEXT,
    has_fusion_pairs         INTEGER DEFAULT 0,
    is_multi_study           INTEGER DEFAULT 0,
    num_slice_directories    INTEGER DEFAULT 1,
    slice_directories        TEXT
);

-- Tabla de pacientes (agrupa estudios por paciente / fantoma)
CREATE TABLE IF NOT EXISTS patients (
    patient_id               TEXT PRIMARY KEY,
    patient_name             TEXT,
    patient_directory        TEXT,
    num_studies              INTEGER DEFAULT 1,
    is_multi_study           INTEGER DEFAULT 0,
    slice_thickness          REAL,
    study_uids               TEXT,
    study_directories        TEXT,
    total_slice_directories  INTEGER DEFAULT 0
);

-- Tabla de series
CREATE TABLE IF NOT EXISTS series (
    series_instance_uid  TEXT PRIMARY KEY,
    study_instance_uid   TEXT NOT NULL,
    modality             TEXT,
    series_date          TEXT,
    series_time          TEXT,
    series_description   TEXT,
    series_number        INTEGER,
    body_part_examined   TEXT,
    patient_position     TEXT,
    num_images           INTEGER DEFAULT 0,
    FOREIGN KEY (study_instance_uid) REFERENCES studies(study_instance_uid)
);

-- Tabla de imágenes (un registro por slice / archivo)
CREATE TABLE IF NOT EXISTS images (
    sop_instance_uid                      TEXT PRIMARY KEY,
    series_instance_uid                   TEXT NOT NULL,
    file_path                             TEXT NOT NULL,
    instance_number                       INTEGER,
    image_type                            TEXT,
    rows                                  INTEGER,
    columns                               INTEGER,
    bits_allocated                        INTEGER,
    bits_stored                           INTEGER,
    pixel_representation                  INTEGER,
    pixel_spacing                         TEXT,
    pixel_spacing_x                       REAL,
    pixel_spacing_y                       REAL,
    slice_thickness                       REAL,
    image_position_patient                TEXT,
    image_position_x                      REAL,
    image_position_y                      REAL,
    image_position_z                      REAL,
    image_orientation_patient             TEXT,
    reconstruction_target_center_patient TEXT,
    reconstruction_target_center_x        REAL,
    reconstruction_target_center_y        REAL,
    reconstruction_target_center_z        REAL,
    rescale_slope                         REAL,
    rescale_intercept                     REAL,
    rescale_type                          TEXT,
    FOREIGN KEY (series_instance_uid) REFERENCES series(series_instance_uid)
);

-- Parámetros específicos de CT
CREATE TABLE IF NOT EXISTS ct_parameters (
    sop_instance_uid             TEXT PRIMARY KEY,
    kvp                          REAL,
    x_ray_tube_current           REAL,
    exposure                     REAL,
    convolution_kernel           TEXT,
    reconstruction_diameter      REAL,
    data_collection_diameter     REAL,
    filter_type                  TEXT,
    ctdi_vol                     REAL,
    single_collimation_width     REAL,
    total_collimation_width      REAL,
    gantry_detector_tilt         REAL,
    table_height                 REAL,
    distance_source_to_detector  REAL,
    distance_source_to_patient   REAL,
    spiral_pitch_factor          REAL,
    table_speed                  REAL,
    exposure_time                REAL,
    FOREIGN KEY (sop_instance_uid) REFERENCES images(sop_instance_uid)
);

-- Parámetros específicos de PET
CREATE TABLE IF NOT EXISTS pet_parameters (
    sop_instance_uid               TEXT PRIMARY KEY,
    units                          TEXT,
    counts_source                  TEXT,
    decay_correction               TEXT,
    reconstruction_method          TEXT,
    scatter_correction_method      TEXT,
    attenuation_correction_method  TEXT,
    number_of_slices               INTEGER,
    FOREIGN KEY (sop_instance_uid) REFERENCES images(sop_instance_uid)
);

-- Información del radiofármaco (por estudio)
CREATE TABLE IF NOT EXISTS radiopharmaceutical_info (
    id                                    INTEGER PRIMARY KEY AUTOINCREMENT,
    study_instance_uid                    TEXT NOT NULL,
    radiopharmaceutical                   TEXT,
    radionuclide_total_dose               REAL,
    radionuclide_half_life                REAL,
    radionuclide_positron_fraction        REAL,
    radiopharmaceutical_start_time        TEXT,
    radiopharmaceutical_start_datetime    TEXT,
    radiopharmaceutical_stop_datetime     TEXT,
    radionuclide_code_value              TEXT,
    radionuclide_code_meaning            TEXT,
    FOREIGN KEY (study_instance_uid) REFERENCES studies(study_instance_uid)
);

-- Pares de directorios CT/PET fusionables
-- Un par fusionable es aquel donde el directorio CT y el directorio PET
-- tienen el mismo número de cortes (slices) y los cortes están alineados
-- correctamente en el eje Z (mismo rango Z y mismo slice thickness).
CREATE TABLE IF NOT EXISTS fusion_pairs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    study_instance_uid       TEXT NOT NULL,
    ct_series_instance_uid   TEXT NOT NULL,
    pet_series_instance_uid  TEXT NOT NULL,
    ct_series_description    TEXT,
    pet_series_description   TEXT,
    ct_directory             TEXT NOT NULL,
    pet_directory            TEXT NOT NULL,
    num_slices               INTEGER NOT NULL,
    slice_thickness          REAL,
    ct_z_min                 REAL,
    ct_z_max                 REAL,
    pet_z_min                REAL,
    pet_z_max                REAL,
    ct_pixel_spacing_x       REAL,
    ct_pixel_spacing_y       REAL,
    pet_pixel_spacing_x      REAL,
    pet_pixel_spacing_y      REAL,
    ct_rows                  INTEGER,
    ct_columns               INTEGER,
    pet_rows                 INTEGER,
    pet_columns              INTEGER,
    is_fusionable            INTEGER DEFAULT 1,
    FOREIGN KEY (study_instance_uid) REFERENCES studies(study_instance_uid),
    FOREIGN KEY (ct_series_instance_uid) REFERENCES series(series_instance_uid),
    FOREIGN KEY (pet_series_instance_uid) REFERENCES series(series_instance_uid)
);
"""


# Funciones auxiliares de extracción
def _safe_get(ds, attr, default=None):
    """Obtener un atributo de un dataset DICOM de forma segura."""
    val = getattr(ds, attr, default)
    if val is None or val == "":
        return default
    # Convertir tipos DICOM especiales a tipos nativos de Python
    if isinstance(val, pydicom.uid.UID):
        return str(val)
    if isinstance(val, pydicom.valuerep.PersonName):
        return str(val)
    if isinstance(val, pydicom.valuerep.DSfloat):
        return float(val)
    if isinstance(val, pydicom.valuerep.IS):
        try:
            return int(val)
        except (ValueError, TypeError):
            return str(val)
    if isinstance(val, (pydicom.multival.MultiValue, list)):
        return [str(v) for v in val]
    return val


def _safe_float(ds, attr, default=None):
    """Obtener un float de un dataset DICOM."""
    val = getattr(ds, attr, None)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(ds, attr, default=None):
    """Obtener un int de un dataset DICOM."""
    val = getattr(ds, attr, None)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_str(ds, attr, default=None):
    """Obtener un string de un dataset DICOM."""
    val = getattr(ds, attr, None)
    if val is None or val == "":
        return default
    return str(val)



# Extracción de metadata DICOM
def extract_study_info(ds):
    """Extrae información a nivel estudio de un dataset DICOM."""
    return {
        "study_instance_uid": _safe_str(ds, "StudyInstanceUID"),
        "study_date": _safe_str(ds, "StudyDate"),
        "study_time": _safe_str(ds, "StudyTime"),
        "study_description": _safe_str(ds, "StudyDescription"),
        "accession_number": _safe_str(ds, "AccessionNumber"),
        "patient_name": _safe_str(ds, "PatientName"),
        "patient_id": _safe_str(ds, "PatientID"),
        "patient_birth_date": _safe_str(ds, "PatientBirthDate"),
        "patient_sex": _safe_str(ds, "PatientSex"),
        "patient_age": _safe_str(ds, "PatientAge"),
        "patient_weight": _safe_float(ds, "PatientWeight"),
        "patient_size": _safe_float(ds, "PatientSize"),
        "institution_name": _safe_str(ds, "InstitutionName"),
        "institution_address": _safe_str(ds, "InstitutionAddress"),
        "manufacturer": _safe_str(ds, "Manufacturer"),
        "manufacturer_model_name": _safe_str(ds, "ManufacturerModelName"),
        "station_name": _safe_str(ds, "StationName"),
        "device_serial_number": _safe_str(ds, "DeviceSerialNumber"),
        "software_versions": str(_safe_get(ds, "SoftwareVersions", "")),
        "protocol_name": _safe_str(ds, "ProtocolName"),
    }


def extract_series_info(ds):
    """Extrae información a nivel serie de un dataset DICOM."""
    return {
        "series_instance_uid": _safe_str(ds, "SeriesInstanceUID"),
        "study_instance_uid": _safe_str(ds, "StudyInstanceUID"),
        "modality": _safe_str(ds, "Modality"),
        "series_date": _safe_str(ds, "SeriesDate"),
        "series_time": _safe_str(ds, "SeriesTime"),
        "series_description": _safe_str(ds, "SeriesDescription"),
        "series_number": _safe_int(ds, "SeriesNumber"),
        "body_part_examined": _safe_str(ds, "BodyPartExamined"),
        "patient_position": _safe_str(ds, "PatientPosition"),
    }


def extract_image_info(ds, file_path):
    """Extrae información a nivel imagen de un dataset DICOM."""
    # Pixel spacing (0028, 0030)
    ps = _safe_get(ds, "PixelSpacing")
    ps_x = float(ps[0]) if ps and len(ps) >= 1 else None
    ps_y = float(ps[1]) if ps and len(ps) >= 2 else None
    ps_str = json.dumps([ps_x, ps_y]) if ps_x is not None and ps_y is not None else None

    # Image position patient (0020, 0032)
    ipp = _safe_get(ds, "ImagePositionPatient")
    ip_x = float(ipp[0]) if ipp and len(ipp) >= 1 else None
    ip_y = float(ipp[1]) if ipp and len(ipp) >= 2 else None
    ip_z = float(ipp[2]) if ipp and len(ipp) >= 3 else None
    ipp_str = json.dumps([ip_x, ip_y, ip_z]) if ip_x is not None and ip_y is not None and ip_z is not None else None

    # Reconstruction target center patient (0018, 9313)
    rtcp = getattr(ds, "ReconstructionTargetCenterPatient", None)
    if rtcp is None and (0x0018, 0x9313) in ds:
        try:
            rtcp = ds[(0x0018, 0x9313)].value
        except Exception:
            rtcp = None

    if rtcp and len(rtcp) >= 3:
        rtc_x = float(rtcp[0])
        rtc_y = float(rtcp[1])
        rtc_z = float(rtcp[2])
        rtcp_str = json.dumps([rtc_x, rtc_y, rtc_z])
    elif rtcp and len(rtcp) == 2:
        rtc_x = float(rtcp[0])
        rtc_y = float(rtcp[1])
        rtc_z = None
        rtcp_str = json.dumps([rtc_x, rtc_y])
    else:
        rtc_x, rtc_y, rtc_z, rtcp_str = None, None, None, None

    # Image orientation
    iop = _safe_get(ds, "ImageOrientationPatient")
    iop_str = json.dumps([float(v) for v in iop]) if iop else None

    # Image type
    it = _safe_get(ds, "ImageType")
    it_str = "\\".join(str(v) for v in it) if it else None

    return {
        "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
        "series_instance_uid": _safe_str(ds, "SeriesInstanceUID"),
        "file_path": file_path,
        "instance_number": _safe_int(ds, "InstanceNumber"),
        "image_type": it_str,
        "rows": _safe_int(ds, "Rows"),
        "columns": _safe_int(ds, "Columns"),
        "bits_allocated": _safe_int(ds, "BitsAllocated"),
        "bits_stored": _safe_int(ds, "BitsStored"),
        "pixel_representation": _safe_int(ds, "PixelRepresentation"),
        "pixel_spacing": ps_str,
        "pixel_spacing_x": ps_x,
        "pixel_spacing_y": ps_y,
        "slice_thickness": _safe_float(ds, "SliceThickness"),
        "image_position_patient": ipp_str,
        "image_position_x": ip_x,
        "image_position_y": ip_y,
        "image_position_z": ip_z,
        "image_orientation_patient": iop_str,
        "reconstruction_target_center_patient": rtcp_str,
        "reconstruction_target_center_x": rtc_x,
        "reconstruction_target_center_y": rtc_y,
        "reconstruction_target_center_z": rtc_z,
        "rescale_slope": _safe_float(ds, "RescaleSlope"),
        "rescale_intercept": _safe_float(ds, "RescaleIntercept"),
        "rescale_type": _safe_str(ds, "RescaleType"),
    }


def extract_ct_parameters(ds):
    """Extrae parámetros específicos de CT."""
    return {
        "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
        "kvp": _safe_float(ds, "KVP"),
        "x_ray_tube_current": _safe_float(ds, "XRayTubeCurrent"),
        "exposure": _safe_float(ds, "Exposure"),
        "convolution_kernel": _safe_str(ds, "ConvolutionKernel"),
        "reconstruction_diameter": _safe_float(ds, "ReconstructionDiameter"),
        "data_collection_diameter": _safe_float(ds, "DataCollectionDiameter"),
        "filter_type": _safe_str(ds, "FilterType"),
        "ctdi_vol": _safe_float(ds, "CTDIvol"),
        "single_collimation_width": _safe_float(ds, "SingleCollimationWidth"),
        "total_collimation_width": _safe_float(ds, "TotalCollimationWidth"),
        "gantry_detector_tilt": _safe_float(ds, "GantryDetectorTilt"),
        "table_height": _safe_float(ds, "TableHeight"),
        "distance_source_to_detector": _safe_float(ds, "DistanceSourceToDetector"),
        "distance_source_to_patient": _safe_float(ds, "DistanceSourceToPatient"),
        "spiral_pitch_factor": _safe_float(ds, "SpiralPitchFactor"),
        "table_speed": _safe_float(ds, "TableSpeed"),
        "exposure_time": _safe_float(ds, "ExposureTime"),
    }


def extract_pet_parameters(ds):
    """Extrae parámetros específicos de PET."""
    return {
        "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
        "units": _safe_str(ds, "Units"),
        "counts_source": _safe_str(ds, "CountsSource"),
        "decay_correction": _safe_str(ds, "DecayCorrection"),
        "reconstruction_method": _safe_str(ds, "ReconstructionMethod"),
        "scatter_correction_method": _safe_str(ds, "ScatterCorrectionMethod"),
        "attenuation_correction_method": _safe_str(ds, "AttenuationCorrectionMethod"),
        "number_of_slices": _safe_int(ds, "NumberOfSlices"),
    }


def extract_radiopharmaceutical_info(ds, study_uid):
    """
    Extrae información del radiofármaco desde RadiopharmaceuticalInformationSequence.

    Esta información es crucial para la calibración PET (cálculo de SUV):
    - Dosis total del radionúclido
    - Vida media del radionúclido
    - Hora de inyección
    """
    rps = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not rps:
        return None

    results = []
    for rp in rps:
        # Extraer código del radionúclido
        rnc_val = None
        rnc_meaning = None
        rncs = getattr(rp, "RadionuclideCodeSequence", None)
        if rncs:
            for rnc in rncs:
                rnc_val = _safe_str(rnc, "CodeValue")
                rnc_meaning = _safe_str(rnc, "CodeMeaning")

        results.append({
            "study_instance_uid": study_uid,
            "radiopharmaceutical": _safe_str(rp, "Radiopharmaceutical"),
            "radionuclide_total_dose": _safe_float(rp, "RadionuclideTotalDose"),
            "radionuclide_half_life": _safe_float(rp, "RadionuclideHalfLife"),
            "radionuclide_positron_fraction": _safe_float(rp, "RadionuclidePositronFraction"),
            "radiopharmaceutical_start_time": _safe_str(rp, "RadiopharmaceuticalStartTime"),
            "radiopharmaceutical_start_datetime": _safe_str(rp, "RadiopharmaceuticalStartDateTime"),
            "radiopharmaceutical_stop_datetime": _safe_str(rp, "RadiopharmaceuticalStopDateTime"),
            "radionuclide_code_value": rnc_val,
            "radionuclide_code_meaning": rnc_meaning,
        })
    return results


# Inserción en SQLite
def _insert_or_ignore(cursor, table, data_dict):
    """INSERT OR IGNORE genérico para un diccionario."""
    cols = ", ".join(data_dict.keys())
    placeholders = ", ".join(["?"] * len(data_dict))
    sql = f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})"
    cursor.execute(sql, list(data_dict.values()))


def _insert_auto(cursor, table, data_dict):
    """INSERT para tablas con AUTOINCREMENT (sin conflicto de PK)."""
    # Quitar 'id' si es autoincrement
    d = {k: v for k, v in data_dict.items() if k != "id"}
    cols = ", ".join(d.keys())
    placeholders = ", ".join(["?"] * len(d))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    cursor.execute(sql, list(d.values()))


# Construcción del árbol de directorios JSON
def build_directory_tree(root_path, dicom_root, file_info_map):
    """
    Construye un arbol de directorios como diccionario anidado.
    """
    basename = os.path.basename(root_path)
    rel_path = os.path.relpath(root_path, dicom_root)

    node = {
        "name": basename,
        "type": "directory",
        "relative_path": rel_path,
        "children": [],
    }

    try:
        entries = sorted(os.listdir(root_path))
    except PermissionError:
        logger.warning("Sin permisos para leer: %s", root_path)
        return node

    for entry in entries:
        entry_path = os.path.join(root_path, entry)

        # Ignorar archivos del sistema
        if entry in IGNORE_FILES or any(entry.startswith(p) for p in IGNORE_PREFIXES):
            continue

        if os.path.isdir(entry_path):
            # Recursar subdirectorios
            child = build_directory_tree(entry_path, dicom_root, file_info_map)
            node["children"].append(child)

        elif os.path.isfile(entry_path):
            _, ext = os.path.splitext(entry)
            child_rel = os.path.relpath(entry_path, dicom_root)

            if ext.lower() in NON_DICOM_EXTENSIONS:
                # CASO: Archivos no-DICOM (PNGs de fusiones guardadas, etc.)
                # Se registran en el árbol pero marcados como no-DICOM.
                child_node = {
                    "name": entry,
                    "type": "file",
                    "relative_path": child_rel,
                    "file_type": "non_dicom",
                    "extension": ext.lower(),
                }
            else:
                # CASO: Archivos DICOM sin extensión (CT000xxx, PT000xxx)
                # Son el formato principal de PET/CT de este hospital.
                info = file_info_map.get(entry_path, {})
                child_node = {
                    "name": entry,
                    "type": "file",
                    "relative_path": child_rel,
                    "file_type": "dicom_no_extension",
                    "modality": info.get("modality"),
                    "sop_instance_uid": info.get("sop_instance_uid"),
                    "series_description": info.get("series_description"),
                    "pixel_spacing": info.get("pixel_spacing"),
                    "image_position_patient": info.get("image_position_patient"),
                    "reconstruction_target_center_patient": info.get("reconstruction_target_center_patient"),
                }

            node["children"].append(child_node)

    return node


# Detección de pares de fusión CT/PET
def find_fusion_pairs(cursor, dicom_dir):
    """
    Identifica pares de series CT y PET que pueden fusionarse en base a
    su pertenencia al mismo estudio y su alineacion espacial en el eje Z.
    """
    cursor.execute("""
        SELECT
            s.series_instance_uid,
            s.study_instance_uid,
            s.modality,
            s.series_description,
            s.num_images,
            (SELECT i.file_path
             FROM images i WHERE i.series_instance_uid = s.series_instance_uid LIMIT 1) AS sample_file_path,
            (SELECT MIN(i.image_position_z) FROM images i WHERE i.series_instance_uid = s.series_instance_uid) AS z_min,
            (SELECT MAX(i.image_position_z) FROM images i WHERE i.series_instance_uid = s.series_instance_uid) AS z_max,
            (SELECT i.slice_thickness FROM images i WHERE i.series_instance_uid = s.series_instance_uid LIMIT 1) AS slice_thickness,
            (SELECT i.pixel_spacing_x FROM images i WHERE i.series_instance_uid = s.series_instance_uid LIMIT 1) AS px_x,
            (SELECT i.pixel_spacing_y FROM images i WHERE i.series_instance_uid = s.series_instance_uid LIMIT 1) AS px_y,
            (SELECT i.rows FROM images i WHERE i.series_instance_uid = s.series_instance_uid LIMIT 1) AS img_rows,
            (SELECT i.columns FROM images i WHERE i.series_instance_uid = s.series_instance_uid LIMIT 1) AS img_cols
        FROM series s
        WHERE s.modality IN ('CT', 'PT')
          AND s.num_images > 1
        ORDER BY s.study_instance_uid, s.modality
    """)
    rows = cursor.fetchall()

    study_dir_series = collections.defaultdict(lambda: {"CT": [], "PT": []})

    for row in rows:
        (series_uid, study_uid, modality, series_desc, num_images,
         sample_file_path, z_min, z_max, slice_thickness,
         px_x, px_y, img_rows, img_cols) = row

        if sample_file_path is None:
            continue

        series_dir = os.path.dirname(sample_file_path)
        parts = series_dir.split("/") if "/" in series_dir else series_dir.split(os.sep)
        parent_dir = "/".join(parts[:-1]) if len(parts) > 1 else series_dir

        desc_upper = (series_desc or "").upper()

        if modality == "CT":
            if any(skip in desc_upper for skip in [
                "TOPOGRAM", "DOSE REPORT", "PATIENT PROTOCOL",
                "RANGE-CT", "RANGE_CT", "STATISTICS", "KEY_IMAGES"
            ]):
                continue
            if num_images <= 1:
                continue

        elif modality == "PT":
            if any(skip in desc_upper for skip in [
                "UNCORRECTED", "STATISTICS", "DOSE REPORT", "KEY_IMAGES", "MU MAP"
            ]):
                continue
            if num_images <= 1:
                continue

        key = (study_uid, parent_dir)
        info = {
            "series_uid": series_uid,
            "series_desc": series_desc or "",
            "num_images": num_images,
            "series_dir": series_dir,
            "z_min": z_min,
            "z_max": z_max,
            "slice_thickness": slice_thickness,
            "px_x": px_x,
            "px_y": px_y,
            "rows": img_rows,
            "cols": img_cols,
        }
        study_dir_series[key][modality].append(info)

    fusion_pairs = []
    studies_with_pairs = set()

    for (study_uid, parent_dir), modalities in study_dir_series.items():
        ct_series = modalities["CT"]
        pt_series = modalities["PT"]

        if not ct_series or not pt_series:
            continue

        for ct in ct_series:
            for pt in pt_series:
                ct_z_min = ct["z_min"]
                ct_z_max = ct["z_max"]
                pt_z_min = pt["z_min"]
                pt_z_max = pt["z_max"]

                if any(v is None for v in [ct_z_min, ct_z_max, pt_z_min, pt_z_max]):
                    continue

                ct_range = sorted([ct_z_min, ct_z_max])
                pt_range = sorted([pt_z_min, pt_z_max])

                overlap_min = max(ct_range[0], pt_range[0])
                overlap_max = min(ct_range[1], pt_range[1])
                overlap_len = overlap_max - overlap_min

                if overlap_len <= 10.0:
                    continue

                if ct["num_images"] == pt["num_images"]:
                    num_slices = ct["num_images"]
                else:
                    num_slices = min(ct["num_images"], pt["num_images"])

                ct_st = ct["slice_thickness"]
                pt_st = pt["slice_thickness"]
                pair_st = pt_st if pt_st is not None else ct_st

                pair_data = {
                    "study_instance_uid": study_uid,
                    "ct_series_instance_uid": ct["series_uid"],
                    "pet_series_instance_uid": pt["series_uid"],
                    "ct_series_description": ct["series_desc"],
                    "pet_series_description": pt["series_desc"],
                    "ct_directory": ct["series_dir"],
                    "pet_directory": pt["series_dir"],
                    "num_slices": num_slices,
                    "slice_thickness": pair_st,
                    "ct_z_min": ct_z_min,
                    "ct_z_max": ct_z_max,
                    "pet_z_min": pt_z_min,
                    "pet_z_max": pt_z_max,
                    "ct_pixel_spacing_x": ct["px_x"],
                    "ct_pixel_spacing_y": ct["px_y"],
                    "pet_pixel_spacing_x": pt["px_x"],
                    "pet_pixel_spacing_y": pt["px_y"],
                    "ct_rows": ct["rows"],
                    "ct_columns": ct["cols"],
                    "pet_rows": pt["rows"],
                    "pet_columns": pt["cols"],
                    "is_fusionable": 1,
                }

                d = {k: v for k, v in pair_data.items() if k != "id"}
                cols = ", ".join(d.keys())
                placeholders = ", ".join(["?"] * len(d))
                cursor.execute(
                    f"INSERT INTO fusion_pairs ({cols}) VALUES ({placeholders})",
                    list(d.values())
                )

                studies_with_pairs.add(study_uid)

                pair_data["summary"] = {
                    "ct_directory": ct["series_dir"],
                    "pet_directory": pt["series_dir"],
                    "ct_series_description": ct["series_desc"],
                    "pet_series_description": pt["series_desc"],
                    "num_slices": num_slices,
                    "slice_thickness": pair_st,
                    "ct_pixel_spacing": [ct["px_x"], ct["px_y"]],
                    "pet_pixel_spacing": [pt["px_x"], pt["px_y"]],
                    "ct_dimensions": [ct["rows"], ct["cols"]],
                    "pet_dimensions": [pt["rows"], pt["cols"]],
                    "z_range": [
                        min(ct_z_min or 0, pt_z_min or 0),
                        max(ct_z_max or 0, pt_z_max or 0),
                    ],
                }

                fusion_pairs.append(pair_data)

                logger.info("  Par fusionable: CT=%s (%d slices) <-> PET=%s (%d slices) en %s",
                            ct["series_desc"], ct["num_images"],
                            pt["series_desc"], pt["num_images"],
                            parent_dir)

    for study_uid in studies_with_pairs:
        cursor.execute(
            "UPDATE studies SET has_fusion_pairs = 1 WHERE study_instance_uid = ?",
            (study_uid,)
        )

    return fusion_pairs


def annotate_tree_with_fusion_pairs(tree_node, fusion_pairs):
    """
    Anota los nodos del arbol JSON con informacion de pares de fusion.
    """
    ct_dirs = {}
    pet_dirs = {}
    for fp in fusion_pairs:
        ct_dir = fp.get("ct_directory", "")
        pet_dir = fp.get("pet_directory", "")
        ct_dirs[ct_dir] = fp
        pet_dirs[pet_dir] = fp

    def _annotate_recursive(node):
        rel_path = node.get("relative_path", "")

        if node.get("type") == "directory":
            if rel_path in ct_dirs:
                fp = ct_dirs[rel_path]
                node["fusion_role"] = "ct"
                node["fusion_pair_directory"] = fp.get("pet_directory")
                node["fusion_pair_description"] = fp.get("pet_series_description")
                node["fusion_num_slices"] = fp.get("num_slices")
                node["fusion_pair_id"] = fusion_pairs.index(fp)

            elif rel_path in pet_dirs:
                fp = pet_dirs[rel_path]
                node["fusion_role"] = "pet"
                node["fusion_pair_directory"] = fp.get("ct_directory")
                node["fusion_pair_description"] = fp.get("ct_series_description")
                node["fusion_num_slices"] = fp.get("num_slices")
                node["fusion_pair_id"] = fusion_pairs.index(fp)

            for child in node.get("children", []):
                _annotate_recursive(child)

    _annotate_recursive(tree_node)


def update_studies_multi_study_info(cursor):
    """
    Analiza la base de datos y actualiza:
      1. En la tabla 'studies':
         - is_multi_study        : 1 si el estudio contiene > 1 directorio de cortes de CT/PET, 0 en caso contrario.
         - num_slice_directories : numero de directorios de cortes distintos asociados al estudio.
         - slice_directories     : JSON array con las rutas relativas de dichos directorios.
      2. En la tabla 'patients':
         - Registra y agrupa los estudios por paciente / fantoma.
         - num_studies           : cantidad de estudios asociados al paciente.
         - is_multi_study        : 1 si num_studies > 1, 0 en caso contrario.
         - study_uids            : lista JSON con los StudyInstanceUIDs.
         - study_directories     : lista JSON con las carpetas de estudios del paciente.
         - total_slice_directories: conteo total de directorios de cortes.
    """
    cursor.execute("""
        SELECT
            se.study_instance_uid,
            i.file_path,
            st.patient_id,
            st.patient_name,
            st.study_description
        FROM images i
        JOIN series se ON i.series_instance_uid = se.series_instance_uid
        JOIN studies st ON se.study_instance_uid = st.study_instance_uid
    """)
    rows = cursor.fetchall()

    study_dirs = {}
    study_meta = {}
    for study_uid, file_path, pid, pname, sdesc in rows:
        if not study_uid or not file_path:
            continue
        s_dir = os.path.dirname(file_path)
        if study_uid not in study_dirs:
            study_dirs[study_uid] = set()
            study_meta[study_uid] = {"pid": pid, "pname": pname, "sdesc": sdesc}
        study_dirs[study_uid].add(s_dir)

    for study_uid, dir_set in study_dirs.items():
        sorted_dirs = sorted(list(dir_set))
        num_dirs = len(sorted_dirs)
        is_multi = 1 if num_dirs > 1 else 0
        cursor.execute("""
            UPDATE studies
            SET is_multi_study = ?,
                num_slice_directories = ?,
                slice_directories = ?
            WHERE study_instance_uid = ?
        """, (is_multi, num_dirs, json.dumps(sorted_dirs), study_uid))

    cursor.execute("""
        SELECT
            se.study_instance_uid,
            AVG(i.slice_thickness) as avg_st,
            COUNT(*) as img_count
        FROM series se
        JOIN images i ON se.series_instance_uid = i.series_instance_uid
        WHERE i.slice_thickness IS NOT NULL AND i.slice_thickness > 0
          AND LOWER(se.series_description) NOT LIKE '%topogram%'
          AND LOWER(se.series_description) NOT LIKE '%scout%'
          AND LOWER(se.series_description) NOT LIKE '%dose%'
          AND LOWER(se.series_description) NOT LIKE '%statistic%'
        GROUP BY se.study_instance_uid, se.series_instance_uid
        ORDER BY img_count DESC
    """)
    st_rows = cursor.fetchall()
    study_st_map = {}
    for suid, avg_st, img_count in st_rows:
        if suid not in study_st_map and avg_st:
            study_st_map[suid] = round(avg_st, 2)

    cursor.execute("""
        SELECT
            study_instance_uid,
            patient_id,
            patient_name,
            study_description,
            slice_directories
        FROM studies
    """)
    study_rows = cursor.fetchall()

    patients_map = {}
    for suid, pid, pname, sdesc, sdirs_json in study_rows:
        sdirs = json.loads(sdirs_json) if sdirs_json else []
        p_dir = ""
        study_folder = ""
        if sdirs:
            first_sdir = sdirs[0]
            if os.path.basename(first_sdir).startswith("SE0") or os.path.basename(first_sdir).startswith("ST0"):
                study_folder = os.path.dirname(first_sdir)
            else:
                study_folder = first_sdir

            parts = sdirs[0].split("/")
            if len(parts) >= 3:
                if parts[1] == "PACIENTES":
                    p_dir = "/".join(parts[:3])
                else:
                    p_dir = "/".join(parts[:2])
            else:
                p_dir = os.path.dirname(sdirs[0])

        pid_clean = (pid or "").strip()
        pname_clean = (pname or "").strip()
        if pid_clean and not (pid_clean.startswith("23.") or pid_clean.startswith("24.")):
            patient_key = pid_clean
            final_pname = pname_clean or pid_clean
        elif p_dir and ("fantoma" in p_dir.lower() or "fwhm" in p_dir.lower()):
            patient_key = os.path.basename(p_dir)
            final_pname = patient_key
        else:
            patient_key = pid_clean or pname_clean or p_dir or "UNKNOWN_PATIENT"
            final_pname = pname_clean or pid_clean or patient_key

        if patient_key not in patients_map:
            patients_map[patient_key] = {
                "patient_id": patient_key,
                "patient_name": final_pname,
                "patient_directories": set(),
                "study_uids": [],
                "studies_items": [],
                "all_slice_dirs": set(),
            }

        if p_dir:
            patients_map[patient_key]["patient_directories"].add(p_dir)
        patients_map[patient_key]["study_uids"].append(suid)
        patients_map[patient_key]["studies_items"].append({
            "study_uid": suid,
            "study_folder": study_folder or (sdirs[0] if sdirs else suid),
            "study_description": sdesc or "",
            "slice_thickness": study_st_map.get(suid),
            "slice_directories": sdirs,
        })
        patients_map[patient_key]["all_slice_dirs"].update(sdirs)

    cursor.execute("DELETE FROM patients")
    for pkey in sorted(patients_map.keys()):
        pinfo = patients_map[pkey]
        st_groups = collections.defaultdict(list)
        for st_item in pinfo["studies_items"]:
            th = st_item["slice_thickness"]
            st_groups[th].append(st_item)

        best_th = None
        best_studies = []
        for th, grp in st_groups.items():
            if len(grp) >= 2 and len(grp) > len(best_studies):
                best_th = th
                best_studies = grp

        is_multi = 1 if len(best_studies) > 1 else 0
        if is_multi:
            matching_uids = [s["study_uid"] for s in best_studies]
            matching_folders = [s["study_folder"] for s in best_studies]
            n_studies = len(best_studies)
            chosen_th = best_th
        else:
            matching_uids = pinfo["study_uids"]
            matching_folders = [s["study_folder"] for s in pinfo["studies_items"]]
            n_studies = len(pinfo["studies_items"])
            chosen_th = pinfo["studies_items"][0]["slice_thickness"] if pinfo["studies_items"] else None

        cursor.execute("""
            INSERT OR REPLACE INTO patients (
                patient_id, patient_name, patient_directory,
                num_studies, is_multi_study, slice_thickness, study_uids,
                study_directories, total_slice_directories
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pinfo["patient_id"],
            pinfo["patient_name"],
            "; ".join(sorted(list(pinfo["patient_directories"]))),
            n_studies,
            is_multi,
            chosen_th,
            json.dumps(matching_uids),
            json.dumps(sorted(matching_folders)),
            len(pinfo["all_slice_dirs"])
        ))


def annotate_tree_with_multi_study(tree_node):
    """
    Anota recursivamente los nodos del arbol JSON con informacion de multi-estudio y multi-paciente.
    """
    def _traverse(node):
        if node.get("type") != "directory":
            return [], []

        children = node.get("children", [])
        direct_files = [c for c in children if c.get("type") == "file"]
        subdirs = [c for c in children if c.get("type") == "directory"]

        medical_files = [
            f for f in direct_files
            if f.get("file_type") in ("dicom_no_extension", "dicom_dcm", "dicom_no_extension_enhanced", "analyze_hdr", "analyze_img")
            or f.get("modality") is not None
        ]
        is_slice_dir = len(medical_files) > 0
        node["is_slice_directory"] = is_slice_dir
        node["num_slices"] = len(medical_files) if is_slice_dir else 0

        direct_modalities = set(f.get("modality") for f in direct_files if f.get("modality"))
        descendant_slice_dirs = []
        descendant_modalities = set(direct_modalities)

        if is_slice_dir:
            descendant_slice_dirs.append(node.get("relative_path"))

        for sd in subdirs:
            sd_slice_dirs, sd_mods = _traverse(sd)
            descendant_slice_dirs.extend(sd_slice_dirs)
            descendant_modalities.update(sd_mods)

        num_slice_dirs = len(descendant_slice_dirs)
        node["slice_directories"] = sorted(list(set(descendant_slice_dirs)))
        node["num_slice_directories"] = num_slice_dirs
        node["is_multi_study"] = bool(num_slice_dirs > 1)
        node["modalities_present"] = sorted(list(descendant_modalities))

        studies_contained = []
        for sd in subdirs:
            if sd.get("slice_directories") or sd.get("is_slice_directory"):
                studies_contained.append({
                    "name": sd.get("name"),
                    "relative_path": sd.get("relative_path"),
                    "slice_directories": sd.get("slice_directories", []),
                    "num_slice_directories": sd.get("num_slice_directories", 0),
                    "modalities_present": sd.get("modalities_present", []),
                })

        if len(studies_contained) > 1 and node.get("name") not in ("PET_CT", "MRI", "PACIENTES"):
            for st_item in studies_contained:
                sd_node = next((sd for sd in subdirs if sd.get("name") == st_item["name"]), None)
                st_item_th = None
                if sd_node:
                    def _find_th(n):
                        for c in n.get("children", []):
                            if c.get("type") == "file" and c.get("slice_thickness") and c.get("slice_thickness") > 0:
                                desc = (c.get("series_description") or "").lower()
                                if "topogram" not in desc and "scout" not in desc and "dose" not in desc:
                                    return round(float(c.get("slice_thickness")), 2)
                            elif c.get("type") == "directory":
                                th = _find_th(c)
                                if th is not None:
                                    return th
                        return None
                    st_item_th = _find_th(sd_node)
                st_item["slice_thickness"] = st_item_th

            st_groups = collections.defaultdict(list)
            for st_item in studies_contained:
                st_groups[st_item["slice_thickness"]].append(st_item)

            best_th = None
            best_studies = []
            for th, grp in st_groups.items():
                if len(grp) >= 2 and len(grp) > len(best_studies):
                    best_th = th
                    best_studies = grp

            if len(best_studies) > 1:
                node["is_patient_node"] = True
                node["num_studies"] = len(best_studies)
                node["studies"] = best_studies
                node["slice_thickness"] = best_th
            else:
                node["is_patient_node"] = False
        elif len(studies_contained) == 1 and not is_slice_dir:
            node["is_patient_node"] = True
            node["num_studies"] = 1
            node["studies"] = studies_contained

        return descendant_slice_dirs, descendant_modalities

    _traverse(tree_node)


def index_pet_ct(dicom_dir, output_dir=None, verbose=False):
    """
    Indexa los estudios PET/CT dentro de la carpeta DICOM.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pet_ct_root = os.path.join(dicom_dir, "PET_CT")
    if not os.path.isdir(pet_ct_root):
        msg = "No se encontro la carpeta PET_CT en: %s" % dicom_dir
        logger.error(msg)
        raise FileNotFoundError(msg)

    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "pet_ct_index.db")
    json_path = os.path.join(output_dir, "pet_ct_tree.json")

    # Crear / conectar base de datos
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)
    conn.commit()

    # Recolectar todos los archivos
    logger.info("Buscando archivos en: %s", pet_ct_root)
    all_files = []
    for dirpath, _dirnames, filenames in os.walk(pet_ct_root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            all_files.append(fpath)

    logger.info("Total de archivos encontrados: %d", len(all_files))

    # Procesar archivos DICOM
    processed = 0
    skipped = 0
    errors = 0
    studies_seen = set()
    series_seen = set()
    radiopharm_studies_seen = set()
    series_image_count = {}  # series_uid -> conteo de imágenes
    file_info_map = {}  # filepath -> info para el JSON

    for fpath in all_files:
        basename = os.path.basename(fpath)

        # Filtrar archivos del sistema
        if basename in IGNORE_FILES or any(basename.startswith(p) for p in IGNORE_PREFIXES):
            skipped += 1
            continue

        # Filtrar extensiones no-DICOM
        _, ext = os.path.splitext(basename)
        if ext.lower() in NON_DICOM_EXTENSIONS:
            skipped += 1
            logger.debug("Ignorando archivo no-DICOM: %s", fpath)
            continue

        # CASO: Archivo DICOM sin extensión (formato PET/CT)
        # Los archivos se nombran CT000xxx (para CT) o PT000xxx (para PET).
        # No tienen extensión .dcm — se leen directamente con pydicom.
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
        except (InvalidDicomError, Exception) as e:
            errors += 1
            logger.warning("Error leyendo DICOM: %s — %s", fpath, e)
            continue

        # Verificar que tiene campos mínimos DICOM
        if not hasattr(ds, "SOPInstanceUID") or not hasattr(ds, "Modality"):
            errors += 1
            logger.warning("Archivo sin metadata DICOM válida: %s", fpath)
            continue

        rel_path = os.path.relpath(fpath, dicom_dir)
        modality = _safe_str(ds, "Modality", "UNKNOWN")
        study_uid = _safe_str(ds, "StudyInstanceUID")
        series_uid = _safe_str(ds, "SeriesInstanceUID")
        sop_uid = _safe_str(ds, "SOPInstanceUID")

        # Insertar estudio (si es nuevo)
        if study_uid and study_uid not in studies_seen:
            study_info = extract_study_info(ds)
            _insert_or_ignore(cursor, "studies", study_info)
            studies_seen.add(study_uid)
            logger.debug("Nuevo estudio: %s — %s", study_uid,
                         study_info.get("study_description"))

        # Insertar serie (si es nueva)
        if series_uid and series_uid not in series_seen:
            series_info = extract_series_info(ds)
            _insert_or_ignore(cursor, "series", series_info)
            series_seen.add(series_uid)
            series_image_count[series_uid] = 0
            logger.debug("Nueva serie: %s — %s (%s)", series_uid,
                         series_info.get("series_description"), modality)

        # Insertar imagen
        image_info = extract_image_info(ds, rel_path)
        _insert_or_ignore(cursor, "images", image_info)
        if series_uid in series_image_count:
            series_image_count[series_uid] += 1

        # Guardar info para el JSON (incluyendo ImagePositionPatient, ReconstructionTargetCenterPatient, PixelSpacing)
        file_info_map[fpath] = {
            "modality": modality,
            "sop_instance_uid": sop_uid,
            "series_description": _safe_str(ds, "SeriesDescription"),
            "pixel_spacing": [image_info["pixel_spacing_x"], image_info["pixel_spacing_y"]] if image_info["pixel_spacing_x"] is not None else None,
            "image_position_patient": [image_info["image_position_x"], image_info["image_position_y"], image_info["image_position_z"]] if image_info["image_position_x"] is not None else None,
            "reconstruction_target_center_patient": [image_info["reconstruction_target_center_x"], image_info["reconstruction_target_center_y"], image_info["reconstruction_target_center_z"]] if image_info["reconstruction_target_center_x"] is not None else None,
        }

        # Insertar parámetros CT
        if modality == "CT":
            ct_params = extract_ct_parameters(ds)
            _insert_or_ignore(cursor, "ct_parameters", ct_params)

        # Insertar parámetros PET
        elif modality == "PT":
            pet_params = extract_pet_parameters(ds)
            _insert_or_ignore(cursor, "pet_parameters", pet_params)

            # Extraer info radiofarmacéutica (una vez por estudio)
            if study_uid and study_uid not in radiopharm_studies_seen:
                rp_list = extract_radiopharmaceutical_info(ds, study_uid)
                if rp_list:
                    for rp_info in rp_list:
                        _insert_auto(cursor, "radiopharmaceutical_info", rp_info)
                    radiopharm_studies_seen.add(study_uid)

        processed += 1
        if verbose and processed % 500 == 0:
            logger.info("Procesados: %d / %d archivos...", processed, len(all_files))

    # Actualizar conteo de imágenes por serie
    for series_uid, count in series_image_count.items():
        cursor.execute(
            "UPDATE series SET num_images = ? WHERE series_instance_uid = ?",
            (count, series_uid)
        )

    conn.commit()

    # Detectar pares de fusión CT/PET
    logger.info("Detectando pares de fusión CT/PET...")
    fusion_pairs = find_fusion_pairs(cursor, dicom_dir)
    num_fusion_pairs = len(fusion_pairs)
    logger.info("Pares de fusión CT/PET encontrados: %d", num_fusion_pairs)

    # Actualizar métricas de multi-estudio en la tabla 'studies'
    logger.info("Actualizando información de multi-estudios en base de datos...")
    update_studies_multi_study_info(cursor)
    conn.commit()

    # Construir árbol JSON
    logger.info("Construyendo árbol de directorios JSON...")
    tree = build_directory_tree(pet_ct_root, dicom_dir, file_info_map)

    # Anotar el árbol con la información de pares de fusión
    annotate_tree_with_fusion_pairs(tree, fusion_pairs)

    # Anotar el árbol con la información de multi-estudio
    annotate_tree_with_multi_study(tree)

    # Conteo de estudios multi-estudio
    cursor.execute("SELECT COUNT(*) FROM studies WHERE is_multi_study = 1")
    total_multi_studies = cursor.fetchone()[0]

    cursor.execute("""
        SELECT study_instance_uid, study_description, is_multi_study, num_slice_directories, slice_directories
        FROM studies
    """)
    studies_summary = []
    for s_uid, s_desc, is_m, n_dirs, dirs_json in cursor.fetchall():
        studies_summary.append({
            "study_instance_uid": s_uid,
            "study_description": s_desc,
            "is_multi_study": bool(is_m),
            "num_slice_directories": n_dirs,
            "slice_directories": json.loads(dirs_json) if dirs_json else [],
        })

    cursor.execute("""
        SELECT patient_id, patient_name, patient_directory, num_studies, is_multi_study, slice_thickness, study_uids, study_directories, total_slice_directories
        FROM patients
        ORDER BY patient_id ASC
    """)
    patients_summary = []
    for pid, pname, pdir, n_std, is_m, st_th, suids_json, sdirs_json, total_sdirs in cursor.fetchall():
        patients_summary.append({
            "patient_id": pid,
            "patient_name": pname,
            "patient_directory": pdir,
            "num_studies": n_std,
            "is_multi_study": bool(is_m),
            "slice_thickness": st_th,
            "study_uids": json.loads(suids_json) if suids_json else [],
            "study_directories": json.loads(sdirs_json) if sdirs_json else [],
            "total_slice_directories": total_sdirs,
        })

    tree_output = {
        "root": "DICOM/PET_CT",
        "scan_date": datetime.now().isoformat(),
        "total_files": processed + skipped,
        "total_dicom_files": processed,
        "total_patients": len(patients_summary),
        "total_studies": len(studies_seen),
        "total_multi_studies": total_multi_studies,
        "total_series": len(series_seen),
        "total_fusion_pairs": num_fusion_pairs,
        "skipped_files": skipped,
        "errors": errors,
        "fusion_pairs": [fp["summary"] for fp in fusion_pairs],
        "patients": patients_summary,
        "studies": studies_summary,
        "tree": tree,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tree_output, f, ensure_ascii=False, indent=2)

    conn.close()

    # Resumen
    logger.info("=" * 60)
    logger.info("PACIENTES Y ESTUDIOS REGISTRADOS (Ordenados por Patient ID)")
    logger.info("=" * 60)
    for p in patients_summary:
        st_tag = f", ST: {p['slice_thickness']:.1f}mm" if p.get("slice_thickness") is not None else ""
        multi_tag = " [MULTI-ESTUDIO]" if p["is_multi_study"] else ""
        logger.info("  - [ID: %s] %s (%d estudio%s%s)%s:",
                    p["patient_id"], p["patient_name"], p["num_studies"],
                    "s" if p["num_studies"] > 1 else "", st_tag, multi_tag)
        for s_dir in p["study_directories"]:
            logger.info("      - %s", s_dir)

    logger.info("=" * 60)
    logger.info("INDEXACIÓN PET/CT COMPLETADA")
    logger.info("=" * 60)
    logger.info("  Archivos DICOM procesados : %d", processed)
    logger.info("  Archivos ignorados        : %d", skipped)
    logger.info("  Errores                   : %d", errors)
    logger.info("  Pacientes registrados     : %d (Multi-estudio: %d)", len(patients_summary), sum(1 for p in patients_summary if p["is_multi_study"]))
    logger.info("  Estudios                  : %d", len(studies_seen))
    logger.info("  Series                    : %d", len(series_seen))
    logger.info("  Pares de fusión CT/PET    : %d", num_fusion_pairs)
    logger.info("  Base de datos             : %s", db_path)
    logger.info("  Árbol JSON                : %s", json_path)

    return db_path, json_path
