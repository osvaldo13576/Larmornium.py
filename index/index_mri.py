#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_mri.py — Indexador de estudios DICOM MRI
================================================

Recorre la carpeta DICOM/MRI/, identifica y procesa múltiples formatos
de archivo de imagen médica provenientes de diferentes centros:

  1. DICOM .dcm           — Philips Achieva (UNAM Neurobiología)
  2. DICOM sin extensión  — Philips Ingenia (I.N. Psiquiatría), Enhanced MR
  3. Analyze .img + .hdr  — Formato volumétrico Analyze 7.5
  4. TIFF .tif            — Exportaciones post-procesadas de MRI
  5. Archivos de análisis — .fig, .png, .pdf (solo en JSON, no en DB)

Genera:
  - mri_index.db  : Base de datos SQLite con metadata de los estudios
  - mri_tree.json : Árbol de directorios en formato JSON

Uso (a traves de larmornium.py):
    python3 larmornium.py index-mri --dicom-dir ./DICOM
    python3 larmornium.py index-mri --dicom-dir ./DICOM --output-dir ./output --verbose
"""

import collections
import json
import logging
import os
import sqlite3
import struct
from datetime import datetime

import pydicom
from pydicom.errors import InvalidDicomError

# Intentar importar tifffile para leer metadatos TIFF
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# Logging
logger = logging.getLogger("index_mri")

# Constantes
IGNORE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("._",)

# Extensiones de archivos de análisis / resultados (no médicos)
ANALYSIS_EXTENSIONS = {".fig", ".png", ".jpg", ".jpeg", ".pdf",
                       ".mat", ".xlsx", ".csv", ".txt"}

# Extensiones de archivos de imagen médica reconocidos
MEDICAL_IMAGE_EXTENSIONS = {".dcm", ".img", ".hdr"}

# Tipos de datos Analyze 7.5
ANALYZE_DATATYPES = {
    0: "DT_UNKNOWN", 1: "DT_BINARY", 2: "DT_UNSIGNED_CHAR",
    4: "DT_SIGNED_SHORT", 8: "DT_SIGNED_INT", 16: "DT_FLOAT",
    32: "DT_COMPLEX", 64: "DT_DOUBLE", 128: "DT_RGB",
}

# Esquema SQLite
SCHEMA_SQL = """
-- Tabla de estudios MRI
CREATE TABLE IF NOT EXISTS studies (
    study_instance_uid       TEXT PRIMARY KEY,
    study_date               TEXT,
    study_time               TEXT,
    study_description        TEXT,
    patient_name             TEXT,
    patient_id               TEXT,
    patient_birth_date       TEXT,
    patient_sex              TEXT,
    patient_age              TEXT,
    patient_weight           REAL,
    institution_name         TEXT,
    institution_address      TEXT,
    manufacturer             TEXT,
    manufacturer_model_name  TEXT,
    station_name             TEXT,
    device_serial_number     TEXT,
    magnetic_field_strength  REAL,
    software_versions        TEXT,
    is_multi_study           INTEGER DEFAULT 0,
    num_slice_directories    INTEGER DEFAULT 1,
    slice_directories        TEXT
);

-- Tabla de pacientes MRI
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

-- Tabla de series MRI
CREATE TABLE IF NOT EXISTS series (
    series_instance_uid  TEXT PRIMARY KEY,
    study_instance_uid   TEXT NOT NULL,
    modality             TEXT,
    series_date          TEXT,
    series_time          TEXT,
    series_description   TEXT,
    series_number        INTEGER,
    body_part_examined   TEXT,
    num_images           INTEGER DEFAULT 0,
    FOREIGN KEY (study_instance_uid) REFERENCES studies(study_instance_uid)
);

-- Tabla de imágenes (un registro por archivo DICOM)
CREATE TABLE IF NOT EXISTS images (
    sop_instance_uid          TEXT PRIMARY KEY,
    series_instance_uid       TEXT,
    file_path                 TEXT NOT NULL,
    file_type                 TEXT NOT NULL,
    instance_number           INTEGER,
    image_type                TEXT,
    rows                      INTEGER,
    columns                   INTEGER,
    bits_allocated            INTEGER,
    bits_stored               INTEGER,
    number_of_frames          INTEGER,
    pixel_spacing_x           REAL,
    pixel_spacing_y           REAL,
    slice_thickness           REAL,
    spacing_between_slices    REAL,
    image_position_x          REAL,
    image_position_y          REAL,
    image_position_z          REAL,
    image_orientation_patient TEXT,
    rescale_slope             REAL,
    rescale_intercept         REAL,
    FOREIGN KEY (series_instance_uid) REFERENCES series(series_instance_uid)
);

-- Parámetros específicos de MR
CREATE TABLE IF NOT EXISTS mr_parameters (
    sop_instance_uid          TEXT PRIMARY KEY,
    scanning_sequence         TEXT,
    sequence_variant          TEXT,
    mr_acquisition_type       TEXT,
    repetition_time           REAL,
    echo_time                 REAL,
    flip_angle                REAL,
    echo_train_length         INTEGER,
    number_of_averages        REAL,
    pixel_bandwidth           REAL,
    imaging_frequency         REAL,
    imaged_nucleus            TEXT,
    acquisition_contrast      TEXT,
    percent_sampling          REAL,
    percent_phase_fov         REAL,
    sar_value                 REAL,
    FOREIGN KEY (sop_instance_uid) REFERENCES images(sop_instance_uid)
);

-- Volúmenes en formato Analyze 7.5 (.img + .hdr)
CREATE TABLE IF NOT EXISTS analyze_volumes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT NOT NULL,
    hdr_path        TEXT NOT NULL,
    img_path        TEXT NOT NULL,
    sizeof_hdr      INTEGER,
    num_dimensions  INTEGER,
    dim_x           INTEGER,
    dim_y           INTEGER,
    dim_z           INTEGER,
    dim_t           INTEGER,
    datatype        INTEGER,
    datatype_name   TEXT,
    bitpix          INTEGER,
    pixdim_x        REAL,
    pixdim_y        REAL,
    pixdim_z        REAL,
    pixdim_t        REAL,
    vox_offset      REAL,
    cal_max         REAL,
    cal_min         REAL,
    img_file_size   INTEGER,
    description     TEXT
);
"""


# Funciones auxiliares de extracción DICOM
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


def _safe_get(ds, attr, default=None):
    val = getattr(ds, attr, default)
    if val is None or val == "":
        return default
    if isinstance(val, (pydicom.multival.MultiValue, list)):
        return [str(v) for v in val]
    return val


# Clasificación de archivos
def classify_file(filepath):
    """
    Clasifica un archivo en su tipo correspondiente.

    Retorna uno de:
      - 'dicom_dcm'                  : Archivo DICOM con extensión .dcm
      - 'dicom_no_extension_enhanced': Archivo DICOM Enhanced MR sin extensión
      - 'analyze_hdr'                : Header Analyze 7.5 (.hdr)
      - 'analyze_img'                : Datos Analyze 7.5 (.img)
      - 'tiff'                       : Archivo TIFF (.tif / .tiff)
      - 'analysis_output'            : Archivos de análisis (.fig, .png, .pdf)
      - 'ignore'                     : Archivos a ignorar
      - 'unknown'                    : No reconocido
    """
    basename = os.path.basename(filepath)

    # Ignorar archivos del sistema
    if basename in IGNORE_FILES or any(basename.startswith(p) for p in IGNORE_PREFIXES):
        return "ignore"

    _, ext = os.path.splitext(basename)
    ext_lower = ext.lower()

    # CASO 1: Archivo DICOM con extensión .dcm
    # Origen: Philips Achieva, UNAM Inst. de Neurobiología
    # Formato: MR Image Storage clásico (single-frame)
    # Archivos: IM-000x-000x.dcm
    if ext_lower == ".dcm":
        return "dicom_dcm"

    # CASO 3a: Archivo Analyze 7.5 — Header (.hdr)
    # Origen: Exportación volumétrica, probablemente Philips
    # El header tiene exactamente 348 bytes y contiene dims,
    # datatype, pixdim, y otros parámetros del volumen.
    if ext_lower == ".hdr":
        return "analyze_hdr"

    # CASO 3b: Archivo Analyze 7.5 — Datos (.img)
    # Origen: Acompaña al .hdr con los datos crudos del volumen.
    # Contiene la matriz de vóxeles sin header.
    if ext_lower == ".img":
        return "analyze_img"

    # CASO 4: Archivos TIFF (.tif / .tiff)
    # Origen: Exportaciones de imagen que NO son DICOM.
    # Se identifican para registrarse en el árbol JSON (como
    # imágenes no-DICOM / análisis) pero NO se incluyen en el
    # índice de estudios DICOM de la base de datos (.db).
    if ext_lower in (".tif", ".tiff"):
        return "tiff"

    # CASO 5: Archivos de análisis (.fig, .png, .pdf, etc.)
    # Origen: Resultados de procesamiento en MATLAB (mapas CoV),
    # imágenes de visualización, reportes PDF.
    # Solo se registran en el JSON, no en la DB.
    if ext_lower in ANALYSIS_EXTENSIONS:
        return "analysis_output"

    # CASO 2: Archivo DICOM Enhanced MR sin extensión
    # Origen: Philips Ingenia, I.N. Psiquiatría
    # Formato: Enhanced MR Image Storage (multi-frame)
    # SOPClassUID: 1.2.840.10008.5.1.4.1.1.4.1
    # Archivos: R1, R2, R3, R4, R5 (sin extensión)
    # Metadata en SharedFunctionalGroupsSequence y
    # PerFrameFunctionalGroupsSequence.
    if ext_lower == "":
        # Sin extensión -> intentar leer como DICOM
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
            if hasattr(ds, "Modality"):
                return "dicom_no_extension_enhanced"
        except Exception:
            pass
        return "unknown"

    return "unknown"


# Extracción de metadata DICOM (classic MR y Enhanced MR)
def extract_study_info(ds):
    """Extrae información a nivel estudio."""
    return {
        "study_instance_uid": _safe_str(ds, "StudyInstanceUID"),
        "study_date": _safe_str(ds, "StudyDate"),
        "study_time": _safe_str(ds, "StudyTime"),
        "study_description": _safe_str(ds, "StudyDescription"),
        "patient_name": _safe_str(ds, "PatientName"),
        "patient_id": _safe_str(ds, "PatientID"),
        "patient_birth_date": _safe_str(ds, "PatientBirthDate"),
        "patient_sex": _safe_str(ds, "PatientSex"),
        "patient_age": _safe_str(ds, "PatientAge"),
        "patient_weight": _safe_float(ds, "PatientWeight"),
        "institution_name": _safe_str(ds, "InstitutionName"),
        "institution_address": _safe_str(ds, "InstitutionAddress"),
        "manufacturer": _safe_str(ds, "Manufacturer"),
        "manufacturer_model_name": _safe_str(ds, "ManufacturerModelName"),
        "station_name": _safe_str(ds, "StationName"),
        "device_serial_number": _safe_str(ds, "DeviceSerialNumber"),
        "magnetic_field_strength": _safe_float(ds, "MagneticFieldStrength"),
        "software_versions": str(_safe_get(ds, "SoftwareVersions", "")),
    }


def extract_series_info(ds):
    """Extrae información a nivel serie."""
    return {
        "series_instance_uid": _safe_str(ds, "SeriesInstanceUID"),
        "study_instance_uid": _safe_str(ds, "StudyInstanceUID"),
        "modality": _safe_str(ds, "Modality"),
        "series_date": _safe_str(ds, "SeriesDate"),
        "series_time": _safe_str(ds, "SeriesTime"),
        "series_description": _safe_str(ds, "SeriesDescription"),
        "series_number": _safe_int(ds, "SeriesNumber"),
        "body_part_examined": _safe_str(ds, "BodyPartExamined"),
    }


def extract_image_info_classic(ds, file_path, file_type):
    """
    Extrae información de imagen para DICOM clásico (single-frame).

    CASO: DICOM .dcm — MR Image Storage
    """
    ps = _safe_get(ds, "PixelSpacing")
    ps_x = float(ps[0]) if ps and len(ps) >= 1 else None
    ps_y = float(ps[1]) if ps and len(ps) >= 2 else None

    ipp = _safe_get(ds, "ImagePositionPatient")
    ip_x = float(ipp[0]) if ipp and len(ipp) >= 1 else None
    ip_y = float(ipp[1]) if ipp and len(ipp) >= 2 else None
    ip_z = float(ipp[2]) if ipp and len(ipp) >= 3 else None

    iop = _safe_get(ds, "ImageOrientationPatient")
    iop_str = json.dumps([float(v) for v in iop]) if iop else None

    it = _safe_get(ds, "ImageType")
    it_str = "\\".join(str(v) for v in it) if it else None

    return {
        "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
        "series_instance_uid": _safe_str(ds, "SeriesInstanceUID"),
        "file_path": file_path,
        "file_type": file_type,
        "instance_number": _safe_int(ds, "InstanceNumber"),
        "image_type": it_str,
        "rows": _safe_int(ds, "Rows"),
        "columns": _safe_int(ds, "Columns"),
        "bits_allocated": _safe_int(ds, "BitsAllocated"),
        "bits_stored": _safe_int(ds, "BitsStored"),
        "number_of_frames": _safe_int(ds, "NumberOfFrames", 1),
        "pixel_spacing_x": ps_x,
        "pixel_spacing_y": ps_y,
        "slice_thickness": _safe_float(ds, "SliceThickness"),
        "spacing_between_slices": _safe_float(ds, "SpacingBetweenSlices"),
        "image_position_x": ip_x,
        "image_position_y": ip_y,
        "image_position_z": ip_z,
        "image_orientation_patient": iop_str,
        "rescale_slope": _safe_float(ds, "RescaleSlope"),
        "rescale_intercept": _safe_float(ds, "RescaleIntercept"),
    }


def extract_image_info_enhanced(ds, file_path, file_type):
    """
    Extrae información de imagen para DICOM Enhanced MR (multi-frame).

    CASO: DICOM sin extensión Enhanced MR (SOPClassUID 1.2.840.10008.5.1.4.1.1.4.1)
    La metadata de adquisición está en SharedFunctionalGroupsSequence
    y PerFrameFunctionalGroupsSequence, no en tags de nivel superior.
    """
    # Intentar extraer pixel spacing y otros parámetros de las
    # Functional Groups Sequences (Enhanced DICOM)
    ps_x, ps_y = None, None
    ip_x, ip_y, ip_z = None, None, None
    iop_str = None
    slice_thickness = _safe_float(ds, "SliceThickness")
    spacing_between = _safe_float(ds, "SpacingBetweenSlices")

    # SharedFunctionalGroupsSequence contiene parámetros compartidos
    sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if sfg:
        sfg0 = sfg[0]
        # Pixel Measures
        pm_seq = getattr(sfg0, "PixelMeasuresSequence", None)
        if pm_seq:
            pm = pm_seq[0]
            ps = _safe_get(pm, "PixelSpacing")
            if ps and len(ps) >= 2:
                ps_x, ps_y = float(ps[0]), float(ps[1])
            if slice_thickness is None:
                slice_thickness = _safe_float(pm, "SliceThickness")
            if spacing_between is None:
                spacing_between = _safe_float(pm, "SpacingBetweenSlices")

    # PerFrameFunctionalGroupsSequence — usar primer frame para posición
    pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if pffg and len(pffg) > 0:
        pf0 = pffg[0]
        pp_seq = getattr(pf0, "PlanePositionSequence", None)
        if pp_seq:
            pp = pp_seq[0]
            ipp = _safe_get(pp, "ImagePositionPatient")
            if ipp and len(ipp) >= 3:
                ip_x, ip_y, ip_z = float(ipp[0]), float(ipp[1]), float(ipp[2])
        po_seq = getattr(pf0, "PlaneOrientationSequence", None)
        if po_seq:
            po = po_seq[0]
            iop = _safe_get(po, "ImageOrientationPatient")
            if iop:
                iop_str = json.dumps([float(v) for v in iop])

    it = _safe_get(ds, "ImageType")
    it_str = "\\".join(str(v) for v in it) if it else None

    return {
        "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
        "series_instance_uid": _safe_str(ds, "SeriesInstanceUID"),
        "file_path": file_path,
        "file_type": file_type,
        "instance_number": _safe_int(ds, "InstanceNumber"),
        "image_type": it_str,
        "rows": _safe_int(ds, "Rows"),
        "columns": _safe_int(ds, "Columns"),
        "bits_allocated": _safe_int(ds, "BitsAllocated"),
        "bits_stored": _safe_int(ds, "BitsStored"),
        "number_of_frames": _safe_int(ds, "NumberOfFrames", 1),
        "pixel_spacing_x": ps_x,
        "pixel_spacing_y": ps_y,
        "slice_thickness": slice_thickness,
        "spacing_between_slices": spacing_between,
        "image_position_x": ip_x,
        "image_position_y": ip_y,
        "image_position_z": ip_z,
        "image_orientation_patient": iop_str,
        "rescale_slope": _safe_float(ds, "RescaleSlope"),
        "rescale_intercept": _safe_float(ds, "RescaleIntercept"),
    }


def extract_mr_parameters_classic(ds):
    """
    Extrae parámetros MR de DICOM clásico (single-frame).

    CASO: DICOM .dcm con tags MR de nivel superior.
    """
    return {
        "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
        "scanning_sequence": _safe_str(ds, "ScanningSequence"),
        "sequence_variant": _safe_str(ds, "SequenceVariant"),
        "mr_acquisition_type": _safe_str(ds, "MRAcquisitionType"),
        "repetition_time": _safe_float(ds, "RepetitionTime"),
        "echo_time": _safe_float(ds, "EchoTime"),
        "flip_angle": _safe_float(ds, "FlipAngle"),
        "echo_train_length": _safe_int(ds, "EchoTrainLength"),
        "number_of_averages": _safe_float(ds, "NumberOfAverages"),
        "pixel_bandwidth": _safe_float(ds, "PixelBandwidth"),
        "imaging_frequency": _safe_float(ds, "ImagingFrequency"),
        "imaged_nucleus": _safe_str(ds, "ImagedNucleus"),
        "acquisition_contrast": _safe_str(ds, "AcquisitionContrast"),
        "percent_sampling": _safe_float(ds, "PercentSampling"),
        "percent_phase_fov": _safe_float(ds, "PercentPhaseFieldOfView"),
        "sar_value": None,
    }


def extract_mr_parameters_enhanced(ds):
    """
    Extrae parámetros MR de DICOM Enhanced MR (multi-frame).

    CASO: DICOM sin extensión Enhanced MR
    Los parámetros de secuencia están dentro de SharedFunctionalGroupsSequence:
    - MRTimingAndRelatedParametersSequence -> TR, FlipAngle, EchoTrainLength
    - MREchoSequence (en PerFrame) -> TE
    - MRImagingModifierSequence -> PixelBandwidth
    """
    result = {
        "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
        "scanning_sequence": _safe_str(ds, "ScanningSequence"),
        "sequence_variant": _safe_str(ds, "SequenceVariant"),
        "mr_acquisition_type": _safe_str(ds, "MRAcquisitionType"),
        "repetition_time": None,
        "echo_time": None,
        "flip_angle": None,
        "echo_train_length": None,
        "number_of_averages": _safe_float(ds, "NumberOfAverages"),
        "pixel_bandwidth": None,
        "imaging_frequency": None,
        "imaged_nucleus": _safe_str(ds, "ImagedNucleus"),
        "acquisition_contrast": _safe_str(ds, "AcquisitionContrast"),
        "percent_sampling": None,
        "percent_phase_fov": None,
        "sar_value": None,
    }

    sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if sfg:
        sfg0 = sfg[0]

        # MRTimingAndRelatedParametersSequence
        timing_seq = getattr(sfg0, "MRTimingAndRelatedParametersSequence", None)
        if timing_seq:
            t = timing_seq[0]
            result["repetition_time"] = _safe_float(t, "RepetitionTime")
            result["flip_angle"] = _safe_float(t, "FlipAngle")
            result["echo_train_length"] = _safe_int(t, "EchoTrainLength")

            # SAR
            sar_seq = getattr(t, "SpecificAbsorptionRateSequence", None)
            if sar_seq:
                result["sar_value"] = _safe_float(sar_seq[0],
                                                  "SpecificAbsorptionRateValue")

        # MRImagingModifierSequence
        mod_seq = getattr(sfg0, "MRImagingModifierSequence", None)
        if mod_seq:
            m = mod_seq[0]
            result["pixel_bandwidth"] = _safe_float(m, "PixelBandwidth")
            result["imaging_frequency"] = _safe_float(m, "TransmitterFrequency")

    # Echo time del primer frame
    pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if pffg and len(pffg) > 0:
        pf0 = pffg[0]
        echo_seq = getattr(pf0, "MREchoSequence", None)
        if echo_seq:
            result["echo_time"] = _safe_float(echo_seq[0], "EffectiveEchoTime")

    return result


# Parser de Analyze 7.5 (.hdr)
def parse_analyze_header(hdr_path):
    """
    Lee y parsea un header Analyze 7.5 (.hdr).

    CASO: Archivos Analyze (.img + .hdr)
    Formato volumétrico clásico con header de 348 bytes.
    Estructura del header (little-endian):
      - Offset 0:   sizeof_hdr (4 bytes, int32) — siempre 348
      - Offset 40:  dim[8] (16 bytes, 8 × int16) — dimensiones [ndim, x, y, z, t, ...]
      - Offset 70:  datatype (2 bytes, int16) — tipo de datos
      - Offset 72:  bitpix (2 bytes, int16) — bits por píxel
      - Offset 76:  pixdim[8] (32 bytes, 8 × float32) — tamaño de vóxel
      - Offset 108: vox_offset (4 bytes, float32) — offset de datos en .img
      - Offset 124: cal_max (4 bytes, float32)
      - Offset 128: cal_min (4 bytes, float32)
      - Offset 148: descrip (80 bytes, char[80])
    """
    with open(hdr_path, "rb") as f:
        data = f.read()

    if len(data) < 348:
        logger.warning("Header Analyze demasiado corto: %s (%d bytes)", hdr_path, len(data))
        return None

    # Detectar endianness (sizeof_hdr debe ser 348)
    sizeof_hdr_le = struct.unpack_from("<i", data, 0)[0]
    sizeof_hdr_be = struct.unpack_from(">i", data, 0)[0]

    if sizeof_hdr_le == 348:
        endian = "<"
    elif sizeof_hdr_be == 348:
        endian = ">"
    else:
        logger.warning("Header Analyze inválido (sizeof_hdr != 348): %s", hdr_path)
        return None

    sizeof_hdr = 348

    # Dimensiones: dim[8] en offset 40
    dims = struct.unpack_from(f"{endian}8h", data, 40)
    ndim = dims[0]
    dim_x = dims[1] if ndim >= 1 else 0
    dim_y = dims[2] if ndim >= 2 else 0
    dim_z = dims[3] if ndim >= 3 else 0
    dim_t = dims[4] if ndim >= 4 else 0

    # Tipo de datos y bits por píxel
    datatype = struct.unpack_from(f"{endian}h", data, 70)[0]
    bitpix = struct.unpack_from(f"{endian}h", data, 72)[0]

    # Tamaño de vóxel: pixdim[8] en offset 76
    pixdim = struct.unpack_from(f"{endian}8f", data, 76)
    pixdim_x = pixdim[1] if ndim >= 1 else 0.0
    pixdim_y = pixdim[2] if ndim >= 2 else 0.0
    pixdim_z = pixdim[3] if ndim >= 3 else 0.0
    pixdim_t = pixdim[4] if ndim >= 4 else 0.0

    # vox_offset
    vox_offset = struct.unpack_from(f"{endian}f", data, 108)[0]

    # cal_max, cal_min
    cal_max = struct.unpack_from(f"{endian}f", data, 124)[0]
    cal_min = struct.unpack_from(f"{endian}f", data, 128)[0]

    # Descripción (80 chars en offset 148)
    descrip_raw = data[148:228]
    descrip = descrip_raw.decode("ascii", errors="replace").rstrip("\x00").strip()

    # Tamaño del archivo .img asociado
    img_path = os.path.splitext(hdr_path)[0] + ".img"
    img_size = os.path.getsize(img_path) if os.path.exists(img_path) else None

    return {
        "hdr_path": hdr_path,
        "img_path": img_path,
        "sizeof_hdr": sizeof_hdr,
        "num_dimensions": ndim,
        "dim_x": dim_x,
        "dim_y": dim_y,
        "dim_z": dim_z,
        "dim_t": dim_t,
        "datatype": datatype,
        "datatype_name": ANALYZE_DATATYPES.get(datatype, f"UNKNOWN({datatype})"),
        "bitpix": bitpix,
        "pixdim_x": pixdim_x,
        "pixdim_y": pixdim_y,
        "pixdim_z": pixdim_z,
        "pixdim_t": pixdim_t,
        "vox_offset": vox_offset,
        "cal_max": cal_max,
        "cal_min": cal_min,
        "img_file_size": img_size,
        "description": descrip if descrip else None,
    }


# Lector de metadata TIFF
def read_tiff_metadata(tiff_path):
    """
    Lee metadata de un archivo TIFF.

    CASO: Archivos TIFF (.tif)
    Exportaciones post-procesadas de las imágenes MRI.
    Pueden ser multi-page (un page por slice).
    Se usa tifffile si está disponible, sino se reporta solo el tamaño.
    """
    result = {
        "file_path": tiff_path,
        "shape": None,
        "dtype": None,
        "num_pages": None,
        "image_width": None,
        "image_height": None,
        "bits_per_sample": None,
        "file_size": os.path.getsize(tiff_path),
    }

    if HAS_TIFFFILE:
        try:
            with tifffile.TiffFile(tiff_path) as tif:
                result["num_pages"] = len(tif.pages)
                if tif.pages:
                    page0 = tif.pages[0]
                    result["image_width"] = page0.imagewidth
                    result["image_height"] = page0.imagelength
                    result["bits_per_sample"] = (
                        page0.bitspersample
                        if hasattr(page0, "bitspersample")
                        else None
                    )
                # Leer array para shape/dtype (solo metadatos)
                series = tif.series
                if series:
                    result["shape"] = json.dumps(list(series[0].shape))
                    result["dtype"] = str(series[0].dtype)
        except Exception as e:
            logger.warning("Error leyendo TIFF %s: %s", tiff_path, e)
    else:
        logger.debug("tifffile no disponible; solo se registra tamaño de %s", tiff_path)

    return result


# Construcción del árbol de directorios JSON
def build_directory_tree(root_path, dicom_root, file_info_map):
    """
    Construye un árbol de directorios como diccionario anidado.
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
            child = build_directory_tree(entry_path, dicom_root, file_info_map)
            node["children"].append(child)

        elif os.path.isfile(entry_path):
            child_rel = os.path.relpath(entry_path, dicom_root)
            info = file_info_map.get(entry_path, {})
            file_type = info.get("file_type", "unknown")

            child_node = {
                "name": entry,
                "type": "file",
                "relative_path": child_rel,
                "file_type": file_type,
            }

            # Enriquecer con info específica según el tipo
            if file_type in ("dicom_dcm", "dicom_no_extension_enhanced"):
                child_node["modality"] = info.get("modality")
                child_node["sop_instance_uid"] = info.get("sop_instance_uid")
                child_node["series_description"] = info.get("series_description")
                child_node["number_of_frames"] = info.get("number_of_frames")
            elif file_type == "analyze_hdr":
                child_node["dimensions"] = info.get("dimensions")
                child_node["voxel_size"] = info.get("voxel_size")
            elif file_type == "analyze_img":
                child_node["file_size"] = info.get("file_size")
            elif file_type == "tiff":
                child_node["shape"] = info.get("shape")
                child_node["file_size"] = info.get("file_size")
            elif file_type == "analysis_output":
                _, ext = os.path.splitext(entry)
                child_node["extension"] = ext.lower()

            node["children"].append(child_node)

    return node


# Inserción en SQLite
def _insert_or_ignore(cursor, table, data_dict):
    cols = ", ".join(data_dict.keys())
    placeholders = ", ".join(["?"] * len(data_dict))
    sql = f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})"
    cursor.execute(sql, list(data_dict.values()))


def _insert_auto(cursor, table, data_dict):
    d = {k: v for k, v in data_dict.items() if k != "id"}
    cols = ", ".join(d.keys())
    placeholders = ", ".join(["?"] * len(d))
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    cursor.execute(sql, list(d.values()))


def update_studies_multi_study_info(cursor):
    """
    Analiza la base de datos y actualiza:
      1. En la tabla 'studies':
         - is_multi_study        : 1 si el estudio contiene > 1 directorio de cortes de MRI, 0 en caso contrario.
         - num_slice_directories : número de directorios de cortes distintos asociados al estudio.
         - slice_directories     : JSON array con las rutas relativas de dichos directorios.
      2. En la tabla 'patients':
         - Registra y agrupa los estudios por paciente/proyecto.
    """
    cursor.execute("""
        SELECT
            se.study_instance_uid,
            i.file_path
        FROM images i
        JOIN series se ON i.series_instance_uid = se.series_instance_uid
    """)
    rows = cursor.fetchall()

    study_dirs = {}
    for study_uid, file_path in rows:
        if not study_uid or not file_path:
            continue
        s_dir = os.path.dirname(file_path)
        if study_uid not in study_dirs:
            study_dirs[study_uid] = set()
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

    # Obtener slice_thickness representativo para cada study_uid
    cursor.execute("""
        SELECT
            se.study_instance_uid,
            AVG(i.slice_thickness) as avg_st,
            COUNT(*) as img_count
        FROM series se
        JOIN images i ON se.series_instance_uid = i.series_instance_uid
        WHERE i.slice_thickness IS NOT NULL AND i.slice_thickness > 0
        GROUP BY se.study_instance_uid, se.series_instance_uid
        ORDER BY img_count DESC
    """)
    st_rows = cursor.fetchall()
    study_st_map = {}
    for suid, avg_st, img_count in st_rows:
        if suid not in study_st_map and avg_st:
            study_st_map[suid] = round(avg_st, 2)

    # Actualizar tabla 'patients' agrupando estrictamente por Patient ID
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
                p_dir = "/".join(parts[:2])  # MRI/ImagenesPsiquiatria o MRI/FRUTAS_RM
            else:
                p_dir = os.path.dirname(sdirs[0])

        # Clave del paciente: usar patient_id DICOM siempre que esté presente
        pid_clean = (pid or "").strip()
        pname_clean = (pname or "").strip()
        if pid_clean and pid_clean not in ("UNKNOWN",):
            patient_key = pid_clean
            final_pname = pname_clean or pid_clean
        elif p_dir and ("fantoma" in p_dir.lower() or "meta" in p_dir.lower() or "fruta" in p_dir.lower()):
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

        # Si el paciente tiene múltiples slice_directories dentro de una sesión compartida,
        # cada subdirectorio de adquisición representa un estudio/adquisición individual
        s_dirs_all = sorted(list(pinfo["all_slice_dirs"]))
        if len(s_dirs_all) > 1 and len(pinfo["studies_items"]) <= 1:
            expanded_items = []
            for sd in s_dirs_all:
                sd_name = os.path.basename(sd)
                expanded_items.append({
                    "study_uid": pinfo["study_uids"][0] if pinfo["study_uids"] else sd,
                    "study_folder": sd,
                    "study_description": sd_name,
                    "slice_thickness": pinfo["studies_items"][0]["slice_thickness"] if pinfo["studies_items"] else None,
                    "slice_directories": [sd],
                })
            pinfo["studies_items"] = expanded_items

        # Agrupar estudios por slice_thickness
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
    Anota recursivamente los nodos del árbol JSON con información de multi-estudio y multi-paciente.
    """
    def _traverse(node):
        if node.get("type") != "directory":
            return [], []

        children = node.get("children", [])
        direct_files = [c for c in children if c.get("type") == "file"]
        subdirs = [c for c in children if c.get("type") == "directory"]

        medical_files = [
            f for f in direct_files
            if f.get("file_type") in ("dicom_dcm", "dicom_no_extension_enhanced", "analyze_hdr", "analyze_img")
            or f.get("modality") is not None
        ]
        is_slice_dir = len(medical_files) > 0
        node["is_slice_directory"] = is_slice_dir
        node["num_slices"] = len(medical_files) if is_slice_dir else 0

        direct_modalities = set(f.get("modality") for f in direct_files if f.get("modality"))
        if not direct_modalities and any(f.get("file_type") in ("analyze_hdr", "analyze_img") for f in direct_files):
            direct_modalities.add("MR")

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

        # Agrupar estudios por slice_thickness
        if len(studies_contained) > 1 and node.get("name") not in ("PET_CT", "MRI", "PACIENTES"):
            for st_item in studies_contained:
                sd_node = next((sd for sd in subdirs if sd.get("name") == st_item["name"]), None)
                st_item_th = None
                if sd_node:
                    def _find_th(n):
                        for c in n.get("children", []):
                            if c.get("type") == "file" and c.get("slice_thickness") and c.get("slice_thickness") > 0:
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


# Función principal de indexación
def index_mri(dicom_dir, output_dir=None, verbose=False):
    """
    Indexa los estudios MRI dentro de la carpeta DICOM.

    Parameters
    ----------
    dicom_dir : str
        Ruta al directorio raíz DICOM (que contiene MRI/).
    output_dir : str, optional
        Directorio donde se guardarán los archivos .db y .json.
    verbose : bool
        Si True, imprime progreso detallado.

    Returns
    -------
    tuple(str, str)
        Rutas absolutas de (db_path, json_path).
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mri_root = os.path.join(dicom_dir, "MRI")
    if not os.path.isdir(mri_root):
        msg = "No se encontro la carpeta MRI en: %s" % dicom_dir
        logger.error(msg)
        raise FileNotFoundError(msg)

    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    db_path = os.path.join(output_dir, "mri_index.db")
    json_path = os.path.join(output_dir, "mri_tree.json")

    # Crear DB
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)
    conn.commit()

    # Recolectar todos los archivos
    logger.info("Buscando archivos en: %s", mri_root)
    all_files = []
    for dirpath, _dirnames, filenames in os.walk(mri_root):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            all_files.append(fpath)

    logger.info("Total de archivos encontrados: %d", len(all_files))

    # Clasificar y procesar
    counters = {
        "dicom_dcm": 0,
        "dicom_no_extension_enhanced": 0,
        "analyze_hdr": 0,
        "analyze_img": 0,
        "tiff": 0,
        "analysis_output": 0,
        "ignore": 0,
        "unknown": 0,
        "errors": 0,
    }
    studies_seen = set()
    series_seen = set()
    series_image_count = {}
    file_info_map = {}  # filepath -> info para el JSON
    processed_hdr_pairs = set()  # Para evitar duplicar pares .hdr/.img

    for fpath in all_files:
        ftype = classify_file(fpath)
        rel_path = os.path.relpath(fpath, dicom_dir)

        if ftype == "ignore":
            counters["ignore"] += 1
            continue

        # CASO 1: DICOM .dcm — MR Image Storage clásico (single-frame)
        # Origen: Philips Achieva, UNAM Inst. de Neurobiología
        # Los archivos tienen extensión .dcm y contienen metadata MR
        # en tags de nivel superior (RepetitionTime, EchoTime, etc.)
        if ftype == "dicom_dcm":
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            except Exception as e:
                counters["errors"] += 1
                logger.warning("Error leyendo DICOM .dcm: %s — %s", fpath, e)
                continue

            study_uid = _safe_str(ds, "StudyInstanceUID")
            series_uid = _safe_str(ds, "SeriesInstanceUID")

            # Estudio
            if study_uid and study_uid not in studies_seen:
                _insert_or_ignore(cursor, "studies", extract_study_info(ds))
                studies_seen.add(study_uid)

            # Serie
            if series_uid and series_uid not in series_seen:
                _insert_or_ignore(cursor, "series", extract_series_info(ds))
                series_seen.add(series_uid)
                series_image_count[series_uid] = 0

            # Imagen
            img_info = extract_image_info_classic(ds, rel_path, "dicom_dcm")
            _insert_or_ignore(cursor, "images", img_info)
            if series_uid in series_image_count:
                series_image_count[series_uid] += 1

            # Parámetros MR
            mr_params = extract_mr_parameters_classic(ds)
            _insert_or_ignore(cursor, "mr_parameters", mr_params)

            # Info para JSON
            file_info_map[fpath] = {
                "file_type": "dicom_dcm",
                "modality": _safe_str(ds, "Modality"),
                "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
                "series_description": _safe_str(ds, "SeriesDescription"),
                "number_of_frames": 1,
            }

            counters["dicom_dcm"] += 1
            logger.debug("DICOM .dcm: %s", rel_path)

        # CASO 2: DICOM sin extensión — Enhanced MR (multi-frame)
        # Origen: Philips Ingenia, I.N. Psiquiatría
        # SOPClassUID: 1.2.840.10008.5.1.4.1.1.4.1
        # Los parámetros de secuencia están en
        # SharedFunctionalGroupsSequence y
        # PerFrameFunctionalGroupsSequence.
        # Archivos nombrados R1, R2, R3, R4, R5 sin extensión.
        elif ftype == "dicom_no_extension_enhanced":
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception as e:
                counters["errors"] += 1
                logger.warning("Error leyendo Enhanced MR: %s — %s", fpath, e)
                continue

            study_uid = _safe_str(ds, "StudyInstanceUID")
            series_uid = _safe_str(ds, "SeriesInstanceUID")

            if study_uid and study_uid not in studies_seen:
                _insert_or_ignore(cursor, "studies", extract_study_info(ds))
                studies_seen.add(study_uid)

            if series_uid and series_uid not in series_seen:
                _insert_or_ignore(cursor, "series", extract_series_info(ds))
                series_seen.add(series_uid)
                series_image_count[series_uid] = 0

            img_info = extract_image_info_enhanced(
                ds, rel_path, "dicom_no_extension_enhanced")
            _insert_or_ignore(cursor, "images", img_info)
            if series_uid in series_image_count:
                series_image_count[series_uid] += 1

            mr_params = extract_mr_parameters_enhanced(ds)
            _insert_or_ignore(cursor, "mr_parameters", mr_params)

            file_info_map[fpath] = {
                "file_type": "dicom_no_extension_enhanced",
                "modality": _safe_str(ds, "Modality"),
                "sop_instance_uid": _safe_str(ds, "SOPInstanceUID"),
                "series_description": _safe_str(ds, "SeriesDescription"),
                "number_of_frames": _safe_int(ds, "NumberOfFrames", 1),
            }

            counters["dicom_no_extension_enhanced"] += 1
            logger.debug("Enhanced MR (sin ext): %s", rel_path)

        # CASO 3a: Analyze 7.5 Header (.hdr)
        # Origen: Exportación volumétrica, probablemente Philips
        # El header es de 348 bytes y contiene las dimensiones del
        # volumen (x,y,z,t), tipo de datos, tamaño de vóxel (pixdim),
        # y offset a los datos en el .img asociado.
        # Se procesa el par .hdr+.img juntos cuando se encuentra el .hdr.
        elif ftype == "analyze_hdr":
            if fpath in processed_hdr_pairs:
                continue

            hdr_info = parse_analyze_header(fpath)
            if hdr_info:
                hdr_info["file_path"] = rel_path
                hdr_info["hdr_path"] = rel_path
                hdr_info["img_path"] = os.path.relpath(hdr_info["img_path"], dicom_dir)
                _insert_auto(cursor, "analyze_volumes", hdr_info)
                processed_hdr_pairs.add(fpath)

                file_info_map[fpath] = {
                    "file_type": "analyze_hdr",
                    "dimensions": f"{hdr_info['dim_x']}x{hdr_info['dim_y']}x{hdr_info['dim_z']}",
                    "voxel_size": f"{hdr_info['pixdim_x']:.4f}x{hdr_info['pixdim_y']:.4f}x{hdr_info['pixdim_z']:.1f}",
                }

            counters["analyze_hdr"] += 1
            logger.debug("Analyze .hdr: %s", rel_path)

        # CASO 3b: Analyze 7.5 Data (.img)
        # Datos crudos del volumen sin header.
        # Se registra en el JSON pero ya fue procesado junto con el .hdr.
        elif ftype == "analyze_img":
            file_info_map[fpath] = {
                "file_type": "analyze_img",
                "file_size": os.path.getsize(fpath),
            }
            counters["analyze_img"] += 1
            logger.debug("Analyze .img: %s", rel_path)

        # CASO 4: Archivos TIFF (.tif / .tiff)
        # Origen: Exportaciones de imagen que NO son DICOM.
        # Solo se registran en el árbol de directorios JSON como referencia.
        # NO se insertan en la base de datos (.db) de estudios DICOM.
        elif ftype == "tiff":
            tiff_info = read_tiff_metadata(fpath)
            file_info_map[fpath] = {
                "file_type": "tiff",
                "is_dicom": False,
                "shape": tiff_info.get("shape"),
                "file_size": tiff_info.get("file_size"),
            }

            counters["tiff"] += 1
            logger.debug("TIFF (no-DICOM, solo en JSON): %s", rel_path)

        # CASO 5: Archivos de análisis (.fig, .png, .pdf, etc.)
        # Origen: Resultados de procesamiento MATLAB (mapas CoV, etc.)
        # Solo se registran en el JSON del árbol como referencia,
        # no se indexan en la base de datos.
        elif ftype == "analysis_output":
            file_info_map[fpath] = {
                "file_type": "analysis_output",
            }
            counters["analysis_output"] += 1
            logger.debug("Análisis: %s", rel_path)

        else:
            file_info_map[fpath] = {"file_type": "unknown"}
            counters["unknown"] += 1
            logger.debug("Desconocido: %s", rel_path)

    # Actualizar conteo de imágenes por serie
    for series_uid, count in series_image_count.items():
        cursor.execute(
            "UPDATE series SET num_images = ? WHERE series_instance_uid = ?",
            (count, series_uid)
        )

    # Actualizar métricas de multi-estudio en la tabla 'studies'
    logger.info("Actualizando información de multi-estudios en base de datos...")
    update_studies_multi_study_info(cursor)
    conn.commit()

    # Construir árbol JSON
    logger.info("Construyendo árbol de directorios JSON...")
    tree = build_directory_tree(mri_root, dicom_dir, file_info_map)

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

    # Solo conteo de imágenes/volúmenes médicos verdaderos (DICOM + Analyze)
    total_medical = (counters["dicom_dcm"] + counters["dicom_no_extension_enhanced"]
                     + counters["analyze_hdr"])

    cursor.execute("SELECT COUNT(*) FROM patients WHERE is_multi_study = 1")
    total_multi_patients = cursor.fetchone()[0]

    tree_output = {
        "root": "DICOM/MRI",
        "scan_date": datetime.now().isoformat(),
        "total_files": sum(counters.values()),
        "total_medical_files": total_medical,
        "total_patients": len(patients_summary),
        "total_multi_patients": total_multi_patients,
        "total_studies": len(studies_seen),
        "total_multi_studies": total_multi_studies,
        "total_series": len(series_seen),
        "file_type_counts": {
            "dicom_dcm": counters["dicom_dcm"],
            "dicom_no_extension_enhanced": counters["dicom_no_extension_enhanced"],
            "analyze_hdr_img_pairs": counters["analyze_hdr"],
            "tiff_non_dicom": counters["tiff"],
            "analysis_output": counters["analysis_output"],
        },
        "errors": counters["errors"],
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
    logger.info("INDEXACIÓN MRI COMPLETADA")
    logger.info("=" * 60)
    logger.info("  DICOM .dcm                  : %d", counters["dicom_dcm"])
    logger.info("  DICOM Enhanced MR (sin ext) : %d", counters["dicom_no_extension_enhanced"])
    logger.info("  Analyze .hdr                : %d", counters["analyze_hdr"])
    logger.info("  Analyze .img                : %d", counters["analyze_img"])
    logger.info("  TIFF (no-DICOM, solo JSON)  : %d", counters["tiff"])
    logger.info("  Archivos de análisis        : %d", counters["analysis_output"])
    logger.info("  Ignorados                   : %d", counters["ignore"])
    logger.info("  Errores                     : %d", counters["errors"])
    logger.info("  Pacientes registrados       : %d (Multi-estudio: %d)", len(patients_summary), total_multi_patients)
    logger.info("  Estudios                    : %d", len(studies_seen))
    logger.info("  Series                      : %d", len(series_seen))
    logger.info("  Base de datos               : %s", db_path)
    logger.info("  Árbol JSON                  : %s", json_path)

    return db_path, json_path
