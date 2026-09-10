#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["QT_API"] = "PySide6"
import hashlib
import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field

from datetime import datetime
import matplotlib.pyplot as plt
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import binary_dilation

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QTimer, QSize, QPointF
from PySide6.QtGui import QIcon, QImage, QPixmap, QMovie, QColor, QPainter, QPen, QBrush, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
_INDEX_DIR = os.path.join(_PROJECT_ROOT, "index")
_PROCESSING_DIR = os.path.join(_PROJECT_ROOT, "processing")
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)
if _INDEX_DIR not in sys.path:
    sys.path.insert(0, _INDEX_DIR)
if _PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _PROCESSING_DIR)

import index_dicom_all  # noqa: E402
import join_mri  # noqa: E402
import join_pet_ct  # noqa: E402
import fusion_pet_ct  # noqa: E402
import render_3d  # noqa: E402
from calcular_HU_CT import calcular_hu_ct  # noqa: E402
from calcular_SUV_PT import calcular_suv_pt  # noqa: E402
import analisis_resolucion_espacial  # noqa: E402
import analisis_uniformidad  # noqa: E402
logger = logging.getLogger("larmornium.gui")

try:
    import vtk
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkWindowedSincPolyDataFilter
    from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
    from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
    from vtkmodules.vtkRenderingCore import (
        vtkActor, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow,
        vtkWindowToImageFilter, vtkVolume, vtkVolumeProperty,
        vtkColorTransferFunction
    )
    from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
    VTK_AVAILABLE = True
except Exception:
    try:
        import vtk
        from vtkmodules.util import numpy_support
        vtkSmartVolumeMapper = getattr(vtk, "vtkSmartVolumeMapper", None) or getattr(vtk, "vtkGPUVolumeRayCastMapper", None)
        vtkPiecewiseFunction = getattr(vtk, "vtkPiecewiseFunction", None)
        vtkColorTransferFunction = getattr(vtk, "vtkColorTransferFunction", None)
        vtkImageData = getattr(vtk, "vtkImageData", None)
        vtkTransform = getattr(vtk, "vtkTransform", None)
        vtkTransformPolyDataFilter = getattr(vtk, "vtkTransformPolyDataFilter", None)
        vtkFlyingEdges3D = getattr(vtk, "vtkFlyingEdges3D", None)
        vtkPolyDataNormals = getattr(vtk, "vtkPolyDataNormals", None)
        vtkWindowedSincPolyDataFilter = getattr(vtk, "vtkWindowedSincPolyDataFilter", None)
        vtkOutlineFilter = getattr(vtk, "vtkOutlineFilter", None)
        vtkActor = getattr(vtk, "vtkActor", None)
        vtkCamera = getattr(vtk, "vtkCamera", None)
        vtkPolyDataMapper = getattr(vtk, "vtkPolyDataMapper", None)
        vtkRenderer = getattr(vtk, "vtkRenderer", None)
        vtkRenderWindow = getattr(vtk, "vtkRenderWindow", None)
        vtkWindowToImageFilter = getattr(vtk, "vtkWindowToImageFilter", None)
        vtkVolume = getattr(vtk, "vtkVolume", None)
        vtkVolumeProperty = getattr(vtk, "vtkVolumeProperty", None)
        VTK_AVAILABLE = vtkSmartVolumeMapper is not None
    except Exception:
        VTK_AVAILABLE = False


LARMORNIUM_FILES_DIRNAME = "larmornium_files"
INDEXED_DIRNAME = "indexed"
COMBINED_DB_FILENAME = "dicom_all_index.db"
COMBINED_JSON_FILENAME = "dicom_all_tree.json"
RECENT_FOLDERS_CONFIG_FILENAME = "larmornium.conf"
MAX_RECENT_FOLDERS = 5

LARMORNIUM_FILES_DIR = os.path.join(_PROJECT_ROOT, LARMORNIUM_FILES_DIRNAME)
RECENT_FOLDERS_CONFIG_PATH = os.path.join(
    LARMORNIUM_FILES_DIR, RECENT_FOLDERS_CONFIG_FILENAME
)

ICON_DIR = os.path.join(_GUI_DIR, "icon")
ICON_FUSION_FOUND_PATH = os.path.join(ICON_DIR, "fusion_found.png")
ICON_FUSION_NOT_FOUND_PATH = os.path.join(ICON_DIR, "fusion_not_found.png")
ICON_CT_VOL_FOUND_PATH = os.path.join(ICON_DIR, "ct_vol_found.png")
ICON_CT_VOL_NOT_FOUND_PATH = os.path.join(ICON_DIR, "ct_vol_not_found.png")
ICON_PET_VOL_FOUND_PATH = os.path.join(ICON_DIR, "pet_vol_found.png")
ICON_PET_VOL_NOT_FOUND_PATH = os.path.join(ICON_DIR, "pet_vol_not_found.png")
ICON_MRI_VOL_FOUND_PATH = os.path.join(ICON_DIR, "mri_vol_found.png")
ICON_MRI_VOL_NOT_FOUND_PATH = os.path.join(ICON_DIR, "mri_vol_not_found.png")
ICON_DICOM_FILE_PATH = os.path.join(ICON_DIR, "dicom_file.png")
ICON_LOADING_PATH = os.path.join(ICON_DIR, "loading.gif")

ICON_SEG_FOUND_PATH = os.path.join(ICON_DIR, "seg_found.png")
if not os.path.isfile(ICON_SEG_FOUND_PATH):
    ICON_SEG_FOUND_PATH = os.path.join(ICON_DIR, "seg_vol_found.png")

ICON_SEG_NOT_FOUND_PATH = os.path.join(ICON_DIR, "seg_not_found.png")
if not os.path.isfile(ICON_SEG_NOT_FOUND_PATH):
    ICON_SEG_NOT_FOUND_PATH = os.path.join(ICON_DIR, "seg_vol_not_found.png")

ICON_WARNING_PATH = os.path.join(ICON_DIR, "warning.png")
ICON_GREEN_PATH = os.path.join(ICON_DIR, "green.png")
ICON_RED_PATH = os.path.join(ICON_DIR, "red.png")


def get_fusion_icon(is_built):
    icon_path = ICON_FUSION_FOUND_PATH if is_built else ICON_FUSION_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_ct_icon(is_built):
    icon_path = ICON_CT_VOL_FOUND_PATH if is_built else ICON_CT_VOL_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_pet_icon(is_built):
    icon_path = ICON_PET_VOL_FOUND_PATH if is_built else ICON_PET_VOL_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_mri_icon(is_built):
    icon_path = ICON_MRI_VOL_FOUND_PATH if is_built else ICON_MRI_VOL_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_warning_icon():
    return QIcon(ICON_WARNING_PATH) if os.path.isfile(ICON_WARNING_PATH) else QIcon()


def get_seg_icon(status):
    if status is True or status == "built":
        icon_path = ICON_SEG_FOUND_PATH
    elif status == "warning":
        icon_path = ICON_WARNING_PATH
    else:
        icon_path = ICON_SEG_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_dicom_file_icon():
    return QIcon(ICON_DICOM_FILE_PATH) if os.path.isfile(ICON_DICOM_FILE_PATH) else QIcon()


def get_directory_id(directory):
    if not directory:
        return ""
    abs_path = os.path.abspath(directory)
    return hashlib.md5(abs_path.encode("utf-8")).hexdigest()


def get_directory_files_dir(directory):
    dir_id = get_directory_id(directory)
    return os.path.join(LARMORNIUM_FILES_DIR, dir_id)


FUSION_VOL_DIRNAME = getattr(join_pet_ct, "FUSION_VOL_DIRNAME", "fusion_vol")
CT_VOL_DIRNAME = getattr(join_pet_ct, "CT_VOL_DIRNAME", "ct_vol")
PET_VOL_DIRNAME = getattr(join_pet_ct, "PET_VOL_DIRNAME", "pet_vol")
MRI_VOL_DIRNAME = getattr(join_mri, "MRI_VOL_DIRNAME", "mri_vol")

# Directorios dedicados para volúmenes de segmentación
FUSION_SEG_VOL_DIRNAME = "fusion_segmentation_vol"
CT_SEG_VOL_DIRNAME = "ct_segmentation_vol"
MRI_SEG_VOL_DIRNAME = "mri_segmentation_vol"
SEG_VOL_DIRNAME = "segmentation_vol"  # Legado
PRINT_3D_DIRNAME = "print_3d_files"


def get_print_3d_dir(directory=None, nii_path=None):
    # Retorna la ruta absoluta del directorio print_3d_files dentro de la carpeta hash del estudio
    if directory:
        base = get_directory_files_dir(directory)
    elif nii_path:
        norm = os.path.normpath(os.path.abspath(nii_path))
        parts = norm.split(os.sep)
        if "larmornium_files" in parts:
            idx = parts.index("larmornium_files")
            if idx + 1 < len(parts):
                base = os.sep.join(parts[:idx + 2])
            else:
                base = os.path.dirname(os.path.dirname(norm))
        else:
            base = os.path.dirname(os.path.dirname(norm))
    else:
        base = LARMORNIUM_FILES_DIR
    out_dir = os.path.join(base, PRINT_3D_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# Claves dedicadas en larmornium.conf
FUSION_SEG_CONFIG_KEY = "fusion_segmentations"
CT_SEG_CONFIG_KEY = "ct_segmentations"
MRI_SEG_CONFIG_KEY = "mri_segmentations"
SEG_CONFIG_KEY = "segmentations"  # Legado


def _seg_config_key(seg_context):
    ctx = str(seg_context or "").lower()
    if ctx == "fusion":
        return FUSION_SEG_CONFIG_KEY
    elif ctx in ("ct",):
        return CT_SEG_CONFIG_KEY
    elif ctx in ("mr", "mri"):
        return MRI_SEG_CONFIG_KEY
    return SEG_CONFIG_KEY


def _seg_dir_name(seg_context):
    ctx = str(seg_context or "").lower()
    if ctx == "fusion":
        return FUSION_SEG_VOL_DIRNAME
    elif ctx in ("ct",):
        return CT_SEG_VOL_DIRNAME
    elif ctx in ("mr", "mri"):
        return MRI_SEG_VOL_DIRNAME
    return SEG_VOL_DIRNAME


def _load_built_segmentations(config_path, seg_context=None):
    if not config_path or not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        if seg_context:
            target_key = _seg_config_key(seg_context)
            res = dict(data.get(target_key, {}))
            return res if isinstance(res, dict) else {}
        else:
            all_segs = {}
            for k in (FUSION_SEG_CONFIG_KEY, CT_SEG_CONFIG_KEY, MRI_SEG_CONFIG_KEY, SEG_CONFIG_KEY):
                sub = data.get(k, {})
                if isinstance(sub, dict):
                    all_segs.update(sub)
            return all_segs
    except Exception:
        return {}


def _mark_segmentation_built(config_path, seg_context, key, record):
    if not config_path:
        return
    try:
        data = {}
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        target_key = _seg_config_key(seg_context)
        built = data.setdefault(target_key, {})
        built[key] = record
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Error guardando registro de segmentación en %s: %s", config_path, exc)


def is_segmentation_built(config_path, seg_context, target_id, organ_key):
    if not target_id or not organ_key or not seg_context:
        return False
    key = f"{target_id}_{organ_key}"
    if config_path:
        built = _load_built_segmentations(config_path, seg_context)
        if key in built and isinstance(built[key], dict):
            rec = built[key]
            if rec.get("status") == "warning" or rec.get("generated") is False:
                return False
            if rec.get("stats", {}).get("num_voxels") == 0:
                return False
            nii_path = rec.get("nii_path", "")
            if nii_path and os.path.isfile(nii_path):
                return True
    if config_path != RECENT_FOLDERS_CONFIG_PATH:
        built_global = _load_built_segmentations(RECENT_FOLDERS_CONFIG_PATH, seg_context)
        if key in built_global and isinstance(built_global[key], dict):
            rec = built_global[key]
            if rec.get("status") == "warning" or rec.get("generated") is False:
                return False
            if rec.get("stats", {}).get("num_voxels") == 0:
                return False
            nii_path = rec.get("nii_path", "")
            return bool(nii_path and os.path.isfile(nii_path))
    return False


def get_segmentation_status(config_path, seg_context, target_id, organ_key):
    if not target_id or not organ_key or not seg_context:
        return "none"
    key = f"{target_id}_{organ_key}"

    rec = None
    if config_path:
        built = _load_built_segmentations(config_path, seg_context)
        if key in built and isinstance(built[key], dict):
            rec = built[key]
    if not rec and config_path != RECENT_FOLDERS_CONFIG_PATH:
        built_global = _load_built_segmentations(RECENT_FOLDERS_CONFIG_PATH, seg_context)
        if key in built_global and isinstance(built_global[key], dict):
            rec = built_global[key]

    if rec and isinstance(rec, dict):
        if rec.get("status") == "warning" or rec.get("generated") is False:
            return "warning"
        num_vox = rec.get("stats", {}).get("num_voxels")
        if num_vox == 0:
            return "warning"
        nii_path = rec.get("nii_path", "")
        if nii_path and os.path.isfile(nii_path):
            return "built"

    return "none"


def get_directory_indexed_dir(directory):
    return os.path.join(get_directory_files_dir(directory), INDEXED_DIRNAME)


def get_directory_fusion_vol_dir(directory):
    return os.path.join(get_directory_files_dir(directory), "fusion_vol")


def get_directory_mri_vol_dir(directory):
    return os.path.join(get_directory_files_dir(directory), "mri_vol")


def get_directory_fusion_seg_vol_dir(directory):
    return os.path.join(get_directory_files_dir(directory), FUSION_SEG_VOL_DIRNAME)


def get_directory_ct_seg_vol_dir(directory):
    return os.path.join(get_directory_files_dir(directory), CT_SEG_VOL_DIRNAME)


def get_directory_mri_seg_vol_dir(directory):
    return os.path.join(get_directory_files_dir(directory), MRI_SEG_VOL_DIRNAME)


MODALITY_PREFIXES = {
    "PET_CT": "pet_ct",
    "MRI": "mri",
}
MODALITY_LABELS = {
    "PET_CT": "PET/CT",
    "MRI": "MRI",
}

NODE_TYPE_MODALITY = "modality"
NODE_TYPE_PATIENT = "patient"
NODE_TYPE_STUDY = "study"
NODE_TYPE_SERIES = "series"

NODE_TYPE_FUSION_CATEGORY = "fusion_category"
NODE_TYPE_FUSION_PATIENT = "fusion_patient"
NODE_TYPE_FUSION_STUDY = "fusion_study"
NODE_TYPE_FUSION_PAIR = "fusion_pair"

NODE_TYPE_MULTI_STUDY_CATEGORY = "multi_study_category"
NODE_TYPE_MULTI_STUDY_PATIENT = "multi_study_patient"
NODE_TYPE_MULTI_STUDY_STUDY = "multi_study_study"
NODE_TYPE_MULTI_STUDY_GROUP = "multi_study_group"
NODE_TYPE_MULTI_STUDY_DIRECTORY = "multi_study_directory"

FUSION_CATEGORY_LABEL = "Estudios con corregistro"
MULTI_STUDY_CATEGORY_LABEL = "Paciente con multi estudio"

MODALITY_CT = "CT"
MODALITY_PT = "PT"

CT_WINDOW_PRESETS = [
    ("Tejido blando", 40, 400),
    ("Hueso", 400, 1800),
    ("Pulmón", -600, 1500),
    ("Cerebro", 40, 80),
    ("Mediastino", 50, 350),
]


class RecentFoldersStore:
    def __init__(self, config_path):
        self.config_path = config_path

    def _load_config(self):
        if not os.path.isfile(self.config_path):
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_config(self, config):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def load(self):
        config = self._load_config()
        folders = config.get("recent_folders", [])
        return [f for f in folders if isinstance(f, str)]

    def load_valid(self, validator_fn=None):
        raw = self.load()
        valid = []
        for p in raw:
            if not os.path.isdir(p):
                continue
            if validator_fn:
                cand_db, _ = validator_fn(p)
                if not cand_db or not os.path.isfile(cand_db):
                    continue
            valid.append(p)
        return valid

    def add(self, folder):
        abs_folder = os.path.abspath(folder)
        folders = self.load()
        if abs_folder in folders:
            folders.remove(abs_folder)
        folders.insert(0, abs_folder)
        folders = folders[:MAX_RECENT_FOLDERS]
        self._save(folders)
        return folders

    def remove(self, folder):
        abs_folder = os.path.abspath(folder)
        folders = self.load()
        if abs_folder in folders:
            folders.remove(abs_folder)
        self._save(folders)
        return folders

    def _save(self, folders):
        config = self._load_config()
        config["recent_folders"] = folders
        self._save_config(config)


@dataclass
class SeriesInfo:
    series_instance_uid: str
    series_description: str
    series_number: object
    modality: str
    num_images: int
    series_directory: str = ""


@dataclass
class StudyInfo:
    study_instance_uid: str
    study_description: str
    study_date: str
    study_directory: str = ""
    series_list: list = field(default_factory=list)


@dataclass
class PatientInfo:
    patient_id: str
    patient_name: str
    studies: list = field(default_factory=list)


@dataclass
class FusionPairInfo:
    id: object
    study_instance_uid: str
    ct_series_instance_uid: str
    pet_series_instance_uid: str
    ct_series_description: str
    pet_series_description: str
    ct_directory: str
    pet_directory: str
    num_slices: int
    slice_thickness: object
    ct_convolution_kernel: str = ""
    pet_reconstruction_method: str = ""
    pet_convolution_kernel: str = ""


@dataclass
class FusionStudyInfo:
    study_instance_uid: str
    study_description: str
    study_date: str
    study_directory: str = ""
    pairs: list = field(default_factory=list)


@dataclass
class FusionPatientInfo:
    patient_id: str
    patient_name: str
    studies: list = field(default_factory=list)


@dataclass
class MultiStudyStudyInfo:
    study_instance_uid: str
    study_description: str
    study_date: str
    study_directory: str = ""
    pairs: list = field(default_factory=list)
    series_list: list = field(default_factory=list)


@dataclass
class MultiStudyPatientInfo:
    patient_id: str
    patient_name: str
    modality: str
    num_studies: int
    study_directories: list = field(default_factory=list)
    studies: list = field(default_factory=list)
    total_pairs: int = 0


def parse_analyze_metadata(row):
    file_path = row.get("file_path") or row.get("hdr_path") or ""
    base = os.path.basename(file_path).replace(".hdr", "").replace(".img", "")
    dir_path = os.path.dirname(file_path)
    dir_name = os.path.basename(dir_path)
    desc = (row.get("description") or "").strip()

    m_dir = re.match(r"^(\d+)_(\d+)_(.*)_(\d{8})$", dir_name)
    m_base = re.match(r"^([A-Za-z0-9_]+)_(\d{8})_(\d+)_(\d+)_(.*)$", base)

    if m_base:
        patient_name = m_base.group(1).replace("_", " ").title()
        study_date = m_base.group(2)
        patient_id = m_base.group(3)
        try:
            series_num = int(m_base.group(4))
        except ValueError:
            series_num = 1
        if not desc:
            desc = m_base.group(5).replace("_", " ")
    elif m_dir:
        patient_id = m_dir.group(1)
        try:
            series_num = int(m_dir.group(2))
        except ValueError:
            series_num = 1
        study_date = m_dir.group(4)
        if not desc:
            desc = m_dir.group(3).replace("_", " ")
        patient_name = patient_id
    else:
        patient_id = dir_name or "ANALYZE_PATIENT"
        patient_name = base.replace("_", " ").title() or patient_id
        study_date = ""
        series_num = 1

    if not desc:
        desc = base

    if study_date:
        hash_seed = f"{patient_id}_{study_date}"
        study_desc = "MRI Analyze 7.5"
    else:
        hash_seed = f"{patient_id}_{dir_name}"
        study_desc = f"MRI Analyze 7.5 ({desc})"

    study_uid = f"analyze.study.{hashlib.sha1(hash_seed.encode('utf-8')).hexdigest()[:16]}"
    series_hash = f"{study_uid}_{series_num}_{file_path}"
    series_uid = f"analyze.series.{hashlib.sha1(series_hash.encode('utf-8')).hexdigest()[:16]}"

    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "study_date": study_date,
        "study_description": study_desc,
        "series_description": desc,
        "series_number": series_num,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "file_path": file_path,
        "dir_path": dir_path,
        "dim_z": int(row.get("dim_z") or 1),
    }


class IndexDataAccess:
    def __init__(self, dicom_root, db_path):
        self.dicom_root = dicom_root
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_source_dicom_dir(self):
        conn = self._connect()
        try:
            table_names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "summary" not in table_names:
                return None
            row = conn.execute(
                "SELECT value FROM summary WHERE key='source_dicom_dir'"
            ).fetchone()
            return row["value"] if row else None
        except Exception:
            return None
        finally:
            conn.close()

    def load_modalities(self):
        modalities = {}
        conn = self._connect()
        try:
            table_names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            for modality, prefix in MODALITY_PREFIXES.items():
                has_prefix_table = (
                    f"{prefix}_studies" in table_names or
                    (prefix == "mri" and (f"{prefix}_analyze_volumes" in table_names or "analyze_volumes" in table_names))
                )
                if has_prefix_table:
                    patients = self._load_modality(conn, prefix)
                    if patients:
                        modalities[modality] = patients
        finally:
            conn.close()
        return modalities

    def _load_modality(self, conn, prefix):
        patients_by_id = {}
        patient_order = []

        patients_table = f"{prefix}_patients"
        table_names = {
            row["name"] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if patients_table in table_names:
            rows = conn.execute(
                f"SELECT patient_id, patient_name FROM {patients_table}"
            )
            for row in rows:
                pid = row["patient_id"] or "(sin identificador)"
                patients_by_id[pid] = PatientInfo(pid, row["patient_name"] or pid)
                patient_order.append(pid)

        studies_table_cols = {
            row[1] for row in conn.execute(f"PRAGMA table_info({prefix}_studies)")
        } if f"{prefix}_studies" in table_names else set()
        has_slice_dirs = "slice_directories" in studies_table_cols
        slice_dirs_expr = ", slice_directories" if has_slice_dirs else ""

        studies_by_uid = {}
        if f"{prefix}_studies" in table_names:
            rows = conn.execute(
                f"SELECT study_instance_uid, study_description, study_date, "
                f"patient_id, patient_name{slice_dirs_expr} FROM {prefix}_studies"
            )
            for row in rows:
                uid = row["study_instance_uid"]
                pid = row["patient_id"] or "(sin identificador)"
                if pid not in patients_by_id:
                    patients_by_id[pid] = PatientInfo(pid, row["patient_name"] or pid)
                    patient_order.append(pid)

                st_dir = ""
                if has_slice_dirs and row["slice_directories"]:
                    try:
                        dirs_list = json.loads(row["slice_directories"])
                        if dirs_list and isinstance(dirs_list, list):
                            st_dir = os.path.dirname(dirs_list[0])
                    except Exception:
                        pass

                study = StudyInfo(
                    study_instance_uid=uid,
                    study_description=row["study_description"] or "(sin descripción)",
                    study_date=row["study_date"] or "",
                    study_directory=st_dir,
                )
                studies_by_uid[uid] = study
                patients_by_id[pid].studies.append(study)

        series_dirs = {}
        images_table = f"{prefix}_images"
        if images_table in table_names:
            img_rows = conn.execute(
                f"SELECT series_instance_uid, file_path FROM {images_table} GROUP BY series_instance_uid"
            ).fetchall()
            for ir in img_rows:
                fp = ir["file_path"]
                if fp:
                    series_dirs[ir["series_instance_uid"]] = os.path.dirname(fp)

        if f"{prefix}_series" in table_names:
            rows = conn.execute(
                f"SELECT series_instance_uid, study_instance_uid, series_description, "
                f"series_number, modality, num_images "
                f"FROM {prefix}_series ORDER BY series_number"
            )
            for row in rows:
                suid = row["study_instance_uid"]
                if suid in studies_by_uid:
                    ser_uid = row["series_instance_uid"]
                    ser_dir = series_dirs.get(ser_uid, "")

                    studies_by_uid[suid].series_list.append(SeriesInfo(
                        series_instance_uid=ser_uid,
                        series_description=row["series_description"] or "(sin descripción)",
                        series_number=row["series_number"],
                        modality=row["modality"] or "",
                        num_images=row["num_images"] or 0,
                        series_directory=ser_dir,
                    ))

        # Cargar volúmenes Analyze 7.5 (.hdr / .img) si existen en la base de datos
        analyze_table = (
            f"{prefix}_analyze_volumes" if f"{prefix}_analyze_volumes" in table_names
            else ("analyze_volumes" if "analyze_volumes" in table_names and prefix == "mri" else None)
        )
        if analyze_table:
            existing_uids = {s.series_instance_uid for st in studies_by_uid.values() for s in st.series_list}
            existing_dirs = {s.series_directory for st in studies_by_uid.values() for s in st.series_list if s.series_directory}
            analyze_rows = conn.execute(
                f"SELECT file_path, hdr_path, img_path, dim_x, dim_y, dim_z, "
                f"pixdim_x, pixdim_y, pixdim_z, description FROM {analyze_table}"
            ).fetchall()
            for a_row in analyze_rows:
                a_meta = parse_analyze_metadata(dict(a_row))
                f_path = a_meta["file_path"]
                d_path = a_meta["dir_path"]
                ser_uid = a_meta["series_uid"]

                if ser_uid in existing_uids or d_path in existing_dirs or f_path in existing_dirs:
                    continue

                pid = a_meta["patient_id"]
                if pid not in patients_by_id:
                    patients_by_id[pid] = PatientInfo(pid, a_meta["patient_name"] or pid)
                    patient_order.append(pid)

                st_uid = a_meta["study_uid"]
                if st_uid not in studies_by_uid:
                    study = StudyInfo(
                        study_instance_uid=st_uid,
                        study_description=a_meta["study_description"],
                        study_date=a_meta["study_date"],
                        study_directory=d_path,
                    )
                    studies_by_uid[st_uid] = study
                    patients_by_id[pid].studies.append(study)

                studies_by_uid[st_uid].series_list.append(SeriesInfo(
                    series_instance_uid=ser_uid,
                    series_description=a_meta["series_description"],
                    series_number=a_meta["series_number"],
                    modality="MRI",
                    num_images=a_meta["dim_z"],
                    series_directory=d_path,
                ))
                existing_uids.add(ser_uid)
                existing_dirs.add(d_path)

        return [patients_by_id[pid] for pid in patient_order]

    def load_series_frames(self, prefix, series_instance_uid):
        conn = self._connect()
        try:
            table_names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            images_table = f"{prefix}_images"
            series_table = f"{prefix}_series"
            studies_table = f"{prefix}_studies"

            if images_table not in table_names:
                return []

            images_cols = {
                row[1] for row in conn.execute(f"PRAGMA table_info({images_table})")
            }
            series_cols = {
                row[1] for row in conn.execute(f"PRAGMA table_info({series_table})")
            }
            study_cols = {
                row[1] for row in conn.execute(f"PRAGMA table_info({studies_table})")
            } if studies_table in table_names else set()

            select_parts = ["img.file_path", "img.instance_number"]
            if "image_position_z" in images_cols:
                select_parts.append("img.image_position_z")
                order_clause = "img.image_position_z DESC, img.instance_number"
            elif "slice_location" in images_cols:
                select_parts.append("img.slice_location")
                order_clause = "img.slice_location, img.instance_number"
            else:
                order_clause = "img.instance_number"

            extra_cols = [
                "rescale_slope", "rescale_intercept", "patient_weight",
                "radionuclide_total_dose", "radionuclide_half_life",
                "radiopharmaceutical_start_time", "series_time",
                "image_position_patient", "pixel_spacing",
                "series_description", "modality", "series_number",
            ]
            for col in extra_cols:
                if col in images_cols:
                    select_parts.append(f"img.{col}")
                elif col in series_cols:
                    select_parts.append(f"ser.{col}")
                elif col in study_cols:
                    select_parts.append(f"st.{col}")

            joins = f"FROM {images_table} img LEFT JOIN {series_table} ser ON ser.series_instance_uid = img.series_instance_uid"
            if studies_table in table_names and "study_instance_uid" in series_cols:
                joins += f" LEFT JOIN {studies_table} st ON st.study_instance_uid = ser.study_instance_uid"

            query = (
                f"SELECT {', '.join(select_parts)} {joins} "
                f"WHERE img.series_instance_uid = ? "
                f"ORDER BY {order_clause}"
            )

            rows = conn.execute(query, (series_instance_uid,)).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                fp = d.get("file_path", "")
                if fp and not os.path.isabs(fp) and self.dicom_root:
                    d["file_path"] = os.path.normpath(os.path.join(self.dicom_root, fp))
                results.append(d)
            return results
        finally:
            conn.close()

    def load_fusion_pairs(self):
        conn = self._connect()
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
            raw_rows = [dict(r) for r in rows]
            join_pet_ct._enrich_pairs_metadata(conn, raw_rows)
        finally:
            conn.close()

        patients_by_id = {}
        patient_order = []
        studies_by_key = {}

        for row in raw_rows:
            pid = row.get("patient_id") or "(sin identificador)"
            pname = row.get("patient_name") or pid
            if pid not in patients_by_id:
                patients_by_id[pid] = FusionPatientInfo(pid, pname)
                patient_order.append(pid)

            study_uid = row["study_instance_uid"]
            study_key = (pid, study_uid)
            if study_key not in studies_by_key:
                ct_dir = row.get("ct_directory") or ""
                st_dir = os.path.dirname(ct_dir) if ct_dir else ""
                study = FusionStudyInfo(
                    study_instance_uid=study_uid,
                    study_description=row.get("study_description") or "(sin descripción)",
                    study_date=row.get("study_date") or "",
                    study_directory=st_dir,
                )
                studies_by_key[study_key] = study
                patients_by_id[pid].studies.append(study)

            studies_by_key[study_key].pairs.append(FusionPairInfo(
                id=row["id"],
                study_instance_uid=study_uid,
                ct_series_instance_uid=row.get("ct_series_instance_uid") or "",
                pet_series_instance_uid=row.get("pet_series_instance_uid") or "",
                ct_series_description=row.get("ct_series_description") or "(sin descripción)",
                pet_series_description=row.get("pet_series_description") or "(sin descripción)",
                ct_directory=row.get("ct_directory") or "",
                pet_directory=row.get("pet_directory") or "",
                num_slices=row.get("num_slices") or 0,
                slice_thickness=row.get("slice_thickness"),
                ct_convolution_kernel=row.get("ct_convolution_kernel", "") or "",
                pet_reconstruction_method=row.get("pet_reconstruction_method", "") or "",
                pet_convolution_kernel=row.get("pet_convolution_kernel", "") or "",
            ))

        return [patients_by_id[pid] for pid in patient_order]

    def load_multi_study_patients(self):
        conn = self._connect()
        try:
            table_names = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            results = []
            for modality, prefix in MODALITY_PREFIXES.items():
                table = f"{prefix}_patients"
                fusion_pairs_table = f"{prefix}_fusion_pairs"
                if table not in table_names or fusion_pairs_table not in table_names:
                    continue

                patient_cols = {
                    r[1] for r in conn.execute(f"PRAGMA table_info({table})")
                }
                has_uids = "study_uids" in patient_cols
                select_cols = "patient_id, patient_name, num_studies, study_directories"
                if has_uids:
                    select_cols += ", study_uids"

                rows = conn.execute(
                    f"SELECT {select_cols} FROM {table} WHERE is_multi_study = 1"
                ).fetchall()

                studies_table = f"{prefix}_studies"
                series_table = f"{prefix}_series"
                images_table = f"{prefix}_images"

                series_dirs = {}
                if images_table in table_names:
                    img_rows = conn.execute(
                        f"SELECT series_instance_uid, file_path FROM {images_table} GROUP BY series_instance_uid"
                    ).fetchall()
                    for ir in img_rows:
                        fp = ir["file_path"]
                        if fp:
                            series_dirs[ir["series_instance_uid"]] = os.path.dirname(fp)

                for row in rows:
                    try:
                        directories = json.loads(row["study_directories"] or "[]")
                    except (ValueError, TypeError):
                        directories = []
                    pid = row["patient_id"] or "(sin identificador)"
                    pname = row["patient_name"] or pid

                    uids = []
                    if has_uids and row["study_uids"]:
                        try:
                            parsed = json.loads(row["study_uids"])
                            if isinstance(parsed, list):
                                uids = parsed
                        except Exception:
                            uids = []

                    seen_uids = set()
                    unique_uids = [u for u in uids if u and not (u in seen_uids or seen_uids.add(u))]

                    studies_list = []
                    st_rows = []
                    if studies_table in table_names:
                        if unique_uids:
                            placeholders = ",".join("?" * len(unique_uids))
                            st_query = f"SELECT * FROM {studies_table} WHERE study_instance_uid IN ({placeholders})"
                            raw_st = conn.execute(st_query, unique_uids).fetchall()
                            st_map = {r["study_instance_uid"]: r for r in raw_st}
                            st_rows = [st_map[u] for u in unique_uids if u in st_map]
                        else:
                            st_rows = conn.execute(
                                f"SELECT * FROM {studies_table} WHERE patient_id = ?", (row["patient_id"],)
                            ).fetchall()

                    for st_r in st_rows:
                        suid = st_r["study_instance_uid"]
                        st_desc = st_r["study_description"] or "(sin descripción)"
                        st_date = st_r["study_date"] or ""
                        st_dir = ""
                        if "slice_directories" in st_r.keys() and st_r["slice_directories"]:
                            try:
                                dlist = json.loads(st_r["slice_directories"])
                                if dlist and isinstance(dlist, list):
                                    st_dir = os.path.dirname(dlist[0])
                            except Exception:
                                pass

                        st_obj = MultiStudyStudyInfo(
                            study_instance_uid=suid,
                            study_description=st_desc,
                            study_date=st_date,
                            study_directory=st_dir,
                        )

                        if series_table in table_names:
                            ser_rows = conn.execute(
                                f"SELECT series_instance_uid, series_description, series_number, modality, num_images "
                                f"FROM {series_table} WHERE study_instance_uid = ? ORDER BY series_number",
                                (suid,)
                            ).fetchall()
                            for sr in ser_rows:
                                s_uid = sr["series_instance_uid"]
                                st_obj.series_list.append(SeriesInfo(
                                    series_instance_uid=s_uid,
                                    series_description=sr["series_description"] or "(sin descripción)",
                                    series_number=sr["series_number"],
                                    modality=sr["modality"] or "",
                                    num_images=sr["num_images"] or 0,
                                    series_directory=series_dirs.get(s_uid, ""),
                                ))

                        fp_rows = conn.execute(
                            f"SELECT * FROM {fusion_pairs_table} WHERE study_instance_uid = ?", (suid,)
                        ).fetchall()
                        raw_fps = [dict(r) for r in fp_rows]
                        join_pet_ct._enrich_pairs_metadata(conn, raw_fps)
                        for fp_r in raw_fps:
                            st_obj.pairs.append(FusionPairInfo(
                                id=fp_r["id"],
                                study_instance_uid=suid,
                                ct_series_instance_uid=fp_r.get("ct_series_instance_uid") or "",
                                pet_series_instance_uid=fp_r.get("pet_series_instance_uid") or "",
                                ct_series_description=fp_r.get("ct_series_description") or "(sin descripción)",
                                pet_series_description=fp_r.get("pet_series_description") or "(sin descripción)",
                                ct_directory=fp_r.get("ct_directory") or "",
                                pet_directory=fp_r.get("pet_directory") or "",
                                num_slices=fp_r.get("num_slices") or 0,
                                slice_thickness=fp_r.get("slice_thickness"),
                                ct_convolution_kernel=fp_r.get("ct_convolution_kernel", "") or "",
                                pet_reconstruction_method=fp_r.get("pet_reconstruction_method", "") or "",
                                pet_convolution_kernel=fp_r.get("pet_convolution_kernel", "") or "",
                            ))

                        # Conservar estudios que contengan series o al menos un par fusionable
                        if st_obj.pairs or st_obj.series_list:
                            studies_list.append(st_obj)

                    # Solo incluir pacientes que tengan al menos un estudio
                    if not studies_list:
                        continue

                    total_pairs = sum(len(s.pairs) for s in studies_list)
                    results.append(MultiStudyPatientInfo(
                        patient_id=pid,
                        patient_name=pname,
                        modality=MODALITY_LABELS.get(modality, modality),
                        num_studies=len(studies_list),
                        study_directories=[d for d in directories if isinstance(d, str)],
                        studies=studies_list,
                        total_pairs=total_pairs,
                    ))
        finally:
            conn.close()
        return results


class _QtLogHandler(logging.Handler):
    def __init__(self, worker):
        super().__init__()
        self._worker = worker
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record):
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self._worker.log_message.emit(message)


class IndexWorker(QObject):
    log_message = Signal(str)
    progress_num = Signal(int, int)
    finished = Signal(bool, str)

    def __init__(self, dicom_dir, output_dir):
        super().__init__()
        self.dicom_dir = dicom_dir
        self.output_dir = output_dir

    def run(self):
        root_logger = logging.getLogger()
        handler = _QtLogHandler(self)
        root_logger.addHandler(handler)
        previous_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        try:
            def _on_progress(current, total):
                self.progress_num.emit(current, total)

            index_dicom_all.index_all(
                self.dicom_dir, self.output_dir, verbose=False, progress_callback=_on_progress
            )
            self.finished.emit(True, "")
        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)


class SingleFusionWorker(QObject):
    log_message = Signal(str)
    finished = Signal(bool, str, str, str, object)

    def __init__(self, dicom_root, db_path, larmornium_files_dir, config_path,
                 study_instance_uid=None, pair=None):
        super().__init__()
        self.dicom_root = dicom_root
        self.db_path = db_path
        self.larmornium_files_dir = larmornium_files_dir
        self.config_path = config_path
        self.study_instance_uid = study_instance_uid
        self.pair = pair

    def run(self):
        try:
            target_pair = self.pair
            study_uid = self.study_instance_uid
            if target_pair is None and study_uid:
                pairs = join_pet_ct.load_fusion_pairs_for_study(self.db_path, study_uid)
                if not pairs:
                    self.finished.emit(False, "No se encontraron pares fusionables para el estudio.", study_uid, "", None)
                    return
                target_pair = pairs[0]

            if target_pair is not None:
                study_uid = target_pair.get("study_instance_uid", study_uid)

            key, record = join_pet_ct.ensure_fusion_volume_for_pair(
                target_pair, self.dicom_root, self.larmornium_files_dir, self.config_path,
                progress_callback=self.log_message.emit,
            )
            volume_data = join_pet_ct.load_fused_volume_data(
                record, self.larmornium_files_dir, self.dicom_root, target_pair
            )
            self.finished.emit(True, "", study_uid, key, volume_data)
        except Exception as exc:
            self.finished.emit(False, str(exc), self.study_instance_uid or "", "", None)


class SingleVolumeWorker(QObject):
    log_message = Signal(str)
    finished = Signal(bool, str, str, str, object)

    def __init__(self, dicom_root, series_dict, larmornium_files_dir, config_path, modality="CT"):
        super().__init__()
        self.dicom_root = dicom_root
        self.series_dict = series_dict
        self.larmornium_files_dir = larmornium_files_dir
        self.config_path = config_path
        self.modality = modality.upper()

    def run(self):
        try:
            ser_uid = self.series_dict.get("series_instance_uid", "")
            if self.modality == "CT":
                uid, record = join_pet_ct.ensure_ct_volume_for_series(
                    self.series_dict, self.dicom_root, self.larmornium_files_dir, self.config_path,
                    progress_callback=self.log_message.emit
                )
                data = join_pet_ct.load_single_volume_data(record, modality="CT")
            else:
                uid, record = join_pet_ct.ensure_pet_volume_for_series(
                    self.series_dict, self.dicom_root, self.larmornium_files_dir, self.config_path,
                    progress_callback=self.log_message.emit
                )
                data = join_pet_ct.load_single_volume_data(record, modality="PET")
            self.finished.emit(True, "", self.modality, uid, data)
        except Exception as exc:
            logger.exception("Error construyendo volumen %s: %s", self.modality, exc)
            self.finished.emit(False, str(exc), self.modality, self.series_dict.get("series_instance_uid", ""), None)


class SingleMRIWorker(QObject):
    log_message = Signal(str)
    finished = Signal(bool, str, str, str, object)

    def __init__(self, dicom_root, target_dict, larmornium_files_dir, config_path):
        super().__init__()
        self.dicom_root = dicom_root
        self.target_dict = target_dict
        self.larmornium_files_dir = larmornium_files_dir
        self.config_path = config_path

    def run(self):
        series_uid = self.target_dict.get("series_instance_uid") or ""
        study_uid = self.target_dict.get("study_instance_uid") or ""
        uid = series_uid or study_uid
        try:
            key, record = join_mri.ensure_mri_volume(
                self.target_dict, self.dicom_root, self.larmornium_files_dir, self.config_path,
                progress_callback=self.log_message.emit
            )
            data = join_mri.load_mri_volume_data(record)
            self.finished.emit(True, "", "MRI", key or uid, data)
        except Exception as exc:
            logger.exception("Error construyendo volumen MRI: %s", exc)
            self.finished.emit(False, str(exc), "MRI", uid, None)


class LoadVolumeWorker(QObject):
    finished = Signal(bool, str, str, str, object)

    def __init__(self, load_fn, kind="", tag="", *args, **kwargs):
        super().__init__()
        self.load_fn = load_fn
        self.kind = kind
        self.tag = tag
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.load_fn(*self.args, **self.kwargs)
            self.finished.emit(True, "", self.kind, self.tag, result)
        except Exception as exc:
            logger.exception("Error cargando volumen en segundo plano (%s): %s", self.kind, exc)
            self.finished.emit(False, str(exc), self.kind, self.tag, None)


class Export3DWorker(QObject):
    finished = Signal(bool, str, object)

    def __init__(self, nii_path, output_path, output_format="stl", scale=1.0, smooth_sigma=0.5, quality=0.3, json_path=None, **format_kwargs):
        super().__init__()
        self.nii_path = nii_path
        self.output_path = output_path
        self.output_format = output_format
        self.scale = scale
        self.smooth_sigma = smooth_sigma
        self.quality = quality
        self.json_path = json_path
        self.format_kwargs = format_kwargs

    def run(self):
        # Ejecuta la conversion de la segmentacion a formato 3D en segundo plano
        try:
            from nii_2_3d import segmentation_to_3d
            res = segmentation_to_3d(
                volume_input=self.nii_path,
                output_path=self.output_path,
                output_format=self.output_format,
                json_path=self.json_path,
                scale=self.scale,
                smooth_sigma=self.smooth_sigma,
                quality=self.quality,
                **self.format_kwargs
            )
            self.finished.emit(True, "", res)
        except Exception as exc:
            logger.exception("Error en exportacion 3D: %s", exc)
            self.finished.emit(False, str(exc), None)


def format_dose_mci(dose_val):
    if dose_val is None:
        return None
    try:
        val = float(dose_val)
        if val <= 0:
            return None
        # En DICOM RadionuclideTotalDose está en Bq (1 mCi = 37 MBq = 3.7e7 Bq)
        if val > 1_000_000:
            mci = val / 37_000_000.0
        elif val > 35:  # Si está en MBq
            mci = val / 37.0
        else:  # Si ya viene en mCi
            mci = val
        return f"{mci:.3f} mCi" if mci < 1.0 else f"{mci:.2f} mCi"
    except (ValueError, TypeError):
        return None


def format_dicom_date(date_val):
    if not date_val:
        return None
    s = str(date_val).strip()
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def fetch_pet_dose_and_date(db_path, study_uid=None, series_uid=None):
    if not db_path or not os.path.isfile(db_path):
        return None, None
    dose = None
    date_val = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables = {r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        for r_table in ("pet_ct_radiopharmaceutical_info", "radiopharmaceutical_info"):
            if r_table in tables:
                if study_uid:
                    row = cursor.execute(
                        f"SELECT radionuclide_total_dose FROM {r_table} WHERE study_instance_uid = ? AND radionuclide_total_dose IS NOT NULL LIMIT 1",
                        (study_uid,)
                    ).fetchone()
                    if row and row[0] is not None:
                        dose = row[0]
                        break
                elif series_uid:
                    for s_table in ("pet_ct_series", "series"):
                        if s_table in tables:
                            row = cursor.execute(
                                f"SELECT study_instance_uid FROM {s_table} WHERE series_instance_uid = ? LIMIT 1",
                                (series_uid,)
                            ).fetchone()
                            if row and row[0]:
                                study_uid = row[0]
                                break
                    if study_uid:
                        row = cursor.execute(
                            f"SELECT radionuclide_total_dose FROM {r_table} WHERE study_instance_uid = ? AND radionuclide_total_dose IS NOT NULL LIMIT 1",
                            (study_uid,)
                        ).fetchone()
                        if row and row[0] is not None:
                            dose = row[0]
                            break

        if study_uid:
            for s_table in ("pet_ct_studies", "studies"):
                if s_table in tables:
                    row = cursor.execute(
                        f"SELECT study_date FROM {s_table} WHERE study_instance_uid = ? LIMIT 1",
                        (study_uid,)
                    ).fetchone()
                    if row and row[0]:
                        date_val = row[0]
                        break
        conn.close()
    except Exception as e:
        logger.debug("Error consultando dosis/fecha en SQLite: %s", e)
    return dose, date_val


def extract_study_pet_max_suv(study_dict):
    if not isinstance(study_dict, dict):
        return 1.0
    mod = str(study_dict.get("modality", "")).upper()
    vd = study_dict.get("volume_data") or {}
    meta = vd.get("metadata") or {}
    candidates = [
        study_dict.get("max_suv"),
        study_dict.get("pet_max_suv"),
        vd.get("max_suv"),
        vd.get("pet_max_suv"),
        meta.get("max_suv"),
        meta.get("pet_max_suv"),
    ]
    for c in candidates:
        if c is not None:
            try:
                fc = float(c)
                if fc > 0:
                    return fc
            except (ValueError, TypeError):
                pass

    if mod == "FUSION":
        pet_vol = vd.get("pet_volume")
        if pet_vol is not None and getattr(pet_vol, "size", 0) > 0:
            try:
                m = float(np.nanmax(pet_vol))
                if m > 0:
                    return m
            except Exception:
                pass
    elif mod in ("PET", "PT"):
        vol = vd.get("volume")
        if vol is not None and getattr(vol, "size", 0) > 0:
            try:
                m = float(np.nanmax(vol))
                if m > 0:
                    return m
            except Exception:
                pass

    return 1.0


class MultiStudyVolumeWorker(QObject):
    progress = Signal(int, int, str)
    log_message = Signal(str)
    finished = Signal(bool, str, list)

    def __init__(self, items, dicom_root, larmornium_files_dir, config_path, db_path=None):
        super().__init__()
        self.items = list(items or [])
        self.dicom_root = dicom_root
        self.larmornium_files_dir = larmornium_files_dir
        self.config_path = config_path
        self.db_path = db_path

    def run(self):
        loaded_studies = []
        total = len(self.items)
        if total == 0:
            self.finished.emit(True, "", [])
            return

        try:
            for idx, item in enumerate(self.items):
                node_type = item.get("type")
                mod = str(item.get("modality") or "").upper()
                label = item.get("label") or item.get("series_description") or "Estudio"
                pat_name = item.get("patient_name") or item.get("patient_id") or "Paciente"
                self.progress.emit(idx + 1, total, f"Cargando {label} ({idx + 1}/{total})...")

                if node_type == NODE_TYPE_FUSION_PAIR or mod == "FUSION":
                    pair_dict = item.get("pair") or item
                    key, record = join_pet_ct.ensure_fusion_volume_for_pair(
                        pair_dict, self.dicom_root, self.larmornium_files_dir, self.config_path,
                        progress_callback=self.log_message.emit
                    )
                    vol_data = join_pet_ct.load_fused_volume_data(
                        record, self.larmornium_files_dir, self.dicom_root, pair_dict
                    )
                    study_uid = pair_dict.get("study_instance_uid") or item.get("study_instance_uid")
                    pet_series_uid = pair_dict.get("pet_series_instance_uid")
                    dose_val, date_val = fetch_pet_dose_and_date(self.db_path, study_uid=study_uid, series_uid=pet_series_uid)
                    if dose_val is None:
                        dose_val = pair_dict.get("radionuclide_total_dose") or (vol_data.get("metadata") or {}).get("radionuclide_total_dose")
                    if date_val is None:
                        date_val = pair_dict.get("study_date") or (vol_data.get("metadata") or {}).get("study_date") or pair_dict.get("acquisition_date")
                    loaded_studies.append({
                        "item_info": item,
                        "volume_data": vol_data,
                        "record": record,
                        "key": key,
                        "modality": "FUSION",
                        "patient_name": pat_name,
                        "description": label,
                        "radionuclide_total_dose": dose_val,
                        "acquisition_date": date_val,
                    })
                elif mod in ("PT", "PET"):
                    uid, record = join_pet_ct.ensure_pet_volume_for_series(
                        item, self.dicom_root, self.larmornium_files_dir, self.config_path,
                        progress_callback=self.log_message.emit
                    )
                    vol_data = join_pet_ct.load_single_volume_data(record, modality="PET")
                    study_uid = item.get("study_instance_uid")
                    series_uid = item.get("series_instance_uid") or uid
                    dose_val, date_val = fetch_pet_dose_and_date(self.db_path, study_uid=study_uid, series_uid=series_uid)
                    if dose_val is None:
                        dose_val = item.get("radionuclide_total_dose") or (vol_data.get("metadata") or {}).get("radionuclide_total_dose")
                    if date_val is None:
                        date_val = item.get("study_date") or (vol_data.get("metadata") or {}).get("study_date") or item.get("acquisition_date")
                    loaded_studies.append({
                        "item_info": item,
                        "volume_data": vol_data,
                        "record": record,
                        "key": uid,
                        "modality": "PET",
                        "patient_name": pat_name,
                        "description": label,
                        "radionuclide_total_dose": dose_val,
                        "acquisition_date": date_val,
                    })
                elif mod in ("MR", "MRI"):
                    key, record = join_mri.ensure_mri_volume(
                        item, self.dicom_root, self.larmornium_files_dir, self.config_path,
                        progress_callback=self.log_message.emit
                    )
                    vol_data = join_mri.load_mri_volume_data(record)
                    loaded_studies.append({
                        "item_info": item,
                        "volume_data": vol_data,
                        "record": record,
                        "key": key,
                        "modality": "MRI",
                        "patient_name": pat_name,
                        "description": label,
                    })
                else:  # CT
                    uid, record = join_pet_ct.ensure_ct_volume_for_series(
                        item, self.dicom_root, self.larmornium_files_dir, self.config_path,
                        progress_callback=self.log_message.emit
                    )
                    vol_data = join_pet_ct.load_single_volume_data(record, modality="CT")
                    loaded_studies.append({
                        "item_info": item,
                        "volume_data": vol_data,
                        "record": record,
                        "key": uid,
                        "modality": "CT",
                        "patient_name": pat_name,
                        "description": label,
                    })

            self.finished.emit(True, "", loaded_studies)
        except Exception as exc:
            logger.exception("Error en carga de volúmenes multi-estudio: %s", exc)
            self.finished.emit(False, str(exc), loaded_studies)


class _HoverImageLabel(QLabel):
    pixel_hovered = Signal(float, float)
    pixel_left = Signal()
    resized = Signal(QSize)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(30, 30)

    def minimumSizeHint(self):
        return QSize(30, 30)

    def sizeHint(self):
        return QSize(100, 100)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit(event.size())

    def mouseMoveEvent(self, event):
        pos = event.position()
        self.pixel_hovered.emit(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.pixel_left.emit()
        super().leaveEvent(event)


class ImageViewer(QWidget):
    PLACEHOLDER_TEXT = "Seleccione una serie o estudio para visualizarlo"
    frame_changed = Signal(str, int, int, object)
    fused_slice_changed = Signal(int, int, str)
    mri_slice_changed = Signal(int, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_fusion_mode = False
        self._is_mri_mode = False
        self._frames = []
        self._modality = None
        self._cache_path = None
        self._cache_dataset = None
        self._cache_pixel_array = None
        self._cache_value_matrix = None
        self._current_pixmap = None
        self._current_value_matrix = None
        self._current_units_label = None

        self._fusion_ct_volume = None
        self._fusion_pet_volume = None
        self._fusion_pet_vmax = 1.0
        self._fusion_pet_max_suv = 1.0
        self._series_pet_max_suv = None
        self._series_pet_vmax = None
        self._user_adjusted_suv = False
        self._fusion_pet_units = "SUV"
        self._fusion_z_positions = []

        self._mri_volume = None
        self._mri_units = None
        self._mri_z_positions = []
        self._mri_window_center = 0.0
        self._mri_window_width = 1.0

        self._num_slices = 0

        self._ct_window_center = 40.0
        self._ct_window_width = 400.0
        self._ct_alpha = 0.6
        self._pet_alpha = 0.4

        self._setup_ui()

    def minimumSizeHint(self):
        return QSize(250, 150)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        self.title_label = QLabel("")
        self.title_label.setStyleSheet("color: #63b3ed; font-weight: bold; font-size: 13px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        self.image_label = _HoverImageLabel(self.PLACEHOLDER_TEXT)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1c23; border: 1px solid #2d3139;")
        self.image_label.setMinimumSize(50, 50)
        self.image_label.pixel_hovered.connect(self._on_pixel_hovered)
        self.image_label.pixel_left.connect(self._on_pixel_left)

        img_overlay_layout = QVBoxLayout(self.image_label)
        img_overlay_layout.setContentsMargins(8, 8, 8, 8)
        img_overlay_layout.addStretch()
        self.lbl_pet_overlay = QLabel("")
        self.lbl_pet_overlay.setStyleSheet("color: #ffff00; font-weight: bold; font-size: 11px; background: transparent;")
        self.lbl_pet_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.lbl_pet_overlay.setWordWrap(True)
        self.lbl_pet_overlay.setVisible(False)
        img_overlay_layout.addWidget(self.lbl_pet_overlay)

        layout.addWidget(self.image_label, 1)

        controls_layout = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.setValue(1)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        controls_layout.addWidget(self.slider)

        self.slice_label = QLabel("0 / 0")
        self.slice_label.setFixedWidth(70)
        self.slice_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls_layout.addWidget(self.slice_label)

        self.value_label = QLabel("")
        self.value_label.setFixedWidth(130)
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet("color: #a0a0a0; font-family: monospace;")
        controls_layout.addWidget(self.value_label)

        layout.addLayout(controls_layout)

        self.window_row = QWidget()
        win_layout = QHBoxLayout(self.window_row)
        win_layout.setContentsMargins(0, 0, 0, 0)
        win_layout.setSpacing(6)

        self.win_label = QLabel("Ventana:")
        win_layout.addWidget(self.win_label)

        self.window_combo = QComboBox()
        for name, c, w in CT_WINDOW_PRESETS:
            self.window_combo.addItem(name, (c, w))
        self.window_combo.currentIndexChanged.connect(self._on_window_preset_changed)
        win_layout.addWidget(self.window_combo)

        self.pet_suv_max_widget = QWidget()
        suv_layout = QHBoxLayout(self.pet_suv_max_widget)
        suv_layout.setContentsMargins(10, 0, 0, 0)
        suv_layout.setSpacing(6)

        self.pet_suv_max_label = QLabel("SUV Máx: 1.00")
        self.pet_suv_max_label.setMinimumWidth(95)
        self.pet_suv_max_slider = QSlider(Qt.Horizontal)
        self.pet_suv_max_slider.setRange(1, 250)
        self.pet_suv_max_slider.setValue(100)
        self.pet_suv_max_slider.setFixedWidth(130)
        self.pet_suv_max_slider.valueChanged.connect(self._on_pet_suv_max_changed)

        suv_layout.addWidget(self.pet_suv_max_label)
        suv_layout.addWidget(self.pet_suv_max_slider)
        win_layout.addWidget(self.pet_suv_max_widget)
        self.pet_suv_max_widget.setVisible(False)

        win_layout.addStretch()
        self.window_row.setVisible(False)
        layout.addWidget(self.window_row)

        self.transparency_row = QWidget()
        trans_layout = QHBoxLayout(self.transparency_row)
        trans_layout.setContentsMargins(0, 0, 0, 0)
        trans_layout.setSpacing(12)

        ct_alpha_label = QLabel("Opacidad CT:")
        self.ct_alpha_slider = QSlider(Qt.Horizontal)
        self.ct_alpha_slider.setRange(0, 100)
        self.ct_alpha_slider.setValue(60)
        self.ct_alpha_slider.setFixedWidth(100)
        self.ct_alpha_slider.valueChanged.connect(self._on_ct_alpha_changed)
        self.ct_alpha_val_label = QLabel("60%")
        self.ct_alpha_val_label.setFixedWidth(35)

        trans_layout.addWidget(ct_alpha_label)
        trans_layout.addWidget(self.ct_alpha_slider)
        trans_layout.addWidget(self.ct_alpha_val_label)

        pet_alpha_label = QLabel("Opacidad PET:")
        self.pet_alpha_slider = QSlider(Qt.Horizontal)
        self.pet_alpha_slider.setRange(0, 100)
        self.pet_alpha_slider.setValue(40)
        self.pet_alpha_slider.setFixedWidth(100)
        self.pet_alpha_slider.valueChanged.connect(self._on_pet_alpha_changed)
        self.pet_alpha_val_label = QLabel("40%")
        self.pet_alpha_val_label.setFixedWidth(35)

        trans_layout.addWidget(pet_alpha_label)
        trans_layout.addWidget(self.pet_alpha_slider)
        trans_layout.addWidget(self.pet_alpha_val_label)

        trans_layout.addStretch()
        self.transparency_row.setVisible(False)
        layout.addWidget(self.transparency_row)

    def _on_ct_alpha_changed(self, value):
        self._ct_alpha = value / 100.0
        self.ct_alpha_val_label.setText("%d%%" % value)
        if self._is_fusion_mode:
            self._render_current_fused_slice()

    def _on_pet_alpha_changed(self, value):
        self._pet_alpha = value / 100.0
        self.pet_alpha_val_label.setText("%d%%" % value)
        if self._is_fusion_mode:
            self._render_current_fused_slice()

    def _on_pet_suv_max_changed(self, value):
        if self._is_fusion_mode and self._fusion_pet_max_suv > 0:
            pct = value / 100.0
            self._fusion_pet_vmax = max(0.01, pct * self._fusion_pet_max_suv)
            self.pet_suv_max_label.setText("SUV Máx: %.2f" % self._fusion_pet_vmax)
            self._render_current_fused_slice()
        elif self._modality == MODALITY_PT or str(self._modality).upper() in ("PT", "PET"):
            self._user_adjusted_suv = True
            if self._series_pet_max_suv is None or self._series_pet_max_suv <= 0:
                if self._cache_value_matrix is not None:
                    pos_vals = self._cache_value_matrix[self._cache_value_matrix > 0]
                    ref = float(np.percentile(pos_vals, 99.5)) if len(pos_vals) > 0 else (
                        float(np.nanmax(self._cache_value_matrix)) if self._cache_value_matrix.size > 0 else 1.0
                    )
                    self._series_pet_max_suv = max(ref, 0.1)
                else:
                    self._series_pet_max_suv = 1.0
            pct = value / 100.0
            self._series_pet_vmax = max(0.01, pct * self._series_pet_max_suv)
            self.pet_suv_max_label.setText("SUV Máx: %.2f" % self._series_pet_vmax)
            if self._cache_value_matrix is not None:
                self._apply_pet_colormap(self._cache_value_matrix)

    def _on_window_preset_changed(self, index):
        if index < 0:
            return
        data = self.window_combo.itemData(index)
        if not data:
            return
        center, width = data
        if self._is_mri_mode:
            self._mri_window_center = float(center)
            self._mri_window_width = float(width)
            self._render_current_mri_slice()
        elif self._is_fusion_mode:
            self._ct_window_center = float(center)
            self._ct_window_width = float(width)
            self._render_current_fused_slice()
        elif self._modality == MODALITY_CT and self._cache_value_matrix is not None:
            self._ct_window_center = float(center)
            self._ct_window_width = float(width)
            self._apply_ct_window(self._cache_value_matrix)

    def _update_pet_overlay(self, dose_val, date_val):
        dose_str = format_dose_mci(dose_val)
        date_str = format_dicom_date(date_val)
        parts = []
        if date_str:
            parts.append(date_str)
        if dose_str:
            parts.append(f"D0: {dose_str}")
        if parts:
            self.lbl_pet_overlay.setText("\n".join(parts))
            self.lbl_pet_overlay.setVisible(True)
        else:
            self.lbl_pet_overlay.clear()
            self.lbl_pet_overlay.setVisible(False)

    def clear(self):
        self._is_fusion_mode = False
        self._is_mri_mode = False
        self._frames = []
        self._modality = None
        self._cache_path = None
        self._cache_dataset = None
        self._cache_pixel_array = None
        self._cache_value_matrix = None
        self._current_pixmap = None
        self._current_value_matrix = None
        self._current_units_label = None

        self._fusion_ct_volume = None
        self._fusion_pet_volume = None
        self._fusion_z_positions = []

        self._mri_volume = None
        self._mri_units = None
        self._mri_z_positions = []

        self._num_slices = 0

        self.slider.setEnabled(False)
        self.slider.blockSignals(True)
        self.slider.setValue(1)
        self.slider.setMaximum(1)
        self.slider.blockSignals(False)

        self.slice_label.setText("0 / 0")
        self.value_label.setText("")
        self.title_label.setText("")
        self.image_label.clear()
        self.image_label.setText(self.PLACEHOLDER_TEXT)
        self._series_pet_max_suv = None
        self._series_pet_vmax = None
        self._user_adjusted_suv = False
        self.window_row.setVisible(False)
        self.win_label.setVisible(True)
        self.window_combo.setVisible(True)
        self.pet_suv_max_widget.setVisible(False)
        self.transparency_row.setVisible(False)
        self._update_pet_overlay(None, None)

    def show_series(self, frames, modality, title="", initial_max_suv=None):
        self._is_fusion_mode = False
        self._is_mri_mode = False
        self._fusion_ct_volume = None
        self._fusion_pet_volume = None
        self._mri_volume = None
        self._frames = frames
        self._modality = modality
        self._cache_path = None
        self._cache_dataset = None
        self._cache_pixel_array = None
        self._cache_value_matrix = None
        self._current_value_matrix = None
        self._current_units_label = None
        self.value_label.setText("")
        self.title_label.setText(f"Serie {modality}: {title}" if title else f"Modalidad {modality}")

        is_ct = (modality == MODALITY_CT or str(modality).upper() == "CT")
        is_pet = (modality == MODALITY_PT or str(modality).upper() in ("PT", "PET"))

        self.window_row.setVisible(is_ct or is_pet)
        self.win_label.setVisible(is_ct)
        self.window_combo.setVisible(is_ct)
        self.pet_suv_max_widget.setVisible(is_pet)
        self.transparency_row.setVisible(False)

        self._user_adjusted_suv = False
        if is_pet:
            val = float(initial_max_suv) if initial_max_suv and float(initial_max_suv) > 0 else None
            self._series_pet_max_suv = val
            self._series_pet_vmax = val
            self.pet_suv_max_slider.blockSignals(True)
            self.pet_suv_max_slider.setRange(1, 250)
            self.pet_suv_max_slider.setValue(100)
            self.pet_suv_max_slider.blockSignals(False)
            if val is not None:
                self.pet_suv_max_label.setText("SUV Máx: %.2f" % val)
            else:
                self.pet_suv_max_label.setText("SUV Máx: ---")
        else:
            self._series_pet_max_suv = None
            self._series_pet_vmax = None

        count = len(frames)
        self.slider.blockSignals(True)
        self.slider.setMinimum(1)
        self.slider.setMaximum(count)
        self.slider.setValue(1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(count > 1)
        self._display_frame(0)

    def set_pet_series_max_suv(self, max_suv):
        val = float(max_suv) if max_suv and float(max_suv) > 0 else 1.0
        self._series_pet_max_suv = val
        if not self._user_adjusted_suv:
            pct = self.pet_suv_max_slider.value() / 100.0 if self.pet_suv_max_slider else 1.0
            self._series_pet_vmax = max(0.01, pct * val)
            self.pet_suv_max_label.setText("SUV Máx: %.2f" % self._series_pet_vmax)
            if (self._modality == MODALITY_PT or str(self._modality).upper() in ("PT", "PET")) and self._cache_value_matrix is not None:
                self._apply_pet_colormap(self._cache_value_matrix)

    def show_fused_volume(self, volume_data):
        if not volume_data or "ct_volume" not in volume_data:
            self.clear()
            self.image_label.setText("No se pudieron cargar los datos del volumen fusionado")
            return

        self._is_fusion_mode = True
        self._modality = "FUSION_PET_CT"
        self.title_label.setText("Fusión PET/CT")
        self._frames = []

        meta = volume_data.get("metadata", {})
        dose_val = volume_data.get("radionuclide_total_dose") or meta.get("radionuclide_total_dose")
        date_val = (
            volume_data.get("acquisition_date")
            or volume_data.get("study_date")
            or meta.get("acquisition_date")
            or meta.get("study_date")
        )
        self._update_pet_overlay(dose_val, date_val)
        self._fusion_ct_volume = volume_data["ct_volume"]
        self._fusion_pet_volume = volume_data["pet_volume"]
        self._fusion_pet_max_suv = float(
            volume_data.get("max_suv") or volume_data.get("pet_max_suv") or (
                float(np.nanmax(self._fusion_pet_volume)) if self._fusion_pet_volume.size > 0 else 1.0
            )
        )
        if self._fusion_pet_max_suv <= 0:
            self._fusion_pet_max_suv = 1.0
        self._fusion_pet_vmax = self._fusion_pet_max_suv
        self._fusion_pet_units = str(volume_data.get("pet_units", "SUV"))
        self._fusion_z_positions = volume_data.get("z_positions", [])
        self._num_slices = int(volume_data.get("num_slices", self._fusion_ct_volume.shape[0]))

        self.window_row.setVisible(True)
        self.win_label.setVisible(True)
        self.window_combo.setVisible(True)
        self.pet_suv_max_widget.setVisible(True)
        self.pet_suv_max_slider.blockSignals(True)
        self.pet_suv_max_slider.setRange(1, 250)
        self.pet_suv_max_slider.setValue(100)
        self.pet_suv_max_slider.blockSignals(False)
        self.pet_suv_max_label.setText("SUV Máx: %.2f" % self._fusion_pet_max_suv)

        self.transparency_row.setVisible(True)

        count = self._num_slices
        self.slider.blockSignals(True)
        self.slider.setMinimum(1)
        self.slider.setMaximum(count)
        self.slider.setValue(count // 2 if count > 0 else 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(count > 1)

        self._render_current_fused_slice()

    def _render_current_fused_slice(self):
        if not self._is_fusion_mode or self._fusion_ct_volume is None:
            return

        slice_idx = self.slider.value() - 1
        total = self._num_slices
        if slice_idx < 0 or slice_idx >= total:
            return

        ct_slice = self._fusion_ct_volume[slice_idx].astype(np.float32)
        pet_slice = self._fusion_pet_volume[slice_idx].astype(np.float32)

        c = self._ct_window_center
        w = max(self._ct_window_width, 1.0)
        ct_norm = np.clip((ct_slice - (c - w / 2.0)) / w, 0.0, 1.0)
        ct_rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

        vmax = max(self._fusion_pet_vmax, 1e-6)
        pet_norm = np.clip(pet_slice / vmax, 0.0, 1.0)
        cmap = plt.get_cmap("hot")
        pet_rgba = cmap(pet_norm)
        pet_rgb = pet_rgba[..., :3]

        alpha_ct = self._ct_alpha
        alpha_pet = self._pet_alpha
        fused_rgb = np.clip(ct_rgb * alpha_ct + pet_rgb * alpha_pet, 0.0, 1.0)

        rgb_uint8 = np.ascontiguousarray((fused_rgb * 255).astype(np.uint8))
        h, w_img, _ = rgb_uint8.shape
        qimg = QImage(rgb_uint8.data, w_img, h, 3 * w_img, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self._current_pixmap = pixmap
        self._current_value_matrix = ct_slice
        self._current_units_label = "HU"

        z_val = self._fusion_z_positions[slice_idx] if slice_idx < len(self._fusion_z_positions) else None
        z_str = "z = %.1f mm" % z_val if z_val is not None else ""
        self.slice_label.setText("%d / %d" % (slice_idx + 1, total))
        self._scale_and_set_pixmap()

        self.fused_slice_changed.emit(slice_idx + 1, total, z_str)

    def show_mri_volume(self, volume_data, title=""):
        if not volume_data or "volume" not in volume_data:
            self.clear()
            self.image_label.setText("No se pudieron cargar los datos del volumen MRI")
            return

        self._is_fusion_mode = False
        self._is_mri_mode = True
        self._modality = "MRI"
        self.title_label.setText(f"Serie MRI: {title}" if title else "Modalidad MRI")
        self._frames = []
        self._fusion_ct_volume = None
        self._fusion_pet_volume = None
        self._mri_volume = volume_data["volume"]
        self._mri_units = volume_data.get("units", "")
        self._mri_z_positions = volume_data.get("z_positions", [])
        self._num_slices = int(volume_data.get("num_slices", self._mri_volume.shape[0]))

        v_min = float(volume_data.get("intensity_min", np.nanmin(self._mri_volume) if self._mri_volume.size > 0 else 0.0))
        v_max = float(volume_data.get("intensity_max", np.nanmax(self._mri_volume) if self._mri_volume.size > 0 else 1.0))
        if v_max <= v_min:
            v_max = v_min + 1.0

        p1 = float(np.percentile(self._mri_volume, 1)) if self._mri_volume.size > 0 else v_min
        p99 = float(np.percentile(self._mri_volume, 99)) if self._mri_volume.size > 0 else v_max
        if p99 <= p1:
            p99 = p1 + 1.0

        # Configurar selector de ventaneo MRI
        self.win_label.setText("Ventana MRI:")
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        mri_presets = [
            ("Contraste óptimo (1-99%)", (p1 + p99) / 2.0, max(p99 - p1, 1.0)),
            ("Por defecto (Completo)", (v_min + v_max) / 2.0, max(v_max - v_min, 1.0)),
            ("Alto contraste", v_min + 0.35 * (v_max - v_min), max(0.50 * (v_max - v_min), 1.0)),
            ("Tejido blando / Cerebro", v_min + 0.45 * (v_max - v_min), max(0.60 * (v_max - v_min), 1.0)),
            ("Señal alta (T2/FLAIR)", v_min + 0.60 * (v_max - v_min), max(0.70 * (v_max - v_min), 1.0)),
            ("Señal baja (T1)", v_min + 0.25 * (v_max - v_min), max(0.45 * (v_max - v_min), 1.0)),
        ]
        for name, c, w in mri_presets:
            self.window_combo.addItem(name, (c, w))
        self.window_combo.setCurrentIndex(0)
        self._mri_window_center = (p1 + p99) / 2.0
        self._mri_window_width = max(p99 - p1, 1.0)
        self.window_combo.blockSignals(False)

        self.window_row.setVisible(True)
        self.win_label.setVisible(True)
        self.window_combo.setVisible(True)
        self.pet_suv_max_widget.setVisible(False)
        self.transparency_row.setVisible(False)

        count = self._num_slices
        self.slider.blockSignals(True)
        self.slider.setMinimum(1)
        self.slider.setMaximum(count)
        self.slider.setValue(count // 2 if count > 0 else 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(count > 1)

        self._render_current_mri_slice()

    def _render_current_mri_slice(self):
        if not self._is_mri_mode or self._mri_volume is None:
            return

        slice_idx = self.slider.value() - 1
        total = self._num_slices
        if slice_idx < 0 or slice_idx >= total:
            return

        slice_data = self._mri_volume[slice_idx].astype(np.float32)

        c = self._mri_window_center
        w = max(self._mri_window_width, 1.0)
        norm = np.clip((slice_data - (c - w / 2.0)) / w, 0.0, 1.0)

        uint8_arr = np.ascontiguousarray((norm * 255).astype(np.uint8))
        h, w_img = uint8_arr.shape[:2]
        qimg = QImage(uint8_arr.data, w_img, h, w_img, QImage.Format_Grayscale8)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._current_value_matrix = slice_data
        self._current_units_label = self._mri_units or ""

        z_val = self._mri_z_positions[slice_idx] if slice_idx < len(self._mri_z_positions) else None
        z_str = "z = %.1f mm" % z_val if z_val is not None else ""
        self.slice_label.setText("%d / %d" % (slice_idx + 1, total))
        self._scale_and_set_pixmap()

        self.mri_slice_changed.emit(slice_idx + 1, total, z_str)

    def _on_slider_changed(self, value):
        if self._is_mri_mode:
            self._render_current_mri_slice()
        elif self._is_fusion_mode:
            self._render_current_fused_slice()
        else:
            self._display_frame(value - 1)

    def _display_frame(self, index):
        if not self._frames or index < 0 or index >= len(self._frames):
            return

        frame = self._frames[index]
        self.slice_label.setText("%d / %d" % (index + 1, len(self._frames)))

        file_path = frame["file_path"]
        self.frame_changed.emit(file_path, index + 1, len(self._frames), frame.get("frame_index"))

        try:
            if self._cache_path != file_path:
                ds = pydicom.dcmread(file_path, force=True)
                self._cache_path = file_path
                self._cache_dataset = ds
                self._cache_pixel_array = ds.pixel_array
                self._cache_value_matrix = None
            else:
                ds = self._cache_dataset

            arr = self._cache_pixel_array
            is_rgb = (arr.ndim == 3 and arr.shape[-1] in (3, 4))
            photo_interp = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()

            if is_rgb or photo_interp in ("RGB", "YBR_FULL", "YBR_FULL_422", "PALETTE COLOR"):
                self._current_value_matrix = arr
                self._current_units_label = "RGB"
                self._display_rgb(arr)

            elif self._modality == MODALITY_CT and photo_interp in ("MONOCHROME1", "MONOCHROME2", ""):
                if self._cache_value_matrix is None:
                    self._cache_value_matrix = calcular_hu_ct(
                        image_path=file_path,
                        rescale_slope=frame.get("rescale_slope"),
                        rescale_intercept=frame.get("rescale_intercept"),
                    )
                self._current_value_matrix = self._cache_value_matrix
                self._current_units_label = "HU"
                self._apply_ct_window(self._cache_value_matrix)
                self._update_pet_overlay(None, None)

            elif (self._modality == MODALITY_PT or str(self._modality).upper() in ("PT", "PET")) and photo_interp in ("MONOCHROME1", "MONOCHROME2", ""):
                if self._cache_value_matrix is None:
                    try:
                        self._cache_value_matrix = calcular_suv_pt(
                            image_path=file_path,
                            rescale_slope=frame.get("rescale_slope"),
                            rescale_intercept=frame.get("rescale_intercept"),
                            patient_weight=frame.get("patient_weight"),
                            radionuclide_total_dose=frame.get("radionuclide_total_dose"),
                            radionuclide_half_life=frame.get("radionuclide_half_life"),
                            radiopharmaceutical_start_time=frame.get("radiopharmaceutical_start_time"),
                            series_time=frame.get("series_time"),
                        )
                    except Exception as e_suv:
                        logger.warning("Error al calcular SUV con calcular_suv_pt, usando pixel_array: %s", e_suv)
                        self._cache_value_matrix = arr.astype(np.float32)
                self._current_value_matrix = self._cache_value_matrix
                self._current_units_label = "SUV"

                if self._series_pet_max_suv is None:
                    pos_vals = self._cache_value_matrix[self._cache_value_matrix > 0]
                    slice_max = float(np.percentile(pos_vals, 99.5)) if len(pos_vals) > 0 else (
                        float(np.nanmax(self._cache_value_matrix)) if self._cache_value_matrix.size > 0 else 1.0
                    )
                    self._series_pet_max_suv = max(slice_max, 0.1)
                    pct = self.pet_suv_max_slider.value() / 100.0
                    self._series_pet_vmax = max(0.01, pct * self._series_pet_max_suv)
                    self.pet_suv_max_label.setText("SUV Máx: %.2f" % self._series_pet_vmax)
                elif not self._user_adjusted_suv:
                    pos_vals = self._cache_value_matrix[self._cache_value_matrix > 0]
                    slice_p99 = float(np.percentile(pos_vals, 99.5)) if len(pos_vals) > 0 else 0.0
                    if slice_p99 > self._series_pet_max_suv:
                        self._series_pet_max_suv = slice_p99
                        pct = self.pet_suv_max_slider.value() / 100.0
                        self._series_pet_vmax = max(0.01, pct * self._series_pet_max_suv)
                        self.pet_suv_max_label.setText("SUV Máx: %.2f" % self._series_pet_vmax)

                self._apply_pet_colormap(self._cache_value_matrix)

                dose_val = frame.get("radionuclide_total_dose")
                if dose_val is None and ds is not None:
                    try:
                        seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
                        if seq and len(seq) > 0:
                            dose_val = getattr(seq[0], "RadionuclideTotalDose", None)
                        if dose_val is None:
                            dose_val = getattr(ds, "RadionuclideTotalDose", None)
                    except Exception:
                        pass
                date_val = (
                    frame.get("study_date")
                    or frame.get("acquisition_date")
                    or frame.get("series_date")
                    or (getattr(ds, "AcquisitionDate", None) if ds is not None else None)
                    or (getattr(ds, "StudyDate", None) if ds is not None else None)
                )
                self._update_pet_overlay(dose_val, date_val)

            else:
                self._current_value_matrix = arr
                self._current_units_label = "val"
                self._display_grayscale(arr)
                self._update_pet_overlay(None, None)

        except Exception as exc:
            self.image_label.setText("Error al cargar la imagen:\n%s" % exc)

    def _apply_ct_window(self, hu_matrix):
        c = self._ct_window_center
        w = max(self._ct_window_width, 1.0)
        norm = np.clip((hu_matrix.astype(np.float32) - (c - w / 2.0)) / w, 0.0, 1.0)
        uint8_arr = np.ascontiguousarray((norm * 255).astype(np.uint8))
        h, w_img = uint8_arr.shape[:2]
        qimg = QImage(uint8_arr.data, w_img, h, w_img, QImage.Format_Grayscale8)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._scale_and_set_pixmap()

    def _apply_pet_colormap(self, suv_matrix):
        if self._series_pet_vmax is not None and self._series_pet_vmax > 0:
            vmax = max(self._series_pet_vmax, 1e-6)
        else:
            pos_vals = suv_matrix[suv_matrix > 0]
            vmax = float(np.percentile(pos_vals, 99.5)) if len(pos_vals) > 0 else 1.0
            vmax = max(vmax, 1e-6)
        norm = np.clip(suv_matrix.astype(np.float32) / vmax, 0.0, 1.0)
        cmap = plt.get_cmap("hot")
        rgba = cmap(norm)
        rgb_uint8 = np.ascontiguousarray((rgba[..., :3] * 255).astype(np.uint8))
        h, w_img, _ = rgb_uint8.shape
        qimg = QImage(rgb_uint8.data, w_img, h, 3 * w_img, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._scale_and_set_pixmap()

    def _display_rgb(self, rgb_arr):
        rgb_uint8 = np.ascontiguousarray(rgb_arr.astype(np.uint8))
        h, w_img = rgb_uint8.shape[:2]
        channels = rgb_uint8.shape[2] if rgb_uint8.ndim == 3 else 1
        if channels == 4:
            qimg = QImage(rgb_uint8.data, w_img, h, 4 * w_img, QImage.Format_RGBA8888)
        elif channels == 3:
            qimg = QImage(rgb_uint8.data, w_img, h, 3 * w_img, QImage.Format_RGB888)
        else:
            qimg = QImage(rgb_uint8.data, w_img, h, w_img, QImage.Format_Grayscale8)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._scale_and_set_pixmap()

    def _display_grayscale(self, arr):
        farr = arr.astype(np.float32)
        amin, amax = float(np.min(farr)), float(np.max(farr))
        rng = amax - amin if amax > amin else 1.0
        norm = np.clip((farr - amin) / rng, 0.0, 1.0)
        uint8_arr = np.ascontiguousarray((norm * 255).astype(np.uint8))
        h, w_img = uint8_arr.shape[:2]
        qimg = QImage(uint8_arr.data, w_img, h, w_img, QImage.Format_Grayscale8)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._scale_and_set_pixmap()

    def _scale_and_set_pixmap(self):
        if self._current_pixmap is None:
            return
        lbl_size = self.image_label.size()
        if lbl_size.width() > 10 and lbl_size.height() > 10:
            scaled = self._current_pixmap.scaled(
                lbl_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_and_set_pixmap()

    def _on_pixel_hovered(self, label_x, label_y):
        if self._current_pixmap is None or self._current_value_matrix is None:
            return
        lbl_size = self.image_label.size()
        pix_size = self._current_pixmap.size()
        scaled = self._current_pixmap.scaled(lbl_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        scaled_size = scaled.size()

        offset_x = (lbl_size.width() - scaled_size.width()) / 2.0
        offset_y = (lbl_size.height() - scaled_size.height()) / 2.0

        if not (offset_x <= label_x < offset_x + scaled_size.width() and
                offset_y <= label_y < offset_y + scaled_size.height()):
            self.value_label.setText("")
            return

        norm_x = (label_x - offset_x) / scaled_size.width()
        norm_y = (label_y - offset_y) / scaled_size.height()

        mat_h, mat_w = self._current_value_matrix.shape[:2]
        col = int(np.clip(norm_x * mat_w, 0, mat_w - 1))
        row = int(np.clip(norm_y * mat_h, 0, mat_h - 1))

        if self._current_value_matrix.ndim == 3:
            pixel = self._current_value_matrix[row, col]
            if len(pixel) >= 3:
                self.value_label.setText(f"RGB: ({pixel[0]}, {pixel[1]}, {pixel[2]})")
            else:
                self.value_label.setText(f"{pixel}")
        else:
            val = float(self._current_value_matrix[row, col])
            units = self._current_units_label or ""
            if units == "HU":
                self.value_label.setText(f"{val:+.1f} HU")
            elif units == "SUV":
                self.value_label.setText(f"{val:.2f} SUV")
            elif units:
                self.value_label.setText(f"{val:.1f} {units}")
            else:
                if abs(val - round(val)) < 1e-4:
                    self.value_label.setText(f"{int(round(val))}")
                else:
                    self.value_label.setText(f"{val:.1f}")

    def _on_pixel_left(self):
        self.value_label.setText("")


class VTKCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #101217; border: 1px solid #2d3139;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(30, 30)
        self._current_pixmap = None

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(40)
        self._resize_timer.timeout.connect(self.render_scene)

        if VTK_AVAILABLE:
            self.renderer = vtkRenderer()
            self.renderer.SetBackground(0.10, 0.11, 0.14)
            self.renderer.AutomaticLightCreationOn()

            self.render_window = vtkRenderWindow()
            self.render_window.SetOffScreenRendering(1)
            self.render_window.SetSize(600, 600)
            self.render_window.AddRenderer(self.renderer)

            self.w2if = vtkWindowToImageFilter()
            self.w2if.SetInput(self.render_window)
        else:
            self.renderer = None
            self.render_window = None
            self.w2if = None

        self._last_mouse_pos = None

    def minimumSizeHint(self):
        return QSize(30, 30)

    def sizeHint(self):
        return QSize(100, 100)

    def render_scene(self):
        if not VTK_AVAILABLE or not self.render_window:
            self.setText("VTK no disponible para renderizado 3D.")
            return

        w = max(self.width(), 100)
        h = max(self.height(), 100)
        self.render_window.SetSize(w, h)
        self.render_window.Render()
        self.w2if.Modified()
        self.w2if.Update()

        vtk_img = self.w2if.GetOutput()
        dims = vtk_img.GetDimensions()
        scalars = vtk_img.GetPointData().GetScalars()
        if scalars is None:
            return

        arr = numpy_support.vtk_to_numpy(scalars).reshape(dims[1], dims[0], -1)
        arr = np.ascontiguousarray(np.flipud(arr))

        img_h, img_w, c = arr.shape
        qimg = QImage(arr.data, img_w, img_h, c * img_w, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg)
        self._scale_and_set_pixmap()

    def _scale_and_set_pixmap(self):
        if self._current_pixmap is None:
            return
        lbl_size = self.size()
        if lbl_size.width() > 10 and lbl_size.height() > 10:
            scaled = self._current_pixmap.scaled(
                lbl_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)

    def mousePressEvent(self, event):
        self._last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if not VTK_AVAILABLE or self._last_mouse_pos is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = pos

        camera = self.renderer.GetActiveCamera()
        if event.buttons() & Qt.LeftButton:
            camera.Azimuth(-dx * 0.5)
            camera.Elevation(dy * 0.5)
            camera.OrthogonalizeViewUp()
            self.render_scene()
        elif event.buttons() & Qt.RightButton:
            factor = 1.0 + dy * 0.01
            if factor > 0:
                camera.Dolly(factor)
                self.renderer.ResetCameraClippingRange()
                self.render_scene()

    def wheelEvent(self, event):
        if not VTK_AVAILABLE:
            return
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        camera = self.renderer.GetActiveCamera()
        camera.Dolly(factor)
        self.renderer.ResetCameraClippingRange()
        self.render_scene()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_and_set_pixmap()
        self._resize_timer.start()


class Viewer3DWidget(QWidget):
    def minimumSizeHint(self):
        return QSize(250, 150)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ct_actor = None
        self._pet_volume = None
        self._outline_actor = None
        self._pet_opacity_func = None
        self._max_suv = 1.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        top_controls_layout = QHBoxLayout()
        self.info_label = QLabel("Render 3D: Sin datos")
        self.info_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        top_controls_layout.addWidget(self.info_label)
        top_controls_layout.addStretch()

        btn_anterior = QPushButton("Anterior")
        btn_anterior.clicked.connect(lambda: self.set_view_preset("anterior"))
        btn_lateral = QPushButton("Lateral")
        btn_lateral.clicked.connect(lambda: self.set_view_preset("lateral_r"))
        btn_superior = QPushButton("Superior")
        btn_superior.clicked.connect(lambda: self.set_view_preset("superior"))
        btn_reset = QPushButton("Reset Cámara")
        btn_reset.clicked.connect(self.reset_camera)

        top_controls_layout.addWidget(btn_anterior)
        top_controls_layout.addWidget(btn_lateral)
        top_controls_layout.addWidget(btn_superior)
        top_controls_layout.addWidget(btn_reset)
        layout.addLayout(top_controls_layout)

        self._ct_opacity_prefix = "Opacidad CT"
        self._pet_opacity_prefix = "Opacidad PET"

        opacity_row = QHBoxLayout()
        self.ct_opacity_widget = QWidget()
        ct_h = QHBoxLayout(self.ct_opacity_widget)
        ct_h.setContentsMargins(0, 0, 0, 0)
        ct_h.setSpacing(6)
        self.ct_opacity_label = QLabel("Opacidad CT: 30%")
        self.ct_opacity_slider = QSlider(Qt.Horizontal)
        self.ct_opacity_slider.setRange(0, 100)
        self.ct_opacity_slider.setValue(30)
        self.ct_opacity_slider.setFixedWidth(100)
        self.ct_opacity_slider.valueChanged.connect(self._on_ct_opacity_changed)
        ct_h.addWidget(self.ct_opacity_label)
        ct_h.addWidget(self.ct_opacity_slider)
        opacity_row.addWidget(self.ct_opacity_widget)

        self.pet_opacity_widget = QWidget()
        pet_h = QHBoxLayout(self.pet_opacity_widget)
        pet_h.setContentsMargins(12, 0, 0, 0)
        pet_h.setSpacing(6)
        self.pet_opacity_label = QLabel("Opacidad PET: 80%")
        self.pet_opacity_slider = QSlider(Qt.Horizontal)
        self.pet_opacity_slider.setRange(0, 100)
        self.pet_opacity_slider.setValue(80)
        self.pet_opacity_slider.setFixedWidth(100)
        self.pet_opacity_slider.valueChanged.connect(self._on_pet_opacity_changed)
        pet_h.addWidget(self.pet_opacity_label)
        pet_h.addWidget(self.pet_opacity_slider)
        opacity_row.addWidget(self.pet_opacity_widget)

        opacity_row.addStretch()
        layout.addLayout(opacity_row)

        self.canvas = VTKCanvas(self)
        layout.addWidget(self.canvas, 1)

    def _on_ct_opacity_changed(self, value):
        self.ct_opacity_label.setText(f"{self._ct_opacity_prefix}: {value}%")
        if self._ct_actor:
            self._ct_actor.GetProperty().SetOpacity(value / 100.0)
            self.canvas.render_scene()

    def _on_pet_opacity_changed(self, value):
        self.pet_opacity_label.setText(f"{self._pet_opacity_prefix}: {value}%")
        if self._pet_opacity_func and self._max_suv > 0:
            scale = value / 100.0
            self._pet_opacity_func.RemoveAllPoints()
            self._pet_opacity_func.AddPoint(0.0, 0.0)
            self._pet_opacity_func.AddPoint(0.05 * self._max_suv, 0.0)
            self._pet_opacity_func.AddPoint(0.20 * self._max_suv, 0.20 * scale)
            self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.60 * scale)
            self._pet_opacity_func.AddPoint(self._max_suv, 0.90 * scale)
            self.canvas.render_scene()

    def clear(self):
        if not VTK_AVAILABLE or not self.canvas.renderer:
            return
        self.canvas.renderer.RemoveAllViewProps()
        self._ct_actor = None
        self._pet_volume = None
        self._outline_actor = None
        self._pet_opacity_func = None
        self.info_label.setText("Render 3D: Sin datos")
        self.ct_opacity_widget.setVisible(False)
        self.pet_opacity_widget.setVisible(False)
        self.canvas.render_scene()

    def show_ct_volume(self, ct_volume, voxel_spacing=None, title=""):
        self.clear()
        if not VTK_AVAILABLE or ct_volume is None or ct_volume.size == 0:
            return

        spacing = voxel_spacing or [1.0, 1.0, 1.0]
        sp_x = float(spacing[0]) if len(spacing) > 0 else 1.0
        sp_y = float(spacing[1]) if len(spacing) > 1 else 1.0
        sp_z = float(spacing[2]) if len(spacing) > 2 else 1.0
        eff_spacing = [sp_x, sp_y, sp_z]

        mask, eff_sp, _ = render_3d.extract_silhouette_mask(
            ct_volume, voxel_spacing=eff_spacing, modality="ct", fast_mode=(ct_volume.shape[0] > 150)
        )
        polydata = render_3d.build_silhouette_polydata(mask, eff_sp or eff_spacing, smoothing_iterations=15)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        self._ct_actor = vtkActor()
        self._ct_actor.SetMapper(mapper)
        prop = self._ct_actor.GetProperty()
        prop.SetColor(0.80, 0.88, 0.96)
        prop.SetOpacity(0.85)
        prop.SetSpecular(0.40)
        prop.SetSpecularPower(30.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()

        self.canvas.renderer.AddActor(self._ct_actor)

        outline = vtkOutlineFilter()
        outline.SetInputData(polydata)
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self._outline_actor = vtkActor()
        self._outline_actor.SetMapper(outline_mapper)
        self._outline_actor.GetProperty().SetColor(0.0, 0.77, 1.0)
        self.canvas.renderer.AddActor(self._outline_actor)

        self.info_label.setText(f"Render 3D: Silueta CT {title}")
        self.ct_opacity_label.setText("Opacidad CT: 85%")
        self.ct_opacity_widget.setVisible(True)
        self.ct_opacity_slider.setValue(85)
        self.pet_opacity_widget.setVisible(False)

        self.reset_camera()

    def show_mri_volume(self, mri_volume, voxel_spacing=None, title=""):
        self.clear()
        if not VTK_AVAILABLE or mri_volume is None or mri_volume.size == 0:
            return

        spacing = voxel_spacing or [1.0, 1.0, 1.0]
        sp_x = float(spacing[0]) if len(spacing) > 0 else 1.0
        sp_y = float(spacing[1]) if len(spacing) > 1 else 1.0
        sp_z = float(spacing[2]) if len(spacing) > 2 else 1.0
        eff_spacing = [sp_x, sp_y, sp_z]

        mask, eff_sp, _ = render_3d.extract_silhouette_mask(
            mri_volume, voxel_spacing=eff_spacing, modality="mri", fast_mode=(mri_volume.shape[0] > 150)
        )
        polydata = render_3d.build_silhouette_polydata(mask, eff_sp or eff_spacing, smoothing_iterations=15)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        self._ct_actor = vtkActor()
        self._ct_actor.SetMapper(mapper)
        prop = self._ct_actor.GetProperty()
        prop.SetColor(0.82, 0.88, 0.94)
        prop.SetOpacity(0.85)
        prop.SetSpecular(0.40)
        prop.SetSpecularPower(30.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()

        self.canvas.renderer.AddActor(self._ct_actor)

        outline = vtkOutlineFilter()
        outline.SetInputData(polydata)
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self._outline_actor = vtkActor()
        self._outline_actor.SetMapper(outline_mapper)
        self._outline_actor.GetProperty().SetColor(0.2, 0.85, 0.4)
        self.canvas.renderer.AddActor(self._outline_actor)

        self.info_label.setText(f"Render 3D: Silueta MRI {title}")
        self.ct_opacity_label.setText("Opacidad MRI: 85%")
        self.ct_opacity_widget.setVisible(True)
        self.ct_opacity_slider.setValue(85)
        self.pet_opacity_widget.setVisible(False)

        self.reset_camera()

    def show_pet_volume(self, pet_volume, voxel_spacing=None, max_suv=None, title=""):
        self.clear()
        if not VTK_AVAILABLE or pet_volume is None or pet_volume.size == 0:
            return

        spacing = voxel_spacing or [1.0, 1.0, 1.0]
        sp_x = float(spacing[0]) if len(spacing) > 0 else 1.0
        sp_y = float(spacing[1]) if len(spacing) > 1 else 1.0
        sp_z = float(spacing[2]) if len(spacing) > 2 else 1.0

        self._max_suv = max_suv if max_suv and max_suv > 0 else (float(np.nanmax(pet_volume)) if pet_volume.size > 0 else 1.0)
        if self._max_suv <= 0:
            self._max_suv = 1.0

        if pet_volume.ndim == 3:
            nz, ny, nx = pet_volume.shape
        elif pet_volume.ndim == 2:
            nz = 1
            ny, nx = pet_volume.shape
            pet_volume = pet_volume.reshape((1, ny, nx))
        else:
            return

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(nx, ny, nz)
        vtk_img.SetSpacing(sp_x, sp_y, sp_z)

        flat_data = np.ascontiguousarray(pet_volume.astype(np.float32)).ravel()
        vtk_arr = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        vol_prop = vtkVolumeProperty()
        vol_prop.ShadeOn()
        vol_prop.SetInterpolationTypeToLinear()

        color_func = vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(0.15 * self._max_suv, 0.8, 0.0, 0.0)
        color_func.AddRGBPoint(0.40 * self._max_suv, 1.0, 0.6, 0.0)
        color_func.AddRGBPoint(0.80 * self._max_suv, 1.0, 1.0, 0.2)
        color_func.AddRGBPoint(self._max_suv, 1.0, 1.0, 0.9)
        vol_prop.SetColor(color_func)

        self._pet_opacity_func = vtkPiecewiseFunction()
        self._pet_opacity_func.AddPoint(0.0, 0.0)
        self._pet_opacity_func.AddPoint(0.05 * self._max_suv, 0.0)
        self._pet_opacity_func.AddPoint(0.20 * self._max_suv, 0.16)
        self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.48)
        self._pet_opacity_func.AddPoint(self._max_suv, 0.72)
        vol_prop.SetScalarOpacity(self._pet_opacity_func)

        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_img)

        self._pet_volume = vtkVolume()
        self._pet_volume.SetMapper(mapper)
        self._pet_volume.SetProperty(vol_prop)

        self.canvas.renderer.AddVolume(self._pet_volume)

        self.info_label.setText(f"Render 3D: Nube Radiactiva PET {title}")
        self.ct_opacity_widget.setVisible(False)
        self.pet_opacity_widget.setVisible(True)
        self.pet_opacity_slider.setValue(80)

        self.reset_camera()

    def show_fused_volume(self, ct_volume, pet_volume, voxel_spacing=None, max_suv=None, title=""):
        self.clear()
        if not VTK_AVAILABLE or ct_volume is None or pet_volume is None:
            return

        spacing = voxel_spacing or [1.0, 1.0, 1.0]
        sp_x = float(spacing[0]) if len(spacing) > 0 else 1.0
        sp_y = float(spacing[1]) if len(spacing) > 1 else 1.0
        sp_z = float(spacing[2]) if len(spacing) > 2 else 1.0
        eff_spacing = [sp_x, sp_y, sp_z]

        mask, eff_sp, _ = render_3d.extract_silhouette_mask(
            ct_volume, voxel_spacing=eff_spacing, modality="ct", fast_mode=(ct_volume.shape[0] > 150)
        )
        polydata = render_3d.build_silhouette_polydata(mask, eff_sp or eff_spacing, smoothing_iterations=15)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        self._ct_actor = vtkActor()
        self._ct_actor.SetMapper(mapper)
        prop = self._ct_actor.GetProperty()
        prop.SetColor(0.80, 0.88, 0.96)
        prop.SetOpacity(0.30)
        prop.SetSpecular(0.40)
        prop.SetSpecularPower(30.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()

        self.canvas.renderer.AddActor(self._ct_actor)

        self._max_suv = max_suv if max_suv and max_suv > 0 else (float(np.nanmax(pet_volume)) if pet_volume.size > 0 else 1.0)
        if self._max_suv <= 0:
            self._max_suv = 1.0

        if pet_volume.ndim == 3:
            nz, ny, nx = pet_volume.shape
        elif pet_volume.ndim == 2:
            nz = 1
            ny, nx = pet_volume.shape
            pet_volume = pet_volume.reshape((1, ny, nx))
        else:
            return

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(nx, ny, nz)
        vtk_img.SetSpacing(sp_x, sp_y, sp_z)

        flat_data = np.ascontiguousarray(pet_volume.astype(np.float32)).ravel()
        vtk_arr = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        vol_prop = vtkVolumeProperty()
        vol_prop.ShadeOn()
        vol_prop.SetInterpolationTypeToLinear()

        color_func = vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(0.15 * self._max_suv, 0.8, 0.0, 0.0)
        color_func.AddRGBPoint(0.40 * self._max_suv, 1.0, 0.6, 0.0)
        color_func.AddRGBPoint(0.80 * self._max_suv, 1.0, 1.0, 0.2)
        color_func.AddRGBPoint(self._max_suv, 1.0, 1.0, 0.9)
        vol_prop.SetColor(color_func)

        self._pet_opacity_func = vtkPiecewiseFunction()
        self._pet_opacity_func.AddPoint(0.0, 0.0)
        self._pet_opacity_func.AddPoint(0.05 * self._max_suv, 0.0)
        self._pet_opacity_func.AddPoint(0.20 * self._max_suv, 0.16)
        self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.48)
        self._pet_opacity_func.AddPoint(self._max_suv, 0.72)
        vol_prop.SetScalarOpacity(self._pet_opacity_func)

        vol_mapper = vtkSmartVolumeMapper()
        vol_mapper.SetInputData(vtk_img)

        self._pet_volume = vtkVolume()
        self._pet_volume.SetMapper(vol_mapper)
        self._pet_volume.SetProperty(vol_prop)

        self.canvas.renderer.AddVolume(self._pet_volume)

        self.info_label.setText(f"Render 3D: Fusión CT + PET {title}")
        self.ct_opacity_widget.setVisible(True)
        self.ct_opacity_slider.setValue(30)
        self.pet_opacity_widget.setVisible(True)
        self.pet_opacity_slider.setValue(80)

        self.reset_camera()

    def reset_camera(self):
        if not VTK_AVAILABLE or not self.canvas.renderer:
            return
        self.canvas.renderer.ResetCamera()
        self.set_view_preset("anterior")

    def set_view_preset(self, preset):
        if not VTK_AVAILABLE or not self.canvas.renderer:
            return
        camera = self.canvas.renderer.GetActiveCamera()
        bounds = self.canvas.renderer.ComputeVisiblePropBounds()
        if bounds[0] > bounds[1]:
            return
        center = [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
        diag = np.sqrt(
            (bounds[1] - bounds[0]) ** 2 +
            (bounds[3] - bounds[2]) ** 2 +
            (bounds[5] - bounds[4]) ** 2
        )
        dist = max(diag * 1.6, 100.0)

        if preset == "anterior":
            camera.SetPosition(center[0], center[1] - dist, center[2])
            camera.SetFocalPoint(*center)
            camera.SetViewUp(0, 0, 1)
        elif preset == "posterior":
            camera.SetPosition(center[0], center[1] + dist, center[2])
            camera.SetFocalPoint(*center)
            camera.SetViewUp(0, 0, 1)
        elif preset == "lateral_r":
            camera.SetPosition(center[0] - dist, center[1], center[2])
            camera.SetFocalPoint(*center)
            camera.SetViewUp(0, 0, 1)
        elif preset == "lateral_l":
            camera.SetPosition(center[0] + dist, center[1], center[2])
            camera.SetFocalPoint(*center)
            camera.SetViewUp(0, 0, 1)
        elif preset == "superior":
            camera.SetPosition(center[0], center[1], center[2] + dist)
            camera.SetFocalPoint(*center)
            camera.SetViewUp(0, 1, 0)

        self.canvas.renderer.ResetCameraClippingRange()
        self.canvas.render_scene()

    def show_segmentation_volume(self, seg_volume, voxel_spacing=None, pet_volume=None, max_suv=None, title="Segmentación"):
        self.clear()
        if not VTK_AVAILABLE or seg_volume is None or seg_volume.size == 0:
            return

        spacing = voxel_spacing or [1.0, 1.0, 1.0]
        sp_x = float(spacing[0]) if len(spacing) > 0 else 1.0
        sp_y = float(spacing[1]) if len(spacing) > 1 else 1.0
        sp_z = float(spacing[2]) if len(spacing) > 2 else 1.0
        eff_spacing = [sp_x, sp_y, sp_z]

        bin_mask = (seg_volume > 0).astype(np.uint8)
        if not np.any(bin_mask):
            self.info_label.setText(f"Render 3D: {title} (Máscara vacía)")
            return

        polydata = render_3d.build_silhouette_polydata(bin_mask, eff_spacing, smoothing_iterations=15)
        if not polydata or polydata.GetNumberOfPoints() == 0:
            self.info_label.setText(f"Render 3D: {title} (Sin geometría)")
            return

        # Centrar la geometría del objeto segmentado exactamente en el origen (0, 0, 0)
        # para que la cámara y todas las rotaciones 3D giren estrictamente con respecto
        # al centro del objeto segmentado y no al centro del cuerpo completo.
        c_x, c_y, c_z = polydata.GetCenter()
        if vtkTransform is not None and vtkTransformPolyDataFilter is not None:
            trans = vtkTransform()
            trans.Translate(-c_x, -c_y, -c_z)
            tfilter = vtkTransformPolyDataFilter()
            tfilter.SetInputData(polydata)
            tfilter.SetTransform(trans)
            tfilter.Update()
            polydata = tfilter.GetOutput()

        has_pet = bool(pet_volume is not None and np.any(pet_volume > 0))
        default_surface_opacity = 0.35 if has_pet else 0.95

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        self._ct_actor = vtkActor()
        self._ct_actor.SetMapper(mapper)
        prop = self._ct_actor.GetProperty()
        prop.SetColor(0.85, 0.85, 0.85)
        prop.SetOpacity(default_surface_opacity)
        prop.SetSpecular(0.45)
        prop.SetSpecularPower(35.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()

        self.canvas.renderer.AddActor(self._ct_actor)

        outline = vtkOutlineFilter()
        outline.SetInputData(polydata)
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self._outline_actor = vtkActor()
        self._outline_actor.SetMapper(outline_mapper)
        if has_pet:
            self._outline_actor.GetProperty().SetColor(1.0, 0.9, 0.0)
        else:
            self._outline_actor.GetProperty().SetColor(0.0, 0.85, 0.95)
        self.canvas.renderer.AddActor(self._outline_actor)

        if has_pet:
            pet_arr = pet_volume.astype(np.float32)
            if pet_arr.shape == bin_mask.shape:
                masked_pet = np.where(bin_mask > 0, pet_arr, 0.0)
            elif pet_arr.shape == (bin_mask.shape[2], bin_mask.shape[1], bin_mask.shape[0]):
                pet_t = np.transpose(pet_arr, (2, 1, 0))
                masked_pet = np.where(bin_mask > 0, pet_t, 0.0)
            else:
                masked_pet = np.where(bin_mask > 0, pet_arr, 0.0)

            eff_max_suv = float(max_suv) if max_suv and max_suv > 0 else (float(np.nanmax(masked_pet)) if masked_pet.size > 0 else 1.0)
            if eff_max_suv <= 0:
                eff_max_suv = 1.0
            self._max_suv = eff_max_suv

            # Acotar el volumen PET a la caja delimitadora del objeto segmentado
            # y desplazar su origen al sistema centrado en (0, 0, 0)
            z_idx, y_idx, x_idx = np.where(bin_mask > 0)
            if len(z_idx) > 0:
                min_z, max_z = int(np.min(z_idx)), int(np.max(z_idx))
                min_y, max_y = int(np.min(y_idx)), int(np.max(y_idx))
                min_x, max_x = int(np.min(x_idx)), int(np.max(x_idx))
                margin = 1
                min_z = max(0, min_z - margin)
                max_z = min(bin_mask.shape[0] - 1, max_z + margin)
                min_y = max(0, min_y - margin)
                max_y = min(bin_mask.shape[1] - 1, max_y + margin)
                min_x = max(0, min_x - margin)
                max_x = min(bin_mask.shape[2] - 1, max_x + margin)

                cropped_pet = masked_pet[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
            else:
                cropped_pet = masked_pet
                min_z, min_y, min_x = 0, 0, 0

            if cropped_pet.ndim == 3:
                nz, ny, nx = cropped_pet.shape
            else:
                nz, ny, nx = 1, cropped_pet.shape[0], cropped_pet.shape[1]
                cropped_pet = cropped_pet.reshape((1, ny, nx))

            vtk_img = vtkImageData()
            vtk_img.SetDimensions(nx, ny, nz)
            vtk_img.SetSpacing(sp_x, sp_y, sp_z)
            vtk_img.SetOrigin(min_x * sp_x - c_x, min_y * sp_y - c_y, min_z * sp_z - c_z)

            flat_data = np.ascontiguousarray(cropped_pet.astype(np.float32)).ravel()
            vtk_arr = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_img.GetPointData().SetScalars(vtk_arr)

            vol_prop = vtkVolumeProperty()
            vol_prop.ShadeOn()
            vol_prop.SetInterpolationTypeToLinear()

            color_func = vtkColorTransferFunction()
            color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color_func.AddRGBPoint(0.12 * self._max_suv, 0.8, 0.0, 0.0)
            color_func.AddRGBPoint(0.35 * self._max_suv, 1.0, 0.55, 0.0)
            color_func.AddRGBPoint(0.75 * self._max_suv, 1.0, 1.0, 0.1)
            color_func.AddRGBPoint(self._max_suv, 1.0, 1.0, 0.95)
            vol_prop.SetColor(color_func)

            self._pet_opacity_func = vtkPiecewiseFunction()
            self._pet_opacity_func.AddPoint(0.0, 0.0)
            self._pet_opacity_func.AddPoint(0.04 * self._max_suv, 0.0)
            self._pet_opacity_func.AddPoint(0.18 * self._max_suv, 0.20)
            self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.55)
            self._pet_opacity_func.AddPoint(self._max_suv, 0.85)
            vol_prop.SetScalarOpacity(self._pet_opacity_func)

            vol_mapper = vtkSmartVolumeMapper()
            vol_mapper.SetInputData(vtk_img)

            self._pet_volume = vtkVolume()
            self._pet_volume.SetMapper(vol_mapper)
            self._pet_volume.SetProperty(vol_prop)
            self.canvas.renderer.AddVolume(self._pet_volume)

            self.info_label.setText(f"Render 3D: {title} + Nube Radioactiva PET")
            self._ct_opacity_prefix = "Opacidad Contorno 3D"
            self._pet_opacity_prefix = "Opacidad Nube PET"
            self.ct_opacity_label.setText(f"Opacidad Contorno 3D: {int(default_surface_opacity * 100)}%")
            self.ct_opacity_widget.setVisible(True)
            self.ct_opacity_slider.setValue(int(default_surface_opacity * 100))
            self.pet_opacity_label.setText("Opacidad Nube PET: 80%")
            self.pet_opacity_widget.setVisible(True)
            self.pet_opacity_slider.setValue(80)
        else:
            self.info_label.setText(f"Render 3D: {title} (Sin Colores)")
            self._ct_opacity_prefix = "Opacidad Segmentación"
            self.ct_opacity_label.setText("Opacidad Segmentación: 95%")
            self.ct_opacity_widget.setVisible(True)
            self.ct_opacity_slider.setValue(95)
            self.pet_opacity_widget.setVisible(False)

        self.reset_camera()


# Visualización multi-estudio: touchpad 3D y mosaicos 2D/3D

class _TouchpadCanvas(QFrame):
    rotate_requested = Signal(float, float)
    pan_requested = Signal(float, float)
    zoom_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setMinimumSize(140, 110)
        self.setMaximumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMouseTracking(True)
        self.setCursor(Qt.OpenHandCursor)
        self._last_pos = None
        self._active_mode = "rotate"
        self._is_dragging = False

    def set_mode(self, mode):
        self._active_mode = mode
        if mode == "rotate":
            self.setCursor(Qt.OpenHandCursor)
        elif mode == "pan":
            self.setCursor(Qt.SizeAllCursor)
        elif mode == "zoom":
            self.setCursor(Qt.SizeVerCursor)
        self.update()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._last_pos = event.position().toPoint()
            self._is_dragging = True
            if self._active_mode == "rotate":
                self.setCursor(Qt.ClosedHandCursor)
            self.update()

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        self._is_dragging = False
        self.set_mode(self._active_mode)
        self.update()

    def mouseMoveEvent(self, event):
        if not self._is_dragging or self._last_pos is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        self._last_pos = pos

        buttons = event.buttons()
        if buttons & Qt.RightButton or self._active_mode == "pan":
            self.pan_requested.emit(float(dx), float(dy))
        elif buttons & Qt.MiddleButton or self._active_mode == "zoom":
            factor = 1.0 + dy * 0.01
            if factor > 0.05:
                self.zoom_requested.emit(float(factor))
        else:  # LeftButton o modo rotate
            self.rotate_requested.emit(float(dx), float(dy))
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self.zoom_requested.emit(float(factor))
        event.accept()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pal = self.palette()
        mid_color = pal.color(QPalette.Mid)
        highlight_color = pal.color(QPalette.Highlight)

        w = self.width()
        h = self.height()
        cx, cy = w / 2.0, h / 2.0

        # Guías suaves con color nativo del sistema
        painter.setPen(QPen(mid_color, 1, Qt.DotLine))
        painter.drawLine(int(cx), 8, int(cx), h - 8)
        painter.drawLine(8, int(cy), w - 8, int(cy))
        painter.drawEllipse(QPointF(cx, cy), min(w, h) * 0.28, min(w, h) * 0.28)

        # Indicador central
        dot_color = highlight_color if self._is_dragging else mid_color
        painter.setBrush(QBrush(dot_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 3.5, 3.5)


class MultiStudy3DTouchpad(QWidget):
    rotate_requested = Signal(float, float)
    pan_requested = Signal(float, float)
    zoom_requested = Signal(float)
    preset_requested = Signal(str)
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Barra de modo: Rotar | Panear | Zoom
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(4)

        self.btn_mode_rot = QPushButton("Rotar")
        self.btn_mode_rot.setCheckable(True)
        self.btn_mode_rot.setChecked(True)
        self.btn_mode_rot.clicked.connect(lambda: self._set_mode("rotate"))

        self.btn_mode_pan = QPushButton("Pan")
        self.btn_mode_pan.setCheckable(True)
        self.btn_mode_pan.clicked.connect(lambda: self._set_mode("pan"))

        self.btn_mode_zoom = QPushButton("Zoom")
        self.btn_mode_zoom.setCheckable(True)
        self.btn_mode_zoom.clicked.connect(lambda: self._set_mode("zoom"))

        mode_row.addWidget(self.btn_mode_rot)
        mode_row.addWidget(self.btn_mode_pan)
        mode_row.addWidget(self.btn_mode_zoom)
        layout.addLayout(mode_row)

        # Superficie táctil Touchpad
        self.touchpad_canvas = _TouchpadCanvas(self)
        self.touchpad_canvas.rotate_requested.connect(self.rotate_requested.emit)
        self.touchpad_canvas.pan_requested.connect(self.pan_requested.emit)
        self.touchpad_canvas.zoom_requested.connect(self.zoom_requested.emit)
        layout.addWidget(self.touchpad_canvas)

        # Botones de orientación rápida y reset
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(3)

        btn_ant = QPushButton("Ant")
        btn_ant.setToolTip("Vista Anterior")
        btn_ant.clicked.connect(lambda: self.preset_requested.emit("anterior"))

        btn_lat = QPushButton("Lat")
        btn_lat.setToolTip("Vista Lateral Derecha")
        btn_lat.clicked.connect(lambda: self.preset_requested.emit("lateral_r"))

        btn_sup = QPushButton("Sup")
        btn_sup.setToolTip("Vista Superior")
        btn_sup.clicked.connect(lambda: self.preset_requested.emit("superior"))

        btn_rst = QPushButton("Reset")
        btn_rst.setToolTip("Restablecer orientación de cámaras")
        btn_rst.clicked.connect(self.reset_requested.emit)

        btn_row.addWidget(btn_ant)
        btn_row.addWidget(btn_lat)
        btn_row.addWidget(btn_sup)
        btn_row.addWidget(btn_rst)
        layout.addLayout(btn_row)

    def _set_mode(self, mode):
        self.btn_mode_rot.setChecked(mode == "rotate")
        self.btn_mode_pan.setChecked(mode == "pan")
        self.btn_mode_zoom.setChecked(mode == "zoom")
        self.touchpad_canvas.set_mode(mode)


class _Mosaic2DTile(QFrame):
    pixel_hovered = Signal(float, float, dict, object)
    pixel_left = Signal()

    def __init__(self, study_dict, parent=None):
        super().__init__(parent)
        self.study_dict = dict(study_dict or {})
        self._current_pixmap = None
        self._current_value_matrix = None
        self._current_pet_matrix = None
        self._shift_range = None
        self._current_pet_suv_max = self.study_dict.get("global_max_suv")
        self._setup_ui()

    def set_shift_range(self, shift_range):
        self._shift_range = shift_range

    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        pat = self.study_dict.get("patient_name", "Paciente")
        desc = self.study_dict.get("description", "Estudio")
        mod = self.study_dict.get("modality", "")

        self.lbl_title = QLabel(f"<b>{pat}</b> - {desc} [{mod}]")
        self.lbl_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_title.setMinimumWidth(0)
        self.lbl_title.setToolTip(f"{pat} | {desc} [{mod}]")
        header.addWidget(self.lbl_title, 1)

        self.lbl_slice = QLabel("0 / 0")
        header.addWidget(self.lbl_slice)
        layout.addLayout(header)

        self.image_label = _HoverImageLabel("Sin datos")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.pixel_hovered.connect(self._on_pixel_hovered)
        self.image_label.pixel_left.connect(self._on_pixel_left)
        self.image_label.resized.connect(lambda _: self._update_display())

        img_overlay_layout = QVBoxLayout(self.image_label)
        img_overlay_layout.setContentsMargins(6, 6, 6, 6)
        img_overlay_layout.addStretch()
        self.lbl_pet_overlay = QLabel("")
        self.lbl_pet_overlay.setStyleSheet("color: #ffff00; font-weight: bold; font-size: 11px; background: transparent;")
        self.lbl_pet_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.lbl_pet_overlay.setWordWrap(True)
        self.lbl_pet_overlay.setVisible(False)
        img_overlay_layout.addWidget(self.lbl_pet_overlay)

        layout.addWidget(self.image_label, 1)

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(120, 120)

    def _on_pixel_hovered(self, x, y):
        self.pixel_hovered.emit(x, y, self.study_dict, self)

    def _on_pixel_left(self):
        self.pixel_left.emit()

    def render_slice(self, slice_ratio, ct_window_center=40.0, ct_window_width=400.0, ct_alpha=0.6, pet_alpha=0.4, pet_suv_max=None):
        vol_data = self.study_dict.get("volume_data") or {}
        modality = self.study_dict.get("modality", "CT").upper()
        rel_idx = None
        if pet_suv_max is not None and float(pet_suv_max) > 0:
            self._current_pet_suv_max = float(pet_suv_max)
        elif self.study_dict.get("global_max_suv"):
            self._current_pet_suv_max = float(self.study_dict.get("global_max_suv"))

        if modality == "FUSION":
            ct_vol = vol_data.get("ct_volume")
            pet_vol = vol_data.get("pet_volume")
            if ct_vol is None or pet_vol is None:
                return
            num_slices = ct_vol.shape[0]
            if num_slices == 0:
                return

            if self._shift_range is not None and self._shift_range[2] > 0:
                s_start, _, s_common = self._shift_range
                rel_idx = min(s_common - 1, max(0, int(round(slice_ratio * (s_common - 1)))))
                slice_idx = min(num_slices - 1, max(0, s_start + rel_idx))
            else:
                slice_idx = min(num_slices - 1, max(0, int(round(slice_ratio * (num_slices - 1)))))

            ct_slice = ct_vol[slice_idx].astype(np.float32)
            pet_slice = pet_vol[slice_idx].astype(np.float32)

            c = ct_window_center
            w = max(ct_window_width, 1.0)
            ct_norm = np.clip((ct_slice - (c - w / 2.0)) / w, 0.0, 1.0)
            ct_rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

            eff_suv = (
                self._current_pet_suv_max
                or self.study_dict.get("global_max_suv")
                or vol_data.get("global_max_suv")
                or vol_data.get("max_suv")
                or 1.0
            )
            vmax = max(float(eff_suv), 1e-6)
            pet_norm = np.clip(pet_slice / vmax, 0.0, 1.0)
            cmap = plt.get_cmap("hot")
            pet_rgba = cmap(pet_norm)
            pet_rgb = pet_rgba[..., :3]

            fused_rgb = np.clip(ct_rgb * ct_alpha + pet_rgb * pet_alpha, 0.0, 1.0)
            rgb_uint8 = np.ascontiguousarray((fused_rgb * 255).astype(np.uint8))
            h, w_img, _ = rgb_uint8.shape
            qimg = QImage(rgb_uint8.data, w_img, h, 3 * w_img, QImage.Format_RGB888)
            self._current_pixmap = QPixmap.fromImage(qimg)
            self._current_value_matrix = ct_slice
            self._current_pet_matrix = pet_slice

        elif modality in ("PT", "PET"):
            vol = vol_data.get("volume")
            if vol is None:
                return
            num_slices = vol.shape[0]
            if num_slices == 0:
                return

            if self._shift_range is not None and self._shift_range[2] > 0:
                s_start, _, s_common = self._shift_range
                rel_idx = min(s_common - 1, max(0, int(round(slice_ratio * (s_common - 1)))))
                slice_idx = min(num_slices - 1, max(0, s_start + rel_idx))
            else:
                slice_idx = min(num_slices - 1, max(0, int(round(slice_ratio * (num_slices - 1)))))

            pet_slice = vol[slice_idx].astype(np.float32)

            eff_suv = (
                self._current_pet_suv_max
                or self.study_dict.get("global_max_suv")
                or vol_data.get("global_max_suv")
                or vol_data.get("max_suv")
                or (np.nanmax(vol) if vol.size > 0 else 1.0)
            )
            vmax = max(float(eff_suv), 1e-6)
            pet_norm = np.clip(pet_slice / vmax, 0.0, 1.0)
            cmap = plt.get_cmap("hot")
            pet_rgba = cmap(pet_norm)
            rgb_uint8 = np.ascontiguousarray((pet_rgba[..., :3] * 255).astype(np.uint8))
            h, w_img, _ = rgb_uint8.shape
            qimg = QImage(rgb_uint8.data, w_img, h, 3 * w_img, QImage.Format_RGB888)
            self._current_pixmap = QPixmap.fromImage(qimg)
            self._current_value_matrix = pet_slice
            self._current_pet_matrix = None

        elif modality in ("MR", "MRI"):
            vol = vol_data.get("volume")
            if vol is None:
                return
            num_slices = vol.shape[0]
            if num_slices == 0:
                return

            if self._shift_range is not None and self._shift_range[2] > 0:
                s_start, _, s_common = self._shift_range
                rel_idx = min(s_common - 1, max(0, int(round(slice_ratio * (s_common - 1)))))
                slice_idx = min(num_slices - 1, max(0, s_start + rel_idx))
            else:
                slice_idx = min(num_slices - 1, max(0, int(round(slice_ratio * (num_slices - 1)))))

            mri_slice = vol[slice_idx].astype(np.float32)

            v_min = float(np.nanmin(mri_slice)) if mri_slice.size > 0 else 0.0
            v_max = float(np.nanmax(mri_slice)) if mri_slice.size > 0 else 1.0
            if v_max <= v_min:
                v_max = v_min + 1.0
            norm = np.clip((mri_slice - v_min) / (v_max - v_min), 0.0, 1.0)
            norm_uint8 = np.ascontiguousarray((norm * 255).astype(np.uint8))
            h, w_img = norm_uint8.shape
            qimg = QImage(norm_uint8.data, w_img, h, w_img, QImage.Format_Grayscale8)
            self._current_pixmap = QPixmap.fromImage(qimg)
            self._current_value_matrix = mri_slice
            self._current_pet_matrix = None

        else:
            vol = vol_data.get("volume")
            if vol is None:
                return
            num_slices = vol.shape[0]
            if num_slices == 0:
                return

            if self._shift_range is not None and self._shift_range[2] > 0:
                s_start, _, s_common = self._shift_range
                rel_idx = min(s_common - 1, max(0, int(round(slice_ratio * (s_common - 1)))))
                slice_idx = min(num_slices - 1, max(0, s_start + rel_idx))
            else:
                slice_idx = min(num_slices - 1, max(0, int(round(slice_ratio * (num_slices - 1)))))

            ct_slice = vol[slice_idx].astype(np.float32)

            c = ct_window_center
            w = max(ct_window_width, 1.0)
            ct_norm = np.clip((ct_slice - (c - w / 2.0)) / w, 0.0, 1.0)
            norm_uint8 = np.ascontiguousarray((ct_norm * 255).astype(np.uint8))
            h, w_img = norm_uint8.shape
            qimg = QImage(norm_uint8.data, w_img, h, w_img, QImage.Format_Grayscale8)
            self._current_pixmap = QPixmap.fromImage(qimg)
            self._current_value_matrix = ct_slice
            self._current_pet_matrix = None

        z_positions = vol_data.get("z_positions") or []
        z_str = ""
        if slice_idx < len(z_positions):
            z_str = f" (z: {z_positions[slice_idx]:.1f} mm)"

        if self._shift_range is not None and self._shift_range[2] > 0 and rel_idx is not None:
            _, _, s_common = self._shift_range
            self.lbl_slice.setText(f"{rel_idx + 1}/{s_common} (orig: {slice_idx + 1}/{num_slices}){z_str}")
        else:
            self.lbl_slice.setText(f"{slice_idx + 1}/{num_slices}{z_str}")

        if modality in ("FUSION", "PT", "PET"):
            metadata = vol_data.get("metadata") or {}
            dose_val = (
                self.study_dict.get("radionuclide_total_dose")
                or vol_data.get("radionuclide_total_dose")
                or metadata.get("radionuclide_total_dose")
            )
            date_val = (
                self.study_dict.get("acquisition_date")
                or self.study_dict.get("study_date")
                or vol_data.get("acquisition_date")
                or vol_data.get("study_date")
                or metadata.get("acquisition_date")
                or metadata.get("study_date")
            )
            dose_str = format_dose_mci(dose_val)
            date_str = format_dicom_date(date_val)
            parts = []
            if date_str:
                parts.append(date_str)
            if dose_str:
                parts.append(f"D0: {dose_str}")
            if parts:
                self.lbl_pet_overlay.setText("\n".join(parts))
                self.lbl_pet_overlay.setVisible(True)
            else:
                self.lbl_pet_overlay.clear()
                self.lbl_pet_overlay.setVisible(False)
        else:
            self.lbl_pet_overlay.clear()
            self.lbl_pet_overlay.setVisible(False)

        self._update_display()

    def _update_display(self):
        if self._current_pixmap and not self._current_pixmap.isNull():
            sz = self.image_label.size()
            if sz.width() > 10 and sz.height() > 10:
                scaled = self._current_pixmap.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def get_pixel_value_at(self, widget_x, widget_y):
        if self._current_value_matrix is None or self._current_pixmap is None or self._current_pixmap.isNull():
            return None
        lbl_w = self.image_label.width()
        lbl_h = self.image_label.height()
        pm_w = self._current_pixmap.width()
        pm_h = self._current_pixmap.height()
        if lbl_w <= 0 or lbl_h <= 0 or pm_w <= 0 or pm_h <= 0:
            return None

        scale = min(lbl_w / pm_w, lbl_h / pm_h)
        disp_w = pm_w * scale
        disp_h = pm_h * scale
        offset_x = (lbl_w - disp_w) / 2.0
        offset_y = (lbl_h - disp_h) / 2.0

        if not (offset_x <= widget_x < offset_x + disp_w and offset_y <= widget_y < offset_y + disp_h):
            return None

        mat_h, mat_w = self._current_value_matrix.shape[:2]
        img_x = int((widget_x - offset_x) / disp_w * mat_w)
        img_y = int((widget_y - offset_y) / disp_h * mat_h)
        img_x = min(mat_w - 1, max(0, img_x))
        img_y = min(mat_h - 1, max(0, img_y))

        val = float(self._current_value_matrix[img_y, img_x])
        pet_val = float(self._current_pet_matrix[img_y, img_x]) if self._current_pet_matrix is not None else None
        return {
            "x": img_x,
            "y": img_y,
            "val": val,
            "pet_val": pet_val,
            "modality": self.study_dict.get("modality", "CT"),
        }


class MultiStudyMosaic2DViewer(QWidget):
    PAGE_SIZE = 6
    page_changed = Signal(int, int)
    pixel_hovered = Signal(dict)
    pixel_left = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._studies = []
        self._shift_ranges = {}
        self._current_page = 0
        self._current_ratio = 0.5
        self._ct_window_center = 40.0
        self._ct_window_width = 400.0
        self._ct_alpha = 0.6
        self._pet_alpha = 0.4
        self._pet_suv_max = None
        self._active_tiles = []
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Barra de paginación superior
        self.nav_bar = QWidget()
        nav_h = QHBoxLayout(self.nav_bar)
        nav_h.setContentsMargins(0, 0, 0, 0)
        nav_h.setSpacing(8)

        self.btn_prev = QPushButton("◀ Anterior")
        self.btn_prev.clicked.connect(self._on_prev_page)
        nav_h.addWidget(self.btn_prev)

        self.lbl_page_info = QLabel("Página 1 de 1 (0 estudios)")
        self.lbl_page_info.setStyleSheet("color: #63b3ed; font-weight: bold;")
        nav_h.addWidget(self.lbl_page_info)

        self.btn_next = QPushButton("Siguiente ▶")
        self.btn_next.clicked.connect(self._on_next_page)
        nav_h.addWidget(self.btn_next)

        nav_h.addStretch()
        layout.addWidget(self.nav_bar)

        # Área de cuadrícula
        self.grid_container = QWidget()
        self.grid_container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        layout.addWidget(self.grid_container, 1)

    def minimumSizeHint(self):
        return QSize(80, 80)

    def sizeHint(self):
        return QSize(300, 200)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for tile in self._active_tiles:
            tile._update_display()

    def total_pages(self):
        return max(1, (len(self._studies) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)

    def set_studies(self, studies):
        self._studies = list(studies or [])
        self._shift_ranges = {}
        self._current_page = 0
        self._refresh_page()

    def clear(self):
        self._shift_ranges = {}
        self._pet_suv_max = None
        self.set_studies([])

    def set_page(self, page_idx):
        if page_idx < 0 or page_idx >= self.total_pages():
            return
        self._current_page = page_idx
        self._refresh_page()

    def _on_prev_page(self):
        if self._current_page > 0:
            self.set_page(self._current_page - 1)

    def _on_next_page(self):
        if self._current_page < self.total_pages() - 1:
            self.set_page(self._current_page + 1)

    def _refresh_page(self):
        # Limpiar tiles anteriores
        for tile in self._active_tiles:
            self.grid_layout.removeWidget(tile)
            tile.setParent(None)
            tile.deleteLater()
        self._active_tiles.clear()

        total = len(self._studies)
        tot_pages = self.total_pages()
        start = self._current_page * self.PAGE_SIZE
        page_items = self._studies[start : start + self.PAGE_SIZE]
        k = len(page_items)

        self.btn_prev.setEnabled(self._current_page > 0)
        self.btn_next.setEnabled(self._current_page < tot_pages - 1)
        self.lbl_page_info.setText(
            f"Página {self._current_page + 1} de {tot_pages} ({total} estudio{'s' if total != 1 else ''})"
        )

        if k == 0:
            self.page_changed.emit(self._current_page, tot_pages)
            return

        if k == 1:
            cols, rows = 1, 1
        elif k == 2:
            cols, rows = 2, 1
        elif k == 3:
            cols, rows = 3, 1
        elif k == 4:
            cols, rows = 2, 2
        else:
            cols, rows = 3, 2

        for idx, study in enumerate(page_items):
            r = idx // cols
            c = idx % cols
            tile = _Mosaic2DTile(study, self.grid_container)
            tile.pixel_hovered.connect(self._on_tile_pixel_hovered)
            tile.pixel_left.connect(self.pixel_left.emit)
            self.grid_layout.addWidget(tile, r, c)
            self._active_tiles.append(tile)

        # Actualizar filas/columnas para expandirse equitativamente
        for r_i in range(rows):
            self.grid_layout.setRowStretch(r_i, 1)
        for c_i in range(cols):
            self.grid_layout.setColumnStretch(c_i, 1)

        self._apply_shift_to_tiles()
        self.render_all()
        self.page_changed.emit(self._current_page, tot_pages)

    def _apply_shift_to_tiles(self):
        start = self._current_page * self.PAGE_SIZE
        for offset, tile in enumerate(self._active_tiles):
            global_idx = start + offset
            rng = self._shift_ranges.get(global_idx)
            tile.set_shift_range(rng)

    def apply_shift_alignment(self, alignment_result):
        self._shift_ranges = dict(alignment_result.slice_ranges or {}) if alignment_result else {}
        self._apply_shift_to_tiles()
        self.render_all()

    def clear_shift_alignment(self):
        self._shift_ranges = {}
        self._apply_shift_to_tiles()
        self.render_all()

    def _on_tile_pixel_hovered(self, x, y, study_dict, tile):
        info = tile.get_pixel_value_at(x, y)
        if info:
            info["patient_name"] = study_dict.get("patient_name", "")
            self.pixel_hovered.emit(info)

    def update_slice_ratio(self, ratio):
        self._current_ratio = max(0.0, min(1.0, float(ratio)))
        for tile in self._active_tiles:
            tile.render_slice(
                self._current_ratio,
                self._ct_window_center,
                self._ct_window_width,
                self._ct_alpha,
                self._pet_alpha,
                self._pet_suv_max,
            )

    def update_ct_window(self, center, width):
        self._ct_window_center = float(center)
        self._ct_window_width = max(1.0, float(width))
        for tile in self._active_tiles:
            tile.render_slice(
                self._current_ratio,
                self._ct_window_center,
                self._ct_window_width,
                self._ct_alpha,
                self._pet_alpha,
                self._pet_suv_max,
            )

    def update_opacity(self, ct_alpha, pet_alpha):
        self._ct_alpha = float(ct_alpha)
        self._pet_alpha = float(pet_alpha)
        for tile in self._active_tiles:
            tile.render_slice(
                self._current_ratio,
                self._ct_window_center,
                self._ct_window_width,
                self._ct_alpha,
                self._pet_alpha,
                self._pet_suv_max,
            )

    def update_pet_suv_max(self, suv_max):
        self._pet_suv_max = float(suv_max) if suv_max is not None and float(suv_max) > 0 else None
        for tile in self._active_tiles:
            tile.render_slice(
                self._current_ratio,
                self._ct_window_center,
                self._ct_window_width,
                self._ct_alpha,
                self._pet_alpha,
                self._pet_suv_max,
            )

    def render_all(self):
        for tile in self._active_tiles:
            tile.render_slice(
                self._current_ratio,
                self._ct_window_center,
                self._ct_window_width,
                self._ct_alpha,
                self._pet_alpha,
                self._pet_suv_max,
            )


class _Mosaic3DTile(QFrame):
    def __init__(self, study_dict, parent=None):
        super().__init__(parent)
        self.study_dict = dict(study_dict or {})
        self._ct_actor = None
        self._pet_volume_actor = None
        self._pet_opacity_func = None
        self._outline_actor = None
        self._max_suv = 1.0
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            _Mosaic3DTile {
                background-color: #12151c;
                border: 1px solid #2d3748;
                border-radius: 6px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        pat = self.study_dict.get("patient_name", "Paciente")
        desc = self.study_dict.get("description", "Estudio")
        mod = self.study_dict.get("modality", "")
        self.lbl_title = QLabel(f"<b>{pat}</b> - {desc} [{mod}] (3D)")
        self.lbl_title.setStyleSheet("color: #63b3ed; font-size: 11px;")
        self.lbl_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_title.setMinimumWidth(0)
        self.lbl_title.setToolTip(f"{pat} | {desc} [{mod}] (3D)")
        header.addWidget(self.lbl_title)
        layout.addLayout(header)

        self.canvas = VTKCanvas(self)
        layout.addWidget(self.canvas, 1)

        self._build_3d_scene()

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(120, 120)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "canvas") and self.canvas:
            self.canvas._scale_and_set_pixmap()

    def _build_3d_scene(self):
        if not VTK_AVAILABLE or not self.canvas.renderer:
            return
        vol_data = self.study_dict.get("volume_data") or {}
        modality = self.study_dict.get("modality", "CT").upper()

        if modality == "FUSION":
            ct_vol = vol_data.get("ct_volume")
            pet_vol = vol_data.get("pet_volume")
            ps = vol_data.get("pixel_spacing") or [1.0, 1.0]
            st = vol_data.get("slice_thickness") or 1.0
            sp = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
            self._max_suv = float(self.study_dict.get("global_max_suv") or vol_data.get("global_max_suv") or vol_data.get("max_suv") or 1.0)
            self._build_fused_scene(ct_vol, pet_vol, sp)
        elif modality in ("PT", "PET"):
            pet_vol = vol_data.get("volume")
            ps = vol_data.get("pixel_spacing") or [1.0, 1.0]
            st = vol_data.get("slice_thickness") or 1.0
            sp = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
            self._max_suv = float(self.study_dict.get("global_max_suv") or vol_data.get("global_max_suv") or vol_data.get("max_suv") or 1.0)
            self._build_pet_scene(pet_vol, sp)
        elif modality in ("MR", "MRI"):
            vol = vol_data.get("volume")
            ps = vol_data.get("pixel_spacing") or [1.0, 1.0]
            st = vol_data.get("slice_thickness") or 1.0
            sp = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
            self._build_mri_scene(vol, sp)
        else:  # CT
            vol = vol_data.get("volume")
            ps = vol_data.get("pixel_spacing") or [1.0, 1.0]
            st = vol_data.get("slice_thickness") or 1.0
            sp = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
            self._build_ct_scene(vol, sp)

    def _build_ct_scene(self, ct_volume, spacing):
        if ct_volume is None or ct_volume.size == 0:
            return
        mask, eff_sp, _ = render_3d.extract_silhouette_mask(
            ct_volume, voxel_spacing=spacing, modality="ct", fast_mode=(ct_volume.shape[0] > 150)
        )
        polydata = render_3d.build_silhouette_polydata(mask, eff_sp or spacing, smoothing_iterations=15)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        self._ct_actor = vtkActor()
        self._ct_actor.SetMapper(mapper)
        prop = self._ct_actor.GetProperty()
        prop.SetColor(0.80, 0.88, 0.96)
        prop.SetOpacity(0.85)
        prop.SetSpecular(0.40)
        prop.SetSpecularPower(30.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()
        self.canvas.renderer.AddActor(self._ct_actor)

        outline = vtkOutlineFilter()
        outline.SetInputData(polydata)
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self._outline_actor = vtkActor()
        self._outline_actor.SetMapper(outline_mapper)
        self._outline_actor.GetProperty().SetColor(0.0, 0.77, 1.0)
        self.canvas.renderer.AddActor(self._outline_actor)

        self.reset_camera()

    def _build_mri_scene(self, mri_volume, spacing):
        if mri_volume is None or mri_volume.size == 0:
            return
        mask, eff_sp, _ = render_3d.extract_silhouette_mask(
            mri_volume, voxel_spacing=spacing, modality="mri", fast_mode=(mri_volume.shape[0] > 150)
        )
        polydata = render_3d.build_silhouette_polydata(mask, eff_sp or spacing, smoothing_iterations=15)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        self._ct_actor = vtkActor()
        self._ct_actor.SetMapper(mapper)
        prop = self._ct_actor.GetProperty()
        prop.SetColor(0.82, 0.88, 0.94)
        prop.SetOpacity(0.85)
        prop.SetSpecular(0.40)
        prop.SetSpecularPower(30.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()
        self.canvas.renderer.AddActor(self._ct_actor)

        outline = vtkOutlineFilter()
        outline.SetInputData(polydata)
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        self._outline_actor = vtkActor()
        self._outline_actor.SetMapper(outline_mapper)
        self._outline_actor.GetProperty().SetColor(0.2, 0.85, 0.4)
        self.canvas.renderer.AddActor(self._outline_actor)

        self.reset_camera()

    def _build_pet_scene(self, pet_volume, spacing):
        if pet_volume is None or pet_volume.size == 0:
            return
        if self._max_suv <= 0:
            self._max_suv = 1.0

        if pet_volume.ndim == 3:
            nz, ny, nx = pet_volume.shape
        elif pet_volume.ndim == 2:
            nz = 1
            ny, nx = pet_volume.shape
            pet_volume = pet_volume.reshape((1, ny, nx))
        else:
            return

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(nx, ny, nz)
        vtk_img.SetSpacing(spacing[0], spacing[1], spacing[2])

        flat_data = np.ascontiguousarray(pet_volume.astype(np.float32)).ravel()
        vtk_arr = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        vol_prop = vtkVolumeProperty()
        vol_prop.ShadeOn()
        vol_prop.SetInterpolationTypeToLinear()

        color_func = vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(0.15 * self._max_suv, 0.8, 0.0, 0.0)
        color_func.AddRGBPoint(0.40 * self._max_suv, 1.0, 0.6, 0.0)
        color_func.AddRGBPoint(0.80 * self._max_suv, 1.0, 1.0, 0.2)
        color_func.AddRGBPoint(self._max_suv, 1.0, 1.0, 0.9)
        vol_prop.SetColor(color_func)

        self._pet_opacity_func = vtkPiecewiseFunction()
        self._pet_opacity_func.AddPoint(0.0, 0.0)
        self._pet_opacity_func.AddPoint(0.05 * self._max_suv, 0.0)
        self._pet_opacity_func.AddPoint(0.20 * self._max_suv, 0.16)
        self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.48)
        self._pet_opacity_func.AddPoint(self._max_suv, 0.72)
        vol_prop.SetScalarOpacity(self._pet_opacity_func)

        mapper = vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_img)

        self._pet_volume_actor = vtkVolume()
        self._pet_volume_actor.SetMapper(mapper)
        self._pet_volume_actor.SetProperty(vol_prop)
        self.canvas.renderer.AddVolume(self._pet_volume_actor)

        self.reset_camera()

    def _build_fused_scene(self, ct_volume, pet_volume, spacing):
        if ct_volume is None or pet_volume is None:
            return
        mask, eff_sp, _ = render_3d.extract_silhouette_mask(
            ct_volume, voxel_spacing=spacing, modality="ct", fast_mode=(ct_volume.shape[0] > 150)
        )
        polydata = render_3d.build_silhouette_polydata(mask, eff_sp or spacing, smoothing_iterations=15)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        self._ct_actor = vtkActor()
        self._ct_actor.SetMapper(mapper)
        prop = self._ct_actor.GetProperty()
        prop.SetColor(0.80, 0.88, 0.96)
        prop.SetOpacity(0.30)
        prop.SetSpecular(0.40)
        prop.SetSpecularPower(30.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()
        self.canvas.renderer.AddActor(self._ct_actor)

        if self._max_suv <= 0:
            self._max_suv = 1.0

        if pet_volume.ndim == 3:
            nz, ny, nx = pet_volume.shape
        elif pet_volume.ndim == 2:
            nz = 1
            ny, nx = pet_volume.shape
            pet_volume = pet_volume.reshape((1, ny, nx))
        else:
            return

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(nx, ny, nz)
        vtk_img.SetSpacing(spacing[0], spacing[1], spacing[2])

        flat_data = np.ascontiguousarray(pet_volume.astype(np.float32)).ravel()
        vtk_arr = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_img.GetPointData().SetScalars(vtk_arr)

        vol_prop = vtkVolumeProperty()
        vol_prop.ShadeOn()
        vol_prop.SetInterpolationTypeToLinear()

        color_func = vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        color_func.AddRGBPoint(0.15 * self._max_suv, 0.8, 0.0, 0.0)
        color_func.AddRGBPoint(0.40 * self._max_suv, 1.0, 0.6, 0.0)
        color_func.AddRGBPoint(0.80 * self._max_suv, 1.0, 1.0, 0.2)
        color_func.AddRGBPoint(self._max_suv, 1.0, 1.0, 0.9)
        vol_prop.SetColor(color_func)

        self._pet_opacity_func = vtkPiecewiseFunction()
        self._pet_opacity_func.AddPoint(0.0, 0.0)
        self._pet_opacity_func.AddPoint(0.05 * self._max_suv, 0.0)
        self._pet_opacity_func.AddPoint(0.20 * self._max_suv, 0.16)
        self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.48)
        self._pet_opacity_func.AddPoint(self._max_suv, 0.72)
        vol_prop.SetScalarOpacity(self._pet_opacity_func)

        pet_mapper = vtkSmartVolumeMapper()
        pet_mapper.SetInputData(vtk_img)

        self._pet_volume_actor = vtkVolume()
        self._pet_volume_actor.SetMapper(pet_mapper)
        self._pet_volume_actor.SetProperty(vol_prop)
        self.canvas.renderer.AddVolume(self._pet_volume_actor)

        self.reset_camera()

    def set_ct_opacity(self, value):
        if self._ct_actor:
            self._ct_actor.GetProperty().SetOpacity(value / 100.0)
            self.canvas.render_scene()

    def set_pet_opacity(self, value):
        if self._pet_opacity_func and self._max_suv > 0:
            scale = value / 100.0
            self._pet_opacity_func.RemoveAllPoints()
            self._pet_opacity_func.AddPoint(0.0, 0.0)
            self._pet_opacity_func.AddPoint(0.05 * self._max_suv, 0.0)
            self._pet_opacity_func.AddPoint(0.20 * self._max_suv, 0.20 * scale)
            self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.60 * scale)
            self._pet_opacity_func.AddPoint(self._max_suv, 0.90 * scale)
            self.canvas.render_scene()

    def set_pet_suv_max(self, max_suv):
        if max_suv is None or float(max_suv) <= 0:
            return
        self._max_suv = float(max_suv)
        if not VTK_AVAILABLE or not hasattr(self, "canvas") or not self.canvas.renderer:
            return
        if self._pet_volume_actor:
            vol_prop = self._pet_volume_actor.GetProperty()
            color_func = vol_prop.GetRGBTransferFunction()
            if color_func:
                color_func.RemoveAllPoints()
                color_func.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                color_func.AddRGBPoint(0.15 * self._max_suv, 0.8, 0.0, 0.0)
                color_func.AddRGBPoint(0.40 * self._max_suv, 1.0, 0.6, 0.0)
                color_func.AddRGBPoint(0.80 * self._max_suv, 1.0, 1.0, 0.2)
                color_func.AddRGBPoint(self._max_suv, 1.0, 1.0, 0.9)
            if self._pet_opacity_func:
                self._pet_opacity_func.RemoveAllPoints()
                self._pet_opacity_func.AddPoint(0.0, 0.0)
                self._pet_opacity_func.AddPoint(0.05 * self._max_suv, 0.0)
                self._pet_opacity_func.AddPoint(0.20 * self._max_suv, 0.16)
                self._pet_opacity_func.AddPoint(0.50 * self._max_suv, 0.48)
                self._pet_opacity_func.AddPoint(self._max_suv, 0.72)
            self.canvas.render_scene()

    def set_view_preset(self, preset_name):
        if not VTK_AVAILABLE or not self.canvas.renderer:
            return
        cam = self.canvas.renderer.GetActiveCamera()
        fp = cam.GetFocalPoint()
        d = cam.GetDistance()
        if preset_name == "anterior":
            cam.SetPosition(fp[0], fp[1] - d, fp[2])
            cam.SetViewUp(0, 0, 1)
        elif preset_name == "lateral_r":
            cam.SetPosition(fp[0] + d, fp[1], fp[2])
            cam.SetViewUp(0, 0, 1)
        elif preset_name == "superior":
            cam.SetPosition(fp[0], fp[1], fp[2] + d)
            cam.SetViewUp(0, 1, 0)
        self.canvas.renderer.ResetCameraClippingRange()
        self.canvas.render_scene()

    def reset_camera(self):
        if VTK_AVAILABLE and self.canvas.renderer:
            self.canvas.renderer.ResetCamera()
            self.set_view_preset("anterior")
            self.canvas.render_scene()


class MultiStudyMosaic3DViewer(QWidget):
    PAGE_SIZE = 6
    page_changed = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._studies = []
        self._current_page = 0
        self._active_tiles = []
        self._ct_opacity = 85
        self._pet_opacity = 80
        self._pet_suv_max = None
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.nav_bar = QWidget()
        nav_h = QHBoxLayout(self.nav_bar)
        nav_h.setContentsMargins(0, 0, 0, 0)
        nav_h.setSpacing(8)

        self.btn_prev = QPushButton("◀ Anterior")
        self.btn_prev.clicked.connect(self._on_prev_page)
        nav_h.addWidget(self.btn_prev)

        self.lbl_page_info = QLabel("Página 1 de 1 (0 estudios 3D)")
        self.lbl_page_info.setStyleSheet("color: #63b3ed; font-weight: bold;")
        nav_h.addWidget(self.lbl_page_info)

        self.btn_next = QPushButton("Siguiente ▶")
        self.btn_next.clicked.connect(self._on_next_page)
        nav_h.addWidget(self.btn_next)

        nav_h.addStretch()
        layout.addWidget(self.nav_bar)

        self.grid_container = QWidget()
        self.grid_container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(6)
        layout.addWidget(self.grid_container, 1)

    def minimumSizeHint(self):
        return QSize(80, 80)

    def sizeHint(self):
        return QSize(300, 200)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for tile in self._active_tiles:
            if hasattr(tile, "canvas") and tile.canvas:
                tile.canvas._scale_and_set_pixmap()

    def total_pages(self):
        return max(1, (len(self._studies) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)

    def set_studies(self, studies):
        self._studies = list(studies or [])
        self._current_page = 0
        self._refresh_page()

    def clear(self):
        self._pet_suv_max = None
        self.set_studies([])

    def set_page(self, page_idx):
        if page_idx < 0 or page_idx >= self.total_pages():
            return
        self._current_page = page_idx
        self._refresh_page()

    def _on_prev_page(self):
        if self._current_page > 0:
            self.set_page(self._current_page - 1)

    def _on_next_page(self):
        if self._current_page < self.total_pages() - 1:
            self.set_page(self._current_page + 1)

    def _refresh_page(self):
        for tile in self._active_tiles:
            self.grid_layout.removeWidget(tile)
            if VTK_AVAILABLE and tile.canvas.renderer:
                tile.canvas.renderer.RemoveAllViewProps()
            tile.setParent(None)
            tile.deleteLater()
        self._active_tiles.clear()

        total = len(self._studies)
        tot_pages = self.total_pages()
        start = self._current_page * self.PAGE_SIZE
        page_items = self._studies[start : start + self.PAGE_SIZE]
        k = len(page_items)

        self.btn_prev.setEnabled(self._current_page > 0)
        self.btn_next.setEnabled(self._current_page < tot_pages - 1)
        self.lbl_page_info.setText(
            f"Página {self._current_page + 1} de {tot_pages} ({total} estudio{'s' if total != 1 else ''} 3D)"
        )

        if k == 0:
            self.page_changed.emit(self._current_page, tot_pages)
            return

        if k == 1:
            cols, rows = 1, 1
        elif k == 2:
            cols, rows = 2, 1
        elif k == 3:
            cols, rows = 3, 1
        elif k == 4:
            cols, rows = 2, 2
        else:
            cols, rows = 3, 2

        for idx, study in enumerate(page_items):
            r = idx // cols
            c = idx % cols
            tile = _Mosaic3DTile(study, self.grid_container)
            if self._pet_suv_max is not None:
                tile.set_pet_suv_max(self._pet_suv_max)
            self.grid_layout.addWidget(tile, r, c)
            self._active_tiles.append(tile)

        for r_i in range(rows):
            self.grid_layout.setRowStretch(r_i, 1)
        for c_i in range(cols):
            self.grid_layout.setColumnStretch(c_i, 1)

        self.render_all()
        self.page_changed.emit(self._current_page, tot_pages)

    def update_pet_suv_max(self, suv_max):
        self._pet_suv_max = float(suv_max) if suv_max is not None and float(suv_max) > 0 else None
        for tile in self._active_tiles:
            if hasattr(tile, "set_pet_suv_max"):
                tile.set_pet_suv_max(suv_max)

    def apply_rotation(self, dx, dy):
        if not VTK_AVAILABLE:
            return
        for tile in self._active_tiles:
            if tile.canvas.renderer:
                camera = tile.canvas.renderer.GetActiveCamera()
                camera.Azimuth(-dx * 0.5)
                camera.Elevation(dy * 0.5)
                camera.OrthogonalizeViewUp()
                tile.canvas.render_scene()

    def apply_pan(self, dx, dy):
        if not VTK_AVAILABLE:
            return
        for tile in self._active_tiles:
            if tile.canvas.renderer:
                camera = tile.canvas.renderer.GetActiveCamera()
                pos = np.array(camera.GetPosition(), dtype=np.float64)
                fp = np.array(camera.GetFocalPoint(), dtype=np.float64)
                vup = np.array(camera.GetViewUp(), dtype=np.float64)
                vdir = fp - pos
                dist = np.linalg.norm(vdir)
                if dist > 0:
                    vdir = vdir / dist
                    vup_norm = vup - np.dot(vup, vdir) * vdir
                    vup_norm = vup_norm / (np.linalg.norm(vup_norm) + 1e-6)
                    vright = np.cross(vdir, vup_norm)
                    delta = (-dx * vright + dy * vup_norm) * (dist * 0.002)
                    camera.SetPosition(*(pos + delta))
                    camera.SetFocalPoint(*(fp + delta))
                    tile.canvas.render_scene()

    def apply_zoom(self, factor):
        if not VTK_AVAILABLE:
            return
        for tile in self._active_tiles:
            if tile.canvas.renderer:
                camera = tile.canvas.renderer.GetActiveCamera()
                camera.Dolly(factor)
                tile.canvas.renderer.ResetCameraClippingRange()
                tile.canvas.render_scene()

    def apply_preset(self, preset_name):
        for tile in self._active_tiles:
            tile.set_view_preset(preset_name)

    def reset_cameras(self):
        for tile in self._active_tiles:
            tile.reset_camera()

    def update_opacity(self, ct_opacity, pet_opacity):
        self._ct_opacity = int(ct_opacity)
        self._pet_opacity = int(pet_opacity)
        for tile in self._active_tiles:
            tile.set_ct_opacity(self._ct_opacity)
            tile.set_pet_opacity(self._pet_opacity)

    def render_all(self):
        for tile in self._active_tiles:
            tile.canvas.render_scene()


class Segmentation2DViewer(QWidget):
    slice_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ct_volume = None
        self._pet_volume = None
        self._mri_volume = None
        self._mask_volume = None
        self._z_positions = []
        self._max_suv = 1.0
        self._num_slices = 0
        self._pet_transparency = 0.5
        self._title = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.info_label = QLabel("Visualización 2D: Sin datos")
        layout.addWidget(self.info_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(280, 280)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: #0b0d11; border: 1px solid #2d3139;")
        layout.addWidget(self.image_label, 1)

        slice_row = QHBoxLayout()
        slice_row.setContentsMargins(0, 0, 0, 0)
        slice_row.setSpacing(8)

        slice_row.addWidget(QLabel("Corte:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(1, 1)
        self.slice_slider.setValue(1)
        self.slice_slider.setEnabled(False)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        slice_row.addWidget(self.slice_slider)

        self.slice_count_label = QLabel("0 / 0")
        slice_row.addWidget(self.slice_count_label)
        layout.addLayout(slice_row)

        self.trans_row = QWidget()
        trans_layout = QHBoxLayout(self.trans_row)
        trans_layout.setContentsMargins(0, 0, 0, 0)
        trans_layout.setSpacing(8)

        self.trans_label = QLabel("Transparencia PET: 50%")
        self.trans_slider = QSlider(Qt.Horizontal)
        self.trans_slider.setRange(0, 100)
        self.trans_slider.setValue(50)
        self.trans_slider.valueChanged.connect(self._on_trans_slider_changed)
        trans_layout.addWidget(self.trans_label)
        trans_layout.addWidget(self.trans_slider)

        self.cb_contour = QCheckBox("Contorno Amarillo")
        self.cb_contour.setChecked(True)
        self.cb_contour.stateChanged.connect(self._render_current_slice)
        trans_layout.addWidget(self.cb_contour)

        layout.addWidget(self.trans_row)
        self.trans_row.setVisible(False)

    def wheelEvent(self, event):
        if self._num_slices > 1:
            delta = event.angleDelta().y()
            if delta > 0:
                self.slice_slider.setValue(min(self._num_slices, self.slice_slider.value() + 1))
            elif delta < 0:
                self.slice_slider.setValue(max(1, self.slice_slider.value() - 1))
            event.accept()
        else:
            super().wheelEvent(event)

    def set_data(self, mask_volume, ct_volume=None, pet_volume=None, mri_volume=None, max_suv=None, z_positions=None, title=""):
        self._title = title
        self._ct_volume = ct_volume
        self._pet_volume = pet_volume
        self._mri_volume = mri_volume
        self._mask_volume = (mask_volume > 0).astype(np.uint8) if mask_volume is not None else None
        self._z_positions = z_positions or []
        self._max_suv = float(max_suv) if max_suv and max_suv > 0 else (
            float(np.nanmax(pet_volume)) if pet_volume is not None and pet_volume.size > 0 else 1.0
        )
        if self._max_suv <= 0:
            self._max_suv = 1.0

        if self._mask_volume is not None:
            self._num_slices = self._mask_volume.shape[0]
        elif self._ct_volume is not None:
            self._num_slices = self._ct_volume.shape[0]
        elif self._mri_volume is not None:
            self._num_slices = self._mri_volume.shape[0]
        else:
            self._num_slices = 0

        has_pet = bool(self._pet_volume is not None and np.any(self._pet_volume > 0))
        self.trans_row.setVisible(has_pet)

        if self._num_slices > 0:
            self.slice_slider.setEnabled(True)
            self.slice_slider.blockSignals(True)
            self.slice_slider.setRange(1, self._num_slices)
            mid = self._find_center_slice_of_mask() or (self._num_slices // 2)
            self.slice_slider.setValue(mid + 1)
            self.slice_slider.blockSignals(False)
            self._render_current_slice()
        else:
            self.clear()

    def _find_center_slice_of_mask(self):
        if self._mask_volume is None:
            return None
        slice_sums = np.sum(self._mask_volume, axis=(1, 2))
        non_zero = np.where(slice_sums > 0)[0]
        if non_zero.size > 0:
            return int(non_zero[len(non_zero) // 2])
        return None

    def _on_slice_slider_changed(self, value):
        self._render_current_slice()
        self.slice_changed.emit(value)

    def _on_trans_slider_changed(self, value):
        self._pet_transparency = value / 100.0
        self.trans_label.setText(f"Transparencia PET: {value}%")
        self._render_current_slice()

    def _render_current_slice(self):
        if self._num_slices <= 0:
            return

        slice_idx = self.slice_slider.value() - 1
        if slice_idx < 0 or slice_idx >= self._num_slices:
            return

        self.slice_count_label.setText(f"{slice_idx + 1} / {self._num_slices}")
        z_str = ""
        if slice_idx < len(self._z_positions):
            z_str = f" (z = {self._z_positions[slice_idx]:.1f} mm)"
        self.info_label.setText(f"Corte 2D {slice_idx + 1}/{self._num_slices}{z_str}: {self._title}")

        rgb = None
        mask_slice = self._mask_volume[slice_idx] if self._mask_volume is not None else None

        if self._ct_volume is not None and self._pet_volume is not None:
            ct_slice = self._ct_volume[slice_idx].astype(np.float32)
            pet_slice = self._pet_volume[slice_idx].astype(np.float32)

            c, w = 40.0, 400.0
            ct_norm = np.clip((ct_slice - (c - w / 2.0)) / w, 0.0, 1.0)
            ct_rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

            pet_norm = np.clip(pet_slice / self._max_suv, 0.0, 1.0)
            cmap = plt.get_cmap("hot")
            pet_rgb = cmap(pet_norm)[..., :3]

            alpha = self._pet_transparency
            rgb = np.clip(ct_rgb * (1.0 - alpha * 0.45) + pet_rgb * alpha, 0.0, 1.0)
        elif self._ct_volume is not None:
            ct_slice = self._ct_volume[slice_idx].astype(np.float32)
            c, w = 40.0, 400.0
            ct_norm = np.clip((ct_slice - (c - w / 2.0)) / w, 0.0, 1.0)
            rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)
        elif self._mri_volume is not None:
            mri_slice = self._mri_volume[slice_idx].astype(np.float32)
            mi, ma = np.nanmin(mri_slice), np.nanmax(mri_slice)
            rng = ma - mi if (ma - mi) > 0 else 1.0
            mri_norm = np.clip((mri_slice - mi) / rng, 0.0, 1.0)
            rgb = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
        elif mask_slice is not None:
            h, w_dims = mask_slice.shape
            rgb = np.zeros((h, w_dims, 3), dtype=np.float32)

        if rgb is None:
            return

        if self.cb_contour.isChecked() and mask_slice is not None and np.any(mask_slice > 0):
            dilated = binary_dilation(mask_slice > 0, iterations=2)
            boundary = dilated ^ (mask_slice > 0)
            rgb[boundary] = [1.0, 1.0, 0.0]

        rgb_uint8 = np.ascontiguousarray((rgb * 255).astype(np.uint8))
        h, w_img, _ = rgb_uint8.shape
        qimg = QImage(rgb_uint8.data, w_img, h, 3 * w_img, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        scaled_pix = pix.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current_slice()

    def clear(self):
        self._ct_volume = None
        self._pet_volume = None
        self._mri_volume = None
        self._mask_volume = None
        self._num_slices = 0
        self.slice_slider.setEnabled(False)
        self.slice_slider.setValue(1)
        self.slice_count_label.setText("0 / 0")
        self.info_label.setText("Visualización 2D: Sin datos")
        self.image_label.clear()
        self.trans_row.setVisible(False)



class SegmentationGraphsViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._mask_volume = None
        self._ct_volume = None
        self._pet_volume = None
        self._mri_volume = None
        self._voxel_spacing = None
        self._max_suv = 1.0
        self._z_positions = []
        self._title = ""
        self._num_slices = 0

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Barra superior: Lista desplegable de selección y NavigationToolbar
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(2, 2, 2, 2)
        top_bar.setSpacing(10)

        lbl = QLabel("Seleccionar gráfica:")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        top_bar.addWidget(lbl)

        self.combo_graphs = QComboBox()
        self.combo_graphs.addItem("SUV promedio en función del corte")
        self.combo_graphs.addItem("HU promedio en función del corte")
        self.combo_graphs.addItem("Histograma de SUV")
        self.combo_graphs.addItem("Histograma de HU")
        self.combo_graphs.setMinimumWidth(260)
        self.combo_graphs.currentIndexChanged.connect(self._on_graph_selected)
        top_bar.addWidget(self.combo_graphs)

        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas, parent=None)
        top_bar.addWidget(self.toolbar)
        top_bar.addStretch()

        main_layout.addLayout(top_bar)
        main_layout.addWidget(self.canvas, 1)

        self.combo_graphs.setCurrentIndex(0)

    def set_data(self, mask_volume, ct_volume=None, pet_volume=None, mri_volume=None,
                 voxel_spacing=None, max_suv=None, z_positions=None, title="Segmentación"):
        self._mask_volume = mask_volume
        self._ct_volume = ct_volume
        self._pet_volume = pet_volume
        self._mri_volume = mri_volume
        self._voxel_spacing = voxel_spacing
        self._max_suv = float(max_suv) if max_suv is not None else 1.0
        self._z_positions = z_positions or []
        self._title = title or "Segmentación"
        self._num_slices = mask_volume.shape[0] if mask_volume is not None else 0

        # Primero se muestra la gráfica del SUV promedio
        self.combo_graphs.blockSignals(True)
        self.combo_graphs.setCurrentIndex(0)
        self.combo_graphs.blockSignals(False)
        self._render_graph(0)

    def _on_graph_selected(self, index):
        self._render_graph(index)

    def refresh(self):
        idx = self.combo_graphs.currentIndex()
        if idx < 0:
            idx = 0
        self._render_graph(idx)

    def clear(self):
        self._mask_volume = None
        self._ct_volume = None
        self._pet_volume = None
        self._mri_volume = None
        self._title = ""
        self._num_slices = 0
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Seleccione o genere una segmentación para visualizar sus gráficas.",
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax.transAxes, fontsize=11)
        self.canvas.draw()

    def _render_graph(self, index):
        self.ax.clear()

        if self._mask_volume is None or np.count_nonzero(self._mask_volume) == 0:
            self.ax.text(0.5, 0.5, "Seleccione o genere una segmentación para visualizar sus gráficas.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=11)
            self.canvas.draw()
            return

        mask = self._mask_volume > 0

        # 0: SUV promedio en función del corte
        if index == 0:
            self._plot_suv_per_slice(mask)
        # 1: HU promedio en función del corte
        elif index == 1:
            self._plot_hu_per_slice(mask)
        # 2: Histograma de SUV
        elif index == 2:
            self._plot_suv_histogram(mask)
        # 3: Histograma de HU
        elif index == 3:
            self._plot_hu_histogram(mask)

        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    def _plot_suv_per_slice(self, mask):
        if self._pet_volume is None or not np.any(self._pet_volume > 0):
            self.ax.text(0.5, 0.5, "Datos de PET / SUV no disponibles para este estudio.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=11)
            return

        slices = []
        means = []
        for z in range(self._num_slices):
            sl_mask = mask[z]
            if np.any(sl_mask):
                slices.append(z + 1)
                means.append(float(np.mean(self._pet_volume[z][sl_mask])))

        if not slices:
            self.ax.text(0.5, 0.5, "No se detectaron cortes con vóxeles segmentados.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=11)
            return

        self.ax.plot(slices, means, marker='o', color='red', label='SUV promedio')
        self.ax.set_title(f"{self._title} - SUV Promedio en función del Corte")
        self.ax.set_xlabel("Corte")
        self.ax.set_ylabel("SUV Promedio")
        self.ax.grid(True)
        self.ax.legend()

    def _plot_hu_per_slice(self, mask):
        if self._ct_volume is None:
            self.ax.text(0.5, 0.5, "Datos de CT / HU no disponibles para este estudio.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=11)
            return

        slices = []
        means = []
        for z in range(self._num_slices):
            sl_mask = mask[z]
            if np.any(sl_mask):
                slices.append(z + 1)
                means.append(float(np.mean(self._ct_volume[z][sl_mask])))

        if not slices:
            self.ax.text(0.5, 0.5, "No se detectaron cortes con vóxeles segmentados.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=11)
            return

        self.ax.plot(slices, means, marker='o', color='blue', label='HU promedio')
        self.ax.set_title(f"{self._title} - HU Promedio en función del Corte")
        self.ax.set_xlabel("Corte")
        self.ax.set_ylabel("HU Promedio")
        self.ax.grid(True)
        self.ax.legend()

    def _plot_suv_histogram(self, mask):
        if self._pet_volume is None or not np.any(self._pet_volume > 0):
            self.ax.text(0.5, 0.5, "Datos de PET / SUV no disponibles para este estudio.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=11)
            return

        voxels = self._pet_volume[mask]
        if voxels.size == 0:
            return

        self.ax.hist(voxels, bins=30, color='red', edgecolor='black', alpha=0.7)
        self.ax.set_title(f"{self._title} - Histograma de SUV")
        self.ax.set_xlabel("SUV")
        self.ax.set_ylabel("Frecuencia")
        self.ax.grid(True)

    def _plot_hu_histogram(self, mask):
        if self._ct_volume is None:
            self.ax.text(0.5, 0.5, "Datos de CT / HU no disponibles para este estudio.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, color='#a0aec0', fontsize=11)
            return

        voxels = self._ct_volume[mask]
        if voxels.size == 0:
            return

        self.ax.hist(voxels, bins=30, color='blue', edgecolor='black', alpha=0.7)
        self.ax.set_title(f"{self._title} - Histograma de HU")
        self.ax.set_xlabel("HU")
        self.ax.set_ylabel("Frecuencia")
        self.ax.grid(True)


class SegmentationDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.subtab_widget = QTabWidget(self)

        self.viewer_2d = Segmentation2DViewer(self)
        self.viewer_3d = Viewer3DWidget(self)
        self.graphs_viewer = SegmentationGraphsViewer(self)

        self.subtab_widget.addTab(self.viewer_2d, "Segmentación 2D")
        self.subtab_widget.addTab(self.viewer_3d, "Segmentación 3D")
        self.subtab_widget.addTab(self.graphs_viewer, "Gráficas")
        self.subtab_widget.currentChanged.connect(self._on_subtab_changed)

        main_layout.addWidget(self.subtab_widget)

    def _on_subtab_changed(self, index):
        if index == 1:
            if hasattr(self.viewer_3d, "canvas") and self.viewer_3d.canvas:
                self.viewer_3d.canvas.render_scene()
        elif index == 0:
            if hasattr(self.viewer_2d, "_render_current_slice"):
                self.viewer_2d._render_current_slice()
        elif index == 2:
            if hasattr(self.graphs_viewer, "refresh"):
                self.graphs_viewer.refresh()

    def show_segmentation(self, seg_volume, ct_volume=None, pet_volume=None, mri_volume=None,
                          voxel_spacing=None, max_suv=None, z_positions=None, title="Segmentación"):
        bin_mask = (seg_volume > 0).astype(np.uint8) if seg_volume is not None else None

        ref_vol = ct_volume if ct_volume is not None else (pet_volume if pet_volume is not None else mri_volume)
        if bin_mask is not None and bin_mask.ndim == 3:
            if ref_vol is not None and ref_vol.ndim == 3:
                if bin_mask.shape == (ref_vol.shape[2], ref_vol.shape[1], ref_vol.shape[0]):
                    bin_mask = np.transpose(bin_mask, (2, 1, 0))
            elif bin_mask.shape[0] > bin_mask.shape[2]:
                bin_mask = np.transpose(bin_mask, (2, 1, 0))

        self.viewer_2d.set_data(
            mask_volume=bin_mask,
            ct_volume=ct_volume,
            pet_volume=pet_volume,
            mri_volume=mri_volume,
            max_suv=max_suv,
            z_positions=z_positions,
            title=title
        )

        self.viewer_3d.show_segmentation_volume(
            seg_volume=bin_mask,
            voxel_spacing=voxel_spacing,
            pet_volume=pet_volume,
            max_suv=max_suv,
            title=title
        )

        self.graphs_viewer.set_data(
            mask_volume=bin_mask,
            ct_volume=ct_volume,
            pet_volume=pet_volume,
            mri_volume=mri_volume,
            voxel_spacing=voxel_spacing,
            max_suv=max_suv,
            z_positions=z_positions,
            title=title
        )

    def clear(self):
        self.viewer_2d.clear()
        self.viewer_3d.clear()
        self.graphs_viewer.clear()


class _DirectoryDropArea(QFrame):
    directory_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Sunken)
        self.setMinimumHeight(75)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setAlignment(Qt.AlignCenter)
        self.label = QLabel("Arrastre aquí una carpeta\npara indexar automáticamente")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isdir(path):
                    event.acceptProposedAction()
                    self.setLineWidth(2)
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isdir(path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setLineWidth(1)
        event.accept()

    def dropEvent(self, event):
        self.setLineWidth(1)
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.isdir(path):
                    event.acceptProposedAction()
                    self.directory_dropped.emit(path)
                    return
        event.ignore()


class StudySelectionPanel(QWidget):
    node_selected = Signal(dict)
    multi_study_selection_changed = Signal(list)
    directory_selected = Signal(str)
    index_requested = Signal(str)
    open_directory_dialog_requested = Signal()
    change_directory_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dicom_root = None
        self._active_multi_study_modality = None
        self._updating_checks = False
        self.recent_store = RecentFoldersStore(RECENT_FOLDERS_CONFIG_PATH)
        self._setup_ui()

    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)

        # Stack superior: alternar entre selector de directorios recientes (0) y árbol de estudios (1)
        self.top_stack = QStackedWidget()

        # Página 0: Directorios recientes, botones y Drag & Drop
        self.recent_page = QWidget()
        recent_layout = QVBoxLayout(self.recent_page)
        recent_layout.setContentsMargins(0, 0, 0, 0)
        recent_layout.setSpacing(6)

        self.recent_label = QLabel("Directorios recientes:")
        recent_layout.addWidget(self.recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.recent_list.itemSelectionChanged.connect(self._on_recent_selection_changed)
        self.recent_list.itemDoubleClicked.connect(self._on_recent_item_double_clicked)
        recent_layout.addWidget(self.recent_list, 1)

        # Botones: Seleccionar directorio e Indexar
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)
        self.btn_select_dir = QPushButton("Seleccionar directorio...")
        self.btn_select_dir.clicked.connect(self._on_select_dir_clicked)
        self.btn_index = QPushButton("Indexar")
        self.btn_index.setEnabled(False)
        self.btn_index.clicked.connect(self._on_index_clicked)
        btn_layout.addWidget(self.btn_select_dir, 1)
        btn_layout.addWidget(self.btn_index, 0)
        recent_layout.addLayout(btn_layout)

        # Cuadro de arrastre (Drag & Drop) para indexación automática
        self.drop_area = _DirectoryDropArea()
        self.drop_area.directory_dropped.connect(self._on_directory_dropped)
        recent_layout.addWidget(self.drop_area)

        # Página 1: Árbol jerárquico de estudios
        self.tree_page = QWidget()
        tree_layout = QVBoxLayout(self.tree_page)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(6)

        tree_top_layout = QHBoxLayout()
        tree_top_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_change_dir = QPushButton("◀ Cambiar directorio")
        self.btn_change_dir.clicked.connect(self._on_change_dir_clicked)
        tree_top_layout.addWidget(self.btn_change_dir)
        tree_top_layout.addStretch(1)
        tree_layout.addLayout(tree_top_layout)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        tree_layout.addWidget(self.tree, 1)

        self.top_stack.addWidget(self.recent_page)
        self.top_stack.addWidget(self.tree_page)
        self.top_stack.setCurrentIndex(0)
        self.layout.addWidget(self.top_stack, 3)

        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Campo", "Valor"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.info_table.setWordWrap(True)
        self.layout.addWidget(self.info_table, 2)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        self.log_output.setStyleSheet("background-color: #000000; color: #00ff00;")
        self.layout.addWidget(self.log_output, 1)

        # Cargar lista de directorios recientes al inicializar
        self.load_recent_folders()

    def load_recent_folders(self):
        self.recent_list.clear()
        folders = self.recent_store.load()
        for folder in folders:
            item = QListWidgetItem(folder)
            item.setToolTip(folder)
            if not os.path.exists(folder):
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.recent_list.addItem(item)
        self.btn_index.setEnabled(False)

    def show_recent_view(self):
        self.top_stack.setCurrentIndex(0)

    def show_tree_view(self):
        self.top_stack.setCurrentIndex(1)

    def set_selected_directory(self, folder):
        if not folder:
            return
        abs_folder = os.path.abspath(folder)
        for i in range(self.recent_list.count()):
            item = self.recent_list.item(i)
            if item.text() == abs_folder or item.text() == folder:
                self.recent_list.setCurrentItem(item)
                self.btn_index.setEnabled(True)
                self.directory_selected.emit(abs_folder)
                return
        item = QListWidgetItem(abs_folder)
        item.setToolTip(abs_folder)
        self.recent_list.insertItem(0, item)
        self.recent_list.setCurrentItem(item)
        self.btn_index.setEnabled(True)
        self.directory_selected.emit(abs_folder)

    def _on_recent_selection_changed(self):
        items = self.recent_list.selectedItems()
        if items and (items[0].flags() & Qt.ItemIsEnabled):
            self.btn_index.setEnabled(True)
            self.directory_selected.emit(items[0].text())
        else:
            self.btn_index.setEnabled(False)

    def _on_recent_item_double_clicked(self, item):
        if item and (item.flags() & Qt.ItemIsEnabled):
            self.index_requested.emit(item.text())

    def _on_select_dir_clicked(self):
        self.open_directory_dialog_requested.emit()

    def _on_index_clicked(self):
        items = self.recent_list.selectedItems()
        if items and (items[0].flags() & Qt.ItemIsEnabled):
            self.index_requested.emit(items[0].text())

    def _on_directory_dropped(self, path):
        self.set_selected_directory(path)
        self.index_requested.emit(path)

    def _on_change_dir_clicked(self):
        self.show_recent_view()
        self.change_directory_requested.emit()

    def append_log(self, text):
        self.log_output.appendPlainText(text)
        self.log_output.ensureCursorVisible()

    def set_info(self, key_value_pairs):
        self.info_table.setRowCount(len(key_value_pairs))
        for row, (field_name, value) in enumerate(key_value_pairs):
            item_f = QTableWidgetItem(str(field_name))
            item_v = QTableWidgetItem(str(value))
            item_f.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item_v.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.info_table.setItem(row, 0, item_f)
            self.info_table.setItem(row, 1, item_v)

    def clear_info(self):
        self.info_table.setRowCount(0)

    def mark_study_built(self, study_instance_uid, pair_key=None):
        icon_found = get_fusion_icon(True)

        def _update_items(parent_item):
            count = parent_item.childCount() if parent_item else self.tree.topLevelItemCount()
            for i in range(count):
                item = parent_item.child(i) if parent_item else self.tree.topLevelItem(i)
                data = item.data(0, Qt.UserRole) or {}
                node_type = data.get("type")

                if node_type in (NODE_TYPE_STUDY, NODE_TYPE_FUSION_STUDY):
                    if data.get("study_instance_uid") == study_instance_uid:
                        item.setIcon(0, icon_found)

                elif node_type == NODE_TYPE_FUSION_PAIR:
                    if pair_key:
                        p = data.get("pair") or {}
                        if join_pet_ct._pair_key(p) == pair_key:
                            item.setIcon(0, icon_found)

                _update_items(item)
        _update_items(None)

    def mark_ct_series_built(self, series_instance_uid):
        icon_found = get_ct_icon(True)

        def _update_items(parent_item):
            count = parent_item.childCount() if parent_item else self.tree.topLevelItemCount()
            for i in range(count):
                item = parent_item.child(i) if parent_item else self.tree.topLevelItem(i)
                data = item.data(0, Qt.UserRole) or {}
                if data.get("type") == NODE_TYPE_SERIES and data.get("series_instance_uid") == series_instance_uid:
                    if not join_pet_ct.is_non_volume_series(data):
                        item.setIcon(0, icon_found)
                _update_items(item)
        _update_items(None)

    def mark_pet_series_built(self, series_instance_uid):
        icon_found = get_pet_icon(True)

        def _update_items(parent_item):
            count = parent_item.childCount() if parent_item else self.tree.topLevelItemCount()
            for i in range(count):
                item = parent_item.child(i) if parent_item else self.tree.topLevelItem(i)
                data = item.data(0, Qt.UserRole) or {}
                if data.get("type") == NODE_TYPE_SERIES and data.get("series_instance_uid") == series_instance_uid:
                    if not join_pet_ct.is_non_volume_series(data):
                        item.setIcon(0, icon_found)
                _update_items(item)
        _update_items(None)

    def mark_mri_built(self, study_instance_uid, series_instance_uid=None):
        icon_found = get_mri_icon(True)

        def _update_items(parent_item):
            count = parent_item.childCount() if parent_item else self.tree.topLevelItemCount()
            for i in range(count):
                item = parent_item.child(i) if parent_item else self.tree.topLevelItem(i)
                data = item.data(0, Qt.UserRole) or {}
                node_type = data.get("type")

                if node_type == NODE_TYPE_STUDY:
                    if study_instance_uid and data.get("study_instance_uid") == study_instance_uid and (
                        data.get("modality") == "MRI" or data.get("prefix") == "mri"
                    ):
                        item.setIcon(0, icon_found)

                elif node_type == NODE_TYPE_SERIES:
                    if series_instance_uid:
                        if data.get("series_instance_uid") == series_instance_uid and (
                            data.get("prefix") == "mri" or str(data.get("modality")).upper() in ("MR", "MRI")
                        ):
                            item.setIcon(0, icon_found)
                    elif study_instance_uid:
                        if data.get("study_instance_uid") == study_instance_uid and (
                            data.get("prefix") == "mri" or str(data.get("modality")).upper() in ("MR", "MRI")
                        ):
                            item.setIcon(0, icon_found)

                _update_items(item)
        _update_items(None)

    def populate_tree(self, modalities, fusion_patients=None, multi_study_patients=None,
                      built_study_uids=None, built_pair_keys=None,
                      built_ct_uids=None, built_pet_uids=None,
                      built_mri_series_uids=None, built_mri_study_uids=None):
        self.show_tree_view()
        self.tree.clear()
        self.clear_info()
        self._active_multi_study_modality = None
        self.multi_study_selection_changed.emit([])
        built_study_uids = built_study_uids or set()
        built_pair_keys = built_pair_keys or set()
        built_ct_uids = built_ct_uids or set()
        built_pet_uids = built_pet_uids or set()
        built_mri_series_uids = built_mri_series_uids or set()
        built_mri_study_uids = built_mri_study_uids or set()

        for modality, patients in modalities.items():
            prefix = MODALITY_PREFIXES[modality]
            modality_item = QTreeWidgetItem([MODALITY_LABELS.get(modality, modality)])
            modality_item.setData(0, Qt.UserRole, {
                "type": NODE_TYPE_MODALITY,
                "label": MODALITY_LABELS.get(modality, modality),
                "modality": modality,
            })
            for patient in patients:
                patient_label = "%s (%s)" % (patient.patient_name, patient.patient_id)
                patient_item = QTreeWidgetItem([patient_label])
                patient_item.setData(0, Qt.UserRole, {
                    "type": NODE_TYPE_PATIENT,
                    "label": patient_label,
                    "patient_name": patient.patient_name,
                    "patient_id": patient.patient_id,
                    "num_studies": len(patient.studies),
                })
                for study in patient.studies:
                    study_label = "%s (%s)" % (study.study_description, study.study_date)
                    study_item = QTreeWidgetItem([study_label])
                    if modality == "PET_CT":
                        is_built = study.study_instance_uid in built_study_uids
                        study_item.setIcon(0, get_fusion_icon(is_built))
                    elif modality == "MRI":
                        is_built = study.study_instance_uid in built_mri_study_uids
                        study_item.setIcon(0, get_mri_icon(is_built))
                    first_ser_dir = study.series_list[0].series_directory if study.series_list else study.study_directory
                    first_ser_uid = study.series_list[0].series_instance_uid if study.series_list else ""
                    study_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_STUDY,
                        "label": study_label,
                        "study_instance_uid": study.study_instance_uid,
                        "series_instance_uid": "",
                        "study_description": study.study_description,
                        "study_date": study.study_date,
                        "study_directory": study.study_directory or first_ser_dir,
                        "series_directory": first_ser_dir,
                        "patient_name": patient.patient_name,
                        "patient_id": patient.patient_id,
                        "num_series": len(study.series_list),
                        "modality": modality,
                        "prefix": prefix,
                    })
                    for series in study.series_list:
                        series_label = "%s - %s (%s, %d imagenes)" % (
                            series.series_number if series.series_number is not None else "-",
                            series.series_description,
                            series.modality,
                            series.num_images,
                        )
                        series_item = QTreeWidgetItem([series_label])
                        ser_mod = str(series.modality).upper()
                        ser_uid = str(series.series_instance_uid)
                        ser_desc = str(series.series_description or "")
                        if join_pet_ct.is_non_volume_series(ser_desc, modality=ser_mod, num_images=series.num_images):
                            series_item.setIcon(0, get_dicom_file_icon())
                        elif ser_mod == "CT":
                            series_item.setIcon(0, get_ct_icon(ser_uid in built_ct_uids))
                        elif ser_mod in ("PT", "PET"):
                            series_item.setIcon(0, get_pet_icon(ser_uid in built_pet_uids))
                        elif ser_mod in ("MR", "MRI") or prefix == "mri":
                            is_built = ser_uid in built_mri_series_uids
                            series_item.setIcon(0, get_mri_icon(is_built))
                        else:
                            series_item.setIcon(0, get_dicom_file_icon())

                        series_item.setData(0, Qt.UserRole, {
                            "type": NODE_TYPE_SERIES,
                            "label": series_label,
                            "prefix": prefix,
                            "series_instance_uid": series.series_instance_uid,
                            "series_number": series.series_number,
                            "series_description": series.series_description,
                            "series_directory": series.series_directory,
                            "study_directory": study.study_directory,
                            "modality": series.modality,
                            "num_images": series.num_images,
                            "patient_name": patient.patient_name,
                            "patient_id": patient.patient_id,
                            "study_description": study.study_description,
                            "study_date": study.study_date,
                            "study_instance_uid": study.study_instance_uid,
                        })
                        study_item.addChild(series_item)
                    patient_item.addChild(study_item)
                modality_item.addChild(patient_item)
            self.tree.addTopLevelItem(modality_item)

        if fusion_patients:
            self.tree.addTopLevelItem(
                self._build_fusion_category_item(fusion_patients, built_study_uids, built_pair_keys)
            )

        if multi_study_patients:
            self.tree.addTopLevelItem(
                self._build_multi_study_category_item(
                    multi_study_patients,
                    built_study_uids=built_study_uids,
                    built_pair_keys=built_pair_keys,
                    built_ct_uids=built_ct_uids,
                    built_pet_uids=built_pet_uids,
                    built_mri_series_uids=built_mri_series_uids,
                    built_mri_study_uids=built_mri_study_uids,
                )
            )

        self.tree.expandToDepth(0)

    def _build_fusion_category_item(self, fusion_patients, built_study_uids=None, built_pair_keys=None):
        built_study_uids = built_study_uids or set()
        built_pair_keys = built_pair_keys or set()
        category_item = QTreeWidgetItem([FUSION_CATEGORY_LABEL])
        category_item.setData(0, Qt.UserRole, {
            "type": NODE_TYPE_FUSION_CATEGORY,
            "label": FUSION_CATEGORY_LABEL,
        })
        for patient in fusion_patients:
            patient_label = "%s (%s)" % (patient.patient_name, patient.patient_id)
            patient_item = QTreeWidgetItem([patient_label])
            patient_item.setData(0, Qt.UserRole, {
                "type": NODE_TYPE_FUSION_PATIENT,
                "label": patient_label,
                "patient_name": patient.patient_name,
                "patient_id": patient.patient_id,
                "num_studies": len(patient.studies),
            })
            for study in patient.studies:
                study_label = "%s (%s)" % (study.study_description, study.study_date)
                study_item = QTreeWidgetItem([study_label])
                is_built = study.study_instance_uid in built_study_uids
                study_item.setIcon(0, get_fusion_icon(is_built))
                pairs_dicts = [p.__dict__ if hasattr(p, '__dict__') else p for p in study.pairs]
                study_item.setData(0, Qt.UserRole, {
                    "type": NODE_TYPE_FUSION_STUDY,
                    "label": study_label,
                    "study_instance_uid": study.study_instance_uid,
                    "study_description": study.study_description,
                    "study_date": study.study_date,
                    "study_directory": study.study_directory,
                    "patient_name": patient.patient_name,
                    "patient_id": patient.patient_id,
                    "pairs": pairs_dicts,
                })
                for pair in study.pairs:
                    pair_dict = pair.__dict__ if hasattr(pair, '__dict__') else pair
                    key = join_pet_ct._pair_key(pair_dict)
                    pair_label = "CT: %s + PET: %s (%d cortes)" % (
                        pair.ct_series_description, pair.pet_series_description,
                        pair.num_slices,
                    )
                    pair_item = QTreeWidgetItem([pair_label])
                    pair_is_built = key in built_pair_keys
                    pair_item.setIcon(0, get_fusion_icon(pair_is_built))
                    pair_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_FUSION_PAIR,
                        "label": pair_label,
                        "study_instance_uid": study.study_instance_uid,
                        "study_description": study.study_description,
                        "study_date": study.study_date,
                        "study_directory": study.study_directory,
                        "patient_name": patient.patient_name,
                        "patient_id": patient.patient_id,
                        "pair": pair_dict,
                    })
                    study_item.addChild(pair_item)
                patient_item.addChild(study_item)
            category_item.addChild(patient_item)
        return category_item

    def _build_multi_study_category_item(
        self, multi_study_patients,
        built_study_uids=None, built_pair_keys=None,
        built_ct_uids=None, built_pet_uids=None,
        built_mri_series_uids=None, built_mri_study_uids=None
    ):
        built_study_uids = built_study_uids or set()
        built_pair_keys = built_pair_keys or set()
        built_ct_uids = built_ct_uids or set()
        built_pet_uids = built_pet_uids or set()

        category_item = QTreeWidgetItem([MULTI_STUDY_CATEGORY_LABEL])
        category_item.setIcon(0, get_fusion_icon(False))
        category_item.setFlags(category_item.flags() & ~Qt.ItemIsUserCheckable)
        category_item.setData(0, Qt.UserRole, {
            "type": NODE_TYPE_MULTI_STUDY_CATEGORY,
            "label": MULTI_STUDY_CATEGORY_LABEL,
        })

        for patient in multi_study_patients:
            studies_list = getattr(patient, "studies", [])
            if not studies_list:
                continue

            patient_label = f"{patient.patient_name} ({patient.patient_id})"
            patient_item = QTreeWidgetItem([patient_label])
            patient_item.setIcon(0, get_dicom_file_icon())
            patient_item.setFlags(patient_item.flags() & ~Qt.ItemIsUserCheckable)
            patient_item.setData(0, Qt.UserRole, {
                "type": NODE_TYPE_MULTI_STUDY_PATIENT,
                "label": patient_label,
                "patient_name": patient.patient_name,
                "patient_id": patient.patient_id,
                "modality": getattr(patient, "modality", "PET/CT"),
                "num_studies": len(studies_list),
                "is_multi_study": True,
            })

            for study in studies_list:
                study_label = (
                    "Estudio: %s (%s)" % (study.study_description, study.study_date)
                    if study.study_date else "Estudio: %s" % study.study_description
                )
                study_item = QTreeWidgetItem([study_label])
                study_item.setFlags(study_item.flags() & ~Qt.ItemIsUserCheckable)
                is_built = study.study_instance_uid in built_study_uids
                study_item.setIcon(0, get_fusion_icon(is_built))

                pairs_dicts = [p.__dict__ if hasattr(p, "__dict__") else p for p in getattr(study, "pairs", [])]
                study_item.setData(0, Qt.UserRole, {
                    "type": NODE_TYPE_MULTI_STUDY_STUDY,
                    "label": study_label,
                    "study_instance_uid": study.study_instance_uid,
                    "study_description": study.study_description,
                    "study_date": study.study_date,
                    "study_directory": study.study_directory,
                    "patient_name": patient.patient_name,
                    "patient_id": patient.patient_id,
                    "pairs": pairs_dicts,
                    "is_multi_study": True,
                })

                # Clasificar series de CT y PET excluyendo no-volumétricas
                ct_series_list = []
                pet_series_list = []
                for ser in getattr(study, "series_list", []):
                    ser_mod = str(ser.modality).upper()
                    ser_desc = str(ser.series_description or "")
                    if join_pet_ct.is_non_volume_series(ser_desc, modality=ser_mod, num_images=ser.num_images):
                        continue
                    if ser_mod == "CT":
                        ct_series_list.append(ser)
                    elif ser_mod in ("PT", "PET"):
                        pet_series_list.append(ser)

                pairs_list = getattr(study, "pairs", [])

                # Contenedor: Series de CT
                if ct_series_list:
                    ct_group_item = QTreeWidgetItem(["Series de CT"])
                    ct_group_item.setIcon(0, get_ct_icon(False))
                    ct_group_item.setFlags(ct_group_item.flags() & ~Qt.ItemIsUserCheckable)
                    ct_group_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_MULTI_STUDY_GROUP,
                        "group_type": "CT",
                        "group_modality": "CT",
                        "label": "Series de CT",
                        "study_instance_uid": study.study_instance_uid,
                        "study_description": study.study_description,
                        "patient_name": patient.patient_name,
                        "patient_id": patient.patient_id,
                        "is_multi_study": True,
                    })
                    for ser in ct_series_list:
                        ser_label = "CT: %s (%d imágenes)" % (
                            ser.series_description, ser.num_images
                        )
                        ser_is_built = ser.series_instance_uid in built_ct_uids
                        ser_item = QTreeWidgetItem([ser_label])
                        ser_item.setIcon(0, get_ct_icon(ser_is_built))
                        ser_item.setFlags((ser_item.flags() | Qt.ItemIsUserCheckable) | Qt.ItemIsEnabled)
                        ser_item.setCheckState(0, Qt.Unchecked)
                        ser_item.setData(0, Qt.UserRole, {
                            "type": NODE_TYPE_SERIES,
                            "label": ser_label,
                            "series_instance_uid": ser.series_instance_uid,
                            "series_description": ser.series_description,
                            "series_number": ser.series_number,
                            "modality": "CT",
                            "num_images": ser.num_images,
                            "series_directory": ser.series_directory,
                            "study_instance_uid": study.study_instance_uid,
                            "study_description": study.study_description,
                            "study_date": study.study_date,
                            "study_directory": study.study_directory,
                            "patient_name": patient.patient_name,
                            "patient_id": patient.patient_id,
                            "prefix": "pet_ct",
                            "is_multi_study": True,
                        })
                        ct_group_item.addChild(ser_item)
                    study_item.addChild(ct_group_item)

                # Contenedor: Series de PET
                if pet_series_list:
                    pet_group_item = QTreeWidgetItem(["Series de PET"])
                    pet_group_item.setIcon(0, get_pet_icon(False))
                    pet_group_item.setFlags(pet_group_item.flags() & ~Qt.ItemIsUserCheckable)
                    pet_group_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_MULTI_STUDY_GROUP,
                        "group_type": "PET",
                        "group_modality": "PET",
                        "label": "Series de PET",
                        "study_instance_uid": study.study_instance_uid,
                        "study_description": study.study_description,
                        "patient_name": patient.patient_name,
                        "patient_id": patient.patient_id,
                        "is_multi_study": True,
                    })
                    for ser in pet_series_list:
                        ser_label = "PET: %s (%d imágenes)" % (
                            ser.series_description, ser.num_images
                        )
                        ser_is_built = ser.series_instance_uid in built_pet_uids
                        ser_item = QTreeWidgetItem([ser_label])
                        ser_item.setIcon(0, get_pet_icon(ser_is_built))
                        ser_item.setFlags((ser_item.flags() | Qt.ItemIsUserCheckable) | Qt.ItemIsEnabled)
                        ser_item.setCheckState(0, Qt.Unchecked)
                        ser_item.setData(0, Qt.UserRole, {
                            "type": NODE_TYPE_SERIES,
                            "label": ser_label,
                            "series_instance_uid": ser.series_instance_uid,
                            "series_description": ser.series_description,
                            "series_number": ser.series_number,
                            "modality": "PT",
                            "num_images": ser.num_images,
                            "series_directory": ser.series_directory,
                            "study_instance_uid": study.study_instance_uid,
                            "study_description": study.study_description,
                            "study_date": study.study_date,
                            "study_directory": study.study_directory,
                            "patient_name": patient.patient_name,
                            "patient_id": patient.patient_id,
                            "prefix": "pet_ct",
                            "is_multi_study": True,
                        })
                        pet_group_item.addChild(ser_item)
                    study_item.addChild(pet_group_item)

                # Contenedor: Pares fusionables
                if pairs_list:
                    fusion_group_item = QTreeWidgetItem(["Pares fusionables"])
                    fusion_group_item.setIcon(0, get_fusion_icon(False))
                    fusion_group_item.setFlags(fusion_group_item.flags() & ~Qt.ItemIsUserCheckable)
                    fusion_group_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_MULTI_STUDY_GROUP,
                        "group_type": "FUSION",
                        "group_modality": "FUSION",
                        "label": "Pares fusionables",
                        "study_instance_uid": study.study_instance_uid,
                        "study_description": study.study_description,
                        "patient_name": patient.patient_name,
                        "patient_id": patient.patient_id,
                        "is_multi_study": True,
                    })
                    for pair in pairs_list:
                        pair_dict = pair.__dict__ if hasattr(pair, "__dict__") else pair
                        key = join_pet_ct._pair_key(pair_dict)
                        pair_is_built = key in built_pair_keys
                        pair_label = "Par: CT: %s + PET: %s (%d cortes)" % (
                            pair.ct_series_description, pair.pet_series_description, pair.num_slices
                        )
                        pair_item = QTreeWidgetItem([pair_label])
                        pair_item.setIcon(0, get_fusion_icon(pair_is_built))
                        pair_item.setFlags((pair_item.flags() | Qt.ItemIsUserCheckable) | Qt.ItemIsEnabled)
                        pair_item.setCheckState(0, Qt.Unchecked)
                        pair_item.setData(0, Qt.UserRole, {
                            "type": NODE_TYPE_FUSION_PAIR,
                            "label": pair_label,
                            "study_instance_uid": study.study_instance_uid,
                            "study_description": study.study_description,
                            "study_date": study.study_date,
                            "study_directory": study.study_directory,
                            "patient_name": patient.patient_name,
                            "patient_id": patient.patient_id,
                            "pair": pair_dict,
                            "pair_key": key,
                            "modality": "FUSION",
                            "is_multi_study": True,
                        })
                        fusion_group_item.addChild(pair_item)
                    study_item.addChild(fusion_group_item)

                patient_item.addChild(study_item)

            category_item.addChild(patient_item)

        return category_item

    def _is_multi_study_item(self, item):
        curr = item
        while curr:
            data = curr.data(0, Qt.UserRole) or {}
            if data.get("type") == NODE_TYPE_MULTI_STUDY_CATEGORY or data.get("is_multi_study"):
                return True
            curr = curr.parent()
        return False

    def _get_item_modality(self, data):
        if not data:
            return None
        node_type = data.get("type")
        if node_type == NODE_TYPE_FUSION_PAIR or data.get("modality") == "FUSION":
            return "FUSION"
        mod = str(data.get("modality") or "").upper()
        if mod in ("PT", "PET"):
            return "PET"
        if mod in ("CT",):
            return "CT"
        return mod or None

    def _collect_all_checked_multi_study_items(self):
        checked = []

        def _scan(node):
            data = node.data(0, Qt.UserRole) or {}
            node_type = data.get("type")
            if node_type in (NODE_TYPE_SERIES, NODE_TYPE_FUSION_PAIR):
                if (node.flags() & Qt.ItemIsUserCheckable) and node.checkState(0) == Qt.Checked:
                    checked.append((node, data))
            for i in range(node.childCount()):
                _scan(node.child(i))

        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            data = top.data(0, Qt.UserRole) or {}
            if data.get("type") == NODE_TYPE_MULTI_STUDY_CATEGORY:
                _scan(top)
                break
        return checked

    def _set_multi_study_checkboxes_enabled(self, target_modality=None):
        def _apply(node):
            data = node.data(0, Qt.UserRole) or {}
            node_type = data.get("type")
            if node_type in (NODE_TYPE_SERIES, NODE_TYPE_FUSION_PAIR):
                mod = self._get_item_modality(data)
                if target_modality is None or mod == target_modality:
                    node.setFlags(node.flags() | Qt.ItemIsEnabled)
                else:
                    node.setFlags(node.flags() & ~Qt.ItemIsEnabled)
                    if node.checkState(0) != Qt.Unchecked:
                        node.setCheckState(0, Qt.Unchecked)
            for i in range(node.childCount()):
                _apply(node.child(i))

        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            data = top.data(0, Qt.UserRole) or {}
            if data.get("type") == NODE_TYPE_MULTI_STUDY_CATEGORY:
                _apply(top)
                break

    def _set_subtree_expanded(self, item, state):
        item.setExpanded(state)
        for i in range(item.childCount()):
            self._set_subtree_expanded(item.child(i), state)

    def _on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole) or {}
        node_type = data.get("type")
        if node_type == NODE_TYPE_MULTI_STUDY_PATIENT:
            is_expanded = item.isExpanded()
            new_state = not is_expanded
            self._set_subtree_expanded(item, new_state)

    def _on_item_changed(self, item, column):
        if column != 0 or self._updating_checks:
            return
        if not self._is_multi_study_item(item):
            return

        self._updating_checks = True
        try:
            checked_items_with_nodes = self._collect_all_checked_multi_study_items()
            if not checked_items_with_nodes:
                self._active_multi_study_modality = None
                self._set_multi_study_checkboxes_enabled(target_modality=None)
                self.multi_study_selection_changed.emit([])
            else:
                active_mod = self._get_item_modality(checked_items_with_nodes[0][1])
                self._active_multi_study_modality = active_mod
                self._set_multi_study_checkboxes_enabled(target_modality=active_mod)

                checked_data = [d for _, d in checked_items_with_nodes]
                self.multi_study_selection_changed.emit(checked_data)
        finally:
            self._updating_checks = False

    def _uncheck_all_multi_study(self):
        def _uncheck(node):
            data = node.data(0, Qt.UserRole) or {}
            node_type = data.get("type")
            if node_type in (NODE_TYPE_SERIES, NODE_TYPE_FUSION_PAIR):
                node.setCheckState(0, Qt.Unchecked)
            for i in range(node.childCount()):
                _uncheck(node.child(i))

        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            data = top.data(0, Qt.UserRole) or {}
            if data.get("type") == NODE_TYPE_MULTI_STUDY_CATEGORY:
                _uncheck(top)
                break

    def clear_multi_study_checks(self):
        self._updating_checks = True
        try:
            self._uncheck_all_multi_study()
            self._active_multi_study_modality = None
            self._set_multi_study_checkboxes_enabled(target_modality=None)
        finally:
            self._updating_checks = False
        self.multi_study_selection_changed.emit([])

    def _on_selection_changed(self):
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        data = item.data(0, Qt.UserRole) or {}

        # Desactivar y deshacer checkboxes si se navega fuera de la rama multi-estudio
        if self._active_multi_study_modality:
            is_ms = self._is_multi_study_item(item)
            if not is_ms:
                self.clear_multi_study_checks()

        if data:
            self.node_selected.emit(data)


class SegmentationWorker(QObject):
    log_message = Signal(str)
    finished = Signal(dict)

    def __init__(self, input_nii_path, organ_key, modality, cuda, fast, output_dir, output_basename,
                 seg_context="", target_id="", series_uid="", organ_display="", pair_key="", pair=None):
        super().__init__()
        self.input_nii_path = input_nii_path
        self.organ_key = organ_key
        self.modality = modality.upper()
        self.cuda = cuda
        self.fast = fast
        self.output_dir = output_dir
        self.output_basename = output_basename
        self.seg_context = seg_context
        self.target_id = target_id
        self.series_uid = series_uid
        self.organ_display = organ_display
        self.pair_key = pair_key
        self.pair = pair

    def run(self):
        try:
            import segmentation_anato_ct_TotalSegmentator as seg_ct
            import segmentation_anato_mri_TotalSegmentator as seg_mri

            self.log_message.emit(f"Iniciando segmentación de {self.organ_key} en {self.modality} ({self.seg_context.upper()})...")
            if self.modality == "CT":
                seg_nii, meta = seg_ct.segment_organ(
                    input_volume=self.input_nii_path,
                    organ=self.organ_key,
                    cuda=self.cuda,
                    fast=self.fast,
                    output_dir=self.output_dir,
                    output_basename=self.output_basename,
                    quiet=True,
                )
            else:
                seg_nii, meta = seg_mri.segment_organ_mri(
                    input_volume=self.input_nii_path,
                    organ=self.organ_key,
                    cuda=self.cuda,
                    fast=self.fast,
                    output_dir=self.output_dir,
                    output_basename=self.output_basename,
                    quiet=True,
                )
            self.finished.emit({
                "success": True,
                "error_message": "",
                "metadata": meta,
                "seg_nii": seg_nii,
                "seg_context": self.seg_context,
                "target_id": self.target_id,
                "series_uid": self.series_uid,
                "organ": self.organ_key,
                "organ_display": self.organ_display,
                "modality": self.modality,
                "pair_key": self.pair_key,
                "pair": self.pair,
            })
        except Exception as exc:
            logger.exception("Error durante segmentación: %s", exc)
            self.finished.emit({
                "success": False,
                "error_message": str(exc),
                "metadata": {},
                "seg_nii": None,
                "seg_context": self.seg_context,
                "target_id": self.target_id,
                "series_uid": self.series_uid,
                "organ": self.organ_key,
                "organ_display": self.organ_display,
                "modality": self.modality,
                "pair_key": self.pair_key,
                "pair": self.pair,
            })


class CollapsibleSection(QWidget):
    def __init__(self, title="Segmentación", parent=None, is_sublevel=False):
        super().__init__(parent)
        self._is_expanded = True
        self._title_text = title
        self._is_sublevel = is_sublevel
        self._is_section_enabled = True

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)

        self.toggle_button = QPushButton()
        self.toggle_button.clicked.connect(self.toggle)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self._update_button_style()
        main_layout.addWidget(self.toggle_button)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        margin = 4 if is_sublevel else 0
        self.content_layout.setContentsMargins(margin, 0, 0, 0)
        self.content_layout.setSpacing(6)
        main_layout.addWidget(self.content_widget)

        self._update_header()

    def _update_button_style(self):
        padding = "5px 8px" if self._is_sublevel else "6px 10px"
        font_weight = "600" if self._is_sublevel else "bold"
        color_style = "" if self._is_section_enabled else "color: #777777;"
        self.toggle_button.setStyleSheet(
            f"QPushButton {{"
            f"  text-align: left;"
            f"  font-weight: {font_weight};"
            f"  padding: {padding};"
            f"  {color_style}"
            f"}}"
        )

    def toggle(self):
        self._is_expanded = not self._is_expanded
        self.content_widget.setVisible(self._is_expanded)
        self._update_header()

    def set_expanded(self, expanded):
        self._is_expanded = bool(expanded)
        self.content_widget.setVisible(self._is_expanded)
        self._update_header()

    def is_expanded(self):
        return self._is_expanded

    def set_title(self, title):
        self._title_text = title
        self._update_header()

    def title(self):
        return self._title_text

    def title_text(self):
        return self._title_text

    def set_status_icon(self, is_valid: bool):
        icon_path = ICON_GREEN_PATH if is_valid else ICON_RED_PATH
        if os.path.isfile(icon_path):
            icon = QIcon()
            icon.addFile(icon_path, QSize(16, 16), QIcon.Normal, QIcon.Off)
            icon.addFile(icon_path, QSize(16, 16), QIcon.Disabled, QIcon.Off)
            self.toggle_button.setIcon(icon)
            self.toggle_button.setIconSize(QSize(14, 14))

    def set_section_enabled(self, enabled: bool):
        self._is_section_enabled = bool(enabled)
        self.set_status_icon(enabled)
        self.content_widget.setEnabled(bool(enabled))
        self._update_button_style()
        tip = "Herramienta disponible" if enabled else "Herramienta no disponible para la selección actual"
        self.toggle_button.setToolTip(tip)

    def is_section_enabled(self):
        return self._is_section_enabled

    def _update_header(self):
        arrow = "▼" if self._is_expanded else "▶"
        self.toggle_button.setText(f"{arrow}  {self._title_text}")


class SpatialResolutionAnalysisWidget(QWidget):
    analysis_completed = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_node_data = None
        self._dicom_root = None
        self._larmornium_files_dir = None
        self._config_path = None
        self._detected_pet_series = None
        self._resolved_pet_nii_path = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Selección de origen del volumen PET
        group_source = QGroupBox("1. Volumen PET de entrada")
        source_layout = QVBoxLayout(group_source)
        source_layout.setSpacing(6)

        self.rb_active_pet = QRadioButton("Serie PET de la sesión")
        self.rb_active_pet.setChecked(True)
        self.rb_active_pet.toggled.connect(self._on_source_type_changed)
        source_layout.addWidget(self.rb_active_pet)

        self.lbl_active_pet = QLabel("No hay serie PET seleccionada en el árbol.")
        self.lbl_active_pet.setWordWrap(True)
        self.lbl_active_pet.setStyleSheet("color: #666; font-style: italic;")
        source_layout.addWidget(self.lbl_active_pet)

        self.rb_file_pet = QRadioButton("Archivo NIfTI manual (.nii / .nii.gz)")
        self.rb_file_pet.toggled.connect(self._on_source_type_changed)
        source_layout.addWidget(self.rb_file_pet)

        self.file_picker_widget = QWidget()
        file_picker_layout = QHBoxLayout(self.file_picker_widget)
        file_picker_layout.setContentsMargins(0, 0, 0, 0)
        file_picker_layout.setSpacing(4)
        self.txt_nii_path = QLineEdit()
        self.txt_nii_path.setPlaceholderText("Seleccionar archivo .nii o .nii.gz...")
        self.btn_browse = QPushButton("Examinar...")
        self.btn_browse.clicked.connect(self._on_browse_file)
        file_picker_layout.addWidget(self.txt_nii_path)
        file_picker_layout.addWidget(self.btn_browse)
        self.file_picker_widget.setVisible(False)
        source_layout.addWidget(self.file_picker_widget)

        self.lbl_vol_info = QLabel("")
        self.lbl_vol_info.setWordWrap(True)
        source_layout.addWidget(self.lbl_vol_info)

        main_layout.addWidget(group_source)

        # Parámetros del análisis
        group_params = QGroupBox("2. Parámetros del análisis")
        params_layout = QVBoxLayout(group_params)
        params_layout.setSpacing(6)

        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Umbral relativo:"))
        self.spn_threshold = QDoubleSpinBox()
        self.spn_threshold.setRange(0.01, 0.99)
        self.spn_threshold.setSingleStep(0.01)
        self.spn_threshold.setValue(0.10)
        thresh_layout.addWidget(self.spn_threshold)
        params_layout.addLayout(thresh_layout)

        slice_layout = QHBoxLayout()
        self.chk_auto_slice = QCheckBox("Corte axial automático (máx. global)")
        self.chk_auto_slice.setChecked(True)
        self.chk_auto_slice.toggled.connect(self._on_auto_slice_toggled)
        slice_layout.addWidget(self.chk_auto_slice)

        self.spn_slice_k = QSpinBox()
        self.spn_slice_k.setRange(0, 9999)
        self.spn_slice_k.setValue(0)
        self.spn_slice_k.setEnabled(False)
        slice_layout.addWidget(self.spn_slice_k)
        params_layout.addLayout(slice_layout)

        params_layout.addWidget(QLabel("Coordenadas de referencia (X, Y):"))
        self.table_coords = QTableWidget(3, 2)
        self.table_coords.setHorizontalHeaderLabels(["X", "Y"])
        self.table_coords.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_coords.setMaximumHeight(120)
        self._populate_default_coords()
        params_layout.addWidget(self.table_coords)

        btn_coords_layout = QHBoxLayout()
        btn_coords_layout.setSpacing(4)
        self.btn_add_coord = QPushButton("+ Añadir")
        self.btn_add_coord.clicked.connect(self._on_add_coord)
        self.btn_del_coord = QPushButton("- Quitar")
        self.btn_del_coord.clicked.connect(self._on_del_coord)
        self.btn_reset_coords = QPushButton("Restablecer")
        self.btn_reset_coords.clicked.connect(self._populate_default_coords)
        btn_coords_layout.addWidget(self.btn_add_coord)
        btn_coords_layout.addWidget(self.btn_del_coord)
        btn_coords_layout.addWidget(self.btn_reset_coords)
        params_layout.addLayout(btn_coords_layout)

        self.btn_run = QPushButton("Ejecutar análisis de resolución")
        self.btn_run.clicked.connect(self._on_run_analysis)
        params_layout.addWidget(self.btn_run)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        params_layout.addWidget(self.lbl_status)

        main_layout.addWidget(group_params)
        main_layout.addStretch()

    def _populate_default_coords(self):
        default_coords = [(0.0, 1.0), (10.0, 0.0), (20.0, 0.0)]
        self.table_coords.setRowCount(len(default_coords))
        for r, (x, y) in enumerate(default_coords):
            item_x = QTableWidgetItem(f"{x:g}")
            item_y = QTableWidgetItem(f"{y:g}")
            item_x.setTextAlignment(Qt.AlignCenter)
            item_y.setTextAlignment(Qt.AlignCenter)
            self.table_coords.setItem(r, 0, item_x)
            self.table_coords.setItem(r, 1, item_y)

    def _on_add_coord(self):
        row = self.table_coords.rowCount()
        self.table_coords.insertRow(row)
        item_x = QTableWidgetItem("0")
        item_y = QTableWidgetItem("0")
        item_x.setTextAlignment(Qt.AlignCenter)
        item_y.setTextAlignment(Qt.AlignCenter)
        self.table_coords.setItem(row, 0, item_x)
        self.table_coords.setItem(row, 1, item_y)

    def _on_del_coord(self):
        current_row = self.table_coords.currentRow()
        if current_row >= 0 and self.table_coords.rowCount() > 1:
            self.table_coords.removeRow(current_row)

    def _get_reference_coords(self):
        coords = []
        for r in range(self.table_coords.rowCount()):
            it_x = self.table_coords.item(r, 0)
            it_y = self.table_coords.item(r, 1)
            try:
                vx = float(it_x.text()) if it_x else 0.0
                vy = float(it_y.text()) if it_y else 0.0
                coords.append((vx, vy))
            except ValueError:
                coords.append((0.0, 0.0))
        return coords

    def _on_auto_slice_toggled(self, checked):
        self.spn_slice_k.setEnabled(not checked)

    def _on_source_type_changed(self):
        is_file = self.rb_file_pet.isChecked()
        self.file_picker_widget.setVisible(is_file)
        self.lbl_active_pet.setVisible(not is_file)
        if is_file:
            path = self.txt_nii_path.text().strip()
            self._check_volume_file(path)
        else:
            self._resolve_active_pet_volume()

    def _on_browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar volumen PET NIfTI",
            "",
            "Archivos NIfTI (*.nii *.nii.gz);;Todos los archivos (*)"
        )
        if path:
            self.txt_nii_path.setText(path)
            self._check_volume_file(path)

    def _check_volume_file(self, file_path):
        if not file_path or not os.path.isfile(file_path):
            self.lbl_vol_info.setText("Archivo no encontrado.")
            return False
        try:
            loaded = nib.load(file_path)
            shape = loaded.shape
            zooms = [round(float(z), 3) for z in loaded.header.get_zooms()[:3]]
            self.lbl_vol_info.setText(
                f"Volumen cargado: {os.path.basename(file_path)}\n"
                f"Dimensiones: {list(shape)} | Espaciado: {zooms} mm"
            )
            if len(shape) >= 3:
                self.spn_slice_k.setMaximum(shape[2] - 1)
            return True
        except Exception as e:
            self.lbl_vol_info.setText(f"Error al leer NIfTI: {e}")
            return False

    def set_active_series(self, node_data, dicom_root, larmornium_files_dir, config_path):
        self._current_node_data = node_data
        self._dicom_root = dicom_root
        self._larmornium_files_dir = larmornium_files_dir or LARMORNIUM_FILES_DIR
        self._config_path = config_path or RECENT_FOLDERS_CONFIG_PATH

        if not node_data:
            self._detected_pet_series = None
            self._resolved_pet_nii_path = None
            self.lbl_active_pet.setText("No hay serie PET seleccionada en el árbol.")
            return

        node_type = node_data.get("type")
        modality = str(node_data.get("modality") or "").upper()
        prefix = node_data.get("prefix")

        pet_series = None
        if modality in ("PT", "PET") or (prefix == "pet_ct" and modality in ("PT", "PET")):
            pet_series = node_data
        elif node_type == NODE_TYPE_FUSION_PAIR:
            pair = node_data.get("pair")
            if pair:
                pet_uid = pair.get("pet_series_instance_uid")
                pet_series = pair.get("pet_series") or {
                    "series_instance_uid": pet_uid,
                    "series_directory": pair.get("pet_directory", ""),
                    "study_instance_uid": pair.get("study_instance_uid", ""),
                }

        self._detected_pet_series = pet_series
        if pet_series:
            desc = pet_series.get("series_description") or pet_series.get("label") or "PET"
            self.lbl_active_pet.setText(f"Serie detectada: {desc}")
            self._resolve_active_pet_volume()
        else:
            self.lbl_active_pet.setText("La serie activa no es de modalidad PET.")

    def _resolve_active_pet_volume(self):
        if not self._detected_pet_series:
            return None
        series = self._detected_pet_series
        series_uid = series.get("series_instance_uid")
        if not series_uid:
            return None

        built = join_pet_ct._load_built_pet_volumes(self._config_path) if hasattr(join_pet_ct, "_load_built_pet_volumes") else {}
        if series_uid in built and os.path.isfile(built[series_uid].get("nii_path", "")):
            self._resolved_pet_nii_path = built[series_uid]["nii_path"]
            self._check_volume_file(self._resolved_pet_nii_path)
            return self._resolved_pet_nii_path

        if self._dicom_root and self._larmornium_files_dir:
            try:
                uid, rec = join_pet_ct.ensure_pet_volume_for_series(
                    series, self._dicom_root, self._larmornium_files_dir, self._config_path
                )
                if rec and rec.get("nii_path") and os.path.isfile(rec["nii_path"]):
                    self._resolved_pet_nii_path = rec["nii_path"]
                    self._check_volume_file(self._resolved_pet_nii_path)
                    return self._resolved_pet_nii_path
            except Exception as e:
                logger.warning("No se pudo autogenerar volumen PET: %s", e)
        return None

    def _build_volume_context(self, vol_path):
        node_data = self._current_node_data or {}
        node_type = node_data.get("type")

        # Caso 1: Par de fusión PET/CT
        if self.rb_active_pet.isChecked() and node_type == NODE_TYPE_FUSION_PAIR:
            pair = node_data.get("pair")
            pair_key = node_data.get("pair_key") or (join_pet_ct._pair_key(pair) if pair else "")
            if pair or pair_key:
                try:
                    built_pairs = join_pet_ct._load_built_pairs(self._config_path or RECENT_FOLDERS_CONFIG_PATH)
                    rec_to_load = built_pairs.get(pair_key) if (pair_key and pair_key in built_pairs) else (pair_key or pair)
                    dir_files_dir = self._larmornium_files_dir or LARMORNIUM_FILES_DIR
                    fused_data = join_pet_ct.load_fused_volume_data(
                        record_or_key=rec_to_load,
                        larmornium_files_dir=dir_files_dir,
                        dicom_root=self._dicom_root,
                        pair=pair
                    )
                    ct_vol = fused_data.get("ct_volume")
                    pet_vol = fused_data.get("pet_volume")
                    max_suv = fused_data.get("max_suv", 1.0)
                    ps = fused_data.get("pixel_spacing", [1.0, 1.0])
                    st = fused_data.get("slice_thickness", 1.0)
                    sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
                    label = node_data.get("label") or node_data.get("series_description") or "Fusión PET/CT"
                    return {
                        "is_fusion": True,
                        "ct_volume": ct_vol,
                        "pet_volume": pet_vol,
                        "voxel_spacing": sp_3d,
                        "max_suv": max_suv,
                        "title": f"Fusión PET/CT ({label})"
                    }
                except Exception as exc:
                    logger.warning("No se pudo cargar datos de fusión para 3D: %s", exc)

        # Caso 2: Serie PET activa
        if self.rb_active_pet.isChecked() and self._detected_pet_series:
            series_uid = self._detected_pet_series.get("series_instance_uid")
            built_pet = join_pet_ct._load_built_pet_volumes(self._config_path or RECENT_FOLDERS_CONFIG_PATH) if hasattr(join_pet_ct, "_load_built_pet_volumes") else {}
            if series_uid and series_uid in built_pet:
                try:
                    pet_data = join_pet_ct.load_single_volume_data(built_pet[series_uid], modality="PET")
                    pet_vol = pet_data.get("volume")
                    ps = pet_data.get("pixel_spacing", [1.0, 1.0])
                    st = pet_data.get("slice_thickness", 1.0)
                    sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
                    max_suv = pet_data.get("max_suv", 1.0)
                    desc = self._detected_pet_series.get("series_description") or self._detected_pet_series.get("label") or "PET"
                    return {
                        "is_fusion": False,
                        "ct_volume": None,
                        "pet_volume": pet_vol,
                        "voxel_spacing": sp_3d,
                        "max_suv": max_suv,
                        "title": f"PET ({desc})"
                    }
                except Exception as exc:
                    logger.warning("No se pudo cargar datos de serie PET para 3D: %s", exc)

        # Caso 3: Archivo NIfTI (o fallback si no se cargó desde DICOM)
        if vol_path and os.path.isfile(vol_path):
            try:
                loaded = nib.load(vol_path)
                raw = loaded.get_fdata()
                if raw.ndim == 3 and raw.shape[0] > raw.shape[2]:
                    pet_vol = np.transpose(raw, (2, 1, 0))
                else:
                    pet_vol = raw
                zooms = [float(v) for v in loaded.header.get_zooms()[:3]]
                sp_3d = [zooms[0], zooms[1], zooms[2]]
                max_suv = float(np.nanmax(pet_vol)) if pet_vol.size > 0 else 1.0
                return {
                    "is_fusion": False,
                    "ct_volume": None,
                    "pet_volume": pet_vol,
                    "voxel_spacing": sp_3d,
                    "max_suv": max_suv,
                    "title": os.path.basename(vol_path)
                }
            except Exception as exc:
                logger.warning("No se pudo cargar NIfTI para contexto 3D: %s", exc)

        return None

    def _on_run_analysis(self):
        if self.rb_active_pet.isChecked():
            vol_path = self._resolved_pet_nii_path or self._resolve_active_pet_volume()
            if not vol_path or not os.path.isfile(vol_path):
                QMessageBox.warning(
                    self,
                    "Volumen PET no disponible",
                    "No se encontró un volumen NIfTI para la serie PET activa.\n"
                    "Puede seleccionar 'Archivo NIfTI manual' para cargar un archivo .nii / .nii.gz."
                )
                return
        else:
            vol_path = self.txt_nii_path.text().strip()
            if not vol_path or not os.path.isfile(vol_path):
                QMessageBox.warning(
                    self,
                    "Archivo no válido",
                    "Seleccione un archivo de volumen NIfTI válido (.nii o .nii.gz)."
                )
                return

        thresh = self.spn_threshold.value()
        slice_k = None if self.chk_auto_slice.isChecked() else self.spn_slice_k.value()
        ref_coords = self._get_reference_coords()

        self.lbl_status.setText("Ejecutando análisis de resolución espacial...")
        QApplication.processEvents()

        try:
            results = analisis_resolucion_espacial.ejecutar_analisis_resolucion_espacial(
                pet_volume=vol_path,
                threshold=thresh,
                slice_k=slice_k,
                reference_coords=ref_coords,
            )
            num_pts = results.get("num_puntos", 0)
            corte_k = results.get("corte_axial_k", 0)
            vol_context = self._build_volume_context(vol_path)
            self.lbl_status.setText(
                f"Análisis completado: {num_pts} punto(s) en corte k={corte_k}.\n"
                f"Resultados mostrados en la pestaña 'Resultados de FWHM'."
            )
            self.analysis_completed.emit(results, vol_context)
        except Exception as e:
            logger.exception("Error al ejecutar análisis de resolución espacial")
            self.lbl_status.setText(f"Error: {e}")
            QMessageBox.critical(self, "Error de análisis", f"Ocurrió un error durante el cálculo:\n{e}")


class UniformityAnalysisWidget(QWidget):
    analysis_completed = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_node_data = None
        self._dicom_root = None
        self._larmornium_files_dir = None
        self._config_path = None
        self._active_pair = None
        self._active_pair_key = ""

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Estudio de fusión activo
        self.lbl_target = QLabel("Seleccione un par o estudio de fusión PET/CT para habilitar.")
        self.lbl_target.setWordWrap(True)
        main_layout.addWidget(self.lbl_target)

        # Regla de segmentación (Fantoma de agua)
        group_rule = QGroupBox("Regla de segmentación (Fantoma de agua)")
        rule_layout = QVBoxLayout(group_rule)
        rule_layout.setSpacing(6)

        morph_row = QHBoxLayout()
        morph_row.addWidget(QLabel("Cadena morfológica:"))
        self.txt_morph_code = QLineEdit("d1d_d1e_d1e_d1d")
        self.txt_morph_code.setToolTip("Código de operaciones morfológicas 2D (ej: d1d_d1e_d1e_d1d)")
        morph_row.addWidget(self.txt_morph_code)
        rule_layout.addLayout(morph_row)

        hu_row = QHBoxLayout()
        hu_row.addWidget(QLabel("HU mín:"))
        self.spn_hu_min = QDoubleSpinBox()
        self.spn_hu_min.setRange(-1000.0, 3000.0)
        self.spn_hu_min.setValue(-125.0)
        self.spn_hu_min.setSingleStep(5.0)
        hu_row.addWidget(self.spn_hu_min)

        hu_row.addWidget(QLabel("HU máx:"))
        self.spn_hu_max = QDoubleSpinBox()
        self.spn_hu_max.setRange(-1000.0, 3000.0)
        self.spn_hu_max.setValue(125.0)
        self.spn_hu_max.setSingleStep(5.0)
        hu_row.addWidget(self.spn_hu_max)
        rule_layout.addLayout(hu_row)

        main_layout.addWidget(group_rule)

        # Botón de ejecución y estado
        self.btn_run = QPushButton("Ejecutar análisis de uniformidad")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run_analysis)
        main_layout.addWidget(self.btn_run)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        main_layout.addWidget(self.lbl_status)

        main_layout.addStretch()

    def set_active_target(self, node_data, dicom_root, larmornium_files_dir, config_path):
        self._current_node_data = node_data
        self._dicom_root = dicom_root
        self._larmornium_files_dir = larmornium_files_dir or LARMORNIUM_FILES_DIR
        self._config_path = config_path or RECENT_FOLDERS_CONFIG_PATH

        if not node_data:
            self._disable_target()
            return

        node_type = node_data.get("type")
        label = node_data.get("label") or node_data.get("series_description") or "Estudio"

        pair = None
        if node_type == NODE_TYPE_FUSION_PAIR:
            pair = node_data.get("pair")

        pair_key = node_data.get("pair_key") or (join_pet_ct._pair_key(pair) if pair else "")

        if not pair and not pair_key:
            self._disable_target()
            return

        self._active_pair = pair
        self._active_pair_key = pair_key
        self.lbl_target.setText(f"Estudio fusión activo: {label}")
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("")

    def _disable_target(self):
        self._active_pair = None
        self._active_pair_key = ""
        self.lbl_target.setText("Seleccione un par o estudio de fusión PET/CT para habilitar.")
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("")

    def _on_run_analysis(self):
        if not self._active_pair and not self._active_pair_key:
            QMessageBox.warning(self, "Sin estudio de fusión", "Seleccione un par o estudio de fusión PET/CT.")
            return

        dir_files_dir = get_directory_files_dir(self._dicom_root) if self._dicom_root else (self._larmornium_files_dir or LARMORNIUM_FILES_DIR)
        pair_key = self._active_pair_key
        pair = self._active_pair

        built_pairs = join_pet_ct._load_built_pairs(self._config_path)
        fusion_nii = None
        if pair_key in built_pairs and os.path.isfile(built_pairs[pair_key].get("nii_path", "")):
            fusion_nii = built_pairs[pair_key]["nii_path"]
        else:
            fallback = os.path.join(dir_files_dir, FUSION_VOL_DIRNAME, f"{pair_key}.nii.gz")
            if os.path.isfile(fallback):
                fusion_nii = fallback

        if not fusion_nii or not os.path.isfile(fusion_nii):
            if pair:
                self.lbl_status.setText("Generando volumen fusionado PET/CT previo al análisis...")
                QApplication.processEvents()
                try:
                    _, rec = join_pet_ct.ensure_fusion_volume_for_pair(
                        pair, self._dicom_root, dir_files_dir, self._config_path
                    )
                    fusion_nii = rec.get("nii_path")
                except Exception as exc:
                    self.lbl_status.setText(f"Error al generar volumen fusionado: {exc}")
                    QMessageBox.critical(self, "Error de fusión", f"No se pudo generar el volumen fusionado:\n{exc}")
                    return
            else:
                QMessageBox.warning(self, "Volumen fusionado no encontrado", "No se encontró el archivo de fusión correspondiente.")
                return

        morph_code = self.txt_morph_code.text().strip() or "d1d_d1e_d1e_d1d"
        hu_min = self.spn_hu_min.value()
        hu_max = self.spn_hu_max.value()

        self.lbl_status.setText("Ejecutando análisis de uniformidad...")
        QApplication.processEvents()

        try:
            results = analisis_uniformidad.ejecutar_analisis_uniformidad(
                fusion_input=fusion_nii,
                hu_min=hu_min,
                hu_max=hu_max,
                morph_code=morph_code,
                output_dir=os.path.join(dir_files_dir, "uniformidad"),
                verbose=True,
            )
            n_slices = results.get("num_slices_analyzed", 0)
            gm = results.get("global_metrics", {})
            self.lbl_status.setText(
                f"Análisis completado: {n_slices} cortes analizados.\n"
                f"Promedio: {gm.get('mean_suv', 0.0):.2f} SUV | CV: {gm.get('cv_percent', 0.0):.2f}%\n"
                f"Resultados mostrados en la pestaña 'Resultados de Uniformidad'."
            )
            vol_context = {
                "pair_key": pair_key,
                "fusion_nii": fusion_nii,
                "node_data": self._current_node_data,
            }
            self.analysis_completed.emit(results, vol_context)
        except Exception as e:
            logger.exception("Error al ejecutar análisis de uniformidad")
            self.lbl_status.setText(f"Error: {e}")
            QMessageBox.critical(self, "Error de análisis", f"Ocurrió un error durante el cálculo de uniformidad:\n{e}")


class FWHMResultsDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._analysis_results = None
        self._vol_context = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal, self)

        # Panel izquierdo: Puntos de máxima actividad y Métricas
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        group_points = QGroupBox("3. Puntos de máxima actividad")
        points_layout = QVBoxLayout(group_points)
        points_layout.setSpacing(4)

        self.points_list = QListWidget()
        self.points_list.currentItemChanged.connect(self._on_point_selected)
        points_layout.addWidget(self.points_list)
        left_layout.addWidget(group_points, 1)

        group_metrics = QGroupBox("Métricas de FWHM")
        metrics_layout = QVBoxLayout(group_metrics)
        metrics_layout.setSpacing(6)

        self.lbl_point_title = QLabel("Seleccione un punto para ver detalles.")
        self.lbl_point_title.setWordWrap(True)
        self.lbl_point_title.setStyleSheet("font-weight: bold;")
        metrics_layout.addWidget(self.lbl_point_title)

        self.lbl_metrics_fwhm = QLabel("")
        self.lbl_metrics_fwhm.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_metrics_fwhm)
        metrics_layout.addStretch()

        left_layout.addWidget(group_metrics, 1)

        left_widget.setMinimumWidth(280)
        left_widget.setMaximumWidth(400)
        splitter.addWidget(left_widget)

        # Panel derecho: Sub-pestañas de visualización
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.viz_subtabs = QTabWidget(right_widget)
        self.viz_subtabs.currentChanged.connect(self._on_subtab_changed)

        # Subtab 0: Perfiles 1D
        tab_1d = QWidget()
        t1d_layout = QVBoxLayout(tab_1d)
        t1d_layout.setContentsMargins(4, 4, 4, 4)
        t1d_layout.setSpacing(4)

        bar_1d = QHBoxLayout()
        bar_1d.addWidget(QLabel("Eje:"))
        self.combo_profile_axis = QComboBox()
        self.combo_profile_axis.addItems(["Todos los ejes (X, Y, Z)", "Eje X", "Eje Y", "Eje Z"])
        self.combo_profile_axis.currentIndexChanged.connect(self._render_profile_1d)
        bar_1d.addWidget(self.combo_profile_axis)
        bar_1d.addStretch()
        t1d_layout.addLayout(bar_1d)

        self.fig_1d = Figure(figsize=(6, 4))
        self.canvas_1d = FigureCanvas(self.fig_1d)
        self.canvas_1d.setMinimumHeight(280)
        self.toolbar_1d = NavigationToolbar(self.canvas_1d, self)
        t1d_layout.addWidget(self.toolbar_1d)
        t1d_layout.addWidget(self.canvas_1d)
        self.viz_subtabs.addTab(tab_1d, "Perfiles 1D")

        # Subtab 1: Patches 2D (40x40)
        tab_2d = QWidget()
        t2d_layout = QVBoxLayout(tab_2d)
        t2d_layout.setContentsMargins(4, 4, 4, 4)
        t2d_layout.setSpacing(4)

        bar_2d = QHBoxLayout()
        bar_2d.addWidget(QLabel("Plano:"))
        self.combo_patch_view = QComboBox()
        self.combo_patch_view.addItems(["Plano XY (Transaxial)", "Plano ZY (Sagital)", "Ambos planos"])
        self.combo_patch_view.currentIndexChanged.connect(self._render_patch_2d)
        bar_2d.addWidget(self.combo_patch_view)
        bar_2d.addStretch()
        t2d_layout.addLayout(bar_2d)

        self.fig_2d = Figure(figsize=(6, 4))
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.canvas_2d.setMinimumHeight(280)
        self.toolbar_2d = NavigationToolbar(self.canvas_2d, self)
        t2d_layout.addWidget(self.toolbar_2d)
        t2d_layout.addWidget(self.canvas_2d)
        self.viz_subtabs.addTab(tab_2d, "Patches 2D (40x40)")

        # Subtab 2: Puntos 3D
        tab_3d = QWidget()
        t3d_layout = QVBoxLayout(tab_3d)
        t3d_layout.setContentsMargins(4, 4, 4, 4)
        t3d_layout.setSpacing(4)

        self.fig_3d = Figure(figsize=(6, 4))
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.canvas_3d.setMinimumHeight(280)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self)
        t3d_layout.addWidget(self.toolbar_3d)
        t3d_layout.addWidget(self.canvas_3d)
        self.viz_subtabs.addTab(tab_3d, "Puntos 3D")

        # Subtab 3: Volumen 3D PET (o Fusión PET/CT)
        self.viewer_3d = Viewer3DWidget(self)
        self.viz_subtabs.addTab(self.viewer_3d, "Volumen 3D PET")

        right_layout.addWidget(self.viz_subtabs)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 700])

        main_layout.addWidget(splitter)

    def _on_subtab_changed(self, index):
        if index == 3:
            if hasattr(self.viewer_3d, "canvas") and self.viewer_3d.canvas:
                self.viewer_3d.canvas.render_scene()
        elif index == 0:
            self._render_profile_1d()
        elif index == 1:
            self._render_patch_2d()
        elif index == 2:
            self._render_3d()

    def display_results(self, results, vol_context=None):
        self._analysis_results = results
        self._vol_context = vol_context
        num_pts = results.get("num_puntos", 0) if results else 0

        self.points_list.clear()

        if num_pts == 0:
            self.lbl_point_title.setText("Sin puntos detectados.")
            self.lbl_metrics_fwhm.setText("No se encontraron puntos de actividad con el umbral especificado.")
            self._clear_plots()
            self.viewer_3d.clear()
            return

        for p in results.get("puntos", []):
            pid = p["punto_id"]
            lbl = p.get("label", "")
            pt = p.get("patient_mm", [0, 0, 0])
            max_v = p.get("valor_maximo", 0.0)
            item_text = f"Punto {pid} [{lbl}]: ({pt[0]:+.1f}, {pt[1]:+.1f}, {pt[2]:+.1f}) mm | Máx={max_v:,.1f}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, pid)
            self.points_list.addItem(item)

        # Cargar visualización 3D del volumen PET o fusión
        if vol_context:
            is_fusion = vol_context.get("is_fusion", False)
            ct_vol = vol_context.get("ct_volume")
            pet_vol = vol_context.get("pet_volume")
            voxel_spacing = vol_context.get("voxel_spacing")
            max_suv = vol_context.get("max_suv")
            title = vol_context.get("title", "")
            if is_fusion and ct_vol is not None and pet_vol is not None:
                self.viewer_3d.show_fused_volume(ct_vol, pet_vol, voxel_spacing=voxel_spacing, max_suv=max_suv, title=title)
            elif pet_vol is not None:
                self.viewer_3d.show_pet_volume(pet_vol, voxel_spacing=voxel_spacing, max_suv=max_suv, title=title)

        if self.points_list.count() > 0:
            self.points_list.setCurrentRow(0)

    def _on_point_selected(self, current, previous):
        if not current or not self._analysis_results:
            return
        pid = current.data(Qt.UserRole)
        puntos = self._analysis_results.get("puntos", [])
        punto = next((p for p in puntos if p["punto_id"] == pid), None)
        if not punto:
            return

        lbl = punto.get("label", "")
        pt = punto.get("patient_mm", [0, 0, 0])
        self.lbl_point_title.setText(f"Punto {pid} [{lbl}] - ({pt[0]:+.2f}, {pt[1]:+.2f}, {pt[2]:+.2f}) mm")

        fx = punto.get("fwhm_x_mm")
        fy = punto.get("fwhm_y_mm")
        fz = punto.get("fwhm_z_mm")
        ft = punto.get("fwhm_transaxial_mm")

        fx_str = f"{fx:.3f} mm" if fx is not None else "No convergió"
        fy_str = f"{fy:.3f} mm" if fy is not None else "No convergió"
        fz_str = f"{fz:.3f} mm" if fz is not None else "No convergió"
        ft_str = f"{ft:.3f} mm" if ft is not None else "No convergió"

        rx = punto.get("ajuste_x", {}).get("r_squared")
        ry = punto.get("ajuste_y", {}).get("r_squared")
        rz = punto.get("ajuste_z", {}).get("r_squared")
        rx_s = f"{rx:.4f}" if rx is not None else "N/A"
        ry_s = f"{ry:.4f}" if ry is not None else "N/A"
        rz_s = f"{rz:.4f}" if rz is not None else "N/A"
        r_str = f"R^2: X={rx_s}, Y={ry_s}, Z={rz_s}"

        metrics_text = (
            f"FWHM X: {fx_str}   |   FWHM Y: {fy_str}\n"
            f"FWHM Z: {fz_str}   |   FWHM Transaxial: {ft_str}\n"
            f"{r_str}"
        )
        self.lbl_metrics_fwhm.setText(metrics_text)

        self._render_profile_1d()
        self._render_patch_2d()
        self._render_3d()

    def _render_profile_1d(self):
        if not self._analysis_results or self.points_list.count() == 0:
            return
        current_item = self.points_list.currentItem()
        if not current_item:
            return
        pid = current_item.data(Qt.UserRole)
        perfiles_punto = self._analysis_results.get("perfiles", {}).get(pid, {})

        self.fig_1d.clear()
        view_mode = self.combo_profile_axis.currentIndex()

        axes_to_plot = ["x", "y", "z"] if view_mode == 0 else [["x", "y", "z"][view_mode - 1]]

        n_plots = len(axes_to_plot)
        colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}

        for i, ax_name in enumerate(axes_to_plot):
            ax = self.fig_1d.add_subplot(n_plots, 1, i + 1)
            ax_data = perfiles_punto.get(ax_name, {})
            x_raw = ax_data.get("x_mm", [])
            y_raw = ax_data.get("actividad", [])
            x_dense = ax_data.get("x_dense_mm", [])
            y_fit = ax_data.get("curva_ajustada", [])
            fwhm_val = ax_data.get("fwhm_mm")
            r2_val = ax_data.get("r_squared")

            if x_raw and y_raw:
                ax.plot(x_raw, y_raw, "o", markersize=3, color=colors.get(ax_name, "tab:blue"), label=f"Perfil {ax_name.upper()}")
            if x_dense and y_fit and ax_data.get("converged"):
                ax.plot(x_dense, y_fit, "-", color="tab:red", linewidth=1.5, label=f"Ajuste (R^2={r2_val:.3f})")

            if fwhm_val is not None and ax_data.get("converged") and y_fit:
                fit_p = self._get_fit_params(pid, ax_name)
                if fit_p:
                    a = fit_p.get("a", 0.0)
                    b = fit_p.get("b", 1.0)
                    c = fit_p.get("c", 0.0)
                    half_height = a + (b - a) / 2.0
                    x1 = c - fwhm_val / 2.0
                    x2 = c + fwhm_val / 2.0
                    ax.plot([x1, x2], [half_height, half_height], color="purple", linewidth=2, marker="|", markersize=6, label=f"FWHM = {fwhm_val:.2f} mm")

            ax.set_ylabel(f"Actividad ({ax_name.upper()})", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=7)
            if i == n_plots - 1:
                ax.set_xlabel("Distancia al centro del pico (mm)", fontsize=8)

        self.fig_1d.tight_layout()
        self.canvas_1d.draw()

    def _get_fit_params(self, pid, axis_name):
        puntos = self._analysis_results.get("puntos", [])
        punto = next((p for p in puntos if p["punto_id"] == pid), None)
        if punto:
            return punto.get(f"ajuste_{axis_name}", {})
        return {}

    def _render_patch_2d(self):
        if not self._analysis_results or self.points_list.count() == 0:
            return
        current_item = self.points_list.currentItem()
        if not current_item:
            return
        pid = current_item.data(Qt.UserRole)
        patch_data = self._analysis_results.get("parches_2d", {}).get(pid, {})
        if not patch_data:
            return

        self.fig_2d.clear()
        view_idx = self.combo_patch_view.currentIndex()

        if view_idx == 0:
            ax = self.fig_2d.add_subplot(111)
            self._draw_single_patch_xy(ax, patch_data)
        elif view_idx == 1:
            ax = self.fig_2d.add_subplot(111)
            self._draw_single_patch_zy(ax, patch_data)
        else:
            ax1 = self.fig_2d.add_subplot(121)
            self._draw_single_patch_xy(ax1, patch_data)
            ax2 = self.fig_2d.add_subplot(122)
            self._draw_single_patch_zy(ax2, patch_data)

        self.fig_2d.tight_layout()
        self.canvas_2d.draw()

    def _draw_single_patch_xy(self, ax, patch_data):
        patch_xy = np.array(patch_data["patch_xy"])
        extent = patch_data["extent_xy_mm"]
        fx = patch_data.get("fwhm_h_xy_mm")
        fy = patch_data.get("fwhm_v_xy_mm")

        ax.imshow(patch_xy, extent=extent, origin="lower", cmap="hot", interpolation="nearest")
        ax.set_aspect("equal")

        if fx is not None:
            hx = fx / 2.0
            ax.plot([-hx, hx], [0, 0], color="cyan", linewidth=2, label=f"FWHM X: {fx:.2f} mm")
            ax.plot([-hx, -hx], [-0.5, 0.5], color="cyan", linewidth=2)
            ax.plot([hx, hx], [-0.5, 0.5], color="cyan", linewidth=2)

        if fy is not None:
            hy = fy / 2.0
            ax.plot([0, 0], [-hy, hy], color="lime", linewidth=2, label=f"FWHM Y: {fy:.2f} mm")
            ax.plot([-0.5, 0.5], [-hy, -hy], color="lime", linewidth=2)
            ax.plot([-0.5, 0.5], [hy, hy], color="lime", linewidth=2)

        ax.plot(0, 0, marker="+", color="white", markersize=6)
        ax.set_title("Plano XY (Transaxial 40x40)", fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7)

    def _draw_single_patch_zy(self, ax, patch_data):
        patch_zy = np.array(patch_data["patch_zy"])
        extent = patch_data["extent_zy_mm"]
        fy = patch_data.get("fwhm_h_zy_mm")
        fz = patch_data.get("fwhm_v_zy_mm")

        ax.imshow(patch_zy, extent=extent, origin="lower", cmap="hot", interpolation="nearest")
        ax.set_aspect("equal")

        if fy is not None:
            hy = fy / 2.0
            ax.plot([-hy, hy], [0, 0], color="cyan", linewidth=2, label=f"FWHM Y: {fy:.2f} mm")
            ax.plot([-hy, -hy], [-0.5, 0.5], color="cyan", linewidth=2)
            ax.plot([hy, hy], [-0.5, 0.5], color="cyan", linewidth=2)

        if fz is not None:
            hz = fz / 2.0
            ax.plot([0, 0], [-hz, hz], color="lime", linewidth=2, label=f"FWHM Z: {fz:.2f} mm")
            ax.plot([-0.5, 0.5], [-hz, -hz], color="lime", linewidth=2)
            ax.plot([-0.5, 0.5], [hz, hz], color="lime", linewidth=2)

        ax.plot(0, 0, marker="+", color="white", markersize=6)
        ax.set_title("Plano ZY (Sagital 40x40)", fontsize=9)
        ax.set_xlabel("Y (mm)", fontsize=8)
        ax.set_ylabel("Z (mm)", fontsize=8)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7)

    def _render_3d(self):
        if not self._analysis_results or self.points_list.count() == 0:
            return
        current_item = self.points_list.currentItem()
        if not current_item:
            return
        selected_pid = current_item.data(Qt.UserRole)
        puntos = self._analysis_results.get("puntos", [])

        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection="3d")

        xs, ys, zs = [], [], []
        for p in puntos:
            pt = p.get("patient_mm", [0, 0, 0])
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
            zs.append(float(pt[2]))

        if xs:
            ax.scatter(xs, ys, zs, c="tab:red", s=70, depthshade=False)

        for p in puntos:
            pt = p.get("patient_mm", [0, 0, 0])
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            lbl = p.get("label", "")
            pid = p["punto_id"]
            ax.text(x, y, z + 0.5, f" P{pid} {lbl}\n ({x:.1f}, {y:.1f}, {z:.1f})", fontsize=7)

        sel_p = next((p for p in puntos if p["punto_id"] == selected_pid), None)
        if sel_p:
            spt = sel_p.get("patient_mm", [0, 0, 0])
            ax.scatter([float(spt[0])], [float(spt[1])], [float(spt[2])], c="cyan", s=150, edgecolors="black", linewidth=1.5)

        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Y (mm)", fontsize=8)
        ax.set_zlabel("Z (mm)", fontsize=8)
        ax.set_title("Ubicación 3D de puntos", fontsize=9)

        self.fig_3d.tight_layout()
        self.canvas_3d.draw()

    def _clear_plots(self):
        self.fig_1d.clear()
        self.canvas_1d.draw()
        self.fig_2d.clear()
        self.canvas_2d.draw()
        self.fig_3d.clear()
        self.canvas_3d.draw()

    def clear(self):
        self.points_list.clear()
        self.lbl_point_title.setText("Seleccione un punto para ver detalles.")
        self.lbl_metrics_fwhm.setText("")
        self._clear_plots()
        self.viewer_3d.clear()


class UniformityResultsDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._analysis_results = None
        self._vol_context = None

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(6)

        # Resumen cuantitativo superior
        self.lbl_summary = QLabel("No hay resultados de análisis de uniformidad disponibles.")
        self.lbl_summary.setWordWrap(True)
        main_layout.addWidget(self.lbl_summary)

        # Sub-pestañas dinámicas para cada operación
        self.subtabs = QTabWidget(self)
        main_layout.addWidget(self.subtabs, 1)

    def display_results(self, results, vol_context=None):
        self._analysis_results = results
        self._vol_context = vol_context
        self.subtabs.clear()

        if not results or results.get("status") != "success":
            self.lbl_summary.setText("No se obtuvieron resultados válidos del análisis.")
            return

        gm = results.get("global_metrics", {})
        n_slices = results.get("num_slices_analyzed", 0)
        n_total = results.get("num_slices_total", 0)
        vol_cm3 = gm.get("volume_cm3", 0.0)
        voxels = gm.get("total_voxels", 0)
        mean_suv = gm.get("mean_suv", 0.0)
        std_suv = gm.get("std_suv", 0.0)
        cv_pct = gm.get("cv_percent", 0.0)

        self.lbl_summary.setText(
            f"Cortes analizados en VOI: {n_slices} / {n_total} | "
            f"Volumen VOI: {vol_cm3:.2f} cm3 ({voxels:,} vóxeles) | "
            f"SUV Promedio global: {mean_suv:.3f} | "
            f"Desv. Est. global: {std_suv:.3f} | "
            f"Coeficiente de variación global: {cv_pct:.2f}%"
        )

        slices = results.get("slice_indices", [])
        operations = results.get("operations", [])

        if not slices or not operations:
            empty_lbl = QLabel("No se generaron cortes con vóxeles suficientes para graficar.")
            empty_lbl.setAlignment(Qt.AlignCenter)
            self.subtabs.addTab(empty_lbl, "Sin datos")
            return

        for op in operations:
            tab_page = QWidget()
            page_layout = QVBoxLayout(tab_page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            page_layout.setSpacing(4)

            fig = Figure(figsize=(6, 4))
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(280)
            toolbar = NavigationToolbar(canvas, tab_page)
            page_layout.addWidget(toolbar)
            page_layout.addWidget(canvas)

            ax = fig.add_subplot(111)
            y_vals = op.get("values", [])
            color = op.get("color", "blue")
            label = op.get("label", "Operación")
            x_label = op.get("x_label", "Corte")
            y_label = op.get("y_label", label)

            ax.plot(slices, y_vals, marker='o', markersize=3, color=color, label=label)
            ax.set_title(f"{label} por corte")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            canvas.draw()

            self.subtabs.addTab(tab_page, label)

    def clear(self):
        self._analysis_results = None
        self._vol_context = None
        self.subtabs.clear()
        self.lbl_summary.setText("No hay resultados de análisis de uniformidad disponibles.")


class ResponsiveProfileCanvas(FigureCanvas):
    def __init__(self, figure):
        super().__init__(figure)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            if self.figure and self.figure.axes:
                self.figure.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.06)
                self.draw_idle()
        except Exception:
            pass


class MultiStudyAnalysisPanel(QWidget):
    load_requested = Signal(list)
    visualize_requested = Signal(list)  # Compatibilidad
    slice_ratio_changed = Signal(float)
    ct_window_changed = Signal(float, float)
    pet_suv_changed = Signal(float)
    opacity_changed = Signal(float, float)
    page_change_requested = Signal(int)
    shift_toggled = Signal(bool)

    touchpad_rotate = Signal(float, float)
    touchpad_pan = Signal(float, float)
    touchpad_zoom = Signal(float)
    touchpad_preset = Signal(str)
    touchpad_reset = Signal()

    uniformity_requested = Signal(list)
    spatial_requested = Signal(list)
    segmentation_requested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked_items = []
        self._active_modality = None
        self._current_page = 0
        self._total_pages = 1
        self._ct_window_center = 40.0
        self._ct_window_width = 400.0
        self._current_pet_suv_max = 1.0
        self._reference_global_max_suv = 1.0
        self._ct_opacity = 60
        self._pet_opacity = 40
        self._profiles_data = {}
        self._profiles_alignment_result = None
        self._profiles_study_info = []
        self._current_slice_ratio = 0.5
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        self.viz_section = CollapsibleSection("Visualización", self, is_sublevel=True)
        self._setup_viz_section()
        self.viz_section.set_expanded(False)
        self.viz_section.set_section_enabled(False)
        main_layout.addWidget(self.viz_section)

        self.uniformity_section = CollapsibleSection("Análisis de uniformidad", self, is_sublevel=True)
        self._setup_uniformity_section()
        self.uniformity_section.set_expanded(False)
        self.uniformity_section.set_section_enabled(False)
        main_layout.addWidget(self.uniformity_section)

        self.spatial_section = CollapsibleSection("Análisis de resolución espacial", self, is_sublevel=True)
        self._setup_spatial_section()
        self.spatial_section.set_expanded(False)
        self.spatial_section.set_section_enabled(False)
        main_layout.addWidget(self.spatial_section)

        self.seg_section = CollapsibleSection("Segmentación", self, is_sublevel=True)
        self._setup_seg_section()
        self.seg_section.set_expanded(False)
        self.seg_section.set_section_enabled(False)
        main_layout.addWidget(self.seg_section)

    def _setup_viz_section(self):
        layout = self.viz_section.content_layout
        self.lbl_viz_info = QLabel("Seleccione una o más series o pares en el árbol para visualización múltiple.")
        self.lbl_viz_info.setWordWrap(True)
        layout.addWidget(self.lbl_viz_info)

        self.lst_viz_items = QListWidget()
        self.lst_viz_items.setMaximumHeight(100)
        self.lst_viz_items.setSelectionMode(QAbstractItemView.NoSelection)
        layout.addWidget(self.lst_viz_items)

        # Botones "Cargar" y "Shift"
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self.btn_load_volumes = QPushButton("Cargar")
        self.btn_load_volumes.setEnabled(False)
        self.btn_load_volumes.clicked.connect(self._on_load_volumes)
        btn_row.addWidget(self.btn_load_volumes, 1)
        self.btn_run_viz = self.btn_load_volumes  # Alias para compatibilidad

        self.btn_shift = QPushButton("Shift")
        self.btn_shift.setCheckable(True)
        self.btn_shift.setEnabled(False)
        self.btn_shift.setToolTip("Alinear cortes omitiendo extremos según perfiles axiales (utiliza canal CT en fusión)")
        self.btn_shift.toggled.connect(self.shift_toggled.emit)
        self.btn_shift.toggled.connect(lambda _: self.update_profile_plot())
        btn_row.addWidget(self.btn_shift, 1)

        layout.addLayout(btn_row)

        # Contenedor de controles interactivos
        self.viz_controls_container = QWidget()
        ctrl_layout = QVBoxLayout(self.viz_controls_container)
        ctrl_layout.setContentsMargins(0, 4, 0, 0)
        ctrl_layout.setSpacing(6)

        # Paginación
        pag_row = QHBoxLayout()
        pag_row.setContentsMargins(0, 0, 0, 0)
        pag_row.setSpacing(4)
        self.btn_pag_prev = QPushButton("◀")
        self.btn_pag_prev.setFixedWidth(30)
        self.btn_pag_prev.clicked.connect(lambda: self.page_change_requested.emit(-1))
        pag_row.addWidget(self.btn_pag_prev)

        self.lbl_pagination = QLabel("Pág. 1 / 1")
        self.lbl_pagination.setAlignment(Qt.AlignCenter)
        pag_row.addWidget(self.lbl_pagination, 1)

        self.btn_pag_next = QPushButton("▶")
        self.btn_pag_next.setFixedWidth(30)
        self.btn_pag_next.clicked.connect(lambda: self.page_change_requested.emit(1))
        pag_row.addWidget(self.btn_pag_next)
        ctrl_layout.addLayout(pag_row)

        # Slider de cortes proporcional
        slice_box = QVBoxLayout()
        slice_box.setSpacing(2)
        self.lbl_slice_sync = QLabel("Corte sincronizado: 50%")
        slice_box.addWidget(self.lbl_slice_sync)

        self.slider_slice = QSlider(Qt.Horizontal)
        self.slider_slice.setRange(0, 1000)
        self.slider_slice.setValue(500)
        self.slider_slice.valueChanged.connect(self._on_slice_slider_changed)
        slice_box.addWidget(self.slider_slice)
        ctrl_layout.addLayout(slice_box)

        # Lectura de píxel / Voxel en tiempo real
        self.lbl_pixel_info = QLabel("Pos: (---, ---) | Valor: --- | W: 400 C: 40")
        self.lbl_pixel_info.setWordWrap(True)
        self.lbl_pixel_info.setFixedHeight(36)
        self.lbl_pixel_info.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lbl_pixel_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lbl_pixel_info.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        ctrl_layout.addWidget(self.lbl_pixel_info)

        # Gráfica de perfil axial multi-estudio (Shift) compacta (~70% reducida) y responsive
        self.profile_container = QWidget()
        self.profile_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        prof_layout = QVBoxLayout(self.profile_container)
        prof_layout.setContentsMargins(0, 2, 0, 2)
        prof_layout.setSpacing(0)

        self.lbl_profile_title = QLabel()
        self.lbl_profile_title.setVisible(False)

        self.fig_profile = Figure(facecolor="#000000")
        self.canvas_profile = ResponsiveProfileCanvas(self.fig_profile)
        self.canvas_profile.setFixedHeight(75)
        self.canvas_profile.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        prof_layout.addWidget(self.canvas_profile)

        ctrl_layout.addWidget(self.profile_container)

        # Ventaneo CT
        self.ct_window_widget = QWidget()
        win_layout = QVBoxLayout(self.ct_window_widget)
        win_layout.setContentsMargins(0, 0, 0, 0)
        win_layout.setSpacing(3)

        lbl_win_title = QLabel("Ventaneo CT:")
        win_layout.addWidget(lbl_win_title)

        self.combo_ct_presets = QComboBox()
        self.combo_ct_presets.addItem("Tejido blando (40 / 400)", (40.0, 400.0))
        self.combo_ct_presets.addItem("Pulmón (-600 / 1500)", (-600.0, 1500.0))
        self.combo_ct_presets.addItem("Hueso (400 / 1800)", (400.0, 1800.0))
        self.combo_ct_presets.addItem("Cerebro (40 / 80)", (40.0, 80.0))
        self.combo_ct_presets.addItem("Abdomen (60 / 400)", (60.0, 400.0))
        self.combo_ct_presets.addItem("Mediastino (50 / 350)", (50.0, 350.0))
        self.combo_ct_presets.addItem("Completo (0 / 2000)", (0.0, 2000.0))
        self.combo_ct_presets.currentIndexChanged.connect(self._on_ct_preset_changed)
        win_layout.addWidget(self.combo_ct_presets)

        cw_row = QHBoxLayout()
        cw_row.setContentsMargins(0, 0, 0, 0)
        cw_row.setSpacing(4)
        self.lbl_center = QLabel("C: 40")
        self.slider_center = QSlider(Qt.Horizontal)
        self.slider_center.setRange(-1000, 2000)
        self.slider_center.setValue(40)
        self.slider_center.valueChanged.connect(self._on_cw_changed)
        cw_row.addWidget(self.lbl_center)
        cw_row.addWidget(self.slider_center)
        win_layout.addLayout(cw_row)

        ww_row = QHBoxLayout()
        ww_row.setContentsMargins(0, 0, 0, 0)
        ww_row.setSpacing(4)
        self.lbl_width = QLabel("W: 400")
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 4000)
        self.slider_width.setValue(400)
        self.slider_width.valueChanged.connect(self._on_cw_changed)
        ww_row.addWidget(self.lbl_width)
        ww_row.addWidget(self.slider_width)
        win_layout.addLayout(ww_row)

        ctrl_layout.addWidget(self.ct_window_widget)

        # Ajuste PET SUV
        self.pet_suv_widget = QWidget()
        pet_suv_layout = QVBoxLayout(self.pet_suv_widget)
        pet_suv_layout.setContentsMargins(0, 0, 0, 0)
        pet_suv_layout.setSpacing(3)

        lbl_suv_title = QLabel("Ajuste PET SUV:")
        pet_suv_layout.addWidget(lbl_suv_title)

        suv_row = QHBoxLayout()
        suv_row.setContentsMargins(0, 0, 0, 0)
        suv_row.setSpacing(4)
        self.lbl_pet_suv_max = QLabel("SUV Máx: 1.00")
        self.lbl_pet_suv_max.setMinimumWidth(85)
        self.slider_pet_suv_max = QSlider(Qt.Horizontal)
        self.slider_pet_suv_max.setRange(1, 200)
        self.slider_pet_suv_max.setValue(20)
        self.slider_pet_suv_max.valueChanged.connect(self._on_pet_suv_slider_changed)
        suv_row.addWidget(self.lbl_pet_suv_max)
        suv_row.addWidget(self.slider_pet_suv_max)
        pet_suv_layout.addLayout(suv_row)

        ctrl_layout.addWidget(self.pet_suv_widget)
        self.pet_suv_widget.setVisible(False)

        # Transparencias (CT y PET)
        self.opacity_widget = QWidget()
        op_layout = QVBoxLayout(self.opacity_widget)
        op_layout.setContentsMargins(0, 0, 0, 0)
        op_layout.setSpacing(3)

        self.ct_op_row_widget = QWidget()
        self.ct_op_row = QHBoxLayout(self.ct_op_row_widget)
        self.ct_op_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_ct_op = QLabel("Opacidad CT: 60%")
        self.slider_ct_op = QSlider(Qt.Horizontal)
        self.slider_ct_op.setRange(0, 100)
        self.slider_ct_op.setValue(60)
        self.slider_ct_op.valueChanged.connect(self._on_opacity_slider_changed)
        self.ct_op_row.addWidget(self.lbl_ct_op)
        self.ct_op_row.addWidget(self.slider_ct_op)
        op_layout.addWidget(self.ct_op_row_widget)

        self.pet_op_row_widget = QWidget()
        self.pet_op_row = QHBoxLayout(self.pet_op_row_widget)
        self.pet_op_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_pet_op = QLabel("Opacidad PET: 40%")
        self.slider_pet_op = QSlider(Qt.Horizontal)
        self.slider_pet_op.setRange(0, 100)
        self.slider_pet_op.setValue(40)
        self.slider_pet_op.valueChanged.connect(self._on_opacity_slider_changed)
        self.pet_op_row.addWidget(self.lbl_pet_op)
        self.pet_op_row.addWidget(self.slider_pet_op)
        op_layout.addWidget(self.pet_op_row_widget)

        ctrl_layout.addWidget(self.opacity_widget)

        # Touchpad 3D
        lbl_touch_title = QLabel("Control 3D Touchpad:")
        ctrl_layout.addWidget(lbl_touch_title)

        self.touchpad_3d = MultiStudy3DTouchpad(self)
        self.touchpad_3d.rotate_requested.connect(self.touchpad_rotate.emit)
        self.touchpad_3d.pan_requested.connect(self.touchpad_pan.emit)
        self.touchpad_3d.zoom_requested.connect(self.touchpad_zoom.emit)
        self.touchpad_3d.preset_requested.connect(self.touchpad_preset.emit)
        self.touchpad_3d.reset_requested.connect(self.touchpad_reset.emit)
        ctrl_layout.addWidget(self.touchpad_3d)

        layout.addWidget(self.viz_controls_container)
        self.viz_controls_container.setVisible(False)

    def _setup_uniformity_section(self):
        layout = self.uniformity_section.content_layout
        self.lbl_uniformity_info = QLabel("Análisis de uniformidad sobre los estudios seleccionados.")
        self.lbl_uniformity_info.setWordWrap(True)
        self.lbl_uniformity_info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.lbl_uniformity_info)

        self.btn_run_uniformity = QPushButton("Ejecutar análisis de uniformidad")
        self.btn_run_uniformity.setEnabled(False)
        self.btn_run_uniformity.clicked.connect(self._on_run_uniformity)
        layout.addWidget(self.btn_run_uniformity)

    def _setup_spatial_section(self):
        layout = self.spatial_section.content_layout
        self.lbl_spatial_info = QLabel("Análisis de resolución espacial (FWHM) para series PET seleccionadas.")
        self.lbl_spatial_info.setWordWrap(True)
        self.lbl_spatial_info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.lbl_spatial_info)

        self.btn_run_spatial = QPushButton("Ejecutar análisis de resolución espacial")
        self.btn_run_spatial.setEnabled(False)
        self.btn_run_spatial.clicked.connect(self._on_run_spatial)
        layout.addWidget(self.btn_run_spatial)

    def _setup_seg_section(self):
        layout = self.seg_section.content_layout
        self.lbl_seg_info = QLabel("Segmentación anatómica en lote para los estudios seleccionados.")
        self.lbl_seg_info.setWordWrap(True)
        self.lbl_seg_info.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.lbl_seg_info)

        self.btn_run_seg = QPushButton("Ejecutar segmentación en lote")
        self.btn_run_seg.setEnabled(False)
        self.btn_run_seg.clicked.connect(self._on_run_seg)
        layout.addWidget(self.btn_run_seg)

    def _on_load_volumes(self):
        if self._checked_items:
            self.load_requested.emit(list(self._checked_items))
            self.visualize_requested.emit(list(self._checked_items))

    def _on_run_viz(self):
        self._on_load_volumes()

    def _on_run_uniformity(self):
        if self._checked_items:
            self.uniformity_requested.emit(list(self._checked_items))

    def _on_run_spatial(self):
        if self._checked_items:
            self.spatial_requested.emit(list(self._checked_items))

    def _on_run_seg(self):
        if self._checked_items:
            self.segmentation_requested.emit(list(self._checked_items))

    def _on_slice_slider_changed(self, value):
        ratio = value / 1000.0
        self._current_slice_ratio = ratio
        self.lbl_slice_sync.setText(f"Corte sincronizado: {int(ratio * 100)}%")
        self.slice_ratio_changed.emit(ratio)

    def _on_ct_preset_changed(self, idx):
        data = self.combo_ct_presets.itemData(idx)
        if data:
            c, w = data
            self.slider_center.blockSignals(True)
            self.slider_width.blockSignals(True)
            self.slider_center.setValue(int(c))
            self.slider_width.setValue(int(w))
            self.lbl_center.setText(f"C: {int(c)}")
            self.lbl_width.setText(f"W: {int(w)}")
            self._ct_window_center = float(c)
            self._ct_window_width = float(w)
            self.slider_center.blockSignals(False)
            self.slider_width.blockSignals(False)
            self.ct_window_changed.emit(self._ct_window_center, self._ct_window_width)

    def _on_cw_changed(self):
        c = float(self.slider_center.value())
        w = float(self.slider_width.value())
        self.lbl_center.setText(f"C: {int(c)}")
        self.lbl_width.setText(f"W: {int(w)}")
        self._ct_window_center = c
        self._ct_window_width = w
        self.ct_window_changed.emit(c, w)

    def _on_opacity_slider_changed(self):
        ct_val = self.slider_ct_op.value()
        pet_val = self.slider_pet_op.value()
        self.lbl_ct_op.setText(f"Opacidad CT: {ct_val}%")
        self.lbl_pet_op.setText(f"Opacidad PET: {pet_val}%")
        self.opacity_changed.emit(ct_val / 100.0, pet_val / 100.0)

    def _on_pet_suv_slider_changed(self, value):
        suv_val = max(0.01, value / 20.0)
        self._current_pet_suv_max = suv_val
        self.lbl_pet_suv_max.setText(f"SUV Máx: {suv_val:.2f}")
        self.pet_suv_changed.emit(suv_val)

    def set_pet_suv_range(self, max_suv):
        val = float(max_suv) if max_suv and float(max_suv) > 0 else 1.0
        self._reference_global_max_suv = val
        self._current_pet_suv_max = val

        slider_max = max(int(round(val * 2.5 * 20.0)), 200)
        slider_val = int(round(val * 20.0))

        self.slider_pet_suv_max.blockSignals(True)
        self.slider_pet_suv_max.setRange(1, slider_max)
        self.slider_pet_suv_max.setValue(slider_val)
        self.slider_pet_suv_max.blockSignals(False)
        self.lbl_pet_suv_max.setText(f"SUV Máx: {val:.2f}")

    def update_pixel_readout(self, info):
        if not info:
            suv_str = f" | SUVmáx: {self._current_pet_suv_max:.2f}" if self._active_modality in ("PET", "PT", "FUSION") else ""
            self.lbl_pixel_info.setText(f"Pos: (---, ---) | Valor: --- | W: {int(self._ct_window_width)} C: {int(self._ct_window_center)}{suv_str}")
            return
        x = info.get("x", 0)
        y = info.get("y", 0)
        hu_val = info.get("val")
        pet_val = info.get("pet_val")
        mod = str(info.get("modality", "")).upper()
        pat = info.get("patient_name", "")

        parts = [f"Pos: ({x}, {y})"]
        if mod == "FUSION":
            if hu_val is not None:
                parts.append(f"{hu_val:.1f} HU (CT)")
            if pet_val is not None:
                parts.append(f"{pet_val:.2f} SUV (PET)")
        elif mod in ("PT", "PET"):
            if hu_val is not None:
                parts.append(f"{hu_val:.2f} SUV")
        elif mod in ("MR", "MRI"):
            if hu_val is not None:
                parts.append(f"{hu_val:.1f}")
        else:  # CT
            if hu_val is not None:
                parts.append(f"{hu_val:.1f} HU")

        if mod in ("PT", "PET"):
            if hasattr(self, "_current_pet_suv_max") and self._current_pet_suv_max:
                parts.append(f"SUVmáx: {self._current_pet_suv_max:.2f}")
        elif mod == "FUSION":
            parts.append(f"W: {int(self._ct_window_width)} C: {int(self._ct_window_center)}")
            if hasattr(self, "_current_pet_suv_max") and self._current_pet_suv_max:
                parts.append(f"SUVmáx: {self._current_pet_suv_max:.2f}")
        else:
            parts.append(f"W: {int(self._ct_window_width)} C: {int(self._ct_window_center)}")

        if pat:
            parts.append(f"[{pat}]")
        self.lbl_pixel_info.setText(" | ".join(parts))

    def set_pagination_info(self, current_page, total_pages):
        self._current_page = current_page
        self._total_pages = max(1, total_pages)
        self.lbl_pagination.setText(f"Pág. {self._current_page + 1} / {self._total_pages}")
        self.btn_pag_prev.setEnabled(self._current_page > 0)
        self.btn_pag_next.setEnabled(self._current_page < self._total_pages - 1)

    def set_study_profiles(self, profiles, alignment_result=None, study_info=None):
        self._profiles_data = dict(profiles or {})
        self._profiles_alignment_result = alignment_result
        self._profiles_study_info = list(study_info or [])
        self.update_profile_plot()

    def clear_profile_plot(self):
        self._profiles_data = {}
        self._profiles_alignment_result = None
        self._profiles_study_info = []
        if hasattr(self, "fig_profile"):
            self.fig_profile.clear()
            ax = self.fig_profile.add_subplot(111)
            ax.set_facecolor("#000000")
            for spine in ax.spines.values():
                spine.set_color("#2d3748")
            ax.tick_params(
                axis="both", which="both",
                left=True, bottom=True, top=False, right=False,
                colors="#2d3748", length=2, width=0.8,
                labelleft=False, labelbottom=False,
                labeltop=False, labelright=False,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            self.fig_profile.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.06)
            self.canvas_profile.draw_idle()

    def update_profile_plot(self):
        if not hasattr(self, "fig_profile"):
            return
        self.fig_profile.clear()
        if not self._profiles_data:
            ax = self.fig_profile.add_subplot(111)
            ax.set_facecolor("#000000")
            for spine in ax.spines.values():
                spine.set_color("#2d3748")
            ax.tick_params(
                axis="both", which="both",
                left=True, bottom=True, top=False, right=False,
                colors="#2d3748", length=2, width=0.8,
                labelleft=False, labelbottom=False,
                labeltop=False, labelright=False,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            self.fig_profile.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.06)
            self.canvas_profile.draw_idle()
            return

        ax = self.fig_profile.add_subplot(111)
        ax.set_facecolor("#000000")

        is_shifted = self.btn_shift.isChecked()
        res = self._profiles_alignment_result

        # Ejes con marco visible pero sin numeración ni etiquetas
        spine_color = "#00e5ff" if (is_shifted and res and res.common_length > 0) else "#4a5568"
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(1.0 if is_shifted else 0.8)

        ax.tick_params(
            axis="both",
            which="both",
            left=True,
            bottom=True,
            top=False,
            right=False,
            colors=spine_color,
            length=2,
            width=0.8,
            labelleft=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Paleta de colores brillantes / neón para las curvas
        bright_colors = [
            "#00f0ff",
            "#ffff00",
            "#00ff66",
            "#ff00aa",
            "#ff7700",
            "#9966ff",
            "#33ccff",
            "#ff3366",
        ]

        all_lens = []
        for idx, (study_i, prof) in enumerate(sorted(self._profiles_data.items())):
            if len(prof) == 0:
                continue
            all_lens.append(len(prof))
            color = bright_colors[idx % len(bright_colors)]

            if is_shifted and res and study_i in res.shifts:
                s = res.shifts[study_i]
                x_coords = np.arange(len(prof)) - s
            else:
                x_coords = np.arange(len(prof))

            ax.plot(x_coords, prof, color=color, linewidth=0.9)

        # Rango horizontal estable para que el observador perciba claramente el movimiento/alineación
        if all_lens:
            max_len = max(all_lens)
            all_shifts = list(res.shifts.values()) if (res and res.shifts) else [0]
            min_s = min(all_shifts)
            max_s = max(all_shifts)
            x_min = min(0, -max_s) - 2
            x_max = max(max_len - 1, max_len - 1 - min_s) + 2
            ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.02, 1.05)

        # Ajuste de márgenes relativo para adaptación responsive al redimensionar ventana
        self.fig_profile.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.06)
        self.canvas_profile.draw_idle()

    def enable_viz_controls(self, enabled=True):
        self.viz_controls_container.setVisible(enabled)
        self.btn_shift.setEnabled(enabled)
        if enabled:
            has_pet = self._active_modality in ("PET", "PT", "FUSION")
            self.pet_suv_widget.setVisible(has_pet)
        else:
            self.btn_shift.blockSignals(True)
            self.btn_shift.setChecked(False)
            self.btn_shift.blockSignals(False)
            self.clear_profile_plot()
            self.pet_suv_widget.setVisible(False)

    def update_selection(self, checked_items):
        self._checked_items = list(checked_items or [])
        self.lst_viz_items.clear()

        if not self._checked_items:
            self._active_modality = None
            self.viz_section.set_title("Visualización")
            self.lbl_viz_info.setText("Seleccione una o más series o pares en el árbol para visualización múltiple.")
            self.btn_load_volumes.setEnabled(False)
            self.btn_shift.setEnabled(False)
            self.btn_shift.blockSignals(True)
            self.btn_shift.setChecked(False)
            self.btn_shift.blockSignals(False)
            self.clear_profile_plot()
            self.viz_section.set_section_enabled(False)
            self.viz_controls_container.setVisible(False)
            self.pet_suv_widget.setVisible(False)

            self.lbl_uniformity_info.setText("Análisis de uniformidad sobre los estudios seleccionados.")
            self.btn_run_uniformity.setEnabled(False)
            self.uniformity_section.set_section_enabled(False)

            self.lbl_spatial_info.setText("Análisis de resolución espacial (FWHM) para series PET seleccionadas.")
            self.btn_run_spatial.setEnabled(False)
            self.spatial_section.set_section_enabled(False)

            self.lbl_seg_info.setText("Segmentación anatómica en lote para los estudios seleccionados.")
            self.btn_run_seg.setEnabled(False)
            self.seg_section.set_section_enabled(False)
            return

        first = self._checked_items[0]
        mod = str(first.get("modality") or "").upper()
        node_type = first.get("type")
        if node_type == NODE_TYPE_FUSION_PAIR or mod == "FUSION":
            modality_key = "FUSION"
            section_title = "Visualización Fusión"
            mod_desc = "pares fusionables"
        elif mod in ("PT", "PET"):
            modality_key = "PET"
            section_title = "Visualización PET"
            mod_desc = "series PET"
        elif mod in ("MR", "MRI"):
            modality_key = "MRI"
            section_title = "Visualización MRI"
            mod_desc = "series MRI"
        elif mod == "CT":
            modality_key = "CT"
            section_title = "Visualización CT"
            mod_desc = "series CT"
        else:
            modality_key = mod
            section_title = f"Visualización {mod}"
            mod_desc = f"series {mod}"

        self._active_modality = modality_key
        self.viz_section.set_title(section_title)

        n = len(self._checked_items)
        self.lbl_viz_info.setText(f"{n} {mod_desc} seleccionada{'s' if n > 1 else ''} para análisis:")
        for item_data in self._checked_items:
            pat = item_data.get("patient_name") or item_data.get("patient_id") or "Paciente"
            lbl = item_data.get("label") or item_data.get("series_description") or "Elemento"
            item_text = f"[{pat}] {lbl}"
            list_item = QListWidgetItem(item_text)
            if modality_key == "CT":
                list_item.setIcon(get_ct_icon(False))
            elif modality_key == "PET":
                list_item.setIcon(get_pet_icon(False))
            elif modality_key == "MRI":
                list_item.setIcon(get_mri_icon(False))
            elif modality_key == "FUSION":
                list_item.setIcon(get_fusion_icon(False))
            self.lst_viz_items.addItem(list_item)

        # Visibilidad de controles según modalidad
        if modality_key == "PET":
            self.ct_window_widget.setVisible(False)
            self.pet_suv_widget.setVisible(True)
            self.ct_op_row_widget.setVisible(False)
            self.pet_op_row_widget.setVisible(True)
        elif modality_key in ("CT", "MRI"):
            self.ct_window_widget.setVisible(True)
            self.pet_suv_widget.setVisible(False)
            self.ct_op_row_widget.setVisible(True)
            self.pet_op_row_widget.setVisible(False)
        else:  # FUSION
            self.ct_window_widget.setVisible(True)
            self.pet_suv_widget.setVisible(True)
            self.ct_op_row_widget.setVisible(True)
            self.pet_op_row_widget.setVisible(True)

        # Visualización: siempre habilitada cuando hay selección multi-estudio
        self.viz_section.set_section_enabled(True)
        self.btn_load_volumes.setEnabled(True)

        # Uniformidad
        # Regla: [multiestudio] CT, MRI o Fusión: habilitada; PET: deshabilitada
        if modality_key in ("CT", "MRI", "FUSION"):
            self.lbl_uniformity_info.setText(f"{n} {mod_desc} disponible{'s' if n > 1 else ''} para análisis de uniformidad.")
            self.uniformity_section.set_section_enabled(True)
            self.btn_run_uniformity.setEnabled(True)
        else:
            self.lbl_uniformity_info.setText("El análisis de uniformidad no está disponible para series PET.")
            self.uniformity_section.set_section_enabled(False)
            self.btn_run_uniformity.setEnabled(False)

        # Resolución espacial (FWHM)
        # Regla: [multiestudio] PET: habilitada; CT, MRI o Fusión: deshabilitada
        if modality_key == "PET":
            self.lbl_spatial_info.setText(f"{n} {mod_desc} disponible{'s' if n > 1 else ''} para resolución espacial.")
            self.spatial_section.set_section_enabled(True)
            self.btn_run_spatial.setEnabled(True)
        else:
            self.lbl_spatial_info.setText("La resolución espacial (FWHM) solo está disponible para series PET.")
            self.spatial_section.set_section_enabled(False)
            self.btn_run_spatial.setEnabled(False)

        # Segmentación
        # Regla: [multiestudio] CT, MRI o Fusión: habilitada; PET: deshabilitada
        if modality_key in ("CT", "MRI", "FUSION"):
            self.lbl_seg_info.setText(f"{n} {mod_desc} disponible{'s' if n > 1 else ''} para segmentación anatómica.")
            self.seg_section.set_section_enabled(True)
            self.btn_run_seg.setEnabled(True)
        else:
            self.lbl_seg_info.setText("La segmentación anatómica no está disponible para series PET.")
            self.seg_section.set_section_enabled(False)
            self.btn_run_seg.setEnabled(False)


class ToolsPanel(QWidget):
    segment_requested = Signal(dict)
    view_3d_requested = Signal(str, str)
    spatial_analysis_completed = Signal(object, object)
    uniformity_analysis_completed = Signal(object, object)
    export_3d_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)

        self._current_node_data = None
        self._current_series_uid = ""
        self._current_study_uid = ""
        self._current_modality = ""
        self._current_pair_key = ""
        self._current_seg_context = ""
        self._current_target_id = ""
        self._config_path = RECENT_FOLDERS_CONFIG_PATH
        self._larmornium_files_dir = LARMORNIUM_FILES_DIR
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Área de scroll principal para que las secciones se desplacen fluidamente
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._layout.addWidget(scroll)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.setSpacing(8)

        # Sección : Segmentación (desplegable verticalmente)
        self.seg_section = CollapsibleSection("Segmentación", container)
        container_layout.addWidget(self.seg_section)
        self.seg_section.set_expanded(False)
        self.seg_section.set_section_enabled(False)

        c_layout = self.seg_section.content_layout

        self.target_label = QLabel("Seleccione una serie CT o MRI para segmentar.")
        self.target_label.setWordWrap(True)
        c_layout.addWidget(self.target_label)

        c_layout.addWidget(QLabel("Estructuras disponibles:"))

        self.organs_list = QListWidget()
        self.organs_list.setIconSize(QSize(20, 20))
        self.organs_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.organs_list.setMinimumHeight(140)
        self.organs_list.setMaximumHeight(220)
        self.organs_list.currentItemChanged.connect(self._on_organ_item_changed)
        self.organs_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        c_layout.addWidget(self.organs_list)

        opt_layout = QHBoxLayout()
        opt_layout.setContentsMargins(0, 0, 0, 0)
        opt_layout.setSpacing(8)

        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:
            pass

        self.cb_cuda = QCheckBox("GPU (CUDA)")
        self.cb_cuda.setChecked(has_cuda)
        self.cb_cuda.setEnabled(has_cuda)
        self.cb_cuda.setToolTip("Usar GPU (CUDA) para la inferencia de la red neuronal" if has_cuda else "GPU (CUDA) no detectada")
        opt_layout.addWidget(self.cb_cuda)

        self.cb_fast = QCheckBox("Rápido (3mm)")
        self.cb_fast.setChecked(False)
        self.cb_fast.setToolTip("Ejecuta TotalSegmentator en resolución reducida de 3mm (más rápido)")
        opt_layout.addWidget(self.cb_fast)
        c_layout.addLayout(opt_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)

        self.btn_segment = QPushButton("Segmentar")
        self.btn_segment.setEnabled(False)
        self.btn_segment.clicked.connect(self._on_segment_clicked)
        btn_layout.addWidget(self.btn_segment)

        self.btn_view_3d = QPushButton("Visualizar 3D")
        self.btn_view_3d.setEnabled(False)
        self.btn_view_3d.clicked.connect(self._on_view_3d_clicked)
        btn_layout.addWidget(self.btn_view_3d)

        c_layout.addLayout(btn_layout)

        self.seg_info_label = QLabel("")
        self.seg_info_label.setWordWrap(True)
        c_layout.addWidget(self.seg_info_label)

        # Seccion : Visor de Segmentacion (habilitada si existen segmentaciones generadas)
        self.seg_viewer_section = CollapsibleSection("Visor de Segmentación", container)
        container_layout.addWidget(self.seg_viewer_section)
        self.seg_viewer_section.set_expanded(True)
        self.seg_viewer_section.set_section_enabled(False)

        sv_layout = self.seg_viewer_section.content_layout

        self.seg_viewer_desc_label = QLabel("Segmentaciones generadas:")
        sv_layout.addWidget(self.seg_viewer_desc_label)

        self.seg_viewer_list = QListWidget()
        self.seg_viewer_list.setIconSize(QSize(20, 20))
        self.seg_viewer_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.seg_viewer_list.setMinimumHeight(120)
        self.seg_viewer_list.setMaximumHeight(200)
        self.seg_viewer_list.currentItemChanged.connect(self._on_seg_viewer_item_changed)
        self.seg_viewer_list.itemClicked.connect(self._on_seg_viewer_item_clicked)
        sv_layout.addWidget(self.seg_viewer_list)

        sv_btn_layout = QHBoxLayout()
        sv_btn_layout.setContentsMargins(0, 0, 0, 0)
        sv_btn_layout.setSpacing(6)

        self.btn_seg_viewer_view = QPushButton("Visualizar")
        self.btn_seg_viewer_view.setEnabled(False)
        self.btn_seg_viewer_view.clicked.connect(self._on_seg_viewer_view_clicked)
        sv_btn_layout.addWidget(self.btn_seg_viewer_view)
        sv_layout.addLayout(sv_btn_layout)

        self.seg_viewer_info_label = QLabel("")
        self.seg_viewer_info_label.setWordWrap(True)
        sv_layout.addWidget(self.seg_viewer_info_label)

        
        self.seg_viewer_table = QTableWidget(0, 2)
        self.seg_viewer_table.setHorizontalHeaderLabels(["Parámetro", "Valor"])
        self.seg_viewer_table.verticalHeader().setVisible(False)
        self.seg_viewer_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.seg_viewer_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.seg_viewer_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.seg_viewer_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.seg_viewer_table.setMinimumHeight(110)
        self.seg_viewer_table.setMaximumHeight(150)
        self.seg_viewer_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.seg_viewer_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.seg_viewer_table.horizontalHeader().setStretchLastSection(False)
        self.seg_viewer_table.setStyleSheet(
            "QTableWidget {"
            "  background-color: #14171f;"
            "  gridline-color: #2b303c;"
            "  border: 1px solid #2b303c;"
            "  font-size: 11px;"
            "}"
            "QHeaderView::section {"
            "  background-color: #1c202a;"
            "  color: #c5cddb;"
            "  padding: 3px 6px;"
            "  font-weight: bold;"
            "  border: 1px solid #2b303c;"
            "}"
        )
        sv_layout.addWidget(self.seg_viewer_table)

        # Controles para exportacion de segmentacion a formato 3D (impresion 3D)
        export_sep = QFrame()
        export_sep.setFrameShape(QFrame.HLine)
        export_sep.setFrameShadow(QFrame.Sunken)
        sv_layout.addWidget(export_sep)

        self.lbl_export_title = QLabel("Exportar para impresión 3D:")
        self.lbl_export_title.setStyleSheet("font-weight: bold; color: #c5cddb; margin-top: 2px;")
        sv_layout.addWidget(self.lbl_export_title)

        export_grid = QGridLayout()
        export_grid.setContentsMargins(0, 0, 0, 0)
        export_grid.setHorizontalSpacing(6)
        export_grid.setVerticalSpacing(4)

        # Formato de salida (STL, OBJ, GLB)
        export_grid.addWidget(QLabel("Formato:"), 0, 0)
        self.cmb_export_format = QComboBox()
        self.cmb_export_format.addItem("STL (.stl)", "stl")
        self.cmb_export_format.addItem("OBJ (.obj)", "obj")
        self.cmb_export_format.addItem("glTF (.glb)", "glb")
        self.cmb_export_format.currentIndexChanged.connect(self._on_export_format_changed)
        export_grid.addWidget(self.cmb_export_format, 0, 1)

        # Factor de escala (1.0=original, 2.0=doble, 0.5=mitad)
        export_grid.addWidget(QLabel("Escala:"), 1, 0)
        self.spn_export_scale = QDoubleSpinBox()
        self.spn_export_scale.setRange(0.05, 20.0)
        self.spn_export_scale.setSingleStep(0.1)
        self.spn_export_scale.setValue(1.0)
        self.spn_export_scale.setDecimals(2)
        self.spn_export_scale.setSuffix("x")
        export_grid.addWidget(self.spn_export_scale, 1, 1)

        # Suavizado gaussiano previo a marching cubes
        export_grid.addWidget(QLabel("Suavizado (σ):"), 2, 0)
        self.spn_export_sigma = QDoubleSpinBox()
        self.spn_export_sigma.setRange(0.0, 5.0)
        self.spn_export_sigma.setSingleStep(0.1)
        self.spn_export_sigma.setValue(0.5)
        self.spn_export_sigma.setDecimals(1)
        export_grid.addWidget(self.spn_export_sigma, 2, 1)

        # Calidad de la malla (0.0=minima/maxima reduccion, 1.0=original, default 0.3)
        export_grid.addWidget(QLabel("Calidad:"), 3, 0)
        self.spn_export_quality = QDoubleSpinBox()
        self.spn_export_quality.setRange(0.0, 1.0)
        self.spn_export_quality.setSingleStep(0.05)
        self.spn_export_quality.setValue(0.3)
        self.spn_export_quality.setDecimals(2)
        self.spn_export_quality.setToolTip("Calidad de la malla: 0.0 (mínima calidad / máxima reducción de triángulos) a 1.0 (calidad original)")
        export_grid.addWidget(self.spn_export_quality, 3, 1)

        # Opcion dinamica segun formato seleccionado
        self.lbl_export_dynamic = QLabel("Modo STL:")
        export_grid.addWidget(self.lbl_export_dynamic, 4, 0)

        self.chk_export_ascii = QCheckBox("Formato ASCII")
        self.chk_export_ascii.setChecked(False)
        self.chk_export_ascii.setToolTip("Exportar STL en formato ASCII (desmarcado: binario)")
        export_grid.addWidget(self.chk_export_ascii, 4, 1)

        self.chk_export_mtl = QCheckBox("Generar .mtl")
        self.chk_export_mtl.setChecked(False)
        self.chk_export_mtl.setToolTip("Generar archivo de material .mtl acompanante")
        self.chk_export_mtl.setVisible(False)
        export_grid.addWidget(self.chk_export_mtl, 4, 1)

        sv_layout.addLayout(export_grid)

        self.btn_export_3d = QPushButton("Exportar 3D")
        self.btn_export_3d.setEnabled(False)
        self.btn_export_3d.clicked.connect(self._on_export_3d_clicked)
        sv_layout.addWidget(self.btn_export_3d)

        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setWordWrap(True)
        self.lbl_export_status.setStyleSheet("font-size: 11px;")
        sv_layout.addWidget(self.lbl_export_status)

        self.spatial_section = CollapsibleSection("Análisis de resolución espacial", container)
        container_layout.addWidget(self.spatial_section)

        self.spatial_res_widget = SpatialResolutionAnalysisWidget(self.spatial_section)
        self.spatial_res_widget.analysis_completed.connect(self.spatial_analysis_completed.emit)
        self.spatial_section.content_layout.addWidget(self.spatial_res_widget)
        self.spatial_section.set_expanded(False)
        self.spatial_section.set_section_enabled(False)

        self.uniformity_section = CollapsibleSection("Análisis de uniformidad", container)
        container_layout.addWidget(self.uniformity_section)

        self.uniformity_widget = UniformityAnalysisWidget(self.uniformity_section)
        self.uniformity_widget.analysis_completed.connect(self.uniformity_analysis_completed.emit)
        self.uniformity_section.content_layout.addWidget(self.uniformity_widget)
        self.uniformity_section.set_expanded(False)
        self.uniformity_section.set_section_enabled(False)

        self.multi_study_section = CollapsibleSection("Análisis de estudios múltiples", container)
        container_layout.addWidget(self.multi_study_section)

        self.multi_study_widget = MultiStudyAnalysisPanel(self.multi_study_section)
        self.multi_study_section.content_layout.addWidget(self.multi_study_widget)
        self.multi_study_section.setEnabled(True)
        self.multi_study_section.set_expanded(True)
        self.multi_study_section.set_status_icon(False)

        container_layout.addStretch()
        scroll.setWidget(container)

        self._dicom_root = None
        self.refresh_segmentation_viewer()

    def sizeHint(self):
        return QSize(310, 800)

    def minimumSizeHint(self):
        return QSize(50, 50)

    def set_multi_study_active(self, active: bool, checked_items: list = None):
        if hasattr(self, "multi_study_section"):
            self.multi_study_section.set_status_icon(bool(active))
        if hasattr(self, "multi_study_widget"):
            self.multi_study_widget.update_selection(checked_items if active else [])

    def update_target(self, node_data, dicom_root, larmornium_files_dir, config_path):
        self._dicom_root = dicom_root
        self._larmornium_files_dir = larmornium_files_dir or LARMORNIUM_FILES_DIR
        self._config_path = config_path or RECENT_FOLDERS_CONFIG_PATH
        self.refresh_segmentation_viewer(dicom_root, self._config_path)

        node_type = node_data.get("type") if node_data else None
        if not node_data or node_data.get("is_multi_study") or node_type not in (NODE_TYPE_SERIES, NODE_TYPE_FUSION_PAIR):
            self._current_node_data = None
            if hasattr(self, "spatial_res_widget"):
                self.spatial_res_widget.set_active_series(None, dicom_root, larmornium_files_dir, config_path)
            if hasattr(self, "uniformity_widget"):
                self.uniformity_widget.set_active_target(None, dicom_root, larmornium_files_dir, config_path)
            self._disable_segmentation()
            self.spatial_section.set_section_enabled(False)
            self.uniformity_section.set_section_enabled(False)
            return

        self._current_node_data = node_data
        self._larmornium_files_dir = larmornium_files_dir or LARMORNIUM_FILES_DIR
        self._config_path = config_path or RECENT_FOLDERS_CONFIG_PATH

        modality = str(node_data.get("modality") or "").upper()
        prefix = node_data.get("prefix")
        label = node_data.get("label") or node_data.get("series_description") or "Serie"

        series_uid = node_data.get("series_instance_uid") or ""
        study_uid = node_data.get("study_instance_uid") or ""

        if node_type == NODE_TYPE_FUSION_PAIR:
            pair = node_data.get("pair")
            pair_key = node_data.get("pair_key") or (join_pet_ct._pair_key(pair) if pair else "")
            if not pair_key:
                self._disable_segmentation()
                self.spatial_section.set_section_enabled(False)
                self.uniformity_section.set_section_enabled(False)
                return
            series_uid = pair.get("ct_series_instance_uid", "") if pair else ""
            study_uid = pair.get("study_instance_uid", "") if pair else ""
            target_modality = "CT"
            seg_context = "fusion"
            target_id = pair_key
            label = f"Fusión PET/CT ({label})"

            # Regla: Se selecciona un par fusionable, entonces se habilita la sección de segmentación y uniformidad
            self.seg_section.set_section_enabled(True)
            self.uniformity_section.set_section_enabled(True)
            self.spatial_section.set_section_enabled(False)

            if hasattr(self, "uniformity_widget"):
                self.uniformity_widget.set_active_target(node_data, dicom_root, self._larmornium_files_dir, self._config_path)
            if hasattr(self, "spatial_res_widget"):
                self.spatial_res_widget.set_active_series(None, dicom_root, self._larmornium_files_dir, self._config_path)

        elif modality in ("CT",) or (prefix == "pet_ct" and modality == "CT"):
            target_modality = "CT"
            seg_context = "ct"
            target_id = series_uid
            label = f"CT ({label})"

            # Regla: Se selecciona una serie de CT o MRI, entonces se habilita la sección de segmentación y se deshabilitan las demás
            self.seg_section.set_section_enabled(True)
            self.spatial_section.set_section_enabled(False)
            self.uniformity_section.set_section_enabled(False)

            if hasattr(self, "spatial_res_widget"):
                self.spatial_res_widget.set_active_series(None, dicom_root, self._larmornium_files_dir, self._config_path)
            if hasattr(self, "uniformity_widget"):
                self.uniformity_widget.set_active_target(None, dicom_root, self._larmornium_files_dir, self._config_path)

        elif modality in ("MR", "MRI") or prefix == "mri":
            target_modality = "MRI"
            seg_context = "mri"
            if not series_uid and study_uid:
                series_uid = study_uid
            target_id = series_uid
            label = f"MRI ({label})"

            # Regla: Se selecciona una serie de CT o MRI, entonces se habilita la sección de segmentación y se deshabilitan las demás
            self.seg_section.set_section_enabled(True)
            self.spatial_section.set_section_enabled(False)
            self.uniformity_section.set_section_enabled(False)

            if hasattr(self, "spatial_res_widget"):
                self.spatial_res_widget.set_active_series(None, dicom_root, self._larmornium_files_dir, self._config_path)
            if hasattr(self, "uniformity_widget"):
                self.uniformity_widget.set_active_target(None, dicom_root, self._larmornium_files_dir, self._config_path)

        elif modality in ("PT", "PET") or (prefix == "pet_ct" and modality in ("PT", "PET")):
            # Regla: Se selecciona una serie de PET, entonces se habilita la sección de análisis de resolución espacial y se deshabilitan las demás
            self.spatial_section.set_section_enabled(True)
            self.seg_section.set_section_enabled(False)
            self.uniformity_section.set_section_enabled(False)

            if hasattr(self, "spatial_res_widget"):
                self.spatial_res_widget.set_active_series(node_data, dicom_root, self._larmornium_files_dir, self._config_path)
            if hasattr(self, "uniformity_widget"):
                self.uniformity_widget.set_active_target(None, dicom_root, self._larmornium_files_dir, self._config_path)
            self._disable_segmentation()
            return

        else:
            self._disable_segmentation()
            self.spatial_section.set_section_enabled(False)
            self.uniformity_section.set_section_enabled(False)
            return

        if not target_id:
            self._disable_segmentation()
            return

        self._current_series_uid = series_uid
        self._current_study_uid = study_uid
        self._current_modality = target_modality
        self._current_pair_key = pair_key if seg_context == "fusion" else ""
        self._current_seg_context = seg_context
        self._current_target_id = target_id

        self.target_label.setText(f"Serie activa [{target_modality}]: {label}")
        self._populate_organs(target_modality, target_id, seg_context=seg_context)

    def _disable_segmentation(self):
        self._current_series_uid = ""
        self._current_study_uid = ""
        self._current_modality = ""
        self._current_pair_key = ""
        self._current_seg_context = ""
        self._current_target_id = ""
        self.target_label.setText("Seleccione una serie CT o MRI para segmentar.")
        self.organs_list.clear()
        self.btn_segment.setEnabled(False)
        self.btn_view_3d.setEnabled(False)
        self.seg_info_label.setText("")
        self.seg_section.set_section_enabled(False)
        if hasattr(self, "uniformity_widget"):
            self.uniformity_widget.set_active_target(None, None, None, None)

    def _populate_organs(self, modality, target_id, seg_context=""):
        self.organs_list.clear()
        import segmentation_anato_ct_TotalSegmentator as seg_ct
        import segmentation_anato_mri_TotalSegmentator as seg_mri

        registry = seg_ct.ORGAN_REGISTRY if modality == "CT" else seg_mri.ORGAN_REGISTRY

        for organ_key, info in registry.items():
            disp_name = info["display_name"]
            status = get_segmentation_status(self._config_path, seg_context, target_id, organ_key)
            item = QListWidgetItem(get_seg_icon(status), disp_name)
            item.setData(Qt.UserRole, organ_key)
            item.setData(Qt.UserRole + 1, info)
            item.setData(Qt.UserRole + 2, status)
            self.organs_list.addItem(item)

        if self.organs_list.count() > 0:
            self.organs_list.setCurrentRow(0)

    def refresh_organ_status(self, organ_key, status="built"):
        if status is True:
            status = "built"
        elif status is False:
            status = "none"
        for i in range(self.organs_list.count()):
            item = self.organs_list.item(i)
            if item.data(Qt.UserRole) == organ_key:
                item.setIcon(get_seg_icon(status))
                item.setData(Qt.UserRole + 2, status)
                if item == self.organs_list.currentItem():
                    self._on_organ_item_changed(item, None)
                break
        self.refresh_segmentation_viewer()

    def _on_organ_item_changed(self, current, previous):
        if not current:
            self.btn_segment.setEnabled(False)
            self.btn_view_3d.setEnabled(False)
            self.seg_info_label.setText("")
            return

        self.btn_segment.setEnabled(True)
        organ_key = current.data(Qt.UserRole)
        info = current.data(Qt.UserRole + 1) or {}
        desc = info.get("description", "")

        status = get_segmentation_status(self._config_path, self._current_seg_context, self._current_target_id, organ_key)
        is_built = (status == "built")
        self.btn_view_3d.setEnabled(is_built)

        if status == "built":
            built = _load_built_segmentations(self._config_path, self._current_seg_context)
            rec = built.get(f"{self._current_target_id}_{organ_key}")
            if not rec and self._config_path != RECENT_FOLDERS_CONFIG_PATH:
                recent_built = _load_built_segmentations(RECENT_FOLDERS_CONFIG_PATH, self._current_seg_context)
                rec = recent_built.get(f"{self._current_target_id}_{organ_key}")
            rec = rec or {}
            stats = rec.get("stats", {})
            vol = stats.get("volume_cm3")
            vol_str = f" | Volumen: {vol:.2f} cm3" if vol is not None else ""
            self.seg_info_label.setText(f"{desc}\n[Estado: Segmentado{vol_str}]")
        elif status == "warning":
            self.seg_info_label.setText(f"{desc}\n[Estado: No se generó la segmentación (0 píxeles detectados). Puede volver a intentarlo]")
        else:
            self.seg_info_label.setText(f"{desc}\n[Estado: No segmentado]")

    def _on_item_double_clicked(self, item):
        if not item:
            return
        status = item.data(Qt.UserRole + 2)
        if status == "built" or status is True:
            self._on_view_3d_clicked()
        else:
            self._on_segment_clicked()

    def _on_segment_clicked(self):
        item = self.organs_list.currentItem()
        if not item or not self._current_target_id:
            return
        organ_key = item.data(Qt.UserRole)
        info = item.data(Qt.UserRole + 1) or {}

        payload = {
            "seg_context": self._current_seg_context,
            "target_id": self._current_target_id,
            "series_uid": self._current_series_uid,
            "study_uid": self._current_study_uid,
            "modality": self._current_modality,
            "pair_key": self._current_pair_key,
            "organ": organ_key,
            "organ_display": info.get("display_name", organ_key),
            "cuda": self.cb_cuda.isChecked(),
            "fast": self.cb_fast.isChecked(),
            "node_data": self._current_node_data,
        }
        self.segment_requested.emit(payload)

    def _on_view_3d_clicked(self):
        item = self.organs_list.currentItem()
        if not item or not self._current_target_id:
            return
        organ_key = item.data(Qt.UserRole)
        built = _load_built_segmentations(self._config_path, self._current_seg_context)
        rec = built.get(f"{self._current_target_id}_{organ_key}")
        if not rec and self._config_path != RECENT_FOLDERS_CONFIG_PATH:
            recent_built = _load_built_segmentations(RECENT_FOLDERS_CONFIG_PATH, self._current_seg_context)
            rec = recent_built.get(f"{self._current_target_id}_{organ_key}")

        if rec and rec.get("nii_path") and os.path.isfile(rec["nii_path"]):
            info = item.data(Qt.UserRole + 1) or {}
            title = f"{info.get('display_name', organ_key)} ({self._current_modality})"
            self.view_3d_requested.emit(rec["nii_path"], title)

    def _get_all_existing_segmentations(self):
        # Recopila segmentaciones registradas tanto en la configuracion global como en la del directorio
        configs_to_check = []
        if self._config_path and os.path.isfile(self._config_path):
            configs_to_check.append(self._config_path)
        if RECENT_FOLDERS_CONFIG_PATH and os.path.isfile(RECENT_FOLDERS_CONFIG_PATH) and RECENT_FOLDERS_CONFIG_PATH not in configs_to_check:
            configs_to_check.append(RECENT_FOLDERS_CONFIG_PATH)
        if hasattr(self, "_dicom_root") and self._dicom_root:
            dir_conf = os.path.join(get_directory_files_dir(self._dicom_root), RECENT_FOLDERS_CONFIG_FILENAME)
            if os.path.isfile(dir_conf) and dir_conf not in configs_to_check:
                configs_to_check.append(dir_conf)

        all_segs = {}
        for cfg in configs_to_check:
            loaded = _load_built_segmentations(cfg)
            for k, v in loaded.items():
                if k not in all_segs and isinstance(v, dict):
                    all_segs[k] = v

        valid_segs = {}
        for key, rec in all_segs.items():
            if not isinstance(rec, dict):
                continue
            if rec.get("status") == "warning" or rec.get("generated") is False:
                continue
            if rec.get("stats", {}).get("num_voxels") == 0:
                continue
            nii_path = rec.get("nii_path", "")
            if nii_path and os.path.isfile(nii_path):
                valid_segs[key] = rec

        return valid_segs

    def refresh_segmentation_viewer(self, dicom_root=None, config_path=None):
        if dicom_root is not None:
            self._dicom_root = dicom_root
        if config_path is not None:
            self._config_path = config_path

        valid_segs = self._get_all_existing_segmentations()

        # Guardar la ruta seleccionada actual para intentar restaurarla
        prev_nii = None
        cur_item = self.seg_viewer_list.currentItem() if hasattr(self, "seg_viewer_list") else None
        if cur_item:
            prev_rec = cur_item.data(Qt.UserRole)
            if prev_rec and isinstance(prev_rec, dict):
                prev_nii = prev_rec.get("nii_path")

        if not hasattr(self, "seg_viewer_list") or not hasattr(self, "seg_viewer_section"):
            return

        self.seg_viewer_list.blockSignals(True)
        self.seg_viewer_list.clear()

        if valid_segs:
            self.seg_viewer_section.set_section_enabled(True)
            self.seg_viewer_desc_label.setText(f"Segmentaciones generadas ({len(valid_segs)}):")

            import segmentation_anato_ct_TotalSegmentator as seg_ct
            import segmentation_anato_mri_TotalSegmentator as seg_mri

            restore_item = None
            for key, rec in valid_segs.items():
                organ = rec.get("organ", "")
                modality = rec.get("modality", "CT")

                # Obtener nombre para mostrar
                disp_name = None
                if modality == "CT" and organ in seg_ct.ORGAN_REGISTRY:
                    disp_name = seg_ct.ORGAN_REGISTRY[organ].get("display_name")
                elif modality in ("MR", "MRI") and organ in seg_mri.ORGAN_REGISTRY:
                    disp_name = seg_mri.ORGAN_REGISTRY[organ].get("display_name")

                if not disp_name and rec.get("json_path") and os.path.isfile(rec["json_path"]):
                    try:
                        with open(rec["json_path"], "r", encoding="utf-8") as jf:
                            jdata = json.load(jf)
                        disp_name = jdata.get("organ_display_name")
                    except Exception:
                        pass

                if not disp_name:
                    disp_name = organ.capitalize() if organ else key

                stats = rec.get("stats", {})
                vol_cm3 = stats.get("volume_cm3")
                vol_txt = f" - {vol_cm3:.1f} cm3" if vol_cm3 is not None else ""
                item_text = f"{disp_name} [{modality}]{vol_txt}"

                icon = get_seg_icon("built")
                item = QListWidgetItem(icon, item_text)
                item.setData(Qt.UserRole, rec)
                item.setData(Qt.UserRole + 1, disp_name)
                tip = f"Estructura: {disp_name}\nModalidad: {modality}\nClave: {key}"
                if vol_cm3 is not None:
                    tip += f"\nVolumen: {vol_cm3:.2f} cm3"
                item.setToolTip(tip)
                self.seg_viewer_list.addItem(item)

                if prev_nii and rec.get("nii_path") == prev_nii:
                    restore_item = item

            self.seg_viewer_list.blockSignals(False)

            if restore_item:
                self.seg_viewer_list.setCurrentItem(restore_item)
                self.btn_seg_viewer_view.setEnabled(True)
                if hasattr(self, "btn_export_3d"):
                    self.btn_export_3d.setEnabled(True)
            else:
                self.seg_viewer_list.setCurrentItem(None)
                self.btn_seg_viewer_view.setEnabled(False)
                if hasattr(self, "btn_export_3d"):
                    self.btn_export_3d.setEnabled(False)
                self.seg_viewer_info_label.setText("Seleccione una segmentación para visualizarla en 3D y 2D.")
                if hasattr(self, "seg_viewer_table"):
                    self.seg_viewer_table.setRowCount(0)
        else:
            self.seg_viewer_section.set_section_enabled(False)
            self.seg_viewer_desc_label.setText("No hay segmentaciones generadas.")
            self.seg_viewer_info_label.setText("No existen segmentaciones generadas en larmornium.conf.")
            if hasattr(self, "seg_viewer_table"):
                self.seg_viewer_table.setRowCount(0)
            self.btn_seg_viewer_view.setEnabled(False)
            if hasattr(self, "btn_export_3d"):
                self.btn_export_3d.setEnabled(False)
            self.seg_viewer_list.blockSignals(False)

    def _on_seg_viewer_item_changed(self, current, previous):
        if not current:
            self.btn_seg_viewer_view.setEnabled(False)
            if hasattr(self, "btn_export_3d"):
                self.btn_export_3d.setEnabled(False)
            self.seg_viewer_info_label.setText("")
            if hasattr(self, "seg_viewer_table"):
                self.seg_viewer_table.setRowCount(0)
            return
        self._update_seg_viewer_details(current)
        self._trigger_seg_viewer_visualization(current)

    def _on_seg_viewer_item_clicked(self, item):
        if not item:
            return
        rec = item.data(Qt.UserRole)
        nii_path = rec.get("nii_path", "") if rec else ""
        if nii_path and nii_path == getattr(self, "_last_requested_seg_path", None):
            return
        self._update_seg_viewer_details(item)
        self._trigger_seg_viewer_visualization(item)

    def _on_seg_viewer_view_clicked(self):
        item = self.seg_viewer_list.currentItem()
        if item:
            self._trigger_seg_viewer_visualization(item, force=True)

    def _update_seg_viewer_details(self, item):
        if not item:
            if hasattr(self, "seg_viewer_table"):
                self.seg_viewer_table.setRowCount(0)
            if hasattr(self, "btn_export_3d"):
                self.btn_export_3d.setEnabled(False)
            return
        rec = item.data(Qt.UserRole)
        if not rec or not isinstance(rec, dict):
            if hasattr(self, "seg_viewer_table"):
                self.seg_viewer_table.setRowCount(0)
            if hasattr(self, "btn_export_3d"):
                self.btn_export_3d.setEnabled(False)
            return
        self.btn_seg_viewer_view.setEnabled(True)
        if hasattr(self, "btn_export_3d"):
            self.btn_export_3d.setEnabled(True)
        organ = rec.get("organ", "")
        modality = rec.get("modality", "CT")
        disp_name = item.data(Qt.UserRole + 1) or organ.capitalize()
        stats = rec.get("stats", {})
        vol_cm3 = stats.get("volume_cm3")
        num_vox = stats.get("num_voxels")
        nii_path = rec.get("nii_path", "")

        self.seg_viewer_info_label.setText(f"Datos de {disp_name} [{modality}]:")

        rows = [
            ("Estructura", str(disp_name)),
            ("Modalidad", str(modality)),
        ]
        if vol_cm3 is not None:
            rows.append(("Volumen", f"{vol_cm3:.2f} cm3"))
        if num_vox is not None:
            rows.append(("Voxeles", f"{num_vox:,}"))
        if rec.get("series_instance_uid"):
            rows.append(("Serie UID", str(rec.get("series_instance_uid"))))
        if rec.get("study_instance_uid"):
            rows.append(("Estudio UID", str(rec.get("study_instance_uid"))))
        if rec.get("timestamp"):
            rows.append(("Fecha", str(rec.get("timestamp"))))
        if nii_path:
            rows.append(("Archivo", os.path.basename(nii_path)))
            rows.append(("Ruta NIfTI", str(nii_path)))
        if rec.get("json_path"):
            rows.append(("Archivo JSON", os.path.basename(rec["json_path"])))

        if hasattr(self, "seg_viewer_table"):
            self.seg_viewer_table.setRowCount(len(rows))
            for r, (param, val) in enumerate(rows):
                it_param = QTableWidgetItem(param)
                it_val = QTableWidgetItem(val)
                it_param.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_val.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                it_val.setToolTip(val)
                self.seg_viewer_table.setItem(r, 0, it_param)
                self.seg_viewer_table.setItem(r, 1, it_val)
            self.seg_viewer_table.resizeColumnsToContents()

    def _trigger_seg_viewer_visualization(self, item, force=False):
        if not item:
            return
        rec = item.data(Qt.UserRole)
        if not rec or not isinstance(rec, dict):
            return
        nii_path = rec.get("nii_path", "")
        if not nii_path or not os.path.isfile(nii_path):
            self.seg_viewer_info_label.setText("Archivo NIfTI no encontrado en disco.")
            return

        if not force and nii_path == getattr(self, "_last_requested_seg_path", None):
            return

        self._last_requested_seg_path = nii_path
        self._update_seg_viewer_details(item)

        organ = rec.get("organ", "")
        modality = rec.get("modality", "CT")
        disp_name = item.data(Qt.UserRole + 1) or organ.capitalize()
        title = f"{disp_name} ({modality})"
        self.view_3d_requested.emit(nii_path, title)

    def _on_export_format_changed(self, index):
        # Actualiza las opciones dinamicas mostradas segun el formato 3D elegido
        fmt = self.cmb_export_format.currentData() if hasattr(self, "cmb_export_format") else "stl"
        if fmt == "stl":
            self.lbl_export_dynamic.setText("Modo STL:")
            self.lbl_export_dynamic.setVisible(True)
            self.chk_export_ascii.setVisible(True)
            self.chk_export_mtl.setVisible(False)
        elif fmt == "obj":
            self.lbl_export_dynamic.setText("Opciones OBJ:")
            self.lbl_export_dynamic.setVisible(True)
            self.chk_export_ascii.setVisible(False)
            self.chk_export_mtl.setVisible(True)
        elif fmt == "glb":
            self.lbl_export_dynamic.setVisible(False)
            self.chk_export_ascii.setVisible(False)
            self.chk_export_mtl.setVisible(False)

    def _on_export_3d_clicked(self):
        # Recopila los parametros y solicita la exportacion de la segmentacion seleccionada
        item = self.seg_viewer_list.currentItem() if hasattr(self, "seg_viewer_list") else None
        if not item:
            return
        rec = item.data(Qt.UserRole)
        if not rec or not isinstance(rec, dict):
            return
        nii_path = rec.get("nii_path", "")
        if not nii_path or not os.path.isfile(nii_path):
            self.set_export_status("Error: Archivo NIfTI no encontrado en disco.", success=False)
            return

        fmt = self.cmb_export_format.currentData() or "stl"
        scale = float(self.spn_export_scale.value())
        smooth_sigma = float(self.spn_export_sigma.value())
        quality = float(self.spn_export_quality.value()) if hasattr(self, "spn_export_quality") else 0.3
        organ = rec.get("organ", "")
        disp_name = item.data(Qt.UserRole + 1) or organ.capitalize()

        payload = {
            "nii_path": nii_path,
            "json_path": rec.get("json_path"),
            "format": fmt,
            "scale": scale,
            "smooth_sigma": smooth_sigma,
            "quality": quality,
            "organ": organ,
            "disp_name": disp_name,
            "rec": rec,
            "ascii_stl": self.chk_export_ascii.isChecked() if fmt == "stl" else False,
            "write_mtl_obj": self.chk_export_mtl.isChecked() if fmt == "obj" else False,
        }
        self.export_3d_requested.emit(payload)

    def set_export_in_progress(self, in_progress, message=""):
        # Actualiza la interfaz durante el progreso de exportacion
        if hasattr(self, "btn_export_3d"):
            has_sel = bool(self.seg_viewer_list.currentItem()) if hasattr(self, "seg_viewer_list") else False
            self.btn_export_3d.setEnabled(not in_progress and has_sel)
            if in_progress:
                self.btn_export_3d.setText("Exportando...")
            else:
                self.btn_export_3d.setText("Exportar 3D")
        if hasattr(self, "lbl_export_status") and message:
            self.lbl_export_status.setText(message)
            self.lbl_export_status.setStyleSheet("color: #00d2ff; font-size: 11px;")

    def set_export_status(self, message, success=True):
        # Muestra mensaje de estado de exportacion con color apropiado
        if hasattr(self, "lbl_export_status"):
            self.lbl_export_status.setText(message)
            color = "#51cf66" if success else "#ff6b6b"
            self.lbl_export_status.setStyleSheet(f"color: {color}; font-size: 11px;")


class LoadingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #101217;")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.gif_label.setStyleSheet("background: transparent;")

        if os.path.isfile(ICON_LOADING_PATH):
            self.movie = QMovie(ICON_LOADING_PATH)
            self.movie.setCacheMode(QMovie.CacheAll)
            self.movie.setBackgroundColor(QColor("#101217"))
            self.movie.setScaledSize(QSize(110, 110))
            self.gif_label.setMovie(self.movie)
        else:
            self.movie = None

        layout.addWidget(self.gif_label)

        self.status_label = QLabel("Procesando...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumWidth(400)
        self.status_label.setStyleSheet("color: #00d2ff; font-size: 14px; font-weight: bold; background: transparent;")
        layout.addWidget(self.status_label)

    def start(self, message="Procesando..."):
        self.status_label.setText(message)
        if self.movie:
            if self.movie.state() != QMovie.Running:
                self.movie.start()
            else:
                self.movie.jumpToFrame(0)

    def set_status(self, message):
        self.status_label.setText(message)
        QApplication.processEvents()

    def stop(self):
        if self.movie:
            self.movie.stop()

    def show_loading(self, message="Procesando..."):
        self.start(message)

    def hide_loading(self):
        self.stop()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Larmornium - Visualización PET/CT y MRI")
        self.resize(1400, 900)

        self.dicom_root = None
        self.current_db_path = None
        self.current_json_path = None
        self.data_access = None
        self._index_thread = None
        self._index_worker = None
        self._fusion_thread = None
        self._fusion_worker = None
        self._volume_thread = None
        self._volume_worker = None
        self._mri_thread = None
        self._mri_worker = None
        self._load_thread = None
        self._load_worker = None
        self._seg_thread = None
        self._seg_worker = None
        self._export_3d_thread = None
        self._export_3d_worker = None
        os.makedirs(LARMORNIUM_FILES_DIR, exist_ok=True)
        self.recent_store = RecentFoldersStore(RECENT_FOLDERS_CONFIG_PATH)

        self.left_panel = StudySelectionPanel()
        self.left_panel.setMinimumWidth(320)
        self.left_panel.node_selected.connect(self._on_node_selected)
        self.left_panel.multi_study_selection_changed.connect(self._on_multi_study_selection_changed)
        self.left_panel.index_requested.connect(self._on_index_requested_from_panel)
        self.left_panel.open_directory_dialog_requested.connect(self._on_open_directory_dialog_from_panel)
        self.left_panel.change_directory_requested.connect(self._on_change_directory_from_panel)
        self._current_node_data = None

        self.viewer_2d = ImageViewer()
        self.viewer_2d.frame_changed.connect(self._on_viewer_frame_changed)
        self.viewer_2d.fused_slice_changed.connect(self._on_viewer_fused_slice_changed)
        self.viewer_2d.mri_slice_changed.connect(self._on_viewer_mri_slice_changed)
        self.viewer = self.viewer_2d

        self.viewer_3d = Viewer3DWidget()

        # Visores para modo multi-estudio en cuadrícula (mosaico)
        self.multi_viewer_2d = MultiStudyMosaic2DViewer()
        self.multi_viewer_3d = MultiStudyMosaic3DViewer()
        self._is_multi_study_mode = False

        # Stacks para alternar entre visor individual (0) y mosaico multi-estudio (1)
        self.stack_2d = QStackedWidget()
        self.stack_2d.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.stack_2d.addWidget(self.viewer_2d)
        self.stack_2d.addWidget(self.multi_viewer_2d)

        self.stack_3d = QStackedWidget()
        self.stack_3d.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.stack_3d.addWidget(self.viewer_3d)
        self.stack_3d.addWidget(self.multi_viewer_3d)

        self.viz_tab_widget = QTabWidget()
        self.viz_tab_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.viz_tab_widget.addTab(self.stack_2d, "Visualización 2D")
        self.viz_tab_widget.addTab(self.stack_3d, "Visualización 3D")
        self.viz_tab_widget.currentChanged.connect(self._on_viz_subtab_changed)

        self.segmentation_display = SegmentationDisplayWidget()
        self.fwhm_results_display = FWHMResultsDisplayWidget()

        self.tab_widget = QTabWidget()
        self.tab_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.tab_widget.addTab(self.viz_tab_widget, "Visualización")
        self.tab_widget.addTab(self.segmentation_display, "Visualización Segmentación")
        self.tab_widget.addTab(self.fwhm_results_display, "Resultados de FWHM")
        self.uniformity_display = UniformityResultsDisplayWidget()
        self.tab_widget.addTab(self.uniformity_display, "Resultados de Uniformidad")
        self.tab_widget.currentChanged.connect(self._on_main_tab_changed)

        self.loading_panel = LoadingPanel()

        self.display_stack = QStackedWidget()
        self.display_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.display_stack.addWidget(self.tab_widget)
        self.display_stack.addWidget(self.loading_panel)
        self.display_stack.setCurrentWidget(self.tab_widget)

        self.tools_panel = ToolsPanel()
        self.tools_panel.segment_requested.connect(self._on_segment_requested)
        self.tools_panel.view_3d_requested.connect(self._on_view_3d_requested)
        self.tools_panel.spatial_analysis_completed.connect(self._on_spatial_analysis_completed)
        self.tools_panel.uniformity_analysis_completed.connect(self._on_uniformity_analysis_completed)
        self.tools_panel.export_3d_requested.connect(self._on_export_3d_requested)

        # Conectar señales del panel de análisis multi-estudio
        ms_w = self.tools_panel.multi_study_widget
        ms_w.load_requested.connect(self._on_load_multi_study_requested)
        ms_w.shift_toggled.connect(self._on_multi_study_shift_toggled)
        ms_w.slice_ratio_changed.connect(self.multi_viewer_2d.update_slice_ratio)
        ms_w.ct_window_changed.connect(self.multi_viewer_2d.update_ct_window)
        ms_w.opacity_changed.connect(self._on_multi_study_opacity_changed)
        ms_w.pet_suv_changed.connect(self._on_multi_study_pet_suv_changed)
        ms_w.page_change_requested.connect(self._on_multi_study_page_change_requested)

        ms_w.touchpad_rotate.connect(self.multi_viewer_3d.apply_rotation)
        ms_w.touchpad_pan.connect(self.multi_viewer_3d.apply_pan)
        ms_w.touchpad_zoom.connect(self.multi_viewer_3d.apply_zoom)
        ms_w.touchpad_preset.connect(self.multi_viewer_3d.apply_preset)
        ms_w.touchpad_reset.connect(self.multi_viewer_3d.reset_cameras)

        self.multi_viewer_2d.page_changed.connect(self._on_multi_study_page_changed_from_2d)
        self.multi_viewer_2d.pixel_hovered.connect(ms_w.update_pixel_readout)
        self.multi_viewer_2d.pixel_left.connect(lambda: ms_w.update_pixel_readout(None))

        self.multi_viewer_3d.page_changed.connect(self._on_multi_study_page_changed_from_3d)

        self.tools_dock = QDockWidget("Herramientas de análisis", self)
        self.tools_dock.setWidget(self.tools_panel)
        self.tools_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tools_dock)
        self.tools_dock.setVisible(True)
        self.resizeDocks([self.tools_dock], [315], Qt.Horizontal)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.display_stack)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([380, 1020])
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self._build_menu()

    def ensure_tools_dock_expanded(self, target_width=315):
        if hasattr(self, "tools_dock") and self.tools_dock is not None:
            cur_w = self.tools_dock.width()
            desired_w = max(cur_w, target_width)
            self.resizeDocks([self.tools_dock], [desired_w], Qt.Horizontal)
            if hasattr(self, "tools_panel") and self.tools_panel is not None:
                self.tools_panel.updateGeometry()
            self.tools_dock.updateGeometry()

    def show_loading(self, message="Procesando..."):
        self.loading_panel.start(message)
        self.display_stack.setCurrentWidget(self.loading_panel)
        QApplication.processEvents()

    def set_loading_status(self, message):
        self.loading_panel.set_status(message)

    def hide_loading(self):
        self.loading_panel.stop()
        self.display_stack.setCurrentWidget(self.tab_widget)
        self.ensure_tools_dock_expanded(315)

    def _start_async_volume_load(self, load_fn, kind, tag, callback, message, *args, **kwargs):
        if self._is_thread_running(self._load_thread):
            try:
                self._load_thread.quit()
                self._load_thread.wait(500)
            except Exception:
                pass

        self._load_callback = callback
        self.show_loading(message)
        self._load_thread = QThread(self)
        self._load_worker = LoadVolumeWorker(load_fn, kind, tag, *args, **kwargs)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_volume_finished)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._load_thread.deleteLater)
        self._load_thread.finished.connect(self._on_load_thread_finished)

        self._load_thread.start()

    def _on_load_volume_finished(self, success, error_msg, kind, tag, data):
        self.hide_loading()
        if success and data is not None:
            if callable(self._load_callback):
                self._load_callback(data)
        elif not success:
            self.left_panel.append_log("Error al cargar volumen (%s): %s" % (kind, error_msg))

    def _on_load_thread_finished(self):
        self._load_thread = None
        self._load_worker = None
        self._load_callback = None

    def _build_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("Archivo")

        self.index_action = file_menu.addAction("Indexar carpeta...")
        self.index_action.triggered.connect(self._on_open_directory_dialog)

        self.recent_menu = file_menu.addMenu("Abrir carpeta reciente")
        self._update_recent_menu()

        self.reindex_action = file_menu.addAction("Reindexar directorio actual")
        self.reindex_action.setEnabled(False)
        self.reindex_action.triggered.connect(self._reindex_current_directory)

        file_menu.addSeparator()
        exit_action = file_menu.addAction("Salir")
        exit_action.triggered.connect(self.close)

        view_menu = menu_bar.addMenu("Ver")
        view_menu.addAction(self.tools_dock.toggleViewAction())

    def _on_open_directory_dialog(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Seleccionar directorio para indexar"
        )
        if directory:
            self._on_directory_selected(directory)

    def _on_open_directory_dialog_from_panel(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Seleccionar directorio para indexar"
        )
        if directory:
            self.left_panel.set_selected_directory(directory)

    def _on_index_requested_from_panel(self, directory):
        if not directory or not os.path.isdir(directory):
            QMessageBox.warning(
                self, "Carpeta no encontrada",
                "La carpeta no existe:\n%s" % (directory or "")
            )
            self.recent_store.remove(directory)
            self._update_recent_menu()
            self.left_panel.load_recent_folders()
            return

        db_path, json_path = self._find_index_for_directory(directory)
        if db_path and os.path.isfile(db_path):
            reply = QMessageBox.question(
                self,
                "Índice existente encontrado",
                "Se ha encontrado un índice previo para:\n%s\n\n¿Desea abrir el índice existente?\n(Haga clic en 'No' para reindexar de nuevo)" % directory,
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._on_directory_selected(directory)
            elif reply == QMessageBox.No:
                self.dicom_root = directory
                self.current_db_path = None
                self.current_json_path = None
                self.reindex_action.setEnabled(False)
                self._remember_folder(directory)
                self._start_indexing(directory)
            else:
                return
        else:
            self.dicom_root = directory
            self.current_db_path = None
            self.current_json_path = None
            self.reindex_action.setEnabled(False)
            self._remember_folder(directory)
            self._start_indexing(directory)

    def _on_change_directory_from_panel(self):
        self.viewer_2d.clear()
        self.viewer_3d.clear()
        self.multi_viewer_2d.clear()
        self.multi_viewer_3d.clear()
        self.segmentation_display.clear()
        self.left_panel.clear_info()
        self.left_panel.load_recent_folders()
        self.left_panel.show_recent_view()
        if self._is_multi_study_mode:
            self.stack_2d.setCurrentIndex(0)
            self.stack_3d.setCurrentIndex(0)
            self._is_multi_study_mode = False
            self.tools_panel.set_multi_study_mode(False)

    def _update_recent_menu(self):
        self.recent_menu.clear()
        valid_folders = self.recent_store.load_valid(self._find_index_for_directory)
        if valid_folders:
            for folder in valid_folders:
                action = self.recent_menu.addAction(folder)
                action.triggered.connect(
                    lambda checked=False, f=folder: self._on_recent_folder_selected(f)
                )
        else:
            disabled_action = self.recent_menu.addAction("No hay carpetas recientes")
            disabled_action.setEnabled(False)

    def _find_index_for_directory(self, directory):
        if not directory or not os.path.isdir(directory):
            return None, None

        abs_dir = os.path.abspath(directory)
        dir_indexed = get_directory_indexed_dir(abs_dir)
        dir_files = get_directory_files_dir(abs_dir)

        cand_db = os.path.join(dir_indexed, COMBINED_DB_FILENAME)
        cand_json = os.path.join(dir_indexed, COMBINED_JSON_FILENAME)
        if os.path.isfile(cand_db):
            return cand_db, (cand_json if os.path.isfile(cand_json) else None)

        cand_db2 = os.path.join(dir_files, COMBINED_DB_FILENAME)
        cand_json2 = os.path.join(dir_files, COMBINED_JSON_FILENAME)
        if os.path.isfile(cand_db2):
            return cand_db2, (cand_json2 if os.path.isfile(cand_json2) else None)

        central_dir = os.path.join(LARMORNIUM_FILES_DIR, "indexed")
        central_db = os.path.join(central_dir, COMBINED_DB_FILENAME)
        central_json = os.path.join(central_dir, COMBINED_JSON_FILENAME)
        if os.path.isfile(central_db):
            try:
                src = IndexDataAccess(abs_dir, central_db).load_source_dicom_dir()
                if src and os.path.abspath(src) == abs_dir:
                    return central_db, (central_json if os.path.isfile(central_json) else None)
            except Exception:
                pass

        for search_dir in (dir_indexed, dir_files, central_dir):
            for db_name, json_name in [("pet_ct_index.db", "pet_ct_tree.json"), ("mri_index.db", "mri_tree.json")]:
                cand_db = os.path.join(search_dir, db_name)
                cand_json = os.path.join(search_dir, json_name)
                if os.path.isfile(cand_db):
                    try:
                        src = IndexDataAccess(abs_dir, cand_db).load_source_dicom_dir()
                        if src and os.path.abspath(src) == abs_dir:
                            return cand_db, (cand_json if os.path.isfile(cand_json) else None)
                    except Exception:
                        pass

        return None, None

    def _on_directory_selected(self, directory):
        if not directory or not os.path.isdir(directory):
            return

        db_path, json_path = self._find_index_for_directory(directory)
        if db_path and os.path.isfile(db_path):
            self.dicom_root = directory
            self.current_db_path = db_path
            self.current_json_path = json_path
            self.reindex_action.setEnabled(True)
            self.left_panel.append_log("Abriendo índice existente: %s" % directory)
            self._remember_folder(directory)
            self._load_index()
        else:
            self.dicom_root = directory
            self.current_db_path = None
            self.current_json_path = None
            self.reindex_action.setEnabled(False)
            self._start_indexing(directory)

    def _on_recent_folder_selected(self, directory):
        if not os.path.isdir(directory):
            QMessageBox.warning(
                self, "Carpeta no encontrada",
                "La carpeta ya no existe:\n%s" % directory
            )
            self.recent_store.remove(directory)
            self._update_recent_menu()
            if hasattr(self, "left_panel"):
                self.left_panel.load_recent_folders()
            return

        db_path, json_path = self._find_index_for_directory(directory)
        if db_path and os.path.isfile(db_path):
            self.dicom_root = directory
            self.current_db_path = db_path
            self.current_json_path = json_path
            self.reindex_action.setEnabled(True)
            self.left_panel.append_log("Abriendo índice existente: %s" % directory)
            self._remember_folder(directory)
            self._load_index()
        else:
            self.left_panel.append_log(
                "No se encontró un índice previo para: %s" % directory
            )
            self.dicom_root = directory
            self.current_db_path = None
            self.current_json_path = None
            self.reindex_action.setEnabled(False)
            self._start_indexing(directory)

    def _remember_folder(self, directory):
        self.recent_store.add(directory)
        self._update_recent_menu()
        if hasattr(self, "left_panel"):
            self.left_panel.load_recent_folders()

    def _reindex_current_directory(self):
        if self.dicom_root:
            self._start_indexing(self.dicom_root)

    def _start_indexing(self, directory):
        abs_dir = os.path.abspath(directory)
        output_dir = get_directory_indexed_dir(abs_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.index_action.setEnabled(False)
        self.reindex_action.setEnabled(False)
        self.left_panel.append_log("Indexando directorio: %s" % directory)
        self.show_loading("0 / 0")

        self._index_thread = QThread(self)
        self._index_worker = IndexWorker(directory, output_dir)
        self._index_worker.moveToThread(self._index_thread)

        self._index_thread.started.connect(self._index_worker.run)
        self._index_worker.log_message.connect(self.left_panel.append_log)
        self._index_worker.progress_num.connect(self._on_index_progress)
        self._index_worker.finished.connect(self._on_indexing_finished)
        self._index_worker.finished.connect(self._index_thread.quit)
        self._index_thread.finished.connect(self._index_thread.deleteLater)
        self._index_thread.finished.connect(self._on_indexing_thread_finished)

        self._index_thread.start()

    def _on_index_progress(self, current, total):
        if total > 0:
            self.set_loading_status(f"{current} / {total}")
        else:
            self.set_loading_status(f"{current}")

    def _on_indexing_thread_finished(self):
        self._index_thread = None
        self._index_worker = None

    def _on_indexing_finished(self, success, error_message):
        self.hide_loading()
        self.index_action.setEnabled(True)
        self.reindex_action.setEnabled(self.dicom_root is not None)

        if not success:
            self.left_panel.append_log("Error durante la indexación: %s" % error_message)
            QMessageBox.critical(self, "Error de indexación", error_message)
            return

        self.left_panel.append_log("Indexación completada.")
        if self.dicom_root:
            db_path, json_path = self._find_index_for_directory(self.dicom_root)
            self.current_db_path = db_path
            self.current_json_path = json_path
            self._remember_folder(self.dicom_root)
            self._load_index()

    def _load_index(self):
        db_path = self.current_db_path
        if not db_path and self.dicom_root:
            db_path, _ = self._find_index_for_directory(self.dicom_root)
            self.current_db_path = db_path
        if not db_path or not os.path.isfile(db_path):
            self.left_panel.append_log(
                "No se encontró el archivo de indexado para: %s" % (self.dicom_root or "")
            )
            return

        self.data_access = IndexDataAccess(self.dicom_root, db_path)
        self._populate_tree()
        self.viewer_2d.clear()
        self.viewer_3d.clear()
        self.segmentation_display.clear()
        if hasattr(self, "tools_panel") and self.tools_panel is not None:
            self.tools_panel.refresh_segmentation_viewer(self.dicom_root)

    def _populate_tree(self):
        db_path = self.current_db_path
        if not db_path and self.dicom_root:
            db_path, _ = self._find_index_for_directory(self.dicom_root)
            self.current_db_path = db_path
        if not db_path or not os.path.isfile(db_path):
            return

        modalities = self.data_access.load_modalities()
        fusion_patients = self.data_access.load_fusion_pairs()
        multi_study_patients = self.data_access.load_multi_study_patients()
        built_study_uids = join_pet_ct.load_fully_built_study_uids(
            db_path, RECENT_FOLDERS_CONFIG_PATH
        )
        built_pair_keys = join_pet_ct.load_built_pair_keys(
            RECENT_FOLDERS_CONFIG_PATH
        )
        built_ct_uids = join_pet_ct.load_built_ct_series_uids(
            RECENT_FOLDERS_CONFIG_PATH
        )
        built_pet_uids = join_pet_ct.load_built_pet_series_uids(
            RECENT_FOLDERS_CONFIG_PATH
        )
        built_mri_series_uids = join_mri.load_built_mri_series_uids(
            RECENT_FOLDERS_CONFIG_PATH
        )
        built_mri_study_uids = join_mri.load_built_mri_study_uids(
            RECENT_FOLDERS_CONFIG_PATH
        )

        self.left_panel.populate_tree(
            modalities, fusion_patients, multi_study_patients,
            built_study_uids, built_pair_keys, built_ct_uids, built_pet_uids,
            built_mri_series_uids, built_mri_study_uids
        )
        self.left_panel.append_log(
            "Árbol de estudios cargado (%d modalidad(es))." % len(modalities)
        )

    def _on_viewer_mri_slice_changed(self, slice_num, total_slices, z_str):
        if self._current_node_data is not None:
            self._current_node_data["mri_slice_num"] = slice_num
            self._current_node_data["mri_total_slices"] = total_slices
            self._current_node_data["mri_z_pos"] = z_str
            self._update_info_table(self._current_node_data)

    def _on_main_tab_changed(self, index):
        if index != 0 and hasattr(self, "left_panel"):
            self.left_panel.clear_multi_study_checks()

    def _on_multi_study_selection_changed(self, checked_items):
        active = len(checked_items) > 0
        if hasattr(self, "tools_panel"):
            self.tools_panel.set_multi_study_active(active, checked_items)
        if not active:
            self._exit_multi_study_mode()

    def _enter_multi_study_mode(self):
        self._is_multi_study_mode = True
        if hasattr(self, "stack_2d"):
            self.stack_2d.setCurrentIndex(1)
        if hasattr(self, "stack_3d"):
            self.stack_3d.setCurrentIndex(1)
        if hasattr(self, "tab_widget") and hasattr(self, "viz_tab_widget"):
            self.tab_widget.setCurrentWidget(self.viz_tab_widget)
            for idx in range(1, self.tab_widget.count()):
                self.tab_widget.setTabVisible(idx, False)
            self.tab_widget.tabBar().setVisible(False)

    def _exit_multi_study_mode(self):
        self._is_multi_study_mode = False
        self._current_loaded_studies = []
        if hasattr(self, "multi_viewer_2d"):
            self.multi_viewer_2d.clear_shift_alignment()
        if hasattr(self, "stack_2d"):
            self.stack_2d.setCurrentIndex(0)
        if hasattr(self, "stack_3d"):
            self.stack_3d.setCurrentIndex(0)
        if hasattr(self, "tab_widget"):
            for idx in range(self.tab_widget.count()):
                self.tab_widget.setTabVisible(idx, True)
            self.tab_widget.tabBar().setVisible(True)
        if hasattr(self, "tools_panel") and hasattr(self.tools_panel, "multi_study_widget"):
            self.tools_panel.multi_study_widget.enable_viz_controls(False)
            self.tools_panel.multi_study_widget.clear_profile_plot()
        self.ensure_tools_dock_expanded(315)

    def _on_load_multi_study_requested(self, items):
        if not items:
            QMessageBox.information(self, "Visualización Multi-estudio", "No hay estudios seleccionados para cargar.")
            return

        self.show_loading(f"Cargando volúmenes multi-estudio (0/{len(items)})...")

        dir_files_dir = get_directory_files_dir(self.dicom_root) if self.dicom_root else LARMORNIUM_FILES_DIR
        db_path = self.current_db_path or (self._find_index_for_directory(self.dicom_root)[0] if self.dicom_root else None)

        self._multi_study_thread = QThread()
        self._multi_study_worker = MultiStudyVolumeWorker(
            items, self.dicom_root, dir_files_dir, RECENT_FOLDERS_CONFIG_PATH, db_path=db_path
        )
        self._multi_study_worker.moveToThread(self._multi_study_thread)
        self._multi_study_thread.started.connect(self._multi_study_worker.run)

        self._multi_study_worker.progress.connect(
            lambda cur, tot, msg: self.set_loading_status(f"{msg}")
        )
        self._multi_study_worker.log_message.connect(self.left_panel.append_log)
        self._multi_study_worker.finished.connect(self._on_multi_study_volumes_loaded)

        self._multi_study_worker.finished.connect(self._multi_study_thread.quit)
        self._multi_study_worker.finished.connect(self._multi_study_worker.deleteLater)
        self._multi_study_thread.finished.connect(self._multi_study_thread.deleteLater)

        self._multi_study_thread.start()

    def _on_multi_study_volumes_loaded(self, success, error_msg, loaded_studies):
        self.hide_loading()
        if not success and not loaded_studies:
            QMessageBox.warning(self, "Error de Carga", f"Ocurrió un error cargando los volúmenes:\n{error_msg}")
            return

        self._enter_multi_study_mode()
        self._current_loaded_studies = list(loaded_studies or [])

        # Buscar el valor máximo de SUV entre todos los estudios con visualización PET o Fusión
        has_pet = any(
            str(s.get("modality", "")).upper() in ("PET", "PT", "FUSION")
            for s in loaded_studies
        )
        global_max_suv = None
        if has_pet:
            suvs = [
                extract_study_pet_max_suv(s)
                for s in loaded_studies
                if str(s.get("modality", "")).upper() in ("PET", "PT", "FUSION")
            ]
            global_max_suv = max(suvs) if suvs else 1.0
            for s in loaded_studies:
                s["global_max_suv"] = global_max_suv
                vd = s.get("volume_data") or {}
                vd["global_max_suv"] = global_max_suv

        self.multi_viewer_2d.clear_shift_alignment()
        self.multi_viewer_2d.set_studies(loaded_studies)
        self.multi_viewer_3d.set_studies(loaded_studies)

        ms_w = self.tools_panel.multi_study_widget
        ms_w.enable_viz_controls(True)
        ms_w.btn_shift.blockSignals(True)
        ms_w.btn_shift.setChecked(False)
        ms_w.btn_shift.blockSignals(False)
        ms_w.set_pagination_info(0, self.multi_viewer_2d.total_pages())

        initial_ratio = ms_w.slider_slice.value() / 1000.0
        self.multi_viewer_2d.update_slice_ratio(initial_ratio)
        self.multi_viewer_2d.update_ct_window(ms_w._ct_window_center, ms_w._ct_window_width)

        # Acomodar el SUVmax global en el slider y en los visores si hay estudios con PET
        if has_pet and global_max_suv is not None:
            ms_w.set_pet_suv_range(global_max_suv)
            self.multi_viewer_2d.update_pet_suv_max(global_max_suv)
            self.multi_viewer_3d.update_pet_suv_max(global_max_suv)

        # Calcular alineación y perfiles axiales para la gráfica del panel
        from processing.vol_shift import align_volumes_by_shift
        self._current_alignment_result = align_volumes_by_shift(loaded_studies)
        study_info = [
            {
                "patient_name": s.get("patient_name") or (s.get("info") or s.get("item_info") or {}).get("patient_name") or f"Estudio {i+1}",
                "modality": s.get("modality") or (s.get("info") or s.get("item_info") or {}).get("modality", ""),
            }
            for i, s in enumerate(loaded_studies)
        ]
        ms_w.set_study_profiles(
            self._current_alignment_result.profiles,
            self._current_alignment_result,
            study_info=study_info,
        )

        self.ensure_tools_dock_expanded(315)
        self.left_panel.append_log(f"Visualización multi-estudio lista: {len(loaded_studies)} volúmenes cargados.")

    def _on_multi_study_shift_toggled(self, active):
        studies = getattr(self, "_current_loaded_studies", None)
        if not studies:
            return
        ms_w = self.tools_panel.multi_study_widget
        if active:
            from processing.vol_shift import align_volumes_by_shift

            result = align_volumes_by_shift(studies)
            self._current_alignment_result = result
            self.multi_viewer_2d.apply_shift_alignment(result)
            ms_w.update_profile_plot()
            self.left_panel.append_log(f"[Shift] Alineación axial activada:\n{result.summary}")
        else:
            self.multi_viewer_2d.clear_shift_alignment()
            ms_w.update_profile_plot()
            self.left_panel.append_log("[Shift] Alineación desactivada. Restaurado rango completo original.")

    def _on_multi_study_pet_suv_changed(self, suv_max):
        if hasattr(self, "multi_viewer_2d"):
            self.multi_viewer_2d.update_pet_suv_max(suv_max)
        if hasattr(self, "multi_viewer_3d"):
            self.multi_viewer_3d.update_pet_suv_max(suv_max)

    def _on_multi_study_opacity_changed(self, ct_alpha, pet_alpha):
        if hasattr(self, "multi_viewer_2d"):
            self.multi_viewer_2d.update_opacity(ct_alpha, pet_alpha)
        if hasattr(self, "multi_viewer_3d"):
            self.multi_viewer_3d.update_opacity(ct_alpha * 100.0, pet_alpha * 100.0)

    def _on_multi_study_page_change_requested(self, delta):
        cur = self.multi_viewer_2d._current_page
        new_page = cur + delta
        self.multi_viewer_2d.set_page(new_page)
        self.multi_viewer_3d.set_page(new_page)
        if hasattr(self, "tools_panel") and hasattr(self.tools_panel, "multi_study_widget"):
            self.tools_panel.multi_study_widget.set_pagination_info(
                self.multi_viewer_2d._current_page, self.multi_viewer_2d.total_pages()
            )

    def _on_multi_study_page_changed_from_2d(self, current_page, total_pages):
        if hasattr(self, "multi_viewer_3d") and self.multi_viewer_3d._current_page != current_page:
            self.multi_viewer_3d.set_page(current_page)
        if hasattr(self, "tools_panel") and hasattr(self.tools_panel, "multi_study_widget"):
            self.tools_panel.multi_study_widget.set_pagination_info(current_page, total_pages)

    def _on_multi_study_page_changed_from_3d(self, current_page, total_pages):
        if hasattr(self, "multi_viewer_2d") and self.multi_viewer_2d._current_page != current_page:
            self.multi_viewer_2d.set_page(current_page)
        if hasattr(self, "tools_panel") and hasattr(self.tools_panel, "multi_study_widget"):
            self.tools_panel.multi_study_widget.set_pagination_info(current_page, total_pages)

    def _on_viz_subtab_changed(self, index):
        if hasattr(self, "left_panel") and not getattr(self, "_is_multi_study_mode", False):
            self.left_panel.clear_multi_study_checks()
        if getattr(self, "_is_multi_study_mode", False):
            if index == 1 and hasattr(self, "multi_viewer_3d"):
                self.multi_viewer_3d.render_all()
            elif index == 0 and hasattr(self, "multi_viewer_2d"):
                self.multi_viewer_2d.render_all()
            return
        if index == 1:
            if hasattr(self.viewer_3d, "canvas") and self.viewer_3d.canvas:
                self.viewer_3d.canvas.render_scene()
        elif index == 0:
            if hasattr(self.viewer_2d, "_is_fusion_mode") and self.viewer_2d._is_fusion_mode:
                self.viewer_2d._render_current_fused_slice()
            elif hasattr(self.viewer_2d, "_is_mri_mode") and self.viewer_2d._is_mri_mode:
                self.viewer_2d._render_current_mri_slice()
            elif hasattr(self.viewer_2d, "_frames") and self.viewer_2d._frames:
                self.viewer_2d._display_frame(self.viewer_2d.slider.value())

    def _on_viewer_frame_changed(self, image_path, slice_num, total_slices, frame_index):
        if self._current_node_data is not None:
            self._current_node_data["current_image_path"] = image_path
            self._current_node_data["current_slice_num"] = slice_num
            self._current_node_data["total_slices"] = total_slices
            self._current_node_data["frame_index"] = frame_index
            self._update_info_table(self._current_node_data)

    def _on_viewer_fused_slice_changed(self, slice_num, total_slices, z_str):
        if self._current_node_data is not None:
            self._current_node_data["fused_slice_num"] = slice_num
            self._current_node_data["fused_total_slices"] = total_slices
            self._current_node_data["fused_z_pos"] = z_str
            self._update_info_table(self._current_node_data)

    def _update_info_table(self, data):
        node_type = data.get("type")
        db_path = self.current_db_path
        if not db_path and self.dicom_root:
            db_path, _ = self._find_index_for_directory(self.dicom_root)

        built_study_uids = join_pet_ct.load_fully_built_study_uids(
            db_path, RECENT_FOLDERS_CONFIG_PATH
        ) if db_path else set()
        built_pair_keys = join_pet_ct.load_built_pair_keys(
            RECENT_FOLDERS_CONFIG_PATH
        )

        def _format_dir(dir_path):
            if not dir_path:
                return self.dicom_root or "-"
            if os.path.isabs(dir_path):
                return dir_path
            if self.dicom_root:
                return os.path.join(self.dicom_root, dir_path)
            return dir_path

        rows = []
        if node_type == NODE_TYPE_STUDY:
            study_uid = data.get("study_instance_uid", "-")
            mod = data.get("modality", "")
            if mod == "MRI" or data.get("prefix") == "mri":
                built_mri_uids = join_mri.load_built_mri_uids(RECENT_FOLDERS_CONFIG_PATH)
                is_built = study_uid in built_mri_uids
                vol_label = "Volumen MRI"
                vol_status = "Generado" if is_built else "No construido"
            else:
                is_built = study_uid in built_study_uids
                vol_label = "Volumen Fusión"
                vol_status = "Generado" if is_built else "No construido"

            rows = [
                ("Elemento", "Estudio DICOM"),
                ("Directorio de Estudio", _format_dir(data.get("study_directory"))),
                ("Paciente", data.get("patient_name", "-")),
                ("ID Paciente", data.get("patient_id", "-")),
                ("Estudio", data.get("study_description", data.get("label", "-"))),
                ("Fecha", data.get("study_date", "-")),
                ("Modalidad", data.get("modality", "-")),
                ("Series", str(data.get("num_series", "-"))),
                ("UID Estudio", study_uid),
                (vol_label, vol_status),
            ]
            if mod == "MRI" or data.get("prefix") == "mri":
                m_slice = data.get("mri_slice_num")
                m_total = data.get("mri_total_slices")
                m_z = data.get("mri_z_pos")
                if m_slice and m_total:
                    corte_str = "%d de %d" % (m_slice, m_total)
                    if m_z:
                        corte_str += " (%s)" % m_z
                    rows.append(("Corte Visualizado", corte_str))
        elif node_type == NODE_TYPE_SERIES:
            curr_img = data.get("current_image_path")
            curr_slice = data.get("current_slice_num")
            total_slices = data.get("total_slices")
            slice_info = "%d de %d" % (curr_slice, total_slices) if curr_slice and total_slices else str(data.get("num_images", "-"))
            rows = [
                ("Elemento", "Serie DICOM"),
                ("Directorio de Estudio", _format_dir(data.get("study_directory"))),
                ("Directorio de Serie", _format_dir(data.get("series_directory"))),
            ]
            if curr_img:
                rows.append(("Imagen Visualizada", _format_dir(curr_img)))

            is_mri = (data.get("modality") in ("MR", "MRI") or data.get("prefix") == "mri")
            if is_mri:
                ser_uid = data.get("series_instance_uid", "-")
                st_uid = data.get("study_instance_uid", "-")
                built_mri_uids = join_mri.load_built_mri_uids(RECENT_FOLDERS_CONFIG_PATH)
                is_built = ser_uid in built_mri_uids or st_uid in built_mri_uids
                rows.append(("Volumen MRI", "Generado" if is_built else "No construido"))
                m_slice = data.get("mri_slice_num")
                m_total = data.get("mri_total_slices")
                m_z = data.get("mri_z_pos")
                if m_slice and m_total:
                    corte_str = "%d de %d" % (m_slice, m_total)
                    if m_z:
                        corte_str += " (%s)" % m_z
                    rows.append(("Corte Visualizado", corte_str))
            else:
                rows.append(("Corte Actual", slice_info))

            rows.extend([
                ("Paciente", data.get("patient_name", "-")),
                ("ID Paciente", data.get("patient_id", "-")),
                ("Estudio", data.get("study_description", "-")),
                ("Serie", data.get("series_description", data.get("label", "-"))),
                ("Número Serie", str(data.get("series_number", "-"))),
                ("Modalidad", data.get("modality", "-")),
                ("UID Serie", data.get("series_instance_uid", "-")),
            ])
        elif node_type == NODE_TYPE_FUSION_STUDY:
            study_uid = data.get("study_instance_uid", "-")
            is_built = study_uid in built_study_uids
            pairs = data.get("pairs", [])
            rows = [
                ("Elemento", "Estudio Corregistrado (PET/CT)"),
                ("Directorio de Estudio", _format_dir(data.get("study_directory"))),
                ("Paciente", data.get("patient_name", "-")),
                ("ID Paciente", data.get("patient_id", "-")),
                ("Estudio", data.get("study_description", data.get("label", "-"))),
                ("Fecha", data.get("study_date", "-")),
                ("Pares fusionables", str(len(pairs))),
                ("UID Estudio", study_uid),
                ("Volumen Fusión", "Generado" if is_built else "Generar al seleccionar"),
            ]
        elif node_type == NODE_TYPE_FUSION_PAIR:
            pair = data.get("pair") or {}
            key = join_pet_ct._pair_key(pair)
            is_built = key in built_pair_keys
            built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
            nii_path = built_pairs.get(key, {}).get("nii_path", "")
            f_slice = data.get("fused_slice_num")
            f_total = data.get("fused_total_slices")
            f_z = data.get("fused_z_pos")

            study_dir = data.get("study_directory") or (
                os.path.dirname(pair.get("ct_directory", "")) if pair.get("ct_directory") else ""
            )
            rows = [
                ("Elemento", "Par Fusionable"),
                ("Directorio de Estudio", _format_dir(study_dir)),
                ("Directorio CT", _format_dir(pair.get("ct_directory"))),
                ("Directorio PET", _format_dir(pair.get("pet_directory"))),
            ]
            if nii_path and os.path.isfile(nii_path):
                rows.append(("Volumen NIfTI (.nii.gz)", nii_path))
            if f_slice and f_total:
                corte_str = "%d de %d" % (f_slice, f_total)
                if f_z:
                    corte_str += " (%s)" % f_z
                rows.append(("Corte Fusionado", corte_str))
            else:
                rows.append(("Cortes emparejados", str(pair.get("num_slices", "-"))))

            rows.extend([
                ("Paciente", data.get("patient_name", "-")),
                ("ID Paciente", data.get("patient_id", "-")),
                ("Serie CT", str(pair.get("ct_series_description", "-"))),
                ("Kérnel CT", str(pair.get("ct_convolution_kernel") or self._resolve_ct_kernel(pair.get("ct_series_instance_uid")) or "-")),
                ("UID Serie CT", str(pair.get("ct_series_instance_uid", "-"))),
                ("Serie PET", str(pair.get("pet_series_description", "-"))),
                ("Método Reconstrucción PET", str(pair.get("pet_reconstruction_method") or self._resolve_pet_recon(pair.get("pet_series_instance_uid")) or "-")),
                ("Kérnel PET", str(pair.get("pet_convolution_kernel") or self._resolve_pet_kernel(pair.get("pet_series_instance_uid"), pair.get("pet_series_description")) or "-")),
                ("UID Serie PET", str(pair.get("pet_series_instance_uid", "-"))),
                ("Clave Par", key),
                ("Volumen Fusión", "Generado" if is_built else "Generar al seleccionar"),
            ])
        elif node_type in (NODE_TYPE_PATIENT, NODE_TYPE_FUSION_PATIENT):
            rows = [
                ("Elemento", "Paciente"),
                ("Directorio Raíz", self.dicom_root or "-"),
                ("Nombre", data.get("patient_name", data.get("label", "-"))),
                ("ID Paciente", data.get("patient_id", "-")),
                ("Estudios", str(data.get("num_studies", "-"))),
            ]
        elif node_type == NODE_TYPE_MODALITY:
            rows = [
                ("Elemento", "Modalidad"),
                ("Directorio Raíz", self.dicom_root or "-"),
                ("Nombre", data.get("label", "-")),
            ]
        elif node_type == NODE_TYPE_MULTI_STUDY_PATIENT:
            rows = [
                ("Elemento", "Paciente Multi-estudio"),
                ("Directorio Raíz", self.dicom_root or "-"),
                ("Nombre", data.get("patient_name", "-")),
                ("ID Paciente", data.get("patient_id", "-")),
                ("Modalidad", data.get("modality", "-")),
                ("Estudios", str(data.get("num_studies", "-"))),
            ]
        elif node_type == NODE_TYPE_MULTI_STUDY_STUDY:
            rows = [
                ("Elemento", "Estudio Multi-estudio"),
                ("Directorio Raíz", self.dicom_root or "-"),
                ("Descripción", data.get("study_description", "-")),
                ("Fecha", data.get("study_date", "-")),
                ("UID Estudio", data.get("study_instance_uid", "-")),
                ("Pares Fusionables", str(len(data.get("pairs", [])))),
            ]
        elif node_type == NODE_TYPE_MULTI_STUDY_GROUP:
            rows = [
                ("Elemento", f"Grupo: {data.get('label', '')}"),
                ("Directorio Raíz", self.dicom_root or "-"),
                ("Modalidad", data.get("group_modality", data.get("group_type", "-"))),
                ("Estudio", data.get("study_description", "-")),
                ("Paciente", data.get("patient_name", "-")),
                ("ID Paciente", data.get("patient_id", "-")),
            ]
        elif node_type == NODE_TYPE_MULTI_STUDY_DIRECTORY:
            rows = [
                ("Elemento", "Directorio de Estudio"),
                ("Directorio", _format_dir(data.get("label", "-"))),
            ]
        elif node_type == NODE_TYPE_FUSION_CATEGORY:
            rows = [
                ("Elemento", "Categoría"),
                ("Directorio Raíz", self.dicom_root or "-"),
                ("Nombre", "Estudios con corregistro"),
            ]

        self.left_panel.set_info(rows)

    def _resolve_ct_kernel(self, ct_uid):
        if not ct_uid:
            return ""
        db_path = self.current_db_path or (self._find_index_for_directory(self.dicom_root)[0] if self.dicom_root else None)
        if db_path and os.path.isfile(db_path):
            try:
                conn = sqlite3.connect(db_path)
                r = conn.execute("""
                    SELECT c.convolution_kernel FROM pet_ct_ct_parameters c
                    JOIN pet_ct_images i ON i.sop_instance_uid = c.sop_instance_uid
                    WHERE i.series_instance_uid = ? LIMIT 1
                """, (ct_uid,)).fetchone()
                conn.close()
                if r and r[0]:
                    return r[0]
            except Exception:
                pass
        return ""

    def _resolve_pet_recon(self, pet_uid):
        if not pet_uid:
            return ""
        db_path = self.current_db_path or (self._find_index_for_directory(self.dicom_root)[0] if self.dicom_root else None)
        if db_path and os.path.isfile(db_path):
            try:
                conn = sqlite3.connect(db_path)
                r = conn.execute("""
                    SELECT p.reconstruction_method FROM pet_ct_pet_parameters p
                    JOIN pet_ct_images i ON i.sop_instance_uid = p.sop_instance_uid
                    WHERE i.series_instance_uid = ? LIMIT 1
                """, (pet_uid,)).fetchone()
                conn.close()
                if r and r[0]:
                    return r[0]
            except Exception:
                pass
        return ""

    def _resolve_pet_kernel(self, pet_uid, pet_desc=""):
        db_path = self.current_db_path or (self._find_index_for_directory(self.dicom_root)[0] if self.dicom_root else None)
        if pet_uid and db_path and os.path.isfile(db_path):
            try:
                conn = sqlite3.connect(db_path)
                r = conn.execute("""
                    SELECT p.convolution_kernel FROM pet_ct_pet_parameters p
                    JOIN pet_ct_images i ON i.sop_instance_uid = p.sop_instance_uid
                    WHERE i.series_instance_uid = ? LIMIT 1
                """, (pet_uid,)).fetchone()
                conn.close()
                if r and r[0]:
                    return r[0]
            except Exception:
                pass
        if pet_desc:
            desc_up = pet_desc.upper()
            if "ALL" in desc_up:
                return "All-pass"
            elif "GAUSS" in desc_up:
                return "XYZ Gauss2.00"
        return ""

    def _on_node_selected(self, data):
        self._current_node_data = dict(data)
        node_type = data.get("type")
        label = data.get("label", "")
        self._update_info_table(self._current_node_data)

        dir_files_dir = get_directory_files_dir(self.dicom_root) if self.dicom_root else LARMORNIUM_FILES_DIR
        self.tools_panel.update_target(data, self.dicom_root, dir_files_dir, RECENT_FOLDERS_CONFIG_PATH)

        # En la rama de multi-estudio no se deben cargar volúmenes ni mostrar imágenes en los visores
        if data.get("is_multi_study") or node_type in (
            NODE_TYPE_MULTI_STUDY_CATEGORY,
            NODE_TYPE_MULTI_STUDY_PATIENT,
            NODE_TYPE_MULTI_STUDY_STUDY,
            NODE_TYPE_MULTI_STUDY_GROUP,
            NODE_TYPE_MULTI_STUDY_DIRECTORY,
        ):
            self.viewer_2d.clear()
            self.viewer_3d.clear()
            self.left_panel.append_log("Multi-estudio seleccionado: %s" % label)
            return

        self._exit_multi_study_mode()

        if hasattr(self, "viz_tab_widget") and hasattr(self, "tab_widget"):
            self.tab_widget.setCurrentWidget(self.viz_tab_widget)

        if node_type == NODE_TYPE_SERIES:
            if data.get("modality") in ("MR", "MRI") or data.get("prefix") == "mri":
                self._handle_mri_selected(data, label)
            else:
                self.left_panel.append_log("Cargando serie: %s ..." % label)
                self._load_series(
                    data["prefix"], data["series_instance_uid"], data.get("modality"), label, node_data=data
                )
        elif node_type == NODE_TYPE_FUSION_PAIR:
            pair = data.get("pair")
            self._handle_fusion_pair_selected(pair, label)
        elif node_type == NODE_TYPE_FUSION_STUDY:
            self.viewer_2d.clear()
            self.viewer_3d.clear()
            self.left_panel.append_log("Estudio con corregistro seleccionado: %s" % label)
        elif node_type == NODE_TYPE_STUDY:
            self.viewer_2d.clear()
            self.viewer_3d.clear()
            self.left_panel.append_log("Estudio seleccionado: %s" % label)
        else:
            self.viewer_2d.clear()
            self.viewer_3d.clear()
            self.left_panel.append_log("Seleccionado: %s" % label)

    def _load_series(self, prefix, series_instance_uid, modality, label, node_data=None):
        if self.data_access is None:
            return
        frames = self.data_access.load_series_frames(prefix, series_instance_uid)
        desc = (node_data.get("series_description") or label) if node_data else label
        ser_mod = str(modality).upper()
        init_suv = None
        if ser_mod in ("PT", "PET"):
            built_pet = join_pet_ct._load_built_pet_volumes(RECENT_FOLDERS_CONFIG_PATH)
            if series_instance_uid in built_pet:
                init_suv = built_pet[series_instance_uid].get("max_suv") or built_pet[series_instance_uid].get("pet_max_suv")

        self.viewer_2d.show_series(frames, modality, title=desc, initial_max_suv=init_suv)
        if frames:
            self.left_panel.append_log(
                "Serie cargada: %s (%d imágenes)" % (label, len(frames))
            )
        else:
            self.left_panel.append_log("Serie sin imágenes disponibles: %s" % label)

        if join_pet_ct.is_non_volume_series(node_data or label, modality=modality):
            self.viewer_3d.clear()
            return

        if ser_mod == "CT":
            built_ct = join_pet_ct._load_built_ct_volumes(RECENT_FOLDERS_CONFIG_PATH)
            if series_instance_uid in built_ct and os.path.isfile(built_ct[series_instance_uid].get("nii_path", "")):
                def _on_ct_loaded(vol_data):
                    ps = vol_data.get("pixel_spacing") or [1.0, 1.0]
                    st = vol_data.get("slice_thickness") or 1.0
                    sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
                    self.viewer_3d.show_ct_volume(vol_data["volume"], sp_3d, title="(%s)" % label)
                    self.left_panel.mark_ct_series_built(series_instance_uid)

                self._start_async_volume_load(
                    join_pet_ct.load_single_volume_data, "CT", series_instance_uid,
                    _on_ct_loaded, "Cargando volumen 3D CT...",
                    built_ct[series_instance_uid], modality="CT"
                )
            elif node_data:
                self._start_single_volume_build(node_data, modality="CT")
        elif ser_mod in ("PT", "PET"):
            built_pet = join_pet_ct._load_built_pet_volumes(RECENT_FOLDERS_CONFIG_PATH)
            if series_instance_uid in built_pet and os.path.isfile(built_pet[series_instance_uid].get("nii_path", "")):
                def _on_pet_loaded(vol_data):
                    ps = vol_data.get("pixel_spacing") or [1.0, 1.0]
                    st = vol_data.get("slice_thickness") or 1.0
                    sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
                    self.viewer_3d.show_pet_volume(vol_data["volume"], sp_3d, max_suv=vol_data["max_suv"], title="(%s)" % label)
                    self.left_panel.mark_pet_series_built(series_instance_uid)
                    if hasattr(self.viewer_2d, "set_pet_series_max_suv"):
                        self.viewer_2d.set_pet_series_max_suv(vol_data.get("max_suv"))

                self._start_async_volume_load(
                    join_pet_ct.load_single_volume_data, "PET", series_instance_uid,
                    _on_pet_loaded, "Cargando volumen 3D PET...",
                    built_pet[series_instance_uid], modality="PET"
                )
            elif node_data:
                self._start_single_volume_build(node_data, modality="PET")
        else:
            self.viewer_3d.clear()

    def _handle_mri_selected(self, data, label):
        node_type = data.get("type")
        series_uid = data.get("series_instance_uid") or ""
        study_uid = data.get("study_instance_uid") or ""

        # Si se seleccionó un estudio pero no se especificó serie_uid, tomar el de la primera serie si existe
        if node_type == NODE_TYPE_STUDY and not series_uid:
            series_uid = data.get("series_instance_uid") or ""

        target_uid = series_uid or study_uid
        if not target_uid:
            self.viewer_2d.clear()
            self.viewer_3d.clear()
            return

        built_mri = join_mri._load_built_mri_volumes(RECENT_FOLDERS_CONFIG_PATH)
        record = None

        # Si tenemos series_uid, buscar estrictamente por series_uid
        if series_uid:
            if series_uid in built_mri and os.path.isfile(built_mri[series_uid].get("nii_path", "")):
                record = built_mri[series_uid]
        elif study_uid:
            if study_uid in built_mri and os.path.isfile(built_mri[study_uid].get("nii_path", "")):
                record = built_mri[study_uid]

        series_desc = (
            data.get("series_description")
            or data.get("label")
            or (record.get("series_description") if record else "")
            or label
        )

        if record:
            self.left_panel.append_log("Cargando volumen MRI existente: %s ..." % label)

            def _on_mri_loaded(volume_data):
                self.viewer_2d.show_mri_volume(volume_data, title=series_desc)
                sp = volume_data.get("voxel_spacing") or [1.0, 1.0, 1.0]
                self.viewer_3d.show_mri_volume(volume_data["volume"], sp, title="(%s)" % series_desc)
                self.left_panel.mark_mri_built(study_uid, series_uid)
                self.left_panel.append_log("Volumen MRI cargado (%d cortes) [%s]." % (
                    volume_data.get("num_slices", 0), series_desc
                ))

            self._start_async_volume_load(
                join_mri.load_mri_volume_data, "MRI", target_uid,
                _on_mri_loaded, "Cargando volumen 3D MRI...",
                record
            )
        else:
            self._start_single_mri_build(data, label)

    def _start_single_mri_build(self, target_data, label):
        if self._is_thread_running(self._mri_thread):
            self.left_panel.append_log("Ya hay un proceso de MRI en ejecución. Espere a que finalice.")
            return

        dir_files_dir = get_directory_files_dir(self.dicom_root)
        self.left_panel.append_log("Iniciando generación de volumen MRI: %s ..." % label)
        self.show_loading("Generando volumen 3D MRI ...")
        self._mri_thread = QThread(self)
        self._mri_worker = SingleMRIWorker(
            self.dicom_root, target_data, dir_files_dir, RECENT_FOLDERS_CONFIG_PATH
        )
        self._mri_worker.moveToThread(self._mri_thread)

        self._mri_thread.started.connect(self._mri_worker.run)
        self._mri_worker.log_message.connect(self.left_panel.append_log)
        self._mri_worker.log_message.connect(self.set_loading_status)
        self._mri_worker.finished.connect(self._on_single_mri_finished)
        self._mri_worker.finished.connect(self._mri_thread.quit)
        self._mri_thread.finished.connect(self._mri_thread.deleteLater)
        self._mri_thread.finished.connect(self._on_mri_thread_finished)

        self._mri_thread.start()

    def _on_mri_thread_finished(self):
        self._mri_thread = None
        self._mri_worker = None

    def _on_single_mri_finished(self, success, error_message, modality, uid, volume_data):
        self.hide_loading()
        if not success:
            self.left_panel.append_log("Error construyendo volumen MRI: %s" % error_message)
            return

        study_uid = self._current_node_data.get("study_instance_uid") if self._current_node_data else ""
        series_uid = self._current_node_data.get("series_instance_uid") if self._current_node_data else uid
        self.left_panel.mark_mri_built(study_uid, series_uid)

        if volume_data and "volume" in volume_data:
            desc = (self._current_node_data.get("series_description") or
                    self._current_node_data.get("label") or "MRI") if self._current_node_data else "MRI"
            self.viewer_2d.show_mri_volume(volume_data, title=desc)
            sp = volume_data.get("voxel_spacing") or [1.0, 1.0, 1.0]
            self.viewer_3d.show_mri_volume(volume_data["volume"], sp, title="(%s)" % desc)
            self.left_panel.append_log(
                "Volumen MRI generado y cargado exitosamente (%d cortes) [%s]." % (
                    volume_data.get("num_slices", 0), desc
                )
            )

        if self._current_node_data is not None:
            self._update_info_table(self._current_node_data)
            dir_files_dir = get_directory_files_dir(self.dicom_root) if self.dicom_root else LARMORNIUM_FILES_DIR
            self.tools_panel.update_target(self._current_node_data, self.dicom_root, dir_files_dir, RECENT_FOLDERS_CONFIG_PATH)

    def _handle_fusion_pair_selected(self, pair, label):
        if not pair:
            return
        key = join_pet_ct._pair_key(pair)
        built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
        dir_files_dir = get_directory_files_dir(self.dicom_root)

        if key in built_pairs and os.path.isfile(built_pairs[key].get("nii_path", "")):
            self.left_panel.append_log("Cargando volumen fusionado existente: %s ..." % label)

            def _on_fused_loaded(volume_data):
                self.viewer_2d.show_fused_volume(volume_data)
                ps = volume_data.get("pixel_spacing") or [1.0, 1.0]
                st = volume_data.get("slice_thickness") or 1.0
                sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
                self.viewer_3d.show_fused_volume(
                    volume_data["ct_volume"], volume_data["pet_volume"],
                    sp_3d, max_suv=volume_data.get("max_suv"),
                    title="(%s)" % label
                )
                study_uid = pair.get("study_instance_uid", "")
                if study_uid:
                    self.left_panel.mark_study_built(study_uid, key)
                self.left_panel.append_log("Volumen fusionado cargado (%d cortes)." % volume_data["num_slices"])

            self._start_async_volume_load(
                join_pet_ct.load_fused_volume_data, "FUSION_PAIR", key,
                _on_fused_loaded, "Cargando volumen fusionado...",
                built_pairs[key], dir_files_dir, self.dicom_root, pair
            )
            return

        self._start_single_fusion(pair=pair)

    def _handle_fusion_study_selected(self, study_uid, pairs, label):
        if not study_uid:
            return
        db_path = self.current_db_path
        if not db_path and self.dicom_root:
            db_path, _ = self._find_index_for_directory(self.dicom_root)
        if not pairs and db_path:
            pairs = join_pet_ct.load_fusion_pairs_for_study(db_path, study_uid)
        if not pairs:
            self.viewer_2d.clear()
            self.viewer_3d.clear()
            self.left_panel.append_log("Seleccionado: %s" % label)
            return

        first_pair = pairs[0]
        key = join_pet_ct._pair_key(first_pair)
        built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
        dir_files_dir = get_directory_files_dir(self.dicom_root)

        if key in built_pairs and os.path.isfile(built_pairs[key].get("nii_path", "")):
            self.left_panel.append_log("Cargando volumen fusionado: %s ..." % label)

            def _on_study_fused_loaded(volume_data):
                self.viewer_2d.show_fused_volume(volume_data)
                ps = volume_data.get("pixel_spacing") or [1.0, 1.0]
                st = volume_data.get("slice_thickness") or 1.0
                sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
                self.viewer_3d.show_fused_volume(
                    volume_data["ct_volume"], volume_data["pet_volume"],
                    sp_3d, max_suv=volume_data.get("max_suv"),
                    title="(%s)" % label
                )
                self.left_panel.mark_study_built(study_uid, key)
                self.left_panel.append_log("Volumen fusionado cargado (%d cortes)." % volume_data["num_slices"])

            self._start_async_volume_load(
                join_pet_ct.load_fused_volume_data, "FUSION_STUDY", key,
                _on_study_fused_loaded, "Cargando volumen fusionado...",
                built_pairs[key], dir_files_dir, self.dicom_root, first_pair
            )
            return

        self._start_single_fusion(study_instance_uid=study_uid, pair=first_pair)

    @staticmethod
    def _is_thread_running(thread):
        if thread is None:
            return False
        try:
            return thread.isRunning()
        except RuntimeError:
            return False

    def _start_single_fusion(self, study_instance_uid=None, pair=None):
        if self._is_thread_running(self._fusion_thread):
            self.left_panel.append_log("Ya hay un proceso de fusión en ejecución. Espere a que finalice.")
            return

        db_path = self.current_db_path
        if not db_path and self.dicom_root:
            db_path, _ = self._find_index_for_directory(self.dicom_root)
        dir_files_dir = get_directory_files_dir(self.dicom_root)

        self.left_panel.append_log("Iniciando generación de volumen fusionado...")
        self.show_loading("Generando volumen fusionado PET/CT ...")
        self._fusion_thread = QThread(self)
        self._fusion_worker = SingleFusionWorker(
            self.dicom_root, db_path, dir_files_dir,
            RECENT_FOLDERS_CONFIG_PATH,
            study_instance_uid=study_instance_uid,
            pair=pair,
        )
        self._fusion_worker.moveToThread(self._fusion_thread)

        self._fusion_thread.started.connect(self._fusion_worker.run)
        self._fusion_worker.log_message.connect(self.left_panel.append_log)
        self._fusion_worker.log_message.connect(self.set_loading_status)
        self._fusion_worker.finished.connect(self._on_single_fusion_finished)
        self._fusion_worker.finished.connect(self._fusion_thread.quit)
        self._fusion_thread.finished.connect(self._fusion_thread.deleteLater)
        self._fusion_thread.finished.connect(self._on_fusion_thread_finished)

        self._fusion_thread.start()

    def _on_fusion_thread_finished(self):
        self._fusion_thread = None
        self._fusion_worker = None

    def _on_single_fusion_finished(self, success, error_message, study_uid, pair_key, volume_data):
        self.hide_loading()
        if not success:
            self.left_panel.append_log("Error en la fusión: %s" % error_message)
            return

        if study_uid:
            self.left_panel.mark_study_built(study_uid, pair_key=pair_key)

        if self._current_node_data is not None:
            self._update_info_table(self._current_node_data)
            dir_files_dir = get_directory_files_dir(self.dicom_root) if self.dicom_root else LARMORNIUM_FILES_DIR
            self.tools_panel.update_target(self._current_node_data, self.dicom_root, dir_files_dir, RECENT_FOLDERS_CONFIG_PATH)

        if volume_data:
            self.viewer_2d.show_fused_volume(volume_data)
            ps = volume_data.get("pixel_spacing") or [1.0, 1.0]
            st = volume_data.get("slice_thickness") or 1.0
            sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]
            self.viewer_3d.show_fused_volume(
                volume_data["ct_volume"], volume_data["pet_volume"],
                sp_3d, max_suv=volume_data.get("max_suv")
            )
            self.left_panel.append_log(
                "Volumen fusionado generado y cargado exitosamente (%d cortes)." % volume_data.get("num_slices", 0)
            )

    def _start_single_volume_build(self, series_data, modality="CT"):
        if self._is_thread_running(self._volume_thread):
            return

        dir_files_dir = get_directory_files_dir(self.dicom_root)
        self.show_loading("Generando volumen 3D %s ..." % modality)
        self._volume_thread = QThread(self)
        self._volume_worker = SingleVolumeWorker(
            self.dicom_root, series_data, dir_files_dir,
            RECENT_FOLDERS_CONFIG_PATH, modality=modality
        )
        self._volume_worker.moveToThread(self._volume_thread)

        self._volume_thread.started.connect(self._volume_worker.run)
        self._volume_worker.log_message.connect(self.left_panel.append_log)
        self._volume_worker.log_message.connect(self.set_loading_status)
        self._volume_worker.finished.connect(self._on_single_volume_finished)
        self._volume_worker.finished.connect(self._volume_thread.quit)
        self._volume_thread.finished.connect(self._volume_thread.deleteLater)
        self._volume_thread.finished.connect(self._on_volume_thread_finished)

        self._volume_thread.start()

    def _on_volume_thread_finished(self):
        self._volume_thread = None
        self._volume_worker = None

    def _on_single_volume_finished(self, success, error_message, modality, series_uid, volume_data):
        self.hide_loading()
        if not success:
            self.left_panel.append_log("Error construyendo volumen %s: %s" % (modality, error_message))
            return

        ps = volume_data.get("pixel_spacing") or [1.0, 1.0] if volume_data else [1.0, 1.0]
        st = volume_data.get("slice_thickness") or 1.0 if volume_data else 1.0
        sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]

        if modality == "CT":
            self.left_panel.mark_ct_series_built(series_uid)
            if volume_data and "volume" in volume_data:
                self.viewer_3d.show_ct_volume(volume_data["volume"], sp_3d)
        elif modality in ("PT", "PET"):
            self.left_panel.mark_pet_series_built(series_uid)
            if volume_data and "volume" in volume_data:
                self.viewer_3d.show_pet_volume(volume_data["volume"], sp_3d, max_suv=volume_data.get("max_suv"))

        if self._current_node_data is not None:
            self._update_info_table(self._current_node_data)
            dir_files_dir = get_directory_files_dir(self.dicom_root) if self.dicom_root else LARMORNIUM_FILES_DIR
            self.tools_panel.update_target(self._current_node_data, self.dicom_root, dir_files_dir, RECENT_FOLDERS_CONFIG_PATH)

    def _on_seg_thread_finished(self):
        self._seg_thread = None
        self._seg_worker = None

    def _on_segment_requested(self, payload):
        if self._is_thread_running(self._seg_thread):
            self.left_panel.append_log("Ya hay un proceso de segmentación en ejecución. Por favor espere.")
            return

        series_uid = payload.get("series_uid")
        modality = str(payload.get("modality", "")).upper()
        organ = payload.get("organ")
        organ_display = payload.get("organ_display", organ)
        cuda = bool(payload.get("cuda", False))
        fast = bool(payload.get("fast", False))

        if not series_uid or not organ:
            self.left_panel.append_log("Error: Serie u órgano no seleccionado para segmentar.")
            return

        dir_files_dir = get_directory_files_dir(self.dicom_root) if self.dicom_root else LARMORNIUM_FILES_DIR
        input_nii = None

        node_data = payload.get("node_data") or {}
        node_type = node_data.get("type")
        is_fusion = (node_type == NODE_TYPE_FUSION_PAIR) or ("Fusión" in str(node_data.get("label", "")))

        pair = None
        if node_type == NODE_TYPE_FUSION_PAIR:
            pair = node_data.get("pair")

        pair_key = payload.get("pair_key") or (join_pet_ct._pair_key(pair) if pair else None)
        if not pair and pair_key:
            built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
            if pair_key in built_pairs:
                pair = built_pairs[pair_key].get("pair")

        if is_fusion and pair and pair_key:
            built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
            fusion_nii = None
            if pair_key in built_pairs and os.path.isfile(built_pairs[pair_key].get("nii_path", "")):
                fusion_nii = built_pairs[pair_key]["nii_path"]
            else:
                fallback_fusion = os.path.join(dir_files_dir, FUSION_VOL_DIRNAME, f"{pair_key}.nii.gz")
                if os.path.isfile(fallback_fusion):
                    fusion_nii = fallback_fusion

            if not fusion_nii or not os.path.isfile(fusion_nii):
                self.show_loading("Generando volumen fusionado previo a la segmentación...")
                try:
                    _, rec = join_pet_ct.ensure_fusion_volume_for_pair(
                        pair, self.dicom_root, dir_files_dir, RECENT_FOLDERS_CONFIG_PATH
                    )
                    fusion_nii = rec.get("nii_path")
                except Exception as exc:
                    self.hide_loading()
                    self.left_panel.append_log(f"Error generando volumen fusionado: {exc}")
                    return

            ct_from_fusion = os.path.join(dir_files_dir, CT_VOL_DIRNAME, f"{pair_key}_ct.nii.gz")
            needs_extract = True
            if os.path.isfile(ct_from_fusion):
                try:
                    existing_nii = nib.load(ct_from_fusion)
                    aff = existing_nii.affine
                    # Si el archivo previo tiene la afín diagonal ficticia (origen 0,0,0 y paso Z positivo), se debe regenerar
                    if not (aff[0, 3] == 0.0 and aff[1, 3] == 0.0 and aff[2, 3] == 0.0 and aff[0, 0] > 0):
                        needs_extract = False
                except Exception:
                    needs_extract = True

            if needs_extract:
                try:
                    fnii = nib.load(fusion_nii)
                    fdata = fnii.get_fdata()
                    ct_3d = fdata[..., 0].astype(np.float32)

                    affine = fnii.affine.copy()
                    if affine[0, 3] == 0.0 and affine[1, 3] == 0.0 and affine[2, 3] == 0.0 and affine[0, 0] > 0:
                        affine = fusion_pet_ct.build_fusion_ct_affine_from_pair(pair, self.dicom_root, dir_files_dir)
                        try:
                            nib.save(nib.Nifti1Image(fdata, affine, fnii.header), fusion_nii)
                        except Exception:
                            pass

                    os.makedirs(os.path.dirname(ct_from_fusion), exist_ok=True)
                    ct_nii = nib.Nifti1Image(ct_3d, affine, fnii.header)
                    ct_nii.header.set_xyzt_units("mm")
                    nib.save(ct_nii, ct_from_fusion)
                except Exception as exc:
                    self.left_panel.append_log(f"Error extrayendo CT de la fusión: {exc}")
                    return
            input_nii = ct_from_fusion
        elif modality == "CT":
            built_ct = join_pet_ct._load_built_ct_volumes(RECENT_FOLDERS_CONFIG_PATH)
            if series_uid in built_ct and os.path.isfile(built_ct[series_uid].get("nii_path", "")):
                input_nii = built_ct[series_uid]["nii_path"]
            else:
                fallback_path = os.path.join(dir_files_dir, CT_VOL_DIRNAME, f"{series_uid}.nii.gz")
                if os.path.isfile(fallback_path):
                    input_nii = fallback_path
        elif modality in ("MR", "MRI"):
            built_mri = join_mri._load_built_mri_volumes(RECENT_FOLDERS_CONFIG_PATH)
            if series_uid in built_mri and os.path.isfile(built_mri[series_uid].get("nii_path", "")):
                input_nii = built_mri[series_uid]["nii_path"]
            else:
                fallback_path = os.path.join(dir_files_dir, MRI_VOL_DIRNAME, f"{series_uid}.nii.gz")
                if os.path.isfile(fallback_path):
                    input_nii = fallback_path

        if not input_nii or not os.path.isfile(input_nii):
            self.left_panel.append_log(
                f"No se encontró el volumen {modality} generado para la serie {series_uid}. "
                "Espere a que termine de generarse o seleccione la serie de nuevo."
            )
            return

        seg_context = payload.get("seg_context") or ("fusion" if is_fusion else ("mri" if modality in ("MR", "MRI") else "ct"))
        target_id = payload.get("target_id") or (pair_key if is_fusion else series_uid)

        seg_dir_name = _seg_dir_name(seg_context)
        output_dir = os.path.join(dir_files_dir, seg_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        output_basename = f"{target_id}_{organ}"

        self.show_loading(f"Segmentando {organ_display} ({modality})...\nEspere un momento.")
        self._seg_thread = QThread(self)
        self._seg_worker = SegmentationWorker(
            input_nii_path=input_nii,
            organ_key=organ,
            modality=modality,
            cuda=cuda,
            fast=fast,
            output_dir=output_dir,
            output_basename=output_basename,
            seg_context=seg_context,
            target_id=target_id,
            series_uid=series_uid,
            organ_display=organ_display,
            pair_key=pair_key,
            pair=pair,
        )
        self._seg_worker.moveToThread(self._seg_thread)
        self._seg_thread.started.connect(self._seg_worker.run)
        self._seg_worker.log_message.connect(self.left_panel.append_log)
        self._seg_worker.finished.connect(self._on_segment_worker_finished)
        self._seg_worker.finished.connect(self._seg_thread.quit)
        self._seg_thread.finished.connect(self._seg_thread.deleteLater)
        self._seg_thread.finished.connect(self._on_seg_thread_finished)
        self._seg_thread.start()

    @Slot(dict)
    def _on_segment_worker_finished(self, res):
        self._on_segment_finished(
            success=res["success"],
            error_message=res["error_message"],
            metadata=res["metadata"],
            seg_nii=res["seg_nii"],
            series_uid=res["series_uid"],
            organ=res["organ"],
            organ_display=res["organ_display"],
            modality=res["modality"],
            pair_key=res.get("pair_key"),
            pair=res.get("pair"),
            seg_context=res.get("seg_context"),
            target_id=res.get("target_id"),
        )

    def _on_segment_finished(self, success, error_message, metadata, seg_nii, series_uid, organ, organ_display, modality, pair_key=None, pair=None, seg_context=None, target_id=None):
        self.hide_loading()
        if not success:
            self.left_panel.append_log(f"Error en segmentación de {organ_display}: {error_message}")
            return

        nii_path = metadata.get("output_nifti")
        if not nii_path and hasattr(seg_nii, "get_filename"):
            nii_path = seg_nii.get_filename()

        effective_context = seg_context or ("fusion" if pair_key else ("mri" if modality in ("MR", "MRI") else "ct"))
        effective_target_id = target_id or (pair_key if effective_context == "fusion" else series_uid)

        stats = metadata.get("segmentation_stats") or metadata.get("stats") or {}
        num_voxels = stats.get("num_voxels")
        if num_voxels is None and seg_nii is not None and hasattr(seg_nii, "get_fdata"):
            try:
                num_voxels = int(np.sum(seg_nii.get_fdata() > 0))
            except Exception:
                num_voxels = 0
        elif num_voxels is None and nii_path and os.path.isfile(nii_path):
            try:
                num_voxels = int(np.sum(nib.load(nii_path).get_fdata() > 0))
            except Exception:
                num_voxels = 0

        is_empty = (num_voxels is not None and num_voxels == 0)

        rec = {
            "series_instance_uid": series_uid,
            "pair_key": pair_key or "",
            "target_id": effective_target_id,
            "seg_context": effective_context,
            "organ": organ,
            "modality": modality,
            "nii_path": nii_path or "",
            "json_path": metadata.get("output_json", ""),
            "stats": stats,
            "status": "warning" if is_empty else "built",
            "generated": not is_empty,
            "timestamp": datetime.now().isoformat(),
        }

        key = f"{effective_target_id}_{organ}"
        _mark_segmentation_built(RECENT_FOLDERS_CONFIG_PATH, effective_context, key, rec)
        if self.dicom_root:
            dir_conf = os.path.join(get_directory_files_dir(self.dicom_root), RECENT_FOLDERS_CONFIG_FILENAME)
            _mark_segmentation_built(dir_conf, effective_context, key, rec)

        if is_empty:
            self.tools_panel.refresh_organ_status(organ, "warning")
            self.left_panel.append_log(
                f"Aviso: No se generó la segmentación de {organ_display} (0 píxeles detectados en el volumen). "
                f"Puede volver a intentar la segmentación."
            )
            return

        self.tools_panel.refresh_organ_status(organ, "built")

        vol_cm3 = stats.get("volume_cm3")
        vol_txt = f" (Volumen: {vol_cm3:.2f} cm3)" if vol_cm3 is not None else ""
        self.left_panel.append_log(f"Segmentación de {organ_display} finalizada con éxito{vol_txt}.")

        # Visualizar en la pestaña de segmentación (2D con contorno amarillo + 3D con nube PET)
        try:
            dir_files_dir = get_directory_files_dir(self.dicom_root) if self.dicom_root else LARMORNIUM_FILES_DIR
            if seg_nii is not None and hasattr(seg_nii, "get_fdata"):
                seg_vol = seg_nii.get_fdata()
                voxel_spacing = [float(v) for v in seg_nii.header.get_zooms()[:3]]
            elif nii_path and os.path.isfile(nii_path):
                loaded_nii = nib.load(nii_path)
                seg_vol = loaded_nii.get_fdata()
                voxel_spacing = [float(v) for v in loaded_nii.header.get_zooms()[:3]]
            else:
                seg_vol = None
                voxel_spacing = None

            if seg_vol is not None:
                if effective_context == "fusion" and (pair_key or pair):
                    built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
                    rec_to_load = built_pairs.get(pair_key) if (pair_key and pair_key in built_pairs) else (pair_key or pair)
                    fused_data = join_pet_ct.load_fused_volume_data(
                        record_or_key=rec_to_load,
                        larmornium_files_dir=dir_files_dir,
                        dicom_root=self.dicom_root,
                        pair=pair
                    )
                    ct_vol = fused_data["ct_volume"]
                    pet_vol = fused_data["pet_volume"]
                    max_suv = fused_data.get("max_suv", 1.0)
                    z_pos = fused_data.get("z_positions", [])
                    ps = fused_data.get("pixel_spacing", [1.0, 1.0])
                    st = fused_data.get("slice_thickness", 1.0)
                    sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]

                    self.segmentation_display.show_segmentation(
                        seg_volume=seg_vol,
                        ct_volume=ct_vol,
                        pet_volume=pet_vol,
                        voxel_spacing=sp_3d,
                        max_suv=max_suv,
                        z_positions=z_pos,
                        title=f"{organ_display} (Fusión PET/CT)"
                    )
                elif effective_context == "ct" or modality == "CT":
                    built_ct = join_pet_ct._load_built_ct_volumes(RECENT_FOLDERS_CONFIG_PATH)
                    ct_vol = None
                    z_pos = []
                    if series_uid in built_ct:
                        ct_data = join_pet_ct.load_single_volume_data(built_ct[series_uid], modality="CT")
                        ct_vol = ct_data["volume"]
                        z_pos = ct_data.get("z_positions", [])
                    self.segmentation_display.show_segmentation(
                        seg_volume=seg_vol,
                        ct_volume=ct_vol,
                        voxel_spacing=voxel_spacing,
                        z_positions=z_pos,
                        title=f"{organ_display} (CT)"
                    )
                elif effective_context == "mri" or modality in ("MR", "MRI"):
                    built_mri = join_mri._load_built_mri_volumes(RECENT_FOLDERS_CONFIG_PATH)
                    mri_vol = None
                    z_pos = []
                    if series_uid in built_mri:
                        mri_data = join_mri.load_mri_volume_data(built_mri[series_uid])
                        mri_vol = mri_data["volume"]
                        z_pos = mri_data.get("z_positions", [])
                    self.segmentation_display.show_segmentation(
                        seg_volume=seg_vol,
                        mri_volume=mri_vol,
                        voxel_spacing=voxel_spacing,
                        z_positions=z_pos,
                        title=f"{organ_display} (MRI)"
                    )
                self.tab_widget.setCurrentWidget(self.segmentation_display)
        except Exception as exc:
            logger.exception("Error al visualizar segmentación: %s", exc)
            self.left_panel.append_log(f"Error al visualizar segmentación: {exc}")

    @staticmethod
    def _load_segmentation_bundle(nii_path, title, dicom_root, node_data):
        # Carga y descompresion pesada de NIfTI ejecutada en hilo secundario
        loaded_nii = nib.load(nii_path)
        seg_vol = loaded_nii.get_fdata()
        voxel_spacing = [float(v) for v in loaded_nii.header.get_zooms()[:3]]
        dir_files_dir = get_directory_files_dir(dicom_root) if dicom_root else LARMORNIUM_FILES_DIR

        curr_node = node_data or {}
        node_type = curr_node.get("type")
        pair = curr_node.get("pair") if node_type == NODE_TYPE_FUSION_PAIR else None

        built_segs = _load_built_segmentations(RECENT_FOLDERS_CONFIG_PATH)
        if dicom_root:
            dir_conf = os.path.join(get_directory_files_dir(dicom_root), RECENT_FOLDERS_CONFIG_FILENAME)
            if os.path.isfile(dir_conf):
                for k, v in _load_built_segmentations(dir_conf).items():
                    if k not in built_segs:
                        built_segs[k] = v

        rec = None
        norm_nii = os.path.realpath(nii_path)
        base_nii = os.path.basename(nii_path)
        for k, v in built_segs.items():
            v_nii = v.get("nii_path", "")
            if v_nii and (os.path.realpath(v_nii) == norm_nii or os.path.basename(v_nii) == base_nii):
                rec = v
                break

        seg_context = (rec.get("seg_context") if rec else None) or (
            "fusion" if (FUSION_SEG_VOL_DIRNAME in nii_path or pair) else (
                "mri" if (MRI_SEG_VOL_DIRNAME in nii_path or curr_node.get("modality") in ("MR", "MRI")) else "ct"
            )
        )
        pair_key = (rec.get("pair_key") if rec else None) or (join_pet_ct._pair_key(pair) if pair else None)

        ct_vol = None
        pet_vol = None
        mri_vol = None
        max_suv = 1.0
        z_pos = []
        sp_3d = voxel_spacing

        if seg_context == "fusion" and (pair_key or pair):
            built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
            if dicom_root:
                dir_conf = os.path.join(get_directory_files_dir(dicom_root), RECENT_FOLDERS_CONFIG_FILENAME)
                if os.path.isfile(dir_conf):
                    dir_pairs = join_pet_ct._load_built_pairs(dir_conf)
                    for k, v in dir_pairs.items():
                        if k not in built_pairs:
                            built_pairs[k] = v
            rec_to_load = built_pairs.get(pair_key) if (pair_key and pair_key in built_pairs) else (pair_key or pair)
            fused_data = join_pet_ct.load_fused_volume_data(
                record_or_key=rec_to_load,
                larmornium_files_dir=dir_files_dir,
                dicom_root=dicom_root,
                pair=pair
            )
            ct_vol = fused_data["ct_volume"]
            pet_vol = fused_data["pet_volume"]
            max_suv = fused_data.get("max_suv", 1.0)
            z_pos = fused_data.get("z_positions", [])
            ps = fused_data.get("pixel_spacing", [1.0, 1.0])
            st = fused_data.get("slice_thickness", 1.0)
            sp_3d = [float(ps[1]) if len(ps) > 1 else float(ps[0]), float(ps[0]), float(st)]

        elif seg_context == "ct" or curr_node.get("modality") == "CT":
            series_uid = (rec.get("series_instance_uid") if rec else None) or curr_node.get("series_instance_uid")
            built_ct = join_pet_ct._load_built_ct_volumes(RECENT_FOLDERS_CONFIG_PATH)
            if dicom_root:
                dir_conf = os.path.join(get_directory_files_dir(dicom_root), RECENT_FOLDERS_CONFIG_FILENAME)
                if os.path.isfile(dir_conf):
                    dir_ct = join_pet_ct._load_built_ct_volumes(dir_conf)
                    for k, v in dir_ct.items():
                        if k not in built_ct:
                            built_ct[k] = v
            if series_uid and series_uid in built_ct:
                ct_data = join_pet_ct.load_single_volume_data(built_ct[series_uid], modality="CT")
                ct_vol = ct_data["volume"]
                z_pos = ct_data.get("z_positions", [])

            # Respaldo si no se pudo cargar desde built_ct: leer input_volume del json
            if ct_vol is None and rec and rec.get("json_path") and os.path.isfile(rec["json_path"]):
                try:
                    with open(rec["json_path"], "r", encoding="utf-8") as jf:
                        jdata = json.load(jf)
                    inp_vol = jdata.get("input_volume", "")
                    if inp_vol and os.path.isfile(inp_vol):
                        loaded_ct_nii = nib.load(inp_vol)
                        ct_vol = loaded_ct_nii.get_fdata()
                except Exception as exc:
                    logger.warning("No se pudo cargar input_volume CT desde json: %s", exc)

        elif seg_context == "mri" or curr_node.get("modality") in ("MR", "MRI"):
            series_uid = (rec.get("series_instance_uid") if rec else None) or curr_node.get("series_instance_uid")
            built_mri = join_mri._load_built_mri_volumes(RECENT_FOLDERS_CONFIG_PATH)
            if dicom_root:
                dir_conf = os.path.join(get_directory_files_dir(dicom_root), RECENT_FOLDERS_CONFIG_FILENAME)
                if os.path.isfile(dir_conf):
                    dir_mri = join_mri._load_built_mri_volumes(dir_conf)
                    for k, v in dir_mri.items():
                        if k not in built_mri:
                            built_mri[k] = v
            if series_uid and series_uid in built_mri:
                mri_data = join_mri.load_mri_volume_data(built_mri[series_uid])
                mri_vol = mri_data["volume"]
                z_pos = mri_data.get("z_positions", [])

            # Respaldo si no se pudo cargar desde built_mri: leer input_volume del json
            if mri_vol is None and rec and rec.get("json_path") and os.path.isfile(rec["json_path"]):
                try:
                    with open(rec["json_path"], "r", encoding="utf-8") as jf:
                        jdata = json.load(jf)
                    inp_vol = jdata.get("input_volume", "")
                    if inp_vol and os.path.isfile(inp_vol):
                        loaded_mri_nii = nib.load(inp_vol)
                        mri_vol = loaded_mri_nii.get_fdata()
                except Exception as exc:
                    logger.warning("No se pudo cargar input_volume MRI desde json: %s", exc)

        return {
            "seg_volume": seg_vol,
            "ct_volume": ct_vol,
            "pet_volume": pet_vol,
            "mri_volume": mri_vol,
            "voxel_spacing": sp_3d,
            "max_suv": max_suv,
            "z_positions": z_pos,
            "title": title
        }

    def _on_view_3d_requested(self, nii_path, title):
        if not nii_path or not os.path.isfile(nii_path):
            self.left_panel.append_log(f"Archivo de segmentación no encontrado: {nii_path}")
            return

        def _on_seg_bundle_loaded(bundle):
            if not bundle:
                return
            try:
                self.segmentation_display.show_segmentation(
                    seg_volume=bundle["seg_volume"],
                    ct_volume=bundle["ct_volume"],
                    pet_volume=bundle["pet_volume"],
                    mri_volume=bundle["mri_volume"],
                    voxel_spacing=bundle["voxel_spacing"],
                    max_suv=bundle["max_suv"],
                    z_positions=bundle["z_positions"],
                    title=bundle["title"]
                )
                self.tab_widget.setCurrentWidget(self.segmentation_display)
                self.segmentation_display.subtab_widget.setCurrentWidget(self.segmentation_display.viewer_3d)
                self.ensure_tools_dock_expanded(315)
                self.left_panel.append_log(f"Cargada segmentación: {bundle['title']}")
            except Exception as exc:
                logger.exception("Error al presentar segmentacion: %s", exc)
                self.left_panel.append_log(f"Error al presentar segmentacion: {exc}")

        self._start_async_volume_load(
            self._load_segmentation_bundle,
            "SEGMENTATION",
            title,
            _on_seg_bundle_loaded,
            f"Cargando {title}...",
            nii_path,
            title,
            self.dicom_root,
            self._current_node_data
        )

    def _on_spatial_analysis_completed(self, results, vol_context):
        try:
            self.fwhm_results_display.display_results(results, vol_context)
            self.tab_widget.setCurrentWidget(self.fwhm_results_display)
            self.ensure_tools_dock_expanded(315)
            n_pts = results.get("num_puntos", 0) if results else 0
            self.left_panel.append_log(
                f"Análisis resolución espacial completado: {n_pts} punto(s). "
                f"Resultados mostrados en 'Resultados de FWHM'."
            )
        except Exception as exc:
            logger.exception("Error al presentar resultados de FWHM: %s", exc)
            self.left_panel.append_log(f"Error al presentar resultados de FWHM: {exc}")

    def _on_uniformity_analysis_completed(self, results, vol_context):
        try:
            self.uniformity_display.display_results(results, vol_context)
            self.tab_widget.setCurrentWidget(self.uniformity_display)
            self.ensure_tools_dock_expanded(315)
            n_slices = results.get("num_slices_analyzed", 0) if results else 0
            gm = results.get("global_metrics", {}) if results else {}
            self.left_panel.append_log(
                f"Análisis de uniformidad completado: {n_slices} cortes analizados. "
                f"Promedio: {gm.get('mean_suv', 0.0):.2f} SUV | CV: {gm.get('cv_percent', 0.0):.2f}%. "
                f"Resultados mostrados en 'Resultados de Uniformidad'."
            )
        except Exception as exc:
            logger.exception("Error al presentar resultados de uniformidad: %s", exc)
            self.left_panel.append_log(f"Error al presentar resultados de uniformidad: {exc}")

    def _on_export_3d_requested(self, payload):
        # Gestiona la exportacion de la segmentacion a formato 3D en segundo plano
        if self._is_thread_running(self._export_3d_thread):
            self.left_panel.append_log("Ya hay una exportación 3D en ejecución. Por favor espere.")
            return

        nii_path = payload.get("nii_path")
        if not nii_path or not os.path.isfile(nii_path):
            self.left_panel.append_log(f"Error: Archivo NIfTI no encontrado: {nii_path}")
            self.tools_panel.set_export_status("Error: Archivo NIfTI no encontrado", success=False)
            return

        fmt = str(payload.get("format", "stl")).lower()
        scale = float(payload.get("scale", 1.0))
        smooth_sigma = float(payload.get("smooth_sigma", 0.5))
        quality = float(payload.get("quality", 0.3))
        disp_name = payload.get("disp_name", "segmentacion")
        json_path = payload.get("json_path")

        # Directorio de salida en larmornium_files/(carpeta_hash)/print_3d_files sin subdirectorios
        print_3d_dir = get_print_3d_dir(directory=self.dicom_root, nii_path=nii_path)

        # Nombre base del archivo de salida
        base_stem = os.path.basename(nii_path)
        if base_stem.endswith(".nii.gz"):
            base_stem = base_stem[:-7]
        elif base_stem.endswith(".nii"):
            base_stem = base_stem[:-4]

        scale_suffix = f"_{scale:g}x" if scale != 1.0 else ""
        out_filename = f"{base_stem}{scale_suffix}.{fmt}"
        output_path = os.path.join(print_3d_dir, out_filename)

        format_kwargs = {}
        if fmt == "stl":
            format_kwargs["ascii_stl"] = bool(payload.get("ascii_stl", False))
        elif fmt == "obj":
            format_kwargs["write_mtl_obj"] = bool(payload.get("write_mtl_obj", False))
        elif fmt == "glb":
            format_kwargs["title_glb"] = disp_name

        self.tools_panel.set_export_in_progress(True, f"Exportando a {fmt.upper()}...")
        self.left_panel.append_log(
            f"Iniciando exportación 3D de {disp_name} a formato {fmt.upper()} "
            f"(Escala: {scale:g}x, Suavizado: {smooth_sigma}, Calidad: {quality:g})..."
        )

        self._export_3d_thread = QThread(self)
        self._export_3d_worker = Export3DWorker(
            nii_path=nii_path,
            output_path=output_path,
            output_format=fmt,
            scale=scale,
            smooth_sigma=smooth_sigma,
            quality=quality,
            json_path=json_path,
            **format_kwargs
        )
        self._export_3d_worker.moveToThread(self._export_3d_thread)
        self._export_3d_thread.started.connect(self._export_3d_worker.run)

        self._export_3d_worker.finished.connect(self._on_export_3d_finished)
        self._export_3d_worker.finished.connect(self._export_3d_thread.quit)
        self._export_3d_thread.finished.connect(self._export_3d_thread.deleteLater)
        self._export_3d_thread.finished.connect(self._on_export_3d_thread_finished)
        self._export_3d_thread.start()

    @Slot(bool, str, object)
    def _on_export_3d_finished(self, success, error_msg, res):
        self.tools_panel.set_export_in_progress(False)
        if success and isinstance(res, dict):
            out_file = res.get("output_path", "")
            size_mb = os.path.getsize(out_file) / (1024 * 1024) if (out_file and os.path.isfile(out_file)) else 0.0
            num_triangles = res.get("num_triangles", 0)
            proc_time = res.get("processing_time_seconds", 0.0)
            self.left_panel.append_log(
                f"Exportación 3D completada con éxito: {os.path.basename(out_file)} "
                f"({size_mb:.2f} MB, {num_triangles:,} triángulos en {proc_time:.2f} s)."
            )
            self.left_panel.append_log(f"Ruta de salida: {out_file}")
            self.tools_panel.set_export_status(
                f"Exportado: {os.path.basename(out_file)} ({size_mb:.2f} MB)",
                success=True
            )
        else:
            self.left_panel.append_log(f"Error en exportación 3D: {error_msg}")
            self.tools_panel.set_export_status(f"Error: {error_msg}", success=False)

    def _on_export_3d_thread_finished(self):
        self._export_3d_thread = None
        self._export_3d_worker = None


def launch_gui():
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    if owns_app:
        sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
