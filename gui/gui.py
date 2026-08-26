#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py - Interfaz gráfica principal de Larmornium
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pydicom

from PySide6.QtCore import Qt, QObject, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QIcon, QImage, QPixmap, QMovie, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
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
import join_pet_ct  # noqa: E402
import render_3d  # noqa: E402
from calcular_HU_CT import calcular_hu_ct  # noqa: E402
from calcular_SUV_PT import calcular_suv_pt  # noqa: E402
logger = logging.getLogger("larmornium.gui")

try:
    import vtk
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction
    from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals, vtkWindowedSincPolyDataFilter
    from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
    from vtkmodules.vtkRenderingCore import (
        vtkActor, vtkCamera, vtkPolyDataMapper, vtkRenderer, vtkRenderWindow,
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
ICON_DICOM_FILE_PATH = os.path.join(ICON_DIR, "dicom_file.png")
ICON_LOADING_PATH = os.path.join(ICON_DIR, "loading.gif")


def get_fusion_icon(is_built):
    icon_path = ICON_FUSION_FOUND_PATH if is_built else ICON_FUSION_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_ct_icon(is_built):
    icon_path = ICON_CT_VOL_FOUND_PATH if is_built else ICON_CT_VOL_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_pet_icon(is_built):
    icon_path = ICON_PET_VOL_FOUND_PATH if is_built else ICON_PET_VOL_NOT_FOUND_PATH
    return QIcon(icon_path) if os.path.isfile(icon_path) else QIcon()


def get_dicom_file_icon():
    return QIcon(ICON_DICOM_FILE_PATH) if os.path.isfile(ICON_DICOM_FILE_PATH) else QIcon()


def get_directory_id(directory):
    """Calcula el identificador único MD5 a partir de la ruta absoluta del directorio."""
    if not directory:
        return ""
    abs_path = os.path.abspath(directory)
    return hashlib.md5(abs_path.encode("utf-8")).hexdigest()


def get_directory_files_dir(directory):
    """Ruta base para un directorio específico dentro de larmornium_files/."""
    dir_id = get_directory_id(directory)
    return os.path.join(LARMORNIUM_FILES_DIR, dir_id)


def get_directory_indexed_dir(directory):
    """Directorio de indexado para una carpeta DICOM específica."""
    return os.path.join(get_directory_files_dir(directory), INDEXED_DIRNAME)


def get_directory_fusion_vol_dir(directory):
    """Directorio de volúmenes fusionados para una carpeta DICOM específica."""
    return os.path.join(get_directory_files_dir(directory), "fusion_vol")


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
    """Gestiona en larmornium.conf la lista de directorios abiertos recientemente."""

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
class MultiStudyPatientInfo:
    patient_id: str
    patient_name: str
    modality: str
    num_studies: int
    study_directories: list = field(default_factory=list)


class IndexDataAccess:
    """Lee la jerarquía de estudios desde la base de datos indexada."""

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
                if f"{prefix}_studies" in table_names:
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
        }
        has_slice_dirs = "slice_directories" in studies_table_cols
        slice_dirs_expr = ", slice_directories" if has_slice_dirs else ""

        studies_by_uid = {}
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
        finally:
            conn.close()

        patients_by_id = {}
        patient_order = []
        studies_by_key = {}

        for row in rows:
            pid = row["patient_id"] or "(sin identificador)"
            pname = row["patient_name"] or pid
            if pid not in patients_by_id:
                patients_by_id[pid] = FusionPatientInfo(pid, pname)
                patient_order.append(pid)

            study_uid = row["study_instance_uid"]
            study_key = (pid, study_uid)
            if study_key not in studies_by_key:
                ct_dir = row["ct_directory"] or ""
                st_dir = os.path.dirname(ct_dir) if ct_dir else ""
                study = FusionStudyInfo(
                    study_instance_uid=study_uid,
                    study_description=row["study_description"] or "(sin descripción)",
                    study_date=row["study_date"] or "",
                    study_directory=st_dir,
                )
                studies_by_key[study_key] = study
                patients_by_id[pid].studies.append(study)

            studies_by_key[study_key].pairs.append(FusionPairInfo(
                id=row["id"],
                study_instance_uid=study_uid,
                ct_series_instance_uid=row["ct_series_instance_uid"] or "",
                pet_series_instance_uid=row["pet_series_instance_uid"] or "",
                ct_series_description=row["ct_series_description"] or "(sin descripción)",
                pet_series_description=row["pet_series_description"] or "(sin descripción)",
                ct_directory=row["ct_directory"] or "",
                pet_directory=row["pet_directory"] or "",
                num_slices=row["num_slices"] or 0,
                slice_thickness=row["slice_thickness"],
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
                if table not in table_names:
                    continue
                rows = conn.execute(
                    f"SELECT patient_id, patient_name, num_studies, study_directories "
                    f"FROM {table} WHERE is_multi_study = 1"
                ).fetchall()
                for row in rows:
                    try:
                        directories = json.loads(row["study_directories"] or "[]")
                    except (ValueError, TypeError):
                        directories = []
                    pid = row["patient_id"] or "(sin identificador)"
                    results.append(MultiStudyPatientInfo(
                        patient_id=pid,
                        patient_name=row["patient_name"] or pid,
                        modality=MODALITY_LABELS.get(modality, modality),
                        num_studies=row["num_studies"] or 0,
                        study_directories=[d for d in directories if isinstance(d, str)],
                    ))
        finally:
            conn.close()
        return results


class _QtLogHandler(logging.Handler):
    """Handler de logging que reenvía cada registro a una señal Qt."""

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
    """Ejecuta la indexación combinada en un hilo separado de la GUI."""

    log_message = Signal(str)
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
            index_dicom_all.index_all(self.dicom_dir, self.output_dir, verbose=False)
            self.finished.emit(True, "")
        except Exception as exc:
            self.finished.emit(False, str(exc))
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(previous_level)


class SingleFusionWorker(QObject):
    """Genera en segundo plano el volumen fusionado de un estudio o par seleccionado."""

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
    """Genera en segundo plano el volumen individual 3D para una serie CT o PET."""

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


class LoadVolumeWorker(QObject):
    """Carga en segundo plano los datos de un volumen existente de disco."""

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


class _HoverImageLabel(QLabel):
    """QLabel que reporta la posición del cursor para mostrar el valor HU/SUV del píxel."""

    pixel_hovered = Signal(float, float)
    pixel_left = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(50, 50)

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

    def mouseMoveEvent(self, event):
        pos = event.position()
        self.pixel_hovered.emit(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.pixel_left.emit()
        super().leaveEvent(event)


class ImageViewer(QWidget):
    """Visualizador 2D central para series individuales y estudios fusionados PET/CT."""

    PLACEHOLDER_TEXT = "Seleccione una serie o estudio fusionado para visualizarlo"
    frame_changed = Signal(str, int, int, object)
    fused_slice_changed = Signal(int, int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_fusion_mode = False
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
        self._fusion_pet_units = "SUV"
        self._fusion_z_positions = []
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

        self.image_label = _HoverImageLabel(self.PLACEHOLDER_TEXT)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1c23; border: 1px solid #2d3139;")
        self.image_label.setMinimumSize(50, 50)
        self.image_label.pixel_hovered.connect(self._on_pixel_hovered)
        self.image_label.pixel_left.connect(self._on_pixel_left)
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

        win_label = QLabel("Ventana CT:")
        win_layout.addWidget(win_label)

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
        self.pet_suv_max_slider = QSlider(Qt.Horizontal)
        self.pet_suv_max_slider.setRange(1, 200)
        self.pet_suv_max_slider.setValue(100)
        self.pet_suv_max_slider.setFixedWidth(120)
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

    def _on_window_preset_changed(self, index):
        center, width = self.window_combo.itemData(index)
        self._ct_window_center = float(center)
        self._ct_window_width = float(width)
        if self._is_fusion_mode:
            self._render_current_fused_slice()
        elif self._modality == MODALITY_CT and self._cache_value_matrix is not None:
            self._apply_ct_window(self._cache_value_matrix)

    def clear(self):
        self._is_fusion_mode = False
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
        self._num_slices = 0

        self.slider.setEnabled(False)
        self.slider.blockSignals(True)
        self.slider.setValue(1)
        self.slider.setMaximum(1)
        self.slider.blockSignals(False)

        self.slice_label.setText("0 / 0")
        self.value_label.setText("")
        self.image_label.clear()
        self.image_label.setText(self.PLACEHOLDER_TEXT)
        self.window_row.setVisible(False)
        self.pet_suv_max_widget.setVisible(False)
        self.transparency_row.setVisible(False)

    def show_series(self, frames, modality):
        self._is_fusion_mode = False
        self._fusion_ct_volume = None
        self._fusion_pet_volume = None
        self._frames = frames
        self._modality = modality
        self._cache_path = None
        self._cache_dataset = None
        self._cache_pixel_array = None
        self._cache_value_matrix = None
        self._current_value_matrix = None
        self._current_units_label = None
        self.value_label.setText("")
        self.window_row.setVisible(modality == MODALITY_CT)
        self.pet_suv_max_widget.setVisible(False)
        self.transparency_row.setVisible(False)

        count = len(frames)
        self.slider.blockSignals(True)
        self.slider.setMinimum(1)
        self.slider.setMaximum(count)
        self.slider.setValue(1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(count > 1)
        self._display_frame(0)

    def show_fused_volume(self, volume_data):
        if not volume_data or "ct_volume" not in volume_data:
            self.clear()
            self.image_label.setText("No se pudieron cargar los datos del volumen fusionado")
            return

        self._is_fusion_mode = True
        self._modality = "FUSION_PET_CT"
        self._frames = []
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
        self.pet_suv_max_widget.setVisible(True)
        self.pet_suv_max_slider.blockSignals(True)
        self.pet_suv_max_slider.setRange(1, 200)
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

    def _on_slider_changed(self, value):
        if self._is_fusion_mode:
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

            elif self._modality == MODALITY_PT and photo_interp in ("MONOCHROME1", "MONOCHROME2", ""):
                if self._cache_value_matrix is None:
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
                self._current_value_matrix = self._cache_value_matrix
                self._current_units_label = "SUV"
                self._apply_pet_colormap(self._cache_value_matrix)

            else:
                self._current_value_matrix = arr
                self._current_units_label = "val"
                self._display_grayscale(arr)

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
            else:
                self.value_label.setText(f"{val:.1f}")

    def _on_pixel_left(self):
        self.value_label.setText("")


class VTKCanvas(QLabel):
    """Canvas de renderizado 3D VTK con soporte para rotación, zoom y paneo por ratón."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #101217; border: 1px solid #2d3139;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(50, 50)
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
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(400, 400)

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
    """Visualizador 3D para CT (silueta), PET (nube de radioactividad) y Fusión PET/CT."""

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
        self.ct_opacity_label.setText(f"Opacidad CT: {value}%")
        if self._ct_actor:
            self._ct_actor.GetProperty().SetOpacity(value / 100.0)
            self.canvas.render_scene()

    def _on_pet_opacity_changed(self, value):
        self.pet_opacity_label.setText(f"Opacidad PET: {value}%")
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


class StudySelectionPanel(QWidget):
    """Panel izquierdo que contiene el árbol de estudios, tabla de metadatos y registro de logs."""

    node_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dicom_root = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree, 3)

        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Campo", "Valor"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.info_table.setWordWrap(True)
        layout.addWidget(self.info_table, 2)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        self.log_output.setStyleSheet("background-color: #14161b; color: #a0a0a0; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_output, 1)

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
        """Actualiza en el árbol el icono del estudio y/o par cuando se genera el volumen fusionado."""
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

    def populate_tree(self, modalities, fusion_patients=None, multi_study_patients=None,
                      built_study_uids=None, built_pair_keys=None,
                      built_ct_uids=None, built_pet_uids=None):
        self.tree.clear()
        self.clear_info()
        built_study_uids = built_study_uids or set()
        built_pair_keys = built_pair_keys or set()
        built_ct_uids = built_ct_uids or set()
        built_pet_uids = built_pet_uids or set()

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
                    study_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_STUDY,
                        "label": study_label,
                        "study_instance_uid": study.study_instance_uid,
                        "study_description": study.study_description,
                        "study_date": study.study_date,
                        "study_directory": study.study_directory,
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
                self._build_multi_study_category_item(multi_study_patients)
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

    def _build_multi_study_category_item(self, multi_study_patients):
        category_item = QTreeWidgetItem([MULTI_STUDY_CATEGORY_LABEL])
        category_item.setData(0, Qt.UserRole, {
            "type": NODE_TYPE_MULTI_STUDY_CATEGORY,
            "label": MULTI_STUDY_CATEGORY_LABEL,
        })
        for patient in multi_study_patients:
            patient_label = "%s (%s, %s, %d estudios)" % (
                patient.patient_name, patient.patient_id,
                patient.modality, patient.num_studies,
            )
            patient_item = QTreeWidgetItem([patient_label])
            patient_item.setData(0, Qt.UserRole, {
                "type": NODE_TYPE_MULTI_STUDY_PATIENT,
                "label": patient_label,
                "patient_name": patient.patient_name,
                "patient_id": patient.patient_id,
                "num_studies": patient.num_studies,
            })
            for directory in patient.study_directories:
                directory_item = QTreeWidgetItem([directory])
                directory_item.setData(0, Qt.UserRole, {
                    "type": NODE_TYPE_MULTI_STUDY_DIRECTORY,
                    "label": directory,
                })
                patient_item.addChild(directory_item)
            category_item.addChild(patient_item)
        return category_item

    def _on_selection_changed(self):
        items = self.tree.selectedItems()
        if not items:
            return
        data = items[0].data(0, Qt.UserRole)
        if data:
            self.node_selected.emit(data)


class ToolsPanel(QWidget):
    """Panel de herramientas de análisis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._placeholder = QLabel("No hay herramientas de análisis disponibles.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)

    def add_tool(self, widget):
        self._placeholder.hide()
        self._layout.addWidget(widget)


class LoadingPanel(QWidget):
    """Panel de carga con animación loading.gif y mensaje de estado del proceso."""

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


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación Larmornium."""

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
        self._load_thread = None
        self._load_worker = None
        os.makedirs(LARMORNIUM_FILES_DIR, exist_ok=True)
        self.recent_store = RecentFoldersStore(RECENT_FOLDERS_CONFIG_PATH)

        self.left_panel = StudySelectionPanel()
        self.left_panel.setMinimumWidth(320)
        self.left_panel.node_selected.connect(self._on_node_selected)
        self._current_node_data = None

        self.viewer_2d = ImageViewer()
        self.viewer_2d.frame_changed.connect(self._on_viewer_frame_changed)
        self.viewer_2d.fused_slice_changed.connect(self._on_viewer_fused_slice_changed)
        self.viewer = self.viewer_2d

        self.viewer_3d = Viewer3DWidget()

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self.viewer_2d, "Visualización 2D")
        self.tab_widget.addTab(self.viewer_3d, "Visualización 3D")

        self.loading_panel = LoadingPanel()

        self.display_stack = QStackedWidget()
        self.display_stack.addWidget(self.tab_widget)
        self.display_stack.addWidget(self.loading_panel)
        self.display_stack.setCurrentWidget(self.tab_widget)

        self.tools_panel = ToolsPanel()
        self.tools_dock = QDockWidget("Herramientas de análisis", self)
        self.tools_dock.setWidget(self.tools_panel)
        self.tools_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tools_dock)
        self.tools_dock.setVisible(False)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.display_stack)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([380, 1020])
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        self._build_menu()

    def show_loading(self, message="Procesando..."):
        self.loading_panel.start(message)
        self.display_stack.setCurrentWidget(self.loading_panel)
        QApplication.processEvents()

    def set_loading_status(self, message):
        self.loading_panel.set_status(message)

    def hide_loading(self):
        self.loading_panel.stop()
        self.display_stack.setCurrentWidget(self.tab_widget)

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
        self.show_loading("Indexando directorio: %s ..." % os.path.basename(directory))

        self._index_thread = QThread(self)
        self._index_worker = IndexWorker(directory, output_dir)
        self._index_worker.moveToThread(self._index_thread)

        self._index_thread.started.connect(self._index_worker.run)
        self._index_worker.log_message.connect(self.left_panel.append_log)
        self._index_worker.log_message.connect(self.set_loading_status)
        self._index_worker.finished.connect(self._on_indexing_finished)
        self._index_worker.finished.connect(self._index_thread.quit)
        self._index_thread.finished.connect(self._index_thread.deleteLater)
        self._index_thread.finished.connect(self._on_indexing_thread_finished)

        self._index_thread.start()

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

        self.left_panel.populate_tree(
            modalities, fusion_patients, multi_study_patients,
            built_study_uids, built_pair_keys, built_ct_uids, built_pet_uids
        )
        self.left_panel.append_log(
            "Árbol de estudios cargado (%d modalidad(es))." % len(modalities)
        )

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
            is_built = study_uid in built_study_uids
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
                ("Volumen Fusión", "Generado" if is_built else "No construido"),
            ]
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
            rows.extend([
                ("Corte Actual", slice_info),
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
                ("UID Serie CT", str(pair.get("ct_series_instance_uid", "-"))),
                ("Serie PET", str(pair.get("pet_series_description", "-"))),
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
                ("Detalle", data.get("label", "-")),
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

    def _on_node_selected(self, data):
        self._current_node_data = dict(data)
        node_type = data.get("type")
        label = data.get("label", "")
        self._update_info_table(self._current_node_data)

        if node_type == NODE_TYPE_SERIES:
            self.left_panel.append_log("Cargando serie: %s ..." % label)
            self._load_series(
                data["prefix"], data["series_instance_uid"], data.get("modality"), label, node_data=data
            )
        elif node_type == NODE_TYPE_FUSION_PAIR:
            pair = data.get("pair")
            self._handle_fusion_pair_selected(pair, label)
        elif node_type == NODE_TYPE_FUSION_STUDY:
            study_uid = data.get("study_instance_uid")
            pairs = data.get("pairs", [])
            self._handle_fusion_study_selected(study_uid, pairs, label)
        elif node_type == NODE_TYPE_STUDY:
            study_uid = data.get("study_instance_uid")
            db_path = self.current_db_path
            if not db_path and self.dicom_root:
                db_path, _ = self._find_index_for_directory(self.dicom_root)
            fusion_pairs = join_pet_ct.load_fusion_pairs_for_study(db_path, study_uid) if db_path else []
            if fusion_pairs:
                self._handle_fusion_study_selected(study_uid, fusion_pairs, label)
            else:
                self.viewer_2d.clear()
                self.viewer_3d.clear()
                self.left_panel.append_log("Seleccionado: %s" % label)
        else:
            self.viewer_2d.clear()
            self.viewer_3d.clear()
            self.left_panel.append_log("Seleccionado: %s" % label)

    def _load_series(self, prefix, series_instance_uid, modality, label, node_data=None):
        if self.data_access is None:
            return
        frames = self.data_access.load_series_frames(prefix, series_instance_uid)
        self.viewer_2d.show_series(frames, modality)
        if frames:
            self.left_panel.append_log(
                "Serie cargada: %s (%d imágenes)" % (label, len(frames))
            )
        else:
            self.left_panel.append_log("Serie sin imágenes disponibles: %s" % label)

        if join_pet_ct.is_non_volume_series(node_data or label, modality=modality):
            self.viewer_3d.clear()
            return

        ser_mod = str(modality).upper()
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

                self._start_async_volume_load(
                    join_pet_ct.load_single_volume_data, "PET", series_instance_uid,
                    _on_pet_loaded, "Cargando volumen 3D PET...",
                    built_pet[series_instance_uid], modality="PET"
                )
            elif node_data:
                self._start_single_volume_build(node_data, modality="PET")
        else:
            self.viewer_3d.clear()

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


def launch_gui():
    """Punto de entrada de la GUI, invocado desde larmornium.py."""
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
