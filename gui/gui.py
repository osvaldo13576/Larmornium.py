#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py - Interfaz grafica principal de Larmornium
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

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
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
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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
from calcular_HU_CT import calcular_hu_ct  # noqa: E402
from calcular_SUV_PT import calcular_suv_pt  # noqa: E402


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
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_config(self, config):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def load(self):
        folders = self._load_config().get("recent_folders", [])
        return [folder for folder in folders if isinstance(folder, str)]

    def load_valid(self, index_finder_fn=None):
        folders = self.load()
        valid = []
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            if index_finder_fn:
                db_path, _ = index_finder_fn(folder)
                if not db_path or not os.path.isfile(db_path):
                    continue
            valid.append(folder)
        if len(valid) != len(folders):
            self._save(valid)
        return valid

    def add(self, directory):
        directory = os.path.abspath(directory)
        folders = [f for f in self.load() if os.path.abspath(f) != directory]
        folders.insert(0, directory)
        folders = folders[:MAX_RECENT_FOLDERS]
        self._save(folders)
        return folders

    def remove(self, directory):
        directory = os.path.abspath(directory)
        folders = [f for f in self.load() if os.path.abspath(f) != directory]
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
    """Lee la jerarquia de estudios desde la base de datos indexada."""

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
            pid = row["patient_id"] or "(sin identificador)"
            if pid not in patients_by_id:
                patients_by_id[pid] = PatientInfo(pid, row["patient_name"] or pid)
                patient_order.append(pid)

            study_dir = ""
            if has_slice_dirs and row["slice_directories"]:
                try:
                    s_dirs = json.loads(row["slice_directories"]) if isinstance(row["slice_directories"], str) else row["slice_directories"]
                    if s_dirs:
                        study_dir = s_dirs[0] if len(s_dirs) == 1 else os.path.commonpath(s_dirs)
                except Exception:
                    pass

            study = StudyInfo(
                study_instance_uid=row["study_instance_uid"],
                study_description=row["study_description"] or "(sin descripcion)",
                study_date=row["study_date"] or "",
                study_directory=study_dir,
            )
            studies_by_uid[study.study_instance_uid] = study
            patients_by_id[pid].studies.append(study)

        frame_counts = self._count_frames_per_series(conn, prefix)

        series_dir_map = {}
        if f"{prefix}_images" in table_names:
            img_rows = conn.execute(
                f"SELECT series_instance_uid, file_path FROM {prefix}_images GROUP BY series_instance_uid"
            )
            for ir in img_rows:
                fp = ir["file_path"] or ""
                series_dir_map[ir["series_instance_uid"]] = os.path.dirname(fp) if fp else ""

        rows = conn.execute(
            f"SELECT series_instance_uid, study_instance_uid, series_description, "
            f"series_number, modality, num_images FROM {prefix}_series"
        )
        for row in rows:
            study = studies_by_uid.get(row["study_instance_uid"])
            if study is None:
                continue
            series_uid = row["series_instance_uid"]
            num_images = frame_counts.get(series_uid, row["num_images"] or 0)
            series_dir = series_dir_map.get(series_uid, "")
            study.series_list.append(SeriesInfo(
                series_instance_uid=series_uid,
                series_description=row["series_description"] or "(sin descripcion)",
                series_number=row["series_number"],
                modality=row["modality"] or "",
                num_images=num_images,
                series_directory=series_dir,
            ))

        return [patients_by_id[pid] for pid in patient_order if patients_by_id[pid].studies]

    def _count_frames_per_series(self, conn, prefix):
        columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({prefix}_images)")
        }
        if "number_of_frames" in columns:
            rows = conn.execute(
                f"SELECT series_instance_uid, "
                f"SUM(CASE WHEN number_of_frames > 1 THEN number_of_frames ELSE 1 END) AS total "
                f"FROM {prefix}_images GROUP BY series_instance_uid"
            )
        else:
            rows = conn.execute(
                f"SELECT series_instance_uid, COUNT(*) AS total "
                f"FROM {prefix}_images GROUP BY series_instance_uid"
            )
        return {row["series_instance_uid"]: row["total"] for row in rows}

    def load_series_frames(self, prefix, series_instance_uid):
        conn = self._connect()
        try:
            columns = {
                row[1] for row in conn.execute(f"PRAGMA table_info({prefix}_images)")
            }
            has_frames = "number_of_frames" in columns
            frames_expr = "number_of_frames" if has_frames else "NULL"
            rows = conn.execute(
                f"SELECT file_path, instance_number, {frames_expr} AS number_of_frames "
                f"FROM {prefix}_images WHERE series_instance_uid = ? "
                f"ORDER BY (instance_number IS NULL), instance_number, file_path",
                (series_instance_uid,),
            ).fetchall()
        finally:
            conn.close()

        frames = []
        for row in rows:
            rel_path = row["file_path"]
            if not rel_path:
                continue
            abs_path = os.path.join(self.dicom_root, rel_path)
            num_frames = row["number_of_frames"] or 1
            if num_frames > 1:
                for frame_index in range(num_frames):
                    frames.append((abs_path, frame_index))
            else:
                frames.append((abs_path, None))
        return frames

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
                "SELECT fp.id, fp.study_instance_uid, fp.ct_series_instance_uid, "
                "fp.pet_series_instance_uid, fp.ct_series_description, "
                "fp.pet_series_description, fp.ct_directory, fp.pet_directory, "
                "fp.num_slices, fp.slice_thickness, "
                "s.study_description, s.study_date, s.patient_id, s.patient_name "
                "FROM pet_ct_fusion_pairs fp "
                "LEFT JOIN pet_ct_studies s ON s.study_instance_uid = fp.study_instance_uid "
                "ORDER BY s.patient_id, fp.study_instance_uid"
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

            study_uid = row["study_instance_uid"] or ""
            study_key = (pid, study_uid)
            if study_key not in studies_by_key:
                st_dir = ""
                if row["ct_directory"]:
                    st_dir = os.path.dirname(row["ct_directory"]) or row["ct_directory"]
                study = FusionStudyInfo(
                    study_instance_uid=study_uid,
                    study_description=row["study_description"] or "(sin descripcion)",
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
                ct_series_description=row["ct_series_description"] or "(sin descripcion)",
                pet_series_description=row["pet_series_description"] or "(sin descripcion)",
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
                "SELECT value FROM summary WHERE key = 'source_dicom_dir'"
            ).fetchone()
            return row["value"] if row else None
        finally:
            conn.close()


class _QtLogHandler(logging.Handler):
    """Handler de logging que reenvia cada registro a una senal Qt."""

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
    """Ejecuta la indexacion combinada en un hilo separado de la GUI."""

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
    finished = Signal(bool, str, str, object)

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
                    self.finished.emit(False, "No se encontraron pares fusionables para el estudio.", study_uid, None)
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
            self.finished.emit(True, "", study_uid, volume_data)
        except Exception as exc:
            self.finished.emit(False, str(exc), self.study_instance_uid or "", None)


class _HoverImageLabel(QLabel):
    """QLabel que reporta la posicion del cursor para mostrar el valor HU/SUV del pixel."""

    pixel_hovered = Signal(float, float)
    pixel_left = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        pos = event.position()
        self.pixel_hovered.emit(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.pixel_left.emit()
        super().leaveEvent(event)


class ImageViewer(QWidget):
    """Visualizador central para series individuales y estudios fusionados PET/CT."""

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
        self._fusion_pet_units = "SUV"
        self._fusion_z_positions = []
        self._current_ct_matrix = None
        self._current_pet_matrix = None
        self._current_slice_index = 0
        self._num_slices = 0

        self._display_offset_x = 0.0
        self._display_offset_y = 0.0
        self._display_scale_x = 1.0
        self._display_scale_y = 1.0

        self.image_label = _HoverImageLabel(self.PLACEHOLDER_TEXT)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.pixel_hovered.connect(self._on_pixel_hovered)
        self.image_label.pixel_left.connect(self._on_pixel_left)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)

        self.position_label = QLabel("")
        self.position_label.setAlignment(Qt.AlignCenter)

        self.value_label = QLabel("")
        self.value_label.setAlignment(Qt.AlignCenter)

        self.window_label = QLabel("Ventaneo CT:")
        self.window_combo = QComboBox()
        for name, center, width in CT_WINDOW_PRESETS:
            self.window_combo.addItem(name, (center, width))
        self.window_combo.currentIndexChanged.connect(self._on_window_changed)

        self.window_row = QWidget()
        window_row_layout = QHBoxLayout(self.window_row)
        window_row_layout.setContentsMargins(0, 0, 0, 0)
        window_row_layout.addWidget(self.window_label)
        window_row_layout.addWidget(self.window_combo, 1)
        self.window_row.setVisible(False)

        self.transparency_row = QWidget()
        transparency_layout = QHBoxLayout(self.transparency_row)
        transparency_layout.setContentsMargins(0, 0, 0, 0)

        self.ct_opacity_label = QLabel("Opacidad CT: 100%")
        self.ct_opacity_slider = QSlider(Qt.Horizontal)
        self.ct_opacity_slider.setRange(0, 100)
        self.ct_opacity_slider.setValue(100)
        self.ct_opacity_slider.valueChanged.connect(self._on_transparency_changed)

        self.pet_opacity_label = QLabel("Opacidad PET: 40%")
        self.pet_opacity_slider = QSlider(Qt.Horizontal)
        self.pet_opacity_slider.setRange(0, 100)
        self.pet_opacity_slider.setValue(40)
        self.pet_opacity_slider.valueChanged.connect(self._on_transparency_changed)

        transparency_layout.addWidget(self.ct_opacity_label)
        transparency_layout.addWidget(self.ct_opacity_slider, 1)
        transparency_layout.addWidget(self.pet_opacity_label)
        transparency_layout.addWidget(self.pet_opacity_slider, 1)
        self.transparency_row.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.window_row)
        layout.addWidget(self.transparency_row)
        layout.addWidget(self.slider)
        layout.addWidget(self.position_label)
        layout.addWidget(self.value_label)

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
        self._fusion_pet_vmax = 1.0
        self._fusion_pet_units = "SUV"
        self._fusion_z_positions = []
        self._current_ct_matrix = None
        self._current_pet_matrix = None
        self._num_slices = 0

        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(self.PLACEHOLDER_TEXT)
        self.slider.blockSignals(True)
        self.slider.setMaximum(1)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.position_label.setText("")
        self.value_label.setText("")
        self.window_row.setVisible(False)
        self.transparency_row.setVisible(False)

    def show_series(self, frames, modality=None):
        if not frames:
            self.clear()
            self.image_label.setText("La serie seleccionada no tiene imagenes")
            return

        self._is_fusion_mode = False
        self._modality = modality
        self._frames = frames
        self._cache_path = None
        self._cache_dataset = None
        self._cache_pixel_array = None
        self._cache_value_matrix = None
        self._current_value_matrix = None
        self._current_units_label = None
        self.value_label.setText("")
        self.window_row.setVisible(modality == MODALITY_CT)
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
        """Carga y muestra un volumen fusionado PET/CT con controles interactivos."""
        if not volume_data or "ct_volume" not in volume_data:
            self.clear()
            self.image_label.setText("No se pudieron cargar los datos del volumen fusionado")
            return

        self._is_fusion_mode = True
        self._modality = "FUSION_PET_CT"
        self._frames = []
        self._fusion_ct_volume = volume_data["ct_volume"]
        self._fusion_pet_volume = volume_data["pet_volume"]
        self._fusion_pet_vmax = float(volume_data.get("pet_vmax", 1.0))
        self._fusion_pet_units = str(volume_data.get("pet_units", "SUV"))
        self._fusion_z_positions = volume_data.get("z_positions", [])
        self._num_slices = int(volume_data.get("num_slices", self._fusion_ct_volume.shape[0]))

        self.window_row.setVisible(True)
        self.transparency_row.setVisible(True)
        self.value_label.setText("")

        self.slider.blockSignals(True)
        self.slider.setMinimum(1)
        self.slider.setMaximum(self._num_slices)
        mid_slice = self._num_slices // 2
        self.slider.setValue(mid_slice + 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(self._num_slices > 1)

        self._display_fused_slice(mid_slice)

    def _on_slider_changed(self, value):
        if self._is_fusion_mode:
            self._display_fused_slice(value - 1)
        else:
            self._display_frame(value - 1)

    def _on_transparency_changed(self, _value):
        ct_val = self.ct_opacity_slider.value()
        pet_val = self.pet_opacity_slider.value()
        self.ct_opacity_label.setText("Opacidad CT: %d%%" % ct_val)
        self.pet_opacity_label.setText("Opacidad PET: %d%%" % pet_val)
        if self._is_fusion_mode:
            self._render_current_fused_slice()

    def _current_window(self):
        data = self.window_combo.currentData()
        if data is None:
            return CT_WINDOW_PRESETS[0][1], CT_WINDOW_PRESETS[0][2]
        return data

    def _on_window_changed(self, _index):
        if self._is_fusion_mode:
            self._render_current_fused_slice()
            return
        if self._modality != MODALITY_CT or self._current_value_matrix is None:
            return
        display_array = self._hu_to_display(self._current_value_matrix)
        image = self._array_to_qimage(display_array, False)
        self._current_pixmap = QPixmap.fromImage(image)
        self._update_pixmap_display()

    def _display_frame(self, index):
        if index < 0 or index >= len(self._frames):
            return
        path, frame_index = self._frames[index]
        try:
            value_matrix, display_array, is_rgb, units_label = self._compute_frame(
                path, frame_index
            )
        except Exception as exc:
            self.image_label.setText("Error al leer la imagen:\n%s" % exc)
            return

        self._current_value_matrix = value_matrix
        self._current_units_label = units_label
        image = self._array_to_qimage(display_array, is_rgb)
        self._current_pixmap = QPixmap.fromImage(image)
        self._update_pixmap_display()
        self.position_label.setText("Imagen %d / %d" % (index + 1, len(self._frames)))
        self.value_label.setText("")
        self.frame_changed.emit(path, index + 1, len(self._frames), frame_index)

    def _display_fused_slice(self, index):
        if index < 0 or index >= self._num_slices:
            return
        self._current_slice_index = index
        self._current_ct_matrix = self._fusion_ct_volume[index]
        self._current_pet_matrix = self._fusion_pet_volume[index]

        self._render_current_fused_slice()

        z_str = ""
        if index < len(self._fusion_z_positions):
            z_pos = self._fusion_z_positions[index]
            z_str = "z = %.1f mm" % z_pos
            self.position_label.setText("Corte %d / %d  (%s)" % (index + 1, self._num_slices, z_str))
        else:
            self.position_label.setText("Corte %d / %d" % (index + 1, self._num_slices))
        self.value_label.setText("")
        self.fused_slice_changed.emit(index + 1, self._num_slices, z_str)

    def _render_current_fused_slice(self):
        if self._current_ct_matrix is None or self._current_pet_matrix is None:
            return
        ct_slice = self._current_ct_matrix
        pet_slice = self._current_pet_matrix

        center, width = self._current_window()
        low = center - width / 2.0
        high = center + width / 2.0
        if high > low:
            ct_norm = np.clip((ct_slice - low) / (high - low), 0.0, 1.0)
        else:
            ct_norm = np.zeros_like(ct_slice)
        ct_rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

        pet_vmax = self._fusion_pet_vmax if self._fusion_pet_vmax > 0 else 1.0
        pet_norm = np.clip(pet_slice / pet_vmax, 0.0, 1.0)
        cmap = plt.colormaps.get("hot", plt.cm.hot)
        pet_rgb = cmap(pet_norm)[:, :, :3]

        ct_alpha = self.ct_opacity_slider.value() / 100.0
        pet_alpha = self.pet_opacity_slider.value() / 100.0

        pet_weight = pet_alpha * pet_norm[:, :, np.newaxis]
        ct_layer = ct_rgb * ct_alpha
        blended = ct_layer * (1.0 - pet_weight) + pet_rgb * pet_weight
        display_array = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)

        image = self._array_to_qimage(display_array, True)
        self._current_pixmap = QPixmap.fromImage(image)
        self._update_pixmap_display()

    def _compute_frame(self, path, frame_index):
        if path != self._cache_path:
            dataset = pydicom.dcmread(path)
            self._cache_path = path
            self._cache_dataset = dataset
            self._cache_pixel_array = dataset.pixel_array
            self._cache_value_matrix = None

        array = self._cache_pixel_array
        if frame_index is not None:
            array = array[frame_index]

        if array.ndim == 3 and array.shape[-1] in (3, 4):
            rgb = array[..., :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return None, rgb, True, None

        if self._modality == MODALITY_CT and frame_index is None:
            if self._cache_value_matrix is None:
                self._cache_value_matrix = calcular_hu_ct(path)
            value_matrix = self._cache_value_matrix
            return value_matrix, self._hu_to_display(value_matrix), False, "HU"

        if self._modality == MODALITY_PT and frame_index is None:
            if self._cache_value_matrix is None:
                self._cache_value_matrix = calcular_suv_pt(path)
            value_matrix = self._cache_value_matrix
            return value_matrix, self._minmax_display(value_matrix), False, "SUV"

        dataset = self._cache_dataset
        array = array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        array = array * slope + intercept
        return None, self._minmax_display(array), False, None

    def _hu_to_display(self, hu_matrix):
        center, width = self._current_window()
        low = center - width / 2.0
        high = center + width / 2.0
        clipped = np.clip(hu_matrix, low, high)
        if high > low:
            display = (clipped - low) / (high - low) * 255.0
        else:
            display = np.zeros_like(clipped)
        return display.astype(np.uint8)

    def _minmax_display(self, matrix):
        matrix = matrix.astype(np.float32)
        matrix_min = float(matrix.min())
        matrix_max = float(matrix.max())
        if matrix_max > matrix_min:
            display = (matrix - matrix_min) / (matrix_max - matrix_min) * 255.0
        else:
            display = np.zeros_like(matrix)
        return display.astype(np.uint8)

    def _array_to_qimage(self, array, is_rgb):
        array = np.ascontiguousarray(array)
        height, width = array.shape[0], array.shape[1]
        if is_rgb:
            image = QImage(array.data, width, height, width * 3, QImage.Format_RGB888)
        else:
            image = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
        return image.copy()

    def _update_pixmap_display(self):
        if self._current_pixmap is None:
            return
        scaled = self._current_pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

        label_size = self.image_label.size()
        self._display_offset_x = max(0.0, (label_size.width() - scaled.width()) / 2.0)
        self._display_offset_y = max(0.0, (label_size.height() - scaled.height()) / 2.0)
        self._display_scale_x = (
            self._current_pixmap.width() / float(scaled.width()) if scaled.width() > 0 else 1.0
        )
        self._display_scale_y = (
            self._current_pixmap.height() / float(scaled.height()) if scaled.height() > 0 else 1.0
        )

    def _on_pixel_hovered(self, x, y):
        if self._is_fusion_mode:
            if self._current_ct_matrix is None or self._current_pet_matrix is None:
                return
            img_x = int((x - self._display_offset_x) * self._display_scale_x)
            img_y = int((y - self._display_offset_y) * self._display_scale_y)
            height, width = self._current_ct_matrix.shape[:2]
            if 0 <= img_x < width and 0 <= img_y < height:
                hu_val = float(self._current_ct_matrix[img_y, img_x])
                pet_val = float(self._current_pet_matrix[img_y, img_x])
                units = self._fusion_pet_units or "SUV"
                self.value_label.setText(
                    "HU: %.1f | %s: %.2f  (x=%d, y=%d)" % (hu_val, units, pet_val, img_x, img_y)
                )
            else:
                self.value_label.setText("")
            return

        if self._current_value_matrix is not None and self._current_units_label is not None:
            img_x = int((x - self._display_offset_x) * self._display_scale_x)
            img_y = int((y - self._display_offset_y) * self._display_scale_y)
            height, width = self._current_value_matrix.shape[:2]
            if 0 <= img_x < width and 0 <= img_y < height:
                value = float(self._current_value_matrix[img_y, img_x])
                self.value_label.setText(
                    "%s: %.1f  (x=%d, y=%d)" % (self._current_units_label, value, img_x, img_y)
                )
            else:
                self.value_label.setText("")

    def _on_pixel_left(self):
        self.value_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap_display()


class StudySelectionPanel(QWidget):
    """Panel izquierdo: arbol jerarquico, tabla de informacion y bitacora."""

    node_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Estudios indexados"])
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)

        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Campo", "Valor"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        self.info_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.info_table.verticalHeader().setVisible(False)
        self.info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.info_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.info_table.setAlternatingRowColors(True)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.info_table)
        splitter.addWidget(self.log_output)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(splitter)

    def append_log(self, message):
        self.log_output.appendPlainText(message)

    def set_info(self, rows):
        """Actualiza la tabla de informacion con pares (campo, valor)."""
        self.info_table.setRowCount(len(rows))
        for row, (field, value) in enumerate(rows):
            item_f = QTableWidgetItem(str(field))
            item_v = QTableWidgetItem(str(value))
            item_f.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item_v.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.info_table.setItem(row, 0, item_f)
            self.info_table.setItem(row, 1, item_v)

    def clear_info(self):
        self.info_table.setRowCount(0)

    def mark_study_built(self, study_instance_uid, pair_key=None):
        """Actualiza en el arbol el nombre del estudio y/o par agregando el prefijo [ok] si falta."""
        def _update_items(parent_item):
            count = parent_item.childCount() if parent_item else self.tree.topLevelItemCount()
            for i in range(count):
                item = parent_item.child(i) if parent_item else self.tree.topLevelItem(i)
                data = item.data(0, Qt.UserRole) or {}
                if data.get("study_instance_uid") == study_instance_uid:
                    current_text = item.text(0)
                    if not current_text.startswith("[ok] "):
                        item.setText(0, "[ok] " + current_text)
                if pair_key and data.get("type") == NODE_TYPE_FUSION_PAIR:
                    p = data.get("pair") or {}
                    if join_pet_ct._pair_key(p) == pair_key:
                        current_text = item.text(0)
                        if not current_text.startswith("[ok] "):
                            item.setText(0, "[ok] " + current_text)
                _update_items(item)
        _update_items(None)

    def populate_tree(self, modalities, fusion_patients=None, multi_study_patients=None,
                      built_study_uids=None, built_pair_keys=None):
        self.tree.clear()
        self.clear_info()
        built_study_uids = built_study_uids or set()
        built_pair_keys = built_pair_keys or set()

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
                    if study.study_instance_uid in built_study_uids:
                        study_label = "[ok] " + study_label
                    study_item = QTreeWidgetItem([study_label])
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
                if study.study_instance_uid in built_study_uids:
                    study_label = "[ok] " + study_label
                study_item = QTreeWidgetItem([study_label])
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
                    if key in built_pair_keys:
                        pair_label = "[ok] " + pair_label
                    pair_item = QTreeWidgetItem([pair_label])
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
    """Panel de herramientas de analisis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._placeholder = QLabel("No hay herramientas de analisis disponibles.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._layout.addWidget(self._placeholder)

    def add_tool(self, widget):
        self._placeholder.hide()
        self._layout.addWidget(widget)


class MainWindow(QMainWindow):
    """Ventana principal de la aplicacion Larmornium."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Larmornium - Visualizacion PET/CT y MRI")
        self.resize(1400, 900)

        self.dicom_root = None
        self.current_db_path = None
        self.current_json_path = None
        self.data_access = None
        self._index_thread = None
        self._index_worker = None
        self._fusion_thread = None
        self._fusion_worker = None
        os.makedirs(LARMORNIUM_FILES_DIR, exist_ok=True)
        self.recent_store = RecentFoldersStore(RECENT_FOLDERS_CONFIG_PATH)

        self.left_panel = StudySelectionPanel()
        self.left_panel.node_selected.connect(self._on_node_selected)
        self._current_node_data = None

        self.viewer = ImageViewer()
        self.viewer.frame_changed.connect(self._on_viewer_frame_changed)
        self.viewer.fused_slice_changed.connect(self._on_viewer_fused_slice_changed)

        self.tools_panel = ToolsPanel()
        self.tools_dock = QDockWidget("Herramientas de analisis", self)
        self.tools_dock.setWidget(self.tools_panel)
        self.tools_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tools_dock)
        self.tools_dock.setVisible(False)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        self._build_menu()

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

        self._index_thread = QThread(self)
        self._index_worker = IndexWorker(directory, output_dir)
        self._index_worker.moveToThread(self._index_thread)

        self._index_thread.started.connect(self._index_worker.run)
        self._index_worker.log_message.connect(self.left_panel.append_log)
        self._index_worker.finished.connect(self._on_indexing_finished)
        self._index_worker.finished.connect(self._index_thread.quit)
        self._index_thread.finished.connect(self._index_thread.deleteLater)
        self._index_thread.finished.connect(self._on_indexing_thread_finished)

        self._index_thread.start()

    def _on_indexing_thread_finished(self):
        self._index_thread = None
        self._index_worker = None

    def _on_indexing_finished(self, success, error_message):
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
        self.viewer.clear()

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
        self.left_panel.populate_tree(
            modalities, fusion_patients, multi_study_patients, built_study_uids, built_pair_keys
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
                ("Volumen Fusión", "[ok] Generado" if is_built else "No construido"),
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
                rows.append(("Imagen Visualizada", curr_img))
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
                ("Volumen Fusión", "[ok] Generado" if is_built else "Generar al seleccionar"),
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
                ("Volumen Fusión", "[ok] Generado" if is_built else "Generar al seleccionar"),
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
                data["prefix"], data["series_instance_uid"], data.get("modality"), label
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
                self.viewer.clear()
                self.left_panel.append_log("Seleccionado: %s" % label)
        else:
            self.viewer.clear()
            self.left_panel.append_log("Seleccionado: %s" % label)

    def _load_series(self, prefix, series_instance_uid, modality, label):
        if self.data_access is None:
            return
        frames = self.data_access.load_series_frames(prefix, series_instance_uid)
        self.viewer.show_series(frames, modality)
        if frames:
            self.left_panel.append_log(
                "Serie cargada: %s (%d imágenes)" % (label, len(frames))
            )
        else:
            self.left_panel.append_log("Serie sin imágenes disponibles: %s" % label)

    def _handle_fusion_pair_selected(self, pair, label):
        if not pair:
            return
        key = join_pet_ct._pair_key(pair)
        built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
        dir_files_dir = get_directory_files_dir(self.dicom_root)

        if key in built_pairs and os.path.isfile(built_pairs[key].get("nii_path", "")):
            self.left_panel.append_log("Cargando volumen fusionado existente: %s ..." % label)
            try:
                volume_data = join_pet_ct.load_fused_volume_data(
                    built_pairs[key], dir_files_dir, self.dicom_root, pair
                )
                self.viewer.show_fused_volume(volume_data)
                study_uid = pair.get("study_instance_uid", "")
                if study_uid:
                    self.left_panel.mark_study_built(study_uid, key)
                self.left_panel.append_log("Volumen fusionado cargado (%d cortes)." % volume_data["num_slices"])
            except Exception as exc:
                self.left_panel.append_log("Error al cargar volumen fusionado: %s" % exc)
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
            self.viewer.clear()
            self.left_panel.append_log("Seleccionado: %s" % label)
            return

        first_pair = pairs[0]
        key = join_pet_ct._pair_key(first_pair)
        built_pairs = join_pet_ct._load_built_pairs(RECENT_FOLDERS_CONFIG_PATH)
        dir_files_dir = get_directory_files_dir(self.dicom_root)

        if key in built_pairs and os.path.isfile(built_pairs[key].get("nii_path", "")):
            self.left_panel.append_log("Cargando volumen fusionado: %s ..." % label)
            try:
                volume_data = join_pet_ct.load_fused_volume_data(
                    built_pairs[key], dir_files_dir, self.dicom_root, first_pair
                )
                self.viewer.show_fused_volume(volume_data)
                self.left_panel.mark_study_built(study_uid, key)
                self.left_panel.append_log("Volumen fusionado cargado (%d cortes)." % volume_data["num_slices"])
            except Exception as exc:
                self.left_panel.append_log("Error al cargar volumen fusionado: %s" % exc)
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
        self._fusion_worker.finished.connect(self._on_single_fusion_finished)
        self._fusion_worker.finished.connect(self._fusion_thread.quit)
        self._fusion_thread.finished.connect(self._fusion_thread.deleteLater)
        self._fusion_thread.finished.connect(self._on_fusion_thread_finished)

        self._fusion_thread.start()

    def _on_fusion_thread_finished(self):
        self._fusion_thread = None
        self._fusion_worker = None

    def _on_single_fusion_finished(self, success, error_message, study_uid, volume_data):
        self._fusion_thread = None
        self._fusion_worker = None

        if not success:
            self.left_panel.append_log("Error en la fusión: %s" % error_message)
            return

        if study_uid:
            self.left_panel.mark_study_built(study_uid)

        if volume_data:
            self.viewer.show_fused_volume(volume_data)
            self.left_panel.append_log(
                "Volumen fusionado generado y cargado exitosamente (%d cortes)." % volume_data.get("num_slices", 0)
            )


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
