#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field

import numpy as np
import pydicom

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Permitir ejecutar este archivo de forma independiente (fuera de
# larmornium.py) asegurando que el paquete de indexadores sea importable.
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
RECENT_FOLDERS_CONFIG_FILENAME = "larmornium.conf"
MAX_RECENT_FOLDERS = 5

# Los archivos .db/.json generados y el archivo de configuración de
# carpetas recientes se guardan siempre en la raíz del proyecto, no en
# la carpeta del estudio DICOM seleccionado.
LARMORNIUM_FILES_DIR = os.path.join(_PROJECT_ROOT, LARMORNIUM_FILES_DIRNAME)
INDEXED_DIR = os.path.join(LARMORNIUM_FILES_DIR, INDEXED_DIRNAME)
COMBINED_DB_PATH = os.path.join(INDEXED_DIR, COMBINED_DB_FILENAME)
RECENT_FOLDERS_CONFIG_PATH = os.path.join(
    LARMORNIUM_FILES_DIR, RECENT_FOLDERS_CONFIG_FILENAME
)

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

# Modalidades DICOM que se calibran a unidades fisicas antes de mostrarse
MODALITY_CT = "CT"
MODALITY_PT = "PT"

# Ventaneos (window center, window width) en HU disponibles para CT
CT_WINDOW_PRESETS = [
    ("Tejido blando", 40, 400),
    ("Hueso", 400, 1800),
    ("Pulmón", -600, 1500),
    ("Cerebro", 40, 80),
    ("Mediastino", 50, 350),
]


# ---------------------------------------------------------------------------
# Carpetas recientemente abiertas (larmornium.conf en la raiz del proyecto)
# ---------------------------------------------------------------------------

class RecentFoldersStore:
    """Guarda en larmornium.conf la lista de directorios de estudios
    abiertos recientemente (los mas recientes primero, maximo
    MAX_RECENT_FOLDERS). Lee/escribe el archivo completo para no perder
    otras secciones (p. ej. "fusion_volumes") que gestionan otros modulos."""

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

    def add(self, directory):
        """Agrega el directorio al inicio de la lista (o lo mueve ahi si
        ya estaba), recorta a MAX_RECENT_FOLDERS y guarda el archivo."""
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


# ---------------------------------------------------------------------------
# Modelo de datos de la jerarquia (modalidad / paciente / estudio / serie)
# ---------------------------------------------------------------------------

@dataclass
class SeriesInfo:
    series_instance_uid: str
    series_description: str
    series_number: object
    modality: str
    num_images: int


@dataclass
class StudyInfo:
    study_instance_uid: str
    study_description: str
    study_date: str
    series_list: list = field(default_factory=list)


@dataclass
class PatientInfo:
    patient_id: str
    patient_name: str
    studies: list = field(default_factory=list)


@dataclass
class FusionPairInfo:
    ct_series_description: str
    pet_series_description: str
    num_slices: int
    slice_thickness: object


@dataclass
class FusionStudyInfo:
    study_instance_uid: str
    study_description: str
    study_date: str
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
    """Lee la jerarquia de estudios y las rutas de imagenes desde el .db
    combinado generado por los scripts de indexacion. No se recorre
    nunca el directorio de estudios directamente."""

    def __init__(self, dicom_root, db_path):
        self.dicom_root = dicom_root
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_modalities(self):
        """Retorna un dict {"PET_CT": [PatientInfo, ...], "MRI": [...]}"""
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

        studies_by_uid = {}
        rows = conn.execute(
            f"SELECT study_instance_uid, study_description, study_date, "
            f"patient_id, patient_name FROM {prefix}_studies"
        )
        for row in rows:
            pid = row["patient_id"] or "(sin identificador)"
            if pid not in patients_by_id:
                patients_by_id[pid] = PatientInfo(pid, row["patient_name"] or pid)
                patient_order.append(pid)

            study = StudyInfo(
                study_instance_uid=row["study_instance_uid"],
                study_description=row["study_description"] or "(sin descripcion)",
                study_date=row["study_date"] or "",
            )
            studies_by_uid[study.study_instance_uid] = study
            patients_by_id[pid].studies.append(study)

        frame_counts = self._count_frames_per_series(conn, prefix)

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
            study.series_list.append(SeriesInfo(
                series_instance_uid=series_uid,
                series_description=row["series_description"] or "(sin descripcion)",
                series_number=row["series_number"],
                modality=row["modality"] or "",
                num_images=num_images,
            ))

        return [patients_by_id[pid] for pid in patient_order if patients_by_id[pid].studies]

    def _count_frames_per_series(self, conn, prefix):
        """Conteod de imágenes se mostrarán realmente por serie, contando
        cada frame de los archivos DICOM multi-frame por separado. Este valor
        es el mismo que produce load_series_frames(), para que la etiqueta
        del arbol coincida siempre con el rango del deslizador."""
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
        """Retorna una lista de (ruta_absoluta, indice_de_frame) ordenada
        para recorrer todas las imagenes de una serie. indice_de_frame es
        None cuando el archivo contiene una sola imagen."""
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
        """Retorna los estudios fusionables (CT+PET) agrupados por
        paciente y estudio, leidos de pet_ct_fusion_pairs."""
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
                "SELECT fp.study_instance_uid, fp.ct_series_description, "
                "fp.pet_series_description, fp.num_slices, fp.slice_thickness, "
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
                study = FusionStudyInfo(
                    study_instance_uid=study_uid,
                    study_description=row["study_description"] or "(sin descripcion)",
                    study_date=row["study_date"] or "",
                )
                studies_by_key[study_key] = study
                patients_by_id[pid].studies.append(study)

            studies_by_key[study_key].pairs.append(FusionPairInfo(
                ct_series_description=row["ct_series_description"] or "(sin descripcion)",
                pet_series_description=row["pet_series_description"] or "(sin descripcion)",
                num_slices=row["num_slices"] or 0,
                slice_thickness=row["slice_thickness"],
            ))

        return [patients_by_id[pid] for pid in patient_order]

    def load_multi_study_patients(self):
        """Retorna los pacientes marcados como multi-estudio (is_multi_study=1)
        en cualquier modalidad, con sus directorios de estudio."""
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


# ---------------------------------------------------------------------------
# Indexacion en segundo plano con salida de log hacia la GUI
# ---------------------------------------------------------------------------

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


class FusionVolumeWorker(QObject):
    """Genera en un hilo separado los volumenes fusionados CT+PET (.nii)
    pendientes, usando join_pet_ct.ensure_fusion_volumes()."""

    log_message = Signal(str)
    finished = Signal(bool, int, str)

    def __init__(self, dicom_root, db_path, larmornium_files_dir, config_path):
        super().__init__()
        self.dicom_root = dicom_root
        self.db_path = db_path
        self.larmornium_files_dir = larmornium_files_dir
        self.config_path = config_path

    def run(self):
        try:
            built = join_pet_ct.ensure_fusion_volumes(
                self.dicom_root, self.db_path, self.larmornium_files_dir,
                self.config_path, progress_callback=self.log_message.emit,
            )
            self.finished.emit(True, built, "")
        except Exception as exc:
            self.finished.emit(False, 0, str(exc))


# ---------------------------------------------------------------------------
# Panel central: visor de imagenes de la serie seleccionada
# ---------------------------------------------------------------------------

class _HoverImageLabel(QLabel):
    """QLabel que reporta la posicion del cursor sobre la imagen para
    poder mostrar el valor HU/SUV del pixel bajo el mouse."""

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
    """Muestra la imagen actual de una serie y permite recorrerla con un
    deslizador cuando tiene mas de una imagen. Las series CT se calibran a
    Unidades Hounsfield (calcular_HU_CT) y admiten elegir un ventaneo; las
    series PET se calibran a SUV (calcular_SUV_PT). El resto de modalidades
    se muestran con un reescalado simple minimo-maximo a 8 bits."""

    PLACEHOLDER_TEXT = "Seleccione una serie para visualizarla"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = []
        self._modality = None
        self._cache_path = None
        self._cache_dataset = None
        self._cache_pixel_array = None
        self._cache_value_matrix = None
        self._current_pixmap = None
        self._current_value_matrix = None
        self._current_units_label = None
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

        self.window_label = QLabel("Ventaneo:")
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

        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.window_row)
        layout.addWidget(self.slider)
        layout.addWidget(self.position_label)
        layout.addWidget(self.value_label)

    def clear(self):
        self._frames = []
        self._modality = None
        self._cache_path = None
        self._cache_dataset = None
        self._cache_pixel_array = None
        self._cache_value_matrix = None
        self._current_pixmap = None
        self._current_value_matrix = None
        self._current_units_label = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(self.PLACEHOLDER_TEXT)
        self.slider.blockSignals(True)
        self.slider.setMaximum(1)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.position_label.setText("")
        self.value_label.setText("")
        self.window_row.setVisible(False)

    def show_series(self, frames, modality=None):
        if not frames:
            self.clear()
            self.image_label.setText("La serie seleccionada no tiene imagenes")
            return

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

        count = len(frames)
        self.slider.blockSignals(True)
        self.slider.setMinimum(1)
        self.slider.setMaximum(count)
        self.slider.setValue(1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(count > 1)
        self._display_frame(0)

    def _on_slider_changed(self, value):
        self._display_frame(value - 1)

    def _current_window(self):
        data = self.window_combo.currentData()
        if data is None:
            return CT_WINDOW_PRESETS[0][1], CT_WINDOW_PRESETS[0][2]
        return data

    def _on_window_changed(self, _index):
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
            # setText() ya limpia el pixmap anterior (QLabel solo puede
            # mostrar texto o pixmap a la vez).
            self.image_label.setText("Error al leer la imagen:\n%s" % exc)
            return

        self._current_value_matrix = value_matrix
        self._current_units_label = units_label
        image = self._array_to_qimage(display_array, is_rgb)
        self._current_pixmap = QPixmap.fromImage(image)
        self._update_pixmap_display()
        self.position_label.setText("Imagen %d / %d" % (index + 1, len(self._frames)))
        self.value_label.setText("")

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

        # Las series CT/PET (num_frames siempre None en pet_ct_images) se
        # calibran con las funciones de processing/ en lugar del reescalado
        # generico usado para el resto de modalidades.
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
        if self._current_value_matrix is None or self._current_units_label is None:
            return
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


# ---------------------------------------------------------------------------
# Panel izquierdo: seleccion de directorio, arbol de estudios y bitacora
# ---------------------------------------------------------------------------

class StudySelectionPanel(QWidget):
    """Panel con el boton de seleccion de directorio, la lista de carpetas
    recientes, el arbol de jerarquia de estudios y la bitacora de salida
    de los scripts."""

    directory_selected = Signal(str)
    recent_folder_selected = Signal(str)
    node_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.select_button = QPushButton("Seleccionar directorio de estudios...")
        self.select_button.clicked.connect(self._on_select_clicked)

        self.recent_label = QLabel("Carpetas recientes:")
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(100)
        self.recent_list.itemActivated.connect(self._on_recent_item_activated)
        self.recent_label.setVisible(False)
        self.recent_list.setVisible(False)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Estudios indexados"])
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)

        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.addWidget(self.select_button)
        tree_layout.addWidget(self.recent_label)
        tree_layout.addWidget(self.recent_list)
        tree_layout.addWidget(self.tree)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(tree_container)
        splitter.addWidget(self.log_output)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _on_select_clicked(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Seleccionar directorio de estudios"
        )
        if directory:
            self.directory_selected.emit(directory)

    def _on_recent_item_activated(self, item):
        directory = item.data(Qt.UserRole)
        if directory:
            self.recent_folder_selected.emit(directory)

    def set_recent_folders(self, folders):
        """Reconstruye la lista de carpetas recientes; se oculta si esta vacia."""
        self.recent_list.clear()
        for folder in folders:
            item = QListWidgetItem(folder)
            item.setData(Qt.UserRole, folder)
            self.recent_list.addItem(item)
        has_folders = bool(folders)
        self.recent_label.setVisible(has_folders)
        self.recent_list.setVisible(has_folders)

    def append_log(self, message):
        self.log_output.appendPlainText(message)

    def set_indexing_enabled(self, enabled):
        self.select_button.setEnabled(enabled)

    def populate_tree(self, modalities, fusion_patients=None, multi_study_patients=None,
                      built_study_uids=None):
        """Reconstruye el arbol a partir del dict {modalidad: [PatientInfo, ...]}
        y, si se proporcionan, agrega las categorias de estudios fusionables
        y de pacientes con multi-estudio al mismo nivel que las modalidades.
        built_study_uids es el conjunto de estudios con volumen fusionado ya
        construido (ver join_pet_ct.load_fully_built_study_uids)."""
        self.tree.clear()
        for modality, patients in modalities.items():
            prefix = MODALITY_PREFIXES[modality]
            modality_item = QTreeWidgetItem([MODALITY_LABELS.get(modality, modality)])
            modality_item.setData(0, Qt.UserRole, {
                "type": NODE_TYPE_MODALITY,
                "label": MODALITY_LABELS.get(modality, modality),
            })
            for patient in patients:
                patient_label = "%s (%s)" % (patient.patient_name, patient.patient_id)
                patient_item = QTreeWidgetItem([patient_label])
                patient_item.setData(0, Qt.UserRole, {
                    "type": NODE_TYPE_PATIENT,
                    "label": patient_label,
                })
                for study in patient.studies:
                    study_label = "%s (%s)" % (study.study_description, study.study_date)
                    study_item = QTreeWidgetItem([study_label])
                    study_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_STUDY,
                        "label": study_label,
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
                            "modality": series.modality,
                        })
                        study_item.addChild(series_item)
                    patient_item.addChild(study_item)
                modality_item.addChild(patient_item)
            self.tree.addTopLevelItem(modality_item)

        if fusion_patients:
            self.tree.addTopLevelItem(
                self._build_fusion_category_item(fusion_patients, built_study_uids)
            )

        if multi_study_patients:
            self.tree.addTopLevelItem(
                self._build_multi_study_category_item(multi_study_patients)
            )

        self.tree.expandToDepth(0)

    def _build_fusion_category_item(self, fusion_patients, built_study_uids=None):
        """Construye la categoria 'Estudios con corregistro' (solo lectura).
        Los estudios cuyo volumen fusionado ya fue construido se muestran con
        el prefijo "[ok]" en su nombre."""
        built_study_uids = built_study_uids or set()
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
            })
            for study in patient.studies:
                study_label = "%s (%s)" % (study.study_description, study.study_date)
                if study.study_instance_uid in built_study_uids:
                    study_label = "[ok] " + study_label
                study_item = QTreeWidgetItem([study_label])
                study_item.setData(0, Qt.UserRole, {
                    "type": NODE_TYPE_FUSION_STUDY,
                    "label": study_label,
                })
                for pair in study.pairs:
                    pair_label = "CT: %s + PET: %s (%d cortes)" % (
                        pair.ct_series_description, pair.pet_series_description,
                        pair.num_slices,
                    )
                    pair_item = QTreeWidgetItem([pair_label])
                    pair_item.setData(0, Qt.UserRole, {
                        "type": NODE_TYPE_FUSION_PAIR,
                        "label": pair_label,
                    })
                    study_item.addChild(pair_item)
                patient_item.addChild(study_item)
            category_item.addChild(patient_item)
        return category_item

    def _build_multi_study_category_item(self, multi_study_patients):
        """Construye la categoria 'Paciente con multi estudio' (solo lectura)."""
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


# ---------------------------------------------------------------------------
# Panel derecho: herramientas de analisis (vacio, oculto por defecto)
# ---------------------------------------------------------------------------

class ToolsPanel(QWidget):
    """Panel de herramientas de analisis. Vacio por ahora; el metodo
    add_tool() permite registrar nuevas herramientas en el futuro sin
    modificar el resto de la GUI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._placeholder = QLabel("No hay herramientas de analisis disponibles.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)

    def add_tool(self, widget):
        """Punto de extension para futuras herramientas de analisis."""
        self._placeholder.hide()
        self._layout.addWidget(widget)


# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Larmornium - Visualizacion PET/CT y MRI")
        self.resize(1400, 900)

        self.dicom_root = None
        self.data_access = None
        self._index_thread = None
        self._index_worker = None
        self._fusion_thread = None
        self._fusion_worker = None
        os.makedirs(LARMORNIUM_FILES_DIR, exist_ok=True)
        self.recent_store = RecentFoldersStore(RECENT_FOLDERS_CONFIG_PATH)

        self.left_panel = StudySelectionPanel()
        self.left_panel.directory_selected.connect(self._on_directory_selected)
        self.left_panel.recent_folder_selected.connect(self._on_recent_folder_selected)
        self.left_panel.node_selected.connect(self._on_node_selected)
        self.left_panel.set_recent_folders(self.recent_store.load())

        self.viewer = ImageViewer()

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
        select_action = file_menu.addAction("Seleccionar directorio de estudios...")
        select_action.triggered.connect(self.left_panel._on_select_clicked)

        self.reindex_action = file_menu.addAction("Reindexar directorio actual")
        self.reindex_action.setEnabled(False)
        self.reindex_action.triggered.connect(self._reindex_current_directory)

        view_menu = menu_bar.addMenu("Ver")
        view_menu.addAction(self.tools_dock.toggleViewAction())

    def _on_directory_selected(self, directory):
        self.dicom_root = directory
        self._remember_folder(directory)
        self._start_indexing(directory)

    def _on_recent_folder_selected(self, directory):
        if not os.path.isdir(directory):
            QMessageBox.warning(
                self, "Carpeta no encontrada",
                "La carpeta ya no existe:\n%s" % directory
            )
            self.left_panel.set_recent_folders(self.recent_store.remove(directory))
            return

        self.dicom_root = directory
        self._remember_folder(directory)

        if self._cached_index_matches(directory):
            self.left_panel.append_log("Abriendo índice existente: %s" % directory)
            self.reindex_action.setEnabled(True)
            self._load_index()
        else:
            self.left_panel.append_log(
                "No se encontró un índice previo para: %s" % directory
            )
            self._start_indexing(directory)

    def _cached_index_matches(self, directory):
        if not os.path.isfile(COMBINED_DB_PATH):
            return False
        source_dir = IndexDataAccess(directory, COMBINED_DB_PATH).load_source_dicom_dir()
        return source_dir is not None and os.path.abspath(source_dir) == os.path.abspath(directory)

    def _remember_folder(self, directory):
        folders = self.recent_store.add(directory)
        self.left_panel.set_recent_folders(folders)

    def _reindex_current_directory(self):
        if self.dicom_root:
            self._start_indexing(self.dicom_root)

    def _start_indexing(self, directory):
        os.makedirs(INDEXED_DIR, exist_ok=True)

        self.left_panel.set_indexing_enabled(False)
        self.reindex_action.setEnabled(False)
        self.left_panel.append_log("Indexando directorio: %s" % directory)

        self._index_thread = QThread(self)
        self._index_worker = IndexWorker(directory, INDEXED_DIR)
        self._index_worker.moveToThread(self._index_thread)

        self._index_thread.started.connect(self._index_worker.run)
        self._index_worker.log_message.connect(self.left_panel.append_log)
        self._index_worker.finished.connect(self._on_indexing_finished)
        self._index_worker.finished.connect(self._index_thread.quit)
        self._index_thread.finished.connect(self._index_thread.deleteLater)

        self._index_thread.start()

    def _on_indexing_finished(self, success, error_message):
        self.left_panel.set_indexing_enabled(True)
        self.reindex_action.setEnabled(self.dicom_root is not None)

        if not success:
            self.left_panel.append_log("Error durante la indexación: %s" % error_message)
            QMessageBox.critical(self, "Error de indexación", error_message)
            return

        self.left_panel.append_log("Indexación completada.")
        self._load_index()

    def _load_index(self):
        if not os.path.isfile(COMBINED_DB_PATH):
            self.left_panel.append_log(
                "No se encontró el archivo de indexado: %s" % COMBINED_DB_PATH
            )
            return

        self.data_access = IndexDataAccess(self.dicom_root, COMBINED_DB_PATH)
        self._populate_tree()
        self.viewer.clear()
        self._start_fusion_volume_build()

    def _populate_tree(self):
        modalities = self.data_access.load_modalities()
        fusion_patients = self.data_access.load_fusion_pairs()
        multi_study_patients = self.data_access.load_multi_study_patients()
        built_study_uids = join_pet_ct.load_fully_built_study_uids(
            COMBINED_DB_PATH, RECENT_FOLDERS_CONFIG_PATH
        )
        self.left_panel.populate_tree(
            modalities, fusion_patients, multi_study_patients, built_study_uids
        )
        self.left_panel.append_log(
            "Árbol de estudios cargado (%d modalidad(es))." % len(modalities)
        )

    def _start_fusion_volume_build(self):
        self._fusion_thread = QThread(self)
        self._fusion_worker = FusionVolumeWorker(
            self.dicom_root, COMBINED_DB_PATH, LARMORNIUM_FILES_DIR,
            RECENT_FOLDERS_CONFIG_PATH,
        )
        self._fusion_worker.moveToThread(self._fusion_thread)

        self._fusion_thread.started.connect(self._fusion_worker.run)
        self._fusion_worker.log_message.connect(self.left_panel.append_log)
        self._fusion_worker.finished.connect(self._on_fusion_build_finished)
        self._fusion_worker.finished.connect(self._fusion_thread.quit)
        self._fusion_thread.finished.connect(self._fusion_thread.deleteLater)

        self._fusion_thread.start()

    def _on_fusion_build_finished(self, success, built_count, error_message):
        if not success:
            self.left_panel.append_log(
                "Error al generar volúmenes fusionados: %s" % error_message
            )
            return
        if built_count:
            self.left_panel.append_log(
                "Volúmenes fusionados generados: %d" % built_count
            )
            self._populate_tree()
        else:
            self.left_panel.append_log("Los volúmenes fusionados ya estaban al día.")

    def _on_node_selected(self, data):
        node_type = data.get("type")
        label = data.get("label", "")

        if node_type == NODE_TYPE_SERIES:
            self.left_panel.append_log("Cargando serie: %s ..." % label)
            self._load_series(
                data["prefix"], data["series_instance_uid"], data.get("modality"), label
            )
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
