#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Módulo para descargar estudios DICOM desde bases de datos públicas.

Soporta dos fuentes:
  - TCIA (The Cancer Imaging Archive): API REST pública sin autenticación.
  - Colecciones Recomendadas: catálogo curado de colecciones TCIA organizadas
    por modalidad (PET/CT y MRI) para acceso rápido.
"""
import io
import os
import re
import zipfile
from functools import lru_cache

import requests
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_GUI_DIR)
ICON_DIR = os.path.join(_GUI_DIR, "icon")
ICON_DICOM_FILE_PATH = os.path.join(ICON_DIR, "dicom_file.png")
ICON_PET_CT_PATH = os.path.join(ICON_DIR, "pet_ct_img.png")
ICON_MRI_PATH = os.path.join(ICON_DIR, "mri_img.png")
ICON_CT_PATH = os.path.join(ICON_DIR, "ct_img.png")
ICON_PET_PATH = os.path.join(ICON_DIR, "pet_img.png")
LARMORNIUM_FILES_DIR = os.path.join(_PROJECT_ROOT, "larmornium_files")
DOWNLOAD_BASE_DIR = os.path.join(LARMORNIUM_FILES_DIR, "downloaded_dicom")

TCIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

CURATED_COLLECTIONS = {
    "PET_CT": [
        {"collection": "Lung-PET-CT-Dx", "desc": "Lung PET/CT Diagnosis"},
        {"collection": "RIDER Lung PET-CT", "desc": "RIDER Lung PET-CT"},
        {"collection": "FDG-PET-CT-Lesions", "desc": "FDG PET/CT Lesions"},
        {"collection": "ACRIN-HNSCC-FDG-PET-CT", "desc": "ACRIN Head Neck PET/CT"},
        {"collection": "ACRIN-NSCLC-FDG-PET", "desc": "ACRIN NSCLC FDG PET"},
        {"collection": "RIDER PHANTOM PET-CT", "desc": "RIDER Phantom PET/CT"},
        {"collection": "QIN PET Phantom", "desc": "QIN PET Phantom"},
        {"collection": "NaF PROSTATE", "desc": "NaF Prostate PET/CT"},
        {"collection": "PSMA-PET-CT-Lesions", "desc": "PSMA PET/CT Lesions"},
        {"collection": "CT-vs-PET-Ventilation-Imaging", "desc": "CT vs PET Ventilation"},
    ],
    "MRI": [
        {"collection": "Duke-Breast-Cancer-MRI", "desc": "Duke Breast Cancer MRI"},
        {"collection": "PROSTATE-MRI", "desc": "Prostate MRI"},
        {"collection": "RIDER Breast MRI", "desc": "RIDER Breast MRI"},
        {"collection": "RIDER PHANTOM MRI", "desc": "RIDER Phantom MRI"},
        {"collection": "QIN Breast DCE-MRI", "desc": "QIN Breast DCE-MRI"},
        {"collection": "QIN-BREAST", "desc": "QIN Breast MRI"},
        {"collection": "ACRIN-Contralateral-Breast-MR", "desc": "ACRIN Contralateral Breast MR"},
        {"collection": "ACRIN-FLT-Breast", "desc": "ACRIN FLT Breast"},
        {"collection": "ISPY1", "desc": "I-SPY 1 Breast MRI"},
        {"collection": "Prostate-3T", "desc": "Prostate 3T MRI"},
        {"collection": "PROSTATEx", "desc": "PROSTATEx MRI"},
        {"collection": "Vestibular-Schwannoma-SEG", "desc": "Vestibular Schwannoma MRI"},
        {"collection": "UPENN-GBM", "desc": "UPenn GBM Brain MRI"},
        {"collection": "ReMIND", "desc": "ReMIND Brain MRI"},
        {"collection": "ISPY2", "desc": "I-SPY 2 Breast MRI"},
    ],
}

ITEMS_PER_PAGE = 20


def _format_file_size(size_bytes):
    if not size_bytes:
        return "0 B"
    try:
        size_bytes = int(size_bytes)
    except (ValueError, TypeError):
        return "0 B"
    if size_bytes < 1024:
        return "%d B" % size_bytes
    elif size_bytes < 1024 * 1024:
        return "%.1f KB" % (size_bytes / 1024)
    elif size_bytes < 1024 * 1024 * 1024:
        return "%.1f MB" % (size_bytes / (1024 * 1024))
    return "%.2f GB" % (size_bytes / (1024 * 1024 * 1024))


def _sanitize_dirname(name):
    name = re.sub(r'[<>:"/\\|?*]', "_", str(name))
    name = re.sub(r"\s+", "_", name.strip())
    return name[:120] if name else "sin_nombre"


def _group_series_into_studies(series_list):
    studies_map = {}
    for s in series_list:
        suid = s.get("StudyInstanceUID") or s.get("SeriesInstanceUID") or "desconocido"
        if suid not in studies_map:
            studies_map[suid] = {
                "StudyInstanceUID": suid,
                "PatientID": s.get("PatientID", "N/A"),
                "StudyDesc": s.get("StudyDesc") or s.get("SeriesDescription") or "Estudio DICOM",
                "StudyDate": s.get("StudyDate", ""),
                "Collection": s.get("Collection", ""),
                "Manufacturer": s.get("Manufacturer", ""),
                "BodyPartExamined": s.get("BodyPartExamined", ""),
                "series": [],
            }
        studies_map[suid]["series"].append(s)

    study_items = []
    for study in studies_map.values():
        study["series"].sort(key=lambda x: int(x.get("SeriesNumber", 0) or 0))
        modalities = sorted(set(s.get("Modality", "") for s in study["series"] if s.get("Modality")))
        study["modalities"] = modalities
        study["total_images"] = sum(int(s.get("ImageCount", 0) or 0) for s in study["series"])
        study["total_size"] = sum(int(s.get("FileSize", 0) or 0) for s in study["series"])

        is_pet_ct = ("PT" in modalities and "CT" in modalities) or ("FUSION" in modalities)
        study["is_pet_ct"] = is_pet_ct

        if is_pet_ct:
            study["modality_label"] = "PET/CT"
            study["modality_dir"] = "PET_CT"
        elif "PT" in modalities:
            study["modality_label"] = "PET"
            study["modality_dir"] = "PET_CT"
        elif "CT" in modalities:
            study["modality_label"] = "CT"
            study["modality_dir"] = "PET_CT"
        elif "MR" in modalities:
            study["modality_label"] = "MRI"
            study["modality_dir"] = "MRI"
        else:
            study["modality_label"] = "/".join(modalities) if modalities else "DICOM"
            study["modality_dir"] = "OTHER"

        study_items.append(study)

    return study_items


class TCIAClient:
    def __init__(self, base_url=TCIA_BASE_URL, timeout=30):
        self._base_url = base_url
        self._timeout = timeout

    def _get_json(self, endpoint, params=None):
        url = "%s/%s" % (self._base_url, endpoint)
        resp = requests.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def get_collections(self):
        data = self._get_json("getCollectionValues")
        return sorted([item["Collection"] for item in data if "Collection" in item])

    def get_series(self, collection, modality=None):
        params = {"Collection": collection}
        if modality:
            params["Modality"] = modality
        return self._get_json("getSeries", params)

    def get_series_by_study(self, study_uid):
        return self._get_json("getSeries", {"StudyInstanceUID": study_uid})

    def download_series(self, series_uid, dest_dir, progress_callback=None):
        url = "%s/getImage" % self._base_url
        params = {"SeriesInstanceUID": series_uid}
        os.makedirs(dest_dir, exist_ok=True)

        with requests.get(url, params=params, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            buffer = io.BytesIO()
            downloaded = 0
            chunk_size = 64 * 1024

            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    buffer.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)

        buffer.seek(0)
        with zipfile.ZipFile(buffer) as zf:
            zf.extractall(dest_dir)

        return dest_dir


class FetchWorker(QObject):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, fetch_fn, *args):
        super().__init__()
        self._fetch_fn = fetch_fn
        self._args = args

    @Slot()
    def run(self):
        try:
            result = self._fetch_fn(*self._args)
            self.finished.emit(result if isinstance(result, list) else [])
        except Exception as exc:
            self.error.emit(str(exc))


class DownloadWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    log_message = Signal(str)

    def __init__(self, client, series_list, dest_dir, study_label):
        super().__init__()
        self._client = client
        self._series_list = series_list
        self._dest_dir = dest_dir
        self._study_label = study_label

    @Slot()
    def run(self):
        try:
            total_series = len(self._series_list)
            self.log_message.emit(
                "Descargando estudio: %s (%d %s)..."
                % (self._study_label, total_series, "serie" if total_series == 1 else "series")
            )
            os.makedirs(self._dest_dir, exist_ok=True)

            for idx, s in enumerate(self._series_list, 1):
                series_uid = s.get("SeriesInstanceUID")
                if not series_uid:
                    continue
                modality = s.get("Modality", "DICOM")
                series_num = str(s.get("SeriesNumber", idx) or idx)
                img_count = int(s.get("ImageCount", 0) or 0)
                series_desc = s.get("SeriesDescription", "")

                sub_name = _sanitize_dirname("%s_%s" % (modality, series_num))
                series_dest = os.path.join(self._dest_dir, sub_name)

                existing = (
                    [f for f in os.listdir(series_dest) if not f.startswith(".")]
                    if os.path.isdir(series_dest)
                    else []
                )
                if existing:
                    self.log_message.emit(
                        "  [%d/%d] %s (serie #%s) ya descargada (%d archivos)."
                        % (idx, total_series, modality, series_num, len(existing))
                    )
                    continue

                desc_info = " - %s" % series_desc if series_desc else ""
                self.log_message.emit(
                    "  [%d/%d] Descargando %s (serie #%s%s, %d img)..."
                    % (idx, total_series, modality, series_num, desc_info, img_count)
                )

                last_pct = [-1]

                def on_progress(downloaded, total):
                    if total > 0:
                        pct = int(downloaded * 100 / total)
                        if pct - last_pct[0] >= 25 or pct == 100:
                            last_pct[0] = pct
                            self.log_message.emit(
                                "    Progreso [%s]: %s / %s (%d%%)"
                                % (modality, _format_file_size(downloaded), _format_file_size(total), pct)
                            )

                self._client.download_series(
                    series_uid, series_dest, progress_callback=on_progress
                )

            self.finished.emit(self._dest_dir)
        except Exception as exc:
            self.error.emit(str(exc))


@lru_cache(maxsize=8)
def _get_scaled_icon_pixmap(icon_path, size=36):
    if os.path.isfile(icon_path):
        pm = QPixmap(icon_path)
        if not pm.isNull():
            return pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    return pm


def _get_modality_icon_path(study_data):
    if study_data.get("is_pet_ct"):
        return ICON_PET_CT_PATH
    modalities = study_data.get("modalities", [])
    mod_dir = study_data.get("modality_dir", "")
    mod_label = study_data.get("modality_label", "")

    if "PT" in modalities and "CT" in modalities:
        return ICON_PET_CT_PATH
    if "MR" in modalities or mod_dir == "MRI" or mod_label == "MRI":
        return ICON_MRI_PATH
    if "CT" in modalities or mod_label == "CT":
        return ICON_CT_PATH
    if "PT" in modalities or mod_label == "PET":
        return ICON_PET_PATH
    return ICON_DICOM_FILE_PATH


class StudyListItemWidget(QWidget):
    def __init__(self, study_data, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        self._lbl_icon = QLabel()
        self._lbl_icon.setFixedSize(36, 36)
        self._lbl_icon.setAlignment(Qt.AlignCenter)
        icon_path = _get_modality_icon_path(study_data)
        self._lbl_icon.setPixmap(_get_scaled_icon_pixmap(icon_path, 36))
        layout.addWidget(self._lbl_icon, 0, Qt.AlignVCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        pid = (study_data.get("PatientID") or "").strip()
        sdesc = (study_data.get("StudyDesc") or "").strip()
        if pid and sdesc and sdesc != "Estudio DICOM" and sdesc != pid:
            title = "%s - %s" % (pid, sdesc)
        elif pid:
            title = pid
        else:
            title = sdesc or "Estudio DICOM"

        self._lbl_title = QLabel(title)
        self._lbl_title.setStyleSheet("font-weight: 600; font-size: 11px;")
        text_layout.addWidget(self._lbl_title)

        mod_label = study_data.get("modality_label", "DICOM")
        num_series = len(study_data.get("series", []))
        total_images = study_data.get("total_images", 0)
        s_str = "1 serie" if num_series == 1 else "%d series" % num_series
        c_str = "1 corte" if total_images == 1 else "%d cortes" % total_images
        info_text = "%s | %s y %s" % (mod_label, s_str, c_str)

        self._lbl_info = QLabel(info_text)
        self._lbl_info.setStyleSheet("font-size: 10px;")
        text_layout.addWidget(self._lbl_info)

        layout.addLayout(text_layout, 1)


class DicomDownloaderWidget(QWidget):
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._client = TCIAClient()
        self._all_studies = []
        self._filtered_studies = []
        self._displayed_count = 0
        self._selected_data = None
        self._fetch_thread = None
        self._fetch_worker = None
        self._download_thread = None
        self._download_worker = None
        self._is_downloading = False
        self._current_filter_text = ""

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        src_layout = QHBoxLayout()
        src_layout.setContentsMargins(0, 0, 0, 0)
        src_layout.addWidget(QLabel("Fuente:"))
        self.cmb_source = QComboBox()
        self.cmb_source.addItem("TCIA (Cancer Imaging Archive)", "tcia")
        self.cmb_source.addItem("Colecciones Recomendadas", "curated")
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        src_layout.addWidget(self.cmb_source, 1)
        main_layout.addLayout(src_layout)

        col_layout = QHBoxLayout()
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.addWidget(QLabel("Colección:"))
        self.cmb_collection = QComboBox()
        self.cmb_collection.setMinimumWidth(120)
        self.cmb_collection.currentIndexChanged.connect(self._on_collection_changed)
        col_layout.addWidget(self.cmb_collection, 1)
        main_layout.addLayout(col_layout)

        mod_layout = QHBoxLayout()
        mod_layout.setContentsMargins(0, 0, 0, 0)
        mod_layout.addWidget(QLabel("Modalidad:"))
        self.cmb_modality = QComboBox()
        self.cmb_modality.addItem("Todas", None)
        self.cmb_modality.addItem("PET/CT", "PET_CT")
        self.cmb_modality.addItem("MR (MRI)", "MR")
        self.cmb_modality.addItem("CT", "CT")
        self.cmb_modality.addItem("PT (PET)", "PT")
        self.cmb_modality.currentIndexChanged.connect(self._on_modality_changed)
        mod_layout.addWidget(self.cmb_modality, 1)
        self._mod_layout_widget = QWidget()
        self._mod_layout_widget.setLayout(mod_layout)
        main_layout.addWidget(self._mod_layout_widget)

        cur_mod_layout = QHBoxLayout()
        cur_mod_layout.setContentsMargins(0, 0, 0, 0)
        cur_mod_layout.addWidget(QLabel("Tipo:"))
        self.cmb_curated_modality = QComboBox()
        self.cmb_curated_modality.addItem("PET/CT", "PET_CT")
        self.cmb_curated_modality.addItem("MRI", "MRI")
        self.cmb_curated_modality.currentIndexChanged.connect(self._on_curated_modality_changed)
        cur_mod_layout.addWidget(self.cmb_curated_modality, 1)
        self._cur_mod_widget = QWidget()
        self._cur_mod_widget.setLayout(cur_mod_layout)
        self._cur_mod_widget.setVisible(False)
        main_layout.addWidget(self._cur_mod_widget)

        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Filtrar por paciente, descripción...")
        self.txt_search.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self.txt_search)
        self.btn_search_clear = QPushButton("X")
        self.btn_search_clear.setFixedWidth(28)
        self.btn_search_clear.clicked.connect(lambda: self.txt_search.clear())
        search_layout.addWidget(self.btn_search_clear)
        main_layout.addLayout(search_layout)

        self.btn_load = QPushButton("Cargar estudios")
        self.btn_load.clicked.connect(self._on_load_clicked)
        main_layout.addWidget(self.btn_load)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 11px;")
        main_layout.addWidget(self.lbl_status)

        self.list_studies = QListWidget()
        self.list_studies.setSelectionMode(QListWidget.SingleSelection)
        self.list_studies.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.list_studies.itemClicked.connect(self._on_item_clicked)
        self.list_studies.currentItemChanged.connect(self._on_current_item_changed)
        main_layout.addWidget(self.list_studies, 1)

        self.txt_detail = QTextEdit()
        self.txt_detail.setReadOnly(True)
        self.txt_detail.setFixedHeight(120)
        self.txt_detail.setPlainText("Seleccione un estudio de la lista para ver detalles.")
        main_layout.addWidget(self.txt_detail)

        self.btn_download = QPushButton("Descargar estudio")
        self.btn_download.setEnabled(False)
        self.btn_download.clicked.connect(self._on_download_clicked)
        main_layout.addWidget(self.btn_download)

        self._on_source_changed(0)

    def _on_source_changed(self, _index):
        source = self.cmb_source.currentData()
        self.list_studies.clear()
        self._all_studies.clear()
        self._filtered_studies.clear()
        self._displayed_count = 0
        self._clear_selection()
        self.lbl_status.setText("")

        if source == "tcia":
            self._mod_layout_widget.setVisible(True)
            self._cur_mod_widget.setVisible(False)
            self._load_tcia_collections()
        else:
            self._mod_layout_widget.setVisible(False)
            self._cur_mod_widget.setVisible(True)
            self._load_curated_collections()

    def _load_tcia_collections(self):
        self.cmb_collection.blockSignals(True)
        self.cmb_collection.clear()
        self.lbl_status.setText("Cargando colecciones de TCIA...")
        self.cmb_collection.blockSignals(False)

        self._start_fetch(
            self._client.get_collections,
            self._on_collections_loaded
        )

    def _on_collections_loaded(self, collections):
        self.cmb_collection.blockSignals(True)
        self.cmb_collection.clear()
        for col_name in collections:
            self.cmb_collection.addItem(col_name, col_name)
        self.cmb_collection.blockSignals(False)
        count = self.cmb_collection.count()
        self.lbl_status.setText("%d colecciones disponibles" % count)

    def _load_curated_collections(self):
        modality_key = self.cmb_curated_modality.currentData() or "PET_CT"
        entries = CURATED_COLLECTIONS.get(modality_key, [])
        self.cmb_collection.blockSignals(True)
        self.cmb_collection.clear()
        for entry in entries:
            self.cmb_collection.addItem(entry["desc"], entry["collection"])
        self.cmb_collection.blockSignals(False)
        self.lbl_status.setText(
            "%d colecciones recomendadas (%s)" % (len(entries), modality_key.replace("_", "/"))
        )

    def _on_curated_modality_changed(self, _index):
        self.list_studies.clear()
        self._all_studies.clear()
        self._filtered_studies.clear()
        self._displayed_count = 0
        self._clear_selection()
        self._load_curated_collections()

    def _on_collection_changed(self, _index):
        self.list_studies.clear()
        self._all_studies.clear()
        self._filtered_studies.clear()
        self._displayed_count = 0
        self._clear_selection()
        self.lbl_status.setText("")

    def _on_modality_changed(self, _index):
        self.list_studies.clear()
        self._all_studies.clear()
        self._filtered_studies.clear()
        self._displayed_count = 0
        self._clear_selection()
        self.lbl_status.setText("")

    def _on_search_changed(self, text):
        self._current_filter_text = text.strip().lower()
        self._apply_filter()

    def _apply_filter(self):
        self.list_studies.clear()
        self._displayed_count = 0

        if self._current_filter_text:
            filtered = [
                s for s in self._all_studies
                if self._matches_filter(s, self._current_filter_text)
            ]
        else:
            filtered = self._all_studies

        self._filtered_studies = filtered
        self._load_next_page()

    def _matches_filter(self, study_data, text):
        fields = [
            study_data.get("PatientID", ""),
            study_data.get("StudyDesc", ""),
            study_data.get("modality_label", ""),
            study_data.get("Collection", ""),
            study_data.get("Manufacturer", ""),
            study_data.get("BodyPartExamined", ""),
        ]
        for s in study_data.get("series", []):
            fields.append(s.get("SeriesDescription", ""))
            fields.append(s.get("Modality", ""))
        combined = " ".join(str(f) for f in fields).lower()
        return text in combined

    def _on_load_clicked(self):
        collection = self.cmb_collection.currentData()
        if not collection:
            self.lbl_status.setText("Seleccione una colección primero.")
            return

        self.list_studies.clear()
        self._all_studies.clear()
        self._filtered_studies.clear()
        self._displayed_count = 0
        self._clear_selection()

        source = self.cmb_source.currentData()
        modality = None
        if source == "tcia":
            mod_data = self.cmb_modality.currentData()
            if mod_data != "PET_CT":
                modality = mod_data

        self.lbl_status.setText("Cargando estudios de %s..." % collection)
        self.btn_load.setEnabled(False)

        def fetch_fn():
            return self._client.get_series(collection, modality)

        self._start_fetch(fetch_fn, self._on_series_loaded)

    def _on_series_loaded(self, series_list):
        self.btn_load.setEnabled(True)
        studies = _group_series_into_studies(series_list)

        source = self.cmb_source.currentData()
        if source == "tcia" and self.cmb_modality.currentData() == "PET_CT":
            studies = [s for s in studies if "PT" in s["modalities"] or "CT" in s["modalities"]]
            studies.sort(key=lambda s: 0 if s.get("is_pet_ct") else 1)

        self._all_studies = studies
        self._filtered_studies = studies
        self._current_filter_text = self.txt_search.text().strip().lower()
        if self._current_filter_text:
            self._apply_filter()
        else:
            self._displayed_count = 0
            self._load_next_page()

        total_series = len(series_list)
        pet_ct_count = sum(1 for s in studies if s.get("is_pet_ct"))
        if pet_ct_count > 0:
            self.lbl_status.setText(
                "%d estudios (%d con PET/CT integrado, %d series)"
                % (len(studies), pet_ct_count, total_series)
            )
        else:
            self.lbl_status.setText(
                "%d estudios (%d series)" % (len(studies), total_series)
            )

    def _load_next_page(self):
        source = getattr(self, "_filtered_studies", self._all_studies)
        start = self._displayed_count
        end = min(start + ITEMS_PER_PAGE, len(source))

        if start >= len(source):
            return

        for i in range(start, end):
            study_data = source[i]
            item = QListWidgetItem()
            item.setData(Qt.UserRole, study_data)
            row_widget = StudyListItemWidget(study_data)
            item.setSizeHint(row_widget.sizeHint())
            self.list_studies.addItem(item)
            self.list_studies.setItemWidget(item, row_widget)

        self._displayed_count = end

        total = len(source)
        self.lbl_status.setText(
            "Mostrando %d de %d estudios" % (self._displayed_count, total)
        )

    def _on_scroll(self, value):
        sb = self.list_studies.verticalScrollBar()
        if sb.maximum() == 0:
            return
        if value >= sb.maximum() - 50:
            self._load_next_page()

    def _on_item_clicked(self, item):
        if item is not None:
            data = item.data(Qt.UserRole)
            if data:
                self._select_study(data)

    def _on_current_item_changed(self, current, _previous):
        if current is not None:
            data = current.data(Qt.UserRole)
            if data:
                self._select_study(data)

    def _select_study(self, study_data):
        self._selected_data = study_data
        self._update_detail_panel(study_data)
        self.btn_download.setEnabled(not self._is_downloading)

    def _clear_selection(self):
        self._selected_data = None
        self.list_studies.clearSelection()
        self.list_studies.setCurrentItem(None)
        self.btn_download.setEnabled(False)
        self.txt_detail.setPlainText("Seleccione un estudio de la lista para ver detalles.")

    def _update_detail_panel(self, study_data):
        patient = study_data.get("PatientID", "N/A")
        study_desc = study_data.get("StudyDesc", "N/A")
        mod_label = study_data.get("modality_label", "N/A")
        total_images = study_data.get("total_images", 0)
        total_size = study_data.get("total_size", 0)
        series_list = study_data.get("series", [])
        num_series = len(series_list)
        collection = study_data.get("Collection", "N/A")
        date = study_data.get("StudyDate", "N/A")
        manufacturer = study_data.get("Manufacturer", "N/A")
        body_part = study_data.get("BodyPartExamined", "N/A")
        is_pet_ct = study_data.get("is_pet_ct", False)

        if is_pet_ct:
            pt_series = [s for s in series_list if s.get("Modality") == "PT"]
            ct_series = [s for s in series_list if s.get("Modality") == "CT"]
            pt_imgs = sum(int(s.get("ImageCount", 0) or 0) for s in pt_series)
            ct_imgs = sum(int(s.get("ImageCount", 0) or 0) for s in ct_series)
            mod_info = "PET/CT integrado (PET: %d ser/%d img | CT: %d ser/%d img)" % (
                len(pt_series), pt_imgs, len(ct_series), ct_imgs
            )
        else:
            mod_info = mod_label

        lines = [
            "%s - %s" % (patient, study_desc),
            "Colección: %s" % collection,
            "Modalidad: %s" % mod_info,
            "Total de series: %d | Cortes/Imágenes: %d" % (num_series, total_images),
        ]
        if total_size:
            lines.append("Tamaño total: %s" % _format_file_size(total_size))
        if date or body_part:
            lines.append("Fecha: %s | Región: %s" % (date, body_part))
        if manufacturer:
            lines.append("Fabricante: %s" % manufacturer)

        lines.append("Series incluidas:")
        for s in series_list[:8]:
            s_num = s.get("SeriesNumber", "?")
            s_mod = s.get("Modality", "")
            s_desc = s.get("SeriesDescription", "")
            s_img = s.get("ImageCount", 0)
            s_size = s.get("FileSize", 0)
            sz_str = " (%s)" % _format_file_size(s_size) if s_size else ""
            lines.append("  - [#%s] %s: %s (%d cortes%s)" % (s_num, s_mod, s_desc, s_img, sz_str))

        if len(series_list) > 8:
            lines.append("  - ... y %d series más" % (len(series_list) - 8))

        self.txt_detail.setPlainText("\n".join(l for l in lines if l))

    def _on_download_clicked(self):
        if self._selected_data is None or self._is_downloading:
            return

        study = self._selected_data
        series_list = list(study.get("series", []))
        study_uid = study.get("StudyInstanceUID")

        if not series_list and study_uid:
            try:
                series_list = self._client.get_series_by_study(study_uid)
            except Exception as e:
                self.log_message.emit("Error al obtener series: %s" % e)
                return

        if study_uid and len(study.get("modalities", [])) == 1 and study["modalities"][0] in ("PT", "CT"):
            try:
                full_series = self._client.get_series_by_study(study_uid)
                if full_series and len(full_series) > len(series_list):
                    series_list = full_series
                    study["series"] = full_series
                    modalities = sorted(set(s.get("Modality", "") for s in full_series if s.get("Modality")))
                    study["modalities"] = modalities
                    if "PT" in modalities and "CT" in modalities:
                        study["is_pet_ct"] = True
                        study["modality_label"] = "PET/CT"
                        study["modality_dir"] = "PET_CT"
            except Exception:
                pass

        if not series_list:
            self.log_message.emit("Error: No hay series disponibles para descargar.")
            return

        modality_dir = study.get("modality_dir", "PET_CT")
        patient_id = study.get("PatientID", "desconocido")
        collection = study.get("Collection", "")
        study_date = study.get("StudyDate", "")

        raw_name = (
            "%s_%s_%s" % (collection, patient_id, study_date)
            if study_date
            else "%s_%s" % (collection, patient_id)
        )
        study_name = _sanitize_dirname(raw_name)
        dest_dir = os.path.join(DOWNLOAD_BASE_DIR, modality_dir, study_name)

        all_downloaded = True
        for s in series_list:
            s_mod = s.get("Modality", "DICOM")
            s_num = str(s.get("SeriesNumber", 0) or 0)
            sub = os.path.join(dest_dir, _sanitize_dirname("%s_%s" % (s_mod, s_num)))
            if not os.path.isdir(sub) or not [f for f in os.listdir(sub) if not f.startswith(".")]:
                all_downloaded = False
                break

        if all_downloaded and os.path.isdir(dest_dir):
            self.log_message.emit("El estudio ya se encuentra completamente descargado en: %s" % dest_dir)
            return

        label = "%s (%s - %s)" % (
            patient_id,
            study.get("modality_label", "DICOM"),
            study.get("StudyDesc", ""),
        )
        self._start_download(series_list, dest_dir, label)

    def _start_download(self, series_list, dest_dir, label):
        self._is_downloading = True
        self.btn_download.setEnabled(False)
        self.btn_download.setText("Descargando...")

        self._download_thread = QThread()
        self._download_worker = DownloadWorker(
            self._client, series_list, dest_dir, label
        )
        self._download_worker.moveToThread(self._download_thread)

        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.log_message.connect(self.log_message.emit)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_worker.error.connect(self._download_thread.quit)
        self._download_thread.finished.connect(self._cleanup_download_thread)

        self._download_thread.start()

    def _on_download_finished(self, path):
        self._is_downloading = False
        self.btn_download.setEnabled(True)
        self.btn_download.setText("Descargar estudio")
        self.log_message.emit(
            "Descarga completada con éxito.\n"
            "Estudio guardado en: %s\n"
            "Para visualizarlo, indexe la carpeta de descargas manualmente." % path
        )

    def _on_download_error(self, error_msg):
        self._is_downloading = False
        self.btn_download.setEnabled(True)
        self.btn_download.setText("Descargar estudio")
        self.log_message.emit("Error en la descarga: %s" % error_msg)

    def _cleanup_download_thread(self):
        self._download_thread = None
        self._download_worker = None

    def _start_fetch(self, fetch_fn, on_success):
        if self._fetch_thread is not None:
            try:
                self._fetch_thread.quit()
                self._fetch_thread.wait(500)
            except Exception:
                pass

        self._fetch_thread = QThread()
        self._fetch_worker = FetchWorker(fetch_fn)
        self._fetch_worker.moveToThread(self._fetch_thread)

        self._fetch_thread.started.connect(self._fetch_worker.run)
        self._fetch_worker.finished.connect(on_success)
        self._fetch_worker.error.connect(self._on_fetch_error)
        self._fetch_worker.finished.connect(self._fetch_thread.quit)
        self._fetch_worker.error.connect(self._fetch_thread.quit)
        self._fetch_thread.finished.connect(self._cleanup_fetch_thread)

        self._fetch_thread.start()

    def _on_fetch_error(self, error_msg):
        self.btn_load.setEnabled(True)
        self.lbl_status.setText("Error: %s" % error_msg)
        self.log_message.emit("Error al consultar la API: %s" % error_msg)

    def _cleanup_fetch_thread(self):
        self._fetch_thread = None
        self._fetch_worker = None

    def closeEvent(self, event):
        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            self._fetch_thread.quit()
            self._fetch_thread.wait(500)
        if self._download_thread is not None and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(500)
        super().closeEvent(event)
