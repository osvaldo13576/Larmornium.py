#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger("gen_volume_MRI")

IGNORE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
IGNORE_PREFIXES = ("._",)
NON_DICOM_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".pdf", ".fig", ".tif",
    ".tiff", ".mat", ".xlsx", ".csv", ".txt",
}


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


def _safe_get(item, tag_or_name, default=None):
    if item is None:
        return default
    val = getattr(item, tag_or_name, None)
    if val is None:
        try:
            val = item[tag_or_name].value
        except (KeyError, TypeError, IndexError):
            return default
    return val


def _read_dicom_file(filepath):
    basename = os.path.basename(filepath)
    if basename in IGNORE_FILES or any(basename.startswith(p) for p in IGNORE_PREFIXES):
        return None
    _, ext = os.path.splitext(basename)
    if ext.lower() in NON_DICOM_EXTENSIONS:
        return None

    try:
        ds = pydicom.dcmread(filepath, force=True)
    except (InvalidDicomError, Exception):
        return None

    if not hasattr(ds, "pixel_array"):
        return None

    modality = getattr(ds, "Modality", None)
    if modality is not None and modality != "MR":
        return None

    return ds


def find_mri_series_from_index(index_json_path=None, dicom_root="./DICOM/MRI"):
    candidate_paths = [
        index_json_path,
        "./output/mri_tree.json",
        "./mri_tree.json",
        os.path.join(dicom_root, "output", "mri_tree.json"),
        os.path.join(os.path.dirname(dicom_root), "output", "mri_tree.json"),
    ]
    json_file = None
    for p in candidate_paths:
        if p and os.path.isfile(p):
            json_file = p
            break

    # Si no existe, intentar ejecutar index_mri
    if not json_file:
        try:
            import index_mri
            os.makedirs("./output", exist_ok=True)
            index_mri.index_mri(dicom_dir=os.path.dirname(os.path.abspath(dicom_root)), output_dir="./output", verbose=False)
            if os.path.isfile("./output/mri_tree.json"):
                json_file = "./output/mri_tree.json"
        except Exception:
            pass

    series_list = []

    if json_file and os.path.isfile(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            tree_root = data.get("tree", {})

            def _traverse(node, current_path=""):
                node_type = node.get("type", "")
                rel_path = node.get("relative_path", "")
                name = node.get("name", "")

                if node_type == "directory":
                    children = node.get("children", [])
                    # Verificar si este directorio contiene directamente archivos médicos
                    med_files = [
                        c for c in children
                        if c.get("file_type") in ("dicom_dcm", "dicom_no_extension_enhanced", "analyze_hdr")
                    ]
                    if med_files:
                        first_f = med_files[0]
                        ftype = first_f.get("file_type", "")
                        is_phantom = (
                            "phantom" in rel_path.lower() or
                            "fantoma" in rel_path.lower() or
                            "meta" in rel_path.lower() or
                            "uniformidad" in rel_path.lower()
                        )

                        fmt = "DICOM_MR"
                        desc = name
                        if ftype == "analyze_hdr":
                            fmt = "Analyze_7.5"
                            desc = f"Analyze 7.5 ({first_f.get('dimensions', '3D')})"
                        elif ftype == "dicom_no_extension_enhanced":
                            fmt = "Enhanced_MR"
                            desc = f"{first_f.get('series_description', name)} ({first_f.get('number_of_frames', 1)} frames)"
                        elif ftype == "dicom_dcm":
                            fmt = "DICOM_MR"
                            desc = f"{name} ({len(med_files)} cortes)"

                        # Ajustar ruta relativa quitando prefijo 'MRI/' si dicom_root ya apunta a MRI
                        clean_rel = rel_path
                        if clean_rel.startswith("MRI/"):
                            clean_rel = clean_rel[4:]

                        full_path = os.path.abspath(os.path.join(dicom_root, clean_rel))
                        if not os.path.exists(full_path) and os.path.exists(os.path.join("./DICOM", rel_path)):
                            full_path = os.path.abspath(os.path.join("./DICOM", rel_path))

                        label_type = "[Fantoma Uniformidad]" if is_phantom else "[Paciente]"
                        display_label = f"{label_type} {name} - {desc}"

                        series_list.append({
                            "name": name,
                            "display_label": display_label,
                            "relative_path": clean_rel,
                            "full_path": full_path,
                            "format": fmt,
                            "is_phantom": is_phantom,
                            "details": first_f,
                        })

                    # Continuar recursión
                    for child in children:
                        _traverse(child, rel_path)

            _traverse(tree_root)
        except Exception as e:
            logger.warning("Error al leer el árbol JSON de MRI: %s", e)

    # Si la lista está vacía, hacer escaneo manual como fallback
    if not series_list and os.path.isdir(dicom_root):
        for root, dirs, files in os.walk(dicom_root):
            valid_files = [f for f in files if not f.startswith(".") and not f.endswith(tuple(NON_DICOM_EXTENSIONS))]
            if valid_files:
                name = os.path.basename(root)
                rel_path = os.path.relpath(root, dicom_root)
                is_phantom = any(k in root.lower() for k in ("phantom", "fantoma", "meta", "uniformidad"))
                has_hdr = any(f.endswith(".hdr") for f in valid_files)
                label_type = "[Fantoma Uniformidad]" if is_phantom else "[Paciente]"
                display_label = f"{label_type} {name} ({len(valid_files)} archivos)"
                series_list.append({
                    "name": name,
                    "display_label": display_label,
                    "relative_path": rel_path,
                    "full_path": os.path.abspath(root),
                    "format": "Analyze_7.5" if has_hdr else "DICOM_MR",
                    "is_phantom": is_phantom,
                })

    return series_list


def _build_affine_from_dicom(first_ds, slice_spacing):
    if hasattr(first_ds, "ImageOrientationPatient") and hasattr(first_ds, "PixelSpacing"):
        iop = [float(v) for v in first_ds.ImageOrientationPatient]
        row_cosine = np.array(iop[0:3], dtype=float)
        col_cosine = np.array(iop[3:6], dtype=float)
        slice_cosine = np.cross(row_cosine, col_cosine)

        ps = first_ds.PixelSpacing
        row_spacing = float(ps[0])  # Spacing along rows (Y direction)
        col_spacing = float(ps[1])  # Spacing along columns (X direction)

        ipp = [float(v) for v in getattr(first_ds, "ImagePositionPatient", [0.0, 0.0, 0.0])]

        affine = np.eye(4)
        # Invertir X e Y para convertir de DICOM LPS a NIfTI RAS+
        affine[0, 0] = -row_cosine[0] * col_spacing
        affine[1, 0] = -row_cosine[1] * col_spacing
        affine[2, 0] =  row_cosine[2] * col_spacing

        affine[0, 1] = -col_cosine[0] * row_spacing
        affine[1, 1] = -col_cosine[1] * row_spacing
        affine[2, 1] =  col_cosine[2] * row_spacing

        affine[0, 2] = -slice_cosine[0] * slice_spacing
        affine[1, 2] = -slice_cosine[1] * slice_spacing
        affine[2, 2] =  slice_cosine[2] * slice_spacing

        affine[0, 3] = -ipp[0]
        affine[1, 3] = -ipp[1]
        affine[2, 3] =  ipp[2]
        return affine
    else:
        affine = np.diag([1.0, 1.0, slice_spacing, 1.0])
        return affine


def _build_affine_from_enhanced_mr(ds):
    row_spacing = None
    col_spacing = None
    slice_spacing = None
    slice_thickness = None

    pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    sfg = getattr(ds, "SharedFunctionalGroupsSequence", None)

    # Intentar extraer PixelMeasuresSequence de PerFrameFunctionalGroupsSequence[0]
    if pffg and len(pffg) > 0:
        pf0 = pffg[0]
        pm_seq = getattr(pf0, "PixelMeasuresSequence", None)
        if pm_seq and len(pm_seq) > 0:
            pm = pm_seq[0]
            ps = getattr(pm, "PixelSpacing", None)
            if ps and len(ps) >= 2:
                row_spacing, col_spacing = float(ps[0]), float(ps[1])
            slice_thickness = _safe_float(pm, "SliceThickness")
            slice_spacing = _safe_float(pm, "SpacingBetweenSlices", slice_thickness)

    # Si falta algún valor, intentar de SharedFunctionalGroupsSequence[0]
    if sfg and len(sfg) > 0:
        pm_seq = getattr(sfg[0], "PixelMeasuresSequence", None)
        if pm_seq and len(pm_seq) > 0:
            pm = pm_seq[0]
            if row_spacing is None or col_spacing is None:
                ps = getattr(pm, "PixelSpacing", None)
                if ps and len(ps) >= 2:
                    row_spacing, col_spacing = float(ps[0]), float(ps[1])
            if slice_thickness is None:
                slice_thickness = _safe_float(pm, "SliceThickness")
            if slice_spacing is None:
                slice_spacing = _safe_float(pm, "SpacingBetweenSlices", slice_thickness)

    # Si aún falta, buscar en tags directos del dataset
    if row_spacing is None or col_spacing is None:
        ps = getattr(ds, "PixelSpacing", None)
        if ps and len(ps) >= 2:
            row_spacing, col_spacing = float(ps[0]), float(ps[1])
    if slice_thickness is None:
        slice_thickness = _safe_float(ds, "SliceThickness")
    if slice_spacing is None:
        slice_spacing = _safe_float(ds, "SpacingBetweenSlices", slice_thickness)

    # Fallback si no se encontró PixelSpacing
    if row_spacing is None or col_spacing is None:
        row_spacing, col_spacing = 1.0, 1.0

    # Extraer orientación (PlaneOrientationSequence) y posición del primer frame (PlanePositionSequence)
    row_cosine = np.array([1.0, 0.0, 0.0], dtype=float)
    col_cosine = np.array([0.0, 1.0, 0.0], dtype=float)
    ipp = [0.0, 0.0, 0.0]

    if pffg and len(pffg) > 0:
        pf0 = pffg[0]
        po_seq = getattr(pf0, "PlaneOrientationSequence", None)
        if po_seq and len(po_seq) > 0:
            iop = getattr(po_seq[0], "ImageOrientationPatient", None)
            if iop and len(iop) >= 6:
                row_cosine = np.array([float(v) for v in iop[0:3]], dtype=float)
                col_cosine = np.array([float(v) for v in iop[3:6]], dtype=float)
        elif sfg and len(sfg) > 0:
            po_seq = getattr(sfg[0], "PlaneOrientationSequence", None)
            if po_seq and len(po_seq) > 0:
                iop = getattr(po_seq[0], "ImageOrientationPatient", None)
                if iop and len(iop) >= 6:
                    row_cosine = np.array([float(v) for v in iop[0:3]], dtype=float)
                    col_cosine = np.array([float(v) for v in iop[3:6]], dtype=float)

        pp_seq = getattr(pf0, "PlanePositionSequence", None)
        if pp_seq and len(pp_seq) > 0:
            pos = getattr(pp_seq[0], "ImagePositionPatient", None)
            if pos and len(pos) >= 3:
                ipp = [float(v) for v in pos]

        # Calcular distancia real entre frames consecutivos proyectada en el vector normal de corte
        slice_normal = np.cross(row_cosine, col_cosine)
        norm_mag = np.linalg.norm(slice_normal)
        if norm_mag > 1e-6:
            slice_normal = slice_normal / norm_mag

        if len(pffg) > 1:
            pp1_seq = getattr(pffg[1], "PlanePositionSequence", None)
            if pp1_seq and len(pp1_seq) > 0:
                pos1 = getattr(pp1_seq[0], "ImagePositionPatient", None)
                if pos1 and len(pos1) >= 3:
                    p0 = np.array(ipp, dtype=float)
                    p1 = np.array([float(v) for v in pos1], dtype=float)
                    # Distancia real proyectada a lo largo del vector normal del corte
                    dist = float(abs(np.dot(p1 - p0, slice_normal)))
                    if dist < 1e-3:
                        dist = float(np.linalg.norm(p1 - p0))
                    if dist > 1e-3:
                        slice_spacing = dist
    else:
        slice_normal = np.cross(row_cosine, col_cosine)

    if slice_spacing is None or slice_spacing < 1e-3:
        slice_spacing = slice_thickness if slice_thickness else 1.0

    slice_cosine = np.cross(row_cosine, col_cosine)

    affine = np.eye(4)
    # Convención NIfTI (RAS+): invertir X e Y respecto a DICOM (LPS)
    affine[0, 0] = -row_cosine[0] * col_spacing
    affine[1, 0] = -row_cosine[1] * col_spacing
    affine[2, 0] =  row_cosine[2] * col_spacing

    affine[0, 1] = -col_cosine[0] * row_spacing
    affine[1, 1] = -col_cosine[1] * row_spacing
    affine[2, 1] =  col_cosine[2] * row_spacing

    affine[0, 2] = -slice_cosine[0] * slice_spacing
    affine[1, 2] = -slice_cosine[1] * slice_spacing
    affine[2, 2] =  slice_cosine[2] * slice_spacing

    affine[0, 3] = -ipp[0]
    affine[1, 3] = -ipp[1]
    affine[2, 3] =  ipp[2]

    zooms = [col_spacing, row_spacing, slice_spacing]
    return affine, zooms


def generate_mri_volume(mri_directory, dicom_root="./DICOM/MRI", output_path=None, verbose=False):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        if not logger.handlers and not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Resolver ruta del directorio o archivo
    if os.path.isabs(mri_directory):
        target_path = mri_directory
    else:
        # Probar primero directo con dicom_root, luego probando en subdirectorios
        target_path = os.path.join(dicom_root, mri_directory)
        if not os.path.exists(target_path):
            candidates = [
                os.path.join("./DICOM", mri_directory),
                os.path.join(dicom_root, mri_directory[4:]) if mri_directory.startswith("MRI/") else "",
                os.path.join(dicom_root, "MRI", mri_directory),
                os.path.join(os.path.dirname(os.path.abspath(dicom_root)), mri_directory),
            ]
            for cand in candidates:
                if cand and os.path.exists(cand):
                    target_path = cand
                    break

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Directorio o archivo MRI no encontrado: {target_path}")

    if os.path.isfile(target_path):
        target_file = target_path
        abs_mri_dir = os.path.dirname(target_file)
    else:
        target_file = None
        abs_mri_dir = target_path

    logger.info("Procesando estudio MRI de: %s", abs_mri_dir)

    # Verificar si hay archivos Analyze (.hdr / .img)
    hdr_files = [
        os.path.join(abs_mri_dir, f) for f in os.listdir(abs_mri_dir)
        if f.endswith(".hdr") and not f.startswith("._")
    ]
    if target_file and target_file.endswith(".hdr"):
        hdr_files = [target_file]

    if hdr_files:
        hdr_path = hdr_files[0]
        logger.info("Detectado archivo Analyze 7.5: %s", hdr_path)
        img_analyze = nib.load(hdr_path)
        data = img_analyze.get_fdata()
        if data.ndim == 4 and data.shape[3] == 1:
            data = np.squeeze(data, axis=3)
        elif data.ndim == 4:
            data = data[..., 0]

        affine = img_analyze.affine
        zooms = [float(v) for v in img_analyze.header.get_zooms()[:3]]
        nifti_img = nib.Nifti1Image(data.astype(np.float32), affine)
        nifti_img.header.set_zooms(tuple(zooms))

        patient_name = os.path.basename(hdr_path).replace(".hdr", "")
        metadata = {
            "source_type": "Analyze_7.5",
            "source_file": os.path.abspath(hdr_path),
            "patient_name": patient_name,
            "dimensions": list(data.shape),
            "voxel_spacing_mm": zooms,
            "intensity_min": float(np.min(data)),
            "intensity_max": float(np.max(data)),
            "intensity_mean": float(np.mean(data)),
            "timestamp": datetime.now().isoformat(),
        }

    else:
        # Leer archivos DICOM
        files_to_read = [target_file] if target_file else [
            os.path.join(abs_mri_dir, f) for f in sorted(os.listdir(abs_mri_dir))
            if os.path.isfile(os.path.join(abs_mri_dir, f))
        ]

        slices = []
        for fpath in files_to_read:
            ds = _read_dicom_file(fpath)
            if ds is not None:
                slices.append(ds)

        if not slices:
            raise ValueError(f"No se encontraron cortes DICOM MR válidos ni archivos Analyze en: {abs_mri_dir}")

        # Caso A: Enhanced MR Multi-frame (un solo archivo con múltiples cortes en pixel_array 3D)
        first_ds = slices[0]
        if hasattr(first_ds, "pixel_array") and first_ds.pixel_array.ndim == 3:
            logger.info("Detectado archivo DICOM Enhanced MR Multi-frame (%d frames)", first_ds.pixel_array.shape[0])
            arr_3d = first_ds.pixel_array.astype(np.float32)  # [frames, rows, cols]
            slope = _safe_float(first_ds, "RescaleSlope", 1.0)
            intercept = _safe_float(first_ds, "RescaleIntercept", 0.0)
            arr_3d = arr_3d * slope + intercept

            # Reorientar a [cols, rows, frames] estándar NIfTI
            data = np.transpose(arr_3d, (2, 1, 0))

            affine, zooms = _build_affine_from_enhanced_mr(first_ds)
            nifti_img = nib.Nifti1Image(data, affine)
            nifti_img.header.set_zooms(tuple(zooms))

            metadata = {
                "source_type": "Enhanced_MR_MultiFrame",
                "source_dir": os.path.abspath(abs_mri_dir),
                "patient_name": _safe_str(first_ds, "PatientName", "ANONYMOUS"),
                "patient_id": _safe_str(first_ds, "PatientID", ""),
                "series_description": _safe_str(first_ds, "SeriesDescription", "Enhanced MR"),
                "dimensions": list(data.shape),
                "voxel_spacing_mm": zooms,
                "num_slices": data.shape[2],
                "intensity_min": float(np.min(data)),
                "intensity_max": float(np.max(data)),
                "intensity_mean": float(np.mean(data)),
                "timestamp": datetime.now().isoformat(),
            }

        else:
            # Caso B: Serie clásica de cortes individuales
            if hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[0], "ImageOrientationPatient"):
                iop = [float(v) for v in slices[0].ImageOrientationPatient]
                normal = np.cross(iop[0:3], iop[3:6])
                norm_mag = np.linalg.norm(normal)
                if norm_mag > 1e-6:
                    normal = normal / norm_mag
                slices.sort(key=lambda s: np.dot([float(v) for v in getattr(s, "ImagePositionPatient", [0, 0, 0])], normal))
            elif hasattr(slices[0], "SliceLocation"):
                slices.sort(key=lambda s: float(s.SliceLocation))
            elif hasattr(slices[0], "InstanceNumber"):
                slices.sort(key=lambda s: int(s.InstanceNumber))

            arrays = []
            for s in slices:
                arr = s.pixel_array.astype(np.float32)
                slope = _safe_float(s, "RescaleSlope", 1.0)
                intercept = _safe_float(s, "RescaleIntercept", 0.0)
                arr = arr * slope + intercept
                arrays.append(arr)

            data = np.stack(arrays, axis=-1)
            data = np.transpose(data, (1, 0, 2))  # [cols, rows, slices]

            if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[1], "ImagePositionPatient"):
                p0 = np.array([float(v) for v in slices[0].ImagePositionPatient], dtype=float)
                p1 = np.array([float(v) for v in slices[1].ImagePositionPatient], dtype=float)
                if hasattr(slices[0], "ImageOrientationPatient"):
                    iop = [float(v) for v in slices[0].ImageOrientationPatient]
                    normal = np.cross(iop[0:3], iop[3:6])
                    norm_mag = np.linalg.norm(normal)
                    if norm_mag > 1e-6:
                        normal = normal / norm_mag
                    slice_spacing = float(abs(np.dot(p1 - p0, normal)))
                else:
                    slice_spacing = float(np.linalg.norm(p1 - p0))
            elif hasattr(slices[0], "SpacingBetweenSlices") and slices[0].SpacingBetweenSlices:
                slice_spacing = float(slices[0].SpacingBetweenSlices)
            elif hasattr(slices[0], "SliceThickness") and slices[0].SliceThickness:
                slice_spacing = float(slices[0].SliceThickness)
            else:
                slice_spacing = 1.0

            ps = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
            row_spacing = float(ps[0])
            col_spacing = float(ps[1])
            zooms = [col_spacing, row_spacing, slice_spacing]

            affine = _build_affine_from_dicom(slices[0], slice_spacing)
            nifti_img = nib.Nifti1Image(data, affine)
            nifti_img.header.set_zooms(tuple(zooms))

            metadata = {
                "source_type": "DICOM_MR_Series",
                "source_dir": os.path.abspath(abs_mri_dir),
                "patient_name": _safe_str(slices[0], "PatientName", "ANONYMOUS"),
                "patient_id": _safe_str(slices[0], "PatientID", ""),
                "series_description": _safe_str(slices[0], "SeriesDescription", ""),
                "magnetic_field_strength": _safe_float(slices[0], "MagneticFieldStrength"),
                "dimensions": list(data.shape),
                "voxel_spacing_mm": zooms,
                "num_slices": len(slices),
                "intensity_min": float(np.min(data)),
                "intensity_max": float(np.max(data)),
                "intensity_mean": float(np.mean(data)),
                "timestamp": datetime.now().isoformat(),
            }

    if output_path is None:
        os.makedirs("./volumes", exist_ok=True)
        safe_name = os.path.basename(os.path.normpath(abs_mri_dir)).replace(" ", "_")
        output_path = os.path.join("./volumes", f"{safe_name}_mri.nii.gz")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    nib.save(nifti_img, output_path)
    logger.info("Volumen NIfTI MRI guardado: %s", output_path)
    metadata["output_nifti"] = os.path.abspath(output_path)

    json_path = output_path.replace(".nii.gz", ".json").replace(".nii", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Metadata JSON guardada: %s", json_path)

    return nifti_img, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generar volumen NIfTI 3D a partir de una serie MRI DICOM o Analyze 7.5"
    )
    parser.add_argument(
        "--mri-dir", "-m", default=None,
        help="Ruta relativa o absoluta al directorio de la serie MRI"
    )
    parser.add_argument(
        "--dicom-root", "-d", default="./DICOM/MRI",
        help="Ruta raíz para series DICOM (default: ./DICOM/MRI)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida del volumen NIfTI (.nii o .nii.gz)"
    )
    parser.add_argument(
        "--list-series", action="store_true",
        help="Listar todas las series MRI y fantomas descubiertos en el índice JSON"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="Imprimir información detallada"
    )
    args = parser.parse_args()

    if args.list_series:
        print("Series MRI detectadas en el repositorio:")
        series = find_mri_series_from_index(dicom_root=args.dicom_root)
        for s in series:
            print(f"  {s['display_label']}")
            print(f"    Ruta: {s['relative_path']} [{s['format']}]")
        sys.exit(0)

    if not args.mri_dir:
        parser.error("Se requiere el argumento --mri-dir/-m o --list-series.")

    nifti_img, metadata = generate_mri_volume(
        mri_directory=args.mri_dir,
        dicom_root=args.dicom_root,
        output_path=args.output,
        verbose=args.verbose,
    )
    print("\n" + "=" * 60)
    print("VOLUMEN MRI GENERADO")
    print("=" * 60)
    print(f"  Paciente/Estudio: {metadata.get('patient_name')}")
    print(f"  Tipo de fuente  : {metadata.get('source_type')}")
    print(f"  Dimensiones     : {metadata.get('dimensions')}")
    print(f"  Espaciado       : {metadata.get('voxel_spacing_mm')} mm")
    print(f"  NIfTI salida    : {metadata.get('output_nifti')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
