#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import logging
import re
import argparse
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

logger = logging.getLogger("fantoma_agua_segmentation")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _create_structuring_element_2d(elem_type: str, radius: int) -> np.ndarray:
    r = max(1, int(radius))
    elem_type = elem_type.lower()

    if elem_type in ("d", "c"):
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return (x * x + y * y) <= (r * r)
    elif elem_type in ("s", "b"):
        size = 2 * r + 1
        return np.ones((size, size), dtype=bool)
    else:
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        return (x * x + y * y) <= (r * r)


def parse_morphological_code(code_str: str):
    if not code_str:
        return []

    tokens = [t.strip().lower() for t in re.split(r"[_\-\,\;\s]+", code_str.strip()) if t.strip()]
    parsed_ops = []

    pattern = re.compile(r"^([a-z])(\d+)([a-z])$")
    for tok in tokens:
        match = pattern.match(tok)
        if match:
            elem_type = match.group(1)
            radius = int(match.group(2))
            op_type = match.group(3)
            parsed_ops.append((elem_type, radius, op_type))
        else:
            logger.warning("Token morfológico desconocido ignorado: '%s'", tok)

    return parsed_ops


def apply_morphological_chain_2d(binary_slice: np.ndarray, morph_code: str = "d1d_d1e_d1e_d1d") -> np.ndarray:
    ops = parse_morphological_code(morph_code)
    if not ops:
        return binary_slice.astype(bool)

    current_mask = binary_slice.astype(bool)
    for elem_type, radius, op_type in ops:
        footprint = _create_structuring_element_2d(elem_type, radius)
        if op_type == "d":
            current_mask = ndi.binary_dilation(current_mask, structure=footprint)
        elif op_type == "e":
            current_mask = ndi.binary_erosion(current_mask, structure=footprint)
        else:
            logger.warning("Operación no soportada '%s', omitiendo paso.", op_type)

    return current_mask


def _segment_water_phantom_core(
    nifti_input: nib.Nifti1Image,
    hu_min: float = -125.0,
    hu_max: float = 125.0,
    morph_code: str = "d1d_d1e_d1e_d1d",
    min_slice_area_px: int = 1500,
    verbose: bool = True
) -> np.ndarray:
    ct_data = nifti_input.get_fdata()
    nx, ny, nz = ct_data.shape
    mask_3d = np.zeros((nx, ny, nz), dtype=np.uint8)

    if verbose:
        logger.info("  Procesando %d cortes axiales (Rango HU: [%.1f, %.1f], Cadena: '%s')...",
                    nz, hu_min, hu_max, morph_code)

    struct_cross = ndi.generate_binary_structure(2, 1)

    # Procesamiento independiente corte a corte (2D)
    for z in range(nz):
        slice_2d = ct_data[:, :, z]

        # Paso 1: Umbralización HU
        mask_hu = (slice_2d >= hu_min) & (slice_2d <= hu_max)
        if not np.any(mask_hu):
            continue

        # Paso 2: Aplicación de la cadena morfológica configurada (ej: d1d_d1e_d1e_d1d)
        mask_cleaned = apply_morphological_chain_2d(mask_hu, morph_code=morph_code)

        # Paso 3: Componentes conexos 2D
        labeled_2d, num_feats = ndi.label(mask_cleaned, structure=struct_cross)
        if num_feats == 0:
            continue

        counts = np.bincount(labeled_2d.flat)
        counts[0] = 0
        if len(counts) <= 1:
            continue

        max_label = counts.argmax()
        max_count = counts[max_label]

        # Validar tamaño mínimo de sección transversal del cilindro
        if max_count < min_slice_area_px:
            continue

        phantom_slice = (labeled_2d == max_label)

        # Paso 4: Relleno de agujeros internos (burbujas o artefactos de reconstrucción)
        phantom_slice_filled = ndi.binary_fill_holes(phantom_slice)

        mask_3d[:, :, z] = phantom_slice_filled.astype(np.uint8)

    # Extracción del componente conexo tridimensional dominante (cilindro principal)
    labeled_3d, num_f_3d = ndi.label(mask_3d)
    if num_f_3d > 0:
        counts_3d = np.bincount(labeled_3d.flat)
        counts_3d[0] = 0
        if len(counts_3d) > 1:
            dominant_label = counts_3d.argmax()
            mask_3d = (labeled_3d == dominant_label).astype(np.uint8)

    return mask_3d


def segment_water_phantom_ct(
    input_volume,
    hu_min: float = -125.0,
    hu_max: float = 125.0,
    morph_code: str = "d1d_d1e_d1e_d1d",
    output_dir: str = None,
    output_basename: str = None,
    quiet: bool = False,
):
    if quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    logger.info("SEGMENTACIÓN CT: FANTOMA DE AGUA (PET/CT)")
    logger.info("  Método        : Umbralización 2D [%.1f, %.1f] HU + Cadena Morfológica", hu_min, hu_max)
    logger.info("  Cadena Morf.  : %s", morph_code)

    # Cargar volumen NIfTI o directorio DICOM
    if isinstance(input_volume, str):
        if os.path.isdir(input_volume):
            from gen_volume_CT import generate_ct_volume
            abs_ct_dir = os.path.abspath(input_volume)
            logger.info("  Directorio DICOM CT : %s", abs_ct_dir)
            nifti_input, _ = generate_ct_volume(abs_ct_dir, dicom_root="", verbose=False)
            input_path = abs_ct_dir
        elif os.path.isfile(input_volume):
            input_path = os.path.abspath(input_volume)
            nifti_input = nib.load(input_path)
            logger.info("  Volumen CT NIfTI    : %s", input_path)
        else:
            raise FileNotFoundError(f"Ruta de entrada no encontrada: {input_volume}")
    else:
        nifti_input = input_volume
        input_path = getattr(input_volume, "get_filename", lambda: None)()
        if input_path:
            input_path = os.path.abspath(input_path)
        logger.info("  Volumen CT          : [objeto nibabel en memoria]")

    input_shape = list(nifti_input.shape)
    affine = nifti_input.affine
    voxel_spacing = [float(v) for v in nifti_input.header.get_zooms()[:3]]

    logger.info("  Dimensiones   : %s", input_shape)
    logger.info("  Espaciado     : %s mm", [round(v, 4) for v in voxel_spacing])

    # Ejecutar algoritmo
    t_start = time.time()
    binary_mask = _segment_water_phantom_core(
        nifti_input=nifti_input,
        hu_min=hu_min,
        hu_max=hu_max,
        morph_code=morph_code,
        verbose=not quiet
    )
    t_elapsed = time.time() - t_start

    # Estadísticas cuantitativas
    total_voxels = int(np.sum(binary_mask > 0))
    voxel_vol_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    total_vol_cm3 = round(total_voxels * voxel_vol_mm3 / 1000.0, 2)

    # Calcular estadísticas HU dentro de la ROI
    ct_data = nifti_input.get_fdata()
    if total_voxels > 0:
        roi_hu_values = ct_data[binary_mask > 0]
        hu_mean = round(float(np.mean(roi_hu_values)), 2)
        hu_std = round(float(np.std(roi_hu_values)), 2)
        hu_min_val = round(float(np.min(roi_hu_values)), 2)
        hu_max_val = round(float(np.max(roi_hu_values)), 2)
        hu_median = round(float(np.median(roi_hu_values)), 2)

        indices = np.argwhere(binary_mask > 0)
        min_idx = indices.min(axis=0).tolist()
        max_idx = indices.max(axis=0).tolist()
        centroid_idx = indices.mean(axis=0).tolist()

        active_z_slices = int(max_idx[2] - min_idx[2] + 1)
        z_min_idx = int(min_idx[2])
        z_max_idx = int(max_idx[2])

        min_mm = [
            round(min_idx[0] * voxel_spacing[0], 2),
            round(min_idx[1] * voxel_spacing[1], 2),
            round(min_idx[2] * voxel_spacing[2], 2),
        ]
        max_mm = [
            round(max_idx[0] * voxel_spacing[0], 2),
            round(max_idx[1] * voxel_spacing[1], 2),
            round(max_idx[2] * voxel_spacing[2], 2),
        ]
        centroid_mm = [
            round(centroid_idx[0] * voxel_spacing[0], 2),
            round(centroid_idx[1] * voxel_spacing[1], 2),
            round(centroid_idx[2] * voxel_spacing[2], 2),
        ]
    else:
        hu_mean = hu_std = hu_min_val = hu_max_val = hu_median = 0.0
        min_idx = max_idx = centroid_idx = [0, 0, 0]
        min_mm = max_mm = centroid_mm = [0.0, 0.0, 0.0]
        active_z_slices = 0
        z_min_idx = z_max_idx = 0

    logger.info("Segmentación completada en %.2f segundos", t_elapsed)
    logger.info("  Vóxeles segmentados : %d", total_voxels)
    logger.info("  Volumen (cm3)       : %.2f", total_vol_cm3)
    logger.info("  Cortes con agua (Z) : %d cortes (Z_min=%d a Z_max=%d)", active_z_slices, z_min_idx, z_max_idx)
    logger.info("  Promedio HU en ROI  : %+.2f HU (+/- %.2f HU)", hu_mean, hu_std)

    seg_nifti = nib.Nifti1Image(binary_mask.astype(np.uint8), affine, nifti_input.header)
    seg_nifti.header.set_data_dtype(np.uint8)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "segmentations")
    os.makedirs(output_dir, exist_ok=True)

    if output_basename is None:
        if input_path:
            base = os.path.splitext(os.path.basename(input_path))[0]
            if base.endswith(".nii"):
                base = os.path.splitext(base)[0]
            output_basename = f"{base}_fantoma_agua"
        else:
            output_basename = f"fantoma_agua_{int(time.time())}"

    nifti_out_path = os.path.join(output_dir, f"{output_basename}.nii.gz")
    json_out_path = os.path.join(output_dir, f"{output_basename}.json")

    nib.save(seg_nifti, nifti_out_path)
    logger.info("Máscara NIfTI guardada: %s", nifti_out_path)

    metadata = {
        "organ": "fantoma_agua",
        "organ_display_name": "Fantoma de Agua",
        "modality": "CT",
        "model": "segmentation_anato_ct_fantoma_agua",
        "morphological_code": morph_code,
        "hu_threshold_range": [hu_min, hu_max],
        "input_volume": input_path,
        "output_nifti": os.path.abspath(nifti_out_path),
        "output_json": os.path.abspath(json_out_path),
        "total_voxels": total_voxels,
        "volume_cm3": total_vol_cm3,
        "segmentation_stats": {
            "num_voxels": total_voxels,
            "volume_cm3": total_vol_cm3,
            "voxel_volume_mm3": voxel_vol_mm3,
        },
        "active_slices_z": active_z_slices,
        "z_slice_range": [z_min_idx, z_max_idx],
        "voxel_spacing_mm": voxel_spacing,
        "dimensions": input_shape,
        "hu_statistics": {
            "mean": hu_mean,
            "std": hu_std,
            "min": hu_min_val,
            "max": hu_max_val,
            "median": hu_median,
        },
        "bounding_box_voxels": {
            "min": min_idx,
            "max": max_idx,
        },
        "bounding_box_mm": {
            "min": min_mm,
            "max": max_mm,
        },
        "centroid_voxels": [round(c, 2) for c in centroid_idx],
        "centroid_mm": centroid_mm,
        "elapsed_seconds": round(t_elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata JSON guardada: %s", json_out_path)

    return seg_nifti, metadata


# Alias para compatibilidad de interfaz con otros segmentadores
segment_organ = segment_water_phantom_ct


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Segmentación de fantomas de agua para estudios de uniformidad PET/CT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python3 segmentation_anato_ct_fantoma_agua.py \\
      --input ./volumes/PET_CT_uniformidad_fantoma_cilindrico_30_01_2023_0_790mCi_SE000001.nii.gz \\
      --output-dir ./segmentations/ \\
      --morph-code d1d_d1e_d1e_d1d
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Ruta al archivo NIfTI del volumen CT (.nii o .nii.gz).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directorio de salida para la máscara NIfTI y metadata JSON (default: ./segmentations/).",
    )
    parser.add_argument(
        "--output-basename", "-b",
        type=str,
        default=None,
        help="Nombre base para los archivos de salida (sin extensión).",
    )
    parser.add_argument(
        "--hu-min",
        type=float,
        default=-125.0,
        help="Límite inferior en Unidades Hounsfield (default: -125.0 HU).",
    )
    parser.add_argument(
        "--hu-max",
        type=float,
        default=125.0,
        help="Límite superior en Unidades Hounsfield (default: +125.0 HU).",
    )
    parser.add_argument(
        "--morph-code", "-m",
        type=str,
        default="d1d_d1e_d1e_d1d",
        help="Cadena de operaciones morfológicas 2D (default: 'd1d_d1e_d1e_d1d').",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suprime la salida informativa en consola.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        segment_water_phantom_ct(
            input_volume=args.input,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            morph_code=args.morph_code,
            output_dir=args.output_dir,
            output_basename=args.output_basename,
            quiet=args.quiet,
        )
    except Exception as e:
        logger.error("Error durante la segmentación: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
