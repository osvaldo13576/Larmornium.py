#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

logger = logging.getLogger("segmentation_anato_mri")

# Registro modular de órganos y fantomas MRI
ORGAN_REGISTRY = {
    "cerebro": {
        "display_name": "Cerebro",
        "roi_subset": ["brain"],
        "task": "total_mr",
        "description": "Segmentación del cerebro completo en MRI",
    },
    "pulmon": {
        "display_name": "Pulmón",
        "roi_subset": ["lung_left", "lung_right"],
        "task": "total_mr",
        "description": "Segmentación de ambos pulmones (izquierdo y derecho) en MRI",
    },
    "higado": {
        "display_name": "Hígado",
        "roi_subset": ["liver"],
        "task": "total_mr",
        "description": "Segmentación del hígado en MRI",
    },
    "rinon": {
        "display_name": "Riñón",
        "roi_subset": ["kidney_left", "kidney_right"],
        "task": "total_mr",
        "description": "Segmentación de ambos riñones (izquierdo y derecho) en MRI",
    },
    "corazon": {
        "display_name": "Corazón",
        "roi_subset": ["heart"],
        "task": "total_mr",
        "description": "Segmentación del corazón completo en MRI",
    },
    "bazo": {
        "display_name": "Bazo",
        "roi_subset": ["spleen"],
        "task": "total_mr",
        "description": "Segmentación del bazo en MRI",
    },
    "pancreas": {
        "display_name": "Páncreas",
        "roi_subset": ["pancreas"],
        "task": "total_mr",
        "description": "Segmentación del páncreas en MRI",
    },
    "vejiga": {
        "display_name": "Vejiga",
        "roi_subset": ["urinary_bladder"],
        "task": "total_mr",
        "description": "Segmentación de la vejiga urinaria en MRI",
    },
    "estomago": {
        "display_name": "Estómago",
        "roi_subset": ["stomach"],
        "task": "total_mr",
        "description": "Segmentación del estómago en MRI",
    },
    "fantoma_uniformidad": {
        "display_name": "Fantoma de Uniformidad",
        "roi_subset": ["phantom_uniformity"],
        "task": "mri_phantom_segmentation",
        "description": "Segmentación del volumen de líquido/gel en fantomas de uniformidad MRI (excluyendo pared plástica)",
    },
}

# Mapeo de alias para compatibilidad flexible en CLI y API
ALIAS_MAP = {
    "brain": "cerebro",
    "pulmón": "pulmon",
    "lungs": "pulmon",
    "lung": "pulmon",
    "hígado": "higado",
    "liver": "higado",
    "riñón": "rinon",
    "rinones": "rinon",
    "riñones": "rinon",
    "kidney": "rinon",
    "kidneys": "rinon",
    "corazón": "corazon",
    "heart": "corazon",
    "spleen": "bazo",
    "páncreas": "pancreas",
    "urinary_bladder": "vejiga",
    "bladder": "vejiga",
    "stomach": "estomago",
    "estómago": "estomago",
    "fantoma": "fantoma_uniformidad",
    "phantom": "fantoma_uniformidad",
    "uniformidad": "fantoma_uniformidad",
    "fantoma_mri": "fantoma_uniformidad",
    "fantoma_agua_mri": "fantoma_uniformidad",
    "mri_phantom": "fantoma_uniformidad",
}


def list_available_organs():
    return list(ORGAN_REGISTRY.keys())


def get_organ_info(organ_name):
    key = str(organ_name).lower().strip()
    key = ALIAS_MAP.get(key, key)
    return ORGAN_REGISTRY.get(key, None)


def _segment_mri_uniformity_phantom(nifti_input, margin_mm=3.0, threshold_ratio=0.30):
    data = nifti_input.get_fdata()
    spacing = [float(v) for v in nifti_input.header.get_zooms()[:3]]

    # Separar señal del fantoma del piso de ruido de fondo (Rayleigh/Rician)
    non_zero = data[data > 0]
    if len(non_zero) == 0:
        return np.zeros_like(data, dtype=np.uint8)

    p95 = float(np.percentile(non_zero, 95))
    thresh = max(p95 * threshold_ratio, 1.0)
    binary = data >= thresh

    # Apertura 3D para eliminar ruido y soportes finos
    struct_3d = ndi.generate_binary_structure(3, 1)
    opened = ndi.binary_opening(binary, structure=struct_3d, iterations=1)
    if not np.any(opened):
        opened = binary

    # Extraer el componente conexo principal (el cuerpo del fantoma)
    labeled, num_features = ndi.label(opened)
    if num_features == 0:
        return np.zeros_like(data, dtype=np.uint8)

    counts = np.bincount(labeled.flat)
    counts[0] = 0
    main_component = (labeled == counts.argmax())

    # Relleno exhaustivo 2D corte a corte y 3D
    struct_2d = ndi.generate_binary_structure(2, 1)
    filled = np.zeros_like(main_component, dtype=bool)
    for z in range(main_component.shape[2]):
        s = main_component[:, :, z]
        if np.any(s):
            s_fill = ndi.binary_fill_holes(s)
            s_closed = ndi.binary_closing(s_fill, structure=struct_2d, iterations=2)
            filled[:, :, z] = s_closed

    filled = ndi.binary_fill_holes(filled)

    # Erosión en plano de corte (2D) para excluir pared plástica exterior
    if margin_mm > 0:
        sx, sy = float(spacing[0]), float(spacing[1])
        min_inplane_spacing = min(sx, sy) if min(sx, sy) > 1e-4 else 1.0
        iter_e = max(int(round(margin_mm / min_inplane_spacing)), 1)
        eroded = np.zeros_like(filled, dtype=bool)
        for z in range(filled.shape[2]):
            s = filled[:, :, z]
            if np.any(s):
                eroded[:, :, z] = ndi.binary_erosion(s, structure=struct_2d, iterations=iter_e)
        result = eroded
    else:
        result = filled

    return (result > 0).astype(np.uint8)


def _compute_segmentation_stats(binary_mask, affine, voxel_spacing, raw_data=None):
    num_voxels = int(np.sum(binary_mask > 0))
    if num_voxels == 0:
        return {
            "num_voxels": 0,
            "volume_cm3": 0.0,
            "physical_dimensions_mm": [0.0, 0.0, 0.0],
            "equivalent_diameter_mm": 0.0,
            "sphericity_aspect_ratio": 0.0,
            "bounding_box_voxel": None,
            "bounding_box_mm": None,
            "centroid_voxel": None,
            "centroid_mm": None,
        }

    vx, vy, vz = [float(v) for v in voxel_spacing[:3]]
    voxel_volume_mm3 = vx * vy * vz
    volume_mm3 = num_voxels * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000.0

    # Bounding box en coordenadas de vóxel
    coords = np.argwhere(binary_mask > 0)
    bb_min_voxel = coords.min(axis=0).tolist()
    bb_max_voxel = coords.max(axis=0).tolist()

    # Dimensiones físicas en mm (X, Y, Z)
    dim_voxels = [bb_max_voxel[i] - bb_min_voxel[i] + 1 for i in range(3)]
    dim_mm = [round(dim_voxels[i] * voxel_spacing[i], 2) for i in range(3)]

    # Diámetro equivalente de esfera: V = 4/3 * pi * R^3 => D = 2 * (3V / 4pi)^(1/3)
    equiv_diameter_mm = round(2.0 * ((3.0 * volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0), 2)
    sphericity_ratio = round(min(dim_mm) / (max(dim_mm) + 1e-6), 3)

    # Centroide en coordenadas de vóxel
    centroid_voxel = coords.mean(axis=0).tolist()

    def voxel_to_mm(v_coords):
        v = np.array([v_coords[0], v_coords[1], v_coords[2], 1.0])
        mm = affine @ v
        return mm[:3].tolist()

    bb_min_mm = voxel_to_mm(bb_min_voxel)
    bb_max_mm = voxel_to_mm(bb_max_voxel)
    centroid_mm = voxel_to_mm(centroid_voxel)

    stats = {
        "num_voxels": int(num_voxels),
        "volume_cm3": round(volume_cm3, 2),
        "physical_dimensions_mm": {
            "dx_mm": dim_mm[0],
            "dy_mm": dim_mm[1],
            "dz_mm": dim_mm[2],
        },
        "equivalent_diameter_mm": equiv_diameter_mm,
        "sphericity_aspect_ratio": sphericity_ratio,
        "bounding_box_voxel": {
            "min": [int(v) for v in bb_min_voxel],
            "max": [int(v) for v in bb_max_voxel],
        },
        "bounding_box_mm": {
            "min": [round(v, 2) for v in bb_min_mm],
            "max": [round(v, 2) for v in bb_max_mm],
        },
        "centroid_voxel": [round(v, 2) for v in centroid_voxel],
        "centroid_mm": [round(v, 2) for v in centroid_mm],
    }

    # Métricas de señal en el volumen segmentado si se suministra raw_data
    if raw_data is not None:
        seg_vals = raw_data[binary_mask > 0]
        if len(seg_vals) > 0:
            mean_sig = float(np.mean(seg_vals))
            std_sig = float(np.std(seg_vals))
            snr = round(mean_sig / (std_sig + 1e-6), 2)
            stats["signal_mean"] = round(mean_sig, 2)
            stats["signal_std"] = round(std_sig, 2)
            stats["signal_snr"] = snr

    return stats


def _get_totalsegmentator_version():
    try:
        import importlib.metadata
        return importlib.metadata.version("TotalSegmentator")
    except Exception:
        return "unknown"


def segment_organ_mri(
    input_volume,
    organ,
    cuda=True,
    fast=False,
    output_dir=None,
    output_basename=None,
    quiet=False,
):
    if not quiet:
        if not logger.handlers and not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(levelname)s: %(message)s"
            )

    # Validar estructura
    raw_key = str(organ).lower().strip()
    organ_key = ALIAS_MAP.get(raw_key, raw_key)
    organ_info = get_organ_info(organ_key)
    if organ_info is None:
        available = ", ".join(
            f'"{k}" ({v["display_name"]})'
            for k, v in ORGAN_REGISTRY.items()
        )
        raise ValueError(
            f'Estructura "{organ}" no reconocida para MRI. '
            f"Estructuras disponibles: {available}"
        )

    roi_subset = organ_info["roi_subset"]
    display_name = organ_info["display_name"]
    task = organ_info.get("task", "total_mr")

    logger.info("SEGMENTACIÓN MRI: %s (%s)", display_name, task)
    logger.info("  Estructura  : %s (%s)", display_name, organ_key)
    logger.info("  Tarea       : %s", task)
    logger.info("  Dispositivo : %s", "GPU (CUDA)" if cuda else "CPU")

    # Cargar volumen de entrada
    if isinstance(input_volume, str):
        if not os.path.isfile(input_volume):
            raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_volume}")
        input_path = os.path.abspath(input_volume)
        nifti_raw = nib.load(input_path)
        logger.info("  Volumen     : %s", input_path)
    else:
        nifti_raw = input_volume
        input_path = getattr(input_volume, "get_filename", lambda: None)()
        if input_path:
            input_path = os.path.abspath(input_path)
        logger.info("  Volumen     : [objeto nibabel en memoria]")

    raw_affine = nifti_raw.affine
    raw_header = nifti_raw.header

    # Verificar y sincronizar espaciado vóxel entre header y norma de columnas de la matriz afín
    header_zooms = [float(v) for v in raw_header.get_zooms()[:3]]
    affine_zooms = [float(np.linalg.norm(raw_affine[:3, i])) for i in range(3)]

    if any(abs(v - 1.0) < 1e-6 for v in header_zooms) and not all(abs(v - 1.0) < 1e-6 for v in affine_zooms):
        voxel_spacing = affine_zooms
    else:
        voxel_spacing = header_zooms

    raw_data = nifti_raw.get_fdata()
    if raw_data.ndim == 4:
        if raw_data.shape[3] == 1:
            raw_data = np.squeeze(raw_data, axis=3)
        else:
            raw_data = raw_data[..., 0]

    nifti_input = nib.Nifti1Image(raw_data.astype(np.float32), raw_affine)
    nifti_input.header.set_zooms(tuple(voxel_spacing))
    input_shape = nifti_input.shape

    logger.info("  Dimensiones : %s", list(input_shape))
    logger.info("  Espaciado   : %s mm", [round(v, 4) for v in voxel_spacing])

    import torch
    import gc

    if cuda and torch.cuda.is_available():
        device = "gpu"
        device_name = torch.cuda.get_device_name(0)
        logger.info("  Aceleración GPU activada: %s (CUDA)", device_name)
    else:
        device = "cpu"
        if cuda and not torch.cuda.is_available():
            logger.warning("  GPU (CUDA) solicitada pero no disponible en PyTorch. Se usará CPU.")
        else:
            logger.info("  Ejecución configurada en CPU.")

    t_start = time.time()

    # Caso A: Segmentación de fantoma de uniformidad MRI
    if task == "mri_phantom_segmentation" or organ_key == "fantoma_uniformidad":
        logger.info("Ejecutando algoritmo de segmentación de Fantoma de Uniformidad MRI...")
        binary_mask = _segment_mri_uniformity_phantom(
            nifti_input,
            margin_mm=3.0,
            threshold_ratio=0.30
        )
        model_name = "Algoritmo Analítico de Uniformidad MRI"
        model_ver = "1.0.0"

    # Caso B: TotalSegmentator MR (total_mr)
    else:
        from totalsegmentator.python_api import totalsegmentator

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Ejecutando TotalSegmentator en %s (task=%s)...", "GPU (CUDA)" if device == "gpu" else "CPU", task)
        try:
            seg_output = totalsegmentator(
                nifti_input,
                None,
                fast=fast,
                task=task,
                roi_subset=roi_subset,
                device=device,
                quiet=quiet,
            )
        except Exception as e:
            if device == "gpu" and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
                logger.warning("Memoria GPU insuficiente (%s). Reintentando en CPU...", e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                device = "cpu"
                seg_output = totalsegmentator(
                    nifti_input,
                    None,
                    fast=fast,
                    task=task,
                    roi_subset=roi_subset,
                    device="cpu",
                    quiet=quiet,
                )
            else:
                raise e

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        seg_data = seg_output.get_fdata()
        binary_mask = (seg_data > 0).astype(np.uint8)
        model_name = "TotalSegmentator"
        model_ver = _get_totalsegmentator_version()

    t_elapsed = time.time() - t_start
    logger.info("Segmentación completada en %.1f segundos (%s)", t_elapsed, "GPU (CUDA)" if device == "gpu" else "CPU")

    # Garantizar estrictamente que el volumen segmentado sea binario (0 y 1)
    binary_mask = np.ascontiguousarray((binary_mask > 0).astype(np.uint8))
    unique_vals = np.unique(binary_mask)
    logger.debug("Valores únicos en máscara binaria MRI: %s (dtype=%s)", unique_vals, binary_mask.dtype)

    # Crear imagen NIfTI con la máscara binaria y espaciado explícito
    seg_nifti = nib.Nifti1Image(binary_mask, raw_affine)
    seg_header = seg_nifti.header
    seg_header.set_zooms(tuple(voxel_spacing))
    seg_header.set_data_dtype(np.uint8)
    seg_header["descrip"] = np.bytes_(f"MRI {organ_key} segmentation"[:80])

    # Calcular estadísticas espaciales y geométricas
    stats = _compute_segmentation_stats(binary_mask, raw_affine, voxel_spacing, raw_data=raw_data)
    logger.info("  Vóxeles segmentados : %d", stats["num_voxels"])
    logger.info("  Volumen (cm3)       : %.2f", stats["volume_cm3"])
    if stats.get("physical_dimensions_mm"):
        p_dim = stats["physical_dimensions_mm"]
        logger.info("  Dimensiones físicas : DX=%.1f mm, DY=%.1f mm, DZ=%.1f mm (Aspecto: %.3f)",
                    p_dim["dx_mm"], p_dim["dy_mm"], p_dim["dz_mm"], stats["sphericity_aspect_ratio"])
    if stats["bounding_box_mm"]:
        logger.info("  Bounding box (mm)   : min=%s  max=%s", stats["bounding_box_mm"]["min"], stats["bounding_box_mm"]["max"])

    if output_dir is None:
        output_dir = os.path.join(".", "segmentations")
    os.makedirs(output_dir, exist_ok=True)

    if output_basename is None:
        if input_path:
            base = os.path.basename(input_path)
            for ext in (".nii.gz", ".nii", ".hdr", ".img"):
                if base.endswith(ext):
                    base = base[:-len(ext)]
                    break
        else:
            base = "mri_volume"
        output_basename = f"{base}_{organ_key}_mri_seg"

    nifti_out_path = os.path.join(output_dir, f"{output_basename}.nii.gz")
    json_out_path = os.path.join(output_dir, f"{output_basename}.json")

    nib.save(seg_nifti, nifti_out_path)
    logger.info("Máscara NIfTI guardada: %s", nifti_out_path)

    metadata = {
        "study_name": output_basename.replace(f"_{organ_key}_mri_seg", ""),
        "modality": "MRI",
        "organ": organ_key,
        "organ_display_name": display_name,
        "organ_description": organ_info["description"],
        "model": model_name,
        "model_version": model_ver,
        "model_task": task,
        "roi_labels_used": roi_subset,
        "device": device,
        "fast_mode": fast,
        "input_volume": input_path,
        "output_nifti": os.path.abspath(nifti_out_path),
        "output_json": os.path.abspath(json_out_path),
        "volume_dimensions": list(input_shape),
        "voxel_spacing_mm": [round(v, 6) for v in voxel_spacing],
        "segmentation_stats": stats,
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": round(t_elapsed, 2),
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Metadata JSON guardada: %s", json_out_path)

    return seg_nifti, metadata


segment_organ = segment_organ_mri


def main():
    organ_choices = list(ORGAN_REGISTRY.keys()) + list(ALIAS_MAP.keys())
    organ_help_lines = []
    for key, info in ORGAN_REGISTRY.items():
        labels = ", ".join(info["roi_subset"])
        organ_help_lines.append(f'  {key:20s} - {info["display_name"]:25s} (labels: {labels})')
    organ_help = "\n".join(organ_help_lines)

    parser = argparse.ArgumentParser(
        description="Segmentación anatómica y de fantomas en MRI con TotalSegmentator y algoritmos especializados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Estructuras disponibles:
{organ_help}

Ejemplos:
  # Segmentar cerebro en MRI con GPU
  python3 segmentation_anato_mri_TotalSegmentator.py \\
      --input ./volumes/mri_cerebro_anatomico.nii.gz \\
      --organ cerebro \\
      --cuda

  # Segmentar fantoma de uniformidad en MRI
  python3 segmentation_anato_mri_TotalSegmentator.py \\
      --input ./volumes/mri_psiquiatria_sinmeta_se.nii.gz \\
      --organ fantoma_uniformidad

  # Listar estructuras disponibles
  python3 segmentation_anato_mri_TotalSegmentator.py --list-organs
        """,
    )
    parser.add_argument("--input", "-i", default=None, help="Ruta al volumen MRI (.nii, .nii.gz, .hdr/.img)")
    parser.add_argument("--organ", "-g", default=None, choices=organ_choices, help="Estructura u órgano a segmentar")
    parser.add_argument("--cuda", action="store_true", default=True, dest="cuda", help="Usar GPU (CUDA)")
    parser.add_argument("--no-cuda", action="store_false", dest="cuda", help="Forzar uso de CPU")
    parser.add_argument("--fast", action="store_true", default=False, help="Modo rápido TotalSegmentator (3mm)")
    parser.add_argument("--output-dir", "-o", default=None, help="Directorio de salida (default: ./segmentations/)")
    parser.add_argument("--output-basename", default=None, help="Nombre base para los archivos de salida")
    parser.add_argument("--quiet", "-q", action="store_true", default=False, help="Suprimir mensajes")
    parser.add_argument("--list-organs", action="store_true", help="Listar estructuras disponibles y salir")

    args = parser.parse_args()

    if args.list_organs:
        print("Estructuras MRI disponibles para segmentación:")
        for key, info in ORGAN_REGISTRY.items():
            labels = ", ".join(info["roi_subset"])
            print(f'  {key:20s} - {info["display_name"]:25s} ({info["description"]}) [labels: {labels}]')
        sys.exit(0)

    if not args.input or not args.organ:
        parser.error("Se requieren los argumentos --input/-i y --organ/-g para realizar la segmentación.")

    _, metadata = segment_organ_mri(
        input_volume=args.input,
        organ=args.organ,
        cuda=args.cuda,
        fast=args.fast,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        quiet=args.quiet,
    )

    stats = metadata["segmentation_stats"]
    print("\nSEGMENTACIÓN MRI COMPLETADA:")
    print(f"  Estructura         : {metadata['organ_display_name']}")
    print(f"  Modelo             : {metadata['model']} v{metadata['model_version']} ({metadata['model_task']})")
    print(f"  Dispositivo        : {metadata['device']}")
    print(f"  Tiempo             : {metadata['processing_time_seconds']} s")
    print(f"  Vóxeles segmentados: {stats['num_voxels']}")
    print(f"  Volumen (cm3)      : {stats['volume_cm3']}")
    if stats.get("physical_dimensions_mm"):
        p_dim = stats["physical_dimensions_mm"]
        print(f"  Dimensiones (mm)   : DX={p_dim['dx_mm']} mm, DY={p_dim['dy_mm']} mm, DZ={p_dim['dz_mm']} mm")
        print(f"  Diámetro equiv.    : {stats['equivalent_diameter_mm']} mm (Esfericidad: {stats['sphericity_aspect_ratio']})")
    if stats.get("signal_mean") is not None:
        print(f"  Señal media        : {stats['signal_mean']} +/- {stats['signal_std']} (SNR: {stats['signal_snr']})")
    if stats["bounding_box_mm"]:
        print(f"  Bounding box (mm)  : min={stats['bounding_box_mm']['min']}")
        print(f"                       max={stats['bounding_box_mm']['max']}")
    if stats["centroid_mm"]:
        print(f"  Centroide (mm)     : {stats['centroid_mm']}")
    print(f"  NIfTI              : {metadata['output_nifti']}")
    print(f"  JSON               : {metadata['output_json']}")


if __name__ == "__main__":
    main()
