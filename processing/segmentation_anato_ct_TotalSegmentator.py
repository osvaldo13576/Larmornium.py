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

logger = logging.getLogger("segmentation_anato_ct")

# Registro modular de órganos
# Para agregar un nuevo órgano, solo se necesita añadir una entrada aquí.
# Cada entrada mapea un identificador interno (sin acentos, minúsculas) a:
#   - display_name : nombre para mostrar al usuario
#   - roi_subset   : lista de etiquetas de TotalSegmentator
#   - task         : tarea de TotalSegmentator (default: "total") o método especializado
#   - description  : descripción breve
ORGAN_REGISTRY = {
    "cerebro": {
        "display_name": "Cerebro",
        "roi_subset": ["brain"],
        "task": "total",
        "description": "Segmentación del cerebro completo",
    },
    "cerebelo": {
        "display_name": "Cerebelo",
        "roi_subset": ["cerebellum"],
        "task": "brain_structures",
        "description": "Segmentación del cerebelo",
    },
    "corazon": {
        "display_name": "Corazón",
        "roi_subset": ["heart"],
        "task": "total",
        "description": "Segmentación del corazón completo",
    },
    "pulmon": {
        "display_name": "Pulmón",
        "roi_subset": [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ],
        "task": "total",
        "description": "Segmentación de ambos pulmones (5 lóbulos)",
    },
    "sistema_respiratorio": {
        "display_name": "Sistema Respiratorio",
        "roi_subset": [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
            "trachea",
        ],
        "task": "total",
        "description": "Segmentación del sistema respiratorio completo (pulmones [5 lóbulos] y tráquea)",
    },
    "higado": {
        "display_name": "Hígado",
        "roi_subset": ["liver"],
        "task": "total",
        "description": "Segmentación del hígado",
    },
    "intestino": {
        "display_name": "Intestino",
        "roi_subset": ["small_bowel", "duodenum", "colon"],
        "task": "total",
        "description": "Segmentación del tracto intestinal (intestino delgado, duodeno y colon)",
    },
    "rinon": {
        "display_name": "Riñón",
        "roi_subset": ["kidney_left", "kidney_right"],
        "task": "total",
        "description": "Segmentación de ambos riñones (izquierdo y derecho)",
    },
    "caja_toracica": {
        "display_name": "Caja Torácica",
        "roi_subset": [
            "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4",
            "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8",
            "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
            "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4",
            "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8",
            "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
            "sternum",
            "costal_cartilages",
            "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4",
            "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8",
            "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
        ],
        "task": "total",
        "description": "Segmentación de la caja torácica (24 costillas, esternón, cartílagos costales y vértebras torácicas T1-T12)",
    },
    "craneo": {
        "display_name": "Cráneo",
        "roi_subset": ["skull"],
        "task": "total",
        "description": "Segmentación del cráneo completo (bóveda y base craneal)",
    },
    "huesos_manos": {
        "display_name": "Huesos de las Manos",
        "roi_subset": ["carpal", "metacarpal", "phalanges_hand"],
        "task": "appendicular_bones",
        "description": "Segmentación de los huesos de ambas manos (carpo, metacarpo y falanges)",
    },
    "huesos_pies": {
        "display_name": "Huesos de los Pies",
        "roi_subset": ["tarsal", "metatarsal", "phalanges_feet"],
        "task": "appendicular_bones",
        "description": "Segmentación de los huesos de ambos pies (tarso, metatarso y falanges)",
    },
    "huesos_extremidades": {
        "display_name": "Huesos de Extremidades",
        "roi_subset": [
            "patella", "tibia", "fibula",
            "tarsal", "metatarsal", "phalanges_feet",
            "ulna", "radius",
            "carpal", "metacarpal", "phalanges_hand"
        ],
        "task": "appendicular_bones",
        "description": "Segmentación de huesos apendiculares (antebrazos, manos, piernas y pies)",
    },
    "huesos": {
        "display_name": "Todos los Huesos del Cuerpo (Esqueleto Completo + Cráneo + Manos + Pies)",
        "roi_subset": [
            "skull",
            "sacrum",
            "vertebrae_S1",
            "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
            "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9",
            "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5",
            "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1",
            "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4",
            "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
            "humerus_left", "humerus_right",
            "scapula_left", "scapula_right",
            "clavicula_left", "clavicula_right",
            "femur_left", "femur_right",
            "hip_left", "hip_right",
            "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4",
            "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8",
            "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
            "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4",
            "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8",
            "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
            "sternum",
            "costal_cartilages",
            "patella", "tibia", "fibula",
            "tarsal", "metatarsal", "phalanges_feet",
            "ulna", "radius",
            "carpal", "metacarpal", "phalanges_hand",
        ],
        "task": "total+appendicular_bones",
        "description": "Segmentación integral de todos los huesos del cuerpo humano (74 estructuras: cráneo, columna, costillas, pelvis, extremidades, manos y pies)",
    },
    "fantoma_agua": {
        "display_name": "Fantoma de Agua",
        "roi_subset": ["water_phantom"],
        "task": "water_phantom_ct",
        "description": "Segmentación del volumen de agua en fantoma cilíndrico de uniformidad (0 HU +/- 125 HU)",
    },
}

# Mapeo de alias para mayor flexibilidad en CLI y API
ALIAS_MAP = {
    "cerebel": "cerebelo",
    "cerebellum": "cerebelo",
    "corazón": "corazon",
    "heart": "corazon",
    "pulmón": "pulmon",
    "lungs": "pulmon",
    "lung": "pulmon",
    "sistema respiratorio": "sistema_respiratorio",
    "respiratorio": "sistema_respiratorio",
    "respiratory_system": "sistema_respiratorio",
    "respiratory": "sistema_respiratorio",
    "aparato_respiratorio": "sistema_respiratorio",
    "aparato respiratorio": "sistema_respiratorio",
    "vias_respiratorias": "sistema_respiratorio",
    "vías_respiratorias": "sistema_respiratorio",
    "vias respiratorias": "sistema_respiratorio",
    "vías respiratorias": "sistema_respiratorio",
    "traquea_pulmones": "sistema_respiratorio",
    "tráquea_pulmones": "sistema_respiratorio",
    "hígado": "higado",
    "liver": "higado",
    "intestinos": "intestino",
    "intestine": "intestino",
    "bowel": "intestino",
    "tracto_intestinal": "intestino",
    "tracto intestinal": "intestino",
    "small_bowel": "intestino",
    "colon": "intestino",
    "duodeno": "intestino",
    "duodenum": "intestino",
    "riñón": "rinon",
    "rinones": "rinon",
    "riñones": "rinon",
    "kidney": "rinon",
    "kidneys": "rinon",
    "caja_torácica": "caja_toracica",
    "caja toracica": "caja_toracica",
    "caja torácica": "caja_toracica",
    "cajatoracica": "caja_toracica",
    "cajatorácica": "caja_toracica",
    "parrilla_costal": "caja_toracica",
    "parrilla costal": "caja_toracica",
    "rib_cage": "caja_toracica",
    "ribcage": "caja_toracica",
    "thoracic_cage": "caja_toracica",
    "torax_oseo": "caja_toracica",
    "tórax_óseo": "caja_toracica",
    "costillas": "caja_toracica",
    "ribs": "caja_toracica",
    "brain": "cerebro",
    "cráneo": "craneo",
    "skull": "craneo",
    "cabeza_huesos": "craneo",
    "manos": "huesos_manos",
    "huesos_mano": "huesos_manos",
    "hands": "huesos_manos",
    "pies": "huesos_pies",
    "huesos_pie": "huesos_pies",
    "feet": "huesos_pies",
    "extremidades": "huesos_extremidades",
    "appendicular_bones": "huesos_extremidades",
    "huesos_cuerpo": "huesos",
    "huesos_del_cuerpo": "huesos",
    "bones": "huesos",
    "all_bones": "huesos",
    "skeleton": "huesos",
    "esqueleto": "huesos",
    "esqueleto_completo": "huesos",
    "fantoma": "fantoma_agua",
    "fantoma_de_agua": "fantoma_agua",
    "water_phantom": "fantoma_agua",
}


def list_available_organs():
    return list(ORGAN_REGISTRY.keys())


def get_organ_info(organ_key):
    if not organ_key:
        return None
    key = organ_key.lower().strip()
    canonical_key = ALIAS_MAP.get(key, key)
    return ORGAN_REGISTRY.get(canonical_key)


def _compute_segmentation_stats(mask_data, affine, voxel_spacing):
    segmented_voxels = np.argwhere(mask_data > 0)

    if len(segmented_voxels) == 0:
        return {
            "num_voxels": 0,
            "volume_cm3": 0.0,
            "bounding_box_voxel": None,
            "bounding_box_mm": None,
            "centroid_voxel": None,
            "centroid_mm": None,
        }

    num_voxels = len(segmented_voxels)

    voxel_volume_mm3 = abs(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    volume_cm3 = (num_voxels * voxel_volume_mm3) / 1000.0

    bb_min_voxel = segmented_voxels.min(axis=0).tolist()
    bb_max_voxel = segmented_voxels.max(axis=0).tolist()

    centroid_voxel = segmented_voxels.mean(axis=0).tolist()

    def voxel_to_mm(voxel_coord):
        coord = np.array([*voxel_coord, 1.0])
        mm = affine @ coord
        return mm[:3].tolist()

    bb_min_mm = voxel_to_mm(bb_min_voxel)
    bb_max_mm = voxel_to_mm(bb_max_voxel)
    centroid_mm = voxel_to_mm(centroid_voxel)

    return {
        "num_voxels": int(num_voxels),
        "volume_cm3": round(volume_cm3, 2),
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


def segment_organ(
    input_volume,
    organ,
    cuda=True,
    fast=False,
    output_dir=None,
    output_basename=None,
    quiet=False,
):
    from totalsegmentator.python_api import totalsegmentator

    if not quiet:
        if not logger.handlers and not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(levelname)s: %(message)s"
            )

    # Validar órgano
    raw_key = organ.lower().strip()
    organ_key = ALIAS_MAP.get(raw_key, raw_key)
    organ_info = get_organ_info(organ_key)
    if organ_info is None:
        available = ", ".join(
            f'"{k}" ({v["display_name"]})'
            for k, v in ORGAN_REGISTRY.items()
        )
        raise ValueError(
            f'Órgano "{organ}" no reconocido. '
            f"Órganos disponibles: {available}"
        )

    roi_subset = organ_info["roi_subset"]
    display_name = organ_info["display_name"]
    task = organ_info.get("task", "total")

    # Fast mode solo es compatible con tareas total
    effective_fast = fast
    if task not in ("total", "total_v1", "total_v3", "total_mr") and fast:
        logger.warning(
            "El modo rápido (fast=True) no está soportado para la tarea '%s'. Ejecutando en resolución completa.",
            task,
        )
        effective_fast = False

    logger.info("SEGMENTACIÓN CT: %s", display_name.upper())
    logger.info("  Órgano      : %s (%s)", display_name, organ_key)
    logger.info("  Tarea/Método: %s", task)
    logger.info("  ROI labels  : %s", roi_subset)
    logger.info("  Dispositivo : %s", "GPU (CUDA)" if cuda else "CPU")
    logger.info("  Modo rápido : %s", "Sí" if effective_fast else "No")

    # Cargar volumen de entrada
    if isinstance(input_volume, str):
        if not os.path.isfile(input_volume):
            raise FileNotFoundError(
                f"Archivo de entrada no encontrado: {input_volume}"
            )
        input_path = os.path.abspath(input_volume)
        nifti_input = nib.load(input_path)
        logger.info("  Volumen     : %s", input_path)
    else:
        nifti_input = input_volume
        input_path = getattr(input_volume, "get_filename", lambda: None)()
        if input_path:
            input_path = os.path.abspath(input_path)
        logger.info("  Volumen     : [objeto nibabel en memoria]")

    input_shape = nifti_input.shape
    affine = nifti_input.affine
    voxel_spacing = [float(v) for v in nifti_input.header.get_zooms()[:3]]

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

    if task == "water_phantom_ct" or organ_key == "fantoma_agua":
        from segmentation_anato_ct_fantoma_agua import segment_water_phantom_ct
        return segment_water_phantom_ct(
            input_volume=nifti_input,
            output_dir=output_dir,
            output_basename=output_basename,
            quiet=quiet,
        )

    model_name = "TotalSegmentator"

    if task == "total+appendicular_bones":
        logger.info("Ejecutando segmentación integral del esqueleto: Tarea Total (Cráneo + Axial + Mayores) + Tarea Huesos Apendiculares (Manos + Pies + Extremidades)...")
        t_start = time.time()

        total_bones = [
            "skull", "sacrum", "vertebrae_S1",
            "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
            "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9",
            "vertebrae_T8", "vertebrae_T7", "vertebrae_T6", "vertebrae_T5",
            "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1",
            "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4",
            "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
            "humerus_left", "humerus_right",
            "scapula_left", "scapula_right",
            "clavicula_left", "clavicula_right",
            "femur_left", "femur_right",
            "hip_left", "hip_right",
            "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4",
            "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8",
            "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
            "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4",
            "rib_right_5", "rib_right_6", "rib_right_7", "rib_right_8",
            "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
            "sternum", "costal_cartilages",
        ]

        # Tarea 'total' (Cráneo, columna, costillas, pelvis, extremidades principales)
        logger.info("  [1/2] Segmentando Cráneo, Columna, Costillas, Pelvis y Extremidades Principales (tarea 'total')...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seg_total = totalsegmentator(
            nifti_input,
            None,
            fast=effective_fast,
            task="total",
            roi_subset=total_bones,
            device=device,
            quiet=quiet,
        )
        mask_total = (seg_total.get_fdata() > 0).astype(np.uint8)

        # Tarea 'appendicular_bones' (Manos, pies, antebrazos, piernas)
        logger.info("  [2/2] Segmentando Huesos de Manos, Pies y Extremidades Apendiculares (tarea 'appendicular_bones')...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seg_app = totalsegmentator(
            nifti_input,
            None,
            task="appendicular_bones",
            device=device,
            quiet=quiet,
        )
        mask_app = (seg_app.get_fdata() > 0).astype(np.uint8)

        # Unión lógica
        binary_mask = ((mask_total > 0) | (mask_app > 0)).astype(np.uint8)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        t_elapsed = time.time() - t_start
        logger.info("Segmentación del esqueleto completo completada en %.1f segundos (%s)", t_elapsed, "GPU (CUDA)" if device == "gpu" else "CPU")
        model_name = "TotalSegmentator (total + appendicular_bones)"

    else:
        # TotalSegmentator solo soporta el argumento roi_subset en tareas de tipo 'total' o 'total_mr'.
        # Para otras tareas (como 'brain_structures' o 'appendicular_bones'), se pasa roi_subset=None y luego se filtran los labels.
        ts_roi_subset = roi_subset if task.startswith("total") else None

        # Ejecutar TotalSegmentator con fallback automático a CPU si falta memoria GPU
        logger.info("Ejecutando TotalSegmentator en %s (task=%s)...", "GPU (CUDA)" if device == "gpu" else "CPU", task)
        t_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            seg_output = totalsegmentator(
                nifti_input,
                None,  # output_path=None: devuelve objeto nibabel
                fast=effective_fast,
                task=task,
                roi_subset=ts_roi_subset,
                device=device,
                quiet=quiet,
            )
        except Exception as e:
            if device == "gpu" and ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
                logger.warning("Memoria GPU insuficiente (%s). Reintentando automáticamente en CPU...", e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                device = "cpu"
                seg_output = totalsegmentator(
                    nifti_input,
                    None,
                    fast=effective_fast,
                    task=task,
                    roi_subset=ts_roi_subset,
                    device="cpu",
                    quiet=quiet,
                )
            else:
                raise e

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        t_elapsed = time.time() - t_start
        logger.info("Segmentación completada en %.1f segundos (%s)", t_elapsed, "GPU (CUDA)" if device == "gpu" else "CPU")

        # Combinar etiquetas en máscara binaria
        seg_data = seg_output.get_fdata()

        from totalsegmentator.map_to_binary import class_map

        if not task.startswith("total") and task in class_map:
            name_to_id = {v: k for k, v in class_map[task].items()}
            target_ids = [name_to_id[name] for name in roi_subset if name in name_to_id]
            if target_ids:
                binary_mask = np.isin(seg_data, target_ids).astype(np.uint8)
            else:
                binary_mask = (seg_data > 0).astype(np.uint8)
        else:
            binary_mask = (seg_data > 0).astype(np.uint8)

    # Garantizar estrictamente que el volumen segmentado sea binario (0 y 1)
    binary_mask = np.ascontiguousarray((binary_mask > 0).astype(np.uint8))
    unique_vals = np.unique(binary_mask)
    logger.debug("Valores únicos en máscara binaria CT: %s (dtype=%s)", unique_vals, binary_mask.dtype)

    # Crear nueva imagen NIfTI con la máscara binaria (uint8)
    seg_nifti = nib.Nifti1Image(binary_mask, affine, nifti_input.header)
    seg_header = seg_nifti.header
    seg_header.set_data_dtype(np.uint8)
    seg_header["descrip"] = np.bytes_(
        f"{organ_key} segmentation"[:80]
    )

    # Calcular estadísticas
    stats = _compute_segmentation_stats(binary_mask, affine, voxel_spacing)

    logger.info("  Vóxeles segmentados : %d", stats["num_voxels"])
    logger.info("  Volumen (cm3)       : %.2f", stats["volume_cm3"])
    if stats["bounding_box_mm"]:
        logger.info(
            "  Bounding box (mm)   : min=%s  max=%s",
            stats["bounding_box_mm"]["min"],
            stats["bounding_box_mm"]["max"],
        )

    if output_dir is None:
        output_dir = os.path.join(".", "segmentations")
    os.makedirs(output_dir, exist_ok=True)

    if output_basename is None:
        if input_path:
            base = os.path.basename(input_path)
            if base.endswith(".nii.gz"):
                base = base[:-7]
            elif base.endswith(".nii"):
                base = base[:-4]
        else:
            base = "ct_volume"
        output_basename = f"{base}_{organ_key}_seg"

    nifti_out_path = os.path.join(output_dir, f"{output_basename}.nii.gz")
    json_out_path = os.path.join(output_dir, f"{output_basename}.json")

    nib.save(seg_nifti, nifti_out_path)
    logger.info("Máscara NIfTI guardada: %s", nifti_out_path)

    metadata = {
        "study_name": output_basename.replace(f"_{organ_key}_seg", ""),
        "organ": organ_key,
        "organ_display_name": display_name,
        "organ_description": organ_info["description"],
        "model": model_name,
        "model_version": _get_totalsegmentator_version() if "TotalSegmentator" in model_name else "1.0",
        "model_task": task,
        "roi_labels_used": roi_subset,
        "device": device,
        "fast_mode": effective_fast,
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


def _get_totalsegmentator_version():
    try:
        import importlib.metadata
        return importlib.metadata.version("TotalSegmentator")
    except Exception:
        return "unknown"


def main():
    organ_choices = list(ORGAN_REGISTRY.keys()) + list(ALIAS_MAP.keys())
    organ_help_lines = []
    for key, info in ORGAN_REGISTRY.items():
        labels = ", ".join(info["roi_subset"])
        task_str = f" [task: {info.get('task', 'total')}]" if info.get("task", "total") != "total" else ""
        organ_help_lines.append(f'  {key:22s} - {info["display_name"]:22s} (labels: {labels}){task_str}')
    organ_help = "\n".join(organ_help_lines)

    parser = argparse.ArgumentParser(
        description="Segmentación anatómica de volúmenes CT con TotalSegmentator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Órganos disponibles:
{organ_help}

Ejemplos:
  # Segmentar corazón con GPU
  python3 segmentation_anato_ct_TotalSegmentator.py \\
      --input ./volumes/paciente_00_ct.nii.gz \\
      --organ corazon \\
      --cuda

  # Segmentar sistema respiratorio (pulmones + tráquea)
  python3 segmentation_anato_ct_TotalSegmentator.py \\
      --input ./volumes/paciente_00_ct.nii.gz \\
      --organ sistema_respiratorio \\
      --cuda

  # Segmentar intestino (delgado, duodeno y colon)
  python3 segmentation_anato_ct_TotalSegmentator.py \\
      --input ./volumes/paciente_00_ct.nii.gz \\
      --organ intestino \\
      --cuda

  # Segmentar caja torácica (costillas, esternón, cartílagos costales y vértebras T1-T12)
  python3 segmentation_anato_ct_TotalSegmentator.py \\
      --input ./volumes/paciente_00_ct.nii.gz \\
      --organ caja_toracica \\
      --cuda

  # Segmentar cerebelo con GPU
  python3 segmentation_anato_ct_TotalSegmentator.py \\
      --input ./volumes/paciente_00_ct.nii.gz \\
      --organ cerebelo \\
      --cuda

  # Segmentar pulmones en modo rápido con CPU
  python3 segmentation_anato_ct_TotalSegmentator.py \\
      --input ./volumes/paciente_00_ct.nii.gz \\
      --organ pulmon \\
      --no-cuda \\
      --fast
        """,
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Ruta al volumen CT en formato NIfTI (.nii o .nii.gz)"
    )
    parser.add_argument(
        "--organ", "-g", default=None, choices=organ_choices,
        help="Órgano a segmentar"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False,
        dest="cuda",
        help="Usar GPU (CUDA) para la segmentación (default: False)"
    )
    parser.add_argument(
        "--no-cuda", action="store_false", dest="cuda",
        help="Forzar uso de CPU"
    )
    parser.add_argument(
        "--fast", action="store_true", default=False,
        help="Modo rápido (resolución 3mm, menos preciso, para tareas compatibles)"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directorio de salida (default: ./segmentations/)"
    )
    parser.add_argument(
        "--output-basename", default=None,
        help="Nombre base para archivos de salida (sin extensión)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", default=False,
        help="Suprimir mensajes de progreso de TotalSegmentator"
    )
    parser.add_argument(
        "--list-organs", action="store_true",
        help="Listar órganos disponibles y salir"
    )

    args = parser.parse_args()

    if args.list_organs:
        print("Órganos disponibles para segmentación:")
        for key, info in ORGAN_REGISTRY.items():
            labels = ", ".join(info["roi_subset"])
            task_str = f" [task: {info.get('task', 'total')}]" if info.get("task", "total") != "total" else ""
            print(f'  {key:22s} - {info["display_name"]:22s} '
                  f'({info["description"]}) [labels: {labels}]{task_str}')
        sys.exit(0)

    if not args.input or not args.organ:
        parser.error("Se requieren los argumentos --input/-i y --organ/-g para realizar la segmentación.")

    _, metadata = segment_organ(
        input_volume=args.input,
        organ=args.organ,
        cuda=args.cuda,
        fast=args.fast,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        quiet=args.quiet,
    )

    stats = metadata["segmentation_stats"]
    print("\nSEGMENTACIÓN COMPLETADA:")
    print(f"  Órgano             : {metadata['organ_display_name']}")
    print(f"  Modelo             : {metadata['model']} v{metadata['model_version']}")
    print(f"  Dispositivo        : {metadata['device']}")
    print(f"  Tiempo             : {metadata['processing_time_seconds']} s")
    print(f"  Vóxeles segmentados: {stats['num_voxels']}")
    print(f"  Volumen (cm3)      : {stats['volume_cm3']}")
    if stats["bounding_box_mm"]:
        print(f"  Bounding box (mm)  : min={stats['bounding_box_mm']['min']}")
        print(f"                       max={stats['bounding_box_mm']['max']}")
    if stats["centroid_mm"]:
        print(f"  Centroide (mm)     : {stats['centroid_mm']}")
    print(f"  NIfTI              : {metadata['output_nifti']}")
    print(f"  JSON               : {metadata['output_json']}")


if __name__ == "__main__":
    main()
