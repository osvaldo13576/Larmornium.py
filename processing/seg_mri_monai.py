#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

logger = logging.getLogger("seg_mri_monai")

# Registro modular de estructuras segmentables en MRI con MONAI
ORGAN_REGISTRY = {
    "cerebro_cerebelo": {
        "display_name": "Cerebro+cerebelo",
        "backend": "monai",
        "task": "wholeBrainSeg_Large_UNEST_segmentation",
        "description": "Segmentación fina de cerebro completo y cerebelo en MRI usando MONAI UNesT 3D Transformer",
    },
}

ALIAS_MAP = {
    "cerebro+cerebelo": "cerebro_cerebelo",
    "cerebro_cerebelo": "cerebro_cerebelo",
    "cerebro y cerebelo": "cerebro_cerebelo",
    "cerebro_y_cerebelo": "cerebro_cerebelo",
    "brain_cerebellum": "cerebro_cerebelo",
    "whole_brain": "cerebro_cerebelo",
    "encefalo": "cerebro_cerebelo",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
LARMORNIUM_FILES_DIR = os.path.join(_PROJECT_ROOT, "larmornium_files")
MONAI_CACHE_DIR = os.path.join(LARMORNIUM_FILES_DIR, "monai_cache")
MONAI_UNEST_DIR = os.path.join(MONAI_CACHE_DIR, "wholeBrainSeg_Large_UNEST_segmentation")
ALL_ENCEPHALON_CLASSES = set(range(1, 133))


def ensure_monai_unest_bundle(bundle_dir=None, quiet=False):
    """Garantiza la disponibilidad del modelo MONAI UNesT en larmornium_files/monai_cache."""
    target_dir = bundle_dir or MONAI_UNEST_DIR
    model_weights = os.path.join(target_dir, "models", "model.pt")
    if os.path.isdir(target_dir) and os.path.isfile(model_weights):
        return target_dir

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    # Migrar desde ~/.cache/monai si ya existe en el equipo
    user_cache = os.path.expanduser("~/.cache/monai/wholeBrainSeg_Large_UNEST_segmentation")
    if os.path.isdir(user_cache) and os.path.isfile(os.path.join(user_cache, "models", "model.pt")):
        if not quiet:
            logger.info("Copiando modelo MONAI UNesT a %s...", target_dir)
        import shutil
        shutil.copytree(user_cache, target_dir, dirs_exist_ok=True)
        return target_dir

    # Descarga automática del bundle si no se encuentra
    if not quiet:
        logger.info("Descargando bundle MONAI UNesT en %s...", os.path.dirname(target_dir))
    import monai.bundle
    monai.bundle.download(
        name="wholeBrainSeg_Large_UNEST_segmentation",
        bundle_dir=os.path.dirname(target_dir),
        progress=not quiet
    )
    return target_dir


def list_available_organs():
    return list(ORGAN_REGISTRY.keys())


def get_organ_info(organ_name):
    key = str(organ_name).lower().strip()
    key = ALIAS_MAP.get(key, key)
    return ORGAN_REGISTRY.get(key, None)


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
    volume_mm3 = num_voxels * (vx * vy * vz)
    volume_cm3 = volume_mm3 / 1000.0

    coords = np.argwhere(binary_mask > 0)
    bb_min_voxel = coords.min(axis=0).tolist()
    bb_max_voxel = coords.max(axis=0).tolist()

    dim_voxels = [bb_max_voxel[i] - bb_min_voxel[i] + 1 for i in range(3)]
    dim_mm = [round(dim_voxels[i] * voxel_spacing[i], 2) for i in range(3)]

    equiv_diameter_mm = round(2.0 * ((3.0 * volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0), 2)
    sphericity_ratio = round(min(dim_mm) / (max(dim_mm) + 1e-6), 3)

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

    if raw_data is not None:
        seg_vals = raw_data[binary_mask > 0]
        if len(seg_vals) > 0:
            mean_sig = float(np.mean(seg_vals))
            std_sig = float(np.std(seg_vals))
            stats["signal_mean"] = round(mean_sig, 2)
            stats["signal_std"] = round(std_sig, 2)
            stats["signal_snr"] = round(mean_sig / (std_sig + 1e-6), 2)

    return stats


def _get_anatomical_brain_envelope(data_f32, raw_affine, voxel_spacing, input_path=None, output_dir=None, cuda=True, quiet=False):
    """
    Obtiene una envolvente anatómica endocraneal de alta fidelidad para acotar
    las predicciones de MONAI UNesT sin amputar la corteza ni generar caras poliédricas.
    """
    import glob

    # 1. Buscar si ya existe una máscara de referencia 'cerebro' de TotalSegmentator
    candidate_paths = []
    if output_dir and os.path.isdir(output_dir):
        candidate_paths.extend(glob.glob(os.path.join(output_dir, "*cerebro*.nii.gz")))
        candidate_paths.extend(glob.glob(os.path.join(output_dir, "*brain*.nii.gz")))
    if input_path:
        parent_dir = os.path.dirname(os.path.dirname(input_path))
        seg_vol_dir = os.path.join(parent_dir, "mri_segmentation_vol")
        if os.path.isdir(seg_vol_dir):
            candidate_paths.extend(glob.glob(os.path.join(seg_vol_dir, "*cerebro*.nii.gz")))

    for p in candidate_paths:
        if os.path.isfile(p) and "cerebelo" not in os.path.basename(p).lower():
            try:
                ref_nii = nib.load(p)
                ref_data = ref_nii.get_fdata() > 0
                if ref_data.shape == data_f32.shape and np.sum(ref_data) > 500000:
                    if not quiet:
                        logger.info("  Usando envolvente cerebral de referencia: %s", os.path.basename(p))
                    struct = ndi.generate_binary_structure(3, 1)
                    dilated = ndi.binary_dilation(ref_data, structure=struct, iterations=2)
                    return ndi.binary_fill_holes(dilated)
            except Exception as e:
                logger.debug("No se pudo cargar máscara de referencia %s: %s", p, e)

    # 2. Si no existe y hay GPU disponible, invocar TotalSegmentator
    import torch
    if cuda and torch.cuda.is_available():
        try:
            from totalsegmentator.python_api import totalsegmentator
            if not quiet:
                logger.info("  Generando envolvente anatómica con TotalSegmentator MR (brain)...")
            nii_in = nib.Nifti1Image(data_f32, raw_affine)
            nii_in.header.set_zooms(tuple(voxel_spacing))
            seg_res = totalsegmentator(
                nii_in,
                roi_subset=["brain"],
                task="total_mr",
                device="gpu",
                quiet=True
            )
            ts_data = seg_res.get_fdata() > 0
            if np.sum(ts_data) > 500000:
                struct = ndi.generate_binary_structure(3, 1)
                dilated = ndi.binary_dilation(ts_data, structure=struct, iterations=2)
                return ndi.binary_fill_holes(dilated)
        except Exception as e:
            logger.warning("No se pudo ejecutar TotalSegmentator para envolvente: %s. Usando fallback morfológico.", e)

    # 3. Fallback morfológico esférico suave (sin cortes poliédricos ni diamantes)
    if not quiet:
        logger.info("  Generando envolvente morfológica esférica continua...")
    vals = data_f32[data_f32 > 0]
    p10 = np.percentile(vals, 10) if len(vals) > 0 else 0
    binary_head = data_f32 > p10
    head_filled = ndi.binary_fill_holes(binary_head)

    struct_sphere = ndi.generate_binary_structure(3, 3)
    opened = ndi.binary_opening(head_filled, structure=struct_sphere, iterations=3)
    lbl, num = ndi.label(opened)
    if num > 0:
        sizes = ndi.sum(opened, lbl, range(1, num + 1))
        brain_core = (lbl == (np.argmax(sizes) + 1))
        # Dilatación esférica con transform de distancia para suavidad anatómica absoluta
        dist = ndi.distance_transform_edt(~brain_core)
        envelope = dist <= 10.0
        return ndi.binary_fill_holes(envelope)

    return np.ones(data_f32.shape, dtype=bool)


def _run_monai_unest_cerebro_cerebelo(
    data_f32,
    raw_affine=None,
    voxel_spacing=(1.0, 1.0, 1.0),
    input_path=None,
    output_dir=None,
    cuda=True,
    fast=False,
    quiet=False
):
    import contextlib
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, EnsureType

    unest_dir = ensure_monai_unest_bundle(quiet=quiet)

    if unest_dir not in sys.path:
        sys.path.insert(0, unest_dir)

    from scripts.networks.unest_base_patch_4 import UNesT

    is_cuda = bool(cuda and torch.cuda.is_available())
    gpu = torch.device("cuda:0" if is_cuda else "cpu")
    cpu = torch.device("cpu")

    if not quiet:
        logger.info("Dispositivo MONAI inferencia: %s | acumulador: %s", gpu, cpu)

    # 1. Obtener envolvente anatómica endocraneal
    envelope = _get_anatomical_brain_envelope(
        data_f32,
        raw_affine=raw_affine if raw_affine is not None else np.eye(4),
        voxel_spacing=voxel_spacing,
        input_path=input_path,
        output_dir=output_dir,
        cuda=cuda,
        quiet=quiet
    )

    # 2. Cargar red UNesT
    net = UNesT(
        in_channels=1,
        out_channels=133,
        patch_size=4,
        depths=[2, 2, 8],
        embed_dim=[128, 256, 512],
        num_heads=[4, 8, 16]
    ).to(gpu)

    weights_path = os.path.join(unest_dir, "models", "model.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Pesos no encontrados en {weights_path}")

    state = torch.load(weights_path, map_location=gpu, weights_only=False)
    state_dict = state["model"] if "model" in state else state
    net.load_state_dict(state_dict)
    net.eval()

    # 3. Región de interés acotada a la cabeza
    nonzero_idx = np.where(data_f32 > 0)
    bbox_min = [max(0, int(np.min(idx)) - 8) for idx in nonzero_idx]
    bbox_max = [min(data_f32.shape[i], int(np.max(idx)) + 8) for i, idx in enumerate(nonzero_idx)]
    cropped = data_f32[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]

    pipeline = Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        EnsureType(data_type="tensor", device=cpu)
    ])
    inp = pipeline(cropped).unsqueeze(0)

    # 4. Inferencia con ventana deslizante optimizada según hardware
    sw_batch_size = 4 if is_cuda else 1
    overlap = 0.5 if (is_cuda and not fast) else 0.25
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if is_cuda
        else contextlib.nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            out = sliding_window_inference(
                inputs=inp,
                roi_size=(96, 96, 96),
                sw_batch_size=sw_batch_size,
                predictor=net,
                overlap=overlap,
                mode="gaussian",
                sw_device=gpu,
                device=cpu,
                progress=not quiet
            )
        pred_crop = torch.argmax(out, dim=1).squeeze(0).numpy().astype(np.uint8)

    multiclass_mask = np.zeros(data_f32.shape, dtype=np.uint8)
    multiclass_mask[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]] = pred_crop

    # 5. Acotar al espacio endocraneal con la envolvente anatómica segura
    multiclass_mask = multiclass_mask * (envelope > 0).astype(np.uint8)

    # 6. Extraer todas las clases encefálicas (cerebro + corteza + cerebelo + tronco)
    raw_mask = np.isin(multiclass_mask, list(ALL_ENCEPHALON_CLASSES)).astype(np.uint8)

    # 7. Filtrar únicamente motas de ruido < 200 vóxeles sin amputar ramas corticales
    lbl, num = ndi.label(raw_mask)
    if num > 0:
        sizes = ndi.sum(raw_mask, lbl, range(1, num + 1))
        valid_lbls = np.where(sizes >= 200)[0] + 1
        organ_mask = np.isin(lbl, valid_lbls)
    else:
        organ_mask = raw_mask.astype(bool)

    # 8. Rellenar cavidades internas para continuidad anatómica perfecta
    organ_mask = ndi.binary_fill_holes(organ_mask).astype(np.uint8)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return organ_mask


def segment_organ_mri(
    input_volume,
    organ="cerebro_cerebelo",
    cuda=True,
    fast=False,
    output_dir=None,
    output_basename=None,
    quiet=False,
):
    if not quiet:
        if not logger.handlers and not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_key = str(organ).lower().strip()
    organ_key = ALIAS_MAP.get(raw_key, raw_key)
    organ_info = get_organ_info(organ_key)

    if organ_info is None:
        available = ", ".join(f'"{k}" ({v["display_name"]})' for k, v in ORGAN_REGISTRY.items())
        raise ValueError(
            f'Estructura "{organ}" no reconocida para MONAI MRI. '
            f"Estructuras disponibles: {available}"
        )

    display_name = organ_info["display_name"]
    task = organ_info.get("task", "wholeBrainSeg_Large_UNEST_segmentation")

    logger.info("SEGMENTACIÓN MRI (MONAI): %s (%s)", display_name, organ_key)
    logger.info("  Estructura  : %s (%s)", display_name, organ_key)
    logger.info("  Modelo      : MONAI UNesT Large 3D Transformer")
    logger.info("  Dispositivo : %s", "GPU (CUDA)" if cuda else "CPU")

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

    header_zooms = [float(v) for v in raw_header.get_zooms()[:3]]
    affine_zooms = [float(np.linalg.norm(raw_affine[:3, i])) for i in range(3)]
    if any(abs(v - 1.0) < 1e-6 for v in header_zooms) and not all(abs(v - 1.0) < 1e-6 for v in affine_zooms):
        voxel_spacing = affine_zooms
    else:
        voxel_spacing = header_zooms

    raw_data = nifti_raw.get_fdata()
    if raw_data.ndim == 4:
        raw_data = np.squeeze(raw_data, axis=3) if raw_data.shape[3] == 1 else raw_data[..., 0]

    t_start = time.time()
    binary_mask = _run_monai_unest_cerebro_cerebelo(
        raw_data.astype(np.float32),
        raw_affine=raw_affine,
        voxel_spacing=voxel_spacing,
        input_path=input_path,
        output_dir=output_dir,
        cuda=cuda,
        fast=fast,
        quiet=quiet,
    )
    t_elapsed = time.time() - t_start

    binary_mask = np.ascontiguousarray((binary_mask > 0).astype(np.uint8))
    seg_nifti = nib.Nifti1Image(binary_mask, raw_affine)
    seg_header = seg_nifti.header
    seg_header.set_zooms(tuple(voxel_spacing))
    seg_header.set_data_dtype(np.uint8)
    seg_header["descrip"] = np.bytes_(f"MRI {organ_key} MONAI segmentation"[:80])

    stats = _compute_segmentation_stats(binary_mask, raw_affine, voxel_spacing, raw_data=raw_data)
    logger.info("Segmentación completada en %.1f s. Vóxeles: %d, Volumen: %.2f cm3",
                t_elapsed, stats["num_voxels"], stats["volume_cm3"])

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
        "model": "MONAI_UNesT",
        "model_version": "Large_133clases",
        "model_task": task,
        "backend": "monai",
        "device": "gpu" if cuda and torch_cuda_available() else "cpu",
        "fast_mode": fast,
        "input_volume": input_path,
        "output_nifti": os.path.abspath(nifti_out_path),
        "output_json": os.path.abspath(json_out_path),
        "volume_dimensions": list(raw_data.shape),
        "voxel_spacing_mm": [round(v, 6) for v in voxel_spacing],
        "segmentation_stats": stats,
        "timestamp": datetime.now().isoformat(),
        "processing_time_seconds": round(t_elapsed, 2),
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("Metadata JSON guardada: %s", json_out_path)

    return seg_nifti, metadata


def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


segment_organ = segment_organ_mri


def main():
    parser = argparse.ArgumentParser(description="Segmentación de Cerebro+Cerebelo en MRI con MONAI UNesT.")
    parser.add_argument("-i", "--input", default=None, help="Ruta al volumen NIfTI de entrada (MRI T1W).")
    parser.add_argument("-g", "--organ", default="cerebro_cerebelo", choices=list(ORGAN_REGISTRY.keys()) + list(ALIAS_MAP.keys()))
    parser.add_argument("-o", "--output-dir", default=None, help="Directorio de salida para NIfTI y JSON.")
    parser.add_argument("--cpu", action="store_true", help="Forzar inferencia en CPU.")
    parser.add_argument("--list-organs", action="store_true", help="Listar estructuras disponibles.")
    args = parser.parse_args()

    if args.list_organs:
        print("Estructuras disponibles en MONAI MRI:")
        for k, v in ORGAN_REGISTRY.items():
            print(f'  {k:20s} - {v["display_name"]} ({v["description"]})')
        return

    if not args.input:
        parser.error("Se requiere el argumento -i/--input para segmentar.")

    segment_organ_mri(
        input_volume=args.input,
        organ=args.organ,
        cuda=not args.cpu,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
