#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import contextlib
import gc
import json
import logging
import os

import time
from datetime import datetime

import nibabel as nib
import nibabel.processing
import numpy as np
import scipy.ndimage as ndi

logger = logging.getLogger("seg_ct_monai")

ORGAN_REGISTRY = {
    "cerebro_cerebelo": {
        "display_name": "Cerebro+cerebelo",
        "backend": "monai",
        "task": "wholeBody_ct_segmentation",
        "target_classes": [50],
        "description": "Segmentación de cerebro completo y cerebelo en CT usando MONAI",
    },
    "corazon_completo": {
        "display_name": "Corazón completo",
        "backend": "monai",
        "task": "wholeBody_ct_segmentation",
        "target_classes": [7, 44, 45, 46, 47, 48, 49],
        "description": "Segmentación del corazón completo (miocardio, cámaras, aorta y arteria pulmonar) en CT usando MONAI",
    },
    "craneo": {
        "display_name": "Cráneo",
        "backend": "monai",
        "task": "wholeBody_ct_segmentation",
        "target_classes": [50, 93, 40, 41],
        "description": "Segmentación completa y de alta resolución de cráneo, base craneal, esqueleto facial y piezas dentales en CT usando MONAI",
    },
}

ALIAS_MAP = {
    "cerebro+cerebelo": "cerebro_cerebelo",
    "cerebro_cerebelo": "cerebro_cerebelo",
    "cerebro y cerebelo": "cerebro_cerebelo",
    "cerebro_y_cerebelo": "cerebro_cerebelo",
    "brain": "cerebro_cerebelo",
    "whole_brain": "cerebro_cerebelo",
    "encefalo": "cerebro_cerebelo",
    "corazon": "corazon_completo",
    "corazón": "corazon_completo",
    "corazon_completo": "corazon_completo",
    "corazón completo": "corazon_completo",
    "heart": "corazon_completo",
    "whole_heart": "corazon_completo",
    "craneo": "craneo",
    "cráneo": "craneo",
    "skull": "craneo",
    "cranium": "craneo",
    "calavera": "craneo",
    "huesos_cabeza": "craneo",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
LARMORNIUM_FILES_DIR = os.path.join(_PROJECT_ROOT, "larmornium_files")
MONAI_CACHE_DIR = os.path.join(LARMORNIUM_FILES_DIR, "monai_cache")
MONAI_WB_DIR = os.path.join(MONAI_CACHE_DIR, "wholeBody_ct_segmentation")


def ensure_monai_bundle(bundle_dir=None, quiet=False):
    """Garantiza la disponibilidad del bundle wholeBody_ct_segmentation de MONAI."""
    target_dir = bundle_dir or MONAI_WB_DIR
    model_weights = os.path.join(target_dir, "models", "model.pt")
    if os.path.isdir(target_dir) and os.path.isfile(model_weights):
        return target_dir

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    user_cache = os.path.expanduser("~/.cache/monai/wholeBody_ct_segmentation")
    if os.path.isdir(user_cache) and os.path.isfile(os.path.join(user_cache, "models", "model.pt")):
        if not quiet:
            logger.info("Copiando modelo MONAI CT a %s...", target_dir)
        import shutil
        shutil.copytree(user_cache, target_dir, dirs_exist_ok=True)
        return target_dir

    if not quiet:
        logger.info("Descargando bundle MONAI wholeBody_ct_segmentation en %s...", os.path.dirname(target_dir))
    import monai.bundle
    monai.bundle.download(
        name="wholeBody_ct_segmentation",
        version="0.2.5",
        bundle_dir=os.path.dirname(target_dir),
        progress=not quiet
    )
    return target_dir



def get_organ_info(organ_name):
    key = str(organ_name).lower().strip()
    key = ALIAS_MAP.get(key, key)
    return ORGAN_REGISTRY.get(key, None)


def _compute_segmentation_stats(binary_mask, affine, voxel_spacing, raw_data=None, nifti_raw=None):
    num_voxels = int(np.sum(binary_mask > 0))
    if num_voxels == 0:
        return {
            "num_voxels": 0,
            "volume_cm3": 0.0,
            "physical_dimensions_mm": {"dx_mm": 0.0, "dy_mm": 0.0, "dz_mm": 0.0},
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
            stats["hu_mean"] = round(float(np.mean(seg_vals)), 2)
            stats["hu_std"] = round(float(np.std(seg_vals)), 2)
            stats["hu_min"] = round(float(np.min(seg_vals)), 2)
            stats["hu_max"] = round(float(np.max(seg_vals)), 2)
    elif nifti_raw is not None:
        z_min_seg = int(bb_min_voxel[2])
        z_max_seg = int(bb_max_voxel[2]) + 1
        sub_raw = np.asarray(nifti_raw.dataobj[:, :, z_min_seg:z_max_seg], dtype=np.float32)
        sub_mask = binary_mask[:, :, z_min_seg:z_max_seg]
        seg_vals = sub_raw[sub_mask > 0]
        if len(seg_vals) > 0:
            stats["hu_mean"] = round(float(np.mean(seg_vals)), 2)
            stats["hu_std"] = round(float(np.std(seg_vals)), 2)
            stats["hu_min"] = round(float(np.min(seg_vals)), 2)
            stats["hu_max"] = round(float(np.max(seg_vals)), 2)

    return stats


def _detect_anatomical_z_range(nifti_raw, organ_key, voxel_spacing):
    """
    Localiza de forma rápida y de bajo consumo de memoria el rango de cortes Z
    donde se ubica la estructura anatómica seleccionada.
    """
    shape = nifti_raw.shape
    Nz = shape[2]
    vz = float(voxel_spacing[2])
    vxy = float(voxel_spacing[0])
    total_z_span_mm = Nz * vz

    # Si el volumen ya es acotado (<300 mm de extensión Z), procesar completo
    if total_z_span_mm <= 300.0 or Nz <= 180:
        return 0, Nz

    step_z = max(1, int(round(4.0 / vz)))
    step_xy = max(1, min(shape[0], shape[1]) // 64)
    sub = np.asarray(nifti_raw.dataobj[::step_xy, ::step_xy, ::step_z], dtype=np.float32)

    body_profile = np.sum(sub > -500, axis=(0, 1))
    valid_k = np.where(body_profile > 10)[0]
    if len(valid_k) == 0:
        return 0, Nz

    k_body_min, k_body_max = int(valid_k.min()), int(valid_k.max())
    z_body_min = k_body_min * step_z
    z_body_max = min(Nz, (k_body_max + 1) * step_z)

    if organ_key in ("cerebro_cerebelo", "craneo"):
        # Detección del ápice pulmonar para delimitar el cuello en barridos de cuerpo entero o tórax
        lung_counts = np.zeros(sub.shape[2])
        for k in range(k_body_min, k_body_max + 1):
            sl = sub[:, :, k]
            if np.sum(sl > -500) > 10:
                body = ndi.binary_fill_holes(sl > -500)
                lung_counts[k] = np.sum(body & (sl >= -950) & (sl <= -400))

        peak_lung = float(lung_counts.max()) if len(lung_counts) > 0 else 0.0
        if peak_lung > 15:
            axcodes = nib.aff2axcodes(nifti_raw.affine)
            is_sup_positive = (axcodes[2] == "S") if len(axcodes) >= 3 else True
            top_lung_k = np.where(lung_counts > peak_lung * 0.15)[0]
            if len(top_lung_k) > 0:
                if is_sup_positive:
                    lung_top_z = int(top_lung_k.max()) * step_z
                    z_start = max(0, lung_top_z - int(np.ceil(20.0 / vz)))
                    z_end = min(Nz, lung_top_z + int(np.ceil(280.0 / vz)))
                else:
                    lung_top_z = int(top_lung_k.min()) * step_z
                    z_start = max(0, lung_top_z - int(np.ceil(280.0 / vz)))
                    z_end = min(Nz, lung_top_z + int(np.ceil(20.0 / vz)))
                return z_start, z_end

        # Si no hay pulmones visibles, buscar la cavidad craneal por densidad ósea y encéfalo
        cranial_z = []
        sub_vxy = vxy * step_xy
        for k in range(k_body_min, k_body_max + 1):
            sl = sub[:, :, k]
            bone = sl > 180
            filled = ndi.binary_fill_holes(bone)
            cavity = filled & (~bone)
            area_cm2 = np.sum(cavity) * (sub_vxy * sub_vxy) / 100.0
            if 30 < area_cm2 < 250:
                brain_vox = np.sum(cavity & (sl >= 15) & (sl <= 55))
                if brain_vox / max(1, np.sum(cavity)) > 0.4:
                    cranial_z.append(k * step_z)

        if len(cranial_z) > 0:
            mid_z = float(np.median(cranial_z))
            z_start = max(0, int(mid_z - 140.0 / vz))
            z_end = min(Nz, int(mid_z + 140.0 / vz))
            return z_start, z_end

        head_height_mm = 240.0
        head_slices = int(np.ceil(head_height_mm / vz))
        z_start = max(0, z_body_max - head_slices)
        z_end = min(Nz, z_body_max + int(np.ceil(15.0 / vz)))
        return z_start, z_end

    elif organ_key == "corazon_completo":
        lung_counts = np.zeros(sub.shape[2])
        for k in range(k_body_min, k_body_max + 1):
            sl = sub[:, :, k]
            if np.sum(sl > -500) > 10:
                body = ndi.binary_fill_holes(sl > -500)
                lung_counts[k] = np.sum(body & (sl >= -950) & (sl <= -400))

        peak_lung = float(lung_counts.max()) if len(lung_counts) > 0 else 0.0
        if peak_lung < 10:
            logger.warning("No se detectó cavidad pulmonar/torácica en el volumen para corazón.")
            return None

        lung_k = np.where(lung_counts > peak_lung * 0.15)[0]
        lung_z_min = int(lung_k.min()) * step_z
        lung_z_max = min(Nz, (int(lung_k.max()) + 1) * step_z)
        z_start = max(0, lung_z_min - int(np.ceil(35.0 / vz)))
        z_end = min(Nz, lung_z_max + int(np.ceil(50.0 / vz)))
        return z_start, z_end

    return 0, Nz


def _run_monai_ct_inference(
    nifti_raw,
    organ_key,
    target_classes,
    voxel_spacing,
    cuda=True,
    fast=False,
    quiet=False,
):
    import torch
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SegResNet
    from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, ScaleIntensity, EnsureType

    bundle_dir = ensure_monai_bundle(quiet=quiet)

    is_cuda = bool(cuda and torch.cuda.is_available())
    gpu = torch.device("cuda:0" if is_cuda else "cpu")
    cpu = torch.device("cpu")

    if not quiet:
        logger.info("Dispositivo MONAI inferencia: %s | acumulador: %s", gpu, cpu)

    # 1. Delimitación anatómica del rango Z
    z_range = _detect_anatomical_z_range(nifti_raw, organ_key, voxel_spacing)
    if z_range is None:
        return np.zeros(nifti_raw.shape[:3], dtype=np.uint8)

    z_start, z_end = z_range
    if not quiet:
        logger.info("Rango Z anatómico seleccionado: cortes %d a %d (total: %d cortes)",
                    z_start, z_end, z_end - z_start)

    # 2. Extracción de cortes relevantes y ajuste del afín
    roi_data = np.asarray(nifti_raw.dataobj[:, :, z_start:z_end], dtype=np.float32)
    roi_affine = nifti_raw.affine.copy()
    roi_affine[:3, 3] = roi_affine[:3, 3] + roi_affine[:3, 2] * z_start

    roi_nii = nib.Nifti1Image(roi_data, roi_affine)
    roi_nii.header.set_zooms(tuple(voxel_spacing[:3]))

    # 3. Orientación canónica y remuestreo a resolución de entrenamiento
    target_spacing = (1.5, 1.5, 1.5)
    roi_ras = nib.as_closest_canonical(roi_nii)
    roi_res = nib.processing.resample_to_output(roi_ras, voxel_sizes=target_spacing, order=1)
    data_res = roi_res.get_fdata().astype(np.float32)

    # Preprocesamiento de intensidad
    pipeline = Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        ScaleIntensity(minv=-1.0, maxv=1.0),
        EnsureType(data_type="tensor", device=cpu),
    ])

    # 4. Recorte del contorno corporal en XY para descartar aire periférico
    body_xy = np.where(data_res > -800)
    if len(body_xy[0]) > 0:
        bx_min = max(0, int(np.min(body_xy[0])) - 8)
        bx_max = min(data_res.shape[0], int(np.max(body_xy[0])) + 8)
        by_min = max(0, int(np.min(body_xy[1])) - 8)
        by_max = min(data_res.shape[1], int(np.max(body_xy[1])) + 8)
    else:
        bx_min, bx_max = 0, data_res.shape[0]
        by_min, by_max = 0, data_res.shape[1]

    cropped_xy = data_res[bx_min:bx_max, by_min:by_max, :]

    # 5. Cargar red SegResNet
    net = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    ).to(gpu)

    weights_file = "model.pt"
    weights_path = os.path.join(bundle_dir, "models", weights_file)
    state = torch.load(weights_path, map_location=gpu, weights_only=False)
    state_dict = state["model"] if "model" in state else state
    net.load_state_dict(state_dict)
    net.eval()

    # 6. Inferencia del modelo SegResNet
    Z_res = cropped_xy.shape[2]
    sw_batch_size = 2 if is_cuda else 1
    overlap = 0.25
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if is_cuda
        else contextlib.nullcontext()
    )

    norm_cropped = pipeline(cropped_xy)
    del cropped_xy
    gc.collect()

    if Z_res <= 220:
        inp_tensor = norm_cropped.unsqueeze(0)
        with torch.no_grad():
            with autocast_ctx:
                out_tensor = sliding_window_inference(
                    inputs=inp_tensor,
                    roi_size=(96, 96, 96),
                    sw_batch_size=sw_batch_size,
                    predictor=net,
                    overlap=overlap,
                    mode="gaussian",
                    sw_device=gpu,
                    device=cpu,
                    progress=False,
                )
        del inp_tensor, norm_cropped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if organ_key in ("cerebro_cerebelo", "craneo"):
            max_organ, _ = torch.max(out_tensor[0, 1:], dim=0)
            pred_res_uint8 = torch.argmax(out_tensor, dim=1).squeeze(0).numpy().astype(np.uint8)
            brain_calib = ((out_tensor[0, 50] == max_organ) & (out_tensor[0, 50] > -1.0)).numpy()
            pred_res_uint8[brain_calib] = 50
        else:
            pred_res_uint8 = torch.argmax(out_tensor, dim=1).squeeze(0).numpy().astype(np.uint8)

        del out_tensor
        gc.collect()
    else:
        pred_res_uint8 = np.zeros(norm_cropped.shape[1:], dtype=np.uint8)
        slab_size = 128
        slab_step = 96
        z_slabs = []
        z_curr = 0
        while z_curr < Z_res:
            z_end_slab = min(Z_res, z_curr + slab_size)
            z_start_slab = max(0, z_end_slab - slab_size)
            z_slabs.append((z_start_slab, z_end_slab))
            if z_end_slab >= Z_res:
                break
            z_curr += slab_step

        for zs, ze in z_slabs:
            inp_slab = norm_cropped[:, :, :, zs:ze].unsqueeze(0)
            with torch.no_grad():
                with autocast_ctx:
                    out_slab = sliding_window_inference(
                        inputs=inp_slab,
                        roi_size=(96, 96, 96),
                        sw_batch_size=sw_batch_size,
                        predictor=net,
                        overlap=overlap,
                        mode="gaussian",
                        sw_device=gpu,
                        device=cpu,
                        progress=False,
                    )
            if organ_key in ("cerebro_cerebelo", "craneo"):
                max_organ, _ = torch.max(out_slab[0, 1:], dim=0)
                slab_pred = torch.argmax(out_slab, dim=1).squeeze(0).numpy().astype(np.uint8)
                brain_calib = ((out_slab[0, 50] == max_organ) & (out_slab[0, 50] > -1.0)).numpy()
                slab_pred[brain_calib] = 50
            else:
                slab_pred = torch.argmax(out_slab, dim=1).squeeze(0).numpy().astype(np.uint8)

            del out_slab, inp_slab
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            target_sub = pred_res_uint8[:, :, zs:ze]
            update_mask = (slab_pred > 0) | (target_sub == 0)
            target_sub[update_mask] = slab_pred[update_mask]

        del norm_cropped
        gc.collect()

    # 7. Extracción de clases de interés y recomposición espacial
    if organ_key == "craneo":
        pred_res_full = np.zeros(data_res.shape, dtype=np.uint8)
        pred_res_full[bx_min:bx_max, by_min:by_max, :] = pred_res_uint8
        del pred_res_uint8, data_res
        gc.collect()

        # 8. Remuestreo de etiquetas multiclase al espacio nativo del subvolumen ROI
        seg_pred_nii = nib.Nifti1Image(pred_res_full, roi_res.affine)
        seg_pred_orig = nib.processing.resample_from_to(seg_pred_nii, roi_nii, order=0)
        pred_native = np.asarray(seg_pred_orig.dataobj, dtype=np.uint8)
        del pred_res_full, seg_pred_nii, seg_pred_orig
        gc.collect()

        vz = float(voxel_spacing[2])
        vy, vx = float(voxel_spacing[1]), float(voxel_spacing[0])

        brain_mask = (pred_native == 50)
        b_coords = np.argwhere(brain_mask)
        if len(b_coords) > 0:
            b_min_z = int(b_coords[:, 2].min())
            b_center_xy = b_coords[:, :2].mean(axis=0)
        else:
            b_min_z = 0
            b_center_xy = np.array([roi_data.shape[0] / 2.0, roi_data.shape[1] / 2.0])

        # Máscara tridimensional del paciente que engloba el encéfalo y excluye camilla y aire
        body_raw = ndi.binary_fill_holes(roi_data > -500)
        lbl_body, num_body = ndi.label(body_raw)
        if num_body > 1 and len(b_coords) > 0:
            brain_lbls = np.unique(lbl_body[brain_mask])
            valid_lbls = brain_lbls[brain_lbls > 0]
            if len(valid_lbls) > 0:
                patient_head_3d = np.isin(lbl_body, valid_lbls)
            else:
                sizes = ndi.sum(body_raw, lbl_body, range(1, num_body + 1))
                patient_head_3d = (lbl_body == (np.argmax(sizes) + 1))
        elif num_body == 1:
            patient_head_3d = (lbl_body > 0)
        else:
            patient_head_3d = body_raw

        # Exclusión de vértebras cervicales y torácicas bajo el encéfalo identificadas por MONAI (C1 a L5)
        zi = np.arange(roi_data.shape[2])[None, None, :]
        vertebrae_raw = np.isin(pred_native, range(18, 42)) & (zi <= (b_min_z + int(round(5.0 / vz))))
        vertebrae_dil = ndi.binary_dilation(vertebrae_raw, iterations=2)

        # Exclusión de huesos apendiculares y torácicos (húmeros, escápulas, clavículas y costillas)
        arms_raw = np.isin(pred_native, [82, 83, 84, 85, 86, 87]) | np.isin(pred_native, range(58, 82))
        arms_dil = ndi.binary_dilation(arms_raw, iterations=3)

        # Límite anatómico inferior (mandíbula termina unos 65 mm bajo la unión craneocervical)
        c1_coords = np.argwhere((pred_native == 41) & (zi <= (b_min_z + int(round(5.0 / vz)))))
        if len(c1_coords) > 0:
            c1_z = int(c1_coords[:, 2].min())
            inf_cutoff = max(0, c1_z - int(round(65.0 / vz)))
        else:
            inf_cutoff = max(0, b_min_z - int(round(95.0 / vz)))

        # Delimitación radial en XY respecto al centro craneal para descartar miembros periféricos
        yi, xi = np.ogrid[:roi_data.shape[0], :roi_data.shape[1]]
        dist_from_center_mm = np.sqrt(((yi - b_center_xy[0]) * vy) ** 2 + ((xi - b_center_xy[1]) * vx) ** 2)
        head_zone_xy = (dist_from_center_mm <= 135.0)[:, :, None]

        # Estructuras óseas del cráneo (bóveda, base, esqueleto facial, mandíbula y piezas dentales)
        candidate_bone = (roi_data >= 120) & patient_head_3d & (~vertebrae_dil) & (~arms_dil) & head_zone_xy
        candidate_bone[:, :, :inf_cutoff] = False

        # Consolidación morfológica 3D de diploe, cavidades paranasales y piezas dentales
        struct = ndi.generate_binary_structure(3, 1)
        skull_closed = ndi.binary_closing(candidate_bone, structure=struct, iterations=3)
        binary_roi = (skull_closed & (roi_data >= 50) & patient_head_3d) | candidate_bone
        binary_roi[:, :, :inf_cutoff] = False

        # Filtrado de componentes espurios menores a 100 vóxeles
        lbl_sk, num_sk = ndi.label(binary_roi)
        if num_sk > 1:
            sizes = ndi.sum(binary_roi, lbl_sk, range(1, num_sk + 1))
            valid = np.where(sizes >= 100)[0] + 1
            binary_roi = np.isin(lbl_sk, valid)

        binary_roi = binary_roi.astype(np.uint8)
        del pred_native, brain_mask, patient_head_3d, vertebrae_raw, vertebrae_dil, arms_raw, arms_dil
        del candidate_bone, skull_closed, lbl_sk
        gc.collect()

    else:
        binary_cropped = np.isin(pred_res_uint8, target_classes).astype(np.uint8)
        del pred_res_uint8
        gc.collect()

        binary_res = np.zeros(data_res.shape, dtype=np.uint8)
        binary_res[bx_min:bx_max, by_min:by_max, :] = binary_cropped
        del binary_cropped, data_res
        gc.collect()

        # 8. Remuestreo al espacio nativo del subvolumen ROI
        seg_roi_res = nib.Nifti1Image(binary_res, roi_res.affine)
        seg_roi_orig = nib.processing.resample_from_to(seg_roi_res, roi_nii, order=0)
        binary_roi = (seg_roi_orig.get_fdata() > 0).astype(np.uint8)
        del seg_roi_res, seg_roi_orig
        gc.collect()

        if organ_key == "cerebro_cerebelo":
            lbl, num = ndi.label(binary_roi > 0)
            if num > 1:
                sizes = ndi.sum(binary_roi > 0, lbl, range(1, num + 1))
                binary_roi = (lbl == (np.argmax(sizes) + 1)).astype(np.uint8)

            for k in range(binary_roi.shape[2]):
                sl = binary_roi[:, :, k]
                if np.sum(sl) > 20:
                    sl_c = ndi.binary_closing(sl, iterations=2)
                    binary_roi[:, :, k] = ndi.binary_fill_holes(sl_c).astype(np.uint8)

            binary_roi = ((binary_roi > 0) & (roi_data >= -10) & (roi_data <= 85)).astype(np.uint8)
            binary_roi = ndi.binary_fill_holes(binary_roi).astype(np.uint8)

            lbl2, num2 = ndi.label(binary_roi > 0)
            if num2 > 1:
                sizes2 = ndi.sum(binary_roi > 0, lbl2, range(1, num2 + 1))
                binary_roi = (lbl2 == (np.argmax(sizes2) + 1)).astype(np.uint8)

    # 9. Inserción de la máscara en las coordenadas nativas del volumen original
    full_mask = np.zeros(nifti_raw.shape[:3], dtype=np.uint8)
    full_mask[:, :, z_start:z_end] = binary_roi
    del binary_roi
    gc.collect()

    # 10. Postprocesamiento morfológico
    if organ_key == "craneo":
        full_mask[:, :, z_start:z_end] = ((full_mask[:, :, z_start:z_end] > 0) & (roi_data >= 200.0)).astype(np.uint8)
    elif organ_key != "cerebro_cerebelo":
        lbl, num = ndi.label(full_mask)
        if num > 0:
            sizes = ndi.sum(full_mask, lbl, range(1, num + 1))
            valid_lbls = np.where(sizes >= 200)[0] + 1
            full_mask = np.isin(lbl, valid_lbls).astype(np.uint8)
        full_mask = ndi.binary_fill_holes(full_mask).astype(np.uint8)

    return full_mask


def segment_organ_ct(
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
            f'Estructura "{organ}" no reconocida para MONAI CT. '
            f"Estructuras disponibles: {available}"
        )

    display_name = organ_info["display_name"]
    task = organ_info.get("task", "wholeBody_ct_segmentation")
    target_classes = organ_info.get("target_classes", [50])

    logger.info("SEGMENTACIÓN CT (MONAI): %s (%s)", display_name, organ_key)
    logger.info("  Estructura  : %s (%s)", display_name, organ_key)
    logger.info("  Modelo      : MONAI SegResNet (wholeBody_ct_segmentation)")
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

    t_start = time.time()
    binary_mask = _run_monai_ct_inference(
        nifti_raw=nifti_raw,
        organ_key=organ_key,
        target_classes=target_classes,
        voxel_spacing=voxel_spacing,
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
    seg_header["descrip"] = np.bytes_(f"CT {organ_key} MONAI segmentation"[:80])

    stats = _compute_segmentation_stats(binary_mask, raw_affine, voxel_spacing, nifti_raw=nifti_raw)
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
            base = "ct_volume"
        output_basename = f"{base}_{organ_key}_ct_seg"

    nifti_out_path = os.path.join(output_dir, f"{output_basename}.nii.gz")
    json_out_path = os.path.join(output_dir, f"{output_basename}.json")
    png_out_path = os.path.join(output_dir, f"{output_basename}.png")

    nib.save(seg_nifti, nifti_out_path)
    logger.info("Máscara NIfTI guardada: %s", nifti_out_path)

    thumbnail_path = None
    try:
        from create_segmentation_thumbnail import create_segmentation_thumbnail
        thumbnail_path = create_segmentation_thumbnail(
            seg_volume=seg_nifti,
            output_path=png_out_path,
            underlay_volume=input_path,
        )
        logger.info("Miniatura de segmentación guardada: %s", thumbnail_path)
    except Exception as exc:
        logger.warning("No se pudo generar la miniatura de segmentación: %s", exc)

    metadata = {
        "study_name": output_basename.replace(f"_{organ_key}_ct_seg", ""),
        "modality": "CT",
        "organ": organ_key,
        "organ_display_name": display_name,
        "organ_description": organ_info["description"],
        "model": "MONAI_SegResNet",
        "model_version": "wholeBody_ct_segmentation_v0.2.5",
        "model_task": task,
        "backend": "monai",
        "device": "gpu" if cuda and torch_cuda_available() else "cpu",
        "fast_mode": fast,
        "input_volume": input_path,
        "output_nifti": os.path.abspath(nifti_out_path),
        "output_json": os.path.abspath(json_out_path),
        "output_thumbnail": os.path.abspath(thumbnail_path) if thumbnail_path else "",
        "volume_dimensions": list(nifti_raw.shape[:3]),
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



def main():
    parser = argparse.ArgumentParser(description="Segmentación de CT (Cerebro+Cerebelo, Corazón completo y Cráneo) con MONAI.")
    parser.add_argument("-i", "--input", default=None, help="Ruta al volumen NIfTI de entrada (CT).")
    parser.add_argument("-g", "--organ", default="cerebro_cerebelo", choices=list(ORGAN_REGISTRY.keys()) + list(ALIAS_MAP.keys()))
    parser.add_argument("-o", "--output-dir", default=None, help="Directorio de salida para NIfTI y JSON.")
    parser.add_argument("--output-basename", default=None, help="Nombre base para los archivos de salida.")
    parser.add_argument("--cpu", action="store_true", help="Forzar inferencia en CPU.")
    parser.add_argument("--fast", action="store_true", help="Modo rápido (resolución 3.0mm).")
    parser.add_argument("--list-organs", action="store_true", help="Listar estructuras disponibles.")
    args = parser.parse_args()

    if args.list_organs:
        print("Estructuras disponibles en MONAI CT:")
        for k, v in ORGAN_REGISTRY.items():
            print(f'  {k:20s} - {v["display_name"]} ({v["description"]})')
        return

    if not args.input:
        parser.error("Se requiere el argumento -i/--input para segmentar.")

    segment_organ_ct(
        input_volume=args.input,
        organ=args.organ,
        cuda=not args.cpu,
        fast=args.fast,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
    )


if __name__ == "__main__":
    main()
