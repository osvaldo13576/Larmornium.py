#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import argparse
import numpy as np
import nibabel as nib

# Asegurar importación de módulos hermanos en 'processing' o en el PATH
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from segmentation_anato_ct_fantoma_agua import segment_water_phantom_ct
except ImportError:
    from processing.segmentation_anato_ct_fantoma_agua import segment_water_phantom_ct

logger = logging.getLogger("analisis_uniformidad")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _calcular_promedio(suv_voxels: np.ndarray) -> float:
    if suv_voxels.size == 0:
        return 0.0
    return float(np.mean(suv_voxels))


def _calcular_desviacion_estandar(suv_voxels: np.ndarray) -> float:
    if suv_voxels.size == 0:
        return 0.0
    return float(np.std(suv_voxels))


def _calcular_coeficiente_variacion(suv_voxels: np.ndarray) -> float:
    if suv_voxels.size == 0:
        return 0.0
    mean_val = float(np.mean(suv_voxels))
    if mean_val == 0.0:
        return 0.0
    return float((np.std(suv_voxels) / mean_val) * 100.0)


# Registro oficial de operaciones de uniformidad por corte:
UNIFORMITY_OPERATIONS = [
    {
        "id": "promedio",
        "label": "Promedio",
        "color": "blue",
        "y_label": "Promedio [SUV]",
        "x_label": "Corte",
        "func": _calcular_promedio,
    },
    {
        "id": "desviacion_estandar",
        "label": "Desviación estándar",
        "color": "green",
        "y_label": "Desviación estándar [SUV]",
        "x_label": "Corte",
        "func": _calcular_desviacion_estandar,
    },
    {
        "id": "coeficiente_variacion",
        "label": "Coeficiente de variación",
        "color": "red",
        "y_label": "Coeficiente de variación [%]",
        "x_label": "Corte",
        "func": _calcular_coeficiente_variacion,
    },
]


def obtener_operaciones_disponibles():
    return list(UNIFORMITY_OPERATIONS)


def ejecutar_analisis_uniformidad(
    fusion_input,
    hu_min: float = -125.0,
    hu_max: float = 125.0,
    morph_code: str = "d1d_d1e_d1e_d1d",
    operations: list = None,
    output_dir: str = None,
    output_basename: str = None,
    dicom_root: str = None,
    larmornium_files_dir: str = None,
    verbose: bool = True,
) -> dict:
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logger.info("Iniciando análisis de uniformidad PET/CT...")
    logger.info("  Regla de segmentación: HU=[%.1f, %.1f], Morfología='%s'", hu_min, hu_max, morph_code)

    ops_to_run = operations if operations is not None else UNIFORMITY_OPERATIONS

    # Extracción y resolución de volúmenes CT y PET
    ct_data_3d = None
    pet_data_3d = None
    affine = np.eye(4)
    header = None
    source_desc = "Memoria"

    if isinstance(fusion_input, str):
        if not os.path.isfile(fusion_input):
            raise FileNotFoundError(f"Archivo de volumen fusionado no encontrado: {fusion_input}")
        source_desc = os.path.basename(fusion_input)
        fnii = nib.load(fusion_input)
        affine = fnii.affine
        header = fnii.header
        fdata = fnii.get_fdata().astype(np.float32)
        if fdata.ndim == 4 and fdata.shape[-1] >= 2:
            ct_data_3d = fdata[..., 0]
            pet_data_3d = fdata[..., 1]
        elif fdata.ndim == 3:
            ct_data_3d = fdata
            pet_data_3d = np.zeros_like(fdata)
        else:
            raise ValueError(f"Dimensiones de NIfTI no compatibles con fusión: {fdata.shape}")
    elif isinstance(fusion_input, nib.Nifti1Image):
        source_desc = "Objeto NIfTI"
        affine = fusion_input.affine
        header = fusion_input.header
        fdata = fusion_input.get_fdata().astype(np.float32)
        if fdata.ndim == 4 and fdata.shape[-1] >= 2:
            ct_data_3d = fdata[..., 0]
            pet_data_3d = fdata[..., 1]
        else:
            raise ValueError(f"Objeto NIfTI no contiene canales CT y PET fusionados: {fdata.shape}")
    elif isinstance(fusion_input, dict):
        if "nii_path" in fusion_input and os.path.isfile(fusion_input["nii_path"]):
            return ejecutar_analisis_uniformidad(
                fusion_input["nii_path"],
                hu_min=hu_min,
                hu_max=hu_max,
                morph_code=morph_code,
                operations=ops_to_run,
                output_dir=output_dir,
                output_basename=output_basename,
                dicom_root=dicom_root,
                larmornium_files_dir=larmornium_files_dir,
                verbose=verbose,
            )
        elif "ct_volume" in fusion_input and "pet_volume" in fusion_input:
            source_desc = "Arreglos en memoria"
            ct_raw = np.asarray(fusion_input["ct_volume"], dtype=np.float32)
            pet_raw = np.asarray(fusion_input["pet_volume"], dtype=np.float32)
            # Si vienen en formato (Z, Y, X) de visualización, transponer a (X, Y, Z)
            if ct_raw.shape[0] < ct_raw.shape[1] and ct_raw.shape[0] < ct_raw.shape[2]:
                ct_data_3d = np.transpose(ct_raw, (2, 1, 0))
                pet_data_3d = np.transpose(pet_raw, (2, 1, 0))
            else:
                ct_data_3d = ct_raw
                pet_data_3d = pet_raw
            if "affine" in fusion_input:
                affine = fusion_input["affine"]
        else:
            raise ValueError("Diccionario de entrada no contiene 'nii_path' ni volúmenes 'ct_volume'/'pet_volume'")
    else:
        raise TypeError(f"Tipo de entrada no soportado para análisis de uniformidad: {type(fusion_input)}")

    nx, ny, nz = ct_data_3d.shape
    logger.info("  Dimensiones del volumen: %dx%dx%d (X, Y, Z) desde '%s'", nx, ny, nz, source_desc)

    # Segmentación del fantoma de agua en el volumen CT
    ct_nii_obj = nib.Nifti1Image(ct_data_3d, affine, header)
    seg_nifti, seg_meta = segment_water_phantom_ct(
        input_volume=ct_nii_obj,
        hu_min=hu_min,
        hu_max=hu_max,
        morph_code=morph_code,
        output_dir=output_dir,
        output_basename=output_basename,
        quiet=not verbose,
    )

    voi_mask_3d = (seg_nifti.get_fdata() > 0).astype(bool)
    total_voxels = int(np.sum(voi_mask_3d))
    logger.info("  VOI creada: %d vóxeles totales (Volumen: %.2f cm3)",
                total_voxels, seg_meta.get("volume_cm3", 0.0))

    if total_voxels == 0:
        logger.warning("No se detectó volumen del fantoma de agua con la regla especificada.")

    # Cálculo de operaciones corte a corte usando el SUV de PET
    active_slices = []
    op_values = {op["id"]: [] for op in ops_to_run}

    for z in range(nz):
        slice_voi = voi_mask_3d[:, :, z]
        if not np.any(slice_voi):
            continue

        suv_in_slice = pet_data_3d[:, :, z][slice_voi]
        # Filtrar posibles valores no finitos (NaN / Inf)
        suv_in_slice = suv_in_slice[np.isfinite(suv_in_slice)]
        if suv_in_slice.size == 0:
            continue

        active_slices.append(z + 1)  # Representación 1-based para ejes de visualización clínica

        for op in ops_to_run:
            op_id = op["id"]
            calc_fn = op["func"]
            try:
                val = float(calc_fn(suv_in_slice))
            except Exception as ex:
                logger.warning("Error al calcular operación '%s' en corte %d: %s", op_id, z + 1, ex)
                val = 0.0
            op_values[op_id].append(val)

    # Métricas globales en toda la VOI 3D
    all_suv = pet_data_3d[voi_mask_3d]
    all_suv = all_suv[np.isfinite(all_suv)]
    if all_suv.size > 0:
        overall_mean = float(np.mean(all_suv))
        overall_std = float(np.std(all_suv))
        overall_cv = float((overall_std / overall_mean * 100.0)) if overall_mean > 0 else 0.0
    else:
        overall_mean = overall_std = overall_cv = 0.0

    # Empaquetar operaciones con sus curvas y metadatos
    operations_result = []
    for op in ops_to_run:
        vals = op_values[op["id"]]
        operations_result.append({
            "id": op["id"],
            "label": op["label"],
            "color": op.get("color", "blue"),
            "y_label": op.get("y_label", op["label"]),
            "x_label": op.get("x_label", "Corte"),
            "values": vals,
            "global_mean": float(np.mean(vals)) if vals else 0.0,
            "global_std": float(np.std(vals)) if vals else 0.0,
            "global_min": float(np.min(vals)) if vals else 0.0,
            "global_max": float(np.max(vals)) if vals else 0.0,
        })

    results = {
        "status": "success",
        "source": source_desc,
        "num_slices_total": nz,
        "num_slices_analyzed": len(active_slices),
        "slice_indices": active_slices,
        "operations": operations_result,
        "global_metrics": {
            "mean_suv": round(overall_mean, 4),
            "std_suv": round(overall_std, 4),
            "cv_percent": round(overall_cv, 2),
            "total_voxels": total_voxels,
            "volume_cm3": seg_meta.get("volume_cm3", 0.0),
            "z_range": seg_meta.get("z_slice_range", [0, 0]),
            "voxel_spacing_mm": seg_meta.get("voxel_spacing_mm", [1.0, 1.0, 1.0]),
        },
        "phantom_metadata": seg_meta,
        "voi_mask": voi_mask_3d,
    }

    logger.info("Análisis de uniformidad finalizado: %d cortes analizados. Promedio global=%.3f SUV, CV=%.2f%%",
                len(active_slices), overall_mean, overall_cv)

    return results


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Análisis de uniformidad PET/CT mediante segmentación de fantoma de agua y métricas por corte.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python3 analisis_uniformidad.py \\
      --fusion ./volumes/FUSION_PET_CT_SE000001_SE000003.nii.gz \\
      --morph-code d1d_d1e_d1e_d1d \\
      --hu-min -125.0 \\
      --hu-max 125.0
        """
    )
    parser.add_argument(
        "--fusion", "-f",
        required=True,
        type=str,
        help="Ruta al volumen NIfTI fusionado (.nii o .nii.gz).",
    )
    parser.add_argument(
        "--morph-code", "-m",
        type=str,
        default="d1d_d1e_d1e_d1d",
        help="Cadena de operaciones morfológicas para segmentación (default: d1d_d1e_d1e_d1d).",
    )
    parser.add_argument(
        "--hu-min",
        type=float,
        default=-125.0,
        help="Umbral mínimo HU para segmentar agua (default: -125.0).",
    )
    parser.add_argument(
        "--hu-max",
        type=float,
        default=125.0,
        help="Umbral máximo HU para segmentar agua (default: 125.0).",
    )
    parser.add_argument(
        "--output-json", "-o",
        type=str,
        default=None,
        help="Ruta para guardar los resultados en formato JSON.",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    results = ejecutar_analisis_uniformidad(
        fusion_input=args.fusion,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        morph_code=args.morph_code,
        verbose=True,
    )

    if args.output_json:
        # Serializar resultados (excluyendo la máscara booleana 3D)
        export_dict = dict(results)
        export_dict.pop("voi_mask", None)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=2, ensure_ascii=False)
        print(f"Resultados guardados en: {args.output_json}")
    else:
        print("\n--- RESUMEN DE ANÁLISIS DE UNIFORMIDAD ---")
        gm = results["global_metrics"]
        print(f"Cortes analizados: {results['num_slices_analyzed']}")
        print(f"Promedio global: {gm['mean_suv']} SUV")
        print(f"Desv. Est. global: {gm['std_suv']} SUV")
        print(f"Coeficiente de variación: {gm['cv_percent']}%")
        print(f"Volumen VOI: {gm['volume_cm3']} cm3 ({gm['total_voxels']} vóxeles)")


if __name__ == "__main__":
    main()
