#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
from datetime import datetime

import nibabel as nib
import numpy as np

try:
    from calcular_fwhm import _extract_profile_1d, _gaussian_model, calcular_fwhm
    from label_locs import DEFAULT_REFERENCE_COORDS, etiquetar_puntos_actividad
    from loc_puntos_actividad import DEFAULT_THRESHOLD, localizar_puntos_actividad
except ImportError:
    from processing.calcular_fwhm import _extract_profile_1d, _gaussian_model, calcular_fwhm
    from processing.label_locs import DEFAULT_REFERENCE_COORDS, etiquetar_puntos_actividad
    from processing.loc_puntos_actividad import DEFAULT_THRESHOLD, localizar_puntos_actividad

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analisis_resolucion_espacial")


def _extract_patch_2d(slice_2d, center_ij, patch_size=40, pad_value=0.0):
    half = patch_size // 2
    ci, cj = int(center_ij[0]), int(center_ij[1])
    h, w = slice_2d.shape

    i_start = ci - half
    i_end = i_start + patch_size
    j_start = cj - half
    j_end = j_start + patch_size

    patch = np.full((patch_size, patch_size), pad_value, dtype=np.float32)

    # Coordenadas válidas en la fuente
    src_i_min = max(0, i_start)
    src_i_max = min(h, i_end)
    src_j_min = max(0, j_start)
    src_j_max = min(w, j_end)

    # Coordenadas correspondientes en el destino
    dst_i_min = src_i_min - i_start
    dst_i_max = dst_i_min + (src_i_max - src_i_min)
    dst_j_min = src_j_min - j_start
    dst_j_max = dst_j_min + (src_j_max - src_j_min)

    if src_i_max > src_i_min and src_j_max > src_j_min:
        patch[dst_i_min:dst_i_max, dst_j_min:dst_j_max] = slice_2d[src_i_min:src_i_max, src_j_min:src_j_max]

    return patch


def ejecutar_analisis_resolucion_espacial(
    pet_volume,
    threshold=DEFAULT_THRESHOLD,
    slice_k=None,
    reference_coords=None,
    patch_size=40,
    min_voxels=1,
    output_dir=None,
    pet_dicom_dir=None,
    verbose=False,
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    if reference_coords is None:
        reference_coords = list(DEFAULT_REFERENCE_COORDS)

    if isinstance(pet_volume, str):
        if not os.path.isfile(pet_volume):
            logger.error("Volumen PET no encontrado: %s", pet_volume)
            raise FileNotFoundError(f"Volumen PET no encontrado: {pet_volume}")
        nii = nib.load(pet_volume)
        vol_path = pet_volume
    elif isinstance(pet_volume, nib.Nifti1Image):
        nii = pet_volume
        vol_path = getattr(nii, "get_filename", lambda: "<en memoria>")()
    else:
        raise TypeError("pet_volume debe ser una ruta (str) o un objeto nibabel.Nifti1Image")

    volume = np.asarray(nii.dataobj, dtype=np.float64)
    affine = nii.affine

    spacing_mm = [
        float(np.linalg.norm(affine[:3, 0])),  # dx (X, columnas)
        float(np.linalg.norm(affine[:3, 1])),  # dy (Y, filas)
        float(np.linalg.norm(affine[:3, 2])),  # dz (Z, cortes)
    ]
    dx, dy, dz = spacing_mm

    logger.info("Iniciando análisis de resolución espacial.")
    logger.info("Dimensiones del volumen: %s | Espaciado: [%.3f, %.3f, %.3f] mm", list(volume.shape), dx, dy, dz)

    # Localizar puntos de máxima actividad en el corte axial
    puntos_detectados = localizar_puntos_actividad(
        pet_volume=nii,
        threshold=threshold,
        output_json=False,
        min_voxels=min_voxels,
        slice_k=slice_k,
        verbose=verbose,
    )

    if not puntos_detectados:
        logger.warning("No se detectaron puntos de actividad con el umbral especificado.")
        return {
            "num_puntos": 0,
            "puntos": [],
            "voxel_spacing_mm": spacing_mm,
            "volume_shape": list(volume.shape),
            "corte_axial_k": slice_k,
            "perfiles": {},
            "parches_2d": {},
        }

    corte_k = int(puntos_detectados[0]["voxel_ijk"][2])

    # Etiquetar puntos según coordenadas de referencia
    puntos_etiquetados = etiquetar_puntos_actividad(
        puntos=puntos_detectados,
        reference_coords=reference_coords,
        output_json=False,
    )

    # Calcular FWHM mediante ajuste Gaussiano en los tres ejes
    resultados_fwhm = calcular_fwhm(
        pet_volume=nii,
        puntos_json=puntos_etiquetados,
        output_json=False,
        pet_dicom_dir=pet_dicom_dir,
        verbose=verbose,
    )

    fwhm_by_id = {r["punto_id"]: r for r in resultados_fwhm}

    puntos_completos = []
    perfiles = {}
    parches_2d = {}

    half_patch = patch_size // 2

    for p in puntos_etiquetados:
        pid = p["punto_id"]
        fwhm_info = fwhm_by_id.get(pid, {})

        punto_entry = dict(p)
        punto_entry.update({
            "fwhm_x_mm": fwhm_info.get("fwhm_x_mm"),
            "fwhm_y_mm": fwhm_info.get("fwhm_y_mm"),
            "fwhm_z_mm": fwhm_info.get("fwhm_z_mm"),
            "fwhm_transaxial_mm": fwhm_info.get("fwhm_transaxial_mm"),
            "ajuste_x": fwhm_info.get("ajuste_x", {}),
            "ajuste_y": fwhm_info.get("ajuste_y", {}),
            "ajuste_z": fwhm_info.get("ajuste_z", {}),
        })
        puntos_completos.append(punto_entry)

        # Extracción de perfiles 1D para visualización
        i0, j0, k0 = [int(v) for v in p["voxel_ijk"]]
        perfiles[pid] = {}

        for axis_idx, (axis_name, ax_sp) in enumerate(zip(["x", "y", "z"], spacing_mm)):
            x_mm, raw_profile = _extract_profile_1d(volume, (i0, j0, k0), axis=axis_idx, spacing_mm=ax_sp)
            fit = fwhm_info.get(f"ajuste_{axis_name}", {})

            # Si el ajuste convergió, evaluamos la función Gaussiana en una malla fina para graficar suave
            if fit.get("converged"):
                x_dense = np.linspace(float(x_mm.min()), float(x_mm.max()), 300)
                y_fit_dense = _gaussian_model(
                    x_dense,
                    float(fit["a"]),
                    float(fit["b"]),
                    float(fit["c"]),
                    float(fit["d"]),
                )
            else:
                x_dense = x_mm
                y_fit_dense = raw_profile

            perfiles[pid][axis_name] = {
                "x_mm": x_mm.tolist(),
                "actividad": raw_profile.tolist(),
                "x_dense_mm": x_dense.tolist(),
                "curva_ajustada": y_fit_dense.tolist(),
                "fwhm_mm": fit.get("fwhm_mm"),
                "r_squared": fit.get("r_squared"),
                "sigma_mm": fit.get("d"),
                "converged": fit.get("converged", False),
            }

        # Extracción de recortes 2D (40x40) en planos XY y ZY con dimensiones físicas reales
        # Plano XY (Corte axial k0):
        # Eje horizontal = X (columna i, espaciado dx)
        # Eje vertical = Y (fila j, espaciado dy)
        slice_xy = volume[:, :, k0].T  # Transpuesto para formato estándar fila=Y, col=X
        # Centro en coordenadas de matriz (fila j0, col i0)
        patch_xy = _extract_patch_2d(slice_xy, (j0, i0), patch_size=patch_size)

        # Dimensiones físicas del parche XY en mm
        # Centrado en (0, 0) para graficar las barras de FWHM
        extent_xy_mm = [
            -half_patch * dx,
            half_patch * dx,
            -half_patch * dy,
            half_patch * dy,
        ]

        # Plano ZY (Corte sagital/axial en columna i0):
        # Eje horizontal = Y (fila j, espaciado dy)
        # Eje vertical = Z (corte k, espaciado dz)
        slice_zy = volume[i0, :, :].T  # Fila = Z, Columna = Y
        patch_zy = _extract_patch_2d(slice_zy, (k0, j0), patch_size=patch_size)

        extent_zy_mm = [
            -half_patch * dy,
            half_patch * dy,
            -half_patch * dz,
            half_patch * dz,
        ]

        parches_2d[pid] = {
            "patch_xy": patch_xy.tolist(),
            "extent_xy_mm": extent_xy_mm,
            "fwhm_h_xy_mm": punto_entry["fwhm_x_mm"],
            "fwhm_v_xy_mm": punto_entry["fwhm_y_mm"],
            "patch_zy": patch_zy.tolist(),
            "extent_zy_mm": extent_zy_mm,
            "fwhm_h_zy_mm": punto_entry["fwhm_y_mm"],
            "fwhm_v_zy_mm": punto_entry["fwhm_z_mm"],
            "spacing_xy": [dx, dy],
            "spacing_zy": [dy, dz],
        }

    resultado_final = {
        "script": "analisis_resolucion_espacial.py",
        "timestamp": datetime.now().isoformat(),
        "input_volume": str(vol_path),
        "volume_shape": list(volume.shape),
        "voxel_spacing_mm": [round(s, 6) for s in spacing_mm],
        "corte_axial_k": corte_k,
        "coordenadas_referencia": [[float(c[0]), float(c[1])] for c in reference_coords],
        "num_puntos": len(puntos_completos),
        "puntos": puntos_completos,
        "perfiles": perfiles,
        "parches_2d": parches_2d,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(str(vol_path)))[0]
        if base_name.endswith(".nii"):
            base_name = os.path.splitext(base_name)[0]
        out_json_path = os.path.join(output_dir, f"{base_name}_analisis_resolucion.json")
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(resultado_final, f, indent=2, ensure_ascii=False)
        logger.info("Resultados completos exportados a: %s", out_json_path)

    return resultado_final


def main():
    parser = argparse.ArgumentParser(
        description="Flujo completo de análisis de resolución espacial en volúmenes PET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pet-volume", required=True,
        help="Ruta al archivo NIfTI (.nii o .nii.gz) del volumen PET"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
        help=f"Umbral relativo [0.0-1.0]. Por defecto: {DEFAULT_THRESHOLD}"
    )
    parser.add_argument(
        "--slice-k", "-k", type=int, default=None,
        help="Índice del corte transaxial a evaluar (por defecto: automático)"
    )
    parser.add_argument(
        "--coords", nargs="+", default=None,
        help="Coordenadas de referencia 'X,Y'. Por defecto: '0,1' '10,0' '20,0'"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directorio donde exportar el JSON de resultados"
    )
    parser.add_argument(
        "--pet-dicom-dir", default=None,
        help="Directorio DICOM de la serie PET para metadatos"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Mostrar mensajes detallados de ejecución"
    )

    args = parser.parse_args()

    ref_coords = None
    if args.coords:
        ref_coords = []
        for pair_str in args.coords:
            parts = pair_str.replace("(", "").replace(")", "").split(",")
            if len(parts) == 2:
                ref_coords.append((float(parts[0].strip()), float(parts[1].strip())))

    res = ejecutar_analisis_resolucion_espacial(
        pet_volume=args.pet_volume,
        threshold=args.threshold,
        slice_k=args.slice_k,
        reference_coords=ref_coords,
        output_dir=args.output_dir,
        pet_dicom_dir=args.pet_dicom_dir,
        verbose=args.verbose,
    )

    print("=" * 70)
    print(f"RESUMEN ANÁLISIS RESOLUCIÓN ESPACIAL - {res['num_puntos']} PUNTO(S)")
    print("=" * 70)
    for p in res["puntos"]:
        lbl = p.get("label", "N/A")
        px, py, pz = p.get("patient_mm", [0, 0, 0])
        fx = p.get("fwhm_x_mm") or 0.0
        fy = p.get("fwhm_y_mm") or 0.0
        fz = p.get("fwhm_z_mm") or 0.0
        ft = p.get("fwhm_transaxial_mm") or 0.0
        print(f"  Punto {p['punto_id']:>2} [{lbl:<10}]: Coord=({px:+7.2f}, {py:+7.2f}, {pz:+7.2f}) mm")
        print(f"       FWHM: X={fx:6.3f} mm, Y={fy:6.3f} mm, Z={fz:6.3f} mm | Transaxial={ft:6.3f} mm")
    print("=" * 75)


if __name__ == "__main__":
    main()
