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
import scipy.ndimage as ndimage

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("loc_puntos_actividad")

# Umbral relativo por defecto (10% del máximo global)
DEFAULT_THRESHOLD = 0.10


def _voxel_to_patient_coords(vox_ijk, affine):
    vox = np.atleast_2d(vox_ijk).astype(float)
    ones = np.ones((vox.shape[0], 1))
    vox_h = np.hstack([vox, ones])
    patient_h = (affine @ vox_h.T).T
    patient = patient_h[:, :3]
    if patient.shape[0] == 1:
        return patient[0]
    return patient


def localizar_puntos_actividad(pet_volume, threshold=DEFAULT_THRESHOLD,
                                output_json=None, min_voxels=1,
                                slice_k=None,
                                verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)

    if isinstance(pet_volume, str):
        if not os.path.isfile(pet_volume):
            logger.error("Archivo NIfTI no encontrado: %s", pet_volume)
            raise FileNotFoundError(f"Archivo NIfTI no encontrado: {pet_volume}")
        logger.info("Cargando volumen PET: %s", pet_volume)
        nii = nib.load(pet_volume)
        volume_path = pet_volume
    elif isinstance(pet_volume, nib.Nifti1Image):
        nii = pet_volume
        volume_path = getattr(nii, "get_filename", lambda: "<en memoria>")()
    else:
        raise TypeError("pet_volume debe ser una ruta (str) o un nibabel.Nifti1Image")

    volume = np.asarray(nii.dataobj, dtype=np.float32)
    affine = nii.affine

    # Extraer espaciado de vóxel desde la matriz afín
    spacing_mm = [
        float(np.linalg.norm(affine[:3, 0])),
        float(np.linalg.norm(affine[:3, 1])),
        float(np.linalg.norm(affine[:3, 2])),
    ]

    global_max = float(volume.max())
    global_min = float(volume.min())

    logger.info("Dimensiones del volumen  : %s", list(volume.shape))
    logger.info("Espaciado vóxel (mm)     : [%.4f, %.4f, %.4f]", *spacing_mm)
    logger.info("Rango de actividad       : [%.2f, %.2f]", global_min, global_max)
    logger.info("Umbral relativo          : %.2f (%.2f del máximo)", threshold, threshold * global_max)

    if global_max <= 0:
        logger.error("El volumen PET tiene todos los valores <= 0. Verifique el archivo de entrada.")
        return []

    # Determinación del corte transaxial (eje Z, corte k)
    num_slices_z = volume.shape[2]
    if slice_k is None:
        # Corte axial que contiene el máximo global de actividad
        max_flat_idx = int(volume.argmax())
        k0 = int(np.unravel_index(max_flat_idx, volume.shape)[2])
        logger.info("Corte axial central seleccionado (máximo global): k = %d / %d", k0, num_slices_z - 1)
    else:
        k0 = int(slice_k)
        if not (0 <= k0 < num_slices_z):
            logger.error("Corte axial especificado k=%d fuera de rango [0, %d]", k0, num_slices_z - 1)
            raise ValueError(f"Corte k={k0} fuera de rango [0, {num_slices_z - 1}]")
        logger.info("Corte axial especificado por usuario: k = %d / %d", k0, num_slices_z - 1)

    # Coordenada Z en espacio del paciente para el corte k0
    pt_z_ref = _voxel_to_patient_coords([0, 0, k0], affine)
    z_ref_mm = round(float(pt_z_ref[2]), 3)
    logger.info("Coordenada Z del paciente para el corte k=%d: z = %+.2f mm", k0, z_ref_mm)

    # Extracción del corte 2D y binarización de componentes conexas
    slice_2d = volume[:, :, k0].astype(np.float32)
    slice_max = float(slice_2d.max())
    umbral_abs = threshold * global_max

    binary_2d = slice_2d >= umbral_abs
    labeled_2d, num_features = ndimage.label(binary_2d)

    logger.info("Componentes conexas encontradas en corte k=%d: %d", k0, num_features)

    if num_features == 0:
        logger.warning("No se encontraron puntos de actividad en el corte k=%d con umbral %.2f.", k0, threshold)
        return []

    # Análisis de cada componente conexa dentro del corte
    puntos = []
    for lab in range(1, num_features + 1):
        mask_2d = (labeled_2d == lab)
        n_vox = int(mask_2d.sum())
        if n_vox < min_voxels:
            logger.debug("Componente %d ignorada: %d vóxeles < mínimo (%d)", lab, n_vox, min_voxels)
            continue

        comp_vals = slice_2d * mask_2d.astype(np.float32)

        # Máximo local dentro del corte
        max_val = float(comp_vals.max())
        max_flat = comp_vals.argmax()
        max_ij = np.unravel_index(max_flat, slice_2d.shape)  # (i, j)
        i0, j0 = int(max_ij[0]), int(max_ij[1])
        max_vox_ijk = [i0, j0, k0]

        # Centro de masa 2D ponderado por actividad
        cm_2d = ndimage.center_of_mass(comp_vals)
        ci, cj = float(cm_2d[0]), float(cm_2d[1])
        cm_vox_ijk = [round(ci, 3), round(cj, 3), float(k0)]

        # Promedio de actividad en la componente
        mean_val = float(comp_vals.sum()) / float(n_vox)

        # Coordenadas del paciente en mm (z_mm común para todos los puntos)
        patient_max = _voxel_to_patient_coords(max_vox_ijk, affine).tolist()
        patient_cm = _voxel_to_patient_coords(cm_vox_ijk, affine).tolist()

        puntos.append({
            "punto_id": lab,
            "voxel_ijk": max_vox_ijk,
            "patient_mm": [round(float(patient_max[0]), 3), round(float(patient_max[1]), 3), z_ref_mm],
            "cm_voxel": cm_vox_ijk,
            "cm_patient_mm": [round(float(patient_cm[0]), 3), round(float(patient_cm[1]), 3), z_ref_mm],
            "valor_maximo": round(max_val, 4),
            "valor_promedio": round(mean_val, 4),
            "num_voxeles": n_vox,
        })

    # Ordenación por valor máximo descendente
    puntos.sort(key=lambda p: -p["valor_maximo"])

    # Reasignación de identificadores consecutivos según actividad
    for idx, p in enumerate(puntos):
        p["punto_id"] = idx + 1

    logger.info("=" * 60)
    logger.info("PUNTOS DE ACTIVIDAD EN CORTE k=%d (z=%+.2f mm): %d", k0, z_ref_mm, len(puntos))
    logger.info("=" * 60)
    for p in puntos:
        logger.info(
            "  Punto %d: Máx=%.2f | Vóxel=(%d, %d, %d) | Coord(mm)=(%+.1f, %+.1f, %+.1f) | Vóxeles=%d",
            p["punto_id"],
            p["valor_maximo"],
            *p["voxel_ijk"],
            *p["patient_mm"],
            p["num_voxeles"],
        )

    if output_json is not False:
        if output_json is None:
            if isinstance(pet_volume, str):
                base = os.path.splitext(os.path.splitext(pet_volume)[0])[0]
                output_json = base + "_puntos_actividad.json"
            else:
                output_json = "./puntos_actividad.json"

        out_dir = os.path.dirname(os.path.abspath(output_json))
        os.makedirs(out_dir, exist_ok=True)

        result = {
            "script": "loc_puntos_actividad.py",
            "timestamp": datetime.now().isoformat(),
            "input_volume": str(volume_path) if volume_path else "<en memoria>",
            "volumen_dimensiones": list(volume.shape),
            "voxel_spacing_mm": [round(s, 6) for s in spacing_mm],
            "corte_axial_k": k0,
            "coordenada_z_paciente_mm": z_ref_mm,
            "mismo_corte_transaxial": True,
            "actividad_maxima_global": round(global_max, 4),
            "actividad_maxima_corte": round(slice_max, 4),
            "umbral_relativo": threshold,
            "umbral_absoluto": round(umbral_abs, 4),
            "num_puntos_encontrados": len(puntos),
            "puntos": puntos,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info("Resultado exportado a: %s", output_json)
    return puntos


def main():
    parser = argparse.ArgumentParser(
        description="Localiza puntos de máxima actividad en volúmenes PET NIfTI "
                    "dentro del mismo corte axial (fantomas de resolución espacial)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  # A partir de un volumen NIfTI ya generado (corte axial de máxima actividad automático):
  python3 loc_puntos_actividad.py \\
      --pet-volume ./volumes/fwhm_pet.nii.gz \\
      --output ./resultados/puntos_actividad.json

  # Especificando manualmente el corte axial k=75:
  python3 loc_puntos_actividad.py \\
      --pet-volume ./volumes/fwhm_pet.nii.gz \\
      --slice-k 75 \\
      --output ./resultados/puntos_actividad.json

  # Generando el volumen PET directamente desde los archivos DICOM:
  python3 loc_puntos_actividad.py \\
      --pet-dir PET_CT/fwhm_puntos_rad/3_puntos_11_01_24/SE000003 \\
      --dicom-root ./DICOM \\
      --threshold 0.10 \\
      --output ./resultados/puntos_actividad.json
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pet-volume",
        help="Ruta al volumen NIfTI PET generado por gen_volume_PET.py "
             "(.nii o .nii.gz)"
    )
    group.add_argument(
        "--pet-dir",
        help="Directorio de la serie PET DICOM (se generará el volumen "
             "automáticamente usando gen_volume_PET.py)"
    )

    parser.add_argument(
        "--dicom-root", default="./DICOM",
        help="Ruta raíz DICOM (requerido con --pet-dir). Por defecto: ./DICOM"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=DEFAULT_THRESHOLD,
        help=f"Umbral relativo al máximo global del volumen PET para definir "
             f"regiones de alta actividad [0.0-1.0]. Por defecto: {DEFAULT_THRESHOLD}"
    )
    parser.add_argument(
        "--min-voxels", type=int, default=1,
        help="Tamaño mínimo de una componente (vóxeles) para considerarse un "
             "punto de actividad válido. Por defecto: 1"
    )
    parser.add_argument(
        "--slice-k", "-k", type=int, default=None,
        help="Índice del corte transaxial (0-indexed) donde ubicar los puntos. "
             "Por defecto: se autodetecta el corte que contiene el máximo global."
    )
    parser.add_argument(
        "--suv", action="store_true",
        help="Convertir el volumen a SUVbw antes de localizar puntos "
             "(solo relevante con --pet-dir)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el archivo JSON con los resultados. "
             "Por defecto se genera junto al archivo NIfTI."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir información detallada"
    )

    args = parser.parse_args()

    if not (0.0 < args.threshold < 1.0):
        parser.error(f"--threshold debe ser un número entre 0.0 y 1.0 (exclusivo). "
                     f"Valor recibido: {args.threshold}")

    if args.pet_volume:
        pet_input = args.pet_volume
    else:
        try:
            from gen_volume_PET import generate_pet_volume
        except ImportError:
            logger.error("No se pudo importar gen_volume_PET.py. "
                         "Asegúrese de que el script esté en el mismo directorio.")
            sys.exit(1)

        logger.info("Generando volumen PET desde DICOM: %s", args.pet_dir)
        nii, _ = generate_pet_volume(
            pet_directory=args.pet_dir,
            dicom_root=args.dicom_root,
            output_path=None,
            convert_suv=args.suv,
            verbose=args.verbose,
        )
        pet_input = nii

    puntos = localizar_puntos_actividad(
        pet_volume=pet_input,
        threshold=args.threshold,
        output_json=args.output,
        min_voxels=args.min_voxels,
        slice_k=args.slice_k,
        verbose=args.verbose,
    )

    print("\n" + "=" * 75)
    print(f"PUNTOS DE MÁXIMA ACTIVIDAD ENCONTRADOS EN EL MISMO CORTE: {len(puntos)}")
    if puntos:
        print(f"Corte axial k = {puntos[0]['voxel_ijk'][2]}  |  Coordenada Z paciente = {puntos[0]['patient_mm'][2]:+.2f} mm")
    print("=" * 75)
    for p in puntos:
        x, y, z = p["patient_mm"]
        i, j, k = p["voxel_ijk"]
        print(f"  Punto {p['punto_id']:>2}:  Vóxel = ({i:>3}, {j:>3}, {k:>3})  |  Coords Paciente = ({x:+8.2f}, {y:+8.2f}, {z:+8.2f}) mm")
        print(f"           Máx   = {p['valor_maximo']:,.2f}  |  Promedio = {p['valor_promedio']:,.2f}  |  Vóxeles = {p['num_voxeles']}")
        print()
    print("=" * 75)
    if args.output:
        print(f"JSON exportado a: {args.output}")


if __name__ == "__main__":
    main()
