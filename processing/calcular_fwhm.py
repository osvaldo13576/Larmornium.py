#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import math
import os
import re
from datetime import datetime

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from scipy.optimize import curve_fit

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("calcular_fwhm")

# Factor de conversión de sigma a FWHM: 2 * sqrt(2 * ln(2))
FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))

# Archivos y extensiones que no corresponden a formato DICOM
_IGNORE_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
_IGNORE_PREFIXES = ("._",)
_NON_DICOM_EXT = {
    ".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff",
    ".mat", ".xlsx", ".csv", ".txt", ".nii", ".gz",
}

# Expresión regular para extraer la fracción del FOV (ej. "1/2", "1/8")
_FOV_REGEX = re.compile(r'\b(1\s*/\s*\d+)\b')


def _extract_series_metadata(pet_dicom_dir):
    if not pet_dicom_dir or not os.path.isdir(pet_dicom_dir):
        return {}

    ds = None
    for fname in sorted(os.listdir(pet_dicom_dir)):
        if fname in _IGNORE_FILES:
            continue
        if any(fname.startswith(p) for p in _IGNORE_PREFIXES):
            continue
        _, ext = os.path.splitext(fname)
        if ext.lower() in _NON_DICOM_EXT:
            continue
        fpath = os.path.join(pet_dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            ds = pydicom.dcmread(fpath, force=True)
            if hasattr(ds, "pixel_array"):
                break
        except (InvalidDicomError, Exception):
            ds = None
            continue

    if ds is None:
        logger.warning("No se pudo leer ningún archivo DICOM de: %s", pet_dicom_dir)
        return {}

    def _safe(attr, default=None):
        val = getattr(ds, attr, None)
        return str(val).strip() if val is not None else default

    series_desc = _safe("SeriesDescription")
    recon_method = _safe("ReconstructionMethod")
    conv_kernel = _safe("ConvolutionKernel")
    protocol = _safe("ProtocolName")

    # Extracción de la fracción del FOV
    fov_fraction = None
    fov_source = None

    # Prioridad 1: SeriesDescription
    if series_desc:
        m = _FOV_REGEX.search(series_desc)
        if m:
            fov_fraction = m.group(1).replace(" ", "")
            fov_source = "SeriesDescription"

    # Prioridad 2: AttenuationCorrectionMethod
    if fov_fraction is None:
        ac_method = _safe("AttenuationCorrectionMethod")
        if ac_method:
            m = _FOV_REGEX.search(ac_method)
            if m:
                fov_fraction = m.group(1).replace(" ", "")
                fov_source = "AttenuationCorrectionMethod"

    # Metadatos espaciales y de grosor de corte
    slice_thick = None
    try:
        raw_st = getattr(ds, "SliceThickness", None)
        if raw_st is not None:
            slice_thick = float(raw_st)
    except (ValueError, TypeError):
        pass

    pixel_sp = None
    try:
        raw_ps = getattr(ds, "PixelSpacing", None)
        if raw_ps is not None:
            pixel_sp = [float(v) for v in raw_ps]
    except (ValueError, TypeError):
        pass

    spacing_between = None
    try:
        raw_sbs = getattr(ds, "SpacingBetweenSlices", None)
        if raw_sbs is not None:
            spacing_between = float(raw_sbs)
    except (ValueError, TypeError):
        pass

    # Cálculo del paso axial IPP (delta Z) entre cortes consecutivos
    delta_z_ipp = None
    try:
        valid_files = [
            os.path.join(pet_dicom_dir, f) for f in sorted(os.listdir(pet_dicom_dir))
            if os.path.isfile(os.path.join(pet_dicom_dir, f))
            and f not in _IGNORE_FILES
            and not any(f.startswith(p) for p in _IGNORE_PREFIXES)
            and os.path.splitext(f)[1].lower() not in _NON_DICOM_EXT
        ]
        if len(valid_files) >= 2:
            ds0 = pydicom.dcmread(valid_files[0], force=True)
            ds1 = pydicom.dcmread(valid_files[1], force=True)
            ipp0 = getattr(ds0, "ImagePositionPatient", None)
            ipp1 = getattr(ds1, "ImagePositionPatient", None)
            if ipp0 is not None and ipp1 is not None:
                delta_z_ipp = round(abs(float(ipp1[2]) - float(ipp0[2])), 4)
    except Exception:
        pass

    meta = {
        "series_description": series_desc,
        "reconstruction_method": recon_method,
        "convolution_kernel": conv_kernel,
        "protocol_name": protocol,
        "fov_fraction": fov_fraction,
        "fov_source": fov_source,
        "slice_thickness_mm": slice_thick,
        "pixel_spacing_mm": pixel_sp,
        "spacing_between_slices_mm": spacing_between,
        "slice_spacing_ipp_mm": delta_z_ipp,
    }

    logger.info("Metadatos de la serie PET:")
    logger.info("  SeriesDescription          : %s", series_desc)
    logger.info("  ReconstructionMethod       : %s", recon_method)
    logger.info("  ConvolutionKernel          : %s", conv_kernel)
    logger.info("  ProtocolName               : %s", protocol)
    logger.info("  FOV (fracción)             : %s  [fuente: %s]", fov_fraction, fov_source)
    logger.info("  SliceThickness (0018,0050) : %s mm", f"{slice_thick:.2f}" if slice_thick is not None else "No definido")
    logger.info("  PixelSpacing (0028,0030)   : %s mm", pixel_sp)
    logger.info("  SpacingBetweenSlices       : %s mm", f"{spacing_between:.2f}" if spacing_between is not None else "No definido")
    logger.info("  Paso Axial IPP (Δz)        : %s mm", f"{delta_z_ipp:.2f}" if delta_z_ipp is not None else "No definido")

    return meta


def _gaussian_model(x, a, b, c, d):
    return a + (b - a) * np.exp(-((x - c) ** 2) / (2.0 * d ** 2))


def _fwhm_from_sigma(sigma_mm):
    return FWHM_FACTOR * abs(sigma_mm)


def _extract_profile_1d(volume, peak_ijk, axis, spacing_mm):
    i0, j0, k0 = peak_ijk
    if axis == 0:
        profile = volume[:, j0, k0].astype(np.float64)
        peak_pos = i0
    elif axis == 1:
        profile = volume[i0, :, k0].astype(np.float64)
        peak_pos = j0
    else:
        profile = volume[i0, j0, :].astype(np.float64)
        peak_pos = k0

    n = len(profile)
    x_vox = np.arange(n, dtype=np.float64)
    x_mm = (x_vox - peak_pos) * spacing_mm  # Centrado en el pico
    return x_mm, profile


def _fit_gaussian_profile(x_mm, profile, peak_mm=0.0, verbose=False):
    result = {
        "a": None, "b": None, "c": None, "d": None,
        "fwhm_mm": None, "fwhm_factor": FWHM_FACTOR,
        "r_squared": None, "rmse": None,
        "converged": False, "error_msg": None,
    }

    if profile is None or len(profile) < 5:
        result["error_msg"] = "Perfil demasiado corto para ajustar"
        return result

    peak_val = float(profile.max())
    baseline_val = float(np.percentile(profile, 10))  # Estimación del fondo

    if peak_val <= baseline_val:
        result["error_msg"] = "El pico no supera el nivel de fondo"
        return result

    # Estimación inicial de sigma a partir del ancho a mitad del máximo
    half_max = (peak_val + baseline_val) / 2.0
    mask_above_half = profile >= half_max
    n_above = int(mask_above_half.sum())
    step = float(x_mm[1] - x_mm[0]) if len(x_mm) > 1 else 1.0
    sigma_init = max(abs(n_above * step / FWHM_FACTOR), abs(step))

    p0 = [baseline_val, peak_val, peak_mm, sigma_init]

    # Límites de parámetros: a en (-inf, inf), b > fondo, c libre, d > 0
    bounds_low = [-np.inf, baseline_val, float(x_mm.min()), 1e-6]
    bounds_high = [np.inf, np.inf, float(x_mm.max()), np.inf]

    try:
        popt, _ = curve_fit(
            _gaussian_model, x_mm, profile,
            p0=p0,
            bounds=(bounds_low, bounds_high),
            maxfev=10000,
        )
        a_fit, b_fit, c_fit, d_fit = popt

        fwhm = _fwhm_from_sigma(d_fit)

        # Bondad del ajuste
        y_pred = _gaussian_model(x_mm, *popt)
        residuals = profile - y_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((profile - profile.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        rmse = float(np.sqrt(ss_res / len(profile)))

        result.update({
            "a": round(float(a_fit), 6),
            "b": round(float(b_fit), 6),
            "c": round(float(c_fit), 6),
            "d": round(float(d_fit), 6),
            "fwhm_mm": round(fwhm, 6),
            "r_squared": round(r2, 6),
            "rmse": round(rmse, 6),
            "converged": True,
            "error_msg": None,
        })

        if verbose:
            logger.debug(
                "    Ajuste: a=%.4f, b=%.4f, c=%.4f mm, d=%.4f mm, FWHM=%.4f mm (R2=%.4f)",
                a_fit, b_fit, c_fit, d_fit, fwhm, r2
            )

    except (RuntimeError, ValueError) as exc:
        result["error_msg"] = str(exc)
        if verbose:
            logger.debug("    Ajuste no convergió: %s", exc)

    return result


def calcular_fwhm(pet_volume, puntos_json, output_json=None,
                  pet_dicom_dir=None, verbose=False):
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Extracción de metadatos de la serie DICOM si se proporciona el directorio
    series_meta = _extract_series_metadata(pet_dicom_dir) if pet_dicom_dir else {}

    if isinstance(pet_volume, str):
        if not os.path.isfile(pet_volume):
            logger.error("Volumen NIfTI no encontrado: %s", pet_volume)
            raise FileNotFoundError(f"Volumen NIfTI no encontrado: {pet_volume}")
        logger.info("Cargando volumen PET: %s", pet_volume)
        nii = nib.load(pet_volume)
        vol_path = pet_volume
    elif isinstance(pet_volume, nib.Nifti1Image):
        nii = pet_volume
        vol_path = getattr(nii, "get_filename", lambda: "<en memoria>")()
    else:
        raise TypeError("pet_volume debe ser str (ruta) o nibabel.Nifti1Image")

    volume = np.asarray(nii.dataobj, dtype=np.float64)
    affine = nii.affine

    # Espaciado de vóxel desde la matriz afín
    spacing_mm = [
        float(np.linalg.norm(affine[:3, 0])),  # dx (eje X, columnas)
        float(np.linalg.norm(affine[:3, 1])),  # dy (eje Y, filas)
        float(np.linalg.norm(affine[:3, 2])),  # dz (eje Z, cortes)
    ]

    logger.info("Dimensiones volumen  : %s", list(volume.shape))
    logger.info("Espaciado vóxel (mm) : [%.4f, %.4f, %.4f]", *spacing_mm)

    if isinstance(puntos_json, str):
        if not os.path.isfile(puntos_json):
            logger.error("Archivo JSON no encontrado: %s", puntos_json)
            raise FileNotFoundError(f"Archivo JSON no encontrado: {puntos_json}")
        with open(puntos_json, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        # Soporta tanto la lista directa como el dict raíz de loc_puntos_actividad.py
        puntos = json_data.get("puntos", json_data) if isinstance(json_data, dict) else json_data
        json_path = puntos_json
    elif isinstance(puntos_json, list):
        puntos = puntos_json
        json_path = "<en memoria>"
    else:
        raise TypeError("puntos_json debe ser str (ruta al JSON) o list de dicts")

    logger.info("Puntos de actividad cargados: %d", len(puntos))

    resultados = []
    for punto in puntos:
        pid = punto.get("punto_id", "?")
        vijk = punto.get("voxel_ijk")
        patient_mm = punto.get("patient_mm")
        val_max = punto.get("valor_maximo")

        if vijk is None:
            logger.warning("Punto %s: sin campo 'voxel_ijk', omitido.", pid)
            continue

        i0, j0, k0 = int(vijk[0]), int(vijk[1]), int(vijk[2])

        logger.info("-" * 60)
        logger.info("Punto %s: vóxel=(%d, %d, %d), patient=(%.1f, %.1f, %.1f) mm",
                    pid, i0, j0, k0, *(patient_mm or [0, 0, 0]))

        resultados_ejes = {}
        fwhm_vals = {}

        for axis_idx, (axis_name, ax_sp) in enumerate(
            zip(["x", "y", "z"], spacing_mm)
        ):
            x_mm, profile = _extract_profile_1d(
                volume, (i0, j0, k0), axis=axis_idx, spacing_mm=ax_sp
            )
            fit = _fit_gaussian_profile(x_mm, profile, peak_mm=0.0, verbose=verbose)

            resultados_ejes[f"ajuste_{axis_name}"] = fit
            fwhm_vals[f"fwhm_{axis_name}_mm"] = fit["fwhm_mm"] if fit["converged"] else None

            status = (
                f"FWHM={fit['fwhm_mm']:.4f} mm (sigma={fit['d']:.4f} mm, R^2={fit['r_squared']:.4f})"
                if fit["converged"]
                else f"no convergió: {fit['error_msg']}"
            )
            logger.info("  Eje %s: %s", axis_name.upper(), status)

        # FWHM transaxial: media de X e Y si ambos convergieron
        fx = fwhm_vals.get("fwhm_x_mm")
        fy = fwhm_vals.get("fwhm_y_mm")
        if fx is not None and fy is not None:
            fwhm_transaxial = round((fx + fy) / 2.0, 6)
        elif fx is not None:
            fwhm_transaxial = fx
        elif fy is not None:
            fwhm_transaxial = fy
        else:
            fwhm_transaxial = None

        res = {
            "punto_id": pid,
            "voxel_ijk": [i0, j0, k0],
            "patient_mm": patient_mm,
            "valor_maximo": val_max,
            **resultados_ejes,
            "fwhm_x_mm": round(fwhm_vals["fwhm_x_mm"], 4) if fwhm_vals["fwhm_x_mm"] is not None else None,
            "fwhm_y_mm": round(fwhm_vals["fwhm_y_mm"], 4) if fwhm_vals["fwhm_y_mm"] is not None else None,
            "fwhm_z_mm": round(fwhm_vals["fwhm_z_mm"], 4) if fwhm_vals["fwhm_z_mm"] is not None else None,
            "fwhm_transaxial_mm": round(fwhm_transaxial, 4) if fwhm_transaxial is not None else None,
        }
        resultados.append(res)

    logger.info("=" * 60)
    logger.info("RESUMEN DE FWHM POR PUNTO")
    logger.info("=" * 60)
    for r in resultados:
        px, py, pz = r.get("patient_mm") or [0, 0, 0]
        logger.info(
            "  Punto %s: (%+.1f, %+.1f, %+.1f) mm | "
            "FWHM X=%.2f  Y=%.2f  Z=%.2f  Transaxial=%.2f (mm)",
            r["punto_id"], px, py, pz,
            r["fwhm_x_mm"] or 0.0,
            r["fwhm_y_mm"] or 0.0,
            r["fwhm_z_mm"] or 0.0,
            r["fwhm_transaxial_mm"] or 0.0,
        )

    if output_json is not False:
        if output_json is None:
            if isinstance(pet_volume, str):
                base = os.path.splitext(os.path.splitext(pet_volume)[0])[0]
                output_json = base + "_fwhm.json"
            else:
                output_json = "./fwhm_resultados.json"

        out_dir = os.path.dirname(os.path.abspath(output_json))
        os.makedirs(out_dir, exist_ok=True)

        output_data = {
            "script": "calcular_fwhm.py",
            "timestamp": datetime.now().isoformat(),
            "model": "f(x) = a + (b-a)*exp(-((x-c)^2)/(2*d^2))",
            "fwhm_formula": "FWHM = 2*sqrt(2*ln(2))*d",
            "fwhm_factor": round(FWHM_FACTOR, 8),
            "input_volume": str(vol_path),
            "input_puntos_json": str(json_path),
            "input_dicom_dir": str(pet_dicom_dir) if pet_dicom_dir else None,
            "voxel_spacing_mm": [round(s, 6) for s in spacing_mm],
            "series_description": series_meta.get("series_description"),
            "reconstruction_method": series_meta.get("reconstruction_method"),
            "convolution_kernel": series_meta.get("convolution_kernel"),
            "protocol_name": series_meta.get("protocol_name"),
            "fov_fraction": series_meta.get("fov_fraction"),
            "fov_source": series_meta.get("fov_source"),
            "num_puntos": len(resultados),
            "resultados": resultados,
        }

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info("Resultados FWHM exportados a: %s", output_json)
    return resultados


def main():
    parser = argparse.ArgumentParser(
        description="Cálculo del FWHM de puntos de actividad en volúmenes PET "
                    "mediante ajuste Gaussiano: f(x)=a+(b-a)*exp(-((x-c)^2)/(2d^2))",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  python3 calcular_fwhm.py \\
      --pet-volume ./volumes/fwhm_3puntos_pet.nii.gz \\
      --puntos-json ./volumes/fwhm_3puntos_puntos_actividad.json \\
      --output ./resultados/fwhm_resultados.json

  python3 calcular_fwhm.py \\
      --pet-volume ./volumes/fwhm_3puntos_pet.nii.gz \\
      --puntos-json ./volumes/fwhm_3puntos_puntos_actividad.json \\
      --verbose
        """
    )
    parser.add_argument(
        "--pet-volume", required=True,
        help="Ruta al volumen NIfTI PET generado por gen_volume_PET.py "
             "(.nii o .nii.gz)"
    )
    parser.add_argument(
        "--puntos-json", required=True,
        help="Ruta al archivo JSON de puntos de actividad generado por "
             "loc_puntos_actividad.py"
    )
    parser.add_argument(
        "--pet-dicom-dir", default=None,
        help="(Opcional) Directorio DICOM original de la serie PET. Si se "
             "proporciona, se extraen SeriesDescription, ReconstructionMethod, "
             "ConvolutionKernel y fracción del FOV (1/2, 1/8, ...) "
             "directamente de los archivos DICOM."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el JSON de resultados FWHM. "
             "Por defecto se crea junto al volumen NIfTI."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Mostrar información detallada del ajuste"
    )

    args = parser.parse_args()

    resultados = calcular_fwhm(
        pet_volume=args.pet_volume,
        puntos_json=args.puntos_json,
        output_json=args.output,
        pet_dicom_dir=args.pet_dicom_dir,
        verbose=args.verbose,
    )

    print("\n" + "=" * 70)
    print(f"RESULTADOS FWHM - {len(resultados)} PUNTO(S)")
    print(f"Modelo: f(x) = a + (b-a) * exp(-(x-c)^2 / (2 * d^2))")
    print(f"FWHM   = 2 * sqrt(2 * ln2) * d  aprox  {FWHM_FACTOR:.4f} * d")
    print("=" * 70)

    if args.pet_dicom_dir:
        sm = _extract_series_metadata(args.pet_dicom_dir)
        if sm:
            print("\n  METADATOS DE LA SERIE PET:")
            print(f"    SeriesDescription   : {sm.get('series_description') or 'N/A'}")
            print(f"    ReconstructionMethod: {sm.get('reconstruction_method') or 'N/A'}")
            print(f"    ConvolutionKernel   : {sm.get('convolution_kernel') or 'N/A'}")
            print(f"    ProtocolName        : {sm.get('protocol_name') or 'N/A'}")
            fov = sm.get('fov_fraction')
            fov_src = sm.get('fov_source')
            print(f"    FOV (fracción)      : {fov or 'N/A'}  [fuente: {fov_src or 'N/A'}]")

    for r in resultados:
        px, py, pz = r.get("patient_mm") or [0, 0, 0]
        print(f"\n  Punto {r['punto_id']:>2}:  ({px:+9.2f}, {py:+9.2f}, {pz:+9.2f}) mm")
        for eje in ("x", "y", "z"):
            fval = r.get(f"fwhm_{eje}_mm")
            fit = r.get(f"ajuste_{eje}", {})
            d_val = fit.get("d")
            r2 = fit.get("r_squared")
            if fval is not None:
                print(f"    FWHM {eje.upper()} = {fval:8.4f} mm  "
                      f"(sigma={d_val:.4f} mm, R^2={r2:.4f})")
            else:
                print(f"    FWHM {eje.upper()} = no convergió: {fit.get('error_msg', 'N/A')}")
        ftrans = r.get("fwhm_transaxial_mm")
        if ftrans is not None:
            print(f"    FWHM Transaxial (X+Y)/2 = {ftrans:.4f} mm")
    print("\n" + "=" * 70)
    if args.output:
        print(f"JSON exportado a: {args.output}")


if __name__ == "__main__":
    main()
