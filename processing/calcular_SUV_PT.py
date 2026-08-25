#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calcular_SUV_PT.py - Cálculo eficiente de Standardized Uptake Value (SUV) para cortes PET

Calcula la matriz 2D de SUV (Standardized Uptake Value normalizado por peso corporal,
SUVbw) para un corte individual de PET (PT) a partir de la ruta de la imagen
(obtenida del archivo .json de indexación) y los parámetros de calibración
física (obtenidos de la base de datos .db de indexación).

Fórmula matemática:
    1. Actividad de la imagen (concentración radiactiva en Bq/mL):
       C_img = PixelArray * RescaleSlope + RescaleIntercept

    2. Tiempo transcurrido desde la inyección:
       delta_t = t_scan - t_injection  (en segundos)

    3. Factor de decaimiento radiactivo:
       D = exp(-ln(2) * delta_t / RadionuclideHalfLife)  [si decay_correction == 'START']
       D = 1.0                                           [si decay_correction == 'ADMIN']

    4. Dosis inyectada corregida por decaimiento:
       A_decayed = RadionuclideTotalDose * D  (en Bq)

    5. Concentración esperada en el cuerpo:
       C_body = A_decayed / (PatientWeight_kg * 1000 g/kg)  (en Bq/g aprox. Bq/mL)

    6. Matriz SUV (SUVbw):
       SUV = C_img / C_body = C_img * (PatientWeight_kg * 1000) / A_decayed

Uso como módulo:
    from calcular_SUV_PT import calcular_suv_pt

    # Con parámetros desacoplados del .db (máxima eficiencia)
    suv_matrix = calcular_suv_pt(
        image_path="PET_CT/PACIENTES/DICOM_ANA_2022/ST000000/SE000001/PT000000",
        rescale_slope=0.0535174,
        rescale_intercept=0.0,
        patient_weight=56.0,
        radionuclide_total_dose=225700000.0,
        radionuclide_half_life=6586.2,
        radiopharmaceutical_start_time="093500.000000",
        series_time="110554.000000",
        dicom_root="./DICOM"
    )

    # O dejando que extraiga los metadatos directamente del DICOM si no se pasan
    suv_matrix = calcular_suv_pt("ruta/a/PT000000")

Uso desde CLI:
    python3 calcular_SUV_PT.py \
        --image-path "PET_CT/PACIENTES/DICOM_ANA_2022/ST000000/SE000001/PT000000" \
        --rescale-slope 0.0535174 \
        --rescale-intercept 0.0 \
        --patient-weight 56.0 \
        --dose 225700000.0 \
        --half-life 6586.2 \
        --injection-time "093500" \
        --scan-time "110554" \
        --dicom-root ./DICOM \
        --stats
"""

import argparse
import datetime
import logging
import os
import sys

import numpy as np
import pydicom

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("calcular_SUV_PT")

# Vida media estándar de fluorodesoxiglucosa (18F) en segundos (109.77 min)
DEFAULT_HALF_LIFE_F18 = 6586.2


def parse_dicom_time_seconds(time_val) -> float:
    """
    Convierte una representación de tiempo DICOM (cadena HHMMSS.frac,
    entero, flotante, time o datetime) a segundos desde la medianoche.

    Parameters
    ----------
    time_val : str, float, int, datetime.time, or datetime.datetime

    Returns
    -------
    float
        Segundos transcurridos desde las 00:00:00 del día.
    """
    if time_val is None or time_val == "":
        return 0.0

    if isinstance(time_val, (datetime.datetime, datetime.time)):
        return time_val.hour * 3600.0 + time_val.minute * 60.0 + time_val.second + time_val.microsecond / 1e6

    if isinstance(time_val, (int, float)):
        # Si ya es un valor en segundos acumulados
        if float(time_val) < 86400.0 and float(time_val) >= 0.0 and not str(int(time_val)).endswith("00"):
            return float(time_val)
        time_val = str(int(time_val)).zfill(6)

    t_str = str(time_val).strip()

    # Si contiene fecha y hora (formato YYYYMMDDHHMMSS o ISO)
    if "T" in t_str:
        t_str = t_str.split("T")[-1]
    elif len(t_str) >= 14 and "." not in t_str[:14] and t_str.isdigit():
        t_str = t_str[8:]

    parts = t_str.split(".")
    base = parts[0].zfill(6)
    usec = float("0." + parts[1]) if len(parts) > 1 else 0.0

    try:
        hours = int(base[0:2])
        minutes = int(base[2:4])
        seconds = int(base[4:6])
        return hours * 3600.0 + minutes * 60.0 + seconds + usec
    except (ValueError, IndexError):
        logger.warning("No se pudo parsear el tiempo DICOM '%s', usando 0.0s", time_val)
        return 0.0


def resolver_ruta_imagen(image_path: str, dicom_root: str = None) -> str:
    """
    Resuelve la ruta completa al archivo DICOM manejando rutas absolutas,
    relativas, sin extensión y con diversas extensiones (.dcm, .ima, .pt).
    """
    candidatos = [image_path]

    if dicom_root:
        candidatos.append(os.path.join(dicom_root, image_path))
        if image_path.startswith("DICOM/") or image_path.startswith("DICOM\\"):
            candidatos.append(os.path.join(dicom_root, image_path[6:]))

    ext_candidatos = []
    for cand in candidatos:
        ext_candidatos.append(cand)
        if not os.path.splitext(cand)[1]:
            ext_candidatos.append(cand + ".dcm")
            ext_candidatos.append(cand + ".DCM")
            ext_candidatos.append(cand + ".ima")
            ext_candidatos.append(cand + ".IMA")
            ext_candidatos.append(cand + ".pt")

    for ruta in ext_candidatos:
        if os.path.isfile(ruta):
            return os.path.abspath(ruta)

    raise FileNotFoundError(
        f"No se pudo encontrar el archivo DICOM de PET: '{image_path}'. "
        f"Rutas probadas: {ext_candidatos}"
    )


def leer_pixels_pet_raw(image_path: str, dicom_root: str = None):
    """
    Lee los píxeles crudos (sin calibrar) de un archivo DICOM de PET y
    extrae los metadatos de calibración física del encabezado como respaldo.

    Returns
    -------
    tuple (numpy.ndarray, dict)
        - raw_pixels: Matriz 2D de tipo float64.
        - header_info: Diccionario con metadatos de calibración.
    """
    ruta_real = resolver_ruta_imagen(image_path, dicom_root)
    try:
        ds = pydicom.dcmread(ruta_real, force=True)
    except Exception as e:
        raise RuntimeError(f"Error al leer archivo DICOM '{ruta_real}': {e}")

    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"El archivo DICOM '{ruta_real}' no contiene 'pixel_array'.")

    raw_pixels = ds.pixel_array.astype(np.float64)

    def _safe_float(ds_obj, attr, default=None):
        val = getattr(ds_obj, attr, None)
        if val is None or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _safe_int(ds_obj, attr, default=None):
        val = getattr(ds_obj, attr, None)
        if val is None or val == "":
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    rad_dose = None
    half_life = DEFAULT_HALF_LIFE_F18
    inj_time = None
    rad_name = None

    if hasattr(ds, "RadiopharmaceuticalInformationSequence") and len(ds.RadiopharmaceuticalInformationSequence) > 0:
        r_seq = ds.RadiopharmaceuticalInformationSequence[0]
        rad_dose = _safe_float(r_seq, "RadionuclideTotalDose", None)
        half_life = _safe_float(r_seq, "RadionuclideHalfLife", DEFAULT_HALF_LIFE_F18)
        inj_time = getattr(r_seq, "RadiopharmaceuticalStartTime", None)
        rad_name = getattr(r_seq, "Radiopharmaceutical", None)

    scan_time = getattr(ds, "SeriesTime", getattr(ds, "AcquisitionTime", getattr(ds, "StudyTime", None)))

    header_info = {
        "file_path": ruta_real,
        "rows": getattr(ds, "Rows", raw_pixels.shape[0]),
        "columns": getattr(ds, "Columns", raw_pixels.shape[1]),
        "rescale_slope": _safe_float(ds, "RescaleSlope", 1.0),
        "rescale_intercept": _safe_float(ds, "RescaleIntercept", 0.0),
        "units": str(getattr(ds, "Units", "BQML")).upper(),
        "decay_correction": str(getattr(ds, "DecayCorrection", "START")).upper(),
        "patient_weight": _safe_float(ds, "PatientWeight", None),
        "radionuclide_total_dose": rad_dose,
        "radionuclide_half_life": half_life,
        "radiopharmaceutical_start_time": inj_time,
        "series_time": scan_time,
        "radiopharmaceutical": rad_name,
        "pixel_spacing": [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0]) if v is not None and v != ""],
        "slice_thickness": _safe_float(ds, "SliceThickness", None),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "instance_number": _safe_int(ds, "InstanceNumber", None),
    }

    return raw_pixels, header_info


def calcular_suv_pt(image_path: str,
                    rescale_slope: float = None,
                    rescale_intercept: float = None,
                    patient_weight: float = None,
                    radionuclide_total_dose: float = None,
                    radionuclide_half_life: float = None,
                    radiopharmaceutical_start_time=None,
                    series_time=None,
                    units: str = "BQML",
                    decay_correction: str = "START",
                    dicom_root: str = None,
                    return_metadata: bool = False):
    """
    Calcula la matriz 2D de Standardized Uptake Value (SUVbw) para un corte de PET.

    Parameters
    ----------
    image_path : str
        Ruta al archivo del corte PET (obtenida del JSON de indexación).
    rescale_slope : float, optional
        Pendiente de calibración (obtenida del .db de indexación).
        Si es None, se lee del encabezado DICOM.
    rescale_intercept : float, optional
        Intercepción de calibración (obtenida del .db de indexación).
        Si es None, se lee del encabezado DICOM (típicamente 0.0).
    patient_weight : float, optional
        Peso del paciente en kilogramos (obtenido del .db de indexación).
    radionuclide_total_dose : float, optional
        Dosis total administrada en Becquerels (Bq) (obtenida del .db de indexación).
    radionuclide_half_life : float, optional
        Vida media del radionúclido en segundos (por defecto: 6586.2 s para 18F).
    radiopharmaceutical_start_time : str or datetime, optional
        Hora de inyección del radiofármaco (ej: "093500.000000").
    series_time : str or datetime, optional
        Hora de adquisición de la serie PET (ej: "110554.000000").
    units : str, optional
        Unidades de actividad de la imagen ("BQML", "GML", "CNTS", por defecto: "BQML").
    decay_correction : str, optional
        Tipo de corrección de decaimiento ("START", "ADMIN", "NONE", por defecto: "START").
    dicom_root : str, optional
        Ruta al directorio raíz DICOM si image_path es relativa.
    return_metadata : bool, optional
        Si True, retorna también un diccionario con información y estadísticas.

    Returns
    -------
    numpy.ndarray o tuple(numpy.ndarray, dict)
        - suv_matrix: Arreglo 2D NumPy (float64) con los valores calibrados de SUV.
        - metadata (si return_metadata=True): Diccionario con detalles de la calibración y estadísticas.
    """
    raw_pixels, header_info = leer_pixels_pet_raw(image_path, dicom_root)

    # Determinar parámetros con respaldo del encabezado DICOM
    slope = float(rescale_slope) if rescale_slope is not None else header_info["rescale_slope"]
    intercept = float(rescale_intercept) if rescale_intercept is not None else header_info["rescale_intercept"]
    weight_kg = float(patient_weight) if patient_weight is not None else (header_info["patient_weight"] or 70.0)
    total_dose = float(radionuclide_total_dose) if radionuclide_total_dose is not None else header_info["radionuclide_total_dose"]
    half_life = float(radionuclide_half_life) if radionuclide_half_life is not None else (header_info["radionuclide_half_life"] or DEFAULT_HALF_LIFE_F18)

    t_inj_raw = radiopharmaceutical_start_time if radiopharmaceutical_start_time is not None else header_info["radiopharmaceutical_start_time"]
    t_scan_raw = series_time if series_time is not None else header_info["series_time"]

    unit_str = str(units or header_info["units"] or "BQML").upper()
    decay_str = str(decay_correction or header_info["decay_correction"] or "START").upper()

    # Concentración de actividad en la imagen (Bq/mL)
    activity_img = raw_pixels * slope + intercept

    # Si las unidades ya están en GML (gramos/mL), la imagen ya representa SUV
    if unit_str == "GML":
        suv_matrix = activity_img
        decay_factor = 1.0
        delta_time_sec = 0.0
        decayed_dose = total_dose or 1.0
        dose_per_g = 1.0
    else:
        # Cálculo de la diferencia temporal (inyección a adquisición)
        t_inj_sec = parse_dicom_time_seconds(t_inj_raw)
        t_scan_sec = parse_dicom_time_seconds(t_scan_raw)

        delta_time_sec = t_scan_sec - t_inj_sec
        # Si la adquisición cruzó la medianoche
        if delta_time_sec < 0:
            delta_time_sec += 86400.0

        # Factor de decaimiento radiactivo
        if decay_str == "ADMIN":
            decay_factor = 1.0
        else:
            if half_life > 0:
                decay_factor = float(np.exp(-np.log(2.0) * delta_time_sec / half_life))
            else:
                decay_factor = 1.0

        # Dosis decaída y concentración corporal esperada
        if total_dose is not None and total_dose > 0:
            decayed_dose = total_dose * decay_factor
            dose_per_g = decayed_dose / (weight_kg * 1000.0)

            if dose_per_g > 0:
                suv_matrix = activity_img / dose_per_g
            else:
                suv_matrix = activity_img
        else:
            logger.warning(
                "No se especificó la dosis total del radiofármaco. "
                "Se devuelve la concentración de actividad sin normalizar por dosis."
            )
            decayed_dose = None
            dose_per_g = None
            suv_matrix = activity_img

    # Evitar valores negativos espurios en SUV por ruido de fondo
    suv_matrix = np.maximum(0.0, suv_matrix)

    if return_metadata:
        positive_suv = suv_matrix[suv_matrix > 0]
        meta = {
            "file_path": header_info["file_path"],
            "shape": list(suv_matrix.shape),
            "rescale_slope_used": slope,
            "rescale_intercept_used": intercept,
            "patient_weight_kg": weight_kg,
            "radionuclide_total_dose_bq": total_dose,
            "decayed_dose_bq": decayed_dose,
            "radionuclide_half_life_sec": half_life,
            "radiopharmaceutical_start_time": t_inj_raw,
            "series_time": t_scan_raw,
            "delta_time_seconds": round(delta_time_sec, 2),
            "delta_time_minutes": round(delta_time_sec / 60.0, 2),
            "decay_factor": round(decay_factor, 6),
            "units": unit_str,
            "decay_correction": decay_str,
            "suv_min": float(np.min(suv_matrix)),
            "suv_max": float(np.max(suv_matrix)),
            "suv_mean": float(np.mean(suv_matrix)),
            "suv_99th": float(np.percentile(suv_matrix, 99.0)) if len(suv_matrix) > 0 else 0.0,
            "suv_mean_positive": float(np.mean(positive_suv)) if len(positive_suv) > 0 else 0.0,
            "pixel_spacing": header_info["pixel_spacing"],
            "slice_thickness": header_info["slice_thickness"],
            "sop_instance_uid": header_info["sop_instance_uid"],
            "instance_number": header_info["instance_number"],
        }
        return suv_matrix, meta

    return suv_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Calcula la matriz 2D de Standardized Uptake Value (SUV) para un corte de PET."
    )
    parser.add_argument(
        "-i", "--image-path", required=True,
        help="Ruta al archivo DICOM del corte PET (ej: PET_CT/paciente_00/osteo1/SE000003/PT000000)"
    )
    parser.add_argument(
        "-s", "--rescale-slope", type=float, default=None,
        help="RescaleSlope obtenido del .db (ej: 0.0535). Si se omite, se lee del DICOM."
    )
    parser.add_argument(
        "-t", "--rescale-intercept", type=float, default=None,
        help="RescaleIntercept obtenido del .db (ej: 0.0). Si se omite, se lee del DICOM."
    )
    parser.add_argument(
        "-w", "--patient-weight", type=float, default=None,
        help="Peso del paciente en kg obtenido del .db (ej: 56.0). Si se omite, se lee del DICOM."
    )
    parser.add_argument(
        "--dose", type=float, default=None,
        help="Dosis inyectada en Bq obtenida del .db (ej: 225700000.0). Si se omite, se lee del DICOM."
    )
    parser.add_argument(
        "--half-life", type=float, default=None,
        help="Vida media del radionúclido en segundos (por defecto: 6586.2 para F-18)."
    )
    parser.add_argument(
        "--injection-time", default=None,
        help="Hora de inyección obtenida del .db (ej: '093500.000000' o '093500')."
    )
    parser.add_argument(
        "--scan-time", default=None,
        help="Hora de escaneo obtenida del .db (ej: '110554.000000' o '110554')."
    )
    parser.add_argument(
        "--units", default="BQML",
        help="Unidades PET ('BQML', 'GML', 'CNTS', por defecto: 'BQML')."
    )
    parser.add_argument(
        "--decay-correction", default="START",
        help="Tipo de corrección de decaimiento ('START', 'ADMIN', 'NONE', por defecto: 'START')."
    )
    parser.add_argument(
        "-d", "--dicom-root", default=None,
        help="Directorio raíz DICOM para rutas relativas."
    )
    parser.add_argument(
        "-o", "--output-npy", default=None,
        help="Ruta para guardar la matriz de salida como archivo binario NumPy (.npy)."
    )
    parser.add_argument(
        "--output-csv", default=None,
        help="Ruta para guardar la matriz de salida como archivo de texto CSV."
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Imprime estadísticas descriptivas de los valores SUV del corte."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Modo detallado de depuración."
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        suv_matrix, meta = calcular_suv_pt(
            image_path=args.image_path,
            rescale_slope=args.rescale_slope,
            rescale_intercept=args.rescale_intercept,
            patient_weight=args.patient_weight,
            radionuclide_total_dose=args.dose,
            radionuclide_half_life=args.half_life,
            radiopharmaceutical_start_time=args.injection_time,
            series_time=args.scan_time,
            units=args.units,
            decay_correction=args.decay_correction,
            dicom_root=args.dicom_root,
            return_metadata=True
        )

        logger.info("Corte PET procesado exitosamente: %s", meta["file_path"])
        logger.info("Dimensiones de la matriz: %s", meta["shape"])
        logger.info("Diferencia de tiempo (inyección a escaneo): %.2f min", meta["delta_time_minutes"])
        logger.info("Factor de decaimiento: %.6f", meta["decay_factor"])

        if args.stats or args.verbose:
            print("\n" + "=" * 60)
            print("  ESTADÍSTICAS DEL CORTE PET (Standardized Uptake Value - SUVbw)")
            print("=" * 60)
            print(f"  Archivo               : {os.path.basename(meta['file_path'])}")
            print(f"  Dimensiones (fil, col): {meta['shape'][0]} x {meta['shape'][1]}")
            print(f"  Peso Paciente         : {meta['patient_weight_kg']:.1f} kg")
            if meta['radionuclide_total_dose_bq']:
                print(f"  Dosis Inyectada       : {meta['radionuclide_total_dose_bq'] / 1e6:.2f} MBq")
            if meta['decayed_dose_bq']:
                print(f"  Dosis Decaída (Scan)  : {meta['decayed_dose_bq'] / 1e6:.2f} MBq")
            print(f"  Tiempo Transcurrido   : {meta['delta_time_minutes']:.2f} min ({meta['delta_time_seconds']:.1f} s)")
            print(f"  SUV Mínimo            : {meta['suv_min']:.4f}")
            print(f"  SUV Máximo (SUVmax)   : {meta['suv_max']:.4f}")
            print(f"  SUV Promedio (SUVmean): {meta['suv_mean']:.4f}")
            print(f"  SUV Percentil 99      : {meta['suv_99th']:.4f}")
            print(f"  PixelSpacing (mm)     : {meta['pixel_spacing']}")
            print("=" * 60 + "\n")

        if args.output_npy:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)
            np.save(args.output_npy, suv_matrix)
            logger.info("Matriz SUV guardada en formato NumPy: %s", args.output_npy)

        if args.output_csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
            np.savetxt(args.output_csv, suv_matrix, delimiter=",", fmt="%.4f")
            logger.info("Matriz SUV guardada en formato CSV: %s", args.output_csv)

    except Exception as e:
        logger.error("Error al calcular SUV: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
