#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import logging
import math
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("label_locs")

DEFAULT_REFERENCE_COORDS = [(0.0, 1.0), (10.0, 0.0), (20.0, 0.0)]


def _format_coord_label(coord):
    x, y = coord
    x_str = f"{x:g}"
    y_str = f"{y:g}"
    return f"({x_str}, {y_str})"


def _radial_distance(point_xy, center_xy=(0.0, 0.0)):
    dx = float(point_xy[0]) - float(center_xy[0])
    dy = float(point_xy[1]) - float(center_xy[1])
    return math.sqrt(dx * dx + dy * dy)


def etiquetar_puntos_actividad(puntos, reference_coords=None, output_json=None):
    if reference_coords is None:
        ref_coords = list(DEFAULT_REFERENCE_COORDS)
    else:
        ref_coords = [(float(c[0]), float(c[1])) for c in reference_coords]

    input_source = "<en memoria>"
    if isinstance(puntos, str):
        if not os.path.isfile(puntos):
            logger.error("Archivo de puntos no encontrado: %s", puntos)
            raise FileNotFoundError(f"Archivo de puntos no encontrado: {puntos}")
        input_source = puntos
        with open(puntos, "r", encoding="utf-8") as f:
            data = json.load(f)
        puntos_list = data.get("puntos", data) if isinstance(data, dict) else data
    elif isinstance(puntos, list):
        puntos_list = [dict(p) for p in puntos]
    else:
        raise TypeError("puntos debe ser una lista de diccionarios o una ruta a archivo JSON")

    n_puntos = len(puntos_list)
    n_refs = len(ref_coords)
    logger.info("Etiquetando %d punto(s) con %d coordenada(s) de referencia.", n_puntos, n_refs)

    if n_puntos == 0:
        return []

    # Extraer coordenadas (x, y) de cada punto detectado (a partir de patient_mm o voxel_ijk)
    puntos_con_pos = []
    for idx, p in enumerate(puntos_list):
        p_copy = dict(p)
        patient_mm = p_copy.get("patient_mm")
        if patient_mm is not None and len(patient_mm) >= 2:
            x_val, y_val = float(patient_mm[0]), float(patient_mm[1])
        else:
            vox = p_copy.get("voxel_ijk", [0, 0, 0])
            x_val, y_val = float(vox[0]), float(vox[1])
        puntos_con_pos.append((idx, p_copy, (x_val, y_val)))

    # Ordenar coordenadas de referencia por su distancia radial al origen (0, 0)
    ref_coords_sorted = sorted(ref_coords, key=lambda c: _radial_distance(c, (0.0, 0.0)))

    # Para los puntos detectados, calcular distancia al centroide o isocentro
    # Si las coordenadas de paciente están centradas en el isocentro, usamos (0, 0)
    # También calculamos distancias al origen
    puntos_ordenados = sorted(
        puntos_con_pos,
        key=lambda item: _radial_distance(item[2], (0.0, 0.0))
    )

    puntos_etiquetados = [None] * n_puntos

    for rank, (orig_idx, p_dict, (px, py)) in enumerate(puntos_ordenados):
        if rank < n_refs:
            ref_c = ref_coords_sorted[rank]
            lbl = _format_coord_label(ref_c)
            p_dict["label"] = lbl
            p_dict["coord_referencia"] = [round(ref_c[0], 4), round(ref_c[1], 4)]
        else:
            p_dict["label"] = f"Punto {orig_idx + 1}"
            p_dict["coord_referencia"] = None

        puntos_etiquetados[orig_idx] = p_dict

    for p in puntos_etiquetados:
        pid = p.get("punto_id", "?")
        lbl = p.get("label", "")
        pt_mm = p.get("patient_mm", [])
        logger.info("  Punto %s: Etiqueta = %s | Coordenadas Paciente = %s", pid, lbl, pt_mm)

    if output_json:
        out_dir = os.path.dirname(os.path.abspath(output_json))
        os.makedirs(out_dir, exist_ok=True)
        out_data = {
            "script": "label_locs.py",
            "timestamp": datetime.now().isoformat(),
            "input_source": input_source,
            "coordenadas_referencia": [[float(c[0]), float(c[1])] for c in ref_coords],
            "num_puntos": len(puntos_etiquetados),
            "puntos": puntos_etiquetados,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        logger.info("Puntos etiquetados guardados en: %s", output_json)

    return puntos_etiquetados


def main():
    parser = argparse.ArgumentParser(
        description="Etiqueta puntos de máxima actividad según coordenadas de referencia",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Ejemplos:
  python3 label_locs.py \\
      --puntos-json ./resultados/puntos_actividad.json \\
      --output ./resultados/puntos_etiquetados.json

  python3 label_locs.py \\
      --puntos-json ./resultados/puntos_actividad.json \\
      --coords "0,1" "10,0" "20,0" \\
      --output ./resultados/puntos_etiquetados.json
        """
    )
    parser.add_argument(
        "--puntos-json", required=True,
        help="Ruta al archivo JSON generado por loc_puntos_actividad.py"
    )
    parser.add_argument(
        "--coords", nargs="+", default=None,
        help="Coordenadas de referencia en formato 'X,Y' (ejemplo: '0,1' '10,0' '20,0'). "
             "Por defecto: (0, 1), (10, 0), (20, 0)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el archivo JSON de puntos etiquetados"
    )

    args = parser.parse_args()

    ref_coords = None
    if args.coords:
        ref_coords = []
        for pair_str in args.coords:
            parts = pair_str.replace("(", "").replace(")", "").split(",")
            if len(parts) == 2:
                ref_coords.append((float(parts[0].strip()), float(parts[1].strip())))
            else:
                parser.error(f"Formato de coordenada inválido: {pair_str}. Use 'X,Y'")

    puntos = etiquetar_puntos_actividad(
        puntos=args.puntos_json,
        reference_coords=ref_coords,
        output_json=args.output,
    )

    print("\n" + "=" * 70)
    print(f"PUNTOS ETIQUETADOS ({len(puntos)})")
    print("=" * 70)
    for p in puntos:
        pid = p.get("punto_id", "?")
        lbl = p.get("label", "")
        pt_mm = p.get("patient_mm", [0, 0, 0])
        val = p.get("valor_maximo", 0.0)
        print(f"  Punto {pid:>2}: Etiqueta = {lbl:<10} | Coords = ({pt_mm[0]:+7.2f}, {pt_mm[1]:+7.2f}, {pt_mm[2]:+7.2f}) mm | Máx = {val:,.2f}")
    print("=" * 70)
    if args.output:
        print(f"JSON exportado a: {args.output}")


if __name__ == "__main__":
    main()
