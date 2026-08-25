#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
larmornium.py - Script principal de Larmornium
=================================================

Punto de entrada unificado para la aplicacion Larmornium.
Centraliza las importaciones, la configuracion de rutas,
y expone los subcomandos:

  - gui           : Lanza la interfaz grafica de visualizacion
  - index         : Ejecuta la indexacion combinada (PET/CT + MRI) desde CLI
  - index-pet-ct  : Ejecuta unicamente la indexacion de estudios PET/CT
  - index-mri     : Ejecuta unicamente la indexacion de estudios MRI

Uso:
    python3 larmornium.py gui
    python3 larmornium.py index --dicom-dir ./DICOM
    python3 larmornium.py index --dicom-dir ./DICOM --output-dir ./output --verbose
    python3 larmornium.py index-pet-ct --dicom-dir ./DICOM
    python3 larmornium.py index-mri --dicom-dir ./DICOM
"""

import argparse
import logging
import os
import sys

# Configurar rutas de importacion para los modulos del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_INDEX_DIR = os.path.join(_PROJECT_ROOT, "index")
_GUI_DIR = os.path.join(_PROJECT_ROOT, "gui")

# Agregar el directorio de indexadores al path si no esta
if _INDEX_DIR not in sys.path:
    sys.path.insert(0, _INDEX_DIR)

# Agregar el directorio de la GUI al path si no esta
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)

# Agregar el directorio raiz del proyecto al path si no esta
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _default_output_dir(dicom_dir, output_dir):
    """Calcula el directorio de salida por defecto: <dicom-dir>/larmornium_files/indexed."""
    if output_dir is not None:
        return output_dir
    return os.path.join(dicom_dir, "larmornium_files", "indexed")


def cmd_gui(args):
    """Lanza la interfaz grafica de Larmornium."""
    from gui import launch_gui
    launch_gui()


def cmd_index(args):
    """Ejecuta la indexacion combinada (PET/CT + MRI) de estudios DICOM desde CLI."""
    import index_dicom_all

    output_dir = _default_output_dir(args.dicom_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_dicom_all.index_all(args.dicom_dir, output_dir, args.verbose)


def cmd_index_pet_ct(args):
    """Ejecuta unicamente la indexacion de estudios PET/CT desde CLI."""
    import index_pet_ct

    output_dir = _default_output_dir(args.dicom_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_pet_ct.index_pet_ct(args.dicom_dir, output_dir, args.verbose)


def cmd_index_mri(args):
    """Ejecuta unicamente la indexacion de estudios MRI desde CLI."""
    import index_mri

    output_dir = _default_output_dir(args.dicom_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_mri.index_mri(args.dicom_dir, output_dir, args.verbose)


def _add_index_arguments(subparser, folder_hint):
    """Agrega los argumentos comunes de indexacion (--dicom-dir, --output-dir, --verbose)."""
    subparser.add_argument(
        "--dicom-dir", required=True,
        help="Ruta al directorio DICOM raiz (que contiene %s)" % folder_hint
    )
    subparser.add_argument(
        "--output-dir", default=None,
        help=("Directorio de salida para .db y .json "
              "(default: <dicom-dir>/larmornium_files/indexed/)")
    )
    subparser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir progreso detallado"
    )


def main():
    """Punto de entrada principal con subcomandos CLI."""
    parser = argparse.ArgumentParser(
        prog="larmornium",
        description="Larmornium — Visualizacion y analisis de estudios PET/CT y MRI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcomandos:
  gui           Lanza la interfaz grafica
  index         Ejecuta la indexacion combinada (PET/CT + MRI)
  index-pet-ct  Ejecuta unicamente la indexacion de estudios PET/CT
  index-mri     Ejecuta unicamente la indexacion de estudios MRI

Ejemplos:
  python3 larmornium.py gui
  python3 larmornium.py index --dicom-dir ./DICOM
  python3 larmornium.py index --dicom-dir ./DICOM --output-dir ./output --verbose
  python3 larmornium.py index-pet-ct --dicom-dir ./DICOM
  python3 larmornium.py index-mri --dicom-dir ./DICOM
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcomando a ejecutar")

    # Subcomando: gui
    gui_parser = subparsers.add_parser(
        "gui",
        help="Lanzar la interfaz grafica de visualizacion"
    )
    gui_parser.set_defaults(func=cmd_gui)

    # Subcomando: index (combinado PET/CT + MRI)
    index_parser = subparsers.add_parser(
        "index",
        help="Indexar estudios DICOM (PET/CT + MRI) desde linea de comandos"
    )
    _add_index_arguments(index_parser, "PET_CT/ y/o MRI/")
    index_parser.set_defaults(func=cmd_index)

    # Subcomando: index-pet-ct
    index_pet_ct_parser = subparsers.add_parser(
        "index-pet-ct",
        help="Indexar unicamente estudios PET/CT desde linea de comandos"
    )
    _add_index_arguments(index_pet_ct_parser, "PET_CT/")
    index_pet_ct_parser.set_defaults(func=cmd_index_pet_ct)

    # Subcomando: index-mri
    index_mri_parser = subparsers.add_parser(
        "index-mri",
        help="Indexar unicamente estudios MRI desde linea de comandos"
    )
    _add_index_arguments(index_mri_parser, "MRI/")
    index_mri_parser.set_defaults(func=cmd_index_mri)

    args = parser.parse_args()

    # Si no se especifica subcomando, lanzar la GUI por defecto
    if not hasattr(args, "func"):
        cmd_gui(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()
