#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys

os.environ["QT_API"] = "PySide6"

# Configurar rutas de importación para los módulos del proyecto
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_INDEX_DIR = os.path.join(_PROJECT_ROOT, "index")
_GUI_DIR = os.path.join(_PROJECT_ROOT, "gui")
_PROCESSING_DIR = os.path.join(_PROJECT_ROOT, "processing")

if _INDEX_DIR not in sys.path:
    sys.path.insert(0, _INDEX_DIR)
if _GUI_DIR not in sys.path:
    sys.path.insert(0, _GUI_DIR)
if _PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _PROCESSING_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import hashlib


def _get_dir_id(dicom_dir):
    abs_path = os.path.abspath(dicom_dir)
    return hashlib.md5(abs_path.encode("utf-8")).hexdigest()


def _default_output_dir(dicom_dir, output_dir):
    if output_dir is not None:
        return output_dir
    dir_id = _get_dir_id(dicom_dir)
    return os.path.join(_PROJECT_ROOT, "larmornium_files", dir_id, "indexed")


def cmd_gui(args):
    from gui import launch_gui
    launch_gui()


def cmd_index(args):
    import index_dicom_all

    output_dir = _default_output_dir(args.dicom_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_dicom_all.index_all(args.dicom_dir, output_dir, args.verbose)


def cmd_index_pet_ct(args):
    import index_pet_ct

    output_dir = _default_output_dir(args.dicom_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_pet_ct.index_pet_ct(args.dicom_dir, output_dir, args.verbose)


def cmd_index_mri(args):
    import index_mri

    output_dir = _default_output_dir(args.dicom_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    index_mri.index_mri(args.dicom_dir, output_dir, args.verbose)


def cmd_gen_ct(args):
    import gen_volume_CT

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    nifti_img, metadata = gen_volume_CT.generate_ct_volume(
        ct_directory=args.ct_dir,
        dicom_root=args.dicom_root,
        output_path=args.output,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("VOLUMEN CT GENERADO")
    print("=" * 60)
    print(f"  Directorio CT      : {metadata['ct_directory']}")
    print(f"  Dimensiones        : {metadata['dimensions']}")
    print(f"  Espaciado (mm)     : {metadata['voxel_spacing_mm']}")
    print(f"  Rango HU           : {metadata['hu_range']}")
    print(f"  Rango Z (mm)       : {metadata['z_range_mm']}")
    print(f"  Serie              : {metadata['series_description']}")
    print(f"  Paciente           : {metadata['patient_name']}")
    print(f"  Archivo NIfTI      : {metadata['output_path']}")
    print("=" * 60)


def cmd_gen_pet(args):
    import gen_volume_PET

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    nifti_img, metadata = gen_volume_PET.generate_pet_volume(
        pet_directory=args.pet_dir,
        dicom_root=args.dicom_root,
        output_path=args.output,
        convert_suv=args.suv,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("VOLUMEN PET GENERADO")
    print("=" * 60)
    print(f"  Directorio PET     : {metadata['pet_directory']}")
    print(f"  Dimensiones        : {metadata['dimensions']}")
    print(f"  Espaciado (mm)     : {metadata['voxel_spacing_mm']}")
    print(f"  Unidades           : {metadata['units']}")
    print(f"  Rango valores      : [{metadata['value_range'][0]:.2f}, {metadata['value_range'][1]:.2f}] {metadata['units']}")
    print(f"  Rango Z (mm)       : {metadata['z_range_mm']}")
    print(f"  Serie              : {metadata['series_description']}")
    print(f"  Paciente           : {metadata['patient_name']}")
    print(f"  Archivo NIfTI      : {metadata['output_path']}")
    if metadata.get("suv_factor"):
        print(f"  Factor SUVbw       : {metadata['suv_factor']:.8f}")
    print("=" * 60)


def cmd_fusion(args):
    import fusion_pet_ct

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_nii = args.output or "fusion_pet_ct.nii.gz"
    output_json = os.path.splitext(output_nii.replace(".nii.gz", ".nii"))[0] + ".json"
    if output_nii.endswith(".nii.gz"):
        output_json = output_nii.replace(".nii.gz", ".json")

    fusion_pet_ct.fuse_and_save_pair(
        ct_dir=args.ct_dir,
        pet_dir=args.pet_dir,
        dicom_root=args.dicom_root,
        output_nii_path=output_nii,
        output_json_path=output_json,
    )

    print("\n" + "=" * 60)
    print("VOLUMEN FUSIONADO PET/CT GENERADO")
    print("=" * 60)
    print(f"  Directorio CT      : {args.ct_dir}")
    print(f"  Directorio PET     : {args.pet_dir}")
    print(f"  Archivo NIfTI      : {os.path.abspath(output_nii)}")
    print(f"  Archivo JSON       : {os.path.abspath(output_json)}")
    print("=" * 60)


def cmd_gen_mri(args):
    import gen_volume_MRI

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    nifti_img, metadata = gen_volume_MRI.generate_mri_volume(
        mri_directory=args.mri_dir,
        dicom_root=args.dicom_root,
        output_path=args.output,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("VOLUMEN MRI GENERADO")
    print("=" * 60)
    print(f"  Estudio / Serie    : {metadata.get('series_description') or metadata.get('patient_name')}")
    print(f"  Tipo fuente        : {metadata.get('source_type')}")
    print(f"  Dimensiones        : {metadata.get('dimensions')}")
    print(f"  Espaciado (mm)     : {metadata.get('voxel_spacing_mm')}")
    print(f"  Rango Intensidad   : [{metadata.get('intensity_min', 0):.1f}, {metadata.get('intensity_max', 0):.1f}]")
    print(f"  Paciente           : {metadata.get('patient_name')}")
    print(f"  Archivo NIfTI      : {metadata.get('output_nifti') or metadata.get('output_path')}")
    print("=" * 60)


def cmd_segment_ct(args):
    import segmentation_anato_ct_TotalSegmentator as seg_ct

    if args.list_organs:
        print("Órganos y estructuras CT disponibles para segmentación:")
        for key, info in seg_ct.ORGAN_REGISTRY.items():
            labels = ", ".join(info["roi_subset"])
            task_str = f" [task: {info.get('task', 'total')}]" if info.get("task", "total") != "total" else ""
            print(f'  {key:20s} - {info["display_name"]:20s} ({info["description"]}) [labels: {labels}]{task_str}')
        return

    if not args.input or not args.organ:
        print("Error: Se requieren los argumentos --input/-i y --organ/-g para realizar la segmentación CT.")
        sys.exit(1)

    _, metadata = seg_ct.segment_organ(
        input_volume=args.input,
        organ=args.organ,
        cuda=args.cuda,
        fast=args.fast,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        quiet=args.quiet,
    )

    stats = metadata.get("segmentation_stats", {})
    print("\nSEGMENTACIÓN CT COMPLETADA:")
    print(f"  Estructura         : {metadata.get('organ_display_name', args.organ)}")
    print(f"  Modelo             : {metadata.get('model', 'TotalSegmentator')} v{metadata.get('model_version', '1.0')}")
    print(f"  Dispositivo        : {metadata.get('device', 'cpu')}")
    print(f"  Tiempo             : {metadata.get('processing_time_seconds', metadata.get('elapsed_seconds', 0))} s")
    print(f"  Vóxeles segmentados: {stats.get('num_voxels', metadata.get('total_voxels', 0))}")
    print(f"  Volumen (cm3)      : {stats.get('volume_cm3', metadata.get('volume_cm3', 0))}")
    if stats.get("bounding_box_mm"):
        print(f"  Bounding box (mm)  : min={stats['bounding_box_mm']['min']}")
        print(f"                       max={stats['bounding_box_mm']['max']}")
    if stats.get("centroid_mm"):
        print(f"  Centroide (mm)     : {stats['centroid_mm']}")
    print(f"  NIfTI              : {metadata.get('output_nifti')}")
    print(f"  JSON               : {metadata.get('output_json')}")


def cmd_segment_mri(args):
    import segmentation_anato_mri_TotalSegmentator as seg_mri

    if args.list_organs:
        print("Estructuras MRI disponibles para segmentación:")
        for key, info in seg_mri.ORGAN_REGISTRY.items():
            labels = ", ".join(info["roi_subset"])
            print(f'  {key:20s} - {info["display_name"]:25s} ({info["description"]}) [labels: {labels}]')
        return

    if not args.input or not args.organ:
        print("Error: Se requieren los argumentos --input/-i y --organ/-g para realizar la segmentación MRI.")
        sys.exit(1)

    _, metadata = seg_mri.segment_organ_mri(
        input_volume=args.input,
        organ=args.organ,
        cuda=args.cuda,
        fast=args.fast,
        output_dir=args.output_dir,
        output_basename=args.output_basename,
        quiet=args.quiet,
    )

    stats = metadata.get("segmentation_stats", {})
    print("\nSEGMENTACIÓN MRI COMPLETADA:")
    print(f"  Estructura         : {metadata.get('organ_display_name', args.organ)}")
    print(f"  Modelo             : {metadata.get('model', 'TotalSegmentator')} v{metadata.get('model_version', '1.0')} ({metadata.get('model_task', 'total_mr')})")
    print(f"  Dispositivo        : {metadata.get('device', 'cpu')}")
    print(f"  Tiempo             : {metadata.get('processing_time_seconds', 0)} s")
    print(f"  Vóxeles segmentados: {stats.get('num_voxels', 0)}")
    print(f"  Volumen (cm3)      : {stats.get('volume_cm3', 0)}")
    if stats.get("physical_dimensions_mm"):
        p_dim = stats["physical_dimensions_mm"]
        print(f"  Dimensiones (mm)   : DX={p_dim['dx_mm']} mm, DY={p_dim['dy_mm']} mm, DZ={p_dim['dz_mm']} mm")
        print(f"  Diámetro equiv.    : {stats.get('equivalent_diameter_mm')} mm (Esfericidad: {stats.get('sphericity_aspect_ratio')})")
    if stats.get("signal_mean") is not None:
        print(f"  Señal media        : {stats['signal_mean']} +/- {stats.get('signal_std', 0)} (SNR: {stats.get('signal_snr')})")
    if stats.get("bounding_box_mm"):
        print(f"  Bounding box (mm)  : min={stats['bounding_box_mm']['min']}")
        print(f"                       max={stats['bounding_box_mm']['max']}")
    if stats.get("centroid_mm"):
        print(f"  Centroide (mm)     : {stats['centroid_mm']}")
    print(f"  NIfTI              : {metadata.get('output_nifti')}")
    print(f"  JSON               : {metadata.get('output_json')}")


def _add_index_arguments(subparser, folder_hint):
    subparser.add_argument(
        "--dicom-dir", required=True,
        help="Ruta al directorio DICOM raiz (que contiene %s)" % folder_hint
    )
    subparser.add_argument(
        "--output-dir", default=None,
        help=("Directorio de salida para .db y .json "
              "(default: ./larmornium_files/<id_directorio>/indexed/)")
    )
    subparser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir progreso detallado"
    )


def main():
    parser = argparse.ArgumentParser(
        prog="larmornium",
        description="Larmornium — Visualizacion y analisis de estudios PET/CT y MRI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcomandos:
  gui           Lanza la interfaz gráfica
  index         Ejecuta la indexación combinada (PET/CT + MRI)
  index-pet-ct  Ejecuta únicamente la indexación de estudios PET/CT
  index-mri     Ejecuta únicamente la indexación de estudios MRI
  gen-ct        Genera un volumen 3D NIfTI a partir de una serie CT DICOM
  gen-pet       Genera un volumen 3D NIfTI a partir de una serie PET DICOM
  gen-mri       Genera un volumen 3D NIfTI a partir de una serie MRI DICOM o Analyze
  fusion        Genera un volumen fusionado PET/CT
  segment-ct    Segmenta órganos o fantomas en volúmenes CT
  segment-mri   Segmenta órganos o fantomas en volúmenes MRI

Ejemplos:
  python3 larmornium.py gui
  python3 larmornium.py index --dicom-dir ./DICOM
  python3 larmornium.py index --dicom-dir ./DICOM --output-dir ./output --verbose
  python3 larmornium.py index-pet-ct --dicom-dir ./DICOM
  python3 larmornium.py index-mri --dicom-dir ./DICOM
  python3 larmornium.py gen-ct --ct-dir PET_CT/paciente/osteo1/SE000001 --dicom-root ./DICOM
  python3 larmornium.py gen-pet --pet-dir PET_CT/paciente/osteo1/SE000003 --dicom-root ./DICOM --suv
  python3 larmornium.py gen-mri --mri-dir "MRI/Phantom_Meta_GE/COR_T1" --dicom-root ./DICOM
  python3 larmornium.py fusion --ct-dir SE000001 --pet-dir SE000003 --dicom-root ./DICOM
  python3 larmornium.py segment-ct --input ./volumes/ct_vol.nii.gz --organ corazon --cuda
  python3 larmornium.py segment-ct --list-organs
  python3 larmornium.py segment-mri --input ./volumes/mri_vol.nii.gz --organ cerebro --cuda
  python3 larmornium.py segment-mri --list-organs
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcomando a ejecutar")

    gui_parser = subparsers.add_parser(
        "gui",
        help="Lanzar la interfaz grafica de visualizacion"
    )
    gui_parser.set_defaults(func=cmd_gui)

    index_parser = subparsers.add_parser(
        "index",
        help="Indexar estudios DICOM (PET/CT + MRI) desde linea de comandos"
    )
    _add_index_arguments(index_parser, "PET_CT/ y/o MRI/")
    index_parser.set_defaults(func=cmd_index)

    index_pet_ct_parser = subparsers.add_parser(
        "index-pet-ct",
        help="Indexar unicamente estudios PET/CT desde linea de comandos"
    )
    _add_index_arguments(index_pet_ct_parser, "PET_CT/")
    index_pet_ct_parser.set_defaults(func=cmd_index_pet_ct)

    index_mri_parser = subparsers.add_parser(
        "index-mri",
        help="Indexar unicamente estudios MRI desde linea de comandos"
    )
    _add_index_arguments(index_mri_parser, "MRI/")
    index_mri_parser.set_defaults(func=cmd_index_mri)

    gen_ct_parser = subparsers.add_parser(
        "gen-ct",
        help="Generar volumen 3D NIfTI a partir de una serie CT DICOM"
    )
    gen_ct_parser.add_argument(
        "--ct-dir", required=True,
        help="Ruta relativa al directorio de la serie CT dentro del DICOM root"
    )
    gen_ct_parser.add_argument(
        "--dicom-root", required=True,
        help="Ruta al directorio raiz DICOM"
    )
    gen_ct_parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el archivo NIfTI (.nii o .nii.gz)"
    )
    gen_ct_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir informacion detallada"
    )
    gen_ct_parser.set_defaults(func=cmd_gen_ct)

    gen_pet_parser = subparsers.add_parser(
        "gen-pet",
        help="Generar volumen 3D NIfTI a partir de una serie PET DICOM"
    )
    gen_pet_parser.add_argument(
        "--pet-dir", required=True,
        help="Ruta relativa al directorio de la serie PET dentro del DICOM root"
    )
    gen_pet_parser.add_argument(
        "--dicom-root", required=True,
        help="Ruta al directorio raiz DICOM"
    )
    gen_pet_parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el archivo NIfTI (.nii o .nii.gz)"
    )
    gen_pet_parser.add_argument(
        "--suv", action="store_true",
        help="Convertir el volumen a SUVbw usando metadatos DICOM"
    )
    gen_pet_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir informacion detallada"
    )
    gen_pet_parser.set_defaults(func=cmd_gen_pet)

    gen_mri_parser = subparsers.add_parser(
        "gen-mri",
        help="Generar volumen 3D NIfTI a partir de una serie MRI DICOM o Analyze"
    )
    gen_mri_parser.add_argument(
        "--mri-dir", "-m", required=True,
        help="Ruta relativa o absoluta al directorio o archivo de la serie MRI"
    )
    gen_mri_parser.add_argument(
        "--dicom-root", "-d", required=True,
        help="Ruta al directorio raiz DICOM"
    )
    gen_mri_parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el archivo NIfTI (.nii o .nii.gz)"
    )
    gen_mri_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir informacion detallada"
    )
    gen_mri_parser.set_defaults(func=cmd_gen_mri)

    fusion_parser = subparsers.add_parser(
        "fusion",
        help="Generar volumen fusionado PET/CT a partir de directorios CT y PET"
    )
    fusion_parser.add_argument(
        "--ct-dir", required=True,
        help="Ruta al directorio de la serie CT"
    )
    fusion_parser.add_argument(
        "--pet-dir", required=True,
        help="Ruta al directorio de la serie PET"
    )
    fusion_parser.add_argument(
        "--dicom-root", required=True,
        help="Ruta al directorio raiz DICOM"
    )
    fusion_parser.add_argument(
        "--output", "-o", default=None,
        help="Ruta de salida para el archivo NIfTI fusionado (.nii.gz)"
    )
    fusion_parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Imprimir informacion detallada"
    )
    fusion_parser.set_defaults(func=cmd_fusion)

    seg_ct_parser = subparsers.add_parser(
        "segment-ct",
        help="Segmentar órganos o fantomas en un volumen CT"
    )
    seg_ct_parser.add_argument(
        "--input", "-i", default=None,
        help="Ruta al volumen CT en formato NIfTI (.nii o .nii.gz)"
    )
    seg_ct_parser.add_argument(
        "--organ", "-g", default=None,
        help="Estructura u órgano a segmentar (ej: cerebro, corazon, sistema_respiratorio, intestino, caja_toracica, fantoma_agua)"
    )
    seg_ct_parser.add_argument(
        "--cuda", action="store_true", default=True, dest="cuda",
        help="Usar GPU (CUDA) si está disponible (default: True)"
    )
    seg_ct_parser.add_argument(
        "--no-cuda", action="store_false", dest="cuda",
        help="Forzar uso de CPU"
    )
    seg_ct_parser.add_argument(
        "--fast", action="store_true", default=False,
        help="Modo rápido de resolución 3mm (TotalSegmentator)"
    )
    seg_ct_parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directorio de salida (default: ./segmentations/)"
    )
    seg_ct_parser.add_argument(
        "--output-basename", default=None,
        help="Nombre base para archivos de salida (sin extensión)"
    )
    seg_ct_parser.add_argument(
        "--quiet", "-q", action="store_true", default=False,
        help="Suprimir mensajes informativos"
    )
    seg_ct_parser.add_argument(
        "--list-organs", action="store_true",
        help="Listar estructuras CT disponibles y salir"
    )
    seg_ct_parser.set_defaults(func=cmd_segment_ct)

    seg_mri_parser = subparsers.add_parser(
        "segment-mri",
        help="Segmentar órganos o fantomas en un volumen MRI"
    )
    seg_mri_parser.add_argument(
        "--input", "-i", default=None,
        help="Ruta al volumen MRI en formato NIfTI (.nii o .nii.gz)"
    )
    seg_mri_parser.add_argument(
        "--organ", "-g", default=None,
        help="Estructura u órgano a segmentar (ej: cerebro, corazon, fantoma_uniformidad)"
    )
    seg_mri_parser.add_argument(
        "--cuda", action="store_true", default=True, dest="cuda",
        help="Usar GPU (CUDA) si está disponible (default: True)"
    )
    seg_mri_parser.add_argument(
        "--no-cuda", action="store_false", dest="cuda",
        help="Forzar uso de CPU"
    )
    seg_mri_parser.add_argument(
        "--fast", action="store_true", default=False,
        help="Modo rápido de resolución 3mm (TotalSegmentator)"
    )
    seg_mri_parser.add_argument(
        "--output-dir", "-o", default=None,
        help="Directorio de salida (default: ./segmentations/)"
    )
    seg_mri_parser.add_argument(
        "--output-basename", default=None,
        help="Nombre base para archivos de salida (sin extensión)"
    )
    seg_mri_parser.add_argument(
        "--quiet", "-q", action="store_true", default=False,
        help="Suprimir mensajes informativos"
    )
    seg_mri_parser.add_argument(
        "--list-organs", action="store_true",
        help="Listar estructuras MRI disponibles y salir"
    )
    seg_mri_parser.set_defaults(func=cmd_segment_mri)

    args = parser.parse_args()

    # Si no se específica subcomando, lanzar la GUI por defecto
    if not hasattr(args, "func"):
        cmd_gui(args)
        return

    args.func(args)


if __name__ == "__main__":
    main()

