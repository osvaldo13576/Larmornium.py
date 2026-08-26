#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_3d.py - Renderizado 3D y extracción de silueta anatómica (CT / MRI).

Genera un render 3D y una malla de superficie poligonal mostrando únicamente la
silueta y el contorno exterior del objeto o cuerpo a partir de un volumen 3D generado
por gen_volume_CT.py o gen_volume_MRI.py (o cualquier archivo NIfTI o DICOM).

La implementación está generalizada para cualquier volumen de Tomografía Computarizada
(CT) o Resonancia Magnética (MRI):
  - CT: Detección de tejido y cuerpo por umbral de Unidades Hounsfield (HU > -300),
    rellenado de cavidades internas (pulmones, aire intestinal, etc.) en 2D/3D y
    selección del componente conectado principal (cuerpo o fantoma) eliminando la camilla.
  - MRI: Umbralización adaptativa (Otsu sobre el fondo y percentiles de señal),
    rellenado morfológico de huecos y extracción del mayor componente conexo.
  - VTK Pipeline: Extracción de isosuperficie 3D de alto rendimiento con
    vtkFlyingEdges3D o vtkMarchingCubes, suavizado vtkWindowedSincPolyDataFilter
    para preservar el volumen sin encogimiento, normales de superficie e iluminación Phong.
  - Exportación: Soporte para guardar en formato STL (.stl), NIfTI (.nii.gz)
    o captura de imagen PNG con renderizado offscreen o ventana interactiva.

Uso desde terminal:
    python3 render_3d.py -i ./volumes/paciente_00_ct.nii.gz --output-stl ./volumes/silueta_paciente_00.stl
    python3 render_3d.py -i ./volumes/340714432_901_ANATOMICO_20101020_mri.nii.gz --output-png ./silueta_mri.png
    python3 render_3d.py -i ./volumes/paciente_00_ct.nii.gz --show

Uso como módulo Python:
    from render_3d import extract_silhouette_mask, build_silhouette_polydata, render_3d_silhouette
    mask, spacing, meta = extract_silhouette_mask(nifti_image, modality="ct")
    polydata = build_silhouette_polydata(mask, spacing)
"""

import argparse
import logging
import os
import sys
import time

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

try:
    import skimage.filters as skf
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import vtk
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersCore import (
        vtkDecimatePro,
        vtkFlyingEdges3D,
        vtkPolyDataNormals,
        vtkWindowedSincPolyDataFilter,
    )
    from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
    from vtkmodules.vtkIOGeometry import vtkSTLWriter
    from vtkmodules.vtkIOImage import vtkPNGWriter
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
    from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkRenderWindowInteractor,
        vtkWindowToImageFilter,
    )
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False

logger = logging.getLogger("render_3d")


def detect_modality(volume_data):
    """
    Determina automáticamente si los datos corresponden a CT o MRI según el rango de valores.
    - CT suele tener valores negativos (aire aprox -1000 HU, tejido aprox 0 a 100 HU, hueso > 400 HU).
    - MRI suele tener intensidades no negativas (0 a varios miles).
    """
    min_val = float(np.min(volume_data))
    max_val = float(np.max(volume_data))
    if min_val < -200.0 or (min_val < -50.0 and max_val > 500.0):
        return "ct"
    return "mri"


def extract_silhouette_mask(volume_input, voxel_spacing=None, modality="auto", hu_threshold=-300.0,
                            fill_holes=True, lcc=True,
                            fast_mode=False, subsample_step=1):
    """
    Genera una máscara binaria 3D correspondiente a la silueta o contorno exterior del
    objeto o cuerpo en el volumen.

    Parameters
    ----------
    volume_input : str, nibabel.Nifti1Image o numpy.ndarray
        Ruta al archivo NIfTI, objeto Nibabel o arreglo 3D numpy.
    voxel_spacing : tuple o list, optional
        [sx, sy, sz] en milímetros.
    modality : str, optional
        'auto', 'ct' o 'mri'. Por defecto 'auto'.
    hu_threshold : float, optional
        Umbral HU para aislar tejido u objeto en CT (por defecto: -300.0 HU).
    fill_holes : bool, optional
        Si True, rellena cavidades y huecos internos para formar una silueta sólida.
    lcc : bool, optional
        Si True, retiene únicamente componentes conexos significativos.
    fast_mode : bool, optional
        Si True, aplica submuestreo para generación rápida.
    subsample_step : int, optional
        Paso de submuestreo espacial (1 = resolución completa, 2 = 2x más rápido).

    Returns
    -------
    tuple (numpy.ndarray, list, dict)
        - mask_data (uint8): Arreglo 3D de 0 y 1 con la silueta.
        - voxel_spacing (list): [sx, sy, sz] en milímetros.
        - metadata (dict): Información técnica sobre la extracción.
    """
    t0 = time.time()
    affine = np.eye(4)

    if isinstance(volume_input, str):
        if not os.path.exists(volume_input):
            raise FileNotFoundError(f"No se encontró el archivo de volumen: {volume_input}")
        nii = nib.load(volume_input)
        raw_data = np.transpose(nii.get_fdata(), (2, 1, 0)).astype(np.float32)
        nii_zooms = [float(v) for v in nii.header.get_zooms()[:3]]
        curr_spacing = voxel_spacing or nii_zooms
        affine = nii.affine
    elif hasattr(volume_input, "get_fdata"):
        raw_data = np.transpose(volume_input.get_fdata(), (2, 1, 0)).astype(np.float32)
        nii_zooms = [float(v) for v in volume_input.header.get_zooms()[:3]]
        curr_spacing = voxel_spacing or nii_zooms
        affine = volume_input.affine
    elif isinstance(volume_input, np.ndarray):
        raw_data = volume_input.astype(np.float32)
        curr_spacing = voxel_spacing or [1.0, 1.0, 1.0]
    else:
        raise ValueError("Tipo de entrada no soportado para volume_input")

    sp_x = float(curr_spacing[0]) if len(curr_spacing) > 0 and float(curr_spacing[0]) > 0 else 1.0
    sp_y = float(curr_spacing[1]) if len(curr_spacing) > 1 and float(curr_spacing[1]) > 0 else 1.0
    sp_z = float(curr_spacing[2]) if len(curr_spacing) > 2 and float(curr_spacing[2]) > 0 else 1.0
    base_spacing = [sp_x, sp_y, sp_z]

    if raw_data.ndim == 4:
        raw_data = np.squeeze(raw_data) if raw_data.shape[-1] == 1 else raw_data[..., 0]

    detected_mod = detect_modality(raw_data) if modality == "auto" else str(modality).lower()

    step = 2 if fast_mode and subsample_step == 1 else max(1, int(subsample_step))
    if step > 1:
        data = raw_data[::step, ::step, ::step]
        eff_spacing = [base_spacing[0] * step, base_spacing[1] * step, base_spacing[2] * step]
    else:
        data = raw_data
        eff_spacing = list(base_spacing)

    smoothed = ndi.gaussian_filter(data.astype(np.float32), sigma=0.8)

    if detected_mod == "ct":
        bin_mask = (smoothed > hu_threshold)
    else:
        nz = smoothed[smoothed > 0]
        if len(nz) > 0:
            if SKIMAGE_AVAILABLE:
                try:
                    thresh_li = float(skf.threshold_li(nz))
                    thresh_tri = float(skf.threshold_triangle(nz))
                    thresh = max(thresh_tri, min(thresh_li, 0.25 * float(np.percentile(nz, 95))))
                except Exception:
                    thresh = float(skf.threshold_otsu(nz)) * 0.4
            else:
                thresh = float(np.percentile(nz, 20))
        else:
            thresh = 0.0
        bin_mask = (smoothed > thresh)

    if fill_holes:
        filled_mask = np.zeros_like(bin_mask, dtype=bool)
        for z in range(bin_mask.shape[2]):
            slice_z = bin_mask[:, :, z]
            if np.any(slice_z):
                filled_mask[:, :, z] = ndi.binary_fill_holes(slice_z)

        for y in range(bin_mask.shape[1]):
            slice_y = filled_mask[:, y, :]
            if np.any(slice_y):
                filled_mask[:, y, :] = ndi.binary_fill_holes(slice_y)

        bin_mask = filled_mask

    struct = ndi.generate_binary_structure(3, 1)
    bin_mask = ndi.binary_closing(bin_mask, structure=struct, iterations=2)
    bin_mask = ndi.binary_opening(bin_mask, structure=struct, iterations=1)

    if lcc:
        labeled, num_features = ndi.label(bin_mask)
        if num_features > 0:
            sizes = ndi.sum(bin_mask, labeled, range(1, num_features + 1))
            max_size = np.max(sizes)
            valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= 0.05 * max_size]
            bin_mask = np.isin(labeled, valid_labels)

    mask_uint8 = bin_mask.astype(np.uint8)
    elapsed = time.time() - t0

    num_voxels = int(np.sum(mask_uint8))
    voxel_vol_cm3 = (eff_spacing[0] * eff_spacing[1] * eff_spacing[2]) / 1000.0
    volume_cm3 = round(num_voxels * voxel_vol_cm3, 2)

    metadata = {
        "modality": detected_mod,
        "original_shape": list(raw_data.shape),
        "processed_shape": list(data.shape),
        "voxel_spacing_mm": eff_spacing,
        "num_voxels_silhouette": num_voxels,
        "volume_cm3": volume_cm3,
        "extraction_time_seconds": round(elapsed, 3),
        "fast_mode": (step > 1),
        "subsample_step": step,
        "affine": affine.tolist() if isinstance(affine, np.ndarray) else affine,
    }

    return mask_uint8, eff_spacing, metadata


def build_silhouette_polydata(mask_data, voxel_spacing=(1.0, 1.0, 1.0),
                              smoothing_iterations=20, pass_band=0.1,
                              decimate_fraction=0.0):
    """
    Construye la malla de superficie 3D (vtkPolyData) a partir de la máscara binaria.

    Parameters
    ----------
    mask_data : numpy.ndarray
        Máscara 3D binaria (uint8 o bool).
    voxel_spacing : tuple o list
        [sx, sy, sz] en milímetros.
    smoothing_iterations : int, optional
        Número de iteraciones del filtro Windowed Sinc (por defecto: 20).
    pass_band : float, optional
        Banda de paso del filtro de suavizado (por defecto: 0.1).
    decimate_fraction : float, optional
        Fracción de simplificación de triángulos (0.0 a 0.8).

    Returns
    -------
    vtkPolyData
        Malla 3D lista para renderizar o exportar a STL.
    """
    if not VTK_AVAILABLE:
        raise RuntimeError("Se requiere la biblioteca VTK para construir vtkPolyData.")

    data = (mask_data > 0).astype(np.uint8)
    if data.ndim == 3:
        nz, ny, nx = data.shape
    elif data.ndim == 2:
        nz = 1
        ny, nx = data.shape
        data = data.reshape((1, ny, nx))
    else:
        return vtk.vtkPolyData()

    if np.sum(data) == 0:
        return vtk.vtkPolyData()

    sp_x = float(voxel_spacing[0]) if len(voxel_spacing) > 0 and float(voxel_spacing[0]) > 0 else 1.0
    sp_y = float(voxel_spacing[1]) if len(voxel_spacing) > 1 and float(voxel_spacing[1]) > 0 else 1.0
    sp_z = float(voxel_spacing[2]) if len(voxel_spacing) > 2 and float(voxel_spacing[2]) > 0 else 1.0

    vtk_img = vtkImageData()
    vtk_img.SetDimensions(nx, ny, nz)
    vtk_img.SetSpacing(sp_x, sp_y, sp_z)
    vtk_img.SetOrigin(0.0, 0.0, 0.0)

    flat_data = np.ascontiguousarray(data).ravel()
    vtk_arr = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_img.GetPointData().SetScalars(vtk_arr)

    surface_extractor = vtkFlyingEdges3D()
    surface_extractor.SetInputData(vtk_img)
    surface_extractor.SetValue(0, 0.5)
    surface_extractor.Update()

    current_output = surface_extractor.GetOutputPort()

    if decimate_fraction > 0.01:
        decimator = vtkDecimatePro()
        decimator.SetInputConnection(current_output)
        decimator.SetTargetReduction(min(0.9, float(decimate_fraction)))
        decimator.PreserveTopologyOn()
        decimator.Update()
        current_output = decimator.GetOutputPort()

    if smoothing_iterations > 0:
        smoother = vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(current_output)
        smoother.SetNumberOfIterations(int(smoothing_iterations))
        smoother.SetPassBand(float(pass_band))
        smoother.BoundarySmoothingOn()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        current_output = smoother.GetOutputPort()

    normals = vtkPolyDataNormals()
    normals.SetInputConnection(current_output)
    normals.SetFeatureAngle(60.0)
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def save_polydata_stl(polydata, output_path):
    """Guarda la malla vtkPolyData en formato STL binario."""
    if not VTK_AVAILABLE:
        raise RuntimeError("VTK es necesario para exportar a STL.")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = vtkSTLWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.SetFileTypeToBinary()
    writer.Write()
    logger.info("Malla STL guardada en: %s", output_path)
    return output_path


def save_mask_nifti(mask_data, output_path, affine=None, voxel_spacing=None):
    """Guarda la máscara binaria en formato NIfTI (.nii.gz)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if affine is None:
        if voxel_spacing is not None:
            affine = np.diag([voxel_spacing[0], voxel_spacing[1], voxel_spacing[2], 1.0])
        else:
            affine = np.eye(4)
    nii = nib.Nifti1Image(mask_data.astype(np.uint8), affine)
    nib.save(nii, output_path)
    logger.info("Máscara NIfTI guardada en: %s", output_path)
    return output_path


def render_3d_silhouette(volume_input, modality="auto", output_stl=None,
                        output_nii=None, output_png=None, show=False,
                        opacity=0.85, color=(0.80, 0.88, 0.96),
                        bg_color=(0.10, 0.11, 0.14), hu_threshold=-300.0,
                        fast_mode=False, smoothing_iterations=25):
    """
    Pipeline completo: extrae la silueta 3D, genera la malla poligonal,
    la exporta o renderiza según los parámetros especificados.

    Parameters
    ----------
    volume_input : str, nibabel.Nifti1Image o numpy.ndarray
        Volumen de entrada.
    modality : str
        'auto', 'ct' o 'mri'.
    output_stl : str, optional
        Ruta para exportar archivo .stl.
    output_nii : str, optional
        Ruta para guardar máscara .nii.gz.
    output_png : str, optional
        Ruta para guardar captura PNG de la silueta renderizada.
    show : bool, optional
        Si True, abre una ventana interactiva VTK.
    opacity : float
        Opacidad de la silueta (0.0 a 1.0).
    color : tuple (R, G, B)
        Color de la superficie (0.0 a 1.0).
    bg_color : tuple (R, G, B)
        Color de fondo del render.
    hu_threshold : float
        Umbral HU para CT.
    fast_mode : bool
        Modo rápido con submuestreo.
    smoothing_iterations : int
        Iteraciones de suavizado Windowed Sinc.

    Returns
    -------
    dict
        Diccionario con 'polydata', 'mask_data', 'metadata', 'output_stl', 'output_png'.
    """
    logger.info("Extrayendo silueta 3D del volumen...")
    mask_data, spacing, meta = extract_silhouette_mask(
        volume_input=volume_input,
        modality=modality,
        hu_threshold=hu_threshold,
        fast_mode=fast_mode
    )

    logger.info("Silueta extraída: %d vóxeles (volumen: %.2f cm³)", meta["num_voxels_silhouette"], meta["volume_cm3"])

    polydata = build_silhouette_polydata(
        mask_data=mask_data,
        voxel_spacing=spacing,
        smoothing_iterations=smoothing_iterations
    )

    result = {
        "metadata": meta,
        "polydata": polydata,
        "mask_data": mask_data,
        "voxel_spacing": spacing,
        "output_stl": None,
        "output_nii": None,
        "output_png": None,
    }

    if output_stl:
        save_polydata_stl(polydata, output_stl)
        result["output_stl"] = output_stl

    if output_nii:
        affine = np.array(meta.get("affine", np.eye(4)))
        save_mask_nifti(mask_data, output_nii, affine=affine, voxel_spacing=spacing)
        result["output_nii"] = output_nii

    if VTK_AVAILABLE and (output_png or show):
        renderer = vtkRenderer()
        renderer.SetBackground(*bg_color)
        renderer.AutomaticLightCreationOn()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        actor = vtkActor()
        actor.SetMapper(mapper)

        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(float(opacity))
        prop.SetSpecular(0.40)
        prop.SetSpecularPower(30.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetInterpolationToPhong()
        prop.BackfaceCullingOff()

        renderer.AddActor(actor)

        outline = vtkOutlineFilter()
        outline.SetInputData(polydata)
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(0.0, 0.77, 1.0)
        outline_actor.GetProperty().SetLineWidth(1.0)
        renderer.AddActor(outline_actor)

        bounds = actor.GetBounds()
        center = [
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ]
        diag = np.sqrt(
            (bounds[1] - bounds[0]) ** 2 +
            (bounds[3] - bounds[2]) ** 2 +
            (bounds[5] - bounds[4]) ** 2
        )
        dist = max(diag * 1.8, 100.0)

        camera = renderer.GetActiveCamera()
        camera.SetPosition(center[0], center[1] - dist, center[2])
        camera.SetFocalPoint(*center)
        camera.SetViewUp(0, 0, 1)
        renderer.ResetCameraClippingRange()

        render_win = vtkRenderWindow()
        render_win.SetSize(1024, 768)
        render_win.AddRenderer(renderer)

        if output_png:
            render_win.SetOffScreenRendering(1)
            render_win.Render()

            w2if = vtkWindowToImageFilter()
            w2if.SetInput(render_win)
            w2if.Update()

            os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
            png_writer = vtkPNGWriter()
            png_writer.SetFileName(output_png)
            png_writer.SetInputConnection(w2if.GetOutputPort())
            png_writer.Write()
            logger.info("Render PNG guardado en: %s", output_png)
            result["output_png"] = output_png

        if show:
            render_win.SetOffScreenRendering(0)
            interactor = vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_win)
            style = vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)

            axes = vtkAxesActor()
            axes.SetTotalLength(25.0, 25.0, 25.0)
            axes.SetXAxisLabelText("R-L")
            axes.SetYAxisLabelText("A-P")
            axes.SetZAxisLabelText("I-S")
            axes_widget = vtkOrientationMarkerWidget()
            axes_widget.SetOrientationMarker(axes)
            axes_widget.SetInteractor(interactor)
            axes_widget.SetViewport(0.0, 0.0, 0.22, 0.22)
            axes_widget.SetEnabled(1)
            axes_widget.InteractiveOff()

            render_win.Render()
            interactor.Start()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="render_3d.py - Renderizado 3D y extracción de silueta anatómica (CT / MRI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python3 render_3d.py -i ./volumes/paciente_00_ct.nii.gz --output-stl ./volumes/silueta_ct.stl
  python3 render_3d.py -i ./volumes/340714432_901_ANATOMICO_20101020_mri.nii.gz --output-png ./silueta_mri.png
  python3 render_3d.py -i ./volumes/paciente_00_ct.nii.gz --show
        """
    )

    parser.add_argument("-i", "--input-volume", required=True,
                        help="Ruta al volumen NIfTI (.nii / .nii.gz) o serie de entrada")
    parser.add_argument("-m", "--modality", choices=["auto", "ct", "mri"], default="auto",
                        help="Modalidad del volumen (por defecto: auto)")
    parser.add_argument("-o", "--output-stl", default=None,
                        help="Ruta para exportar la malla 3D en formato STL")
    parser.add_argument("--output-nii", default=None,
                        help="Ruta para guardar la máscara binaria de la silueta en NIfTI")
    parser.add_argument("--output-png", default=None,
                        help="Ruta para guardar una captura del render 3D en formato PNG")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Abrir ventana interactiva VTK para rotar y visualizar la silueta 3D")
    parser.add_argument("--hu-threshold", type=float, default=-300.0,
                        help="Umbral de Unidades Hounsfield para aislar el cuerpo en CT (por defecto: -300)")
    parser.add_argument("--opacity", type=float, default=0.75,
                        help="Opacidad de la silueta en el render 3D (0.0 a 1.0, por defecto: 0.75)")
    parser.add_argument("--fast", action="store_true",
                        help="Modo rápido con submuestreo espacial para generación instantánea")
    parser.add_argument("--smooth-iter", type=int, default=20,
                        help="Número de iteraciones del filtro de suavizado de malla (por defecto: 20)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Imprimir información detallada en consola")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    res = render_3d_silhouette(
        volume_input=args.input_volume,
        modality=args.modality,
        output_stl=args.output_stl,
        output_nii=args.output_nii,
        output_png=args.output_png,
        show=args.show,
        opacity=args.opacity,
        hu_threshold=args.hu_threshold,
        fast_mode=args.fast,
        smoothing_iterations=args.smooth_iter
    )

    meta = res["metadata"]
    print("\n" + "=" * 60)
    print("  RESULTADO DE EXTRACCIÓN DE SILUETA 3D")
    print("=" * 60)
    print(f"  Modalidad detectada : {meta['modality'].upper()}")
    print(f"  Dimensiones volumen : {meta['original_shape']}")
    print(f"  Espaciado vóxel (mm): {meta['voxel_spacing_mm']}")
    print(f"  Vóxeles silueta     : {meta['num_voxels_silhouette']:,}")
    print(f"  Volumen físico      : {meta['volume_cm3']} cm³")
    print(f"  Tiempo de proceso   : {meta['extraction_time_seconds']} s")
    if res["output_stl"]:
        print(f"  Archivo STL guardado: {res['output_stl']}")
    if res["output_nii"]:
        print(f"  Máscara NIfTI       : {res['output_nii']}")
    if res["output_png"]:
        print(f"  Captura PNG         : {res['output_png']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
