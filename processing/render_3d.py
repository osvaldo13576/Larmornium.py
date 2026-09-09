#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
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
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkPolyDataMapper,
        vtkRenderer,
        vtkRenderWindow,
        vtkRenderWindowInteractor,
        vtkTextActor,
        vtkWindowToImageFilter,
    )
    VTK_AVAILABLE = True
except Exception:
    VTK_AVAILABLE = False

logger = logging.getLogger("render_3d")


def _triangle_threshold(data):
    counts, bin_edges = np.histogram(data, bins=256)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    peak_idx = int(np.argmax(counts))
    nz_bins = np.where(counts > 0)[0]
    if len(nz_bins) == 0:
        return 0.0
    end_idx = int(nz_bins[-1])
    if peak_idx >= end_idx:
        return float(bin_centers[peak_idx])
    x1, y1 = float(peak_idx), float(counts[peak_idx])
    x2, y2 = float(end_idx), float(counts[end_idx])
    dx, dy = x2 - x1, y2 - y1
    indices = np.arange(peak_idx, end_idx + 1, dtype=np.float64)
    y_vals = counts[peak_idx:end_idx + 1].astype(np.float64)
    dist = np.abs(dy * indices - dx * y_vals + x2 * y1 - y2 * x1)
    best_bin = peak_idx + int(np.argmax(dist))
    return float(bin_centers[best_bin])


def _otsu_threshold(data):
    counts, bin_edges = np.histogram(data, bins=256)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    total = np.sum(counts)
    if total == 0:
        return 0.0
    weight1 = np.cumsum(counts)
    weight2 = total - weight1
    sum_all = np.sum(counts * bin_centers)
    sum1 = np.cumsum(counts * bin_centers)
    valid = (weight1 > 0) & (weight2 > 0)
    if not np.any(valid):
        return float(bin_centers[int(np.argmax(counts))])
    mean1 = np.zeros_like(sum1, dtype=np.float64)
    mean2 = np.zeros_like(sum1, dtype=np.float64)
    mean1[valid] = sum1[valid] / weight1[valid]
    mean2[valid] = (sum_all - sum1[valid]) / weight2[valid]
    variance = np.zeros_like(sum1, dtype=np.float64)
    variance[valid] = weight1[valid] * weight2[valid] * ((mean1[valid] - mean2[valid]) ** 2)
    return float(bin_centers[int(np.argmax(variance))])


def compute_optimal_mri_threshold(volume_data):
    nz = volume_data[volume_data > 0]
    if len(nz) == 0:
        return 0.0

    if SKIMAGE_AVAILABLE:
        try:
            tri = float(skf.threshold_triangle(nz))
            otsu = float(skf.threshold_otsu(nz))
        except Exception:
            tri = _triangle_threshold(nz)
            otsu = _otsu_threshold(nz)
    else:
        tri = _triangle_threshold(nz)
        otsu = _otsu_threshold(nz)

    if otsu > tri:
        # El 10% del rango entre el límite de ruido (triangle) y la bimodalidad (Otsu)
        # elimina por completo el ruido de fondo y camilla sin cortar partes finas.
        thresh = tri + 0.10 * (otsu - tri)
    else:
        thresh = tri

    return float(thresh)


def detect_modality(volume_data):
    min_val = float(np.min(volume_data))
    max_val = float(np.max(volume_data))
    if min_val < -200.0 or (min_val < -50.0 and max_val > 500.0):
        return "ct"
    return "mri"


def extract_silhouette_mask(volume_input, voxel_spacing=None, modality="auto", hu_threshold=-300.0,
                            mri_threshold=None, fill_holes=True, lcc=True,
                            only_largest=False, min_volume_cm3=0.2,
                            fast_mode=False, subsample_step=1):
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

    # Suavizado anisotrópico respetando la resolución física de los vóxeles:
    # En el plano (X, Y) se atenúa ruido de alta frecuencia manteniendo bordes nítidos.
    # A lo largo del eje entre cortes (Z), se limita la dispersión para no fusionar cortes ni partes finas.
    sp_x_eff, sp_y_eff, sp_z_eff = eff_spacing
    sigma_xy = 0.6
    sigma_z = max(0.15, min(0.6, 0.6 * min(sp_x_eff, sp_y_eff) / max(sp_z_eff, 1e-3)))
    smoothed = ndi.gaussian_filter(data.astype(np.float32), sigma=(sigma_z, sigma_xy, sigma_xy))

    if detected_mod == "ct":
        thresh_used = float(hu_threshold)
        bin_mask = (smoothed > hu_threshold)
    else:
        if mri_threshold is not None:
            thresh_used = float(mri_threshold)
        else:
            thresh_used = compute_optimal_mri_threshold(smoothed)
        bin_mask = (smoothed > thresh_used)

    if fill_holes:
        if detected_mod == "ct":
            # En CT, rellenado axial (Z) para cavidades internas como pulmones, seguido de 3D
            filled_mask = np.zeros_like(bin_mask, dtype=bool)
            for z in range(bin_mask.shape[0]):
                if np.any(bin_mask[z]):
                    filled_mask[z] = ndi.binary_fill_holes(bin_mask[z])
            bin_mask = ndi.binary_fill_holes(filled_mask)
        else:
            # En MRI, rellenado estrictamente 3D para preservar concavidades abiertas, espacios
            # entre hojas de corona (piña) y separar objetos vecinos sin crear cilindros artificiales.
            bin_mask = ndi.binary_fill_holes(bin_mask)

    # Filtrado de ruido y preservación de múltiples objetos reales:
    # NO se aplican binary_closing o binary_opening agresivos en 3D que erosionan hojas
    # finas o puentean objetos separados.
    voxel_vol_cm3 = (eff_spacing[0] * eff_spacing[1] * eff_spacing[2]) / 1000.0
    min_voxels = max(20, int(round(min_volume_cm3 / max(voxel_vol_cm3, 1e-6))))

    num_objects = 0
    if lcc:
        labeled, num_features = ndi.label(bin_mask)
        if num_features > 0:
            sizes = ndi.sum(bin_mask, labeled, range(1, num_features + 1))
            if only_largest:
                max_idx = int(np.argmax(sizes)) + 1
                bin_mask = (labeled == max_idx)
                num_objects = 1
            else:
                # Conservar todos los objetos con volumen físico significativo, eliminando polvo/ruido
                valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
                if valid_labels:
                    bin_mask = np.isin(labeled, valid_labels)
                    num_objects = len(valid_labels)
                else:
                    max_idx = int(np.argmax(sizes)) + 1
                    bin_mask = (labeled == max_idx)
                    num_objects = 1
    else:
        labeled, num_objects = ndi.label(bin_mask)

    mask_uint8 = bin_mask.astype(np.uint8)
    elapsed = time.time() - t0

    num_voxels = int(np.sum(mask_uint8))
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
        "threshold_used": round(thresh_used, 2),
        "num_objects": int(num_objects),
    }

    return mask_uint8, eff_spacing, metadata


def build_silhouette_polydata(mask_data, voxel_spacing=(1.0, 1.0, 1.0),
                              smoothing_iterations=20, pass_band=0.1,
                              decimate_fraction=0.0):
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
                        mri_threshold=None, min_volume_cm3=0.2, only_largest=False,
                        fast_mode=False, smoothing_iterations=20):
    logger.info("Extrayendo silueta 3D del volumen...")
    mask_data, spacing, meta = extract_silhouette_mask(
        volume_input=volume_input,
        modality=modality,
        hu_threshold=hu_threshold,
        mri_threshold=mri_threshold,
        min_volume_cm3=min_volume_cm3,
        only_largest=only_largest,
        fast_mode=fast_mode
    )

    logger.info("Silueta extraída: %d vóxeles en %d objeto(s) (volumen: %.2f cm3, umbral: %s)",
                meta["num_voxels_silhouette"], meta.get("num_objects", 1), meta["volume_cm3"], meta.get("threshold_used", "N/A"))

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
            w2i = vtkWindowToImageFilter()
            w2i.SetInput(render_win)
            w2i.Update()
            writer = vtkPNGWriter()
            os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
            writer.SetFileName(output_png)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            logger.info("Captura PNG guardada en: %s", output_png)
            result["output_png"] = output_png

        if show:
            interactor = vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_win)
            style = vtkInteractorStyleTrackballCamera()
            interactor.SetInteractorStyle(style)
            interactor.Initialize()

            text_actor = vtkTextActor()
            text_str = (
                f"Larmornium 3D | {meta['modality'].upper()}\n"
                f"Objetos: {meta.get('num_objects', 1)} | Vóxeles: {meta['num_voxels_silhouette']:,}\n"
                f"Volumen: {meta['volume_cm3']} cm3 | Umbral: {meta.get('threshold_used', 'N/A')}"
            )
            text_actor.SetInput(text_str)
            text_prop = text_actor.GetTextProperty()
            text_prop.SetFontSize(14)
            text_prop.SetColor(0.85, 0.85, 0.90)
            text_prop.SetFontFamilyToArial()
            text_actor.SetPosition(15, 15)
            renderer.AddActor2D(text_actor)

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
    parser.add_argument("--mri-threshold", type=float, default=None,
                        help="Umbral de intensidad para MRI (por defecto: cálculo adaptativo automático)")
    parser.add_argument("--min-volume-cm3", type=float, default=0.2,
                        help="Volumen mínimo en cm3 para conservar objetos reales (por defecto: 0.2)")
    parser.add_argument("--only-largest", action="store_true",
                        help="Conservar únicamente el objeto de mayor tamaño (desactiva multi-objeto)")
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
        mri_threshold=args.mri_threshold,
        min_volume_cm3=args.min_volume_cm3,
        only_largest=args.only_largest,
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
    print(f"  Objetos detectados  : {meta.get('num_objects', 1)}")
    print(f"  Umbral aplicado     : {meta.get('threshold_used', 'N/A')}")
    print(f"  Vóxeles silueta     : {meta['num_voxels_silhouette']:,}")
    print(f"  Volumen físico      : {meta['volume_cm3']} cm3")
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
