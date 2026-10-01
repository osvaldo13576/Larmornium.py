import os
import json
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from PIL import Image


PALETTE = [
    np.array([0, 230, 140], dtype=np.float32),
    np.array([255, 170, 0], dtype=np.float32),
    np.array([50, 180, 255], dtype=np.float32),
    np.array([175, 110, 255], dtype=np.float32),
    np.array([255, 80, 110], dtype=np.float32),
    np.array([255, 225, 50], dtype=np.float32),
]


def _normalize_grayscale(slice_2d):
    valid = slice_2d[np.isfinite(slice_2d)]
    if valid.size == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)
    p1, p99 = np.percentile(valid, (1.0, 99.0))
    if p99 > p1:
        norm = np.clip((slice_2d - p1) / (p99 - p1), 0.0, 1.0)
    else:
        norm = np.zeros_like(slice_2d)
    return (norm * 255.0).astype(np.uint8)


def _determine_output_path(seg_input, output_path):
    if output_path:
        return os.path.abspath(output_path)

    path_str = None
    if isinstance(seg_input, (str, Path)):
        path_str = str(seg_input)
    elif hasattr(seg_input, "get_filename") and seg_input.get_filename():
        path_str = seg_input.get_filename()

    if not path_str:
        raise ValueError("output_path es requerido cuando seg_volume no proviene de un archivo en disco.")

    if path_str.endswith(".nii.gz"):
        return os.path.abspath(path_str[:-7] + ".png")
    elif path_str.endswith(".nii"):
        return os.path.abspath(path_str[:-4] + ".png")
    return os.path.abspath(os.path.splitext(path_str)[0] + ".png")


def _find_underlay_path_from_metadata(seg_path):
    if not seg_path or not isinstance(seg_path, (str, Path)):
        return None
    p_str = str(seg_path)
    if p_str.endswith(".nii.gz"):
        json_path = p_str[:-7] + ".json"
    elif p_str.endswith(".nii"):
        json_path = p_str[:-4] + ".json"
    else:
        json_path = os.path.splitext(p_str)[0] + ".json"

    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            input_vol = meta.get("input_volume")
            if input_vol and os.path.isfile(input_vol):
                return input_vol
        except Exception:
            pass
    return None


def _load_volume_data(volume_input):
    if isinstance(volume_input, (str, Path)):
        nii = nib.load(str(volume_input))
        data = nii.get_fdata()
        return np.squeeze(data), nii
    elif hasattr(volume_input, "get_fdata"):
        data = volume_input.get_fdata()
        return np.squeeze(data), volume_input
    elif isinstance(volume_input, np.ndarray):
        return np.squeeze(volume_input), None
    raise TypeError(f"Tipo de volumen no soportado: {type(volume_input)}")


def create_segmentation_thumbnail(
    seg_volume,
    output_path=None,
    underlay_volume=None,
    target_size=(256, 256),
):
    out_png = _determine_output_path(seg_volume, output_path)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    seg_data, _ = _load_volume_data(seg_volume)
    if seg_data.ndim != 3:
        if seg_data.ndim > 3:
            seg_data = seg_data[..., 0]
        else:
            raise ValueError(f"La máscara debe ser tridimensional, recibida dimensión {seg_data.ndim}")

    non_zero_voxels = np.count_nonzero(seg_data > 0)
    if non_zero_voxels == 0:
        empty_img = np.full((target_size[1], target_size[0], 3), (20, 22, 26), dtype=np.uint8)
        Image.fromarray(empty_img).save(out_png)
        return out_png

    c0 = np.sum(seg_data > 0, axis=(1, 2))
    c1 = np.sum(seg_data > 0, axis=(0, 2))
    c2 = np.sum(seg_data > 0, axis=(0, 1))

    max0 = c0.max() if c0.size > 0 else 0
    max1 = c1.max() if c1.size > 0 else 0
    max2 = c2.max() if c2.size > 0 else 0

    areas = [max0, max1, max2]
    max_area = max(areas)

    # Priorizar vista axial si representa al menos el 70% del area maxima; en caso contrario coronal
    if max2 >= 0.70 * max_area:
        best_axis = 2
        best_slice_idx = int(np.argmax(c2))
        seg_slice = seg_data[:, :, best_slice_idx]
    elif max1 >= 0.85 * max_area:
        best_axis = 1
        best_slice_idx = int(np.argmax(c1))
        seg_slice = seg_data[:, best_slice_idx, :]
    else:
        best_axis = int(np.argmax(areas))
        if best_axis == 0:
            best_slice_idx = int(np.argmax(c0))
            seg_slice = seg_data[best_slice_idx, :, :]
        elif best_axis == 1:
            best_slice_idx = int(np.argmax(c1))
            seg_slice = seg_data[:, best_slice_idx, :]
        else:
            best_slice_idx = int(np.argmax(c2))
            seg_slice = seg_data[:, :, best_slice_idx]

    # Rotacion estandar radiologica (superior / anterior hacia arriba)
    seg_slice = np.rot90(seg_slice, 1)

    # Intentar resolver volumen anatomico base si no fue suministrado directamente
    if underlay_volume is None and isinstance(seg_volume, (str, Path)):
        underlay_volume = _find_underlay_path_from_metadata(seg_volume)

    underlay_data = None
    if underlay_volume is not None:
        try:
            u_data, _ = _load_volume_data(underlay_volume)
            if u_data.ndim == 3 and u_data.shape == seg_data.shape:
                underlay_data = u_data
        except Exception:
            underlay_data = None

    h, w = seg_slice.shape
    mask_binary = seg_slice > 0

    if underlay_data is not None:
        if best_axis == 0:
            u_slice = underlay_data[best_slice_idx, :, :]
        elif best_axis == 1:
            u_slice = underlay_data[:, best_slice_idx, :]
        else:
            u_slice = underlay_data[:, :, best_slice_idx]
        u_slice = np.rot90(u_slice, 1)

        gray = _normalize_grayscale(u_slice)
        rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

        unique_labels = [lbl for lbl in np.unique(seg_slice) if lbl > 0]
        alpha = 0.45
        for idx, lbl in enumerate(unique_labels):
            lbl_mask = seg_slice == lbl
            color = PALETTE[idx % len(PALETTE)]
            rgb[lbl_mask] = (1.0 - alpha) * rgb[lbl_mask] + alpha * color

        # Contorno delimitador nitido
        dilated = ndi.binary_dilation(mask_binary, iterations=1)
        edge = dilated ^ mask_binary
        rgb[edge] = [100.0, 255.0, 200.0]
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    else:
        rgb = np.full((h, w, 3), (18, 19, 22), dtype=np.uint8)
        unique_labels = [lbl for lbl in np.unique(seg_slice) if lbl > 0]
        for idx, lbl in enumerate(unique_labels):
            lbl_mask = seg_slice == lbl
            color = PALETTE[idx % len(PALETTE)].astype(np.uint8)
            rgb[lbl_mask] = color

        dilated = ndi.binary_dilation(mask_binary, iterations=1)
        edge = dilated ^ mask_binary
        rgb[edge] = [120, 255, 210]

    # Encuadre centrado con margen proporcional
    ys, xs = np.where(mask_binary)
    if len(ys) > 0:
        cy = int((ys.min() + ys.max()) // 2)
        cx = int((xs.min() + xs.max()) // 2)
        box_h = int(ys.max() - ys.min() + 1)
        box_w = int(xs.max() - xs.min() + 1)
        side = int(max(box_h, box_w) * 1.30)
        side = max(side, 32)

        y0 = cy - side // 2
        y1 = y0 + side
        x0 = cx - side // 2
        x1 = x0 + side

        # Relleno seguro si el encuadre excede los limites del corte
        pad_top = max(0, -y0)
        pad_bottom = max(0, y1 - h)
        pad_left = max(0, -x0)
        pad_right = max(0, x1 - w)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            bg_color = (0, 0, 0) if underlay_data is not None else (18, 19, 22)
            padded_rgb = np.full(
                (h + pad_top + pad_bottom, w + pad_left + pad_right, 3),
                bg_color,
                dtype=np.uint8,
            )
            padded_rgb[pad_top : pad_top + h, pad_left : pad_left + w] = rgb
            crop_y0 = y0 + pad_top
            crop_x0 = x0 + pad_left
            crop = padded_rgb[crop_y0 : crop_y0 + side, crop_x0 : crop_x0 + side]
        else:
            crop = rgb[y0:y1, x0:x1]
    else:
        crop = rgb

    pil_img = Image.fromarray(crop).resize(target_size, Image.Resampling.LANCZOS)
    pil_img.save(out_png, format="PNG")
    return out_png


def main():
    parser = argparse.ArgumentParser(
        description="Genera una miniatura 256x256 de una segmentacion medica NIfTI en su plano optimo."
    )
    parser.add_argument("input", help="Ruta al archivo de segmentacion NIfTI (.nii o .nii.gz)")
    parser.add_argument("-o", "--output", default=None, help="Ruta de salida de la imagen PNG")
    parser.add_argument("-u", "--underlay", default=None, help="Ruta opcional al volumen anatomico base")
    parser.add_argument("--size", type=int, default=256, help="Tamano cuadrado de salida en pixeles (por defecto: 256)")
    args = parser.parse_args()

    out_file = create_segmentation_thumbnail(
        seg_volume=args.input,
        output_path=args.output,
        underlay_volume=args.underlay,
        target_size=(args.size, args.size),
    )
    print(f"Miniatura generada exitosamente: {out_file}")


if __name__ == "__main__":
    main()
