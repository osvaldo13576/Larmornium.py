from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ShiftAlignmentResult:
    common_length: int
    slice_ranges: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    shifts: Dict[int, int] = field(default_factory=dict)
    correlations: Dict[int, float] = field(default_factory=dict)
    omit_start: Dict[int, int] = field(default_factory=dict)
    omit_end: Dict[int, int] = field(default_factory=dict)
    profiles: Dict[int, np.ndarray] = field(default_factory=dict)
    summary: str = ""

    def get_range_for_study(self, study_idx: int) -> Optional[Tuple[int, int, int]]:
        return self.slice_ranges.get(study_idx)


def extract_slice_profile(volume: Optional[np.ndarray], modality: str = "CT") -> np.ndarray:
    if volume is None or volume.size == 0 or volume.ndim < 3:
        return np.zeros(0, dtype=np.float32)

    num_slices = volume.shape[0]
    profile = np.zeros(num_slices, dtype=np.float32)
    mod = str(modality or "CT").upper()

    is_ct = mod in ("CT", "FUSION")

    for z in range(num_slices):
        slice_z = volume[z]
        if slice_z.size == 0:
            profile[z] = 0.0
            continue

        if is_ct:
            # En CT, valores > -500 HU representan tejido o fantoma frente a aire exterior (-1000 HU)
            fg_mask = slice_z > -500.0
            fg_count = np.count_nonzero(fg_mask)
            if fg_count > 0:
                fg_mean = float(np.mean(slice_z[fg_mask]))
                # Masa ponderada: número de vóxeles * intensidad por encima de aire base (-1000)
                profile[z] = float(fg_count) * (fg_mean + 1000.0)
            else:
                profile[z] = float(np.mean(slice_z))
        else:
            # PET / MRI
            vmax = float(np.nanmax(volume)) if volume.size > 0 else 0.0
            thresh = 0.05 * vmax if vmax > 0 else 0.0
            fg_mask = slice_z > thresh
            fg_count = np.count_nonzero(fg_mask)
            if fg_count > 0:
                fg_mean = float(np.mean(slice_z[fg_mask]))
                profile[z] = float(fg_count) * fg_mean
            else:
                profile[z] = float(np.mean(slice_z))

    # Normalización min-max a [0.0, 1.0]
    p_min = float(np.min(profile))
    p_max = float(np.max(profile))
    if p_max - p_min > 1e-6:
        profile = (profile - p_min) / (p_max - p_min)
    elif p_max > 1e-6:
        profile = profile / p_max

    return profile.astype(np.float32)


def compute_relative_shift(
    profile_ref: np.ndarray,
    profile_target: np.ndarray,
    max_shift: int = 25,
) -> Tuple[int, float]:
    z_ref = len(profile_ref)
    z_tgt = len(profile_target)

    if z_ref == 0 or z_tgt == 0:
        return 0, 0.0

    # Limitar el rango de búsqueda para mantener al menos una solapa razonable
    min_len = min(z_ref, z_tgt)
    max_s = min(max_shift, max(1, min_len // 2))

    best_shift = 0
    best_corr = -1.0

    for s in range(-max_s, max_s + 1):
        ref_start = max(0, -s)
        ref_end = min(z_ref, z_tgt - s)
        tgt_start = ref_start + s
        tgt_end = ref_end + s

        overlap_len = ref_end - ref_start
        if overlap_len < 10 or overlap_len < 0.25 * min_len:
            continue

        sub_ref = profile_ref[ref_start:ref_end]
        sub_tgt = profile_target[tgt_start:tgt_end]

        ref_c = sub_ref - np.mean(sub_ref)
        tgt_c = sub_tgt - np.mean(sub_tgt)

        norm_ref = np.linalg.norm(ref_c)
        norm_tgt = np.linalg.norm(tgt_c)

        if norm_ref > 1e-6 and norm_tgt > 1e-6:
            corr = float(np.dot(ref_c, tgt_c) / (norm_ref * norm_tgt))
        else:
            # Si uno de los perfiles es plano, medir diferencia absoluta media
            mae = float(np.mean(np.abs(sub_ref - sub_tgt)))
            corr = max(0.0, 1.0 - mae)

        if corr > best_corr:
            best_corr = corr
            best_shift = s

    if best_corr < 0.0:
        best_shift = 0
        best_corr = 1.0

    return best_shift, best_corr


def align_volumes_by_shift(
    loaded_studies: List[Dict[str, Any]],
    reference_idx: int = 0,
    max_shift: int = 25,
) -> ShiftAlignmentResult:
    if not loaded_studies:
        return ShiftAlignmentResult(common_length=0, summary="No hay estudios cargados.")

    n = len(loaded_studies)
    ref_i = max(0, min(reference_idx, n - 1))

    # Extraer perfiles y longitudes Z
    profiles: List[np.ndarray] = []
    slice_counts: List[int] = []

    for i, study in enumerate(loaded_studies):
        modality = str(
            study.get("modality")
            or (study.get("info") or study.get("item_info") or {}).get("modality")
            or ""
        ).upper()
        vol_data = study.get("volume_data") if isinstance(study.get("volume_data"), dict) else study

        # En caso de fusión, se toma el volumen de CT
        if modality == "FUSION":
            vol = vol_data.get("ct_volume")
            if vol is None:
                vol = vol_data.get("volume")
            prof_mod = "CT"
        elif modality in ("PT", "PET"):
            vol = vol_data.get("volume")
            if vol is None:
                vol = vol_data.get("pet_volume")
            prof_mod = "PET"
        elif modality in ("MR", "MRI"):
            vol = vol_data.get("volume")
            prof_mod = "MRI"
        else:
            vol = vol_data.get("volume")
            if vol is None:
                vol = vol_data.get("ct_volume")
            prof_mod = "CT"

        if vol is not None and vol.ndim >= 3:
            prof = extract_slice_profile(vol, modality=prof_mod)
            count = vol.shape[0]
        else:
            prof = np.zeros(0, dtype=np.float32)
            count = 0

        profiles.append(prof)
        slice_counts.append(count)

    ref_profile = profiles[ref_i]

    # Calcular desplazamientos relativos respecto a la referencia
    shifts: Dict[int, int] = {}
    correlations: Dict[int, float] = {}

    for i in range(n):
        if i == ref_i:
            shifts[i] = 0
            correlations[i] = 1.0
        else:
            tgt_profile = profiles[i]
            if len(ref_profile) > 0 and len(tgt_profile) > 0:
                s, c = compute_relative_shift(ref_profile, tgt_profile, max_shift=max_shift)
                shifts[i] = s
                correlations[i] = c
            else:
                shifts[i] = 0
                correlations[i] = 0.0

    # Calcular ventana común y cortes a omitir
    # La coordenada de alineación común satisface: 0 <= z_k < Z_k, donde z_k = z_aligned + s_k
    min_shift = min(shifts.values()) if shifts else 0
    z_aligned_min = -min_shift

    upper_bounds = [
        slice_counts[i] - shifts[i] - 1
        for i in range(n)
        if slice_counts[i] > 0
    ]
    z_aligned_max = min(upper_bounds) if upper_bounds else 0

    common_length = max(0, z_aligned_max - z_aligned_min + 1)

    slice_ranges: Dict[int, Tuple[int, int, int]] = {}
    omit_start: Dict[int, int] = {}
    omit_end: Dict[int, int] = {}

    summary_lines = [
        f"Alineación Shift calculada para {n} estudio{'s' if n != 1 else ''}:",
        f"  - Longitud común sincronizada: {common_length} cortes",
    ]

    for i in range(n):
        total_z = slice_counts[i]
        if total_z == 0 or common_length == 0:
            slice_ranges[i] = (0, 0, 0)
            omit_start[i] = 0
            omit_end[i] = 0
            continue

        start_idx = z_aligned_min + shifts[i]
        end_idx = start_idx + common_length - 1
        om_start = start_idx
        om_end = max(0, total_z - 1 - end_idx)

        slice_ranges[i] = (start_idx, end_idx, common_length)
        omit_start[i] = om_start
        omit_end[i] = om_end

        study_name = loaded_studies[i].get("patient_name") or f"Estudio {i + 1}"
        desc = loaded_studies[i].get("description") or ""
        mod = loaded_studies[i].get("modality") or ""
        corr_str = f"{correlations[i]:.3f}" if i != ref_i else "1.000 (Ref)"
        shift_sign = f"+{shifts[i]}" if shifts[i] > 0 else f"{shifts[i]}"

        summary_lines.append(
            f"  [{i + 1}] {study_name} ({desc}) [{mod}]: "
            f"Shift={shift_sign} | Corr={corr_str} | "
            f"Cortes [{start_idx + 1}..{end_idx + 1}]/{total_z} "
            f"(Omitidos: {om_start} iniciales, {om_end} finales)"
        )

    profiles_dict = {i: profiles[i] for i in range(n)}

    return ShiftAlignmentResult(
        common_length=common_length,
        slice_ranges=slice_ranges,
        shifts=shifts,
        correlations=correlations,
        omit_start=omit_start,
        omit_end=omit_end,
        profiles=profiles_dict,
        summary="\n".join(summary_lines),
    )
