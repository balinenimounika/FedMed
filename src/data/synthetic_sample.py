"""
Synthetic Brain MRI Generator Module
------------------------------------
Generates realistic synthetic 3D BraTS-compatible multi-modal volumes (FLAIR, T1, T1ce, T2)
and ground-truth segmentation masks (NCR/NET, ED, ET) saved as valid NIfTI (.nii.gz) files.
Enables immediate testing and verification without requiring immediate multi-gigabyte downloads.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import nibabel as nib


def generate_synthetic_patient(
    patient_id: str,
    output_dir: Path,
    shape: Tuple[int, int, int] = (155, 240, 240),
    seed: int = 42
) -> Path:
    """
    Creates a synthetic patient folder with 4 MRI modalities and 1 segmentation mask:
      - <patient_id>_flair.nii.gz
      - <patient_id>_t1.nii.gz
      - <patient_id>_t1ce.nii.gz
      - <patient_id>_t2.nii.gz
      - <patient_id>_seg.nii.gz
    
    Data characteristics:
      - Realistic brain ellipsoid inside empty zero air background
      - Anatomical distinction (ventricles, white/gray matter variations)
      - Focal multi-compartment brain tumor:
        - Label 1: Necrotic tumor core
        - Label 2: Peritumoral edema ring
        - Label 4: Enhancing tumor rim
    """
    rng = np.random.RandomState(seed)
    p_dir = output_dir / patient_id
    p_dir.mkdir(parents=True, exist_ok=True)

    d, h, w = shape  # Typically 155, 240, 240 in BraTS
    zc, yc, xc = d // 2, h // 2, w // 2

    # Coordinate grids
    z, y, x = np.ogrid[:d, :h, :w]

    # Brain ellipsoid mask
    rad_z, rad_y, rad_x = d * 0.36, h * 0.34, w * 0.30
    brain_dist = ((z - zc) / rad_z)**2 + ((y - yc) / rad_y)**2 + ((x - xc) / rad_x)**2
    brain_mask = brain_dist <= 1.0

    # Internal ventricles (fluid - dark on T1, bright on T2)
    vent_dist = ((z - zc) / (rad_z * 0.35))**2 + ((y - yc) / (rad_y * 0.2))**2 + ((x - xc) / (rad_x * 0.15))**2
    ventricles_mask = (vent_dist <= 1.0) & brain_mask

    # Tumor center with random perturbation
    tz = zc + int(rng.uniform(-0.15, 0.15) * d)
    ty = yc + int(rng.uniform(-0.12, 0.12) * h)
    tx = xc + int(rng.uniform(-0.12, 0.12) * w)

    tumor_dist = np.sqrt((z - tz)**2 + (y - ty)**2 + (x - tx)**2)

    # Multi-compartment tumor radii
    edema_radius = rng.uniform(16, 22)
    enhancing_radius = rng.uniform(10, 15)
    core_radius = rng.uniform(4, 8)

    edema_mask = (tumor_dist <= edema_radius) & brain_mask
    enhancing_mask = (tumor_dist <= enhancing_radius) & (tumor_dist > core_radius) & brain_mask
    core_mask = (tumor_dist <= core_radius) & brain_mask

    # Segmentation labels: 0=background, 1=core, 2=edema, 4=enhancing tumor (BraTS format)
    seg = np.zeros(shape, dtype=np.uint8)
    seg[edema_mask] = 2
    seg[enhancing_mask] = 4
    seg[core_mask] = 1

    # Base brain tissue intensities
    base_brain = np.zeros(shape, dtype=np.float32)
    base_brain[brain_mask] = rng.normal(loc=120.0, scale=15.0, size=np.sum(brain_mask))

    # Modality 1: T1 (Fat bright, CSF/ventricles dark, tumor hypo-intense)
    t1 = base_brain.copy()
    t1[ventricles_mask] = 30.0 + rng.normal(0, 5, np.sum(ventricles_mask))
    t1[edema_mask] = 70.0 + rng.normal(0, 8, np.sum(edema_mask))
    t1[core_mask] = 40.0 + rng.normal(0, 5, np.sum(core_mask))
    t1 = np.clip(t1, 0, None)

    # Modality 2: T1ce (T1 with contrast: Enhancing tumor rim glows very bright)
    t1ce = t1.copy()
    t1ce[enhancing_mask] = 220.0 + rng.normal(0, 20, np.sum(enhancing_mask))

    # Modality 3: T2 (Water/fluid/edema bright, ventricles very bright)
    t2 = base_brain.copy()
    t2[brain_mask] = 100.0 + rng.normal(0, 12, np.sum(brain_mask))
    t2[ventricles_mask] = 240.0 + rng.normal(0, 15, np.sum(ventricles_mask))
    t2[edema_mask] = 210.0 + rng.normal(0, 18, np.sum(edema_mask))
    t2[enhancing_mask] = 170.0 + rng.normal(0, 15, np.sum(enhancing_mask))
    t2[core_mask] = 80.0 + rng.normal(0, 10, np.sum(core_mask))

    # Modality 4: FLAIR (Ventricles/fluid suppressed/dark, edema and tumor hyper-intense)
    flair = base_brain.copy()
    flair[brain_mask] = 90.0 + rng.normal(0, 10, np.sum(brain_mask))
    flair[ventricles_mask] = 25.0 + rng.normal(0, 4, np.sum(ventricles_mask))  # Attenuated fluid
    flair[edema_mask] = 230.0 + rng.normal(0, 20, np.sum(edema_mask))          # Bright edema
    flair[enhancing_mask] = 190.0 + rng.normal(0, 15, np.sum(enhancing_mask))
    flair[core_mask] = 60.0 + rng.normal(0, 8, np.sum(core_mask))

    # NIfTI affine matrix (1mm isotropic standard orientation)
    affine = np.eye(4, dtype=np.float32)

    # Transpose arrays from (Z, Y, X) to (X, Y, Z) for standard NIfTI saving
    nib.save(nib.Nifti1Image(np.transpose(t1, (2, 1, 0)), affine), str(p_dir / f"{patient_id}_t1.nii.gz"))
    nib.save(nib.Nifti1Image(np.transpose(t1ce, (2, 1, 0)), affine), str(p_dir / f"{patient_id}_t1ce.nii.gz"))
    nib.save(nib.Nifti1Image(np.transpose(t2, (2, 1, 0)), affine), str(p_dir / f"{patient_id}_t2.nii.gz"))
    nib.save(nib.Nifti1Image(np.transpose(flair, (2, 1, 0)), affine), str(p_dir / f"{patient_id}_flair.nii.gz"))
    nib.save(nib.Nifti1Image(np.transpose(seg, (2, 1, 0)), affine), str(p_dir / f"{patient_id}_seg.nii.gz"))

    return p_dir
