"""
Dataset Reader Module
---------------------
Discovers, validates, and safely loads 3D NIfTI (.nii, .nii.gz) MRI volumes
and corresponding ground-truth segmentation masks.
Supports standard BraTS multi-file format and Medical Decathlon 4D format.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import nibabel as nib
import numpy as np


@dataclass
class PatientScan:
    """Represents a validated patient scan with paths to its modalities and mask."""
    patient_id: str
    modality_paths: Dict[str, Path]  # e.g., {'flair': path, 't1': path, 't1ce': path, 't2': path}
    mask_path: Path
    is_4d_combined: bool = False
    original_shape: Optional[Tuple[int, ...]] = None


class DatasetReader:
    """
    Scans a raw data directory, discovers patient volumes, checks correspondence,
    and isolates valid samples from missing/corrupted files safely.
    """

    EXPECTED_MODALITIES = ["flair", "t1", "t1ce", "t2"]

    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)

    def discover_patients(self) -> Tuple[List[PatientScan], List[Dict[str, Any]]]:
        """
        Scans raw_data_dir for patient records.
        Returns:
            valid_patients: list of verified PatientScan objects
            invalid_patients: list of dicts with patient_id, reason, and missing/corrupted details
        """
        valid_patients: List[PatientScan] = []
        invalid_patients: List[Dict[str, Any]] = []

        if not self.raw_data_dir.exists():
            return valid_patients, [{"error": f"Raw data directory does not exist: {self.raw_data_dir}"}]

        # Check if Medical Decathlon format (imagesTr & labelsTr)
        msd_images = self.raw_data_dir / "imagesTr"
        msd_labels = self.raw_data_dir / "labelsTr"
        if msd_images.is_dir() and msd_labels.is_dir():
            return self._discover_msd_format(msd_images, msd_labels)

        # Standard BraTS format: patient directories (e.g. BraTS20_Training_001/)
        patient_dirs = [d for d in self.raw_data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

        # If no subdirectories found, also check for flat structure
        if not patient_dirs:
            return self._discover_flat_format()

        for p_dir in sorted(patient_dirs):
            patient_id = p_dir.name
            modality_files: Dict[str, Path] = {}
            mask_file: Optional[Path] = None

            # Collect .nii and .nii.gz files in patient folder
            nii_files = list(p_dir.glob("*.nii*"))

            for file_path in nii_files:
                fname_lower = file_path.name.lower()
                # Remove extension for matching
                clean_name = fname_lower.replace(".nii.gz", "").replace(".nii", "")

                if "seg" in clean_name or "mask" in clean_name:
                    mask_file = file_path
                elif "flair" in clean_name:
                    modality_files["flair"] = file_path
                elif "t1ce" in clean_name or "t1gd" in clean_name:
                    modality_files["t1ce"] = file_path
                elif "t1" in clean_name and "t1ce" not in clean_name and "t1gd" not in clean_name:
                    modality_files["t1"] = file_path
                elif "t2" in clean_name:
                    modality_files["t2"] = file_path

            # Validation checks
            missing_modalities = [m for m in self.EXPECTED_MODALITIES if m not in modality_files]
            if missing_modalities or mask_file is None:
                invalid_patients.append({
                    "patient_id": patient_id,
                    "reason": "Missing required files",
                    "missing_modalities": missing_modalities,
                    "mask_found": mask_file is not None
                })
                continue

            # Verify file integrity and dimension match
            is_valid, validation_info = self._verify_patient_files(patient_id, modality_files, mask_file)
            if is_valid:
                valid_patients.append(PatientScan(
                    patient_id=patient_id,
                    modality_paths=modality_files,
                    mask_path=mask_file,
                    is_4d_combined=False,
                    original_shape=validation_info.get("shape")
                ))
            else:
                invalid_patients.append(validation_info)

        return valid_patients, invalid_patients

    def _discover_msd_format(self, images_dir: Path, labels_dir: Path) -> Tuple[List[PatientScan], List[Dict[str, Any]]]:
        """Handles Medical Segmentation Decathlon (MSD) Task01_BrainTumour 4D format."""
        valid_patients: List[PatientScan] = []
        invalid_patients: List[Dict[str, Any]] = []

        image_files = sorted(list(images_dir.glob("*.nii*")))
        for img_path in image_files:
            if img_path.name.startswith("."):
                continue
            patient_id = img_path.name.split(".")[0]
            # Expected label path
            label_name = img_path.name
            mask_path = labels_dir / label_name

            if not mask_path.exists():
                invalid_patients.append({
                    "patient_id": patient_id,
                    "reason": "Missing segmentation mask in labelsTr",
                    "mask_path": str(mask_path)
                })
                continue

            try:
                img_nii = nib.load(str(img_path))
                mask_nii = nib.load(str(mask_path))
                img_shape = img_nii.shape
                mask_shape = mask_nii.shape

                # 4D image has shape (H, W, D, 4) or (4, H, W, D)
                spatial_shape = img_shape[:3] if len(img_shape) == 4 and img_shape[3] == 4 else img_shape[1:]
                if spatial_shape != mask_shape:
                    invalid_patients.append({
                        "patient_id": patient_id,
                        "reason": f"Shape mismatch: Image {img_shape} vs Mask {mask_shape}"
                    })
                    continue

                valid_patients.append(PatientScan(
                    patient_id=patient_id,
                    modality_paths={"combined_4d": img_path},
                    mask_path=mask_path,
                    is_4d_combined=True,
                    original_shape=spatial_shape
                ))
            except Exception as e:
                invalid_patients.append({
                    "patient_id": patient_id,
                    "reason": f"Corrupted NIfTI file: {str(e)}"
                })

        return valid_patients, invalid_patients

    def _discover_flat_format(self) -> Tuple[List[PatientScan], List[Dict[str, Any]]]:
        """Handles single directory containing all files with naming pattern `<patient_id>_<modality>.nii.gz`."""
        valid_patients: List[PatientScan] = []
        invalid_patients: List[Dict[str, Any]] = []

        # Group by common prefix
        all_nii = list(self.raw_data_dir.glob("*.nii*"))
        if not all_nii:
            return valid_patients, [{"error": f"No NIfTI (.nii, .nii.gz) files found in {self.raw_data_dir}"}]

        groups: Dict[str, Dict[str, Path]] = {}
        for f in all_nii:
            fname = f.name.replace(".nii.gz", "").replace(".nii", "")
            for mod in self.EXPECTED_MODALITIES + ["seg", "mask"]:
                if f"_{mod}" in fname.lower():
                    pid = fname.lower().split(f"_{mod}")[0]
                    if pid not in groups:
                        groups[pid] = {}
                    groups[pid][mod] = f
                    break

        for pid, files in groups.items():
            mask_file = files.get("seg") or files.get("mask")
            mod_files = {m: files[m] for m in self.EXPECTED_MODALITIES if m in files}
            missing = [m for m in self.EXPECTED_MODALITIES if m not in mod_files]

            if missing or mask_file is None:
                invalid_patients.append({
                    "patient_id": pid,
                    "reason": "Missing required modalities or mask",
                    "missing": missing,
                    "mask_found": mask_file is not None
                })
                continue

            is_valid, validation_info = self._verify_patient_files(pid, mod_files, mask_file)
            if is_valid:
                valid_patients.append(PatientScan(
                    patient_id=pid,
                    modality_paths=mod_files,
                    mask_path=mask_file,
                    is_4d_combined=False,
                    original_shape=validation_info.get("shape")
                ))
            else:
                invalid_patients.append(validation_info)

        return valid_patients, invalid_patients

    def _verify_patient_files(
        self,
        patient_id: str,
        modality_files: Dict[str, Path],
        mask_file: Path
    ) -> Tuple[bool, Dict[str, Any]]:
        """Safely verifies file readability, non-empty header, and spatial dimension consistency."""
        try:
            # Check mask
            mask_nii = nib.load(str(mask_file))
            mask_shape = mask_nii.shape
            if len(mask_shape) != 3:
                return False, {
                    "patient_id": patient_id,
                    "reason": f"Mask is not 3D. Shape: {mask_shape}"
                }

            # Check each modality
            for mod_name in self.EXPECTED_MODALITIES:
                mod_path = modality_files[mod_name]
                mod_nii = nib.load(str(mod_path))
                mod_shape = mod_nii.shape

                if mod_shape != mask_shape:
                    return False, {
                        "patient_id": patient_id,
                        "reason": f"Dimension mismatch: {mod_name} shape {mod_shape} != mask shape {mask_shape}"
                    }

            return True, {
                "patient_id": patient_id,
                "shape": mask_shape
            }
        except Exception as e:
            return False, {
                "patient_id": patient_id,
                "reason": f"Corrupted NIfTI file: {str(e)}"
            }

    @classmethod
    def load_patient_raw_arrays(cls, patient: PatientScan) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the 4 MRI modalities and segmentation mask into numpy arrays.
        Output:
            image_array: shape (4, D, H, W), float32
            mask_array: shape (D, H, W), int32
        """
        mask_nii = nib.load(str(patient.mask_path))
        mask_arr = np.asanyarray(mask_nii.dataobj).astype(np.int32)

        if patient.is_4d_combined:
            # 4D combined file
            img_nii = nib.load(str(patient.modality_paths["combined_4d"]))
            img_arr = np.asanyarray(img_nii.dataobj).astype(np.float32)
            if img_arr.shape[-1] == 4:
                # Shape: (H, W, D, 4) -> (4, H, W, D)
                img_arr = np.moveaxis(img_arr, -1, 0)
        else:
            # Load 4 separate modalities in fixed order: flair, t1, t1ce, t2
            channel_arrays = []
            for mod in cls.EXPECTED_MODALITIES:
                nii = nib.load(str(patient.modality_paths[mod]))
                arr = np.asanyarray(nii.dataobj).astype(np.float32)
                channel_arrays.append(arr)
            img_arr = np.stack(channel_arrays, axis=0)  # (4, H, W, D)

        # Standardize axis order to (C, D, H, W) where D is depth (axial slices)
        # In NIfTI, dataobj is typically (X, Y, Z) = (Width, Height, Depth)
        # We reorder so spatial axes are (D, H, W):
        # image: (4, X, Y, Z) -> (4, Z, Y, X)
        # mask: (X, Y, Z) -> (Z, Y, X)
        img_arr = np.transpose(img_arr, (0, 3, 2, 1))
        mask_arr = np.transpose(mask_arr, (2, 1, 0))

        return img_arr, mask_arr
