"""
Script 02: Preprocess Dataset
-----------------------------
Discovers raw 3D MRI scans, validates correspondence, crops non-zero brain,
normalizes voxel intensities, resizes to target shape, re-maps labels, saves compressed
volumes to data/processed/, and generates leak-free train/val/test splits.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from configs.dataset_config import DEFAULT_CONFIG
from src.data.dataset_reader import DatasetReader
from src.data.preprocessor import VolumePreprocessor
from src.data.split_generator import SplitGenerator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"--- {desc} ---")
        return iterable


def main():
    config = DEFAULT_CONFIG
    raw_dir = Path(config.raw_data_dir)
    processed_dir = Path(config.processed_data_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print(" FedMed: Week 1 - 3D Brain MRI Preprocessing Pipeline")
    print(f" Raw Directory:        {raw_dir}")
    print(f" Processed Directory:  {processed_dir}")
    print(f" Target Shape:         {config.target_shape}")
    print(f" Normalization:        {config.norm_method}")
    print(f" Bounding Box Crop:    {config.crop_to_nonzero_brain}")
    print(f" Label Remapping:      {config.remap_labels} (BraTS 4 -> 3)")
    print("=======================================================\n")

    # Step 1: Discover and validate raw patient scans
    print("[1/4] Scanning and validating raw dataset files...")
    reader = DatasetReader(raw_dir)
    valid_patients, invalid_patients = reader.discover_patients()

    print(f"  -> Discovered total patients/records: {len(valid_patients) + len(invalid_patients)}")
    print(f"  -> Valid patients:                   {len(valid_patients)}")
    print(f"  -> Invalid/Incomplete patients:      {len(invalid_patients)}")

    if invalid_patients:
        print("\n[Notice] Details on invalid/missing records:")
        for inv in invalid_patients:
            print(f"  * Patient '{inv.get('patient_id', 'unknown')}': {inv.get('reason', 'unspecified error')}")

    if len(valid_patients) == 0:
        print("\n[ERROR] No valid patient scans found in raw directory!")
        print(f"Please place BraTS dataset in: {raw_dir}")
        print("Or generate synthetic test scans using: python scripts/01_generate_synthetic_data.py")
        sys.exit(1)

    # Step 2: Preprocess each valid patient
    print("\n[2/4] Executing 3D Preprocessing Pipeline...")
    preprocessor = VolumePreprocessor(config)
    dataset_metadata = {}

    for patient in tqdm(valid_patients, desc="Preprocessing Volumes"):
        pid = patient.patient_id
        # Safe raw loading
        try:
            image_raw, mask_raw = DatasetReader.load_patient_raw_arrays(patient)
            processed_img, processed_mask, meta = preprocessor.process_patient(image_raw, mask_raw)

            # Save as compressed .npz
            out_file = processed_dir / f"{pid}.npz"
            np.savez_compressed(
                out_file,
                image=processed_img,
                mask=processed_mask
            )

            meta["processed_file"] = str(out_file.relative_to(config.processed_data_dir.parent))
            dataset_metadata[pid] = meta

        except Exception as e:
            print(f"\n[Warning] Failed to process patient {pid}: {str(e)}")

    # Save summary metadata JSON
    meta_json_path = processed_dir / "preprocessing_metadata.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        # Convert any tuples/arrays to serializable types
        serializable_meta = {}
        for pid, m in dataset_metadata.items():
            serializable_meta[pid] = {
                k: list(v) if isinstance(v, (tuple, list, np.ndarray)) else v
                for k, v in m.items()
            }
        json.dump(serializable_meta, f, indent=2)

    print(f"\n  -> Processed {len(dataset_metadata)} patients successfully.")
    print(f"  -> Saved metadata summary to: {meta_json_path}")

    # Step 3: Generate leak-free train/val/test splits
    print("\n[3/4] Generating patient-level Train / Val / Test splits (Zero Leakage)...")
    split_gen = SplitGenerator(config)
    splits = split_gen.create_splits(list(dataset_metadata.keys()))

    print(f"  -> Training Patients:   {len(splits['train'])} ({splits['train']})")
    print(f"  -> Validation Patients: {len(splits['val'])} ({splits['val']})")
    print(f"  -> Test Patients:       {len(splits['test'])} ({splits['test']})")
    print(f"  -> Split manifests saved to: {config.splits_dir}")

    # Step 4: Split integrity check
    print("\n[4/4] Verifying split isolation...")
    SplitGenerator.verify_no_leakage(splits)
    print("  -> [PASS] Verified zero patient overlap between splits.")

    print("\n=======================================================")
    print(" [SUCCESS] Preprocessing completed successfully!")
    print(" Next Step: Run 'python scripts/03_verify_dataset.py' to generate verification report.")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
