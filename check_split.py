"""
Read-Only Split Verification Script
Checks the actual contents of train.json, val.json, and test.json.
Performs pairwise intersection checks, patient counts, and validation assertions.
DOES NOT MODIFY ANY FILES.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def main():
    splits_dir = Path("data/splits")
    train_file = splits_dir / "train.json"
    val_file = splits_dir / "val.json"
    test_file = splits_dir / "test.json"

    # Read train.json
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    train_patients = train_data.get("patient_ids", [])

    # Read val.json
    with open(val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    val_patients = val_data.get("patient_ids", [])

    # Read test.json
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_patients = test_data.get("patient_ids", [])

    # Sets for overlap verification
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    all_patients = train_patients + val_patients + test_patients
    unique_patients = set(all_patients)

    # Overlap computations
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    has_overlap = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    # Verification checks
    counts_valid = (len(train_patients) == 4 and len(val_patients) == 1 and len(test_patients) == 1)
    totals_valid = (len(all_patients) == 6 and len(unique_patients) == 6)
    splits_valid = counts_valid and totals_valid and (not has_overlap)

    # Output formatted report
    print("=======================================================")
    print(" FedMed: Train/Validation/Test Split Verification")
    print("=======================================================\n")

    print("TRAIN PATIENTS:")
    print(train_patients)
    print()

    print("VALIDATION PATIENTS:")
    print(val_patients)
    print()

    print("TEST PATIENTS:")
    print(test_patients)
    print()

    print("-------------------------------------------------------")
    print("Patient Counts")
    print("-------------------------------------------------------")
    print(f"Training:   {len(train_patients)}")
    print(f"Validation: {len(val_patients)}")
    print(f"Test:       {len(test_patients)}")
    print(f"Total:      {len(all_patients)}")
    print(f"Unique:     {len(unique_patients)}")
    print()

    print("-------------------------------------------------------")
    print("Overlap Check")
    print("-------------------------------------------------------")
    print(f"Train ∩ Validation: {train_val_overlap}")
    print(f"Train ∩ Test:        {train_test_overlap}")
    print(f"Validation ∩ Test:   {val_test_overlap}")
    print()

    print("-------------------------------------------------------")
    print("FINAL RESULT")
    print("-------------------------------------------------------")
    print(f"Patient overlap: {'FOUND' if has_overlap else 'NONE'}")
    print(f"Split verification: {'PASSED' if splits_valid else 'FAILED'}")
    print(f"Dataset split is valid: {'YES' if splits_valid else 'NO'}")

if __name__ == "__main__":
    main()
