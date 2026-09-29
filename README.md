# FedMed: Cross-Silo Federated Learning Engine
## Week 1: 3D Brain MRI Dataset Collection, Preprocessing & PyTorch Pipeline

This repository contains the complete dataset collection, validation, preprocessing, and PyTorch DataLoader pipeline for **Week 1** of **FedMed**. The processed output is directly prepared to train a baseline **3D U-Net** architecture.

---

## 1. Project Directory Structure

```text
FedMed/
├── configs/
│   ├── __init__.py
│   └── dataset_config.py        # Centralized settings: target shape, modalities, normalization, splits
├── data/
│   ├── raw/                     # Raw NIfTI (.nii, .nii.gz) files (BraTS / Decathlon / Synthetic)
│   │   └── README.md            # Detailed instructions on dataset download and placement
│   ├── processed/               # Preprocessed compressed .npz archives per patient
│   │   └── preprocessing_metadata.json
│   ├── splits/                  # Patient-level leak-free split manifests
│   │   ├── train.json
│   │   ├── val.json
│   │   ├── test.json
│   │   └── splits_summary.json
│   └── visualizations/          # Generated slice and mask overlay preview images
├── scripts/
│   ├── 01_generate_synthetic_data.py # Rapid demo generator for instant testing
│   ├── 02_preprocess_dataset.py      # Core preprocessing & splitting pipeline runner
│   ├── 03_verify_dataset.py          # Validation checks & Week 1 report generator
│   └── 04_visualize_samples.py       # Generates MRI + mask overlay figures
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── dataset_reader.py    # Discovers, validates, and loads NIfTI volumes safely
│       ├── preprocessor.py      # Bounding box crop, z-score normalization, 3D nearest-neighbor mask resampling
│       ├── split_generator.py   # Patient-level leak-free partitioner (fixed seed 42)
│       ├── mri_dataset.py       # Memory-efficient PyTorch Dataset (lazy loading) & DataLoader
│       └── synthetic_sample.py  # Generates realistic synthetic BraTS NIfTI scans
├── requirements.txt             # Minimal, pinned dependencies
└── README.md                    # Complete documentation & commands
```

---

## 2. Dataset Acquisition & Placement

### Recommended Dataset: BraTS (Brain Tumor Segmentation Challenge)
- **Source**: MICCAI BraTS 2020 / BraTS 2021 on Kaggle or RSNA-MICCAI
  - Kaggle link: [BraTS 2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
  - Synapse link: [Synapse BraTS 2020 Challenge](https://www.synapse.org/#!Synapse:syn25829420/wiki/610863)
  - Alternative (Decathlon): [Medical Segmentation Decathlon Task01_BrainTumour](http://medicaldecathlon.com/)

### Required Modalities & Mask per Patient
Each patient scan requires 4 co-registered MRI sequences and 1 ground-truth segmentation mask:
1. `flair.nii.gz`: Fluid Attenuated Inversion Recovery (highlights peritumoral edema)
2. `t1.nii.gz`: T1-weighted native image (delineates anatomical structures)
3. `t1ce.nii.gz`: T1-weighted contrast-enhanced (highlights vascularized enhancing tumor rim)
4. `t2.nii.gz`: T2-weighted image (highlights fluid and non-enhancing tumor core)
5. `seg.nii.gz`: Ground-truth segmentation mask

### Placement in `data/raw/`
Extract patient folders directly into `data/raw/`:
```text
data/raw/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
```

---

## 3. Preprocessing Highlights

1. **Non-Zero Brain Bounding Box Cropping**: Removes unnecessary dark air background outside the skull across all modalities, reducing memory usage and computation.
2. **Consistent 3D Resampling**: Resizes volumes to target shape `(128, 128, 128)`. MRI image channels are resampled with trilinear interpolation; **segmentation masks are resampled strictly with nearest-neighbor interpolation** to preserve discrete class labels.
3. **Class Label Remapping**: Raw BraTS labels `{0, 1, 2, 4}` are standardized to `{0, 1, 2, 3}`:
   - `0`: Background
   - `1`: Necrotic / Non-Enhancing Core (NCR/NET)
   - `2`: Peritumoral Edema (ED)
   - `3`: Enhancing Tumor (ET)
4. **Intensity Normalization**: Channel-wise Z-score normalization computed strictly on non-zero brain voxels with `[0.5%, 99.5%]` percentile outlier clipping. Background voxels remain 0.
5. **Memory-Efficient PyTorch Dataset**: Preprocessed volumes are saved as individual compressed `.npz` files. `MRIDataset` performs lazy per-sample disk loading during PyTorch batch generation, preventing RAM exhaustion on large 3D medical datasets.
6. **Leak-Free Partitioning**: Patient-level splitting (`train.json`, `val.json`, `test.json`) with fixed random seed (`42`) guarantees zero slice or patient leakage across training, validation, and testing sets.

---

## 4. Windows VS Code Execution Commands

Open PowerShell in the `FedMed` project folder inside VS Code:

### Step 1: Create Virtual Environment (Python 3.12)
```powershell
py -3.12 -m venv venv
```

### Step 2: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*

### Step 3: Install Required Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: (Optional) Generate Synthetic Test Scans
*Skip this step if you have already extracted real BraTS data into `data/raw/`:*
```powershell
python scripts/01_generate_synthetic_data.py --num_patients 6
```

### Step 5: Run Dataset Preprocessing & Splitting
```powershell
python scripts/02_preprocess_dataset.py
```

### Step 6: Run Dataset Verification & DataLoader Tests
```powershell
python scripts/03_verify_dataset.py
```

### Step 7: Run Visualization Script
```powershell
python scripts/04_visualize_samples.py
```
Preview the saved visualization at: `data/visualizations/sample_slice_visualization.png`.
