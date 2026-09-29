# BraTS / Brain MRI Raw Dataset Directory

Place your raw Brain MRI NIfTI (.nii or .nii.gz) files in this directory.

## Option A: BraTS 2020 / 2021 Dataset (Recommended)
1. **Source**: BraTS Challenge on Kaggle or RSNA-MICCAI
   - Kaggle: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
   - Synapse: https://www.synapse.org/#!Synapse:syn25829420/wiki/610863
2. **Directory Structure**:
   Extract each patient as a folder inside `data/raw/`:
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

## Option B: Medical Segmentation Decathlon (Task01_BrainTumour)
1. **Source**: MSD Official Site (http://medicaldecathlon.com/)
2. **Directory Structure**:
   ```text
   data/raw/
   ├── imagesTr/
   │   ├── BRATS_001.nii.gz
   │   └── BRATS_002.nii.gz
   └── labelsTr/
       ├── BRATS_001.nii.gz
       └── BRATS_002.nii.gz
   ```

## Option C: Out-of-the-Box Synthetic BraTS Generation
If you want to test and verify the entire pipeline before downloading 10GB+ of real MRI scans:
```powershell
.\venv\Scripts\python.exe scripts/01_generate_synthetic_data.py --num_patients 6
```
This automatically produces 6 valid BraTS patients with full multi-modality volumes and ground truth masks in `data/raw/`.
