"""
Script 04: Visualize Preprocessed Brain MRI & Segmentation Mask
--------------------------------------------------------------
Generates a multi-panel visualization showing:
1. 2D axial slices across modalities (FLAIR, T1, T1ce, T2)
2. Corresponding ground-truth segmentation mask
3. High-contrast MRI + Segmentation mask overlay
Saves the visualization to data/visualizations/sample_slice_visualization.png.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend safe for terminals and remote servers
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from configs.dataset_config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Visualize preprocessed 3D MRI volume slice and mask overlay.")
    parser.add_argument("--patient_id", type=str, default=None, help="Patient ID to visualize (default: first available)")
    parser.add_argument("--slice_idx", type=int, default=None, help="Axial slice index (default: slice with maximum tumor)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    processed_dir = Path(config.processed_data_dir)
    vis_dir = Path(config.visualizations_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(list(processed_dir.glob("*.npz")))
    if not npz_files:
        print(f"[ERROR] No preprocessed files found in {processed_dir}. Run scripts/02_preprocess_dataset.py first.")
        sys.exit(1)

    if args.patient_id is not None:
        target_file = processed_dir / f"{args.patient_id}.npz"
        if not target_file.exists():
            print(f"[ERROR] File for patient '{args.patient_id}' not found at {target_file}")
            sys.exit(1)
        patient_file = target_file
    else:
        patient_file = npz_files[0]

    patient_id = patient_file.stem
    print(f"\n[Visualization] Loading patient: {patient_id} ...")

    with np.load(patient_file) as data:
        image = data["image"]  # (4, D, H, W)
        mask = data["mask"]    # (D, H, W)

    num_channels, depth, height, width = image.shape

    # Determine slice with maximum tumor burden
    if args.slice_idx is not None:
        slice_idx = max(0, min(args.slice_idx, depth - 1))
    else:
        # Sum non-zero mask voxels across each depth slice
        tumor_per_slice = np.sum(mask > 0, axis=(1, 2))
        if np.max(tumor_per_slice) > 0:
            slice_idx = int(np.argmax(tumor_per_slice))
        else:
            slice_idx = depth // 2

    print(f"  -> Visualizing axial slice {slice_idx}/{depth} (Maximum tumor cross-section)")

    # Modalities: [flair, t1, t1ce, t2]
    flair_slice = image[0, slice_idx, :, :]
    t1_slice = image[1, slice_idx, :, :]
    t1ce_slice = image[2, slice_idx, :, :]
    t2_slice = image[3, slice_idx, :, :]
    mask_slice = mask[slice_idx, :, :]

    # Setup color palette for segmentation mask
    # 0: background (transparent), 1: Core (red), 2: Edema (green), 3: Enhancing (yellow)
    colors = [
        (0.0, 0.0, 0.0, 0.0),    # 0: transparent
        (0.9, 0.2, 0.2, 0.85),   # 1: Red (NCR/NET)
        (0.2, 0.8, 0.2, 0.85),   # 2: Green (ED)
        (1.0, 0.85, 0.0, 0.85),  # 3: Yellow (ET)
    ]
    custom_cmap = ListedColormap(colors)

    # Plot 6 panels: FLAIR, T1, T1ce, T2, Ground-truth Mask, MRI + Mask Overlay
    fig, axes = plt.subplots(1, 6, figsize=(22, 4.2), dpi=150)
    plt.subplots_adjust(wspace=0.15)

    # 1. FLAIR
    axes[0].imshow(flair_slice, cmap="gray", origin="lower")
    axes[0].set_title(f"FLAIR (Modality 0)\nSlice {slice_idx}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. T1
    axes[1].imshow(t1_slice, cmap="gray", origin="lower")
    axes[1].set_title(f"T1 (Modality 1)\nSlice {slice_idx}", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. T1ce
    axes[2].imshow(t1ce_slice, cmap="gray", origin="lower")
    axes[2].set_title(f"T1ce (Modality 2)\nSlice {slice_idx}", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    # 4. T2
    axes[3].imshow(t2_slice, cmap="gray", origin="lower")
    axes[3].set_title(f"T2 (Modality 3)\nSlice {slice_idx}", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # 5. Mask
    axes[4].imshow(mask_slice, cmap=custom_cmap, vmin=0, vmax=3, origin="lower")
    axes[4].set_title("Ground-Truth Mask\n(Discrete Classes)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    # 6. Overlay (T1ce + Mask)
    axes[5].imshow(t1ce_slice, cmap="gray", origin="lower")
    masked_data = np.ma.masked_where(mask_slice == 0, mask_slice)
    axes[5].imshow(masked_data, cmap=custom_cmap, vmin=0, vmax=3, alpha=0.6, origin="lower")
    axes[5].set_title("MRI + Mask Overlay\n(T1ce + Multi-class)", fontsize=11, fontweight="bold")
    axes[5].axis("off")

    # Legend for classes
    patch_bg = mpatches.Patch(color="black", label="0: Background")
    patch_ncr = mpatches.Patch(color=(0.9, 0.2, 0.2), label="1: Necrotic Core (NCR/NET)")
    patch_ed = mpatches.Patch(color=(0.2, 0.8, 0.2), label="2: Edema (ED)")
    patch_et = mpatches.Patch(color=(1.0, 0.85, 0.0), label="3: Enhancing Tumor (ET)")
    fig.legend(
        handles=[patch_ncr, patch_ed, patch_et],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        fontsize=10
    )

    out_plot_path = vis_dir / f"{patient_id}_slice_{slice_idx}_visualization.png"
    plt.savefig(out_plot_path, bbox_inches="tight")
    plt.close()

    # Also save a canonical copy as sample_slice_visualization.png
    canonical_path = vis_dir / "sample_slice_visualization.png"
    import shutil
    shutil.copyfile(out_plot_path, canonical_path)

    print(f"\n[SUCCESS] Visualization figure saved to:")
    print(f"  -> {out_plot_path}")
    print(f"  -> {canonical_path}\n")


if __name__ == "__main__":
    main()
