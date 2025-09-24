#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_and_visualize_metrics_diagnostic.py

This script provides a comprehensive, step-by-step analysis of image similarity,
with extensive diagnostic logging to trace data transformations and metric calculations.
It is designed to debug inconsistencies in metric results by showing the state of
the data at every critical point.

Key Features:
- Accepts GT, test, and mask files via command-line.
- **Intensive Diagnostics:** Prints array statistics at each processing stage.
- **Independent Percentile Normalization:** Implements a robust normalization by
  clipping each image to its 1st and 99th percentile values before scaling to [0, 1].
  This method ensures ZERO information leak between the test and ground truth images.
- **Numerically Consistent Metrics:** Uses the precise calculation logic from your
  benchmark script to ensure results are comparable.
- **Visual & Numerical Output:** Saves final metrics to CSV, 3D error maps to NIfTI,
  and 2D slice snapshots for visual inspection.
"""
import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim as ssim_pt

# Attempt to import LPIPS, handle failure gracefully
try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None

# ----------------------------------------------------------------------#
# Logging and Device Configuration
# ----------------------------------------------------------------------#
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on device: {DEVICE}")

LPIPS_MODEL = None
if lpips_lib:
    try:
        LPIPS_MODEL = lpips_lib.LPIPS(net="alex", spatial=True).to(DEVICE).eval()
        for p in LPIPS_MODEL.parameters():
            p.requires_grad_(False)
        logging.info("LPIPS model loaded (AlexNet, spatial=True)")
    except Exception as e:
        logging.error(f"Failed to load LPIPS model: {e}. LPIPS calculation disabled.")
        LPIPS_MODEL = None
else:
    logging.warning("LPIPS library not found. LPIPS calculation disabled.")


# ----------------------------------------------------------------------#
# Helper Functions
# ----------------------------------------------------------------------#

def print_stats(
    arr: Union[np.ndarray, torch.Tensor],
    name: str,
    mask: Union[np.ndarray, torch.Tensor] = None
):
    """Helper function to print detailed statistics of a tensor or array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy().astype(bool)

    header = f" DIAGNOSTICS FOR: {name} "
    logging.debug("=" * 20 + header + "=" * 20)
    logging.debug(f"Shape: {arr.shape}, DType: {arr.dtype}")
    logging.debug(f"Full Volume -> Min: {arr.min():.6f}, Max: {arr.max():.6f}, Mean: {arr.mean():.6f}")
    if mask is not None and mask.any():
        masked_arr = arr[mask]
        logging.debug(f"Masked Area -> Min: {masked_arr.min():.6f}, Max: {masked_arr.max():.6f}, Mean: {masked_arr.mean():.6f}")
    logging.debug("=" * (42 + len(header)) + "\n")


# ----------------------------------------------------------------------#
# Core Logic (for numerical consistency with Script 1)
# ----------------------------------------------------------------------#

class PSNR:
    """Calculates PSNR map, setting MSE=0 to PSNR=0.0 for robust map saving."""
    def __init__(self, max_val: float = 1.0):
        self.max_val = max_val

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        img1, img2 = img1.float(), img2.float()
        mse = F.mse_loss(img1, img2, reduction='none')
        mse_clipped = torch.clamp(mse, min=1e-12)
        psnr_map = 20 * torch.log10(self.max_val / torch.sqrt(mse_clipped))
        psnr_map[mse < 1e-12] = 0.0
        return psnr_map

PSNR_CALC = PSNR(max_val=1.0)

def load_image(fp: Path) -> sitk.Image:
    """Loads a medical image using SimpleITK."""
    return sitk.ReadImage(str(fp), sitk.sitkFloat64)

def conform_image(moving_img: sitk.Image, reference_img: sitk.Image, is_mask: bool = False) -> sitk.Image:
    """Resamples (conforms) a moving image to the space of a reference image."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resampler.Execute(moving_img)


def calc_metrics_and_maps(
    gt_np: np.ndarray,
    test_np: np.ndarray,
    mask_np: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Computes MAE, PSNR, SSIM, and LPIPS with extensive diagnostic logging.
    This version uses independent percentile normalization for fair comparison.
    """
    logging.info("--- Entering Metric Calculation Stage ---")
    
    # --- CHANGE: Perform Independent Percentile Normalization ---
    logging.info("Performing independent percentile normalization...")

    def percentile_normalize(img_array: np.ndarray, name: str) -> np.ndarray:
        p1, p99 = np.percentile(img_array, [1, 99])
        logging.debug(f"Normalization range for '{name}': [{p1:.4f}, {p99:.4f}] (1st-99th percentile)")
        clipped_array = np.clip(img_array, p1, p99)
        denominator = p99 - p1
        if denominator < 1e-12:
            return np.zeros_like(img_array)
        return (clipped_array - p1) / denominator

    gt_np_norm = percentile_normalize(gt_np, "Ground Truth")
    test_np_norm = percentile_normalize(test_np, "Test Image")

    gt_t_norm = torch.from_numpy(gt_np_norm).to(DEVICE).float()
    test_t_norm = torch.from_numpy(test_np_norm).to(DEVICE).float()
    mask_t = torch.from_numpy(mask_np).to(DEVICE).bool()

    print_stats(gt_t_norm, "Ground Truth Tensor (Post-Percentile-Normalization)")
    print_stats(test_t_norm, "Test Tensor (Post-Percentile-Normalization)")

    s1, s2, s3 = gt_t_norm.shape

    logging.info("Calculating MAE map...")
    mae_map = F.l1_loss(test_t_norm, gt_t_norm, reduction='none').cpu()
    print_stats(mae_map, "MAE Map", mask=mask_t)
    mae_scalar = mae_map[mask_t.cpu()].mean().item()
    logging.info(f"Final Masked MAE: {mae_scalar:.6f}")

    logging.info("Calculating PSNR map...")
    psnr_map = PSNR_CALC(test_t_norm, gt_t_norm).cpu()
    print_stats(psnr_map, "PSNR Map", mask=mask_t)
    psnr_scalar = psnr_map[mask_t.cpu()].mean().item()
    logging.info(f"Final Masked PSNR: {psnr_scalar:.6f}")

    logging.info("Calculating SSIM map across all 3 axes...")
    ssim_map = torch.zeros_like(gt_t_norm, device='cpu')
    pad = 5
    with torch.no_grad():
        for axis_idx, axis_dim, slicer in [
            (0, s1, lambda vol, i: vol[i, :, :]),
            (1, s2, lambda vol, i: vol[:, i, :]),
            (2, s3, lambda vol, i: vol[:, :, i])]:
            for i in range(axis_dim):
                sl_t = slicer(test_t_norm, i).unsqueeze(0).unsqueeze(0)
                sl_g = slicer(gt_t_norm, i).unsqueeze(0).unsqueeze(0)
                sl_t = F.pad(sl_t, (pad, pad, pad, pad), mode='replicate')
                sl_g = F.pad(sl_g, (pad, pad, pad, pad), mode='replicate')
                ssim2d = ssim_pt(sl_t, sl_g, data_range=1.0, size_average=False).cpu().squeeze()
                if axis_idx == 0:   ssim_map[i, :, :] += ssim2d
                elif axis_idx == 1: ssim_map[:, i, :] += ssim2d
                else:               ssim_map[:, :, i] += ssim2d
    ssim_map /= 3.0
    print_stats(ssim_map, "SSIM Map", mask=mask_t)
    ssim_scalar = ssim_map[mask_t.cpu()].mean().item()
    logging.info(f"Final Masked SSIM: {ssim_scalar:.6f}")

    lpips_map = torch.zeros_like(gt_t_norm, device='cpu')
    lpips_scalar = float('nan')
    if LPIPS_MODEL:
        logging.info("Calculating LPIPS map across all 3 axes...")
        lpips_scores_avg = []
        with torch.no_grad():
            for axis_idx, (axis_dim, slicer) in enumerate([
                (s1, lambda vol, i: vol[i, :, :]),
                (s2, lambda vol, i: vol[:, i, :]),
                (s3, lambda vol, i: vol[:, :, i])]):
                for i in range(axis_dim):
                    mask2d = slicer(mask_t, i)
                    if not bool(mask2d.any()): continue
                    t2d = slicer(test_t_norm, i).unsqueeze(0).unsqueeze(0)
                    g2d = slicer(gt_t_norm, i).unsqueeze(0).unsqueeze(0)
                    lpips2d = LPIPS_MODEL(t2d.repeat(1, 3, 1, 1), g2d.repeat(1, 3, 1, 1), normalize=True).cpu().squeeze()
                    if lpips2d.shape != mask2d.shape:
                        lpips2d = F.interpolate(lpips2d[None, None], size=mask2d.shape, mode='bilinear', align_corners=False).squeeze()
                    lpips_scores_avg.append(lpips2d[mask2d.cpu()].mean().item())
                    if axis_idx == 0:   lpips_map[i, :, :] += lpips2d
                    elif axis_idx == 1: lpips_map[:, i, :] += lpips2d
                    else:               lpips_map[:, :, i] += lpips2d
        lpips_map /= 3.0
        print_stats(lpips_map, "LPIPS Map", mask=mask_t)
        if lpips_scores_avg:
            lpips_scalar = np.mean(lpips_scores_avg)
        logging.info(f"Final Masked LPIPS: {lpips_scalar:.6f}")

    scalar_metrics = {"MAE": mae_scalar, "PSNR": psnr_scalar, "SSIM": ssim_scalar, "LPIPS": lpips_scalar}
    mask_cpu_float = mask_t.cpu().float().numpy()
    metric_maps = {
        "MAE": mae_map.numpy() * mask_cpu_float,
        "PSNR": psnr_map.numpy() * mask_cpu_float,
        "SSIM": ssim_map.numpy() * mask_cpu_float,
        "LPIPS": lpips_map.numpy() * mask_cpu_float if LPIPS_MODEL else np.zeros_like(gt_np)
    }
    return scalar_metrics, metric_maps


# ----------------------------------------------------------------------#
# Visualization and I/O
# ----------------------------------------------------------------------#

def save_nifti(data: np.ndarray, reference_img: sitk.Image, filename: str):
    out_img = sitk.GetImageFromArray(data)
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, filename)
    logging.info(f"Saved NIfTI map to: {filename}")

def save_combined_slices(
    gt_img: np.ndarray, test_img: np.ndarray, maps: Dict[str, np.ndarray],
    scalar_metrics: Dict[str, float], output_folder: str, filename_base: str,
    vmin: float, vmax: float
):
    s1, s2, s3 = gt_img.shape
    positions = [0.3, 0.4, 0.5, 0.6, 0.7]
    os.makedirs(output_folder, exist_ok=True)
    for axis, axis_size, axis_name in [(0, s1, 'Axial'), (1, s2, 'Coronal'), (2, s3, 'Sagittal')]:
        for pos in positions:
            slice_idx = int(pos * axis_size)
            gt_slice, test_slice = np.take(gt_img, slice_idx, axis=axis), np.take(test_img, slice_idx, axis=axis)
            mae_slice, lpips_slice = np.take(maps['MAE'], slice_idx, axis=axis), np.take(maps['LPIPS'], slice_idx, axis=axis)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='white')
            fig.suptitle(f'{axis_name} View (Slice {slice_idx} at {pos:.0%})', fontsize=16, y=0.95)
            axes[0, 0].imshow(gt_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax); axes[0, 0].set_title('Ground Truth (Raw)')
            axes[0, 1].imshow(test_slice.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax); axes[0, 1].set_title('Test Image (Raw)')
            im1 = axes[1, 0].imshow(mae_slice.T, cmap='inferno', origin='lower'); axes[1, 0].set_title(f"MAE (Avg: {scalar_metrics['MAE']:.4f})"); fig.colorbar(im1, ax=axes[1, 0])
            im2 = axes[1, 1].imshow(lpips_slice.T, cmap='viridis', origin='lower'); axes[1, 1].set_title(f"LPIPS (Avg: {scalar_metrics.get('LPIPS', float('nan')):.4f})"); fig.colorbar(im2, ax=axes[1, 1])
            for ax in axes.flatten(): ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig(os.path.join(output_folder, f"{filename_base}_{axis_name}_{int(pos*100)}.png"), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
    logging.info(f"Saved combined slice snapshots to: {output_folder}")


# ----------------------------------------------------------------------#
# Main Execution Logic
# ----------------------------------------------------------------------#

# GT: /data/dadmah/gonwal2/Documents/SuperResolution/LUSENBRINK_001_T1w_pp0.nii.gz
# MASK: /data/dadmah/gonwal2/Documents/SuperResolution/sr_benchmark/LUSENBRINK_001_T1w_pp1/mask/mask.nii.gz
# SMORE_X2: /data/dadmah/gonwal2/Documents/SuperResolution/Lusenbrink_smore_z2/LUSENBRINK_001_T1w_pp0_downsample-z2/LUSENBRINK_001_T1w_pp0_downsample-z2_smore4.nii.gz
# SMORE_X3: /data/dadmah/gonwal2/Documents/SuperResolution/Lusenbrink_smore_z3/LUSENBRINK_001_T1w_pp0_downsample-z3/LUSENBRINK_001_T1w_pp0_downsample-z3_smore4.nii.gz
# RAVEN_X2: /data/dadmah/gonwal2/Documents/SuperResolution/Lusenbrink_smore_z2/LUSENBRINK_001_T1w_pp0_downsample-z2/LUSENBRINK_001_T1w_pp0_downsample-z2_RAVEN.nii.gz
# RAVEN_X3: /data/dadmah/gonwal2/Documents/SuperResolution/Lusenbrink_smore_z2/LUSENBRINK_001_T1w_pp0_downsample-z3/LUSENBRINK_001_T1w_pp0_downsample-z3_RAVEN.nii.gz
# FM_X2: /data/dadmah/gonwal2/Documents/SuperResolution/Lusenbrink_smore_z2/LUSENBRINK_001_T1w_pp0_downsample-z3/LUSENBRINK_001_T1w_pp0_downsample-z2_RAVENFM.nii.gz
def main():
    parser = argparse.ArgumentParser(description="Compute and visualize similarity metrics with intensive diagnostics.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gt", type=str, default="/data/dadmah/gonwal2/Documents/SuperResolution/LUSENBRINK_001_T1w_pp0.nii.gz", help="Path to the ground truth image.")
    parser.add_argument("--test", type=str, default="/data/dadmah/gonwal2/Documents/SuperResolution/LUSENBRINK_001_T1w_pp0_SynthSR.nii.gz", help="Path to the test (predicted) image.")
    parser.add_argument("--mask", type=str, default="/data/dadmah/gonwal2/Documents/overnight_2431/t2w_0p64_FONDUE_LT_mask.nii.gz", help="Path to the foreground mask.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results. Defaults to a subfolder next to the test image.")
    args = parser.parse_args()

    start_time = time.time()

    # --- 1. Load and Conform Images ---
    logging.info("--- Loading and Conforming Images ---")
    gt_sitk = load_image(Path(args.gt))
    test_sitk_orig = load_image(Path(args.test))
    mask_sitk_orig = load_image(Path(args.mask))

    test_sitk = conform_image(test_sitk_orig, gt_sitk)
    mask_sitk = conform_image(mask_sitk_orig, gt_sitk, is_mask=True)

    gt_np_raw = sitk.GetArrayFromImage(gt_sitk)
    test_np_raw = sitk.GetArrayFromImage(test_sitk)
    mask_np = sitk.GetArrayFromImage(mask_sitk) > 0
    
    # --- DIAGNOSTICS: RAW DATA (as NumPy) ---
    print_stats(gt_np_raw, "RAW Ground Truth (from disk)", mask=mask_np)
    print_stats(test_np_raw, "RAW Test Image (from disk, after conforming)", mask=mask_np)

    if mask_np.sum() == 0:
        logging.error("Mask is empty after conforming. Cannot proceed.")
        return

    # --- 2. Run Metric Calculation ---
    # The function now handles the independent percentile normalization internally.
    # We pass the raw numpy arrays for a fair comparison.
    scalar_metrics, metric_maps = calc_metrics_and_maps(gt_np_raw, test_np_raw, mask_np)

    # --- 3. Save Outputs ---
    logging.info("--- Saving All Outputs ---")
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_base_dir = Path(args.test).parent / f"{Path(args.test).stem}_analysis_{timestamp}"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([scalar_metrics])
    csv_path = output_base_dir / "similarity_metrics.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.6g")
    
    logging.info("--- FINAL METRICS ---")
    logging.info("\n" + metrics_df.to_string(index=False))
    logging.info(f"\nMetrics saved to {csv_path}")

    for name, map_data in metric_maps.items():
        if name == "LPIPS" and not LPIPS_MODEL: continue
        map_filename = str(output_base_dir / f"{Path(args.test).stem}_{name.lower()}_map.nii.gz")
        save_nifti(map_data, gt_sitk, map_filename)

    # For visualization, we show the raw images to see what they look like before any processing
    vmin, vmax = np.percentile(gt_np_raw[mask_np], [1, 99])
    save_combined_slices(
        gt_img=gt_np_raw, test_img=test_np_raw, maps=metric_maps,
        scalar_metrics=scalar_metrics, output_folder=str(output_base_dir / "slice_snapshots"),
        filename_base=Path(args.test).stem, vmin=vmin, vmax=vmax
    )

    logging.info(f"\n✨ Analysis complete. Total time: {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
