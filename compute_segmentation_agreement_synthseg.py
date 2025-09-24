#!/usr/bin/env python

"""
Computes a comprehensive set of segmentation similarity metrics between a ground
truth mask and two test masks.

This script first conforms the test masks to the ground truth's physical space
using nearest neighbor interpolation to ensure a valid comparison.
The final metrics table is both printed to the screen and saved to a CSV file.
"""

import sys
import os
import argparse
import SimpleITK as sitk
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# --- Label and Group Definitions ---
LABELS_DICT = {
    2: "Left Cerebral White Matter", 3: "Left Cerebral Cortex", 4: "Left Lateral Ventricle",
    5: "Left Inf Lat Ventricle", 7: "Left Cerebellum White Matter", 8: "Left Cerebellum Cortex",
    10: "Left Thalamus", 11: "Left Caudate", 12: "Left Putamen", 13: "Left Pallidum",
    14: "3rd Ventricle", 15: "4th Ventricle", 16: "Brain-Stem", 17: "Left Hippocampus",
    18: "Left Amygdala", 24: "CSF", 26: "Left Accumbens Area", 28: "Left Ventral DC",
    41: "Right Cerebral White Matter", 42: "Right Cerebral Cortex", 43: "Right Lateral Ventricle",
    44: "Right Inf Lat Ventricle", 46: "Right Cerebellum White Matter", 47: "Right Cerebellum Cortex",
    49: "Right Thalamus", 50: "Right Caudate", 51: "Right Putamen", 52: "Right Pallidum",
    53: "Right Hippocampus", 54: "Right Amygdala", 58: "Right Accumbens Area",
    60: "Right Ventral DC"
}
GROUPS_DICT = {
    "Gray Matter": [3, 42, 10, 49, 11, 50, 12, 51, 13, 52, 18, 54, 26, 58, 28, 60],
    "White Matter": [2, 41], "Ventricles": [4, 43, 5, 44, 14, 15, 24],
    "Hippocampus": [17, 53], "Cerebellum GM": [8, 47], "Cerebellum WM": [7, 46],
    "Brain-Stem": [16]
}

# --- Metric Calculation Functions ---
def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Computes a panel of metrics for a given ground truth and prediction mask."""
    epsilon = 1e-8
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()
    
    vol_gt = np.sum(gt_flat)
    vol_pred = np.sum(pred_flat)
    intersection = np.sum(gt_flat * pred_flat)
    union = vol_gt + vol_pred - intersection

    if not np.any(gt_mask) and not np.any(pred_mask):
        return {
            'Dice': 1.0, 'IoU': 1.0, 'Kappa': 1.0,
            'Sensitivity': 1.0, 'Precision': 1.0, 'VolumeSimilarity': 1.0
        }
        
    metrics = {
        'Dice': (2. * intersection + epsilon) / (vol_gt + vol_pred + epsilon),
        'IoU': (intersection + epsilon) / (union + epsilon),
        'Kappa': cohen_kappa_score(gt_flat, pred_flat),
        'Sensitivity': (intersection + epsilon) / (vol_gt + epsilon),
        'Precision': (intersection + epsilon) / (vol_pred + epsilon),
        'VolumeSimilarity': 1.0 - (abs(vol_gt - vol_pred) / (vol_gt + vol_pred + epsilon))
    }
    return metrics

# --- Main Processing Function ---
def main():
    """Main function to orchestrate the loading, conforming, processing, and reporting."""
    parser = argparse.ArgumentParser(
        description="Compute comprehensive metrics for brain segmentation masks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ground_truth", type=str, default="/data/dadmah/gonwal2/Documents/overnight_2431/t2w_0p64_FONDUE_LT_mask.nii.gz", help="Path to the ground truth mask (e.g., ground_truth.nii.gz).")
    parser.add_argument("--test1", type=str, default="/data/dadmah/gonwal2/Documents/overnight_2431/t2w_1p28_nlmupsample_reg_synthseg.nii.gz", help="Path to the first test mask (e.g., prediction_A.nii.gz).")
    parser.add_argument("--test2", type=str, default="/data/dadmah/gonwal2/Documents/overnight_2431/t2w_1p28_nlmupsample_synthseg.nii.gz", help="Path to the second test mask (e.g., prediction_B.nii.gz).")
    args = parser.parse_args()

    print("🧠 Loading images...")
    try:
        gt_img = sitk.ReadImage(args.ground_truth, sitk.sitkUInt32)
        test1_img = sitk.ReadImage(args.test1, sitk.sitkUInt32)
        test2_img = sitk.ReadImage(args.test2, sitk.sitkUInt32)
    except Exception as e:
        print(f"❌ Error reading files: {e}", file=sys.stderr)
        sys.exit(1)

    print("🔄 Conforming test masks to ground truth space...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(gt_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    test1_conformed_img = resampler.Execute(test1_img)
    test2_conformed_img = resampler.Execute(test2_img)
    
    gt_array = sitk.GetArrayFromImage(gt_img)
    test1_array = sitk.GetArrayFromImage(test1_conformed_img)
    test2_array = sitk.GetArrayFromImage(test2_conformed_img)
    print(f"✅ Images loaded and conformed. Shape: {gt_array.shape}")
    
    results_data = []
    test1_name = os.path.basename(args.test1).split('.')[0]
    test2_name = os.path.basename(args.test2).split('.')[0]

    all_label_defs = [("Individual", LABELS_DICT), ("Group", GROUPS_DICT)]

    for type_name, label_dict in all_label_defs:
        print(f"\n📊 Calculating metrics for {type_name} structures...")
        for name, label_ids in label_dict.items():
            if type_name == "Individual":
                struct_name, label_id = label_ids, name
                gt_mask = (gt_array == label_id)
                test1_mask = (test1_array == label_id)
                test2_mask = (test2_array == label_id)
            else: # Group
                struct_name = name
                gt_mask = np.isin(gt_array, label_ids)
                test1_mask = np.isin(test1_array, label_ids)
                test2_mask = np.isin(test2_array, label_ids)

            if not np.any(gt_mask):
                continue
            
            metrics1 = calculate_metrics(gt_mask, test1_mask)
            metrics2 = calculate_metrics(gt_mask, test2_mask)
            
            row = {"Type": type_name, "Structure": struct_name}
            for key, val in metrics1.items():
                row[f"{test1_name}_{key}"] = val
            for key, val in metrics2.items():
                row[f"{test2_name}_{key}"] = val
            results_data.append(row)

    if not results_data:
        print("No valid labels found in the ground truth mask. No metrics to compute.")
        return

    # --- Display and Save Results ---
    df = pd.DataFrame(results_data).set_index(['Type', 'Structure'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 180)
    pd.options.display.float_format = '{:,.4f}'.format

    print("\n--- ✅ Segmentation Similarity Metrics ---")
    print(df.sort_index())
    print("-----------------------------------------\n")

    # Save the DataFrame to a CSV file
    output_filename = '/data/dadmah/gonwal2/Documents/overnight_2431/segmentation_metrics_nlmreg_vs_nlm.csv'
    df.to_csv(output_filename)
    print(f"📄 Metrics successfully saved to: {output_filename}")


if __name__ == "__main__":
    main()


