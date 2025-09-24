#!/usr/bin/env python3
import csv
import os
import random

# ---------------------------
# Configuration and settings
# ---------------------------
INPUT_CSV = "/data/dadmah/gonwal2/Documents/SuperResolution/networks/taming-transformers3d_local/dataset_generators_hdf5_3d/exvivo_all.csv"
TARGET_CSV = "/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_pp1_datasplit_csvs/"
# Use the same folder as INPUT_CSV for output files.
input_dir = os.path.dirname(INPUT_CSV)
output_dir = os.path.dirname(TARGET_CSV)
# Output file names for the "pre" CSV files (detailed version with header)
TRAIN_PRE_CSV = os.path.join(input_dir, "DHDATASET_train_pre.csv")
VAL_PRE_CSV   = os.path.join(input_dir, "DHDATASET_val_pre.csv")
TEST_PRE_CSV  = os.path.join(input_dir, "DHDATASET_test_pre.csv")

# Output file names for the final CSV files (one column only, no header)
TRAIN_CSV = os.path.join(output_dir, "DHDATASET_train.csv")
VAL_CSV   = os.path.join(output_dir, "DHDATASET_val.csv")
TEST_CSV  = os.path.join(output_dir, "DHDATASET_test.csv")

# Splitting ratios (subject-wise)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
# (The test ratio is implied: 1 - (TRAIN_RATIO + VAL_RATIO))

# Set random seed for reproducibility
random.seed(42)

# ---------------------------
# Read and filter the CSV file
# ---------------------------
# The input CSV is assumed to have a header: Subject,File Path,Image Description.
# We discard rows where the image description contains "Neuromel" (case insensitive).
# For each subject, we then select one row for T1w and one row for T2w.
# If multiple rows exist for a modality, the one with the last filename (alphabetically) is chosen.

# Dictionary to collect rows per subject and modality.
# Structure: { subject: { "T1w": [row, ...], "T2w": [row, ...] } }
subject_modality_rows = {}

with open(INPUT_CSV, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        subject = row["Subject"]
        file_path = row["File Path"]
        image_desc = row["Image Description"]

        # Discard any row whose description contains "Neuromel" (case insensitive)
        if "neuromel" in image_desc.lower():
            continue

        # Determine the modality by checking for "T1w" or "T2w" in the description.
        modality = None
        if "t1w" in image_desc.lower():
            modality = "T1w"
        elif "t2w" in image_desc.lower():
            modality = "T2w"
        else:
            continue

        # Initialize dictionary for the subject if needed.
        if subject not in subject_modality_rows:
            subject_modality_rows[subject] = {"T1w": [], "T2w": []}

        subject_modality_rows[subject][modality].append(row)

# ---------------------------
# Select one row per modality per subject
# ---------------------------
final_subject_data = {}

for subject, modality_dict in subject_modality_rows.items():
    t1w_rows = modality_dict["T1w"]
    t2w_rows = modality_dict["T2w"]

    # Only consider subjects with at least one row for both modalities.
    if not t1w_rows or not t2w_rows:
        continue

    # Helper function to extract the filename from the file path.
    def get_filename(row):
        return os.path.basename(row["File Path"])

    # Sort (alphabetically by filename) and select the last row for each modality.
    selected_t1w = sorted(t1w_rows, key=get_filename)[-1]
    selected_t2w = sorted(t2w_rows, key=get_filename)[-1]

    final_subject_data[subject] = {"T1w": selected_t1w, "T2w": selected_t2w}

# ---------------------------
# Subject-wise random splitting
# ---------------------------
all_subjects = list(final_subject_data.keys())
random.shuffle(all_subjects)

n_subjects = len(all_subjects)
n_train = int(TRAIN_RATIO * n_subjects)
n_val = int(VAL_RATIO * n_subjects)
# The remaining subjects go to the test set.
train_subjects = all_subjects[:n_train]
val_subjects = all_subjects[n_train:n_train+n_val]
test_subjects = all_subjects[n_train+n_val:]

print(f"Total subjects with both T1w and T2w: {n_subjects}")
print(f"Train: {len(train_subjects)} subjects")
print(f"Validation: {len(val_subjects)} subjects")
print(f"Test: {len(test_subjects)} subjects")

# ---------------------------
# Write out the pre CSV files (detailed with header)
# ---------------------------
HEADER = ["Subject", "File Path", "Image Description"]

def write_pre_csv(filename, subjects_list):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for subj in subjects_list:
            t1w_row = final_subject_data[subj]["T1w"]
            t2w_row = final_subject_data[subj]["T2w"]
            writer.writerow([t1w_row["Subject"], t1w_row["File Path"], t1w_row["Image Description"]])
            writer.writerow([t2w_row["Subject"], t2w_row["File Path"], t2w_row["Image Description"]])

write_pre_csv(TRAIN_PRE_CSV, train_subjects)
write_pre_csv(VAL_PRE_CSV, val_subjects)
write_pre_csv(TEST_PRE_CSV, test_subjects)

# ---------------------------
# Write out the final CSV files (one column only, no header)
# ---------------------------
def write_final_csv(filename, subjects_list):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for subj in subjects_list:
            t1w_row = final_subject_data[subj]["T1w"]
            t2w_row = final_subject_data[subj]["T2w"]
            # Write only the full absolute path (ensure it is absolute)
            writer.writerow([os.path.abspath(t1w_row["File Path"])])
            writer.writerow([os.path.abspath(t2w_row["File Path"])])

write_final_csv(TRAIN_CSV, train_subjects)
write_final_csv(VAL_CSV, val_subjects)
write_final_csv(TEST_CSV, test_subjects)

print("Data splitting complete.")
print("Pre files (with headers) saved as:")
print(f"  {TRAIN_PRE_CSV}")
print(f"  {VAL_PRE_CSV}")
print(f"  {TEST_PRE_CSV}")
print("Final files (one column, no header) saved as:")
print(f"  {TRAIN_CSV}")
print(f"  {VAL_CSV}")
print(f"  {TEST_CSV}")
