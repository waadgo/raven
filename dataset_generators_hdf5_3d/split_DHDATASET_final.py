#!/usr/bin/env python3
import os
import csv
import random
import glob

# ---------------------------
# Configuration and settings
# ---------------------------
# Folder containing the preprocessed files (with naming pattern: SUBJECTID_MODAITY_pp1.nii.gz)
INPUT_FOLDER = "/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_EXVIVO_pp1"

# Folder where the CSV files will be saved.
TARGET_CSV = "/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_pp1_datasplit_csvs"
TARGET_CSV2 = "/data/dadmah/gonwal2/Documents/SuperResolution/networks/taming-transformers3d_local/dataset_generators_hdf5_3d/"
os.makedirs(TARGET_CSV, exist_ok=True)

# Splitting ratios (subject-wise)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
# (Test subjects = remaining subjects)

# Output file names for the "pre" CSV files (detailed version with header)
TRAIN_PRE_CSV = os.path.join(TARGET_CSV2, "DHDATASET_train_pre.csv")
VAL_PRE_CSV   = os.path.join(TARGET_CSV2, "DHDATASET_val_pre.csv")
TEST_PRE_CSV  = os.path.join(TARGET_CSV2, "DHDATASET_test_pre.csv")

# Output file names for the final CSV files (one column only, no header)
TRAIN_CSV = os.path.join(TARGET_CSV, "DHDATASET_train.csv")
VAL_CSV   = os.path.join(TARGET_CSV, "DHDATASET_val.csv")
TEST_CSV  = os.path.join(TARGET_CSV, "DHDATASET_test.csv")

# Set random seed for reproducibility
random.seed(42)

# ---------------------------
# Read and group files from the folder
# ---------------------------
# Expected file name format: SUBJECTID_MODAITY_pp1.nii.gz
# Example: DH1197_20220926_T1w_pp1.nii.gz
# Where:
#   - Subject ID is the concatenation of the first two tokens: "DH1197_20220926"
#   - Modality is the third token: "T1w" or "T2w"
#   - "pp1" is a literal suffix indicating preprocessing.

subject_modality_rows = {}  # Structure: { subject: { "T1w": [row, ...], "T2w": [row, ...] } }

# List all .nii.gz files in the input folder
file_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.nii.gz"))

for file_path in file_paths:
    base_name = os.path.basename(file_path)
    # Split by underscore. For a valid filename we expect at least 4 parts.
    parts = base_name.split("_")
    if len(parts) < 4:
        continue  # Skip files that do not match the expected pattern

    # Construct subject id from first two tokens
    subject = f"{parts[0]}_{parts[1]}"
    modality = parts[2]  # Expecting e.g. "T1w" or "T2w"
    # Ensure modality is one of the expected values
    if modality not in ["T1w", "T2w"]:
        continue

    # Prepare a row dictionary with the three columns
    # (We use modality as the image description.)
    row = {"Subject": subject, "File Path": file_path, "Image Description": modality}

    # Initialize subject dictionary if needed.
    if subject not in subject_modality_rows:
        subject_modality_rows[subject] = {"T1w": [], "T2w": []}
    subject_modality_rows[subject][modality].append(row)

# ---------------------------
# Select one file per modality per subject
# ---------------------------
final_subject_data = {}
for subject, modality_dict in subject_modality_rows.items():
    t1w_rows = modality_dict["T1w"]
    t2w_rows = modality_dict["T2w"]

    # Only consider subjects that have at least one file for both modalities.
    if not t1w_rows or not t2w_rows:
        continue

    # Helper function: extract the filename from the file path.
    def get_filename(row):
        return os.path.basename(row["File Path"])

    # Sort alphabetically by filename and select the last file for each modality.
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
            # Write only the full absolute path (ensuring it is absolute)
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
