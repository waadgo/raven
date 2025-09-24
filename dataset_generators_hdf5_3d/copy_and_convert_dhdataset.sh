#!/bin/bash
# This script processes the pre CSV files containing detailed file info.
# For each row (after the header), it:
#   - Determines the subject and modality (T1w or T2w)
#   - Copies the .mnc file to the destination folder with the name: {SUBJECT_ID}_{MODALITY}_pp0.mnc
#   - Converts the copied file to NIfTI format (.nii) using mnc2nii,
#     then compresses the .nii file to .nii.gz using gzip,
#     resulting in a final file named {SUBJECT_ID}_{MODALITY}_pp0.nii.gz.
#   - The intermediate .nii file is removed after compression.
#
# Adjust the paths below as necessary.

# ---------------------------
# Configuration and settings
# ---------------------------
# Folder where the pre CSV files are stored.
#CSV_DIR="/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_pp1_datasplit_csvs"
CSV_DIR="/data/dadmah/gonwal2/Documents/SuperResolution/networks/taming-transformers3d_local/dataset_generators_hdf5_3d"
# Pre CSV file names (with headers) for train, val, and test splits.
CSV_FILES=(
    "${CSV_DIR}/DHDATASET_train_pre.csv"
    "${CSV_DIR}/DHDATASET_val_pre.csv"
    "${CSV_DIR}/DHDATASET_test_pre.csv"
)

# Destination directories:
# All .mnc files will be copied (renamed) here.
DEST_MNC="/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_EXVIVO_mnc"
# The converted .nii.gz files will be placed here.
DEST_NII="/data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_EXVIVO"

# Create destination directories if they do not exist.
mkdir -p "$DEST_MNC"
mkdir -p "$DEST_NII"

# ---------------------------
# Processing CSV files
# ---------------------------
echo "Starting processing of pre CSV files..."

for csv_file in "${CSV_FILES[@]}"; do
    if [ ! -f "$csv_file" ]; then
        echo "CSV file '$csv_file' not found. Skipping."
        continue
    fi

    echo "Processing CSV file: $csv_file"
    
    # Skip the header and process each subsequent line.
    tail -n +2 "$csv_file" | while IFS=',' read -r subject file_path image_desc; do
        # Remove any surrounding quotes (if present)
        subject=$(echo "$subject" | sed 's/^"//; s/"$//')
        file_path=$(echo "$file_path" | sed 's/^"//; s/"$//')
        image_desc=$(echo "$image_desc" | sed 's/^"//; s/"$//')
        
        # Verify the file exists.
        if [ ! -f "$file_path" ]; then
            echo "File '$file_path' not found. Skipping."
            continue
        fi
        
        # Determine modality from the image description (case-insensitive)
        modality=""
        lower_desc=$(echo "$image_desc" | tr '[:upper:]' '[:lower:]')
        if echo "$lower_desc" | grep -q "t1w"; then
            modality="T1w"
        elif echo "$lower_desc" | grep -q "t2w"; then
            modality="T2w"
        else
            echo "Could not determine modality from image description: '$image_desc'. Skipping."
            continue
        fi

        # Build the new filenames according to the naming system:
        # {SUBJECT_ID}_{MODALITY}_pp0.{ext}
        new_mnc_filename="${subject}_${modality}_pp0.mnc"
        # Final output: .nii.gz
        new_nii_filename="${subject}_${modality}_pp0.nii.gz"

        dest_mnc_path="${DEST_MNC}/${new_mnc_filename}"
        dest_nii_path="${DEST_NII}/${new_nii_filename}"
        
        # Temporary .nii filename (before gzip compression)
        temp_nii_path="${DEST_NII}/${subject}_${modality}_pp0.nii"

        echo "Processing subject '$subject' modality '$modality'"
        echo "  Original file: $file_path"
        echo "  Copying to:    $dest_mnc_path"
        
        # Copy the .mnc file to the destination with the new name.
        cp "$file_path" "$dest_mnc_path"
        if [ $? -ne 0 ]; then
            echo "Error copying '$file_path' to '$dest_mnc_path'. Skipping conversion."
            continue
        fi

        echo "  Converting '$dest_mnc_path' to NIfTI format as '$temp_nii_path'"
        # Convert the .mnc file to .nii format using mnc2nii.
        mnc2nii "$dest_mnc_path" "$temp_nii_path"
        if [ $? -ne 0 ]; then
            echo "Conversion failed for '$dest_mnc_path'."
            continue
        fi

        echo "  Compressing '$temp_nii_path' to NIfTI GZ format"
        # Compress the .nii file to .nii.gz using gzip.
        gzip "$temp_nii_path"
        if [ $? -ne 0 ]; then
            echo "Compression failed for '$temp_nii_path'."
            continue
        fi

        # gzip renames the file to ${temp_nii_path}.gz and removes the original .nii.
        compressed_file="${temp_nii_path}.gz"
        if [ "$compressed_file" != "$dest_nii_path" ]; then
            mv "$compressed_file" "$dest_nii_path"
        fi

        echo "  Completed processing for $subject $modality"
        echo "  Final file: $dest_nii_path"
        echo "---------------------------------------------"
    done
done

echo "All files processed."
