#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# A Bash script that:
#   1) Reads file paths (e.g. .nii.gz) from a CSV.
#   2) For each file, performs denoising (no upsampling) only if the final .nii.gz
#      does not already exist.
#      - If the final .nii.gz exists, it skips denoising.
#      - Otherwise, it removes any partial attempts and runs fondue_eval_simpleitk.py.
#
# Resume-friendly design: if the script crashes or is canceled, re-run it,
# and it will pick up where it left off (skipping completed volumes).
#
# Usage:
#   ./run_denoise_resume.sh <den_method_name> <csv_file> <output_folder>
#
# Example:
#   ./run_denoise_resume.sh FONDUE_LT /data/dadmah/gonwal2/Documents/datasets/Datasets_raw/exvivo_pp0.csv /data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_EXVIVO_pp1
#
# Note:
#   The input files are expected to have filenames ending with "_pp0.nii.gz".
#   The script will replace "_pp0" with "_pp1" in the output filename and will
#   not append the denoising method name.
# -----------------------------------------------------------------------------

# Exit if any command fails
set -e

# 1) Extract arguments
den_method_name="$1"    # e.g. FONDUE_LT
csv_file="$2"           # e.g. /data/dadmah/gonwal2/Documents/datasets/Datasets_raw/exvivo_pp0.csv
output_folder_denoised="$3"   # e.g. /data/dadmah/gonwal2/Documents/datasets/Datasets_raw/ALL_EXVIVO_pp1
if [ -z "${den_method_name}" ] || [ -z "${csv_file}" ] || [ -z "${output_folder_denoised}" ]; then
    echo "Usage: $0 <den_method_name> <csv_file> <output_folder>"
    exit 1
fi

# 2) Define output directory for the denoised results (provided as argument)
mkdir -p "${output_folder_denoised}"

echo "======================================================================"
echo "Reading from CSV:          ${csv_file}"
echo "Denoised results folder:   ${output_folder_denoised}"
echo "Denoise method:            ${den_method_name}"
echo "======================================================================"

# 3) Read CSV line by line
while IFS= read -r input_file
do
    # Skip empty lines
    if [ -z "$input_file" ]; then
        continue
    fi

    # Extract base filename, e.g. foo_pp0.nii.gz => foo_pp0
    base_filename="$(basename "$input_file")"
    # Remove the .nii.gz extension in one pass
    name_no_ext="${base_filename%.nii.gz}"
    # Replace _pp0 with _pp1 in the name
    new_name="${name_no_ext/_pp0/_pp1}"

    # Construct final denoised output name:
    # e.g. .../foo_T1w_pp1.nii.gz
    out_file_denoised="${output_folder_denoised}/${new_name}.nii.gz"

    echo "-----------------------------------------------------------------"
    echo "Denoising input:  ${input_file}"
    echo "Denoised output:  ${out_file_denoised}"
    echo "Denoise method:   ${den_method_name}"
    echo "-----------------------------------------------------------------"

    # Check if the final denoised file already exists
    if [ -f "${out_file_denoised}" ]; then
        echo "Denoised file already exists at ${out_file_denoised}."
        echo "Skipping denoising step."
    else
        # Remove any partial .nii or .nii.gz leftover
        partial_prefix="${output_folder_denoised}/${new_name}"
        echo "No final .nii.gz found, removing partial files matching:"
        echo "${partial_prefix}.nii*"
        rm -f "${partial_prefix}.nii" "${partial_prefix}.nii.gz" 2>/dev/null || true

        # Now run the denoising script
        cd /data/dadmah/gonwal2/Documents/denoising/FONDUE_v1_1
        python ./fondue_eval_simpleitk.py \
            --in_name "${input_file}" \
            --out_name "${out_file_denoised}" \
            --save_new_input False \
            --intensity_range_mode 2 \
            --robust_rescale_input False \
            --name "${den_method_name}" \
            --batch_size 4
    fi

    echo "========== Done processing ${input_file} =========="
    echo "Denoised file:  ${out_file_denoised}"
    echo

done < "${csv_file}"

echo "All lines from ${csv_file} have been processed!"

