#!/bin/bash
# This script searches through subject folders under the BASE_DIR,
# extracts the image description from each .mnc file using mincheader and grep,
# and writes the results to a CSV file only if the description contains
# T1w, T2w, or FLAIR (case insensitive).

# Base directory containing subject folders
BASE_DIR="/data/dadmah/ex_vivo_DBCBB_Data/MNC_Files"
# Name (and path) of the output CSV file
OUTPUT_CSV="exvivo_all.csv"

# Write a header to the CSV file (optional)
echo "Subject,File Path,Image Description" > "$OUTPUT_CSV"

# Loop over each subject directory in BASE_DIR
for subject_dir in "$BASE_DIR"/*; do
    # Ensure that the item is a directory
    if [ -d "$subject_dir" ]; then
        # Get the subject name from the directory name
        subject=$(basename "$subject_dir")
        
        # Loop over each .mnc file in the subject directory
        for mnc_file in "$subject_dir"/*.mnc; do
            # Check if the file exists (in case there are no .mnc files)
            if [ -f "$mnc_file" ]; then
                # Extract the image description using mincheader and grep.
                # The sed command strips off everything up to and including the last colon and space.
                img_desc=$(mincheader "$mnc_file" | grep 'acquisition:series_description' | sed 's/.*: //')
                
                # Only add the entry if the image description contains T1w, T2w, or FLAIR (case insensitive)
                if echo "$img_desc" | grep -qiE 'T1w|T2w|FLAIR'; then
                    echo "\"$subject\",\"$mnc_file\",\"$img_desc\"" >> "$OUTPUT_CSV"
                fi
            fi
        done
    fi
done

echo "CSV file has been generated at: $(realpath "$OUTPUT_CSV")"

