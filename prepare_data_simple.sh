#!/bin/bash

# Simple data preparation script for aneurysm detection
# Organizes existing NIfTI files into train/val/test directories

BASE_DIR="/Users/owner/work"
DATA_DIR="$BASE_DIR/data"
OUTPUT_DIR="$BASE_DIR/aneurysm_dataset"

# Create directories
mkdir -p "$OUTPUT_DIR/train"
mkdir -p "$OUTPUT_DIR/val"
mkdir -p "$OUTPUT_DIR/test"

echo "Organizing aneurysm data..."

# Count total NIfTI files (excluding segmentation masks)
TOTAL_FILES=$(ls $DATA_DIR/*.nii 2>/dev/null | grep -v "_cowseg" | wc -l | tr -d ' ')
echo "Found $TOTAL_FILES NIfTI files"

# Calculate split sizes (75/5/20)
TRAIN_SIZE=$((TOTAL_FILES * 75 / 100))
VAL_SIZE=$((TOTAL_FILES * 5 / 100))
TEST_SIZE=$((TOTAL_FILES - TRAIN_SIZE - VAL_SIZE))

echo "Split: Train=$TRAIN_SIZE, Val=$VAL_SIZE, Test=$TEST_SIZE"

# Create file list and shuffle
ls $DATA_DIR/*.nii 2>/dev/null | grep -v "_cowseg" | sort -R > /tmp/nifti_files.txt

# Split files
head -n $TRAIN_SIZE /tmp/nifti_files.txt | while read file; do
    filename=$(basename "$file")
    cp "$file" "$OUTPUT_DIR/train/"
    # Check for corresponding segmentation
    seg_file="${file%.nii}_cowseg.nii"
    if [ -f "$seg_file" ]; then
        cp "$seg_file" "$OUTPUT_DIR/train/"
    fi
done

tail -n +$((TRAIN_SIZE + 1)) /tmp/nifti_files.txt | head -n $VAL_SIZE | while read file; do
    filename=$(basename "$file")
    cp "$file" "$OUTPUT_DIR/val/"
    # Check for corresponding segmentation
    seg_file="${file%.nii}_cowseg.nii"
    if [ -f "$seg_file" ]; then
        cp "$seg_file" "$OUTPUT_DIR/val/"
    fi
done

tail -n $TEST_SIZE /tmp/nifti_files.txt | while read file; do
    filename=$(basename "$file")
    cp "$file" "$OUTPUT_DIR/test/"
    # Check for corresponding segmentation
    seg_file="${file%.nii}_cowseg.nii"
    if [ -f "$seg_file" ]; then
        cp "$seg_file" "$OUTPUT_DIR/test/"
    fi
done

# Create dataset info
cat > "$OUTPUT_DIR/dataset_info.txt" << EOF
RSNA Aneurysm Detection Dataset
================================
Data Directory: $OUTPUT_DIR

Split Summary:
- Training: $(ls $OUTPUT_DIR/train/*.nii | wc -l) files
- Validation: $(ls $OUTPUT_DIR/val/*.nii | wc -l) files
- Testing: $(ls $OUTPUT_DIR/test/*.nii | wc -l) files

Task: Binary Classification (Aneurysm Detection)
Metrics: AUC, Sensitivity, Specificity, FPR, TNR

Created: $(date)
EOF

echo "Data preparation complete!"
cat "$OUTPUT_DIR/dataset_info.txt"