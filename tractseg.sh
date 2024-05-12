#!/bin/bash
set -e

if [[ $# -ne 3 ]]; then
  echo "Usage: ./sh2peaks_batch.sh <PEAKS_PATH> <MASKS_PATH> <TRACTSEG_OUTPUT_PATH>"
  echo "  where PEAKS_PATH is the path to the folder into which FOD peaks were saved"
  echo "  and MASKS_PATH is the folder in which you saved the brain masks"
  echo "  and TRACTSEG_OUTPUT_PATH is the folder to save output into."
  exit
fi

PEAKS_PATH=$1
MASKS_PATH=$2
TRACTSEG_OUTPUT_PATH=$3

if [ ! -d "$PEAKS_PATH" ]; then
    echo "Peaks directory not found"
    exit 1
fi

mkdir -p $TRACTSEG_OUTPUT_PATH

# Estimate a response function for each image
for PEAKS in "$PEAKS_PATH"/*_peaks.nii.gz; do
  BASENAME=$(basename $PEAKS _peaks.nii.gz)
  MASK=$MASKS_PATH/${BASENAME}_mask.nii.gz
  OUTPUT_DIR="$TRACTSEG_OUTPUT_PATH"/${BASENAME}/
  mkdir -p "$OUTPUT_DIR"

  # tractseg needs us to compute three things before doing tractography: tract segmentations,
  # bundle segmentations, and TOMs (tract orientation maps).
  # after getting those three things then from those it performs tractography.

  TRACT_SEGS="$OUTPUT_DIR/bundle_segmentations"
  END_SEGS="$OUTPUT_DIR/endings_segmentations"
  TOMS="$OUTPUT_DIR/TOM"
  TRACTS="$OUTPUT_DIR/TOM_trackings"

  if [ -d $TRACT_SEGS ]; then
    echo "Bundle segmentations already exist at {$TRACT_SEGS}, skipping tract_segmentation."
  else
    TractSeg --output_type tract_segmentation -i "$PEAKS" -o "$OUTPUT_DIR" --brain_mask "$MASK"
  fi

  if [ -d $END_SEGS ]; then
    echo "Ending segmentations already exist at {$END_SEGS}, skipping endings_segmentation."
  else
    TractSeg --output_type endings_segmentation -i "$PEAKS" -o "$OUTPUT_DIR" --brain_mask "$MASK"
  fi

  if [ -d $TOMS ]; then
    echo "TOMs already exist at {$TOMS}, skipping TOM computation."
  else
    TractSeg --output_type TOM -i "$PEAKS" -o "$OUTPUT_DIR" --brain_mask "$MASK"
  fi

  if [ -d $TRACTS ]; then
    echo "Tracts already exist at {$TRACTS}, skipping tractography."
  else
    Tracking --algorithm prob -i "$PEAKS" -o "$OUTPUT_DIR"
  fi

  echo -e "\n\n===== Done processing $BASENAME =====\n\n"
done