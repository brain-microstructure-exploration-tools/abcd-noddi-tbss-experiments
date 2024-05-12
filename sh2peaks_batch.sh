#!/bin/bash
set -e

if [[ $# -ne 3 ]]; then
  echo "Usage: ./sh2peaks_batch.sh <CSD_OUTPUT_PATH> <MASKS_PATH> <DESTINATION_PATH>"
  echo "  where CSD_OUTPUT_PATH is the path to the folder into which CSD output was placed"
  echo "  and MASKS_PATH is the folder in which you saved the brain masks"
  echo "  and DESTINATION_PATH is the folder in which you want to save peaks."
  exit
fi

CSD_OUTPUT_PATH=$1
MASKS_PATH=$2
PEAKS_PATH=$3

FOD_PATH=$CSD_OUTPUT_PATH/fod

if [ ! -d "$FOD_PATH" ]; then
    echo "FOD directory not found"
    exit 1
fi

mkdir -p $PEAKS_PATH

# Estimate a response function for each image
for FOD in "$FOD_PATH"/*_fod.nii.gz; do
  BASENAME=$(basename $FOD _fod.nii.gz)
  MASK=$MASKS_PATH/${BASENAME}_mask.nii.gz
  OUTPUT="$PEAKS_PATH"/${BASENAME}_peaks.nii.gz
  if [ -f "$OUTPUT" ]; then
    echo "Skipping ${BASENAME} because this file already exists: ${OUTPUT}"
    continue
  fi
  sh2peaks $FOD $OUTPUT -mask $MASK
done