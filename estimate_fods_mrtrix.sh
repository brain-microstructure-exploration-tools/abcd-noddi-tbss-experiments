#!/bin/bash
set -e

if [[ $# -ne 3 ]]; then
  echo "Usage: ./estimate_fods_mrtrix.sh <EXTRACTED_IMAGES_PATH> <MASKS_PATH> <DESTINATION_PATH>"
  echo "  where EXTRACTED_IMAGES_PATH is the path to the folder in which downloaded ABCD images were extracted"
  echo "  and MASKS_PATH is the folder in which you saved the brain masks"
  echo "  and DESTINATION_PATH is the folder in which you want to save FODs and response functions."
  exit
fi

EXTRACTED_IMAGES_PATH=$1
MASKS_PATH=$2
DESTINATION_PATH=$3

RESPONSE_PATH=$DESTINATION_PATH/response_functions
AVERAGE_RESPONSE=$DESTINATION_PATH/average_response.txt
FOD_PATH=$DESTINATION_PATH/fod

mkdir -p $RESPONSE_PATH
mkdir -p $FOD_PATH

# Estimate a response function for each image
for DWI_PATH in "$EXTRACTED_IMAGES_PATH"/*/*/*/dwi/; do
  DWI=$DWI_PATH/*.nii # we assume there is a unique .nii file in there, same for bval, bvec
  BVAL=$DWI_PATH/*.bval
  BVEC=$DWI_PATH/*.bvec
  BASENAME=$(basename $DWI .nii)
  MASK=$MASKS_PATH/${BASENAME}_mask.nii.gz
  dwi2response tournier $DWI $RESPONSE_PATH/$BASENAME.txt -fslgrad $BVEC $BVAL -mask $MASK
done

# Taking advice from
# https://mrtrix.readthedocs.io/en/latest/fixel_based_analysis/st_fibre_density_cross-section.html?highlight=average_response#computing-an-average-white-matter-response-function
# we compute an average response function
responsemean $RESPONSE_PATH/*.txt $AVERAGE_RESPONSE

# Estimate a FOD for each image
for DWI_PATH in "$EXTRACTED_IMAGES_PATH"/*/*/*/dwi/; do
  DWI=$DWI_PATH/*.nii # we assume there is a unique .nii file in there, same for bval, bvec
  BVAL=$DWI_PATH/*.bval
  BVEC=$DWI_PATH/*.bvec
  BASENAME=$(basename $DWI .nii)
  MASK=$MASKS_PATH/${BASENAME}_mask.nii.gz
  dwi2fod csd $DWI $AVERAGE_RESPONSE $FOD_PATH/${BASENAME}_fod.nii.gz -fslgrad $BVEC $BVAL -mask $MASK
done
