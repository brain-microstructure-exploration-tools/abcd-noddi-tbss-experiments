#!/bin/bash
set -e

if [[ $# -ne 3 ]]; then
  echo "Usage: ./generate_population_template.sh <FOD_DIR> <MASK_DIR> <DESTINATION_PATH>"
  echo "  where FOD_DIR is the path to the fiber orientation distributions generated using one of the estimate_fods scripts"
  echo "  and MASK_DIR is the destination folder you used during brain extraction"
  echo "  and DESTINATION_PATH is the folder in which you want to save the template construction results."
  exit
fi

# This is the voxel spacing for the output, fixed to the known spacing of ABCD images
# See https://abcdstudy.org/images/Protocol_Imaging_Sequences.pdf
VOXEL_SIZE=1.7

# We assume that the images are already rigidly co-registered (ABCD images are)
# It may be sensible to still include an affine registration and change this to affine_nonlinear
# because we might expect very slight brain volume increases... but I have found this
# to cause more trouble than it is worth for the small overall brain volume variation.
REG_TYPE=nonlinear

FOD_DIR=$1
MASK_INPUT_DIR=$2
OUTPUT_DIR=$3
FOD_STAGING_DIR=$OUTPUT_DIR/input_fods/
MASK_STAGING_DIR=$OUTPUT_DIR/input_masks/

# create a staging directory where mrtrix will find FOD images
mkdir -p $FOD_STAGING_DIR
for FILE in "$FOD_DIR"/*_fod.nii.gz; do
    BASENAME=$(basename $FILE _fod.nii.gz)
    ln -sr $FILE ${FOD_STAGING_DIR}/${BASENAME}.nii.gz
done

# create a staging directory where mrtrix will find masks
mkdir -p $MASK_STAGING_DIR
for FILE in "$MASK_INPUT_DIR"/*_mask.nii.gz; do
    BASENAME=$(basename $FILE _mask.nii.gz)
    ln -sr $FILE ${MASK_STAGING_DIR}/${BASENAME}.nii.gz
done

# run the mrtrix command
population_template $FOD_STAGING_DIR $OUTPUT_DIR/template.nii.gz -mask_dir $MASK_STAGING_DIR -warp_dir $OUTPUT_DIR/warps -transformed_dir $OUTPUT_DIR/transformed_images -template_mask $OUTPUT_DIR/template_mask.nii.gz -voxel_size $VOXEL_SIZE -type $REG_TYPE
