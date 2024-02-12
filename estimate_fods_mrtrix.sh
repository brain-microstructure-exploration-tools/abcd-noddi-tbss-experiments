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

FOD_PATH=$DESTINATION_PATH/fod
RESPONSE_PATH=$DESTINATION_PATH/subject_response_functions
AVERAGE_RESPONSE_WM_DIR=$DESTINATION_PATH/group_response_functions/WM/
# AVERAGE_RESPONSE_GM_DIR=$DESTINATION_PATH/group_response_functions/GM/ # (not using these so commented out)
# AVERAGE_RESPONSE_CSF_DIR=$DESTINATION_PATH/group_response_functions/CSF/

mkdir -p $RESPONSE_PATH/WM
mkdir -p $RESPONSE_PATH/GM
mkdir -p $RESPONSE_PATH/CSF
mkdir -p $AVERAGE_RESPONSE_WM_DIR
# mkdir -p $AVERAGE_RESPONSE_GM_DIR
# mkdir -p $AVERAGE_RESPONSE_CSF_DIR
mkdir -p $FOD_PATH

SITE_TABLE=$EXTRACTED_IMAGES_PATH/site_table.csv

# Estimate a response function for each image
for DWI_PATH in "$EXTRACTED_IMAGES_PATH"/*/*/*/dwi/; do
  DWI=$DWI_PATH/*.nii.gz # we assume there is a unique .nii.gz file in there, same for bval, bvec
  BVAL=$DWI_PATH/*.bval
  BVEC=$DWI_PATH/*.bvec
  BASENAME=$(basename $DWI .nii.gz)
  MASK=$MASKS_PATH/${BASENAME}_mask.nii.gz
  dwi2response dhollander $DWI $RESPONSE_PATH/WM/$BASENAME.txt $RESPONSE_PATH/GM/$BASENAME.txt $RESPONSE_PATH/CSF/$BASENAME.txt -fslgrad $BVEC $BVAL -mask $MASK
done

# Taking advice from
# https://mrtrix.readthedocs.io/en/latest/fixel_based_analysis/st_fibre_density_cross-section.html?highlight=average_response#computing-an-average-white-matter-response-function
# we compute an average response function
# But we do this within each study site.
SITES=$(python site_table_helper.py $SITE_TABLE get_sites)
for SITE in $SITES; do
  SUBJECT_BASENAMES=$(python site_table_helper.py $SITE_TABLE get_subjects --site $SITE)
  for BASENAME in $SUBJECT_BASENAMES; do
    SUBJECT_RESPONSES+="$RESPONSE_PATH/WM/$BASENAME.txt "
  done
  responsemean $SUBJECT_RESPONSES $AVERAGE_RESPONSE_WM_DIR/$SITE.txt -legacy
done

# Estimate a FOD for each image
for DWI_PATH in "$EXTRACTED_IMAGES_PATH"/*/*/*/dwi/; do
  DWI=$DWI_PATH/*.nii.gz # we assume there is a unique .nii.gz file in there, same for bval, bvec
  BVAL=$DWI_PATH/*.bval
  BVEC=$DWI_PATH/*.bvec
  BASENAME=$(basename $DWI .nii.gz)
  MASK=$MASKS_PATH/${BASENAME}_mask.nii.gz
  SITE=$(python site_table_helper.py $SITE_TABLE get_site --subject $BASENAME)
  dwi2fod msmt_csd $DWI $AVERAGE_RESPONSE_WM_DIR/$SITE.txt $FOD_PATH/${BASENAME}_fod.nii.gz -fslgrad $BVEC $BVAL -mask $MASK
done
