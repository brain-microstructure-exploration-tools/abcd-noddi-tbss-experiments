#!/bin/bash
set -e

# The base directory to which the archive files were downloaded
ARCHIVES_PATH="/home/ebrahim/data/abcd/Package_1217694/fmriresults01/abcd-mproc-release5"

if [[ $# -ne 1 ]]; then
  echo "Usage: ./01_extract_images.sh  <DESTINATION_PATH>"
  echo "  where DESTINATION_PATH is the directory to which you want to dump the extracted images."
  exit
fi

# The directory in which we want to put the extracted images
DESTINATION_PATH=$1


for A in "$ARCHIVES_PATH"/*.tgz ; do
  D=$DESTINATION_PATH/$(basename $A .tgz)
  if [ -d $D ]
  then
    continue
  fi
  mkdir $D
  tar -xvf $A --directory $D
done
