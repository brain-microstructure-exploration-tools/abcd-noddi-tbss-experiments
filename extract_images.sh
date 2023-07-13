#!/bin/bash
set -e

if [[ $# -ne 2 ]]; then
  echo "Usage: ./01_extract_images.sh <ARCHIVES_PATH> <DESTINATION_PATH>"
  echo "  where ARCHIVES_PATH is the directory to which directory to which the archive files were downloaded"
  echo "  and DESTINATION_PATH is the directory to which you want to dump the extracted images."
  exit
fi

# The base directory to which the archive files were downloaded
ARCHIVES_PATH=$1

# The directory in which we want to put the extracted images
DESTINATION_PATH=$2


for A in "$ARCHIVES_PATH"/*.tgz ; do
  D=$DESTINATION_PATH/$(basename $A .tgz)
  if [ -d $D ]
  then
    continue
  fi
  mkdir $D
  tar -xvf $A --directory $D
done
