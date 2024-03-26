from pathlib import Path
import argparse
import conversion  # Get this from https://github.com/pnlbwh/conversion
from common import get_unique_file_with_extension

parser = argparse.ArgumentParser(
    description="generates nrrd headers for extracted ABCD diffusion images"
)
parser.add_argument(
    "extracted_images_path",
    type=str,
    help="path to folder in which downloaded ABCD images were extracted",
)
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)

for dwi_nii_directory in extracted_images_path.glob("*/*/*/dwi/"):
    nii_file_path = get_unique_file_with_extension(dwi_nii_directory, "nii.gz")
    # (Note that getting a unique file like this wouldn't work in general on an ABCD download if someone extracted everything to the same
    # target folder instead of creating one folder for each archive as I did.)
    output_nhdr_path = dwi_nii_directory / (nii_file_path.stem + ".nhdr")
    conversion.nhdr_write(
        str(nii_file_path),
        str(get_unique_file_with_extension(dwi_nii_directory, "bval")),
        str(get_unique_file_with_extension(dwi_nii_directory, "bvec")),
        str(output_nhdr_path),
    )
