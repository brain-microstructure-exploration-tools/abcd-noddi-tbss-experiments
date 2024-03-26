from pathlib import Path
import argparse

import dipy.io.image
import dipy.io.gradients

from common import get_unique_file_with_extension


parser = argparse.ArgumentParser(
    description="generates average b0 images for ABCD diffusion images"
)
parser.add_argument(
    "extracted_images_path",
    type=str,
    help="path to folder in which downloaded ABCD images were extracted",
)
parser.add_argument(
    "output_dir", type=str, help="path to folder in which to save the b0 average images"
)
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
output_dir = Path(args.output_dir)

for dwi_nii_directory in extracted_images_path.glob("*/*/*/dwi/"):
    nii_path = get_unique_file_with_extension(dwi_nii_directory, "nii.gz")
    # (Note that getting a unique file like this wouldn't work in general on an ABCD download if someone extracted everything to the same
    # target folder instead of creating one folder for each archive as I did.)
    bval_path = get_unique_file_with_extension(dwi_nii_directory, "bval")
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, "bvec")

    output_image_path = output_dir / (nii_path.name)

    if output_image_path.exists():
        print(f"found {nii_path.name}; skipping.")
        continue

    data, affine, img = dipy.io.image.load_nifti(str(nii_path), return_img=True)
    bvals, bvecs = dipy.io.read_bvals_bvecs(str(bval_path), str(bvec_path))

    gtab = dipy.io.gradients.gradient_table(bvals, bvecs)
    mean_of_b0s = data[:, :, :, gtab.b0s_mask].mean(axis=3)

    dipy.io.image.save_nifti(output_image_path, mean_of_b0s, affine, img.header)
    print(f"processed {nii_path.name}")
