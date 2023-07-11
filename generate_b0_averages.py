from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table

parser = argparse.ArgumentParser(description='generates average b0 images for ABCD diffusion images')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
parser.add_argument('output_dir', type=str, help='path to folder in which to save the b0 average images')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
output_dir = Path(args.output_dir)

def get_unique_file_with_extension(directory_path : Path, extension : str) -> Path:
    l = list(directory_path.glob(f'*.{extension}'))
    if len(l) == 0 :
        raise Exception(f"No {extension} file was found in {directory_path}")
    if len(l) > 1 :
        raise Exception(f"Multiple {extension} files were found in {directory_path}")
    return l[0]

for dwi_nii_directory in extracted_images_path.glob('*/*/*/dwi/'):

    nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii')
    bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

    output_image_path = output_dir/(nii_path.stem + '.nii.gz')

    data, affine, img = load_nifti(str(nii_path), return_img=True)
    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs)

    mean_of_b0s = data[:,:,:,gtab.b0s_mask].mean(axis=3)

    save_nifti(output_image_path, mean_of_b0s, affine, img.header)
    print(f"processed {nii_path.stem}")