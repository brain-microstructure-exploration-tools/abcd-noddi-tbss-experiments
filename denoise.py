from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from common import get_unique_file_with_extension
from dipy.denoise.patch2self import patch2self
import multiprocessing


parser = argparse.ArgumentParser(description='denoise extraced DWIs using Patch2Self. REPLACES ORIGINAL IMAGES.')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)

def denoise_dwi_nii_directory(dwi_nii_directory):
    nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii.gz')
    # (Note that getting a unique file like this wouldn't work in general on an ABCD download if someone extracted everything to the same
    # target folder instead of creating one folder for each archive as I did.)
    basename = nii_path.name.split('.')[0]
    bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs)

    data, affine, img = load_nifti(str(nii_path), return_img=True)

    print(f"{basename}: Denoising...")

    data_denoised = patch2self(
        data,
        bvals,
        shift_intensity=True,
        clip_negative_vals=False,
        b0_threshold=5,
        verbose=False
    )

    print(f"{basename}: Saving...")
    save_nifti(nii_path, data_denoised, affine, img.header)
    print(f'{basename}: Done! Replaced {nii_path}')


pool = multiprocessing.Pool(processes=3)
pool.map(denoise_dwi_nii_directory, extracted_images_path.glob('*/*/*/dwi/'))