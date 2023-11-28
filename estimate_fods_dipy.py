from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from common import get_unique_file_with_extension, get_dipy_to_mrtrix_permutation
from dipy.reconst.csdeconv import recursive_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
import numpy as np

print("WARNING: This script is being developed and there are known issues with it: https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments/issues/1")

# === Parse args ===

parser = argparse.ArgumentParser(description='estimates fiber orientation distribution images using constrained spherical deconvolution')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
parser.add_argument('masks_path', type=str, help='path to folder containing brain masks. the mask of x.nii is expected to be named x_mask.nii.gz')
parser.add_argument('dti_path', type=str, help='path to folder in which dti fit was saved')
parser.add_argument('output_dir', type=str, help='path to folder in which to save the estimated FODs')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
masks_path = Path(args.masks_path)
dti_path = Path(args.dti_path)
output_dir = Path(args.output_dir)
output_dir_fod = output_dir/'fod'
output_dir_fod.mkdir(exist_ok=True)

# === Iterate through images, performing CSD and saving results ===

for dwi_nii_directory in extracted_images_path.glob('*/*/*/dwi/'):

    nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii')

    subject_output_file = output_dir_fod/(nii_path.stem + '_fod.nii.gz')
    if subject_output_file.exists():
        print(f"Skipping {nii_path.stem} and assuming it was already processed since the following output file exists:\n{subject_output_file}")
        continue

    bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    bvecs[:,0] = -bvecs[:,0] # https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments/issues/7
    gtab = gradient_table(bvals, bvecs)

    mask_path = masks_path/(nii_path.stem + '_mask.nii.gz')

    data, affine, img = load_nifti(str(nii_path), return_img=True)
    mask_data, mask_affine, mask_img = load_nifti(str(mask_path), return_img=True)

    subject_dti_dir = dti_path/(nii_path.stem)
    if not subject_dti_dir.exists():
        print(f"Could not find directory {subject_dti_dir}. Did you complete the DTI fit for this subject?")
        print(f"(skipping {nii_path.stem})")
        continue
    fa_path = subject_dti_dir/(nii_path.stem + '_fa.nii.gz')
    fa_data, fa_affine, fa_img = load_nifti(str(fa_path), return_img=True)

    md_path = subject_dti_dir/(nii_path.stem + '_md.nii.gz')
    md_data, md_affine, md_img = load_nifti(str(md_path), return_img=True)

    print(f"applying CSD to estimate FOD for {nii_path.stem}...")

    # below approach and parameters are taken from the example https://dipy.org/documentation/1.1.0./examples_built/reconst_csd/
    wm_mask = (np.logical_or(fa_data >= 0.4, (np.logical_and(fa_data >= 0.15, md_data >= 0.0011))))
    response = recursive_response(gtab, data, mask=wm_mask, sh_order=8, peak_thr=0.01, init_fa=0.08, init_trace=0.0021, iter=8, convergence=0.001, parallel=True)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
    csd_fit = csd_model.fit(data, mask=mask_data)

    csd_shm_coeff_mrtrix = csd_fit.shm_coeff[:,:,:,get_dipy_to_mrtrix_permutation(8)]
    print(f"processed {nii_path.stem}\nsaving data...")
    save_nifti(subject_output_file, csd_shm_coeff_mrtrix, affine, img.header)
    print('saved!')


