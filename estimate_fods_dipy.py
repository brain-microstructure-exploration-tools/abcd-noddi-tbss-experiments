from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from common import get_unique_file_with_extension, write_dipy_response, read_dipy_response, aggregate_dipy_response_functions_workflow
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.shm import convert_sh_descoteaux_tournier
import numpy as np
from dipy.reconst.csdeconv import mask_for_response_ssst, response_from_mask_ssst
import warnings
import pandas as pd

# === Parse args ===

parser = argparse.ArgumentParser(description='estimates fiber orientation distribution images using constrained spherical deconvolution')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
parser.add_argument('masks_path', type=str, help='path to folder containing brain masks. the mask of x.nii is expected to be named x_mask.nii.gz')
parser.add_argument('output_dir', type=str, help='path to folder in which to save the estimated FODs')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
masks_path = Path(args.masks_path)
output_dir = Path(args.output_dir)
output_dir_fod = output_dir/'fod'
output_dir_response = output_dir/'subject_response_functions'
average_response_dir = output_dir/'group_response_functions'
output_dir_fod.mkdir(exist_ok=True)
output_dir_response.mkdir(exist_ok=True)
average_response_dir.mkdir(exist_ok=True)
site_df = pd.read_csv(extracted_images_path/'site_table.csv')

# === Iterate through images, estimating response functions. ===

for dwi_nii_directory in extracted_images_path.glob('*/*/*/dwi/'):

    nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii.gz')
    # (Note that getting a unique file like this wouldn't work in general on an ABCD download if someone extracted everything to the same
    # target folder instead of creating one folder for each archive as I did.)
    basename = nii_path.name.split('.')[0]

    response_output_file = output_dir_response/(basename + '.txt')
    if response_output_file.exists():
        print(f"Skipping response function estimate for {basename} since the following output file exists:\n{response_output_file}")
        continue

    bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    bvecs[:,0] = -bvecs[:,0] # https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments/issues/7
    gtab = gradient_table(bvals, bvecs)

    mask_path = masks_path/(basename + '_mask.nii.gz')

    data, affine, img = load_nifti(str(nii_path), return_img=True)
    mask_data, mask_affine, mask_img = load_nifti(str(mask_path), return_img=True)

    print(f"estimating response function for {basename}...")

    # b-values above 1200 aren't great for DTI estimation. dipy uses DTI to model response functions.
    low_b_mask = gtab.bvals <= 1200
    gtab_low_b = gradient_table(bvals[low_b_mask], bvecs[low_b_mask])
    data_low_b = data[...,low_b_mask]

    # determine roi center based on brain mask
    i,j,k = np.where(mask_data)
    roi_center=np.round([i.mean(), j.mean(), k.mean()]).astype(int) # COM of mask
    roi_radii=[(indices.max() - indices.min())//4 for indices in (i,j,k)]

    # get mask of voxels to use for response estimate
    mask_for_response = mask_for_response_ssst(
        gtab_low_b,
        data_low_b,
        roi_center = roi_center,
        roi_radii = roi_radii,
        fa_thr=0.8,
    )
    mask_for_response *= mask_data # ensure we stay inside brain mask (almost certainly we already were but just in case)
    if mask_for_response.sum() < 100:
        warnings.warn("There are less than 100 voxels in the mask used to estimate the response function.")
        # If this happens maybe we need to decrease fa_thr or check what might be the issue
    
    # perform response function estimate using the selected voxels
    response, ratio = response_from_mask_ssst(
        gtab_low_b,
        data_low_b,
        mask_for_response,
    )

    if ratio > 0.3:
        warnings.warn("Ratio of response diffusion tensor eigenvalues is greater than 0.3. For a response function we expect more prolateness. Something may be wrong.")

    write_dipy_response(response, response_output_file)

# === Aggregate response functions ===
print("aggregating estimated response functions by study site....")
for site,group in site_df.groupby('site_id_l'):
    aggregate_dipy_response_functions_workflow(
        [output_dir_response/(basename + '.txt') for basename in group.basename],
        average_response_dir/(str(site)+'.txt')
    )


# === Iterate through images, performing CSD to compute FODs. ===
    
for dwi_nii_directory in extracted_images_path.glob('*/*/*/dwi/'):

    nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii.gz')
    basename = nii_path.name.split('.')[0]

    subject_output_file = output_dir_fod/(basename + '_fod.nii.gz')
    if subject_output_file.exists():
        print(f"Skipping {basename} and assuming it was already processed since the following output file exists:\n{subject_output_file}")
        continue

    bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    bvecs[:,0] = -bvecs[:,0] # https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments/issues/7
    gtab = gradient_table(bvals, bvecs)

    mask_path = masks_path/(basename + '_mask.nii.gz')

    data, affine, img = load_nifti(str(nii_path), return_img=True)
    mask_data, mask_affine, mask_img = load_nifti(str(mask_path), return_img=True)

    site = site_df[site_df.basename == basename].site_id_l.item()
    print(f"applying CSD to estimate FOD for {basename} (site {site})...")

    # below approach and parameters are taken from the example https://dipy.org/documentation/1.1.0./examples_built/reconst_csd/
    response = read_dipy_response(average_response_dir/(str(site)+'.txt'))
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
    csd_fit = csd_model.fit(data, mask=mask_data)

    csd_shm_coeff_mrtrix = convert_sh_descoteaux_tournier(csd_fit.shm_coeff)
    print(f"processed {basename}\nsaving data...")
    save_nifti(subject_output_file, csd_shm_coeff_mrtrix, affine, img.header)
    print('saved!')