# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This is a notebook to explore ways of adding response function averaging to `estimate_fods_dipy.py`.
#
# This also looks at how to do multi shell single tissue csd using `dipy.reconst.csdeconv`.

# %%
from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from common import get_unique_file_with_extension
from dipy.reconst.csdeconv import recursive_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import AxSymShResponse
import numpy as np

# %%
extracted_images_path = Path('extracted_images/')
masks_path = Path('hdbet_output/')
dti_path = Path('dti_output/')
output_dir = Path('csd_output_dipy/')
output_dir_fod = output_dir/'fod'
output_dir_fod.mkdir(exist_ok=True)

# %%
dwi_nii_directory = np.random.choice(list(extracted_images_path.glob('*/*/*/dwi/')))
dwi_nii_directory = Path('extracted_images/NDARINV1JXDFV9Z_baselineYear1Arm1_ABCD-MPROC-DTI_20161206184105/sub-NDARINV1JXDFV9Z/ses-baselineYear1Arm1/dwi/')

# %%
nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii')
print(f"INPUT_DIR={dwi_nii_directory}")
print(f"STEM={nii_path.stem}")
subject_output_file = output_dir_fod/(nii_path.stem + '_fod.nii.gz')
if subject_output_file.exists():
        print(f"Warning: output for subject {nii_path.stem} already exists at the file\n\t{subject_output_file}")

mask_path = masks_path/(nii_path.stem + '_mask.nii.gz')

data, affine, img = load_nifti(str(nii_path), return_img=True)
mask_data, mask_affine, mask_img = load_nifti(str(mask_path), return_img=True)

bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')
bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))

# %%
bvecs[:,0] = -bvecs[:,0]
print("Negating the x coordinate of all the b-vectors!")

# %%
gtab = gradient_table(bvals, bvecs)

subject_dti_dir = dti_path/(nii_path.stem)
if not subject_dti_dir.exists():
    raise Exception(f"Could not find directory {subject_dti_dir}. Did you complete the DTI fit for this subject?")
fa_path = subject_dti_dir/(nii_path.stem + '_fa.nii.gz')
fa_data, fa_affine, fa_img = load_nifti(str(fa_path), return_img=True)

md_path = subject_dti_dir/(nii_path.stem + '_md.nii.gz')
md_data, md_affine, md_img = load_nifti(str(md_path), return_img=True)

# %%
# Warning this cell takes a long time to run and is not needed for the things that follow.

# below approach and parameters are taken from the example https://dipy.org/documentation/1.1.0./examples_built/reconst_csd/
# wm_mask = (np.logical_or(fa_data >= 0.4, (np.logical_and(fa_data >= 0.15, md_data >= 0.0011))))
# response = recursive_response(gtab, data, mask=wm_mask, sh_order=8, peak_thr=0.01, init_fa=0.08, init_trace=0.0021, iter=2, convergence=0.1, parallel=True)

# %% [markdown]
# ----
#
# No more using `recursive_response` because I am not convinced that it handles multi-shell data correctly. Not using `dipy.reconst.mcsd` for multishell data because I don't understand what's going on with volume fractions with that approach (they come out not making sense), the code isn't as clear to me what it's doing, and I suspect it's too new and not sufficiently tested for me to use without issues. Instead what I'm going to try is to use multiple shells without multiple tissue compartments and still use `dipy.reconst.csdeconv` to do it. [This](https://github.com/dipy/dipy/discussions/3037#discussioncomment-8148506) is the advice I am following.

# %%
from dipy.reconst.csdeconv import mask_for_response_ssst, response_from_mask_ssst
import matplotlib.pyplot as plt
import warnings

# %%
low_b_mask = gtab.bvals <= 1200
gtab_low_b = gradient_table(bvals[low_b_mask], bvecs[low_b_mask])
data_low_b = data[...,low_b_mask]

# %%
# roi center based on brain mask

i,j,k = np.where(mask_data)
roi_center=np.round([i.mean(), j.mean(), k.mean()]).astype(int) # COM of mask
roi_radii=[(indices.max() - indices.min())//4 for indices in (i,j,k)]

# %%
mask_for_response = mask_for_response_ssst(
    gtab_low_b,
    data_low_b,
    roi_center = roi_center,
    roi_radii = roi_radii,
    fa_thr=0.8,
)
mask_for_response *= mask_data # ensure we stay inside brain mask (almost certainly we already were but just in case)

# %%
if mask_for_response.sum() < 100:
    warnings.warn("There are less than 100 voxels in the mask used to estimate the response function.")
    # If this happens maybe we need to decrease fa_thr or check what might be the issue

# %%
response, ratio = response_from_mask_ssst(
    gtab_low_b,
    data_low_b,
    mask_for_response,
)

# %%
if ratio > 0.3:
    warnings.warn("Ratio of response diffusion tensor eigenvalues is greater than 0.3. For a response function we expect more prolateness. Something may be wrong.")

# %%
subject_output_file

# %%
sh_order = 8
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
csd_fit = csd_model.fit(data, mask=mask_data)
csd_shm_coeff = csd_fit.shm_coeff
subject_output_file_alt = subject_output_file.parent / f"{subject_output_file.name.split('.')[0]}.nii.gz"
from dipy.reconst.shm import sph_harm_ind_list
def get_dipy_to_mrtrix_permutation(sh_order):
    m,l = sph_harm_ind_list(sh_order)
    basis_indices = list(zip(l,m)) # dipy basis ordering
    dimensionality = len(basis_indices)
    basis_indices_permuted = list(zip(l,-m)) # mrtrix basis ordering
    permutation = [basis_indices.index(basis_indices_permuted[i]) for i in range(dimensionality)] # dipy to mrtrix permution
    return permutation
csd_shm_coeff_mrtrix = csd_shm_coeff[:,:,:,get_dipy_to_mrtrix_permutation(sh_order)]

save_nifti(subject_output_file_alt, csd_shm_coeff_mrtrix, affine, img.header)
print(f'saved {subject_output_file_alt}')

# %%

# %% [markdown]
# old stuff below
#
# ---

# %%
sh_order = 8
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
csd_fit = csd_model.fit(data, mask=mask_data)
csd_shm_coeff = csd_fit.shm_coeff
subject_output_file_alt = subject_output_file.parent / f"{subject_output_file.name.split('.')[0]}_dipyTaxResponse_mrtrixConverted_bvecXflipped.nii.gz"
from dipy.reconst.shm import sph_harm_ind_list
def get_dipy_to_mrtrix_permutation(sh_order):
    m,l = sph_harm_ind_list(sh_order)
    basis_indices = list(zip(l,m)) # dipy basis ordering
    dimensionality = len(basis_indices)
    basis_indices_permuted = list(zip(l,-m)) # mrtrix basis ordering
    permutation = [basis_indices.index(basis_indices_permuted[i]) for i in range(dimensionality)] # dipy to mrtrix permution
    return permutation
csd_shm_coeff_mrtrix = csd_shm_coeff[:,:,:,get_dipy_to_mrtrix_permutation(sh_order)]

save_nifti(subject_output_file_alt, csd_shm_coeff_mrtrix, affine, img.header)
print(f'saved {subject_output_file_alt}')

# %% [markdown]
# Below are legacy parts of this notebook.
#
# ----

# %%
with open(Path('csd_output_mrtrix/average_response.txt')) as f:
    mrtrix_response_function_string = f.readlines()[-1]
print(list(map(float,mrtrix_response_function_string.strip().split(' '))))

# %%
response_mrtrix = AxSymShResponse(300,np.array(list(map(float,mrtrix_response_function_string.strip().split(' ')))))

# %% [markdown]
# The 300 looks like it's coming from nowhere, but actually in dipy, the `response.S0` is only saved [here](https://github.com/dipy/dipy/blob/1d558fce90d38e71cc16ed4c7e48bf1116ae459b/dipy/reconst/csdeconv.py#L262) when you construct the `ConstrainedSphericalDeconvModel` object. When you call the `fit` method you can see [here](https://github.com/dipy/dipy/blob/1d558fce90d38e71cc16ed4c7e48bf1116ae459b/dipy/reconst/csdeconv.py#L289-L293) that the S0 value not used anywhere. It's only used in the `predict` method, which we are not really using here because we are not trying to simulate signals right now. So I think for the CSD fit to generate a FOD, the S0 value doesn't really matter.

# %%
sh_order = 8
csd_model = ConstrainedSphericalDeconvModel(gtab, response_mrtrix, sh_order=sh_order)

# %%
csd_fit = csd_model.fit(data, mask=mask_data)

# %%
subject_output_file_alt = subject_output_file.parent / f"{subject_output_file.name.split('.')[0]}_mrtrixResponse.nii.gz"
subject_output_file_alt

# %%
csd_fit

# %%
csd_shm_coeff = csd_fit.shm_coeff

from dipy.reconst.shm import sph_harm_ind_list
def get_dipy_to_mrtrix_permutation(sh_order):
    m,l = sph_harm_ind_list(sh_order)
    basis_indices = list(zip(l,m)) # dipy basis ordering
    dimensionality = len(basis_indices)
    basis_indices_permuted = list(zip(l,-m)) # mrtrix basis ordering
    permutation = [basis_indices.index(basis_indices_permuted[i]) for i in range(dimensionality)] # dipy to mrtrix permution
    return permutation
csd_shm_coeff_mrtrix = csd_shm_coeff[:,:,:,get_dipy_to_mrtrix_permutation(sh_order)]
subject_output_file_alt2 = subject_output_file.parent / f"{subject_output_file.name.split('.')[0]}_mrtrixResponse_mrtrixConverted.nii.gz"

save_nifti(subject_output_file_alt, csd_shm_coeff, affine, img.header)
print(f'saved {subject_output_file_alt}')
save_nifti(subject_output_file_alt2, csd_shm_coeff_mrtrix, affine, img.header)
print(f'saved {subject_output_file_alt2}')

# %% [markdown]
# I visually inspected three images using `mrview`:
# - the mrtrix reconstruction using the mrtrix estimated average response function
# - the dipy reconstruction using the dipy estimated average response function
# - the dipy reconstruction using the mrtrix estimated average response function
#
# The latter two were both pretty bad looking (e.g. qualtiatively when looking at corpus callosum where the FODs should be extra sharp and clean) so I don't think my issue is only with response function generation.
#
# Maybe the dipy spherical harmonic basis order is different? So when I save the 45-channel coefficients list perhaps it's being misinterpreted by mrview? Based on [my investigation](https://github.com/dipy/dipy/discussions/2861#discussioncomment-7256439), no:

# %%
from dipy.reconst.shm import sph_harm_ind_list
m,l = sph_harm_ind_list(8)
print(list(zip(l,m)))
