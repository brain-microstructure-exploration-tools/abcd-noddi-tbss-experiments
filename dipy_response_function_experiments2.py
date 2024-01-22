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
from common import get_unique_file_with_extension, read_dipy_response, write_dipy_response
from dipy.reconst.csdeconv import recursive_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import AxSymShResponse
from dipy.reconst.shm import convert_sh_descoteaux_tournier
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
sh_order = 8
csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
csd_fit = csd_model.fit(data, mask=mask_data)
csd_shm_coeff = csd_fit.shm_coeff
csd_shm_coeff_mrtrix = convert_sh_descoteaux_tournier(csd_shm_coeff)

save_nifti(subject_output_file, csd_shm_coeff_mrtrix, affine, img.header)
print(f'saved {subject_output_file}')

# %% [markdown]
# ---
#
# **How do we average response functions when they are in the DIPY representation of a diffusion tensor?**
#
# There is a beautiful way of interpolating or averaging diffusion tensors which is based on log-euclidean metrics, described by [Arsigny 2006](https://doi.org/10.1002/mrm.20965). A log-euclidean metric is a metric on the space of $n\times n$ real positive definite symmetric matrices (called simply *tensors* in that work) that is defined by
# $$d(S_1,S_2)=|| \log(S_1) - \log(S_2) ||$$
# where $\log$ is the matrix logarithm and where $||\cdot||$ is just any old vector space norm on the vector space of symmetric matrices under addition. This makes sense because in fact the $\log$ of a tensor will always be a symmetric matrix. The simplest case is where $||\cdot||$ is the frobenius norm (in which case for a tensor you could write it as $||S||=\sqrt{\operatorname{Trace}(S^2)}$) and that is also a great case because you get very nice invariance properties from the metric. Anyway as we can see in Arsigny eqn [3] the Frechet mean of tensors with this metric is given by simply taking the ordinary additive in matrix-log space.
#
# Taking matrix log is super easy when you have a tensor: just diagonalize it to $S=RDR^{-1}$ and you are guaranteed $R$ orthogonal and $D$ having real positive values (the eigenvalues) down the diagonal. Then your log is
# $$
# \log(S) = R\log(D)R^{-1}
# $$
# where $\log(D)$ is simply a component-wise log of the diagonal matrix $D$.
#
# In my case the response functions used by dipy are represnted as eigenvalues $(\lambda_1, \lambda_2, \lambda_3=\lambda_2)$, with $\lambda_1$ being larger so that it is a prolate tensor, and also an overall non-diffusion weighted signal level $S_0$. Eigenvectors are not specified because a response function is always for a z-axis oriented pure fiber population. You can see [here](https://github.com/dipy/dipy/blob/9153a852a511b07ffc141becfbd9a96ca42e9a90/dipy/reconst/mcsd.py#L478-L481) or [here](https://github.com/dipy/dipy/blob/9153a852a511b07ffc141becfbd9a96ca42e9a90/dipy/reconst/csdeconv.py#L459-L461) that dipy just fixes the eigenvectors and [computes](https://github.com/dipy/dipy/blob/9153a852a511b07ffc141becfbd9a96ca42e9a90/dipy/sims/voxel.py#L321) a response signal from the diffusion tensor representation. So if I have a family of response functions given by $(\lambda_1^{(i)}, \lambda_2^{(i)}, S_0^{(i)})$ for subjects $i\in\{1,\ldots, N\}$ then
#
# - the diffusion tensors represented by the eigenvalue pairs are all oriented the same way and so we can think of all diffusion tensors as being diagonalized by the same $R$. therefore the log-euclidean mean of the diffusion tensors is 
#   $$\exp\left(\frac{1}{N}\sum_i R\log(\operatorname{diag}( \lambda_1^{(i)},\lambda_2^{(i)},\lambda_2^{(i)} ))R^{-1}\right)\\
#   =R\exp\left(\frac{1}{N}\sum_i\operatorname{diag}( \log(\lambda_1^{(i)}),\log(\lambda_2^{(i)}),\log(\lambda_2^{(i)}) )\right)R^{-1}\\
#   =R\exp\left(\operatorname{diag}\left( \log\left(\left(\prod_i\lambda_1^{(i)}\right)^{\frac{1}{N}}\right),\log\left(\left(\prod_i\lambda_2^{(i)}\right)^{\frac{1}{N}}\right),\log\left(\left(\prod_i\lambda_2^{(i)}\right)^{\frac{1}{N}}\right) \right)\right)R^{-1}\\
#   =R\operatorname{diag}\left( \left(\prod_i\lambda_1^{(i)}\right)^{\frac{1}{N}},\left(\prod_i\lambda_2^{(i)}\right)^{\frac{1}{N}},\left(\prod_i\lambda_2^{(i)}\right)^{\frac{1}{N}} \right)R^{-1}
#   $$
#   which is exactly the diffusion tensor represented by the geometric mean of eigenvalues across subjects. The signals
# - and the non-diffusion-weighted signals can simply be averaged (averaging them is IMO similar to averaging the 0th coefficient of the spherical harmonic representation of response functions).
#
# So actually a good way to do group averages is to simply take the geometric mean of the eigenvalues and arithmetic mean of the signal:
# $$
# \left(
# \left(\prod_{i=1}^N\lambda_1^{(i)}\right)^{\frac{1}{N}},
# \left(\prod_{i=1}^N\lambda_2^{(i)}\right)^{\frac{1}{N}},
# \frac{1}{N}\sum_{i=1}^N S_0^{(i)}
# \right)
# $$
# This is effectively achieving the log-euclidean mean of Arsigny 2006.

# %% [markdown]
# ---
#
# briefly testing out this method of aggregation

# %%
response_dir = Path('./test_response_agg')

# %%
# randomly jitter the response function we have to create an artifical family of response functions

for i in range(1000):
    new_response = (
        response[0] * np.exp(np.random.randn(3)/5), # the second and third eigenvals should always be equal but whatevs
        np.float32(response[1]) * np.exp(np.random.randn()/10)
    )
    write_dipy_response(new_response, response_dir/f"response{i}.txt")


# %%
def aggregate_dipy_response_functions(response_functions):
    """ Aggregate a collection of dipy dti response functions.

    This is useful for taking an average response function over a group when doing a population study.

    The method is to take the geometric mean of the eigenvalues and the arithmetic mean of the non-diffusion-weighted signal.
    The reason for taking the geometric mean of the eigenvalues is that this is exactly taking the Frechet mean of the diffusion
    tensors that are represented by the eigenvalues when the tensors are treated as elements of the standard log-euclidean metric space
    described in 

        Arsigny, Vincent, et al. "Log‐Euclidean metrics for fast and simple calculus on diffusion tensors."
        Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance
        in Medicine 56.2 (2006): 411-421.

    Args:
        response_functions: a sequence of response functions. a response function is the sort of thing
            returned by dipy.reconst.csdeconv.response_from_mask_ssst
    Returns: an aggregate response function.
    """
    
    evals_array = np.array([rf[0] for rf in response_functions])
    S0_array = np.array([rf[1] for rf in response_functions])
    return np.exp(np.log(evals_array).mean(axis=0)), np.mean(S0_array)

def aggregate_dipy_response_functions_workflow(response_dir, output_path):
    """ Aggregate a collection of dipy dti response functions in a directory and write the output to a file.

    This is useful for taking an average response function over a group when doing a population study.

    See aggregate_dipy_response_functions for more details.

    Args:
        response_dir: a directory containing response functions as text files. (for example they
            could be written out by write_dipy_response)
        output_path: file path at which to save the aggregate response
    """
    response_file_paths = list(response_dir.glob("*.txt"))
    if len(response_file_paths) == 0 :
        raise FileNotFoundError(f"No response files found in {response_dir}")
    response_functions = [read_dipy_response(response_file_path) for response_file_path in response_file_paths]
    aggregate_response_function =  aggregate_dipy_response_functions(response_functions)
    write_dipy_response(aggregate_response_function, output_path)


# %%
aggregate_dipy_response_functions_workflow(response_dir, "/home/ebrahimebrahim/Desktop/agg.txt")
