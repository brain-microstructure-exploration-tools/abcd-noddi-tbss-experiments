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
# The goal of this notebook is to examine a FOD output and visualize it without using any purpose made visualization tool, requiring me to prove that I understand how to interpret the FOD coefficients.

# %%
# !pip install "itkwidgets[notebook]==1.0a37"
# !pip install "numpy==1.23.5"

# %%
from pathlib import Path
import numpy as np
import nibabel as nib
from itkwidgets import view
from scipy.special import sph_harm

# %%
fod_path = Path('csd_output_mrtrix_msmt/fod/WM/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi_wmfod.nii.gz')
dwi_path = Path('extracted_images/NDARINV1JXDFV9Z_baselineYear1Arm1_ABCD-MPROC-DTI_20161206184105/sub-NDARINV1JXDFV9Z/ses-baselineYear1Arm1/dwi/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi.nii')

# %%
dwi = nib.load(dwi_path)
dwi_array = dwi.get_fdata()

# %%
img = nib.load(fod_path)
img_array = img.get_fdata()

# %%
num_theta = 100
num_phi = 100
thetas = np.linspace(0,np.pi,num=num_theta,endpoint=False)
phis = np.linspace(0,2*np.pi,num=num_phi,endpoint=False)
th, ph = np.meshgrid(thetas, phis)

# %%
sphere_points = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1).reshape(-1,3)


# %%
def sph_harm_l_m(l_max):
    for l in range(0,l_max+1,2):
        for m in range(-l,l+1):
            yield l,m
l, m = np.array(list(sph_harm_l_m(8)),dtype=int).T


# %%
# follows https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html#storage-conventions
def sph_harm_real(m,l,ph,th):
    y = sph_harm(m,l,ph,th)
    ynegm = sph_harm(-m,l,ph,th)
    y = np.where(m<0,np.sqrt(2)*np.imag(ynegm),y)
    y = np.where(m>0,np.sqrt(2)*np.real(y),y)
    return np.real_if_close(y)


# %%
sph_harm_vals = sph_harm_real(m[:,np.newaxis],l[:,np.newaxis],ph.reshape(1,-1),th.reshape(1,-1))


# %%
def view_voxel(i,j,k):
    fod_vals = (img_array[i,j,k] @ sph_harm_vals)
    scaled_sphere_pts = fod_vals[:,np.newaxis] * sphere_points
    view(np.concatenate([scaled_sphere_pts]*3,axis=0))


# %%
dwi_array[79,87,84,0]

# %% [markdown]
# Hmm I can't even seem to get array values to match what I see in mrview when I do
#
# ```sh
# mrview extracted_images/NDARINV1JXDFV9Z_baselineYear1Arm1_ABCD-MPROC-DTI_20161206184105/sub-NDARINV1JXDFV9Z/ses-baselineYear1Arm1/dwi/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi.nii -odf.load_sh csd_output_mrtrix_msmt/fod/WM/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi_wmfod.nii.gz
# ```

# %%
view_voxel(84,78,85)
