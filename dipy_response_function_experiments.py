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
# This is a temporary notebook to see what's wrong with the approach in `estimate_fods_dipy.py` and why it differs so wildly from the mrtrix3 results. See [#1](https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments/issues/1).
#
# Here is how to use mrview to get the visualization going:
# ```sh
# INPUT_DIR=extracted_images/NDARINV1JXDFV9Z_2YearFollowUpYArm1_ABCD-MPROC-DTI_20181219171951/sub-NDARINV1JXDFV9Z/ses-2YearFollowUpYArm1/dwi/
# STEM=sub-NDARINV1JXDFV9Z_ses-2YearFollowUpYArm1_run-01_dwi
# mrview $INPUT_DIR/$STEM.nii -odf.load_sh csd_output_dipy/fod/${STEM}_fod.nii.gz
# ```
#
# The first thing to look at is the response function. What happens if I use in the dipy approach the same response function that was generated using the mrtrix method.

# %%
from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from common import get_unique_file_with_extension
from dipy.reconst.csdeconv import recursive_response
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
import numpy as np

# %%
extracted_images_path = Path('extracted_images/')
masks_path = Path('hdbet_output/')
dti_path = Path('dti_output/')
output_dir = Path('csd_output_dipy/')
output_dir_fod = output_dir/'fod'
output_dir_fod.mkdir(exist_ok=True)

# %% [markdown]
#
