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

# %%
from pathlib import Path
from dipy.io.image import load_nifti
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
sns.set()

# %%
data_root = Path("/data/ebrahim-data/abcd/registration-experiments/2024-01/")
degree_powers_dipy_path = data_root/"degree_powers_dipy"
degree_powers_mrtrix_path = data_root/"degree_powers_mrtrix"
dti_path = data_root/"dti_output"
masks_path = data_root/"hdbet_output"
table_path = data_root/"extracted_images/site_table.csv"

# %%
site_table = pd.read_csv(table_path)


# %%
class DegreePowersImageLoader:
    def __init__(self, degree_powers_path:Path):
        self.img_dir = degree_powers_path/"degree_power_images"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image subdirectory not found: {self.img_dir}")
        l_values_file_path = degree_powers_path/"l_values.txt"
        if not l_values_file_path.exists():
            raise FileNotFoundError(f"Could not find l-values text file at {l_values_file_path}")
        with open(l_values_file_path) as f:
            self.l_values = eval(f.readline())

    def load_image(self, basename:str):
        img_path = self.img_dir/f"{basename}_degreepowers.nii.gz"
        if not img_path.exists():
            raise FileNotFoundError(f"Couldn't find degree power image file {img_path}.")
        fa_path = dti_path/basename/f"{basename}_fa.nii.gz"
        if not fa_path.exists():
            raise FileNotFoundError(f"Couldn't find FA image file {fa_path}.")
        mask_path = masks_path/f"{basename}_mask.nii.gz"
        
        data, affine = load_nifti(img_path)
        fa_data, fa_affine = load_nifti(fa_path)
        mask_data, mask_affine = load_nifti(mask_path)

        wm_mask = (mask_data > 0) & (fa_data > 0.78)

        return data, wm_mask

    def get_wm_averages(self, basename:str):
        data, wm_mask = self.load_image(basename)
        # shape of data should be (140,140,140,5) where the last axis is l-values which are listed in self.l_values
        wm_averages = data[wm_mask].mean(axis=0) # shape should be 5
        assert(len(wm_averages)==len(self.l_values))
        return self.l_values, wm_averages      


# %%
dp_loader_dipy = DegreePowersImageLoader(degree_powers_dipy_path)
dp_loader_mrtrix = DegreePowersImageLoader(degree_powers_mrtrix_path)

# %%
data_dict = defaultdict(list)
for basename in site_table.basename:
    data_dict["basename"].append(basename)
    for software_name, image_loader in [("dipy", dp_loader_dipy), ("mrtrix", dp_loader_mrtrix):
    l_values, wm_averages = image_loader.get_wm_averages(basename)
    for i,l in enumerate(l_values):
        colname = f"DPWMA_l{l}_{software_name}"
        data_dict[colname].append(wm_averages[i])
