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
# # Table of Contents
# * [Without normalizing](#Without-normalizing)
# 	* [one way ANOVA test for scanner effects](#one-way-ANOVA-test-for-scanner-effects)
# * [With normalizing](#With-normalizing)
# 	* [one way ANOVA test for scanner effects](#one-way-ANOVA-test-for-scanner-effects)

# %%
from pathlib import Path
from dipy.io.image import load_nifti
import pandas as pd
import numpy as np
import scipy.stats
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
sns.set_theme(style='white')

# %%
data_root = Path("/data/ebrahim-data/abcd/registration-experiments/2024-01/")
degree_powers_dipy_path = data_root/"degree_powers_dipy"
degree_powers_mrtrix_path = data_root/"degree_powers_mrtrix"
dti_path = data_root/"dti_output"
masks_path = data_root/"hdbet_output"
site_table_path = data_root/"extracted_images/site_table.csv"
DPWMA_table_path = data_root/"DPWMA_table.csv" # DPWMA = "degree powers white matter average" ; this is where to write/cache the output
img_output_path = Path("./degree_powers_variance_by_scanner_img_ouput/")
img_output_path.mkdir(exist_ok=True)

# %%
site_table = pd.read_csv(site_table_path)


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

        self.index_of_l0 = self.l_values.index(0)

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
        dp = data[wm_mask] # degree powers; shape (N,5)
        wm_averages = dp.mean(axis=0) # shape should be 5
        dp_normalized = dp / dp[:,[self.index_of_l0]] # normalized degree powers; shape (N,5)
        wm_averages_of_normalized_FOD = dp_normalized.mean(axis=0) # shape should be 5
        assert(len(wm_averages)==len(self.l_values))
        return self.l_values, wm_averages, wm_averages_of_normalized_FOD


# %%
if DPWMA_table_path.exists():
    print(f"Reading degree power averages table from cache at {DPWMA_table_path}.")
    DPWMA_table = pd.read_csv(DPWMA_table_path)
else:
    print("Computing degree power averages...")
    dp_loader_dipy = DegreePowersImageLoader(degree_powers_dipy_path)
    dp_loader_mrtrix = DegreePowersImageLoader(degree_powers_mrtrix_path)
    
    data_dict = defaultdict(list)
    for basename in tqdm(site_table.basename):
        data_dict["basename"].append(basename)
        for software_name, image_loader in [("dipy", dp_loader_dipy), ("mrtrix", dp_loader_mrtrix)]:
            l_values, wm_averages, wm_averages_of_normalized_FOD = image_loader.get_wm_averages(basename)
            for i,l in enumerate(l_values):
                data_dict[f"DPWMA_l{l}_{software_name}"].append(wm_averages[i])
                data_dict[f"NDPWMA_l{l}_{software_name}"].append(wm_averages_of_normalized_FOD[i]) # "N" in column name means "normalized"
    
    DPWMA_table = site_table.merge(pd.DataFrame(data_dict), on='basename')
    print(f"Done. Caching result at {DPWMA_table_path}.")
    DPWMA_table.to_csv(DPWMA_table_path, index=False)

# %% [markdown]
# # Without normalizing
# First we look at unnormalized FODs, where I think that the FOD magnitude corresponds to an estimate WM volume fraction.

# %% [markdown]
# There is a huge outlier we are going to omit:

# %%
compute_zscores = lambda s : (s - s.mean()) / s.std()
for software_name in ["dipy","mrtrix"]:
    DPWMA_table[f"l0_zscore_{software_name}"] = compute_zscores(DPWMA_table[f"DPWMA_l0_{software_name}"])
outlier_threshold_zscore = 10
outlier_mask = (DPWMA_table[f"l0_zscore_dipy"] > outlier_threshold_zscore) | (DPWMA_table[f"l0_zscore_mrtrix"] > outlier_threshold_zscore)
DPWMA_table[outlier_mask][["basename","l0_zscore_dipy","l0_zscore_mrtrix"]]

# %%
DPWMA_table[outlier_mask].basename.item()

# %% [markdown]
# It is a ridiculous outlier for the mrtrix-generated result only, but we want to include the same images in the analysis for both, so we will omit this image altogether here.

# %%
DPWMA_table.drop(DPWMA_table[outlier_mask].index, inplace=True)

# %% [markdown]
# We take the square roots of the degree powers (aka band energies) so that they are back into the same "units" as the spherical harmonic coefficients and so that the histograms are easier to read. (The squaring pushes outliers further out and makes it difficult to differentiate histograms with different spreads on the same plot using a common binning.)
#
# So here "**R**DPWMA" stands for "square **Root** of Degree Power White Matter Average".

# %%
l_values = DegreePowersImageLoader(degree_powers_dipy_path).l_values
DPWMA_table['manufacturer'] = DPWMA_table.mri_info_manufacturer.apply(lambda s : s.lower().split(" ")[0])

for l in l_values:
    for software_name in ["dipy","mrtrix"]:
        DPWMA_table[f"RDPWMA_l{l}_{software_name}"] = DPWMA_table[f"DPWMA_l{l}_{software_name}"].apply(lambda x : np.sqrt(x))

for l in l_values:
    x_max = max(DPWMA_table[f"RDPWMA_l{l}_dipy"].max(), DPWMA_table[f"RDPWMA_l{l}_mrtrix"].max())
    for software_name in ["dipy","mrtrix"]:
        plt.figure(figsize=(10,4.8))
        sns.histplot(
            data = DPWMA_table,
            x = f"RDPWMA_l{l}_{software_name}",
            hue = 'manufacturer',
            stat = 'density',
            common_norm=False,
            element = "poly"
        )
        plt.xlabel(None)
        plt.xlim(0,x_max)
        plt.title(f"WM-average of l={l} root-power of {software_name}-generated FOD")
        plt.savefig(img_output_path/f"RDPWMA_l{l}_{software_name}_by_scanner.png")
        plt.show()

# %% [markdown]
# ## one way ANOVA test for scanner effects
#
# From the above plots we clearly aren't looking at normally distributed data. Taking logs helps as we can see from the histograms:

# %%
l_values = DegreePowersImageLoader(degree_powers_dipy_path).l_values
DPWMA_table['manufacturer'] = DPWMA_table.mri_info_manufacturer.apply(lambda s : s.lower().split(" ")[0])

for l in l_values:
    for software_name in ["dipy","mrtrix"]:
        DPWMA_table[f"LDPWMA_l{l}_{software_name}"] = DPWMA_table[f"DPWMA_l{l}_{software_name}"].apply(lambda x : np.log(x))

for l in l_values:
    x_min = min(DPWMA_table[f"LDPWMA_l{l}_dipy"].min(), DPWMA_table[f"LDPWMA_l{l}_mrtrix"].min())
    x_max = max(DPWMA_table[f"LDPWMA_l{l}_dipy"].max(), DPWMA_table[f"LDPWMA_l{l}_mrtrix"].max())
    for software_name in ["dipy","mrtrix"]:
        plt.figure(figsize=(10,4.8))
        sns.histplot(
            data = DPWMA_table,
            x = f"LDPWMA_l{l}_{software_name}",
            hue = 'manufacturer',
            stat = 'density',
            common_norm=False,
            element = "poly"
        )
        plt.xlabel(None)
        plt.xlim(x_min,x_max)
        plt.title(f"WM-average of l={l} log-power of {software_name}-generated FOD")
        plt.savefig(img_output_path/f"LDPWMA_l{l}_{software_name}_by_scanner.png")
        plt.show()

# %% [markdown]
# Based on these histograms, we don't expect variances within groups to be equal to one another, so we should use Welch's one way anova rather than classic one way anova. 

# %%
anova_data_dict = defaultdict(list)
for l in l_values:
    for software_name in ["dipy","mrtrix"]:
        aov = pg.welch_anova(
            dv=f"LDPWMA_l{l}_{software_name}",
            between="manufacturer",
            data=DPWMA_table,
        )
        anova_data_dict['l_value'].append(l)
        anova_data_dict['software'].append(software_name)
        anova_data_dict['p_uncorrected'].append(aov['p-unc'].item())
        anova_data_dict['partial_eta_squared'].append(aov['np2'].item())
anova_df = pd.DataFrame(anova_data_dict)
anova_df['p_corrected'] = anova_df['p_uncorrected']*len(anova_df)
anova_df['scanner_effect_significant'] = anova_df['p_corrected'] < 0.05
anova_df.to_csv(data_root/"log_degree_power_scanner_effect_welch_anova.csv", index=False)
anova_df

# %%
print(anova_df[['l_value', 'software', 'p_corrected', 'scanner_effect_significant', 'partial_eta_squared']].to_markdown(index=False))

# %% [markdown]
# Here we see the significance of scanner effects, as well as the effect size in the form of the partial eta squared values.
#
# One thing to note is that Bonferroni correcting the p-vals is overly conservative here because the tests are not independent. The tests are clealry not independent because for example FODs are constrained to be nonnegative so the l=0 coefficient has to respond to the magnitudes of the other coefficients to meet that constraint.
#
# Still, we see significant and large scanner effects, as was visually evident from the histograms.
#
# The scanner effects are much larger for mrtrix-generated FODs than they are for dipy-generated ones.

# %% [markdown]
# # With normalizing
#
# Now we look at normalized FODs, where I've rescaled at each voxel such that the integral of a FOD is always 1. (Or maybe not 1 but at least some common global constant.)

# %%
print(f"Reading degree power averages table from cache at {DPWMA_table_path}.")
DPWMA_table = pd.read_csv(DPWMA_table_path)

# %%
l_values = DegreePowersImageLoader(degree_powers_dipy_path).l_values
DPWMA_table['manufacturer'] = DPWMA_table.mri_info_manufacturer.apply(lambda s : s.lower().split(" ")[0])

for l in l_values:
    for software_name in ["dipy","mrtrix"]:
        DPWMA_table[f"RNDPWMA_l{l}_{software_name}"] = DPWMA_table[f"NDPWMA_l{l}_{software_name}"].apply(lambda x : np.sqrt(x))

for l in l_values:
    x_min = min(DPWMA_table[f"RNDPWMA_l{l}_dipy"].min(), DPWMA_table[f"RNDPWMA_l{l}_mrtrix"].min())
    x_max = max(DPWMA_table[f"RNDPWMA_l{l}_dipy"].max(), DPWMA_table[f"RNDPWMA_l{l}_mrtrix"].max())
    for software_name in ["dipy","mrtrix"]:
        plt.figure(figsize=(10,4.8))
        sns.histplot(
            data = DPWMA_table,
            x = f"RNDPWMA_l{l}_{software_name}",
            hue = 'manufacturer',
            stat = 'density',
            common_norm=False,
            element = "poly"
        )
        plt.xlabel(None)
        plt.xlim(x_min-0.01,x_max+0.01)
        plt.title(f"WM-average of l={l} root-power of {software_name}-generated normalized FOD")
        plt.savefig(img_output_path/f"RNDPWMA_l{l}_{software_name}_by_scanner.png")
        plt.show()

# %% [markdown]
# ## one way ANOVA test for scanner effects
# This time the square-root distributions are more normal looking that with the unnormalized FOD coeffs, and I found that taking logs here doesn't help as much (it produces visually similarly-imperfectly-normal histograms to the ones just above). So we work with the square roots of the degree powers, not the logs.

# %%
anova_data_dict = defaultdict(list)
for l in l_values:
    if l==0: continue # Skip l=0 since it's all 1's after normalizing
    for software_name in ["dipy","mrtrix"]:
        aov = pg.welch_anova(
            dv=f"RNDPWMA_l{l}_{software_name}",
            between="manufacturer",
            data=DPWMA_table,
        )
        anova_data_dict['l_value'].append(l)
        anova_data_dict['software'].append(software_name)
        anova_data_dict['p_uncorrected'].append(aov['p-unc'].item())
        anova_data_dict['partial_eta_squared'].append(aov['np2'].item())
anova_df = pd.DataFrame(anova_data_dict)
anova_df['p_corrected'] = anova_df['p_uncorrected']*len(anova_df)
anova_df['scanner_effect_significant'] = anova_df['p_corrected'] < 0.05
anova_df.to_csv(data_root/"root_normalized_degree_power_scanner_effect_welch_anova.csv", index=False)
anova_df

# %%
print(anova_df[['l_value', 'software', 'p_corrected', 'scanner_effect_significant', 'partial_eta_squared']].to_markdown(index=False))

# %% [markdown]
# After normalizing, there are definitely still large scanner effects.
#
# And now the effects are more equally large between dipy and mrtrix.
