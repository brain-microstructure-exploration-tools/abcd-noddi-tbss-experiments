# abcd-noddi-tbss-experiments

This is an experiment in running part of a NODDI TBSS pipeline on diffusion imaging from the ABCD study dataset.
The purpose is mainly for me to become familiar with some of the tools involved in the pipeline.

This is a WIP. The TBSS part does not exist yet.

## The data

This is being applied to DMRI images from ABCD release 5.0, those categorized under _RELEASE 5.0 MINIMALLY PROCESSED IMAGING DATA RECOMMENDED - JUNE 2023_. Running `report_image_history.py` on the `fmriresults01.txt` table downloaded from the ABCD Study dataset, we see the following processing steps listed as having already been done to the data:

- eddy-current correction
- motion correction
- B0 inhomogeneity correction
- gradient unwarp
- replacement of bad slice-frames
- between scan motion correction
- rigid body registration to atlas and resampling to 1.7mm^3 LPI (requires rigid registration to T1 - see included json for matrix values)

That last item on rigid body registration and resampling might be irrelevant in our case, because if you run `dump_registration_matrices.py` on the extracted images you see that all the registration matrices are identity.

## Setup

Set up two python 3.8 environments, a main one and one for dmipy.
Install the required packages as follows:
```sh
python3.8 -m venv .venv
. .venv/bin/activate
pip install git+https://github.com/dipy/dipy.git@871b498c2bab8ad1c4961e2dbd594063bd877440
pip install -r requirements.txt
deactivate
python3.8 -m venv .venv_dmipy
. .venv_dmipy/bin/activate
pip install -r requirements_dmipy.txt
deactivate
```

Use the main virtual environment for most steps, but use the dmipy environment for the NODDI computation steps.

Some steps are based on MRtrix3 and require that you [install MRtrix3](https://mrtrix.readthedocs.io/en/latest/installation/before_install.html#before-installing) and configure your environemnt to find the MRtrix3 executables.

## Extracting files

The downloaded data is in the form of `tgz` files that can be extracted to image files with bvals and bvecs provided as separate text files in FSL format.

Since the ABCD Study is a multi-site study, we will need to keep track of study site to do some of the image processing correctly. This is done in a csv table with at least the following columns:

- `filename`: This column contains the full filenames of the tgz files to be extracted.
- `site_id_l`: This column contains the ID for the study site at which each image was acquired. It can be found in the table `abcd_y_lt` of the ABCD 5.0/5.1 release.
- `mri_info_manufacturer`: This columns contains the manufacturer name for the MRI scanner for each image. It can be found in the table `mri_y_adm_info` of the ABCD 5.0/5.1 release.

Run the following command with `TABLE_PATH` replaced by a path to the csv table just described and with `ARCHIVES_PATH` replaced by a path to the directory containing downloaded ABCD image `tgz` files:

```sh
mkdir extracted_images
python ./extract_images.py  TABLE_PATH  ARCHIVES_PATH  extracted_images/
```

Besides extracting archives, this will write a table to the `extracted_images/` directory that extends the input table by a column that indicates image file basenames.

Any scans coming from the Phillips scanner must now be consolidated as the scanner generates two image files per acquisition:
```sh
python concatenate_scans.py extracted_images/
```

## Denoise

Apply Patch2Self denoising. This step is optional and it currently _replaces_ the original image files with the denoised ones.

```sh
cp -r extracted_images/ extracted_images_before_denoising/ # Optionally, preserve the originals
python denoise.py extracted_images/
```

## Generating NRRD Headers

Install the tool at [this repo](https://github.com/pnlbwh/conversion):
```sh
git clone https://github.com/pnlbwh/conversion.git
pip install conversion/
```

Use the script to generate all NRRD headers:
```sh
python generate_nrrd_headers.py extracted_images/
```
Having the NRRD headers allows the DWIs to be loaded in 3D Slicer and recognized as DWIs.

## Brain extraction

Average the b0 images within each DWI sequence:
```sh
mkdir b0_averages
python generate_b0_averages.py extracted_images/ b0_averages/
```

Install [HD-BET](https://github.com/MIC-DKFZ/HD-BET):
```sh
git clone https://github.com/MIC-DKFZ/HD-BET
pip install -e HD-BET
```

Run brain extraction:
```sh
mkdir hdbet_output/
hd-bet -i b0_averages/ -o hdbet_output/
```

## Perform DTI fit

This is optional.

```sh
mkdir dti_output/
python fit_dti.py extracted_images/ hdbet_output/ dti_output/
```

## Perform NODDI fit

The NODDI fit takes a while to run. Almost 3 hours per image with parallel processing enabled on my 12-core machine.

It uses [dmipy](https://github.com/AthenaEPI/dmipy), following [this tutorial](https://nbviewer.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_setting_up_acquisition_scheme.ipynb) for creating the acquisition scheme object and [this tutorial](https://nbviewer.org/github/AthenaEPI/dmipy/blob/master/examples/example_noddi_watson.ipynb) for constructing a Watson NODDI model.

Remember to use the dmipy python environment that you set up above, not the main python environment.

```sh
mkdir noddi_output/
python fit_watson_noddi.py extracted_images/ hdbet_output/ noddi_output/
```

## Estimate FODs

Here we estimate fiber orientation distributions (FODs) using CSD. We can use MRtrix3 or dipy for this. We save the output FODs as a 3D image of functions representated in a basis of real spherical harmonics. Regardless of whether MRtrix3 or dipy is used, the output saved is in terms of a basis of real spherical harmonics that follows the MRtrix3 convention. In the convention, the basis functions $Y^\text{(mrtrix)}_{lm}(\theta,\phi)$ are given in terms of the complex spherical harmonics $Y^m_l(\theta,\phi)$ as follows:
```math
Y^\text{(mrtrix)}_{lm} =
\left\{
\begin{array}{ll}
\sqrt{2}\ \textrm{Im}[Y^{-m}_l] & \text{if $m<0$}\\
Y^0_l & \text{if $m=0$}\\
\sqrt{2}\ \textrm{Re}[Y^{m}_l] & \text{if $m>0$}
\end{array}
\right.
```

### MRTrix3 FODs


To carry out the MRtrix3 processing:

```sh
mkdir csd_output/
./estimate_fods_mrtrix.sh extracted_images/ hdbet_output/ csd_output/
```

### DIPY FODs

To carry out the DIPY processing instead:

```sh
mkdir csd_output/
python estimate_fods_dipy.py extracted_images/ hdbet_output/ csd_output/
```

Note: This pipeline is designed to work with the preprocessed ABCD images, and we have found that for these images the DIPY processing script must flip the x-axis of the b-vectors. [It's not clear why.](https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments/issues/7#issuecomment-1828736081)

Note: The output FODs are saved in the form of spherical harmonic coefficients using the conventions of MRtrix3, regardless of whether DIPY or MRtrix3 is used.

## Compute degree powers

In this step we compute "degree powers" of FODs, going from FOD images to degree power images.

The "power" at a specific degree $l$ is the square-sum over all indices $m$ of the spherical harmonic coefficinets $c^m_l$ at that given $l$:
```math
p_l = \sum_{m=-l}^{l} (c^m_l)^2
```
The idea comes from [Bloy 2010](https://doi.org/10.1109/ISBI.2010.5490161).
This quantity $p_l$ is a useful derived feature because it is invariant to rotations of the function represented by spherical harmonic coefficents $c^m_l$.

To compute degree powers:
```sh
mkdir degree_powers/
python compute_degree_powers.py csd_output/fod degree_powers/
```

To then normalize them to have the same spherical integral:
```sh
mkdir degree_powers_normalized/
python normalize_degree_powers.py degree_powers/
```
This will remove the l=0 channel since that would be a constant image.
This step simply replaces $c^m_l$ by $c^m_l/c^0_0$ and provides the "degree power" that would result from that:
```math
p_l^\text{(normalized)} = \sum_{m=-l}^{l} (c^m_l/c^0_0)^2 = p_l/p_0
```
This is a type of normalization because the integral of a FOD over the sphere is proportional to $c^0_0$.

## Compute TOMs and tracts

This step is not part of the NODDI-TBSS pipeline, but is included here for our convenience because we needed it for something unrelated.
This step requires MRtrix3.
The method has been ported over from [this script](https://github.com/brain-microstructure-exploration-tools/abcd-registration-experiments/blob/7e78678297f4eeba317418e6c313260ca642227a/evaluation_framework/single_subject_tractography.py).

Install TractSeg (it can go into the main python environment being used for most of these steps):
```sh
pip install TractSeg
```

Compute peaks:
```
mkdir fod_peaks/
./sh2peaks_batch.sh csd_output/ hdbet_output/ fod_peaks/
```

Run TractSeg:
```sh
mkdir tractseg_output/
./tractseg.sh fod_peaks/ hdbet_output/ tractseg_output/
```



## Generate a population template

Here we use MRtrix3 to generate a FOD population template. Then:

```sh
mkdir population_template
./generate_population_template.sh csd_output/fod/ hdbet_output/ population_template/
```

This process can pause for user keystroke repeatedly if there are implausible seeming registrations. If you want to stop MRtrix3 `population_template` from pausing every time it detects an implausible registration, add the `-linear_no_pause` flag to the command in `generate_population_template.sh`. Probably it would be [better to debug the situation](https://community.mrtrix.org/t/population-template-error-on-linear-transformation/4081/2).

## Coregister DTI and NODDI images using the population template

Again using MRtrix3:

```sh
mkdir dti_output_warped
mkdir noddi_output_warped
./transform_to_template.sh dti_output/ dti_output_warped/ population_template_mrtrix/warps/ population_template_mrtrix/template.nii.gz
./transform_to_template.sh noddi_output/ noddi_output_warped/ population_template_mrtrix/warps/ population_template_mrtrix/template.nii.gz
```

## TBSS

I haven't done this part yet, but I think a good place to start looking is the great work here: [generalized TBSS pipeline](https://github.com/pnlbwh/tbss)
