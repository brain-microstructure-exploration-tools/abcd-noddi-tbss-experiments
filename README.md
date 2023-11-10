# abcd-noddi-tbss-experiments

This is an experiment in running part of a NODDI TBSS pipeline on diffusion imaging from the ABCD study dataset.
The purpose is mainly for me to become familiar with some of the tools involved in the pipeline.

This is a WIP and is currently stalled, possibly to be resumed later. The TBSS part does not exist yet, nor does registration and template generation.

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

Set up a python 3.8 environment and install the required packages:
```sh
pip install -r requirements.txt
```
Some steps are based on MRtrix3 and require that you [install MRtrix3](https://mrtrix.readthedocs.io/en/latest/installation/before_install.html#before-installing) and configure your environemnt to find the MRtrix3 executables.

## Extracting files

The downloaded data is in the form of `tgz` files that can be extracted to `nii` image files with bvals and bvecs provided as separate text files.

Run the following command with `ARCHIVES_PATH` replaced by a path to the directory containing downloaded ABCD image `tgz` files:

```sh
mkdir extracted_images
./extract_images.sh  ARCHIVES_PATH extracted_images/
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

```sh
mkdir dti_output/
python fit_dti.py extracted_images/ hdbet_output/ dti_output/
```

## Perform NODDI fit

The NODDI fit takes a while to run. Almost 3 hours per image with parallel processing enabled on my 12-core machine.

It uses [dmipy](https://github.com/AthenaEPI/dmipy), following [this tutorial](https://nbviewer.org/github/AthenaEPI/dmipy/blob/master/examples/tutorial_setting_up_acquisition_scheme.ipynb) for creating the acquisition scheme object and [this tutorial](https://nbviewer.org/github/AthenaEPI/dmipy/blob/master/examples/example_noddi_watson.ipynb) for constructing a Watson NODDI model.

```sh
mkdir noddi_output/
python fit_watson_noddi.py extracted_images/ hdbet_output/ noddi_output/
```

## Estimate FODs

Here we estimate fiber orientation distributions (FODs) using CSD.
We use MRtrix3 for this. To carry out the MRtrix3 processing:

```sh
mkdir csd_output/
./estimate_fods_mrtrix.sh extracted_images/ hdbet_output/ csd_output/
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
