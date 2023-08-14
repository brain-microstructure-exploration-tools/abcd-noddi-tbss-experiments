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

This section will use a proper `requirements.txt` at some point but for now:
```sh
pip install dmipy pathos numba dipy pandas numpy
```

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
We can use MRtrix3 or dipy for this. We include both approaches here, but currently recommend the MRtrix3 approach.

To carry out the MRtrix3 processing, first [install MRtrix3](https://mrtrix.readthedocs.io/en/latest/installation/before_install.html#before-installing) and make sure your environment is configured to find the MRtrix3 executables. Then:

```sh
mkdir csd_output/
./estimate_fods_mrtrix.sh extracted_images/ hdbet_output/ csd_output/
```

To instead carry out dipy processing, first make sure you did the DTI fitting above (the FA is used to support the automatic response function estimation algorithm). Then:

```sh
mkdir csd_output/
python estimate_fods_dipy.py extracted_images/ hdbet_output/ dti_output/ csd_output/
```

These two approaches use different algorithms for response function estimation. The dipy approach given here in `estimate_fods_dipy.py` does not work with a common group mean response function, instead using subject-level estimated response functions for doing CSD for each subject. This might be a bad thing; we need to think about it. The dipy approach here is also slower. However the dipy implementation is a little more pleasant to study, being part of a more general and elegant framework.

## Generate a population template

Here we use MRtrix3 to generate a FOD population template. Again, first ensure that [MRtrix3](https://mrtrix.readthedocs.io/en/latest/installation/before_install.html#before-installing) is installed and that your environment is configured to find the MRtrix3 executables. Then:

```sh
mkdir population_template
./generate_population_template.sh csd_output/fod/ hdbet_output/ population_template/
```

## TBSS

I haven't done this part yet, but I think a good place to start looking is the great work here: [generalized TBSS pipeline](https://github.com/pnlbwh/tbss)
