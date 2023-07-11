# abcd-noddi-tbss-experiments

This is an experiment in running part of a NODDI TBSS pipeline on diffusion imaging from the ABCD study dataset.
The purpose is mainly for me to become familiar with some of the tools involved in the pipeline.

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

## Extracting

The downloaded data is in the form of `tgz` files that can be extracted to `nii` image files with bvals and bvecs provided as separate text files.

Set `ARCHIVES_PATH` in `./01_extract_images.sh` to point to downloaded ABCD images.

```sh
mkdir extracted_images
./extract_images.sh extracted_images/
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

