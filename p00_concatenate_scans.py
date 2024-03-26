import argparse
import shutil
import os
from pathlib import Path
import pandas as pd
from common import get_unique_file_with_extension
from dipy.io import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
import numpy as np

parser = argparse.ArgumentParser(
    description="concatenate Phillips dMRI scans from extracted ABCD Study images"
)
parser.add_argument(
    "extracted_images_path",
    type=str,
    help="path to folder in which downloaded ABCD images were extracted",
)
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
table_path = extracted_images_path / "site_table.csv"

df = pd.read_csv(table_path)


def write_bvals(bvals_array, filepath):
    """Write a bvals numpy array into the typical fsl format"""
    with open(filepath, "w") as f:
        print(" ".join(map(str, bvals_array.astype(int).tolist())), file=f, end=" ")


def write_bvecs(bvecs_array, filepath):
    """Write a bvecs numpy array into the typical fsl format"""
    with open(filepath, "w") as f:
        for i in range(3):
            print(" ".join(map(str, bvecs_array[:, i].tolist())), file=f, end=" \n")


for subdir in extracted_images_path.iterdir():
    if not subdir.is_dir():
        continue
    dwi_dir_list = list(subdir.glob("*/*/dwi/"))
    if len(dwi_dir_list) != 1:
        print(
            f"Encountered directory {subdir} in the extracted images path with unexpected contents. Skipping this one."
        )
        # (Note that for a custom ABCD download+extraction this could fail. E.g. if someone extracted everything to the same
        # target folder instead of creating one folder for each archive as I did. Then you could have multiple time points or multiple modalities
        # showing up here in dwi_dir_list.)
        continue
    dwi_dir = dwi_dir_list[0]
    subject_level_dirname = dwi_dir.parent.parent.stem
    timepoint_level_dirname = dwi_dir.parent.stem
    dwi_dirs_for_this_scan = list(
        extracted_images_path.glob(
            f"*/{subject_level_dirname}/{timepoint_level_dirname}/dwi"
        )
    )
    row_lookup = df[df.filename == f"{subdir.stem}.tgz"]

    if len(row_lookup) < 1:
        raise Exception(
            f"Could not find a row in {table_path} for filename {subdir.stem}.tgz, even though there is an extracted directory {subdir}."
        )
    if len(row_lookup) > 1:
        raise Exception(
            f"Table {table_path} has multiple rows for filename {subdir.stem}.tgz."
        )
    is_phillips = row_lookup.iloc[0].mri_info_manufacturer == "Philips Medical Systems"

    if len(dwi_dirs_for_this_scan) == 1:
        # This is not multiple scans, so continue.
        continue

    if not is_phillips:
        raise Exception(
            f"Weird. Found multiple directories in {extracted_images_path} containing subdirectories {subject_level_dirname}/{timepoint_level_dirname}, but this is not supposed to be a Phillips scan so this is unexpected."
        )

    print("Concatenating:")
    for p in dwi_dirs_for_this_scan:
        print("\t", p, sep="")

    if len(dwi_dirs_for_this_scan) != 2:
        raise Exception(
            f"Expected there to be exactly two parts to this Phillips scan."
        )

    dwi_dir1, dwi_dir2 = dwi_dirs_for_this_scan
    bval1, bvec1 = read_bvals_bvecs(
        str(get_unique_file_with_extension(dwi_dir1, "bval")),
        str(get_unique_file_with_extension(dwi_dir1, "bvec")),
    )
    bval2, bvec2 = read_bvals_bvecs(
        str(get_unique_file_with_extension(dwi_dir2, "bval")),
        str(get_unique_file_with_extension(dwi_dir2, "bvec")),
    )
    img1_path = get_unique_file_with_extension(dwi_dir1, "nii.gz")
    img2_path = get_unique_file_with_extension(dwi_dir2, "nii.gz")
    data1, affine1, img1 = load_nifti(str(img1_path), return_img=True)
    data2, affine2, img2 = load_nifti(str(img2_path), return_img=True)

    basename_split = img1_path.name.split(".")[0].split("_")
    if not "run" in basename_split[2]:
        raise Exception(f"Unexpected file naming for image: {img1_path}")
    basename_split[2] = "run-concatenated"
    basename = "_".join(basename_split)

    archive_name1 = dwi_dir1.parent.parent.parent.name  # name of the tgz file
    archive_name2 = dwi_dir2.parent.parent.parent.name  # name of the tgz file
    timestamp1 = archive_name1.split("_")[-1]
    timestamp2 = archive_name2.split("_")[-1]
    archive_name_split = archive_name1.split("_")
    archive_name_split[-1] = f"{timestamp1}-{timestamp2}"
    archive_name = "_".join(archive_name_split)
    dwi_dir_cat = (
        extracted_images_path
        / archive_name
        / (dwi_dir1.parent.parent.name)
        / (dwi_dir1.parent.name)
        / (dwi_dir1.name)
    )

    bval_cat = np.concatenate([bval1, bval2])
    bvec_cat = np.concatenate([bvec1, bvec2])
    data_cat = np.concatenate([data1, data2], axis=-1)
    bval_cat_path = dwi_dir_cat / f"{basename}.bval"
    bvec_cat_path = dwi_dir_cat / f"{basename}.bvec"
    img_cat_path = dwi_dir_cat / f"{basename}.nii.gz"

    json1_path = get_unique_file_with_extension(dwi_dir1, "json")
    json2_path = get_unique_file_with_extension(dwi_dir2, "json")
    json1_dst_path = dwi_dir_cat / (json1_path.name)
    json2_dst_path = dwi_dir_cat / (json2_path.name)

    df_before_cat = df

    df_after_cat = df.set_index("filename")
    row1 = df_after_cat.loc[f"{archive_name1}.tgz"]
    row2 = df_after_cat.loc[f"{archive_name2}.tgz"]
    if row1.site_id_l != row2.site_id_l:
        raise Exception(
            f"Site ID in the table {table_path} differs between {archive_name1} and {archive_name2}. These should have been two parts of the same scanning session, so somthing is wrong."
        )
    if row1.mri_info_manufacturer != row2.mri_info_manufacturer:
        raise Exception(
            f"MRI manufacturer in the table {table_path} differs between {archive_name1} and {archive_name2}. These should have been two parts of the same scanning session, so somthing is wrong."
        )
    row_cat = pd.Series(
        {
            "site_id_l": row1.site_id_l,
            "mri_info_manufacturer": row1.mri_info_manufacturer,
            "fmriresults01_id": (row1.fmriresults01_id, row2.fmriresults01_id),
            "basename": basename,
        },
        name=f"{archive_name}.tgz",  # filename column
    )
    df_after_cat = pd.concat([df_after_cat, row_cat.to_frame().T])
    df_after_cat.drop(
        index=[f"{archive_name1}.tgz", f"{archive_name2}.tgz"], inplace=True
    )
    df_after_cat.reset_index(names="filename", inplace=True)

    # These are the directories into which the scans were originally extracted.
    # This pattern of deleting these parent directories only works in this particular setup
    # where we extracted each archive into its own folder. Does not generalize well to other
    # approaches to extractint the data.
    dwi_dir_parent1 = dwi_dir1.parent.parent.parent
    dwi_dir_parent2 = dwi_dir2.parent.parent.parent
    dwi_dir_parent_cat = extracted_images_path / archive_name

    try:
        dwi_dir_cat.mkdir(parents=True)
        write_bvals(bval_cat, bval_cat_path)
        write_bvecs(bvec_cat, bvec_cat_path)
        save_nifti(str(img_cat_path), data_cat, affine1, img1.header)
        shutil.copy(json1_path, json1_dst_path)
        shutil.copy(json2_path, json2_dst_path)
        everything_exists = (
            dwi_dir_cat.exists()
            and bval_cat_path.exists()
            and bvec_cat_path.exists()
            and img_cat_path.exists()
            and json1_dst_path.exists()
            and json2_dst_path.exists()
        )
        assert everything_exists
        df = df_after_cat
        df.to_csv(table_path, index=False)
        shutil.rmtree(dwi_dir_parent1)
        shutil.rmtree(dwi_dir_parent2)
    except Exception as e:
        error_msg = (
            f"Error encountered while writing concatenated files to {dwi_dir_cat}."
        )
        if dwi_dir_parent1.exists() and dwi_dir_parent2.exists():
            if dwi_dir_parent_cat.exists():
                shutil.rmtree(dwi_dir_parent_cat)
                df_before_cat.to_csv(
                    table_path, index=False
                )  # restore table in case it was updated
                error_msg += f"\nCleanup after error is successful: The directory {dwi_dir_cat} has been removed and the original source directories are still present. The table {table_path} has been restored."
        else:  # This should never happen unless shutil.rmtree fails in some horrible way.
            error_msg += "\nCleanup failed!"
            for dir in (dwi_dir_parent1, dwi_dir_parent2):
                if not dir.exists():
                    error_msg += f"\n\tThe original image directory {dir} was removed."
                else:
                    error_msg += f"\n\tThe original image directory {dir} still exists."
            error_msg += f"\nThe attempted concatenated directory {dwi_dir_parent_cat} has been kept to avoid further data loss. Please inspect and clean up manually."
        raise Exception(error_msg) from e
