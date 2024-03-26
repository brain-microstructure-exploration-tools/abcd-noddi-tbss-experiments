import argparse
import json
import numpy as np
from common import get_degree_powers
from pathlib import Path
from dipy.io.image import load_nifti, save_nifti

# === Parse args ===

parser = argparse.ArgumentParser(
    description="computes degree powers from fiber orientation distributions"
)
parser.add_argument(
    "fod_path",
    type=str,
    help="path to folder containing fiber orientation distribution images",
)
parser.add_argument(
    "output_dir",
    type=str,
    help="path to folder in which to save the computed degree power images",
)
args = parser.parse_args()
fod_path = Path(args.fod_path)
output_dir = Path(args.output_dir)
l_values_path = output_dir / "l_values.txt"
degree_power_image_path = output_dir / "degree_power_images"
degree_power_image_path.mkdir(exist_ok=True)

if not fod_path.exists():
    raise FileNotFoundError(f"Could not find {fod_path}")

first_l_values = None
if l_values_path.exists():
    with open(l_values_path, "r") as l_values_file:
        first_l_values = np.array(json.load(l_values_file), dtype=int)

for fod_filepath in fod_path.glob("*.nii.gz"):
    basename = "_".join(
        str(fod_filepath.name).split("_")[:-1]
    )  # remove the trailing _fod.nii.gz
    output_file_path = degree_power_image_path / f"{basename}_degreepowers.nii.gz"
    if output_file_path.exists():
        print(
            f"Skipping degree power computation for {basename} since the following output file exists:\n{output_file_path}"
        )
        continue

    fod_data, affine, img = load_nifti(fod_filepath, return_img=True)

    l_values, degree_powers = get_degree_powers(fod_data)

    # save l-values to file if we didn't already. if we did, then ensure consistency. l-values should be same for all subjects
    if first_l_values is None:
        first_l_values = l_values
        with open(l_values_path, "w") as l_values_file:
            json.dump(
                l_values.astype(
                    int
                ).tolist(),  # list of ints suitable for json serialization
                l_values_file,
            )
    else:
        if not (l_values == first_l_values).all():
            raise Exception(
                "Inconsistency detected in the l-values list that describes the degree power image channel."
            )

    save_nifti(output_file_path, degree_powers, affine, img.header)
    print(f"computed and saved degree powers for for {basename}...")
print("done")
