from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from common import get_unique_file_with_extension
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.distributions.distribute_models import SD1WatsonDistributed
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues



# === Parse args ===

parser = argparse.ArgumentParser(description='fits Watson-NODDI model to ABCD DWI images')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
parser.add_argument('masks_path', type=str, help='path to folder containing brain masks. the mask of x.nii is expected to be named x_mask.nii.gz')
parser.add_argument('output_dir', type=str, help='path to folder in which to save the fitted NODDI results')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
masks_path = Path(args.masks_path)
output_dir = Path(args.output_dir)

# === Set up Watson NODDI model ===
# following https://nbviewer.org/github/AthenaEPI/dmipy/blob/master/examples/example_noddi_watson.ipynb

ball = gaussian_models.G1Ball()
stick = cylinder_models.C1Stick()
zeppelin = gaussian_models.G2Zeppelin()
watson_dispersed_bundle = SD1WatsonDistributed(models=[stick, zeppelin])
watson_dispersed_bundle.set_tortuous_parameter('G2Zeppelin_1_lambda_perp','C1Stick_1_lambda_par','partial_volume_0')
watson_dispersed_bundle.set_equal_parameter('G2Zeppelin_1_lambda_par', 'C1Stick_1_lambda_par')
watson_dispersed_bundle.set_fixed_parameter('G2Zeppelin_1_lambda_par', 1.7e-9)
NODDI_mod = MultiCompartmentModel(models=[ball, watson_dispersed_bundle])
NODDI_mod.set_fixed_parameter('G1Ball_1_lambda_iso', 3e-9)


# === Iterate through images, performing NODDI fit and saving results ===

for dwi_nii_directory in extracted_images_path.glob('*/*/*/dwi/'):

    nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii')

    subject_output_dir = output_dir/(nii_path.stem)
    if subject_output_dir.exists():
        print(f"Skipping {nii_path.stem} and assuming it was already processed since the following output directory exists:\n{subject_output_dir}")
        continue
    subject_output_dir.mkdir()
    def generate_output_filepath(noddi_output_name):
        return subject_output_dir/(f"{nii_path.stem}_{noddi_output_name}.nii.gz")

    bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    bvals_SI = bvals * 1e6  # now given in SI units as s/m^2
    acq_scheme = acquisition_scheme_from_bvalues(bvals_SI, bvecs)

    mask_path = masks_path/(nii_path.stem + '_mask.nii.gz')

    data, affine, img = load_nifti(str(nii_path), return_img=True)
    mask_data, mask_affine, mask_img = load_nifti(str(mask_path), return_img=True)

    print(f"performing NODDI fit for {nii_path.stem}...")

    NODDI_fit = NODDI_mod.fit(acq_scheme, data, mask=mask_data)

    fitted_parameters = NODDI_fit.fitted_parameters
    odi = fitted_parameters['SD1WatsonDistributed_1_SD1Watson_1_odi']
    vf_intra = fitted_parameters['SD1WatsonDistributed_1_partial_volume_0'] * fitted_parameters['partial_volume_1']
    vf_iso = fitted_parameters['partial_volume_0']

    mse = NODDI_fit.mean_squared_error(data)
    r2 = NODDI_fit.R2_coefficient_of_determination(data)

    print(f"processed {nii_path.stem}\nsaving data...")
    save_nifti(generate_output_filepath('odi'), odi, affine, img.header)
    save_nifti(generate_output_filepath('vf_intra'), vf_intra, affine, img.header)
    save_nifti(generate_output_filepath('vf_iso'), vf_iso, affine, img.header)
    save_nifti(generate_output_filepath('mse'), mse, affine, img.header)
    save_nifti(generate_output_filepath('r2'), r2, affine, img.header)
    print('saved!')


