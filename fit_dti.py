from pathlib import Path
import argparse
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from common import get_unique_file_with_extension
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity



# === Parse args ===

parser = argparse.ArgumentParser(description='fits diffusion tensors to a diffusion weighted image')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
parser.add_argument('masks_path', type=str, help='path to folder containing brain masks. the mask of x.nii is expected to be named x_mask.nii.gz')
parser.add_argument('output_dir', type=str, help='path to folder in which to save the DTIs')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
masks_path = Path(args.masks_path)
output_dir = Path(args.output_dir)

# === Iterate through images, performing NODDI fit and saving results ===

for dwi_nii_directory in extracted_images_path.glob('*/*/*/dwi/'):

    nii_path = get_unique_file_with_extension(dwi_nii_directory, 'nii.gz')
    # (Note that getting a unique file like this wouldn't work in general on an ABCD download if someone extracted everything to the same
    # target folder instead of creating one folder for each archive as I did.)
    basename = nii_path.name.split('.')[0]

    subject_output_dir = output_dir/basename
    if subject_output_dir.exists():
        print(f"Skipping {basename} and assuming it was already processed since the following output directory exists:\n{subject_output_dir}")
        continue
    subject_output_dir.mkdir()
    def generate_output_filepath(output_name):
        return subject_output_dir/(f"{basename}_{output_name}.nii.gz")

    bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
    bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

    bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
    gtab = gradient_table(bvals, bvecs)

    mask_path = masks_path/(basename + '_mask.nii.gz')

    data, affine, img = load_nifti(str(nii_path), return_img=True)
    mask_data, mask_affine, mask_img = load_nifti(str(mask_path), return_img=True)

    print(f"fitting DTI for {basename}...")

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask_data)

    dti_lotri = tenfit.lower_triangular() # lotri means lower triangular, in dipy order (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)
    evals = tenfit.evals # eigenvalues
    evecs = tenfit.evecs # eigenvectors
    fa = fractional_anisotropy(evals)
    md = mean_diffusivity(evals)


    print(f"processed {basename}\nsaving data...")
    save_nifti(generate_output_filepath('dti_lotri'), dti_lotri, affine, img.header)
    save_nifti(generate_output_filepath('evals'), evals, affine, img.header)
    save_nifti(generate_output_filepath('evecs'), evecs, affine, img.header)
    save_nifti(generate_output_filepath('fa'), fa, affine, img.header)
    save_nifti(generate_output_filepath('md'), md, affine, img.header)
    print('saved!')


