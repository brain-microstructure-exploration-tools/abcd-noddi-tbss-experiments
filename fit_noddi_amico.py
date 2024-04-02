from pathlib import Path
import shutil
import argparse
import amico
from common import get_unique_file_with_extension



# === Parse args ===

parser = argparse.ArgumentParser(description='fits NODDI model to ABCD DWI images using AMICO')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
parser.add_argument('masks_path', type=str, help='path to folder containing brain masks. the mask of x.nii.gz is expected to be named x_mask.nii.gz')
parser.add_argument('output_dir', type=str, help='path to folder in which to save the fitted NODDI results')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)
masks_path = Path(args.masks_path)
output_dir = Path(args.output_dir)

# === Set up AMICO ===
# following https://github.com/daducci/AMICO/wiki/NODDI

amico.setup()
kernels_generated = False


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

    try:
        amico_workspace = subject_output_dir/'amico_workspace'
        amico_workspace.mkdir(exist_ok=True)

        bval_path = get_unique_file_with_extension(dwi_nii_directory, 'bval')
        bvec_path = get_unique_file_with_extension(dwi_nii_directory, 'bvec')

        scheme_path = amico_workspace/f'{basename}.scheme'
        amico.util.fsl2scheme(bval_path, bvec_path, schemeFilename=scheme_path)

        mask_path = masks_path/(basename + '_mask.nii.gz')

        ae = amico.Evaluation()
        ae.load_data(nii_path, scheme_path, mask_path, replace_bad_voxels=0)
        ae.set_model('NODDI')
        ae.generate_kernels(regenerate=(not kernels_generated)) # We only need to generate kernels once because the b-values are the same for all of ABCD.
        kernels_generated = True
        ae.load_kernels()
        ae.fit()
        ae.save_results(path_suffix=basename)
        amico_results_dir = Path('.')/'AMICO'/f'NODDI_{basename}'

        for amico_output_file in amico_results_dir.iterdir():
            filename = amico_output_file.name
            filename = filename.replace('fit', basename)
            shutil.move(amico_output_file, subject_output_dir/filename)
        shutil.rmtree(Path('.')/'AMICO')
    except Exception as e:
        shutil.rmtree(subject_output_dir) # Remove subject output dir so that upon re-run it doesn't think this was already processed successfully.
        raise e