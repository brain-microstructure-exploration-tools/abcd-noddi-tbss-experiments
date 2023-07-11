from pathlib import Path
import argparse
import conversion # Get this from https://github.com/pnlbwh/conversion


parser = argparse.ArgumentParser(description='generates nrrd headers for extracted ABCD diffusion images')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)

def get_unique_file_with_extension(directory_path : Path, extension : str) -> Path:
    l = list(directory_path.glob(f'*.{extension}'))
    if len(l) == 0 :
        raise Exception(f"No {extension} file was found in {directory_path}")
    if len(l) > 1 :
        raise Exception(f"Multiple {extension} files were found in {directory_path}")
    return l[0]

for dwi_nii_directory in extracted_images_path.glob('*/*/*/dwi/'):
    nii_file_path = get_unique_file_with_extension(dwi_nii_directory, 'nii')
    output_nhdr_path = dwi_nii_directory/(nii_file_path.stem + '.nhdr')
    conversion.nhdr_write(
        str(nii_file_path),
        str(get_unique_file_with_extension(dwi_nii_directory, 'bval')),
        str(get_unique_file_with_extension(dwi_nii_directory, 'bvec')),
        str(output_nhdr_path),
    )