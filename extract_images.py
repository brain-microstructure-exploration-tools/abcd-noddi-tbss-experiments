import argparse
import shutil
import os
import tarfile
import gzip
from pathlib import Path
import pandas as pd

# === Parse args ===

parser = argparse.ArgumentParser(description='extract downloaded ABCD Study images')
parser.add_argument('table_path', type=str, help='path to csv table with a filename and site_id_l column')
parser.add_argument('archive_path', type=str, help='path to folder ABCD images were downloaded (this folder would contain the .tgz files)')
parser.add_argument('output_path', type=str, help='path to folder in which to put extracted ABCD images')
args = parser.parse_args()
table_path = Path(args.table_path)
archive_path = Path(args.archive_path)
output_path = Path(args.output_path)

output_path.mkdir(exist_ok=True)

# === Extract what is in the table's filename column ===

df = pd.read_csv(table_path)
filenames = []
basenames = []
for filename in df.filename:

    archive_file_path = archive_path/filename
    extraction_target = output_path/(Path(filename).stem)
    
    if not archive_file_path.exists():
        raise FileNotFoundError(f"File {archive_file_path} not found")
    
    # Extract
    if not extraction_target.exists():
        with tarfile.open(archive_file_path, "r:gz") as tar:
            tar.extractall(path=extraction_target)
    else:
        print(f"Found that {extraction_target} already exists. Skipping extraction.")

    # Get the path and name of the image file
    niigz_path_list = list(extraction_target.glob('*/*/dwi/*.nii.gz'))
    nii_path_list = list(extraction_target.glob('*/*/dwi/*.nii'))
    img_path_list = nii_path_list + niigz_path_list
    if len(img_path_list) == 0:
        raise FileNotFoundError(f"Couldn't find any image file in {extraction_target}")
    if len(img_path_list) > 1:
        img_path_list_formatted = '\n\t'.join(map(str,img_path_list))
        raise Exception(f"Multiple image files were found in {extraction_target}:\n\t{img_path_list_formatted}")
    img_file_path = img_path_list[0]
    basename = img_file_path.name.split('.')[0]
    filenames.append(filename)
    basenames.append(basename)

    # Recompress just the nii to an nii.gz
    if len(nii_path_list) > 0:
        assert(len(nii_path_list)==1) # this must be true now
        nii_file_path = nii_path_list[0]
        with open(nii_file_path, 'rb') as f_in:
            with gzip.open(str(nii_file_path) + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)    
        # Remove the original nii
        os.remove(nii_file_path)
    else:
        assert(len(niigz_path_list)==1) # this must be true now
        print(f"Found that {niigz_path_list[0]} already exists. Skipping recompression.")
    
df_with_basenames = pd.merge(df, pd.DataFrame({'filename':filenames,'basename':basenames}))
df_with_basenames.to_csv(output_path/'site_table.csv', index=False)