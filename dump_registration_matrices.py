from pathlib import Path
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='dumps out the registration matrices found in json files in extracted ABCD diffusion images')
parser.add_argument('extracted_images_path', type=str, help='path to folder in which downloaded ABCD images were extracted')
args = parser.parse_args()
extracted_images_path = Path(args.extracted_images_path)

registration_matrix_key = 'registration_matrix_T1'
for json_file_path in extracted_images_path.glob('*/*/*/dwi/*.json'):
    with open(json_file_path, 'r') as f:
        d = json.load(f)
        if registration_matrix_key not in d:
            print(f"Unexpected json file found that does not have {registration_matrix_key}: {json_file_path.name}")
            continue
        print(np.array(d[registration_matrix_key]))