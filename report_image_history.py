import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='prints out DTI image history from the ABCD Study table fmriresults01.txt')
parser.add_argument('table_path', type=str, help='path to fmriresults01.txt')
args = parser.parse_args()
fmriresults01_table_path = args.table_path

df = pd.read_csv(fmriresults01_table_path, delimiter='\t')
dti_mask = df.scan_type.str.contains('dti', case=False)
df = df[dti_mask]

columns_to_report_on = [col for col in df.columns if "pipeline" in col]
columns_to_report_on.append('image_history')
for col in columns_to_report_on:
    print(f"Values of {col} that occur for DMRI images:\n\t", end='')
    print('\n\t'.join([str(val) for val in df[col].unique()]))