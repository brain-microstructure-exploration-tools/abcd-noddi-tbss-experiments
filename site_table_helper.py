import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='helper used by other scripts for working with the study site table')
parser.add_argument('table_path', type=str, help='path to the study sites table that was output by the extract_images script')
parser.add_argument('command', type=str,
    help="one of get_site, get_sites, or get_subjects"
)
parser.add_argument('--subject', type=str,
    help="to specify a subject when using get_site. specify by providing image file basename."
)
parser.add_argument('--site', type=str,
    help="to specify a site when using get_subjects. specify by providing site id."
)

if __name__=="__main__":
    args = parser.parse_args()
    table_path = args.table_path
    cmd = args.command
    df = pd.read_csv(table_path)

    if cmd == 'get_sites':
        print(' '.join(df.site_id_l.unique()))
    
    elif cmd == 'get_site':
        basename = args.subject
        if basename is None:
            raise ValueError("The \"--subject\" argument is needed for get_site.")
        sites = df[df.basename == basename].site_id_l
        if len(sites) > 1:
            raise Exception(f"Multiple sites associated with subject {basename}")
        if len(sites) < 1:
            raise Exception(f"Unable to find in the table the subject {basename}")
        print(sites.item())
    
    elif cmd == 'get_subjects':
        site = args.site
        if site is None:
            raise ValueError("The \"--site\" argument is needed for get_subjects.")
        basenames = df[df.site_id_l == site].basename
        print(' '.join(basenames))

    else:
        raise ValueError(f"Unknown command: {cmd}")