import csv
import os
from glob import glob

# Paths
original_annots_fpath = 'H:/Luna16_Data/DeepLung/CSVFILES/annotations.csv'

# Iterate through folds
for fold in range(10):
    subset_fpath = f'H:/Luna16_Data/DeepLung/preprocess/subset{fold}'
    subset_fnames = sorted(glob(os.path.join(subset_fpath, '*_clean.npy')))

    # Extract seriesuid by removing path and suffix
    subset_fnames = [os.path.basename(f).rsplit('_clean.npy')[0] for f in subset_fnames]

    # Prepare filenames for fold-specific annotations and series IDs
    fold_annotations_fname = f'annotations{fold}.csv'
    fold_annotations = []
    fold_annots_writer = csv.writer(open(os.path.join(subset_fpath, fold_annotations_fname), 'w', newline=''),
                                    delimiter='\n')

    fold_sids_fname = f'seriesids{fold}.csv'
    fold_sids = []
    fold_sids_writer = csv.writer(open(os.path.join(subset_fpath, fold_sids_fname), 'w', newline=''), delimiter='\n')

    # Read original annotations and filter by series IDs in the current subset
    with open(original_annots_fpath, 'r') as orig_file:
        orig_csv_rdr = csv.reader(orig_file, delimiter=',')
        for row in orig_csv_rdr:
            curr_sid = row[0]
            if curr_sid in subset_fnames:
                fold_annotations.append((",").join(row))
                if curr_sid not in fold_sids:
                    fold_sids.append(curr_sid)

    # Write the series IDs and annotations for the current fold
    fold_sids_writer.writerow(fold_sids)
    print(f'Written {os.path.join(subset_fpath, fold_sids_fname)}')
    fold_annots_writer.writerow(fold_annotations)
    print(f'Written {os.path.join(subset_fpath, fold_annotations_fname)}')
