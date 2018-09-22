

import re
import os
import csv
import nrrd
import shutil

import pandas as pd

from collections import OrderedDict


def _check_text(text_elem):
    # Return <int> if able to convert, else <str>.

    return int(text_elem) if text_elem.isdigit() else text_elem

def sample_num(sample):

    sample_ids = re.split('(\d+)', sample)
    for elem in sample_ids:
        if elem.isdigit():
            return int(elem)

    return None


def natural_keys(text):
    """
    """

    return [_check_text(text_elem) for text_elem in re.split('(\d+)', text)]


def relative_paths(path_to_dir, sorted=True, target_format=None):
    # Return a list of relative paths to all files in directory.

    file_names = os.listdir(path_to_dir)

    rel_paths = []
    for file_name in file_names:

        rel_path = os.path.join(path_to_dir, file_name)

        if os.path.isfile(rel_path) and rel_path.endswith(target_format):
            rel_paths.append(rel_path)

        if sorted:
            rel_paths.sort(key=natural_keys)

    return rel_paths


def read_matlab():

    # In case need to convert raw data files.
    pass


def read_nrrd():

    # In case need to convert raw data files.
    pass


def write_nrrd():

    # In case need to convert raw data files.
    pass


def matlab_to_nrrd():

    # In case need to convert raw data files.
    pass


# Assumes the input CSV has at least 2 columns: "Image" and "Mask"
# These columns indicate the location of the image file and mask file, respectively
# Additionally, this script uses 2 additonal Columns: "Patient" and "Reader"
# These columns indicate the name of the patient (i.e. the image), the reader (i.e. the segmentation), if
# these columns are omitted, a value is automatically generated ("Patient" = "Pt <Pt_index>", "Reader" = "N/A")
def read_samples(path_image_dir, path_mask_dir, target_format='nrrd'):
    """Generate dictionary of locations to image and corresponding mask."""

    sample_paths = relative_paths(
        path_image_dir, sorted=True, target_format=target_format
    )
    mask_paths = relative_paths(
        path_mask_dir, sorted=True, target_format=target_format
    )
    samples = []
    for sample, mask in zip(sample_paths, mask_paths):
        samples.append(
            OrderedDict(
                Image=sample, Mask=mask, Patient=sample_num(sample), Reader=''
            )
        )
    return samples


def write_features(path_to_file, results):

    with open(path_to_file, mode='w') as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=list(results[0].keys()),
            restval='',
            extrasaction='raise',
            lineterminator='\n'
        )
        writer.writeheader()
        writer.writerows(results)

    return None


def read_prelim_result(path_to_file):

    result = OrderedDict(pd.read_csv(path_to_file).values)
    return result


def write_prelim_result(path_to_file, feature_vector):
    """Store results in temporary separate files to prevent write conflicts
    # This allows for the extraction to be interrupted. Upon restarting,
    # already processed cases are found in the TEMP_DIR directory and
    # loaded instead of re-extracted.
    """

    with open(path_to_file, 'w') as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=list(feature_vector.keys()),
            lineterminator='\n'
        )
        writer.writeheader()
        writer.writerow(feature_vector)

    return None


if __name__ == '__main__':
    # TODO: Clean up

    path_ct_dir = './../../data/images/stacks_ct/cropped_ct'
    path_pet_dir = './../../data/images/stacks_pet/cropped_pet'
    path_masks_dir = './../../data/images/masks/prep_masks'

    samples = read_samples(path_ct_dir, path_masks_dir)
    print(samples[0].keys())
