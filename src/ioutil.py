

import re
import os
import csv
import nrrd
import shutil

import pandas as pd

from collections import OrderedDict


# Checkout: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def _atoi(text):

    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    """

    return [_atoi(c) for c in re.split('(\d+)', text)]


def _relative_paths(path_to_dir, sorted=True, target_format=None):
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

    sample_paths = _relative_paths(
        path_image_dir, sorted=True, target_format=target_format
    )
    mask_paths = _relative_paths(
        path_mask_dir, sorted=True, target_format=target_format
    )
    samples = []
    for sample, mask in zip(sample_paths, mask_paths):
        pass
        #samples.append(
        #    OrderedDict(
        #        Image=sample, Mask=mask, Patient=sample_num, Reader='n/a'
        #    )
        #)
    return samples

        # NOTE: Return a pandas dataframe with
        # index = patient num
        # col image = location img file
        # col mask = location of mask file

        # Extract List of cases
        #cases = []
        #with open(INPUTCSV, 'r') as inFile:
        #    cr = csv.DictReader(inFile, lineterminator='\n')
        #    cases = []
        #    for row_idx, row in enumerate(cr, start=1):
        #        # If not included, add a "Patient" and "Reader" column.
        #        if 'Patient' not in row:
        #            row['Patient'] = row_idx
        #        if 'Reader' not in row:
        #            row['Reader'] = 'N-A'
        #        cases.append(row)
        #return cases


def write_features():

    pass



if __name__ == '__main__':

    path_ct_dir = './../../data/images/stacks_ct/cropped_ct'
    path_pet_dir = './../../data/images/stacks_pet/cropped_pet'
    path_masks_dir = './../../data/images/masks/prep_masks'

    read_samples(path_ct_dir, path_masks_dir)
