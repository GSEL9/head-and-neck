

import re
import os
import csv
import nrrd
import shutil

import pandas as pd
import scipy.io as sio

from joblib import Parallel, delayed

from collections import OrderedDict


def _typecheck(item):
    # Return <int> if able to convert, else <str>.

    return int(item) if item.isdigit() else item


def sample_num(sample):

    sample_ids = re.split('(\d+)', sample)
    for elem in sample_ids:
        if elem.isdigit():
            return int(elem)

    return None


def natural_keys(text):
    """
    """

    return [_typecheck(item) for item in re.split('(\d+)', text)]


def relative_paths(path_to_dir, sorted=True, target_format=None):
    """Produce a list of relative paths to all files in directory."""

    if not os.path.isdir(path_to_dir):
        raise RuntimeError('Invalid path {}'.format(path_to_dir))

    file_names = os.listdir(path_to_dir)

    rel_paths = []
    for file_name in file_names:

        rel_path = os.path.join(path_to_dir, file_name)

        if os.path.isfile(rel_path) and rel_path.endswith(target_format):
            rel_paths.append(rel_path)

        if sorted:
            rel_paths.sort(key=natural_keys)

    return rel_paths


def read_matlab(path_to_dir):

    rel_paths = relative_paths(path_to_dir, target_format='mat')
    images = Parallel(n_jobs=n_jobs, verbose=verbose)(
        sio.loadmat(rel_path) for rel_path in rel_paths
    )
    return images


def read_nrrd():

    # In case need to convert raw data files.
    pass


def write_nrrd(path_to_dir, images):

    for image in images:
        nrrd.write(image)

    return None


def matlab_to_nrrd(path_mat_dir, path_nrrd_dir):

    mat_images = read_matlab(path_mat_dir)
    write_nrrd(path_nrrd_dir, mat_images)

    return None


# Assumes the input CSV has at least 2 columns: "Image" and "Mask"
# These columns indicate the location of the image file and mask file, respectively
# Additionally, this script uses 2 additonal Columns: "Patient" and "Reader"
# These columns indicate the name of the patient (i.e. the image), the reader (i.e. the segmentation), if
# these columns are omitted, a value is automatically generated ("Patient" = "Pt <Pt_index>", "Reader" = "N/A")
def sample_paths(path_image_dir, path_mask_dir, target_format='nrrd'):
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

    #frame = pd.DataFrame([result for result in results])
    #frame.to_csv(path_to_file)

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

    with open(path_to_file, mode='r'):

        results = pd.read_csv(path_to_file).to_dict()

    return results


def write_prelim_results(path_to_file, results):
    """Store results in temporary separate files to prevent write conflicts
    # This allows for the extraction to be interrupted. Upon restarting,
    # already processed cases are found in the TEMP_DIR directory and
    # loaded instead of re-extracted.
    """

    with open(path_to_file, 'w') as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=list(results.keys()), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerow(results)

    return None


# ERROR:
def write_comparison_results(path_to_file, results):
    """Writes model copmarison results to CSV file."""

    data = pd.DataFrame([result for result in results])

    data.to_csv(path_to_file)

    return None


def setup_tempdir(tempdir, root=None):
    """Returns path and sets up directory if non-existent."""

    if root is None:
        root = os.getcwd()

    path_tempdir = os.path.join(root, tempdir)
    if not os.path.isdir(path_tempdir):
        os.mkdir(path_tempdir)

    return path_tempdir


def teardown_tempdir(path_to_dir):
    """Removes directory even if not empty."""

    shutil.rmtree(path_to_dir)

    return None


if __name__ == '__main__':

    path_ct_dir = './../../data/images/stacks_ct/cropped_ct'
    path_pet_dir = './../../data/images/stacks_pet/cropped_pet'
    path_masks_dir = './../../data/images/masks/prep_masks'

    path_raw_ct_dir = './../../data/images/stacks_ct/raw_ct'

    #samples = read_samples(path_ct_dir, path_masks_dir)
    #print(samples[0].keys())

    read_matlab(path_raw_ct_dir)
