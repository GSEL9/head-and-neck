

import re
import os
import csv
import nrrd
import shutil

import numpy as np
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

    if target_format is None:
        raise ValueError('Must specify target format')

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


def matlab_to_nrrd(path_mat, path_nrrd, path_mask=None, modality=None):
    """Converts MATLAB formatted images to NRRD format.

    Kwargs:
        path_mask (str):
        modality (str, {`mask`, `PET`, `CT`}):

    """

    if os.path.isfile(path_mat):
        image_data = sio.loadmat(path_mat)
        image = image_data[modality]
        nrrd.write(path_nrrd, image)

        if path_mask is not None:
            mask = np.copy(image)
            mask[np.isnan(image)] = 0
            mask[mask != 0] = 1
            nrrd.write(path_mask, mask)

    elif os.path.isdir(path_mat) and os.path.isdir(path_nrrd):

        mat_rel_paths = relative_paths(path_mat, target_format='mat')
        nrrd_rel_paths = relative_paths(path_mat, target_format='mat')

        images = Parallel(n_jobs=n_jobs, verbose=verbose)(
            sio.loadmat(rel_path) for rel_path in rel_paths
        )
        for num, image in enumerate(images):
            nrrd.write(nrrd_rel_paths[num], image[modality])

    else:
        raise RuntimeError('Unable to locate:\n{}\n{}'
                           ''.format(path_mat, path_nrrd))

    return None


def sample_paths(path_image_dir, path_mask_dir, target_format=None):
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


def read_prelim_result(path_to_file):
    """Read temporary stored results."""

    results = pd.read_csv(path_to_file, index_col=False)

    return OrderedDict(zip(*(results.columns, results.values[0])))


def write_prelim_results(path_to_file, results):
    """Store results in temporary separate files to prevent write conflicts."""

    with open(path_to_file, 'w') as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=list(results.keys()), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerow(results)

    return None


def write_final_results(path_to_file, results):
    """Write the total collection of results to disk."""

    if isinstance(results, pd.DataFrame):
        results.to_csv(path_to_file, columns=results.columns)
    else:
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
    # NOTE: Converting P58
    raw_p58_ct = './../../data/images/p58/P058CT.mat'
    raw_p58_pet = './../../data/images/p58/P058PET.mat'
    raw_p58_mask = './../../data/images/p58/P058mask.mat'

    prep_p58_ct = './../../data/images/p58/P058CT.nrrd'
    prep_p58_pet = './../../data/images/p58/P058PET.nrrd'
    prep_p58_mask = './../../data/images/p58/P058mask.nrrd'

    matlab_to_nrrd(
        raw_p58_mask, prep_p58_mask, modality='mask'
    )
    matlab_to_nrrd(
        raw_p58_ct, prep_p58_ct, modality='CT',
        path_mask='./../../data/images/p58/P058PET.nrrd'
    )
    matlab_to_nrrd(
        raw_p58_pet, prep_p58_pet, modality='PET'
    )
