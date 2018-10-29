# -*- coding: utf-8 -*-
#
# feature_extraction.py
#

"""
Setup for conducting parallel feature extraction of images compatible with
PyRadiomics.

The feature extraction procedure assumes a parameter file and a list with
references to all images samples. Extracted features are stored in a temporary
directory created immideately after execution and deleted after the process is
complete. If the process was incomplete, the temporary directory presist.
Hence, it is possible to reenter an aborted process from last stored results.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import time
import utils

import logging
import radiomics
import threading
import ioutil
import shutil
import csv

import pandas as pd
import SimpleITK as sitk

from datetime import datetime
from joblib import Parallel, delayed
from collections import OrderedDict
from multiprocessing import cpu_count
from radiomics.featureextractor import RadiomicsFeaturesExtractor


threading.current_thread().name = 'Main'

TMP_EXTRACTION_DIR = 'tmp_feature_extraction'


def feature_extraction(param_file, samples, verbose=0, n_jobs=None, **kwargs):
    """Extract features from PyRadimoics compatible images. Preliminary results
    are stored for re-entering the process in case of abortion.

    Args:
        param_file (str):
        samples (list):

    Kwargs:
        verbose (int): The level of verbosity during extraction.
        n_jobs (int): The number of CPUs to distribute the extraction process.
            Defaults to available - 1.

    Returns:
        (dict): The extracted features.

    """

    global TMP_EXTRACTION_DIR

    if not os.path.isfile(param_file):
        raise ValueError('Invalid path param file: {}'.format(param_file))

    threading.current_thread().name = 'Main'

    # Setup temporary directory to store preliminary results.
    path_tempdir = ioutil.setup_tempdir(TMP_EXTRACTION_DIR)

    # Set number of available CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    if verbose > 0:
        print('Initiated feature extraction.')

    # Extract features.
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_extract_features)(
            param_file, sample, path_tempdir, verbose=verbose
        ) for sample in samples
    )
    # Clean up temporary directory when process complete.
    ioutil.teardown_tempdir(TMP_EXTRACTION_DIR)

    return results


def _extract_features(param_file, case, path_tempdir, verbose=0):
    # The work function to extract features from a single sample.
    # Inputs:
    # param_file (str):
    # case (str):
    # path_tempdir (str):
    # verbose (int):

    features = OrderedDict(case)
    ptLogger = logging.getLogger('radiomics.batch')

    try:
        threading.current_thread().name = case['Patient']

        case_file = ('_').join(('features', str(case['Patient']), '.csv'))
        path_case_file = os.path.join(path_tempdir, case_file)

        if os.path.isfile(path_case_file):
            # Load results stored prior to process abortion.
            features = ioutil.read_prelim_result(path_case_file)

            if verbose > 1:
                print('Loading previously extracted features.')

            ptLogger.info('Case %s already processed', case['Patient'])

        else:
            # Extract features.
            extractor = RadiomicsFeaturesExtractor(param_file)

            if verbose > 1:
                print('Extracting features.')

            start_time = datetime.now()
            features.update(
                extractor.execute(case['Image'], case['Mask']),
                label=case.get('Label', None)
            )
            delta_time = datetime.now() - start_time

            if verbose > 1:
                print('Writing preliminary results.')

            # Write preliminary results to disk.
            ioutil.write_prelim_results(path_case_file, features)

            delta_t = datetime.now() - start_time
            ptLogger.info('Case %s processed in %s', case['Patient'], delta_t)

    except Exception:
        ptLogger.error('Feature extraction failed!', exc_info=True)

    return features


if __name__ == '__main__':

    path_ct_dir = './../../data/images/ct_cropped/'
    path_pet_dir = './../../data/images/pet_cropped/'
    path_masks_dir = './../../data/images/masks_cropped/'

    base_path = './../../data/extraction_settings/discr/'

    ct_fnames = [
        'ct_extract_set_discr1.yaml',
        'ct_extract_set_discr2.yaml',
        'ct_extract_set_discr3.yaml',
        'ct_extract_set_discr4.yaml',
    ]
    pet_fnames = [
        'pet_extract_set_discr1.yaml',
        'pet_extract_set_discr2.yaml',
        'pet_extract_set_discr3.yaml',
        'pet_extract_set_discr4.yaml',
    ]
    ct_param_files = [
        os.path.join(base_path, ct_fname) for ct_fname in ct_fnames
    ]
    pet_param_files = [
        os.path.join(base_path, pet_fname) for pet_fname in pet_fnames
    ]
    path_raw_ct_features = [
        './../../data/outputs/ct_features/raw_ct_features1.csv',
        './../../data/outputs/ct_features/raw_ct_features2.csv',
        './../../data/outputs/ct_features/raw_ct_features3.csv',
        './../../data/outputs/ct_features/raw_ct_features4.csv',
    ]
    path_raw_pet_features = [
        './../../data/outputs/pet_features/raw_pet_features1.csv',
        './../../data/outputs/pet_features/raw_pet_features2.csv',
        './../../data/outputs/pet_features/raw_pet_features3.csv',
        './../../data/outputs/pet_features/raw_pet_features4.csv',
    ]

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    paths_ct_samples = ioutil.sample_paths(
        path_ct_dir, path_masks_dir, target_format='nrrd'
    )
    paths_pet_samples = ioutil.sample_paths(
        path_pet_dir, path_masks_dir, target_format='nrrd'
    )

    utils.setup_logger()
    """
    for num, pet_param_file in enumerate(pet_param_files):

        #utils.setup_logger(unique_fname)

        print('Run: {}\nParam file: {}'.format(num+1, pet_param_file))

        start_time = datetime.now()
        raw_pet_outputs = feature_extraction(pet_param_file, paths_pet_samples)
        print('Features extracted in: {}'.format(datetime.now() - start_time))

        ioutil.write_final_results(path_raw_pet_features[num], raw_pet_outputs)
    """

    for num, ct_param_file in enumerate(ct_param_files):

        #utils.setup_logger(unique_fname)

        print('Run: {}\nParam file: {}'.format(num+1, ct_param_file))

        start_time = datetime.now()
        raw_ct_outputs = feature_extraction(ct_param_file, paths_ct_samples)
        print('Features extracted in: {}'.format(datetime.now() - start_time))

        ioutil.write_final_results(path_raw_ct_features[num], raw_ct_outputs)
