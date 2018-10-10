# -*- coding: utf-8 -*-
#
# feature_extraction.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import time
import ioutil
import shutil
import csv
import pandas as pd
import os
from datetime import datetime
import SimpleITK as sitk
from joblib import Parallel, delayed
from collections import OrderedDict
import radiomics
import threading

from multiprocessing import cpu_count
from radiomics.featureextractor import RadiomicsFeaturesExtractor


threading.current_thread().name = 'Main'


TMP_EXTRACTION_DIR = 'tmp_feature_extraction'


def feature_extraction(param_file, samples, verbose=0, n_jobs=None, **kwargs):

    global TMP_EXTRACTION_DIR

    if not os.path.isfile(param_file):
        raise ValueError('Invalid path param file: {}'.format(param_file))

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

    features = OrderedDict(case)

    try:
        threading.current_thread().name = case['Patient']

        case_file = ('_').join(('features', str(case['Patient']), '.csv'))
        path_case_file = os.path.join(path_tempdir, case_file)

        if os.path.isfile(path_case_file):
            # Load results stored prior to process abortion.
            features = ioutil.read_prelim_result(path_case_file)

            if verbose > 1:
                print('Loading previously extracted features.')

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

    except:
        raise RuntimeError('Unable to extract features.')

    return features


if __name__ == '__main__':

    path_ct_dir = './../../data/images/ct_cropped_prep/'
    path_pet_dir = './../../data/images/pet_cropped_prep/'
    path_masks_dir = './../../data/images/masks_cropped_prep/'

    ct_param_files = [
        './../../data/extraction_settings/ct_extract_settings1.yaml',
        './../../data/extraction_settings/ct_extract_settings2.yaml',
        './../../data/extraction_settings/ct_extract_settings3.yaml',
        './../../data/extraction_settings/ct_extract_settings4.yaml',
        './../../data/extraction_settings/ct_extract_settings5.yaml'
    ]
    pet_param_files = [
        './../../data/extraction_settings/pet_extract_settings1.yaml',
        './../../data/extraction_settings/pet_extract_settings2.yaml',
        './../../data/extraction_settings/pet_extract_settings3.yaml',
        './../../data/extraction_settings/pet_extract_settings4.yaml',
        './../../data/extraction_settings/pet_extract_settings5.yaml'
    ]
    path_raw_ct_features = [
        './../../data/outputs/ct_feature_extraction/raw_ct_features1.csv',
        './../../data/outputs/ct_feature_extraction/raw_ct_features2.csv',
        './../../data/outputs/ct_feature_extraction/raw_ct_features3.csv',
        './../../data/outputs/ct_feature_extraction/raw_ct_features4.csv',
        './../../data/outputs/ct_feature_extraction/raw_ct_features5.csv'
    ]
    path_raw_pet_features = [
        './../../data/outputs/pet_feature_extraction/raw_pet_features1.csv',
        './../../data/outputs/pet_feature_extraction/raw_pet_features2.csv',
        './../../data/outputs/pet_feature_extraction/raw_pet_features3.csv',
        './../../data/outputs/pet_feature_extraction/raw_pet_features4.csv',
        './../../data/outputs/pet_feature_extraction/raw_pet_features5.csv'
    ]

    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    paths_ct_samples = ioutil.sample_paths(
        path_ct_dir, path_masks_dir, target_format='nrrd'
    )
    paths_pet_samples = ioutil.sample_paths(
        path_pet_dir, path_masks_dir, target_format='nrrd'
    )

    for num, _ in enumerate(paths_pet_samples):

        print('Starting run: {}\nParam file: {}'.format(num, param_file))

        start_time = datetime.now()
        raw_pet_outputs = feature_extraction(pet_param_files[num], paths_pet_samples)
        print('Duration feature extraction: {}'.format(datetime.now() - start_time))

        ioutil.write_final_results(path_raw_pet_features[num], raw_pet_outputs)
        print('Duration extraction process: {}'.format(datetime.now() - start_time))


    for num, _ in enumerate(paths_ct_samples):

        print('Starting run: {}\nParam file: {}'.format(num, param_file))

        start_time = datetime.now()
        raw_ct_outputs = feature_extraction(ct_param_files[num], paths_ct_samples)
        print('Duration feature extraction: {}'.format(datetime.now() - start_time))

        ioutil.write_final_results(path_raw_ct_features[num], raw_ct_outputs)
        print('Duration extraction process: {}'.format(datetime.now() - start_time))
