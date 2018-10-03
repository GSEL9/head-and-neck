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

    # Setup temporary directory.
    path_tempdir = ioutil.setup_tempdir(TMP_EXTRACTION_DIR)

    # Set number of CPUs.
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
    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(TMP_RESULTS_DIR)

    return results


def _extract_features(param_file, case, path_tempdir, verbose=0):

    features = OrderedDict(case)

    try:
        # Set thread name to case name.
        threading.current_thread().name = case['Patient']

        case_file = ('_').join(('features', str(case['Patient']), '.csv'))
        path_case_file = os.path.join(path_tempdir, case_file)

        # Load results stored prior to process abortion.
        if os.path.isfile(path_case_file):
            features = ioutil.read_prelim_result(path_case_file)

            if verbose > 1:
                print('Loading previously extracted features.')

        # Extract features.
        else:
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
    # TEMP: demo run

    import ioutil
    import postprep

    path_raw_pet_features = [
        './../../data/outputs/pet_feature_extraction/raw_features2.csv',
        './../../data/outputs/pet_feature_extraction/raw_features3.csv',
        './../../data/outputs/pet_feature_extraction/raw_features4.csv',
        './../../data/outputs/pet_feature_extraction/raw_features5.csv'
    ]

    path_ct_dir = './../../data/images/ct_cropped_prep/'
    path_pet_dir = './../../data/images/pet_cropped_prep/'
    path_masks_dir = './../../data/images/masks_cropped_prep/'

    param_file2 = './../../data/extraction_settings/extract_settings2.yaml'
    param_file3 = './../../data/extraction_settings/extract_settings3.yaml'
    param_file4 = './../../data/extraction_settings/extract_settings4.yaml'
    param_file5 = './../../data/extraction_settings/extract_settings5.yaml'
    param_files = [
        param_file2, param_file3, param_file4, param_file5
    ]
    # Ensure the entire extraction is handled on 1 thread
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    paths_pet_samples = ioutil.sample_paths(
        path_pet_dir, path_masks_dir, target_format='nrrd'
    )
    for num, param_file in enumerate(param_files):

        raw_pet_outputs = feature_extraction(param_file, paths_pet_samples)
        ioutil.write_final_results(path_raw_pet_features[num], raw_pet_outputs)
