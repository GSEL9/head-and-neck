
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

    # Setup temporary directory.
    path_tempdir = ioutil.setup_tempdir(TMP_EXTRACTION_DIR)

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    # Extract features.
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_extract_features)(
            param_file, sample, path_tempdir, verbose=verbose
        ) for sample in samples
    )

    return results


def _extract_features(param_file, case, path_tempdir, verbose=0):

    features = OrderedDict(case)

    try:
        # Set thread name to patient name.
        threading.current_thread().name = case['Patient']

        case_file = ('_').join(('features', str(case['Patient']), '.csv'))
        path_case_file = os.path.join(path_tempdir, case_file)

        # Load results stored prior to process abortion. Necessary to write
        # complete feature set.
        if os.path.isfile(path_case_file):
            features = ioutil.read_prelim_result(path_case_file)

        # Extract features.
        else:
            extractor = RadiomicsFeaturesExtractor(param_file)

            start_time = datetime.now()
            features.update(
                extractor.execute(case['Image'], case['Mask']),
                label=case.get('Label', None)
            )
            delta_time = datetime.now() - start_time

            # Write preliminary results to disk.
            ioutil.write_prelim_results(path_case_file, features)

    except:
        raise RuntimeError('Unable to extract features.')

    return features


if __name__ == '__main__':
    # TODO: Prepate P58 images. Create a notebook.
    # TODO: Experimental setup for feature extraction
    # 1. Create a notebook to print dynamic range(?) = number of voxel
    # intensites per image. Save the results in `images` dir.
    # 2. Determine a binwidth such that range/binwidth remains approximately
    # the same across images.
    # 3. Extract and process data into a set for analysis.


    import ioutil
    import postprep

    path_ct_dir = './../../data/images/stacks_ct/cropped_ct'
    path_pet_dir = './../../data/images/stacks_pet/cropped_pet'
    path_masks_dir = './../../data/images/masks/prep_masks'

    path_ct_features = './../../data/results/feature_extraction/features_ct/raw_features1.csv'
    path_pet_features = './../../data/results/feature_extraction/features_pet/raw_features1.csv'

    param_file = './../../data/extraction_settings/example.yaml'

    # Ensure the entire extraction is handled on 1 thread
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    paths_to_samples = ioutil.sample_paths(path_ct_dir, path_masks_dir)

    # ERROR: Proper reading prelim results.
    # Extracting raw features.
    results = feature_extraction(param_file, paths_to_samples[:6])

    # Writing raw features to disk.
    ioutil.write_final_results(path_ct_features, results)

    drop_cols = [
        'Image', 'Mask', 'Patient', 'Reader', 'label', 'general'
    ]
    features = postprep.check_features(path_ct_features, drop_cols=drop_cols)
    ioutil.write_final_results(
        './../../data/results/feature_extraction/features_ct/prep_features1.csv', 
        features
    )
