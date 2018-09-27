
import time
import shutil
import csv
import pandas as pd
import logging
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


class InfoFilter(logging.Filter):
    # Define filter that allows messages from specified filter and level INFO
    # and up, and level WARNING and up from other loggers.

    def __init__(self, name):

        super(InfoFilter, self).__init__(name)
        self.level = logging.WARNING

    def filter(self, record):

        if record.levelno >= self.level:
            return True
        if record.name == self.name and record.levelno >= logging.INFO:
            return True

        return False


# Adding the filter to the first handler of the radiomics logger limits the info messages on the output to just those
# from radiomics.batch, but warnings and errors from the entire library are also printed to the output. This does not
# affect the amount of logging stored in the log file.
def initiate_logging(path_to_file):

    log_handler = logging.FileHandler(filename=path_to_file, mode='a')
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(
        '%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s')
    )
    rad_logger = radiomics.logger
    rad_logger.addHandler(log_handler)
    # Handler printing to the output
    outputhandler = rad_logger.handlers[0]
    outputhandler.setFormatter(
        logging.Formatter(
            '[%(asctime)-.19s] (%(threadName)s) %(name)s: %(message)s'
        )
    )
    # Ensures that INFO messages are being passed to the filter.
    outputhandler.setLevel(logging.INFO)
    # NOTE: Include InfoFilter instance.
    outputhandler.addFilter(InfoFilter('radiomics.batch'))

    logging.getLogger('radiomics.batch').debug('Logging init')

    return None


def _setup_tempdir(root=None, tempdir=None):
    # Returns path and sets up directory if necessary.

    if root is None:
        root = os.getcwd()

    if tempdir is None:
        tempdir = 'feature_extraction_tmp'

    path_tempdir = os.path.join(root, tempdir)
    if not os.path.isdir(path_tempdir):
        os.mkdir(path_tempdir)

    return path_tempdir


def _teardown_tempdir(path_to_dir):
    # Removes directory even if not empty.

    shutil.rmtree(path_to_dir)

    return None


def feature_extraction(param_file, samples, verbose=0, n_jobs=None):

    initiate_logging(os.path.join(os.getcwd(), 'extraction_log.txt'))

    logger = logging.getLogger('radiomics.batch')
    logger.info('pyradiomics version: {}'.format(radiomics.__version__))

    # Setup temporary directory.
    path_tempdir = _setup_tempdir()

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    # Extract features.
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_extract_features)(
            param_file, sample, path_tempdir
        ) for sample in samples
    )
    # Remove temporary directory if process completed succesfully.
    _teardown_tempdir(path_tempdir)

    return results


def _extract_features(param_file, case, path_tempdir):

    feature_tensor = OrderedDict(case)
    # Monitor extraction procedure.
    _logger = logging.getLogger('radiomics.batch')

    try:
        # Set thread name to patient name.
        threading.current_thread().name = case['Patient']

        case_file = ('_').join(('features', str(case['Patient']), '.csv'))
        path_case_file = os.path.join(path_tempdir, case_file)

        # Read results stored prior to process abortion.
        if os.path.isfile(path_case_file):
            feature_tensor = ioutil.read_prelim_result(path_case_file)
            _logger.info('Already processed case {}'.format(case['Patient']))

        else:
            # Extract features.
            start_time = datetime.now()
            extractor = RadiomicsFeaturesExtractor(param_file)
            feature_tensor.update(
                extractor.execute(
                    case['Image'], case['Mask']
                ),
                label=case.get('Label', None)
            )
            delta_time = datetime.now() - start_time
            _logger.info('Case {} processed in {}'.format(case['Patient'],
                                                          delta_time))
            # Write preliminary results to disk.
            ioutil.write_prelim_result(path_case_file, feature_tensor)

    except Exception:
        _logger.error('Feature extraction failed', exc_info=True)

    return feature_tensor


if __name__ == '__main__':
    # TODO: Setup logger.
    # TODO: Checking that path to param file exists.
    # TODO: Clarify meaning of settings params in extraction params file.

    import ioutil
    import postprep

    path_ct_dir = './../../data/images/stacks_ct/cropped_ct'
    path_pet_dir = './../../data/images/stacks_pet/cropped_pet'
    path_masks_dir = './../../data/images/masks/prep_masks'

    path_ct_features = './../../data/images/features_ct/raw_features1.csv'
    path_pet_features = './../../data/images/features_pet/raw_features1.csv'

    param_file = './../../data/images/extraction_settings/ct_extract_settings1.yaml'

    # Ensure the entire extraction is handled on 1 thread
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    paths_to_samples = ioutil.sample_paths(path_ct_dir, path_masks_dir)

    # Extracting raw features.
    results = feature_extraction(param_file, paths_to_samples[:3])

    # Writing raw features to disk.
    ioutil.write_features(path_ct_features, results)

    # Reading raw features.
    path_ct_features = './../../data/images/features_ct/raw_features1.csv'
    drop_cols = [
        'Image', 'Mask', 'Patient', 'Reader', 'label', 'general'
    ]

    # Processing raw features.
    features = postprep.check_extracted_features(
        path_ct_features, drop_cols=drop_cols
    )

    # Writing processed features to disk.
    features.to_csv(
        './../../data/images/features_ct/prep_features1.csv',
        columns=features.columns
    )
