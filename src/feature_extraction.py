# -*- coding: utf-8 -*-
#
# feature_extraction.py
#

"""
Parallel batch processing of image feature extraction.


Steps taken inside pyradiomics feature extractor .execute():
-----------------------------------------------------------------------
1. Image and mask are loaded and normalized/resampled if necessary.
2. Validity of ROI is checked using :py:func:`~imageoperations.checkMask`,
    which also computes and returns the bounding box.
3. If enabled, provenance information is calculated and stored as part of the
    result. (Not available in voxel-based extraction)
4. Shape features are calculated on a cropped (no padding) version of the
    original image. (Not available in voxel-based extraction)
5. If enabled, resegment the mask based upon the range specified in
    ``resegmentRange`` (default None: resegmentation disabled).
6. Other enabled feature classes are calculated using all specified image
    types in ``_enabledImageTypes``. Images are cropped to tumor mask
    (no padding) after application of any filter and before being passed to the
    feature class.
7. The calculated features are returned as ``collections.OrderedDict``.

"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import csv
import ioutil
import shutil
import logging
import radiomics
import threading

import SimpleITK as sitk

from collections import OrderedDict
from datetime import datetime
from multiprocessing import cpu_count, Pool
from radiomics.featureextractor import RadiomicsFeaturesExtractor


threading.current_thread().name = 'Main'


# File variables
ROOT = os.getcwd()
# Parameter file
PARAMS = './../../data/extraction_settings/ct_extract_settings1.yaml'
# Location of output log file.
LOG = os.path.join(ROOT, 'log.txt')

# Assumes the input CSV has at least 2 columns: "Image" and "Mask"
# These columns indicate the location of the image file and mask file, respectively
# Additionally, this script uses 2 additonal Columns: "Patient" and "Reader"
# These columns indicate the name of the patient (i.e. the image), the reader (i.e. the segmentation), if
# these columns are omitted, a value is automatically generated ("Patient" = "Pt <Pt_index>", "Reader" = "N/A")
INPUTCSV = os.path.join(ROOT, 'testCases.csv')

path_ct_dir = './../../data/images/ct_stacks/cropped_ct/'
#path_pet_dir = './../../data/images/pet_stacks/cropped_per/'
path_masks_dir = './../../data/images/masks/prep_masks/'
rel_ct_paths, rel_mask_paths = ioutil.relative_paths_pairwise(
    path_ct_dir, path_masks_dir, target_format='nrrd'
)
OUTPUTCSV = os.path.join(ROOT, 'results.csv')


# Parallel processing variables.
TEMP_DIR = '_temp'
# Remove temporary directory when results have been successfully stored into one file.
REMOVE_TEMP_DIR = True
# Number of processors to use keeping one spare processor for alternate work.
NUM_OF_WORKERS = cpu_count() - 1
# In case only one processor is available: ensure that it is used.
if NUM_OF_WORKERS < 1:
    NUM_OF_WORKERS = 1

# Header labels of all extracted features
HEADERS = None


# Creates a log file in the root folder. Setup logging:
rlogger = radiomics.logger
log_handler = logging.FileHandler(filename=LOG, mode='a')
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(
    logging.Formatter(
        '%(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'
    )
)
rlogger.addHandler(log_handler)


# TODO: To utils.
class InfoFilter(logging.Filter):
    """An information filter enabling messages from a specified filter in
    additon to level INFO and up. Enables level WARNING and up from other
    loggers.

    Args:
        name (str): Name of the filter to enable messages from.

    Attributes:
        level ():

    """

    def __init__(self, name):

        super(InfoFilter, self).__init__(name)

        self.level = logging.WARNING

    def filter(self, record):
        """

        Args:
            record ():

        Returns:
            (bool):

        """

        if record.levelno >= self.level:
            return True
            if record.name == self.name and record.levelno >= logging.INFO:
                return True
            # else:
            return False


# Adding the filter to the first handler of the radiomics logger limits the
# info messages on the output to just those from radiomics.batch, but warnings
# and errors from the entire library are also printed to the output. This does
# not affect the amount of logging stored in the log file.
# Handler printing to the output:
output_handler = rlogger.handlers[0]
output_handler.setFormatter(
    logging.Formatter(
        '[%(asctime)-.19s] (%(threadName)s) %(name)s: %(message)s'
    )
)
# NOTE: Ensures that INFO messages are being passed to the filter.
output_handler.setLevel(logging.INFO)
output_handler.addFilter(InfoFilter('radiomics.batch'))

logging.getLogger('radiomics.batch').debug('Logging init')


# TODO: Refactor
# NOTE: The run functino accepts one case. With the pool.map, the function is
# used multiple times in parallel with one case each.
def run(case):

    # Collects params from global scope.
    global PARAMS, ROOT, TEMP_DIR

    ptLogger = logging.getLogger('radiomics.batch')
    feature_vector = OrderedDict(case)

    # Set thread name to patient name
    try:
        threading.current_thread().name = case['Patient']

        filename = r'features_' + str(case['Reader']) + '_' + str(case['Patient']) + '.csv'
        output_filename = os.path.join(ROOT, TEMP_DIR, filename)

    # Output already generated, load result (prevents re-extraction in case of interrupted process)
    if os.path.isfile(output_filename):

        with open(output_filename, 'w') as outputFile:

            reader = csv.reader(outputFile)
            headers = reader.rows[0]
            values = reader.rows[1]
            feature_vector = OrderedDict(zip(headers, values))

        ptLogger.info('Patient %s read by %s already processed...', case['Patient'], case['Reader'])

    else:
        star_time = datetime.now()

        #imageFilepath = case['Image']  # Required
        #maskFilepath = case['Mask']  # Required
        #label = case.get('Label', None)  # Optional

        # Instantiate Radiomics Feature extractor and extract features.
        extractor = RadiomicsFeaturesExtractor(PARAMS)
        feature_vector.update(
            extractor.execute(path_to_image, path_to_mask), label=None#label
        )

        # Store results in temporary separate files to prevent write conflicts
        # This allows for the extraction to be interrupted. Upon restarting, already processed cases are found in the
        # TEMP_DIR directory and loaded instead of re-extracted.
        with open(output_filename, 'w') as outputFile:
            writer = csv.DictWriter(
                outputFile,
                fieldnames=list(feature_vector.keys()),
                lineterminator='\n'
            )
            writer.writeheader()
            writer.writerow(feature_vector)

      # Display message
      delta_time = datetime.now() - star_time

      #ptLogger.info('Patient %s read by %s processed in %s', case['Patient'], case['Reader'], delta_t)

    except Exception:
        ptLogger.error('Feature extraction failed!', exc_info=True)

    return feature_vector


# TODO: to ioutil.
def _writeResults(path_to_file, features, headers=None):

    global HEADERS, OUTPUTCSV

    # Use the lock to prevent write access conflicts
    try:
        with open(path_to_file, 'a') as outfile:

            writer = csv.writer(outfile, lineterminator='\n')

            # NB: Make sure that header status is changed after intial writing.
            if headers is None:
                writer.writerow(list(features.keys()))

            row = []
            for header in headers:
                row.append(features.get(header, 'N/A'))
                writer.writerow(row)

    except Exception:
        error_msg = 'error writing results'
        logging.getLogger('radiomics.batch').error(error_msg, exc_info=True)

    #return None


if __name__ == '__main__':
    logger = logging.getLogger('radiomics.batch')

    # Ensure the entire extraction is handled on 1 thread
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    # Setup the pool processing:

    logger.info('pyradiomics version: {}'.format(radiomics.__version__))
    logger.info('Loading CSV...')

    # Extract List of cases
    try:
        with open(INPUTCSV, 'r') as inFile:
            # Create an object that operates like a regular reader but maps the
            # information in each row to an OrderedDict whose keys are given by
            # the optional fieldnames parameter.
            cr = csv.DictReader(inFile, lineterminator='\n')
            cases = []
            for row_idx, row in enumerate(cr, start=1):
                # If not included, add a "Patient" and "Reader" column.
                if 'Patient' not in row:
                    row['Patient'] = row_idx
                if 'Reader' not in row:
                    row['Reader'] = 'N-A'
                cases.append(row)

    except Exception:
        logger.error('CSV READ FAILED', exc_info=True)

    logger.info('Loaded %d jobs', len(cases))

    # Make temporary output directory if necessary
    if not os.path.isdir(os.path.join(ROOT, TEMP_DIR)):
        logger.info('Creating temporary output directory %s', os.path.join(ROOT, TEMP_DIR))
        os.mkdir(os.path.join(ROOT, TEMP_DIR))

    # Start parallel processing:
    logger.info('Starting parralel pool with %d workers out of %d CPUs', NUM_OF_WORKERS, cpu_count())
    # Running the Pool
    pool = Pool(NUM_OF_WORKERS)
    results = pool.map(run, cases)

    try:
        # Store all results into 1 file
        with open(OUTPUTCSV, mode='w') as outputFile:
            writer = csv.DictWriter(
                outputFile, fieldnames=list(results[0].keys()), restval='',
                # raise error when a case contains more headers than first case
                extrasaction='raise', lineterminator='\n'
            )
        writer.writeheader()
        writer.writerows(results)

        if REMOVE_TEMP_DIR:
            logger.info('Removing temporary directory %s (contains individual case results files)', os.path.join(ROOT, TEMP_DIR))
            shutil.rmtree(os.path.join(ROOT, TEMP_DIR))

    except Exception:
        logger.error('Error storing results into single file!', exc_info=True)
