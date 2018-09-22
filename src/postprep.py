import ioutil
import numpy as np
import pandas as pd
import logging

from sklearn.preprocessing import Imputer


def check_missing_values(features, imputer='mean'):

    num_missing = np.sum(features.isnull().values)
    #logger = logging.getLogger('postprocessing')
    #logger.info('num missing values: {}'.format())

    # Replace missing values with imputated values.
    if num_missing > 0:
        imputer = Imputer(strategy=imputer, axis=1)
        features.update(imputer.fit_transform(features.values))
        pass
    return features


def check_dtypes(features, valid_dtypes=None):
    # Encode dtypes

    if valid_dtypes is None:
        pass


def _filter_columns(features, drop_cols):

    _drop_cols = []
    for label in drop_cols:
        _drop_cols.extend(list(features.filter(regex=label).columns))

    features.drop(_drop_cols, axis=1, inplace=True)

    return None


def check_extracted_features(path_to_file, drop_cols=None):

    features = pd.read_csv(path_to_file)
    # Set patient number index.
    features.index  = [ioutil.sample_num(sample) for sample in features.Image]

    if drop_cols is not None:
        _filter_columns(features, drop_cols)

    features = check_missing_values(features)
    #check_dtypes(features)

    return features



def preprocessing_report():
    # Generate a report describing preprocessing steps
    # https://docs.python.org/3/howto/logging-cookbook.html
    pass


if __name__ == '__main__':

    path_ct_features = './../../data/images/features_ct/features1.csv'
    drop_cols = [
        'Image', 'Mask', 'Patient', 'Reader', 'label', 'general'
    ]
    features = check_extracted_features(path_ct_features, drop_cols=drop_cols)
    features.to_csv(
        './../../data/images/features_ct/prep_features1.csv',
        columns=features.columns
    )
