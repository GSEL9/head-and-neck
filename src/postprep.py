import ioutil
import numpy as np
import pandas as pd
import logging

from sklearn.preprocessing import Imputer



# NB ERROR:
def check_dtypes(features, valid_dtypes=None):
    # Encode dtypes

    if valid_dtypes is None:
        pass

def preprocessing_report():
    # Generate a report describing preprocessing steps
    # https://docs.python.org/3/howto/logging-cookbook.html
    pass


class PostProcessor:

    DROP_COLUMNS = [
        'Image', 'Mask', 'Patient', 'Reader', 'label', 'general'
    ]

    def __init__(self, path_to_features=None, verbose=0):

        self.verbose = verbose
        self.data = [
            pd.read_csv(fpath, index_col=0) for fpath in path_to_features
        ]

        # NOTE:
        self._steps = None

    def check_features(self, steps='all'):

        if steps == 'all':

            self.filter_columns()
            self.check_missing_values()
            #self.check_dtypes(features)

    def filter_columns(self):

        for feature_set in self.data:

            _drop_cols = []
            for label in self.DROP_COLUMNS:
                _drop_cols.extend(list(feature_set.filter(regex=label).columns))

            feature_set.drop(_drop_cols, axis=1, inplace=True)

            if self.verbose > 0:
                print('Dropped columns including: {}'.format(self.DROP_COLUMNS))

        return self

    def check_missing_values(self, imputer='mean'):

        for feature_set in self.data:

            missing_feats = feature_set.iloc[:, [103, 1391, 1483, 1575, 1667, 1759, 1851, 1943]]

            for col in missing_feats.columns:
                print(col)

            num_missing = np.sum(feature_set.isnull().values)

            #missing_cols =
            #if missing_cols > 0:

            #missing_rows = 
            #if missing_rows > 0:


            #print(feature_set)
            #print(np.where(feature_set.isnull()))
            #for row in range(feature_set.shape[0]):
            #    if feature_set.loc[row, :].isnull().sum() > 0:
            #        print(np.where(feature_set.loc[row, :].isnull()))
            # Replace missing values with imputated values.
            #if num_missing > 0:
            #    imputer = Imputer(strategy=imputer, axis=1)
            #    features.update(imputer.fit_transform(features.values))

            return self


if __name__ == '__main__':

    path_pet_features = './../../data/outputs/pet_feature_extraction/raw_pet_features5.csv'
    post_prep = PostProcessor([path_pet_features])
    post_prep.check_features()
