import ioutil
import numpy as np
import pandas as pd
import logging


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
        'Image', 'Mask', 'Reader', 'label', 'general'
    ]

    def __init__(self, path_to_features=None, verbose=0):

        self.verbose = verbose
        self.data = {
            num: pd.read_csv(fpath, index_col=0)
            for num, fpath in enumerate(path_to_features)
        }

        # NOTE:
        self._steps = None

    def rename_columns(self, labels=None, add_extend=None):

        # Assign new column labels to each data set.
        if labels is not None:
            for feature_set in self.data.values():
                feature_set.columns = labels

        # Modify column labels with extension.
        elif add_extend is not None:

            for data in self.data.values():
                new_labels = [
                    '{}_{}'.format(add_extend, col) for col in data.columns
                ]
                data.columns = new_labels

        return self


    def check_identifiers(self, id_col=None, target_id=None):

        if id_col is not None:

            for feature_set in self.data.values():

                feature_set.index = feature_set.loc[:, id_col]
                if id_col in feature_set.columns:
                    feature_set.drop(id_col, axis=1, inplace=True)

        # Check matching identifier.
        if target_id is not None:

            for feature_set in self.data.values():

                if sum(feature_set.index != target_id) > 0:
                    raise RuntimeError('Different samples in feature set and '
                                       'reference samples!')
                else:
                    pass

        return self

    def check_features(self, steps='all'):

        if steps == 'all':

            self.filter_columns()
            self.impute_missing_values()
            #self.check_dtypes(features)

    def filter_columns(self, columns=None):

        for feature_set in self.data.values():

            # Drop default columns.
            if columns is None:

                columns = []
                for label in self.DROP_COLUMNS:
                    columns.extend(
                        list(feature_set.filter(regex=label).columns)
                    )
            feature_set.drop(columns, axis=1, inplace=True)

            if self.verbose > 0:
                print('Dropped columns including: {}'.format(self.DROP_COLUMNS))

        return self

    def impute_missing_values(self, imputer=0, thresh=2):

        for feature_set in self.data.values():

            where_missing = np.where(feature_set.isnull())
            if isinstance(imputer, (int, float)):
                feature_set.iloc[where_missing] = imputer
            else:
                raise NotImplementerError('imputer not implemented')

            return self


if __name__ == '__main__':

    path_pet_features = './../../data/outputs/pet_feature_extraction/raw_pet_features1.csv'
    post_prep = PostProcessor([path_pet_features])
    post_prep.check_features()
