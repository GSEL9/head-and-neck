# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
Algorithm: model comparison
1. for each model
2.     for each random state
           # Enter into nested cross val (\approx unbiased model preformance)
3.         for each outer k-fold
4.             best model, best features = inner k-folds (grid search CV)
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import shutil
import ioutil
import logging
import pathlib

import numpy as np
import pandas as pd

from datetime import datetime
from collections import Counter, OrderedDict

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

from multiprocessing import cpu_count

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


EXP_RESULTS = 'model_comparison_tmp'


class PathTracker:

    def __init__(self, root=None):

        if root is None:
            self.root = os.getcwd()
        else:
            self.root = root

        # NOTE:
        self.path = None
        self.prev_path = None

    def dir_down(self, extension):

        if self.prev_path is None:
            self.prev_path = self.root
        else:
            self.prev_path = self.path

        self.path = os.path.join(self.prev_path, extension)

        return self

    def dir_up(self, extension):

        self.prev_path, _ = os.path.split(self.path)

        self.dir_down(extension)

        return self

    def same_dir(self, extension):

        self.path, _ = os.path.split(self.path)

        self.dir_down(extension)

        return self

    def produce(self):

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        return self

    def reset_to_root(self):

        self.path, self.prev_path = None, None

        return self

    def teardown(self):
        """Removes directory even if not empty."""

        shutil.rmtree(self.root)

        return self


def multi_intersect(arrays):
    """Determines the intersection between multiple sets."""

    sets = [set(array) for array in arrays]
    matches = set.intersection(*sets)

    return list(matches)


def write_comparison_results(path_to_file, results):
    """Writes model copmarison results to CSV file."""

    data = []
    for name, experiments in results.items():

        frame = pd.DataFrame([experiment for experiment in experiments])
        frame.index = [name] * frame.shape[0]
        data.append(frame)

    output = pd.concat(data)
    output.to_csv(path_to_file, sep=',')

    return None


def model_comparison(*args, verbose=2, n_jobs=None, **kwargs):
    # Collecting repeated average performance data of optimal models.
    estimators, param_grids, X, y, random_states, n_splits = args

    global EXP_RESULTS

    #logger = logging.getLogger(__name__)
    #logger.info('Model comparison')

    path_tracker = PathTracker()

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    experiment, comparison_results = None, {}
    for name, estimator in estimators.items():

        path_tracker.dir_down('{}/{}'.format(EXP_RESULTS, name)).produce()

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(param_grids[name])

        # Repeated experimental results.
        comparison_results[estimator.__name__] = joblib.Parallel(
            n_jobs=n_jobs, verbose=verbose
        )(
            joblib.delayed(nested_cross_val)(
                X, y, estimator, hparam_grid, n_splits, random_state,
                path_tracker, verbose=verbose
            ) for random_state in random_states
        )
        path_tracker.reset_to_root()

    # Write results to disk.
    ioutil.write_comparison_results(
        './comparison_results.csv', comparison_results
    )
    # Remove temporary directory if process completed succesfully.
    path_tracker.teardown()

    return None


def nested_cross_val(*args, verbose=1, **kwargs):
    # Collecting average performance data of optimal model.
    X, y, estimator, hparam_grid, n_splits, random_state, path_tracker = args

    # Outer cross-validation loop.
    kfolds = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    train_scores, test_scores, best_features = [], [], []
    for fold_num, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Determine best model and feature subset.
        best_model, sel_features = grid_search(
            estimator, hparam_grid, X_train, y_train, n_splits, random_state
        )
        # Z-scores and selected features from inner CV loop.
        X_train_std_sub, X_test_std_sub = train_test_z_scores(
            X_train[:, sel_features], X_test[:, sel_features]
        )
        best_model.fit(X_train_std_sub, y_train)
        # Aggregate model predictions with hparams and feature subset.
        train_scores.append(
            roc_auc_score(y_train, best_model.predict(X_train_std_sub))
        )
        test_scores.append(
            roc_auc_score(y_test, best_model.predict(X_test_std_sub))
        )
        best_features.append(sel_features)

        # TODO:
        # Write preliminary outer fold results.

    return {
        'experiment_id': random_state,
        'best_params': best_model.get_params(),
        'avg_test_score': np.mean(test_scores),
        'avg_train_score': np.mean(train_scores),
        'best_features': multi_intersect(best_features)
    }


def grid_search(*args, **kwargs):
    # TEMP:
    estimator, hparam_grid, X, y, n_splits, random_state = args

    best_score, best_model, best_features = 0.0, None, None
    for combo_num, hparams in enumerate(hparam_grid):

        # Setup model.
        model = estimator(**hparams, random_state=random_state)

        # Inner cross-validation loop.
        kfolds = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
        # Determine average model performance for each hparam combo.
        train_scores, test_scores, sel_features = [], [], []
        for num, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_score, test_score, features = select_fit_predict(
                model, X_train, X_test, y_train, y_test, random_state
            )
            train_scores.append(train_score), test_scores.append(test_score)
            sel_features.append(features)

        # Write preliminary inner fold results.
        # TODO:
        #path_results_file = os.path.join(inner_X)
        #ioutil.write_prelim_result(path_results_file, results)

        # Update globally improved score from hparam combo and feature subset.
        if np.mean(test_scores) > best_score:
            best_score = np.mean(test_scores)
            best_model, best_features = model, sel_features

    return best_model, multi_intersect(sel_features)


def select_fit_predict(*args, **kwargs):
    # TEMP:
    model, X_train, X_test, y_train, y_test, random_state = args

    # Z-scores.
    X_train_std, X_test_std = train_test_z_scores(X_train, X_test)

    # NB: Ensure parallel feature sel. Save all selected features to disk.

    # Feature selection based on training set to avoid information leakage.
    concensus_support = np.arange(X_train.shape[1])
    X_train_std_sub = X_train_std[:, concensus_support]
    X_test_std_sub = X_test_std[:, concensus_support]

    model.fit(X_train_std_sub, y_train)
    # Aggregate model predictions with hparams combo selected feature subset.
    train_score = roc_auc_score(y_train, model.predict(X_train_std_sub))
    test_score = roc_auc_score(y_test, model.predict(X_test_std_sub))

    return train_score, test_score, concensus_support


def train_test_z_scores(X_train, X_test):
    """Compute Z-scores for training and test sets.

    Args:
        X_train (array-like): Training set.
        X_test (array-like): Test set.

    Returns:
        (tuple): Standard score values for training and test set.

    """

    # NOTE: Transform test data with training parameters set avoiding
    # information leakage.

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


if __name__ == '__main__':
    # TODO: Setup temporary directories:
    # model_comparison_tmp
    #   exp_X_round_Y (X=model name, Y=random_state)
    #

    # NOTE: Use Elastic and RF because indicated with good performance in unbiased
    # studies?

    # NOTE: Dealing with class imbalance:
    # * Better to use ROC on imbalanced data sets
    # * In scikit-learn: Assign larger penalty to wrong predictions on the
    #   minority class with class_weight='balanced' among model params.
    # * Upsampling of minority class/downsamlping majority class/generation of
    #   synthetic samples. See scikit-learn resample function to upsample minority
    #  function.
    # * Generate synthetic samples with SMOTE algorithm (p. 216).

    import feature_selection

    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import ElasticNet

    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score

    cancer = load_breast_cancer()
    # NB: roc_auc_score requires binary <int> target values.
    y = cancer.target
    X = cancer.data

    #n_splits = 10
    # random_states = np.arange(100)

    # TEMP:
    n_splits = 2
    random_states = np.arange(3)

    estimators = {
        'logreg': LogisticRegression,
        'elnet': ElasticNet
    }
    param_grids = {
        'logreg': {
            'C': [0.001, 0.05, 0.1], 'fit_intercept': [True, False]
        },
        'elnet': {
            'alpha': [0.05, 0.1], 'l1_ratio':[0.1, 0.5]
        }
    }
    results = model_comparison(
        estimators, param_grids, X, y, random_states, n_splits
    )
