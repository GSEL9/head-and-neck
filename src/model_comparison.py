# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import shutil
import logging

import numpy as np
import pandas as pd

from multiprocessing import cpu_count

from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ParameterGrid


TMP_RESULTS = 'tmp_model_comparison'


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


def model_comparison(*args, verbose=2, score_func=None, n_jobs=None, **kwargs):
    # Collecting repeated average performance data of optimal models.
    estimators, param_grids, X, y, random_states, n_splits = args

    global TMP_RESULTS

    #logger = logging.getLogger(__name__)
    #logger.info('Model comparison')

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    experiment, comparison_results = None, {}
    for name, estimator in estimators.items():

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(param_grids[name])

        # Repeated experimental results.
        comparison_results[estimator.__name__] = joblib.Parallel(
            n_jobs=n_jobs, verbose=verbose
        )(
            joblib.delayed(nested_cross_val)(
                X, y, estimator, hparam_grid, n_splits, random_state,
                verbose=verbose, score_func=score_func
            ) for random_state in random_states
        )

    # Write results to disk.
    write_comparison_results(
        './comparison_results.csv', comparison_results
    )
    return comparison_results


def nested_cross_val(*args, verbose=1, score_func=None, **kwargs):
    # Collecting average performance data of optimal model.
    X, y, estimator, hparam_grid, n_splits, random_state = args

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
            estimator, hparam_grid, X_train, y_train, n_splits, random_state,
            score_func=score_func
        )
        train_score, test_score, _ = select_fit_predict(
            best_model, X_train[:, sel_features], X_test[:, sel_features],
            y_train, y_test, random_state, score_func=score_func,
            select_feats=False
        )
        train_scores.append(train_score), test_scores.append(test_score)
        best_features.append(sel_features)

    return {
        'experiment_id': random_state,
        'best_params': best_model.get_params(),
        'avg_test_score': np.mean(test_scores),
        'avg_train_score': np.mean(train_scores),
        'best_features': multi_intersect(best_features)
    }


def grid_search(*args, score_func=None, **kwargs):
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
                model, X_train, X_test, y_train, y_test, random_state,
                select_feats=True, score_func=score_func
            )
            train_scores.append(train_score), test_scores.append(test_score)
            sel_features.append(features)

        # Update globally improved score from hparam combo and feature subset.
        if np.mean(test_scores) > best_score:
            best_score = np.mean(test_scores)
            best_model, best_features = model, sel_features

    return best_model, multi_intersect(sel_features)


def select_fit_predict(*args, select_feats=True, score_func=None, **kwargs):
    model, X_train, X_test, y_train, y_test, random_state = args

    support = None
    if select_feats:
        X_train_std, X_test_std, support = select_features(X_train, X_test)
    else:
        X_train_std, X_test_std = train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)

    # Aggregate model predictions with hparams combo selected feature subset.
    train_score = score_func(y_train, model.predict(X_train_std))
    test_score = score_func(y_test, model.predict(X_test_std))

    return train_score, test_score, support


# TODO: Feature selection.
def select_features(X_train, X_test):

    support = np.arange(X_train.shape[1])

    # Z-scores.
    X_train_std, X_test_std = train_test_z_scores(X_train, X_test)

    # Feature selection based on training set to avoid information leakage.
    X_train_std_sub = X_train_std[:, support]
    X_test_std_sub = X_test_std[:, support]

    return X_train_std_sub, X_test_std_sub, support


def train_test_z_scores(X_train, X_test):
    """Compute Z-scores for training and test sets.

    Args:
        X_train (array-like): Training set.
        X_test (array-like): Test set.

    Returns:
        (tuple): Standard score values for training and test set.

    """

    # NOTE: Avoid leakage by transforming test data with training params.
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


if __name__ == '__main__':
    # NB:
    # Setup temp dirs holding prelim results.
    # Implement feature selection.

    # TODO checkout:
    # * ElasticNet + RF
    # * Upsampling/resampling
    # * Generate synthetic samples with SMOTE algorithm (p. 216).

    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import ElasticNet

    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score

    cancer = load_breast_cancer()
    # NB: roc_auc_score requires binary <int> target values.
    y = cancer.target
    X = cancer.data

    # SETUP
    # NOTE: Number of CV folds
    n_splits = 2

    # NOTE: Number of experiments
    random_states = np.arange(3)

    estimators = {
        'logreg': LogisticRegression,
        'elnet': ElasticNet
    }
    param_grids = {
        'logreg': {
            'C': [0.001, 0.05, 0.1]
        },
        'elnet': {
            'alpha': [0.05, 0.1], 'l1_ratio':[0.1, 0.5]
        }
    }
    results = model_comparison(
        estimators, param_grids, X, y, random_states, n_splits,
        score_func=roc_auc_score
    )
