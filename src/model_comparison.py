# -*- coding: utf-8 -*-
#
# model_comparison.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import utils
import ioutil
import shutil
import logging
import model_selection
import feature_selection

import numpy as np
import pandas as pd

from multiprocessing import cpu_count

from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid


PATH_TMP_RESULTS = 'tmp_model_comparison'


def model_comparison(*args, verbose=0, score_func=None, n_jobs=None, **kwargs):
    # Collecting repeated average performance data of optimal models.
    (
        comparison_scheme, estimators, selectors, param_grids, X, y,
        random_states, n_splits
    ) = args

    global PATH_TMP_RESULTS

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for name, estimator in estimators.items():

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(param_grids[name])
        for label, selector in selectors.items():

            # Repeated experimental results.
            results.extend(
                joblib.Parallel(
                    n_jobs=n_jobs, verbose=verbose
                )(
                    joblib.delayed(comparison_scheme)(
                    X, y, estimator, selector, hparam_grid, selector, n_splits,
                    random_state, verbose=verbose, score_func=score_func
                ) for random_state in random_states
            )
        )
    return results


if __name__ == '__main__':
    # NB:
    # Setup temp dirs holding prelim results.
    # Implement feature selection.

    # TODO checkout:
    # * ElasticNet + RF
    # * Upsampling/resampling
    # * Generate synthetic samples with SMOTE algorithm (p. 216).
    # * Display models vs feature sel in heat map with performance.
    # * Display model performance as function of num selected features.

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score

    cancer = load_breast_cancer()
    # NB: roc_auc_score requires binary <int> target values.
    y = cancer.target
    X = cancer.data

    # NOTE: Number of CV folds.
    # * Num outer train samples = (n_splits - 1) * 200 / n_splits
    # * Num outer test sapmles = 200 / n_splits
    # * Num inner train samples = (n_splits - 1) * Num outer train samples / n_splits
    # * Num inner test samples = Num outer train samples / n_splits
    n_splits = 5

    # NOTE: Number of experiments
    random_states = np.arange(3)

    estimators = {
        'rforest': RandomForestClassifier,
        'logreg': LogisticRegression
    }
    param_grids = {
        'logreg': {
            'C': [0.001, 0.05, 0.1]
        },
        'rforest': {
            'n_estimators': [10, 15]
        }
    }
    selectors = {
        'dummy': feature_selection.dummy
    }
    #selection_scheme = model_selection.nested_cross_val
    selection_scheme = model_selection.bootstrap_point632plus
    results = model_comparison(
        selection_scheme, estimators, selectors, param_grids, X, y,
        random_states, n_splits, score_func=roc_auc_score
    )
    ioutil.write_comparison_results(
        './../../data/results/model_comparison/model_comparison_results.csv',
        results
    )
    # Create final heat map:
    # * Group results according to classifier + feature selector
    # * Average scores across all random states
