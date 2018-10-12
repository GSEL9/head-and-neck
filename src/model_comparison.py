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


TMP_RESULTS_DIR = 'tmp_model_comparison'


def model_comparison(*args, verbose=1, score_func=None, n_jobs=None, **kwargs):
    """Collecting repeated average performance measures of selected models.

    """
    (
        comparison_scheme, X, y, estimators, estimator_params,
        selectors, selector_params, random_states, n_splits
    ) = args

    global TMP_RESULTS_DIR

    # Setup temporary directory.
    path_tempdir = ioutil.setup_tempdir(TMP_RESULTS_DIR)

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for estimator_name, estimator in estimators.items():

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(estimator_params[estimator_name])

        for selector_name, selector_func in selectors.items():
            selector = {
                'name': selector_name, 'func': selector_func,
                'params': selector_params[selector_name]
            }
            # Repeated experimental results.
            results.extend(
                joblib.Parallel(
                    n_jobs=n_jobs, verbose=verbose
                )(
                    joblib.delayed(comparison_scheme)(
                        X, y, estimator, hparam_grid, selector, n_splits,
                        random_state, path_tempdir, verbose=verbose,
                        score_func=score_func
                    ) for random_state in random_states
                )
            )
    # Remove temporary directory if process completed succesfully.
    #ioutil.teardown_tempdir(TMP_RESULTS_DIR)

    return results


if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.datasets import load_breast_cancer

    # NB ERROR:
    # For imbalanced datasets, the Average Precision metric is sometimes a
    # better alternative to the AUROC. The AP score is the area under the precision-recall curve.
    # https://stats.stackexchange.com/questions/222558/classification-evaluation-metrics-for-highly-imbalanced-data
    # REF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4403252/
    from sklearn.metrics import roc_auc_score

    df_X = pd.read_csv('')
    df_y = pd.read_csv('')

    n_splits = 2
    random_states = np.arange(3)

    estimators = {
        'rforest': RandomForestClassifier,
        'logreg': LogisticRegression
    }
    estimator_params = {
        'logreg': {'C': [0.001, 0.05, 0.1]},
        'rforest': {'n_estimators': [10, 15]}
    }
    selectors = {
        'sff': feature_selection.forward_floating,
        'relieff': feature_selection.relieff,
        'var_thresh': feature_selection.variance_threshold,
        'anovaf': feature_selection.anova_fvalue,
        'mutual_info': feature_selection.mutual_info,
        #'permut_imp': feature_selection.permutation_importance
    }
    selector_params = {
        'sff': {
            'model': RandomForestClassifier(n_estimators=10, random_state=0),
            'k': 10, 'cv': 2, 'scoring': 'accuracy'#'roc_auc'
        },
        'relieff': {'k': 10, 'n_neighbors': 100},
        'var_thresh': {'alpha': 0.05},
        'anovaf': {'alpha': 0.05},
        'mutual_info': {'n_neighbors': 3, 'thresh': 0.1},
        #'permut_imp': {
        #    'model': RandomForestClassifier(n_estimators=10, random_state=0),
        #    'thresh': 0, 'nreps': 2
        #}
    }
    selection_scheme = model_selection.nested_cross_val
    #selection_scheme = model_selection.bootstrap_point632plus
    #results = model_comparison(
    #    selection_scheme, X, y, estimators, estimator_params, selectors,
    #    selector_params, random_states, n_splits, score_func=roc_auc_score
    #)
    #ioutil.write_final_results(
    #    './../../data/outputs/model_comparison/test.csv',
    #    results
    #)
