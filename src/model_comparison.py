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

from datetime import datetime
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

        print('Running estimator: {}\n{}'.format(estimator.__name__, '-' * 30))

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(estimator_params[estimator_name])

        for selector_name, selector_func in selectors.items():

            print('Running selector: {}\n{}'.format(selector_name, '-' * 30))

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
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVC, LinearSVC
    from pyearth import Earth
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score

    df_X = pd.read_csv(
        './../../data/to_analysis/squareroot_/ct3_pet0_clinical.csv',
        index_col=0
    )
    X = df_X.values

    df_y_pfs = pd.read_csv(
        './../../data/to_analysis/target_pfs.csv', index_col=0
    )
    y = np.squeeze(df_y_pfs.values)

    n_splits = 2
    random_states = np.arange(5)

    # NB NB NB NB NB NB NB NB NB NB NB
    # * Revisit model comparison procedure to check validity of each step. Pay
    #   particular attention to steps in retaining feature subsets.


    # NOTE:
    # Errors in A analysis:
    # * Did not dummy encode clinical data.

    estimators = {
        #'lda': LinearDiscriminantAnalysis,
        #'logreg': LogisticRegression,
        # NB: warnings.warn('Y residual constant at iteration %s' % k)
        #'pls': PLSRegression,
        'adaboost': AdaBoostClassifier,
        # NOTE: May report colinear vars.
        #'gnb': GaussianNB,
        #'svc': SVC,
        #'lin_svc': LinearSVC,
    }

    # NOTE: Hparam setup.
    # Use same param settings across models attempting to ensure `fair` grounds
    # for comparison.
    K, CV, SEED = 20, 4, 0

    PRIORS = [0.224, 0.776]
    N_ESTIMATORS = [5, 50, 100, 500, 1000]
    LEARNINGR_RATE = [0.05, 0.2, 0.5, 0.7, 1]
    TOL = [1e-7, 1e-5, 0.0001, 0.001, 0.01, 0.1]
    C = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0, 1000.0]

    SCORE = 'roc_auc'
    PENALTY = ['l1', 'l2']
    CLASS_WEIGHT = ['balanced']

    logreg_l1 = LogisticRegression(
        penalty='l1', class_weight='balanced', random_state=SEED
    )
    logreg_l2 = LogisticRegression(
        penalty='l2', class_weight='balanced', random_state=SEED
    )
    rf_model = RandomForestClassifier(
        n_estimators=50, class_weight='balanced', random_state=SEED
    )

    hparams = {
        'lda': {
            'n_components': N_ESTIMATORS,
            'tol': TOL, 'priors': [PRIORS],
        },
        'logreg': {
            'C': C, 'solver': ['liblinear'], 'penalty': PENALTY,
            'class_weight': CLASS_WEIGHT,
        },
        'pls': {
            'n_components': [5, 30, 70, 100, 150], 'tol': TOL, 'max_iter': [500]
        },
        'adaboost': {
            'base_estimator': [logreg_l2],
            'learning_rate': LEARNINGR_RATE, 'n_estimators': N_ESTIMATORS,
        },
        'svc': {
            'kernel': ['rbf'], 'C': C, 'gamma': [0.001, 0.01, 0.05, 0.1, 0.2],
            'cache_size': [30, 50, 70, 200, 300], 'degree': [2, 3],
            'class_weight': CLASS_WEIGHT
        },
        'lin_svc': {
            'C': C, 'class_weight': CLASS_WEIGHT, 'penalty': PENALTY,
            'dual': [False], 'tol': TOL,
        },
        'gnb': {'priors': [PRIORS]},
    }

    selectors = {
        #'rf_permut_imp': feature_selection.permutation_importance,
        #'ff_logregl1': feature_selection.forward_floating,
        #'ff_logregl2': feature_selection.forward_floating,
        #'ff_rf': feature_selection.forward_floating,
        'var_thresh': feature_selection.variance_threshold,
        #'relieff': feature_selection.relieff,
        #'mutual_info': feature_selection.mutual_info,
    }

    selector_params = {
        # Wrapper methods:
        'rf_permut_imp': {'model': rf_model, 'thresh': 0.0, 'nreps': 1},
        'ff_rf': {'model': rf_model, 'k': K, 'cv': CV, 'scoring': SCORE},
        'ff_logregl1': {'model': logreg_l1, 'k': K, 'cv': CV, 'scoring': SCORE},
        'ff_logregl2': {'model': logreg_l2, 'k': K, 'cv': CV, 'scoring': SCORE
        },
        # Filter methods:
        'var_thresh': {'alpha': 0.05},
        'relieff': {'k': K, 'n_neighbors': 20},
        'mutual_info': {'n_neighbors': 20, 'thresh': 0.05},
    }

    selection_scheme = model_selection.nested_cross_val

    #"""

    # TODO:
    start_time = datetime.now()
    results = model_comparison(
        selection_scheme, X, y, estimators, hparams, selectors,
        selector_params, random_states, n_splits, score_func=roc_auc_score
    )
    print('Execution time: {}'.format(datetime.now() - start_time))

    print(results)
    #ioutil.write_final_results(
    #    './../../data/outputs/model_comparison/lbp/ct0_pet0_clinical.csv',
    #    results
    #)
    #"""
    """

    ref_feature_dir = './../../data/to_analysis'
    ref_results_dir = './../../data/outputs/model_comparison'

    filter_cats = [
        label for label in os.listdir(ref_feature_dir)
        if not label.endswith('.csv') and not label.startswith('.')
    ]
    df_y = np.squeeze(
        pd.read_csv('./../../data/to_analysis/target.csv', index_col=0).values
    )

    for filter_cat in filter_cats:

        path_filter_cat_feature_sets = ioutil.relative_paths(
            os.path.join(ref_feature_dir, filter_cat), target_format='.csv'
        )
        for num, path_feature_set in enumerate(path_filter_cat_feature_sets):

            X = pd.read_csv(path_feature_set, index_col=0).values

            start_time = datetime.now()

            # TODO: Plot a progress bar.
            # print('Collecting results:')
            results = model_comparison(
                selection_scheme, X, y, estimators, hparams, selectors,
                selector_params, random_states, n_splits,
                score_func=roc_auc_score
            )
            # TODO: Print collection time.
            path_results = os.path.join(
                ref_results_dir, filter_cat, os.path.basename(path_feature_set)
            )
            ioutil.write_final_results(path_results, results)
            print('Saving results to: {}'.format(path_results))
    """
