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


def _cleanup(results, path_to_results):

    ioutil.write_final_results(path_to_results, results)

    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(TMP_RESULTS_DIR)

    return results


def model_comparison(*args, verbose=1, score_func=None, n_jobs=None, **kwargs):
    """Collecting repeated average performance measures of selected models.

    """
    (
        comparison_scheme, X, y, estimators, estimator_params, selectors,
        fs_params, random_states, n_splits, path_to_results
    ) = args

    global TMP_RESULTS_DIR

    # Setup temporary directory.
    path_tempdir = ioutil.setup_tempdir(TMP_RESULTS_DIR, root='.')

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    results = []
    for estimator_name, estimator in estimators.items():

        print('Running estimator: {}\n{}'.format(estimator.__name__, '-' * 30))

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(estimator_params[estimator_name])

        for fs_name, fs_func in selectors.items():

            print('Running selector: {}\n{}'.format(fs_name, '-' * 30))

            selector = {
                'name': fs_name, 'func': fs_func, 'params': fs_params[fs_name]
            }
            # Repeating experiments.
            results.extend(
                joblib.Parallel(
                    n_jobs=n_jobs, verbose=verbose
                )(
                    joblib.delayed(comparison_scheme)(
                        X, y, estimator, hparam_grid, selector, n_splits,
                        random_state, path_tempdir, verbose=verbose,
                        score_func=score_func, n_jobs=n_jobs
                    ) for random_state in random_states
                )
            )
    results = _cleanup(results, path_to_results)

    return results


if __name__ == '__main__':
    import os
    import utils
    import ioutil
    import feature_selection

    import numpy as np
    import pandas as pd

    from datetime import datetime
    from model_comparison import model_comparison
    from model_selection import nested_point632plus

    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # Setup:
    K, SEED, N_REPS = 10, 0, 50

    # Priors summing to 1.0.
    PFS_PRIORS = [0.677, 0.323]
    LRC_PRIORS = [0.753, 0.247]

    MAX_ITER = [800]

    N_ESTIMATORS = [20, 50, 100, 500, 1000]
    LEARNINGR_RATE = [0.001, 0.05, 0.2, 0.6, 1]

    TOL = [0.001, 0.01, 0.1, 0.3, 0.7, 1]
    C = [0.001, 0.01, 0.1, 1, 10, 100]

    SCORE = roc_auc_score

    PENALTY = ['l2']
    CLASS_WEIGHT = ['balanced']

    # Number of experiments.
    n_experiments = 10

    np.random.seed(SEED)
    random_states = np.random.randint(1000, size=n_experiments)

    logreg_l1 = LogisticRegression(
        penalty='l1', class_weight='balanced', random_state=SEED,
        solver='liblinear'
    )
    logreg_l2 = LogisticRegression(
        penalty='l2', class_weight='balanced', random_state=SEED,
        solver='liblinear'
    )
    rf_model = RandomForestClassifier(
        n_estimators=30, class_weight='balanced', random_state=SEED
    )

    selectors = {
        # Wrapper methods:
        'logregl1_permut_imp': feature_selection.permutation_importance,
        'rf_permut_imp': feature_selection.permutation_importance,

        # Filter methods:
        'var_thresh': feature_selection.variance_threshold,
        'relieff': feature_selection.relieff,
        'mutual_info': feature_selection.mutual_info
    }
    selector_params = {
        'logregl1_permut_imp': {'model': logreg_l1, 'thresh': 0.0, 'nreps': 1},
        'rf_permut_imp': {'model': rf_model, 'thresh': 0.0, 'nreps': 1},

        'var_thresh': {'alpha': 0.05},
        'relieff': {'k': K, 'n_neighbors': 20},
        'mutual_info': {'n_neighbors': 20, 'thresh': 0.05},
    }

    df_y_pfs = pd.read_csv(
        './../../data/to_analysis/target_pfs.csv', index_col=0
    )
    df_y_lrc = pd.read_csv(
        './../../data/to_analysis/target_lrc.csv', index_col=0
    )
    y_pfs, y_lrc = np.squeeze(df_y_pfs.values), np.squeeze(df_y_lrc.values)

    estimators = {
        # NB: Reports colinear variables.
        'lda': LinearDiscriminantAnalysis,
        'logreg': LogisticRegression,
        # NOTE: Skip for now
        #'svc': SVC,
        'gnb': GaussianNB,
        # NB: warnings.warn('Y residual constant at iteration %s' % k)
        'pls': PLSRegression,
        'qda': QuadraticDiscriminantAnalysis,
    }

    hparams = {
        'lda': {
            # NOTE: n_components determined in model selection
            'n_components': [None], 'tol': TOL, 'priors': [PFS_PRIORS],
        },
        'qda': {
            'priors': [PFS_PRIORS], 'tol': TOL
        },
        'pls': {
            'n_components': [None], 'tol': TOL,
        },
        'gnb': {
            'priors': [PFS_PRIORS]
        },
        'logreg': {
            'C': C, 'solver': ['sag'], 'penalty': PENALTY,
            'class_weight': CLASS_WEIGHT, 'max_iter': MAX_ITER
        },
        'svc': {
            'kernel': ['rbf'], 'C': C,
            'gamma': [0.001, 0.01, 0.1, 0.5, 1],
            'cache_size': [20, 100, 300, 500], 'degree': [2, 3],
            'class_weight': CLASS_WEIGHT
        },
    }

    comparison_scheme = nested_point632plus
    df_X = pd.read_csv(
        './../../data/to_analysis/lbp/ct4_pet0_clinical.csv', index_col=0
    )
    #X = pd.read_csv(
    #    './../../data/to_analysis/lbp/ct4_pet1_clinical.csv', index_col=0
    #)
    X = df_X.values

    path_to_results = './../../data/outputs/model_comparison_pfs/ct4_pet0_clinical.csv'

    start_time = datetime.now()
    print('Started model comparison experiments')
    results = model_comparison(
        comparison_scheme, X, y_pfs, estimators, hparams, selectors,
        selector_params, random_states, N_REPS, path_to_results,
        score_func=SCORE
    )
    print('Ended model comparison experiments in {}'
          ''.format(datetime.now() - start_time))
