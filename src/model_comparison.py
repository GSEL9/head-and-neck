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
        comparison_scheme, X, y, estimators, estimator_params, selectors,
        fs_params, random_states, n_splits
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
    # Remove temporary directory if process completed succesfully.
    ioutil.teardown_tempdir(TMP_RESULTS_DIR)

    return results


if __name__ == '__main__':
    import numpy as np
    import pandas as pd



    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import matthews_corrcoef

    from sklearn.svm import SVC, LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


    # NOTE:
    # * Experiment with catboost/adaboost/gradient boost after ID best base
    #   estimator.

    # Setup:
    K, CV, SEED = 20, 4, 0

    # Number of experiments.
    n_experiments = 20

    np.random.seed(SEED)
    random_states = np.random.randint(1000, size=n_experiments)

    """
    20 CPUs:
        * LDA
        * LogReg
        * SVC

    8 CPUs:
        * GNB
        * PLS

    Filter pri:
        * Exponential
        * See Alise...
    """

    estimators = {
        # NB: Reports colinear variables.
        #'lda': LinearDiscriminantAnalysis,
        #'logreg': LogisticRegression,
        # NB: warnings.warn('Y residual constant at iteration %s' % k)
        #'pls': PLSRegression,
        #'adaboost': AdaBoostClassifier,
        #'gnb': GaussianNB,
        'svc': SVC,
    }
    # Priors summing to 1.0.
    PFS_PRIORS = [0.677, 0.323]
    LRC_PRIORS = [0.753, 0.247]

    MAX_ITER = [800]

    N_ESTIMATORS = [20, 50, 100, 200, 500, 1000]
    LEARNINGR_RATE = [0.001, 0.05, 0.2, 0.6, 1, 3]

    TOL = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.7, 1]
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    SCORE = roc_auc_score

    PENALTY = ['l1', 'l2']
    CLASS_WEIGHT = ['balanced']

    logreg_l1 = LogisticRegression(
        penalty='l1', class_weight='balanced', random_state=SEED,
        solver='liblinear'
    )
    logreg_l2 = LogisticRegression(
        penalty='l2', class_weight='balanced', random_state=SEED,
        solver='liblinear'
    )
    rf_model = RandomForestClassifier(
        n_estimators=50, class_weight='balanced', random_state=SEED
    )

    hparams = {
        'lda': {
            # NOTE: n_components determined in model selection
            "n_components": [None], 'tol': TOL, 'priors': [PFS_PRIORS],
        },
        'logreg': {
            'C': C, 'solver': ['liblinear'], 'penalty': PENALTY,
            'class_weight': CLASS_WEIGHT, 'max_iter': MAX_ITER
        },
        'pls': {
            # NOTE: n_components determined in model selection
            "n_components": [None], 'tol': TOL,
        },
        'adaboost': {
            'base_estimator': [logreg_l2],
            'learning_rate': LEARNINGR_RATE, 'n_estimators': N_ESTIMATORS,
        },
        'svc': {
            'kernel': ['rbf'], 'C': C,
            'gamma': [0.0001, 0.001, 0.01, 0.1, 0.3, 0.7, 1],
            'cache_size': [20, 100, 300, 500], 'degree': [2, 3],
            'class_weight': CLASS_WEIGHT
        },
        'lin_svc': {
            'C': C, 'class_weight': CLASS_WEIGHT, 'penalty': PENALTY,
            'dual': [False], 'tol': TOL,
        },
        'gnb': {'priors': [PFS_PRIORS]},
    }

    selectors = {
        'ff_logregl1': feature_selection.forward_floating,
        #'ff_logregl2': feature_selection.forward_floating,
        #'rf_permut_imp': feature_selection.permutation_importance,

        #'var_thresh': feature_selection.variance_threshold,
        #'relieff': feature_selection.relieff,
        #'mutual_info': feature_selection.mutual_info,
    }

    selector_params = {
        # Wrapper methods:
        'ff_logregl1': {'model': logreg_l1, 'k': K, 'cv': 2, 'scoring': SCORE},
        'ff_logregl2': {'model': logreg_l2, 'k': K, 'cv': 2, 'scoring': SCORE},
        'rf_permut_imp': {'model': rf_model, 'thresh': 0.0, 'nreps': 1},

        # Filter methods:
        'var_thresh': {'alpha': 0.05},
        'relieff': {'k': K, 'n_neighbors': 20},
        'mutual_info': {'n_neighbors': 20, 'thresh': 0.05},
    }

    selection_scheme = model_selection.nested_cross_val

    ref_feature_dir = './../../data/to_analysis'
    ref_results_pfs_dir = './../../data/outputs/model_comparison_pfs'
    ref_results_lrc_dir = './../../data/outputs/model_comparison_lrc'

    df_y_pfs = pd.read_csv(
        './../../data/to_analysis/target_pfs.csv', index_col=0
    )
    df_y_lrc = pd.read_csv(
        './../../data/to_analysis/target_lrc.csv', index_col=0
    )
    y_pfs, y_lrc = np.squeeze(df_y_pfs.values), np.squeeze(df_y_lrc.values)


    dirnames = utils.listdir(ref_feature_dir)

    # TODO: Benchmark big O for time budget estimation.
    # * SVC classifier with L1 logreg
    for dirname in dirnames[:1]:

        file_paths = ioutil.relative_paths(
            os.path.join(ref_feature_dir, dirname), target_format='.csv'
        )
        for path_to_file in file_paths[:1]:


            X = pd.read_csv(path_to_file, index_col=0).values

            # path_pfs_results = TODO: where to save results

            # NOTE: PFS
            pfs_results = model_comparison(
                selection_scheme, X, y_pfs, estimators, hparams, selectors,
                selector_params, random_states, CV, score_func=SCORE
            )
            # Write results for each analyzed data set of current filter and
            # discr combo.
            ioutil.write_final_results(path_pfs_results, pfs_results)


            # path_lrc_results = TODO: where to save results

            """
            NB: Use different priors ([]).

            # NOTE: LRC
            lrc_results = model_comparison(
                selection_scheme, X, y_lrc, estimators, hparams, selectors,
                selector_params, random_states, CV, score_func=SCORE_FUNC
            )
            # Write results for each analyzed data set of current filter and
            # discr combo.
            ioutil.write_final_results(path_lrc_results, lrc_results)
            """
