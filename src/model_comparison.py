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

    n_splits = 4
    random_states = np.arange(5)


    # TODO:
    # * Run PLSR, LogReg, LDA and AdaBoost across all discr and filter combos
    #   with Relieff, LogReg1 and RF permutation importance.
    estimators = {
        'logreg': LogisticRegression,
        #'rf': RandomForestClassifier,
        #'knn': KNeighborsClassifier,
        'adaboost': AdaBoostClassifier,
        #'dtree': DecisionTreeClassifier,
        #'gaussianb': GaussianNB,
        #'svc': SVC,
        #'linsvc': LinearSVC,
        #'mlp': MLPClassifier,

        # NB: Reports colinear variables.
        'lda': LinearDiscriminantAnalysis,
        # NB: Reports colinear variables.
        #'qda': QuadraticDiscriminantAnalysis,
        # NB: warnings.warn('Y residual constant at iteration %s' % k)
        'pls': PLSRegression,

        # ERROR: Wrong number of columns in X. Reshape your data.
        #'mars': Earth,
    }
    hparams = {
        'logreg': {
            'C': [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'], 'class_weight': ['balanced'],
        },
        'rf': {
            'n_estimators': [5, 50, 100, 150],
            'max_depth': [10, 50, 100, 500, None],
        },
        'knn': {
            'leaf_size': [10, 20, 30, 40],
            'n_neighbors': [2, 5, 10, 15, 20]
        },
        'adaboost': {
            'base_estimator': [LogisticRegression(class_weight='balanced')],
            'learning_rate': [0.05, 0.5, 1],
            'n_estimators': [5, 50, 100, 500, 1000],
        },
        'dtree': {
            'max_depth': [10, 50, 100, 500, None],
            'class_weight': ['balanced']
        },
        'lda': {
            'n_components': [1, 10, 50, 100],
            'tol': [0.0001, 0.00001, 0.001, 0.01]
        },
        'qda': {
            'tol': [0.0001, 0.00001, 0.001, 0.01],
            'reg_param': [0.1, 0.5, 0.7, 0.9],
            'priors': [0.776,0.224]
        },
        'mlp': {
            'hidden_layer_sizes': [5, 10, 50, 100, 150, 200],
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'tol': [0.0001, 0.00001, 0.001, 0.01],
            'max_iter': [300]
        },
        'mars': {
            'penalty' : [0.01, 0.05, 0.1, 0.5],
            'minspan_alpha' : [1, 3, 5, 10]
        },
        'pls': {
            'n_components': [1, 10, 50, 100],
            'tol': [0.0001, 0.00001, 0.001, 0.01],
            'scale': [False]
        },
        'svc': {
            'class_weight': ['balanced'],
            'gamma': [0.001, 0.05, 0.1, 0.5],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        },
        'linsvc': {
            'dual': [False],
            'class_weight': ['balanced'],
            'C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'tol': [0.00001, 0.0001, 0.001, 0.1, 1]
        },
    }

    # NOTE:
    # * Variance threshold may produce colinear features.
    # * Random forest may fit to noise without any FS.
    # * If model performing worse than average: probably adjusts to noise.
    # * How about combining AdaBoostClassifier with a GridSearchCV of
    #   LogisticRegression in a meta classifier?

    selectors = {
        #'dummy': feature_selection.dummy,
        #'var_thresh': feature_selection.variance_threshold,
        #'ff_logregl1': feature_selection.forward_floating,
        #'ff_logregl2': feature_selection.forward_floating,
        'rf_permut_imp': feature_selection.permutation_importance,
        #'relieff': feature_selection.relieff,
        #'ff_rf': feature_selection.forward_floating,
        # ERROR ANOVAF: Reports constant features.
        #'anovaf': feature_selection.anova_fvalue,
        #'mutual_info': feature_selection.mutual_info,
    }
    selector_params = {
        'var_thresh': {'alpha': 0.05},
        'logregl1': {
            'model': LogisticRegression(penalty='l1', class_weight='balanced'),
            'k': 10, 'cv': 5, 'scoring': 'roc_auc'
        },

        'ff_rf': {
            'model': RandomForestClassifier(
                n_estimators=50, random_state=0, class_weight='balanced'
            ),
            'k': 10, 'cv': 2, 'scoring': 'roc_auc'
        },
        'relieff': {'k': 30, 'n_neighbors': 5},
        'anovaf': {'alpha': 0.05},
        'mutual_info': {'n_neighbors': 5, 'thresh': 0.05},

        'logregl2': {
            'model': LogisticRegression(penalty='l2', class_weight='balanced'),
            'k': 10, 'cv': 5, 'scoring': 'roc_auc'
        },
        'rf_permut_imp': {
            'model': RandomForestClassifier(n_estimators=50, random_state=0),
            'thresh': 1e-5, 'nreps': 5
        },
        'dummy': {},
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

    ioutil.write_final_results(
        './../../data/outputs/model_comparison/lbp/ct0_pet0_clinical.csv',
        results
    )
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

            results = model_comparison(
                selection_scheme, X, y, estimators, hparams, selectors,
                selector_params, random_states, n_splits,
                score_func=roc_auc_score
            )
            path_results = os.path.join(
                ref_results_dir, filter_cat, os.path.basename(path_feature_set)
            )
            ioutil.write_final_results(path_results, results)
    """
