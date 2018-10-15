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
    from skrebate import ReliefF

    # NB ERROR:
    # For imbalanced datasets, the Average Precision metric is sometimes a
    # better alternative to the AUROC. The AP score is the area under the precision-recall curve.
    # https://stats.stackexchange.com/questions/222558/classification-evaluation-metrics-for-highly-imbalanced-data
    # REF: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4403252/
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import cohen_kappa_score

    df_X = pd.read_csv(
        './../../data/to_analysis/lbp/ct0_pet4_clinical.csv', index_col=0
    )
    df_y = pd.read_csv('./../../data/to_analysis/target.csv', index_col=0)
    X = df_X.values
    y = np.squeeze(df_y.values)

    # TODO: Time one

    n_splits = 5
    random_states = np.arange(50)

    estimators = {
        'logreg_l1': LogisticRegression,
        #'logreg_l2': LogisticRegression,
        #'rf': RandomForestClassifier,
        #'knn': KNeighborsClassifier,
        #'adaboost': AdaBoostClassifier,
        #'dtree': DecisionTreeClassifier,
        #'gaussianb': GaussianNB,
        #'svc': SVC,
        #'linsvc': LinearSVC,
        #'mlp': MLPClassifier,

        # TypeError: 1D weights expected when shapes of a and weights differ.
        # Error is linked to numpy.
        #'lda': LinearDiscriminantAnalysis,

        # NB: QDA reports colinear variables. Multicollinearity means that your
        # predictors are correlated. Why is this bad? Because LDA, like
        # regression techniques involves computing a matrix inversion, which is
        # inaccurate if the determinant is close to 0 (i.e. two or more
        # variables are almost a linear combination of each other
        #'qda': QuadraticDiscriminantAnalysis,

        # WARNING: warnings.warn('Y residual constant at iteration %s' % k)
        #'pls': PLSRegression,

        # ERROR: Wrong number of columns in X. Reshape your data.
        #'mars': Earth,
    }
    hparams = {
        'logreg': {
            'C': [0.0001, 0.001,0.005, 0.01,0.05, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'fit_intercept': [True], 'solver': ['liblinear'],
            'penalty': ['l1', 'l2'], 'class_weight': ['balanced'],
        },
        'rf': {
            'n_estimators': [5, 10, 20, 100, 500],
            'max_depth': [10, 50, 100, 500, 1000],
        },
        'knn': {
            'leaf_size': [10, 30, 50],
            'n_neighbors': [2, 5, 10, 15, 20]
        },
        'adaboost': {
            'learning_rate': [0.05, 0.5, 1, 2, 3],
            'n_estimators': [100, 500, 1000]
        },
        'dtree': {
            'max_depth': [10, 50, 100, 500, 1000],
            'class_weight': ['balanced']
        },
        'gaussianb': {},
        'lda': {
            'solver' : ['lsqr'], 'shrinkage': [0.1, 0.5, 0.8],
            'priors': [0.776,0.224], 'n_components': [5, 10, 20, 30],
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
            'tol': [0.0001, 0.00001, 0.001, 0.01], 'max_iter': [300]
        },
        'mars': {
            'penalty' : [0.01, 0.05, 0.1, 0.5], 'minspan_alpha' : [1, 3, 5, 10]
        },
        'pls': {
            'n_components': [1, 10, 100, 500, 1000], 'tol' : [1e-6, 1e-5, 1e-3],
            'scale': [False]
        },
        'svc': {
            'class_weight': ['balanced'], 'gamma': [0.001, 0.05, 0.1, 0.5],
            'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        },
        'linsvc': {
            'dual': [False], 'class_weight': ['balanced'],
            'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'tol': [0.00001, 0.0001, 0.001, 0.1, 1]
        },
    }
    selectors = {
        #'dummy': feature_selection.dummy,
        'corr_thresh': feature_selection.correlation_threshold,
        'var_thresh': feature_selection.variance_threshold,
        'logregl1': feature_selection.forward_floating,
        'relieff': feature_selection.relieff,
        'sff': feature_selection.forward_floating,
        # ERROR ANOVAF: Reports constant features.
        #'anovaf': feature_selection.anova_fvalue,
        'mutual_info': feature_selection.mutual_info,
        'rf_permut_imp': feature_selection.permutation_importance
    }
    selector_params = {
        'dummy': {},
        'sff': {
            'model': RandomForestClassifier(
                n_estimators=50, random_state=0, class_weight='balanced'
            ),
            'k': 10, 'cv': 2, 'scoring': 'roc_auc'
        },
        'relieff': {'k': 10, 'n_neighbors': 5},
        'var_thresh': {'alpha': 0.05},
        'anovaf': {'alpha': 0.05},
        'mutual_info': {'n_neighbors': 5, 'thresh': 0.05},
        'logregl1': {
            'model': LogisticRegression(penalty='l1', class_weight='balanced'),
            'k': 10, 'cv': 5, 'scoring': 'accuracy'
        },
        'rf_permut_imp': {
            'model': RandomForestClassifier(n_estimators=50, random_state=0),
            'thresh': 0.05, 'nreps': 2
        }
    }
    selection_scheme = model_selection.nested_cross_val
    #"""

    # TODO: Time single run and multiply with 19 filters * 25 feature sets to
    # obtain total run time estimate.
    start_time = datetime.now()
    results = model_comparison(
        selection_scheme, X, y, estimators, hparams, selectors,
        selector_params, random_states, n_splits, score_func=roc_auc_score
    )
    print('Execution time: {}'.format(datetime.now() - start_time))

    ioutil.write_final_results(
        './../../data/outputs/model_comparison/ct0_pet0_clinical.csv',
        results
    )
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
