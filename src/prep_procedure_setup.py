# -*- coding: utf-8 -*-
#
# model_comparison_experiments.py
#

"""
Setup model comparison experiments.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np
import pandas as pd


def target(path_to_target, index_col=0):

    var = pd.read_csv(path_to_target, index_col=index_col)
    return np.squeeze(var.values)


def feature_set(path_to_data, index_col=0):

    data = pd.read_csv(path_to_data, index_col=index_col)
    return data.values


if __name__ == '__main__':

    import os
    import feature_selection

    from datetime import datetime
    from model_comparison import model_comparison
    from model_selection import nested_point632plus

    from sklearn.metrics import roc_auc_score

    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # Setup: number of target features, random seed, number of OOB splits.
    K, SEED, N_REPS = 10, 0, 50

    # Shared hyperparameters:
    MAX_ITER = [800]
    PENALTY = ['l2']
    CLASS_WEIGHT = ['balanced']
    # Priors for both target variables summing to 1.0.
    PFS_PRIORS = [0.677, 0.323]
    LRC_PRIORS = [0.753, 0.247]
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    TOL = [0.001, 0.01, 0.1, 0.3, 0.7, 1]
    N_ESTIMATORS = [20, 50, 100, 500, 1000]
    LEARNINGR_RATE = [0.001, 0.05, 0.2, 0.6, 1]

    # Loss function.
    SCORE = roc_auc_score

    # Repeatability and reproducibility.
    np.random.seed(SEED)

    # Number of experiments.
    n_experiments = 10
    random_states = np.random.randint(1000, size=n_experiments)

    # Wrapped algorithms in feature selection.
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
    # Feature selection algorithms.
    selectors = {
        # Wrapper methods:
        'logregl1_permut_imp': feature_selection.permutation_importance,
        'rf_permut_imp': feature_selection.permutation_importance,
        # Filter methods:
        'var_thresh': feature_selection.variance_threshold,
        'relieff': feature_selection.relieff,
        'mutual_info': feature_selection.mutual_info
    }
    # Feature selection parameters.
    selector_params = {
        'logregl1_permut_imp': {
            'model': logreg_l1, 'thresh': 0.0, 'nreps': 1
            },
        'rf_permut_imp': {'model': rf_model, 'thresh': 0.0, 'nreps': 1},
        'mutual_info': {'n_neighbors': 20, 'thresh': 0.05},
        'relieff': {'k': K, 'n_neighbors': 20},
        'var_thresh': {'alpha': 0.05},
    }
    # Classification algorithms.
    estimators = {
        'lda': LinearDiscriminantAnalysis,
        'logreg': LogisticRegression,
        'gnb': GaussianNB,
        'pls': PLSRegression,
        'qda': QuadraticDiscriminantAnalysis,
    }
    # Hyperparameters classification algorithms.
    hparams = {
        'lda': {
            # NOTE: n_components determined in model selection work function.
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
    }
    # Set comparison procedure.
    comparison_scheme = nested_point632plus

    BASE_PATH = './../../data'
    TARGET_SET = 'ct4_pet0_clinical.csv'

    # Target variables.
    y_pfs = target('./../../data/to_analysis/target_pfs.csv')
    y_lrc = target('./../../data/to_analysis/target_lrc.csv')

    # Data set.
    X = feature_set('./../../data/to_analysis/lbp/ct4_pet0_clinical.csv')
    #X = feature_set('./../../data/to_analysis/lbp/ct4_pet1_clinical.csv')

    # Location to store results.
    path_to_results = os.path.join(
        BASE_PATH, 'outputs/model_comparison_pfs', TARGET_SET
    )
    
