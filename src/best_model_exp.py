# -*- coding: utf-8 -*-
#
# best_model_inpsection.py
#

"""
Model retraining and evaluation.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np
import pandas as pd


if __name__ == '__main__':

    import feature_selection

    from sklearn import metrics
    from model_selection import point632plus
    from model_comparison import model_comparison

    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression

    # Setup: number of target features, random seed, number of OOB splits.
    K, SEED, N_REPS = 15, 0, 50

    MAX_ITER = [1200]

    # Repeatability and reproducibility.
    np.random.seed(SEED)

    # Number of repeated experiments.
    n_experiments = 10
    random_states = np.random.randint(1000, size=n_experiments)

    path_to_data = './../../data/to_analysis/lbp/ct3_pet1_clinical.csv'
    path_to_pfstarget = './../../data/to_analysis/target_pfs.csv'
    path_to_lrctarget = './../../data/to_analysis/target_lrc.csv'

    y_pfs = np.squeeze(pd.read_csv(path_to_pfstarget, index_col=0).values)
    y_lrc = np.squeeze(pd.read_csv(path_to_lrctarget, index_col=0).values)

    X = np.array(pd.read_csv(path_to_data, index_col=0).values, dtype=float)

    # Feature selection setup.
    selectors = {'var_thresh': feature_selection.variance_threshold}
    selector_params = {'var_thresh': {'alpha': 0.05}}

    # Classification setup.
    estimators = {'logreg': LogisticRegression, 'pls': PLSRegression}
    hparams = {
        'pls': {
            'max_iter': MAX_ITER,
            'tol': [0.001, 0.01, 0.1, 1, 10, 100],
            'n_components': [None],
        },
        'logreg': {
            'max_iter': MAX_ITER,
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['sag'], 'penalty': ['l2'], 'class_weight': ['balanced'],
        },
    }

    #score_metrics = [
    #    metric.roc_auc_score,
    #]
    #for score_metric in score_metrics:

    # Location to store results.
    path_to_lrcresults = './../../data/outputs/best_model_inspection/lrc_ct3_pet1.csv'
    path_to_pfsresults = './../../data/outputs/best_model_inspection/pfs_ct3_pet1.csv'

    results_pfs = model_comparison(
        point632plus, X, y_pfs, estimators, hparams, selectors,
        selector_params, random_states, N_REPS, path_to_pfsresults,
        score_func=metrics.roc_auc_score
    )
