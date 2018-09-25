import os
import utils
import shutil
import logging
import operator

import model_selection
import feature_selection

import numpy as np
import pandas as pd

from multiprocessing import cpu_count

from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


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
            verbose=verbose, score_func=score_func
        )
        train_score, test_score = utils.scale_fit_predict(
            best_model, X_train[:, sel_features], X_test[:, sel_features],
            y_train, y_test, random_state, score_func=score_func
        )
        train_scores.append(train_score), test_scores.append(test_score)
        best_features.append(sel_features)

    return {
        'experiment_id': random_state,
        'best_params': best_model.get_params(),
        'avg_test_score': np.mean(test_scores),
        'avg_train_score': np.mean(train_scores),
        'best_features': utils.multi_intersect(best_features)
    }


def grid_search(*args, score_func=None, n_jobs=1, verbose=0, **kwargs):
    estimator, hparam_grid, X, y, n_splits, random_state = args

    best_score, best_model, best_features = 0.0, None, None
    train_scores, test_scores, sel_features = [], [], []
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

            X_train_sub, X_test_sub, support = feature_selection.dummy(
                X_train, X_test, y_train
            )
            train_score, test_score = utils.scale_fit_predict(
                model, X_train_sub, X_test_sub, y_train, y_test,
                random_state, score_func=score_func
            )
            train_scores.append(train_score), test_scores.append(test_score)
            sel_features.append(support)

        if np.mean(test_scores) > best_score:
            best_score = np.mean(test_scores)
            best_model, best_features = model, sel_features

    return best_model, utils.multi_intersect(sel_features)
