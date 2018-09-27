import os
import utils
import ioutil
import shutil
import logging
import operator

import model_selection
import feature_selection

import numpy as np
import pandas as pd

from collections import OrderedDict
from multiprocessing import cpu_count

from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


PATH_TMP_RESULTS = ''


def nested_cross_val(*args, verbose=1, score_func=None, **kwargs):
    """A nested cross validation scheme comprising (1) an inner cross
    validation loop to tune hyperparameters and select the best model, (2) an
    outer cross validation loop to evaluate the model selected by the inner
    cross validation scheme.

    """
    (
        X, y, estimator, selector, hparam_grid, selector, n_splits,
        random_state
    ) = args

    global PATH_TMP_RESULTS

    results = {'experiment_id': random_state}

    # Outer cross-validation loop.
    kfolds = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    train_scores, test_scores, best_features = [], [], []
    for fold_num, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Determine best model and feature subset.
        best_model, sel_features = grid_search_cv(
            estimator, hparam_grid, selector, X_train, y_train, n_splits,
            random_state, verbose=verbose, score_func=score_func
        )
        train_score, test_score = utils.scale_fit_predict(
            best_model, X_train[:, sel_features], X_test[:, sel_features],
            y_train, y_test, random_state, score_func=score_func
        )
        train_scores.append(train_score), test_scores.append(test_score)
        best_features.append(sel_features)

    results.update(
        {
            'feature_selection'
            'best_params': best_model.get_params(),
            'avg_test_score': np.mean(test_scores),
            'avg_train_score': np.mean(train_scores),
            'best_features': utils.multi_intersect(best_features)
        }
    )
    # Write preliminary results to disk.
    # NB: ioutil.write_prelim_result(path_case_file, result)

    return results


def grid_search_cv(*args, score_func=None, n_jobs=1, verbose=0, **kwargs):
    """Exhaustive search in estimator hyperparameter space to derive optimal
    combination with respect to scoring function.

    """
    estimator, hparam_grid, selector, X, y, n_splits, random_state = args

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

            # NOTE: Standardizing in feature sel function.
            X_train_sub, X_test_sub, support = selector(
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


def bootstrap_point632plus(*args, verbose=1, score_func=None, **kwargs):
    """A out-of-bag bootstrap scheme to select optimal classifier based on
    .632+ estimator.

    """
    (
        X, y, estimator, selector, hparam_grid, selector, n_splits,
        random_state
    ) = args
    boot = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    results = {'experiment_id': random_state}

    best_score, best_model, best_features = 0.0, None, None
    for combo_num, hparams in enumerate(hparam_grid):

        # Setup model.
        model = estimator(**hparams, random_state=random_state)

        scores, sel_features = [], []
        for split_num, (train_idx, test_idx) in enumerate(boot.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # NOTE: Standardizing in feature sel function.
            X_train_sub, X_test_sub, support = selector(
                X_train, X_test, y_train
            )
            sel_features.append(support)

            # Aggregate model predictions.
            model.fit(X_train_sub, y_train)
            y_train_pred = model.predict(X_train_sub)
            train_score = 1.0 - score_func(y_train, y_train_pred)
            test_score = 1.0 - score_func(y_test, model.predict(X_test_sub))

            train_error, test_error = 1.0 - train_score, 1.0 - test_score
            # Compute .632+ score.
            weight = utils.omega(
                utils.rel_overfit_rate(
                    train_error, test_error, utils.no_info_rate(
                        y_train, y_train_pred
                    )
                )
            )
            scores.append(
                utils.point632p_score(weight, train_error, test_error)
            )
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_model, best_features = model, sel_features
    # TODO: Add na
    results.update(
        {
            'feature_selection'
            'point632plus_score': best_score,
            'best_params': best_model.get_params(),
            'best_features': utils.multi_intersect(best_features)
        }
    )
    # Write preliminary results to disk.
    #ioutil.write_prelim_result(path_case_file, results)

    return results


if __name__ == '__main__':

    np.random.seed(1)
    y_true = np.random.randint(0, 2, 10)
    y_pred = np.random.randint(0, 2, 10)

    utils.no_info_rate(y_true, y_pred)
