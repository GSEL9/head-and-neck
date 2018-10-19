# -*- coding: utf-8 -*-
#
# model_selection.py
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
import operator

import model_selection
import feature_selection

import numpy as np
import pandas as pd

from datetime import datetime
from collections import OrderedDict
from multiprocessing import cpu_count

from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def _update_prelim_results(results, path_tempdir, random_state, *args):
    # Update results <dict> container and write preliminary results to disk.
    (
        estimator, selector, best_params, avg_test_scores, avg_train_scores,
        best_features
    ) = args
    results.update(
        {
            'model': estimator.__name__,
            'selector': selector['name'],
            'best_params': best_params,
            'avg_test_score': avg_test_scores,
            'avg_train_score': avg_train_scores,
            'best_features': best_features,
            'num_features': np.size(best_features)
        }
    )
    # Write preliminary results to disk.
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    ioutil.write_prelim_results(path_case_file, results)

    return results


def nested_cross_val(*args, verbose=1, score_func=None, **kwargs):
    """A nested cross validation scheme comprising (1) an inner cross
    validation loop to tune hyperparameters and select the best model, (2) an
    outer cross validation loop to evaluate the model selected by the inner
    cross validation scheme.

    """
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    results = {'experiment_id': random_state}
    feature_votes = feature_selection.FeatureVotings(
        nfeatures=X.shape[1], thresh=n_splits-1
    )
    # Outer cross-validation loop.
    kfolds = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    train_scores, test_scores, opt_hparams = [], [], []
    for fold_num, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Determine best model and feature subset.
        best_model, best_support = grid_search_cv(
            estimator, hparam_grid, selector, X_train, y_train, n_splits,
            random_state, verbose=verbose, score_func=score_func
        )
        train_score, test_score = utils.scale_fit_predict(
            best_model, X_train[:, best_support], X_test[:, best_support],
            y_train, y_test, score_func=score_func
        )
        # Bookkeeping of best feature subset in each fold.
        feature_votes.update_votes(best_support)
        opt_hparams.append(best_model.get_params())
        train_scores.append(train_score), test_scores.append(test_score)

    # NOTE: Obtaining a different set of hparams for each fold. Selecting mode
    # of hparams as opt hparam settings.
    mode_hparams = max(opt_hparams, key=opt_hparams.count)

    end_results = _update_prelim_results(
        results, path_tempdir, random_state, estimator, selector, mode_hparams,
        np.mean(test_scores), np.mean(train_scores), feature_votes.major_votes
    )
    return end_results


def grid_search_cv(*args, score_func=None, n_jobs=1, verbose=0, **kwargs):
    """Exhaustive search in estimator hyperparameter space to derive optimal
    combination with respect to scoring function.

    """
    estimator, hparam_grid, selector, X, y, n_splits, random_state = args

    best_test_score, best_model, best_support = 0, [], []
    for combo_num, hparams in enumerate(hparam_grid):

        # Setup:
        try:
            model = estimator(**hparams, random_state=random_state)
        except:
            model = estimator(**hparams)
        # NOTE: Use thresh = n_splits in case all features are selected in each
        # round by default mechanism.
        feature_votes = feature_selection.FeatureVotings(
            nfeatures=X.shape[1], thresh=n_splits-1
        )
        # Inner cross-validation loop.
        kfolds = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
        # Determine average model performance for each hparam combo.
        train_scores, test_scores = [], []
        for num, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # NOTE: Standardizing in feature sel function.
            start_time = datetime.now()
            X_train_sub, X_test_sub, support = selector['func'](
                (X_train, X_test, y_train, y_test), **selector['params']
            )
            #print('Feature selection completed in: {}'.format(datetime.now() - start_time))
            train_score, test_score = utils.scale_fit_predict(
                model, X_train_sub, X_test_sub, y_train, y_test,
                score_func=score_func
            )
            # Bookkeeping of features selected in each fold.
            feature_votes.update_votes(support)
            train_scores.append(train_score), test_scores.append(test_score)

        if np.mean(test_scores) > best_test_score:
            best_test_score = np.mean(test_scores)
            best_support = feature_votes.major_votes #consensus_votes
            try:
                best_model = estimator(**hparams, random_state=random_state)
            except:
                best_model = estimator(**hparams)

    return best_model, best_support



def bootstrap_point632plus(*args, verbose=1, score_func=None, **kwargs):
    """A out-of-bag bootstrap scheme to select optimal classifier based on
    .632+ estimator.

    """
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    results = {'experiment_id': random_state}

    boot = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    avg_test_error, avg_train_error, best_model, best_support = 1, 1, [], []
    for combo_num, hparams in enumerate(hparam_grid):

        test_errors, train_errors, support = _boot_validation(
            estimator, hparams, selector, boot, X, y, random_state,
            score_func=score_func, n_jobs=1, verbose=0, **kwargs
        )
        # Determine the optimal hparam combo.
        if np.mean(test_errors) < avg_test_error:
            avg_test_error = np.mean(test_errors)
            avg_train_error = np.mean(train_errors)
            best_model = estimator(**hparams, random_state=random_state)
            best_support = support

    end_results = _update_prelim_results(
        results, path_tempdir, random_state, estimator, selector,
        best_model.get_params(), avg_test_error, avg_train_error, best_support
    )
    return end_results


def _boot_validation(*args, score_func=None, n_jobs=1, verbose=0, **kwargs):

    estimator, hparams, selector, boot, X, y, random_state = args

    # Setup:
    model = estimator(**hparams, random_state=random_state)
    feature_votes = feature_selection.FeatureVotings(nfeatures=X.shape[1])

    train_errors, test_errors = [], []
    for split_num, (train_idx, test_idx) in enumerate(boot.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # NOTE: Standardizing in feature sel function.
        X_train_sub, X_test_sub, support = selector['func'](
            (X_train, X_test, y_train, y_test), **selector['params']
        )
        model.fit(X_train_sub, y_train)
        # Aggregate model predictions.
        y_test_pred = model.predict(X_test_sub)
        y_train_pred = model.predict(X_train_sub)
        test_score = score_func(y_test, y_test_pred)
        train_score = score_func(y_train, y_train_pred)

        # Compute train and test errors.
        train_errors.append(
            utils.point632p_score(
                y_train, y_train_pred, train_score, test_score
            )
        )
        test_errors.append(
            utils.point632p_score(
                y_test, y_test_pred, train_score, test_score
            )
        )
        # Bookkeeping of features selected in each fold.
        feature_votes.update_votes(support)

    return train_errors, test_errors, feature_votes.consensus_votes
