# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""
Frameworks for performing model selection.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import os
import utils
import ioutil
import model_selection
import feature_selection

import numpy as np
import pandas as pd

from datetime import datetime
from collections import OrderedDict
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold


THRESH = 1


def _check_estimator(nfeatures, hparams, estimator, random_state):

    # Using all available features after feature selection.
    if 'n_components' in hparams:
        if nfeatures - 1 < 1:
            hparams['n_components'] = 1
        else:
            hparams['n_components'] = nfeatures - 1
    # If stochastic algorithms.
    try:
        model = estimator(**hparams, random_state=random_state)
    except:
        model = estimator(**hparams)

    try:
        model.n_jobs = -1
    except:
        pass

    return model


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


def nested_cross_val(*args, verbose=1, n_jobs=None, score_func=None, **kwargs):
    """A nested cross validation scheme inclusive feature selection.

    Args:
        X (array-like):
        y (array-like)

    Kwargs:

    Returns:
        (dict):

    """
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        if verbose > 0:
            print('Reloading previous results')

    else:
        if verbose > 0:
            start_time = datetime.now()
        results = _nested_cross_val(
            *args, verbose=verbose, score_func=score_func, n_jobs=n_jobs,
            **kwargs
        )
        if verbose > 0:
            delta_time = datetime.now() - start_time
            print('Collected results in: {}'.format(delta_time))

    return results


def _nested_cross_val(*args, verbose=1, n_jobs=None, score_func=None, **kwargs):
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    global THRESH

    # Setup:
    results = {'experiment_id': random_state}
    features = np.zeros(X.shape[1], dtype=int)

    # Outer cross-validation loop.
    kfolds = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )
    train_scores, test_scores, opt_hparams = [], [], []
    for fold_num, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):

        if verbose > 0:
            print('Outer loop fold {}'.format(fold_num))

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Determine best model and feature subset.
        best_model, best_support = grid_search_cv(
            estimator, hparam_grid, selector, X_train, y_train, n_splits,
            random_state, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        train_score, test_score = utils.scale_fit_predict(
            best_model, X_train[:, best_support], X_test[:, best_support],
            y_train, y_test, score_func=score_func
        )
        # Bookeeping of best feature subset in each fold.
        #feature_votes.update_votes(best_support)
        features[best_support] += 1
        opt_hparams.append(best_model.get_params())
        train_scores.append(train_score), test_scores.append(test_score)

    # NOTE: Obtaining a different set of hparams for each fold. Selecting mode
    # of hparams as opt hparam settings.
    mode_hparams = max(opt_hparams, key=opt_hparams.count)

    end_results = _update_prelim_results(
        results, path_tempdir, random_state, estimator, selector, mode_hparams,
        np.mean(test_scores), np.mean(train_scores),
        np.squeeze(np.where(features >= THRESH))
    )
    return end_results


def grid_search_cv(*args, score_func=None, n_jobs=None, verbose=2, **kwargs):
    """Exhaustive search in estimator hyperparameter space to derive optimal
    combination with respect to scoring function.

    """
    estimator, hparam_grid, selector, X, y, n_splits, random_state = args

    global THRESH

    if verbose > 0:
        start_time = datetime.now()
        print('Entering grid search')

    best_test_score, best_model, best_support = 0, [], []
    for combo_num, hparams in enumerate(hparam_grid):

        if verbose > 1:
            print('Hyperparameter combo {}'.format(combo_num))

        # Setup:
        features = np.zeros(X.shape[1], dtype=int)

        kfolds = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
        train_scores, test_scores = [], []
        for num, (train_idx, test_idx) in enumerate(kfolds.split(X, y)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # NOTE: Standardizing in feature sel function.
            X_train_sub, X_test_sub, support = selector['func'](
                (X_train, X_test, y_train, y_test), **selector['params']
            )
            model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
            train_score, test_score = utils.scale_fit_predict(
                model, X_train_sub, X_test_sub, y_train, y_test,
                score_func=score_func
            )
            # Bookkeeping of features selected in each fold.
            features[support] += 1
            train_scores.append(train_score), test_scores.append(test_score)

        if np.mean(test_scores) > best_test_score:
            best_test_score = np.mean(test_scores)
            best_support = np.squeeze(np.where(features >= THRESH))
            best_model = _check_estimator(
                np.size(support), hparams, estimator, random_state=random_state
            )
    if verbose > 0:
        print('Exiting grid search at {}'.format(datetime.now() - start_time))

    return best_model, best_support


def nested_point632plus(*args, verbose=1, n_jobs=1, score_func=None):
    """

    """
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        if verbose > 0:
            print('Reloading previous results')

    else:
        if verbose > 0:
            start_time = datetime.now()
            print('Entering nested procedure with ID: {}'.format(random_state))
        results = _nested_point632plus(
            *args, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        if verbose > 0:
            delta_time = datetime.now() - start_time
            print('Collected results in: {}'.format(delta_time))

    return results


def _nested_point632plus(*args, n_jobs=None, score_func=None, **kwargs):
    # The worker function for the nested .632+ Out-of-Bag model comparison
    # scheme.
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    global THRESH

    # Setup:
    results = {'experiment_id': random_state}
    # Producing N bootstrap samples.
    outer_sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    sel_features = np.zeros(X.shape[1], dtype=int)
    # Outer loop for best models average performance.
    train_errors, test_errors, opt_hparams = [], [], []
    for num, (train_idx, test_idx) in enumerate(outer_sampler.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        min_test_error = 1
        inner_sampler = utils.BootstrapOutOfBag(
            n_splits=n_splits, random_state=random_state
        )
        # Inner loop for model selection.
        best_hparams, best_support = [], []
        for combo_num, hparams in enumerate(hparam_grid):
            # NOTE: Standardizing in feature sel function.
            test_errors, train_errors, support = oob_resampling(
                estimator, hparams, selector, inner_sampler, X_train, y_train,
                random_state, score_func=score_func, n_jobs=n_jobs,
            )
            # Determine the optimal hparam combo.
            if np.mean(test_errors) < min_test_error:
                min_test_error = np.mean(test_errors)
                best_hparams, best_support = hparams, support

        # Best model.
        best_model = _check_estimator(
            np.size(best_support), best_hparams, estimator,
            random_state=random_state
        )
        train_error, test_error = utils.scale_fit_predict632(
            best_model, X_train[:, best_support], X_test[:, best_support],
            y_train, y_test, score_func=score_func
        )
        train_errors.append(train_error), test_errors.append(test_error)
        # Bookeeping of best feature subset and hparams in each fold.
        sel_features[best_support] += 1
        opt_hparams.append(best_model.get_params())

    # NOTE: Selecting mode of hparams as opt hparam settings.
    best_model_hparams = max(opt_hparams, key=opt_hparams.count)

    end_results = _update_prelim_results(
        results, path_tempdir, random_state, estimator, selector,
        best_model_hparams, np.mean(test_errors), np.mean(train_errors),
        np.squeeze(np.where(sel_features >= THRESH))
    )
    return end_results


def point632plus(*args, verbose=1, n_jobs=1, score_func=None):
    """Perform a hyperparameter grid search with the .632+ bootstrap estimator.

    """
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    path_case_file = os.path.join(
        path_tempdir, '{}_{}_{}'.format(
            estimator.__name__, selector['name'], random_state
        )
    )
    if os.path.isfile(path_case_file):
        results = ioutil.read_prelim_result(path_case_file)
        if verbose > 0:
            print('Reloading previous results')

    else:
        if verbose > 0:
            start_time = datetime.now()
        results = _point632plus(
            *args, verbose=verbose, score_func=score_func, n_jobs=n_jobs
        )
        if verbose > 0:
            delta_time = datetime.now() - start_time
            print('Collected results in: {}'.format(delta_time))

    return results


def _point632plus(*args, verbose=1, score_func=None, n_jobs=1):
    """A out-of-bag bootstrap scheme to select optimal classifier based on
    .632+ estimator.

    """
    (
        X, y, estimator, hparam_grid, selector, n_splits, random_state,
        path_tempdir
    ) = args

    # Setup:
    results = {'experiment_id': random_state}
    # N splits produces N bootstrap samples
    oob_sampler = utils.BootstrapOutOfBag(
        n_splits=n_splits, random_state=random_state
    )
    min_test_error, min_train_error = 1, 1
    best_model, best_support = [], []
    for combo_num, hparams in enumerate(hparam_grid):

        test_errors, train_errors, support = oob_resampling(
            estimator, hparams, selector, oob_sampler, X, y, random_state,
            score_func=score_func, n_jobs=n_jobs, verbose=verbose,
        )
        # Determine the optimal hparam combo.
        if np.mean(test_errors) < min_test_error:
            min_test_error = np.mean(test_errors)
            min_train_error = np.mean(train_errors)
            best_hparams, best_support = hparams, support

    end_results = _update_prelim_results(
        results, path_tempdir, random_state, estimator, selector,
        best_hparams, min_test_error, min_train_error, best_support
    )
    return end_results


def oob_resampling(*args, score_func=None, n_jobs=1, verbose=0):
    # Computes the .632+ score for OOB splits.
    (
        estimator, hparams, selector, oob_sampler, X, y, random_state
    ) = args

    global THRESH

    # Setup:
    features = np.zeros(X.shape[1], dtype=int)

    train_errors, test_errors = [], []
    for split_num, (train_idx, test_idx) in enumerate(oob_sampler.split(X, y)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # NOTE: Standardizing in feature sel function.
        X_train_sub, X_test_sub, support = selector['func'](
            (X_train, X_test, y_train, y_test), **selector['params']
        )
        model = _check_estimator(
            np.size(support), hparams, estimator, random_state=random_state
        )
        train_error, test_error = utils.scale_fit_predict632(
            model, X_train_sub, X_test_sub, y_train,
            y_test, score_func=score_func
        )
        train_errors.append(train_error), test_errors.append(test_error)
        # Bookkeeping of features selected in each fold.
        features[support] += 1

    return train_errors, test_errors, np.squeeze(np.where(features >= THRESH))
