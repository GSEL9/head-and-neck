# -*- coding: utf-8 -*-
#
# feature_selection.py
#

"""
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import utils

import numpy as np
import pandas as pd

from ReliefF import ReliefF
from multiprocessing import cpu_count
from sklearn import feature_selection

from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

from mlxtend.evaluate import feature_importance_permutation
from mlxtend.feature_selection import SequentialFeatureSelector


SEED = 0
METRIC = 'accuracy'


class FeatureVotings:

    def __init__(self, nfeatures=None, thresh=1):

        self.nfeatures = nfeatures
        self.thresh = thresh

        # NOTE:
        self.feature_votes = None
        self.selected_supports = None

    @property
    def consensus_votes(self):
        """Retains only the feaures selected in each round."""

        support_matches = utils.multi_intersect(self.selected_supports)

        # At least one feature was selected commonly in all sessions.
        if np.size(support_matches) > 0:
            return support_matches
        else:
            raise RuntimeError('No feature consensus. Tip: Adjust threshold.')

    @property
    def major_votes(self):
        """Retains only the k highest voted feaures."""

        support_matches = []
        for feature, nvotes in self.feature_votes.items():

            if nvotes >= self.thresh:
                support_matches.append(feature)

        if np.size(support_matches) > 0:
            return support_matches
        else:
            raise RuntimeError('No feature consensus')


    def update_votes(self, support):
        """Update votes per feature in derived support."""

        # NOTE: Updated at instantiation only.
        if self.feature_votes is None:
            self.feature_votes = {num: 0 for num in range(self.nfeatures)}

        if self.selected_supports is None:
            self.selected_supports = []

        for feature_num in support:
            self.feature_votes[feature_num] += 1

        self.selected_supports.append(support)

        return self


class CorrelationThreshold:

    def __init__(self, threshold=0.0):

        self.threshold = threshold

        # NOTE:
        self._data = None
        self._support = None

    def fit(self, X, y=None, **kwargs):

        if isinstance(X, pd.DataFrame):
            self._data = X
        else:
            self._data = pd.DataFrame(X, columns=np.arange(X.shape[1]))

        # Create correlation matrix.
        corr_mat = self._data.corr().abs()

        # Select upper triangle of correlation matrix.
        upper = corr_mat.where(
            np.triu(np.ones(np.shape(corr_mat)), k=1).astype(np.bool)
        )
        # Find index of feature columns with correlation > thresh.
        corr_cols = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        self._data.drop(self._data.columns[corr_cols], axis=1, inplace=True)

        return self

    def get_support(self, **kwargs):

        return self._data.columns


def dummy(data, **kwargs):

    X_train, X_test, y_train, y_test = data

    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    support = np.arange(X_train.shape[1])

    return X_train_std[:, support], X_test_std[:, support], support


def correlation_threshold(data, alpha=0.05):

    X_train, X_test, y_train, y_test = data

    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = CorrelationThreshold(threshold=alpha)

    # NOTE: Cannot filter variance from standardized data.
    selector.fit(X_train, y_train)
    support = selector.get_support(indices=True)

    # NB: Default mechanism is to include all features is none were selected.
    if np.size(support) < 1:
        support = np.arange(X_train.shape[1], dtype=int)

    return X_train_std[:, support], X_test_std[:, support], support


def variance_threshold(data, alpha=0.05):
    """A wrapper of scikit-learn VarianceThreshold."""

    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = feature_selection.VarianceThreshold(threshold=alpha)
    # NOTE: Cannot filter variance from standardized data.
    selector.fit(X_train, y_train)
    support = selector.get_support(indices=True)

    # NB: Default mechanism is to include all features is none were selected.
    if np.size(support) < 1:
        support = np.arange(X_train.shape[1], dtype=int)

    return X_train_std[:, support], X_test_std[:, support], support


def anova_fvalue(data, alpha=0.05):
    """A wrapper of scikit-learn ANOVA F-value."""

    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    _, pvalues = feature_selection.f_classif(X_train, y_train)
    support = np.squeeze(np.where(pvalues <= alpha))

    # NB: Default mechanism is to include all features is none were selected.
    if np.size(support) < 1:
        support = np.arange(X_train.shape[1], dtype=int)

    return X_train_std[:, support], X_test_std[:, support], support


def mutual_info(data, n_neighbors=3, thresh=0.05):
    """A wrapper of scikit-learn mutual information feature selector."""

    X_train, X_test, y_train, y_test = data

    global SEED

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    mut_info = feature_selection.mutual_info_classif(
        X_train_std, y_train, n_neighbors=n_neighbors, random_state=SEED
    )
    # NOTE: Retain features contributing above threshold to model performance.
    support = np.squeeze(np.argwhere(mut_info > thresh))

    # NB: Default mechanism is to include all features is none were selected.
    if np.size(support) < 1:
        support = np.arange(X_train.shape[1], dtype=int)

    return X_train_std[:, support], X_test_std[:, support], support


def relieff(data, n_neighbors=20, k=10):
    """A wrapper of the ReliefF algorithm.

    Args:
        n_neighbors (int): The number of neighbors to consider when assigning
            feature importance scores.

    """
    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = ReliefF(n_neighbors=n_neighbors)
    selector.fit(X_train_std, y_train)

    support = selector.top_features[:k]

    # NB: Default mechanism is to include all features is none were selected.
    if np.size(support) < 1:
        support = np.arange(X_train.shape[1], dtype=int)

    return X_train_std[:, support], X_test_std[:, support], support


def forward_floating(data, scoring=None, model=None, k=3, cv=10):
    """A wrapper of mlxtend Sequential Forward Floating Selection algorithm.

    """
    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    # NOTE: Nested calls not supported by multiprocessing => joblib converts
    # into sequential code (thus, default n_jobs=1).
    #n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()
    n_jobs = 1

    selector = SequentialFeatureSelector(
        model, k_features=k, forward=True, floating=True, scoring=scoring,
        cv=cv, n_jobs=n_jobs
    )
    selector.fit(X_train_std, y_train)

    support = selector.k_feature_idx_

    # NB: Default mechanism is to include all features is none were selected.
    if np.size(support) < 1:
        support = np.arange(X_train.shape[1], dtype=int)

    return X_train_std[:, support], X_test_std[:, support], support


def permutation_importance(data, model=None, thresh=0, nreps=5):
    """A wrapper of mlxtend feature importance permutation algorithm.

    """

    X_train, X_test, y_train, y_test = data

    global METRIC, SEED

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)

    imp, _ = feature_importance_permutation(
        predict_method=model.predict, X=X_test_std, y=y_test,
        metric=METRIC, num_rounds=nreps, seed=SEED
    )
    support = np.squeeze(np.where(imp > thresh))

    # NB: Default mechanism is to include all features is none were selected.
    if np.size(support) < 1:
        support = np.arange(X_train.shape[1], dtype=int)

    return X_train_std[:, support], X_test_std[:, support], support
