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

from ReliefF import ReliefF
from multiprocessing import cpu_count
from sklearn import feature_selection

from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier

from mlxtend.evaluate import feature_importance_permutation
from mlxtend.feature_selection import SequentialFeatureSelector


def variance_threshold(data, alpha=0.05):
    """A wrapper of scikit-learn VarianceThreshold."""

    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = feature_selection.VarianceThreshold(threshold=alpha)
    selector.fit(X_train_std, y_train)
    support = selector.get_support(indices=True)

    return X_train_std[:, support], X_test_std[:, support], support


def anova_fvalue(data, alpha=0.05):
    """A wrapper of scikit-learn ANOVA F-value."""

    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    _, pvalues = feature_selection.f_classif(X_train_std, y_train)
    support = np.squeeze(np.where(pvalues <= alpha))

    return X_train_std[:, support], X_test_std[:, support], support


def relieff(data, n_neighbors=100, k=10):
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

    return X_train_std[:, support], X_test_std[:, support], support


def permutation_importance(data, model=None, thresh=0, nreps=100):
    """"""

    _metric = 'accuracy'

    X_train, X_test, y_train, y_test = data

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    model.fit(X_train_std, y_train)

    imp, _ = feature_importance_permutation(
        predict_method=model.predict, X=X_test_std, y=y_test,
        metric=_metric,
        num_rounds=nreps, seed=0
    )
    # NOTE: Retain features contributing above threshold to model performance.
    support = np.squeeze(np.argwhere(imp > thresh))

    return X_train_std[:, support], X_test_std[:, support], support


if __name__ == '__main__':

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    cancer = load_breast_cancer()

    # NB: roc_auc_score requires binary <int> target values.
    y = cancer.target
    X = cancer.data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf_clf = RandomForestClassifier(random_state=0)
    X_train_sub, X_test_sub, support = permutation_importance(
        (X_train, X_test, y_train, y_test), model=rf_clf,
    )
    print(support)
