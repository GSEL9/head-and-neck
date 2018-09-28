import utils

import numpy as np

from ReliefF import ReliefF
from multiprocessing import cpu_count
from sklearn import feature_selection

from sklearn.metrics import make_scorer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

from mlxtend.feature_selection import SequentialFeatureSelector


# NOTE: Filter methods.
def variance_threshold(**kwargs):
    """A wrapper of scikit-learn VarianceThreshold."""

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(
        kwargs['X_train'], kwargs['X_test']
    )
    selector = feature_selection.VarianceThreshold(threshold=kwargs['alpha'])
    selector.fit(X_train_std, kwargs['y_train'])
    support = selector.get_support(indices=True)

    return X_train_std[:, support], X_test_std[:, support], support


def chi2(**kwargs):
    """A wrapper of scikit-learn chi2."""

    # NB: Cannot handle negative features.

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(
        kwargs['X_train'], kwargs['X_test']
    )
    _, pvalues = feature_selection.chi2(X_train_std, y_train)
    support = np.where(pvalues <= threshold)

    return X_train_std[:, support], X_test_std[:, support], support


def anova_fvalue(**kwargs):
    """A wrapper of scikit-learn ANOVA F-value."""

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(
        kwargs['X_train'], kwargs['X_test']
    )
    _, pvalues = feature_selection.f_classif(X_train_std, y_train)
    support = np.where(pvalues <= alpha)

    return X_train_std[:, support], X_test_std[:, support], support


# NOTE: Wrapper methods.
def false_positive_rates(X_train, X_test, y_train, scorer=None, alpha=0.05):
    """A wrapper of scikit-learn False Positive Rate test."""

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = feature_selection.SelectFpr(scorer, alpha=alpha)
    selector.fit(X_train_std, y_train)
    support = np.where(selector.pvalues_ <= alpha)

    return X_train_std[:, support], X_test_std[:, support], support


def relieff(X_train, X_test, y_train, n_neighbors=100, k=10):
    """A wrapper of the ReliefF algorithm.

    Args:
        n_neighbors (int): The number of neighbors to consider when assigning
            feature importance scores.

    """

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    selector = ReliefF(n_neighbors=n_neighbors)
    selector.fit(X_train_std, y_train)

    support = selector.top_features[:k]

    return X_train_std[:, support], X_test_std[:, support], support


def sequential_forward_floating(**kwargs):
    """A wrapper of mlxtend Sequential Forward Floating Selection algorithm."""

    scorer = make_scorer(kwargs['scoring'])

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(
        kwargs['X_train'], kwargs['X_test']
    )

    # Set number of CPUs.
    n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    selector = SequentialFeatureSelector(
        kwargs['score_model'], k_features=kwargs['k'], forward=True,
        floating=True, scoring=scorer, cv=['cv'], n_jobs=n_jobs
    )
    selector.fit(X_train_std, kwargs['y_train'])

    #support = selector.k_feature_idx_

    #return X_train_std[:, support], X_test_std[:, support], support


def feature_evluation():

    # Use /Drop-out feature importanceFeature importance permutation on subset
    # of features associated with the with best model after model comparison
    # experiments.

    pass


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
    score_model = ElasticNet(random_state=42)

    X_train_sub, X_test_sub, support = relieff(
        X_train=X_train, X_test=X_test, y_train=y_train, k=10, n_neighbors=100
    )
    print(support)
