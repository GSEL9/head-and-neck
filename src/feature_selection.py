import utils
import numpy as np

from sklearn import feature_selection


# Filter selection algorithms:
# * feature_selection.VarianceThreshold
#   - Select feature based on variance threshold
# * feature_selection.chi2 (OBS: Cannot handle negative features)
#   - Select feature based on p-values > 0.05
# * feature_selection.mutual_info_classif
# * feature_selection.f_classif
#   - Select feature based on p-values > 0.05
# feature_selection.SelectFpr
#   - Select feature based on p-values > 0.05
# feature_selection.SelectFwe
#   - Select feature based on p-values > 0.05
# ReliefF


# Wrapped selection algorithms:
# * mlxtend.SequentialFeatureSelector
#   - Sequential Forward Floating Selection


# Embedded selection algorithms:
# feature_selection.SelectFromModel
#   - Grid Search CV + ElasticNet w/ hparam grid
#   - Grid Search CV + ElasticNet w/ hparam grid (OBS: Permutation importance)



# NOTE: Targeting robust features, should therefore select concensus of all
# methods?

# NOTE Experiments:
# * Compare outcome of grouping features into categories with using complete
#   feature pool when selecting features.



def feature_evluation():

    # Use /Drop-out feature importanceFeature importance permutation on subset
    # of features associated with the with best model after model comparison
    # experiments.

    pass



# NOTE:
# * On Comparison of Feature Selection Algorithms recommend IN scheme for
#   feaure sel in cross-val.
# * Feature sel best practices: https://medium.com/ai%C2%B3-theory-practice-business/three-effective-feature-selection-strategies-e1f86f331fb1
# * With p >> N: variance and overfitting are major concerns. Highly regularized
#   approaches often become the methods of choice.
# * Ridge regression with λ = 0.001 successfully exploits the correlation in the
#   features when p < N, but cannot do so when p ≫ N. There is not enough
#   information in the relatively small number of samples to efficiently
#   estimate the high-dimensional covariance matrix
# * Various Discriminant Analysis classifiers: https://github.com/manhtuhtk/mlpy/blob/master/mlpy/da.py
# * Yet despite the high dimensionality, radial kernels (Section 12.3.3)
#   sometimes deliver superior results in these high dimensional problems. The
#   radial kernel tends to dampen inner products between points far away from
#   each other, which in turn leads to robustness to outliers. This occurs
#   often in high dimensions, and may explain the positive results.


# TODO: Feature selection.
def dummy(X_train, X_test, y_train):

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    # Feature selection based on training set to avoid information leakage.
    support = np.arange(X_train.shape[1])

    return X_train_std[:, support], X_test_std[:, support], support




if __name__ == '__main__':

    pass
