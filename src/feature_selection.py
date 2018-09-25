import utils
import numpy as np


# TODO: Feature selection.
def dummy(X_train, X_test, y_train):

    # Z-scores.
    X_train_std, X_test_std = utils.train_test_z_scores(X_train, X_test)

    # Feature selection based on training set to avoid information leakage.
    support = np.arange(X_train.shape[1])

    return X_train_std[:, support], X_test_std[:, support], support


if __name__ == '__main__':

    pass
