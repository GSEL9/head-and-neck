# new_features = select_from_pool(feature_pool, to_retain)

import logging

import numpy as np

from joblib import Parallel, delayed
from scipy import stats
from collections import Counter
from collections import OrderedDict
from datetime import datetime
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid


from multiprocessing import cpu_count

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# logger = logging.GetLogger('')


def multi_intersect(arrays):
    """Determines the intersection between multiple sets."""

    sets = [set(array) for array in arrays]
    matches = set.intersection(*sets)

    return list(matches)


def model_comparison(*args, verbose=0, n_jobs=None, **kwargs):
    # TEMP:
    # AIM: Collecting repeated average performance data of optimal models.
    estimators, param_grids, X, y, random_states, n_splits = args

    logger = logging.getLogger(__name__)

    # Set number of CPUs.
    if n_jobs is None:
        n_jobs = cpu_count() - 1 if cpu_count() > 1 else cpu_count()

    comparison_results = {}
    for name, estimator in estimators.items():

        logger.info('Initiated model comparison with: `name`'.format(name))

        if verbose:
            print('Experiment initiated with: `{}`\n{}'.format(name, '_' * 40))

        # Setup hyperparameter grid.
        hparam_grid = ParameterGrid(param_grids[name])

        # Experimental results of model performance and selected feaures.
        comparison_results[name] = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(nested_cross_val)(
                X, y, estimator, hparam_grid, n_splits, random_state,
                verbose=verbose
            ) for random_state in random_states
        )

    return comparison_results


def nested_cross_val(*args, verbose=1, **kwargs):
    # TEMP:
    # AIM: Collecting average performance data of optimal model.
    X, y, estimator, hparam_grid, n_splits, random_state = args

    logger = logging.getLogger(__name__)
    logger.info('Initiated nested cross validation.')

    if verbose > 0:
        print('* Starting cross validation sequence {}'.format(random_state))
        start_time = datetime.now()

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
            estimator, hparam_grid, X_train, y_train, n_splits, random_state
        )
        # Z-scores and selected features from inner CV loop.
        X_train_std_sub, X_test_std_sub = train_test_z_scores(
            X_train[:, sel_features], X_test[:, sel_features]
        )
        best_model.fit(X_train_std_sub, y_train)
        # Aggregate model predictions with hparams and feature subset.
        train_scores.append(
            roc_auc_score(y_train, best_model.predict(X_train_std_sub))
        )
        test_scores.append(
            roc_auc_score(y_test, best_model.predict(X_test_std_sub))
        )
        best_features.append(sel_features)

    if verbose > 0:
        delta_time = datetime.now() - start_time
        print('* Nested cross validation complete in {}'.format(delta_time))
        print('* Mean test score: {}\n'.format(np.mean(test_scores)))

    return {
        'id': random_state,
        'best_model': best_model,
        'avg_test_score': np.mean(test_scores),
        'avg_train_score': np.mean(train_scores),
        'best_features': multi_intersect(best_features)
    }


def grid_search(*args, **kwargs):
    # TEMP:
    estimator, hparam_grid, X, y, n_splits, random_state = args

    logger = logging.getLogger(__name__)
    logger.info('Initiated grid search.')

    best_score, best_model, best_features = 0.0, None, None
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

            train_score, test_score, features = select_fit_predict(
                model, X_train, X_test, y_train, y_test, random_state
            )
            train_scores.append(train_score), test_scores.append(test_score)
            sel_features.append(features)

        # Update globally improved score from hparam combo and feature subset.
        if np.mean(test_scores) > best_score:
            best_score = np.mean(test_scores)
            best_model, best_features = model, sel_features

    return best_model, multi_intersect(sel_features)


def select_fit_predict(*args, **kwargs):
    # TEMP:
    model, X_train, X_test, y_train, y_test, random_state = args

    # Z-scores.
    X_train_std, X_test_std = train_test_z_scores(X_train, X_test)

    # Feature selection based on training set to avoid information leakage.
    concensus_support = np.arange(X_train.shape[1])
    X_train_std_sub = X_train_std[:, concensus_support]
    X_test_std_sub = X_test_std[:, concensus_support]

    model.fit(X_train_std_sub, y_train)
    # Aggregate model predictions with hparams combo selected feature subset.
    train_score = roc_auc_score(y_train, model.predict(X_train_std_sub))
    test_score = roc_auc_score(y_test, model.predict(X_test_std_sub))

    return train_score, test_score, concensus_support


def train_test_z_scores(X_train, X_test):
    """Compute Z-scores for training and test sets.

    Args:
        X_train (array-like): Training set.
        X_test (array-like): Test set.

    Returns:
        (tuple): Standard score values for training and test set.

    """

    # NOTE: Transform test data with training parameters set avoiding
    # information leakage.

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


if __name__ == '__main__':
    # NOTE: Use Elastic and RF because indicated with good performance in unbiased
    # studies?

    # NB: The purpose of a nested CV is not to select the parameters, but to
    # have an unbiased evaluation of what is the expected accuracy of your
    # algorithms. Cross-validation is a technique for estimating the
    # generalisation performance for a method of generating a model, rather
    # than of the model itself.

    # NOTE: Dealing with class imbalance:
    # * Better to use ROC on imbalanced data sets
    # * In scikit-learn: Assign larger penalty to wrong predictions on the
    #   minority class with class_weight='balanced' among model params.
    # * Upsampling of minority class/downsamlping majority class/generation of
    #   synthetic samples. See scikit-learn resample function to upsample minority
    #  function.
    # * Generate synthetic samples with SMOTE algorithm (p. 216).

    # NOTE:
    # Selecting features from training set only because should not use
    # test data in any part of training procedure. Cannot let information
    # about the full dataset leack into cross-validation to prevent
    # overfitting. Thus, must re-select feaures in each cross-validation
    # iteration.

    # TODO: Parallelize for models only first. For each round of parallelizing
    # obtains a list => nested parallelizing gives nested lists. PArallelizing
    # models gives lists with results for each model.
    # NOTE: Parallel
    # https://zacharyst.com/2016/03/31/parallelize-a-multifunction-argument-in-python/

    # NOTE: Nested CV
    # https://stats.stackexchange.com/questions/136296/implementation-of-nested-cross-validation

    # NOTE: Cross validation should always be the outer most loop
    # in any machine learning algorithm. So, split the data into 5
    # sets. For every set you choose as your test set (1/5), fit
    # the model after doing a feature selection on the training set
    # (4/5). Repeat this for all the CV folds - here you have 5 folds.
    # Now once the CV procedure is complete, you have an estimate of
    # your model's accuracy, which is a simple average of your
    # individual CV fold's accuracy. As far as the final set of
    # features for training the model on the complete set of data is
    # concerned, do the following to select the final set of features:
    # Each time you do a CV on a fold as outlined above, vote for
    # the features that you selected in that particular fold. At the
    # end of 5 fold CV, select a particular number of features that
    # have the top votes. Use the above selected set of features to
    # do one final procedure of feature selection and then train the
    # model on the complete data (combined of all 5 folds) and move
    # the model to production.

    # TODO: Logging and type checking

    import feature_selection

    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import roc_auc_score

    cancer = load_breast_cancer()
    # NB: roc_auc_score requires binary <int> target values.
    y = cancer.target
    X = cancer.data

    #n_splits = 10
    # random_states = np.arange(100)

    # TEMP:
    n_splits = 2
    random_states = np.arange(2)

    estimators = {
        'logreg': LogisticRegression
    }
    param_grids = {
        'logreg': {
            'C': [0.001, 0.05, 0.1], 'fit_intercept': [True, False]
        },
    }
    results = model_comparison(
        estimators, param_grids, X, y, random_states, n_splits
    )

    print(results)
