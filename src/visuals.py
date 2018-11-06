# -*- coding: utf-8 -*-
#
# visuals.py
#

"""
Visualization tools.
"""

__author__ = 'Severin Langberg'
__email__ = 'langberg91@gmail.com'


import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


# NOTE: Globals.
np.set_printoptions(precision=2)


def roc_cv():
    """Receiver Operating Characteristic (ROC) with cross validation."""

    # from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

    pass


def model_learning(model, X, y, cv=10, scoring='rocs', nsplits=20):

    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.01, 1.0, nsplits)
    )
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(
        train_sizes, train_mean, '--', color='#111111',  label='Training score'
    )
    plt.plot(
        train_sizes, test_mean, color='#111111', label='Cross-validation score'
    )
    # Draw bands
    plt.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std,
        color='#DDDDDD'
    )
    plt.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std,
        color='#DDDDDD'
    )
    # Create plot
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size'), plt.ylabel('Score'), plt.legend(loc='best')
    plt.tight_layout()

    return plt


def param_validation(model, X, y, param_name, params, cv=10, scoring='rocs'):

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(
        model, X, y,
         param_name=param_name,
         param_range=params,
         cv=cv, scoring=scoring, n_jobs=-1
    )
    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(params, train_mean, label='Training score', color='k')
    plt.plot(
        params, test_mean, label='Cross-validation score', color='dimgrey'
    )

    # Plot accurancy bands for training and test sets
    plt.fill_between(
        params, train_mean - train_std, train_mean + train_std, color='gray'
    )
    plt.fill_between(
        params, test_mean - test_std, test_mean + test_std, color='gainsboro'
    )
    # Create plot
    plt.title('Validation Curve')
    plt.xlabel('Hyperparameter'), plt.ylabel('Score')
    plt.tight_layout()
    plt.legend(loc='best')

    return plt


def confusion_matrix(y_test, y_pred, classes, title='Confusion matrix'):

    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title), plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return plt


def feature_rankings():

    pass


def score_metrics():


if __name__ == '__main__':

    import pandas as pd

    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_decomposition import PLSRegression

    path_to_data = './../../data/to_analysis/lbp/ct3_pet1_clinical.csv'
    path_to_pfstarget = './../../data/to_analysis/target_pfs.csv'
    path_to_lrctarget = './../../data/to_analysis/target_lrc.csv'

    SEED = 0
    SCORING = 'roc_auc'

    TOL = 1e-5
    MAX_ITER = 800
    PFS_PRIORS = [0.677, 0.323]
    LRC_PRIORS = [0.753, 0.247]

    X = np.array(
        pd.read_csv(path_to_data, index_col=0).values, dtype=float
    )
    y_pfs = np.squeeze(
        pd.read_csv(path_to_pfstarget, index_col=0).values
    )
    y_lrc = np.squeeze(
        pd.read_csv(path_to_lrctarget, index_col=0).values
    )

    # Learning curves
    """
    params = np.logspace(1e-4, 1e2, 10)
    model = LogisticRegression(
        penalty='l2', tol=TOL, C=1.0, intercept_scaling=1,
        class_weight=PFS_PRIORS, random_state=SEED, solver='sag',
        max_iter=MAX_ITER, n_jobs=-1
    )
    #model = PLSRegression(
    #    n_components=2, scale=True, max_iter=MAX_ITER, tol=TOL, copy=True
    #)
    plt = param_validation(model, X, y_pfs, param_name='C', params=params, scoring=SCORING)
    plt.show()
    """

    # Plot confusion matrix for PFS and LRC together:
    """
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    classes = np.unique(y)
    plt.figure()
    plt = plot_confusion_matrix(
        y_test, y_pred, classes=class_names,
    )
    plt.show()
    """
