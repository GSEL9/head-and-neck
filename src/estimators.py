import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC, LinearSVC
from pyearth import Earth
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from skrebate import ReliefF


estimators = {
    'logreg_l1': LogisticRegression,
    'logreg_l2': LogisticRegression,
    'rf': RandomForestClassifier,
    'knn': KNeighborsClassifier,
    'adaboost': AdaBoostClassifier,
    'dtree': DecisionTreeClassifier,
    'gaussianb': GaussianNB,
    'lda': LinearDiscriminantAnalysis,
    'qda': QuadraticDiscriminantAnalysis,
    'mlp': MLPClassifier,
    'mars': Earth,
    'pls': PLSRegression,
    'svc': SVC,
    'linsvc': LinearSVC,
}
hparams = {
    'logreg_1': {
        'C': [0.0001, 0.001,0.005, 0.01,0.05, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'fit_intercept': [True], 'solver': ['liblinear'], 'penalty': ['l1'],
        'class_weight': ['balanced'],
    },
    'logreg_l2' : {
        'C': [0.0001, 0.001,0.005, 0.01,0.05, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'fit_intercept': [True], 'solver': ['liblinear'], 'penalty': ['l2'],
        'class_weight': ['balanced'],
    },
    'rf': {
        'n_estimators': [5, 10, 20, 100, 500],
        'max_depth': [1, 5, 10, 100],
    },
    'knn': {
        'leaf_size': [10, 30, 50],
        'n_neighbors': [2, 5, 10, 15, 20]
    },
    'adaboost': {
        'learning_rate': [0.05, 0.5, 1, 2, 3],
        'n_estimators': [100, 500, 1000]
    },
    'dtree': {
        'max_depth': [1, 5, 10, 100],
        'class_weight': ['balanced']
    },
    'gaussianb': {
    },
    'lda': {
        'solver' : ['lsqr'], 'shrinkage': [0.1, 0.5, 0.8],
        'priors': [0.776,0.224] 'n_components': [5, 10, 20, 30],
        'tol': [0.0001, 0.00001, 0.001, 0.01]
    },
    'qda': {
        'tol': [0.0001, 0.00001, 0.001, 0.01],
        'reg_param': [0.1, 0.5, 0.7, 0.9],
        'priors': [0.776,0.224]
    },
    'mlp': {
        'hidden_layer_sizes': [5, 10, 50, 100, 150, 200],
        'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'tol': [0.0001, 0.00001, 0.001, 0.01]
    },
    'mars': {
        'penalty' : [0.01, 0.05, 0.1, 0.5], 'minspan_alpha' : [1, 3, 5, 10]
    },
    'pls': {
        'n_components': [1, 10, 100, 500, 1000, 1500],
        'tol' : [1e-6, 1e-5, 1e-3], 'scale': [False]
    },
    'svc': {
        'class_weight': ['balanced'], 'gamma': [0.001, 0.05, 0.1, 0.5],
        'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'cache_range': [50, 100, 200, 300]
    },
    'linsvc': {
        'dual': [False], 'class_weight': ['balanced'],
        'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'tol': [0.00001, 0.0001, 0.001, 0.1, 1]
    },
}
