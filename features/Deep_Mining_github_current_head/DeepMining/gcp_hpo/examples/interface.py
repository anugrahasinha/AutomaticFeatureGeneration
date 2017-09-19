"""
Simple examples to show how to use SmartSearch.
"""

from gcp_hpo.smart_search import SmartSearch

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import numpy as np


def hpo_custom_function():
    parameters = {'kernel': ['cat', ['rbf', 'poly']],
                  'd': ['int', [1, 3]],
                  'C': ['float', [1, 10]]}

    def scoring_function(x):
        return [0.5]

    search = SmartSearch(parameters, model='GP', estimator=scoring_function, n_iter=20)
    search._fit()


def hpo_sklearn_pipeline():
    iris = load_digits()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier(n_estimators=20)

    # specify parameters and distributions to sample from
    parameters = {"max_depth": ['int', [3, 3]],
                  "max_features": ['int', [1, 11]],
                  "min_samples_split": ['int', [1, 11]],
                  "min_samples_leaf": ['int', [1, 11]],
                  "bootstrap": ['cat', [True, False]],
                  "criterion": ['cat', ["gini", "entropy"]]}

    search = SmartSearch(parameters, estimator=clf, X=X, y=y, n_iter=20)
    search._fit()


if __name__ == "__main__":

    print 'Run examples'
    print '\nHyper-parameter optimization with a custom function'
    hpo_custom_function()
    print '\nHyper-parameter optimization with an sklearn pipeline' \
          ' and the Iris dataset'
    hpo_sklearn_pipeline()