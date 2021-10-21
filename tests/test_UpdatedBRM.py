#!/usr/bin/env python
"""Testing suite for the UpdatedBRM class.
"""

# Imports.
import os
import sys

# Changing home directory from pytest's default to current project's home
# directory.
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

# Continuing imports.
import pytest
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
# Importing custom class.
from Classes.UpdatedBRM import UpdatedBRM

__author__ = "Jorge Ciprian and Alan Barlandas"
__credits__ = ["Jorge Ciprian", "Alan Barlandas"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

def test_correct_class_creation():
    """
    Testing correct class instantiation.
    """
    brm = UpdatedBRM()
    assert brm.classifier_count == 100

def test_fit_predict_correct_euclidean_distances():
    """
    Testing correct fit and predict functionality.
    """
    # Creating mock source data.
    data_X = pd.DataFrame({'SepalLength': [5.1,4.9,4.6,5.0,5.4,6.5,6.6,6.8,6.0,7.0],
              'SepalWidth': [3.5,3.0,3.1,3.6,3.9,2.8,2.9,2.8,2.9,3.2],
              'PetalLength': [1.4,1.4,1.5,1.4,1.7,4.6,4.6,4.8,4.5,4.7],
              'PetalWidth': [0.2]*5 + [1.5,1.3,1.4,1.5,1.4]})
    data_y = pd.DataFrame({'Class': ['positive']*5 + ['negative']*5})
    # Creating brm instance.
    brm = UpdatedBRM()
    # Fitting BRM.
    brm.fit(euclidean_distances, data_X, data_y.to_numpy().flatten())
    # Generating predictions.
    y_pred = brm.score_samples(data_X, euclidean_distances)
    # Getting AUC score.
    auc = roc_auc_score(data_y, y_pred)
    if(auc <= 0.5):
        auc = 1 - auc
    assert auc == 1.0

def test_fit_predict_correct_cosine_distances():
    """
    Testing correct fit and predict functionality.
    """
    # Creating mock source data.
    data_X = pd.DataFrame({'SepalLength': [5.1,4.9,4.6,5.0,5.4,6.5,6.6,6.8,6.0,7.0],
              'SepalWidth': [3.5,3.0,3.1,3.6,3.9,2.8,2.9,2.8,2.9,3.2],
              'PetalLength': [1.4,1.4,1.5,1.4,1.7,4.6,4.6,4.8,4.5,4.7],
              'PetalWidth': [0.2]*5 + [1.5,1.3,1.4,1.5,1.4]})
    data_y = pd.DataFrame({'Class': ['positive']*5 + ['negative']*5})
    # Creating brm instance.
    brm = UpdatedBRM()
    # Fitting BRM.
    brm.fit(cosine_distances, data_X, data_y.to_numpy().flatten())
    # Generating predictions.
    y_pred = brm.score_samples(data_X, cosine_distances)
    # Getting AUC score.
    auc = roc_auc_score(data_y, y_pred)
    if(auc <= 0.5):
        auc = 1 - auc
    assert round(auc, 2) == 0.84
