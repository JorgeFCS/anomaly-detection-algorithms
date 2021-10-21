#!/usr/bin/env python
"""Class that extends on the BRM class, removing automatic normalization on
training and allowing for any sklearn pairwise metric.
"""

# Imports.
import math
import random
import numpy as np
import pandas as pd
from brminer import BRM
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Setting random seed for reproducibility.
random.seed(42)
np.random.seed(42)

__author__ = "Jorge Ciprian and Alan Barlandas"
__credits__ = ["Jorge Ciprian", "Alan Barlandas"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

class UpdatedBRM(BRM, object):
    """
    Class that extends the BRM's "fit" method.

    Inherits form the original BRM class and overrides the original "fit"
    method, removing automatic scaling/normalization and accepting any of
    sklearns pairwise metrics for dissimilarity computation.
    """
    def __init__(self, classifier_count=100, bootstrap_sample_percent=100,
                 use_bootstrap_sample_count=False, bootstrap_sample_count=0,
                 use_past_even_queue=False, max_event_count=3, alpha=0.5,
                 user_threshold=95):
        # Calling the constructor to parent class.
        super(UpdatedBRM, self).__init__(classifier_count=100, bootstrap_sample_percent=100,
                     use_bootstrap_sample_count=False, bootstrap_sample_count=0,
                     use_past_even_queue=False, max_event_count=3, alpha=0.5,
                     user_threshold=95)

    def fit(self, X, y=None):
        """
        Overriding BRM's fit method.
        """
        # Check that X and y have correct shape.
        if y is not None:
            X_train, y_train = check_X_y(X, y)
        else:
             X_train = check_array(X)
        # Initializing variables.
        self._similarity_sum = 0
        self._is_threshold_Computed = False
        self.n_features_in_ = X_train.shape[1]
        # Validating correct vector shape.
        if self.n_features_in_ < 1:
            raise ValueError('Unable to instantiate the train dataset - Empty vector')
        X_train = pd.DataFrame(X_train)
        self._max_dissimilarity = math.sqrt(self.n_features_in_)
        self._sd = np.empty(0)
        sampleSize = int(self.bootstrap_sample_count) if (self.use_bootstrap_sample_count) else int(0.01 * self.bootstrap_sample_percent * len(X_train));
        self._centers = np.empty((0, sampleSize, self.n_features_in_))
        list_instances = X_train.values.tolist()
        for i in range(0, self.classifier_count):
            centers = random.choices(list_instances, k=sampleSize)
            self._centers = np.insert(self._centers, i, centers, axis=0)
            self._sd = np.insert(self._sd, i, 2*(np.mean(euclidean_distances(centers, centers))/self._max_dissimilarity)**2)
        return self

    def score_samples(self, X):
        """
        Overriding the score_samples method.
        """
        X_test = np.array(X)
        result = []
        batch_size = 100
        for i in range(min(len(X_test), batch_size), len(X_test) + batch_size, batch_size):
            current_X_test = X_test[[j for j in range(max(0, i-batch_size), min(i, len(X_test)))]]
            current_similarity = np.average([np.exp(-np.power(np.amin(euclidean_distances(current_X_test, self._centers[i]), axis=1)/self._max_dissimilarity, 2)/(self._sd[i])) for i in range(len(self._centers))], axis=0)
            result = result + [j for j in list(map(self._evaluate, current_similarity))]
        return result
