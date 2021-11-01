#!/usr/bin/env python
"""Genertic utility functions that can be used in multiple parts of the code.
"""


import os
import Orange
import itertools
import numpy as np
import pandas as pd
from os import listdir
import statsmodels.api as sa
import matplotlib.pyplot as plt
from os.path import isfile, join
import statsmodels.formula.api as sfa
from sklearn.metrics import roc_auc_score
from autorank import autorank

from sklearn.preprocessing import StandardScaler, MinMaxScaler

__author__ = "Jorge Ciprian and Alan Barlandas"
__credits__ = ["Jorge Ciprian", "Alan Barlandas"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"


def open_datasets(path):
    """
    Function that reads the contents of a .dat file and returns training and
    testing partitions as Pandas dataframes.
    """
    # Getting list of files.
    file_list = [f for f in listdir(path) if isfile(join(path, f))]
    # Initializing variables.
    cols = []
    skip_lines = 0
    files_to_read = {}
    for file_name in file_list:
        if("tst" in file_name): # If it is the test file.
            # Adding file name to files to read as CSV.
            files_to_read["test"] = path+file_name
            # Opening file.
            file = open(path+file_name, "r")
            for line in file:
                if("@attribute" in line or "@relation" in line or "@data" in line):
                    # Adding to counter for skip lines.
                    skip_lines+=1
                elif("@input" in line):
                    # Separating and removing everything before first space.
                    trim_line = line.split(' ', 1)[1].lower()
                    # Removing all spaces now.
                    trim_line = trim_line.replace(' ', '')
                    # Removing end of line character.
                    trim_line = trim_line.replace('\n', '')
                    # Getting column names - separating by commas.
                    cols = cols + trim_line.split(',')
                    # Adding skipline counter.
                    skip_lines+=1
                elif("@output" in line):
                    # Separating and removing everything before first space.
                    trim_line = line.split(' ', 1)[1].lower()
                    # Removing end of line character.
                    trim_line = trim_line.replace('\n', '')
                    # Adding name to column list.
                    cols.append(trim_line)
                    # Adding skipline counter.
                    skip_lines+=1
                else:
                    pass
            # Closing file.
            file.close()
    # Generating train file name.
    files_to_read["train"] = files_to_read["test"].replace('tst', 'tra')
    # Reading files as CSV.
    train_df = pd.read_csv(files_to_read["train"], skiprows=skip_lines,
                           names=cols, skipinitialspace=True)
    test_df = pd.read_csv(files_to_read["test"], skiprows=skip_lines,
                          names=cols, skipinitialspace=True)
    return train_df, test_df

def prepare_datasets(train_df, test_df):
    """
    Function that returns the datasets as X (features) and y (target) partitions
    with one-hot encoding, min-max scaling and standard scaling.
    """
    # Initializing variables.
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    # Separating dataset into X (features) and y (target) partitions.
    X_train = train_df.drop(['class'], axis=1)
    y_train = train_df[['class']]
    X_test = test_df.drop(['class'], axis=1)
    y_test = test_df[['class']]
    # One-hot encoding.
    X_train_pre = pd.get_dummies(X_train)
    X_test_pre = pd.get_dummies(X_test)
    # Min-max scaling.
    X_train_minmax = minmax_scaler.fit_transform(X_train_pre)
    X_test_minmax = minmax_scaler.fit_transform(X_test_pre)
    # Standard scaling.
    X_train_stand = standard_scaler.fit_transform(X_train_pre)
    X_test_stand = standard_scaler.fit_transform(X_test_pre)
    return X_train_pre, y_train, X_test_pre, y_test, X_train_minmax, X_test_minmax, X_train_stand, X_test_stand

def get_auc_score(model, X_train, y_train, X_test, y_test, distance=None):
    """
    Function that trains a model, evaluates it and returns the AUC score.
    """
    if distance is None:
        model.fit(X_train, y_train.to_numpy().flatten())
        y_pred = model.score_samples(X_test)
    else:
        model.fit(distance, X_train, y_train.to_numpy().flatten())
        y_pred = model.score_samples(X_test, distance)
    auc = roc_auc_score(y_test, y_pred)
    if(auc <= 0.5):
        auc = 1 - auc
    return auc


def box_plot(dataframe):
    """
    Function that returns a box plot. The input is the column is the data of the models after realized the friedman and posthoc method.
    It returns the boxplot.
    """
    boxplot = dataframe.boxplot(rot=45)
    print(type(boxplot))
    plt.title("Boxplot_methods")
    plt.xlabel("Algorithm")
    plt.ylabel("AUC score")
    plt.savefig("../Results/boxplots.png")
    plt.show()
    plt.close()

def get_all_dirs(path):
    """
    Function that gets all the directory names inside a given path.
    """
    dir_list = [name for name in os.listdir(path) if os.path.isdir(path+name)]
    return dir_list



def saveCD(data, name='test', title='CD_DIAGRAM'):
    """
    Function that creates the CD diagram using the Friedman test and the post hoc test Nemenyi.
    It shows the CD_diagram
    """
    models = list(data.model)
    data = data.drop(columns=['model'])
    values = data.values
    values = values.T
    data = pd.DataFrame(values, columns=models)
    result = autorank(data, alpha=0.05, verbose=False)
    print(result)
    critical_distance = result.cd
    rankdf = result.rankdf
    avranks = rankdf.meanrank
    ranks = list(avranks.values)
    names = list(avranks.index)
    names = names[:60]
    avranks = ranks[:60]
    Orange.evaluation.graph_ranks(avranks, names, cd=critical_distance, width=10, textspace=1.5, labels=True)
    plt.suptitle(title)
    plt.savefig("../Results/CD_diagram.png")
    plt.show()
    plt.close()