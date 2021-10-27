#!/usr/bin/env python
"""Genertic utility functions that can be used in multiple parts of the code.
"""

# Imports. --> Revisar que todos sean necesarios.
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
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.preprocessing import StandardScaler, MinMaxScaler

__author__ = "Jorge Ciprian and Alan Barlandas"
__credits__ = ["Jorge Ciprian", "Alan Barlandas"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

# def open_datasets(mypath):
#     """
#     Function that opens a .dat file and return the dataframe with training and testing
#     """
#     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#     final_doc_tra = []
#     final_doc_tst = []
#     for file in onlyfiles:
#         if "tst" in file:
#             archive = open("{}/{}".format(mypath,file))
#             for line in archive:
#                 if "@attribute" in line or "@relation" in line or "@data" in line: #Avoiding not necessary lines
#                     continue
#                 elif "@input" in line:
#                     aux_line = line[8:-2]
#                     #final_doc += aux_line.strip()
#                 elif "@output" in line:
#                     #aux_line = aux_line + ", " + line[9:-1]
#                     aux_line = aux_line + ", " + line[-6:-1].lower()
#                     #print("Line position: ", line[-6:-1].lower())
#                 else:
#                     final_doc_tst.append(line[:-1].split(", "))
#         elif "tra" in file:
#             archive = open("{}/{}".format(mypath,file))
#             for line in archive:
#                 if line[0] == "@":
#                     continue
#                 else:
#                     final_doc_tra.append(line[:-1].split(", "))
#     print("Aux line: ", aux_line)
#     df_training = pd.DataFrame(final_doc_tra,columns = list(aux_line.split(", ")))
#     df_testing = pd.DataFrame(final_doc_tst,columns = list(aux_line.split(", ")))
#     return df_training, df_testing

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

def cd_diagram(names,avranks):
    """
    function to create cd_diagrams. It recieves the methods names and the values obtained as the input, it returns the
    cd_diagram.
    """
    names = names
    avranks = avranks
    cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets --> Entiendo que esto se cambia por lo que te da Keel.
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    # Hay que guardar el plot como imagen.
    plt.show()

# ESto va en un test_utils.py
#cd_diagram(["first", "third", "second", "fourth" ],[1.9, 3.2, 2.8, 3.3 ])

def box_plot(dataframe,column_name):
    """
    Function that returns a box plot. The input is the column is the data of the models after realized the friedman and posthoc method.
    It returns the boxplot.
    """
    boxplot = dataframe.boxplot(column=column_name)
    plt.show()

def get_all_dirs(path):
    """
    Function that gets all the directory names inside a given path.
    """
    dir_list = [name for name in os.listdir(path) if os.path.isdir(path+name)]
    return dir_list
