#!/usr/bin/env python
"""Main file that executes code for either getting AUC results for all the models
and datasets, or plots the results of statistical tests. Calls the corresponding
functions.
"""

# Imports.
import configparser
import pandas as pd
from tqdm import tqdm
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
# Importing custom functions and classes.
from Classes.UpdatedBRM import UpdatedBRM
from Functions.utils import box_plot, get_all_dirs, open_datasets, prepare_datasets, get_auc_score, saveCD

__author__ = "Jorge Ciprian and Alan Barlandas"
__credits__ = ["Jorge Ciprian", "Alan Barlandas"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

def main():
    # Loading configuration file.
    config = configparser.ConfigParser()
    config.read('config.ini')
    task = config['CONFIG'].get('task')
    data_base_path = config['CONFIG'].get('dataset_path')
    # Perform according to task.
    if task == "AUC":
        # Initializing index list.
        indexes = []
        values = []
        # Initializing dissimilarities list.
        dis_list = [euclidean_distances, cosine_distances, manhattan_distances]
        # Initializing models.
        model_list = [IsolationForest(random_state=42), GaussianMixture(n_components=2, random_state=0),
                      UpdatedBRM(), OneClassSVM(gamma='auto')]
        # Initializing model names array.
        model_names = ["ISOF", "GMM", "BRM", "ocSVM"]
        dis_names = ["_euc", "_cos", "_man"]
        # Getting the list of all datasets.
        dir_list = get_all_dirs(data_base_path)
        # Iterating over the first 60 datasets.
        print("Processing datasets. This may take some minutes.")
        for i in tqdm(range(60)):
            print("Processing dataset ", dir_list[i], "...")
            # Getting train and test partitions.
            train_df, test_df = open_datasets(data_base_path+dir_list[i]+"/")
            # Getting the pre-processed partitions.
            X_train, y_train, X_test, y_test, X_train_minmax, X_test_minmax, X_train_stand, X_test_stand =\
            prepare_datasets(train_df, test_df)
            # Iterating over all models for this dataset.
            # Resetting results dictionary.
            results_row = {}
            for j, model in enumerate(model_list):
                # For isolation forest and GMM.
                if j < 2:
                    # First with no pre-processing.
                    auc = get_auc_score(model_list[j], X_train, y_train, X_test, y_test)
                    results_row[model_names[j]] = auc
                    # With minmax pre-processing.
                    auc = get_auc_score(model_list[j], X_train_minmax, y_train, X_test_minmax, y_test)
                    results_row[model_names[j]+"_minmax"] = auc
                    # With standard scaling.
                    auc = get_auc_score(model_list[j], X_train_stand, y_train, X_test_stand, y_test)
                    results_row[model_names[j]+"_stand"] = auc
                elif j == 2: # For BRM
                    # With different distances.
                    for k in range(3):
                        # No pre-processing.
                        auc = get_auc_score(model_list[j], X_train, y_train, X_test, y_test, dis_list[k])
                        results_row[model_names[j]+dis_names[k]] = auc
                        # Minmax.
                        auc = get_auc_score(model_list[j], X_train_minmax, y_train, X_test_minmax, y_test, dis_list[k])
                        results_row[model_names[j]+dis_names[k]+"_minmax"] = auc
                        # Standard scaling.
                        auc = get_auc_score(model_list[j], X_train_stand, y_train, X_test_stand, y_test, dis_list[k])
                        results_row[model_names[j]+dis_names[k]+"_stand"] = auc
                else: # For ocSVM
                    auc = get_auc_score(model_list[j], X_train, y_train, X_test, y_test)
                    results_row[model_names[j]] = auc
            # Appending dictionary to results list.
            values.append(results_row)
            # Adding dataset name to index list.
            indexes.append(dir_list[i])
        # Generating dataframe.
        results_df = pd.DataFrame.from_records(values)
        results_df.index = indexes
        # Saving dataframe.
        results_df.to_csv("./Results/auc_scores.csv")
    elif task == "plot":
        print("Plotting boxplot")
        df_box_plot = pd.read_csv("./Results/auc_scores.csv")
        box_plot(df_box_plot)
        print("Plotting CD diagram")
        cd_diag = pd.read_csv("./Results/auc_scores_transpose.csv")
        saveCD(cd_diag)
    else:
        print("Invalid task! Valid options: AUC or plot.")

if __name__ == '__main__':
    main()
