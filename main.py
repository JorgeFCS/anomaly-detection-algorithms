#!/usr/bin/env python
"""Main file that executes code for either getting AUC results for all the models
and datasets, or plots the results of statistical tests. Calls the corresponding
functions.
"""

# Imports.
import configparser
# Importing custom functions.
from Functions.utils import get_all_dirs, open_datasets

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
    # Getting the list of all datasets.
    dir_list = get_all_dirs(data_base_path)
    for i in range(60):
        #print("Directory: ", dir_list[i])
        # Loading current dataset.
        if(dir_list[i] == "abalone-3_vs_11"):
            train, test = open_datasets(data_base_path+dir_list[i]+"/")
            #print(train)
            #print(test)

if __name__ == '__main__':
    main()