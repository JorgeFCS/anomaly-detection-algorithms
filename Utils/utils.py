#!/usr/bin/env python
"""Genertic utility functions that can be used in multiple parts of the code.
"""
#imports
from os import listdir
from os.path import isfile, join
import pandas as pd 
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
import itertools
import statsmodels.api as sa
import statsmodels.formula.api as sfa
import Orange
import matplotlib.pyplot as plt

__author__ = "Jorge Ciprian and Alan Barlandas"
__credits__ = ["Jorge Ciprian", "Alan Barlandas"]
__license__ = "MIT"
__version__ = "0.1.0"
__status__ = "Development"

def open_datasets(mypath):
    """
    Function that opens a .dat file and return the dataframe with training and testing
    """
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    final_doc_tra = []
    final_doc_tst = []
    for file in onlyfiles:
        if "tst" in file:    
            archive = open("{}/{}".format(mypath,file))   
            for line in archive:
                if "@attribute" in line or "@relation" in line or "@data" in line: #Avoiding not necessary lines 
                    continue
                elif "@input" in line:
                    aux_line = line[8:-2] 
                    #final_doc += aux_line.strip()
                elif "@outputs" in line:
                    aux_line = aux_line + ", " + line[9:-1]
                else:
                    final_doc_tst.append(line[:-1].split(", "))
        elif "tra" in file:
            archive = open("{}/{}".format(mypath,file))
            for line in archive:
                if line[0] == "@":
                    continue
                else:
                    final_doc_tra.append(line[:-1].split(", "))
    df_training = pd.DataFrame(final_doc_tra,columns = list(aux_line.split(", ")))
    df_testing = pd.DataFrame(final_doc_tst,columns = list(aux_line.split(", ")))
    return df_training,df_testing

def friedman_posthoc(dataframe):
    """
    Friedman test with a posthoc test
    """
    #no usar aún, tengo un problema con llamar las columnas de los dataframes, grupo que usemos
    f_test = friedmanchisquare(groups)
    f_res = pd.DataFrame({'test':'Friedman','statistic':f_test[0],'pvalue':f_test[0]},index=[0])
    wilc_test = [wilcoxon(dataframe[i],dataframe[j]) for i,j in itertools.combinations(dataframe.columns,2)]    
    w_res = pd.DataFrame(wilc_test)
    w_res['test'] = ["wilcoxon " + i+" vs "+j for i,j in itertools.combinations(dataframe.columns,2)]
    return pd.concat([f_res,w_res])


def cd_diagram(names,avranks):
    """
    function to create cd_diagrams.
    """
    names = names
    avranks = avranks
    cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
    plt.show()
    return cd