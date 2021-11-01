#!/usr/bin/env python
"""Testing suite for the functions inside utils.py.
"""

# Imports.
import os
import sys
import pytest
from pandas._testing import assert_frame_equal
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Changing home directory from pytest's default to current project's home
# directory.
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

# Importing functions.
from Functions.utils import *

# Testing the get_all_dirs function.
def test_get_all_dirs_correct(tmp_path):
    """
    Testing the get_all_dirs function with correct input and existing
    directories.
    """
    # Creating temporal directory.
    temp_dir = tmp_path / "sub"
    temp_dir.mkdir()
    # Creating mock sub-directories.
    for i in range(3):
        dir_name = str(i) + "_dir"
        sub_dir = temp_dir / dir_name
        sub_dir.mkdir()
    dir_list = get_all_dirs(temp_dir.as_posix()+"/")
    assert set(dir_list) == set(['0_dir', '1_dir', '2_dir'])

def test_get_all_dirs_mixed_contents(tmp_path):
    """
    Testing the get_all_dirs function with mixed contents between files and
    directories.
    """
    # Creating temporal directory.
    temp_dir = tmp_path / "sub"
    temp_dir.mkdir()
    # Creating mock text file.
    file_path = temp_dir / "hello.txt"
    file_path.write_text("hi!")
    # Creating mock sub-directories.
    for i in range(3):
        dir_name = str(i) + "_dir"
        sub_dir = temp_dir / dir_name
        sub_dir.mkdir()
    dir_list = get_all_dirs(temp_dir.as_posix()+"/")
    assert set(dir_list) == set(['0_dir', '1_dir', '2_dir'])

# Testing the open_datasets function.
def test_open_datasets_correct(tmp_path):
    """
    Testing the open_datasets function with correct functionality.
    """
    # Creating mock content.
    content = """@relation abalone-3_vs_11
@attribute Sex {M, F, I}
@attribute Length real [0.075, 0.815]
@attribute Diameter real [0.055, 0.65]
@attribute Height real [0.0, 1.13]
@attribute Whole_weight real [0.002, 2.8255]
@attribute Shucked_weight real [0.001, 1.488]
@attribute Viscera_weight real [5.0E-4, 0.76]
@attribute Shell_weight real [0.0015, 1.005]
@attribute Class {positive, negative}
@inputs Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight
@outputs Class
@data
I, 0.11, 0.09, 0.03, 0.008, 0.0025, 0.002, 0.003, positive
I, 0.165, 0.12, 0.03, 0.0215, 0.007, 0.005, 0.005, positive
"""
    # Creating reference dataset.
    ref_df = pd.DataFrame({"sex": ["I", "I"],
                           "length": [0.11, 0.165],
                           "diameter": [0.09, 0.12],
                           "height": [0.03, 0.03],
                           "whole_weight": [0.008, 0.0215],
                           "shucked_weight": [0.0025, 0.007],
                           "viscera_weight": [0.002, 0.005],
                           "shell_weight": [0.003, 0.005],
                           "class": ["positive", "positive"]})
    # Creating temporal directory.
    temp_dir = tmp_path / "sub"
    temp_dir.mkdir()
    # Creating mock training file.
    file_path = temp_dir / "tra.dat"
    file_path.write_text(content)
    # Creating mock testing file.
    file_path = temp_dir / "tst.dat"
    file_path.write_text(content)
    # Reading training and testing sets.
    train, test = open_datasets(temp_dir.as_posix()+"/")
    assert_frame_equal(train, ref_df, check_column_type=False)
    assert_frame_equal(test, ref_df, check_column_type=False)


def test_boxplot_correct_input():
    """
    Testing the box_plot function with the correct dataframe input
    """
    cancer = load_breast_cancer()
    df1 = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    box_p = box_plot(df1)
    assert type(box_p) != type(df1)

def test_boxplot_wrong_input_string():
    """
    Testing the box_plot function with the wrong dataframe input (string)
    """
    cancer = load_breast_cancer()
    df1 = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    with pytest.raises(AttributeError):
        box_p = box_plot("hola mundo")

def test_boxplot_wrong_input_integer():
    """
    Testing the box_plot function with the wrong dataframe input (integer)
    """
    cancer = load_breast_cancer()
    df1 = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    with pytest.raises(AttributeError):
        box_p = box_plot(5)

def test_boxplot_wrong_input_boolean():
    """
    Testing the box_plot function with the wrong dataframe input (boolean)
    """
    cancer = load_breast_cancer()
    df1 = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    with pytest.raises(AttributeError):
        box_p = box_plot(True)

def test_boxplot_wrong_input_boolean():
    """
    Testing the box_plot function with the wrong dataframe input (boolean)
    """
    cancer = load_breast_cancer()
    df1 = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    with pytest.raises(AttributeError):
        box_p = box_plot(True)

def test_boxplot_wrong_input_float():
    """
    Testing the box_plot function with the wrong dataframe input (float)
    """
    cancer = load_breast_cancer()
    df1 = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    with pytest.raises(AttributeError):
        box_p = box_plot(5.9)

def test_CD_diagram_wrong_input_string():
    """
    Testing the CD_diagram function with the wrong dataframe input (string)
    """
    with pytest.raises(AttributeError):
        cd_diagram = saveCD("hola mundo")

def test_CD_diagram_wrong_input_integer():
    """
    Testing the CD_diagram function with the wrong dataframe input (integer)
    """
    with pytest.raises(AttributeError):
        cd_diagram = saveCD(5)

def test_CD_diagram_wrong_input_boolean():
    """
    Testing the CD_diagram function with the wrong dataframe input (boolean)
    """
    with pytest.raises(AttributeError):
        cd_diagram = saveCD(True)

def test_CD_diagram_wrong_input_float():
    """
    Testing the CD_diagram function with the wrong dataframe input (float)
    """
    with pytest.raises(AttributeError):
        cd_diagram = saveCD(5.5)

def test_CD_diagram_wrong_input_none():
    """
    Testing the CD_diagram function with the wrong dataframe input (Missing data)
    """
    with pytest.raises(TypeError):
        cd_diagram = saveCD()