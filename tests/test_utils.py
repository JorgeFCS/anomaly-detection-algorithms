#!/usr/bin/env python
"""Testing suite for the functions inside utils.py.
"""

# Imports.
import os
import sys
import pytest

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
    print(dir_list)
    assert set(dir_list) == set(['0_dir', '1_dir', '2_dir']) 
