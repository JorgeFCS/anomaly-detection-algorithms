# anomaly-detection-algorithms
Implementation and evaluation of the BRM, GMM, ISOF, and ocSVM anomaly detection algorithms over 60 datasets. We obtain the AUC scores for each method and dataset, testing as well different data pre-processing techniques (min-max and standard scaling). For the BRM method, we evaluate it with three dissimilarity metrics (euclidean, cosine, and manhattan distances). We then perform a Friedman and Nemenyi post-hoc tests and visualize the results through boxplots and CD diagrams.

## System requirements

The code for the present assignment was developed and tested under Python 3.9.6 and Ubuntu 16.04.6 LTS. We provide a requirements.txt file for ease of use to install the required dependencies.

To install the required dependencies using the provided *requirements.txt* file, go to the directory that contains this file, initialize your virtual environment, and type the following in the command line:

```
pip install -r requirements.txt
```

## Running the program

In order to run the program, please modify the *config.ini* file to select the task that you wish to perform ('AUC' for training the model and saving the AUC results, 'plot' to generate the test results and the output plots), and to modify the path to the source datasets, if necessary.

Once the *config.ini* file is ready, you just need to go to the root directory (the one that contains the *main.py* file) and type the following in the command line:

```
python main.py
```

Depending on your Python version, you might need to type it as follows:

```
python3 main.py
```

If you wish to run the tests that we provide, in the root directory you just need to type the following:

```
pytest
```

All the files generated by this program will be automatically saved inside the *Results* directory.

