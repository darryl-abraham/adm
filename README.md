# Algorithmics for Data Mining
## Project 1: Exploratory Data Analysis and Data Preprocessing

### Files

#### `eda.ipynb`
This notebook contains the exploratory data analysis. The notebook is divided into the following sections:
1. Data Import
2. Data Overview
3. Univariate Analysis
4. Multivariate Analysis
5. Outlier Detection
6. Principal Component Analysis

#### `preprocessing.py`
This script contains the data preprocessing methods, instantiated by the `Preprocessing` class. The methods are:
1. Get univariate outliers
2. Get multivariate outliers
3. K-NN imputation
4. Scale data
5. Save

Full pipeline can be run by inserting the path to data in the Preprocessing class instantiation (end of script) and running the script.
Preprocessed data will be saved in the same directory as the script.