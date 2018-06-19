# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 08:19:17 2018

@author: Juan Antonio Morales
"""

### 1. libraries ###
import warnings # remove warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from utils import process_emp_title


### 2. read data ###
print("Reading Data...")

loans = pd.read_csv("/media/juanan/DATA/loan_data_analysis/data/clean/loans.csv.gz")


### 3. cleaning data ###
print("Starting Cleaning Data...")

## filter only Fully Paid and Charged Off loans
loans = loans[(loans['loan_status'] == "Fully Paid") | (loans['loan_status'] == "Charged Off")]
# target to integer
loan_status_dict = {'Fully Paid': 0,
                    'Charged Off': 1}

loans['loan_status'] = loans['loan_status'].map(lambda i: loan_status_dict[i])

## removing columns with 25% of NA's
loans[loans == ""] = np.nan
na_percentage = loans.isnull().sum()/loans.shape[0] #number of NA's per column

loans = loans.loc[:,na_percentage < 0.25] 

## removing features with just one value
number_of_unique_values_per_column = loans.apply(lambda i: i.nunique(), axis = 0)
loans = loans.loc[:,number_of_unique_values_per_column != 1]


### 4. processing numeric variables ###
print("Processing numeric variables...")

numeric_variables = loans._get_numeric_data().columns
## fill NA's to the median
loans[numeric_variables] = loans[numeric_variables].apply(lambda i: i.fillna(i.median()), axis = 1)


### 5. processing categorical variables ###
print("Processing categorical variables...")

categorical_variables = loans.select_dtypes(include="object").columns
## fill NA's to last valid observation
loans[categorical_variables] = loans[categorical_variables].fillna(method = "ffill")
## process emp_title variable
loans["emp_title"] = process_emp_title(loans, n = 1)

### 6. writing clean data ###
print("Writing data to disk...")
print("Final dataset with {} rows and {} columns".format(loans.shape[0], loans.shape[1]))
loans.to_csv("/media/juanan/DATA/loan_data_analysis/data/loans_processed.csv", sep = "^", index = False)
print("Done!")