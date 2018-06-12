#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:17:04 2018

@author: Juan Antonio Morales
"""

### 1. Libraries ###

import pandas as pd

from src.utils import categorical_to_numeric, reject_outliers
from src.models.logistic_regression import logistic_regression


if __name__ == '__main__':

    ### 2. read data ###
    loans = pd.read_csv("/media/juanan/DATA/loan_data_analysis/data/loans_processed.csv", sep = "^")\
                        .sample(20000)
    
    ### 3. pre-processing
    # numeric variables
    numeric_variables = loans._get_numeric_data().columns
    loans = reject_outliers(loans, numeric_variables, z_score = 3.5)
    
    # categorical variables
    categorical_variables = loans.select_dtypes(include="object").columns
    
    for variable in categorical_variables:
        loans[variable] = categorical_to_numeric(loans, variable, "loan_status")
    
    ### 4. logistic regression
    print("LOGISTIC REGRESSION MODEL ----------------------------")
    log_reg, logit_metrics = logistic_regression(loans, normalization = "standard")
    
