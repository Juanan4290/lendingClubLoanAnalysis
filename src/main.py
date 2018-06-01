#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:17:04 2018

@author: Juan Antonio Morales
"""

### 1. Libraries ###

import pandas as pd
import numpy as np

from src.models.logistic_regression import logistic_regression
from src.feature_engineering.grade import feature_grade


if __name__ == '__main__':

    ### 2. read data ###
    loans = pd.read_csv("/media/juanan/DATA/loan_data_analysis/data/loans_processed.csv", sep = "^").sample(200000, random_state=4290)
    
    ### 3. feature engineering
    no_important_features = ["bc_open_to_buy", "delinq_2yrs", "fico_range_low", "funded_amnt_inv", "funded_amnt",
                             "num_accts_ever_120_pd", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
                             "num_sats", "open_acc", "pub_rec", "pub_rec_bankruptcies", "revol_bal",
                             "tax_liens", "tot_coll_amt", "total_bal_ex_mort", "last_fico_range_high"]
    
    loans = loans.drop(no_important_features, axis = 1)
    
    loans['fico_range_square'] = np.square(loans['fico_range_high'])
    loans['fico_range_cos'] = np.cos(loans['fico_range_high'])
    loans['fico_range_sin'] = np.sin(loans['fico_range_high'])
    
    ### 4. logistic regression
    log_reg = logistic_regression(loans, normalization = "robust", C = 0.5)
    
    
