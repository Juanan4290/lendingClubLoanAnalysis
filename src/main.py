#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:17:04 2018

@author: Juan Antonio Morales
"""

### 1. Libraries ###

import pandas as pd

from src.models.logistic_regression import logistic_regression
from src.feature_engineering.grade import feature_grade


if __name__ == '__main__':

    ### 2. read data ###
    loans = pd.read_csv("../data/loans_processed.csv", sep = "^").sample(200000, random_state=4290)
    
    ### 3. feature engineering
    no_important_features = ["bc_open_to_buy", "delinq_2yrs", "fico_range_low", "funded_amnt_inv", "funded_amnt",
                             "num_accts_ever_120_pd", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
                             "num_sats", "open_acc", "pub_rec", "pub_rec_bankruptcies", "revol_bal",
                             "tax_liens", "tot_coll_amt", "total_bal_ex_mort"]
    
    loans = loans.drop(no_important_features, axis = 1)
    
    loans = feature_grade(loans)
    
    ### 4. logistic regression
    logistic_regression(loans, normalization = "robust")
    
    
