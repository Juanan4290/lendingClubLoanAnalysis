#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:17:04 2018

@author: Juan Antonio Morales
"""

### 1. Libraries ###

import numpy as np
import pandas as pd

from src.preprocessing_functions import categorical_to_numeric, reject_outliers
from src.models.logistic_regression import logistic_regression
from src.models.random_forest import random_forest
from src.models.xg_boost import xg_boost
from src.models.nn_autoencoder import nn_autoencoder

import pickle
import sys

if __name__ == '__main__':

    ### 2. read data ###
    loans = pd.read_csv("/media/juanan/DATA/loan_data_analysis/data/loans_processed.csv", sep = "^")
          
    ### 3. pre-processing
    # numeric variables
    numeric_variables = loans._get_numeric_data().columns
    loans = reject_outliers(loans, numeric_variables, z_score = 3.5)
    
    # categorical variables
    categorical_variables = loans.select_dtypes(include="object").columns
    
    for variable in categorical_variables:
        loans[variable] = categorical_to_numeric(loans, variable, "loan_status")
    
    # sort columns in alphabetical order
    loans = loans[sorted(loans.columns)]
    
    ### 4a. logistic regression
    print("LOGISTIC REGRESSION MODEL ----------------------------")
    log_reg, logit_metrics = logistic_regression(loans, normalization = "standard")
    #sys.exit()
    ### 4b. random forest
    print("RANDOM FOREST MODEL ----------------------------")
    random_forest, rf_metrics = random_forest(loans)
    
    ### 4c. xg boost
    print("XG BOOST MODEL ----------------------------")
    xg_boost, xg_metrics = xg_boost(loans)
    
    ### 4d. autoencoder
    print("AUTOENCODER FOR FEATURE EXTRACTION -------------------")
    nn_logit, nn_logit_metrics = nn_autoencoder(loans, 150, 150, 50, 64, 0.001, "minMax")
    
    
    ### 5. output
    # metrics
    metrics = np.transpose(pd.concat([logit_metrics, 
                                      rf_metrics,
                                      xg_metrics,
                                      nn_logit_metrics], axis = 1))
    
    metrics.columns = ["auc_train", "auc_test", "accuracy_train", "accuracy_test",
                       "recall_train", "recall_test", "precision_train", "precision_test"]
    metrics = metrics.rename(index={0: "logit",
                                    1: "rf",
                                    2: "xg",
                                    3: "autoencoder_logit"})
    
    metrics.to_csv("../output/metrics.csv", sep = "^", index = False)
    
    # saving models
    pickle.dump(log_reg, open("../output/models/logistic_regression_model.sav", "wb"))
    pickle.dump(random_forest, open("../output/models/random_forest_model.sav", "wb"))
    pickle.dump(xg_boost, open("../output/models/xg_boost_model.sav", "wb"))
    pickle.dump(nn_logit, open("../output/models/nn_logit_model.sav", "wb"))