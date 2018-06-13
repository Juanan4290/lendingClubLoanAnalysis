# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:26:06 2018

@author: Juan Antonio Morales
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.utils import normalize_variables, model_evaluation

def logistic_regression(data, normalization = "standard"):
    """
    Parameters
    ---------
    data: DataFrame to fit logistic regression
    normalization: type of normalization to perform: "robust", "standard" and "minMax"
    
    Returns
    ---------
    result: logistic regression model and evaluation model: 
            AUC in train and test, confusion matrix, accuracy, 
            recall and precision (treshold = 0.5)
    """
    ### pre-process
    print("Preprocessing...")
    # normalization
    data = normalize_variables(data, normalization)
        
    # train/test split
    X = data.loc[:, data.columns!='loan_status']
    y = data['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    ### model
    print("Fitting logistic regression...")
    # logistic regression
    log_reg = LogisticRegression(penalty='l2', C = 10)
    log_reg.fit(X_train, y_train)
    
    ### evaluation
    print("Evaluation...")
    # scores
    y_scores_train = pd.DataFrame(log_reg.predict_proba(X_train)).loc[:,1]
    y_scores_test = pd.DataFrame(log_reg.predict_proba(X_test)).loc[:,1]
    
    # writing scores
    y_scores_train.to_csv("../output/scores/y_scores_train_logit.csv", sep = "^")
    y_scores_test.to_csv("../output/scores/y_scores_test_logit.csv", sep = "^")
    
    metrics = model_evaluation(y_train, y_test, y_scores_train, y_scores_test)
    
    return log_reg, metrics