#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:30:03 2018

@author: Juan Antonio Morales
"""

import pandas as pd

import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.utils import model_evaluation

def xg_boost(data):
    """
    Parameters
    ---------
    data: DataFrame to fit xg boost
    
    Returns
    ---------
    result: xg boost model and evaluation model: 
            AUC in train and test, confusion matrix, accuracy, 
            recall and precision (treshold = 0.5)
    """
    ### pre-process
    print("Preprocessing...")
        
    # train/test split
    X = data.loc[:, data.columns!='loan_status']
    y = data['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    X_train = X_train.as_matrix()
    X_test = X_test.as_matrix()
    
    ### model
    print("Fitting xg boost...")
    # logistic regression
    xg_boost = xgb.XGBClassifier(
            colsample_bylevel=0.6, colsample_bytree=0.5, gamma=0.5,
            learning_rate=0.2, max_depth=6, min_child_weight=7,
            reg_lambda=100, subsample=1, n_estimators=200
            )
    xg_boost.fit(X_train, y_train)
    
    ### evaluation
    print("Evaluation...")
    # scores
    y_scores_train = pd.DataFrame(xg_boost.predict_proba(X_train)).loc[:,1]
    y_scores_test = pd.DataFrame(xg_boost.predict_proba(X_test)).loc[:,1]
    
    # writing scores
    y_scores_train.to_csv("../output/scores/y_scores_train_xg.csv", sep = "^")
    y_scores_test.to_csv("../output/scores/y_scores_test_xg.csv", sep = "^")
    
    metrics = model_evaluation(y_train, y_test, y_scores_train, y_scores_test)
    
    return xg_boost, metrics