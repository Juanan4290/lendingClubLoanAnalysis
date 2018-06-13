#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:30:03 2018

@author: juanan
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
            base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
            colsample_bytree=0.5, gamma=0.25, learning_rate=0.1,
            max_delta_step=0, max_depth=6, min_child_weight=10.0, missing=None,
            n_estimators=200, n_jobs=1, nthread=None,
            objective='binary:logistic', random_state=0, reg_alpha=0,
            reg_lambda=50.0, scale_pos_weight=1, seed=None, silent=True,
            subsample=1.0
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