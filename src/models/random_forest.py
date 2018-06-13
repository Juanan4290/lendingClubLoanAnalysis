#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:30:34 2018

@author: juanan
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.utils import model_evaluation

def random_forest(data):
    """
    Parameters
    ---------
    data: DataFrame to fit random forest
    
    Returns
    ---------
    result: random forest model and evaluation model: 
            AUC in train and test, confusion matrix, accuracy, 
            recall and precision (treshold = 0.5)
    """
    ### pre-process
    print("Preprocessing...")
        
    # train/test split
    X = data.loc[:, data.columns!='loan_status']
    y = data['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    ### model
    print("Fitting random forest...")
    # logistic regression
    random_forest = RandomForestClassifier(
            bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features=9, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=9, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=4,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False
            )
    random_forest.fit(X_train, y_train)
    
    ### evaluation
    print("Evaluation...")
    # scores
    y_scores_train = pd.DataFrame(random_forest.predict_proba(X_train)).loc[:,1]
    y_scores_test = pd.DataFrame(random_forest.predict_proba(X_test)).loc[:,1]
    
    # writing scores
    y_scores_train.to_csv("../output/scores/y_scores_train_rf.csv", sep = "^")
    y_scores_test.to_csv("../output/scores/y_scores_test_rf.csv", sep = "^")
    
    metrics = model_evaluation(y_train, y_test, y_scores_train, y_scores_test)
    
    return random_forest, metrics