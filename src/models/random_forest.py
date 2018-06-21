#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 08:30:34 2018

@author: Juan Antonio Morales
"""

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.model_launcher import model_evaluation

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state = 4290)
    
    ### model
    print("Fitting random forest...")
    # logistic regression
    random_forest = RandomForestClassifier(
            bootstrap=True, criterion='entropy', max_depth=10, max_features=10, 
            min_samples_leaf=2, min_samples_split=3, n_estimators=200
            )
    random_forest.fit(X_train, y_train)
    
    ### evaluation
    print("Evaluation...")
    # scores
    y_scores_train = pd.DataFrame(y_train.reset_index())
    y_scores_train["scores"] = pd.DataFrame(random_forest.predict_proba(X_train)).loc[:,1]
    y_scores_train.columns = ["id","loan_status","scores"]
    
    y_scores_test = pd.DataFrame(y_test.reset_index())
    y_scores_test["scores"] = pd.DataFrame(random_forest.predict_proba(X_test)).loc[:,1]
    y_scores_test.columns = ["id","loan_status","scores"]
    
    # writing scores
    y_scores_train.to_csv("../output/scores/y_scores_train_rf.csv", sep = "^", index = False)
    y_scores_test.to_csv("../output/scores/y_scores_test_rf.csv", sep = "^", index = False)
    
    metrics = model_evaluation(y_train, y_test, y_scores_train["scores"], y_scores_test["scores"])
    
    return random_forest, metrics