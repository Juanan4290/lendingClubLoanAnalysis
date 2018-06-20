#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:48:05 2018

@author: Juan Antonio Morales
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, \
                            accuracy_score

from src.preprocessing_functions import categorical_to_numeric, normalize_variables

def model_evaluation(y_train, y_test, y_scores_train, y_scores_test, threshold = 0.5):
    """
    Parameters
    ---------
    y_train: true labels of the train set.
    y_test: true labels of the test set.
    y_scores_train: model scores for the train set prediction
    y_cores_test: model scores for the test set prediction
    threshold: boundary to classify predictions
    
    Returns:
    ---------
    result:list with the following metrics: 
          [auc_train, auc_test, accuracy_train, accuracy_test, 
          recall_train, recall_test, precision_train, precision_test]
    """
    
    # predictions
    y_train_pred = y_scores_train > threshold
    y_test_pred = y_scores_test > threshold
    
    # auc
    auc_train = roc_auc_score(y_train, y_scores_train)
    auc_test = roc_auc_score(y_test, y_scores_test)
    
    print("AUC:")
    print("AUC in the train set: {}".format(auc_train))
    print("AUC in the test set: {}".format(auc_test))
    print("------------------------")
    
    # confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # accuracy
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print("Accuracy: {}".format(accuracy_test))
    # recall
    recall_train = precision_recall_fscore_support(y_train, y_train_pred)[0][1]
    recall_test = precision_recall_fscore_support(y_test, y_test_pred)[0][1]
    print("Recall: {}".format(recall_test))
    # precision
    precision_train = precision_recall_fscore_support(y_train, y_train_pred)[1][1]
    precision_test = precision_recall_fscore_support(y_test, y_test_pred)[1][1]
    print("Precision: {}".format(precision_test))
    print("------------------------")
    
    result = pd.Series([auc_train, auc_test, accuracy_train, accuracy_test, recall_train, recall_test,
              precision_train, precision_test])
    
    return result

def api_predict(test_json):
    """
    Parameters
    ----------
    test_json: json with the test data we want to predict
    
    returns:
    ----------
    scores: test data scores with logistic regression, random forest and xg boost models
    """
    
    ## pre-processing data
    print("Pre-processing data...")
    # transform json to dataframe to preprocess data for feeding models
    test = pd.read_json(test_json, orient='records')
    
    # order columns in alphabetical order
    test = test[sorted(test.columns)]
    
    # get ids
    ids = test["id"]
    test = test.drop("id", axis = 1)
    
    # removing target
    if "loan_status" in test.columns:
        test = test.drop("loan_status", axis = 1)    
    
    ## pre-processing
    
    # get numeric and categorical variables
    numeric_variables = test._get_numeric_data().columns
    # remove "id" from numeric columns
    numeric_variables = [variable for variable in numeric_variables if variable != "id"]
    categorical_variables = test.select_dtypes(include="object").columns
    
    # reading numeric stats from training stage
    numeric_stats = pd.read_csv("./output/numeric_stats_in_training_for_new_data.csv", sep = "^")
    # normalization
    for variable in numeric_variables:
        mean = float(numeric_stats[numeric_stats["numeric_variable"] == variable]["mean"])
        std = float(numeric_stats[numeric_stats["numeric_variable"] == variable]["std"])
        test[variable] = test[variable].map(lambda i: (i - mean)/std)
    
    # reading categorical dictionary from training stage
    categorical_dict = pickle.load(open("./output/categorical_dict.pkl", "rb"))
    
    # pre-processing categorical data
    for variable in categorical_variables:
        test[variable] = test[variable].map(lambda i: categorical_dict[variable][i])
    
    ## loading models
    print("Loading models...")
    logistic_regression = pickle.load(open("./output/models/logistic_regression_model.sav","rb"))
    random_forest = pickle.load(open("./output/models/random_forest_model.sav","rb"))
    xg_boost = pickle.load(open("./output/models/xg_boost_model.sav","rb"))
    
    ## predictions
    print("Models have been loaded...doing predictions now...")
    logit_predictions = list(pd.DataFrame(logistic_regression.predict_proba(test)).loc[:,1])
    rf_predictions = list(pd.DataFrame(random_forest.predict_proba(test)).loc[:,1])
    xg_predictions = list(pd.DataFrame(xg_boost.predict_proba(np.matrix(test))).loc[:,1])

    scores = pd.DataFrame({"id": ids,
                           "logit": logit_predictions,
                           "rf": rf_predictions,
                           "xg": xg_predictions})
        
    return scores.to_json(orient = "records")