#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 19:04:10 2018

@author: Juan Antonio Morales
"""

import pandas as pd

from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, \
                            accuracy_score

def reject_outliers(data, numeric_features, z_score = 2):
    """
    Parameters
    ---------
    data: DataFrame to remove outliers
    numeric_features: features for rejecting outliers. They have to be numerical
    z_score: number of standard deviations from the mean to consider an observation as outlier
    
    Returns
    ---------
    result: DataFrame without outliers in the input features
    """    
    outliers_indexes = []
    
    for col in numeric_features:
        outliers_from_col = data[scale(data[col]) > z_score].index
        
        outliers_indexes.extend(outliers_from_col)
    
    indexes_to_remove = list(set(outliers_indexes))
    indexes_to_remove_mask = data.index.isin(indexes_to_remove)
    result = data[~indexes_to_remove_mask]
    
    return result


def normalize_variables(data, normalization = "robust"):
    """
    Parameters
    ---------
    data: DataFrame to normalize
    normalization: type of normalization to perform: "robust", "standard" and "minMax"
    
    Returns
    ---------
    result: DataFrame with normalized variables
    """
    
    # numeric variables except target
    variables = data.loc[:,data.columns != "loan_status"]
    variables = variables._get_numeric_data().columns
    
    # normalization methods
    robust = RobustScaler()
    standard = StandardScaler()
    minMax = MinMaxScaler()
    
    normalization_dict = {"robust": robust,
                          "standard": standard,
                          "minMax": minMax}
    
    scaler = normalization_dict[normalization]
    
    # normalization
    print(scaler)
    scaler.fit(data[variables])
    data[variables] = scaler.transform(data[variables])
    
    return data


def categorical_to_numeric(data, categorical_variable, target):
    """
    Parameters
    ---------
    data: DataFrame for transforming categorical to numeric
    categorical_variable: variable we want to transform to the mean value of the target.
    target: target of the data
    
    Returns:
    ---------
    result: numeric variable        
    """    
    
    categorical_dict =  dict(data.groupby(categorical_variable)[target].mean())
    
    result = data[categorical_variable].map(lambda i: categorical_dict[i])
    
    return result


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
