# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 09:26:06 2018

@author: Juan Antonio Morales
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support, accuracy_score

from src.utils import normalize_variables

def logistic_regression(data, C = 1, normalization = "robust"):
    """
    Parameters
    ---------
    data: DataFrame to fit logistic regression
    C = regularization parameter
    normalization: type of normalization to perform: "robust", "standard" and "minMax"
    
    Returns
    ---------
    result: evaluation model: AUC in train and test, confusion matrix, accuracy, recall and precision (treshold = 0.5)
    """
    ### pre-process
    print("Preprocessing...")
    # normalization
    data = normalize_variables(data, normalization)
    
    # one-hot-encoding
    categorical_variables = data.select_dtypes(include="object").columns
    data = pd.get_dummies(data, categorical_variables)
    
    # train/test split
    X = data.loc[:, data.columns!='loan_status']
    y = data['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4290)
    
    ### model
    print("Fitting the model...")
    # logistic regression
    log_reg = LogisticRegression(penalty='l2', C = C)
    log_reg.fit(X_train, y_train)
    
    ### evaluation
    print("Evaluation...")
    # scores
    y_scores_train = pd.DataFrame(log_reg.predict_proba(X_train)).loc[:,1]
    y_scores_test = pd.DataFrame(log_reg.predict_proba(X_test)).loc[:,1]

    # auc
    auc_train = roc_auc_score(y_train, y_scores_train)
    auc_test = roc_auc_score(y_test, y_scores_test)
    
    print("AUC:")
    print("AUC in the train set: {}".format(auc_train))
    print("AUC in the test set: {}".format(auc_test))
    print("------------------------")
    
    # confusion matrix and accuracy
    y_test_pred = log_reg.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Accuracy: {}".format(accuracy_score(y_test, y_test_pred)))
    print("Recall: {}".format(precision_recall_fscore_support(y_test, y_test_pred)[0]))
    print("Precision: {}".format(precision_recall_fscore_support(y_test, y_test_pred)[1]))
    print("------------------------")
    